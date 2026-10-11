"""Hermes Improvements — Vector Memory Module (v4.0.0 Architecture).

Features:
- SQLite backing store with WAL mode (zero new dependencies, pure Python stdlib).
- Strategy pattern: EmbeddingBackend with SentenceTransformersBackend and TfidfBackend.
- Clean isolation: vocabulary/drift mechanics isolated strictly inside TfidfBackend.
- Model lineage: every stored vector records model_id + dimension (no mixed metric spaces).
- SHA-256 content_hash deduplication and update skips.
- Sentence-boundary smart truncation.
- Automatic one-shot migration from legacy JSON memory_vectors.db.
- Atomic sync without orphaned keys.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Helpers: Truncation, Hashes, Env Configuration ───────────────────────────

def _content_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def smart_truncate(text: str, max_chars: int = 200, suffix: str = " […]") -> str:
    """Truncate text cleanly at sentence or word boundaries."""
    if not text or len(text) <= max_chars:
        return text
    limit = max(10, max_chars - len(suffix))
    cut = text[:limit]
    for sep in (". ", "! ", "? ", "\n"):
        idx = cut.rfind(sep)
        if idx > int(max_chars * 0.5):
            return cut[: idx + 1].rstrip() + suffix
    idx = cut.rfind(" ")
    if idx > int(max_chars * 0.3):
        return cut[:idx].rstrip() + suffix
    return cut.rstrip() + suffix


def _get_preview_limit() -> int:
    try:
        val = int(os.environ.get("HERMES_MEMORY_PREVIEW_CHARS", 200))
        return max(50, val)
    except (ValueError, TypeError):
        return 200


def _get_max_vocab() -> int:
    try:
        val = int(os.environ.get("HERMES_TFIDF_MAX_VOCAB", 512))
        return max(128, val)
    except (ValueError, TypeError):
        return 512


def _get_model_retry_ttl() -> float:
    try:
        val = float(os.environ.get("HERMES_MODEL_RETRY_TTL", 300.0))
        return max(0.0, val)
    except (ValueError, TypeError):
        return 300.0


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors. Refuses zero padding across spaces."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── Embedding Backend Interface & Implementations ────────────────────────────

class EmbeddingBackend:
    """Abstract interface for embedding generation."""

    @property
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformersBackend(EmbeddingBackend):
    """SentenceTransformers neural embedding backend (384 dimensions by default)."""

    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer
        self._model_path = model_path
        self._model = SentenceTransformer(model_path)
        self._model_id = f"st:{Path(model_path).name if Path(model_path).exists() else model_path}"
        test_vec = self._model.encode("test", show_progress_bar=False)
        self._dimension = len(test_vec)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode(text, show_progress_bar=False)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class TfidfBackend(EmbeddingBackend):
    """Isolated pure-Python fallback embedding backend (TF-IDF + Trigram cold start)."""

    _STOPWORDS = {
        "și", "în", "la", "de", "cu", "pe", "un", "o", "că", "din", "este",
        "sunt", "sau", "dar", "nu", "mai", "se", "sa", "să", "care", "pentru",
        "după", "prin", "între", "când", "dacă", "cum", "tot", "toți",
        "the", "a", "an", "is", "in", "on", "at", "to", "of", "and", "or",
        "not", "for", "with", "from", "that", "this", "it", "be", "are",
    }

    def __init__(self, corpus_texts: Optional[List[str]] = None):
        self._vocab: Dict[str, int] = {}
        self._df_counts: Dict[str, int] = {}
        self._n_docs = 0
        if corpus_texts:
            self.rebuild_vocab(corpus_texts)

    @property
    def model_id(self) -> str:
        return "tfidf:internal"

    @property
    def dimension(self) -> int:
        return len(self._vocab) if self._vocab else 128

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ0-9_]{3,}", text.lower())
        return [t for t in tokens if t not in self._STOPWORDS]

    def rebuild_vocab(self, corpus_texts: List[str]):
        """Rebuild stable sorted vocabulary from indexed documents."""
        df: Counter = Counter()
        self._n_docs = len(corpus_texts)
        if self._n_docs == 0:
            self._vocab = {}
            self._df_counts = {}
            return

        for text in corpus_texts:
            words = set(self._tokenize(text))
            df.update(words)

        max_vocab = _get_max_vocab()
        top_words = sorted([w for w, _ in df.most_common(max_vocab)])
        self._vocab = {word: idx for idx, word in enumerate(top_words)}
        self._df_counts = dict(df)

    def _trigram_embed(self, text: str, dim: int = 128) -> List[float]:
        vec = [0.0] * dim
        words = text.lower().split()
        for pos, word in enumerate(words[:200]):
            weight = 1.0 / (1.0 + pos * 0.05)
            for i in range(len(word) - 2):
                trigram = word[i : i + 3]
                h = int.from_bytes(hashlib.sha256(trigram.encode("utf-8")).digest()[:4], "little") % dim
                vec[h] += weight * 0.1
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, text: str) -> List[float]:
        if not self._vocab:
            return self._trigram_embed(text, dim=128)

        n_docs = max(self._n_docs, 1)
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(self._vocab)

        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        max_tf = max(tf.values())

        vec = [0.0] * len(self._vocab)
        for word, idx in self._vocab.items():
            if word in tf:
                tf_val = tf[word] / max_tf
                idf_val = math.log((n_docs + 1) / (self._df_counts.get(word, 0) + 1)) + 1
                vec[idx] = tf_val * idf_val

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ─── SQLite Vector Serialization Helpers ─────────────────────────────────────

def _pack_vector(vec: List[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack_vector(blob: bytes) -> List[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return arr.tolist()


# ─── Main VectorMemoryStore (v4.0.0) ──────────────────────────────────────────

class VectorMemoryStore:
    """Semantic vector memory store backed by SQLite with automatic backend fallback."""

    _cached_backend: Optional[EmbeddingBackend] = None
    _backend_failed_time: float = 0.0

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.memory_dir / "memory_vectors.sqlite"
        self._init_db()
        self._migrate_legacy_json_if_needed()

        # Initialize backend
        self._backend = self._resolve_backend()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    text_preview TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    model_id TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_model ON memories(model_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(updated_at);")

    def _resolve_backend(self) -> EmbeddingBackend:
        """Resolve embedding backend: SentenceTransformers if available, else TF-IDF."""
        now = time.time()
        ttl = _get_model_retry_ttl()
        cls = VectorMemoryStore

        if cls._cached_backend is None or (
            isinstance(cls._cached_backend, TfidfBackend)
            and (now - cls._backend_failed_time) > ttl
        ):
            default_path = str(Path.home() / ".semantic_search/models/all-MiniLM-L6-v2")
            model_path = os.environ.get("HERMES_EMBEDDING_MODEL", default_path)
            try:
                backend = SentenceTransformersBackend(model_path)
                cls._cached_backend = backend
                logger.info("✅ SentenceTransformers backend active (%s)", backend.model_id)
                return backend
            except Exception as e:
                logger.warning("SentenceTransformers unavailable, activating TF-IDF backend: %s", e)
                cls._backend_failed_time = now

        if cls._cached_backend and isinstance(cls._cached_backend, SentenceTransformersBackend):
            return cls._cached_backend

        # Fallback to TF-IDF backend with corpus from DB
        corpus = self._get_all_texts()
        backend = TfidfBackend(corpus)
        cls._cached_backend = backend
        return backend

    def _get_all_texts(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT text FROM memories").fetchall()
            return [r["text"] for r in rows]

    def _migrate_legacy_json_if_needed(self):
        """One-shot migration from legacy memory_vectors.db JSON file."""
        legacy_json = self.memory_dir / "memory_vectors.db"
        if not legacy_json.exists():
            return

        try:
            with open(legacy_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("index", {})
            if entries:
                lim = _get_preview_limit()
                now = time.time()
                to_insert = []
                for k, v in entries.items():
                    text = v.get("text") or v.get("text_preview", "")
                    if not text:
                        continue
                    preview = v.get("text_preview") or smart_truncate(text, max_chars=lim)
                    h = v.get("content_hash") or _content_hash(text)
                    emb = v.get("embedding") or []
                    meta = v.get("metadata") or {}
                    ts = v.get("timestamp", now)
                    blob = _pack_vector(emb) if emb else b""
                    to_insert.append((k, text, preview, h, blob, "legacy:migrated", len(emb), json.dumps(meta), ts, ts))

                if to_insert:
                    with self._get_connection() as conn:
                        conn.executemany("""
                            INSERT OR REPLACE INTO memories
                            (key, text, text_preview, content_hash, vector, model_id, dim, metadata, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, to_insert)
                logger.info("✅ Migrated %d legacy JSON memories into SQLite", len(to_insert))

            # Backup legacy file safely
            bak_path = legacy_json.with_name(f"memory_vectors.db.migrated.{int(time.time())}.bak")
            legacy_json.replace(bak_path)
        except Exception as e:
            logger.error("Legacy JSON migration failed: %s", e)

    # ─── Store API ────────────────────────────────────────────────────────────

    def add(
        self,
        key: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        auto_save: bool = True,
    ):
        """Index a memory entry."""
        if not text:
            return

        meta = metadata or {}
        chash = _content_hash(text)
        preview = smart_truncate(text, max_chars=_get_preview_limit())
        now = time.time()

        # Update TF-IDF vocabulary if fallback is active
        if isinstance(self._backend, TfidfBackend):
            corpus = self._get_all_texts()
            corpus.append(text)
            self._backend.rebuild_vocab(corpus)

        vec = self._backend.embed(text)
        blob = _pack_vector(vec)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (key, text, text_preview, content_hash, vector, model_id, dim, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (key, text, preview, chash, blob, self._backend.model_id, len(vec), json.dumps(meta), now, now))

    def bulk_add(
        self,
        entries: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        auto_save: bool = True,
    ):
        """Batch ingestion in a single transaction with single-pass embedding."""
        if not entries:
            return

        lim = _get_preview_limit()
        now = time.time()

        # Rebuild TF-IDF vocab once before embedding if using fallback
        if isinstance(self._backend, TfidfBackend):
            corpus = self._get_all_texts()
            corpus.extend(item[1] for item in entries if item[1])
            self._backend.rebuild_vocab(corpus)

        texts = [item[1] for item in entries]
        vecs = self._backend.embed_batch(texts)

        rows = []
        for (key, text, meta), vec in zip(entries, vecs):
            m = meta or {}
            preview = smart_truncate(text, max_chars=lim)
            chash = _content_hash(text)
            blob = _pack_vector(vec)
            rows.append((key, text, preview, chash, blob, self._backend.model_id, len(vec), json.dumps(m), now, now))

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO memories
                (key, text, text_preview, content_hash, vector, model_id, dim, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

    def update(
        self,
        key: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        auto_save: bool = True,
    ):
        """Update an existing memory entry; skips re-embedding if content hash is unchanged."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT content_hash, metadata FROM memories WHERE key = ?", (key,)).fetchone()

        chash = _content_hash(text)
        if row and row["content_hash"] == chash:
            # Hash unchanged: update metadata only without re-embedding
            now = time.time()
            merged_meta = json.loads(row["metadata"])
            if metadata:
                merged_meta.update(metadata)
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE memories SET metadata = ?, updated_at = ? WHERE key = ?",
                    (json.dumps(merged_meta), now, key),
                )
            return

        # Content changed: full update
        self.add(key, text, metadata=metadata, auto_save=auto_save)

    def remove(self, key: str):
        """Delete an entry by key."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memories WHERE key = ?", (key,))

    def clear(self):
        """Remove all entries."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memories")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory entry by key."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT key, text, text_preview, metadata, updated_at FROM memories WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            return {
                "key": row["key"],
                "text": row["text"],
                "text_preview": row["text_preview"],
                "metadata": json.loads(row["metadata"]),
                "timestamp": row["updated_at"],
            }

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        since: Optional[float] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search with metadata filtering, model lineage check and atomic re-embedding."""
        if not query.strip():
            return []

        # Refresh backend if needed
        self._backend = self._resolve_backend()
        query_vec = self._backend.embed(query)
        target_model = self._backend.model_id
        target_dim = self._backend.dimension

        with self._get_connection() as conn:
            query_sql = "SELECT key, text, text_preview, vector, model_id, dim, metadata, updated_at FROM memories WHERE 1=1"
            params: List[Any] = []
            if since is not None:
                query_sql += " AND updated_at >= ?"
                params.append(since)
            rows = conn.execute(query_sql, params).fetchall()

        scores: List[Tuple[str, float, sqlite3.Row]] = []
        stale_reembeds = []

        for r in rows:
            meta = json.loads(r["metadata"])
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue

            row_model = r["model_id"]
            row_dim = r["dim"]

            if row_model != target_model or row_dim != target_dim:
                # Re-embed stale vectors from authoritative complete text
                new_vec = self._backend.embed(r["text"])
                stale_reembeds.append((r["key"], _pack_vector(new_vec), target_model, len(new_vec)))
                stored_vec = new_vec
            else:
                stored_vec = _unpack_vector(r["vector"])

            score = _cosine_similarity(query_vec, stored_vec)
            if score >= min_score:
                scores.append((r["key"], score, r))

        # Asynchronously update stale rows in batch if any occurred
        if stale_reembeds:
            with self._get_connection() as conn:
                conn.executemany(
                    "UPDATE memories SET vector = ?, model_id = ?, dim = ? WHERE key = ?",
                    [(b, m, d, k) for k, b, m, d in stale_reembeds],
                )

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for key, score, r in scores[:top_k]:
            results.append({
                "key": key,
                "score": round(score, 4),
                "text": r["text"],
                "text_preview": r["text_preview"],
                "metadata": json.loads(r["metadata"]),
                "timestamp": r["updated_at"],
            })
        return results

    def sync_from_memory(self, memory_entries: List[Dict[str, Any]]):
        """Atomic synchronization from Hermes memories without creating orphaned keys."""
        if not memory_entries:
            return

        valid_keys_per_target: Dict[str, set] = {}
        to_upsert = []

        for entry in memory_entries:
            action = entry.get("action", "")
            target = entry.get("target", "")
            content = entry.get("content", "")

            if not content or not target:
                continue

            key = f"{target}:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"
            valid_keys_per_target.setdefault(target, set())

            if action == "remove":
                self.remove(key)
            else:
                valid_keys_per_target[target].add(key)
                to_upsert.append((key, content, {"target": target, "action": action}))

        if to_upsert:
            self.bulk_add(to_upsert)

        # Atomic prune of orphaned keys in this target domain
        for target, valid_keys in valid_keys_per_target.items():
            with self._get_connection() as conn:
                existing = conn.execute("SELECT key, metadata FROM memories").fetchall()
                for r in existing:
                    meta = json.loads(r["metadata"])
                    if meta.get("target") == target and r["key"] not in valid_keys:
                        conn.execute("DELETE FROM memories WHERE key = ?", (r["key"],))

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory store statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()["cnt"]
            models = conn.execute("SELECT model_id, COUNT(*) as cnt FROM memories GROUP BY model_id").fetchall()

        return {
            "total_entries": total,
            "storage_backend": "sqlite",
            "active_model": self._backend.model_id,
            "active_dim": self._backend.dimension,
            "models_distribution": {r["model_id"]: r["cnt"] for r in models},
        }
