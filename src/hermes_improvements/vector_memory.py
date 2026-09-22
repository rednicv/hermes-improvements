#!/usr/bin/env python3
"""
Vector Memory Store — Semantic Search on Memory Embeddings

Provides: VectorMemoryStore class for embedding-based semantic memory search.
Stores vector embeddings of memory entries and retrieves similar ones via
cosine similarity.

Design:
  - Lightweight — uses simple numpy-based approach when available,
    falls back to pure-Python cosine similarity
  - Persistent — saves embeddings to disk alongside the vector DB
  - Integrated — hooks into Hermes' existing memory system
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class VectorMemoryStore:
    """
    Semantic search over memory entries using vector embeddings.

    Stores embedding vectors for each memory entry and provides
    cosine-similarity-based retrieval of semantically similar items.

    The actual embedding computation relies on numpy when available;
    otherwise falls back to a simplified bag-of-words representation.

    File layout:
        {memory_dir}/
            memory_vectors.db   — JSON file with {key: {embedding, metadata, timestamp}}
            vector_config.json   — configuration and stats
    """

    def __init__(self, memory_dir: Path):
        """
        Args:
            memory_dir: Path to Hermes memories directory
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.memory_dir / "memory_vectors.db"
        self.config_path = self.memory_dir / "vector_config.json"

        # In-memory index: key -> {embedding, metadata, timestamp}
        self._index: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

        self._load()

    # ─── Persistence ──────────────────────────────────────────────────

    def _load(self):
        """Load existing embeddings from disk."""
        if self._loaded:
            return

        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    self._index = json.load(f)
                logger.debug(
                    "Loaded %d vector entries from %s",
                    len(self._index), self.db_path
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load vector DB: %s. Starting fresh.", e)
                self._index = {}
        else:
            logger.debug("No existing vector DB at %s, starting fresh", self.db_path)

        # Auto-index MEMORY.md if memory_vectors.db is empty
        if not self._index:
            memory_file = self.memory_dir / "MEMORY.md"
            if memory_file.exists():
                try:
                    lines = memory_file.read_text(encoding="utf-8").splitlines()
                    entries = [
                        {"action": "add", "target": "MEMORY.md", "content": line.strip()}
                        for line in lines if line.strip()
                    ]
                    if entries:
                        self.sync_from_memory(entries)
                except Exception as e:
                    logger.warning("Failed to auto-index MEMORY.md: %s", e)

        self._loaded = True

    _save_lock = None  # lazy-init class-level lock

    @classmethod
    def _get_save_lock(cls):
        if cls._save_lock is None:
            import threading
            cls._save_lock = threading.Lock()
        return cls._save_lock

    def _save(self):
        """Persist index to disk atomically with thread + inter-process safety."""
        import os, fcntl
        lock_path = self.db_path.with_suffix(".lock")
        with self._get_save_lock():  # thread-safety (same process)
            with open(lock_path, 'w') as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)  # inter-process safety
                try:
                    # Use PID-based temp name so concurrent processes don't clash
                    pid = os.getpid()
                    tmp = self.db_path.with_suffix(f".{pid}.tmp")
                    with open(tmp, 'w') as f:
                        json.dump(self._index, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    if tmp.exists():
                        tmp.replace(self.db_path)
                except OSError as e:
                    logger.warning("Vector DB save transient error (race): %s", e)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    # ─── Embedding ────────────────────────────────────────────────────

    _DEFAULT_MODEL = str(Path.home() / ".semantic_search/models/all-MiniLM-L6-v2")
    _MODEL_PATH = os.environ.get("HERMES_EMBEDDING_MODEL", _DEFAULT_MODEL)
    _model = None  # lazy-loaded, shared across instances

    @classmethod
    def _get_model(cls):
        """Lazy-load SentenceTransformer model (cached at class level)."""
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                path = cls._MODEL_PATH
                if not Path(path).exists() and not os.environ.get("HERMES_EMBEDDING_MODEL"):
                    path = "all-MiniLM-L6-v2"
                cls._model = SentenceTransformer(path)
                logger.info("✅ SentenceTransformer model loaded (%s)", path)
            except Exception as e:
                logger.warning("SentenceTransformer unavailable, using TF-IDF: %s", e)
                cls._model = False  # Mark as unavailable
        return cls._model if cls._model else None
    _STOPWORDS = {
        "și", "în", "la", "de", "cu", "pe", "un", "o", "că", "din", "este",
        "sunt", "sau", "dar", "nu", "mai", "se", "sa", "care", "pentru",
        "după", "prin", "între", "când", "dacă", "cum", "tot", "toți",
        "the", "a", "an", "is", "in", "on", "at", "to", "of", "and", "or",
        "not", "for", "with", "from", "that", "this", "it", "be", "are",
    }

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, split, remove stopwords and short tokens."""
        import re
        tokens = re.findall(r"[a-zA-ZăâîșțĂÂÎȘȚ]{3,}", text.lower())
        return [t for t in tokens if t not in self._STOPWORDS]

    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from all indexed texts. Returns word→index map."""
        from collections import Counter
        df: Counter = Counter()  # document frequency
        n_docs = len(self._index)
        if n_docs == 0:
            return {}
        for entry in self._index.values():
            words = set(self._tokenize(entry.get("text_preview", "")))
            df.update(words)
        # Keep top-512 words by document frequency, excluding hapax
        vocab = {
            word: idx
            for idx, (word, count) in enumerate(df.most_common(512))
            if count > 1 or n_docs <= 5
        }
        return vocab

    def _compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding. Priority:
        1. SentenceTransformer (all-MiniLM-L6-v2) — 384 dims, best quality
        2. TF-IDF over indexed vocabulary — 512 dims, good for known words
        3. Trigram hash fallback — 128 dims, cold start only
        """
        model = self._get_model()
        if model:
            try:
                vec = model.encode(text, show_progress_bar=False)
                return vec.tolist()
            except Exception as e:
                logger.warning("SentenceTransformer encode failed: %s", e)
        # Fallback to TF-IDF
        return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str) -> List[float]:
        import math
        vocab = self._build_vocab()
        if not vocab:
            return self._trigram_embed(text, dim=128)

        n_docs = max(len(self._index), 1)
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(vocab)

        # TF: term frequency in this text
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        max_tf = max(tf.values())

        # IDF from existing index
        df_counts: Dict[str, int] = {}
        for entry in self._index.values():
            for w in set(self._tokenize(entry.get("text_preview", ""))):
                df_counts[w] = df_counts.get(w, 0) + 1

        vec = [0.0] * len(vocab)
        for word, idx in vocab.items():
            if word in tf:
                tf_val = tf[word] / max_tf
                idf_val = math.log((n_docs + 1) / (df_counts.get(word, 0) + 1)) + 1
                vec[idx] = tf_val * idf_val

        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _trigram_embed(self, text: str, dim: int = 128) -> List[float]:
        """Fallback: character trigram hashing (used when index is empty)."""
        vec = [0.0] * dim
        words = text.lower().split()
        for pos, word in enumerate(words[:200]):
            weight = 1.0 / (1.0 + pos * 0.05)
            for i in range(len(word) - 2):
                h = hash(word[i:i + 3]) % dim
                vec[h] += weight * 0.1
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # ─── Index Operations ─────────────────────────────────────────────

    def add(self, key: str, text: str, metadata: Optional[Dict] = None, auto_save: bool = True):
        """
        Index a memory entry for semantic search.

        Args:
            key: Unique identifier (e.g., memory entry id)
            text: The memory content to embed
            metadata: Optional additional data to store
            auto_save: Write to disk immediately (default True). Set False when batch-adding.
        """
        embedding = self._compute_embedding(text)

        self._index[key] = {
            "embedding": embedding,
            "metadata": metadata or {},
            "text_preview": text[:200],
            "timestamp": time.time(),
            "updated_at": time.time(),
        }

        if auto_save:
            self._save()

    def update(self, key: str, text: str, metadata: Optional[Dict] = None, auto_save: bool = True):
        """Update an existing entry or add if new."""
        existing = self._index.get(key, {})
        new_meta = {**existing.get("metadata", {}), **(metadata or {})}
        self.add(key, text, new_meta, auto_save=auto_save)

    def remove(self, key: str):
        """Remove an entry from the index."""
        if key in self._index:
            del self._index[key]
            self._save()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over indexed entries.

        Args:
            query: Search text to embed and compare
            top_k: Number of top results to return

        Returns:
            List of dicts: [{key, score, metadata, text_preview}, ...]
            sorted by similarity score (descending)
        """
        if not self._index:
            return []

        query_vec = self._compute_embedding(query)

        scores = []
        for key, entry in self._index.items():
            emb = entry.get("embedding")
            if emb and len(emb) != len(query_vec):
                text = entry.get("text_preview", "")
                if text:
                    emb = self._compute_embedding(text)
                    entry["embedding"] = emb
            if emb and len(emb) == len(query_vec):
                score = self._cosine_similarity(query_vec, emb)
                scores.append((key, score, entry))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for key, score, entry in scores[:top_k]:
            results.append({
                "key": key,
                "score": round(score, 4),
                "metadata": entry.get("metadata", {}),
                "text_preview": entry.get("text_preview", ""),
                "timestamp": entry.get("timestamp", 0),
            })

        return results

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an entry by key."""
        entry = self._index.get(key)
        if entry:
            return {
                "key": key,
                "metadata": entry.get("metadata", {}),
                "text_preview": entry.get("text_preview", ""),
                "timestamp": entry.get("timestamp", 0),
            }
        return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity. Handles different lengths via zero-padding."""
        if len(a) != len(b):
            # Pad shorter vector with zeros
            n = max(len(a), len(b))
            a = a + [0.0] * (n - len(a))
            b = b + [0.0] * (n - len(b))
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ─── Stats & Management ───────────────────────────────────────────

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        return {
            "total_entries": len(self._index),
            "db_path": str(self.db_path),
            "db_size_mb": round(
                self.db_path.stat().st_size / (1024 * 1024), 2
            ) if self.db_path.exists() else 0,
            "loaded": self._loaded,
        }

    def clear(self):
        """Remove all entries."""
        self._index = {}
        self._save()
        logger.info("Vector memory index cleared")

    def sync_from_memory(self, memory_entries: List[Dict[str, Any]]):
        """
        Bulk-sync from Hermes' standard memory storage.
        Batch saves — writes to disk only once at the end.

        Args:
            memory_entries: List of {action, target, content, ...} entries
        """
        for entry in memory_entries:
            action = entry.get("action", "")
            target = entry.get("target", "")
            content = entry.get("content", "")

            if not content:
                continue

            key = f"{target}:{hash(content) & 0xFFFFFFFF:08x}"

            if action == "remove":
                self.remove(key)
            else:
                self.update(key, content, {
                    "action": action,
                    "target": target,
                }, auto_save=False)  # No individual saves

        # Single batch save at the end
        if memory_entries:
            self._save()

        logger.debug("Synced %d entries into vector index", len(memory_entries))
