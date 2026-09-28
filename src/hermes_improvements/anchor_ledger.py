#!/usr/bin/env python3
"""
Anchor Ledger — Web-Evidence Memory with Drift Detection for Hermes Agent

Provides:
  - AnchorLedger: Persistent SQLite/JSON-backed store for web research anchors.
  - AnchorRecord: Represents a verified web evidence snapshot.
  - DriftReport: Details discrepancies between historical baseline and live probe.

Design:
  - Snapshots URL, HTTP status, timestamp, key figures, content hash, and claim class.
  - Claim classes: VERIFIED-LIVE, UNVERIFIED, BOT_WALL_403, DELETED_404.
  - Automatically flags price/metric/naming drift across research passes.
  - Integrates seamlessly with VectorMemoryStore and Hermes adaptive turn context.
"""

import json
import os
import logging
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

CLAIM_VERIFIED_LIVE = "VERIFIED-LIVE"
CLAIM_UNVERIFIED = "UNVERIFIED"
CLAIM_BOT_WALL = "BOT_WALL_403"
CLAIM_NOT_FOUND = "DELETED_404"


@dataclass
class AnchorRecord:
    url: str
    http_status: int
    timestamp: float
    claim_class: str
    key_figures: Dict[str, Any]
    content_hash: str
    title: str = ""
    summary: str = ""
    last_verified: float = 0.0
    verification_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    url: str
    has_drift: bool
    status_changed: bool
    figures_changed: Dict[str, Tuple[Any, Any]]  # key -> (old_val, new_val)
    claim_class_changed: Optional[Tuple[str, str]]
    details: List[str]


class AnchorLedger:
    """
    Persistent store tracking web-evidence anchors and detecting drift over time.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
            storage_dir = home / "memories"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "anchor_ledger.db"
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anchor_records (
                    url TEXT PRIMARY KEY,
                    http_status INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    claim_class TEXT NOT NULL,
                    key_figures TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    last_verified REAL NOT NULL,
                    verification_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    drift_details TEXT NOT NULL,
                    FOREIGN KEY(url) REFERENCES anchor_records(url)
                )
            """)
            conn.commit()

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def record_probe(
        self,
        url: str,
        http_status: int,
        content: str = "",
        key_figures: Optional[Dict[str, Any]] = None,
        title: str = "",
        summary: str = "",
        claim_class: Optional[str] = None
    ) -> Tuple[AnchorRecord, Optional[DriftReport]]:
        """
        Record a live web probe attempt. Checks for drift against existing baseline.
        """
        now = time.time()
        key_figures = key_figures or {}
        content_hash = self.compute_hash(content) if content else ""

        if claim_class is None:
            if http_status in (200, 301, 302):
                claim_class = CLAIM_VERIFIED_LIVE
            elif http_status in (403, 429):
                claim_class = CLAIM_BOT_WALL
            elif http_status == 404:
                claim_class = CLAIM_NOT_FOUND
            else:
                claim_class = CLAIM_UNVERIFIED

        existing = self.get_anchor(url)
        drift_report = None

        if existing:
            # Check for drift
            drift_details = []
            status_changed = (existing.http_status != http_status)
            if status_changed:
                drift_details.append(f"HTTP status changed from {existing.http_status} to {http_status}")

            claim_changed = (existing.claim_class != claim_class)
            claim_tuple = (existing.claim_class, claim_class) if claim_changed else None
            if claim_changed:
                drift_details.append(f"Claim class changed from {existing.claim_class} to {claim_class}")

            fig_changes = {}
            for k, new_v in key_figures.items():
                old_v = existing.key_figures.get(k)
                if old_v != new_v:
                    fig_changes[k] = (old_v, new_v)
                    drift_details.append(f"Figure '{k}' changed from '{old_v}' to '{new_v}'")

            if content_hash and existing.content_hash and content_hash != existing.content_hash and not fig_changes:
                drift_details.append("Content hash changed (content updated)")

            has_drift = bool(drift_details)
            if has_drift:
                drift_report = DriftReport(
                    url=url,
                    has_drift=True,
                    status_changed=status_changed,
                    figures_changed=fig_changes,
                    claim_class_changed=claim_tuple,
                    details=drift_details
                )
                self._record_drift(url, now, json.dumps(drift_details))

            # Update existing record
            updated_figures = {**existing.key_figures, **key_figures}
            record = AnchorRecord(
                url=url,
                http_status=http_status,
                timestamp=existing.timestamp,
                claim_class=claim_class,
                key_figures=updated_figures,
                content_hash=content_hash or existing.content_hash,
                title=title or existing.title,
                summary=summary or existing.summary,
                last_verified=now,
                verification_count=existing.verification_count + 1
            )
        else:
            record = AnchorRecord(
                url=url,
                http_status=http_status,
                timestamp=now,
                claim_class=claim_class,
                key_figures=key_figures,
                content_hash=content_hash,
                title=title,
                summary=summary,
                last_verified=now,
                verification_count=1
            )

        self._save_anchor(record)
        return record, drift_report

    def _save_anchor(self, record: AnchorRecord):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO anchor_records
                (url, http_status, timestamp, claim_class, key_figures, content_hash, title, summary, last_verified, verification_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    http_status=excluded.http_status,
                    claim_class=excluded.claim_class,
                    key_figures=excluded.key_figures,
                    content_hash=excluded.content_hash,
                    title=COALESCE(NULLIF(excluded.title, ''), title),
                    summary=COALESCE(NULLIF(excluded.summary, ''), summary),
                    last_verified=excluded.last_verified,
                    verification_count=excluded.verification_count
            """, (
                record.url,
                record.http_status,
                record.timestamp,
                record.claim_class,
                json.dumps(record.key_figures),
                record.content_hash,
                record.title,
                record.summary,
                record.last_verified,
                record.verification_count
            ))
            conn.commit()

    def _record_drift(self, url: str, timestamp: float, details_json: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO drift_history (url, timestamp, drift_details) VALUES (?, ?, ?)",
                (url, timestamp, details_json)
            )
            conn.commit()

    def get_anchor(self, url: str) -> Optional[AnchorRecord]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM anchor_records WHERE url = ?", (url,))
            row = cur.fetchone()
            if not row:
                return None
            return AnchorRecord(
                url=row["url"],
                http_status=row["http_status"],
                timestamp=row["timestamp"],
                claim_class=row["claim_class"],
                key_figures=json.loads(row["key_figures"]),
                content_hash=row["content_hash"],
                title=row["title"],
                summary=row["summary"],
                last_verified=row["last_verified"],
                verification_count=row["verification_count"]
            )

    def search_anchors(self, query: str, limit: int = 5) -> List[AnchorRecord]:
        with self._get_conn() as conn:
            cur = conn.execute("""
                SELECT * FROM anchor_records
                WHERE url LIKE ? OR title LIKE ? OR summary LIKE ? OR key_figures LIKE ?
                ORDER BY last_verified DESC LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit))
            records = []
            for row in cur.fetchall():
                records.append(AnchorRecord(
                    url=row["url"],
                    http_status=row["http_status"],
                    timestamp=row["timestamp"],
                    claim_class=row["claim_class"],
                    key_figures=json.loads(row["key_figures"]),
                    content_hash=row["content_hash"],
                    title=row["title"],
                    summary=row["summary"],
                    last_verified=row["last_verified"],
                    verification_count=row["verification_count"]
                ))
            return records

    def format_context_block(self, urls: Optional[List[str]] = None, max_entries: int = 5) -> str:
        """
        Format verified anchors and drift alerts for insertion into Hermes turn context.
        """
        with self._get_conn() as conn:
            if urls:
                placeholders = ",".join(["?"] * len(urls))
                cur = conn.execute(f"SELECT * FROM anchor_records WHERE url IN ({placeholders})", urls)
            else:
                cur = conn.execute("SELECT * FROM anchor_records ORDER BY last_verified DESC LIMIT ?", (max_entries,))

            rows = cur.fetchall()
            if not rows:
                return ""

            lines = ["## Verified Web Evidence (Anchor Ledger)"]
            for row in rows:
                fig_str = ", ".join([f"{k}: {v}" for k, v in json.loads(row["key_figures"]).items()])
                fig_clause = f" [{fig_str}]" if fig_str else ""
                lines.append(f"- [{row['claim_class']}] {row['url']} ({row['http_status']}){fig_clause}")

            return "\n".join(lines)
