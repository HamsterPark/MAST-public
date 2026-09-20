"""SQLite store initialisation + connection management for v2 logging.

The v2 system lives in its own file (default ``experiments/mast_experiments_v2.db``)
so it does not collide with the vendored v1 ``mast_experiments.db``.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from mast.logging.v2.schema import ALL_DDL, REQUIRED_PRAGMAS

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = "mast_experiments_v2.db"


class ExperimentStoreV2:
    """The v2 experiment record store backed by a single SQLite WAL file."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-thread connection cache for repository methods that don't take a conn explicitly.
        self._tls = threading.local()
        self._init_schema()

    # ── Initialisation ────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self.connect() as conn:
            for ddl in ALL_DDL:
                conn.executescript(ddl)
            # Stamp schema version row (idempotent).
            row = conn.execute(
                "SELECT version FROM schema_versions WHERE version = ?",
                ("2.0.0",),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_versions(version, applied_at, description, applied_by) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        "2.0.0",
                        datetime.now(timezone.utc).isoformat(),
                        "Initial v2 schema (compass terminal plan, 2026-05-19): "
                        "13 entities, append-only triggers, ULID, HLC, materialised "
                        "rollups, generated columns for hot filters.",
                        "system",
                    ),
                )
            conn.commit()

    # ── Connection helpers ────────────────────────────────────────────

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Context-managed connection. Commits on exit unless an exception is raised."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        for pragma in REQUIRED_PRAGMAS:
            conn.execute(pragma)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def raw(self) -> sqlite3.Connection:
        """Return a thread-local raw connection. Caller manages commits."""
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.db_path), check_same_thread=False)
            c.row_factory = sqlite3.Row
            for pragma in REQUIRED_PRAGMAS:
                c.execute(pragma)
            self._tls.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._tls, "conn", None)
        if c is not None:
            try:
                c.close()
            finally:
                self._tls.conn = None

    # ── Schema info ───────────────────────────────────────────────────

    def schema_version(self) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT version FROM schema_versions ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
            return row["version"] if row else None

    def table_counts(self) -> dict[str, int]:
        names = (
            "schema_versions campaigns samples plans experiments instrument_states "
            "actions scan_files observations events approvals reviews claims entity_refs "
            "evidence_edges audit_log policies mv_campaign_stats"
        ).split()
        with self.connect() as conn:
            return {
                n: conn.execute(f"SELECT COUNT(*) AS c FROM {n}").fetchone()["c"]
                for n in names
            }


def open_store(db_path: str | Path | None = None) -> ExperimentStoreV2:
    """Open (and lazily initialise) the v2 store.

    If *db_path* is None, looks in $MAST_DATA_DIR/experiments/, else cwd.
    """
    if db_path is None:
        import os
        root = Path(os.environ.get("MAST_DATA_DIR", "."))
        db_path = root / "experiments" / DEFAULT_DB_NAME
    return ExperimentStoreV2(db_path)
