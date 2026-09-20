"""MemoryStore — persistent, filesystem-style agent memory in the experiment DB.

Design: docs/v2/design/agentic-cognition.md §1.

The memory lives in the SAME SQLite database as the experiment record (so a
project's "cognitive context" is part of its record and exports with it), but
this class owns its own tables via ``CREATE TABLE IF NOT EXISTS`` — it does not
modify ``ExperimentStorage``. Entries are addressed filesystem-style by
``(namespace, path)`` so re-writing a path UPDATES it, and a ``MEMORY.md``-style
index can be loaded into an agent's context at session start.

Everything is best-effort and never raises on a bad row / odd type — agent
memory must never crash a task. ``path`` is sanitised (no traversal / separators)
so a memory write can't escape its logical namespace.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KINDS = ("note", "insight", "summary", "hypothesis", "protocol", "dream",
         "brainstorm")
_PATH_RE = re.compile(r"^[\w./\-一-鿿]{1,200}$")


def sanitize_path(path: str) -> str:
    """Coerce *path* to a safe filesystem-style key (no traversal/separators abuse)."""
    p = str(path or "").strip().strip("/")
    p = p.replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    if ".." in p.split("/"):
        p = "/".join(seg for seg in p.split("/") if seg != "..")
    if not p or not _PATH_RE.match(p):
        # fall back to a slugged form rather than reject (memory must persist)
        p = re.sub(r"[^\w.\-一-鿿]+", "_", p).strip("_/") or "note"
    return p[:200]


class MemoryStore:
    """Filesystem-style persistent memory backed by the experiment SQLite DB."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    @classmethod
    def from_storage(cls, storage) -> "MemoryStore":
        """Build from an ExperimentStorage (shares its database file)."""
        return cls(getattr(storage, "_db_path"))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace     TEXT NOT NULL,
                    path          TEXT NOT NULL,
                    title         TEXT NOT NULL DEFAULT '',
                    content       TEXT NOT NULL,
                    kind          TEXT NOT NULL DEFAULT 'note',
                    tags          TEXT NOT NULL DEFAULT '[]',
                    experiment_id TEXT,
                    author        TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    pinned        INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(namespace, path)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_ns "
                         "ON memory(namespace, updated_at)")

    # ── write / upsert ────────────────────────────────────────────────
    def write(self, namespace: str, path: str, content: str, *, title: str = "",
              kind: str = "note", tags: list | None = None,
              experiment_id: str | None = None, author: str = "",
              pinned: bool = False) -> dict:
        ns = str(namespace or "global").strip() or "global"
        p = sanitize_path(path)
        k = kind if kind in KINDS else "note"
        now = datetime.now().isoformat()
        tags_json = json.dumps(list(tags or []), ensure_ascii=False)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, created_at FROM memory WHERE namespace=? AND path=?",
                (ns, p)).fetchone()
            if row:
                conn.execute(
                    "UPDATE memory SET title=?, content=?, kind=?, tags=?, "
                    "experiment_id=?, author=?, updated_at=?, pinned=? "
                    "WHERE id=?",
                    (title, content, k, tags_json, experiment_id, author, now,
                     1 if pinned else 0, row["id"]))
                mid = row["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO memory (namespace, path, title, content, kind, "
                    "tags, experiment_id, author, created_at, updated_at, pinned) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (ns, p, title, content, k, tags_json, experiment_id, author,
                     now, now, 1 if pinned else 0))
                mid = int(cur.lastrowid)
        return {"id": mid, "namespace": ns, "path": p}

    # ── read / list / search ──────────────────────────────────────────
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except Exception:
            d["tags"] = []
        d["pinned"] = bool(d.get("pinned"))
        return d

    def read(self, namespace: str, path: str) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM memory WHERE namespace=? AND path=?",
                             (namespace, sanitize_path(path))).fetchone()
        return self._row(r) if r else None

    def get(self, mem_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM memory WHERE id=?", (mem_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, namespace: str | None = None, *, kind: str | None = None,
             tag: str | None = None, limit: int = 200) -> list[dict]:
        q = "SELECT * FROM memory"
        clauses, params = [], []
        if namespace is not None:
            clauses.append("namespace=?"); params.append(namespace)
        if kind is not None:
            clauses.append("kind=?"); params.append(kind)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY pinned DESC, updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = [self._row(r) for r in conn.execute(q, params).fetchall()]
        if tag:
            rows = [r for r in rows if tag in (r.get("tags") or [])]
        return rows

    def search(self, query: str, *, namespace: str | None = None,
               limit: int = 20) -> list[dict]:
        """Substring search over title+content+tags (case-insensitive)."""
        ql = f"%{query.lower()}%"
        q = ("SELECT * FROM memory WHERE (LOWER(title) LIKE ? OR LOWER(content) "
             "LIKE ? OR LOWER(tags) LIKE ?)")
        params: list = [ql, ql, ql]
        if namespace is not None:
            q += " AND namespace=?"; params.append(namespace)
        q += " ORDER BY pinned DESC, updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            return [self._row(r) for r in conn.execute(q, params).fetchall()]

    def delete(self, namespace: str, path: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memory WHERE namespace=? AND path=?",
                               (namespace, sanitize_path(path)))
            return cur.rowcount > 0

    def pin(self, mem_id: int, on: bool = True) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE memory SET pinned=? WHERE id=?",
                         (1 if on else 0, mem_id))

    def namespaces(self) -> list[str]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT namespace FROM memory ORDER BY namespace").fetchall()]

    # ── MEMORY.md-style index for session-start injection ─────────────
    def index_markdown(self, namespace: str | None = None, *, max_lines: int = 60) -> str:
        """One-line-per-entry index (pinned first, then most-recent) to load into
        an agent's context at session start, like a coding agent's memory file."""
        rows = self.list(namespace, limit=max_lines)
        if not rows:
            return "# MEMORY\n(空 — 还没有持久化记忆)\n"
        lines = ["# MEMORY (持久化记忆索引 — 用 memory_read(path) 取细节)"]
        for r in rows:
            star = "📌 " if r["pinned"] else ""
            ns = "" if (namespace or r["namespace"] == "global") else f"[{r['namespace']}] "
            title = r["title"] or (r["content"][:60].replace("\n", " "))
            lines.append(f"- {star}{ns}`{r['path']}` ({r['kind']}) — {title}")
        return "\n".join(lines) + "\n"


__all__ = ["MemoryStore", "sanitize_path", "KINDS"]
