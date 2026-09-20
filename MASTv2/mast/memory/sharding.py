"""Conversation phase-sharding (docs/v2/design/agentic-cognition.md §2).

A long conversation is split into *phases*; when a phase ends (manually, or by
an auto-heuristic) a compressed summary of that phase is produced and stored —
both into the ``conversation_phase`` table and as a ``kind="summary"`` memory
entry — so later context can use the summary instead of the full transcript,
avoiding context blow-up while preserving the thread.

Reads the ``conversation_log`` written by ExperimentStorage; owns its own
``conversation_phase`` table (CREATE IF NOT EXISTS) so it doesn't touch
storage.py. The summariser defaults to a dependency-free rule-based extractor so
it works offline; the GUI may inject an LLM summariser for richer summaries.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# A summariser turns a list of {role, content} messages into a short string.
Summarizer = Callable[[list[dict]], str]

AUTO_SHARD_MSG_THRESHOLD = 24   # messages since phase start before auto-sharding


def rule_based_summary(messages: list[dict]) -> str:
    """Offline summary: first ask, last reply, counts, salient bullets."""
    if not messages:
        return "(空阶段)"
    users = [m for m in messages if m.get("role") == "user"]
    asst = [m for m in messages if m.get("role") == "assistant"]
    first_ask = (users[0]["content"] if users else "").strip().replace("\n", " ")
    last_reply = (asst[-1]["content"] if asst else "").strip().replace("\n", " ")
    parts = [f"阶段含 {len(messages)} 条消息({len(users)} 用户 / {len(asst)} 助手)。"]
    if first_ask:
        parts.append(f"起始诉求: {first_ask[:160]}")
    if last_reply:
        parts.append(f"最后结论: {last_reply[:160]}")
    return " ".join(parts)


class PhaseManager:
    """Tracks + summarises conversation phases for one DB (one experiment record)."""

    def __init__(self, db_path: str | Path, *, memory_store=None,
                 summarizer: Summarizer | None = None):
        self._db_path = Path(db_path)
        self._memory = memory_store
        self._summarizer = summarizer or rule_based_summary
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_phase (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id  TEXT,
                    phase_index    INTEGER NOT NULL,
                    title          TEXT NOT NULL DEFAULT '',
                    started_msg_id INTEGER,
                    ended_msg_id   INTEGER,
                    summary        TEXT NOT NULL DEFAULT '',
                    created_at     TEXT NOT NULL
                )
            """)

    def _max_conv_id(self, experiment_id: str | None) -> int:
        try:
            with self._connect() as conn:
                if experiment_id:
                    r = conn.execute(
                        "SELECT MAX(id) FROM conversation_log WHERE experiment_id=?",
                        (experiment_id,)).fetchone()
                else:
                    r = conn.execute("SELECT MAX(id) FROM conversation_log").fetchone()
            return int(r[0]) if r and r[0] is not None else 0
        except sqlite3.OperationalError:
            return 0  # conversation_log not created yet

    # ── phases ────────────────────────────────────────────────────────
    def current_phase(self, experiment_id: str | None = None) -> dict | None:
        with self._connect() as conn:
            if experiment_id:
                r = conn.execute(
                    "SELECT * FROM conversation_phase WHERE ended_msg_id IS NULL "
                    "AND experiment_id IS ? ORDER BY phase_index DESC LIMIT 1",
                    (experiment_id,)).fetchone()
            else:
                r = conn.execute(
                    "SELECT * FROM conversation_phase WHERE ended_msg_id IS NULL "
                    "ORDER BY phase_index DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def start_phase(self, title: str = "", experiment_id: str | None = None) -> dict:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT MAX(phase_index) FROM conversation_phase WHERE experiment_id IS ?",
                (experiment_id,)).fetchone()
            idx = (int(r[0]) + 1) if (r and r[0] is not None) else 0
            now = datetime.now().isoformat()
            start_id = self._max_conv_id(experiment_id)
            cur = conn.execute(
                "INSERT INTO conversation_phase "
                "(experiment_id, phase_index, title, started_msg_id, created_at) "
                "VALUES (?,?,?,?,?)",
                (experiment_id, idx, title, start_id, now))
            pid = int(cur.lastrowid)
        logger.info("conversation phase %d started (%s)", idx, title or "untitled")
        return {"id": pid, "phase_index": idx, "title": title}

    def _phase_messages(self, phase: dict) -> list[dict]:
        start = phase.get("started_msg_id") or 0
        end = phase.get("ended_msg_id")
        exp = phase.get("experiment_id")
        q = "SELECT role, content FROM conversation_log WHERE id > ?"
        params: list = [start]
        if end is not None:
            q += " AND id <= ?"; params.append(end)
        if exp:
            q += " AND experiment_id IS ?"; params.append(exp)
        q += " ORDER BY id"
        try:
            with self._connect() as conn:
                return [dict(r) for r in conn.execute(q, params).fetchall()]
        except sqlite3.OperationalError:
            return []

    def end_phase(self, experiment_id: str | None = None, *,
                  summarize: bool = True) -> dict | None:
        """Close the open phase, summarise it, and persist the summary (also as
        a memory entry). Returns the closed phase dict or None if none open."""
        phase = self.current_phase(experiment_id)
        if phase is None:
            return None
        end_id = self._max_conv_id(experiment_id)
        phase["ended_msg_id"] = end_id
        summary = ""
        if summarize:
            try:
                summary = self._summarizer(self._phase_messages(phase)) or ""
            except Exception as exc:
                logger.debug("phase summariser failed: %s", exc)
                summary = rule_based_summary(self._phase_messages(phase))
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversation_phase SET ended_msg_id=?, summary=? WHERE id=?",
                (end_id, summary, phase["id"]))
        # also store as a memory summary so dreaming / context-assembly can use it
        if summary and self._memory is not None:
            try:
                ns = f"experiment:{experiment_id}" if experiment_id else "global"
                self._memory.write(
                    ns, f"phases/phase-{phase['phase_index']}.md", summary,
                    title=phase.get("title") or f"Phase {phase['phase_index']}",
                    kind="summary", experiment_id=experiment_id, author="shard")
            except Exception as exc:
                logger.debug("phase summary → memory failed: %s", exc)
        phase["summary"] = summary
        return phase

    def maybe_auto_shard(self, experiment_id: str | None = None, *,
                         threshold: int = AUTO_SHARD_MSG_THRESHOLD) -> dict | None:
        """If the open phase has accumulated >= threshold messages, end it and
        start a fresh one. Returns the NEW phase if a shard happened, else None."""
        phase = self.current_phase(experiment_id)
        if phase is None:
            return self.start_phase("", experiment_id)
        n = self._max_conv_id(experiment_id) - (phase.get("started_msg_id") or 0)
        if n >= threshold:
            self.end_phase(experiment_id, summarize=True)
            return self.start_phase("", experiment_id)
        return None

    def list_phases(self, experiment_id: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if experiment_id:
                rows = conn.execute(
                    "SELECT * FROM conversation_phase WHERE experiment_id IS ? "
                    "ORDER BY phase_index", (experiment_id,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM conversation_phase ORDER BY phase_index").fetchall()
        return [dict(r) for r in rows]


__all__ = ["PhaseManager", "rule_based_summary", "Summarizer",
           "AUTO_SHARD_MSG_THRESHOLD"]
