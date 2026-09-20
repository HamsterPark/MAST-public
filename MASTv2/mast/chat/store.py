"""ConversationStore — durable index of chat conversations.

Mirrors the ``MemoryStore`` pattern (``mast/memory/store.py``): lives in the SAME
experiment SQLite DB, owns its own table via ``CREATE TABLE IF NOT EXISTS``, uses
WAL, and never raises on a bad row (chat must keep working).

A *conversation* is just a stable LangGraph ``thread_id`` plus display metadata.
The actual message history lives in the checkpointer keyed by that ``thread_id``;
this store only indexes (id ↔ agent ↔ thread ↔ title) so the GUI can list /
switch / rename / delete and resume across restarts. Both **private** chats
(私聊 a single agent) and **group** chats (群聊 the supervisor/orchestrator) are
rows in this one table, distinguished by ``kind``.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

KINDS = ("private", "group")

# Per-conversation transcript row cap. The transcript shares the experiment DB
# file, and a long-lived resumed 群聊 appends indefinitely (every message / tool
# call / interrupt / done), so an unbounded table would slowly bloat the shared
# DB. append_message trims the oldest rows beyond this cap on write. Comfortably
# above messages_for's 4000-row reconnect window so a replay always sees a full
# tail; one orchestrator run is bounded by recursion_limit (~hundreds of rows),
# so this still holds many resumed turns.
_MAX_TRANSCRIPT_ROWS = 8000


class ConversationStore:
    """SQLite-backed conversation index, sharing the experiment DB file."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    @classmethod
    def from_storage(cls, storage) -> "ConversationStore":
        """Build from an ExperimentStorage, but in a SEPARATE sibling DB file
        (``mast_conversations.db``) so chat data is physically decoupled from the
        experiment-records DB: independent WAL/locks, and an experiment backup no
        longer drags the whole transcript along. Conversations that used to live
        in the shared experiment DB are migrated once on first build."""
        exp_db = Path(getattr(storage, "_db_path"))
        conv_db = exp_db.parent / "mast_conversations.db"
        inst = cls(conv_db)
        inst._migrate_from_legacy_db(exp_db)
        return inst

    def _migrate_from_legacy_db(self, legacy_db_path) -> None:
        """Best-effort, idempotent, SELF-HEALING migration of conversations +
        transcript from the OLD shared experiment DB into this dedicated file.

        Copies by per-row set-difference (only ids / (id,seq) not already here),
        so it is safe to run on EVERY boot and, crucially, does NOT rely on a
        "new DB is empty" guard:
          • self-heals a partially-failed first migration — a default conversation
            seeded by the chat engine right after build no longer strands legacy
            rows (the old COUNT(*)>0 guard skipped migration forever once any row
            existed, permanently hiding the user's old chats);
          • picks up rows that landed in the legacy DB after the split.
        Tolerates older legacy schemas (missing columns → take this table's
        DEFAULT). Once every legacy conversation is present here, the legacy chat
        tables are DROPPED (best-effort) so chat data truly leaves the experiment
        DB (the point of the split) and later boots short-circuit. NEVER raises —
        a failed migration must not block boot."""
        try:
            legacy = Path(legacy_db_path)
            if not legacy.exists() or legacy.resolve() == self._db_path.resolve():
                return
            conn = self._connect()
            try:
                conn.execute("ATTACH DATABASE ? AS legacy", (str(legacy),))
                if not conn.execute(
                    "SELECT 1 FROM legacy.sqlite_master "
                    "WHERE type='table' AND name='conversations'").fetchone():
                    return  # legacy never had a chat table (or already migrated+dropped)

                # Column-intersection so an older legacy schema (missing
                # experiment_id / archived / kind) degrades instead of throwing.
                lcols = {r["name"] for r in conn.execute(
                    "PRAGMA legacy.table_info(conversations)").fetchall()}
                conv_cols = [c for c in (
                    "conversation_id", "agent_id", "thread_id", "title", "kind",
                    "created_at", "updated_at", "last_message_preview",
                    "experiment_id", "sample_id", "archived") if c in lcols]
                cl = ", ".join(conv_cols)
                migrated = conn.execute(
                    f"INSERT INTO conversations ({cl}) SELECT {cl} "
                    "FROM legacy.conversations lc WHERE lc.conversation_id NOT IN "
                    "(SELECT conversation_id FROM conversations)").rowcount

                if conn.execute(
                    "SELECT 1 FROM legacy.sqlite_master WHERE type='table' "
                    "AND name='conversation_messages'").fetchone():
                    mcols = {r["name"] for r in conn.execute(
                        "PRAGMA legacy.table_info(conversation_messages)").fetchall()}
                    msg_cols = [c for c in (
                        "conversation_id", "seq", "agent_id", "role", "kind", "text", "t")
                        if c in mcols]
                    if {"conversation_id", "seq"} <= set(msg_cols):
                        ml = ", ".join(msg_cols)
                        conn.execute(
                            f"INSERT INTO conversation_messages ({ml}) SELECT {ml} "
                            "FROM legacy.conversation_messages lm WHERE NOT EXISTS "
                            "(SELECT 1 FROM conversation_messages m WHERE "
                            "m.conversation_id=lm.conversation_id AND m.seq=lm.seq)")
                conn.commit()
                if migrated:
                    logger.info("ConversationStore: migrated %d conversation(s) from "
                                "legacy experiment DB %s → %s", migrated, legacy,
                                self._db_path)

                # Drop the legacy copies once parity is reached, so chat data
                # actually leaves the experiment DB. Best-effort: a write-lock
                # clash just defers the drop to a later boot (migration re-runs).
                remaining = conn.execute(
                    "SELECT COUNT(*) AS c FROM legacy.conversations lc WHERE "
                    "lc.conversation_id NOT IN (SELECT conversation_id FROM "
                    "conversations)").fetchone()["c"]
                if remaining == 0:
                    try:
                        conn.execute("DROP TABLE IF EXISTS legacy.conversation_messages")
                        conn.execute("DROP TABLE IF EXISTS legacy.conversations")
                        conn.commit()
                        logger.info("ConversationStore: dropped legacy chat tables "
                                    "from %s (chat data now lives only in %s)",
                                    legacy, self._db_path)
                    except Exception as exc:  # noqa: BLE001 — drop is cleanup, non-fatal
                        logger.debug("legacy chat table drop deferred: %s", exc)
                conn.execute("DETACH DATABASE legacy")
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 — migration must never block boot
            logger.warning("ConversationStore legacy migration skipped: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        # Serialize concurrent writers (e.g. two run-task streams persisting into
        # the same group conversation) instead of failing fast with 'database is
        # locked' — bounded so a stuck writer can never hang a request forever.
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id      TEXT PRIMARY KEY,
                    agent_id             TEXT NOT NULL,
                    thread_id            TEXT NOT NULL UNIQUE,
                    title                TEXT NOT NULL DEFAULT '新对话',
                    kind                 TEXT NOT NULL DEFAULT 'private',
                    created_at           TEXT NOT NULL,
                    updated_at           TEXT NOT NULL,
                    last_message_preview TEXT NOT NULL DEFAULT '',
                    experiment_id        TEXT,
                    sample_id            TEXT,
                    archived             INTEGER NOT NULL DEFAULT 0
                )
            """)
            # BEFORE the indexes: an index over a migrated column can only be
            # built once ALTER TABLE has added it to an older DB.
            self._migrate(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_updated "
                         "ON conversations(kind, archived, updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_sample "
                         "ON conversations(sample_id, kind, updated_at)")
            # Durable per-conversation transcript. The checkpointer is the source
            # of truth for a PRIVATE chat's message state, but the GROUP (群聊)
            # orchestrator run streams a RICHER per-agent transcript (subgraph
            # buckets) than the top-level checkpoint channel can reconstruct — so
            # the run-task bridge persists each rendered entry here. This is what
            # lets a 群聊 survive a tab switch / reload (re-read on reconnect) and
            # lets each agent's 群聊 contributions surface in its per-agent view
            # (agent_id-filtered). append_message trims each conversation to the
            # most recent _MAX_TRANSCRIPT_ROWS so this shared DB can't bloat.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    conversation_id TEXT NOT NULL,
                    seq             INTEGER NOT NULL,
                    agent_id        TEXT NOT NULL DEFAULT '',
                    role            TEXT NOT NULL DEFAULT '',
                    kind            TEXT NOT NULL DEFAULT 'message',
                    text            TEXT NOT NULL DEFAULT '',
                    t               REAL NOT NULL DEFAULT 0,
                    -- Structured sidecar (JSON) for rows whose `text` is a
                    -- SUMMARY rather than the whole story — currently a tool
                    -- call's full arguments. Without it,
                    -- summarising the transcript would mean DELETING the
                    -- arguments from the only durable record of the run.
                    meta            TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (conversation_id, seq)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_convmsg_agent "
                         "ON conversation_messages(agent_id, kind, t)")

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns missing from an OLDER conversations / messages table.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing table, so a new
        column reaches an already-deployed DB only through ALTER TABLE. Same
        inline PRAGMA-then-ALTER shape ``ExperimentStorage._migrate`` uses (that
        is how ``actions.sample_id`` was added).

        ``sample_id``: a chat now hangs off the
        SAMPLE it was started on, so the operator sees the real hierarchy
        实验 → 若干样品 → 每个样品若干群聊. NULL on every pre-existing row and on
        any chat started with no sample active — those stay listable and render
        in an explicit "未归属样品" bucket, never hidden.

        ``conversation_messages.meta``: a tool-call row's
        ``text`` is now a one-line Chinese summary, so the arguments it no longer
        spells out live here as JSON. Empty on every pre-existing row — those
        rows keep rendering exactly as they did, just with no 参数 to expand.
        The PRAGMA returns nothing when the table does not exist yet, which is
        why the ALTER is guarded on a non-empty column set rather than on the
        column's absence alone."""
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(conversations)").fetchall()}
        if "sample_id" not in cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN sample_id TEXT")
        mcols = {r[1] for r in conn.execute(
            "PRAGMA table_info(conversation_messages)").fetchall()}
        if mcols and "meta" not in mcols:
            conn.execute("ALTER TABLE conversation_messages "
                         "ADD COLUMN meta TEXT NOT NULL DEFAULT ''")

    # ── create / read / list ──────────────────────────────────────────
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["archived"] = bool(d.get("archived"))
        return d

    def create(self, agent_id: str, *, kind: str = "private",
               title: str | None = None, experiment_id: str | None = None,
               sample_id: str | None = None, thread_id: str | None = None) -> dict:
        """Create a new conversation; returns its row dict.

        ``experiment_id`` / ``sample_id`` place the chat in the operator's mental
        hierarchy 实验 → 样品 → 群聊. Both are
        WHERE-IT-WAS-STARTED tags, not a lease: a chat is never hidden because
        its sample ended, and either may be NULL (no experiment/sample active)."""
        k = kind if kind in KINDS else "private"
        cid = uuid.uuid4().hex
        tid = thread_id or f"conv-{agent_id}-{cid[:12]}"
        now = datetime.now().isoformat()
        ttl = title or ("新对话" if k == "private" else "新任务")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, agent_id, thread_id, "
                "title, kind, created_at, updated_at, last_message_preview, "
                "experiment_id, sample_id, archived) VALUES (?,?,?,?,?,?,?,?,?,?,0)",
                (cid, agent_id, tid, ttl, k, now, now, "", experiment_id, sample_id))
        return {"conversation_id": cid, "agent_id": agent_id, "thread_id": tid,
                "title": ttl, "kind": k, "created_at": now, "updated_at": now,
                "last_message_preview": "", "experiment_id": experiment_id,
                "sample_id": sample_id, "archived": False}

    def get(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM conversations WHERE conversation_id=?",
                             (conversation_id,)).fetchone()
        return self._row(r) if r else None

    def get_by_thread(self, thread_id: str) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM conversations WHERE thread_id=?",
                             (thread_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, *, kind: str | None = None, agent_id: str | None = None,
             experiment_id: str | None = None, sample_id: str | None = None,
             include_archived: bool = False, limit: int = 200) -> list[dict]:
        """List conversations, newest first.

        ``experiment_id`` / ``sample_id`` are OPT-IN drill-down filters .
        They default to None = no filter, so the plain list still spans every
        experiment and sample — a chat is grouped by where it started, never
        confined to it. Pass the literal string ``"none"`` to select the
        UNASSIGNED bucket (rows whose column is NULL), which is what old
        pre-#17 conversations land in."""
        q = "SELECT * FROM conversations"
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind=?"); params.append(kind)
        if agent_id is not None:
            clauses.append("agent_id=?"); params.append(agent_id)
        for col, val in (("experiment_id", experiment_id), ("sample_id", sample_id)):
            if val is None:
                continue
            if val == "none":
                clauses.append(f"{col} IS NULL")
            else:
                clauses.append(f"{col}=?"); params.append(val)
        if not include_archived:
            clauses.append("archived=0")
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            return [self._row(r) for r in conn.execute(q, params).fetchall()]

    # ── mutate ─────────────────────────────────────────────────────────
    def rename(self, conversation_id: str, title: str) -> bool:
        title = (title or "").strip()[:200] or "新对话"
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE conversation_id=?",
                (title, now, conversation_id))
            return cur.rowcount > 0

    def touch(self, conversation_id: str, *, preview: str | None = None,
              title: str | None = None) -> None:
        """Bump updated_at (+ optional preview/title) after a turn completes."""
        now = datetime.now().isoformat()
        sets = ["updated_at=?"]
        params: list = [now]
        if preview is not None:
            sets.append("last_message_preview=?")
            params.append((preview or "").strip().replace("\n", " ")[:160])
        if title is not None:
            sets.append("title=?")
            params.append((title or "").strip()[:200] or "新对话")
        params.append(conversation_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE conversations SET {', '.join(sets)} "
                         "WHERE conversation_id=?", params)

    def archive(self, conversation_id: str, on: bool = True) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET archived=? WHERE conversation_id=?",
                         (1 if on else 0, conversation_id))

    def delete(self, conversation_id: str, *, checkpointer=None) -> bool:
        """Delete a conversation row and (best-effort) purge its checkpoint thread.

        ``checkpointer`` (a langgraph saver) — if given and it exposes
        ``delete_thread``, the durable thread is purged so a deleted chat doesn't
        leave orphan checkpoint rows. Callers MUST guard against deleting a thread
        that is mid-run (see ConversationEngine).
        """
        row = self.get(conversation_id)
        if row is None:
            return False
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE conversation_id=?",
                         (conversation_id,))
            # Cascade the durable transcript so a deleted 群聊 leaves no orphans.
            conn.execute("DELETE FROM conversation_messages WHERE conversation_id=?",
                         (conversation_id,))
        if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
            try:
                checkpointer.delete_thread(row["thread_id"])
            except Exception as exc:  # noqa: BLE001 — orphan checkpoint is non-fatal
                logger.debug("delete_thread(%s) failed: %s", row["thread_id"], exc)
        return True

    # ── transcript (durable per-conversation messages) ─────────────────
    # Used by the 群聊 (group orchestrator) run-task bridge: each rendered SSE
    # entry is appended here so the conversation survives a tab switch / reload
    # (re-read on reconnect) and so an agent's 群聊 contributions surface in its
    # per-agent view. Append-only + best-effort: a transcript write must NEVER
    # break the live stream, so callers wrap these in try/except.
    # NB: the notifier is a module-level function (below the class) rather than a
    # method, so a store built in a test without an EventBus is unaffected.

    def append_message(self, conversation_id: str, *, kind: str = "message",
                       agent_id: str = "", role: str = "", text: str = "",
                       t: float | None = None, meta: str = "",
                       notify: bool = True) -> int:
        """Append one transcript entry; returns its per-conversation seq.

        ``kind`` discriminates how the entry renders on reconnect
        (operator | message | status | interrupt | compaction | done).
        ``agent_id`` is the owning agent for a per-agent ``message`` (the
        per-agent feed filters on it); empty for operator/status rows.

        ``compaction`` marks the point where the context
        middleware replaced a stretch of an agent's history with a summary. It
        is a real transcript row precisely BECAUSE the replay is otherwise
        indistinguishable from an unedited conversation — the operator must be
        able to see that the history above the marker is not what was said.

        seq allocation is ATOMIC: ``MAX(seq)+1`` is computed and inserted in a
        SINGLE statement under sqlite's write lock, so two concurrent writers on
        the same conversation (e.g. two run-task streams resuming one group
        thread) can never read the same MAX and collide on the (conversation_id,
        seq) primary key — which would silently drop a message. busy_timeout
        serializes the writers; a brief lock contention retries rather than raising.

        The conversation is trimmed to the most recent ``_MAX_TRANSCRIPT_ROWS``
        on write so an indefinitely-resumed 群聊 can't bloat the shared DB
        (seq is monotonic, so the threshold delete is a cheap indexed no-op until
        the cap is exceeded).

        ``notify=False`` skips the EventBus transcript notification — the ROW is
        written exactly the same, only the broadcast is withheld. It exists for a
        writer that has its OWN event type and would otherwise emit two events per
        row: 旁白 (``kind='narration'``, see ``chat/narration.py``) appends
        hundreds of rows in one long composite, and this bus replays only the last
        100 events, so the duplicate would evict hardware_state / anomaly /
        current_monitor from every reconnecting client's replay. Anything that
        appends a row a 群聊 viewer must see leaves this True.
        """
        import time as _t
        ts = float(t if t is not None else _t.time())
        params = (conversation_id, agent_id or "", role or "", kind or "message",
                  (text or "")[:8000], ts, (meta or "")[:64_000], conversation_id)
        for attempt in range(6):
            try:
                with self._connect() as conn:
                    cur = conn.execute(
                        "INSERT INTO conversation_messages "
                        "(conversation_id, seq, agent_id, role, kind, text, t, meta) "
                        "SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ?, ?, ?, ? "
                        "FROM conversation_messages WHERE conversation_id=? "
                        "RETURNING seq",
                        params)
                    row = cur.fetchone()
                    seq = int(row["seq"]) if row else 1
                    if seq > _MAX_TRANSCRIPT_ROWS:
                        conn.execute(
                            "DELETE FROM conversation_messages "
                            "WHERE conversation_id=? AND seq <= ?",
                            (conversation_id, seq - _MAX_TRANSCRIPT_ROWS))
                # Guarded HERE as well as inside the notifier. The invariant
                # ("a notification must never cost a message") belongs to this
                # call site: by this point the INSERT has committed, so an
                # exception escaping would hand the caller a failure for a row
                # that IS written — and a caller that retries then duplicates it.
                # The notifier's own try/except protects against a broken bus;
                # this protects against a broken notifier (a signature change,
                # an import raising before its try block is even entered).
                try:
                    if notify:
                        _publish_transcript_append(conversation_id, seq, kind,
                                                   agent_id, role, ts)
                except Exception:  # noqa: BLE001
                    logger.debug("transcript append notify raised", exc_info=True)
                return seq
            except sqlite3.OperationalError as exc:  # 'database is locked' under load
                if "locked" in str(exc).lower() and attempt < 5:
                    continue
                raise
        return 0  # pragma: no cover - loop always returns or raises

    def messages_for(self, conversation_id: str, *, limit: int = 4000) -> list[dict]:
        """The NEWEST ``limit`` transcript entries for one conversation, returned
        oldest-first — the reconnect-replay read. Reading newest-N (not oldest-N)
        matters on a long-lived resumed group chat: an operator reconnecting wants
        the latest activity, not the first ``limit`` rows with the tail dropped."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id, seq, agent_id, role, kind, text, t, meta "
                "FROM conversation_messages WHERE conversation_id=? "
                "ORDER BY seq DESC LIMIT ?", (conversation_id, int(limit))).fetchall()
        return [dict(r) for r in reversed(rows)]

    def messages_since(self, conversation_id: str, after_seq: int = 0,
                       *, limit: int = 5000) -> list[dict]:
        """``seq > after_seq`` 的转录条目，**最旧优先** —— 增量导出的读法。

        与 :meth:`messages_for` 相反：那个读最新 N 条（重连回放要的是最近活动），
        导出要的是"上次导到哪、之后有什么"。

        这个读法是文件夹能保住完整历史的原因：``append_message`` 超过
        ``_MAX_TRANSCRIPT_ROWS``(8000) 会裁掉最旧的行，而按 seq 增量追加导出的
        文件不会 —— 只要导出跑过一次，那段历史就永久留在实验文件夹里，哪怕 DB
        里已经没有了。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id, seq, agent_id, role, kind, text, t, meta "
                "FROM conversation_messages WHERE conversation_id=? AND seq>? "
                "ORDER BY seq ASC LIMIT ?",
                (conversation_id, int(after_seq), int(limit))).fetchall()
        return [dict(r) for r in rows]

    def updated_since(self, iso_ts: str, *, limit: int = 200) -> list[dict]:
        """``updated_at > iso_ts`` 的会话（走 idx_conv_updated 索引）。

        导出 tick 用它决定"这一轮有没有东西需要导" —— 一条索引查询，
        没变化时开销可忽略。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE updated_at > ? "
                "ORDER BY updated_at ASC LIMIT ?", (str(iso_ts or ""), int(limit))
            ).fetchall()
        return [self._row(r) for r in rows]

    def agent_activity(self, agent_id: str, *, limit: int = 200) -> list[dict]:
        """An agent's 群聊 message contributions across ALL group runs, newest
        first, enriched with the owning conversation's title. Powers the per-agent
        view's read-only "群聊中的活动" feed (the bridge that was missing)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT m.conversation_id AS conversation_id, m.seq AS seq, "
                "m.agent_id AS agent_id, m.role AS role, m.kind AS kind, "
                "m.text AS text, m.t AS t, m.meta AS meta, "
                "c.title AS conversation_title "
                "FROM conversation_messages m "
                "LEFT JOIN conversations c "
                "  ON c.conversation_id = m.conversation_id "
                "WHERE m.agent_id=? AND m.kind='message' "
                "ORDER BY m.t DESC, m.seq DESC LIMIT ?",
                (agent_id, int(limit))).fetchall()
        return [dict(r) for r in rows]


def _publish_transcript_append(conversation_id: str, seq: int, kind: str,
                               agent_id: str, role: str, t: float) -> None:
    """Tell the EventBus a transcript row landed, so other viewers can catch up.

    「多个电脑上打开远程窗口时，对话不同步显示」. The transcript
    was read once on mount and then only on an explicit 「刷新进展」 click, so a
    second machine watching the same group chat simply never saw new messages.

    Deliberately carries **no text** — only the cursor (conversation + seq) and
    enough shape to decide whether this viewer cares. A client that does gets the
    row through the existing authenticated read; putting the body on a broadcast
    bus would widen who can see a conversation's content for no benefit.

    Best-effort in every direction: no bus, a bus that raises, an import failure
    — none of them may break a transcript write. A dropped notification costs a
    late refresh; a raised one would cost the message.
    """
    try:
        from mast.core.events import Event, EventBus, EventType

        EventBus.get().publish(Event(
            type=EventType.EXPERIMENT_UPDATE,
            data={"scope": "transcript", "conversation_id": conversation_id,
                  "seq": int(seq), "kind": kind or "message",
                  "agent_id": agent_id or "", "role": role or "", "t": float(t)},
        ))
    except Exception:  # noqa: BLE001 — a notification must never cost a message
        logger.debug("transcript append notify failed", exc_info=True)


def publish_private_turn_finished(conversation_id: str, agent_id: str) -> None:
    """一次**私聊**（一对一对话）的回合写完了，告诉别的窗口去重读。

    为什么私聊要单独一条：``_publish_transcript_append`` 挂在
    :meth:`ConversationStore.append_message` 上，而私聊的正文根本不走这张表 ——
    它落在 LangGraph 的 checkpointer 里（``/agents/{id}/messages`` 读的就是那里）。
    于是群聊转录有推送、私聊一条都没有，前端那一页只在**自己**发完消息之后才刷新。
    「智能体代理对话不会自动刷新」。

    ``scope`` 刻意不是 ``"transcript"``：``RunTaskPanel`` 用
    ``d.scope !== "transcript"`` 做第一道过滤，借用那个值会让它把私聊回合
    当成群聊游标去处理。一个新语义就配一个新名字，别挤进旧的那个。

    与 :func:`_publish_transcript_append` 同样**不带正文**，只带「哪个会话动了」——
    要看内容的客户端走它自己那条已鉴权的读端点。

    Best-effort：没有 bus / bus 抛了 / import 失败，一条都不许把回合毁掉。
    掉一次通知的代价是晚几秒刷新（会话列表那条 6 s 轮询是兜底），
    抛一次的代价是用户的那轮对话。
    """
    try:
        from mast.core.events import Event, EventBus, EventType

        EventBus.get().publish(Event(
            type=EventType.EXPERIMENT_UPDATE,
            data={"scope": "private_turn",
                  "conversation_id": str(conversation_id or ""),
                  "agent_id": str(agent_id or "")},
        ))
    except Exception:  # noqa: BLE001 — a notification must never cost a turn
        logger.debug("private turn notify failed", exc_info=True)


__all__ = ["ConversationStore", "KINDS", "publish_private_turn_finished"]
