"""``chat_messages`` —— 结构化消息正文的持久层（对话历史的新家）。

为什么需要它
------------
今天私聊的正文**只**活在 LangGraph 的 checkpointer 里（``chat/store.py`` 自述：
「一个会话就是一个稳定的 thread_id 加一点展示用元数据，真正的历史活在 checkpointer
里」）。这条设计已经付过两次代价：

* **通知断链**：群聊转录落库时 ``append_message`` 会广播游标，私聊那条路上一个通知
  都发不出去——因为 ``conversation_messages`` 表里根本没有私聊正文；
* **唯一副本**：换掉 checkpointer 历史即归零（``tests/v2/agents/contract/
  test_checkpoint_is_the_message_store.py`` 钉着这条）。真机上有数月的会话。

与 ``conversation_messages`` 的分工（两张表并存，不是取代）
----------------------------------------------------------
============================  ==========================================
``conversation_messages``     **给人看的**：UI 渲染、重连回放、跨窗口推送。
（``chat/store.py``）         一行一帧，文本已经渲染过，8000 行上限，旧的会被裁掉。
``chat_messages``（本模块）   **给模型看的**：结构化正文，tool_calls / tool 结果 /
                              thinking / reasoning_content 一个不丢。**只追加，
                              永不裁剪**。
============================  ==========================================

同一件事在两张表里各存一份不是冗余：一个可以为了显示而摘要、裁剪、重写，另一个不行。
把它们合成一张表的每一次尝试，最后都会在「转录要裁剪」与「上下文不能丢」之间二选一。

三条不变式
----------
1. **只追加**。压缩不删行——插一条 ``kind='compaction_summary'``、meta 记
   ``covers_seq``，工作集从最后一条摘要往后取。全量历史永远在盘上。
   （对照：checkpointer 每个 super-step **重写整个通道值**，追加行严格更省。）
2. **seq 原子分配**。``MAX(seq)+1`` 与 INSERT 在同一条语句里，两个并发写者不可能
   读到同一个 MAX 而撞主键——那会**静默丢一条消息**。抄 ``append_message`` 的形状。
3. **坏行不抛**。聊天必须继续工作。读到一行反序列化不了就跳过它并记一条 debug，
   不能让一条脏数据把整个会话变成打不开的。

⚠️ ``reasoning_content`` 必须原样回存（放在 ``meta``）。Kimi/DeepSeek/Qwen/GLM 的
多轮 tool_call 在第二轮会因为它缺失而 400（``thinking is enabled but
reasoning_content is missing``）——今天靠 ``reasoning_chat_model.py`` 的补丁在模型层
兜着，二期换自研 provider client 时序列化层要直接负责。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: 一条消息正文的上限。远大于 ``conversation_messages`` 的 8000 字符——那张表存的是
#: 渲染后的摘要，这张存的是模型真正读到的东西，截断它等于篡改上下文。
_MAX_CONTENT = 1_000_000

#: 行类型。``message`` 是普通消息；``compaction_summary`` 是压缩摘要（它**替代**
#: 一段历史进入模型上下文，但不删除那段历史）。
KINDS = ("message", "compaction_summary")

ROLES = ("system", "user", "assistant", "tool")


class MessageStore:
    """``chat_messages`` 表 —— 与 :class:`~mast.chat.store.ConversationStore`
    共用同一个 SQLite 文件（``mast_conversations.db``）。

    共用文件而不是共用表：两张表的生命周期规则相反（一个裁剪、一个不裁），但它们
    总是被一起备份、一起删除、一起迁移，分成两个文件只会让「删了会话正文还在」这类
    不一致多一个发生的地方。
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    @classmethod
    def from_conversation_store(cls, store) -> "MessageStore":
        """建在 ``ConversationStore`` 的同一个 DB 文件上。

        取的是它的私有 ``_db_path``：这是**刻意**的耦合——两张表必须在一个文件里，
        让调用方各自算一遍路径，迟早会有一处算错而把正文写进另一个文件。
        """
        return cls(Path(getattr(store, "_db_path")))

    # ── plumbing ───────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        # 与 ConversationStore 同值：并发写者排队而不是 fail-fast 抛
        # 'database is locked'，但有界，卡住的写者不能永远挂住一个请求。
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    conversation_id TEXT NOT NULL,
                    seq             INTEGER NOT NULL,
                    role            TEXT NOT NULL DEFAULT 'assistant',
                    -- 内容块列表的 JSON：[{"type":"text","text":…},
                    -- {"type":"thinking",…}, {"type":"image_ref","path":…}]。
                    -- 存块而不是纯字符串，因为 thinking 与图片引用都必须能原样回放。
                    content         TEXT NOT NULL DEFAULT '[]',
                    -- assistant 行的工具调用：[{"id","name","args"}]
                    tool_calls      TEXT NOT NULL DEFAULT '',
                    -- tool 行与 assistant 行配对用；孤儿 tool 结果会让 provider 400
                    tool_call_id    TEXT NOT NULL DEFAULT '',
                    name            TEXT NOT NULL DEFAULT '',
                    kind            TEXT NOT NULL DEFAULT 'message',
                    -- JSON：model_id / usage / reasoning_content / 时钟 /
                    -- 压缩的 covers_seq。**reasoning_content 丢了会让第二轮 400。**
                    meta            TEXT NOT NULL DEFAULT '',
                    created_at      REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (conversation_id, seq)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chatmsg_kind "
                         "ON chat_messages(conversation_id, kind, seq)")

    # ── write ──────────────────────────────────────────────────────────
    def append(self, conversation_id: str, *, role: str = "assistant",
               content: Any = "", tool_calls: Any = None, tool_call_id: str = "",
               name: str = "", kind: str = "message", meta: Any = None,
               created_at: float | None = None) -> int:
        """追加一行，返回它的 per-conversation ``seq``（失败返回 0）。

        ``content`` 接受字符串或内容块列表；字符串会被包成一个 text 块，这样读端
        只需要处理一种形状。
        """
        rows = self.append_many(conversation_id, [{
            "role": role, "content": content, "tool_calls": tool_calls,
            "tool_call_id": tool_call_id, "name": name, "kind": kind,
            "meta": meta, "created_at": created_at,
        }])
        return rows[-1] if rows else 0

    def append_many(self, conversation_id: str,
                    messages: Iterable[dict]) -> list[int]:
        """批量追加（一个回合的全部新消息）。返回分配到的 seq 列表。

        逐条 INSERT 而不是 executemany：seq 是 ``MAX(seq)+1`` 单语句分配的，
        executemany 会让同一批里的后几条读到同一个 MAX。
        """
        out: list[int] = []
        payloads = [self._row_params(conversation_id, m) for m in messages]
        if not payloads:
            return out
        for params in payloads:
            seq = self._insert_one(params)
            if seq:
                out.append(seq)
        return out

    def _row_params(self, conversation_id: str, m: dict) -> tuple:
        content = m.get("content", "")
        blocks = content if isinstance(content, list) else (
            [{"type": "text", "text": str(content)}] if str(content) else [])
        tc = m.get("tool_calls") or None
        meta = m.get("meta") or None
        ts = m.get("created_at")
        role = str(m.get("role") or "assistant")
        kind = str(m.get("kind") or "message")
        return (
            conversation_id,
            role if role in ROLES else "assistant",
            _dumps(blocks)[:_MAX_CONTENT],
            _dumps(tc) if tc else "",
            str(m.get("tool_call_id") or "")[:200],
            str(m.get("name") or "")[:200],
            kind if kind in KINDS else "message",
            _dumps(meta) if meta else "",
            float(ts if ts is not None else time.time()),
            conversation_id,
        )

    def _insert_one(self, params: tuple) -> int:
        for _attempt in range(6):
            try:
                with self._connect() as conn:
                    cur = conn.execute(
                        "INSERT INTO chat_messages (conversation_id, seq, role, "
                        "content, tool_calls, tool_call_id, name, kind, meta, "
                        "created_at) "
                        "SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ?, ?, ?, ?, ?, ? "
                        "FROM chat_messages WHERE conversation_id=? "
                        "RETURNING seq",
                        params)
                    row = cur.fetchone()
                    return int(row["seq"]) if row else 0
            except sqlite3.OperationalError as exc:  # locked / busy
                logger.debug("chat_messages insert retry: %s", exc)
                time.sleep(0.05)
            except Exception as exc:  # noqa: BLE001 — 聊天不能因为落库失败而中断
                logger.warning("chat_messages insert failed: %s", exc)
                return 0
        logger.warning("chat_messages insert gave up after retries")
        return 0

    def mark_compaction(self, conversation_id: str, *, summary: str,
                        covers_from: int, covers_to: int,
                        meta: dict | None = None) -> int:
        """记一条压缩摘要——**不删除**被它覆盖的那些行。

        这是本模块与 checkpointer 最重要的一处分歧。LangGraph 的压缩靠
        ``RemoveMessage(REMOVE_ALL_MESSAGES)`` 把历史从通道里真的删掉；那条通道
        又是唯一副本，于是「压缩」等于「销毁」。这里压缩只是移动读游标。
        """
        payload = dict(meta or {})
        payload["covers_seq"] = [int(covers_from), int(covers_to)]
        return self.append(conversation_id, role="system", content=summary,
                           kind="compaction_summary", meta=payload)

    # ── read ───────────────────────────────────────────────────────────
    def load(self, conversation_id: str, *, limit: int | None = None) -> list[dict]:
        """全量历史（含被压缩覆盖的部分），按 seq 升序。"""
        sql = ("SELECT * FROM chat_messages WHERE conversation_id=? "
               "ORDER BY seq ASC")
        args: tuple = (conversation_id,)
        if limit:
            sql += " LIMIT ?"
            args = (conversation_id, int(limit))
        return self._query(sql, args)

    def load_working_set(self, conversation_id: str) -> list[dict]:
        """**模型这一轮真正该看到的**：最后一条压缩摘要 + 其后的全部行。

        没有摘要就是全量。这个函数替代的是 ``graph.get_state(cfg).values
        ["messages"]`` —— 续聊因此不再是「从 checkpoint 重放」，而是「读一段行」。
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT MAX(seq) AS s FROM chat_messages "
                    "WHERE conversation_id=? AND kind='compaction_summary'",
                    (conversation_id,)).fetchone()
                cutoff = int(row["s"]) if row and row["s"] is not None else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("load_working_set cutoff failed: %s", exc)
            cutoff = 0
        return self._query(
            "SELECT * FROM chat_messages WHERE conversation_id=? AND seq >= ? "
            "ORDER BY seq ASC", (conversation_id, cutoff))

    def _query(self, sql: str, args: tuple) -> list[dict]:
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, args).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_messages query failed: %s", exc)
            return []
        out: list[dict] = []
        for r in rows:
            try:
                out.append(_row_to_dict(r))
            except Exception as exc:  # noqa: BLE001 — 一行坏了不该毁掉整个会话
                logger.debug("skipping unreadable chat_messages row: %s", exc)
        return out

    def count(self, conversation_id: str) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM chat_messages WHERE conversation_id=?",
                    (conversation_id,)).fetchone()
                return int(row["n"]) if row else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_messages count failed: %s", exc)
            return 0

    def max_seq(self, conversation_id: str) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS s FROM chat_messages "
                    "WHERE conversation_id=?", (conversation_id,)).fetchone()
                return int(row["s"]) if row else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_messages max_seq failed: %s", exc)
            return 0

    def conversation_ids(self) -> list[str]:
        try:
            with self._connect() as conn:
                return [r["conversation_id"] for r in conn.execute(
                    "SELECT DISTINCT conversation_id FROM chat_messages").fetchall()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_messages conversation_ids failed: %s", exc)
            return []

    def source_keys(self, conversation_id: str) -> set[str]:
        """已镜像进来的消息身份集合（见 :func:`source_key`）。

        用身份而不是行数来判断「哪些是新的」：checkpointer 里的消息列表会**变短**
        （压缩把一段历史换成一条摘要），按位置比对会在压缩之后把整段历史重新镜像
        一遍。这是 dedupe 必须按身份做的全部理由。
        """
        out: set[str] = set()
        for row in self.load(conversation_id):
            key = (row.get("meta") or {}).get("src_key")
            if key:
                out.add(str(key))
        return out

    def delete(self, conversation_id: str) -> int:
        """删掉一个会话的全部正文。返回删除行数。

        ⚠️ 与 ``ConversationStore.delete`` 不同，这里**不需要**调用方额外传什么
        东西——正文与身份在同一个 DB，删除是一条语句。那条「忘了传 checkpointer
        就留下孤儿正文」的老缝在这个模型下不存在。
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM chat_messages WHERE conversation_id=?",
                    (conversation_id,))
                return int(cur.rowcount or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_messages delete failed: %s", exc)
            return 0


# ── serialisation: LangChain 消息 ↔ 行 ────────────────────────────────
#
# 一期这两个函数是 langchain 消息与行之间的桥；二期换自研消息 dataclass 时**只改
# 这两个函数，表结构一个字不动** —— schema 刻意是 provider 中立的 JSON。

def _dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return ""


def _loads(raw: str, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return fallback


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "seq": int(row["seq"]),
        "role": row["role"],
        "content": _loads(row["content"], []),
        "tool_calls": _loads(row["tool_calls"], []) or [],
        "tool_call_id": row["tool_call_id"] or "",
        "name": row["name"] or "",
        "kind": row["kind"] or "message",
        "meta": _loads(row["meta"], {}) or {},
        "created_at": float(row["created_at"] or 0.0),
    }


def message_to_row(msg: Any) -> dict:
    """一条 LangChain 消息 → 可以喂给 :meth:`MessageStore.append` 的 dict。

    ``reasoning_content`` 从 ``additional_kwargs`` / ``response_metadata`` 两处捞
    ——不同 provider 放的位置不一样，而它缺失会让下一轮 400。
    """
    role = {"human": "user", "ai": "assistant", "tool": "tool",
            "system": "system"}.get(getattr(msg, "type", ""), "assistant")

    content = getattr(msg, "content", "")
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}] if content else []
    elif isinstance(content, list):
        blocks = [b if isinstance(b, dict) else {"type": "text", "text": str(b)}
                  for b in content]
    else:  # pragma: no cover — provider 返回了没见过的形状
        blocks = [{"type": "text", "text": str(content)}]

    ak = getattr(msg, "additional_kwargs", None) or {}
    rm = getattr(msg, "response_metadata", None) or {}
    meta: dict = {}
    for key in ("reasoning_content", "reasoning"):
        val = ak.get(key) or rm.get(key)
        if val:
            meta["reasoning_content"] = val
            break
    if getattr(msg, "id", None):
        meta["src_id"] = msg.id
    for key in ("model_name", "model", "finish_reason"):
        if rm.get(key):
            meta.setdefault("model", {})[key] = rm[key]
    if getattr(msg, "usage_metadata", None):
        meta["usage"] = msg.usage_metadata
    # 压缩摘要在 additional_kwargs 上盖了章 —— 迁移进来的历史要保住这个标记，
    # 否则「这段是摘要不是原话」的信息就没了。
    if ak.get("mast_compaction"):
        meta["mast_compaction"] = ak["mast_compaction"]

    return {
        "role": role,
        "content": blocks,
        "tool_calls": list(getattr(msg, "tool_calls", None) or []),
        "tool_call_id": getattr(msg, "tool_call_id", "") or "",
        "name": getattr(msg, "name", "") or "",
        "kind": ("compaction_summary" if ak.get("mast_compaction") else "message"),
        "meta": meta,
    }


def row_to_message(row: dict):
    """行 → LangChain 消息。反序列化不了就抛，由 ``_query`` 的调用方跳过该行。"""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    blocks = row.get("content") or []
    text = "".join(b.get("text", "") for b in blocks
                   if isinstance(b, dict) and b.get("type") == "text")
    meta = row.get("meta") or {}
    extra: dict = {}
    if meta.get("reasoning_content"):
        extra["reasoning_content"] = meta["reasoning_content"]
    if meta.get("mast_compaction"):
        extra["mast_compaction"] = meta["mast_compaction"]

    role = row.get("role")
    if role == "user":
        return HumanMessage(content=text)
    if role == "system":
        return SystemMessage(content=text, additional_kwargs=extra)
    if role == "tool":
        return ToolMessage(content=text, tool_call_id=row.get("tool_call_id") or "",
                           name=row.get("name") or "")
    return AIMessage(content=text, tool_calls=list(row.get("tool_calls") or []),
                     additional_kwargs=extra)


def rows_to_messages(rows: Iterable[dict]) -> list:
    """一串行 → 一串消息，坏行跳过（聊天必须继续工作）。"""
    out = []
    for r in rows:
        try:
            out.append(row_to_message(r))
        except Exception as exc:  # noqa: BLE001
            logger.debug("skipping unconvertible row %s: %s", r.get("seq"), exc)
    return out


def source_key(msg: Any) -> str:
    """一条消息的稳定身份，用于「这条镜像过没有」。

    优先用 LangChain 自己的 ``id``（图里流转的消息都有）。没有 id 的（手工构造、
    从元组升格、旧 checkpoint 里的老消息）退回内容指纹——指纹带上 role 与
    ``tool_call_id``，因为「同一句话由 user 说和由 assistant 说」是两条消息，
    而两条内容相同的 tool 结果靠配对 id 区分。
    """
    mid = getattr(msg, "id", None)
    if mid:
        return f"id:{mid}"
    import hashlib

    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        content = _dumps(content)
    raw = "\x1f".join([
        str(getattr(msg, "type", "")),
        str(getattr(msg, "tool_call_id", "") or ""),
        _dumps(list(getattr(msg, "tool_calls", None) or [])),
        content,
    ])
    return "h:" + hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:24]


def mirror(store: "MessageStore", conversation_id: str,
           messages: Iterable[Any]) -> int:
    """把一段 LangChain 消息**增量**镜像进 ``chat_messages``，返回新写入的条数。

    这是 strangler 迁移第 0 步的双写入口：checkpointer 仍是读路径的真源，这里只
    往新家复制一份。幂等——重复调用不会重复写。

    **不抛**。镜像失败最坏是「新家少一段历史」，而抛出去会毁掉一轮对话；两者不是
    一个量级。失败会记 warning，导入账本可以事后补。
    """
    try:
        seen = store.source_keys(conversation_id)
        rows: list[dict] = []
        for msg in messages:
            key = source_key(msg)
            if key in seen:
                continue
            seen.add(key)          # 同一批里出现两条一样的也只写一条
            row = message_to_row(msg)
            row.setdefault("meta", {})
            row["meta"]["src_key"] = key
            rows.append(row)
        if not rows:
            return 0
        return len(store.append_many(conversation_id, rows))
    except Exception as exc:  # noqa: BLE001 — 双写绝不许毁掉一轮对话
        logger.warning("mirror into chat_messages failed for %s: %s",
                       conversation_id, exc)
        return 0


__all__ = [
    "MessageStore",
    "message_to_row",
    "row_to_message",
    "rows_to_messages",
    "source_key",
    "mirror",
    "KINDS",
    "ROLES",
]
