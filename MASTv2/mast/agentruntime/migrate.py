"""checkpoint → ``chat_messages`` 的一次性幂等导入。

为什么这个工具的**发布时机**是硬约束
------------------------------------
反序列化 checkpoint 需要 langgraph 的序列化器在场。所以：

    **导入必须在「requirements 里删掉 langgraph」之前的版本里发布并跑完。**

这就是本模块要检查的东西——它要确认真机与开发机的每一个 thread 都
导入完成，才允许动依赖。顺序搞反的代价是：真机上数月的私聊历史再也读不出来，因为
唯一能解读那份数据的库已经被卸了。

三条纪律
--------
1. **绝不写旧库**。以 ``file:…?mode=ro`` 只读 URI 打开 checkpoint 文件。旧库也
   **永不自动删除**——账本全绿且稳定两个版本之后，由用户手动归档。真机上那是数月
   的会话，而文件很小。
2. **单 thread 失败隔离**。一个坏 checkpoint 不能拖垮整批；失败记进账本，可单独重试。
3. **失败不阻塞启动**。导入是启动钩子，出问题最坏是「新家少一段更早的历史」——那时
   读路径还在 checkpointer 上，有数周窗口对账修复。

顺带做的一件事：**孤儿 tool 结果清洗**
--------------------------------------
旧 checkpoint 里躺着一批没有配对 AIMessage 的 ``ToolMessage``——``Command(graph=
Command.PARENT)`` 短路子图时，发起交棒的那条 AIMessage 留在子图命名空间没有传播，
只有 ToolMessage 进了父通道。今天靠 ``ToolPairGuardMiddleware`` 在送给 provider 之前
把它们剥掉；导入是那段救援逻辑**最后一次上岗**：孤儿在这里被降级成普通文本消息，
新家里不会再有孤儿。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from mast.agentruntime.persist import MessageStore, message_to_row, source_key

logger = logging.getLogger(__name__)

#: 导入来源标记。每个被导入的会话在正文最前面得到一行，说明这段历史是从
#: checkpoint 搬过来的。**回放必须可见、不可伪装** —— 与压缩标记同一条原则：
#: 一段被搬运过的历史，读的人有权知道它被搬运过。
IMPORT_MARKER_KIND = "message"
IMPORT_MARKER_TEXT = "（以下历史自 LangGraph checkpoint 导入）"


class MigrationLedger:
    """per-thread 导入账本 —— 幂等判据，也是「什么时候可以删旧库」的客观依据。"""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_message_migration (
                    thread_id       TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL DEFAULT '',
                    status          TEXT NOT NULL DEFAULT 'pending',
                    imported        INTEGER NOT NULL DEFAULT 0,
                    note            TEXT NOT NULL DEFAULT '',
                    updated_at      REAL NOT NULL DEFAULT 0
                )
            """)

    def mark(self, thread_id: str, *, conversation_id: str = "", status: str,
             imported: int = 0, note: str = "") -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO chat_message_migration (thread_id, conversation_id,"
                    " status, imported, note, updated_at) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(thread_id) DO UPDATE SET conversation_id=excluded."
                    "conversation_id, status=excluded.status, imported=excluded."
                    "imported, note=excluded.note, updated_at=excluded.updated_at",
                    (thread_id, conversation_id, status, int(imported),
                     note[:500], time.time()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("migration ledger write failed for %s: %s", thread_id, exc)

    def status_of(self, thread_id: str) -> str:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM chat_message_migration WHERE thread_id=?",
                    (thread_id,)).fetchone()
                return str(row["status"]) if row else ""
        except Exception:  # noqa: BLE001
            return ""

    def summary(self) -> dict:
        """账本汇总——删除闸门读的就是它。"""
        out = {"done": 0, "failed": 0, "skipped": 0, "pending": 0, "imported": 0}
        try:
            with self._connect() as conn:
                for row in conn.execute(
                        "SELECT status, COUNT(*) AS n, SUM(imported) AS m "
                        "FROM chat_message_migration GROUP BY status").fetchall():
                    out[str(row["status"])] = int(row["n"])
                    out["imported"] += int(row["m"] or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("migration ledger summary failed: %s", exc)
        return out

    def is_complete(self) -> bool:
        """没有 pending / failed 才算完成。**空账本不算完成**——「一次都没跑」和
        「跑完了没有可导的」在这里必须分开，否则删除闸门会被一个从未运行过的
        账本放行。"""
        s = self.summary()
        if s["failed"] or s["pending"]:
            return False
        return (s["done"] + s["skipped"]) > 0


# ── checkpoint 读取（只读，绝不写） ───────────────────────────────────
def _thread_ids(checkpoint_db: Path) -> list[str]:
    """列出 checkpoint 库里的全部 thread_id。只读打开。"""
    uri = f"file:{checkpoint_db.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints").fetchall()
            return [str(r["thread_id"]) for r in rows if r["thread_id"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("cannot enumerate checkpoint threads in %s: %s",
                       checkpoint_db, exc)
        return []


def _messages_for_thread(checkpoint_db: Path, thread_id: str) -> list:
    """读一个 thread 最新 checkpoint 里的 ``messages`` 通道。

    用 langgraph 自己的 ``SqliteSaver`` 反序列化——通道值是它的私有序列化格式，
    手工解析会在第一个非平凡消息上出错。连接是只读的。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    uri = f"file:{checkpoint_db.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        if tup is None:
            return []
        values = getattr(tup, "checkpoint", None) or {}
        channels = values.get("channel_values") or {}
        return list(channels.get("messages") or [])
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def drop_orphan_tool_results(messages: list) -> tuple[list, int]:
    """把没有配对 AIMessage 的 ``ToolMessage`` 降级成普通文本消息。

    ``ToolPairGuardMiddleware`` 救援逻辑的最后一次上岗（见模块 docstring）。降级而
    不是删除：那些 tool 结果里有真实的实验产出，删掉等于凭空少一段历史；而保留成
    ``ToolMessage`` 会让新家继续持有一颗对 provider 的定时炸弹。
    """
    known: set[str] = set()
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            tid = tc.get("id") if isinstance(tc, dict) else None
            if tid:
                known.add(str(tid))

    from langchain_core.messages import AIMessage

    out, fixed = [], 0
    for m in messages:
        if getattr(m, "type", "") == "tool":
            tid = str(getattr(m, "tool_call_id", "") or "")
            if tid not in known:
                content = getattr(m, "content", "")
                name = getattr(m, "name", "") or "tool"
                out.append(AIMessage(
                    content=f"[{name} 的结果（导入时缺少配对的调用记录）]\n{content}"))
                fixed += 1
                continue
        out.append(m)
    return out, fixed


# ── 导入 ──────────────────────────────────────────────────────────────
def import_thread(*, checkpoint_db: Path, thread_id: str, conversation_id: str,
                  message_store: MessageStore, ledger: MigrationLedger) -> int:
    """导入一个 thread。返回新写入的行数。**不抛**——失败记账本。"""
    if ledger.status_of(thread_id) == "done":
        return 0
    try:
        msgs = _messages_for_thread(checkpoint_db, thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("read checkpoint thread %s failed: %s", thread_id, exc)
        ledger.mark(thread_id, conversation_id=conversation_id, status="failed",
                    note=f"read: {type(exc).__name__}: {exc}")
        return 0

    if not msgs:
        ledger.mark(thread_id, conversation_id=conversation_id, status="skipped",
                    note="checkpoint 里没有 messages 通道")
        return 0

    try:
        msgs, orphans = drop_orphan_tool_results(msgs)
        seen = message_store.source_keys(conversation_id)
        rows: list[dict] = []

        # 顺序哨兵：新家里已经有**不是导入来的**行，说明双写先于导入跑了。
        # 那样搬进来的老历史会排在今天的新消息后面（seq 单调追加），模型读到的
        # 会是一段前后颠倒的对话。生产上这不该发生 —— 导入在
        # ``_build_chat_engine`` 里、在能够双写的引擎被构造出来**之前**执行，
        # 顺序由构造保证。留这条日志，是因为一旦有人调换了那两行，症状（模型
        # 偶尔答非所问）离原因太远，没人会想到是这里。
        if message_store.count(conversation_id) > 0 and not any(
                (r.get("meta") or {}).get("imported_from")
                for r in message_store.load(conversation_id)):
            logger.warning(
                "conversation %s already has non-imported rows — the dual-write "
                "ran BEFORE the import, so imported history will land after "
                "today's messages. Check _build_chat_engine's ordering.",
                conversation_id)

        # 导入标记：只在这个会话的新家还完全是空的时候插一次。
        if not seen and message_store.count(conversation_id) == 0:
            rows.append({
                "role": "system", "content": IMPORT_MARKER_TEXT,
                "kind": IMPORT_MARKER_KIND,
                "meta": {"imported_from": "langgraph_checkpoint",
                         "thread_id": thread_id, "src_key": f"import:{thread_id}"},
            })

        for m in msgs:
            key = source_key(m)
            if key in seen:
                continue
            seen.add(key)
            row = message_to_row(m)
            row.setdefault("meta", {})
            row["meta"]["src_key"] = key
            row["meta"]["imported_from"] = "langgraph_checkpoint"
            rows.append(row)

        written = len(message_store.append_many(conversation_id, rows)) if rows else 0
        ledger.mark(thread_id, conversation_id=conversation_id, status="done",
                    imported=written,
                    note=(f"orphan tool results repaired: {orphans}" if orphans
                          else ""))
        return written
    except Exception as exc:  # noqa: BLE001
        logger.warning("import thread %s failed: %s", thread_id, exc)
        ledger.mark(thread_id, conversation_id=conversation_id, status="failed",
                    note=f"write: {type(exc).__name__}: {exc}")
        return 0


def import_all(*, checkpoint_db: str | Path, conversation_store,
               message_store: MessageStore | None = None,
               ledger: MigrationLedger | None = None) -> dict:
    """把 checkpoint 库里所有**认得出会话的** thread 导进新家。

    认不出会话的 thread（会话已删、CLI 的 ``cli-1``、后台 run 的 ``bg-*``）记
    ``skipped``：它们没有归属，导进来会变成一堆无主正文。

    返回账本汇总。**不抛**——这是启动钩子，导入失败不能挡住服务起来。
    """
    ckpt = Path(checkpoint_db)
    message_store = message_store or MessageStore.from_conversation_store(
        conversation_store)
    ledger = ledger or MigrationLedger(Path(getattr(conversation_store, "_db_path")))

    if not ckpt.is_file():
        logger.info("no checkpoint db at %s — nothing to import", ckpt)
        return ledger.summary()

    # thread_id → conversation_id（会话表是权威：一个 thread 只属于一个会话）
    by_thread: dict[str, str] = {}
    try:
        for kind in ("private", "group"):
            for conv in (conversation_store.list(kind=kind) or []):
                tid = conv.get("thread_id")
                if tid:
                    by_thread[str(tid)] = str(conv["conversation_id"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("cannot list conversations for import: %s", exc)
        return ledger.summary()

    for thread_id in _thread_ids(ckpt):
        cid = by_thread.get(thread_id)
        if not cid:
            if ledger.status_of(thread_id) != "skipped":
                ledger.mark(thread_id, status="skipped",
                            note="没有对应的会话行（已删 / CLI / 后台 run）")
            continue
        import_thread(checkpoint_db=ckpt, thread_id=thread_id,
                      conversation_id=cid, message_store=message_store,
                      ledger=ledger)

    summary = ledger.summary()
    logger.info("chat history import: %s", json.dumps(summary, ensure_ascii=False))
    return summary


__all__ = [
    "MigrationLedger",
    "import_all",
    "import_thread",
    "drop_orphan_tool_results",
    "IMPORT_MARKER_TEXT",
]
