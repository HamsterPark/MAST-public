"""ConductStore —— conduct 的**进度真源**(SQLite 三张表)。

设计:``docs/v2/design/campaign_director_design.md`` §4.7 / §6。
持久化分工照 ``planning/plan_store.py``:SQLite 是真源,实验文件夹
(``conduct/<id>/spec_vNNN.md`` + ``progress.jsonl``)是人读视图,恢复**不读它**。
文件夹渲染在 :mod:`mast.conduct.journal`,由本模块的 ``observer`` 单点驱动
(见下面第四条)。

## 三张表,三种东西

* ``conducts`` —— **状态**,一行一个 conduct。核心字段**单写者 = Director**。
* ``conduct_events`` —— **审计流**,append-only,永不 UPDATE。
* ``conduct_ops`` —— **意图队列**,API 线程往里写,Director 线程消费。
  两个线程都要动状态,而 SQLite 不会替我们仲裁语义 —— 队列就是那个仲裁。

## 三条结构纪律(不是约定,是代码形状)

**一、状态改动只有一扇门。** :meth:`ConductStore.record` 同时写事件和改字段,
同一个事务。没有 ``set_status()`` 这种裸方法 —— 有的话,总有一次改动会不留痕,
而事后对账时「状态是怎么变成这样的」就永远答不上来。

**二、单活跃不变式由数据库执行。** ``active_slot`` 非终态=1、终态=NULL,加
``UNIQUE(active_slot)``(SQLite 的 UNIQUE 允许多个 NULL)。所以「同时两个
conduct 在跑」不是靠调用方检查,是 INSERT 直接失败。
**推论(照设计 §4.7 字面实现,值得知道)**:``draft`` 也是非终态,所以一个未
了结的 conduct 在场时连第二份草稿都建不了。要建就先把旧的 abort 掉 ——
「明拒」优于「静默并发」(§10-3)。

**三、异常态必须带理由。** 转入 ``halted_estop`` / ``waiting_*`` / ``paused``
/ ``yielding`` / ``recovery_pending`` / ``aborted`` 而 ``status_reason`` 为空
⇒ 直接抛。一个没有「为什么」的暂停,人只能去猜或者重启进程 —— 这正是
「能挂不能解」这类缺陷的一半。

**四、外溢也只有一扇门。** 人读副本(``progress.jsonl``)与 WS 状态帧都挂在
:meth:`ConductStore.record` **提交之后**的 ``observer`` 上,而不是散在
Director 的几十个调用点上。理由是本仓「每页各自记得」那类缺陷:靠人肉在 N 个
调用点各加一行,漏掉的那几个要等到真机上才发现。挂在这里,**凡是改了状态的
都必然外溢**,漏不掉。

观察者在事务提交后才被调用(触发 refetch 的客户端一定读得到那一行),而且
它抛出的任何异常都被吞掉并记日志 —— 写一份人读副本失败,绝不该回滚一次
真源写入。

## 时钟

``clock`` 可注入(epoch 秒)。ISO 时间戳由**同一个** clock 派生,不另调
``datetime.now()`` —— 两个时间源会让 hold_s / stale_after / renotify 的测试
在真机上对不上账。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── 闭集 ──────────────────────────────────────────────────────────────────

#: 顶层状态(设计 §5)。detour 是**正交标志**不是状态 —— 绕道中同样有
#: RUNNING/WAITING 子状态,并列会造成双重身份。
STATUSES = (
    "draft", "approved", "running", "waiting_operator", "waiting_condition",
    "yielding", "paused", "halted_estop", "recovery_pending",
    "completed", "aborted",
)

#: 终态。进终态 ⇒ 让出 active_slot。
TERMINAL_STATUSES = ("completed", "aborted")

#: 转入这些状态必须给 ``status_reason``。
REASON_REQUIRED_STATUSES = (
    "waiting_operator", "waiting_condition", "yielding", "paused",
    "halted_estop", "recovery_pending", "aborted",
)

#: 事件种类(设计 §4.7)。写死成闭集:一个拼错的 kind 不会报错,只会让按 kind
#: 查的那份对账永远少一类 —— 而少的那类恰恰是出事时要看的。
EVENT_KINDS = (
    "created", "approved", "adopted", "status_change",
    "step_started", "step_finished", "step_interrupted",
    "gate_evaluated", "decision_overridden",
    "wait_entered", "wait_ack", "wait_condition_met",
    "wait_released", "wait_waived",
    "detour_entered", "detour_returned",
    "estop_seen", "estop_cleared",
    "escalation_started", "escalation_verdict",
    "recovery_item", "op_consumed", "op_rejected",
    "budget_tick", "heartbeat_stall", "aborted", "completed",
    # supervised 的撤销窗开着，这一 tick 没有点火（2026-08-27）。只在**进入**
    # 窗口时记一次，不是每 tick 一条 —— 一条每分钟重复的审计行是一条没人读的
    # 审计行。
    "ignition_held",
)

#: 用户意图(设计 §4.7 conduct_ops)。
OPS = ("pause", "resume", "abort", "takeover", "ack", "set_attended",
       "waive_condition", "override_decision")

#: 消费优先级:abort 最高,并且**吞掉同批其余**(由 Director 实现,这里只定序)。
OP_PRIORITY = {"abort": 0, "takeover": 1, "pause": 2, "resume": 3,
               "ack": 4, "waive_condition": 5, "override_decision": 6,
               "set_attended": 7}

#: 必须带 reason 的意图。abort / waive_condition / override_decision 都是「人做了一个
#: 会被追问的决定」,理由缺席就等于事后没人说得清为什么。
OPS_REASON_REQUIRED = ("abort", "waive_condition", "override_decision")

#: ``record()`` 能改的列。**白名单** —— 写错列名要当场炸,不能静默丢一次改动
#: (「加一个可编辑的键永远是双边动作」在本仓已经出现过四次)。
MUTABLE_COLUMNS = (
    "status", "status_reason", "stage_idx", "step_idx", "detour",
    "evidence_epoch", "attended", "active_wait", "active_run_id",
    "budget_spent_usd", "llm_wakes", "approved_by", "approved_at",
    "params", "spec_version",
)

#: 存成 JSON 文本的列(调用方给 python 对象,这里序列化;读回时反序列化)。
_JSON_COLUMNS = {"detour": "detour_json", "active_wait": "active_wait_json",
                 "llm_wakes": "llm_wakes_json", "params": "params_json"}


class ConductStoreError(RuntimeError):
    """本模块的基类异常。"""


class ActiveConductExists(ConductStoreError):
    """单活跃不变式被撞上了。API 层据此回 409。"""


class UnknownConduct(ConductStoreError):
    """conduct_id 不存在。**不返回 None 冒充「什么都没发生」。**"""


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class ConductStore:
    """conduct 状态的读写口。

    ``db_path`` **必填、无默认** —— 没有「不传就用全局实验库」这条逃生门。
    测试污染真实数据在本仓发生过五次,而每一次的入口都是一个善意的默认路径。
    """

    def __init__(self, db_path: "str | Path", *,
                 clock: "Callable[[], float] | None" = None,
                 observer: "Callable[[dict], None] | None" = None):
        if not db_path or not str(db_path).strip():
            raise ValueError(
                "ConductStore 需要显式 db_path —— 这里没有默认路径。"
                "测试请用 tmp_path,生产请传实验库路径。")
        if str(db_path).strip() == ":memory:":
            # 每个方法各开一条连接,``:memory:`` 会给出**各自独立**的空库:
            # 写进去、读不出来,而且一个错误都不报。
            raise ValueError(
                "ConductStore 不支持 ':memory:'(每条连接一个独立空库,"
                "写了读不到且不报错)。测试请用 tmp_path 下的真实文件。")
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock: Callable[[], float] = clock or __import__("time").time
        #: 提交后的外溢口(人读副本 + WS 帧)。默认 ``None`` = 什么都不外溢,
        #: 于是纯存储测试不需要任何文件系统。见模块 docstring 第四条。
        self._observer = observer
        self._ensure_tables()

    def set_observer(self, observer: "Callable[[dict], None] | None") -> None:
        """换外溢口。接线顺序常常是「先有 store 才有能观察它的东西」。"""
        self._observer = observer

    # ── 时间(单一来源)────────────────────────────────────────────────

    def now_epoch(self) -> float:
        return float(self._clock())

    def now_iso(self) -> str:
        return datetime.fromtimestamp(self.now_epoch()).isoformat()

    # ── 连接 ──────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        # 5 s 是 python sqlite3 的默认 busy timeout(见 logging/storage.py 里那段
        # 更正过的注释);写在这里只为把值钉死,免得有人传 timeout=0。
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_tables(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conducts (
                    conduct_id      TEXT PRIMARY KEY,
                    experiment_id    TEXT NOT NULL,
                    spec_id          TEXT NOT NULL,
                    spec_version     INTEGER NOT NULL,
                    params_json      TEXT NOT NULL DEFAULT '{}',
                    status           TEXT NOT NULL,
                    status_reason    TEXT NOT NULL DEFAULT '',
                    stage_idx        INTEGER NOT NULL DEFAULT 0,
                    step_idx         INTEGER NOT NULL DEFAULT 0,
                    detour_json      TEXT,
                    evidence_epoch   INTEGER NOT NULL DEFAULT 0,
                    attended         INTEGER NOT NULL DEFAULT 1,
                    active_wait_json TEXT,
                    active_run_id    TEXT NOT NULL DEFAULT '',
                    heartbeat_at     REAL,
                    budget_spent_usd REAL NOT NULL DEFAULT 0,
                    llm_wakes_json   TEXT NOT NULL DEFAULT '{}',
                    approved_by      TEXT NOT NULL DEFAULT '',
                    approved_at      TEXT NOT NULL DEFAULT '',
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL,
                    -- 单活跃不变式:非终态=1,终态=NULL。UNIQUE 允许多个 NULL,
                    -- 所以「至多一个没了结的 conduct」由数据库执行,不靠调用方。
                    active_slot      INTEGER,
                    UNIQUE(active_slot)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conducts_status "
                         "ON conducts(status)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conduct_events (
                    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    conduct_id  TEXT NOT NULL,
                    ts           TEXT NOT NULL,
                    kind         TEXT NOT NULL,
                    stage_id     TEXT NOT NULL DEFAULT '',
                    step_id      TEXT NOT NULL DEFAULT '',
                    run_id       TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_conduct "
                         "ON conduct_events(conduct_id, event_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_kind "
                         "ON conduct_events(conduct_id, kind)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conduct_ops (
                    op_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    conduct_id  TEXT NOT NULL,
                    op           TEXT NOT NULL,
                    args_json    TEXT NOT NULL DEFAULT '{}',
                    requested_by TEXT NOT NULL DEFAULT '',
                    requested_at TEXT NOT NULL,
                    consumed_at  TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ops_pending "
                         "ON conduct_ops(conduct_id, consumed_at, op_id)")

    # ── 建 ────────────────────────────────────────────────────────────

    def create(self, *, experiment_id: str, spec_id: str, spec_version: int,
               params: "dict | None" = None, attended: bool = True,
               created_by: str = "", conduct_id: "str | None" = None) -> str:
        """建一份 DRAFT。返回 conduct_id。

        撞上单活跃不变式抛 :class:`ActiveConductExists`(API 层 → 409)。
        """
        if not experiment_id:
            raise ValueError("conduct 必须绑定 experiment_id —— "
                             "一份没有实验归属的 conduct,产物没有地方落")
        if not spec_id:
            raise ValueError("spec_id 必填")
        cid = conduct_id or _new_id()
        now = self.now_iso()
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT INTO conducts (conduct_id, experiment_id, spec_id,"
                    " spec_version, params_json, status, stage_idx, step_idx,"
                    " attended, created_at, updated_at, active_slot)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,1)",
                    (cid, experiment_id, spec_id, int(spec_version),
                     json.dumps(params or {}, ensure_ascii=False, default=str),
                     "draft", 0, 0, 1 if attended else 0, now, now))
                event_id = self._insert_event(conn, cid, "created", payload={
                    "spec_id": spec_id, "spec_version": int(spec_version),
                    "experiment_id": experiment_id, "created_by": created_by})
        except sqlite3.IntegrityError as exc:
            if "active_slot" in str(exc):
                cur = self.active()
                raise ActiveConductExists(
                    "已经有一个未了结的 conduct"
                    + (f"({cur['conduct_id']},status={cur['status']})" if cur else "")
                    + " —— 单活跃不变式。先把它 abort 或等它跑完。") from exc
            raise
        self._emit({"conduct_id": cid, "event_id": event_id, "kind": "created",
                    "ts": now, "stage_id": "", "step_id": "", "run_id": "",
                    "payload": {"spec_id": spec_id, "experiment_id": experiment_id,
                                "created_by": created_by},
                    "status_before": "", "status_after": "draft",
                    "status_changed": True})
        return cid

    # ── 读 ────────────────────────────────────────────────────────────

    def get(self, conduct_id: str) -> "dict | None":
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM conducts WHERE conduct_id=?",
                               (conduct_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def active(self) -> "dict | None":
        """当前那个未了结的 conduct(至多一个,由 UNIQUE 保证)。"""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM conducts WHERE active_slot IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 1").fetchone()
        return self._row_to_dict(row) if row else None

    def list_conducts(self, *, status: "str | None" = None,
                       limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM conducts"
        args: list = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def last_event(self, conduct_id: str, kind: str) -> "dict | None":
        """某一 kind 的**最新**一条事件，没有则 None。

        为什么单开一个方法：:meth:`events` 是**升序 + LIMIT**，拿到的是最早那些
        （它自己的 docstring 点名了「靠 ``limit=50`` 再取尾」这个写法）。撤销窗
        要读的恰恰是「最后一次批准」——一份 pause → 再 approve 的 conduct 上，
        取错了就会拿一个早已过去的 ``ignite_at``，窗口静默失效。

        一次 SQL 直接问最大 ``event_id``，不在 Python 里排序、不分页。
        """
        from contextlib import closing as _closing

        with _closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM conduct_events WHERE conduct_id=? AND kind=? "
                "ORDER BY event_id DESC LIMIT 1",
                (conduct_id, str(kind))).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["payload"] = _loads(d.pop("payload_json", "{}"))
        return d

    def events(self, conduct_id: str, *, kind: "str | None" = None,
               run_id: "str | None" = None, after_event_id: int = 0,
               limit: int = 500) -> list[dict]:
        """审计流,**按 event_id 升序**,取最早的 ``limit`` 条。

        ⚠️ 顺序是升序 + LIMIT ⇒ 拿到的是**最早**那些,不是最新那些。想问「当前
        这一步是什么时候开始的」不能靠 ``limit=50`` 再 reverse —— 跑过 50 步之后
        那样问到的是第 50 步的时刻,而它会被当成当前步的时刻,于是停滞告警拿一个
        几小时前的开始时刻去判一个刚开始的步。**按 ``run_id`` 问**:那是唯一能
        把「这一步」和「某一步」分开的键。
        """
        sql = "SELECT * FROM conduct_events WHERE conduct_id=? AND event_id>?"
        args: list = [conduct_id, int(after_event_id)]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if run_id is not None:
            sql += " AND run_id=?"
            args.append(run_id)
        sql += " ORDER BY event_id LIMIT ?"
        args.append(int(limit))
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = _loads(d.pop("payload_json", "{}"))
            out.append(d)
        return out

    # ── 改(唯一一扇门)──────────────────────────────────────────────

    def record(self, conduct_id: str, kind: str, *,
               changes: "dict | None" = None,
               stage_id: str = "", step_id: str = "", run_id: str = "",
               payload: "dict | None" = None) -> int:
        """写一条事件,并在**同一个事务**里应用状态改动。返回 event_id。

        这是改 ``conducts`` 行的唯一方法(``touch_heartbeat`` 除外,它不是状态)。
        崩在中间整体回滚 —— 不会出现「事件说进了等待态,状态行还在 RUNNING」。

        ``changes`` 的键取自 :data:`MUTABLE_COLUMNS`;写错名字**抛异常**,
        不静默丢弃。JSON 列(detour / active_wait / llm_wakes / params)直接
        给 python 对象。

        改 ``status`` 时:
        * 值必须在 :data:`STATUSES` 里;
        * 异常态必须同时给非空 ``status_reason``;
        * ``active_slot`` 自动跟着终态收放 —— 让出 slot 这件事绝不该靠调用方记得;
        * 事件 payload 自动带上 ``status_from`` / ``status_to``(审计要的是转移,
          不是终值)。
        """
        if kind not in EVENT_KINDS:
            raise ValueError(f"未知事件种类 {kind!r} —— 闭集见 EVENT_KINDS")
        changes = dict(changes or {})
        payload = dict(payload or {})
        unknown = [k for k in changes if k not in MUTABLE_COLUMNS]
        if unknown:
            raise ValueError(f"不可改的列: {unknown};白名单见 MUTABLE_COLUMNS")

        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT status, status_reason FROM conducts WHERE conduct_id=?",
                (conduct_id,)).fetchone()
            if row is None:
                raise UnknownConduct(f"conduct {conduct_id!r} 不存在")
            old_status = str(row["status"])

            sets: list[str] = []
            args: list = []
            new_status = changes.get("status")
            if old_status in TERMINAL_STATUSES and new_status not in (None, old_status):
                # 终态是终态。让一个 ABORTED 的 conduct 复活,等于让它的中止序列
                # (确认式退针那一段)变成一段没有结论的历史。要接着做就新建一份。
                raise ConductStoreError(
                    f"conduct {conduct_id} 已是终态 {old_status},不能改成 "
                    f"{new_status} —— 终态是终态,要接着做请新建一份")
            if new_status is not None:
                if new_status not in STATUSES:
                    raise ValueError(f"未知状态 {new_status!r} —— 闭集见 STATUSES")
                reason = changes.get("status_reason",
                                     row["status_reason"] if new_status == old_status
                                     else "")
                if new_status in REASON_REQUIRED_STATUSES and not str(reason).strip():
                    raise ValueError(
                        f"转入 {new_status} 必须给 status_reason —— "
                        f"一个没有「为什么」的停,人只能靠猜或者重启进程")
                changes["status_reason"] = reason
                sets.append("active_slot=?")
                args.append(None if new_status in TERMINAL_STATUSES else 1)
                payload.setdefault("status_from", old_status)
                payload.setdefault("status_to", new_status)

            for col, val in changes.items():
                if col in _JSON_COLUMNS:
                    sets.append(f"{_JSON_COLUMNS[col]}=?")
                    args.append(None if val is None
                                else json.dumps(val, ensure_ascii=False, default=str))
                elif col == "attended":
                    sets.append("attended=?")
                    args.append(1 if val else 0)
                else:
                    sets.append(f"{col}=?")
                    args.append(val)

            sets.append("updated_at=?")
            args.append(self.now_iso())
            args.append(conduct_id)
            try:
                conn.execute(
                    f"UPDATE conducts SET {', '.join(sets)} WHERE conduct_id=?",
                    args)
            except sqlite3.IntegrityError as exc:
                if "active_slot" in str(exc):
                    raise ActiveConductExists(
                        "另一个 conduct 已占着活跃位,这一个不能转回非终态"
                    ) from exc
                raise
            event_id = self._insert_event(conn, conduct_id, kind,
                                          stage_id=stage_id, step_id=step_id,
                                          run_id=run_id, payload=payload)
            status_after = str(new_status if new_status is not None else old_status)
            ts = self.now_iso()
        # 提交之后才外溢:一个收到帧就回头读的客户端,必然读得到这一行。
        self._emit({"conduct_id": conduct_id, "event_id": event_id, "kind": kind,
                    "ts": ts, "stage_id": stage_id, "step_id": step_id,
                    "run_id": run_id, "payload": dict(payload),
                    "status_before": old_status, "status_after": status_after,
                    "status_changed": new_status is not None
                                      and new_status != old_status})
        return event_id

    def touch_heartbeat(self, conduct_id: str) -> float:
        """写一次心跳(独立小事务)。返回写入的 epoch 秒。

        语义 =「决策循环活着」,**不是**「步在推进」。执行长步时它不更新是设计:
        API 层靠 ``heartbeat 年龄`` + ``active_run_id`` + 步的 started_at 去区分
        「正常长步」和「线程死了」。把这两件事混成一个数,就再也分不出来了。
        """
        ts = self.now_epoch()
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE conducts SET heartbeat_at=? WHERE conduct_id=?",
                (ts, conduct_id))
            if cur.rowcount == 0:
                raise UnknownConduct(f"conduct {conduct_id!r} 不存在")
        return ts

    # ── 意图队列 ──────────────────────────────────────────────────────

    def enqueue_op(self, conduct_id: str, op: str, *,
                   args: "dict | None" = None, requested_by: str = "") -> int:
        """API 线程写意图。返回 op_id。

        Director 下一 tick 消费。**唯一的例外通道不在这里**:abort 还要由 API
        线程立刻置 per-run abort Event —— Director 卡在 ``executor.run`` 里时
        tick 不会来,而 abort 按钮必须不经 tick 生效。这条由 API 层实现,
        本方法只负责让意图落表。
        """
        if op not in OPS:
            raise ValueError(f"未知意图 {op!r} —— 闭集见 OPS")
        payload = dict(args or {})
        if op in OPS_REASON_REQUIRED and not str(payload.get("reason", "")).strip():
            raise ValueError(f"{op} 必须带 reason")
        if op in ("ack", "waive_condition") and not str(payload.get("wait_id", "")).strip():
            # 对旧等待点的 ack 必须能被认出来:wait_id 每次等待唯一。
            raise ValueError(f"{op} 必须带 wait_id")
        if op == "override_decision" and not str(payload.get("decision_id", "")).strip():
            # 与 wait_id 同一条纪律,防的是同一件事:**对一个已经翻篇的判定说
            # 「继续」**。``decision_id`` 是那一次 ``gate_evaluated`` 的 event_id,
            # 每判一次就换一个 —— 所以一次放行只解得开**它自己那一次**。
            raise ValueError("override_decision 必须带 decision_id"
                             "(那一次 gate_evaluated 的 event_id)")
        if self.get(conduct_id) is None:
            raise UnknownConduct(f"conduct {conduct_id!r} 不存在")
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO conduct_ops (conduct_id, op, args_json,"
                " requested_by, requested_at) VALUES (?,?,?,?,?)",
                (conduct_id, op,
                 json.dumps(payload, ensure_ascii=False, default=str),
                 requested_by, self.now_iso()))
            return int(cur.lastrowid)

    def pending_ops(self, conduct_id: str) -> list[dict]:
        """未消费的意图,**按优先级再按入队序**(abort > takeover > pause > …)。"""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM conduct_ops WHERE conduct_id=? AND consumed_at IS NULL"
                " ORDER BY op_id", (conduct_id,)).fetchall()
        ops = []
        for r in rows:
            d = dict(r)
            d["args"] = _loads(d.pop("args_json", "{}"))
            ops.append(d)
        ops.sort(key=lambda d: (OP_PRIORITY.get(d["op"], 99), d["op_id"]))
        return ops

    def consume_op(self, op_id: int) -> bool:
        """标记已消费。**幂等**:第二次返回 False,不是异常也不是又消费一遍。

        重放一次 pause 只是多余,重放一次 abort 就是在别人已经重启之后再中止
        一次 —— 所以「消费过没有」必须是可问的,而不是靠调用方记账。
        """
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE conduct_ops SET consumed_at=? "
                "WHERE op_id=? AND consumed_at IS NULL",
                (self.now_iso(), int(op_id)))
            return cur.rowcount > 0

    # ── 内部 ──────────────────────────────────────────────────────────

    def _emit(self, event: dict) -> None:
        """把一条已提交的事件交给外溢口。**永不抛、永不改变真源。**

        观察者跑在**调用线程**上(Director 的 tick 线程或 API 线程),所以它只该
        做便宜的事:追加一行 jsonl、往 EventBus 上打一帧。任何慢活(尤其是任何
        模型调用)都必须自己另起线程,否则一次状态转移会卡在一次推送后面。
        """
        obs = self._observer
        if obs is None:
            return
        try:
            obs(dict(event))
        except Exception as exc:  # noqa: BLE001
            logger.warning("conduct 事件外溢失败(真源已提交,状态照旧): %s", exc)

    def _insert_event(self, conn: sqlite3.Connection, conduct_id: str, kind: str,
                      *, stage_id: str = "", step_id: str = "", run_id: str = "",
                      payload: "dict | None" = None) -> int:
        cur = conn.execute(
            "INSERT INTO conduct_events (conduct_id, ts, kind, stage_id,"
            " step_id, run_id, payload_json) VALUES (?,?,?,?,?,?,?)",
            (conduct_id, self.now_iso(), kind, stage_id, step_id, run_id,
             json.dumps(payload or {}, ensure_ascii=False, default=str)))
        return int(cur.lastrowid)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for name, col in _JSON_COLUMNS.items():
            raw = d.pop(col, None)
            d[name] = _loads(raw) if raw else (None if name in
                                               ("detour", "active_wait") else {})
        d["attended"] = bool(d.get("attended"))
        d["is_terminal"] = d.get("status") in TERMINAL_STATUSES
        return d


def _loads(raw) -> Any:
    if raw in (None, ""):
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        # 存进去的一定是我们自己 dumps 的;解不开说明库被外部动过。
        # 返回 None 而不是 {}:「解不开」不是「空」。
        logger.warning("conduct store: JSON 解析失败,按「读不到」处理: %r", raw)
        return None


__all__ = [
    "STATUSES", "TERMINAL_STATUSES", "REASON_REQUIRED_STATUSES", "EVENT_KINDS",
    "OPS", "OP_PRIORITY", "OPS_REASON_REQUIRED", "MUTABLE_COLUMNS",
    "ConductStore", "ConductStoreError", "ActiveConductExists",
    "UnknownConduct",
]
