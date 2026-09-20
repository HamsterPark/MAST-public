"""ConductDirector —— 驱动多天 conduct 的那条常驻线程。

设计:``campaign_director_design.md`` §3(七个架构判断)、§5(状态转移表)、
§6(tick 行为规范)、§10(陷阱清单)。

## 与 SafetyWatchdog 的区别(这条必须在最前面)

两者都是「代码线程驱动仪器」,但权限**根本不同**:

* **watchdog 是紧急救济** —— 有权走裸 ``urgent_call``,因为它要在一次卡死的
  事务前面把针拿开;
* **Director 是常规驱动** —— 每一步都必须走 ``executor.run`` 的完整安全管道
  (registry → 状态刷新 → SafetyGuard → 快照),**永远没有裸 TCP 权**。

把这两句话分开写,是因为「反正都是代码驱动仪器」这个念头会把它们混成一个,
而混起来的那一刻,conduct 就获得了绕过安全门的能力。

## 诚实的短板(§3-7)

``executor.run`` 是同步的。一次卡死的 TCP 事务会**挂住这条线程**,而杀线程会
永久损坏 Nanonis 端口。所以这里**不做「超时杀步」,也不假装能解**:
``timeout_s`` 只喂给 API 层的停滞告警,heartbeat 只证明决策循环还在转。
停滞期间护针的是 watchdog —— 一个可接受的降级,写下来免得有人以为它是个 bug。

## 一个 tick 做什么(§6)

1. 写 heartbeat(独立小事务);
2. 消费 ``conduct_ops``(优先级 abort > takeover > pause > resume > ack >
   waive > set_attended;abort **吞掉同批其余**);
3. pre-flight:急停闩 → 闩清转恢复 → SAFE 模式对账 → 值守位;
4. 按状态分派;
5. 节奏。

**RUNNING 态步间不 sleep** 的实现方式:``step_tick()` 每次只推进**一件事**并
返回 :class:`TickReport`;线程循环看 ``report.sleep_hint`` 决定睡多久,RUNNING
下推进成功即 0 —— 于是「每步之间重跑第 2-3 步」自动成立(下一圈从 heartbeat
重新走一遍),急停和暂停必然在步边界生效。这比在一个 tick 里内联循环更容易
测,也少一处「忘了重查闩」的入口。
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from mast.conduct import recovery, rules
from mast.conduct.ports import (
    AbortRegistry,
    LatchState,
    LoggingNotifier,
    NullLatch,
    NullTemperature,
    Notification,
    StepOutcome,
)
from mast.conduct.spec import (
    AT_ENTRY_GATE,
    ConductSpec,
    StageSpec,
    StepSpec,
)
from mast.conduct.store import ConductStore, TERMINAL_STATUSES

logger = logging.getLogger(__name__)

#: ``step_idx == -1`` = 停在阶段的 entry_gate 上,还没进第一步。
#:
#: 定义搬去了 :mod:`mast.conduct.spec`(M4-a):恢复清算也要懂这个约定,
#: 而一个约定两份字面量,改了一份另一份会安静地按旧定义继续算。这里 re-export
#: 是为了不动任何 ``from mast.conduct.director import AT_ENTRY_GATE``。
_ = AT_ENTRY_GATE


#: 投影落空时那半句话 —— **它指向模板,不指向机器**。
#:
#: 「spec 点名了一个这一步不产出的字段」是配置错误(校验器规则⑤该在 approve 时
#: 拦下),而「这一步没有产出」是仪器那一侧的事实。两句话让用户去查的地方完全
#: 不一样,所以措辞必须分开 —— 凌晨三点照着「没有产出」去查机器,而根因在模板里
#: 一个拼错的字段名,那一晚就白花了。
#:
#: 单拎成常量,一半是为了单一真源,另一半是为了它**可被替换验证**:把它换成
#: 通用措辞,那条钉着这个区分的测试必须变红。
PROJECTION_MISS_BLAME = "**这是 spec 点错了字段,不是这一步没产出**"


def _project(values: dict, fields) -> "tuple[dict, list]":
    """按 ``EvidenceSpec.fields`` 投影。→ ``(选出来的, 点名了却没有的)``。

    ``fields`` 为空 = 不投影,整包给出去(今天的默认,不改任何现有闸门的行为)。

    投影按**顶层键**做。嵌套字段(``a.b``)由判据自己的路径查找处理 ——
    在这里再实现一次路径解析,就是同一个动作的第二份实现。
    """
    if not fields:
        return dict(values), []
    picked = {k: values[k] for k in fields if k in values}
    return picked, [k for k in fields if k not in values]


#: 连续 busy 超过这么久 ⇒ YIELDING + 通知(§5)。
BUSY_YIELD_AFTER_S = 600.0

#: 「这一代证据是从什么时候开始的」在事件 payload 里的键。**专用**,只有真正
#: 建立代次边界的三个事件(``adopted`` / ``detour_entered`` / 重启清算 A4)带它。
#:
#: 专用是必须的,不是讲究:曾经用「payload 里同时有 evidence_epoch 和 at」当标记,
#: 而 ``step_finished`` 两样都有 —— 于是每步跑完都算一次代次边界,证据窗口一路
#: 跟着最后一步往前爬。见 :meth:`ConductDirector._epoch_started_at`。
EPOCH_START_KEY = "epoch_started_at"

#: 哪些意图在哪些状态下有意义。**不在表里 = 显式拒绝 + op_rejected 留痕**,
#: 不是静默 no-op(§5 末:「不可能」转移每条钉进穷举测试)。
_DRIVEN = frozenset({"running", "waiting_operator", "waiting_condition",
                     "yielding", "recovery_pending"})
_NON_TERMINAL = frozenset({"draft", "approved", "running", "waiting_operator",
                           "waiting_condition", "yielding", "paused",
                           "halted_estop", "recovery_pending"})
OP_VALID_STATUSES: "dict[str, frozenset[str]]" = {
    # pause/takeover 只在 Director 真的在驱动时有意义。DRAFT 还没被采纳,
    # PAUSED 已经停了,HALTED_ESTOP 由闩说了算(解闩的路在 safety 路由上)。
    "pause": _DRIVEN,
    "takeover": _DRIVEN,
    "resume": frozenset({"paused"}),
    # abort 必须**从任何非终态都够得着** —— 这是「能停不能解」的反面。
    "abort": _NON_TERMINAL,
    "set_attended": _NON_TERMINAL,
    "ack": frozenset({"waiting_operator", "waiting_condition"}),
    "waive_condition": frozenset({"waiting_operator", "waiting_condition"}),
    # 闸门判定停下来之后那条**留痕的解锁路**。只在 WAITING_OPERATOR 有意义:
    # 它解的是一次裁决,而裁决转人只会转到这一个状态。
    #
    # 它**不是**无人值守时多出来的一条出口:意图只能由人经 UI 写进 conduct_ops,
    # Director 自己永远不会生成它。``attended`` 位一个字都不用改。
    "override_decision": frozenset({"waiting_operator"}),
}

#: 恢复自检清单(§8)。无接触档 + 接触档;等待态重启只做无接触档。
NO_CONTACT_CHECKS = ("A6_leftovers", "A5_spec", "A4_epoch", "A1_link", "A2_temp")
CONTACT_CHECKS = ("A3_tip",)


@dataclass
class TickReport:
    """一个 tick 干了什么 —— 给测试、日志和将来的面板看。"""

    conduct_id: str = ""
    status_before: str = ""
    status_after: str = ""
    actions: list = field(default_factory=list)
    #: 建议睡多久(秒)。0 = 立刻再来一圈(RUNNING 连续推进)。
    sleep_hint: float = 0.0
    #: 本 tick **没能做**的检查(诚实报告,不当成做过了)。
    not_checked: list = field(default_factory=list)

    def did(self, what: str) -> "TickReport":
        self.actions.append(what)
        return self


class ConductDirector:
    """状态机的执行者。**不自动启动** —— 构造 + 显式 :meth:`start`。

    所有外界接口都是注入的(executor / 急停闩 / 温度 / 通知 / 时钟 /
    技能元数据 / 花销),原因见 :mod:`mast.conduct.ports`。
    """

    def __init__(
        self,
        store: ConductStore,
        *,
        spec_provider: "Callable[[str], ConductSpec]",
        executor,
        latch=None,
        temperature=None,
        notifier=None,
        abort_registry: "AbortRegistry | None" = None,
        clock: "Callable[[], float] | None" = None,
        decide_route: "Callable[[dict, dict], dict] | None" = None,
        analyses_get: "Callable[[str], Callable[[dict], dict]] | None" = None,
        skill_meta: "Callable[[str], Any] | None" = None,
        cost_reader: "Callable[[str], float | None] | None" = None,
        lock_probe: "Callable[[], bool] | None" = None,
        recovery_probe: "Callable[[str], str] | None" = None,
        link_probe: "Callable[[], dict] | None" = None,
        monitor_probe: "Callable[[str, float], dict | None] | None" = None,
        frame_metrics_probe: "Callable[[str, float], dict | None] | None" = None,
        escalation_advisor: "Callable[[Any], Any] | None" = None,
    ):
        self.store = store
        self._spec_provider = spec_provider
        self.executor = executor
        self.latch = latch or NullLatch()
        self.temperature = temperature or NullTemperature()
        self.notifier = notifier or LoggingNotifier()
        self.aborts = abort_registry or AbortRegistry()
        self._clock: Callable[[], float] = clock or store.now_epoch
        self._decide_route = decide_route
        self._skill_meta = skill_meta
        self._cost_reader = cost_reader
        self._lock_probe = lock_probe
        self._recovery_probe = recovery_probe
        #: A1 连接探活(M4-a)。签名 ``() -> {role: bool | None}``,
        #: 其中 ``None`` = **这个 role 探不出来**(不是「它挂了」也不是「它活着」)。
        #: 没注入 ⇒ A1 报「读不到」,不报通过。
        self._link_probe = link_probe
        #: 两个**外部证据**探针。签名都是 ``(selector, since_epoch_s) -> dict | None``,
        #: 而 ``None`` 的意思是**读不到**(不是「没有事件」/「没有帧」)。
        #: 没注入 ⇒ 对应的证据源报缺席,闸门走保守去向。
        self._monitor_probe = monitor_probe
        self._frame_metrics_probe = frame_metrics_probe
        #: L2 值守席(``mast.conduct.l2_seat``)。签名 ``(EscalationContext) ->
        #: EscalationAdvice``,失败一律抛 ``SeatUnavailable``。
        #: **没注入 ⇒ escalate 策略退回「停下来问人」并把这句话说出来** ——
        #: 不假装升级过了。
        self._escalation_advisor = escalation_advisor
        if analyses_get is None:
            from mast.conduct.analyses import get as _get
            analyses_get = _get
        self._analyses_get = analyses_get

        self._thread: "threading.Thread | None" = None
        self._stop = threading.Event()
        #: 连续 busy 起始时刻(进程内的量;重启后重新计时 —— 它只影响
        #: 「什么时候让路」,不影响任何一步扫在哪儿)。
        self._busy_since: "float | None" = None

    # ── 线程 ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """起线程。**只有显式调用才会跑** —— runtime 挂点是 M1-c 的事。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="conduct-director",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """请线程停。**不杀线程** —— 它可能正卡在一次 TCP 事务里,强杀会永久
        损坏 Nanonis 端口(§3-7)。停不下来就如实记一条日志。"""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout_s)
            if t.is_alive():
                logger.warning("conduct director 线程没在 %.0f s 内退出 —— "
                               "多半卡在一次长步里,不强杀", timeout_s)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                report = self.step_tick()
            except Exception as exc:  # noqa: BLE001 —— 一个 tick 崩了不该带走线程
                logger.exception("conduct tick 失败: %s", exc)
                self._stop.wait(5.0)
                continue
            self._stop.wait(max(0.0, report.sleep_hint))

    # ── 一个 tick(§6)────────────────────────────────────────────────

    def step_tick(self) -> TickReport:
        """推进一件事。测试手驱它,线程循环也调它。"""
        row = self.store.active()
        if row is None:
            return TickReport(actions=["no_active_conduct"], sleep_hint=60.0)
        cid = row["conduct_id"]
        rep = TickReport(conduct_id=cid, status_before=row["status"])
        # 每 tick 清一次「仪器归 watchdog」的旗。忘了清的后果是**退针永远不发生**
        # —— 一面粘住的安全旗比没有旗更坏,因为它看起来是在保护什么。
        self._rescue_owned_elsewhere = False

        # 1. heartbeat —— 独立小事务。语义是「决策循环活着」,不是「步在推进」。
        self.store.touch_heartbeat(cid)

        # 2. 意图队列
        self._consume_ops(cid, rep)
        row = self.store.get(cid)
        if row["status"] in TERMINAL_STATUSES:
            rep.status_after = row["status"]
            rep.sleep_hint = 60.0
            return rep

        # 3. pre-flight
        if self._preflight(cid, row, rep):
            row = self.store.get(cid)
            rep.status_after = row["status"]
            rep.sleep_hint = 60.0
            return rep
        row = self.store.get(cid)

        # 4. 分派
        try:
            self._dispatch(cid, row, rep)
        finally:
            after = self.store.get(cid)
            rep.status_after = after["status"] if after else ""
        # 5. 节奏
        #
        # 2026-08-20:``tick_interval_s`` 在此之前是个**死字段** —— 它定义在
        # ``ConductBudget`` 上、有 5-15 s 的区间校验、被面板和停滞告警读去当
        # 参数,而**这条循环从来没读过它**:RUNNING 态的 sleep_hint 一直是 0,
        # 也就是「一步接一步、中间不歇」。
        #
        # 那不是一个无害的偏差。零延迟意味着:意图队列里的 pause/abort 只能在
        # 两步之间那一瞬被看到;心跳按 tick 写,而停滞告警的判据是
        # ``max(3×tick_interval, 步超时+余量)`` —— 一个从没生效过的 tick 值
        # 让那道告警的分母是想象出来的。
        #
        # 现在按 spec 说的走。它不会拖慢什么:一步动辄几十秒到几分钟,这里的
        # 5-15 s 只落在**步与步之间**,而那正是用户按下暂停后期望被听见的地方。
        if rep.sleep_hint == 0.0:
            spec = self._spec_or_none(row)
            if spec is None:
                rep.sleep_hint = 60.0
            elif rep.status_after == "running":
                rep.sleep_hint = float(spec.budgets.tick_interval_s)
            else:
                rep.sleep_hint = float(spec.budgets.wait_tick_interval_s)
        return rep

    # ── 2. 意图 ──────────────────────────────────────────────────────

    def _consume_ops(self, cid: str, rep: TickReport) -> None:
        ops = self.store.pending_ops(cid)      # 已按优先级排好
        aborting = False
        for op in ops:
            name = op["op"]
            if aborting:
                # abort 吞掉同批其余:一个已经决定中止的 conduct,再去执行
                # 「继续」「换值守位」只是给日志添乱。**但要留痕**。
                self.store.consume_op(op["op_id"])
                self.store.record(cid, "op_rejected", payload={
                    "op": name, "op_id": op["op_id"],
                    "why": "同批已有 abort,其余意图作废"})
                rep.did(f"op_swallowed:{name}")
                continue
            self.store.consume_op(op["op_id"])
            status = self.store.get(cid)["status"]
            allowed = OP_VALID_STATUSES.get(name, frozenset())
            if status not in allowed:
                # 「不可能」的转移:**显式拒绝 + 留痕**,不静默 no-op。
                # 静默 no-op 会让用户按了按钮却什么都没发生,而且事后查不到。
                self.store.record(cid, "op_rejected", payload={
                    "op": name, "op_id": op["op_id"], "status": status,
                    "why": f"{name} 在 {status} 状态下没有意义;"
                           f"允许的状态是 {sorted(allowed)}"})
                rep.did(f"op_rejected:{name}")
                continue
            self.store.record(cid, "op_consumed", payload={
                "op": name, "op_id": op["op_id"], "args": op["args"],
                "requested_by": op["requested_by"]})
            if name == "abort":
                aborting = True
            self._apply_op(cid, name, op, rep)

    def _apply_op(self, cid: str, name: str, op: dict, rep: TickReport) -> None:
        args = op.get("args") or {}
        by = op.get("requested_by") or "operator"
        row = self.store.get(cid)
        if name == "abort":
            self._abort(cid, row, str(args.get("reason") or "用户中止"), by, rep)
        elif name in ("pause", "takeover"):
            # pause ≠ abort:**不动针**。用户按暂停常常正是为了手动干预。
            self._signal_abort_event(row)
            reason = (f"{'接管' if name == 'takeover' else '暂停'} by {by}"
                      + (f": {args.get('note')}" if args.get("note") else ""))
            self.store.record(cid, "status_change",
                              changes={"status": "paused", "status_reason": reason},
                              payload={"takeover": name == "takeover", "by": by})
            rep.did(name)
        elif name == "resume":
            # 暂停期间世界未知(人可能动过仪器)⇒ 统一走自检,快则秒过。
            self.store.record(cid, "status_change", changes={
                "status": "recovery_pending",
                "status_reason": f"resume by {by} —— 暂停期间世界未知,先自检"})
            rep.did("resume")
        elif name == "set_attended":
            self.store.record(cid, "status_change",
                              changes={"attended": bool(args.get("attended", True))},
                              payload={"attended": bool(args.get("attended", True)),
                                       "by": by})
            rep.did("set_attended")
        elif name in ("ack", "waive_condition"):
            self._apply_wait_op(cid, name, args, by, rep)
        elif name == "override_decision":
            self._apply_override_decision(cid, args, by, rep)

    def _apply_override_decision(self, cid: str, args: dict, by: str,
                             rep: TickReport) -> None:
        """用户看过一次闸门判定之后说「继续」。

        ## 它解的是**这一次判定**,不是这道闸

        ``decision_id`` 是那一次 ``gate_evaluated`` 的 event_id —— 每判一次换一个,
        与 ``wait_id`` 同一条纪律。所以:

        * 对着一个已经翻篇的判定说「继续」会被拒(``op_rejected`` 留痕);
        * 下一次走到同一道闸,照样重新判。**它不是跳过闸门的万能钥匙。**

        ## 闸门的裁决与人的决定分开记

        ``gate_evaluated`` 那一条**原样留着**(它说的是闸门当时判了什么),人的决定
        另起一条 ``decision_overridden``。合成一条的话,事后对账会看到一道**从不判 fail
        的闸** —— 而那正是最需要复核的那种记录。

        ## 为什么没有「继续 = 去修针」那一档

        被判 ``wait_operator`` 的闸门里,确实有本来要去修针的(比如 rule 判不了、
        而它的 fail 路由指向 ``detour``)。但**「去修针」不是对这道闸那个问题的
        回答** —— 它是另一个动作,而且是一次几小时、要人到场两次的动作。
        把它塞进一个叫「继续」的按钮里,语义对不上;而给 ``DetourPolicy.triggers``
        再加一个今天还没有执行力的生产方,更是把一句安心话变成两句。
        **要推翻这条得先回答**:一个显式的 ``request_detour`` 意图(带自己的确认),
        与「人自己接管去修」相比多买到了什么。
        """
        row = self.store.get(cid)
        pending = self.pending_decision(cid)
        want = str(args.get("decision_id") or "")
        if not pending:
            self.store.record(cid, "op_rejected", payload={
                "op": "override_decision", "by": by,
                "why": "当前没有停在一个可放行的闸门判定上"
                       f"(状态 {row.get('status')!r}"
                       + ("、而且停的是一个 wait 步" if row.get("active_wait") else "")
                       + ")"})
            rep.did("op_rejected:override_decision")
            return
        if want != str(pending.get("decision_id")):
            # 对旧判定的放行必须被认出来 —— 与 ack 的 wait_id 同一个理由。
            self.store.record(cid, "op_rejected", payload={
                "op": "override_decision", "by": by,
                "why": f"decision_id {want!r} 与当前判定 "
                       f"{pending.get('decision_id')!r} 不匹配"})
            rep.did("op_rejected:override_decision")
            return
        spec = self._spec_or_none(row)
        if spec is None:
            self.store.record(cid, "op_rejected", payload={
                "op": "override_decision", "by": by,
                "why": f"取不到模板 {row.get('spec_id')!r},放行之后不知道往哪走"})
            rep.did("op_rejected:override_decision")
            return
        reason = str(args.get("reason") or "")
        kind = str(pending.get("kind") or "")
        what = (f"闸门 {pending.get('gate_id')}" if kind == "gate"
                else f"恢复自检 {pending.get('item')}")
        # **原判定原样带着** —— 人推翻的是它,而事后对账要看得见被推翻的是什么。
        # 合成一条的话,记录里会出现一道从不判 fail 的闸 / 一项从不失败的自检,
        # 而那正是最该复核的那种记录。
        self.store.record(
            cid, "decision_overridden", stage_id=str(pending.get("stage_id") or ""),
            changes={"status": "running" if kind == "gate" else "recovery_pending",
                     # **持续显示**:面板照 status_reason 印,不是按下那一刻闪一次。
                     "status_reason": ("" if kind == "gate" else
                                       f"{what} 的判定由 {by} 放行,继续自检:{reason}")},
            payload={"kind": kind, "decision_id": pending.get("decision_id"),
                     "gate_id": pending.get("gate_id"),
                     "item": pending.get("item"),
                     "which": pending.get("which"),
                     # 放行只在**当前代次**有效 —— 一次重启或一次绕道之后世界
                     # 是新的,上一代那句「就地续跑吧」不该跟着继承。
                     "epoch": int(row.get("evidence_epoch") or 0),
                     "overridden_verdict": pending.get("verdict"),
                     "overridden_reason": pending.get("reason"),
                     "by": by, "reason": reason, "at": self._clock()})
        if kind == "gate":
            self._gate_passed(cid, self.store.get(cid), spec,
                              str(pending.get("which") or ""), rep)
        else:
            # 恢复自检:把那一项记成**被人放行**,清单接着往下走。
            #
            # 不是记成 ``pass`` —— 「探针说过了」与「人说算了」是两句话,而后者
            # 是要被复核的那一句。``_recovery_step`` 的 ``done`` 集按 item 建,
            # 所以这一条同样让它不再重跑。
            self.store.record(cid, "recovery_item", payload={
                "item": pending.get("item"), "verdict": "overridden",
                "detail": f"原判定 {pending.get('verdict')}:"
                          f"{pending.get('reason')} —— 由 {by} 放行:{reason}",
                "by": by})
        self._notify(cid, "decision_overridden",
                     f"用户 {by} 放行了 {what} 的这一次判定"
                     f"({pending.get('verdict')}):{reason}", "warn")
        rep.did("override_decision")

    def _apply_wait_op(self, cid: str, name: str, args: dict, by: str,
                       rep: TickReport) -> None:
        row = self.store.get(cid)
        wait = row.get("active_wait") or {}
        want = str(args.get("wait_id") or "")
        if not wait or wait.get("wait_id") != want:
            # 对旧等待点的 ack 必须被认出来 —— wait_id 每次等待唯一。
            self.store.record(cid, "op_rejected", payload={
                "op": name, "why": f"wait_id {want!r} 与当前等待 "
                                   f"{wait.get('wait_id')!r} 不匹配"})
            rep.did(f"op_rejected:{name}")
            return
        now = self._clock()
        if name == "ack":
            wait["ack_at"] = now
            wait["ack_by"] = by
            self.store.record(cid, "wait_ack", changes={"active_wait": wait},
                              payload={"wait_id": want, "by": by})
            rep.did("ack")
        else:
            # waive:把 condition 闸标成「证据由人提供」。**不是默默放行** ——
            # 面板要持续显示这个标记,报告也带。存在的理由是「能停不能解=死锁」:
            # 传感器读不到时 condition 会永远 stale,人必须有一条留痕的解锁路。
            wait["waived_by"] = by
            wait["waived_at"] = now
            wait["waive_reason"] = str(args.get("reason") or "")
            self.store.record(cid, "wait_waived", changes={"active_wait": wait},
                              payload={"wait_id": want, "by": by,
                                       "reason": wait["waive_reason"]})
            rep.did("waive_condition")

    # ── 3. pre-flight ────────────────────────────────────────────────

    def _preflight(self, cid: str, row: dict, rep: TickReport) -> bool:
        """返回 True = 本 tick 到此为止。"""
        state = self._latch_state()
        status = row["status"]
        if state.latched:
            if status != "halted_estop":
                self._signal_abort_event(row)
                why = state.why or "急停闩挂着(闩没给原因)"
                self.store.record(cid, "estop_seen", changes={
                    "status": "halted_estop", "status_reason": why},
                    payload={"why": why, "abort_set": state.abort_set})
                self._notify(cid, "estop", f"急停闩挂着,conduct 停在原地:{why}",
                             severity="crit")
                rep.did("halted_estop")
            else:
                rep.did("estop_still_latched")
            return True
        if status == "halted_estop":
            # 闩清 ≠ 针没事。必须走自检,不许直接续跑。
            self.store.record(cid, "estop_cleared", changes={
                "status": "recovery_pending",
                "status_reason": "急停闩已清 —— 闩清不等于针没事,先自检"})
            rep.did("estop_cleared")
            return False

        # SAFE 模式对账(§6-3c):下一步是 DANGEROUS 而阶段没声明对应 capability
        # ⇒ 停下来问人,**不静默跳过**。
        if self._skill_meta is None:
            rep.not_checked.append(
                "SAFE 模式对账 —— 没注入 skill_meta(approve 时 validator 已按注册表"
                "查过规则③,这里是纵深防御)")
        else:
            bad = self._safe_mode_violation(row)
            if bad:
                self.store.record(cid, "status_change", changes={
                    "status": "waiting_operator", "status_reason": bad})
                self._notify(cid, "safe_mode", bad, severity="warn")
                rep.did("safe_mode_block")
                return True
        return False

    #: 这一 tick 里仪器归 watchdog(针尖事件/撞针)。由 ``_dispatch`` 那条路立起,
    #: 每个 tick 开头清掉 —— 一面忘了清的旗会让退针**永远**不发生。
    _rescue_owned_elsewhere: bool = False

    def _latch_state(self) -> LatchState:
        try:
            state = self.latch.state()
        except Exception as exc:  # noqa: BLE001
            # 闩读不到 ⇒ **不假设它没挂**。但也不能凭空判它挂着(那样解不开),
            # 所以:如实记一条,按没挂继续,并让告警把这件事顶出去。
            logger.warning("急停闩读不到(按未挂继续,但这是一次读失败): %s", exc)
            return LatchState()
        if isinstance(state, LatchState):
            return state
        return LatchState(latched=bool(state.get("latched")),
                          abort_set=bool(state.get("abort_set")),
                          why=str(state.get("why") or ""))

    def _safe_mode_violation(self, row: dict) -> str:
        spec = self._spec_or_none(row)
        if spec is None:
            return ""
        pos = self._position(spec, row)
        if pos is None or pos[2] is None:
            return ""
        stage, _kind, step = pos
        if step is None or not step.touches_hardware:
            return ""
        meta = self._skill_meta(step.skill) if self._skill_meta else None
        if meta is None:
            return (f"下一步 {step.step_id} 要跑的技能 {step.skill!r} 在注册表里找不到 "
                    f"—— 这台机器上跑不了它")
        level = str(getattr(getattr(meta, "safety_level", None), "value", "")).lower()
        if level != "dangerous":
            return ""
        needed = frozenset(getattr(meta, "capabilities", frozenset()) or ())
        missing = needed - stage.capabilities
        if missing or not needed:
            return (f"下一步 {step.step_id} 是 DANGEROUS 技能 {step.skill!r},"
                    f"而阶段 {stage.stage_id} 没声明 "
                    f"{sorted(missing) or '任何 capability'} —— 停下来问人,"
                    f"不静默跳过")
        return ""

    # ── 4. 分派 ──────────────────────────────────────────────────────

    def _ignition_hold(self, cid: str, row: dict, rep: TickReport) -> "float | None":
        """窗口还没到 ⇒ 返回要睡多久；可以点火 ⇒ 返回 ``None``。

        窗口写在最后一条 ``approved`` 事件的 payload 里（``ignite_at``，绝对
        时刻）。**没有那个键就是可以点火** —— 旧事件、人批、窗口为 0 三种情况
        都长这样，而它们的正确处理都是「立刻」。

        只在**进入**窗口时记一次事件、通知一次人。每 tick 一条的审计行是没人
        读的审计行，每分钟一次的通知是会被静音的通知，而被静音的通知等于没有
        通知 —— 这条纪律在唤醒调度器的熔断上已经写过一次。
        """
        try:
            # ``events()`` 是升序 + LIMIT ⇒ 取尾拿到的是**最早**那些。一份
            # pause → 再 approve 的 conduct 上，那会读到一个早已过去的
            # ``ignite_at``，撤销窗静默失效。
            last = self.store.last_event(cid, "approved")
        except Exception as exc:  # noqa: BLE001 — 读不到审计流不该拦住点火
            logger.debug("conduct %s: 读 approved 事件失败(%s) —— 按可点火处理", cid, exc)
            return None
        if last is None:
            return None
        payload = last.get("payload") or {}
        raw = payload.get("ignite_at")
        if raw is None:
            return None
        try:
            ignite_at = float(raw)
        except Exception:  # noqa: BLE001
            # **读不懂的时刻不是「可以点火」**，但也不能让一份 conduct 卡死在
            # 一个坏字段上。如实记一行，然后按可点火处理 —— 与「读不到 ≠ 通过」
            # 不同：这里没有安全含义，撤销窗保护的是「人来得及后悔」，而人随时
            # 还可以 abort。
            logger.warning("conduct %s: ignite_at 读不懂(%r) —— 按可点火处理", cid, raw)
            return None

        now = self._clock()
        remaining = ignite_at - now
        if remaining <= 0:
            return None

        # 「这个窗口通知过没有」—— 比的是**这一次批准**，不是「这份 conduct
        # 有史以来」。后者会让 pause → 再 approve 的第二个窗口既不记事件也
        # **不通知用户**：窗口仍然生效，但没人被告知可以撤回，而那条通知
        # 正是这一档的交付物。
        already = False
        try:
            held = self.store.last_event(cid, "ignition_held")
            already = bool(held) and float(
                (held.get("payload") or {}).get("ignite_at") or 0.0) == ignite_at
        except Exception:  # noqa: BLE001
            already = False
        if not already:
            by = str(payload.get("by") or "?")
            secs = int(max(0.0, float(payload.get("ignition_delay_s") or remaining)))
            try:
                self.store.record(cid, "ignition_held",
                                  payload={"by": by, "ignite_at": ignite_at,
                                           "ignition_delay_s": float(secs)})
            except Exception as exc:  # noqa: BLE001
                logger.debug("conduct %s: ignition_held 记不上(%s)", cid, exc)
            self._notify(cid, "ignition_window",
                         f"{by} 已批准这份 conduct，{secs} 秒后点火；"
                         "这段时间里 abort 能把它撤回。")
        rep.did("ignition_held")
        # 睡到窗口结束，但不超过一个 tick 的常规间隔 —— 一份 conduct 在等点火
        # 的时候，abort 意图仍然要被及时消费。
        return max(1.0, min(remaining, 60.0))

    def _dispatch(self, cid: str, row: dict, rep: TickReport) -> None:
        status = row["status"]
        if status in ("draft", "paused"):
            rep.did(f"idle:{status}")
            rep.sleep_hint = 60.0
            return
        if status == "approved":
            # ── 撤销窗（2026-08-27） ─────────────────────────────────
            # ``supervised`` 这一档的全部意义就是这段等待：agent 可以批，人不
            # 必在场，但**来得及后悔**。在这段代码之前，``ignition_delay_s`` /
            # ``ApprovalVerdict.deferred`` 只有生产方没有消费方 —— 两条 approve
            # 路径都只读 ``allowed``，而这里无条件采纳。于是 supervised 在行为
            # 上等于 autonomous，同时 403 的文案还在推荐它。
            #
            # 撤回机制本来就在：``abort`` 在 approved 态是合法操作，走 ``_abort``
            # 进终态。缺的只是「采纳时尊重 ignite_at」这一处。
            held = self._ignition_hold(cid, row, rep)
            if held is not None:
                rep.sleep_hint = held
                return
            spec = self._spec(row)
            # 起点也要跳过治疗段:它排在最前只是因为绕道要按 stage_index 找得到
            # 它,不是因为 conduct 该从换样品修针开始。
            idx = 0
            while idx < len(spec.stages) and getattr(
                    spec.stages[idx], "entered_only_by_detour", False):
                idx += 1
            first = spec.stages[idx] if idx < len(spec.stages) else None
            self.store.record(cid, "adopted", changes={
                "status": "running", "stage_idx": idx,
                "step_idx": AT_ENTRY_GATE if (first and first.entry_gate) else 0},
                # 代次边界①:第一代从被采纳那一刻开始。外部证据(监控告警、
                # 扫描帧)只有时间戳,代次归属靠这个时刻派生 —— 见
                # ``_epoch_started_at``。不写它,第一代就永远答不出开始时刻,
                # 于是所有外部证据都被判成跨代次而**安静地**转人。
                payload={"evidence_epoch": int(row["evidence_epoch"]),
                         EPOCH_START_KEY: self._clock()})
            rep.did("adopted")
            return
        if status == "yielding":
            free = True if self._lock_probe is None else bool(self._lock_probe())
            if free:
                self._busy_since = None
                self.store.record(cid, "status_change", changes={
                    "status": "running", "status_reason": ""})
                rep.did("yield_released")
            else:
                rep.did("still_yielding")
                rep.sleep_hint = 60.0
            return
        if status == "recovery_pending":
            self._recovery_step(cid, row, rep)
            return
        if status in ("waiting_operator", "waiting_condition"):
            # WAITING_OPERATOR 有**两个来源**,别混:
            #   ① 一个 ``wait`` 步 —— 有 active_wait,每 tick 要评双闸;
            #   ② 一次「这事得人来定」的裁决(闸门判不了、步失败、超预算、
            #      自检没过……)—— **没有** active_wait,没有可轮询的闸,
            #      出路是用户的 resume/abort/takeover 意图。
            # 把②当成①的异常,会让每一 tick 都去改写 status_reason,
            # 于是「为什么停」这句话被一条通用的错误消息盖掉 —— 而那句话正是
            # 用户唯一能看的东西。
            if row.get("active_wait"):
                self._evaluate_wait(cid, row, rep)
            else:
                rep.did("waiting_for_operator_decision")
                rep.sleep_hint = 60.0
            return
        if status == "running":
            if self._budget_exceeded(cid, row, rep):
                return
            self._run_position(cid, row, rep)
            return
        rep.did(f"unhandled:{status}")

    def _budget_exceeded(self, cid: str, row: dict, rep: TickReport) -> bool:
        spec = self._spec(row)
        if self._cost_reader is None:
            rep.not_checked.append("USD 预算 —— 没注入 cost_reader;"
                                   "花销以 api_cost_recorder 实测为准,不自造汇率")
            return False
        spent = self._cost_reader(cid)
        if spent is None:
            rep.not_checked.append("USD 预算 —— 读不到花销(读不到 ≠ 花了 0)")
            return False
        if float(spent) <= spec.budgets.usd_max:
            return False
        # 超预算**不硬停**:一个跑了六小时的实验因为差几毛钱被砍掉,比超支更贵。
        self.store.record(cid, "budget_tick", changes={
            "status": "waiting_operator", "budget_spent_usd": float(spent),
            "status_reason": f"花销 ${spent:.2f} 超过上限 "
                             f"${spec.budgets.usd_max:.2f} —— 停下来问人,不硬停"})
        self._notify(cid, "budget", f"conduct 花销 ${spent:.2f} 超上限", "warn")
        rep.did("budget_exceeded")
        return True

    # ── 位置解释(AT_ENTRY_GATE 的唯一懂它的地方)────────────────────

    def _position(self, spec: ConductSpec, row: dict):
        """→ ``(stage, kind, step)``;kind ∈ entry_gate/step/exit_gate/done。"""
        si = int(row["stage_idx"])
        if si >= len(spec.stages):
            return None
        stage = spec.stages[si]
        steps = stage.all_steps
        pi = int(row["step_idx"])
        if pi == AT_ENTRY_GATE:
            return stage, "entry_gate", None
        if pi >= len(steps):
            return stage, "exit_gate", None
        return stage, "step", steps[pi]

    def _run_position(self, cid: str, row: dict, rep: TickReport) -> None:
        spec = self._spec(row)
        pos = self._position(spec, row)
        if pos is None:
            self.store.record(cid, "completed", changes={"status": "completed"})
            self._notify(cid, "completed", "conduct 全部阶段完成")
            rep.did("completed")
            return
        stage, kind, step = pos
        if kind == "entry_gate":
            self._run_gate(cid, row, spec, stage, stage.entry_gate, "entry", rep)
        elif kind == "exit_gate":
            if stage.exit_gate is None:
                self._advance_stage(cid, row, spec, rep)
            else:
                self._run_gate(cid, row, spec, stage, stage.exit_gate, "exit", rep)
        else:
            self._execute_step(cid, row, spec, stage, step, rep)

    # ── 闸门 ─────────────────────────────────────────────────────────

    def _run_gate(self, cid, row, spec, stage, gate, which, rep) -> None:
        evidence, missing = self._collect_evidence(cid, row, gate)
        decide = self._decide_route
        wakes = dict(row.get("llm_wakes") or {})
        used = 0
        cap = int(spec.budgets.llm_wakes_per_stage_max)
        no_judge = "llm 判决器未接入"
        if gate.kind == "llm":
            used = int(wakes.get(stage.stage_id, 0))
            if decide is None:
                rep.did("llm_seat_not_wired")
            elif used >= cap:
                # 唤醒预算用完 ⇒ 判不了,而不是「再唤一次」也不是「就当过了」。
                # **这条不靠调用方记得**:去向由 ``gate`` 自己声明的保守出口决定,
                # 而且计价单位是**阶段**(按 stage_id 记账,不拿 run 数或 tick 数
                # 当代理 —— 那两个由别的东西决定,与「这一段问过几次」无关)。
                decide = None
                # 去向一样,但**说的话要是真的**:一句「判决器未接入」印在一台
                # 明明接好了判决器的机器上,会把人送去查一根没断的线。
                no_judge = (f"阶段 {stage.stage_id} 的 LLM 唤醒预算已用完"
                            f"({used}/{cap})")
                rep.did("llm_wake_budget_spent")
        result = rules.evaluate_gate(gate, evidence, attended=bool(row["attended"]),
                                     missing=missing, decide_route=decide,
                                     no_judge_reason=no_judge)
        if result.llm_used:
            wakes[stage.stage_id] = int(wakes.get(stage.stage_id, 0)) + 1
        payload = {"gate_id": gate.gate_id, "which": which,
                   "kind": gate.kind,
                   "verdict": result.verdict, "route": result.route,
                   "reason": result.reason,
                   "rule_state": result.rule_state,
                   "escaped": result.escaped,
                   "evidence_missing": result.evidence_missing,
                   "missing": list(missing)}
        if gate.kind == "llm":
            # **判决要留痕到人眼前**:哪个模型答的、怎么解析出来的、这一段的
            # 唤醒预算还剩多少。少了这一块,一条 LLM 判决在面板上与一条 rule
            # 判定长得一模一样 —— 而这两者事后要做的核对完全不同。
            payload["llm"] = dict(result.llm_audit or {})
            payload["llm"]["wakes_used"] = int(wakes.get(stage.stage_id, used))
            payload["llm"]["wakes_max"] = cap
            payload["llm"]["responsibility"] = str(
                (gate.llm_node or {}).get("responsibility") or "")
        event_id = self.store.record(
            cid, "gate_evaluated", stage_id=stage.stage_id,
            changes={"llm_wakes": wakes} if result.llm_used else None,
            payload=payload)
        rep.did(f"gate:{gate.gate_id}={result.verdict}")
        row = self.store.get(cid)
        if result.verdict == "pass":
            self._gate_passed(cid, row, spec, which, rep)
            return
        if result.verdict == "wait_operator":
            # **停在这里的那一刻,就要留下「怎么才能继续」所需的一切。**
            # ``decision_id`` = 这一次判定的 ``gate_evaluated`` event_id;用户
            # 的放行必须指名它(见 :meth:`_apply_override_decision`)。没有它,
            # 一份被闸门停住的 conduct 就只剩 abort 与 takeover 两条出路 ——
            # 「能停不能解」在本仓已经付过一次账。
            self._wait_for_operator(
                cid, f"闸门 {gate.gate_id}: {result.reason}", rep,
                decision={"kind": "gate", "decision_id": int(event_id),
                          "gate_id": gate.gate_id, "stage_id": stage.stage_id,
                          "which": which, "verdict": result.verdict,
                          "reason": result.reason})
            return
        if result.verdict == "detour":
            self._enter_detour(cid, row, spec, f"闸门 {gate.gate_id}: {result.reason}",
                               rep)
            return
        self._stage_failed(cid, row, spec, stage,
                           f"闸门 {gate.gate_id} 判定不通过: {result.reason}", rep)

    def _gate_passed(self, cid, row, spec, which: str, rep) -> None:
        """一道闸门放行之后位置怎么走。**只有一份实现** —— 闸门自己判 pass 与
        用户放行走的是同一段代码,否则两条路迟早会在「entry 闸放行之后 step_idx
        是 0 还是 -1」这种地方分岔,而分岔的那一侧没有测试。"""
        if which == "entry":
            self.store.record(cid, "status_change", changes={"step_idx": 0})
        elif which == "step":
            pass            # 位置在 _step_ok 里已经推进过了,放行就是什么都不做
        else:
            self._advance_stage(cid, row, spec, rep)

    def _collect_evidence(self, cid: str, row: dict, gate) -> tuple[dict, tuple]:
        """按 source/selector/max_age_s/min_epoch 收证据。

        **结构过滤**:收不到的进 ``missing``,闸门根本看不到它们,所以不可能拿
        过期或跨代次的证据判 —— 「在自己刚炸出来的坑上判针尖」因此做不到。
        """
        values: dict = {}
        missing: list[str] = []
        now = self._clock()
        produced = self._produced(cid)
        epoch = int(row["evidence_epoch"])
        for ev in gate.evidence:
            if ev.source == "step_data":
                entry = produced.get(ev.selector)
                if entry is None:
                    missing.append(f"step_data:{ev.selector}(没有这一步的产出)")
                    continue
                if ev.max_age_s is not None and (now - entry["ts"]) > ev.max_age_s:
                    missing.append(
                        f"step_data:{ev.selector}(产出已过 "
                        f"{now - entry['ts']:.0f}s > {ev.max_age_s:.0f}s)")
                    continue
                if ev.min_epoch == "current" and int(entry["epoch"]) != epoch:
                    missing.append(
                        f"step_data:{ev.selector}(第 {entry['epoch']} 代证据,"
                        f"当前第 {epoch} 代)")
                    continue
                picked, absent = _project(entry["values"], ev.fields)
                if absent:
                    missing.append(
                        f"step_data:{ev.selector} 的投影落空:{sorted(absent)} "
                        f"—— 这一步产出的是 {sorted(entry['values'])};"
                        f"{PROJECTION_MISS_BLAME}")
                    continue
                values.update(picked)
            elif ev.source == "temperature":
                reading = self.temperature.read()
                fresh = reading.freshness(ev.max_age_s or 0.0)
                if fresh != "fresh":
                    missing.append(f"temperature({fresh}:"
                                   f"{reading.reason or '太旧'})")
                    continue
                values["temperature_k"] = reading.value_k
            elif ev.source in ("monitor_events", "frame_metrics"):
                self._collect_external(cid, row, ev, now, epoch, values, missing)
            else:
                # ``verify_verdict`` 还没接:全仓**没有裁决的持久化真源**
                # (grep 无 verdict registry)。裁决要么在步产出里 —— 那已经是
                # ``step_data``,接成第二个源就是同一个事实的两个真源;要么在
                # S2 逐偏压账本那样的**文件**里,而那条路要先有一份「裁决登记」
                # 的设计。**报成缺席**而不是当没这回事。
                missing.append(f"{ev.source}(本期未接入:全仓没有裁决持久化真源)")
        return values, tuple(missing)

    # ── 外部证据(不是 conduct 自己产的那些)────────────────────────
    #
    # step_data 是 conduct 自己的步产出,Director 在 ``step_finished`` 上**亲手
    # 盖了** ``evidence_epoch``。外部证据没有那一章:监控告警和扫描文件都是别的
    # 链路产的,它们只有**时间戳**。
    #
    # 于是代次这一问只能**派生**:一条时间戳晚于「当前代次开始时刻」的记录属于
    # 当前代次。派生的前提是那个时刻答得出来 —— 答不出来就是**判不了**,
    # 而不是「就当它是当前代次」。「限额计价单位由别处决定就必须派生」是同一条:
    # 别拿 run 数、tick 数或者「反正最近」当代次的代理。

    def _epoch_started_at(self, cid: str, epoch: int) -> "float | None":
        """第 ``epoch`` 代是什么时候开始的。**读不到回 None**。

        从审计流找:凡是**建立代次边界**的事件(``adopted`` / ``detour_entered`` /
        重启清算的 A4)都在 payload 里写下
        ``{"evidence_epoch": n, EPOCH_START_KEY: t}``。取代次相符的最后一条。

        ## 为什么是一个**专用键**,而不是「payload 里同时有 evidence_epoch 和 at」

        因为那个组合**不是这件事独有的**:``step_finished`` 的 payload 里两样
        都有(``{"ok":…, "at": …, "evidence_epoch": …}``)。用它当标记,每一步跑完
        都会被读成一次代次边界,于是「这一代从什么时候开始」一路跟着最后一步的
        完成时刻往前爬 —— 证据窗口越缩越窄,五分钟前那条 critical 告警因为一分钟
        前有一步跑完了而被排除在外。**方向还是开的**:漏掉告警,不是多报告警。

        这个 bug 是变异测试逼出来的(把 ``max_age`` 那道下界拿掉,测试本该变红却
        没有 —— 一查,窗口下界根本不是代次开始时刻)。标记要**专用**,别拿一个
        「碰巧只有它有」的组合当标记。

        ## 为什么不从 ``changes`` 里找

        ``conduct_events`` **不持久化 changes**(只有 kind/ts/stage/step/run/
        payload)。靠 changes 找等于靠一个不存在的列找,而那种找法会安静地永远
        找不到 —— 每条外部证据都被判成跨代次,闸门每次转人,没人知道为什么。
        """
        want = None
        for ev in self.store.events(cid, limit=4000):
            p = ev.get("payload") or {}
            if EPOCH_START_KEY not in p or "evidence_epoch" not in p:
                continue
            try:
                if int(p["evidence_epoch"]) == int(epoch):
                    want = float(p[EPOCH_START_KEY])
            except (TypeError, ValueError):
                # 写坏了的一条不该让整个查询无声地退回 None —— 跳过它,
                # 继续找后面的。找不到才是 None。
                continue
        return want

    def _collect_external(self, cid, row, ev, now: float, epoch: int,
                          values: dict, missing: list) -> None:
        """一条外部证据:探针没接 / 读不到 / 跨代次 —— 三种都进 ``missing``。

        **绝不把「读不到」折成空列表。** 一个空的告警列表会被闸门读成「查过了,
        这段时间没有事件」,而真相是「监控根本没在跑」—— 那两件事对「要不要继续
        往下扫」给的是相反的答案。
        """
        probe = (self._monitor_probe if ev.source == "monitor_events"
                 else self._frame_metrics_probe)
        if probe is None:
            missing.append(f"{ev.source}(本期未接入探针)")
            return
        started = self._epoch_started_at(cid, epoch)
        if started is None:
            # 代次答不出来 ⇒ 这条证据的代次归属答不出来 ⇒ **判不了**。
            missing.append(f"{ev.source}(答不出第 {epoch} 代是什么时候开始的,"
                           f"无法判定证据是不是当代的)")
            return
        # 窗口 = 代次开始 与 max_age 两个下界里**晚的那个**。两道都要:代次管
        # 「针换过没有」,年龄管「这条消息还新鲜吗」,互相替代不了。
        since = started
        if ev.max_age_s is not None:
            since = max(since, now - float(ev.max_age_s))
        try:
            got = probe(ev.selector, since)
        except Exception as exc:  # noqa: BLE001 —— 探针炸了是「读不到」,不是「没有」
            missing.append(f"{ev.source}(探针抛异常: {exc})")
            return
        if got is None:
            missing.append(f"{ev.source}(读不到 —— 读不到不等于没有)")
            return
        if not isinstance(got, dict):
            missing.append(f"{ev.source}(探针回了 {type(got).__name__},不是证据 dict)")
            return
        values.update(got)

    def _produced(self, cid: str) -> dict:
        """从审计流重建每一步的产出。

        产出不另开一张表:``step_finished`` 事件本来就是 append-only 的真相,
        而且**天然跨重启存活**。重跑一步会写第二条,按序覆盖 —— 后写的赢,
        与执行顺序一致。
        """
        out: dict = {}
        for ev in self.store.events(cid, kind="step_finished", limit=2000):
            p = ev.get("payload") or {}
            sid = ev.get("step_id") or p.get("step_id")
            if not sid:
                continue
            out[sid] = {"values": dict(p.get("produces") or {}),
                        "ts": float(p.get("at") or 0.0),
                        "epoch": int(p.get("evidence_epoch") or 0)}
        return out

    def _produced_flat(self, cid: str) -> dict:
        return self._flat_with_frames(cid)[0]

    def _flat_with_frames(self, cid: str) -> "tuple[dict, dict]":
        """``(扁平产出表, {扁平键: (产出步, 那一步声明的 coord_epoch 或 None)})``。

        ## 「那一步声明的 coord_epoch」是什么

        产出步如果在自己的 data 里放了一个 ``coord_epoch``,那就是它在**声明自己
        的输出属于哪一代坐标系**(``analyses.plan_bias_series`` 逐字这么做:
        「查不到代次就不盖,不盖 0」;``FindCleanSpot`` 的 provenance 块里也有)。
        没放 = 它从没声称过一个坐标系,那就没有东西可核对 —— 那是**没检查**,
        不是检查通过,由 :meth:`_check_coord_frame` 如实记一条。

        **为什么归属由产出方声明,而不是引擎去猜**:引擎看到的只是一个名字。
        ``positions_json`` 是坐标、``n_points`` 是个计数,而绑定这一层分不开它们。
        一条「凡是 steps.* 绑定都查代次」的规则会在计数上误拦,一条「都不查」的
        规则会把坐标喂进硬件 —— 两个都错。产出方知道自己产的是什么。
        """
        flat: dict = {}
        frames: dict = {}
        for sid, entry in self._produced(cid).items():
            stamp = entry["values"].get("coord_epoch")
            for k, v in entry["values"].items():
                key = f"steps.{sid}.{k}"
                flat[key] = v
                frames[key] = (sid, stamp)
        return flat, frames

    # ── 步 ───────────────────────────────────────────────────────────

    def _execute_step(self, cid, row, spec, stage, step: StepSpec, rep) -> None:
        if step.kind == "wait":
            self._enter_wait(cid, row, stage, step, rep)
            return
        frame_notes: list = []
        try:
            params = self._resolve_params(cid, row, step, notes=frame_notes)
        except KeyError as exc:
            # 绑定解析失败 = 步失败,**没有默认值兜底**。一个「取不到就用 0」
            # 的绑定,会把一次读失败变成一次看起来正常的运行。
            self._step_failed(cid, row, spec, stage, step, str(exc), rep)
            return
        for note in frame_notes:
            if note.get("unprotected"):
                # 「这一次没有代次保护」进 TickReport 的 not_checked —— 与
                # 「核对过、对得上」在事后必须分得开。
                rep.not_checked.append(
                    f"{step.step_id} 的绑定 {note['binding']}:{note['why']}")

        if step.kind == "analysis":
            self._run_analysis(cid, row, spec, stage, step, params, rep,
                               frame_notes=frame_notes)
            return
        self._run_skill(cid, row, spec, stage, step, params, rep,
                        frame_notes=frame_notes)

    def _resolve_params(self, cid: str, row: dict, step: StepSpec,
                        *, flat: "dict | None" = None,
                        frames: "dict | None" = None,
                        notes: "list | None" = None) -> dict:
        """把一步的参数解出来。

        ``flat`` 只给恢复复验步用(见 ``_check_tip``):它们的产出在审计流里带着
        ``recovery.A3#n.`` 前缀,而 spec 里的绑定写的是兄弟步的**本名** ——
        前缀是执行体加的,不该漏到模板里去。

        ``notes`` 收**代次核对说了什么**(见 :meth:`_check_coord_frame`),由调用方
        放进 ``step_started`` 的 payload:「这一次没有代次保护」必须留在纸面上,
        否则它和「核对过、对得上」在事后看长得一模一样。
        """
        params = dict(step.params)
        if flat is None:
            flat, auto_frames = self._flat_with_frames(cid)
        else:
            flat, auto_frames = dict(flat), {}
        frames = auto_frames if frames is None else dict(frames)
        camp_params = dict(row.get("params") or {})
        for name, ref in step.bindings.items():
            if ref.startswith("params."):
                key = ref[len("params."):]
                if key not in camp_params:
                    raise KeyError(f"绑定 {name}←{ref} 取不到:conduct 参数里没有 "
                                   f"{key!r}(有 {sorted(camp_params)})")
                params[name] = camp_params[key]
            else:
                if ref not in flat:
                    raise KeyError(f"绑定 {name}←{ref} 取不到:上游还没产出它,"
                                   f"而 binding 没有默认值兜底")
                self._check_coord_frame(step, name, ref, frames.get(ref), notes)
                params[name] = flat[ref]
        return params

    def _check_coord_frame(self, step: StepSpec, name: str, ref: str,
                           frame, notes: "list | None") -> None:
        """一条 ``steps.*`` 绑定的**坐标代次**核对(修复项 的消费侧,引擎这一层)。

        ## 为什么是 ``coord_epoch`` 而不是 ``evidence_epoch``

        两个代次名字像,量的是两件事,而**这里只有一个是对的**:

        * ``coord_epoch`` = 本作用域粗动过几次(COUNT ``coarse_move`` 标记)。
          横向粗动之后同一个 ``(x, y)`` 指的是**另一片表面** —— 那正是一个坐标
          会变错的唯一原因。它由记录派生,**跨进程重启不变**。
        * ``evidence_epoch`` = conduct 自己的账,只在**重启清算**和**进绕道**时
          +1(全仓只有两处写它)。

        于是拿 ``evidence_epoch`` 当绑定的守卫会**两头错**:粗动不 bump 它 ⇒
        真正的危险漏过;重启/绕道 bump 它而坐标其实好好的 ⇒ 平白拦下一份能续的
        conduct。**一个稳定地在量另一件事的判据,可重复也没用。**

        ## 四态各走各的,不折叠

        照 :mod:`mast.core.coord_epoch` 的闭集,而且**尊重它 docstring 里的告诫**
        (「不要把 UNVERIFIABLE 和 UNSTAMPED 当成陈旧拒掉:那会让『读不到记录』
        变成一道解不开的闸」):

        * ``MATCH``        —— 放行;
        * ``STALE``        —— **拒绝**(抛 KeyError ⇒ 步失败 ⇒ 阶段 ``on_fail``)。
          绑定喂的是「往哪儿开」,不是「要不要继续」;闸门才有「判不了」那一档,
          参数没有,只有「不开」。该模块自己也写着:陈旧坐标的处置只有拒绝,
          **既不夹紧也不换算**;
        * ``UNVERIFIABLE`` —— **放行 + 记一条**。记录存储读不到时拦下来,会造出
          一道用户解不开的闸(那正是上面那句告诫);而真正把针开过去的那一层
          (``SpectroscopyAtPositions`` 只放行确凿的 MATCH)比这里严,严格该待在
          硬件边界上,不该在这里再摞一层解不开的;
        * ``UNSTAMPED``    —— **放行 + 记一条**。产出方从没声称过一个坐标系,
          没有东西可核对。这是**没检查**,不是检查通过 —— 所以要留痕。
        """
        stamp = frame[1] if frame else None
        producer = frame[0] if frame else "?"
        from mast.core import coord_epoch as ce

        verdict = ce.verify(stamp, what=f"{producer} 产出的 {ref}")
        if verdict.state == ce.MATCH:
            if notes is not None:
                notes.append({"binding": name, "ref": ref, "state": verdict.state,
                              "coord_epoch": verdict.stamped})
            return
        if verdict.state == ce.STALE:
            raise KeyError(
                f"绑定 {name}←{ref} **拒绝**:{verdict.message}"
                f"(步 {step.step_id} 会把这个数下发到仪器)")
        if notes is not None:
            notes.append({"binding": name, "ref": ref, "state": verdict.state,
                          "coord_epoch": verdict.stamped,
                          "unprotected": True, "why": verdict.message})

    def _run_analysis(self, cid, row, spec, stage, step, params, rep,
                      *, frame_notes: "list | None" = None) -> None:
        try:
            fn = self._analyses_get(step.analysis_fn)
            produced = fn(params) or {}
        except Exception as exc:  # noqa: BLE001 —— 分析失败是步失败,不是崩溃
            self._step_failed(cid, row, spec, stage, step,
                              f"分析 {step.analysis_fn} 失败: {exc}", rep)
            return
        gap = [k for k in step.produces if k not in produced]
        if gap:
            # 声明了产出却没给 —— 下游那条 binding 会在运行时取不到。
            self._step_failed(cid, row, spec, stage, step,
                              f"分析 {step.analysis_fn} 没有产出它声明的 {gap}", rep)
            return
        self._step_ok(cid, row, stage, step, produced, rep, run_id="", spec=spec)

    def _run_skill(self, cid, row, spec, stage, step, params, rep,
                   *, frame_notes: "list | None" = None) -> None:
        run_id = uuid.uuid4().hex[:12]
        self.aborts.register(run_id)
        self.store.record(cid, "step_started", stage_id=stage.stage_id,
                          step_id=step.step_id, run_id=run_id,
                          changes={"active_run_id": run_id},
                          payload={"skill": step.skill, "params": params,
                                   # 坐标代次核对说了什么 —— **包括「这一次没有
                                   # 代次保护」**。少了它,事后对账分不出「核对过、
                                   # 对得上」与「根本没有章可核对」。
                                   "coord_frames": list(frame_notes or ()),
                                   "at": self._clock()})
        try:
            outcome = self.executor.run(step.skill, params, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            outcome = StepOutcome(ok=False, error=f"executor 抛异常: {exc}",
                                  run_id=run_id)
        finally:
            # 注销一律走 finally:成败都清。留着的话,下一次 abort 会置一个
            # 早就结束的 run 的 Event,而真正在跑的那个没人管。
            self.aborts.unregister(run_id)
            self.store.record(cid, "status_change", changes={"active_run_id": ""},
                              payload={"cleared_run_id": run_id})

        if outcome.busy:
            self._on_busy(cid, stage, step, rep)
            return
        if outcome.ok:
            self._busy_since = None
            self._step_ok(cid, self.store.get(cid), stage, step,
                          dict(outcome.data or {}), rep, run_id=run_id,
                          spec=spec)
            return
        why = outcome.error or "步失败"
        if outcome.tip_event or outcome.crash:
            # 针尖事件/撞针必然挂闩 ⇒ 下一 tick 的 pre-flight 会拦下来。
            # 这里只如实记账,**不抢救济**(watchdog 的处置序列是独立的)。
            #
            # 这面旗给 :meth:`_wait_for_operator` 看:那里 2026-08-21 补了
            # 「等人之前先退针」,而这一路**不许**退 —— 处置归 watchdog,
            # 两个主人同时动仪器比不退更坏。
            #
            # 为什么不读急停闩:闩由 watchdog 挂,而这一 tick 里它可能还没挂上
            # (测试替身里更是从来不挂)。**要读的是「这一步出了针尖事件」这个
            # 事实本身**,不是它的下游后果。
            self._rescue_owned_elsewhere = True
            why = ("扫描中检测到针尖事件" if outcome.tip_event else "检测到撞针") \
                  + (f": {outcome.error}" if outcome.error else "")
        self._step_failed(cid, self.store.get(cid), spec, stage, step, why, rep,
                          run_id=run_id)

    def _on_busy(self, cid, stage, step, rep) -> None:
        now = self._clock()
        if self._busy_since is None:
            self._busy_since = now
        waited = now - self._busy_since
        if waited >= BUSY_YIELD_AFTER_S:
            self.store.record(cid, "status_change", changes={
                "status": "yielding",
                "status_reason": f"仪器令牌被别的链路占用超过 "
                                 f"{waited / 60:.0f} min —— 让路,不排队"})
            self._notify(cid, "yielding", "别的链路长期占用仪器,conduct 让路",
                         "warn")
            rep.did("yielding")
            return
        # 拒绝不排队:本 tick 放弃,下 tick 再试。
        rep.did("busy_retry_next_tick")
        rep.sleep_hint = 5.0

    def _step_ok(self, cid, row, stage, step, produced: dict, rep, *, run_id,
                 spec=None) -> None:
        self.store.record(cid, "step_finished", stage_id=stage.stage_id,
                          step_id=step.step_id, run_id=run_id,
                          changes={"step_idx": int(row["step_idx"]) + 1},
                          payload={"ok": True, "produces": produced,
                                   "at": self._clock(),
                                   "evidence_epoch": int(row["evidence_epoch"])})
        rep.did(f"step_ok:{step.step_id}")
        # 步级闸门:这一步成功之后**立刻**判一次。位置已经推进过了,所以
        # ``pass`` 什么都不用做;其余裁决与阶段闸门同义(等人/绕道/阶段失败)。
        #
        # 为什么要有它:阶段只有首尾两个闸位,而修针这种流程中间就得判 ——
        # forge 没成还继续走完「换回样品 → 等人 → 进针 → 复验」,是白烧一次
        # 用户往返。以前的替代办法是把阶段拆两半,但绕道的返回时机按**阶段**
        # 算,拆开之后第二半永远不跑。
        if step.gate is not None and spec is not None:
            self._run_gate(cid, self.store.get(cid), spec, stage, step.gate,
                           "step", rep)

    def _step_failed(self, cid, row, spec, stage, step, why, rep, *, run_id="") -> None:
        attempts = self._attempts(cid, step.step_id)
        self.store.record(cid, "step_finished", stage_id=stage.stage_id,
                          step_id=step.step_id, run_id=run_id,
                          payload={"ok": False, "why": why, "attempts": attempts,
                                   "at": self._clock(),
                                   "evidence_epoch": int(row["evidence_epoch"])})
        if step.optional:
            # optional 步失败不拖垮阶段(如逐点质量评估:谱已经采到了,
            # 判不了就是判不了)。
            self.store.record(cid, "status_change",
                              changes={"step_idx": int(row["step_idx"]) + 1})
            rep.did(f"step_failed_optional:{step.step_id}")
            return
        if attempts <= step.retries:
            rep.did(f"step_retry:{step.step_id}({attempts}/{step.retries})")
            return
        self._stage_failed(cid, self.store.get(cid), spec, stage,
                           f"步 {step.step_id} 失败: {why}", rep)

    def _attempts(self, cid: str, step_id: str) -> int:
        """这一步**自上次成功以来**已经试了几次。

        从审计流数,不在内存里记:重启之后内存计数会归零,于是一个反复失败的步
        会获得一整套新的重试预算。
        """
        n = 0
        for ev in self.store.events(cid, limit=4000):
            if ev.get("step_id") != step_id:
                continue
            if ev["kind"] == "step_started":
                n += 1
            elif ev["kind"] == "step_finished" and (ev.get("payload") or {}).get("ok"):
                n = 0
        return max(n, 1)

    # ── 阶段推进 / 失败 / 绕道 ───────────────────────────────────────

    def _advance_stage(self, cid, row, spec, rep) -> None:
        detour = row.get("detour") or None
        si = int(row["stage_idx"])
        # 只有**绕道目标阶段自己**跑完才算绕道结束。以前这里是「detour 活跃 +
        # 任意阶段跑完 ⇒ 立刻返回」,于是治疗段一旦拆成两个 stage,第二个永远
        # 不跑 —— 而且没有任何地方会说话。
        target_idx = (spec.stage_index(spec.detour.target_stage)
                      if str(spec.detour.target_stage).strip() else None)
        if detour and target_idx is not None and si == target_idx:
            self._return_from_detour(cid, row, spec, detour, rep)
            return
        nxt = si + 1
        # 治疗段正常流程一步都不进 —— 它只等绕道。跳过它不是优化,是语义:
        # 排在最前会让每次 conduct 先做一整轮换样品修针,排在最后会在实验做完
        # 之后再修一次针,两个都是荒谬的。见 StageSpec.entered_only_by_detour。
        while nxt < len(spec.stages) and getattr(
                spec.stages[nxt], "entered_only_by_detour", False):
            rep.did(f"stage_skipped_detour_only:{spec.stages[nxt].stage_id}")
            nxt += 1
        if nxt >= len(spec.stages):
            self.store.record(cid, "completed", changes={
                "status": "completed", "stage_idx": nxt})
            self._notify(cid, "completed", "conduct 全部阶段完成")
            rep.did("completed")
            return
        stage = spec.stages[nxt]
        self.store.record(cid, "status_change", changes={
            "stage_idx": nxt,
            "step_idx": AT_ENTRY_GATE if stage.entry_gate else 0})
        rep.did(f"stage_advance:{stage.stage_id}")

    def _stage_failed(self, cid, row, spec, stage: StageSpec, why: str, rep) -> None:
        then = stage.on_fail.then
        if then == "detour":
            self._enter_detour(cid, row, spec, why, rep)
        elif then == "abort":
            self._abort(cid, row, f"阶段 {stage.stage_id} 失败: {why}", "policy", rep)
        elif then == "skip":
            rep.did(f"stage_skipped:{stage.stage_id}")
            self.store.record(cid, "status_change", stage_id=stage.stage_id,
                              payload={"skipped_because": why})
            self._advance_stage(cid, self.store.get(cid), spec, rep)
        elif then == "escalate":
            self._escalate(cid, row, spec, stage, why, rep)
        else:
            self._wait_for_operator(cid, why, rep)

    # ── L2 值守席 ────────────────────────────────────────────────────

    def _escalate(self, cid, row, spec, stage: StageSpec, why: str, rep) -> None:
        """叫醒一次只读诊断,拿一个**模板已经授权过**的处置。

        四条纪律,每一条都对应一种「看起来像做过了」的失败:

        1. **没席位 ⇒ 说出来**。advisor 没注入时不假装升级过,退回问人并写明
           原因 —— 一句「已升级」而实际什么都没发生,会让下一个人去查诊断结果。
        2. **每阶段每证据代次至多一次**,次数从审计流数(重启不归零)。一次
           L2 是一整个 agent run;让它在一个反复失败的阶段上循环,烧的是钱和
           时间,而每一轮的输入几乎一样。
        3. **越权 ⇒ 转人**,并且**按越权记**。模型提议了一件模板没授权的事,
           这件事本身值得被看见;把它悄悄改写成 ``wait_operator``,记录里会
           长得像「模型建议叫人」。
        4. **判不了 ⇒ 转人**(``SeatUnavailable``:席位坏了/超时/回不出闭集)。
           fail_silent 在这里的伤害面正好是「少拿一条建议」,这也是 L2 敢用
           一次 agent run 的全部理由。
        """
        allowed = tuple(sorted(stage.allowed_escalations))
        if self._escalation_advisor is None:
            self._wait_for_operator(
                cid, f"{why}(阶段策略是 escalate,但这台机器上没有接 L2 诊断席 —— "
                     f"停下来问人)", rep)
            return

        epoch = int(row.get("evidence_epoch") or 0)
        used = 0
        for ev in self.store.events(cid, kind="escalation_started", limit=500):
            p = ev.get("payload") or {}
            if p.get("stage_id") == stage.stage_id and int(p.get("epoch") or 0) == epoch:
                used += 1
        if used >= 1:
            self._wait_for_operator(
                cid, f"{why}(这一阶段在本证据代次里已经诊断过一次 —— "
                     f"同样的输入不会给出不同的答案,停下来问人)", rep)
            return

        from mast.conduct.l2_seat import EscalationContext
        from mast.conduct.llm_seat import SeatUnavailable

        # 先记「叫醒了」,再去问。顺序是有意的:如果这次调用把进程带走了,
        # 重启之后计数里仍然有这一条 —— 否则一个每次都让 Director 崩溃的
        # 诊断会在每次重启后重来一遍。
        self.store.record(cid, "escalation_started", stage_id=stage.stage_id,
                          payload={"stage_id": stage.stage_id, "epoch": epoch,
                                   "why": why, "allowed": list(allowed)})

        recent = tuple(
            f"{ev.get('kind')}: {str((ev.get('payload') or {}).get('why') or '')[:80]}"
            for ev in self.store.events(cid, limit=12))
        ctx = EscalationContext(
            conduct_id=cid, stage_id=stage.stage_id, why=why,
            attempts=used + 1, evidence_epoch=epoch,
            allowed=allowed, recent=recent)

        try:
            advice = self._escalation_advisor(ctx)
        except SeatUnavailable as exc:
            self.store.record(cid, "escalation_verdict", stage_id=stage.stage_id,
                              payload={"stage_id": stage.stage_id, "epoch": epoch,
                                       "ok": False, "why": why,
                                       "unavailable": str(exc)})
            self._wait_for_operator(cid, f"{why}(L2 诊断没成: {exc})", rep)
            return
        except Exception as exc:  # noqa: BLE001 —— 顾问坏了不该弄停指挥线程
            self.store.record(cid, "escalation_verdict", stage_id=stage.stage_id,
                              payload={"stage_id": stage.stage_id, "epoch": epoch,
                                       "ok": False, "why": why,
                                       "error": str(exc)})
            self._wait_for_operator(cid, f"{why}(L2 诊断抛异常: {exc})", rep)
            return

        route = str(getattr(advice, "route", "") or "")
        self.store.record(cid, "escalation_verdict", stage_id=stage.stage_id,
                          payload={"stage_id": stage.stage_id, "epoch": epoch,
                                   "ok": True, "why": why, "route": route,
                                   "reason": getattr(advice, "reason", ""),
                                   "looked_at": list(getattr(advice, "looked_at", ())),
                                   "allowed": list(allowed)})
        rep.did(f"escalation:{stage.stage_id}:{route}")

        # 包络内 ⇒ 直接执行。这就是「授权包络」的全部含义:模板作者写进
        # allowed_escalations 的处置,这一席说了算。
        if route == "detour":
            self._enter_detour(cid, row, spec, f"L2: {advice.reason or why}", rep)
        elif route == "abort":
            self._abort(cid, row, f"L2 判定中止 {stage.stage_id}: "
                                  f"{advice.reason or why}", "l2", rep)
        elif route == "skip_stage":
            if stage.mandatory:
                # `mandatory` 在此之前是个零执行读者的字段(只有 API 与 journal
                # 拿去显示)。它的语义正是这一句:必做的段不许被跳过 —— 哪怕
                # 模板把 skip_stage 写进了 allowed。
                self._wait_for_operator(
                    cid, f"{why}(L2 建议跳过 {stage.stage_id},但它是 mandatory —— "
                         f"必做的段不许跳,停下来问人)", rep)
                return
            rep.did(f"stage_skipped:{stage.stage_id}")
            self.store.record(cid, "status_change", stage_id=stage.stage_id,
                              payload={"skipped_because": f"L2: {advice.reason or why}"})
            self._advance_stage(cid, self.store.get(cid), spec, rep)
        elif route == "continue_retry":
            # 什么都不做 = 下一 tick 从同一个位置再来一次。**不清计数**:
            # 重试预算是防「反复失败」的,不该被一次诊断重置。
            self.store.record(cid, "status_change", stage_id=stage.stage_id,
                              payload={"l2_retry": advice.reason or why})
        else:
            self._wait_for_operator(
                cid, f"{why}(L2 判定 {route or '(空)'} —— 交给人)", rep)

    def _enter_detour(self, cid, row, spec, reason: str, rep) -> None:
        policy = spec.detour
        target_idx = spec.stage_index(policy.target_stage) if policy.target_stage else None
        if target_idx is None:
            self._wait_for_operator(
                cid, f"{reason} —— 本 spec 没有修针段,绕道无处可去,请人处理", rep)
            return
        done = len(self.store.events(cid, kind="detour_entered", limit=500))
        if done >= policy.max_detours_per_conduct:
            # 熔断:修针 ping-pong 会烧一整夜机时和一根针。
            self._wait_for_operator(
                cid, f"{reason} —— 绕道次数已达上限 "
                     f"{policy.max_detours_per_conduct},熔断,请人处理", rep)
            return
        epoch = int(row["evidence_epoch"]) + 1
        detour = {"return_stage_idx": int(row["stage_idx"]),
                  "return_step_idx": int(row["step_idx"]),
                  "reason": reason, "entered_at": self._clock()}
        # 进 detour 即 bump epoch:坏针之前采的证据全部作废。这是「在自己刚炸
        # 出来的坑上判针尖」的结构化预防 —— 闸门只认当前代次的证据。
        #
        # 跳目标阶段第 0 步。校验器保证的**不再是**「第 0 步是退针技能」,而是
        # 「退针之前没有会改变表面的动作」(§5b)—— 只读诊断可以排在退针前面,
        # 那是「先确认真是针坏了」与「立刻付换样品的钱」之间的差别。
        target = spec.stages[target_idx]
        self.store.record(cid, "detour_entered", changes={
            "detour": detour, "evidence_epoch": epoch,
            "stage_idx": target_idx,
            "step_idx": AT_ENTRY_GATE if target.entry_gate else 0},
            # 代次边界②。``at`` 与 ``evidence_epoch`` **成对**才有用:一个说
            # 「第几代」,一个说「从什么时候起」。少了 ``at``,外部证据的代次
            # 归属就派生不出来。
            payload={"reason": reason, "evidence_epoch": epoch,
                     EPOCH_START_KEY: self._clock()})
        rep.did("detour_entered")

    def _return_from_detour(self, cid, row, spec, detour: dict, rep) -> None:
        back = int(detour.get("return_stage_idx", 0))
        stage = spec.stages[back] if back < len(spec.stages) else None
        on_return = spec.detour.on_return
        if on_return == "restart_stage":
            step_idx = AT_ENTRY_GATE if (stage and stage.entry_gate) else 0
        elif on_return == "resume_step":
            step_idx = int(detour.get("return_step_idx", 0))
        else:  # gate_recheck —— 默认:回去先过一遍入口闸门再续
            step_idx = AT_ENTRY_GATE if (stage and stage.entry_gate) else \
                int(detour.get("return_step_idx", 0))
        self.store.record(cid, "detour_returned", changes={
            "detour": None, "stage_idx": back, "step_idx": step_idx},
            payload={"on_return": on_return})
        rep.did("detour_returned")

    # ── 等待 ─────────────────────────────────────────────────────────

    def _enter_wait(self, cid, row, stage, step: StepSpec, rep) -> None:
        w = step.wait
        wait_id = uuid.uuid4().hex[:8]
        now = self._clock()
        cond_numbers: "dict | None" = None
        if w.condition is not None:
            try:
                cond_numbers = w.condition.resolve(dict(row.get("params") or {}))
            except KeyError as exc:
                # 条件的数取不到 ⇒ **步失败**,不进等待。拿模板里的占位值去等,
                # 等的就不是人填的那个条件了 —— 而且那件事在任何日志上都对不出来。
                spec = self._spec(row)
                self._step_failed(cid, row, spec, stage, step, str(exc), rep)
                return
        active = {"wait_id": wait_id, "kind": w.kind, "message": w.message,
                  "ack_required": bool(w.ack_required),
                  "ack_at": None, "ack_by": "",
                  "condition_met_since": None, "last_notified_at": now,
                  "entered_at": now, "step_id": step.step_id,
                  "last_reading": None, "last_reading_age_s": None}
        if cond_numbers is not None:
            # 解出来的数**存进等待记录**:判定、面板、审计从此看同一组数字,
            # 而不是各自再从 spec+params 推一遍(推两遍就会有两个答案)。
            active["cond_value"] = cond_numbers["value"]
            active["cond_stale_after_s"] = cond_numbers["stale_after_s"]
            active["cond_hold_s"] = cond_numbers["hold_s"]
        status = "waiting_operator" if w.ack_required else "waiting_condition"
        self.store.record(cid, "wait_entered", stage_id=stage.stage_id,
                          step_id=step.step_id, changes={
                              "status": status, "active_wait": active,
                              "status_reason": w.message},
                          payload={"wait_id": wait_id, "kind": w.kind})
        req_id = self.notifier.request_operator_action(Notification(
            kind="wait", conduct_id=cid, message=w.message,
            severity="info", payload={"wait_id": wait_id, "kind": w.kind}))
        if req_id:
            # 记下外部请求 id。重播与**重启后重发**都先拿它问一句「那条还开着
            # 吗」—— 幂等靠一个真的 id,不靠比对文案(文案里带着「还缺什么」,
            # 每次都不一样)。
            active["request_id"] = str(req_id)
            self.store.record(cid, "status_change",
                              changes={"active_wait": active},
                              payload={"wait_id": wait_id, "request_id": req_id})
        rep.did(f"wait_entered:{wait_id}")

    def _evaluate_wait(self, cid, row, rep) -> None:
        spec = self._spec(row)
        active = dict(row.get("active_wait") or {})
        if not active:
            # 走不到这里(``_dispatch`` 已经分流)。真到了就是库被外部动过 ——
            # **不改状态**:把「为什么停」那句话保住,只留一条日志。
            logger.warning("conduct %s 在 %s 却没有 active_wait —— 不动状态",
                           cid, row["status"])
            rep.did("wait_record_missing")
            return
        pos = self._position(spec, row)
        wait_spec = pos[2].wait if (pos and pos[2] is not None) else None
        now = self._clock()
        changed = False

        ack_ok = (not active.get("ack_required")) or active.get("ack_at") is not None

        cond_ok = True
        cond = wait_spec.condition if wait_spec else None
        if cond is not None:
            if active.get("waived_at"):
                cond_ok = True
                rep.did("condition_waived")
            else:
                # 用**进等待那一刻解出来的**三个数(见 ``_enter_wait``)。
                # 老记录(没有解析结果)退回 spec 上的字面值 —— 那些条件本来就
                # 没有绑定,字面值就是它的真值。
                c_value = float(active.get("cond_value", cond.value))
                c_stale = float(active.get("cond_stale_after_s",
                                           cond.stale_after_s))
                c_hold = float(active.get("cond_hold_s", cond.hold_s))
                reading = self.temperature.read()
                fresh = reading.freshness(c_stale)
                active["last_reading"] = reading.value_k
                active["last_reading_age_s"] = reading.age_s
                changed = True
                if fresh != "fresh":
                    # **stale = 读不到 ≠ 没到。** 绝不当「条件不满足」干等下去。
                    cond_ok = False
                    if row["status"] == "waiting_condition":
                        self.store.record(cid, "status_change", changes={
                            "status": "waiting_operator",
                            "active_wait": active,
                            "status_reason": f"温度读数{'太旧' if fresh == 'stale' else '读不到'}"
                                             f"({reading.reason or ''}) —— "
                                             f"读不到不等于没到,请人来看"})
                        self._notify(cid, "stale",
                                     "温度读不到/太旧,等待降级为要人来看", "warn")
                        rep.did("condition_stale")
                        return
                    rep.did("condition_stale_still_waiting")
                else:
                    ok = (reading.value_k <= c_value if cond.op == "<="
                          else reading.value_k >= c_value)
                    if ok:
                        if active.get("condition_met_since") is None:
                            active["condition_met_since"] = now
                            self.store.record(cid, "wait_condition_met",
                                              changes={"active_wait": active},
                                              payload={"value": reading.value_k,
                                                       "threshold": c_value})
                            changed = False
                        cond_ok = (now - float(active["condition_met_since"])
                                   >= c_hold)
                    else:
                        # 回升即清:降到位又升回去不算到位。
                        if active.get("condition_met_since") is not None:
                            active["condition_met_since"] = None
                            changed = True
                        cond_ok = False

        if ack_ok and cond_ok:
            step_idx = int(row["step_idx"]) + 1
            self.store.record(cid, "wait_released", changes={
                "status": "running", "status_reason": "",
                "active_wait": None, "step_idx": step_idx},
                payload={"wait_id": active.get("wait_id"),
                         "ack_by": active.get("ack_by"),
                         "waived_by": active.get("waived_by")})
            rep.did("wait_released")
            return

        # 还缺哪个闸,说清楚(面板照这个显示)。
        lacking = []
        if not ack_ok:
            lacking.append("人的确认")
        if not cond_ok:
            lacking.append("物理条件")
        renotify = wait_spec.renotify_every_s if wait_spec else 14400.0
        last = float(active.get("last_notified_at") or 0.0)
        if now - last >= renotify:
            active["last_notified_at"] = now
            changed = True
            req_id = self.notifier.request_operator_action(Notification(
                kind="wait_renotify", conduct_id=cid,
                message=f"{active.get('message', '')}(还缺:{'、'.join(lacking)})",
                payload={"wait_id": active.get("wait_id"),
                         "request_id": active.get("request_id") or ""}))
            if req_id:
                active["request_id"] = str(req_id)
            rep.did("renotified")
        if wait_spec and wait_spec.max_wait_s is not None:
            waited = now - float(active.get("entered_at") or now)
            if waited > wait_spec.max_wait_s and not active.get("max_wait_notified"):
                active["max_wait_notified"] = True
                changed = True
                # 超时只**升级通知,不放弃** —— 等人没有 fail-closed。
                self._notify(cid, "wait_overdue",
                             f"等待已超过 {wait_spec.max_wait_s / 3600:.1f} h,"
                             f"仍在等(还缺:{'、'.join(lacking)})", "warn")
                rep.did("max_wait_notified")
        if changed:
            self.store.record(cid, "status_change", changes={"active_wait": active},
                              payload={"lacking": lacking})
        rep.did(f"waiting:{'+'.join(lacking)}")
        rep.sleep_hint = spec.budgets.wait_tick_interval_s

    # ── 恢复自检(§8)────────────────────────────────────────────────

    def reconcile_after_restart(self) -> "TickReport":
        """进程重启后调一次(挂点在 ``service.start``,**起线程之前**)。

        §5 的 restart 行:非终态 → RECOVERY_PENDING,**但 PAUSED / DRAFT /
        APPROVED 原样保留**(尊重人的暂停;没被采纳的还没开始)。
        等待态的 ``active_wait`` 留在库里,自检完了会回到等待 —— 「等换样品」的
        语义不该被一次重启变成「重新开始」。

        ## 为什么按**状态**枚举,而不是问一句 ``store.active()``

        ``active()`` 问的是「谁占着活跃位」,清算要问的是「谁还没了结」。两句话
        在不变式成立时同义,而清算恰恰是**不变式可能已经不成立**的那一刻:活跃位
        由 ``record()`` 跟着状态自动收放,一次崩在中间的写、一次外部改库、一份
        旧版本写下的行,都能留下「状态非终态、活跃位是 NULL」的孤儿。那时
        ``active()`` 回 ``None``,而 ``None`` 会被读成「没有要清算的」——
        **「读不到」当成「没有」**,一份跑了两天的 conduct 就那样安静地停在
        那里。所以这里按 :data:`~mast.conduct.store.STATUSES` 里的非终态逐个
        列,列到的孤儿**明说**(§10-3:要么明拒要么明确排队,不许静默)。
        """
        rep = TickReport()
        rows: list[dict] = []
        seen: set = set()
        for status in _NON_TERMINAL:
            for row in self.store.list_conducts(status=status, limit=200):
                if row["conduct_id"] in seen:
                    continue
                seen.add(row["conduct_id"])
                rows.append(row)
        if not rows:
            return rep.did("no_active_conduct")
        rows.sort(key=lambda r: str(r.get("created_at") or ""))
        for row in rows:
            self._reconcile_one(row, rep)
        return rep

    def _reconcile_one(self, row: dict, rep: TickReport) -> None:
        cid = row["conduct_id"]
        status = str(row["status"])
        if not rep.conduct_id:
            rep.conduct_id = cid
            rep.status_before = status
        orphan = row.get("active_slot") is None
        if status in ("paused", "draft", "approved"):
            if orphan:
                # 让它自己说出来。这一行**不去修**活跃位:改它要么经 ``record``
                # 的状态转移(而这三种状态不该被清算改),要么裸写一列 ——
                # 而「状态改动只有一扇门」是 store 的第一条纪律。
                rep.not_checked.append(
                    f"conduct {cid} 状态是 {status} 却没占活跃位 —— "
                    f"单活跃不变式对不上账,别的 conduct 可能因此建得出来")
            rep.did(f"restart_keeps:{status}")
            return
        try:
            self.store.record(cid, "status_change", changes={
                "status": "recovery_pending",
                "status_reason": f"进程重启(重启前是 {status}) —— 统一自检"})
        except Exception as exc:  # noqa: BLE001
            # 最典型的一种:另一个 conduct 占着活跃位,这一个转不回非终态。
            # **明说**,不吞 —— 吞掉的话屏幕上会有一份永远停在旧状态、
            # 而且没有任何解释的 conduct。
            rep.not_checked.append(
                f"conduct {cid}({status})搬不进 RECOVERY_PENDING: {exc}")
            self._notify(cid, "recovery_blocked",
                         f"重启清算搬不动这份 conduct({status}): {exc}", "crit")
            rep.did(f"restart_stuck:{cid}")
            return
        if rep.status_after in ("", status):
            rep.status_after = "recovery_pending"
        rep.did("restart_to_recovery")

    def _recovery_step(self, cid, row, rep) -> None:
        """一 tick 做一项。自检项也是步,跨 tick。"""
        spec = self._spec_or_none(row)
        done = {(ev.get("payload") or {}).get("item")
                for ev in self.store.events(cid, kind="recovery_item", limit=200)}
        waiting = bool(row.get("active_wait"))
        # 等待态重启只做无接触档:A3 需要进针,会把「等换样品」直接打破。
        checklist = list(NO_CONTACT_CHECKS) + ([] if waiting else list(CONTACT_CHECKS))
        for item in checklist:
            if item in done:
                continue
            verdict, detail = self._run_recovery_item(cid, row, spec, item)
            if verdict == "pending":
                # 这一项还没做完(接触档跨多个 tick)—— **不落 recovery_item**,
                # 否则下一 tick 会把它当成做过了。
                rep.did(f"recovery:{item}=pending")
                return
            event_id = self.store.record(cid, "recovery_item",
                                         payload={"item": item, "verdict": verdict,
                                                  "detail": detail})
            rep.did(f"recovery:{item}={verdict}")
            if verdict != "pass":
                # fail 与「读不到」都停下来等人 —— **读不到不等于通过**,
                # 而且这条无视 auto_resume(预授权只授权「通过后不打扰」)。
                #
                # 停下来的同时留下**怎么才能继续**:自检停的这一档在 M4-a 里
                # 同样只有 abort 与 takeover 两条出路,而 takeover→resume 会
                # 再跑一遍同一项、再停在同一个地方。那是一道解不开的闸,
                # 而且这条线是本仓自己接的(A3 判针坏那条路)。
                self._wait_for_operator(
                    cid, f"恢复自检 {item}:{detail}", rep,
                    decision={"kind": "recovery", "decision_id": int(event_id),
                              "item": item, "verdict": verdict, "reason": detail})
            return
        # 全过
        if waiting:
            back = "waiting_operator"
            self.store.record(cid, "status_change", changes={
                "status": back,
                "status_reason": "自检通过,回到重启前的等待(等待不该被重启抹掉)"})
            rep.did("recovery_back_to_wait")
            return
        self._resume_after_recovery(cid, self.store.get(cid), spec, rep)

    # ── 自检全过之后:从上一个已通过的闸门续跑(§0 / §8 A4)────────────

    def _resume_point_overridden(self, cid: str) -> bool:
        """人放行过「续跑点接不上」这一条吗(**只算当前这一代**)。

        按 ``evidence_epoch`` 划界:一次新的重启或一次绕道都会 bump 代次,
        而那之后的世界是新的 —— 上一代那句「就地续跑吧」不该跟着一起继承。
        少了这条限定,一次放行就变成了永久豁免。
        """
        row = self.store.get(cid) or {}
        epoch = int(row.get("evidence_epoch") or 0)
        for ev in self.store.events(cid, kind="decision_overridden", limit=500):
            p = ev.get("payload") or {}
            if p.get("item") == "resume_point" and int(p.get("epoch", epoch)) == epoch:
                return True
        return False

    def _resume_after_recovery(self, cid, row, spec, rep) -> None:
        """定续跑点 → 按预授权位决定问不问人。

        顺序是**先定点再问预授权**,不能反过来:``auto_resume`` 授权的是
        「通过后不打扰」,而「这台机器算不出一个自洽的续跑点」根本还没走到
        「通过」那一步。反过来写的话,一份开了预授权的 conduct 会带着一个
        接不上的位置直接 RUNNING —— 而它会在下一道闸门上以「证据缺席」的形态
        停住,那句话指向的是判定,不是真正的原因。
        """
        if spec is None:
            self._wait_for_operator(cid, "恢复自检全过,但取不到模板,"
                                         "定不出续跑点 —— 请人处理", rep)
            return
        plan = recovery.resume_plan(
            spec, row, self.store.events(cid, kind="gate_evaluated", limit=2000),
            skill_meta=self._skill_meta)
        waived = self._resume_point_overridden(cid)
        event_id = self.store.record(cid, "recovery_item", payload={
            "item": "resume_point",
            "verdict": ("overridden_earlier" if (plan.blocked and waived)
                        else "blocked" if plan.blocked else "pass"),
            "detail": plan.blocked or plan.why,
            "stage_idx": plan.stage_idx, "step_idx": plan.step_idx,
            "rewound": bool(plan.rewound), "blockers": list(plan.blockers)})
        if plan.blocked and not waived:
            # 接不上也要有一条出路 —— 否则 takeover→resume 会再算一遍同一个
            # 「接不上」再停在同一个地方,而那正是「能停不能解」。
            # 放行的语义是**明确的**:就地续跑,不回退。
            self._wait_for_operator(
                cid, f"恢复自检全过,但续跑点接不上:{plan.blocked}", rep,
                decision={"kind": "recovery", "decision_id": int(event_id),
                          "item": "resume_point", "verdict": "blocked",
                          "reason": plan.blocked})
            return
        if plan.blocked and waived:
            # 人放行过了 ⇒ 就地续跑。**位置一个数都不动**:回退是被否掉的那个
            # 动作(它要重放等人步或改表面的动作),人放行的是「不回退,接着跑」。
            rep.did("resume_point_overridden")
        changes: dict = {}
        if plan.rewound:
            changes["step_idx"] = plan.step_idx
            rep.did(f"recovery_rewound:{plan.step_idx}")
        if spec.auto_resume_after_recovery:
            changes.update({"status": "running", "status_reason": ""})
            self.store.record(cid, "status_change", changes=changes,
                              payload={"resume": plan.why})
            rep.did("recovery_auto_resume")
            return
        if changes:
            self.store.record(cid, "status_change", changes=changes,
                              payload={"resume": plan.why})
        self._wait_for_operator(
            cid, f"恢复自检全部通过,等一句「继续」(未开启 auto_resume)。"
                 f"续跑点:{plan.why}", rep)

    def _run_recovery_item(self, cid, row, spec, item: str) -> tuple[str, str]:
        if item == "A6_leftovers":
            run_id = str(row.get("active_run_id") or "")
            if run_id:
                # 进程死在步中 ⇒ 那一步的产出**不可信、不重放**。
                # composite sidecar 按 (name, run_id) 分键,天然作废。
                self.store.record(cid, "step_interrupted", run_id=run_id,
                                  changes={"active_run_id": ""},
                                  payload={"why": "进程死在这一步里,产出不可信,"
                                                  "不重放"})
                return "pass", f"清算了中断的 run {run_id}(产出作废)"
            return "pass", "没有遗留的 run"
        if item == "A5_spec":
            if spec is None:
                return "fail", f"找不到模板 {row['spec_id']!r}"
            if int(spec.spec_version) != int(row["spec_version"]):
                return "fail", (f"模板版本对不上:库里第 {row['spec_version']} 版,"
                                f"代码里第 {spec.spec_version} 版 —— **不自动迁移**")
            return "pass", f"模板 {spec.spec_id} v{spec.spec_version}"
        if item == "A4_epoch":
            from mast.core.coord_epoch import read_current_epoch
            epoch = read_current_epoch()
            # 重启一律 bump 证据代次:重启前采的证据不再参与闸门判定。
            #
            # 代次边界③。``coord_epoch`` 与 ``evidence_epoch`` 是**两个不同的
            # 代次**,别看名字像就当一个:前者管「粗动之后同一个 (x,y) 还是不是
            # 同一片表面」,后者管「针换过没有」。这里一次写两个,各写各的。
            self.store.record(cid, "status_change", changes={
                "evidence_epoch": int(row["evidence_epoch"]) + 1},
                payload={"why": "重启清算:重启前的证据不再参与闸门",
                         "coord_epoch": epoch,
                         "evidence_epoch": int(row["evidence_epoch"]) + 1,
                         EPOCH_START_KEY: self._clock()})
            return "pass", f"证据代次 +1;坐标代次={epoch}"
        # 注入的 ``recovery_probe`` 只在测试里用:它是替身,一条 TCP 都不发。
        # 生产走下面三条真路。有替身就以替身为准(测试要能脚本化三个分支)。
        if self._recovery_probe is not None:
            try:
                verdict = str(self._recovery_probe(item))
            except Exception as exc:  # noqa: BLE001
                return "unreadable", f"{item} 探针抛异常: {exc}"
            if verdict not in ("pass", "fail", "unreadable"):
                return "unreadable", f"{item} 探针回了一个不认识的值 {verdict!r}"
            return verdict, f"{item} 探针判定为 {verdict}"
        if item == "A1_link":
            return self._check_link()
        if item == "A2_temp":
            return self._check_temperature(row, spec)
        if item == "A3_tip":
            return self._check_tip(cid, row, spec)
        return "unreadable", f"{item} 没有实现,读不到不等于通过"

    # ── A1:连接 ─────────────────────────────────────────────────────

    def _check_link(self) -> tuple[str, str]:
        """各 role 只读探活。

        三分支各不相同,而且**「读不到」与「挂了」共用同一个去向但不共用同一句
        话**:去向都是等人,而人要做的事不一样 —— 一个是去看 Nanonis 与
        NI Service Locator,一个是去看这台机器上探针为什么装不上。
        (netstat 会误导:端口 LISTEN 着,而 NI Service Locator 停了 ⇒ TCP 全失败。)
        """
        if self._link_probe is None:
            return "unreadable", ("A1 没有连接探针可用 —— 读不到不等于通过")
        try:
            roles = dict(self._link_probe() or {})
        except Exception as exc:  # noqa: BLE001
            return "unreadable", f"A1 连接探活抛异常: {exc}"
        if not roles:
            return "unreadable", "A1 连接探活一个 role 都没报 —— 读不到"
        down = sorted(r for r, ok in roles.items() if ok is False)
        unknown = sorted(r for r, ok in roles.items() if ok is None)
        if down:
            return "fail", (f"这些 role 不通:{down} —— 查 Nanonis 是不是开着、"
                            f"NI Service Locator 是不是启用(netstat 会误导:"
                            f"端口 LISTEN 着也可能全失败)")
        if unknown:
            return "unreadable", (f"这些 role 探不出来:{unknown} —— "
                                  f"「探不出来」既不是通、也不是不通")
        return "pass", f"{len(roles)} 个 role 全部响应"

    # ── A2:温度 ─────────────────────────────────────────────────────

    def _check_temperature(self, row: dict, spec) -> tuple[str, str]:
        """(value, age_s) + 合理窗对账。

        两问,答不出来的那一问**说出来**:

        1. **读得到吗、新不新鲜** —— 这一问永远有答案(温度口三态自带),
           ``unknown`` / ``stale`` ⇒ 读不到,不是「条件不满足」;
        2. **值在这份 conduct 声明的工作点里吗** —— 这一问要 spec 说话。
           没声明 ⇒ 如实报**没检查**(进 ``not_checked`` 的那句话进 detail),
           而不是把「没问」印成「问过了」。
        """
        window = recovery.temperature_window(spec, row.get("params") or {}) \
            if spec is not None else recovery.TempWindow(None, "取不到模板")
        stale_after = 600.0
        try:
            reading = self.temperature.read()
        except Exception as exc:  # noqa: BLE001
            return "unreadable", f"A2 温度口抛异常: {exc}"
        fresh = reading.freshness(stale_after)
        if fresh != "fresh":
            return "unreadable", (
                f"温度{'读不到' if fresh == 'unknown' else '读数太旧'}"
                f"({reading.reason or fresh})—— 另一个串口程序 在跑吗?"
                f"**读不到既不是到了也不是没到**")
        value = float(reading.value_k)
        if not window.declared:
            return "pass", (f"温度 {value:.2f} K(读得到、够新)。"
                            f"合理窗**没检查**:{window.source}")
        if value > float(window.ceiling_k):
            return "fail", (f"温度 {value:.2f} K 超出声明的工作点 "
                            f"{window.ceiling_k:g} K({window.source})—— "
                            f"样品可能被动过,或者还在升温/没降回来")
        return "pass", (f"温度 {value:.2f} K ≤ {window.ceiling_k:g} K"
                        f"({window.source})")

    # ── A3:针尖重验(接触档)──────────────────────────────────────

    def _check_tip(self, cid, row, spec) -> tuple[str, str]:
        """按 spec 声明的复验步跑一遍,三态裁决。

        **一 tick 跑一步**:复验是接触档,每一步都要走 ``executor.run`` 的完整
        安全管道并受 ``instrument_lock`` 仲裁,一个 tick 里连着跑完等于把急停和
        abort 关在门外(它们只在步边界生效)。所以没跑完时回 ``pending``,
        由 :meth:`_recovery_step` 下一 tick 再来。
        """
        policy = getattr(spec, "recovery", None) if spec is not None else None
        checks = tuple(getattr(policy, "tip_check", ()) or ())
        if not checks:
            # **这一条是刻意的缺席,不是忘了。** 一份没有声明针尖复验的模板,
            # 重启之后没有任何东西知道针还能不能用 —— 那就说出来、停下来。
            return "unreadable", (
                "本 spec 没有声明恢复期针尖复验(RecoveryPolicy.tip_check 是空的)"
                " —— 重启之后没有任何证据说得清针还能不能用,读不到不等于通过")
        attempt = self._tip_attempt(cid)
        done_ids = self._produced(cid)
        epoch = int(row["evidence_epoch"])
        for step in checks:
            key = self._tip_step_id(step, attempt)
            if key in done_ids and done_ids[key]["epoch"] == epoch:
                continue
            tip_flat, tip_frames = self._tip_flat(cid, checks, attempt)
            ok, why = self._run_recovery_check_step(
                cid, row, step, key, flat=tip_flat, frames=tip_frames)
            if why == "__busy__":
                # 仪器被别的链路占着 —— 「拒绝不排队」纪律,下一 tick 再试。
                # 这**不消耗**判不了的重试预算:一次正常的并发仲裁不是一次判决。
                return "pending", f"复验步 {step.step_id} 撞上仪器令牌,下一 tick 再试"
            if not ok:
                # 复验步自己跑挂了 ⇒ **判不了**,不是「针坏了」。
                # 一次 executor 失败与一帧看不清是两件事,而它们的去向不同。
                return self._tip_undecidable(cid, attempt, policy,
                                             f"复验步 {step.step_id} 没跑成:{why}")
            return "pending", f"复验步 {step.step_id} 已完成,下一 tick 继续"
        # 全部跑完 ⇒ 判
        evidence = self._tip_evidence(cid, checks, attempt)
        state = rules.evaluate(policy.tip_rule, evidence)
        if state == rules.TRUE:
            return "pass", f"针尖复验通过({policy.tip_rule})"
        if state == rules.FALSE:
            # ⚠️ **判坏 ⇒ 停下来问人,不进绕道。** 完整理由见 :meth:`_tip_bad_reason`。
            return "fail", self._tip_bad_reason(row, evidence)
        return self._tip_undecidable(cid, attempt, policy,
                                     "复验判不了(证据不足或读数缺席)")

    def _tip_evidence(self, cid, checks, attempt: int) -> dict:
        """复验各步产出的并集 —— ``tip_rule`` 判的就是这一份。"""
        produced = self._produced(cid)
        out: dict = {}
        for step in checks:
            entry = produced.get(self._tip_step_id(step, attempt))
            if entry:
                out.update(entry["values"])
        return out

    def _tip_flat(self, cid, checks, attempt: int) -> "tuple[dict, dict]":
        """复验步之间互相绑定用的 ``(扁平表, 代次归属表)``:兄弟步的产出按**本名**
        也挂一份。

        代次归属跟着一起挂 —— 少了它,复验步之间那条
        ``center_x_m ← steps.R.00_clean_spot.x_m`` 会在「产出方明明盖了章」的情况下
        被当成没盖章放行(``FindCleanSpot`` 的 provenance 块里就有 ``coord_epoch``)。
        """
        flat, frames = self._flat_with_frames(cid)
        produced = self._produced(cid)
        for step in checks:
            entry = produced.get(self._tip_step_id(step, attempt))
            if not entry:
                continue
            stamp = entry["values"].get("coord_epoch")
            for name, value in entry["values"].items():
                key = f"steps.{step.step_id}.{name}"
                flat[key] = value
                frames[key] = (step.step_id, stamp)
        return flat, frames

    def _tip_attempt(self, cid: str) -> int:
        """A3 已经整串重跑过几次(0 = 第一次)。从审计流数,不在内存里记。"""
        n = 0
        for ev in self.store.events(cid, kind="recovery_item", limit=500):
            p = ev.get("payload") or {}
            if p.get("item") == "A3_tip_retry":
                n += 1
        return n

    @staticmethod
    def _tip_step_id(step: StepSpec, attempt: int) -> str:
        """复验步在审计流里的名字。带 attempt ⇒ 重跑不会读到上一轮的读数。"""
        return f"recovery.A3#{attempt}.{step.step_id}"

    def _run_recovery_check_step(self, cid, row, step: StepSpec, key: str,
                                 *, flat: "dict | None" = None,
                                 frames: "dict | None" = None
                                 ) -> tuple[bool, str]:
        """跑一个复验步(经 ``executor.run``),把产出记进审计流。

        **不动 ``step_idx``** —— 自检不是流程,位置一个数都不该被它改。
        """
        frame_notes: list = []
        try:
            params = self._resolve_params(cid, row, step, flat=flat,
                                          frames=frames, notes=frame_notes)
        except KeyError as exc:
            return False, str(exc)
        if step.kind == "analysis":
            try:
                produced = self._analyses_get(step.analysis_fn)(params) or {}
            except Exception as exc:  # noqa: BLE001
                return False, f"分析 {step.analysis_fn} 失败: {exc}"
            self._record_recovery_produce(cid, key, produced, run_id="")
            return True, ""
        run_id = uuid.uuid4().hex[:12]
        self.aborts.register(run_id)
        self.store.record(cid, "step_started", step_id=key, run_id=run_id,
                          changes={"active_run_id": run_id},
                          payload={"skill": step.skill, "params": params,
                                   "coord_frames": list(frame_notes),
                                   "recovery_item": "A3_tip", "at": self._clock()})
        try:
            outcome = self.executor.run(step.skill, params, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            outcome = StepOutcome(ok=False, error=f"executor 抛异常: {exc}",
                                  run_id=run_id)
        finally:
            self.aborts.unregister(run_id)
            self.store.record(cid, "status_change", changes={"active_run_id": ""},
                              payload={"cleared_run_id": run_id})
        if outcome.busy:
            # 仪器被别人占着**不是**复验失败,也不是针坏 —— 下一 tick 再来。
            return False, "__busy__"
        if not outcome.ok:
            return False, outcome.error or "复验步失败"
        self._record_recovery_produce(cid, key, dict(outcome.data or {}),
                                      run_id=run_id)
        return True, ""

    def _record_recovery_produce(self, cid, key: str, produced: dict,
                                 *, run_id: str) -> None:
        row = self.store.get(cid)
        self.store.record(cid, "step_finished", step_id=key, run_id=run_id,
                          payload={"ok": True, "produces": produced,
                                   "recovery_item": "A3_tip", "at": self._clock(),
                                   "evidence_epoch": int(row["evidence_epoch"])})

    def _tip_undecidable(self, cid, attempt: int, policy,
                         why: str) -> tuple[str, str]:
        """判不了:预算内就整串再跑一次,否则停下来问人。

        设计 §8 写的是「换位重试 1 次→仍不可判→有人:ask_user;无人:
        WAITING_OPERATOR」。**换位是复验步自己的事**(它下一轮会重新选点);
        而 ``ask_user`` 这条路 Director 走不了 —— 它是同步 HITL 工具,而等人
        绝不走 ``hitl_bridge``(进程本地、重启即丢、900 s fail-closed,三条性质
        对一份可能等一夜的请求全是错的)。所以两种值守模式**去向相同**
        (WAITING_OPERATOR),差别只在通知的分量,由 ``_wait_for_operator``
        那条心愿单承担。这是与设计措辞的一处偏差,写在这里免得被当成漏实现。
        """
        retries = int(getattr(policy, "tip_retries", 0) or 0)
        if attempt < retries:
            self.store.record(cid, "recovery_item", payload={
                "item": "A3_tip_retry", "verdict": "unreadable",
                "detail": f"{why} —— 第 {attempt + 1}/{retries} 次重跑"})
            return "pending", f"{why},整串重跑一次"
        return "unreadable", (f"{why};重跑 {retries} 次仍判不了。"
                              f"**不可判 ≠ 好 ≠ 坏** —— 请人看一眼")

    def _tip_bad_reason(self, row: dict, evidence: "dict | None" = None) -> str:
        """A3 判针坏之后那句话。**去向是等人,不是绕道** —— 这是一个裁决。

        ## 凭什么不把它接到修针段上

        修针段的第一个动作是十几分钟的只读跨点复测。三条理由,一条比一条硬:

        1. **这个问题问晚了一跳。** 跨点复测不是这根针重启后挨的第一次扫 ——
           A3 自己才是(它就是进针 + 移位 + 扫一帧)。而两者的选点面对的是同一
           片盲区:撞针历史只活在 ``core/tip_crash_tracker`` 里,进程内、TTL
           30 分钟、粗动即清空,**而且撞针不写地图标记**(``full_scan`` 只调
           ``record_crash``),所以 ``FindCleanSpot`` 从来就不知道这件事。
           于是「凭什么再扫十几分钟」若成立,「凭什么扫这一帧」同样成立 ——
           拦住绕道并不能躲开那个风险,只会把它藏到上一跳去。
        2. **夜里进绕道买不到东西。** 修针段第三步就是一次 ``max_wait_s=None``
           的等人换样品。也就是说凌晨三点绕进去,跑完复测和退针就停在人闸上。
           比起当场停下,它多买到的只有「用户早上少花十五分钟」,而付出的是
           一次无人看管、拿一根刚判坏的针、在一张看不见坑的地图上做的接触操作。
        3. **绕道自己的诊断闸在这条路上说的话是假的。** 那道闸判 fail 时印的是
           「两份证据打架:上游判了针坏才绕进来,而跨点复测说不是针」——
           而恢复路上的「上游」是 A3 的**单点**读数。单点与跨点之间没有那种
           对峙关系,于是三个闭集结果里有两个会落进一句不成立的解释。

        **要推翻这条,得先回答**:①撞针历史怎么在重启之后还问得到(持久化它是
        一件独立的事);②在那两道人闸之前先跑那十几分钟,除了省用户十五分钟
        还买到了什么。若将来 A3 本身就是跨点的,那该改的是修针段的第 0 步,
        不是这一条。
        """
        # 选点时地图读不到 ⇒ 这个 fail **也可能是这片表面**。说出来,别让一句
        # 「针坏了」把用户直接推去换样品。("map_known=false 的意思是不知道,
        # 不是干净" —— FindCleanSpot 自己的措辞。)
        caveat = ""
        if evidence is not None and evidence.get("map_known") is False:
            caveat = ("。⚠️ 选点时读不到实验记录(map_known=false)—— "
                      "这一帧扫的地方是不是干净的,谁也不知道,所以这个「坏」"
                      "也可能是表面")
        return (f"针尖复验判**针坏了**(阶段 {row.get('stage_idx')}/步 "
                f"{row.get('step_idx')}){caveat}。**不自动进修针段** —— 修针段"
                f"第三步就是一次没有上限的等人换样品,夜里绕进去只能停在那儿;"
                f"而绕道前的十几分钟只读复测,选点靠的撞针记忆在重启之后是空的"
                f"(进程内、不落盘、撞针也不写地图标记)。"
                f"请人看一眼复验读数再决定修不修")

    # ── 中止序列 ─────────────────────────────────────────────────────

    def _abort(self, cid, row, reason: str, by: str, rep) -> None:
        self._signal_abort_event(row)
        status = row["status"]
        retract = ""
        if status in ("waiting_operator", "waiting_condition"):
            # 等待态里针已经退了(校验器规则①结构保证),直达 ABORTED。
            retract = "等待态,针已退"
        else:
            retract = self._confirmed_retract(cid, rep)
        self.store.record(cid, "aborted", changes={
            "status": "aborted", "active_run_id": "", "active_wait": None,
            "status_reason": f"{reason}(by {by});退针:{retract}"},
            payload={"by": by, "reason": reason, "retract": retract})
        self._notify(cid, "aborted", f"conduct 已中止:{reason};退针:{retract}",
                     "warn")
        rep.did("aborted")

    def _confirmed_retract(self, cid: str, rep) -> str:
        """中止序列里的确认式退针。失败**重试一次 + 大声通知**,再失败就交给
        watchdog —— 但绝不假装退成功了。"""
        last = ""
        for attempt in (1, 2):
            run_id = uuid.uuid4().hex[:12]
            try:
                outcome = self.executor.run("SafeRetract", {}, run_id=run_id)
            except Exception as exc:  # noqa: BLE001
                outcome = StepOutcome(ok=False, error=str(exc))
            self.store.record(cid, "step_finished", step_id="abort_retract",
                              run_id=run_id,
                              payload={"ok": bool(outcome.ok), "attempt": attempt,
                                       "why": outcome.error, "at": self._clock()})
            if outcome.ok:
                return f"已确认(第 {attempt} 次)"
            last = outcome.error or "未确认"
        self._notify(cid, "retract_failed",
                     f"中止时退针未确认({last}) —— watchdog 兜底,请人确认针的状态",
                     "crit")
        return f"**未确认**({last})"

    def _signal_abort_event(self, row: dict) -> None:
        run_id = str(row.get("active_run_id") or "")
        if run_id:
            self.aborts.signal(run_id)

    # ── 小工具 ───────────────────────────────────────────────────────

    def _wait_for_operator(self, cid: str, reason: str, rep,
                           *, decision: "dict | None" = None) -> None:
        """停下来问人。``decision`` 非空 = **人有一条可以说「继续」的路**。

        WAITING_OPERATOR 有两个来源(见 ``_dispatch`` 那段注释),而它们的**出口
        不同**:一个 ``wait`` 步等的是 ack + 物理条件,可轮询;一次裁决转人等的是
        **人的判断**,没有可轮询的闸。后者在 M4-a 之前只有 abort 与 takeover 两条
        出路 —— 也就是说,一份被闸门停在半夜的 conduct,早上人看过觉得没问题,
        能做的只有**放弃它**或**永久接管**。

        这正是本仓 2026-08-13 付过账的那个形状(端口抖 21 s → 判成硬故障 →
        退针 + 挂闩,而解闩的函数是死代码 ⇒ 仪器完好、机器锁死到重启)。
        ``decision`` 就是那条留痕的解锁路,形状照 ``waive_condition``。
        """
        # ── 等人之前先把针拿开 ─────────────────────────────────────
        #
        # 2026-08-21 查出的缺口：确认式退针此前**只挂在中止序列上**
        # （:meth:`_confirmed_retract` 的唯一调用方是 :meth:`_abort`），而
        # WAITING_OPERATOR 不退 —— 于是一份被闸门停在半夜的 conduct，会把针尖
        # 留在隧道结上等人来看，可能几小时。
        #
        # 校验器的规则①（``wait_without_retract_confirm``）说的正是这件事，
        # 但它只管模板里的 ``wait`` **步**；这里是**裁决转人**那条路，
        # 模板侧的闸够不着它。所以补在引擎这一层：15 个调用方共用这一个口子，
        # 补一处就全覆盖。
        #
        # 三条纪律：
        #
        # 1. **退针失败不阻塞进等待。** 人更需要被叫来，而不是被一个失败的退针
        #    挡在门外 —— 但那句「没退成」必须出现在 status_reason 里，
        #    不许静默（``_confirmed_retract`` 自己会重试一次并大声通知）。
        # 2. **已经在等人就不重复退。** 这个函数在少数路径上可能被再次调用；
        #    退针是静态态，重退无害但会刷一串没用的审计行。
        # 3. **没有执行器就说没退。** 「没退成」与「没试过」都不许被写成「退了」。
        retract = ""
        row = self.store.get(cid) or {}
        if self._rescue_owned_elsewhere or self._latch_state().latched:
            # **闩挂着的时候不许碰仪器。** 针尖事件/撞针必然挂闩，而那一路的处置
            # 序列归 watchdog —— Director 再去退一次针，就是两个主人同时动仪器。
            # `test_a_tip_event_is_recorded_but_the_director_does_not_grab_the_rescue`
            # 钉的就是这条：「这里只如实记账，**不抢救济**」。
            retract = "**没退**(急停闩挂着,处置归 watchdog,Director 不抢救济)"
        elif str(row.get("status") or "") == "waiting_operator":
            retract = "已在等人,不重复退针"
        elif getattr(self, "executor", None) is None:
            retract = "**没试过**(这份 Director 没有执行器)"
        else:
            try:
                retract = self._confirmed_retract(cid, rep)
            except Exception as exc:  # noqa: BLE001 — 退不成也要把人叫来
                retract = f"**没退成**:{type(exc).__name__}: {exc}"
        reason = f"{reason}(等人前退针:{retract})"

        self.store.record(cid, "status_change",
                          changes={"status": "waiting_operator",
                                   "status_reason": reason},
                          payload={"decision": dict(decision)} if decision else None)
        self._notify(cid, "wait_operator", reason, "warn")
        rep.did("wait_operator")

    def pending_decision(self, cid: str) -> "dict | None":
        """这份 conduct 现在停在一个**人可以放行的判定**上吗(没有就 ``None``)。

        从审计流取,不加列:最后一条把它搬进 ``waiting_operator`` 的
        ``status_change``,payload 里带 ``decision`` 就是。**不靠解析
        ``status_reason`` 的文案** —— 那是给人读的字符串,改一个字这里就静默失灵
        (本仓为手工维护的名字清单付过至少四次学费)。

        面板与 API 也读它:「还能不能继续」必须是可问的,而不是让用户去猜。
        """
        row = self.store.get(cid) or {}
        if row.get("status") != "waiting_operator" or row.get("active_wait"):
            return None
        for ev in reversed(self.store.events(cid, kind="status_change", limit=2000)):
            p = ev.get("payload") or {}
            if p.get("status_to") != "waiting_operator":
                continue
            decision = p.get("decision")
            return dict(decision) if isinstance(decision, dict) else None
        return None

    def _notify(self, cid: str, kind: str, message: str,
                severity: str = "info") -> None:
        try:
            self.notifier.notify(Notification(kind=kind, conduct_id=cid,
                                              message=message, severity=severity))
        except Exception as exc:  # noqa: BLE001 —— 通知发不出去不该改变状态机
            logger.warning("conduct 通知发送失败(状态照旧): %s", exc)

    def _spec(self, row: dict) -> ConductSpec:
        return self._spec_provider(row["spec_id"])

    def _spec_or_none(self, row: dict) -> "ConductSpec | None":
        try:
            return self._spec_provider(row["spec_id"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("取不到模板 %r: %s", row.get("spec_id"), exc)
            return None


__all__ = ["ConductDirector", "TickReport", "AT_ENTRY_GATE", "EPOCH_START_KEY",
           "OP_VALID_STATUSES", "BUSY_YIELD_AFTER_S",
           "NO_CONTACT_CHECKS", "CONTACT_CHECKS"]
