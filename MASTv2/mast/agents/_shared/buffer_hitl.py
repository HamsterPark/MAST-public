"""缓冲区关键事件只记录与通知，不在此处中断工具调用。

订阅 TIP_QUALITY_DROP / E_STOP，记录 event_refs、日志和诊断台账。
本模块不调用 interrupt，也不关闭工具闸门；wrap_tool_call 透传。
事件日志、ReadHardwareEvents 和面板仍可读取事件。

正常进针、脉冲和扎针会产生瞬态，通知与拒绝型防护必须分别处理。
E_STOP / 用户中止走 register_critical_hook → abort 闩锁 → skill_adapter
的独立通道。撞针状态机、SafetyGate、粗动电压限制、针尖包络和其他前置
条件继续各自执行；本模块不替代或削弱它们。

若改变通知策略，需要评估可检测的危险、现有拒绝型防护的覆盖与误报代价。
REMEDY_TOOL_NAMES / REMEDY_TAGS 是退针词汇定义，与 instrument_lock 的绕过表
保持一致。GATE_* 及 gate_states/reset_all_gates/resolve_all_gates 为调用方
保留兼容接口，返回没有待处理闸门的开放状态。"""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware import AgentMiddleware

from mast.buffer.schemas import Severity, VisionEvent, VisionEventType

if TYPE_CHECKING:
    import asyncio

    from langgraph.runtime import Runtime

    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Defaults — 订阅哪些事件、哪个 severity 起算「值得单独通知一句」
# ─────────────────────────────────────────────────────────────────────

#: 订阅表。两个 kind 都**只是被记录和通知**;E_STOP 的实际停止力在 abort 闩锁上
#: (见模块 docstring),不在这里。
DEFAULT_HITL_EVENT_KINDS: tuple[VisionEventType, ...] = (
    VisionEventType.TIP_QUALITY_DROP,
    VisionEventType.E_STOP,
)

#: 通知门槛。低于它的事件仍然进 ``event_refs``(记录一个字节不少),只是不单独
#: 在诊断台账里占一行 —— 否则 2553 条 warn 会把台账刷没用。
DEFAULT_MIN_SEVERITY: Severity = Severity.CRITICAL


# ─────────────────────────────────────────────────────────────────────
# 分类 —— 只影响**通知怎么写**,不再影响「拦不拦」
# ─────────────────────────────────────────────────────────────────────
#
# ⑰ 之前这一段决定的是「这条事件能不能停住用户」。现在它决定的是通知里那句话
# 怎么措辞:「前放到轨」和「视觉觉得针尖变差了」对用户是完全不同的两件事,通知
# 把它们印成同一句话等于没通知。判据本身一个字没改 —— 改的是它的下游。

# ⑰-C2(2026-08-09):**这四样的定义搬到了 ``mast.core.tip_intent``**,这里只是
# re-export。搬家的理由是方向:⑭ 当时把它们写在这里,因为那时唯一的消费者是确认框;
# ⑰ 把确认框整条割掉之后,本模块只拿它**给通知措辞**,而真正会**停下仪器**的消费者
# 是 ``runtime.make_tip_halt_hook`` —— 让「会停仪器的那一方」到一个「只写措辞的模块」
# 里 import 判据,方向是反的。
#
# 名字保留在这里,是因为外部(测试、将来的读者)按这个位置找过它们;判据本身只有
# ``core/tip_intent.py`` 那一份,连同它的翻盘观测。
from mast.core.tip_intent import (  # noqa: E402
    SUSTAINED_PHYSICAL_SIGNALS,
    TRANSIENT_PHYSICAL_SIGNALS,
    _PHYSICAL_SIGNAL_FALLBACK,
    exempt_during_tip_work,
    physical_current_signals,
)

#: 永远单独通知的 kind:人在喊停 / 已经出事。它们不看 payload。
ALWAYS_ESCALATE_KINDS: frozenset[VisionEventType] = frozenset({
    VisionEventType.E_STOP,
    VisionEventType.EMERGENCY_RETRACT_NEEDED,
})


def classify_event(ev: "VisionEvent") -> str:
    """事件归类,给通知用。永不抛。

    返回 ``e_stop`` / ``crash`` / ``physical_sustained`` /
    ``physical_transient`` / ``tip_work_transient`` / ``morphology`` /
    ``other`` 之一。

    两种「读不出来」是**两句话**,不是一句:``kind`` 认不出来(包括压根不是事件
    对象)是 ``other``;分类**抛了异常**才落回 ``morphology`` —— 后者是旧的
    fail-quiet 默认(最安静的一类)。合成一句会让「这是个没见过的事件类型」和
    「分类器坏了」在台账里长得一模一样。
    """
    try:
        kind = getattr(ev, "kind", None)
        if kind is VisionEventType.E_STOP:
            return "e_stop"
        if kind is VisionEventType.EMERGENCY_RETRACT_NEEDED:
            return "crash"
        if kind is not VisionEventType.TIP_QUALITY_DROP:
            return "other"
        signal = str((getattr(ev, "payload", None) or {}).get("signal") or "")
        if signal not in physical_current_signals():
            return "morphology"
        if signal in SUSTAINED_PHYSICAL_SIGNALS:
            return "physical_sustained"
        if exempt_during_tip_work(signal):
            return "tip_work_transient"
        return "physical_transient"
    except Exception:  # noqa: BLE001
        logger.debug("buffer_hitl: 事件分类失败", exc_info=True)
        return "morphology"


#: 各类事件的一句话通知前缀。键是 :func:`classify_event` 的返回值。
_NOTICE_ZH: dict[str, str] = {
    "e_stop": "急停事件(停止动作走的是 abort 闩锁,不是这里)",
    "crash": "撞针/应急退针事件(处置走撞针状态机)",
    "physical_sustained": "持续型物理越界(贴轨/信号链冻结)",
    "physical_transient": "瞬变型物理事件",
    "tip_work_transient": "修针期间的瞬变(本职动作的签名)",
    "morphology": "针尖/图像形态判定",
    "other": "缓冲区关键事件",
}


def escalates_to_operator(ev: "VisionEvent") -> bool:  # noqa: ARG001
    """这条事件会不会产生 ``interrupt()`` / 关掉工具闸门。

    **恒为 ``False``。** 保留这个函数是因为它回答的问题没变、而答案变了 ——
    删掉它会让「答案是 no」这件事没有落点,也会让下面这段翻盘条件没有归属。

    要让它重新有可能返回 ``True``,先回答模块 docstring 里的三个问题
    (哪一次真阳性 / 为什么拒绝型防护接不住 / 误报率是多少)。
    截至 2026-08-08:第一个问题的答案是「零例」。
    """
    return False


# ─────────────────────────────────────────────────────────────────────
# Remedy 词汇 —— 闸门没了,但这份定义还是树里的单一真源
# ─────────────────────────────────────────────────────────────────────

#: 「这个动作是解药,不是继续推进实验」。本模块已不再消费它(工具闸门割掉了),
#: 但 ``core/instrument_lock`` 的 ``BYPASS_NAMES`` 是它的姊妹表,而
#: ``tests/v2/unit/test_forensics_20260727_safety_gates.py`` 钉着「表里每个名字都
#: 是真技能」。定义留在这里。
REMEDY_TOOL_NAMES: frozenset[str] = frozenset({
    "StopScan",
    "StopSTS",
    "StopMotor",
    "StopAutoApproach",
    "WithdrawTip",
    "TryEngageController",
    "ApproachTip",
    "EmergencyRetract",
    "SafeRetract",
})

#: 不靠名字、靠 metadata 标签认出来的解药。与 ``instrument_lock.BYPASS_TAGS`` 同义。
REMEDY_TAGS: frozenset[str] = frozenset({"retract", "emergency", "withdraw"})


# ─────────────────────────────────────────────────────────────────────
# 兼容外壳 —— 外面还有三个调用方,它们现在得到的答案是「没有闸门」
# ─────────────────────────────────────────────────────────────────────

#: 历史上用户用来重开闸门的入口。闸门已经不存在,路由仍然挂着(前端在读),
#: 现在它是一个诚实的 no-op:清 0 个闸门。
GATE_STATE_PATH: str = "/agents/hitl-gates"
GATE_RESOLVE_PATH: str = "/agents/hitl-gates/resolve"
GATE_API_PREFIX: str = "/api"


def gate_resolve_url() -> str:
    """历史上「重开闸门」的完整路径。保留给 API 路由与其 schema。"""
    return f"{GATE_API_PREFIX}{GATE_RESOLVE_PATH}"


#: 每个活着的中间件,弱引用。留着是为了 :func:`gate_states` 还能如实回答
#: 「有几个中间件在跑、它们各自记录了多少条不打断的事件」。
_LIVE_GATES: "weakref.WeakSet[BufferHITLMiddleware]" = weakref.WeakSet()


def _each_gate():
    try:
        return list(_LIVE_GATES)
    except Exception:  # pragma: no cover — defensive
        return []


def reset_all_gates(why: str = "") -> int:  # noqa: ARG001
    """历史入口:新 run 开始时重置闸门。现在恒返回 0(没有闸门可重置)。

    保留是因为 ``api/routes/orchestrator`` 在每个新任务开头调它,并把返回值写进
    日志。返回 0 是**如实回答**,不是失败。"""
    return 0


def resolve_all_gates(why: str = "") -> int:  # noqa: ARG001
    """历史入口:用户显式重开所有闸门。现在恒返回 0(没有闸门被关着)。"""
    return 0


def gate_states() -> list[dict[str, Any]]:
    """每个活着的中间件的快照。``closed`` 恒为 ``False``。

    ``recorded_not_escalated`` 是这里唯一还在动的数 —— 它是「没有弹窗」与
    「没有事件」的区别,而这两件事从外面看必须不一样。"""
    out = []
    for gate in _each_gate():
        try:
            out.append(gate.gate_state())
        except Exception:  # noqa: BLE001
            pass
    return out


# ─────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────

class BufferHITLMiddleware(AgentMiddleware):
    """每次 LLM 调用前非阻塞地看一眼缓冲区队列:**记录 + 通知,不打断**。

    生命周期:

      * ``__init__`` —— 每个订阅的 kind 调一次 ``buffer.subscribe(kind)``。
      * ``close()`` —— 逐个 ``buffer.unsubscribe``,否则 BufferService 会一直往
        一个没人读的队列里投递(泄漏)。

    名字保留 ``BufferHITL``:它仍然是缓冲区事件进入 agent 状态的那条路,只是
    HITL 的那一半被割掉了。改名会连带动 5 个文件的 import 和两条 API schema,
    而这条改动的重点不在名字上。
    """

    def __init__(
        self,
        buffer: "BufferService",
        *,
        event_kinds: tuple[VisionEventType, ...] = DEFAULT_HITL_EVENT_KINDS,
        min_severity: Severity = DEFAULT_MIN_SEVERITY,
        get_mode: Callable[[], Any] | None = None,
    ):
        """构建并立刻注册订阅。

        Parameters
        ----------
        buffer:
            共享的 ``BufferService``。
        event_kinds:
            订阅哪些事件类型。默认 ``(TIP_QUALITY_DROP, E_STOP)``。
        min_severity:
            低于它的事件仍然进 ``event_refs``,只是不单独通知一行。
        get_mode:
            实时操作模式读取器。只用来让通知里的「建议动作」不自相矛盾:
            SAFE 模式下 MAST 本来就拒绝修针,通知里还建议 ``tip_prep`` 就是在
            推荐一件当前模式不允许自动执行的事。``None`` → 模式盲默认。

        Note
        ----
        ⑰ 删掉了 ``interrupt_fn`` 参数。它曾经是测试注入 ``interrupt`` 的口子;
        现在这个模块根本不 import ``langgraph.types``,**没有调用点**。留一个
        没人用的注入口会让代码读起来像还能打断。
        """
        super().__init__()
        # 不放进 _init_state:这是**整个中间件生命周期**的累计数,不是某一轮的
        # 状态。按 run 清零会抹掉唯一一个能说明「安静的那条路正在被走」的数字,
        # 而这恰恰是用户在被告知「告警没了」之后会问的第一件事。
        self._recorded_not_escalated = 0
        self._init_state()
        self._buffer = buffer
        self._event_kinds = tuple(event_kinds)
        self._min_severity = min_severity
        self._get_mode = get_mode
        # 每个 kind 一个队列。队列满时 buffer 自己丢最老的,这里不会 OOM。
        self._subs: dict[VisionEventType, "asyncio.Queue[VisionEvent]"] = {
            kind: buffer.subscribe(kind) for kind in self._event_kinds
        }
        self._closed = False
        try:
            _LIVE_GATES.add(self)
        except TypeError:  # pragma: no cover — non-weakrefable base class
            logger.debug("buffer_hitl: 实例不可弱引用,gate_states 看不到它")

    def _init_state(self) -> None:
        """一轮里的暂存区。

        ⑰ 之前这里还有 ``_unresolved`` / ``_awaiting`` / ``_degraded`` /
        三个 ``_reask_*`` / ``_blocked_tools`` —— 全是审批对话的状态机,随打断链
        一起删掉了。留下来的只有 ``_staged``:队列是 consume-once 的,而
        ``before_model`` 的返回值只有正常返回时才会被 langgraph 合并进 state,
        所以中途抛异常时暂存区保证事件 id 不丢。
        """
        self._staged: list[tuple[VisionEventType, VisionEvent]] = []

    @property
    def name(self) -> str:
        return "BufferHITLMiddleware"

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """退订。可以重复调用。"""
        if self._closed:
            return
        for kind, q in self._subs.items():
            try:
                self._buffer.unsubscribe(kind, q)
            except Exception:  # pragma: no cover — defensive
                logger.exception("buffer.unsubscribe failed for %s", kind)
        self._subs.clear()
        self._closed = True

    def __del__(self):  # pragma: no cover — best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ── Hook ─────────────────────────────────────────────────────────

    def before_model(
        self,
        state: Any,
        runtime: "Runtime",  # noqa: ARG002 — required by base signature
    ) -> dict[str, Any] | None:
        """非阻塞地把每个订阅队列抽干,记录 + 通知,**永远不打断**。

        返回 ``{"event_refs": [...]}``;``MASTState`` 的 ``dedupe_event_refs``
        reducer 会把它合并到已有 refs 上,所以重复调用不会重复堆积。
        """
        if self._closed:
            return None

        # 把 consume-once 队列抽干进暂存区(while 循环折叠一次突发:两帧之间来的
        # 两条事件都要进 event_refs)。
        for kind, q in self._subs.items():
            while True:
                try:
                    ev = q.get_nowait()
                except Exception:
                    # asyncio.QueueEmpty 是预期哨兵;宽捕获是为了不在模块顶层
                    # import asyncio。
                    break
                self._staged.append((kind, ev))

        if not self._staged:
            return None

        # **每一条**暂存事件都进 state.event_refs —— 包括(现在是全部)不打断的
        # 那些。⑰ 割掉的是打断,不是记录。
        new_refs = [ev.event_id for _, ev in self._staged]
        for _kind, ev in self._staged:
            if self._is_notable(ev, _kind):
                self._notify(ev)
        self._staged.clear()
        return {"event_refs": new_refs} if new_refs else None

    # ── 通知 ─────────────────────────────────────────────────────────

    def _notify(self, ev: VisionEvent) -> None:
        """一条关键事件的通知:日志 + 诊断台账。永不抛。

        诊断台账(``artifacts/diagnostics/refusals.jsonl`` + 诊断面板)是这条
        改动的**审计落点**:「没弹窗」与「没发生」从外面看必须不一样。
        """
        self._recorded_not_escalated += 1
        cls = classify_event(ev)
        signal = (getattr(ev, "payload", None) or {}).get("signal")
        kind_txt = getattr(ev.kind, "value", ev.kind)
        logger.warning(
            "buffer_hitl 记录但不打断:%s(%s, signal=%s, %s)—— "
            "关键事件已退出确认框链路,事件仍在缓冲区、event_refs 与面板里。",
            kind_txt, _NOTICE_ZH.get(cls, cls), signal, ev.cause_ref,
        )
        try:
            from mast.core.diagnostics import record as _diag

            _diag(
                "notice_only", f"buffer:{kind_txt}",
                f"{_NOTICE_ZH.get(cls, cls)} —— 只通知不打断"
                f"(建议动作:{_suggest_action(ev.kind, self._safe_mode())})",
                event_class=cls,
                signal=signal,
                severity=getattr(ev.severity, "value", ev.severity),
                seqno=getattr(ev, "seqno", None),
                cause_ref=getattr(ev, "cause_ref", None),
                summary_zh=(getattr(ev, "payload", None) or {}).get("summary_zh"),
            )
        except Exception:  # noqa: BLE001 — 台账写不进去绝不能反噬事件记录
            logger.debug("buffer_hitl: 诊断台账写入失败", exc_info=True)

    # ── 工具闸门(已割掉)──────────────────────────────────────────────

    def wrap_tool_call(self, request, handler):
        """无条件透传工具调用。
        
        此模块仅记录和通知，即使存在旧式未解决事件属性也不能重新激活工具闸门。"""
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        """:meth:`wrap_tool_call` 的异步孪生。同样无条件透传。"""
        return await handler(request)

    # ── 只读快照 ──────────────────────────────────────────────────────

    def gate_state(self) -> dict[str, Any]:
        """JSON 安全的快照 —— 给测试、``ReadHardwareEvents`` 和用户遥测。

        键集刻意与 ⑰ 之前一致(``api/schemas_agents_control.HITLGateState`` 和
        前端在读),值是现在的事实:没有闸门、没有待答、没有降级。
        """
        return {
            "closed": False,
            "unresolved": 0,
            "awaiting": 0,
            "degraded": False,
            "kinds": [],
            "reask_armed": False,
            "reask_used": False,
            "blocked_tools": [],
            # 本中间件建立以来,记录了但没打断的关键事件条数。刻意面向用户:
            # 「没有弹窗」和「没有事件」从外面看必须不一样。
            "recorded_not_escalated": self._recorded_not_escalated,
        }

    @staticmethod
    def _is_remedy(tool) -> bool:
        """这个工具是不是「解药」(停止 / 退针 / 读 / 修针)。

        ⑰ 之后**本模块不再消费这个判据**(工具闸门割掉了)。留着是因为
        ``core/instrument_lock`` 的 ``BYPASS_TAGS``/``BYPASS_NAMES`` 是同一份
        词汇的另一半,而 ``test_forensics_20260727_safety_gates`` 与
        ``test_safe_mode_audit_fixes`` 对着这里核它们。定义在这儿,别的地方抄。
        """
        name = getattr(tool, "name", "") or ""
        if name in REMEDY_TOOL_NAMES:
            return True
        meta = (getattr(tool, "metadata", None) or {}).get("skill_metadata")
        if meta is None:
            return True  # 不是技能工具 —— handoff / buffer / 分析
        category = getattr(getattr(meta, "category", None), "value", None)
        if str(category).lower() in ("read", "analysis"):
            return True
        tags = {str(t).lower() for t in (getattr(meta, "tags", None) or ())}
        if tags & REMEDY_TAGS:
            return True
        caps = getattr(meta, "capabilities", None) or frozenset()
        try:
            from mast.core.safety import CAP_BIAS_PULSE, CAP_TIP_SHAPING

            return bool({CAP_BIAS_PULSE, CAP_TIP_SHAPING} & set(caps))
        except Exception:  # pragma: no cover — defensive
            return False

    # ── Internals ────────────────────────────────────────────────────

    def _is_notable(self, ev: VisionEvent, kind: VisionEventType) -> bool:
        """这条事件值不值得在台账里单独占一行(不是「拦不拦」)。"""
        if kind not in self._event_kinds:
            return False
        return _severity_at_least(ev.severity, self._min_severity)

    def _safe_mode(self) -> bool:
        """当前是不是 SAFE 模式。永不抛 —— 读模式失败不该影响一条通知。"""
        if self._get_mode is None:
            return False
        try:
            from mast.core.types import OperatingMode
            return OperatingMode.coerce(self._get_mode()) is OperatingMode.SAFE
        except Exception:  # noqa: BLE001
            return False


# Severity ordering — info < warn < critical.
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARN: 1,
    Severity.CRITICAL: 2,
}


def _severity_at_least(have: Severity, threshold: Severity) -> bool:
    """纯比较器。Severity 是 str-Enum,不能直接用 ``<``。"""
    return _SEVERITY_ORDER[have] >= _SEVERITY_ORDER[threshold]


def _suggest_action(kind: VisionEventType, safe_mode: bool = False) -> str:
    """事件类型 → 通知里给人的建议动作。**只是建议,没有任何东西在等它。**

    ``safe_mode`` 只改一个答案,而且只因为 SAFE 模式改变了 MAST 自己肯做什么:
    在一个信念块写着「不修针」、SafetyGate 硬拦脉冲的模式里建议 ``tip_prep``,
    是在推荐一件系统自己不肯做的事 —— 用户看到过并提了 #43。
    """
    if kind is VisionEventType.E_STOP:
        return "halt"
    if kind is VisionEventType.TIP_QUALITY_DROP:
        return "manual_tip_check" if safe_mode else "tip_prep"
    if kind is VisionEventType.EMERGENCY_RETRACT_NEEDED:
        return "retract"
    return "review"


# ─────────────────────────────────────────────────────────────────────
# Factory — preferred public entry point
# ─────────────────────────────────────────────────────────────────────

def make_buffer_hitl_middleware(
    *,
    buffer: "BufferService",
    event_kinds: tuple[VisionEventType, ...] = DEFAULT_HITL_EVENT_KINDS,
    min_severity: Severity = DEFAULT_MIN_SEVERITY,
    get_mode: Callable[[], Any] | None = None,
) -> BufferHITLMiddleware:
    """按默认值构建 :class:`BufferHITLMiddleware`。

    与 :func:`make_handoff` / :func:`make_buffer_tools` 保持同一命名约定。
    """
    return BufferHITLMiddleware(
        buffer=buffer,
        event_kinds=event_kinds,
        min_severity=min_severity,
        get_mode=get_mode,
    )


__all__ = [
    "BufferHITLMiddleware",
    "make_buffer_hitl_middleware",
    "DEFAULT_HITL_EVENT_KINDS",
    "DEFAULT_MIN_SEVERITY",
    "ALWAYS_ESCALATE_KINDS",
    "SUSTAINED_PHYSICAL_SIGNALS",
    "TRANSIENT_PHYSICAL_SIGNALS",
    "classify_event",
    "escalates_to_operator",
    "exempt_during_tip_work",
    "physical_current_signals",
    "REMEDY_TOOL_NAMES",
    "REMEDY_TAGS",
    "GATE_STATE_PATH",
    "GATE_RESOLVE_PATH",
    "GATE_API_PREFIX",
    "gate_resolve_url",
    "reset_all_gates",
    "resolve_all_gates",
    "gate_states",
]
