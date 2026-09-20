"""``instrument_control`` 的 v2 装配 —— 唯一会驱动仪器的那一个。

为什么它单独一个文件
--------------------
另外六个 agent 的零件形状一致（``build_tools`` / ``SYSTEM_PROMPT`` / 通用中间件），
一份 :mod:`~mast.agentruntime.assembly` 就够。IC 不一样：它的中间件栈里有四件**硬件
安全件**，每一件都要运行时注入的可调用对象：

============================  ==================================================
``SafetyGateMiddleware``      ``limits`` / ``get_state`` / ``get_mode`` / ``recorder``
``BufferHITLMiddleware``      ``buffer`` / ``get_mode``
``AlertDeliveryMiddleware``   （读告警表，无参）
``ModeBeliefMiddleware``      ``get_mode``
============================  ==================================================

★ 「第二个真源」这件事，怎么处理的
----------------------------------
这份名单是 ``instrument_control/graph.py`` 里那张表的**第二份拷贝**，而
「名单抄第二遍就会漂」是本仓记录在案的事故形状 —— 尤其危险在这里：漂移的表现是
**守卫看起来在、实际没挂**。

处理办法不是「小心一点」，是 :func:`parity_report` ——它拿 IC **自己 build 之后**记在
``prompts.builds`` 台账里的中间件名单（按挂载顺序）来核对这一份。
``tests/v2/agents/contract/test_ic_assembly.py`` 把它钉成闸门：IC 的表一改，这里不跟着改
就红。**两份拷贝 + 一道自动对账**，是这个仓库对付同类问题的既有手法（``_AUTO_BG_MARKER``
的字面量 parity 测试同形）。

刻意不带过来的三条，各有理由
----------------------------
* ``ToolPairGuardMiddleware`` —— 它唯一的存在理由是修 ``Command(graph=PARENT)`` 短路
  留下的孤儿；新循环里每个 tool_call 在同一次 ``run()`` 里得到配对回应，孤儿在构造上
  不可能。留着它只会让人以为它还有用。
* ``ToolCallLimit`` / ``ModelCallLimit`` —— 限流是 ``AgentLoop`` 的内建，单位是调用数
  而不是 super-step。两套叠着，先撞上的那个说了算，而「哪个先撞」取决于换算率。
* ``AnthropicPromptCaching`` —— 生产默认 Kimi，这条只在 Claude 时挂；缓存断点属于
  provider 适配层（二期）。

⚠️ 上真机之前
-------------
装配好**不等于**可以上机。SafetyGate 是 794 行的状态前置条件与包络裁决，它在桥上跑
和在原生栈上跑是两条路径。计划里那条没有变：**IC 的 v2 私聊要先过安全线测试组
（`test_forensics_20260727_safety_gates` 等）才准上真机**，而且开关默认关。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from mast.agentruntime.loop import AgentLoop, CallLimits
from mast.agentruntime.middleware import Middleware
from mast.agentruntime.model import ChatModelPort, LangChainModelPort
from mast.agentruntime.tools import ToolSpec, spec_from_langchain_tool

logger = logging.getLogger(__name__)

AGENT_ID = "instrument_control"

#: 刻意不带过来的中间件（理由见模块 docstring）。parity 对账时从 IC 的名单里扣掉
#: 这些再比 —— 名字写在这里，就没有「悄悄少挂了一条」这种可能。
DELIBERATELY_OMITTED: frozenset[str] = frozenset({
    "ToolPairGuardMiddleware",          # 新循环里孤儿构造上不可能
    "ToolCallLimitMiddleware",          # AgentLoop 内建（单位是调用数）
    "ModelCallLimitMiddleware",         # 同上
    "AnthropicPromptCachingMiddleware",  # provider 适配层（二期）
})

#: 硬件安全件。**少任何一条都不许装配** —— 见 :func:`build_instrument_loop`。
REQUIRED_SAFETY: tuple[str, ...] = (
    "SafetyGateMiddleware",
    "AlertDeliveryMiddleware",
    "ModeBeliefMiddleware",
)


def build_instrument_middleware(
    *, buf: Any = None, get_state: Callable[[], Any] | None = None,
    get_mode: Callable[[], Any] | None = None,
    registry: Any = None, safety_limits: Any = None,
    safety_recorder: Any = None, turn_recorder: Any = None,
    enable_hitl: bool = True, interrupt_on: dict | None = None,
    extra: Iterable[Any] = (), visibility_mw: Any = None,
) -> list:
    """IC 的中间件表（**langchain 实例，未桥接**），顺序与 ``graph.py`` 一致。

    顺序在 ``wrap_*`` 的洋葱里是有意义的，注释里那些「排在 X 之后 = 在它里面」的
    论证全部依赖它 —— 所以这里照抄顺序，parity 测试也比顺序。

    ``visibility_mw`` 由调用方传进来，因为它**与工具表耦合**
    （``make_tool_visibility_middleware(agent, tools, registry)`` 要先有 tools）。
    ⚠️ 这一条是 parity 闸门第一次跑就抓到的缺口 —— 它在 IC 的实际名单里而我漏了。
    IC 有约 280 个工具，没有它模型每次都看见全部。
    """
    from mast.agents._shared.alert_delivery_mw import AlertDeliveryMiddleware
    from mast.agents._shared.experiment_prefs import ExperimentPrefsMiddleware
    from mast.agents._shared.instrument_profile_mw import (
        InstrumentProfileMiddleware,
    )
    from mast.agents._shared.live_state_mw import LiveStateMiddleware
    from mast.agents._shared.mode_mw import ModeBeliefMiddleware
    from mast.agents._shared.prefill_guard_mw import ClaudePrefillGuardMiddleware
    from mast.agents._shared.safety_mw import SafetyGateMiddleware
    from mast.agents._shared.stall_guard_mw import StallGuardMiddleware
    from mast.agents._shared.upstream_mw import UpstreamArtifactMiddleware
    from mast.agents._shared.vision_mw import SkillImageMiddleware
    from mast.config import SafetyLimits

    mw: list = [
        *(extra or []),
        UpstreamArtifactMiddleware(AGENT_ID),
        ExperimentPrefsMiddleware(),
        InstrumentProfileMiddleware(),
        LiveStateMiddleware(get_state=get_state),
        ModeBeliefMiddleware(get_mode=get_mode),
        AlertDeliveryMiddleware(),
        StallGuardMiddleware(agent_name=AGENT_ID),
        SafetyGateMiddleware(
            limits=safety_limits or SafetyLimits(),
            get_state=get_state, get_mode=get_mode, recorder=safety_recorder),
    ]

    if buf is not None:
        # CRITICAL 事件（E_STOP / 针尖质量骤降）落地的那一刻停下来。
        from mast.agents._shared.buffer_hitl import make_buffer_hitl_middleware

        mw.append(make_buffer_hitl_middleware(buffer=buf, get_mode=get_mode))

    if enable_hitl:
        # ⑰(2026-08-08)：审批 → 提醒。判据没删也没放宽，它搬进了
        # ``core.auto_approval.would_have_asked``；这条中间件做的是「执行+留痕+通知」。
        # 排在 SafetyGate **之后** = 在它里面：通知说的是「不再等批准、直接执行」，
        # 若 SafetyGate 随后拒了，那句话就成了假话。
        from mast.agents._shared.auto_approval_mw import (
            AutoApprovalNoticeMiddleware,
        )
        # 2026-08-27：这个判定从 IC 的 ``graph.py`` 搬到了 ``_shared/hitl_map.py``。
        # 之前这里 import 的是一个**即将被删的图模块** —— 而这个判定本身一行
        # langgraph 都没有（判据是技能的 ``safety_level``）。抄一份到这里会造出
        # 第二个真源，而它漂掉的症状是「本来该通知的没通知」。
        from mast.agents._shared.hitl_map import derive_hitl_map

        gated = interrupt_on if interrupt_on is not None else derive_hitl_map(
            registry, owner="agentruntime.ic_assembly")
        mw.append(AutoApprovalNoticeMiddleware(
            get_mode=get_mode, extra_names=frozenset(gated.keys())))

    if visibility_mw is not None:
        # 位置照抄 IC：其余注入器之后（目录块要贴在 system 末尾，那是模型最后读到
        # 的一段）。收窄是**只减不加**的子集操作，永远合法。
        mw.append(visibility_mw)

    mw.append(SkillImageMiddleware())
    mw.append(ClaudePrefillGuardMiddleware())

    from mast.agents._shared.message_clock_mw import MessageClockMiddleware

    mw.append(MessageClockMiddleware())

    if turn_recorder is not None:
        from mast.agents._shared.recorder_mw import RecorderMiddleware

        mw.append(RecorderMiddleware(agent_id=AGENT_ID, recorder=turn_recorder))
    return mw


class MissingSafetyMiddleware(RuntimeError):
    """装配 IC 时少了硬件安全件 —— 拒绝，而不是「先跑起来再说」。"""


def build_instrument_loop(
    *, buf: Any = None, model: Any = None,
    get_state: Callable[[], Any] | None = None,
    get_mode: Callable[[], Any] | None = None,
    context_provider: Callable[[], Any] | None = None,
    registry: Any = None, safety_limits: Any = None,
    safety_recorder: Any = None, turn_recorder: Any = None,
    enable_hitl: bool = True, interrupt_on: dict | None = None,
    extra_middleware: Iterable[Any] = (),
    max_model_calls: int = 30, max_tool_calls: int = 80,
    system_suffix: str = "",
) -> AgentLoop:
    """装配 IC 的 v2 循环。

    ★ **少一件安全件就抛**（:class:`MissingSafetyMiddleware`），不是记一条 warning
    继续。一个「安全件没挂上但 agent 照跑」的 IC 循环，症状是**它一切正常，直到某次
    动作本该被拒**——那时代价已经落在硬件上了。
    """
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT
    from mast.agentruntime.compat import bridge_all

    # 工具先建：可见性中间件与工具表耦合（它按调用收窄模型看得见的那一份）。
    lc_tools, visibility_mw = _instrument_tool_surface(buf, registry,
                                                       context_provider)

    lc_mw = build_instrument_middleware(
        buf=buf, get_state=get_state, get_mode=get_mode, registry=registry,
        safety_limits=safety_limits, safety_recorder=safety_recorder,
        turn_recorder=turn_recorder, enable_hitl=enable_hitl,
        interrupt_on=interrupt_on, extra=extra_middleware,
        visibility_mw=visibility_mw)

    names = {type(m).__name__ for m in lc_mw}
    missing = [n for n in REQUIRED_SAFETY if n not in names]
    if missing:
        raise MissingSafetyMiddleware(
            f"instrument_control 的循环缺少硬件安全中间件：{missing}。"
            "拒绝装配 —— 一个安全件没挂上但照跑的 IC，症状是它一切正常，"
            "直到某次动作本该被拒。")

    tools = _as_specs(lc_tools)
    prompt = _instrument_prompt(SYSTEM_PROMPT) + (system_suffix or "")

    return AgentLoop(
        name=AGENT_ID,
        model=_as_port(model),
        tools=tools, system_prompt=prompt,
        middleware=bridge_all(lc_mw, agent_id=AGENT_ID),
        limits=CallLimits(max_model_calls=max_model_calls,
                          max_tool_calls=max_tool_calls))


def parity_report(*, buf: Any = None, registry: Any = None) -> dict:
    """把这份名单与 **IC 自己 build 之后记下的那份**对账。

    返回 ``{"ours": [...], "theirs": [...], "missing": [...], "extra": [...]}``。
    ``missing`` = IC 有而我们没有且**不在** :data:`DELIBERATELY_OMITTED` 里的
    —— 那就是漂移，测试据此变红。

    ``theirs`` 取自 ``prompts.builds.last_build("instrument_control").middleware``
    （按挂载顺序），而不是重新读一遍源码：台账记的是 ``create_agent`` **实际收到**
    的那个列表，比任何静态解析都可信。

    ★ 这道闸门**会随删除退役，那是设计不是遗漏**（2026-08-27 副本试验确认）
    ---------------------------------------------------------------------
    参照物是 IC 的 ``graph.py`` 跑一次 ``build()`` 之后留下的台账。那个文件将随
    langgraph 一起移除，删掉之后 ``theirs`` 恒为空 —— 副本里
    ``TestParityWithTheRealBuild`` 的三条会红，而它们红得**正确**：一份对账在只剩
    一份账本的那天就没有工作了。

    删除时应当**删掉那三条测试**，而不是想办法让它们绿（比如把 IC 的表抄一份进
    测试当参照 —— 那正是这道闸门当初要消灭的第二真源）。

    退役之后，它守的那个不变式由**另外两样**接着守，两样都已经在：

    * :class:`MissingSafetyMiddleware` —— 装配时少一件安全件就**抛**，不是记
      warning 继续。它答的是「安全件齐不齐」，而那本来就是这道闸门真正关心的事；
    * :data:`DELIBERATELY_OMITTED` —— 「就是没加」与「漏了」的区别写在代码里。

    换句话说：parity 答的是「两份有没有分叉」，删除之后**没有两份了**；而「这一份
    够不够安全」从来是另一条守卫的工作。
    """
    from mast.prompts.builds import last_build

    rec = last_build(AGENT_ID)
    theirs = list(getattr(rec, "middleware", ()) or ())
    # 用与真实装配**同一条路**造这份名单（含与工具表耦合的可见性中间件），
    # 否则对账对的是一个比生产少几条的影子。
    _tools, visibility_mw = _instrument_tool_surface(buf, registry, lambda: None)
    ours = [type(m).__name__ for m in build_instrument_middleware(
        buf=buf, registry=registry, visibility_mw=visibility_mw)]

    theirs_wanted = [n for n in theirs if n not in DELIBERATELY_OMITTED]
    missing = [n for n in theirs_wanted if n not in ours]
    extra = [n for n in ours if n not in theirs]
    return {"ours": ours, "theirs": theirs, "theirs_wanted": theirs_wanted,
            "missing": missing, "extra": extra}


# ── 内部 ───────────────────────────────────────────────────────────────
def _as_port(model: Any) -> ChatModelPort:
    """Wrap unless it already IS a port.

    ``ChatModelPort`` is a structural Protocol (``invoke`` + ``stream``) and every
    LangChain ``BaseChatModel`` has both, so ``isinstance(model, ChatModelPort)`` said
    yes to raw LangChain models and the loop handed them a ``ModelRequest``
    (``ValueError: Invalid input type … ModelRequest``). A LangChain model is told
    apart by ``bind_tools`` — the one thing a port never has.
    """
    if model is None:
        return _port(None)
    if isinstance(model, LangChainModelPort):
        return model
    if hasattr(model, "bind_tools"):
        return LangChainModelPort(model)
    return model if isinstance(model, ChatModelPort) else _port(model)


def _port(model: Any) -> ChatModelPort:
    if model is not None:
        return LangChainModelPort(model)
    from mast.agents._shared.models import make_chat_model

    return LangChainModelPort(make_chat_model(AGENT_ID))


def _instrument_prompt(default: str) -> str:
    try:
        from mast.prompts.registry import resolve as resolve_prompt

        base = resolve_prompt(f"agent.{AGENT_ID}.system", default)
    except Exception as exc:  # noqa: BLE001 — 覆写层坏了也要能跑
        logger.warning("prompt override unavailable for %s: %s", AGENT_ID, exc)
        base = default
    return base + _manual_index_block()


def _manual_index_block() -> str:
    """Nanonis 模块速查 —— 旧图拼在系统提示末尾的那 1218 字符。

    ★ 为什么这里少了它（2026-08-27 实测发现）
    ---------------------------------------
    拿 IC 两边的提示词对字符数：

        旧图 14285 = agent.instrument_control.system(13067)
                    + agent.instrument_control.manual_index(1218)
        新装配 13067                                    ★ 少 1218

    工具数完全一致（401），少的正好是这一块 —— **告诉模型这台仪器有哪些 Nanonis
    模块、各管什么**的那份速查。

    这是 literature 少 187 字符（``unavailable_note``）的同一个形状，但落在
    **唯一一个会动机器的 agent** 上：模型不知道模块布局时，它不会报错，它会去猜。

    ⚠️ 顺带更正闸门里此前写下的一条：「对照不覆盖 instrument_control（旧路径要真
    硬件栈）」—— **不成立**。实测两边都在无硬件下 build 成功（旧图 459 技能、新装配
    401 工具）。IC 因此被排除在对照之外，而它恰恰是最该对的那个。

    与旧图**同一条纪律**：速查是可选的，取不到就留空，绝不因此挡住装配 ——
    「manual 缺席」和「agent 起不来」不该是同一件事。
    """
    try:
        from mast.knowledge.nanonis_manual import modules_index

        idx = modules_index()
    except Exception as exc:  # noqa: BLE001 — 手册是可选的，永不阻塞装配
        logger.debug("nanonis manual index unavailable: %s", exc)
        return ""
    return f"\n\n# Nanonis 模块速查\n\n{idx}" if idx else ""


def _instrument_tool_surface(buf, registry, context_provider,
                             tool_packs: bool = True):
    """``(langchain 工具列表, 可见性中间件)``。

    照抄 IC 的顺序：先 ``build_tools``，再挂上 tool-finder 工具，最后用**完整的**
    工具表造可见性中间件。顺序不能换 —— 中间件按调用收窄的是它建立时看到的那一份
    目录，漏了 finder 工具，模型就没法把收窄了的部分找回来。

    **交棒工具不给**（私聊里没有父编排器）。
    """
    from mast.agents._shared.tool_finder_tools import make_tool_finder_tools
    from mast.agents._shared.tool_visibility_mw import (
        make_tool_visibility_middleware,
    )
    from mast.agents.instrument_control.tools import build_tools

    # 签名是 ``build_tools(buf, context_provider, registry=None, targets=…)``
    # —— context_provider 是**位置参数**，按名传会 TypeError。
    raw = [t for t in build_tools(buf, context_provider, registry=registry)
           if not str(getattr(t, "name", "")).startswith("handoff_to_")]

    catalog_box: dict = {}
    raw = raw + make_tool_finder_tools(lambda: catalog_box.get("catalog"))
    try:
        visibility_mw, catalog = make_tool_visibility_middleware(
            AGENT_ID, raw, registry, enabled=bool(tool_packs))
        catalog_box["catalog"] = catalog
    except Exception as exc:  # noqa: BLE001 — 收窄失败 = 全都看得见，不是不能跑
        logger.warning("tool visibility unavailable for %s: %s", AGENT_ID, exc)
        visibility_mw = None
    return raw, visibility_mw


def _as_specs(lc_tools) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for t in lc_tools:
        try:
            out.append(spec_from_langchain_tool(t, touches_instrument=True))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping IC tool %s: %s",
                           getattr(t, "name", "?"), exc)
    return out


__all__ = [
    "build_instrument_loop",
    "build_instrument_middleware",
    "parity_report",
    "MissingSafetyMiddleware",
    "DELIBERATELY_OMITTED",
    "REQUIRED_SAFETY",
    "AGENT_ID",
]
