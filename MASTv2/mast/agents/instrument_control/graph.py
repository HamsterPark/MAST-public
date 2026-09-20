"""Instrument-Control agent — Phase 4 real implementation.

Replaces the Phase 3 stub. Uses langchain.agents.create_agent + middleware
stack per compass §4.4. The build() signature stays compatible:

    build(buf, context_provider, registry=None, model=None,
          checkpointer=None) -> CompiledStateGraph

`model` defaults to mast.agents._shared.models.AGENT_MODEL["instrument_control"]
(Sonnet 4.6); pass GenericFakeChatModel in tests.

`context_provider` is a callable returning an ExecutionContext-like object
each invocation (the wrap_skill adapter calls it).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_anthropic import ChatAnthropic

from mast.agents._shared.alert_delivery_mw import AlertDeliveryMiddleware
from mast.agents._shared.call_limits import DEFAULT_MODEL_CALLS_PER_RUN
from mast.agents._shared.auto_approval_mw import AutoApprovalNoticeMiddleware
from mast.agents._shared.experiment_prefs import ExperimentPrefsMiddleware
from mast.agents._shared.hitl_map import (
    DEFAULT_HITL_DECISIONS,
    HITL_DECISION_OVERRIDES,
    derive_hitl_map,
)
from mast.agents._shared.instrument_profile_mw import InstrumentProfileMiddleware
from mast.agents._shared.live_state_mw import LiveStateMiddleware
from mast.agents._shared.mode_mw import ModeBeliefMiddleware
from mast.agents._shared.models import AGENT_MODEL, make_chat_model
from mast.agents._shared.safety_mw import SafetyGateMiddleware
from mast.agents._shared.stall_guard_mw import StallGuardMiddleware
from mast.agents._shared.upstream_mw import UpstreamArtifactMiddleware
from mast.agents.state import AgentSubState
from mast.config import SafetyLimits

from mast.prompts.builds import record_build
from mast.prompts.registry import resolve as resolve_prompt
from .prompts import SYSTEM_PROMPT
from .tools import build_tools, discover_instrument_skills

if TYPE_CHECKING:
    from mast.buffer.service import BufferService
    from mast.core.registry import SkillRegistry
    from mast.core.types import HardwareState

logger = logging.getLogger(__name__)


# ⑰(2026-08-08)/2026-08-27：这张表与下面的推导函数**已搬到**
# ``_shared/hitl_map.py``。它一行 langgraph 都没有（判据是技能自己的
# ``metadata().safety_level``），住在这个图文件里只是因为这张图是第一个需要它的人
# —— 而新运行时的 ``agentruntime/ic_assembly.py`` 为了拿它，曾经得 import 这个
# 即将被删的模块。判据抄第二遍的症状是「本来该通知的没通知」，而那种漏法在日志里
# 和「今天没有 DANGEROUS 技能」长得一模一样，所以是搬走而不是复制。
#
# 下面三行是**再导出**，旧名字原样保留 —— 既有调用点与测试一个都不用改。
_HITL_DECISION_OVERRIDES = HITL_DECISION_OVERRIDES
_DEFAULT_HITL_DECISIONS = DEFAULT_HITL_DECISIONS
_derive_hitl_map = derive_hitl_map


def build(
    buf: "BufferService | None",
    context_provider: Callable[[], Any],
    *,
    registry: "SkillRegistry | None" = None,
    model: Any | None = None,
    safety_limits: SafetyLimits | None = None,
    get_state: Callable[[], "HardwareState"] | None = None,
    get_mode: Callable[[], Any] | None = None,
    checkpointer: Any = None,
    max_model_calls: int = 40,
    max_tool_calls: int = 120,
    # 出厂值从 call_limits 派生 —— **别在这里再写一个字面量**。
    # 2026-08-10 之前这六张图各自硬编码 30,和 call_limits 的常量是六份副本;
    # 把常量从 30 提到 500 时,这六处会静静地留在 30。
    max_model_calls_per_run: int = DEFAULT_MODEL_CALLS_PER_RUN,
    max_tool_calls_per_run: int = 80,
    enable_hitl: bool = True,
    interrupt_on: dict | None = None,
    extra_tools: list | None = None,
    post_hook=None,
    recorder=None,
    safety_recorder=None,
    turn_recorder=None,
    standalone: bool = False,
    extra_middleware: list | None = None,
    tool_packs: bool | None = None,
):
    """Build the Instrument-Control agent.

    Args:
        buf:               BufferService for read_latest_tip_status etc.
                           (None disables buffer tools — useful in pure
                           offline tests).
        context_provider:  callable() -> ExecutionContext (or FakeCtx in tests)
        registry:          SkillRegistry; default is auto-discovered
                           from `mast.skills.builtins`.
        model:             ChatAnthropic instance OR a fake chat model for tests.
                           Default: ChatAnthropic via AGENT_MODEL["instrument_control"].
        safety_limits:     SafetyLimits override; default uses code defaults
                           (compass-blueprint values).
        get_state:         optional callable returning HardwareState (used by
                           SafetyGateMiddleware for state-precondition layer)
        checkpointer:      LangGraph checkpointer (SqliteSaver / InMemorySaver)
        max_model_calls:   ModelCallLimitMiddleware cap (R7 loop guard)
        max_tool_calls:    ToolCallLimitMiddleware cap (R7 loop guard)
        enable_hitl:       wire AutoApprovalNoticeMiddleware — DANGEROUS 技能与
                           SEMI 电脉冲**照跑,并在诊断台账里留一行**。⑰ 之前它
                           挂的是 HumanInTheLoopMiddleware(弹框等人批);名字保留
                           是因为六处调用方按名传,而它回答的问题没变。
                           False = 连通知都不发(纯离线测试)。
        interrupt_on:      显式覆盖「哪些技能算 DANGEROUS」的名单(skill_name ->
                           {...})。默认由 _derive_hitl_map(registry) 从 metadata
                           派生。⑰ 之后这个名单**只影响启动日志**:运行时的判据是
                           mast.core.auto_approval.would_have_asked,它直接读
                           metadata.safety_level(同一个真源,不经过这张表)。
        extra_tools:       extra NON-hardware tools to merge in (e.g. the shared
                           persistent-memory tools from agents._shared). Passed
                           in by the orchestrator/GUI so this agent never imports
                           a sibling agent. These are plain @tool callables with
                           no skill metadata, so SafetyGateMiddleware.wrap_tool_call
                           sees meta_obj is None and passes them through untouched
                           — the instrument-safety path is unaffected. None = none.
    """
    if registry is None:
        registry = discover_instrument_skills()
    if model is None:
        model = make_chat_model(
            "instrument_control",
            # 8192 而不是 4096:技能工坊的 spec JSON 走 tool call 的 arguments,
            # 一份 20 节点的 spec 就有 2000+ token,而它要和这一轮回答的其余部分
            # 共用这个上限。**截断落在 arguments 中间产生的是非法 JSON** —— 一个
            # 困惑的失败,不是干净的失败(DP 加编程能力时同一处、同一个理由)。
            max_tokens=8192,
            temperature=0.2,
        )

    # standalone=True (private 1:1 chat, no orchestrator parent) suppresses the
    # handoff tools — a handoff_to_supervisor with no parent graph errors.
    _targets = () if standalone else ("supervisor", "data_processing")
    tools = build_tools(buf, context_provider, registry=registry, targets=_targets,
                        post_hook=post_hook, recorder=recorder) + list(extra_tools or [])

    # ── 按需加载：目录 + 两个元工具（2026-08-24）────────────────────────
    #
    # 实测这个 agent 的工具 schema 是 354 021 字符 —— 静态系统提示词的 18.2 倍，
    # 而且不是「几个大工具」的问题（中位数 545 字符，最大的 20 个只占 24.4%）。
    # 所以压缩描述治不了它，只能让大多数工具**不出现在这一次调用里**。
    #
    # 这是**目录**不是**门禁**：ToolNode 下面注册的仍然是全部工具，SafetyGate /
    # validator / 自主度策略一个字节没动。2026-08-20「工具面全开」那条裁决讲的是
    # 权限，这里改的是可见性。
    if tool_packs is None:
        # 没显式传就读用户的设置（默认开）。两条建图路径（私聊 runtime、群聊
        # orchestrator）都不必各传一次 —— 「每页各自记得」的接线是这个仓反复漏
        # 掉的形状，一处漏掉的症状是「私聊省了、群聊没省」，而那从外面看不出来。
        try:
            from mast.webui.settings_store import settings_store_for_runtime
            _v = settings_store_for_runtime().get("tool_packs_enabled")
            tool_packs = True if _v is None else bool(_v)
        except Exception:  # noqa: BLE001 — 读不到设置就按默认开
            tool_packs = True

    _catalog_box: dict = {}
    from mast.agents._shared.tool_finder_tools import make_tool_finder_tools
    from mast.agents._shared.tool_visibility_mw import (
        make_tool_visibility_middleware,
    )
    tools = tools + make_tool_finder_tools(lambda: _catalog_box.get("catalog"))
    _visibility_mw, _catalog = make_tool_visibility_middleware(
        "instrument_control", tools, registry, enabled=bool(tool_packs))
    _catalog_box["catalog"] = _catalog

    middleware = [
        # extra_middleware (compaction + memory recall, supplied by the caller)
        # runs OUTERMOST: compaction.before_model trims the history before any
        # other middleware/model sees it; memory recall appends its block first.
        *(extra_middleware or []),
        # UpstreamArtifactMiddleware — what the agents before IC produced: the
        # plan to execute, the last scan. Before this, a plan handed over by
        # experiment_design reached IC as the sentence "ExperimentPlan ready for
        # execution" and nothing else.
        UpstreamArtifactMiddleware("instrument_control"),
        # ExperimentPrefsMiddleware — inject the operator's default-parameter
        # preferences (设置 → 实验默认参数, #147). Placed BEFORE live-state so the
        # magnitude/live-state block stays the freshest (last) context; a no-op
        # when nothing is set, live-read (no rebuild on a settings change).
        ExperimentPrefsMiddleware(),
        # InstrumentProfileMiddleware — inject this rig's 进/退针机理 + 退针方向 /
        # lock-in dI/dV 参数 + learned 到样品标定值, so the agent reasons about
        # 换样品退针 / 进针判距离 with real instrument knowledge (co-design 2026-07-20).
        # Live-read holder (mast.core.instrument_profile); no rebuild on a change.
        InstrumentProfileMiddleware(),
        # LiveStateMiddleware — runs after the injected ones, so the appended
        # state block reaches the inner SafetyGateMiddleware's logging too.
        LiveStateMiddleware(get_state=get_state),
        # ModeBeliefMiddleware — append the operating-mode belief block AFTER the
        # live-state block, so it is the freshest (last) context the model sees.
        # In SAFE it tells the agent the tip is fine and to just keep running
        # experiments (overriding the static prompt's tip-repair guidance); in
        # SEMI it steers toward experimenting + pulse-needs-confirmation; AUTO /
        # no get_mode appends nothing.
        ModeBeliefMiddleware(get_mode=get_mode),
        # AlertDeliveryMiddleware — 电流监控告警的**送达**(2026-08-10)。
        #
        # 排在这里 = 块比 live-state 和模式信念都新,是模型看到的最后一段事实性
        # 上下文。这正是它要的位置:它说的是「就在刚才发生了什么」。
        #
        # 为什么需要它:2026-08-08 一条 CRITICAL saturation 报出来之后 agent 又扫了
        # 13 分钟。探测层全对(规则准、处方对、emitted_buffer=true),断的是送达 ——
        # buffer_hitl 把事件 id 写进 state["event_refs"],而 event_refs 全树**没有
        # 读者**;WARN 则压根不进 buffer。见本中间件的模块 docstring(四条断点)。
        #
        # 不带 buf 的条件:它读的是**告警表**,与视觉缓冲区无关。纯离线测试里
        # 库不存在,``_apply`` 整段包在 try 里,退化成 no-op。
        AlertDeliveryMiddleware(),
        # StallGuardMiddleware — break a same-tool/same-error retry spin before it
        # burns the recursion budget (2026-07-06: the agent looped on bias_nonzero
        # / comms timeouts up to the 150-step limit). Nudge-only: it appends a
        # stop-and-escalate directive to the per-call request; it never crashes a
        # run, and the orchestrator recursion limit remains the hard backstop.
        StallGuardMiddleware(agent_name="instrument_control"),
        SafetyGateMiddleware(
            limits=safety_limits or SafetyLimits(),
            get_state=get_state,
            get_mode=get_mode,
            recorder=safety_recorder,
        ),
    ]
    # Buffer-driven runtime HITL (2026-06-11): pause the (parent) planner graph
    # the instant a CRITICAL event lands — an E_STOP (a future PHYSICAL e-stop
    # button will emit a buffer E_STOP event) or a CRITICAL tip-quality drop.
    # before_model peeks the buffer queues non-blocking, so a clean run is
    # unaffected; the interrupt bubbles up to the supervisor for operator
    # resume. Wired only when a buffer is present (None in pure-offline tests).
    if buf is not None:
        from mast.agents._shared.buffer_hitl import make_buffer_hitl_middleware
        # get_mode so the interrupt's `suggested_action` matches what MAST will
        # actually do: SAFE mode refuses tip conditioning, so it must not tell
        # the operator to run one ().
        middleware.append(make_buffer_hitl_middleware(buffer=buf,
                                                      get_mode=get_mode))
    if enable_hitl:
        # ⑰(2026-08-08)审批 → 提醒。这里曾经挂着两个会 ``interrupt()`` 的中间件:
        # ``HumanInTheLoopMiddleware(interrupt_on=_derive_hitl_map(registry))``
        # (每个 DANGEROUS 技能一次审批框)和 ``ModeGatedPulseHITLMiddleware``
        # (SEMI 模式下每一发电脉冲一次)。两天实机里它们的成绩是**零真阳性**,
        # 把自动运行打成了值守运行,定案整条打断链割掉。
        #
        # 判据没删也没放宽 —— 它搬进了 ``mast.core.auto_approval.would_have_asked``,
        # 由这条通知中间件、``skill_adapter``(写 approvals 审计行)和
        # ``core/executor`` + ``core/execution_context`` 共用同一份答案。
        #
        # 参数名 ``enable_hitl`` / ``interrupt_on`` 保留:上游有六处调用方按名传
        # (runtime 两条链、orchestrator、多份测试),而它们表达的意思没变 ——
        # 「这条 agent 要不要对 DANGEROUS 动作做点什么」。做的事从「拦住等人」
        # 变成了「照跑并通知」。
        #
        # 排在 SafetyGateMiddleware **之后** = 在它里面:通知说的是「不再等批准,
        # 直接执行」,若 SafetyGate 随后拒了,那句话就成了假话。
        _gated = interrupt_on if interrupt_on is not None else _derive_hitl_map(registry)
        middleware.append(AutoApprovalNoticeMiddleware(
            get_mode=get_mode,
            # 生产上这份名单与 would_have_asked 同源(都是 safety_level==DANGEROUS),
            # 所以传进去只是**冗余一致**;它存在的意义是让
            # ``build(..., interrupt_on={...})`` 这个参数继续真的有作用 —— 一个
            # 读起来像可调项、调了却没反应的参数是本仓踩过的坑。名单只能加不能减。
            extra_names=frozenset(_gated.keys()),
        ))
        logger.info(
            "instrument_control: %d 个 DANGEROUS 技能改为「执行+留痕+通知」(不再弹框): %s",
            len(_gated), sorted(_gated.keys()),
        )
    # Loop guards. ``thread_limit`` is cumulative across the whole conversation
    # (a long-session backstop); ``run_limit`` resets each agent INVOCATION and is
    # the real circuit-breaker against an in-turn ReAct spin (e.g. retrying a
    # precondition/safety-blocked tool). On hitting the limit the agent ENDS
    # gracefully instead of spinning until the global recursion_limit kills the
    # whole run (the GraphRecursionError the operator hit).
    #
    # CORRECTION (2026-07-28): exit_behavior="end" does NOT "return control to
    # the supervisor" — the earlier comment here, and dispatch_walkthrough.html
    # §5.1 item 14, both said so and both were wrong. LangChain's limit
    # middlewares return ``{"jump_to": "end"}``
    # (``model_call_limit.py:208`` / ``tool_call_limit.py:451-456``), which is
    # control flow INSIDE the create_agent subgraph. The parent graph adds these
    # subgraphs as bare nodes with no outgoing edge, so when the subgraph ends
    # the branch simply stops: the supervisor is not re-entered, and
    # ``graph.stream()`` runs dry without raising. The ONLY way back is the
    # model calling a handoff tool.
    #
    # That is a real fail-silent termination and it is now caught by the driver,
    # not by this comment: ``api/routes/orchestrator.py`` latches
    # ``active_agent == "__end__"`` and reports ``failed=True`` when the stream
    # is exhausted without it.
    middleware.extend([
        ToolCallLimitMiddleware(
            thread_limit=max_tool_calls, run_limit=max_tool_calls_per_run,
            exit_behavior="end"),
        ModelCallLimitMiddleware(
            thread_limit=max_model_calls, run_limit=max_model_calls_per_run),
    ])
    # AnthropicPromptCachingMiddleware is appropriate for ChatAnthropic only;
    # tests with GenericFakeChatModel should not include it.
    if isinstance(model, ChatAnthropic):
        try:
            from langchain_anthropic.middleware.prompt_caching import (
                AnthropicPromptCachingMiddleware,
            )
            middleware.append(
                AnthropicPromptCachingMiddleware(
                    ttl="1h",
                    min_messages_to_cache=1,
                    unsupported_model_behavior="ignore",
                )
            )
        except ImportError:
            logger.debug("AnthropicPromptCachingMiddleware not available; skipping cache middleware.")

    # F5 residual (2026-06-08): guard no-prefill Claude models — see prefill_guard_mw.
    from mast.agents._shared.prefill_guard_mw import ClaudePrefillGuardMiddleware
    from mast.agents._shared.tool_pair_guard_mw import ToolPairGuardMiddleware
    from mast.agents._shared.vision_mw import SkillImageMiddleware
    # ToolPairGuard (added OUTER of the prefill guard so prefill keeps the final
    # ends-on-user say for Claude): strips the UNMATCHED handoff tool messages the
    # group orchestrator leaves in the parent channel — a handoff returns
    # Command(goto=PARENT) which short-circuits the subgraph, dropping the AIMessage
    # that CALLED handoff, so only the orphan handoff ToolMessage reaches the parent
    # transcript. The next agent seeded with that history else 400s the provider with
    # "tool_call_id is not found".
    # ToolVisibilityMiddleware — 收窄这一次调用能看见的工具 schema（2026-08-24）。
    #
    # 位置：其余注入器之后（目录块要贴在 system 末尾，那是模型最后读到的一段），
    # 缓存中间件之内（工具前缀要在它打断点之前定下来）。收窄是**只减不加**的
    # 子集操作，永远合法 —— factory 只校验中间件*加进来*的工具。
    middleware.append(_visibility_mw)
    middleware.append(ToolPairGuardMiddleware())
    # 图像通道的出站那一半:把技能挂在 ToolMessage 上的图像**路径**读成 data URI
    # 挂进本次请求。模型不支持视觉 / 没有图 / 图读不出来 → 原样返回,退化成纯文本
    # (理由与形状见 vision_mw 模块文档)。装在 prefill guard 之前,让 prefill
    # 仍然拥有 ends-on-user 的最终发言权。
    middleware.append(SkillImageMiddleware())
    middleware.append(ClaudePrefillGuardMiddleware())

    # 给这一轮新产生的消息盖 wall-clock 时刻，让转录和旁白排得到一起去。
    # 2026-08-12 要求：agent 的发言也带上时间——
    # 而 `frontend/src/lib/narration.ts` 里那句「要让它变精确，得给
    # render_history 的每条消息加稳定时间戳」说的正是它。
    # 无条件挂：它不发请求、不改内容，只在 additional_kwargs 上加一个数；
    # 而「只在某些配置下才有时间戳」会让前端那半永远处在「有时有、有时没有」
    # 的状态，那种缺席没人查得动。
    from mast.agents._shared.message_clock_mw import MessageClockMiddleware
    middleware.append(MessageClockMiddleware())

    # Training-log: capture each agent turn (reasoning + tool_calls + usage) when
    # the GUI injects a recorder. No-op (None) keeps the agent untouched.
    if turn_recorder is not None:
        from mast.agents._shared.recorder_mw import RecorderMiddleware
        middleware.append(RecorderMiddleware(agent_id="instrument_control", recorder=turn_recorder))

    # Append the deduped Nanonis software-manual module index ONCE to the system
    # prompt (manual Integration A) — NOT per tool description (review COST-1:
    # 176 tools → 8 distinct hints → ~9k duplicated tokens/turn). Per-skill depth
    # is fetched on demand via the `nanonis_manual` tool. Graceful: empty string
    # when the manual hasn't been extracted on this machine.
    _base_prompt = resolve_prompt("agent.instrument_control.system", SYSTEM_PROMPT)
    _manual_index = ""
    try:
        from mast.knowledge.nanonis_manual import modules_index as _nm_index
        _idx = _nm_index()
        if _idx:
            _manual_index = "\n\n# Nanonis 模块速查\n\n" + _idx
    except Exception:  # noqa: BLE001 - manual is optional; never block agent build
        pass
    _system_prompt = _base_prompt + _manual_index

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=_system_prompt,
        middleware=middleware,
        name="instrument_control",
        checkpointer=checkpointer,
        # The artifact channel lives in state. WITHOUT a schema the subgraph
        # compiles with LangChain's bare AgentState, every channel below
        # "messages" is undeclared, and a tool's Command(update={...}) for it is
        # dropped with NO error -- which is why the skill adapter's
        # executed_skills / scan_paths / composite_progress writes never actually
        # landed either (verified empirically 2026-07-29).
        state_schema=AgentSubState,
    )
    setattr(agent, "mast_tool_catalog", _catalog)
    record_build("instrument_control", middleware=middleware, tools=tools,
                 tool_catalog=_catalog,
                 system_blocks=[("agent.instrument_control.system", _base_prompt),
                                ("agent.instrument_control.manual_index", _manual_index)],
                 standalone=standalone)
    logger.info("instrument_control agent built: %d tools (%d core visible, "
                "%d chars schema → %d), %d middlewares",
                len(tools), len(_catalog.core), _catalog.total_chars,
                _catalog.core_chars, len(middleware))
    return agent


__all__ = ["build", "_derive_hitl_map"]
