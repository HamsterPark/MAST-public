"""Literature Reading agent — Phase 6 real implementation.

Replaces the Phase-3 stub. Uses langchain.agents.create_agent + middleware
stack per compass §4.6. The build() signature mirrors data_processing:

    build(buf, *, model=None, checkpointer=None,
          max_model_calls=40, max_tool_calls=120) -> CompiledStateGraph

`model` defaults to make_chat_model("literature"), i.e. AGENT_MODEL["literature"]
(Kimi K3 per operator request 2026-07-18 — served via the OpenAI-compatible
ChatOpenAI subclass, NOT ChatAnthropic). Pass a GenericFakeChatModel in tests, or
an explicit ChatAnthropic instance to run on Claude.

Literature does NOT include SafetyGateMiddleware or HumanInTheLoopMiddleware
— it is a read-only corpus/web search agent that never commands the instrument.

Note: AnthropicPromptCachingMiddleware is attached ONLY when `model` is a real
ChatAnthropic instance (e.g. an explicitly injected Claude model). With the
default Kimi K3 the cache middleware is intentionally skipped, since Moonshot's
OpenAI-compatible endpoint does not honour the Anthropic cache-control headers.

P1 library curation: build_tools() also attaches the library tools
(lib_list/create/switch/add/remove/search, fetch_paper_abstract,
propose_citations, literature_priors). These are BOUNDED curation tools that
touch only the JSON registry + the local parquet index — never the instrument —
so they intentionally stay OFF the SafetyGate / HITL instrument-safety path
(design G8). The ToolCallLimit / ModelCallLimit loop guards still apply to them.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from mast.agents._shared.upstream_mw import UpstreamArtifactMiddleware
from mast.agents.state import AgentSubState
from mast.agents._shared.call_limits import (
    DEFAULT_MODEL_CALLS_PER_RUN,
    make_call_limit_middleware,
)
from mast.agents._shared.stall_guard_mw import StallGuardMiddleware
from mast.agents._shared.models import AGENT_MODEL, make_chat_model

from mast.prompts.builds import record_build
from mast.prompts.registry import resolve as resolve_prompt
from .prompts import SYSTEM_PROMPT
from .tools import build_tools, tool_availability, unavailable_tools_note

if False:  # TYPE_CHECKING — avoid runtime import of BufferService
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)


def build(
    buf: "BufferService | None",
    *,
    model: Any | None = None,
    checkpointer: Any = None,
    max_model_calls: int | None = 40,
    max_tool_calls: int | None = 120,
    # 出厂值从 call_limits 派生 —— **别在这里再写一个字面量**。
    # 2026-08-10 之前这六张图各自硬编码 30,和 call_limits 的常量是六份副本;
    # 把常量从 30 提到 500 时,这六处会静静地留在 30。
    max_model_calls_per_run: int = DEFAULT_MODEL_CALLS_PER_RUN,
    max_tool_calls_per_run: int = 80,
    extra_tools: list | None = None,
    turn_recorder=None,
    standalone: bool = False,
    extra_middleware: list | None = None,
):
    """Build the Literature Reading agent.

    Args:
        buf:             BufferService for live tip/scan context reads.
                         None disables buffer tools (useful in offline tests).
        model:           A langchain chat model OR a fake chat model for tests.
                         Default: make_chat_model("literature") →
                         AGENT_MODEL["literature"] (Kimi K3; served as a
                         ChatOpenAI subclass, not ChatAnthropic). Pass an
                         explicit ChatAnthropic to run on Claude (only then
                         is prompt caching attached).
        checkpointer:    LangGraph checkpointer (SqliteSaver / InMemorySaver).
        max_model_calls: ModelCallLimitMiddleware cap (R7 loop guard).
        max_tool_calls:  ToolCallLimitMiddleware cap (R7 loop guard).
        extra_tools:     extra non-hardware tools to merge in (e.g. the shared
                         persistent-memory tools from agents._shared). These are
                         passed in by the orchestrator/GUI so this agent never
                         imports a sibling agent. None = none.

    Returns:
        CompiledStateGraph (LangGraph agent ready to invoke).
    """
    if model is None:
        model = make_chat_model(
            "literature",
            max_tokens=4096,
            temperature=0.2,  # literature reading — low temp, focused reasoning
        )

    # Dead-tool detection: probe dependency-gated tools ONCE, swap
    # the dead ones for same-named "unavailable" placeholders, and inject a
    # one-time constraints note so the agent is told up front rather than burning
    # a turn per call rediscovering that web_search / search_papers are down.
    unavailable = tool_availability()
    tools = build_tools(buf, unavailable=unavailable) + list(extra_tools or [])
    if standalone:
        tools = [t for t in tools if not str(getattr(t, "name", "")).startswith("handoff_to_")]

    _base_prompt = resolve_prompt("agent.literature.system", SYSTEM_PROMPT)
    _unavail_note = unavailable_tools_note(unavailable)
    system_prompt = _base_prompt + _unavail_note

    middleware = [
        *(extra_middleware or []),
        # Tell this agent what the agents before it produced. Without it
        # the only thing crossing a handoff is one free-text sentence.
        UpstreamArtifactMiddleware("literature"),
        # StallGuard — end a same-tool/same-error retry spin.
        # It used to be wired into instrument_control ONLY, so the other five
        # agents could spin on an identical failure until the recursion cap killed
        # the run with "任务步数达到上限" and nothing to show for it. A spin is not
        # an instrument problem; any agent can have one.
        StallGuardMiddleware(agent_name="literature"),
        *make_call_limit_middleware(
            max_model_calls=max_model_calls, max_tool_calls=max_tool_calls,
            max_model_calls_per_run=max_model_calls_per_run,
            max_tool_calls_per_run=max_tool_calls_per_run),
    ]

    # AnthropicPromptCachingMiddleware only for the real ChatAnthropic model;
    # GenericFakeChatModel in tests must NOT include it (unsupported).
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
            logger.debug(
                "AnthropicPromptCachingMiddleware not available; skipping cache middleware."
            )

    # F5 residual (2026-06-08): never leave the model with an assistant-last
    # message — no-prefill Claude models (sonnet-4-6) 400 when the orchestrator
    # hands off with a trailing "[SUPERVISOR → …]" AIMessage. No-op otherwise.
    from mast.agents._shared.prefill_guard_mw import ClaudePrefillGuardMiddleware
    from mast.agents._shared.tool_pair_guard_mw import ToolPairGuardMiddleware
    from mast.agents._shared.vision_mw import SkillImageMiddleware
    # ToolPairGuard (outer of prefill so prefill keeps the final ends-on-user say):
    # strips the unmatched handoff tool messages the group orchestrator leaves in the
    # parent channel so a provider never 400s "tool_call_id is not found" — see
    # tool_pair_guard_mw module doc.
    middleware.append(ToolPairGuardMiddleware())
    # 图像通道的出站那一半:把技能挂在 ToolMessage 上的图像**路径**读成 data URI
    # 挂进本次请求。模型不支持视觉 / 没有图 / 图读不出来 → 原样返回,退化成纯文本
    # (理由与形状见 vision_mw 模块文档)。装在 prefill guard 之前,让 prefill
    # 仍然拥有 ends-on-user 的最终发言权。
    middleware.append(SkillImageMiddleware())
    middleware.append(ClaudePrefillGuardMiddleware())

    # Training-log: capture each agent turn (reasoning + tool_calls + usage) when
    # the GUI injects a recorder. No-op (None) keeps the agent untouched.
    if turn_recorder is not None:
        from mast.agents._shared.recorder_mw import RecorderMiddleware
        middleware.append(RecorderMiddleware(agent_id="literature", recorder=turn_recorder))

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        name="literature",
        checkpointer=checkpointer,
        # The artifact channel lives in state. WITHOUT a schema the
        # subgraph compiles with LangChain's bare AgentState, every
        # channel below "messages" is undeclared, and a tool's
        # Command(update={...}) for it is dropped with NO error --
        # which is why the skill adapter's executed_skills /
        # composite_progress writes never actually landed either
        # (verified empirically 2026-07-29).
        state_schema=AgentSubState,
    )

    record_build("literature", middleware=middleware, tools=tools,
                 system_blocks=[("agent.literature.system", _base_prompt),
                                ("agent.literature.unavailable_note", _unavail_note)],
                 standalone=standalone)

    logger.info(
        "literature agent built: %d tools, %d middlewares",
        len(tools),
        len(middleware),
    )
    return agent


__all__ = ["build"]
