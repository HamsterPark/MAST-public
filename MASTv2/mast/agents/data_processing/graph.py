"""Data-Processing agent — Phase 5 real implementation.

Replaces the Phase-3 stub. Uses langchain.agents.create_agent + middleware
stack per compass §4.6. The build() signature mirrors experiment_design:

    build(buf, *, model=None, checkpointer=None,
          max_model_calls=40, max_tool_calls=120) -> CompiledStateGraph

`model` defaults to ChatAnthropic(model=AGENT_MODEL["data_processing"])
(Sonnet 4.6); pass a GenericFakeChatModel in tests.

Unlike IC, DP does NOT include SafetyGateMiddleware or HumanInTheLoopMiddleware
— it is a read-only post-processing agent that never commands the instrument.
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
from .tools import build_tools

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
    """Build the Data-Processing agent.

    Args:
        buf:             BufferService for live tip/scan context reads.
                         None disables buffer tools (useful in offline tests).
        model:           ChatAnthropic instance OR a fake chat model for tests.
                         Default: ChatAnthropic via AGENT_MODEL["data_processing"].
        checkpointer:    LangGraph checkpointer (SqliteSaver / InMemorySaver).
        max_model_calls: ModelCallLimitMiddleware cap (R7 loop guard).
        max_tool_calls:  ToolCallLimitMiddleware cap (R7 loop guard).
        extra_tools:     extra non-hardware tools to merge in (e.g. the shared
                         persistent-memory tools from agents._shared). Passed in
                         by the orchestrator/GUI so this agent never imports a
                         sibling agent. None = none.

    Returns:
        CompiledStateGraph (LangGraph agent ready to invoke).
    """
    if model is None:
        model = make_chat_model(
            "data_processing",
            # 8192（2026-08-19，原 4096）：py_run 让这个 agent 开始产出**代码**，
            # 而代码和回答的其余部分要一起装进这个上限。一段 150 行的分析脚本约
            # 1500–2500 token，JSON 转义换行和引号之后更多；截断发生在 tool call
            # 的 arguments 中间时产生的是非法 JSON —— 表现为 provider 报错或解析
            # 失败，一个困惑的失败而不是干净的失败。4096 是这个 agent 只输出数字
            # 摘要时定的。
            max_tokens=8192,
            temperature=0.1,  # analytical — low temperature, no creative variance
        )

    tools = build_tools(buf) + list(extra_tools or [])
    if standalone:
        tools = [t for t in tools if not str(getattr(t, "name", "")).startswith("handoff_to_")]

    middleware = [
        *(extra_middleware or []),
        # Tell this agent what the agents before it produced. Without it
        # the only thing crossing a handoff is one free-text sentence.
        UpstreamArtifactMiddleware("data_processing"),
        # StallGuard — end a same-tool/same-error retry spin.
        # It used to be wired into instrument_control ONLY, so the other five
        # agents could spin on an identical failure until the recursion cap killed
        # the run with "任务步数达到上限" and nothing to show for it. A spin is not
        # an instrument problem; any agent can have one.
        StallGuardMiddleware(agent_name="data_processing"),
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

    # F5 residual (2026-06-08): guard no-prefill Claude models — see prefill_guard_mw.
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
        middleware.append(RecorderMiddleware(agent_id="data_processing", recorder=turn_recorder))

    _system_prompt = resolve_prompt("agent.data_processing.system", SYSTEM_PROMPT)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=_system_prompt,
        middleware=middleware,
        name="data_processing",
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

    record_build("data_processing", middleware=middleware, tools=tools,
                 system_blocks=[("agent.data_processing.system", _system_prompt)],
                 standalone=standalone)
    logger.info(
        "data_processing agent built: %d tools, %d middlewares",
        len(tools),
        len(middleware),
    )
    return agent


__all__ = ["build"]
