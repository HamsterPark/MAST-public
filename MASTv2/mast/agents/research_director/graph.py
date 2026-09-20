"""科研策划 agent（research_director，RD）—— 决策链的上半环。

全自动实验室的下半环（方案 → 执行 → 记录）在 2026-08 已经通了；这个 agent 补的是
上半环：**科学目标自己生成、自己迭代**。它消费文献报告与既往实验记录，产出/修订
一份 research campaign（假设 / 目标 / 谱系），再把一份委托交给 experiment_design。

## 建图形状：照 literature 的先例，不照 instrument_control 的

    build(buf, *, model=None, checkpointer=None,
          max_model_calls=40, max_tool_calls=120) -> CompiledStateGraph

RD **不装 SafetyGateMiddleware，也不装 HumanInTheLoopMiddleware** —— 与 literature
同一个理由：它是一个只读记录/文档、写自己那张表的 agent，从不驱动仪器。安全闸门装
在这里不会更安全，只会让「哪些 agent 会碰硬件」这件事变得看不清楚。

``AnthropicPromptCachingMiddleware`` 只在 ``model`` 真的是 ``ChatAnthropic`` 时挂上：
默认模型走 OpenAI 兼容端点，那边不认 Anthropic 的 cache-control 头。

## 为什么它可以后台跑

``core/background_runs.BACKGROUNDABLE`` 的判据是「**会不会在没人看着的时候让模型
决定动仪器**」，不是「后台线程不许碰硬件」。RD 的工具面里没有任何执行面，所以它进
得来 —— 走的是判据，不是例外。判据原文在那张表上面。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from mast.agents._shared.call_limits import (
    DEFAULT_MODEL_CALLS_PER_RUN,
    make_call_limit_middleware,
)
from mast.agents._shared.models import make_chat_model
from mast.agents._shared.stall_guard_mw import StallGuardMiddleware
from mast.agents._shared.upstream_mw import UpstreamArtifactMiddleware
from mast.agents.state import AgentSubState
from mast.prompts.builds import record_build
from mast.prompts.registry import resolve as resolve_prompt

from .prompts import SYSTEM_PROMPT
from .tools import build_tools

if False:  # TYPE_CHECKING — 避免运行时 import BufferService
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)


def build(
    buf: "BufferService | None" = None,
    *,
    model: Any | None = None,
    checkpointer: Any = None,
    max_model_calls: int | None = 40,
    max_tool_calls: int | None = 120,
    # 出厂值从 call_limits 派生 —— **别在这里再写一个字面量**（2026-08-10：
    # 六张图各自硬编码 30，把常量提到 500 时那六处会静静地留在 30）。
    max_model_calls_per_run: int = DEFAULT_MODEL_CALLS_PER_RUN,
    max_tool_calls_per_run: int = 80,
    extra_tools: list | None = None,
    turn_recorder=None,
    standalone: bool = False,
    extra_middleware: list | None = None,
):
    """建科研策划 agent。

    Args:
        buf:          BufferService。**这个 agent 不用它** —— 只为与其它 agent 的
                      build() 签名一致（编排器一视同仁地传下来）。针尖状态与扫描
                      进度不是 campaign 层的输入。
        model:        langchain chat model，或测试里的 fake。默认
                      ``make_chat_model("research_director")``。
        checkpointer: LangGraph checkpointer。
        max_model_calls / max_tool_calls: R7 循环闸。
        extra_tools:  编排器/GUI 注入的共享工具（memory + documents + ask_user +
                      conduct 一族）。**这是它不 import 兄弟 agent 的原因** ——
                      共享的东西从 ``agents._shared.*`` 来，由外面发下来。
        standalone:   True 时剥掉 handoff 工具（私聊里没有可交接的对象）。

    Returns:
        CompiledStateGraph。
    """
    if model is None:
        model = make_chat_model(
            "research_director",
            max_tokens=4096,
            # 科研策划是判断题不是创作题：低温度，但比文献略高一点，
            # 因为「这条线索值不值得追」本来就需要一点发散。
            temperature=0.3,
        )

    tools = build_tools(buf) + list(extra_tools or [])
    if standalone:
        tools = [t for t in tools if not str(getattr(t, "name", "")).startswith("handoff_to_")]

    system_prompt = resolve_prompt("agent.research_director.system", SYSTEM_PROMPT)

    middleware = [
        *(extra_middleware or []),
        # 上游产物：文献报告 / 既往方案 / 分析结果 / 它自己上一轮的纲领。
        # 没有这一层，交接过来的只有一句自由文本。
        UpstreamArtifactMiddleware("research_director"),
        # 同一个工具、同一个错误反复重试时收尾（）。任何 agent 都会
        # 打转，不只是仪器那一个。
        StallGuardMiddleware(agent_name="research_director"),
        *make_call_limit_middleware(
            max_model_calls=max_model_calls, max_tool_calls=max_tool_calls,
            max_model_calls_per_run=max_model_calls_per_run,
            max_tool_calls_per_run=max_tool_calls_per_run),
    ]

    # 只有真 ChatAnthropic 才挂缓存中间件；测试里的 GenericFakeChatModel 不支持。
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
            logger.debug("AnthropicPromptCachingMiddleware 不可用，跳过缓存中间件。")

    from mast.agents._shared.prefill_guard_mw import ClaudePrefillGuardMiddleware
    from mast.agents._shared.tool_pair_guard_mw import ToolPairGuardMiddleware
    # ToolPairGuard 装在 prefill 外层（让 prefill 保留 ends-on-user 的最终发言权）：
    # 群聊编排器会在 parent 通道里留下配不上对的 handoff tool message，不清掉的话
    # provider 会 400 "tool_call_id is not found"（）。
    middleware.append(ToolPairGuardMiddleware())
    # 没有 SkillImageMiddleware：这个 agent 不跑技能，也就不会有技能挂图像路径。
    middleware.append(ClaudePrefillGuardMiddleware())

    if turn_recorder is not None:
        from mast.agents._shared.recorder_mw import RecorderMiddleware
        middleware.append(
            RecorderMiddleware(agent_id="research_director", recorder=turn_recorder))

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        name="research_director",
        checkpointer=checkpointer,
        # 产物通道住在 state 里。**没有 state_schema 时，子图会用 LangChain 的裸
        # AgentState 编译，messages 以下的每个通道都没有声明，工具的
        # Command(update={...}) 会被静默丢弃**（2026-07-29 实测）。
        # campaign_request_plan 写的就是这样一个通道。
        state_schema=AgentSubState,
    )

    record_build("research_director", middleware=middleware, tools=tools,
                 system_blocks=[("agent.research_director.system", system_prompt)],
                 standalone=standalone)

    logger.info(
        "research_director agent built: %d tools, %d middlewares",
        len(tools), len(middleware),
    )
    return agent


__all__ = ["build"]
