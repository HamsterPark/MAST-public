"""后台运行的 v2 装配（编排器第一站）。

为什么编排器首秀选后台
----------------------
它是唯一一个**天然隔离**的编排面：

* 状态隔离 —— 后台 run 有自己的 thread_id 与自己的 saver，前台看不见它；
* **零硬件** —— ``BACKGROUNDABLE`` 永不包含 ``instrument_control``（v2.md 不变式 10：
  驱动仪器的每一步都要能指出「谁点的头」）；
* **无 HITL** —— 后台流本来就跳过 interrupt，所以编排级的暂停可以推迟到群聊那一步；
* 失败模式温和 —— 一次后台分析失败，前台无感。

也就是说：编排器的路由、并发原语、护栏、产物直达这四件事，可以在一个碰不到硬件、
碰不到用户、碰不到前台状态的地方先跑熟。

与 ``BackgroundRunManager`` 的关系
----------------------------------
一行都不改它。那 560 行（独立 thread_id + 独立 saver + daemon 线程 + 结果流回同一
transcript）零 langgraph 依赖，而且它的 ``run_fn`` 本来就是注入的——这个模块提供的
就是另一个 ``run_fn``。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Sequence

from mast.agentruntime.context import RunContext
from mast.agentruntime.events import BranchDoneEvent, MessageEvent, RunEvent
from mast.agentruntime.loop import AgentLoop
from mast.agentruntime.orchestrator import OrchestratorLoop, TaskResult
from mast.agentruntime.state import BranchInput, BranchOutcome, TaskState

logger = logging.getLogger(__name__)


def make_branch_runner(
    *, build_loop: Callable[[str], AgentLoop],
    on_event: Callable[[RunEvent], None] | None = None,
) -> Callable[[BranchInput, RunContext], BranchOutcome]:
    """把「装配一个 AgentLoop」变成编排器要的 ``run_branch``。

    ``build_loop`` 注入而不是内建：后台、群聊、私聊三条路装配 agent 的方式不同
    （中间件栈、预算、是否给交棒工具），而编排逻辑对此**不该有意见**。
    """
    def _run(payload: BranchInput, ctx: RunContext) -> BranchOutcome:
        loop = build_loop(payload.agent_id)
        gen = loop.run(list(payload.messages), ctx)
        while True:
            try:
                event = next(gen)
            except StopIteration as stop:
                result = stop.value
                break
            if on_event is not None:
                try:
                    on_event(event)
                except Exception as exc:  # noqa: BLE001 — sink 坏了不该毁掉分支
                    logger.debug("background event sink failed: %s", exc)

        return BranchOutcome(
            agent_id=payload.agent_id,
            outcome=result.outcome,
            final_text=result.final_text,
            stop_reason=result.stop_reason,
            handoff_target=(result.handoff.target if result.handoff else ""),
            handoff_reason=(result.handoff.reason if result.handoff else ""),
            state_delta=dict(result.state_delta),
            new_messages=list(result.new_messages),
            model_calls=result.counters.model_calls,
            tool_calls=result.counters.tool_calls,
        )
    return _run


def run_background_task(
    *, instruction: str, agents: Sequence[str], emit: Callable[..., None],
    abort: Any, router_model: Any, seed_artifacts: dict | None = None,
    seed_notice: str = "", run_id: str = "",
    build_loop: Callable[[str], AgentLoop] | None = None,
    max_model_calls: int = 12, max_tool_calls: int = 40,
) -> str:
    """跑一次 v2 后台编排，返回最终文本。

    ``emit(agent_id, role, text)`` 是 ``BackgroundRunManager`` 的既有协议——**原样
    沿用**，所以结果照旧流回同一个 transcript，前端不知道换了引擎。
    """
    from mast.agentruntime.assembly import build_agent_loop
    from mast.agentruntime.routing import make_llm_router

    targets = [a for a in agents if a]
    if not targets:
        raise RuntimeError("背景运行没有可派发的 agent")

    def _default_build(agent_id: str) -> AgentLoop:
        return build_agent_loop(agent_id, buf=None,
                                max_model_calls=max_model_calls,
                                max_tool_calls=max_tool_calls)

    # 三个接收点，各管一段，**不重叠** —— 事件只有一条通道（生成器的 yield），
    # 但这条通道有三层：agent 循环的、编排器的、以及编排器收尾时经 ctx 发的那一条。
    # 谁处理什么必须写死，否则同一条 BranchDoneEvent 会被记两遍。

    def _on_branch_event(event: RunEvent) -> None:
        """**只**管 agent 自己的正文。分支结局归下面的主循环管。"""
        if isinstance(event, MessageEvent) and event.text:
            emit(event.agent or "_supervisor", "assistant", event.text)

    def _on_orchestrator_event(event: RunEvent) -> None:
        """**只**管分支结局。编排器为每条分支重发一次 BranchDoneEvent —— 那是它的
        事件流对消费者的完整承诺，也是这里唯一该读的地方。"""
        if isinstance(event, BranchDoneEvent) and event.outcome not in (
                "handoff", "final"):
            # ★ 非正常结局必须出现在 transcript 里。旧世界这类分支什么都不说，
            #   于是一次「限流停住」和一次「干完了」在记录上分不开。
            emit(event.agent or "_supervisor", "assistant",
                 f"（{event.agent} 提前结束：{event.stop_reason or event.outcome}）")

    state = TaskState(run_id=run_id, artifacts=dict(seed_artifacts or {}))
    if seed_notice:
        state.messages.append({"role": "system", "content": seed_notice})

    loop = OrchestratorLoop(
        route=make_llm_router(router_model, targets),
        run_branch=make_branch_runner(build_loop=build_loop or _default_build,
                                      on_event=_on_branch_event),
        # 后台名单里本来就没有 IC，但闸门照挂：**调度器是同一个**，
        # 把约束写在这里而不是「后台反正没有 IC 所以不用管」。
        instrument_agents=("instrument_control",),
    )

    # ctx.emit 留空：编排器只用它发收尾的 ``DoneEvent``，而这个函数自己就返回
    # 最终文本 —— 再往 transcript 记一遍是重复。
    #
    # ★ ``ask_human`` 也留空，而且是**刻意的**（2026-08-27 写下来）。
    #
    # 后台 run 没有用户在看：它的定义就是「前台不受影响地跑完」，可后台化名单
    # 永不含 instrument_control（零硬件），编排级 HITL 按计划推迟到群聊那一面。
    # 一个没人会去回答的提问，挂在审批面板上 900 秒然后 fail-closed，比当场告诉
    # agent「这条入口没有提问通道、请按你的保守默认继续」更糟。
    #
    # CLI（``pipeline/main.py`` 的 v2 那一半）走的也是这个函数，同理：一次性命令
    # 行调用没有可以弹卡片的界面。
    #
    # **写下来是因为「就是没有」和「漏了」在代码里长得一模一样** —— 而这次迁移刚
    # 因为一个「挂点在、没人实现」栽过一次（``ask_user`` 在 v2 上问不出去，且这一轮
    # 以 final 正常收场）。要接的话：``from .pause import attach``，一行。
    ctx = RunContext(run_id=run_id, abort=abort)
    gen = loop.run_task(state, instruction, ctx)
    result: TaskResult | None = None
    while True:
        try:
            event = next(gen)
        except StopIteration as stop:
            result = stop.value
            break
        _on_orchestrator_event(event)
        if abort.is_set():
            gen.close()
            break

    if result is None:
        return ""
    if result.outcome not in ("completed",) and result.stop_reason:
        emit("_supervisor", "assistant", f"（后台运行结束：{result.stop_reason}）")
    return result.final_text or ""


__all__ = ["run_background_task", "make_branch_runner"]
