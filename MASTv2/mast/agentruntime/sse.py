"""群聊 SSE：把 :class:`OrchestratorLoop` 的事件流翻译成前端在消费的帧。

这是把编排器迁出 LangGraph 的改动里最大的一块。

住在这里而不是 ``api/routes/orchestrator.py`` 的两个理由
--------------------------------------------------------
1. 改动面越小越好，尽量不动 ``api/routes/orchestrator.py``；
2. 更重要的：这段翻译**本来就该是运行时的一部分**。今天桥里那 900 行帧拼装之所以那么
   长，是因为它在**推断**——agent 名字靠切 namespace 字符串、去重靠攒 ``msg.id``、
   「分支死没死」靠闩住 ``active_agent == "__end__"`` 再解析 dispatch note。事件流自己
   说得清的东西，不需要在消费端重新猜一遍。

省掉的三样（不是简化，是那三件事不存在了）
------------------------------------------
* ``_agent_from_namespace`` —— 事件自带 ``agent``；
* ``seen`` 消息去重集合 —— 事件只发一次（不存在跨 subgraph 边界的 re-emit）；
* ``saw_end`` 闩锁 + dispatch-note 解析 —— :class:`BranchDoneEvent` 直接带 outcome。
  **这一条是本次迁移最想要的东西**：一条死掉的分支和一次真正的完成，在旧桥里可观测
  输出完全相同（见 ``tests/v2/agents/contract/test_silent_branch_death.py``）。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Generator, Iterable, Sequence

from mast.agentruntime.context import RunContext
from mast.agentruntime.events import (
    BranchDoneEvent,
    DoneEvent,
    HandoffEvent,
    MessageEvent,
    RunEvent,
    StatusEvent,
    StepEvent,
    ToolStartEvent,
)
from mast.agentruntime.orchestrator import HOP_HARD_CAP, OrchestratorLoop, TaskResult
from mast.agentruntime.state import TaskState

logger = logging.getLogger(__name__)

#: 一条帧里正文的上限。与旧桥的 ``_SSE_TEXT_MAX`` 同值 —— 帧只要渲染得出来，
#: 完整正文在转录行里（那边的上限是另一个数，刻意的）。
SSE_TEXT_MAX = 8_000


def _clip(text: str, limit: int = SSE_TEXT_MAX) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    # 把截断写进字符串本身 —— 一段被裁过的正文，读的人有权知道它被裁过。
    return text[:limit] + f"\n…（已截断，完整 {len(text)} 字见转录）"


class PendingFrames:
    """已经生成、但还没轮到被 yield 出去的帧 —— **可以被别的线程排空**。

    为什么需要它（2026-08-27）
    --------------------------
    ``drive_group_run`` 是一个同步生成器。它阻塞在 ``next(gen)`` 里的时候，**它
    yield 的任何东西都出不去** —— 而「原地阻塞等人」正是这次迁移用来替掉 interrupt
    重放的东西。两个实测症状同源：

    * 工具帧在工具**跑完之后**才出去（0.611 s 的工具，帧在 0.612 s）——
      循环层已经改成「执行前先 yield」，但群聊多一层批处理；
    * HITL 等待期间「已等 N 秒」的 beat 帧**一条都出不去**。

    连接本身不静默（``api/sse.with_heartbeat`` 在自己的线程上发 ``: ping``），所以这
    不是那种表面「看起来死了」的假死，而是**活着但不说话**。

    修法不是让生成器少阻塞（那等于退回重放），而是**换一条不经过它的出口**：帧在
    产生它的那条线程上就地生成并存进这里，路由层按时钟把它排空。生成器自己在步与
    步之间也排空同一个容器 —— **同一把锁、同一个队列**，所以一条帧不会发两次，也
    不会因为「谁先到」而丢。

    为什么转换与落库放在**产生帧的线程**上
    --------------------------------------
    原来 ``inbox`` 攒的是**事件**，转成帧（含 ``persist``）在主线程做。那样的话
    「帧出得去」就还是要等主线程解除阻塞 —— 什么也没解决。所以转换提前到入队时。

    这样做安全，是**核实过的**而不是假定的：``ConversationStore.append_message``
    的 docstring 明写它为并发写设计（``MAX(seq)+1`` 与 INSERT 在**单条语句**里、
    走 sqlite 写锁，``busy_timeout`` 串行化竞争者），连接也是 ``check_same_thread=
    False``。顺带的好处：转录行落库的时刻从「工具结束后」变成「工具开始时」，与帧
    一致 —— 转录里的时间戳因此不再比事实晚一整次工具调用。
    """

    __slots__ = ("_items", "_lock")

    def __init__(self) -> None:
        self._items: list[dict] = []
        self._lock = threading.Lock()

    def push(self, frame: dict | None) -> None:
        if not frame:
            return
        with self._lock:
            self._items.append(frame)

    def drain(self) -> list[dict]:
        """取走目前攒着的全部帧（取走即清空）。空了返回空列表，不抛。"""
        with self._lock:
            if not self._items:
                return []
            out, self._items = self._items, []
            return out

    def __len__(self) -> int:      # 便于测试与日志说「还压着几条」
        with self._lock:
            return len(self._items)


def drive_group_run(
    *, instruction: str, agents: Sequence[str], router_model: Any,
    state: TaskState | None = None,
    build_loop: Callable[[str], Any] | None = None,
    persist: Callable[..., None] | None = None,
    abort: Any = None,
    hold: Callable[[str], Iterable[dict]] | None = None,
    progress_limit: int = HOP_HARD_CAP,
    max_model_calls: int = 25, max_tool_calls: int = 60,
    hitl_store: Any = None, thread_id: str = "",
    pending: "PendingFrames | None" = None,
) -> Generator[dict, None, TaskResult]:
    """驱动一次群聊任务，逐个 yield **帧 dict**（调用方负责 SSE 序列化）。

    ``hold(agent_id)`` 是用户暂停的钩子：它 yield 出来的帧照发，返回即继续。
    惰性生成器天然支持这件事——不拉下一个事件，这一跳就停在那里。

    ``persist(kind, agent=…, role=…, text=…, meta=…)`` 与旧桥同签名，所以转录落库
    那一侧一行都不用改。

    ``hitl_store`` 是 ``CoreRuntime._orch_interrupts``。给了它，agent 才问得了用户
    （``ask_user`` / DANGEROUS 审批）；**不给就是「这条入口没有提问通道」** —— 那是
    今天的行为，也是实测出来最该修的一条：不接的话 ``ask_user`` 会把「问不出去」转成
    一句提示，这一轮以 ``final`` 正常收场，而**用户永远看不到那个问题**。

    ``pending`` 是**阻塞期间帧的第二条出口**（见 :class:`PendingFrames`）。
    -------------------------------------------------------------------
    不给也能跑：这个生成器自己在步与步之间排空它，行为与从前一致。给了的话，路由层
    可以在**这个生成器还阻塞着**的时候按时钟把它排空 —— 那是工具帧与 HITL「已等 N
    秒」帧唯一的出路，因为一个阻塞在 ``next()`` 里的同步生成器 yield 不出东西。

    ⚠️ 这里曾写着一条降级（2026-08-27 当天修掉）
    -------------------------------------------
    原文是：「等待期间的心跳帧是补发的，不是实时的……正确的修法在路由层，**翻开群
    聊开关之前必须做**」。那条修法就是 ``pending`` + ``api/sse.with_heartbeat`` 的
    ``on_idle``，现已落地，所以这段从「已知降级」变成「已修，附机理」。

    留着这段字是因为**机理本身没变**：同步生成器阻塞时 yield 不出东西，这是「不重放」
    换来的硬约束。变的只是帧不再**只有**生成器这一条出口。谁将来在这条链上加新的帧，
    要问的仍是那个问题：**它会不会正好在一次阻塞中间产生**。
    """
    from mast.agentruntime.assembly import build_agent_loop
    from mast.agentruntime.background import make_branch_runner
    from mast.agentruntime.routing import make_llm_router

    targets = [a for a in agents if a]
    if not targets:
        raise RuntimeError("群聊没有可派发的 agent")

    # ★ 早拒绝，而不是跑到一半炸。
    #
    # 默认的 ``build_agent_loop`` 拒绝 ``instrument_control``（它要一整套硬件安全
    # 中间件，而那些需要运行时注入的 ``get_state`` / ``get_mode`` / ``registry`` /
    # ``buf``）。如果放它进目标名单而调用方没给能装配它的 ``build_loop``，失败会发生在
    # **第一次路由到它的时候** —— 那时用户已经看着这次运行跑了几跳、前面的工作可能
    # 已经改过磁盘状态，而报错读起来像「某个 agent 挂了」而不是「这条路本来就不该走」。
    #
    # 所以这里在开跑前检查一次：**调用方给了能装配仪器 agent 的工厂了吗**。
    # 给了就放行（工厂自己会因为少安全件而抛），没给就现在说清楚。
    from mast.chat.engine_v2 import HARDWARE_AGENTS

    blocked = [a for a in targets if a in HARDWARE_AGENTS]
    if blocked and build_loop is None:
        raise RuntimeError(
            f"v2 群聊要派发 {'、'.join(blocked)}，但调用方没有提供能装配它们的 "
            "``build_loop``。它们驱动仪器，必须带完整的硬件安全中间件"
            "（SafetyGate / AlertDelivery / ModeBelief，有 buffer 时还有 BufferHITL），"
            "而那些需要运行时注入的 get_state / get_mode / registry / buf。"
            "请传 build_loop，或用旧引擎跑这次任务。")

    st = state or TaskState()
    counters = {"hops": 0, "model_calls": 0, "tool_calls": 0}

    def _default_build(agent_id: str):
        return build_agent_loop(agent_id, buf=None,
                                max_model_calls=max_model_calls,
                                max_tool_calls=max_tool_calls)

    # 分支内部的事件（agent 正文、工具起止）与 HITL 心跳都存进**同一个**容器：
    # 分支跑在线程池里、心跳产生在主线程的阻塞中间，两者都不能直接 yield。
    #
    # 转成帧（含落库）就在产生它的那条线程上做，不留到主线程 —— 留到主线程的话，
    # 「帧出得去」还是要等主线程解除阻塞，那等于什么也没解决。见 PendingFrames。
    box = pending if pending is not None else PendingFrames()
    loop = OrchestratorLoop(
        route=make_llm_router(router_model, targets),
        run_branch=make_branch_runner(
            build_loop=build_loop or _default_build,
            on_event=lambda ev: box.push(_branch_frame(ev, persist))),
        instrument_agents=("instrument_control",),
    )

    ctx = RunContext(abort=abort, run_id=st.run_id)
    if hitl_store:
        from mast.agentruntime.pause import attach

        attach(ctx, hitl_store, owner="_supervisor",
               thread_id=thread_id or st.run_id, on_beat=box.push)
    gen = loop.run_task(st, instruction, ctx)
    result: TaskResult | None = None

    while True:
        # 先把攒下的发完，再拉编排器的下一个 —— 顺序因此与发生顺序一致。
        # 路由层可能已经按时钟取走了一部分（那正是它存在的意义），取走即清空，
        # 所以同一条帧不会在这里再发一遍。
        for frame in box.drain():
            yield frame

        try:
            event = next(gen)
        except StopIteration as stop:
            result = stop.value
            break

        if abort is not None and abort.is_set():
            gen.close()
            break

        for frame in _orchestrator_frames(event, counters, progress_limit, persist):
            yield frame

        # 用户暂停：在 agent 边界停住（不拉下一个事件 = 编排器暂停）。
        if hold is not None and isinstance(event, HandoffEvent):
            for frame in (hold(event.to_agent) or ()):
                yield frame

    for frame in box.drain():          # 收尾时把剩下的发完
        yield frame

    return result or TaskResult(outcome="aborted", stop_reason="中止")


def _branch_frame(event: RunEvent, persist) -> dict | None:
    """将 agent 分支内部事件转换为可发送的帧。

    工具起止应及时产生可见事件，复用 narrate_tool_call 和 summarize_args，
    保持群聊与私聊的措辞及参数展示一致。分支线程就地调用本函数，帧暂存于
    PendingFrames，由路由层定时排空，避免同步生成器阻塞期间事件滞留。
    with_heartbeat 的 ping 只维持连接，不能替代工具进度帧。
    """
    if isinstance(event, MessageEvent) and event.text:
        text = _clip(event.text)
        _try_persist(persist, "message", agent=event.agent, role="agent", text=text)
        return {"kind": "message", "agent": event.agent, "role": "agent",
                "text": text, "t": event.t or time.time()}

    if isinstance(event, ToolStartEvent):
        return _tool_frame(event, persist)
    return None


def _tool_frame(event: "ToolStartEvent", persist) -> dict | None:
    """一次工具调用 → 与 v1 **同形**的 ``role="tool"`` 帧。

    ``args`` 从事件的 ``preview`` 里取不回结构（那是给日志看的截断串），所以带上
    原始 args：``ToolStartEvent`` 里有 ``args``（没有的话退回 preview 文本）。
    """
    name = getattr(event, "name", "") or "tool"
    args = getattr(event, "args", None)
    if not isinstance(args, dict):
        args = {}
    try:
        from mast.api.tool_narration import narrate_tool_call, summarize_args

        summary = narrate_tool_call(name, args)
        args_json, clipped = summarize_args(args)
    except Exception as exc:  # noqa: BLE001 — 措辞层坏了不该毁掉一条 run
        logger.debug("tool narration unavailable: %s", exc)
        summary, args_json, clipped = f"调用 {name}", "", False

    _try_persist(persist, "message", agent=event.agent, role="tool",
                 text=summary, meta={"tool": name, "args": args_json,
                                     "args_clipped": clipped})
    return {"kind": "message", "agent": event.agent, "role": "tool",
            "text": summary, "tool": name, "args": args_json,
            "args_clipped": clipped, "t": event.t or time.time()}


def _orchestrator_frames(event: RunEvent, counters: dict, limit: int,
                         persist) -> list[dict]:
    """编排器自己的事件 → 帧。"""
    now = event.t or time.time()

    if isinstance(event, HandoffEvent):
        text = f"[{event.from_agent} → {event.to_agent}] {event.reason}".strip()
        _try_persist(persist, "handoff", agent=event.to_agent, text=text)
        return [{"kind": "handoff", "from_agent": event.from_agent,
                 "to_agent": event.to_agent, "reason": event.reason, "t": now}]

    if isinstance(event, StatusEvent):
        _try_persist(persist, "status", agent=event.agent, text=event.text)
        return [{"kind": "status", "agent": event.agent, "text": event.text,
                 "t": now}]

    if isinstance(event, StepEvent):
        counters["hops"] = event.hops or counters["hops"]
        counters["model_calls"] += event.model_calls
        counters["tool_calls"] += event.tool_calls
        return [{"kind": "status", "subkind": "progress", "text": "",
                 "step": counters["hops"], "step_limit": limit,
                 "progress": counters["hops"], "progress_limit": limit,
                 "progress_unit": "hop", "t": now}]

    if isinstance(event, BranchDoneEvent):
        # ★ 本次迁移最想要的那个帧。旧桥里没有对应物：不交棒的分支什么都不说，
        #   于是「死分支」与「任务完成」可观测输出完全相同。
        if event.outcome in ("handoff", "final"):
            return []                  # 正常结束不必打扰用户
        text = f"（{event.agent} 提前结束：{event.stop_reason or event.outcome}）"
        _try_persist(persist, "status", agent=event.agent, text=text)
        return [{"kind": "status", "agent": event.agent, "text": text,
                 "subkind": "branch_stopped", "outcome": event.outcome, "t": now}]

    if isinstance(event, DoneEvent):
        return []                      # 收尾帧由调用方按它自己的契约发
    return []


def _try_persist(persist, kind: str, **kw) -> None:
    if persist is None:
        return
    try:
        persist(kind, **kw)
    except Exception as exc:  # noqa: BLE001 — 落库失败不该中断一次运行
        logger.debug("group transcript persist failed (%s): %s", kind, exc)


__all__ = ["drive_group_run", "PendingFrames", "SSE_TEXT_MAX"]
