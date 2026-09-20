"""运行时事件 —— agent 循环与编排器**主动说出来**的东西。

与今天的区别在一个词：**主动**
------------------------------
现在的 SSE 帧是驱动层从 ``graph.stream()`` 的 chunk 里**推断**出来的：
agent 名字靠切 namespace 字符串（``"instrument_control:<uuid>"``）、去重靠攒
``msg.id`` 集合、「这一批结束了没有」靠数 chunk、而「分支是不是死了」靠闩住
``active_agent == "__end__"`` 再解析 dispatch note——**从残骸反推**。

推断出来的东西有两个毛病：它会在框架换实现时安静地失准（``stream_mode="updates"``
跨 subgraph 边界 re-emit、id 被重新分配都发生过），而且**推断不出没有痕迹的事**——
一条不交棒就结束的分支不留任何痕迹，于是「死分支」与「任务完成」在像素层面相同。

这里的事件由产生它的那一层直接构造：谁产生、属于哪个 agent、是什么结局，都是字段
而不是线索。

字段命名刻意贴近现有 SSE 帧
--------------------------
迁移期间新旧引擎要同时喂同一个前端。所以 ``kind`` / ``agent`` / ``role`` / ``text``
这些名字原样沿用；**新增**的是运行时无关的进度字段（见 :class:`StepEvent`），旧引擎
也会填。前端因此只改一次、两代同协议。

⚠️ 一条纪律：事件是**只读快照**。消费者不得改它——今天 SSE 桥里那种「先攒着、回头
补一个字段」的写法，会让「这条消息发出去的时候到底长什么样」变得无法回答。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

#: 分支/回合的结局。**这是 outcome 枚举的全部取值** —— 「静默死掉」不在其中，
#: 因为新循环里不存在「什么都不说就结束」这个可能。
Outcome = Literal[
    "handoff",   # 调了交棒工具：正常把控制权交回编排器
    "final",     # 模型给出了不带工具调用的答复：这一跳的交付物
    "limit",     # 撞到 model_calls / tool_calls 上限
    "stalled",   # StallGuard 判定原地打转，主动停机
    "aborted",   # 用户按了停止
    "error",     # 未预期的异常（已被捕获并带出原因）
]


@dataclass(frozen=True, slots=True)
class RunEvent:
    """所有事件的基类。``kind`` 是 SSE 帧上的那个 kind，原样沿用。"""

    kind: str = "event"
    agent: str = ""
    t: float = 0.0


@dataclass(frozen=True, slots=True)
class MessageEvent(RunEvent):
    """一条完整消息（对应今天的 ``kind="message"`` 帧）。"""

    kind: str = "message"
    role: str = "assistant"
    text: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenEvent(RunEvent):
    """文本增量。今天只有语音链消费它（``stream_mode=["updates","messages"]``）。"""

    kind: str = "token"
    text: str = ""


@dataclass(frozen=True, slots=True)
class ToolStartEvent(RunEvent):
    kind: str = "tool_start"
    name: str = ""
    preview: str = ""
    tool_call_id: str = ""
    #: 结构化参数。``preview`` 是**给日志看的截断串**，从它反解不出结构 ——
    #: 而群聊面板的「参数」折叠面板要的是真的 dict：
    #: ``narrate_tool_call`` 按参数名挑措辞，``summarize_args`` 输出带缩进的 JSON
    #: 并**报告**自己截了多少，而不是默默截断。
    args: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolEndEvent(RunEvent):
    kind: str = "tool_end"
    name: str = ""
    preview: str = ""
    tool_call_id: str = ""
    ok: bool = True


@dataclass(frozen=True, slots=True)
class StatusEvent(RunEvent):
    """状态行（等待审批的心跳、park 说明、软预算警告…）。"""

    kind: str = "status"
    text: str = ""
    subkind: str = ""


@dataclass(frozen=True, slots=True)
class InterruptEvent(RunEvent):
    """需要用户回答。字段与 ``hitl_bridge`` 归一化后的 dict 一一对应——那一层
    整体保留，所以形状不能变。"""

    kind: str = "interrupt"
    interrupt_id: str = ""
    interrupt_kind: str = ""          # ask_user | dangerous | workflow_human
    skill: str = ""
    params: dict = field(default_factory=dict)
    rationale: str = ""
    allowed_decisions: list = field(default_factory=list)
    ask: dict = field(default_factory=dict)
    thread_id: str = ""


@dataclass(frozen=True, slots=True)
class HandoffEvent(RunEvent):
    kind: str = "handoff"
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BranchDoneEvent(RunEvent):
    """**新增，而且是这次迁移最想要的那一个。**

    一条分支结束时必发，带着它的 :data:`Outcome`。今天没有对应物：不交棒的分支
    什么都不说，驱动层只能从 state 残骸里猜，猜的结果与「真完成」无法区分
    （见 ``tests/v2/agents/contract/test_silent_branch_death.py``）。

    ``stop_reason`` 是给**人**看的一句话，不是错误码：用户要能据此决定下一步。
    """

    kind: str = "branch_done"
    outcome: str = "final"
    stop_reason: str = ""
    final_text: str = ""


@dataclass(frozen=True, slots=True)
class CompactionEvent(RunEvent):
    """上下文被压缩过。**必须可见**——一段被摘要替换过的历史，读的人有权知道。"""

    kind: str = "compaction"
    removed: int = 0
    kept: int = 0
    est_tokens: int = 0


@dataclass(frozen=True, slots=True)
class StepEvent(RunEvent):
    """进度 —— **单位是真实单位**。

    今天前端收到的是 ``step`` / ``step_limit``，语义是「super-step 数 / recursion
    limit」。那个换算率随中间件数量漂移：同一个字面量 50 在 2026-06 买到约 9 次
    工具调用，2026-08 只买到 3 次，而**没有任何一次 diff 里出现过「限额被改小」**
    （见 ``test_superstep_pricing.py``）。

    这里发的是被预算的那个东西本身：模型调用数、工具调用数、编排跳数。
    ``unit`` 说明前端该拿哪一个当进度条——迁移期间旧引擎也填这三个字段（用它自己
    的口径），前端因此只改一次。
    """

    kind: str = "step"
    hops: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    limit: int = 0
    unit: str = "model_call"


@dataclass(frozen=True, slots=True)
class DoneEvent(RunEvent):
    """整个 run 的收尾帧。"""

    kind: str = "done"
    completed: bool = True
    failed: bool = False
    aborted: bool = False
    stop_reason: str = ""
    final_text: str = ""


def to_frame(event: RunEvent) -> dict:
    """事件 → SSE 帧 dict（前端今天就在消费的那个形状）。

    只做形状转换，不做任何补充推断。``None`` 与空串的区别在这里被保留：前端的
    ``typeof f?.step === "number"`` 这类判断依赖它。
    """
    from dataclasses import asdict

    out = {k: v for k, v in asdict(event).items() if v not in ((), [], {})}
    out["kind"] = event.kind
    if not out.get("t"):
        out.pop("t", None)
    if not out.get("agent"):
        out.pop("agent", None)
    return out


__all__ = [
    "Outcome",
    "RunEvent",
    "MessageEvent",
    "TokenEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "StatusEvent",
    "InterruptEvent",
    "HandoffEvent",
    "BranchDoneEvent",
    "CompactionEvent",
    "StepEvent",
    "DoneEvent",
    "to_frame",
]
