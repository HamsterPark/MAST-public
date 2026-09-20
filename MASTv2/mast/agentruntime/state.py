"""``TaskState`` —— 一次编排任务的状态。**编排器单线程独占。**

替代的是什么
------------
``agents/state.py`` 的 ``MASTState`` / ``AgentSubState``：35 个 ``Annotated`` reducer
通道、7 个自研 reducer、以及围绕它们长出来的一整套不变式。那些东西不是设计出来的，
是被 LangGraph 的通道语义**逼**出来的：

* 「每个被并发写的 key 必须有 reducer，否则 ``InvalidUpdateError``」——因为并行分支
  各自往同一个通道写；
* 「子图每个 channel 的 reducer 必须幂等」——因为子图 state 会经加法 reducer 合并回
  父图，agent 不交棒自然结束时 ``visit_count`` 一跳记两次（fail_silent_end 台账 621 次）；
* 「``budget_remaining_usd`` **刻意没有** reducer，这一处必须反着用上面那条不变式」
  ——因为给 ``NotRequired`` 通道加 reducer 会把缺省从「不存在」变成类型零值，而 0.0
  对一个 ``<= 0 → END`` 的闸门就是「预算耗尽」，**每个 run 第一跳就死**（试过，63 个
  测试当场红）。

这三条在新模型下**全部消失**，因为「并发写同一个通道」这个物理事实不存在了：并发的
只有 :class:`~mast.agentruntime.loop.AgentLoop` 的**纯返回值**，而写 state 的永远是
编排器那一个线程。合并发生在一个地方（:func:`merge_branch_results`），看得见、读得懂、
可以单独测。

分支输入是显式快照
------------------
``Send(payload)`` 的载荷是字面量输入、不经 reducer、也看不到同一个 ``Command`` 里的
``update``——所以生产代码必须手工写 ``{**state, **returned, "messages": kept + msgs_out}``。
那行看着像冗余的展开，其实是那条语义的唯一补偿；漏了它分支就拿着派发之前的世界干活，
**而且不报错**。这里 :class:`BranchInput` 的构造函数明写它携带什么，漏了是 TypeError
而不是「安静地少给一半」。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: 跨 agent 搬运的产物字段。与 ``artifact_channel.CARRIED_FIELDS`` **同一批名字**，
#: 但角色变了：那边是「海关要抄哪些通道」，这里只是「合并时认得哪些键」——运输本身
#: 由返回值完成，不需要海关。
ARTIFACT_FIELDS: tuple[str, ...] = (
    "research_campaign",
    "literature_report",
    "experiment_plan",
    "draft",
    "review",
    "analysis",
    "last_scan",
    "scan_id",
)

#: 审计类列表：追加并按值去重。对应旧世界的 ``dedupe_append``。
_AUDIT_LIST_FIELDS = ("executed_skills", "scan_paths", "error_log", "event_refs")

#: 字典类：右覆盖左。对应旧世界的 ``merge_dicts``。
_MERGE_DICT_FIELDS = ("scan_metadata", "pending_activations", "composite_progress")


@dataclass
class TaskState:
    """一次编排任务的全部状态。

    ⚠️ **只有编排器线程写它。** 分支拿到的是快照（:class:`BranchInput`），返回的是
    值（``AgentRunResult``）；合并回来这一步只在 :func:`merge_branch_results` 里发生。
    """

    #: 父转录：用户指令、路由备注、每次交棒的一句话。
    #: agent 的实质产出**不在这里**——那些走 artifacts 指针，正文在文档存储里。
    messages: list = field(default_factory=list)

    #: 每个 agent 被进入过几次。普通 dict：**「重置」就是赋一个新 dict**。
    #: 旧世界这是个加法 reducer 通道，传 `{}` 是 no-op 而不是重置，于是计数终身
    #: 累加，越过 total-40 之后每轮新任务立刻 END（会话砖化）。
    visit_count: dict = field(default_factory=dict)

    #: 产物指针（不是正文）。
    artifacts: dict = field(default_factory=dict)

    #: 审计与进度。
    executed_skills: list = field(default_factory=list)
    scan_paths: list = field(default_factory=list)
    error_log: list = field(default_factory=list)
    event_refs: list = field(default_factory=list)
    scan_metadata: dict = field(default_factory=dict)
    composite_progress: dict = field(default_factory=dict)
    pending_activations: dict = field(default_factory=dict)

    #: 本跳的路由意图（来自上一跳各分支的 handoff）。
    #: 旧世界这是 ``routing_hints`` 通道；现在它就是上一批返回值，用完即弃。
    routing_hints: list = field(default_factory=list)

    #: 身份。
    experiment_id: str = ""
    sample_id: str = ""
    run_id: str = ""

    #: 预算闸门。``None`` = **未接线**，不是 0。
    #:
    #: 这个显式的 ``| None`` 是旧世界那条「此处必须反着用不变式」的替身：在
    #: ``NotRequired[Annotated[float, last_wins]]`` 里，缺省会变成 ``0.0``，而对一个
    #: ``<= 0 → END`` 的闸门来说 0.0 不是无害默认，它是「预算耗尽」。整类陷阱在
    #: dataclass 下不存在——缺省就是我写在这里的这个值。
    budget_remaining_usd: float | None = None

    def snapshot(self) -> "TaskState":
        """深一层的拷贝，用于给分支做输入。"""
        return TaskState(
            messages=list(self.messages), visit_count=dict(self.visit_count),
            artifacts=dict(self.artifacts),
            executed_skills=list(self.executed_skills),
            scan_paths=list(self.scan_paths), error_log=list(self.error_log),
            event_refs=list(self.event_refs),
            scan_metadata=dict(self.scan_metadata),
            composite_progress=dict(self.composite_progress),
            pending_activations=dict(self.pending_activations),
            routing_hints=list(self.routing_hints),
            experiment_id=self.experiment_id, sample_id=self.sample_id,
            run_id=self.run_id, budget_remaining_usd=self.budget_remaining_usd)


@dataclass
class BranchInput:
    """派发给一条分支的**显式**输入快照。

    「分支看到的就是派发时刻的世界」在这里是构造函数的正常职责，而不是一条要记住
    的框架语义。
    """

    agent_id: str
    messages: list
    artifacts: dict = field(default_factory=dict)
    experiment_id: str = ""
    sample_id: str = ""
    run_id: str = ""
    reason: str = ""

    @classmethod
    def from_state(cls, state: TaskState, agent_id: str, *,
                   extra_messages: Iterable[Any] = (), reason: str = "",
                   ) -> "BranchInput":
        """从当前 state 造一份分支输入。

        ``extra_messages`` 是**本跳新增**的消息（supervisor 的派发备注）。旧世界这
        一步要手工写进 Send payload，漏了分支就看不见自己为什么被叫起来。
        """
        return cls(agent_id=agent_id,
                   messages=[*state.messages, *extra_messages],
                   artifacts=dict(state.artifacts),
                   experiment_id=state.experiment_id, sample_id=state.sample_id,
                   run_id=state.run_id, reason=reason)


@dataclass
class BranchOutcome:
    """一条分支跑完之后，编排器需要知道的一切。"""

    agent_id: str
    outcome: str = "final"
    final_text: str = ""
    stop_reason: str = ""
    handoff_target: str = ""
    handoff_reason: str = ""
    state_delta: dict = field(default_factory=dict)
    new_messages: list = field(default_factory=list)
    model_calls: int = 0
    tool_calls: int = 0


def merge_branch_results(state: TaskState, outcomes: Iterable[BranchOutcome],
                         ) -> TaskState:
    """把一批分支结果合并进 state。**全系统唯一一处写 state 的地方。**

    对比旧世界：同一件事分散在 7 个 reducer 里，每个都要独立地想清楚并发语义，
    而「哪些字段有 reducer」与「哪些字段会被并发写」是两张必须手工对齐的表——
    对不齐的后果是 ``InvalidUpdateError``，**每次并行 run 必崩**。

    这里并发不存在（分支返回的是值），所以合并顺序 = 传进来的顺序，确定且可测。
    """
    for out in outcomes:
        if out is None:
            continue
        delta = out.state_delta or {}

        # 产物：后写覆盖先写。分支之间产物字段本来就少有重叠（每个 agent 产它自己
        # 那一种），重叠时「这一批里最后一个说的算」是可解释的规则。
        for key in ARTIFACT_FIELDS:
            if key in delta and delta[key] is not None:
                state.artifacts[key] = delta[key]

        for key in _AUDIT_LIST_FIELDS:
            incoming = delta.get(key)
            if not incoming:
                continue
            target = getattr(state, key)
            for item in (incoming if isinstance(incoming, list) else [incoming]):
                if item not in target:      # 去重：同一路径被两条分支报上来只记一次
                    target.append(item)

        for key in _MERGE_DICT_FIELDS:
            incoming = delta.get(key)
            if isinstance(incoming, dict) and incoming:
                getattr(state, key).update(incoming)

        # 跳数：每条分支记一跳。**加在这里，而且只加一次** —— 旧世界靠
        # `sum_int_dicts` 加法 reducer，子图 state 合并回父图时会让「不交棒的自然
        # 结束」多记一跳（实测 literature:1 交棒 / literature:2 不交棒）。
        state.visit_count[out.agent_id] = state.visit_count.get(out.agent_id, 0) + 1

        # 分支的实质消息不进父转录（parent 只留路由与交接），但交接语要留下。
        if out.handoff_reason:
            state.routing_hints.append(
                {"from": out.agent_id, "target": out.handoff_target,
                 "reason": out.handoff_reason})

        # 未接线的预算保持 None；接线了才做减法。
        spent = delta.get("spent_usd")
        if spent and state.budget_remaining_usd is not None:
            state.budget_remaining_usd -= float(spent)

    return state


def state_from_dict(data: dict) -> TaskState:
    """从一个朴素 dict 造 TaskState（读快照、测试用）。未知键**忽略并记一条 debug**。

    忽略而不是抛：快照可能来自旧版本，多一个字段不该让会话打不开。记一条 debug
    是为了「少读了什么」有痕迹可查——静默丢弃正是这次迁移要根除的东西。
    """
    known = {f.name for f in fields(TaskState)}
    unknown = sorted(set(data or {}) - known)
    if unknown:
        logger.debug("state_from_dict ignoring unknown keys: %s", unknown)
    return TaskState(**{k: v for k, v in (data or {}).items() if k in known})


__all__ = [
    "TaskState",
    "BranchInput",
    "BranchOutcome",
    "merge_branch_results",
    "state_from_dict",
    "ARTIFACT_FIELDS",
]
