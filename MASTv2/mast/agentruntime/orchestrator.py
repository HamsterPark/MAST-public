"""``OrchestratorLoop`` —— 显式 while 循环，替代 supervisor ``StateGraph``。

承认现实
--------
今天的 orchestrator 图只有**一条静态边**（``START → supervisor``）；七个 agent 节点
没有出边，全部路由由 ``Command(goto=…)`` 在运行时决定。也就是说：那张图早已不是一张
图，它是一个 while 循环外面套了一层图的语法。这个模块把那层语法去掉，换来三样东西：

* **分支结局是值**。不交棒的分支不再静默停住——它返回一个 outcome，编排器必须处置。
* **产物是返回值**。不再需要 ``artifact_channel`` 那张「海关」（663 行）把字段从子图
  命名空间抄过 ``Command.PARENT`` 边界。
* **仪器互斥是显式调度约束**，而不是「barrier 恰好保证了它」。

关于 barrier
------------
super-step barrier 是 BSP 的**正确语义**，不是缺陷——仓库里论证过三次，这里不推翻。
一期保留等价的「批内 join」：一批分支全部结束才进下一跳。差别在于**它成了我们的
join**：等待期间可以 drain 插话、可以发心跳、惰性生成器天然支持 hold。滚动派发（先回
先派）留作后续开关，因为一旦开了，「同一时刻至多一个仪器 agent」就只剩
:meth:`OrchestratorLoop._instrument_gate` 这一道防线了。

不在这里的东西
--------------
路由用的 LLM 分类复用 ``agents/_shared/llm_route.py``（它已经是 provider 中立的
分层降级，重写一遍只会把 ``with_structured_output`` 那个六家五样的坑再踩一次）。
后台剥离仍然交给 ``core/background_runs.BackgroundRunManager``（560 行，零 langgraph）。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterable, Sequence

from mast.agentruntime.context import RunAborted, RunContext
from mast.agentruntime.events import (
    BranchDoneEvent,
    DoneEvent,
    HandoffEvent,
    RunEvent,
    StatusEvent,
    StepEvent,
)
from mast.agentruntime.state import (
    BranchInput,
    BranchOutcome,
    TaskState,
    merge_branch_results,
)

logger = logging.getLogger(__name__)

# ── 循环护栏（值取自 orchestrator/graph.py，逐条对齐） ────────────────
#: 全部 key 的跳数之和。
HOP_HARD_CAP = 60
#: 软提醒阈值：到 80% 时告诉 supervisor「预算快用完了，收敛」。
HOP_SOFT_THRESHOLD = int(HOP_HARD_CAP * 0.8)
#: 单个 agent 被进入的上限。
AGENT_HARD_CAP = 10
#: supervisor 自身的调度次数上限。
SUPERVISOR_HARD_CAP = 30
#: 一批 fan-out 最多几路。
MAX_PARALLEL = 4

#: supervisor 在 visit_count 里的 key。**它不算 agent** —— 把它算进 per-agent 上限
#: 是旧世界那个 cap 误触发的一半原因。
SUPERVISOR_KEY = "supervisor"


@dataclass
class RouteDecision:
    """路由器说了什么。``targets`` 空 = 结束。"""

    targets: list[str] = field(default_factory=list)
    reason: str = ""
    direct_answer: str = ""


@dataclass
class TaskResult:
    outcome: str = "completed"          # completed | limit | aborted | error
    final_text: str = ""
    stop_reason: str = ""
    hops: int = 0
    state: TaskState | None = None


class OrchestratorLoop:
    """把若干 :class:`~mast.agentruntime.loop.AgentLoop` 编排起来。

    ``run_branch`` 是注入的：编排器不关心一条分支是怎么跑的（本地 AgentLoop、
    远端、还是测试替身），只关心它**返回一个 :class:`BranchOutcome`**。这让
    编排逻辑可以脱离模型单独测。
    """

    def __init__(self, *, route: Callable[[TaskState, RunContext], RouteDecision],
                 run_branch: Callable[[BranchInput, RunContext], BranchOutcome],
                 instrument_agents: Iterable[str] = ("instrument_control",),
                 max_parallel: int = MAX_PARALLEL,
                 drain_interjections: Callable[[], list[str]] | None = None):
        self._route = route
        self._run_branch = run_branch
        self._instrument = set(instrument_agents)
        self._max_parallel = max(1, int(max_parallel))
        self._drain = drain_interjections

    # ── 主循环 ─────────────────────────────────────────────────────────
    def run_task(self, state: TaskState, instruction: str,
                 ctx: RunContext | None = None,
                 ) -> Generator[RunEvent, None, TaskResult]:
        ctx = ctx or RunContext(run_id=state.run_id)
        if instruction:
            state.messages.append({"role": "user", "content": instruction})
        hops = 0
        final_text = ""

        try:
            while True:
                if ctx.aborted():
                    return self._done(ctx, "aborted", final_text,
                                      "用户中止了本次运行。", hops, state)

                # ① 硬闸：跳数与预算 ────────────────────────────────
                stop = self._hard_gate(state)
                if stop:
                    return self._done(ctx, "limit", final_text, stop, hops, state)

                # ② 插话：**每一跳**都 drain，而不是只在批次之间 ──────
                for text in (self._drain() if self._drain else []):
                    state.messages.append({"role": "user", "content": text})
                    yield StatusEvent(text=f"用户插话：{text}",
                                      subkind="interjection", t=time.time())

                # ③ 软提醒 ────────────────────────────────────────────
                warn = self._soft_warning(state)
                if warn:
                    yield StatusEvent(text=warn, subkind="budget", t=time.time())

                # ④ 路由 ─────────────────────────────────────────────
                try:
                    decision = self._route(state, ctx)
                except RunAborted:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning("routing failed: %s", exc)
                    return self._done(ctx, "error", final_text,
                                      f"路由失败：{type(exc).__name__}: {exc}",
                                      hops, state)

                if decision.direct_answer:
                    final_text = decision.direct_answer
                    state.messages.append({"role": "assistant",
                                           "content": decision.direct_answer})
                    return self._done(ctx, "completed", final_text, "", hops, state)

                targets = self._admit(decision.targets, state)
                if not targets:
                    return self._done(ctx, "completed", final_text,
                                      decision.reason, hops, state)

                # ⑤ 派发 ─────────────────────────────────────────────
                hops += 1
                state.visit_count[SUPERVISOR_KEY] = (
                    state.visit_count.get(SUPERVISOR_KEY, 0) + 1)
                for t in targets:
                    yield HandoffEvent(from_agent=SUPERVISOR_KEY, to_agent=t,
                                       reason=decision.reason, t=time.time())

                outcomes = self._dispatch(targets, state, decision, ctx)

                # ⑥ 汇合：**唯一一处写 state 的地方** ─────────────────
                merge_branch_results(state, outcomes)
                for out in outcomes:
                    if out.final_text:
                        final_text = out.final_text
                        state.messages.append(
                            {"role": "assistant",
                             "content": f"[{out.agent_id}] {out.final_text}"})
                    # ★ 分支的结局一律显式广播。旧世界里不交棒的分支什么都不说，
                    #   于是「死分支」与「任务完成」像素级相同。
                    yield BranchDoneEvent(agent=out.agent_id, outcome=out.outcome,
                                          stop_reason=out.stop_reason,
                                          final_text=out.final_text, t=time.time())

                yield StepEvent(hops=hops,
                                model_calls=sum(o.model_calls for o in outcomes),
                                tool_calls=sum(o.tool_calls for o in outcomes),
                                limit=HOP_HARD_CAP, unit="hop", t=time.time())

        except RunAborted:
            return self._done(ctx, "aborted", final_text,
                              "用户中止了本次运行。", hops, state)

    # ── 闸门 ───────────────────────────────────────────────────────────
    def _dimensions(self, state: TaskState) -> tuple[int, int, str, int]:
        """``(总跳数, 最忙 agent 的次数, 它的名字, supervisor 次数)``。

        一个 helper，因为硬闸、软提醒、日志三处**必须**对「supervisor 算不算
        agent」有同一个答案。旧世界这里错过一次：算进去了，于是 per-agent 上限
        误触发。
        """
        counts = {k: int(v or 0) for k, v in (state.visit_count or {}).items()}
        total = sum(counts.values())
        agents = {k: n for k, n in counts.items() if k != SUPERVISOR_KEY}
        if agents:
            worst_name = max(agents, key=lambda k: agents[k])
            return total, agents[worst_name], worst_name, counts.get(SUPERVISOR_KEY, 0)
        return total, 0, "", counts.get(SUPERVISOR_KEY, 0)

    def _hard_gate(self, state: TaskState) -> str:
        total, worst, worst_name, sup = self._dimensions(state)
        if total > HOP_HARD_CAP:
            return (f"编排跳数达到上限（{total}/{HOP_HARD_CAP}）。"
                    "已完成的部分保留，未完成的请重新下发一个更聚焦的任务。")
        if worst >= AGENT_HARD_CAP:
            return (f"{worst_name} 已被进入 {worst}/{AGENT_HARD_CAP} 次，"
                    "再派它多半是在原地打转。")
        if sup >= SUPERVISOR_HARD_CAP:
            return f"编排器自身调度已达 {sup}/{SUPERVISOR_HARD_CAP} 次。"
        if state.budget_remaining_usd is not None and state.budget_remaining_usd <= 0:
            # None 与 0.0 在这里是**两件事**：前者是「没接线」，后者是「花完了」。
            return "本次运行的预算已用尽。"
        return ""

    def _soft_warning(self, state: TaskState) -> str:
        total, _worst, _name, _sup = self._dimensions(state)
        if total == HOP_SOFT_THRESHOLD:      # 只在跨过阈值那一跳说一次
            return (f"已用 {total}/{HOP_HARD_CAP} 跳（≥80%），"
                    "循环预算即将耗尽，请开始收敛。")
        return ""

    def _admit(self, targets: Sequence[str], state: TaskState) -> list[str]:
        """把路由器给的目标规整成一批可派发的。

        三件事：去重（顺带让「一批里至多一个 IC」成立）、限宽、**仪器互斥**。
        """
        seen: list[str] = []
        for name in targets or ():
            name = str(name or "").strip()
            if name and name not in seen:
                seen.append(name)
        return self._instrument_gate(seen)[: self._max_parallel]

    def _instrument_gate(self, targets: list[str]) -> list[str]:
        """★ 一批里**至多一个**驱动仪器的 agent。

        旧世界这条是两个机制拼出来的：批内靠 ``_coerce_targets`` 的通用去重（IC
        至多一次只是去重的**副产品**），批间靠 super-step barrier（框架送的，代码
        里一行都没有，只在注释里存在）。

        这里把它写成显式规则。一期保留批内 join，所以它与现状等价；真正成为**唯一**
        保证是在未来开滚动派发的时候——那时这道闸就是拦得住并发仪器动作的全部。

        技能粒度的令牌（``core.instrument_lock``）仍在，但它保证的不是同一件事：
        barrier 保证第二个 IC **根本不会被派发**，令牌只保证「被派发了会在技能边界
        被拒」——后者意味着 run 已经在跑了，操作体验差得多。
        """
        out: list[str] = []
        instrument_seen = False
        for name in targets:
            if name in self._instrument:
                if instrument_seen:
                    logger.info("instrument gate: dropping a second %s from this "
                                "fan-out", name)
                    continue
                instrument_seen = True
            out.append(name)
        return out

    # ── 派发 ───────────────────────────────────────────────────────────
    def _dispatch(self, targets: list[str], state: TaskState,
                  decision: RouteDecision, ctx: RunContext) -> list[BranchOutcome]:
        """跑一批分支并等它们全部结束（批内 join）。

        单目标不起线程池：绝大多数跳是单目标，为它付一个线程池的建立成本没有道理，
        而且**单线程路径下工具仍然跑在调用线程上** —— 那正是「停止」够得着技能的
        原因，不该因为走了编排器就丢掉。
        """
        note = {"role": "assistant",
                "content": f"[SUPERVISOR] {decision.reason}"} if decision.reason else None
        extra = [note] if note else []

        def _one(agent_id: str) -> BranchOutcome:
            branch_ctx = ctx.for_agent(agent_id)
            payload = BranchInput.from_state(state, agent_id, extra_messages=extra,
                                             reason=decision.reason)
            try:
                return self._run_branch(payload, branch_ctx)
            except RunAborted:
                return BranchOutcome(agent_id=agent_id, outcome="aborted",
                                     stop_reason="用户中止了本次运行。")
            except Exception as exc:  # noqa: BLE001
                logger.warning("branch %s raised: %s", agent_id, exc)
                return BranchOutcome(
                    agent_id=agent_id, outcome="error",
                    stop_reason=f"{type(exc).__name__}: {exc}")

        if len(targets) == 1:
            return [_one(targets[0])]
        with ThreadPoolExecutor(max_workers=len(targets),
                                thread_name_prefix="orch-branch") as pool:
            return list(pool.map(_one, targets))

    # ── 收尾 ───────────────────────────────────────────────────────────
    def _done(self, ctx: RunContext, outcome: str, final_text: str,
              stop_reason: str, hops: int, state: TaskState) -> TaskResult:
        ctx.send(DoneEvent(completed=outcome == "completed",
                           failed=outcome in ("error", "limit"),
                           aborted=outcome == "aborted",
                           stop_reason=stop_reason, final_text=final_text,
                           t=time.time()))
        return TaskResult(outcome=outcome, final_text=final_text,
                          stop_reason=stop_reason, hops=hops, state=state)


__all__ = [
    "OrchestratorLoop",
    "RouteDecision",
    "TaskResult",
    "HOP_HARD_CAP",
    "AGENT_HARD_CAP",
    "SUPERVISOR_HARD_CAP",
    "MAX_PARALLEL",
]
