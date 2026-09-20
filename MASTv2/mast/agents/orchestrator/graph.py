"""Orchestrator — Phase 7 real implementation.

Replaces the Phase-3 stub. Builds a top-level StateGraph that routes between
7 agent subgraphs based on structured-output classification of the user's
goal and the current state. (research_director joined 2026-08-21 — the Campaign
layer, i.e. the half of the decision loop that decides what is worth doing at
all; it holds no hardware tool and is routed to only when the ask is a research
DIRECTION rather than a task.)

Compass §4.3 pattern:
  - Custom StateGraph with a supervisor node (NOT langgraph-supervisor lib —
    deprecated in favour of tool-handoff pattern).
  - Sonnet 4.6 default; dynamic Opus 4.7 escalation on decision-critical paths.
  - Triple loop guard:
      1. recursion_limit=50 — bound as a build-time DEFAULT via
         compiled.with_config(recursion_limit=50) in build() (LangGraph has no
         compile() param for it; it is an invoke-time config defaulting to 25).
         An explicit invoke-time recursion_limit still overrides this default.
      2. ModelCallLimit + ToolCallLimit on each agent (per-agent middleware)
      3. visit_count ceilings in supervisor_node (this file) — THREE dimensions
         since 2026-07-30: total hops (60), any ONE real agent (10), and the
         supervisor's own key (30). The per-agent dimension had been removed in
         2026-06-29 because it misfired; both of its root causes are now fixed
         (cross-task accumulation, and the supervisor key sharing the agents'
         ceiling), and the numbers come from the billing ledger rather than from
         a guess. See the constants for the measurements.
  - Budget hard-gate: state["budget_remaining_usd"] <= 0 → END. This was a
    CALLER-SEEDED guard () and an audit on 2026-07-30 found that no
    caller anywhere in the tree ever seeded it — inert for its whole life, while
    one measured run spent $30.66 in an hour on a loop of SUCCESSES that
    StallGuard cannot see. It is now refreshed every hop from ``budget_probe``, a
    zero-arg callable the host supplies (wired from mast.billing.run_meter, whose
    docstring states the attribution caveat). The graph still performs no billing
    and imports nothing from the billing layer; absent a probe the gate behaves
    exactly as before.
  - Budget SOFT-limit (review MEDIUM #7): a non-blocking, fire-once warning
    injected into the message stream at ~80% consumption of whichever loop-guard
    ceiling is reached first — hop total >= 48 of 60, or one real agent >= 7 of
    10 — so the agent converges before the hard gate aborts it mid-thought.
    Idempotent via a unique message marker — adds NO new MASTState field /
    reducer. (A "$ budget soft limit" was REMOVED 2026-06-10: it keyed off
    budget_initial_usd, which is NOT a declared MASTState channel and so never
    survived a real graph hop — dead code. The hard USD gate above is unaffected.)

Orchestrator is the SOLE agent allowed to import sibling agents — see the
agent_boundary 钩子（不随仓） exception. Each agent's build() factory is
imported lazily inside `build()` so partial-graph tests stay light.
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from mast.agents._shared.models import (
    AGENT_MODEL,
    make_chat_model,
    model_id_of as _model_id_of,
    rejects_forced_tool_choice as _rejects_forced_tool_choice,
)


class _SkipTier1(Exception):
    """Internal control flow: this model would 400 on a forced tool call, so the
    function_calling tier is skipped rather than attempted-and-caught."""
from mast.agents._shared.artifact_channel import (
    render_upstream_block as _render_upstream,
)
from mast.agents._shared.roster import AGENT_NAMES
from mast.agents.state import MASTState
from mast.prompts.registry import resolve as _resolve_prompt

if TYPE_CHECKING:
    from mast.admin.override_store import ConfigOverrideRegistry
    from mast.config import SafetyLimits
    from mast.core.types import HardwareState

logger = logging.getLogger(__name__)


def _effective_safety_limits(
    safety_limits: "SafetyLimits | None",
    override_registry: "ConfigOverrideRegistry | None",
) -> "SafetyLimits | None":
    """Merge admin SafetyLimits overrides on top of the supplied (code-default) limits.

    Why this lives in the orchestrator (review: admin safety override lost on the
    multi-agent path): instrument_control.build() exposes a ``safety_limits`` knob
    and a ``get_state`` knob, but does NOT accept a ConfigOverrideRegistry — its
    SafetyGateMiddleware is constructed with ``registry=None``, so the admin
    JSON-override merge that the manual executor path enjoys (via
    core.safety.SafetyGuard → _get_effective_limits) never ran on the autonomous
    agent path. We therefore resolve the EFFECTIVE limits HERE and hand the merged
    object down as ``safety_limits`` so an admin-tightened bias/current/Z/scan-size
    cap is honoured by SafetyGateMiddleware.check_global_bounds on the agent path
    too — without touching instrument_control/graph.py.

    Returns:
        - the merged SafetyLimits when a registry with safety_limits overrides is
          present;
        - the unmerged ``safety_limits`` (possibly None → IC build uses its own
          default) otherwise.

    Fail-OPEN to the supplied limits on any merge failure: a corrupt override file
    must never disable the agent (which would crash build_ic and silently take the
    instrument agent offline), but the failure is logged at WARNING (mirrors
    core.safety._get_effective_limits, security ).
    """
    if override_registry is None:
        return safety_limits
    try:
        from mast.config import SafetyLimits as _SafetyLimits
        base = safety_limits if safety_limits is not None else _SafetyLimits()
        ovr = override_registry.get_safety_limits()
        if ovr:
            merged = base.model_copy(update=ovr)
            logger.info(
                "orchestrator: applied %d admin SafetyLimits override(s) to the "
                "instrument_control agent path: %s",
                len(ovr), sorted(ovr.keys()),
            )
            return merged
        return base
    except Exception as exc:  # noqa: BLE001 — never disable the agent on a bad file
        logger.warning(
            "orchestrator: admin SafetyLimits override merge failed (%s); "
            "passing un-merged limits to instrument_control", exc,
        )
        return safety_limits


#: 名单本身住在 ``_shared/roster.py``（2026-08-27 搬走）—— 「有哪几个 agent」是
#: 系统的组成，不是这张图的实现细节，而 ``api/routes/orchestrator.py`` 为了拿它
#: 曾经得 import 这个即将被删的模块。这里保留旧名字，调用点一个都不用改。
_AGENT_NAMES = AGENT_NAMES

# ── Budget soft-limit (review: MEDIUM #7) ──
# Hard gates END the run; the soft limit is a NON-blocking, fire-once nudge at
# ~80% consumption so the agent steers toward a conclusion before the hard gate
# trips and aborts mid-thought. The unique marker below is what makes the
# warning idempotent: we scan existing messages for it and skip re-injecting,
# so no new MASTState field / reducer is needed (keeps the contract intact).
_SOFTLIMIT_MARKER = "[SUPERVISOR:BUDGET_SOFTLIMIT]"
# ── Loop-guard ceilings (three dimensions, restored + retuned 2026-07-30) ──
#
# The counter all three read only became meaningful again on 2026-07-29:
# `visit_count` accumulated for the lifetime of a thread because run_task's
# "reset" wrote an empty dict into an ADDING reducer (see sum_int_dicts), and it
# was still double-counting until AgentSubState narrowed the subgraph schema on
# 2026-07-30 — a branch that ended WITHOUT handing off merged its whole state
# back through the adding reducer, so one hop was charged twice (measured; the
# diagnostics ledger has 621 fail_silent_end events, every one of them inflating
# this counter). Thresholds set against a counter that reads double are
# meaningless, which is why the per-agent dimension could not come back before
# that fix landed.
#
# Why the per-agent cap was removed on 2026-06-29, and what was ACTUALLY wrong:
# it fired on a FRESH task in a long conversation. Two causes, only one of which
# was understood at the time —
#   1. cross-task accumulation (the reducer bug above) — fixed 2026-07-29;
#   2. **the `supervisor` key was counted against the same per-agent ceiling**,
#      and it is structurally the largest key by a wide margin: the supervisor is
#      re-entered on EVERY hop, so it is ~half of `total` on its own.
#      `_shared/handoff.py` says so in a comment ("the per-agent guard (cap 8) —
#      for which the supervisor key is the binding constraint") and the ledger
#      confirms it: a clean six-stage pipeline puts `supervisor` at 7, a
#      LEGITIMATE 4-agent run with one revision cycle at >=16, and the $30.66
#      runaway at >=34. A cap of 8 shared with the supervisor could not survive
#      even the clean case.
# So the cap returns SPLIT: real agents get their own ceiling, and the supervisor
# gets a much higher one of its own — which is also the better ping-pong detector,
# since 34 routing hops is exactly what the runaway looked like.
_HOP_HARD_CAP = 60                                        # sum over ALL keys
_HOP_SOFT_THRESHOLD = int(_HOP_HARD_CAP * 0.8)            # 48
#: Per REAL agent (supervisor excluded). Measured peak on a legitimate run: ~5.
_AGENT_HARD_CAP = 10
_AGENT_SOFT_THRESHOLD = 7
#: The supervisor's own key, on its own ceiling — the ping-pong detector.
_SUPERVISOR_HARD_CAP = 30
#: The key that is NOT an agent. Spelled once; every guard excludes it by name.
_SUPERVISOR_KEY = "supervisor"


def _guard_dimensions(vc: dict) -> "tuple[int, int, str, int]":
    """``(total, worst_agent_count, worst_agent_name, supervisor_count)``.

    One helper so the hard gate, the soft warning and the dispatch log cannot
    drift apart on how `visit_count` is read — in particular on the one thing
    that has already gone wrong once: whether `supervisor` counts as an agent.
    It does not.
    """
    counts = {k: int(v or 0) for k, v in (vc or {}).items()}
    total = sum(counts.values())
    agents = {k: n for k, n in counts.items() if k != _SUPERVISOR_KEY}
    if agents:
        worst_name = max(agents, key=lambda k: agents[k])
        worst = agents[worst_name]
    else:
        worst_name, worst = "", 0
    return total, worst, worst_name, counts.get(_SUPERVISOR_KEY, 0)

# ── Parent-channel size management (2026-07-29) ─────────────────────────────
# Until now NOTHING trimmed the parent `messages` channel. It got away with that
# for one accidental reason: `visit_count` never reset, so a long-lived thread
# hit the hop guard and died before the channel could grow. Fixing that reset
# (which had been bricking group conversations at 40 cumulative hops) removes the
# only thing that bounded this channel, so the bound has to become real.
#
# A deterministic keep-last-N, not an LLM summary, and that is a deliberate
# choice about WHAT lives here: since the artifact channel landed, the parent
# carries routing notes and one-sentence handoffs — the actual products travel as
# typed pointers in their own channels and are NOT touched by this. There is no
# prose here worth paying a summariser call (or a summariser failure) for.
# Agent-internal history, which IS worth summarising, is handled inside each
# subgraph by the compaction middleware.
_PARENT_MSG_SOFT_CAP = 160     # start pruning past this many messages
_PARENT_MSG_KEEP = 80          # …down to this many of the most recent
#: Marker so the prune note is greppable in a transcript and idempotent-ish.
_PARENT_PRUNE_MARKER = "[SUPERVISOR:CONTEXT_TRIMMED]"
#: Must MATCH ``compaction_mw.COMPACTION_META_KEY``. Spelled as a literal for the
#: same reason the SSE bridge does: neither side imports the other's module. A
#: parity test pins the three spellings together.
_COMPACTION_META_KEY = "mast_compaction"


def _plan_parent_prune(messages: list) -> "tuple[set, Any]":
    """Which parent messages to drop, plus the note that records the drop.

    Returns ``(ids_to_remove, note_or_None)``. Empty set + None when under the
    cap, which is the overwhelmingly common case.

    Two rules that are not obvious:

      * the FIRST message is always kept — it is the operator's instruction, and
        a run that forgets what it was asked to do is worse than a long one;
      * a message with no ``id`` cannot be removed (``RemoveMessage`` addresses by
        id), so it is left alone rather than silently counted as pruned.

    The note is a real message rather than a silent deletion because a transcript
    that quietly loses its middle is indistinguishable from one that never had
    it — the same reason compaction is stamped rather than applied invisibly.
    """
    try:
        msgs = list(messages or [])
        if len(msgs) <= _PARENT_MSG_SOFT_CAP:
            return set(), None
        head = msgs[:1]                       # the operator's original ask
        tail = msgs[-_PARENT_MSG_KEEP:]
        keep = {id(m) for m in head} | {id(m) for m in tail}
        drop_ids = {mid for m in msgs
                    if id(m) not in keep and (mid := getattr(m, "id", None))}
        if not drop_ids:
            return set(), None
        note = AIMessage(
            content=(
                f"{_PARENT_PRUNE_MARKER} 群聊主线已裁掉 {len(drop_ids)} 条较早的路由/交接消息"
                f"（保留最初的指令与最近 {_PARENT_MSG_KEEP} 条）。各智能体的**产物没有受影响**："
                "文献报告 / 实验方案 / 分析结果 / 草稿 / 评审仍在各自的产物通道与磁盘上，"
                "需要就按上下文里给出的 doc_id 读回。"),
            # Ride the EXISTING compaction-visibility chain (SSE frame →
            # transcript kind → operator panel), which until now had nothing real
            # to report: group compaction ran inside subgraphs and its output was
            # discarded at the parent boundary, so the indicator was wired to an
            # event that never reached here.
            #
            # ``mode`` matters. That chain's wording is "已被摘要替代", and this is
            # NOT a summary — it is a deterministic drop with no LLM involved.
            # Reusing the sentence would be a lie of exactly the kind this
            # codebase keeps paying for, so the renderer branches on it.
            additional_kwargs={_COMPACTION_META_KEY: {
                "mode": "parent_prune",
                "removed": len(drop_ids),
                "kept": min(len(msgs), _PARENT_MSG_KEEP),
                "before": len(msgs),
            }},
        )
        return drop_ids, note
    except Exception as exc:  # noqa: BLE001 — housekeeping must never break routing
        logger.debug("parent prune planning failed: %s", exc)
        return set(), None
# The per-agent hop cap was removed 2026-06-29 and RESTORED (split, retuned)
# 2026-07-30 — the full history, both root causes and the measured numbers behind
# the new thresholds are with the constants above.
# NB: a "$ budget soft limit" (warn at <20% of budget_initial_usd) was REMOVED
# — budget_initial_usd is not a declared MASTState channel,
# so it never propagated through the real graph and the dimension was dead code.
# The hard USD gate (budget_remaining_usd <= 0 → END) is a real channel, and as of
# 2026-07-30 it is finally SEEDED + refreshed by a caller-supplied probe
# (`build(budget_probe=…)`, fed by mast.billing.run_meter) instead of sitting
# inert. See the probe's docstring for the accounting caveat.


# ── Activation gating: park instead of dispatching into fiction (2026-07-30) ──
# docs/v2/design/wakeup_scheduling.md §3. Two layers, in this order:
#
#   1. a DETERMINISTIC pre-filter — a hard dependency is missing, so dispatching
#      cannot produce anything but an invented artifact. No model call: this is not
#      a judgement. (`artifact_channel.REQUIRES`, deliberately only two entries.)
#   2. only if layer 1 passes and something SOFT is missing, ASK the agent, with
#        the asymmetry of failure stated in the question (see _shared/activation).
#
# The most important property of the whole mechanism is that a park is VISIBLE.
# This repo has paid for "nothing happened" and "it hung" looking identical (the
# 2026-07-28 fail-silent work; 621 events in the diagnostics ledger), and a parked
# agent is exactly that shape unless it is announced. So every park writes a
# transcript line AND a state entry — and W3 adds the disk board + UI row that make
# it survive the run.
_PARK_MARKER = "[SUPERVISOR:PARKED]"
#: Marks a delivery from a woken background run, so an operator reading the
#: transcript can tell where a product that nobody in this run produced came from.
_RETURN_MARKER = "[SUPERVISOR:WOKEN-RETURN]"

#: ``pending_activations`` statuses written by this node.
#: ``waiting`` = parked; ``asked`` = we asked and it chose to start, recorded so a
#: later hop in the SAME run does not pay for the same question again (a soft miss
#: is the common case, so without this every dispatch re-asks — unbounded cost and
#: latency for a decision already made).
_PARK_WAITING = "waiting"
_PARK_ASKED = "asked"


def _goal_text(state: MASTState, limit: int = 600) -> str:
    """The operator's original ask, for putting in the activation question.

    The FIRST human message, not the latest: the question is "can you do THIS TASK
    with what exists", and by the time an agent is parked the tail of the channel is
    routing notes and handoff sentences. ``_plan_parent_prune`` keeps the first
    message forever for the same reason.

    Since 2026-08-27 an explicit ``goal.text`` wins over the message scan. The
    scan was always an approximation: ``run_task`` prepends resume / operator-reply
    context (``_resume_lead_messages``), so on a resumed thread the "first human
    message" is a resume block or the PREVIOUS task's ask — not this run's. When
    the caller states the goal, we use what they stated.
    """
    stated = str((_goal_dict(state)).get("text") or "").strip()
    if stated:
        return stated[:limit]
    for m in (state.get("messages") or []):
        if "human" in m.__class__.__name__.lower():
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content
                                   if isinstance(b, dict) and b.get("type") == "text")
            text = str(content).strip()
            if text:
                return text[:limit]
    return ""


# ── 目标终止判据 (2026-08-27) ────────────────────────────────────────────
#
# 在此之前，「这次任务做完了没有」全部写在路由提示词里（`# 自主推进流水线` 那一
# 节），由模型自己判；图里没有任何一条边在检查它，代码层的终止只有跳数/预算/
# 递归三个熔断。于是同一个缺口生出两个方向相反的老病：**太早停**（模型说「完
# 了」——fail_silent 那 621 条）与**停不下来**（每次唤醒是新 run，per-run 的熔断
# 全归零，而每一步都「成功」，StallGuard 按重复失败签名结构上看不见它）。
#
# 这三个标记不是装饰：它们让「为什么停 / 为什么没停」在 transcript 里可读，
# 也让「本 run 问过没有」不必新开一个 state 字段（与软限警告同一个技巧）。
_GOAL_MARKER = "[SUPERVISOR:GOAL]"
_GOAL_ASKED_MARKER = "[SUPERVISOR:GOAL_ASKED]"


def _goal_dict(state: MASTState) -> dict:
    g = (state or {}).get("goal")
    return g if isinstance(g, dict) else {}


def _goal_verdict(state: MASTState, *, ask_enabled: bool = True):
    """本 run 的判据这一刻怎么看，没有判据时回 ``None``。

    ``None`` 是**逐字节保持今天行为**的那条路：没给 ``done_when`` 的调用方
    （也就是今天所有的调用方）在下面每一处都会因为它而整段跳过。

    判据读不懂时回一个 ``unknown`` 而不是 ``None`` —— 「写坏了」与「没写」是两
    件事，前者要在 transcript 里说出来。两者对路由的影响相同（都由模型说了算），
    但只有一种会被人看见并去修。
    """
    g = _goal_dict(state)
    if not g.get("done_when"):
        return None
    try:
        from mast.agents._shared.artifact_channel import _present_in_state
        from mast.goals import (
            UNKNOWN,
            GoalVerdict,
            evaluate_done_when,
            normalise_done_when,
        )
        from mast.goals.sources import make_collector
    except Exception as exc:  # noqa: BLE001 — 路由永远不因为判据模块死掉
        logger.debug("goal gate unavailable: %s", exc)
        return None

    spec, errs = normalise_done_when(g.get("done_when"))
    if errs:
        logger.warning("goal done_when 读不懂，本 run 按无判据走: %s", errs)
        return GoalVerdict(verdict=UNKNOWN, reason="判据读不懂：" + "；".join(errs))

    answered = (g.get("operator_confirmed") or {}).get("answer")
    collect = make_collector(
        baseline=g.get("baseline"),
        # 按**这一 run 的纲领**数论断。不带作用域的话 ``claims_supported``
        # 数的是全库历史 —— 库里早有两条别的实验的 supported 论断，
        # ``min_count: 2`` 第一跳就满足。
        campaign_id=_campaign_id_of(state),
        # state 里带着 = 铁证（这一 run 里刚交过班）。state 里**没有**证明不了
        # 什么 —— 新 run 的 state 本来就空 —— 所以那一侧由收集器落到磁盘。
        state_present=lambda f: _present_in_state(state, f),
        operator_answer=answered,
        askable=bool(ask_enabled),
    )
    return evaluate_done_when(spec, collect)


def _goal_blocked(state: MASTState) -> str:
    """有没有东西正卡着（等资料 / 等人）。有则**允许**结束，并说明卡在哪。

    一个被搁置的 agent 既不是完成也不是失败。判据没满足而它在等，这时逼着继续
    派发只会得到同一个结果 —— 那是把一次诚实的等待变成一个死循环。
    """
    waiting = [
        a for a, e in (state.get("pending_activations") or {}).items()
        if isinstance(e, dict) and e.get("status") == _PARK_WAITING
    ]
    if state.get("pending_user_question"):
        waiting.append("（在等用户答复）")
    return "、".join(waiting)


def _goal_already_asked(state: MASTState) -> bool:
    """本 run 问过用户没有 —— **每 run 最多一次**。

    没有这个上限，一个坚持要结束的模型与一道坚持不让结束的闸会互相顶下去，
    直到撞上 ``_SUPERVISOR_HARD_CAP``：那样也能停，但停在一个看不懂的地方，
    而且中间每一跳都在花钱。上限的代价是「有时会在判据没满足时结束」，但那一次
    结束是**大声的**（见 ``_goal_note``），不是静默的。

    读的是 ``goal.asked``，**不是**消息流里的标记（2026-08-28 修）。第一版还扫
    ``messages`` 找 ``_GOAL_ASKED_MARKER``，想的是「checkpoint 回来时 goal.asked
    可能没跟上」；但 ``messages`` 用 ``add_messages``、**跨任务**留在同一个
    checkpoint 线程里，于是同一个群聊里的第二个任务一开始就是「已经问过了」——
    这道闸对它从来没生效过。而 ``goal.asked`` 是随 ``Command.update`` 写进通道
    的，同一 run 内的 resume 照样读得到，跨任务则被 ``initial_state`` 的显式
    清除带走。标记仍然写进 transcript（那是给人读的），只是不再拿它做决定。
    """
    return bool((_goal_dict(state)).get("asked"))


def _goal_note(verdict, decision: str, extra: str = "") -> AIMessage:
    """决定性的那一跳在 transcript 上留下的一行。

    「什么都没发生」和「它挂了」不能长得一样 —— 这条纪律在 park 上已经付过一次
    学费，目标闸门同样：一次由代码做出的结束/不结束，必须能被读出来。
    """
    head = {
        "done": "目标判据已全部满足，任务结束。",
        "hold": "路由说该结束了，但目标判据没满足 —— 先问用户。",
        "unknown": "目标判据这一刻读不到，按模型判断走（下面这句是留痕）。",
        # blocked 与 unknown **不是**一回事：判据读得清清楚楚，只是没满足，而有
        # 东西正卡着。用 unknown 那句话，用户和下一跳的路由模型都会读成
        # 「目标系统坏了」，而不是「有 park 在等」。
        "blocked_ok": "判据没满足，但有东西正卡着 —— 这既不是完成也不是失败。",
        "ended_unmet": "判据未满足仍结束（本 run 已经问过一次用户）。",
        "no_ask": "判据未满足，但这次运行没法向用户提问（没有 checkpointer）。",
    }.get(decision, decision)
    body = f"{_GOAL_MARKER} {head}"
    if verdict is not None:
        body += f"（{verdict.satisfied}/{verdict.total} 满足；{verdict.reason}）"
    if extra:
        body += f" {extra}"
    return AIMessage(content=body)


def _goal_record(verdict, decision: str, state: MASTState) -> None:
    """把这次求值记进诊断台账。**永不抛。**

    用户问「它为什么停了 / 为什么没停」时看的就是这本账（``kinds=("goal_verdict",)``），
    与 fail_silent 那 621 条同一本 —— 那正是这道闸要治的病。
    """
    try:
        from mast.core import diagnostics

        diagnostics.record(
            "goal_verdict",
            subject=str(state.get("run_id") or state.get("experiment_id") or ""),
            reason=(verdict.reason if verdict is not None else ""),
            # 字段**摊平**：台账是一行一条、要被 grep 的东西，嵌套一层 dict
            # 会让「查所有 decision=hold 的行」变成一件要写脚本的事。
            decision=decision,
            verdict=(verdict.verdict if verdict is not None else "none"),
            satisfied=(verdict.satisfied if verdict is not None else 0),
            total=(verdict.total if verdict is not None else 0),
            unmet=[i.text for i in (verdict.unmet if verdict else [])],
            unknown=[i.text for i in (verdict.unknowns if verdict else [])],
        )
    except Exception as exc:  # noqa: BLE001 — 记账坏了不许影响路由
        logger.debug("goal diagnostics failed: %s", exc)


def _goal_hold_question(verdict, state: MASTState, valid_targets) -> dict:
    """判据没满足却要结束时，问用户的那张卡。**由代码生成，不由模型生成。**

    选项里每一条未满足的产物判据都对应「继续：派 X」，其中 X 取自
    ``artifact_channel.producer_of`` —— 不是让模型再想一次该派谁，而是把
    「谁产出这个东西」这个**已经登记过**的事实摆出来。``routes`` 把选项映射回
    agent 名，用户选完就变成一条确定性的交接提示，不再经路由模型。
    """
    try:
        from mast.agents._shared.artifact_channel import field_label, producer_of
    except Exception:  # noqa: BLE001
        field_label = producer_of = None  # type: ignore[assignment]

    unmet_items = list(verdict.unmet if verdict is not None else [])

    # 只差「用户确认」这一条 ⇒ 问的不是「继续还是结束」，而是「够了吗」。
    # 同一张问答预算（每 run 一次）、同一条 replay 安全的路径，只是卡面不同 ——
    # 问错问题比不问更糟：一个已经做完全部工作的 run 被问「要不要继续推进」，
    # 用户只能靠猜我们在说什么。
    if unmet_items and all(i.kind == "operator_confirmed" for i in unmet_items):
        return {
            "question": ("目标里写着「要你点头才算完」。该做的都做完了"
                         f"（{verdict.satisfied}/{verdict.total} 项判据满足）——够了吗？"),
            "options": ["够了，就到这里", "还不够，继续"],
            "kind": "goal_confirm",
            "routes": {},
        }

    options: list[str] = []
    routes: dict[str, str] = {}
    for item in unmet_items:
        field = str(item.args.get("field") or "")
        who = ""
        if field and producer_of is not None:
            try:
                who = producer_of(field)
            except Exception:  # noqa: BLE001
                who = ""
        if who and who in (valid_targets or ()) and who not in routes.values():
            label = field_label(field) if field_label is not None else field
            opt = f"继续：派 {who}（产出{label}）"
            options.append(opt)
            routes[opt] = who
    end_opt = "就此结束（记录为未达成）"
    options.append(end_opt)

    unmet = "；".join(i.text for i in unmet_items) or "（无）"
    question = (
        f"路由判断这次任务可以结束了，但目标判据还差：{unmet}。"
        "要继续推进，还是就此结束？"
    )
    return {"question": question, "options": options, "kind": "goal_hold",
            "routes": routes}


def _agent_tools_for(agent: str) -> "set[str] | None":
    """The agent's REAL tool names, for rendering its context block truthfully.

    Rule 1 of the question design: the "what you already have" block in the
    question must be produced by the same renderer, with the same tool list, as the
    block the agent will actually receive — otherwise it decides from one
    description of the world and then works from another. Degrades to None (the
    canonical hint) rather than raising, because a routing decision must not depend
    on tool introspection succeeding.
    """
    try:
        from mast.agents._shared.artifacts import agent_tool_names

        return agent_tool_names().get(agent) or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("park: tool names unavailable for %s: %s", agent, exc)
        return None


def _artifact_is_newer(incoming: Any, current: Any) -> bool:
    """Should ``incoming`` replace ``current`` in a product channel?

    The channels use ``last_wins``, which is right for concurrent writes inside one
    super-step but wrong for a delivery from OUTSIDE: a woken run's products were
    produced from a snapshot taken when it was spawned, and the mainline may have
    moved on since. Blind last-wins there would overwrite a NEWER pointer with an
    older one — the agent would then be shown a stale summary and, worse, a stale
    ``doc_id`` version to continue from.

    Compare by version when both sides expose one; otherwise accept only into an
    empty slot. Refusing an ambiguous overwrite is the safe direction: the woken
    run's product is still on disk and still discoverable, whereas a clobbered
    pointer is silently gone.
    """
    if current is None or current == "" or current == {}:
        return True
    if incoming is None:
        return False

    def _v(obj) -> "int | None":
        for attr in ("version",):
            val = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
            if isinstance(val, int):
                return val
        return None

    iv, cv = _v(incoming), _v(current)
    if iv is None or cv is None:
        return False
    return iv > cv


def _drain_return_inbox(state: MASTState) -> "tuple[dict, list[str]]":
    """Merge woken runs' products back into the mainline. ``({field: val}, notes)``.

    Drained at ``_dispatch``, which the module's own comment calls the one choke point
    every dispatch passes through — so a future routing branch cannot be added that
    silently drops deliveries.

    A woken run cannot write the mainline's checkpoint (checkpoints are per-thread and
    two writers corrupt it), so it posts to the durable inbox and this is the pickup.
    Without it a woken agent's work would exist on disk and be invisible to everyone,
    which is the same as not having done it.
    """
    try:
        from mast.core.park_board import board
    except Exception as exc:  # noqa: BLE001
        logger.debug("return inbox unavailable: %s", exc)
        return {}, []
    try:
        entries = board().drain_returns()
    except Exception as exc:  # noqa: BLE001 — routing never dies over the mailbox
        logger.debug("return inbox drain failed: %s", exc)
        return {}, []
    if not entries:
        return {}, []

    merged: dict[str, Any] = {}
    notes: list[str] = []
    for entry in entries:
        agent = entry.get("agent") or "?"
        taken: list[str] = []
        skipped: list[str] = []
        for field, value in (entry.get("artifacts") or {}).items():
            current = merged.get(field, _get_state_field(state, field))
            if _artifact_is_newer(value, current):
                merged[field] = value
                taken.append(field)
            else:
                skipped.append(field)
        if taken or skipped:
            note = f"{_RETURN_MARKER} 后台唤醒的 {agent} 已回收产物:{'、'.join(taken) or '（无新版本）'}"
            if skipped:
                note += f"；{'、'.join(skipped)} 主线已有更新的版本,未覆盖"
            notes.append(note)
        # Close the park this run was woken for. Until this point the park sat at
        # ``woken``, which is deliberately NOT terminal: a detached run on an
        # InMemorySaver can die and take its work with it, and a board that called
        # ``woken`` finished would have no record that the chain broke.
        pid = entry.get("park_id")
        if pid:
            try:
                board().mark_done(pid, note=f"产物已回主线（run {entry.get('run_id')}）")
            except Exception as exc:  # noqa: BLE001
                logger.debug("cannot close park %s: %s", pid, exc)
    if merged:
        logger.info("return inbox: merged %s from %d woken run(s)",
                    sorted(merged), len(entries))
    return merged, notes


def _get_state_field(state: MASTState, field: str) -> Any:
    try:
        return state.get(field)
    except Exception:  # noqa: BLE001
        return None


def _campaign_id_of(state: MASTState) -> str:
    """这一 run 属于哪条科研纲领 —— 冻进 park 用。

    ``research_campaign`` 可能是 :class:`CampaignRef`，也可能是一个从
    checkpoint 里解回来的 dict（同一份数据两种形状是本仓的常态），两种都要认；
    认不出来就回空串，**不猜**。猜错的代价是把 A 纲领的判据套到 B 的 park 上，
    而那种错不会报错，只会静默地不唤醒。
    """
    ref = (state or {}).get("research_campaign")
    if ref is None:
        return ""
    cid = ref.get("campaign_id") if isinstance(ref, dict) else getattr(
        ref, "campaign_id", "")
    return str(cid or "")


def _persist_park(agent: str, waiting_for: list[str], reason: str,
                  instruction: str, *, hard: bool, campaign_id: str = "") -> str:
    """Write the park to the DURABLE board and return its ``park_id`` ("" on failure).

    The board — not ``MASTState`` — is the authority, because the whole point of a park
    is to be woken later and when the system is idle there is no state to hold it. The
    state entry is a per-run cache; **writes go to the board only**, so the graph
    thread and the wake scheduler thread cannot drift apart on the decline count.

    Best-effort: if the board cannot be written the park still happens in state and is
    still announced in the transcript. Degrading to a park that does not survive the
    run is much better than failing the routing decision — and it is visible either
    way, which is the property that matters.
    """
    try:
        from mast.core.park_board import board

        rec = board().park(agent, waiting_for=list(waiting_for), reason=reason,
                           instruction=instruction, hard=hard,
                           campaign_id=campaign_id)
        return str(rec.get("park_id") or "")
    except Exception as exc:  # noqa: BLE001 — routing never dies over bookkeeping
        logger.warning("park board write failed for %s: %s", agent, exc)
        return ""


def _plan_activation(state: MASTState, targets: list[str], *, ask_model,
                     instruction: str, gate=None) -> "tuple[list[str], list[dict], dict]":
    """Split ``targets`` into (dispatchable, parked) and return the state delta.

    ``parked`` entries are ``{agent, waiting_for, reason, hard}``. The returned dict
    is the ``pending_activations`` delta to merge.

    ``gate`` is a zero-arg callable → bool, read live from settings. **Absent or
    False → this whole mechanism is OFF and every target dispatches**, byte-for-byte
    the pre-2026-07-30 behaviour. Same shape and same reason as ``background_gate``
    just above: a new routing behaviour that changes what happens by default changes
    it for every existing caller and test at once, and this particular behaviour
    (not dispatching an agent) is the one the design itself flags as most likely to
    become a silent death channel. It gets a switch.

    Never raises: any failure degrades to "dispatch everything". A scheduler that
    parks an agent because its own readiness check crashed is strictly worse than
    one that lets an imperfect run proceed visibly.
    """
    if not gate:
        return list(targets), [], {}
    try:
        if not gate():
            return list(targets), [], {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("activation gate read failed (treated as OFF): %s", exc)
        return list(targets), [], {}
    try:
        from mast.agents._shared.activation import ask_should_start
        from mast.agents._shared.artifact_channel import readiness
    except Exception as exc:  # noqa: BLE001
        logger.debug("activation gating unavailable: %s", exc)
        return list(targets), [], {}

    import time as _time

    prior = state.get("pending_activations") or {}
    go: list[str] = []
    parked: list[dict] = []
    delta: dict[str, Any] = {}

    for agent in targets:
        try:
            r = readiness(agent, state)
        except Exception as exc:  # noqa: BLE001
            logger.debug("readiness failed for %s: %s", agent, exc)
            go.append(agent)
            continue

        if r["missing_hard"]:
            # Layer 1. No model call — "review a manuscript that does not exist"
            # has no judgement in it.
            entry = {
                "status": _PARK_WAITING,
                "waiting_for": list(r["missing_hard"]),
                "reason": "缺少硬依赖,现在派发只能产出编造的内容",
                "hard": True,
                "at": _time.time(),
            }
            entry["park_id"] = _persist_park(
                agent, entry["waiting_for"], entry["reason"], instruction, hard=True,
                campaign_id=_campaign_id_of(state))
            delta[agent] = entry
            parked.append({"agent": agent, "hard": True, **entry})
            continue

        soft = list(r["missing_soft"])
        unknown = list(r["unknown"])
        if not soft and not unknown:
            go.append(agent)
            continue

        was = prior.get(agent) or {}
        if was.get("status") in (_PARK_ASKED, _PARK_WAITING):
            # Already decided this run. Re-asking would spend a model call and
            # several seconds per dispatch to re-derive an answer we hold, and a
            # soft miss is the ordinary case rather than the exception.
            if was.get("status") == _PARK_WAITING:
                parked.append({"agent": agent, "hard": False,
                               "waiting_for": list(was.get("waiting_for") or soft),
                               "reason": str(was.get("reason") or "")})
            else:
                go.append(agent)
            continue

        # Layer 2 — ask the agent itself.
        try:
            decision = ask_should_start(
                ask_model, agent=agent, instruction=instruction, state=state,
                missing_soft=soft, unknown=unknown,
                available_tools=_agent_tools_for(agent))
        except Exception as exc:  # noqa: BLE001 — activation.ask never raises, belt+braces
            logger.debug("activation ask failed for %s: %s", agent, exc)
            go.append(agent)
            continue

        if decision.get("action") == "wait":
            entry = {
                "status": _PARK_WAITING,
                "waiting_for": list(decision.get("waiting_for") or soft),
                "reason": str(decision.get("reason") or ""),
                "hard": False,
                "at": _time.time(),
            }
            entry["park_id"] = _persist_park(
                agent, entry["waiting_for"], entry["reason"], instruction, hard=False,
                campaign_id=_campaign_id_of(state))
            delta[agent] = entry
            parked.append({"agent": agent, "hard": False, **entry})
        else:
            delta[agent] = {"status": _PARK_ASKED, "at": _time.time(),
                            "reason": str(decision.get("reason") or ""),
                            "waiting_for": []}
            go.append(agent)

    return go, parked, delta


def _park_note(parked: list[dict]) -> AIMessage:
    """The transcript line that keeps a park from being a silent death.

    Names the agent, what it waits for, and who produces that — an operator reading
    「什么都没发生」 must be able to tell WHY without opening a log.
    """
    from mast.agents._shared.artifact_channel import field_label, producer_of

    lines = [f"{_PARK_MARKER} 以下智能体本轮**没有派发**,在等资料到位:"]
    for p in parked:
        waits = "、".join(
            f"{field_label(f)}"
            + (f"（{producer_of(f)} 产出）" if producer_of(f) else "")
            for f in (p.get("waiting_for") or [])) or "（未记录）"
        why = "硬依赖缺失" if p.get("hard") else (p.get("reason") or "自行选择等待")
        lines.append(f"  · {p['agent']} —— 等 {waits};原因:{why}")
    lines.append("等到的时候会重新询问它是否醒来;也可以在「待唤醒」面板里查看或直接催办。")
    return AIMessage(content="\n".join(lines))


def _parked_context_line(state: MASTState) -> str:
    """What the ROUTER is told about currently-parked agents.

    Without this the supervisor re-picks a parked agent every hop (each time paying
    a readiness check that short-circuits) or reads the absence of its output as
    "that stage is done" and finishes early. The scheduler only gained product
    visibility on 2026-07-30; this rides the same channel, for the same reason —
    it is being asked to decide something it cannot currently observe.
    """
    parked = {
        a: e for a, e in (state.get("pending_activations") or {}).items()
        if isinstance(e, dict) and e.get("status") == _PARK_WAITING
    }
    if not parked:
        return ""
    from mast.agents._shared.artifact_channel import field_label

    lines = ["## 当前被搁置（等资料）的智能体",
             "它们**不是**已完成,也不是失败 —— 是缺上游资料在等。"
             "缺的东西到位之前,再派发它们只会得到同一个结果;"
             "**但也不要把它们的空白当成任务完成**。"]
    for agent, e in parked.items():
        waits = "、".join(field_label(f) for f in (e.get("waiting_for") or [])) or "?"
        lines.append(f"- {agent}:等 {waits}")
    return "\n".join(lines)


def _already_soft_warned(messages) -> bool:
    """True if the fire-once budget soft-limit warning is already in the stream.

    Idempotency without a dedicated state flag: the warning carries a unique
    marker substring; if any prior message contains it we must not re-inject.
    Tolerant of non-string message content (lists/dicts) — only str content can
    match the marker.
    """
    for m in messages or []:
        content = getattr(m, "content", None)
        if isinstance(content, str) and _SOFTLIMIT_MARKER in content:
            return True
    return False


def _budget_softlimit_warning(
    state: MASTState, total: int,
    *, worst_agent: int = 0, worst_agent_name: str = "",
) -> AIMessage | None:
    """Return a one-shot soft-limit warning AIMessage, or None.

    Non-blocking, read-only over state. Fires when EITHER loop-guard dimension
    crosses ~80% consumption AND the warning has not already been injected.
    Returns None otherwise (no warning needed yet, or already warned). Never
    changes the hard-gate decision — the caller has already passed the hard gates
    by the time this runs.

    Dimensions, mirroring the ceilings that actually exist:
      1. ``total`` hops >= 80% of ``_HOP_HARD_CAP`` (60 → 48);
      2. one REAL agent's visits >= ``_AGENT_SOFT_THRESHOLD`` (7 of 10).

    Dimension 2 came back on 2026-07-30 with the per-agent cap it mirrors. The
    supervisor's key is excluded from it by ``_guard_dimensions`` — counting it
    was half of why the old cap misfired.

    NB: a third "remaining USD < 20% of budget_initial_usd"
    dimension USED to live here but was DEAD CODE — ``budget_initial_usd`` is not
    a declared ``MASTState`` channel, so inside the real compiled graph LangGraph
    drops it and ``state.get("budget_initial_usd")`` is always None (verified:
    only declared channels propagate). Without a stored initial reference the
    "80% consumed" fraction is undefined, so the dollar soft limit could never
    fire on a real run. The hard USD gate (``budget_remaining_usd <= 0`` →
    END, a real declared channel) is unaffected and still honoured by
    ``supervisor_node``. The dollar dimension was removed rather than wired
    because adding a new state channel is out of scope here; if a $ soft limit is
    wanted later, declare ``budget_initial_usd`` in ``MASTState`` first.
    """
    if _already_soft_warned(state.get("messages")):
        return None

    reason: str | None = None

    # Dimension 1: total inter-agent hops approaching the total loop-guard cap.
    if total >= _HOP_SOFT_THRESHOLD:
        reason = (
            f"已用 {total}/{_HOP_HARD_CAP} 跳(≥80%),循环预算即将耗尽"
        )
    # Dimension 2: ONE agent being re-entered over and over — the shape of a
    # two-agent ping-pong, which the total can stay well under.
    elif worst_agent >= _AGENT_SOFT_THRESHOLD and worst_agent_name:
        reason = (
            f"{worst_agent_name} 已被进入 {worst_agent}/{_AGENT_HARD_CAP} 次"
            "(单个智能体上限的 ≥70%),再返工几轮就会被强制中断"
        )

    if reason is None:
        return None
    return AIMessage(
        content=(
            f"{_SOFTLIMIT_MARKER} 软限警告:{reason}。"
            "请尽快收敛到结论,避免在硬性预算上限处被强制中断。"
        )
    )


class Route(TypedDict):
    next_agent: Literal[
        "research_director",
        "literature",
        "experiment_design",
        "instrument_control",
        "data_processing",
        "paper_writing",
        "paper_review",
        "__end__",
    ]
    reason: str


# ── Parallel dispatch (2026-07-11, ) ─────────────────────────
# The supervisor was strictly serial hub-and-spoke: ONE agent per hop, so a
# literature survey could not run while the instrument scanned, and a 6-phase
# campaign paid the sum of every phase's latency.
#
# SAFETY ARGUMENT for running agents concurrently (operator-confirmed): the
# instrument is touched by exactly ONE agent — instrument_control. No other
# agent holds a Nanonis handle or a hardware skill, so no two concurrent
# branches can contend for the instrument. The remaining agents are pure
# compute / network (literature search, offline analysis, drafting, review).
# The one invariant that still MUST hold is therefore mechanical:
#   ► instrument_control may appear AT MOST ONCE in a fan-out (enforced by
#     de-duplication — a set can't hold it twice), and there is never a second
#     concurrent IC because LangGraph's super-step barrier makes every branch
#     finish before the supervisor dispatches again.
#
# Fan-out width is capped (_MAX_PARALLEL) so a confused router cannot light up
# the whole roster at once and multiply token spend.
_MAX_PARALLEL = 4


# ── Automatic backgrounding (conservative whitelist; ) ─────
# When the router pairs an INDEPENDENT analysis agent with instrument_control in
# ONE fan-out, that analysis is a long side-task the operator wants running WHILE
# they keep working the instrument — but the super-step barrier makes the whole
# batch (incl. the fast instrument turn) wait for it. When the auto-background
# gate is ON, such an agent is peeled off to a DETACHED background run (the bridge
# spawns it) so the instrument foreground stays responsive.
#
# The whitelist is deliberately just {literature}: a background run has ISOLATED
# state (its own thread_id + checkpointer), so an agent that consumes prior
# artifacts from the LIVE thread — paper_writing needs the draft, paper_review the
# draft, data_processing the saved scan — would run against EMPTY state and produce
# garbage. literature is the one agent whose work is self-contained from the goal
# (it surveys papers), so it is the only SAFE auto-background target.
# instrument_control is NEVER eligible (it IS the foreground hardware agent).
# Policy is PREFER-MISSING: any doubt → don't background (falls back to the normal
# foreground fan-out, which already works). The gate DEFAULTS OFF.
_AUTO_BG_WHITELIST = frozenset({"literature"})
# Unique ASCII marker the supervisor emits so the bridge (routes/orchestrator)
# knows which agents to spawn as background runs. Distinct from the "[SUPERVISOR →
# …]" dispatch note so a text parse can't confuse the two.
_AUTO_BG_MARKER = "[SUPERVISOR::AUTO_BACKGROUND]"


def _split_auto_background(
    targets: list[str], background_gate, is_slow_type=None
) -> "tuple[list[str], list[str]]":
    """Split router ``targets`` into (foreground, background) per the conservative
    auto-background policy.

    ``background`` is non-empty ONLY when ALL hold:
      * ``background_gate`` is wired AND returns True (default: no gate → OFF);
      * ``instrument_control`` is in the batch (there IS foreground hardware work
        the barrier would otherwise block — no point detaching otherwise);
      * a whitelisted independent agent (``literature``) is in the batch.

    ``is_slow_type`` (optional, item ②) is a duration advisor ``(agent_type) ->
    bool | None``: True = historically slow (worth detaching), False = historically
    FAST (skip the detach overhead → keep foreground), None = too few samples
    (prefer-missing → the static whitelist decides). It only ever REMOVES a
    needless detach of a proven-fast whitelisted agent; it NEVER promotes a
    non-whitelisted (state-unsafe) agent into ``background``. Absent / None advisor
    → behaviour is byte-for-byte the pre-② whitelist policy.

    Fail-safe / prefer-missing: any gate/advisor error, or an empty resulting
    foreground, returns ``(targets, [])`` — i.e. the unchanged normal fan-out.
    instrument_control can never be in ``background`` (it is not in the whitelist)."""
    if not background_gate:
        return targets, []
    try:
        if not background_gate():
            return targets, []
    except Exception:  # noqa: BLE001 — a gate glitch must never drop or misroute work
        return targets, []
    if "instrument_control" not in targets:
        return targets, []
    bg = [t for t in targets if t in _AUTO_BG_WHITELIST]
    # item ② — duration refinement: drop a whitelisted candidate from the
    # background set ONLY when the advisor is CONFIDENT it is fast (verdict is
    # False). Unknown (None, too few samples) or slow (True) keeps the pre-②
    # whitelist behaviour of detaching it.
    if bg and callable(is_slow_type):
        kept: list[str] = []
        for t in bg:
            try:
                verdict = is_slow_type(t)
            except Exception:  # noqa: BLE001 — advisor glitch must never drop work
                verdict = None
            if verdict is False:
                continue  # proven fast → not worth detaching, keep foreground
            kept.append(t)
        bg = kept
    if not bg:
        return targets, []
    fg = [t for t in targets if t not in bg]
    if not fg:  # never leave the foreground empty (IC is present, so unreachable)
        return targets, []
    return fg, bg


class _RouteClarify(TypedDict, total=False):
    """The OPTIONAL half of :class:`ParallelRoute` — see its docstring.

    Split into a ``total=False`` base rather than annotated ``NotRequired[...]``
    because this module runs under ``from __future__ import annotations``: every
    annotation is a string, ``TypedDict`` cannot see through
    ``"NotRequired[str]"`` when it computes ``__required_keys__``, and the fields
    come out REQUIRED. Verified, not assumed — and the failure mode is the whole
    reason these are optional: a required clarify_question forces every provider
    to invent one on every route.
    """

    # Set INSTEAD of dispatching, when the goal is too ambiguous to route: the
    # supervisor asks the operator rather than guessing or quietly ending.
    clarify_question: str
    clarify_options: list[str]


class ParallelRoute(_RouteClarify):
    """Multi-target routing decision. ``next_agents`` may name 1..N agents to run
    CONCURRENTLY, or the single sentinel "__end__" to finish.

    The two ``clarify_*`` keys are optional and MUST stay that way. This schema is
    the one place all six providers have to agree on, and the tiered router exists
    because they disagree about everything else (F5); a router that omits them —
    every provider that ignores an optional field, plus tier 3, which cannot
    express them at all — must keep routing exactly as before.
    """

    next_agents: list[str]
    reason: str


_ROUTER_PROMPT = """你是 MAST 编排器。把用户的目标拆成阶段，派给**一个或多个** agent ——
把**此刻就能开工**的每一个都点名。

  - **research_director** — 科研策划（Campaign 层「为什么做」）。定/改一条**科研
    纲领**：可证伪的假设、什么算答完了、这条线和既往工作的谱系关系。用它的场合是
    用户给的是一个**科学方向**而不是一件具体的事（「接下来该研究什么」「这批数据
    说明我们的假设错了吗」「把这条线继续往下推」），或者要**回顾/迭代**已有纲领。
    它产出的是一份**委托**，交给 experiment_design 起草方案。
    ⚠️ 一件已经说清楚的具体活（「扫一张图」「查一下文献」「写份报告」）**不要**
    先绕它一趟 —— 那只是多烧一轮，纲领层对它没有任何东西可加。
  - **literature** — 检索 / 阅读文献、抽取实验协议。
  - **experiment_design** — 把研究问题变成实验方案；**也用来登记**关于 MAST 自身的
    升级建议 / 心愿单 / 功能请求（它有 report_upgrade_idea 可以归档）。
  - **instrument_control** — 驱动 SPM（扫描、STS、针尖/马达），做**扫描进行中**的
    在线判读（找平整区、评团簇圆度、查这一帧有没有撞针），**以及**管理实验/样品
    生命周期（新建/结束/重命名实验或样品）。**所有硬件与实时技能都归它。**
  - **data_processing** — 离线分析**已经存盘**的扫描 / STS 文件（2D FFT、扣平面、
    缺陷检测、拼图、在已存盘 .sxm 里找平整区、评团簇圆度）。**不做实时硬件读数。**
  - **paper_writing** — 把实验记录组装成报告 / 手稿章节。
  - **paper_review** — 对草稿做方法学 / 数据 / 引文的核查。
  - **__end__** — 任务完成，**或**卡住等人。

派单规则：
  - **research_director 只在来的是「研究方向」而不是「一件活」时才走**
    （「接下来做什么」/「这条线还值得做吗」/「把假设更新一下」）。
    它会读既往纲领与记录，产出/修订纲领并把委托交给 experiment_design。
  - 用户问「关于 X 别人做过什么」→ 先 **literature**：设计之前先收集先验。
  - literature 之后交 experiment_design，除非用户明确说跳过规划。
  - experiment_design 出了 ExperimentPlan 之后，交 instrument_control 执行。
  - instrument_control 把要求的扫描/STS **全部**做完之后，交 data_processing。
    **它的报告里如果还有没做完的测量**（要「5 个点」只报了 2 个），
    **派回 instrument_control** 并点名缺哪几个 —— data_processing 碰不到仪器，
    补不了那些数据。
  - 写报告的流程：分析之后交 paper_writing。**只有用户要了审稿**
    （「审一下」/「把关」/「要投出去」）才加 paper_review；
    一份内部实验报告写完就直接 __end__。
  - 硬熔断只有一道：一个 thread 内**所有 agent 的总跳数超过 {hop_hard_cap} 即 END**
    （visit_count 之和触发）。它是防跑飞的兜底，**不是可以用满的配额**。
    正常任务里每个 agent 走 1 次，个别 2 次；到第 3 次就该怀疑是在原地打转
    （见下方「每个阶段只走一遍」）。
  - 用户的目标**完全达成**，或者**卡住了**，就派 __end__。

# 自主推进流水线（autonomously）—— 不要等人催

**整条研究环归你。** 拿到目标之后，你自己把它推过每一个阶段；用户**不应该**
需要一个阶段一个阶段地催（查文献 / 写方案 / 写报告 / 并行指挥 —— 这些是**你**
该主动发起的，不是他该提醒的）：
  - 任何**新的**研究问题：**先派 literature** 收集先验 —— 哪怕用户只说了目标、
    没说「搜论文」。
  - 把研究目标变成方案：执行之前交 **experiment_design** 出一份 ExperimentPlan，
    不必等人说「写个方案」。
  - **一次实验没被分析并写下来，就不算做完。** data_processing 分析完之后，
    **主动**派 **paper_writing** 起草报告 —— 不要停在分析上干等「写个报告」。
    然后 __end__：审稿不是默认动作（见「每个阶段只走一遍」）。
  - **互不依赖的活并行跑**（见下面「并行派单 / Parallel dispatch」），不要串起来 ——
    例如 instrument_control 在扫描时同时跑 literature；仪器在采下一张时同时离线
    分析已存盘的那一张。
  - **阶段之间的空档不等于任务完成。** 目标里还有没开工的阶段、而你又没有卡在
    「等人答复 / 等硬件」上，就自己派下一个阶段，不要 __end__。
  - 实验/样品生命周期（新建/结束/重命名 实验或样品）→ instrument_control。
  - **读实时扫描通道**的撞针检查 → instrument_control（要硬件）。
    找平整区 / 评团簇圆度**两个 agent 都能做**：当前实时扫描找 instrument_control，
    已存盘的 .sxm 找 data_processing —— 两边都有这些技能，别硬性二选一。
  - **登记关于 MAST 自身的升级建议 / 心愿单 /「希望有个…skill」不是闲聊** ——
    派给 experiment_design，让它用 report_upgrade_idea 归档，**绝不要** __end__。
  - **中断后续跑**（一个已经有 agent 活动的线程上收到「继续」/「continue」）：
    **不要假设上一个阶段做完了**。去看**最后那个 agent 的末尾消息** —— 如果流断掉
    时它的计划还有剩余步骤（例如「第 4 点完成。最后第 5 点」之后再无下文），
    就**派回那个 agent**，并在 reason 里把没做完的那一步写清楚。
    只有当一个 agent **明确报告完成**时，才算那个阶段做完了。

# 每个阶段只走一遍（这一段和上一段同等重要）

上面说的是**不要等人催**，不是**尽量多做**。默认形状是：

    （仅当来的是科研方向）科研策划 →
    文献 → 计划 → 仪器 → 分析 → 报告 → __end__
                                    └─（仅当用户要求）→ 审稿 → 改一次 → __end__

**科研策划不是默认的第一站。** 它只在「要做什么科学问题」还没定的时候走一次；
问题已经定了（哪怕只是一句「测一下这个样品的能隙」），就直接从文献或计划开始。

**每个阶段走一遍就够了。一遍过是正常结果，不是敷衍。**

- **先看「已有产物」那一块再决定派谁。** 上下文里如果有那一块，它列的是此刻**真实
  存在**的产物（文献报告 / 实验方案 / 扫描 / 分析结果 / 草稿 / 评审，带 doc_id）。
  某个阶段的产物**已经在那里**，就不要再派那个阶段 —— 这不是推测，是读数。
  反过来也成立：那一块里**没有**某样东西，就不要假设它已经做完了。
  它是空的或者根本没出现，只说明目前还没有任何产物，不说明别的。
- 一个 agent 交回结果后，除非它**明确报告了失败**、或用户提出了新要求，
  否则不要再派给它。「结果可以更好」不是重派的理由。
- **审稿是可选的，不是默认的**。只有用户明确要了（「审一下」「帮我把把关」
  「要投出去」）才派 paper_review。他说「写一份实验报告」，那就是写完就 __end__ ——
  内部实验报告的读者是他本人，他自己会看。
- **paper_writing ↔ paper_review 最多一个来回，且改完不再复审**：
  审一次 → 改一次 → __end__。**不要**把改后的稿子再送回 paper_review 确认，
  那一轮几乎不会改变结论，却要烧掉和前面全部工作相当的步数。
  审稿方若仍有意见，把它作为遗留问题写进交付说明交给用户。
  报告的最终标准由人定，不由两个 agent 互相说服。
- 同一个 agent 在一个任务里被派到**第 3 次**时，先停下来问：这次和上次有什么
  实质不同？答不上来就 __end__ 并如实说明卡在哪。（唯一的硬熔断是总跳数 > 40；
  它是防跑飞的兜底，不是配额 —— 真走到那里意味着早就该停了。）
- 用户没要求的阶段**不要自己加**。他说「扫图 + STS + 报告」，那就没有审稿也可以；
  他说「搜一下文献」，那就搜完交回，不要顺势去写实验方案。
- 任务完成的判据是「**用户要求的每个阶段都走过一遍且没有硬失败**」，
  不是「所有 agent 都没意见了」。后者永远不会到来。

拿不准该不该再派一轮时：**__end__**，并说明当前状态。用户再说一句话就能继续，
而多跑的那一轮要占仪器、烧 token、还常常引出更多不必要的工作。

# 并行派单（Parallel dispatch）

**把此刻能推进的每一个 agent 都点名** —— 它们**并发**跑，而且**整批跑完**你才会
被再次问到。这是让一次漫长的仪器运行不再挡住其它所有事的办法。

  - **活互不依赖时并行派**：instrument_control 在扫描时同时跑文献调研；仪器在采
    下一张时同时离线分析已存盘的那一张。
  - **下一步依赖上一步的产物时，只派一个**：data_processing 要扫描文件先存在；
    paper_writing 要分析结果；paper_review 要草稿。
    **绝不要把一条依赖链扇出去** —— 那只会让下游 agent 因为缺输入而失败。
  - instrument_control 是**唯一**碰仪器的 agent，所以把它和任何别的 agent 配对
    都是安全的。**同一批里不要点它两次。**
  - **BACKGROUND（真后台，突破本批 barrier）**：一件确实很长、且**独立**的支线活
    （大规模文献调研、离线分析一批存盘扫描），而用户想让它跑着、自己继续跟仪器
    对话 —— 这种**不要用扇出**（扇出的整批仍要跑完你才会被再问）。
    派给 **instrument_control**：它会用 ``spawn_background_task`` 把这件支线活甩到
    一个**脱离本批**的后台运行里，实时仪器对话保持响应。
    用户明确说要「一边…一边跟仪器聊」时，**优先用这条**，而不是把长调研和仪器
    扇在同一批里。
  - **一次最多 {max_parallel} 个 agent。**

**拿不准就问，不要猜。** 指令说的是**真活**、但你分不清是哪一件（它合理地可以指
两件不同的事），或者缺了一个**本该由用户定**的取舍（用哪个样品 / 哪个折中 /
做到什么程度）—— **不要靠掷硬币选一个 agent**。
返回 next_agents `["__end__"]`，同时给 clarify_question（**一个**具体问题）和
clarify_options（2–4 个具体选项）。整条运行会**暂停**等用户答复，然后你拿着
答案再派。

**下面这些不要用 clarify_ 系列：**
  - 打招呼、闲聊、道谢这类对话 —— 直接 __end__ 并写 reason，系统会替你生成回答；
  - 关于 MAST 自身的问题，或任何你从当前上下文里就能回答的；
  - 只是**范围宽**但**明确是一件活**的指令 —— 派下去，让那个 agent 自己收窄。

暂停整条运行是**昂贵**的，而一个在用户看来显而易见的问题，会教他忽略下一个问题。
**拿不准的时候，派下去。**

输出 next_agents（**一个列表**，哪怕只有一个 agent）+ reason（一句话）。
Examples:
  {"next_agents": ["instrument_control", "literature"], "reason": "扫描与文献调研互不依赖，可并行"}
  {"next_agents": ["data_processing"], "reason": "需要先分析已保存的扫描文件"}
  {"next_agents": ["__end__"], "reason": "目标已完成"}
  {"next_agents": ["__end__"], "reason": "需要用户先确定测量目标",
   "clarify_question": "这次是想先看形貌还是先测能谱？两者的针尖状态要求不同。",
   "clarify_options": ["先扫形貌（快，风险低）", "先测 dI/dV（需要更稳的针尖）"]}"""


# Not an f-string (the body carries literal JSON braces the model must copy), so
# the width cap is substituted after the fact.
_ROUTER_PROMPT = _ROUTER_PROMPT.replace("{max_parallel}", str(_MAX_PARALLEL))
# 提示词里写死的 40 与代码里的 60 已经岔开一阵子了（常量 2026-07-30 从 40 改到
# 60，提示词没跟）。让它从常量派生 —— 「改常量而行为不跟着变」是本仓抓过的形状，
# 这里是它的孪生：改了常量而**说给模型听的那句话**不跟着变。
_ROUTER_PROMPT = _ROUTER_PROMPT.replace("{hop_hard_cap}", str(_HOP_HARD_CAP))


# ── Provider-agnostic supervisor routing ──
# LangChain's default with_structured_output(Route) uses response_format
# (json_schema / json_object), which is NOT portable across the 6 providers MAST
# advertises. Full v2 testing on the simulator showed the supervisor routed
# correctly ONLY on the default Kimi; every other provider failed differently:
#   deepseek → "This response_format type is unavailable now"
#   qwen     → "'messages' must contain the word 'json' ... response_format json_object"
#   glm      → returned a `Route(literature, "…")` STRING (right decision, wrong format)
#   sonnet   → "This model does not support assistant message prefill"
#   minimax  → "tool result's tool id … not found" (replayed tool ids in history)
# i.e. switching the orchestrator/agent models off Kimi made the whole multi-agent
# system silently route to __end__ (or 400) without ever dispatching an agent.
# The fix below is provider-portable: (1) function_calling structured output on a
# TEXT-flattened message list (no response_format, no prefill, no replayed tool
# ids); (2) plain-text JSON with a tolerant parser as fallback.
_ROUTER_JSON_HINT = (
    "Respond with ONLY a single json object and nothing else, of the form: "
    '{"next_agents": ["<one or more of research_director|literature|'
    'experiment_design|instrument_control|data_processing|paper_writing|'
    'paper_review>"], '
    '"reason": "<one sentence>"}. '
    'List EVERY agent that can start work now (they run concurrently); use '
    '["__end__"] when the goal is complete. The key MUST be "next_agents" and '
    "its value MUST be a json array."
)


# P2-A (2026-06-11): the hardened provider-agnostic routing machinery moved to
# mast.agents._shared.llm_route (flatten / tolerant parse / two-tier decision)
# so the workflow `llm` node shares ONE implementation with the supervisor.
# These thin wrappers keep the original local names/signatures (and behavior —
# plus a stricter off-enum guard on the structured tier).
from mast.agents._shared.llm_route import (  # noqa: E402
    flatten_messages_for_router as _shared_flatten_messages,
    parse_route_text as _shared_parse_route_text,
    route_decision as _shared_route_decision,
)

_VALID_ROUTE_TARGETS = frozenset(list(_AGENT_NAMES) + ["__end__"])


def _flatten_messages_for_router(messages) -> list[dict]:
    return _shared_flatten_messages(messages)


def _parse_route_text(text: str) -> dict | None:
    parsed, _path = _shared_parse_route_text(
        text, _VALID_ROUTE_TARGETS, field="next_agent")
    return parsed


def _route_decision(supervisor_model, routing_messages) -> dict:
    """Provider-agnostic supervisor routing → {"next_agent", "reason"}.

    SINGLE-target. Kept as-is: it is the 2026-06-08 F5 provider-portability fix
    (function_calling on flattened text + tolerant text-JSON fallback) that made
    routing work on all 6 providers, and it is still the fallback tier for
    :func:`_parallel_route_decision` as well as the API used by existing tests.

    Raises only when BOTH the function_calling and the text-JSON fallback yield
    nothing parseable (the caller then ENDs gracefully)."""
    decision, _path = _shared_route_decision(
        supervisor_model, routing_messages,
        schema=Route, valid_targets=_VALID_ROUTE_TARGETS,
        json_hint=_ROUTER_JSON_HINT, field="next_agent")
    return decision


def _coerce_targets(raw: Any, valid: frozenset[str]) -> list[str]:
    """Normalise whatever the router produced into a clean, ordered target list.

    Accepts a list, a single string, or a comma/space-separated string (models
    improvise). Drops unknown names, de-duplicates (which is also what enforces
    "instrument_control at most once" in a fan-out), and caps the width.
    ``__end__`` is absorbing: if the router names it at all, the run ends —
    "finish" plus "do more work" is incoherent, and ending is the safe read.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.replace(",", " ").split()]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw]
    else:
        return []
    out: list[str] = []
    for name in items:
        if name and name in valid and name not in out:
            out.append(name)
    if "__end__" in out:
        return ["__end__"]
    return out[:_MAX_PARALLEL]


def _clarify_fields(decision: dict) -> dict:
    """The optional ``clarify_*`` pair, normalised — or ``{}`` if not asked for.

    Returning an EMPTY dict when there is no question is the contract: the
    caller spreads this into its result, so a router that never heard of these
    keys produces byte-identical output to before. Junk (a bare string where a
    list belongs, an empty question with options) collapses to ``{}`` too —
    half a question is not a question.
    """
    q = str(decision.get("clarify_question") or "").strip()
    if not q:
        return {}
    raw = decision.get("clarify_options")
    if isinstance(raw, str):
        raw = [raw]
    opts = [str(o).strip() for o in (raw or []) if str(o).strip()]
    return {"clarify_question": q, "clarify_options": opts}


def _parallel_route_decision(supervisor_model, routing_messages) -> dict:
    """Provider-agnostic MULTI-target routing → {"next_agents": [...], "reason"}.

    Same two-tier, provider-portable strategy as :func:`_route_decision`
    (function_calling on a TEXT-flattened list, then plain-text JSON with a
    tolerant parse) — those constraints are load-bearing, see the F5 note above.

    Three tiers of degradation, so a router that cannot do the new shape still
    routes rather than dying:
      1. function_calling with the ParallelRoute schema → a real list;
      2. plain-text JSON: accept ``next_agents`` (list) OR a legacy
         ``next_agent`` (scalar) — some providers stubbornly answer the older
         shape, and one agent is a perfectly good answer;
      3. the proven SINGLE-target path (_route_decision) — never worse than
         the pre-parallel behaviour.

    Raises only when every tier fails (the caller then ENDs gracefully).
    """
    flat = _shared_flatten_messages(routing_messages)

    # Structured function-call output forces tool_choice to the schema tool.
    # Always-on reasoning endpoints that reject forced tool choice skip this
    # tier and use text-JSON directly, avoiding a predictably failing request.
    _skip_t1 = _rejects_forced_tool_choice(_model_id_of(supervisor_model))
    if _skip_t1:
        logger.debug("parallel route: skipping function_calling tier — %s "
                     "rejects forced tool_choice while thinking is on",
                     _model_id_of(supervisor_model))
    try:
        if _skip_t1:
            raise _SkipTier1
        decision = supervisor_model.with_structured_output(
            ParallelRoute, method="function_calling"
        ).invoke(flat)
        if hasattr(decision, "model_dump"):
            decision = decision.model_dump()
        elif hasattr(decision, "dict") and not isinstance(decision, dict):
            decision = decision.dict()
        if isinstance(decision, dict):
            targets = _coerce_targets(decision.get("next_agents"),
                                      _VALID_ROUTE_TARGETS)
            if targets:
                return {"next_agents": targets,
                        "reason": str(decision.get("reason", "") or ""),
                        **_clarify_fields(decision)}
            logger.info("structured parallel route empty/off-enum (%r); falling back",
                        decision.get("next_agents"))
    except _SkipTier1:
        pass                       # deliberate skip; already logged at debug
    except Exception as exc:  # noqa: BLE001
        logger.info("function_calling parallel route failed (%s); trying text JSON",
                    exc)

    # Tier 2 — plain-text JSON, tolerant of the legacy singular key.
    try:
        resp = supervisor_model.invoke(
            flat + [{"role": "user", "content": _ROUTER_JSON_HINT}])
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
        text = str(content)
        dec = _json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _end = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            raw = obj.get("next_agents", obj.get("next_agent"))
            targets = _coerce_targets(raw, _VALID_ROUTE_TARGETS)
            if targets:
                return {"next_agents": targets,
                        "reason": str(obj.get("reason", "") or ""),
                        **_clarify_fields(obj)}
    except Exception as exc:  # noqa: BLE001
        logger.info("text-JSON parallel route failed (%s); trying single-target",
                    exc)

    # Tier 3 — the proven single-target router. Never worse than before parallel.
    single = _route_decision(supervisor_model, routing_messages)
    target = single.get("next_agent") or "__end__"
    return {"next_agents": _coerce_targets(target, _VALID_ROUTE_TARGETS) or ["__end__"],
            "reason": str(single.get("reason", "") or "")}


def _direct_answer(model, messages) -> str:
    """Answer the user's goal directly when no sub-agent is dispatched (F8).

    The supervisor is a pure router; when it ends a run WITHOUT routing to any
    agent (a conversational / meta goal — e.g. "what sub-agents do you
    coordinate?"), the user otherwise sees only the bare "[SUPERVISOR → __end__]"
    route note. This produces a real, concise reply instead. Provider-agnostic
    (flattened plain chat, ends on a user turn so no Claude prefill); returns ""
    on any failure so the caller falls back to the route note (no regression).
    """
    if model is None:
        return ""
    try:
        flat = _flatten_messages_for_router(messages)
        sys_msg = {
            "role": "system",
            "content": (
                "You are MAST Orchestrator, coordinator of 7 sub-agents "
                "(research_director, literature, experiment_design, "
                "instrument_control, data_processing, paper_writing, "
                "paper_review). The user's request "
                "needs no sub-agent dispatch — answer it directly, concisely, in "
                "the user's language."
            ),
        }
        resp = model.invoke([sys_msg] + flat)
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
        # Coalesce None → "" BEFORE str() so a content-less response yields the
        # empty-answer fallback (route note) instead of a literal "None" message.
        text = str(content).strip() if content is not None else ""
        return "" if text == "None" else text
    except Exception as e:  # noqa: BLE001
        logger.info("supervisor _direct_answer failed: %s", e)
        return ""


def _drain_interjections(control_provider) -> "tuple[list, list[str]]":
    """Non-blocking drain of operator interjections from the control provider.

    Returns ``(messages, directed_targets)``:
      * ``messages`` — HumanMessages to inject into the stream so the next agent
        the supervisor routes to actually sees the operator's mid-task input;
      * ``directed_targets`` — agent ids the operator ADDRESSED explicitly (via
        POST /agents/<id>/interject). The supervisor dispatches to these
        DETERMINISTICALLY (a hard @agent route) instead of leaving the LLM router
        to infer intent from the "(指向 X)" text — which it often ignored, the
        "@agent 不管用" report . Empty when the operator broadcast to
        all / addressed the supervisor, or the provider surfaced no targets.

    NB: must stay non-blocking — graph nodes may not sleep / block 钩子
    enforce this on agents/**/graph.py).
    """
    if control_provider is None:
        return [], []
    try:
        ctrl = control_provider() or {}
    except Exception as exc:  # pragma: no cover — provider is best-effort
        logger.debug("control_provider failed: %s", exc)
        return [], []
    texts = ctrl.get("interjections") or []
    out = []
    for t in texts:
        t = str(t).strip()
        if t:
            out.append(HumanMessage(content=f"[用户插话] {t}"))
    targets = [str(a).strip() for a in (ctrl.get("directed_targets") or []) if str(a).strip()]
    if out:
        logger.info("supervisor: injected %d operator interjection(s)%s", len(out),
                    f" (directed → {targets})" if targets else "")
    return out, targets


def _time_now() -> float:
    import time as _t

    return _t.time()


def _ask_operator_node(state: MASTState) -> Command:
    """Pause the whole graph on the supervisor's question, then route on.

    A LangGraph node, not a tool: the supervisor holds no tools, and giving it
    one would mean rebuilding it as an agent — which would take the tiered,
    provider-portable router with it.

    Nothing here runs before ``interrupt()`` except reading state, which is the
    entire reason this is its own node: a resume replays the node from the top,
    and a replay that costs an LLM call (as it would inside ``supervisor_node``)
    can come back with a different decision and never reach the interrupt again,
    silently dropping the answer.

    The answer goes back as a HumanMessage so the next routing pass picks it up
    through ``_trim_for_routing`` like any other operator input — no new channel,
    and the transcript shows the exchange where it happened.
    """
    q = state.get("pending_user_question") or {}
    question = str(q.get("question") or "").strip()
    if not question:
        # Nothing to ask (state cleared, or a resumed old checkpoint) — do not
        # pause on an empty card; hand control straight back.
        return Command(goto="supervisor", update={"pending_user_question": None})
    options = [{"label": str(o)} for o in (q.get("options") or []) if str(o).strip()]
    payload = {
        "kind": "ask_user",
        "question": question,
        "header": "需要你决定",
        "options": options,
        "multi_select": False,
        "allow_custom": True,
        # The supervisor has no fallback plan to fall back TO — it could not
        # route in the first place. Continuing on silence would just re-enter the
        # same coin flip this node exists to avoid.
        "timeout_action": "halt",
        # Self-report: this is a PARENT-graph node, so its namespace is empty and
        # the publisher's owner fallback would file the question under
        # instrument_control — an agent that never asked it.
        "agent_id": "_supervisor",
    }
    try:
        from langgraph.errors import GraphInterrupt
        from langgraph.types import interrupt
    except ImportError:  # pragma: no cover — langgraph is pinned in v2
        answer = None
    else:
        try:
            answer = interrupt(payload)
        except GraphInterrupt:
            raise                       # control flow — NEVER swallow
        except Exception as exc:  # noqa: BLE001 — outside a graph runtime
            logger.debug("ask_operator outside graph runtime: %s", exc)
            answer = None

    picked_list: list[str] = []
    if not isinstance(answer, dict):
        note = ("[用户] （提问通道不可用，未获答复）")
    else:
        picked_list = [str(x) for x in (answer.get("selected") or []) if str(x).strip()]
        picked = "、".join(picked_list)
        custom = str(answer.get("custom_text") or "").strip()
        extra = str(answer.get("note") or "").strip()
        body = "；".join(p for p in (picked, custom, extra) if p) or "（未给出内容）"
        note = f"[用户回答] {body}"

    # ── 目标闸门的两种卡片 (2026-08-27) ──────────────────────────────
    #
    # 普通的澄清问题只需要把答案放回消息流，让下一次路由自己读。目标闸门的两张
    # 卡不一样：它们问的是**结构化**的东西（继续/结束、够了/还不够），所以答案
    # 也要落成结构化的结果，而不是指望路由模型第二次把同一句话读对。
    #
    # ``goal_hold`` 选中的「继续：派 X」直接变成一条 ``routing_hints`` ——
    # 走的是既有的交接提示分发路径（已经过全部熔断），**不再花一次路由调用**。
    # 选「就此结束」则什么都不加：本 run 的 ``goal.asked`` 已置位，下一跳模型
    # 再说 __end__ 时 Gate 2 会大声放行。
    extra_update: dict = {}
    kind = str(q.get("kind") or "")
    if kind == "goal_hold":
        routes = q.get("routes") or {}
        wanted = [routes[o] for o in picked_list if o in routes]
        if wanted:
            extra_update["routing_hints"] = wanted[:1]
            note += f"（→ 继续派 {wanted[0]}）"
    elif kind == "goal_confirm":
        # 「够了吗」——只有明确选中肯定项才算 yes。**读不到 / 没选 / 自定义
        # 文本一律记 no**：这是一个安全计数器方向的字段，把「没答」读成「答应了」
        # 会让一次沉默变成一次结束。
        said_yes = any(o.startswith("够了") for o in picked_list)
        g = state.get("goal")
        g = dict(g) if isinstance(g, dict) else {}
        g["operator_confirmed"] = {"answer": "yes" if said_yes else "no",
                                   "at": _time_now()}
        extra_update["goal"] = g

    return Command(
        goto="supervisor",
        update={
            "messages": [HumanMessage(content=note)],
            "pending_user_question": None,
            "active_agent": "supervisor",
            **extra_update,
        },
    )


def _supervisor_node_factory(
    supervisor_model, control_provider=None, wired_agents=_AGENT_NAMES,
    background_gate=None, is_slow_type=None, budget_probe=None,
    activation_gate=None, ask_operator_enabled=True,
):
    """Returns a supervisor_node bound to the given model.

    ``budget_probe`` (optional) is a zero-arg callable → remaining USD for this
    run, or ``None`` when spend cannot be read. It is what finally connects the
    long-inert ``budget_remaining_usd`` gate to real accounting; the graph stays
    free of any billing import. Absent → the gate behaves exactly as before
    (honours whatever the caller seeded, inert if nothing was).

    ``background_gate`` (optional) is a zero-arg callable → bool. When it is wired
    AND returns True, the conservative auto-background policy peels a whitelisted
    independent analysis agent (literature) off a fan-out that also contains
    instrument_control, signalling the bridge to run it as a detached background
    run (see :func:`_split_auto_background`). None / absent → OFF (default), and
    routing behaviour is byte-for-byte unchanged.

    The model is expected to support `.with_structured_output(Route)`. Real
    ChatAnthropic does; tests can pass a stub or use the no-llm path below.

    ``control_provider`` (optional) is a zero-arg callable returning a dict;
    its ``interjections`` key (list[str]) is drained each step and injected
    into the message stream as operator input. The GUI wires this to the
    Agents-tab interject queue so operator messages reach a running task.

    ``wired_agents`` () is the set of agent node names ACTUALLY added
    to the parent graph (``build()``'s ``include_agents``). The supervisor must
    validate every routing decision against THIS set, not the full
    ``_AGENT_NAMES`` roster: when only a subset of agents is wired, an LLM router
    (or an agent's routing_hint) can name a valid-but-absent agent, and
    ``Command(goto=<absent_node>)`` makes LangGraph raise at runtime
    (``Node '<x>' not found``) — taking the whole task down. Routing to an
    unwired agent now falls through to END instead of crashing. Defaults to the
    whole ``_AGENT_NAMES`` roster so existing callers/tests are unaffected.
    """
    # Normalise to a frozenset of names we may actually dispatch to. An empty /
    # None override would mean "no agents wired" → every route must END.
    valid_targets = frozenset(wired_agents or ())

    def _trim_for_routing(messages, max_tokens: int = 120000):
        """Ephemerally cap the messages fed to the supervisor's routing / direct-
        answer LLM call so a pathologically long group conversation can never
        overflow the router model's window (supervisor-level compaction). Keeps
        the most-recent messages; does NOT mutate the persisted MASTState (which
        is already bounded by the visit_count loop guard). Best-effort."""
        try:
            from langchain_core.messages.utils import (
                count_tokens_approximately, trim_messages,
            )
            msgs = list(messages or [])
            if count_tokens_approximately(msgs) <= max_tokens:
                return msgs
            return trim_messages(
                msgs, max_tokens=max_tokens,
                token_counter=count_tokens_approximately,
                strategy="last", include_system=False, allow_partial=False)
        except Exception:  # noqa: BLE001 — routing must never break on a trim glitch
            msgs = list(messages or [])
            return msgs[-40:] if len(msgs) > 40 else msgs

    def _dispatch(state: MASTState, targets: list[str], new_msgs: list,
                  *, visit_delta: dict[str, int],
                  extra: dict[str, Any] | None = None) -> Command:
        """Route to ONE agent, or fan OUT to several that run concurrently.

        Single target → a plain ``Command(goto=...)``: LangGraph applies the
        update to the channels first, so the agent sees the fresh state. Nothing
        changes for the serial path.

        Several targets → ``Command(goto=[Send(t, payload), ...])``. Send passes
        each node an EXPLICIT input, so the payload must already carry the
        messages this step is adding (spike-verified: a fanned-out node sees the
        Send payload, not the post-update channel state — pass ``state`` alone and
        the branches would never see the operator's interjection or the routing
        note). ``update`` is still returned so the same messages land in the
        durable channel exactly once (add_messages de-dupes by id).

        ``extra`` merges additional channel writes into the same update. It exists
        so control-plane values (the refreshed budget, park bookkeeping) ride the
        ONE choke point rather than being copy-pasted into every return branch —
        the same argument the parent-prune housekeeping below is built on.
        """
        base_msgs = list(state.get("messages", []))
        # Parent-channel housekeeping — the ONE choke point every dispatch passes
        # through, so it cannot be forgotten on a new routing branch.
        drop_ids, prune_note = _plan_parent_prune(base_msgs)
        msgs_out = list(new_msgs)
        if prune_note is not None:
            msgs_out.append(prune_note)
        # Products from runs that were WOKEN while this thread was idle. Drained here
        # for exactly the reason the prune above is here: this is the one place every
        # dispatch passes through. Merged by version rather than last-wins — see
        # _artifact_is_newer.
        returned, return_notes = _drain_return_inbox(state)
        for text in return_notes:
            msgs_out.append(AIMessage(content=text))
        update: dict[str, Any] = {
            "messages": msgs_out + [RemoveMessage(id=i) for i in drop_ids],
            "active_agent": " ‖ ".join(targets),
            "routing_hints": None,     # None CLEARS the channel (merge_routing_hints)
            "visit_count": visit_delta,
            **returned,
        }
        if extra:
            update.update(extra)
        # Hop-distribution telemetry (2026-07-30). The cheapest possible probe for
        # a number nobody has: the REAL per-run hop distribution. `logging/v2`'s
        # `trajectory_steps` would be the proper home, but its schema is built and
        # `begin_trajectory` is called and the table still holds 0 rows (while
        # `actions` in the same DB has 183) — that path has never been proven to
        # work end to end. One log line needs no schema at all, and this function
        # is the single place every dispatch passes through, so no future routing
        # branch can be added that skips it. The thresholds above were set by
        # inference from the billing ledger; in a couple of weeks these lines
        # replace inference with measurement.
        try:
            _t, _wa, _wn, _sup = _guard_dimensions(state.get("visit_count") or {})
            logger.info(
                "supervisor hops: total=%d worst_agent=%s:%d supervisor=%d "
                "delta=%s → %s",
                _t, _wn or "-", _wa, _sup, visit_delta, targets,
            )
        except Exception:  # noqa: BLE001 — telemetry never breaks routing
            pass
        if len(targets) == 1:
            return Command(goto=targets[0], update=update)
        # A Send payload is a LITERAL message list, so the RemoveMessage markers
        # must not travel in it — apply the prune to the list instead.
        kept = [m for m in base_msgs if getattr(m, "id", None) not in drop_ids]
        # A Send payload is EXPLICIT — a fanned-out node sees this dict, not the
        # post-update channel state — so the freshly-returned products have to be
        # overlaid here too, or the branches would run against the pre-delivery world
        # while the durable channel already held the new pointers.
        payload = {**state, **returned, "messages": kept + msgs_out}
        logger.info("supervisor: parallel dispatch → %s", targets)
        return Command(goto=[Send(t, payload) for t in targets], update=update)

    def supervisor_node(state: MASTState) -> Command:
        # ── Loop / budget guards ──
        # Runs on EVERY inter-agent hop: all handoffs (incl. sibling→sibling)
        # route through this node now, so the guard cannot be bypassed
        #. visit_count accumulates per agent via the
        # sum_int_dicts reducer, so per_agent_max reflects the real running
        # total (). NB: every visit_count write below is a +1 DELTA,
        # NOT an absolute snapshot — the adding reducer would otherwise
        # double-count.
        vc = state.get("visit_count") or {}
        # ONE reader of visit_count for all three dimensions, so the gate, the
        # soft warning and the dispatch log cannot disagree about whether the
        # `supervisor` key is an agent (it is not — see the constants).
        total, worst_agent, worst_agent_name, sup_hops = _guard_dimensions(vc)
        # NB: the literal `40` that used to be here was the actual gate while
        # `_HOP_HARD_CAP` fed only the soft warning — so retuning the constant
        # changed nothing and the tree read as if it did. Fixed 2026-07-30.
        if total > _HOP_HARD_CAP:
            return Command(
                goto=END,
                update={
                    "messages": [AIMessage(content=(
                        f"[SUPERVISOR] Loop guard tripped: 总跳数 {total} > "
                        f"{_HOP_HARD_CAP}。"))],
                    "active_agent": "__end__",
                    "routing_hints": None,
                },
            )
        if worst_agent > _AGENT_HARD_CAP:
            return Command(
                goto=END,
                update={
                    "messages": [AIMessage(content=(
                        f"[SUPERVISOR] Loop guard tripped: {worst_agent_name} 被进入 "
                        f"{worst_agent} 次 > {_AGENT_HARD_CAP}（单个智能体上限）。"
                        "总跳数还没超,但反复进同一个智能体是返工死循环的形状。"))],
                    "active_agent": "__end__",
                    "routing_hints": None,
                },
            )
        if sup_hops > _SUPERVISOR_HARD_CAP:
            return Command(
                goto=END,
                update={
                    "messages": [AIMessage(content=(
                        f"[SUPERVISOR] Loop guard tripped: 路由次数 {sup_hops} > "
                        f"{_SUPERVISOR_HARD_CAP}（编排器自身上限 —— 乒乓探测器）。"))],
                    "active_agent": "__end__",
                    "routing_hints": None,
                },
            )
        # Budget hard-gate (). Was CALLER-SEEDED and, as an audit
        # found on 2026-07-30, seeded by NOBODY — inert for its whole life while
        # one measured run spent $30.66 (see mast.billing.run_meter). It is now
        # refreshed here from ``budget_probe``, a zero-arg callable the host wires
        # in; the graph itself still does no billing and imports nothing from the
        # billing layer.
        #
        # Ordering matters: refresh BEFORE comparing, so the gate acts on this
        # hop's spend rather than a value one hop stale. ``None`` from the probe
        # (no ledger / unreadable) leaves whatever the caller seeded — a billing
        # hiccup must not fabricate a budget, in either direction.
        budget = state.get("budget_remaining_usd")
        if budget_probe is not None:
            try:
                fresh = budget_probe()
            except Exception as exc:  # noqa: BLE001 — routing never dies over billing
                logger.debug("budget probe failed: %s", exc)
                fresh = None
            if fresh is not None:
                budget = float(fresh)
        if budget is not None and budget <= 0:
            return Command(
                goto=END,
                update={
                    # "Budget exhausted" stays in the text verbatim: it is the
                    # marker tests and log greps key on, exactly like "Loop guard
                    # tripped" above. The Chinese half is for the operator.
                    "messages": [AIMessage(content=(
                        "[SUPERVISOR] Budget exhausted — 预算已用尽,终止本次任务。"
                        "（本 run 的上限由 orchestrator_run_budget_usd 设定;"
                        "计量口径见 mast/billing/run_meter.py。）"))],
                    "active_agent": "__end__",
                    "routing_hints": None,
                    "budget_remaining_usd": 0.0,
                },
            )

        # The refreshed budget rides every dispatch through ``extra`` (the one
        # choke point) so the value in state — and therefore in the UI and on the
        # next hop — is this hop's, not one hop stale. Empty when the probe could
        # not read spend, which leaves the channel untouched rather than writing a
        # made-up number.
        control_extra: dict[str, Any] = {}
        if budget is not None:
            control_extra["budget_remaining_usd"] = budget

        # ── Operator interjections (non-blocking drain) ──
        # Injected BEFORE routing so the supervisor's decision accounts for
        # them, and persisted into state so the chosen agent sees them too.
        # ``directed_targets`` are agents the operator ADDRESSED explicitly
        # (@agent) — dispatched deterministically below, not left to the router.
        injected, directed_targets = _drain_interjections(control_provider)

        # ── Budget soft-limit nudge (review MEDIUM #7) ──
        # Hard gates above already END at <=0 / loop-cap; this is a NON-blocking,
        # fire-once warning at ~80% consumption so the agent converges before the
        # hard gate aborts it mid-thought. Prepended to `injected` so it rides
        # the existing message-injection path on EVERY non-hard-gate return
        # branch (hint dispatch, no-model END, router error, normal route) and is
        # also visible to the LLM router's decision this same step. Idempotency
        # comes from a unique marker scanned in the message stream — no new
        # MASTState field / reducer (visit_count + sum_int_dicts untouched).
        soft = _budget_softlimit_warning(
            state, total, worst_agent=worst_agent, worst_agent_name=worst_agent_name)
        if soft is not None:
            injected = [soft] + injected
            logger.info(
                "supervisor: injected budget soft-limit warning (total=%d)", total,
            )

        # ── 目标终止判据 (2026-08-27) ───────────────────────────────────
        # 没给 ``done_when`` ⇒ ``_goal_verdict`` 回 None ⇒ 下面每一处整段跳过，
        # 本节点逐字节等于这道闸出现之前。今天所有调用方都是这一支。
        #
        # 顺序：先抓基线再求值。基线是「目标被设定的那一刻」，不是「进程启动
        # 的那一刻」—— 续接的群聊线程里躺着上一个任务的 literature_report，
        # 实验文件夹里躺着上周的草稿；不抓基线的话目标会在第一跳就「达成」，
        # 那是「太早停」换了个方式复现。
        _goal = _goal_dict(state)
        _goal_out: dict = {}
        if _goal.get("done_when") and _goal.get("baseline") is None:
            _base = None
            try:
                from mast.agents._shared.artifact_channel import _present_in_state
                from mast.goals.sources import snapshot_baseline

                _base = snapshot_baseline(
                    state_present=lambda f: _present_in_state(state, f))
            except Exception as exc:  # noqa: BLE001 — 抓不到基线不许拦住路由
                logger.debug("goal baseline snapshot failed: %s", exc)
                _base = None
            if _base is not None:
                # **抓不到就留 None，下一跳再抓。** 写一个 `{}` 进去等于说
                # 「目标设定那一刻世界是空的」，于是任何一份历史产物都算「新的」
                # —— 一次产物索引读失败就把目标变成了永远已达成。
                _goal_out["baseline"] = _base
                _goal = {**_goal, **_goal_out}
                # 本跳后续（含判据求值与 Send 载荷）按新基线走，不等下一跳。
                state = {**state, "goal": _goal}
        goal_verdict = _goal_verdict(state, ask_enabled=ask_operator_enabled)
        if goal_verdict is not None:
            _goal_out["last_verdict"] = goal_verdict.as_dict()
        if _goal_out:
            control_extra = {**control_extra, "goal": {**_goal, **_goal_out}}

        # ── Honour the agents' routing hints (; plural 2026-07-11) ──
        # A handoff toward a sibling agent appends its intended next hop to
        # state["routing_hints"] and routes here so the guards above run. Valid
        # hints dispatch straight through (clearing the channel) WITHOUT spending
        # an LLM router call — the agents already decided, and the guards above
        # have vetted the hop. Several hints (from several branches that just
        # finished in parallel) fan out again: agents each naming a next hop IS a
        # parallel request. (Operator interjections are still surfaced into the
        # message stream so the next agents see them.)
        #
        # Validate against the WIRED set (), not the full roster: a hint
        # toward an agent that wasn't included in this build would route to a
        # non-existent node and crash. Unwired hints are dropped; if nothing valid
        # remains we fall through to the LLM router (the stale channel is cleared
        # on every branch below).
        agent_hints = [h for h in (state.get("routing_hints") or []) if h in valid_targets]
        # Operator @agent directions dispatch DETERMINISTICALLY (a hard route),
        # taking precedence over the LLM router — the whole point of "@agent" is
        # that it reaches THAT agent, not one the router guessed . They
        # ride the same guarded hint-dispatch path as agent handoffs. Order:
        # operator's explicit targets first, then agent-requested hops; deduped +
        # capped (dedupe is also what keeps instrument_control to at most one).
        op_hints = [h for h in directed_targets if h in valid_targets]
        hints: list[str] = []
        for h in op_hints + agent_hints:
            if h not in hints:
                hints.append(h)
        hints = hints[:_MAX_PARALLEL]

        # ── Gate 1：判据满足 ⇒ 确定性结束，**不问模型** ─────────────────
        # 优先级链（固定，测试钉着）：硬熔断 > 预算闸 > 用户 @agent >
        # **这里** > agent 交接提示 > 路由模型 > Gate 2（hold）。
        #
        # 输给用户的 @agent：那是一次显式指令，「@agent 就要到达那个 agent」
        # 是既有规则，目标达成也不该改写它。
        # 赢过 agent 的交接提示：用户的判据说停就停 —— 提示词里那条「用户
        # 没要求的阶段不要自己加」说的是同一件事，只是这一次由代码执行。
        if goal_verdict is not None and goal_verdict.is_done and not op_hints:
            _goal_record(goal_verdict, "done", state)
            logger.info("supervisor: goal done (%d/%d) → END",
                        goal_verdict.satisfied, goal_verdict.total)
            return Command(
                goto=END,
                update={
                    "messages": injected + [_goal_note(goal_verdict, "done")],
                    "active_agent": "__end__",
                    "routing_hints": None,
                    "visit_count": {"supervisor": 1},
                    **({"goal": control_extra["goal"]}
                       if "goal" in control_extra else {}),
                },
            )

        if hints:
            src = "operator" if op_hints else "agent"
            hint_msgs = list(injected)
            # ── Activation gating on AGENT-requested hops only (2026-07-30) ──
            # An operator's explicit @agent is NEVER parked. That is the same rule
            # auto-background already follows: "@agent" means it reaches THAT agent,
            # not one something else decided for them . If the operator
            # wants the reviewer run with no draft on disk, that is their call to
            # make and their result to read — and the agent will say so honestly.
            #
            # An agent's own next-hop hint IS gated: a handing-off agent is guessing
            # about a sibling's inputs, which is exactly the guess readiness knows
            # the answer to.
            gated = [h for h in hints if h not in op_hints]
            allowed, parked, park_delta = _plan_activation(
                state, gated, ask_model=supervisor_model,
                instruction=_goal_text(state), gate=activation_gate)
            if parked:
                hint_msgs.append(_park_note(parked))
                logger.info("supervisor: parked agent-requested %s",
                            [p["agent"] for p in parked])
            if park_delta:
                control_extra = {**control_extra, "pending_activations": park_delta}
            # Preserve the original order (operator targets first) rather than
            # concatenating, so a fan-out's agent order stays stable and testable.
            keep = set(op_hints) | set(allowed)
            hints = [h for h in hints if h in keep]
            if not hints:
                waiting = "、".join(p["agent"] for p in parked) or "（无）"
                return Command(
                    goto=END,
                    update={
                        "messages": hint_msgs + [AIMessage(content=(
                            "[SUPERVISOR → __end__] 交接过来的下一步无法派发:"
                            f"{waiting} 在等上游资料。这既不是失败也不是完成 —— "
                            "资料到位后会重新询问它是否醒来。"))],
                        "active_agent": "__end__",
                        "routing_hints": None,
                        "visit_count": {"supervisor": 1},
                        **({"pending_activations": park_delta} if park_delta else {}),
                    },
                )
            note = AIMessage(content=f"[SUPERVISOR → {' ‖ '.join(hints)}] (per {src} request)")
            # supervisor's own hop is +1; each AGENT-requested hint was already
            # counted by its handoff, but an OPERATOR-directed target was counted
            # by nobody — give it its own +1 so the loop guard stays honest.
            visit_delta: dict[str, int] = {"supervisor": 1}
            for h in op_hints:
                if h in hints:
                    visit_delta[h] = visit_delta.get(h, 0) + 1
            return _dispatch(state, hints, hint_msgs + [note],
                             visit_delta=visit_delta, extra=control_extra)

        # ── Decide next agent via structured-output classifier ──
        if supervisor_model is None:
            # Test path: no LLM, just bounce to END once. Still persist any
            # injected interjections so tests can assert they landed. The stale
            # hint channel is CLEARED here too: reaching this branch means the
            # hints above were all unwired/invalid, and leaving them in state
            # would resurrect them on the next supervisor visit.
            return Command(
                goto=END,
                update={
                    "messages": injected + [AIMessage(content="[SUPERVISOR] No model configured; ending.")],
                    "active_agent": "__end__",
                    "routing_hints": None,
                    "visit_count": {"supervisor": 1},
                },
            )

        try:
            # What already EXISTS, handed to the router (2026-07-30).
            #
            # Until now the scheduler was the one role in the system that could not
            # see the products: `CONSUMES` covered the six agents and
            # UpstreamArtifactMiddleware mounts on their subgraphs, while this node
            # is a bare function outside the middleware stack. So the only evidence
            # the router had about what had been produced was a one-sentence
            # handoff `reason` in the message stream — which is exactly why its
            # "每个阶段只走一遍" rule had to be a behavioural guess rather than a
            # check against reality.
            #
            # No tool list is passed (this node holds no tools), so the block
            # renders summaries and file paths and NO readback instructions — the
            # renderer refuses to name a tool the reader cannot call.
            env_block = _render_upstream(state, "supervisor", available_tools=set())
            # Which agents are currently PARKED (2026-07-30). Same argument as the
            # product block above, one step further: without it the router re-picks a
            # parked agent every hop, or reads its missing output as "that stage is
            # finished" and ends early. A parked agent is neither done nor failed, and
            # nothing else in the router's context can express that.
            park_block = _parked_context_line(state)
            # 目标判据块（2026-08-27）。与上面两块同一性质：**事实上下文**，
            # 不是说服。判断仍然由代码做 —— 模型看懂看不懂都不改变结论 ——
            # 但让它看见「代码这一刻认为还差什么」，可以省掉一次本来会被
            # Gate 2 拦下来再问人的往返。
            goal_block = ""
            if goal_verdict is not None:
                try:
                    from mast.goals import render_goal_block

                    goal_block = render_goal_block(goal_verdict,
                                                   goal_text=_goal_text(state))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("goal block render failed: %s", exc)
            routing_messages = (
                [{"role": "system", "content": _resolve_prompt(
                    "orchestrator.router.system", _ROUTER_PROMPT)}]
                + ([{"role": "system", "content": env_block}] if env_block else [])
                + ([{"role": "system", "content": park_block}] if park_block else [])
                + ([{"role": "system", "content": goal_block}] if goal_block else [])
                + _trim_for_routing(state.get("messages", []))
                + injected
            )
            # Provider-agnostic routing: the default
            # with_structured_output(Route) only worked on Kimi;
            # _parallel_route_decision preserves that hard-won portability
            # (function_calling on flattened text → tolerant text-JSON → the
            # proven single-target router) while allowing a LIST of agents to be
            # dispatched concurrently. It always returns a normalised dict or
            # raises (→ graceful END below); no AttributeError can escape it.
            decision = _parallel_route_decision(supervisor_model, routing_messages)
            targets = _coerce_targets(decision.get("next_agents"), valid_targets)
            reason = decision.get("reason", "") or ""
        except Exception as e:
            logger.warning("supervisor routing failed: %s", e)
            return Command(
                goto=END,
                update={
                    "messages": injected + [AIMessage(content=f"[SUPERVISOR] Routing error: {e}")],
                    "active_agent": "__end__",
                    "routing_hints": None,
                    "visit_count": {"supervisor": 1},
                },
            )

        # _coerce_targets already dropped anything outside the WIRED set (review
        # #53: a valid agent NAME that wasn't included in this build is not a real
        # node — dispatching to it crashes LangGraph), de-duplicated (which is
        # what keeps instrument_control from appearing twice in one fan-out), and
        # absorbed "__end__". An empty list = the router named nothing
        # dispatchable → END, exactly as an explicit "__end__" would.
        if not targets or targets == ["__end__"]:
            # ASK BEFORE GUESSING (2026-08-01). The router can now say "this is
            # too ambiguous to route" instead of picking an agent on a coin flip
            # or ending with a plausible-sounding direct answer to a question it
            # did not actually understand. Checked BEFORE _direct_answer, which
            # is the specific behaviour being replaced: answering anyway.
            #
            # Handed to a separate node rather than interrupted here — this node
            # replays from the top on resume and would re-run its LLM route
            # call, possibly not reaching the interrupt the second time and
            # dropping the operator's answer on the floor.
            _clarify_q = str(decision.get("clarify_question") or "").strip()
            if _clarify_q and ask_operator_enabled:
                logger.info("supervisor: asking the operator to clarify — %s", _clarify_q)
                return Command(
                    goto="ask_operator",
                    update={
                        "messages": injected + [AIMessage(
                            content=f"[SUPERVISOR → 用户] {_clarify_q}")],
                        "pending_user_question": {
                            "question": _clarify_q,
                            "options": list(decision.get("clarify_options") or []),
                        },
                        "active_agent": "ask_operator",
                        "routing_hints": None,
                        "visit_count": {"supervisor": 1},
                    },
                )
            # ── Gate 2：判据没满足就不许静默结束 (2026-08-27) ───────────
            # 这里是 fail_silent 的正面：模型判断「做完了」，而代码知道还没有。
            #
            # **不 nudge、不重问模型。** 模型在这一跳已经看过判据块（上面
            # ``goal_block``）才说的 __end__；拿同样的证据再问一次，只是多花一
            # 跳一次调用，而且那正是「用提示词说服模型」——本仓记过四次的反面
            # 教材。转给用户是确定性的、replay 安全的，而且天然有界。
            #
            # 三种放行：
            #   * 有东西卡着（park / 待答问题）—— 那既不是完成也不是失败，
            #     逼着继续派发只会得到同一个结果；
            #   * 本 run 已经问过一次 —— 上限，见 ``_goal_already_asked``；
            #   * 问不出去（没有 checkpointer，interrupt 无处可停）。
            # 三种都**大声**放行：一句 [SUPERVISOR:GOAL] + 一条诊断台账。
            if (goal_verdict is not None
                    and goal_verdict.verdict == "not_done"):
                _blocked = _goal_blocked(state)
                if _blocked:
                    _goal_record(goal_verdict, "blocked_ok", state)
                    injected = injected + [_goal_note(
                        goal_verdict, "blocked_ok",
                        extra=f"（在等：{_blocked}）")]
                elif not ask_operator_enabled:
                    _goal_record(goal_verdict, "no_ask", state)
                    injected = injected + [_goal_note(goal_verdict, "no_ask")]
                elif _goal_already_asked(state):
                    _goal_record(goal_verdict, "ended_unmet", state)
                    injected = injected + [_goal_note(goal_verdict, "ended_unmet")]
                else:
                    _q = _goal_hold_question(goal_verdict, state, valid_targets)
                    _goal_record(goal_verdict, "hold", state)
                    logger.info("supervisor: goal not met (%d/%d) → 问用户",
                                goal_verdict.satisfied, goal_verdict.total)
                    _g_now = {**_goal_dict(state),
                              **(control_extra.get("goal") or {}),
                              "asked": True}
                    return Command(
                        goto="ask_operator",
                        update={
                            "messages": injected + [
                                _goal_note(goal_verdict, "hold"),
                                AIMessage(content=(
                                    f"{_GOAL_ASKED_MARKER} {_q['question']}")),
                            ],
                            "pending_user_question": _q,
                            "goal": _g_now,
                            "active_agent": "ask_operator",
                            "routing_hints": None,
                            "visit_count": {"supervisor": 1},
                        },
                    )
            elif goal_verdict is not None and goal_verdict.verdict == "unknown":
                # 判不了 ⇒ **保持今天的行为**（模型说了算），但把「读不到」
                # 这件事说出来。读不到被当成答案是本仓一天犯四次的错。
                _goal_record(goal_verdict, "unknown", state)
                injected = injected + [_goal_note(goal_verdict, "unknown")]

            final_msgs = injected + [
                AIMessage(content=f"[SUPERVISOR → __end__] {reason}")]
            # F8 (2026-06-08): ending WITHOUT having dispatched any agent means
            # the router judged this a conversational/meta goal needing no
            # sub-agent. Synthesize a real answer so the user doesn't just get the
            # route note. Guarded (only when no agent ran — agents otherwise
            # produce the answer) and fail-safe (_direct_answer "" → route note).
            prior_agents = [k for k in (state.get("visit_count") or {})
                            if k != "supervisor"]
            if not prior_agents:
                answer = _direct_answer(
                    supervisor_model,
                    _trim_for_routing(state.get("messages", [])) + injected)
                if answer:
                    final_msgs = final_msgs + [AIMessage(content=answer)]
            return Command(
                goto=END,
                update={
                    "messages": final_msgs,
                    "active_agent": "__end__",
                    "routing_hints": None,
                    "visit_count": {"supervisor": 1},
                },
            )

        # Conservative auto-background: peel a whitelisted
        # INDEPENDENT analysis agent (literature) off the fan-out to a DETACHED
        # background run when it's paired with instrument_control, so the fast
        # instrument turn isn't barrier-blocked behind a long survey. Gate OFF
        # (default) → fg_targets == targets, bg_agents == [] → ZERO behaviour
        # change. The supervisor only SIGNALS via a marker message; the bridge
        # (routes/orchestrator) spawns the background run on seeing it. This runs
        # ONLY on the LLM-router branch — an operator @agent direction or an agent
        # handoff hint (handled earlier) is an EXPLICIT choice and is never
        # auto-backgrounded, which is also the operator's override.
        fg_targets, bg_agents = _split_auto_background(targets, background_gate,
                                                       is_slow_type)
        # ── Activation gating (2026-07-30) ──
        # Applied to ROUTER-chosen targets: the router picked from the goal, and
        # whether the upstream products those targets need actually exist is a fact
        # it may not have. Runs BEFORE visit_delta is built, so a parked agent does
        # NOT spend a hop — it never ran, and charging it would let repeated
        # re-selection exhaust the loop budget on work that never happened.
        fg_targets, parked, park_delta = _plan_activation(
            state, fg_targets, ask_model=supervisor_model,
            instruction=_goal_text(state), gate=activation_gate)
        note_msgs = list(injected)
        if parked:
            note_msgs.append(_park_note(parked))
            logger.info("supervisor: parked %s (dispatching %s)",
                        [p["agent"] for p in parked], fg_targets)
        if park_delta:
            control_extra["pending_activations"] = park_delta
        if bg_agents:
            note_msgs.append(AIMessage(
                content=f"{_AUTO_BG_MARKER} {','.join(bg_agents)} :: {reason}"))
            logger.info("supervisor: auto-background %s (foreground=%s)",
                        bg_agents, fg_targets)
        if not fg_targets:
            # Every FOREGROUND target was parked. This ends the run — but not
            # silently: the park note above is in the transcript, the entries are in
            # state (and from W3 on the disk board + the UI's 待唤醒 row). "Nothing
            # happened" and "it hung" must never look the same, so the END message
            # names who is waiting and for what.
            #
            # Any auto-background marker was already appended, so a detached run the
            # bridge was told to spawn still gets spawned; only the foreground has
            # nothing left to do. Reaching _dispatch with an empty target list would
            # instead build an empty Send list — a route to nowhere.
            waiting = "、".join(p["agent"] for p in parked) or "（无）"
            return Command(
                goto=END,
                update={
                    "messages": note_msgs + [AIMessage(content=(
                        "[SUPERVISOR → __end__] 本轮没有可以派发的智能体:"
                        f"{waiting} 都在等上游资料。这既不是失败也不是完成 —— "
                        "资料到位后它们会被重新询问是否醒来。"))],
                    "active_agent": "__end__",
                    "routing_hints": None,
                    "visit_count": {"supervisor": 1},
                    **({"pending_activations": park_delta} if park_delta else {}),
                },
            )
        note = AIMessage(content=f"[SUPERVISOR → {' ‖ '.join(fg_targets)}] {reason}")
        note_msgs.append(note)
        # +1 DELTA for the supervisor's own visit, plus +1 for EACH FOREGROUND
        # agent — the backgrounded agent runs in its OWN detached run with its own
        # loop guards, so it is NOT counted against this thread's hop budget. The
        # sum_int_dicts reducer accumulates the deltas ().
        visit_delta: dict[str, int] = {"supervisor": 1}
        for t in fg_targets:
            visit_delta[t] = visit_delta.get(t, 0) + 1
        return _dispatch(state, fg_targets, note_msgs, visit_delta=visit_delta,
                         extra=control_extra)

    return supervisor_node


def build(
    buf=None,
    *,
    supervisor_model: Any | None = None,
    agent_model_overrides: dict[str, Any] | None = None,
    checkpointer: Any = None,
    include_agents: tuple[str, ...] = _AGENT_NAMES,
    context_provider=None,
    control_provider=None,
    enable_hitl: bool = True,
    memory_tools: list | None = None,
    instrument_extra_tools: list | None = None,
    experiment_design_extra_tools: list | None = None,
    instrument_post_hook=None,
    get_state: "Callable[[], HardwareState] | None" = None,
    get_mode: "Callable[[], Any] | None" = None,
    safety_limits: "SafetyLimits | None" = None,
    override_registry: "ConfigOverrideRegistry | None" = None,
    # 群聊 IC 的技能注册表。**不传 = 回落 discover()**,那只 walk
    # ``skills.builtins`` + ``skills.composite`` 两个包 —— 于是运行期注册的东西
    # (builder 页做的声明式 composite、custom .py、覆盖层、agent 自己铸的技能)
    # 群聊 IC 一个都看不到,而私聊看得到。同一个 agent 两个入口两套能力,谁也不会
    # 收到报错。2026-08-25 补上。
    instrument_registry: Any | None = None,
    recorder=None,
    safety_recorder=None,
    turn_recorder=None,
    agent_extra_middleware: list | None = None,
    agent_call_limits: dict | None = None,
    background_gate=None,
    is_slow_type=None,
    budget_probe=None,
    activation_gate=None,
):
    """Build the top-level Orchestrator graph with 7 agent sub-graphs.

    ``agent_extra_middleware``: optional shared middleware list (context
    compaction + cross-conversation memory recall) attached to EVERY wired agent
    via its build() ``extra_middleware`` param — so 群聊 gets the same modern
    context management as the private chat (私聊).

    Args:
        buf:                       BufferService passed to each agent's build()
        supervisor_model:          ChatAnthropic / fake; default Sonnet 4.6.
                                   Pass None to skip routing (tests).
        agent_model_overrides:     {agent_name: model} for per-agent overrides
        checkpointer:              LangGraph checkpointer
        include_agents:            subset of agent names to wire in (default: all 7)
        context_provider:          required by instrument_control if included
        control_provider:          optional zero-arg callable → dict; its
                                   ``interjections`` key (list[str]) is drained
                                   each supervisor step and injected as operator
                                   input. Non-blocking by contract.
        memory_tools:              optional list of shared persistent-memory
                                   tools (from agents._shared.memory_tools) to
                                   attach to EVERY wired agent via its build()
                                   ``extra_tools`` param. This is how the 6 agents
                                   get cross-session read/write memory without any
                                   agent importing a sibling — the tools come from
                                   the shared module and are passed in here.
        get_state:                 optional zero-arg callable returning the live
                                   HardwareState. THREADED DOWN to the
                                   instrument_control agent's build() so
                                   SafetyGateMiddleware's Layer-2 state-precondition
                                   check fires on the autonomous agent path (e.g.
                                   a skill flagged ``z_controller_off`` is blocked
                                   while the controller is ON; the tip must be
                                   withdrawn before a coarse approach). WITHOUT
                                   this, the agent path saw no live state and the
                                   precondition layer was a silent no-op while the
                                   manual executor path enforced it — the bug this
                                   wiring fixes. The GUI injects ``state.snapshot``;
                                   the CLI a fake-state snapshot. None → IC build
                                   leaves the precondition layer inert (offline
                                   tests).
        safety_limits:             optional SafetyLimits passed down to the
                                   instrument_control agent's SafetyGateMiddleware
                                   global-bounds layer. Merged with admin overrides
                                   from ``override_registry`` (below) before being
                                   handed down. None → code defaults.
        override_registry:         optional ConfigOverrideRegistry (admin JSON
                                   override layer). Used HERE to merge admin
                                   SafetyLimits overrides into ``safety_limits``
                                   before threading the EFFECTIVE limits to the
                                   instrument_control agent — instrument_control's
                                   build() exposes no registry knob, so without
                                   this an admin-tightened bias/current/Z/scan cap
                                   took effect only on the manual executor path and
                                   was silently dropped on the multi-agent path
                                   (the review finding). None → no admin merge.

    Returns:
        CompiledStateGraph — top-level orchestrator.
    """
    overrides = agent_model_overrides or {}
    _cl = agent_call_limits or {}  # per-agent call-limit overrides (durable chat/group turns off the thread cap)
    if supervisor_model is None and not overrides.get("__supervisor_no_model__"):
        try:
            # NB: the supervisor routes via with_structured_output(Route), which
            # on Anthropic forces a tool call — and Claude extended thinking is
            # INCOMPATIBLE with forced tool_choice (the call errors and the
            # supervisor silently routes to END → task dies; review 2.1.13 #4).
            # So we do NOT enable extended thinking here. Routing is a cheap
            # classification, not reasoning-heavy; the default Kimi orchestrator
            # model already reasons intrinsically at full strength regardless.
            supervisor_model = make_chat_model(
                "orchestrator",
                max_tokens=2048,
                temperature=0.1,
            )
        except Exception:
            # Tests may run without API key — keep supervisor_model = None which
            # triggers the no-model END path inside supervisor_node.
            supervisor_model = None

    # ``agent_extra_middleware`` may be a plain LIST (one shared stack, the
    # historical shape) or a CALLABLE ``agent_id -> list``. The callable exists
    # because compaction has to be sized per agent: one instance built for the
    # orchestrator's model was handed to all six, so an agent overridden onto a
    # smaller-window model was sized for someone else's window. Callable form is
    # what core/runtime passes; the list form is kept for tests and any caller
    # that genuinely wants one shared stack.
    def _extra_mw(agent_name: str):
        if callable(agent_extra_middleware):
            try:
                return agent_extra_middleware(agent_name)
            except Exception as exc:  # noqa: BLE001 — never fail a build over context mw
                logger.warning("agent_extra_middleware(%s) failed: %s", agent_name, exc)
                return []
        return agent_extra_middleware

    g: StateGraph = StateGraph(MASTState)
    # Pass include_agents so the supervisor only routes to nodes that are
    # actually wired (). Routing to a valid-but-unwired agent would
    # otherwise crash LangGraph with a missing-node error.
    # Asking the operator to disambiguate needs somewhere to pause. Without a
    # checkpointer the graph cannot be resumed at all, so the supervisor must
    # keep its old behaviour (answer / end) rather than route to a node whose
    # interrupt nothing could ever answer.
    _ask_operator_enabled = checkpointer is not None
    g.add_node(
        "supervisor",
        _supervisor_node_factory(
            supervisor_model, control_provider, wired_agents=tuple(include_agents),
            background_gate=background_gate, is_slow_type=is_slow_type,
            budget_probe=budget_probe, activation_gate=activation_gate,
            ask_operator_enabled=_ask_operator_enabled,
        ),
    )
    if _ask_operator_enabled:
        g.add_node("ask_operator", _ask_operator_node)

    # ── conduct 工具：**每个** agent 都拿到 ─────────────────────────────
    #
    # 2026-08-20 的权限裁决：把关不靠「不给工具」，靠服务端的包络与自主度策略
    # （见 mast/agents/_shared/conduct_tools.py 的模块 docstring）。按角色裁剪
    # 工具面这件事，维护成本随能力数 × 角色数增长，而且挡不住真正该挡的东西
    # —— 一个模型换条路照样能表达同一个意图，被挡住的只是「它想不到」。
    #
    # 署名按 agent 分：conduct 的审计流里因此答得出「这一步是谁点的头」。
    def _conduct_tools(agent_name: str) -> list:
        try:
            from mast.agents._shared.conduct_tools import make_conduct_tools

            return make_conduct_tools(agent_name)
        except Exception as exc:  # noqa: BLE001
            # 起不来不该拦住整张图 —— conduct 引擎默认就是关着的，
            # 而这些工具在引擎关着时仍然能读，所以这里失败是真的异常。
            logger.warning("conduct 工具建不出来（本次这个 agent 没有它们）: %s", exc)
            return []

    # 技能市场：搜全集 + 推荐订阅。订阅列表把工具面收窄到用户在用的那些，
    # 于是 agent 从此看不见自己没有的能力 —— 这一族补的就是那个洞。
    # **只能推荐，不能改自己的工具面**（见 market_tools 模块 docstring）。
    def _market_tools(agent_name: str) -> list:
        try:
            from mast.agents._shared.market_tools import make_market_tools

            return make_market_tools(agent_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("市场工具建不出来（本次这个 agent 没有它们）: %s", exc)
            return []

    def _shared(agent_name: str) -> list:
        return ((memory_tools or []) + _conduct_tools(agent_name)
                + _market_tools(agent_name))

    # Lazy-import each agent's build() — keeps the orchestrator module light when
    # only a subset is included. agent_boundary hook explicitly whitelists
    # orchestrator for these cross-agent imports.
    if "research_director" in include_agents:
        from mast.agents.research_director.graph import build as build_rd
        # Campaign 层。工具面很短（campaign 读写 + 既往记录只读 + 交接），
        # 硬件技能一个都没有 —— 与 literature 同一类，不需要 context_provider、
        # 不需要 get_state、不需要 safety_limits。
        g.add_node("research_director", build_rd(buf, model=overrides.get("research_director"),
                                                 extra_tools=_shared("research_director"),
                                                 turn_recorder=turn_recorder, extra_middleware=_extra_mw("research_director"), **_cl))
    if "literature" in include_agents:
        from mast.agents.literature.graph import build as build_lit
        g.add_node("literature", build_lit(buf, model=overrides.get("literature"),
                                           extra_tools=_shared("literature"),
                                           turn_recorder=turn_recorder, extra_middleware=_extra_mw("literature"), **_cl))
    if "experiment_design" in include_agents:
        from mast.agents.experiment_design.graph import build as build_xd
        # XD also gets the experiment/sample LIFECYCLE meta-tools (start/end/rename
        # experiment+sample). Without them, "新建实验/新建样品" routed here looped on
        # describe_skills/query_past_experiments (introspection) until the recursion
        # cap — it had no way to actually create the record (2026-06-30 fix).
        g.add_node("experiment_design", build_xd(buf, model=overrides.get("experiment_design"),
                                                 extra_tools=_shared("experiment_design") + (experiment_design_extra_tools or []),
                                                 turn_recorder=turn_recorder, extra_middleware=_extra_mw("experiment_design"), **_cl))
    if "instrument_control" in include_agents:
        from mast.agents.instrument_control.graph import build as build_ic
        if context_provider is None:
            raise ValueError(
                "context_provider required when 'instrument_control' is included"
            )
        # Thread the live-state + safety knobs down to the instrument agent's
        # safety stack (review: admin override + live-state injection lost on the
        # multi-agent path). instrument_control.build() exposes get_state +
        # safety_limits; we resolve the EFFECTIVE limits (code defaults merged with
        # admin SafetyLimits overrides — IC's build has no registry knob) and pass
        # get_state straight through. This makes:
        #   • SafetyGateMiddleware Layer-2 state preconditions (z_controller_off →
        #     withdraw-before-coarse-approach, scan_not_running, …) actually fire
        #     on the autonomous agent path — previously a silent no-op there; and
        #   • admin-tightened bias/current/Z/scan-size caps honoured on the agent
        #     path, not just the manual executor path.
        _effective_limits = _effective_safety_limits(safety_limits, override_registry)
        # HITL must stay ON for the production hardware path: a prior hardcoded
        # enable_hitl=False silently removed human approval for every DANGEROUS
        # instrument action (EmergencyRetract / AutoApproach / TipShape /
        # BiasPulse / MotorMove / SetBias…) on the live multi-agent path, while
        # the prompt still told the model approval was required (review
        # 2026-05-30 CRITICAL). It is now an explicit parameter (default True);
        # only an LLM-less test harness passes False. HITL interrupts require the
        # graph be compiled with a checkpointer — build() warns if it isn't.
        g.add_node(
            "instrument_control",
            build_ic(
                buf,
                context_provider=context_provider,
                registry=instrument_registry,
                model=overrides.get("instrument_control"),
                enable_hitl=enable_hitl,
                # IC also gets the instrument meta-tools (experiment/sample session
                # management, scan/knowledge/navigation/plan) in 群聊 — at parity
                # with its private chat. Without this the group IC had no
                # start_experiment/start_sample/rename_* and misread "新建实验/样品"
                # as a Nanonis field it couldn't set.
                extra_tools=_shared("instrument_control") + (instrument_extra_tools or []),
                post_hook=instrument_post_hook,
                recorder=recorder,
                safety_recorder=safety_recorder,
                turn_recorder=turn_recorder,
                get_state=get_state,
                # Global operating mode (safe/semi/auto) → IC's tip-processing
                # gate (SafetyGate Layer-0d + pulse HITL + belief). Only IC needs
                # it; the other agents don't touch the tip.
                get_mode=get_mode,
                safety_limits=_effective_limits,
                extra_middleware=_extra_mw("instrument_control"),
                **_cl,
            ),
        )
    if "data_processing" in include_agents:
        from mast.agents.data_processing.graph import build as build_dp
        g.add_node("data_processing", build_dp(buf, model=overrides.get("data_processing"),
                                               extra_tools=_shared("data_processing"),
                                               turn_recorder=turn_recorder, extra_middleware=_extra_mw("data_processing"), **_cl))
    if "paper_writing" in include_agents:
        from mast.agents.paper_writing.graph import build as build_pw
        g.add_node("paper_writing", build_pw(buf, model=overrides.get("paper_writing"),
                                             extra_tools=_shared("paper_writing"),
                                             turn_recorder=turn_recorder, extra_middleware=_extra_mw("paper_writing"), **_cl))
    if "paper_review" in include_agents:
        from mast.agents.paper_review.graph import build as build_pr
        g.add_node("paper_review", build_pr(buf, model=overrides.get("paper_review"),
                                            extra_tools=_shared("paper_review"),
                                            turn_recorder=turn_recorder, extra_middleware=_extra_mw("paper_review"), **_cl))

    g.add_edge(START, "supervisor")
    # Each agent's graph contains a handoff tool that emits Command(goto="supervisor")
    # via Command.PARENT — that's the return path. No explicit edges needed.

    # ⑰(2026-08-08):这条警告原文是「没有 checkpointer,DANGEROUS 技能的审批就
    # pause/resume 不了」。DANGEROUS 审批已经不存在(它现在直接执行 + 留痕 + 通知,
    # 见 mast/core/auto_approval.py),所以那句话变成了假话 —— 而它指挥的是
    # 部署者的行动。
    #
    # 警告本身**保留**,因为触发条件没变:没有 checkpointer,`interrupt()` 就没法
    # pause/resume,而树里仍有两个 interrupt 生产者(ask_user、工作流 human 节点)。
    # 换掉的只是它举的例子。
    if enable_hitl and "instrument_control" in include_agents and checkpointer is None:
        logger.warning(
            "Orchestrator: no checkpointer was provided — interrupt()-based pauses "
            "cannot resume. The remaining producers are ask_user (an agent asking "
            "the operator a question) and a workflow `human` node; without a "
            "checkpointer both are dead ends. Pass one (e.g. MemorySaver)."
        )
    compiled = g.compile(checkpointer=checkpointer)
    # Bind recursion_limit=50 as a build-time DEFAULT (). LangGraph
    # has no compile() recursion_limit param — it is purely an invoke-time config
    # (default 25). Without this, any caller that invokes WITHOUT passing
    # recursion_limit (tests, library users) gets 25, which is too tight for the
    # 6-agent topology (each hop spends supervisor + agent ≈ 2 steps). The GUI's
    # main loop already passes 50 explicitly; with_config makes 50 the default for
    # everyone else too, while an explicit invoke-time recursion_limit still wins.
    compiled = compiled.with_config(recursion_limit=50)
    logger.info("Orchestrator built with %d agents wired (hitl=%s, recursion_limit=50)",
                len(include_agents), enable_hitl)
    return compiled


__all__ = ["build", "Route"]
