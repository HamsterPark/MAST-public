"""Shared constructor for the two per-agent call-limit middlewares.

Why this exists
---------------
Every agent wires a ``ToolCallLimitMiddleware`` + ``ModelCallLimitMiddleware``
pair as its R7 loop guard. Two knobs matter, and they are easy to get subtly
wrong across six near-identical call sites:

* ``thread_limit`` — CUMULATIVE across the whole conversation thread. It is
  persisted in the checkpoint (``thread_model_call_count`` is a ``PrivateStateAttr``)
  and is **never reset**. Fine as a one-shot task backstop; a footgun on a
  DURABLE multi-turn chat / group thread, where it silently bricks the
  conversation once the running total is reached — every later turn then
  short-circuits to a "Model call limits exceeded" message in ``before_model``
  WITHOUT ever calling the model. With the 2026-06-16 conversation-runtime
  refactor (one durable thread per conversation, reused across turns) the old
  default of 40 cumulative model calls started bricking normal long chats after
  a few dozen turns. That is the bug this module exists to fix.
* ``run_limit`` — resets each agent INVOCATION (``run_*_count`` is an
  ``UntrackedValue``). This is the real in-turn circuit-breaker against a ReAct
  spin (retrying a precondition/safety-blocked tool), and it is what the durable
  chat/group paths rely on instead of the cumulative cap. Context growth is
  handled separately by the compaction/summarization middleware, so the thread
  cap is not needed to bound the conversation.

So the private-chat / group-chat paths pass ``max_*_calls=None`` to DISABLE the
cumulative thread cap and lean on the per-run cap; one-shot paths (composite
agent nodes, orchestrator run-task) may keep a thread cap.

This helper centralises three fiddly invariants the raw middleware enforces:

1. At least one of (``thread_limit``, ``run_limit``) must be non-None, else the
   middleware constructor raises ``ValueError``. We keep the run limit a
   positive int always, so disabling the thread cap is always safe.
2. ``ToolCallLimitMiddleware`` additionally requires ``run_limit <= thread_limit``
   when both are set — we clamp the run limit down so a caller passing a small
   thread cap (e.g. the composite agent node's 40) never trips that ValueError.
3. ``None`` / 0 / negative thread values all mean "no cumulative cap".
"""
from __future__ import annotations

import logging

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

logger = logging.getLogger(__name__)

# Per-run defaults (reset every invocation). Loose enough that a healthy single
# turn never hits them, tight enough to break a genuine in-turn spin.
#
# 模型调用预算需要容纳包含多次读回、动作和验证的回合，同时保持有限上限，
# 防止重复唤醒或失控循环无限消耗模型调用并持续驱动仪器。
# 这里的默认值是工程预算，不是实验标定值；改变它时必须同步检查递归预算。
#
# ⚠️ 改这个数**必须**同时看 :data:`MAX_RECURSION_LIMIT` —— 见那里。
DEFAULT_MODEL_CALLS_PER_RUN = 500
DEFAULT_TOOL_CALLS_PER_RUN = 80

#: ``0`` 的含义:**不限**。
#:
#: 不能真的传 ``None`` —— ``make_call_limit_middleware`` 的不变式 1 要求
#: (thread_limit, run_limit) 至少一个非 None,而聊天路径**刻意关掉了 thread 上限**,
#: 于是两个都 None 会让中间件构造器直接 ``ValueError``,agent 根本建不起来。
#:
#: 所以「不限」= 一个**够不到**的数。一百万次模型调用在一个回合里不可能发生
#: (按最快的模型也是以天计、以千美元计),而它保住了那条不变式。
#:
#: ⚠️ 这个语义是**新的**(2026-08-10)。在此之前 ``0`` 走的是 ``v > 0 else default``
#: —— **想「不设上限」的人填 0,拿到的是 30**。那正是 ``_pos()`` 那个陷阱的形状,
#: 而本仓今天已经为它付过三次学费(ForgeAuTip 的 per_run、看门狗的 disable(for_s)、
#: 还有这里)。**0 现在真的表示不限,而且会大声说。**
UNLIMITED_RUN_CALLS = 1_000_000


def _norm_thread(v: "int | None") -> "int | None":
    """0 / negative / None / non-int → None (no cumulative cap); positive → itself."""
    if v is None:
        return None
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _norm_run(v: "int | None", default: int) -> int:
    """Always a positive int — the run cap is the guaranteed non-None limit.

    三态,而不是两态:

    * ``> 0``   → 就用它;
    * ``== 0``  → **不限**(:data:`UNLIMITED_RUN_CALLS`),并**说一句** ——
      这是用户表达「不要设上限」的方式,不该被悄悄读成「用出厂值」;
    * ``< 0`` / 读不懂 / ``None`` → 出厂值。负数不是「关掉」,是**打错了**,
      而把打错的值悄悄换成出厂值正是 ``lookup(name) || DEFAULT`` 那个形状 ——
      所以它也说一句。
    """
    if v is None:
        return default
    try:
        v = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning(
            "call limits: 上限值 %r 读不懂 —— 改用出厂值 %d(不是静默回退,这一行就是告知)",
            v, default)
        return default
    if v > 0:
        return v
    if v == 0:
        logger.warning(
            "call limits: 每轮上限被设为 0 = **不限**(实际用 %d,一个够不到的数)。"
            "这道刹车现在是关的 —— 一个失控的循环可以一直烧钱、一直驱动仪器。",
            UNLIMITED_RUN_CALLS)
        return UNLIMITED_RUN_CALLS
    logger.warning(
        "call limits: 上限值 %d 是负数 —— 那不是「关掉」(关掉请填 0),"
        "按打错处理,改用出厂值 %d。", v, default)
    return default


def make_call_limit_middleware(
    *,
    max_model_calls: "int | None" = 40,
    max_tool_calls: "int | None" = 120,
    max_model_calls_per_run: "int | None" = DEFAULT_MODEL_CALLS_PER_RUN,
    max_tool_calls_per_run: "int | None" = DEFAULT_TOOL_CALLS_PER_RUN,
    tool_exit_behavior: str = "continue",
) -> list:
    """Return ``[ToolCallLimitMiddleware, ModelCallLimitMiddleware]``.

    ``max_model_calls`` / ``max_tool_calls`` are the CUMULATIVE thread caps; pass
    ``None`` (or 0) to disable them — durable chat/group threads do this so a long
    conversation is never bricked. ``*_per_run`` are the per-invocation caps and
    are always kept positive so at least one limit is always set. The tool run
    cap is clamped down to the tool thread cap when the latter is smaller
    (``ToolCallLimitMiddleware`` invariant).

    ``tool_exit_behavior``: ``"end"`` (IC — stop the run) or ``"continue"``
    (default — block the exceeded tool, let the agent keep going).
    """
    m_thread = _norm_thread(max_model_calls)
    t_thread = _norm_thread(max_tool_calls)
    m_run = _norm_run(max_model_calls_per_run, DEFAULT_MODEL_CALLS_PER_RUN)
    t_run = _norm_run(max_tool_calls_per_run, DEFAULT_TOOL_CALLS_PER_RUN)

    # ToolCallLimitMiddleware requires run_limit <= thread_limit when both set.
    if t_thread is not None and t_run > t_thread:
        t_run = t_thread
    # ModelCallLimitMiddleware has no such cross-check, but keep it consistent so
    # a tiny thread cap can't be nullified by a larger run cap.
    if m_thread is not None and m_run > m_thread:
        m_run = m_thread

    return [
        ToolCallLimitMiddleware(
            thread_limit=t_thread, run_limit=t_run,
            exit_behavior=tool_exit_behavior),
        ModelCallLimitMiddleware(
            thread_limit=m_thread, run_limit=m_run),
    ]


# Recursion budget is derived from the compiled graph in units of tool cycles.
# LangGraph counts super-steps, not calls: before_model and after_model hooks
# contribute separate nodes, while wrap hooks do not. A typical cycle costs
# n_before_model + 1 model + n_after_model + 1 tools.
#
# The recursion limit must allow the readable call-limit and stall guards to
# run before GraphRecursionError can end the turn. Deriving it from actual graph
# structure keeps middleware additions from silently reducing usable calls.
# Floor and cap bound pathological graph sizes; the cap must remain compatible
# with DEFAULT_MODEL_CALLS_PER_RUN. Tests check that coupled invariant.
MIN_RECURSION_LIMIT = 50
MAX_RECURSION_LIMIT = 9000


def super_steps_per_tool_call(compiled_graph) -> "int | None":
    """Super-steps ONE model→tool round costs on ``compiled_graph``.

    ``len(nodes) - 2`` (drop ``__start__`` / ``__end__``). Verified against a
    real streamed run for both the bare (5) and production (11) IC graph shapes
    — see ``tests/v2/unit/agents/test_chat_recursion_budget.py``, which measures
    it by streaming rather than trusting this arithmetic.

    Returns None when the graph cannot be introspected; callers fall back.
    """
    try:
        nodes = list(compiled_graph.get_graph().nodes)
    except Exception:  # noqa: BLE001 — never break a turn over a budget estimate
        return None
    n = len(nodes) - 2
    return n if n >= 2 else None


def turn_super_steps(steps_per_cycle: int, tool_calls: int) -> int:
    """Total super-steps a turn making ``tool_calls`` sequential tool rounds costs.

    ``steps_per_cycle * (tool_calls + 1) - 1``: one extra model call produces the
    final text answer, and it is not followed by a ``tools`` node. Exact against
    measurement at both graph shapes (bare: 5n+4, production: 11n+10).
    """
    return steps_per_cycle * (max(0, int(tool_calls)) + 1) - 1


def derive_recursion_limit(
    compiled_graph,
    *,
    model_calls_per_run: int = DEFAULT_MODEL_CALLS_PER_RUN,
    fallback: int = 300,
) -> int:
    """A super-step budget that lets the READABLE guards fire first.

    Sized so ``ModelCallLimitMiddleware(run_limit=model_calls_per_run)`` — which
    ends the turn with a message an operator can act on — is the binding limit,
    and ``recursion_limit`` goes back to being the backstop it is documented to
    be rather than the primary (and only) stop.

    ``+1`` covers the limit middleware's own jump-to-end super-step.
    """
    per_cycle = super_steps_per_tool_call(compiled_graph)
    if per_cycle is None:
        return max(MIN_RECURSION_LIMIT, min(MAX_RECURSION_LIMIT, int(fallback)))
    calls = _norm_run(model_calls_per_run, DEFAULT_MODEL_CALLS_PER_RUN)
    want = per_cycle * (calls + 1)
    return max(MIN_RECURSION_LIMIT, min(MAX_RECURSION_LIMIT, want))


__all__ = [
    "make_call_limit_middleware",
    "derive_recursion_limit",
    "super_steps_per_tool_call",
    "turn_super_steps",
    "DEFAULT_MODEL_CALLS_PER_RUN",
    "DEFAULT_TOOL_CALLS_PER_RUN",
    "MIN_RECURSION_LIMIT",
    "MAX_RECURSION_LIMIT",
]
