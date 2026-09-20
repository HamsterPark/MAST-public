"""Asking an agent whether it should start now, or wait for something to arrive.

``docs/v2/design/wakeup_scheduling.md`` §3. Two questions, one engine:

  (A) **before dispatch** — "you are missing X. Start anyway, or wait?"
  (B) **after something arrived** — "the thing you were waiting for is here.
      Wake up, or keep waiting?"

The design brief driving this module:

    一个 agent 也应该被问一个问题:现在环境有变化了,你要 wakeup 吗?他可以选择
    不要。LLM 可以做这些判断,**关键是要设计好发送给他们的问题**。

So the prompt text below is the substance of this module, not decoration. Five
rules it follows, each paid for by a specific past failure:

1. **State facts, never hypotheticals.** The "what you already have" block is
   rendered by ``render_upstream_block`` — the SAME renderer that builds the block
   the agent will really receive if it starts. Two wordings of the same thing means
   deciding from A and then working from B.
2. **Name what is missing AND who produces it.** "缺少一些信息" earns an equally
   vague answer back.
3. **Put the ASYMMETRY OF FAILURE in the question.** This is the dividing line
   between a scheduler and a silent death channel. Biasing toward "wait" produces a
   deadlock nobody sees; biasing toward "start" produces one imperfect but VISIBLE
   artifact with its limitations written down. So the prompt says, in as many
   words, that starting is the default — and every failure path in this module
   returns ``start`` for the same reason.
4. **``waiting_for`` must be a CLOSED SET.** Free text cannot be matched against a
   future arrival, so a free-text wait is a wait that never ends. The set is
   enforced HERE rather than trusted from the model, which also means the
   degraded text-parse path cannot produce an unmatchable value.
5. **Tell it how long it has waited, how often it has declined, and when the wait
   expires.** An agent that does not know it has already declined three times
   cannot make a better decision than it made the first time.

Engine choice: ``_shared.llm_route.route_decision`` — already hardened across all
six providers (function_calling on flattened text → tolerant text-JSON → single
unambiguous bare name), and its own docstring offers itself to "any future bounded
decision point". **Never call ``with_structured_output`` directly**; the comment in
``cognition_llm.py`` is the standing evidence ("NOT with_structured_output — that is
not provider-portable across MAST's 6 providers") and ``orchestrator/graph.py``
still carries the six distinct failure modes that taught it.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from mast.agents._shared.artifact_channel import (
    WAITABLE_FIELDS,
    field_label,
    producer_of,
    render_upstream_block,
)

logger = logging.getLogger(__name__)

#: What the model may answer. Deliberately two words: "start" and "wait" are the
#: only actions that differ in what happens next.
ACTIONS: tuple[str, ...] = ("start", "wait")


class StartOrWait(TypedDict):
    """Structured-output schema for question (A)."""

    action: Literal["start", "wait"]
    waiting_for: list[str]
    reason: str


class WakeOrKeepWaiting(TypedDict):
    """Structured-output schema for question (B)."""

    action: Literal["start", "wait"]
    waiting_for: list[str]
    reason: str


_JSON_HINT = (
    "Respond with ONLY a single json object and nothing else, of the form: "
    '{"action": "start"|"wait", "waiting_for": ["<zero or more of '
    + "|".join(WAITABLE_FIELDS) + '>"], "reason": "<one sentence>"}. '
    'The key MUST be "action". Use "start" unless the missing item would make '
    'your output guesswork. "waiting_for" MUST be a json array drawn only from '
    "the listed names — free text there can never be matched against a future "
    "arrival, so it would mean waiting forever."
)


def _clamp_waiting_for(raw: Any, fallback: "list[str] | tuple[str, ...]") -> list[str]:
    """Force ``waiting_for`` into the closed set. Never trusts the model.

    Rule 4 enforced on our side of the boundary. A value outside
    ``WAITABLE_FIELDS`` cannot be matched against any future arrival, so accepting
    one would create a park that is structurally unwakeable — the single worst
    outcome this whole mechanism can produce.

    An empty or fully-invalid list falls back to what the agent was just TOLD it
    was missing. That is the only defensible guess: those are the fields the
    question named, they are all in the closed set, and "wait for nothing in
    particular" is not a wait that can end.
    """
    out: list[str] = []
    if isinstance(raw, str):
        raw = [raw]
    for item in (raw or []):
        name = str(item).strip()
        if name in WAITABLE_FIELDS and name not in out:
            out.append(name)
    if not out:
        out = [f for f in fallback if f in WAITABLE_FIELDS]
    return out


def _missing_block(fields: "list[str] | tuple[str, ...]") -> str:
    """The "what you do NOT have" section — every line naming its producer (rule 2)."""
    lines = []
    for f in fields:
        who = producer_of(f)
        by = f"(通常由 {who} 产出)" if who else ""
        lines.append(f"- {field_label(f)} `{f}` {by}")
    return "\n".join(lines)


def _closed_set_line() -> str:
    return "{" + " | ".join(WAITABLE_FIELDS) + "}"


def _have_block(state: Any, agent: str, available_tools: "set[str] | None") -> str:
    """Rule 1: the SAME renderer that builds the agent's real context block.

    If these two ever diverge the agent decides from one description of the world
    and then works from another.
    """
    try:
        block = render_upstream_block(state, agent, available_tools=available_tools)
    except Exception as exc:  # noqa: BLE001
        logger.debug("activation: upstream render failed: %s", exc)
        block = ""
    return block or "（本实验目前还没有任何上游产物。这是确认过的空,不是查询失败。）"


def build_start_question(*, agent: str, instruction: str, state: Any,
                         missing_soft: "list[str] | tuple[str, ...]",
                         unknown: "list[str] | tuple[str, ...]" = (),
                         available_tools: "set[str] | None" = None) -> str:
    """Question (A). Only asked when nothing HARD is missing — a hard miss is
    decided without a model call, because it is not a judgement."""
    parts = [
        f"你是 {agent}。本轮任务:{instruction}".rstrip(),
        "",
        "【现在环境里已经有的】",
        _have_block(state, agent, available_tools),
        "",
        "【还没有的】",
        _missing_block(missing_soft) or "-（无）",
    ]
    if unknown:
        # Rule 1 again: an unreadable store is NOT an absent product, and saying so
        # is the difference between a decision and a decision made on a false
        # premise.
        parts += [
            "",
            "【这次查不到的】(注意:是查询失败,**不等于**没有)",
            _missing_block(unknown),
        ]
    parts += [
        "",
        "【要你决定的】",
        "现在就开工,还是等某样东西到位再开工?",
        "",
        "判据(重要):",
        "- **默认是现在就开工。** 一份基于现有资料、并且明确写清了局限的产物,"
        "比一个不知何时才醒的等待有用得多。",
        "- 只有在「缺的东西会让你的产出变成猜测」时才选等待。",
        f"- 选等待就必须说明等什么,且只能从这个清单里选:{_closed_set_line()}",
        "  —— 自由文本没法在新资料到位时被匹配上,等于永远不会被叫醒。",
    ]
    return "\n".join(parts)


def build_wake_question(*, agent: str, waiting_for: "list[str] | tuple[str, ...]",
                        state: Any, arrived: "list[str] | tuple[str, ...]",
                        waited_human: str, declines: int, deadline_human: str,
                        instruction: str = "",
                        available_tools: "set[str] | None" = None) -> str:
    """Question (B). Rule 5 is the whole reason for the age / declines / deadline
    arguments: an agent that does not know it has declined three times cannot make
    a better decision than it made the first time."""
    waited_on = "、".join(field_label(f) for f in waiting_for) or "（未记录）"
    parts = [
        f"你是 {agent},之前选择了等待,等的是:{waited_on}。",
        f"已经等了 {waited_human};你之前拒绝过 {declines} 次。",
    ]
    if instruction:
        parts.append(f"当时的任务是:{instruction}")
    parts += [
        "",
        "【刚刚到位的】",
        _missing_block(arrived) or "-（本轮没有新产物到位）",
        "",
        "【现在环境里全部已有的】",
        _have_block(state, agent, available_tools),
        "",
        "【要你决定的】",
        "现在醒来干活,还是继续等?",
        "- 你等的东西**已经到位了** → 除非有别的具体理由,应当醒(选 start)。",
        f"- 继续等就要重新说明等什么(同一个闭集):{_closed_set_line()}",
        f"- **注意**:你已经拒绝 {declines} 次。再拒绝会让这次等待更接近超时上限"
        f"（{deadline_human}）,超时后它会**浮到用户面前**,不会自动继续。",
    ]
    return "\n".join(parts)


def _ask(model, question: str, schema, fallback_waiting: "list[str] | tuple[str, ...]",
         *, what: str) -> dict:
    """Run one bounded decision. ALWAYS returns a decision; never raises.

    ``start`` on every failure path, and that is rule 3 as code rather than as
    advice: if the model is unreachable, or answers unparseably, or times out, the
    choice is between a wait nobody can see and one visible imperfect artifact.
    A scheduler that deadlocks when its LLM hiccups is worse than no scheduler.
    """
    from mast.agents._shared.llm_route import route_decision

    if model is None:
        return {"action": "start", "waiting_for": [], "reason": "no model available",
                "parse_path": "no_model"}
    msgs = [{"role": "user", "content": question}]
    try:
        decision, path = route_decision(
            model, msgs, schema=schema, valid_targets=ACTIONS,
            json_hint=_JSON_HINT, field="action")
    except Exception as exc:  # noqa: BLE001
        logger.info("activation %s: undecidable (%s) → defaulting to start", what, exc)
        return {"action": "start", "waiting_for": [],
                "reason": f"decision failed ({type(exc).__name__}); defaulted to start",
                "parse_path": "error"}
    action = str(decision.get("action") or "start")
    if action not in ACTIONS:
        action = "start"
    out = {
        "action": action,
        "waiting_for": (_clamp_waiting_for(decision.get("waiting_for"),
                                          fallback_waiting)
                        if action == "wait" else []),
        "reason": str(decision.get("reason") or "")[:500],
        "parse_path": path,
    }
    logger.info("activation %s: %s (%s) waiting_for=%s",
                what, out["action"], path, out["waiting_for"])
    return out


def ask_should_start(model, *, agent: str, instruction: str, state: Any,
                     missing_soft: "list[str] | tuple[str, ...]",
                     unknown: "list[str] | tuple[str, ...]" = (),
                     available_tools: "set[str] | None" = None) -> dict:
    """Question (A): dispatch now, or park? → ``{action, waiting_for, reason, parse_path}``.

    Only meaningful when ``missing_soft`` is non-empty; with nothing missing there
    is nothing to decide and the caller must not spend a model call.
    """
    if not missing_soft and not unknown:
        return {"action": "start", "waiting_for": [], "reason": "nothing missing",
                "parse_path": "skipped"}
    q = build_start_question(agent=agent, instruction=instruction, state=state,
                             missing_soft=missing_soft, unknown=unknown,
                             available_tools=available_tools)
    return _ask(model, q, StartOrWait, list(missing_soft) + list(unknown),
                what=f"start?{agent}")


def ask_should_wake(model, *, agent: str, waiting_for: "list[str] | tuple[str, ...]",
                    state: Any, arrived: "list[str] | tuple[str, ...]",
                    waited_human: str, declines: int, deadline_human: str,
                    instruction: str = "",
                    available_tools: "set[str] | None" = None) -> dict:
    """Question (B): wake now, or keep waiting? Same return shape as (A)."""
    q = build_wake_question(agent=agent, waiting_for=waiting_for, state=state,
                            arrived=arrived, waited_human=waited_human,
                            declines=declines, deadline_human=deadline_human,
                            instruction=instruction, available_tools=available_tools)
    return _ask(model, q, WakeOrKeepWaiting, waiting_for, what=f"wake?{agent}")


def humanize_age(seconds: float) -> str:
    """"3 分钟" / "2 天". For rule 5 — a raw epoch delta is not an answer."""
    s = max(0.0, float(seconds or 0.0))
    if s < 90:
        return "不到 1 分钟"
    if s < 5400:
        return f"{int(s // 60)} 分钟"
    if s < 172800:
        return f"{int(s // 3600)} 小时"
    return f"{int(s // 86400)} 天"


__all__ = [
    "ACTIONS", "StartOrWait", "WakeOrKeepWaiting",
    "ask_should_start", "ask_should_wake",
    "build_start_question", "build_wake_question", "humanize_age",
]
