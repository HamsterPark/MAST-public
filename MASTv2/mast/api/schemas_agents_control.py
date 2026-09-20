"""Pydantic schemas for the agents orchestrator-control routes (HITL resolve /
hold / release / interject).

These endpoints are LIVE-ONLY: they relay onto the running orchestrator's
in-process operator-control state on the live app (``ctx.live_app``):

  * ``_orch_interrupts``   — the LangGraph HITL gate store (pending / resolved /
                             events), filled when an IC subgraph pauses on a
                             DANGEROUS-skill approval and BLOCKS the worker.
  * ``_agents_api_state``  — operator-control dict (lock / holds / interjects /
                             interrupts-audit) the supervisor worker drains.
  * ``_build_decision``    — translates an operator verdict into a LangGraph HITL
                             Decision dict (approve / reject / edit).

In standalone mode (no live app / no running orchestrator) every handler
degrades to a typed no-op (``ok=False, degraded=True``) — never a 500. The
authority (verdict validation, arg coercion, SafetyGate re-check, waking the
blocked worker) stays entirely in the live core; the API only relays.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── POST /agents/{agent_id}/interrupts/{interrupt_id}/resolve ────────────────
class ResolveInterruptRequest(BaseModel):
    """Operator verdict for a pending HITL / DANGEROUS-skill interrupt.

    ``decision`` mirrors the live HITL verb set: approve (run as-is), reject
    (cancel), edit (replace the tool args with ``edited_args`` then re-validate).
    A composite ``workflow_human`` node instead takes a ROUTE NAME, and an
    ``ask_user`` question takes the literal verb ``answer`` with the answer
    carried in ``selected`` / ``custom_text``.
    """

    decision: str = Field(..., description="approve | reject | edit | <route name> | answer")
    edited_args: Optional[dict[str, Any]] = Field(
        default=None,
        description="Full replacement tool args (decision=edit only; merged over "
        "the original args by the live core, then SafetyGate-revalidated).",
    )
    comment: Optional[str] = Field(
        default=None, description="Operator reason / note (recorded for audit)."
    )
    # ask_user only. Deliberately NOT folded into ``edited_args``: that field
    # means "rewrite this tool call's arguments" and triggers a SafetyGate
    # re-check, so reusing it for an answer would leave every audit row
    # ambiguous about which of the two the operator actually did.
    selected: Optional[list[str]] = Field(
        default=None,
        description="ask_user only: option labels the operator chose (single-select "
        "questions carry at most one).",
    )
    custom_text: Optional[str] = Field(
        default=None,
        description="ask_user only: the operator's own answer when they did not "
        "pick one of the offered options (or the question was open-ended).",
    )


class ResolveInterruptResponse(BaseModel):
    ok: bool = False
    applied: bool = False
    # applied | no_pending_interrupt | route_not_allowed | answer_invalid | degraded
    status: str = "degraded"
    agent_id: str = ""
    interrupt_id: str = ""
    decision: str = ""
    decision_type: Optional[str] = None  # approve | reject | edit (live core verdict)
    detail: Optional[str] = None
    degraded: bool = True


# ── POST /agents/{agent_id}/hold  +  /agents/{agent_id}/release ──────────────
class HoldResponse(BaseModel):
    ok: bool = False
    agent_id: str = ""
    held: bool = False
    detail: Optional[str] = None
    degraded: bool = True


# ── POST /agents/{agent_id}/interject ───────────────────────────────────────
class InterjectRequest(BaseModel):
    text: str = Field(..., description="Operator interjection delivered to the "
                      "running supervisor on its next super-step.")


class InterjectResponse(BaseModel):
    ok: bool = False
    agent_id: str = ""
    system_addendum_id: Optional[str] = None
    detail: Optional[str] = None
    degraded: bool = True


# ── GET /agents/hitl-gates  +  POST /agents/hitl-gates/resolve ──────────────
# NOT live-app relays. The critical-event gate lives on middleware instances
# inside compiled graphs, reachable through the process-wide weak registry in
# ``agents/_shared/buffer_hitl`` — so these two work in exactly the situation
# that motivates them (a gate held by the PRIVATE-CHAT graph, where there is no
# orchestrator run and therefore nothing that would ever reset it).
class HITLGateState(BaseModel):
    """⑰(2026-08-08)之后:这些字段**恒为「没有闸门」**,只有最后一个还在动。

    打断链整个割掉了(见 ``agents/_shared/buffer_hitl``),所以 ``closed`` /
    ``unresolved`` / ``awaiting`` / ``degraded`` / 两个 ``reask_*`` 现在是历史形状
    的兼容外壳 —— 前端和 ``ReadHardwareEvents`` 在读它们,删字段会当场打断消费方。
    """

    closed: bool = False
    unresolved: int = 0
    awaiting: int = 0
    degraded: bool = False
    kinds: list[str] = Field(default_factory=list)
    #: True when the gate will ask the operator again by itself on the next
    #: refused tool. False means nothing further happens without an explicit
    #: clear — which is the whole reason this endpoint exists.
    reask_armed: bool = False
    reask_used: bool = False
    blocked_tools: list[str] = Field(default_factory=list)
    #: 记录了但没打断的关键事件条数(本中间件建立以来)。
    #:
    #: **⑰ 之后这是本模型里唯一还有信息量的字段**,而它差点在这里被吃掉:
    #: ``gate_states()`` 从一开始就返回它,但这个 model 没有声明 ⇒ pydantic 默认
    #: ``extra=ignore`` ⇒ 端点静默地把它丢了,而两侧都看不出问题(中间件以为报了,
    #: 前端以为没有)。它必须到得了用户那里,因为「没有弹窗」和「没有事件」从外面
    #: 看必须不一样 —— 这正是验收这条改动时该问的问题。
    recorded_not_escalated: int = 0


class HITLGatesResponse(BaseModel):
    ok: bool = True
    gates: list[HITLGateState] = Field(default_factory=list)
    #: How many live gates are currently refusing forward tool calls.
    closed: int = 0
    detail: Optional[str] = None
    degraded: bool = False


class ResolveGatesRequest(BaseModel):
    note: Optional[str] = Field(
        default=None,
        description="Operator reason, logged with the reopen (e.g. 「阈值未标定造成"
        "的误报，针尖实际正常」).",
    )


class ResolveGatesResponse(BaseModel):
    ok: bool = False
    #: Gates that were actually holding something when this ran. 0 is a
    #: successful no-op, not a failure — say so rather than reporting an error.
    reopened: int = 0
    gates_seen: int = 0
    detail: Optional[str] = None
    degraded: bool = True


__all__ = [
    "ResolveInterruptRequest",
    "ResolveInterruptResponse",
    "HoldResponse",
    "InterjectRequest",
    "InterjectResponse",
    "HITLGateState",
    "HITLGatesResponse",
    "ResolveGatesRequest",
    "ResolveGatesResponse",
]
