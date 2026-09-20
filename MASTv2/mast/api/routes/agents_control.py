"""Parity Wave — agents orchestrator-control domain (HITL resolve / hold /
release / interject).

Re-exposes the operator-control actions whose LOGIC lives on the LIVE app's
running orchestrator but whose UI was lost in the Gradio→TS rewrite. Each handler
is a THIN relay onto the live app (``ctx.live_app``); no SafetyGate re-check and
no thread waking policy lives here — that authority stays in the core.

Verdict translation lives in :mod:`mast.core.hitl_decision` (pure functions).
It used to be reached via ``getattr(app, "_build_decision")``, but the Gradio
removal (``7aa1996``) deleted that method without migrating it, so the lookup
returned ``None`` on EVERY call and this endpoint silently answered "live
decision builder unavailable" for every request. Translation is pure; gating
it behind a live-object lookup was the design error.

Endpoints:
  * POST /api/agents/{agent_id}/interrupts/{interrupt_id}/resolve
        — resolve a pending HITL / DANGEROUS-skill interrupt (approve/reject/edit)
          or a composite ``workflow_human`` node (verdict = route name).
          Relays into the live app's ``_orch_interrupts`` store: hands the
          decision to the BLOCKED worker and wakes it.
  * POST /api/agents/{agent_id}/hold      — pause an agent (set hold flag).
  * POST /api/agents/{agent_id}/release   — un-hold an agent.
        Both mutate the live app's ``_agents_api_state["holds"]`` the supervisor
        worker polls in ``_wait_while_held``.
  * POST /api/agents/{agent_id}/interject — queue an operator interjection the
        supervisor drains on its next super-step (``_orch_control_provider``).

LIVE-ONLY + GRACEFUL DEGRADATION (house rule 2): these need the running
orchestrator. This router must boot STANDALONE with no live core wired. When the
live app / the needed state / the running orchestrator is absent, or any relay
raises, the handler returns a valid degraded body (``degraded=True``, never
``ok``) — NEVER a 500. Heavy backends are reached lazily via ``getattr`` on the
live app inside the handler; no gradio import, no orchestrator construction here.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_agents_control import (
    HITLGatesResponse,
    HITLGateState,
    HoldResponse,
    InterjectRequest,
    InterjectResponse,
    ResolveGatesRequest,
    ResolveGatesResponse,
    ResolveInterruptRequest,
    ResolveInterruptResponse,
)
from mast.core.hitl_decision import (
    build_ask_answer as _default_ask_answer,
    build_workflow_route as _default_workflow_route,
    enforce_allowed as _default_enforce_allowed,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents_control"])

# Recognised agent ids (mirrors gui/app.py:_AGENTS_IDS). Kept local so the router
# imports nothing heavy / gradio at module load. ``__all__`` is the hold/release
# broadcast target the live ``_wait_while_held`` understands.
_AGENT_IDS = (
    "_supervisor",
    "research_director",
    "literature",
    "experiment_design",
    "instrument_control",
    "data_processing",
    "paper_writing",
    "paper_review",
    "buffer_summarizer",
)


def _record_approval(app, interrupt_id: str, pending: dict, body, decision) -> None:
    """Write the operator's HITL verdict into the v2 ``approvals`` table.

    The table, its schema and ``ApprovalService.issue()`` have all existed since
    the v2 logging layer landed. **Nothing ever called it.** 2026-07-27
    forensics: 11 CONFIRM-level dangerous operations ran that session and
    ``approvals`` had 0 rows — from the audit trail there is no way to establish
    who authorised any of them, or whether anyone did.

    Recorded on the resolve path rather than inside the middleware because THIS
    is where a human actually decides: the request carries the verdict, the
    edited arguments and the operator's comment, and it only gets here when the
    interrupt was live.

    Best-effort by construction: an audit write must never be able to block, or
    fail, an approval the operator has already given. Failures log at warning —
    not debug — because a silent audit gap is how this one lasted so long.
    """
    try:
        # The live repos the runtime already opened (CoreRuntime._v2_repos).
        # Do NOT open_live_v2() here — that creates a fresh campaign/sample/
        # experiment, so every approval would scatter into its own experiment.
        repos = getattr(app, "_v2_repos", None)
        svc = getattr(repos, "approvals", None) if repos is not None else None
        if svc is None:
            return
        action_id = str(pending.get("action_id") or pending.get("event_id")
                        or interrupt_id)
        verdict = str(body.decision or "")
        note = (body.comment or "").strip()
        evidence = {
            "verdict": verdict,
            "skill": pending.get("skill"),
            "params": pending.get("params") or {},
            "edited_args": body.edited_args or {},
            "comment": note,
            "decision_type": (decision.get("type")
                              if isinstance(decision, dict) else None),
            "interrupt_kind": pending.get("kind"),
        }
        import json as _json
        svc.issue(
            action_id=action_id,
            # The operator is anonymous in this build — no per-user auth exists.
            # Recording "operator" is honest; inventing a user id would not be.
            approver_id="operator",
            approver_kind="human_operator",
            approval_method="gui_click",
            approval_evidence=_json.dumps(evidence, ensure_ascii=False,
                                          default=str)[:4000],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("approval audit write failed for %s: %s", interrupt_id, exc)


def _live_app(ctx: Any):
    """Best-effort handle to the running app (None in standalone). Never raises."""
    return getattr(ctx, "live_app", None) or getattr(ctx, "app", None)


# ── POST /agents/{agent_id}/interrupts/{interrupt_id}/resolve ────────────────
@router.post(
    "/agents/{agent_id}/interrupts/{interrupt_id}/resolve",
    response_model=ResolveInterruptResponse,
)
def resolve_interrupt(
    agent_id: str, interrupt_id: str, body: ResolveInterruptRequest, request: Request
) -> ResolveInterruptResponse:
    """Resolve a pending HITL / DANGEROUS-skill interrupt (approve/reject/edit).

    LIVE-ONLY: the orchestrator pauses the IC subgraph on a DANGEROUS approval,
    publishes a pending entry into ``live_app._orch_interrupts`` and BLOCKS the
    worker. Here we look up that pending entry, translate the verdict via the
    live ``_build_decision`` (approve→{type:approve}, reject→{type:reject,...},
    edit→{type:edit, edited_action:{name,args}} with the full merged args), store
    it in ``resolved`` and set the worker's Event so it resumes the same thread
    via Command(resume=…). The SafetyGate re-validates edited args in the core.

    Degrades (no 500) when the live app / store / pending id is absent: an
    already-resolved/unknown id or no running gate ⇒ ``ok=False`` with an honest
    status, mirroring the live route's 409 semantics as a typed body."""
    ctx = request.app.state.ctx
    app = _live_app(ctx)
    base = dict(
        agent_id=agent_id, interrupt_id=interrupt_id, decision=body.decision or ""
    )
    if app is None:
        return ResolveInterruptResponse(
            **base, status="degraded", detail="no live app (standalone)", degraded=True
        )

    store = getattr(app, "_orch_interrupts", None)
    if not store:
        # No HITL gate store ⇒ orchestrator never built / no checkpointer.
        return ResolveInterruptResponse(
            **base, status="degraded", detail="orchestrator HITL gate not wired",
            degraded=True,
        )

    try:
        lock = store.get("lock")
        if lock is not None:
            with lock:
                pending = store.get("pending", {}).get(interrupt_id)
                ev = store.get("events", {}).get(interrupt_id)
        else:  # pragma: no cover - lock always present on a live store
            pending = store.get("pending", {}).get(interrupt_id)
            ev = store.get("events", {}).get(interrupt_id)

        # Audit record on the live operator-control state (best-effort, mirrors
        # the live route writing st["interrupts"][event_id]).
        api_state = getattr(app, "_agents_api_state", None)
        if isinstance(api_state, dict) and isinstance(api_state.get("interrupts"), dict):
            api_state["interrupts"][interrupt_id] = {
                "verdict": body.decision,
                "params": body.edited_args,
                "reason": body.comment or "",
                "t": _time.time(),
                "applied": pending is not None,
            }

        if pending is None or ev is None:
            # Already resolved, expired with the run, or unknown id.
            return ResolveInterruptResponse(
                **base,
                status="no_pending_interrupt",
                detail="No live pending interrupt with this id — it may already "
                "be resolved or the run ended.",
                degraded=True,
            )

        # ── Composite ``workflow_human`` node ─────────────────────────────
        # Its verdict is a ROUTE NAME and it resumes with {"route","note"},
        # not an approve/reject Decision. This must branch BEFORE the verdict
        # translation: a route name looks like an unknown verdict, which would
        # collapse to {"type":"reject"} and leave interpreter.py reading
        # decision["route"] == None → RuntimeError inside the workflow.
        if pending.get("kind") == "workflow_human":
            decision, route_err = _default_workflow_route(
                body.decision, pending.get("routes"), body.comment or "",
            )
            if route_err is not None:
                # Deliberately do NOT write ``resolved`` or set the Event —
                # the worker stays blocked so the operator can pick a valid
                # route and retry, instead of the run dying on a typo.
                return ResolveInterruptResponse(
                    **base, status="route_not_allowed", detail=route_err,
                    degraded=True,
                )
        # ── ``ask_user`` — the agent asked the OPERATOR a question ────────
        # Same shape of exception as workflow_human and for the same reason:
        # the verdict is not approve/reject/edit, so the translation below
        # would read the answer as an unknown verb and collapse it to reject.
        # Here that would be worse than a wrong decision — the tool would
        # receive a reject Decision where it expects an answer dict.
        elif pending.get("kind") == "ask_user":
            decision, ans_err = _default_ask_answer(
                body.selected, body.custom_text, body.comment or "",
                pending.get("ask") or {},
            )
            if ans_err is not None:
                # Same "leave the worker blocked" contract as an unknown route:
                # an invalid answer is a correctable mistake, not a reason to
                # kill a run that is still sitting there waiting for one.
                return ResolveInterruptResponse(
                    **base, status="answer_invalid", detail=ans_err,
                    degraded=True,
                )
        else:
            # Translate the verdict with the pure core implementation, which
            # first enforces THIS interrupt's advertised ``allowed_decisions``
            # (so an operator cannot 'edit' an EmergencyRetract that only
            # permits 'approve' and thereby rewrite its parameters — a real
            # safety bypass), and coerces edited string params back to their
            # original numeric types so SafetyGate's re-check actually sees
            # numbers.
            #
            # This used to be ``getattr(app, "_build_decision", None)`` with a
            # degrade branch. No real object ever provided that attribute after
            # 7aa1996, so EVERY resolve degraded and no approval worked for ~5
            # weeks. The lookup is gone on purpose: translation is pure, and a
            # live override would also be a way to bypass the allow-list above.
            decision = _default_enforce_allowed(
                body.decision,
                pending.get("allowed_decisions"),
                pending.get("skill"),
                pending.get("params") or {},
                body.edited_args,
                body.comment or "",
            )

        # Hand the decision to the blocked worker and wake it.
        if lock is not None:
            with lock:
                store["resolved"][interrupt_id] = decision
        else:  # pragma: no cover
            store["resolved"][interrupt_id] = decision
        ev.set()

        logger.info(
            "agents_control: interrupt %s resolved (agent=%s) verdict=%r type=%s",
            interrupt_id, agent_id, body.decision,
            decision.get("type") if isinstance(decision, dict) else None,
        )
        _record_approval(app, interrupt_id, pending, body, decision)
        return ResolveInterruptResponse(
            ok=True,
            applied=True,
            status="applied",
            agent_id=agent_id,
            interrupt_id=interrupt_id,
            decision=body.decision or "",
            decision_type=(decision.get("type") if isinstance(decision, dict) else None),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("agents_control: resolve interrupt failed (%s): %s",
                       interrupt_id, exc)
        return ResolveInterruptResponse(
            **base, status="degraded", detail=str(exc), degraded=True
        )


# ── GET /agents/hitl-gates  +  POST /agents/hitl-gates/resolve ──────────────
#
# The out-of-band operator entry onto the CRITICAL-EVENT tool gate
# (``agents/_shared/buffer_hitl``). Unlike everything else in this module these
# are NOT live-app relays: the gate lives on middleware instances inside compiled
# graphs and is reached through that module's process-wide weak registry. That
# is deliberate — the case they exist for is a gate held by the PRIVATE-CHAT
# graph, where ``ctx.live_app`` may be wired but there is no orchestrator run,
# and a run boundary was the only thing in the tree that ever reset a gate.
#
# WHY (2026-08-04, reproduced on the instrument): one press of 拒绝 on a
# ``buffer_hitl`` interrupt in the main chat wedged instrument_control until the
# process restarted. Rejecting holds the gate — correct — but the interrupt is
# never raised again (rising-edge events, and ``before_model`` returns early once
# its queues are empty), and approving an interrupt was the ONLY reopen an
# operator could reach. ``reset_all_gates`` existed, with exactly one caller, on
# a path the main chat never takes. A new conversation does not help either:
# ``ConversationEngine`` caches one graph per agent_id, so every conversation
# shares the same middleware instance.
#
# The paths are LITERALS here and constants in ``buffer_hitl`` (which writes them
# into the refusal message the agent reads). tests/v2/unit/api/test_agents_control
# pins the two against each other — a promise printed in one place and served in
# another is exactly what rotted last time.
@router.get("/agents/hitl-gates", response_model=HITLGatesResponse)
def hitl_gates(request: Request) -> HITLGatesResponse:  # noqa: ARG001 — uniform signature
    """Report every live critical-event gate: is anything refusing tool calls?

    Read-only and process-local; needs no live app. ``reask_armed`` is the field
    that matters operationally — it is the difference between "it will ask you
    again by itself" and "nothing more happens until you clear it"."""
    try:
        from mast.agents._shared.buffer_hitl import gate_states
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.debug("agents_control: buffer_hitl unavailable: %s", exc)
        return HITLGatesResponse(ok=False, detail=f"gate registry unavailable: {exc}",
                                 degraded=True)
    try:
        states = [HITLGateState(**s) for s in gate_states()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("agents_control: gate_states failed: %s", exc)
        return HITLGatesResponse(ok=False, detail=str(exc), degraded=True)
    return HITLGatesResponse(
        gates=states, closed=sum(1 for s in states if s.closed),
    )


@router.post("/agents/hitl-gates/resolve", response_model=ResolveGatesResponse)
def resolve_hitl_gates(
    body: ResolveGatesRequest, request: Request  # noqa: ARG001 — uniform signature
) -> ResolveGatesResponse:
    """Operator clears the critical-event gate — the reopen that needs no interrupt.

    This is an OPERATOR action by construction: it is an HTTP endpoint, so no
    agent can reach it, and reopening its own gate is precisely what an agent
    must not be able to do. ``reopened=0`` means nothing was holding — a
    successful no-op, reported as ``ok`` rather than dressed up as an error.

    ``resolve_all_gates`` and not ``reset_all_gates``: the verbs mean different
    things and only one of them is what happened here. ``resolve`` is "the
    operator dealt with it" — it opens the gate and leaves the interrupt
    bookkeeping alone. ``reset`` additionally drops ``_awaiting`` /
    ``_interrupted_ids``, which is right at a RUN boundary (nothing carries over)
    and wrong here, where a pause may still be live in a chat turn the operator
    can still answer.
    """
    note = (body.note or "").strip()
    try:
        from mast.agents._shared.buffer_hitl import (
            gate_resolve_url,
            gate_states,
            resolve_all_gates,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.debug("agents_control: buffer_hitl unavailable: %s", exc)
        return ResolveGatesResponse(detail=f"gate registry unavailable: {exc}",
                                    degraded=True)
    try:
        seen = len(gate_states())
        why = f"operator cleared via {gate_resolve_url()}" + (f": {note}" if note else "")
        n = resolve_all_gates(why)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agents_control: resolve gates failed: %s", exc)
        return ResolveGatesResponse(detail=str(exc), degraded=True)
    # WARNING, not info: this overrides a refusal the operator themselves gave
    # earlier, and the run is about to be allowed to drive the instrument on data
    # something judged untrustworthy. It belongs in the log at the same level as
    # the block it undoes.
    logger.warning(
        "agents_control: operator cleared %d/%d critical-event gate(s)%s",
        n, seen, f" — {note}" if note else "",
    )
    try:
        from mast.core.diagnostics import record as _diag

        _diag("hitl_gate_operator_clear", "agents_control",
              "用户显式清除了关键事件闸门（此前仍未解决）",
              reopened=n, gates_seen=seen, note=note)
    except Exception as exc:  # noqa: BLE001 — diagnostics never breaks the action
        logger.debug("agents_control: gate-clear diagnostic skipped: %s", exc)
    return ResolveGatesResponse(
        ok=True, reopened=n, gates_seen=seen, degraded=False,
        detail=("没有闸门处于关闭状态（无需清除）" if n == 0 else None),
    )


# ── POST /agents/{agent_id}/hold  +  /agents/{agent_id}/release ──────────────
def _set_hold(request: Request, agent_id: str, held: bool) -> HoldResponse:
    """Shared relay for hold/release onto live ``_agents_api_state['holds']``."""
    ctx = request.app.state.ctx
    app = _live_app(ctx)
    if agent_id not in _AGENT_IDS and agent_id != "__all__":
        # Unknown id would write a hold key the worker never checks — reject
        # honestly (typed, not a 404, to keep the degrade-safe contract).
        return HoldResponse(
            agent_id=agent_id, held=False,
            detail=f"unknown agent {agent_id}", degraded=True,
        )
    if app is None:
        return HoldResponse(
            agent_id=agent_id, held=held,
            detail="no live app (standalone)", degraded=True,
        )
    api_state = getattr(app, "_agents_api_state", None)
    if not isinstance(api_state, dict) or not isinstance(api_state.get("holds"), dict):
        return HoldResponse(
            agent_id=agent_id, held=held,
            detail="operator-control state not wired", degraded=True,
        )
    try:
        # Mutate under the shared state lock so the run-task worker's _honor_holds
        # read sees a consistent holds map (and to match the interject drain).
        import contextlib as _ctxlib
        lock = api_state.get("lock")
        with (lock or _ctxlib.nullcontext()):
            holds = api_state["holds"]
            if held:
                holds[agent_id] = True
            else:
                # Drop the key on un-hold so a stale True can't linger (mirrors live).
                holds.pop(agent_id, None)
        logger.info("agents_control: %s hold=%s", agent_id, held)
        return HoldResponse(ok=True, agent_id=agent_id, held=held, degraded=False)
    except Exception as exc:
        logger.warning("agents_control: set hold failed (%s): %s", agent_id, exc)
        return HoldResponse(agent_id=agent_id, held=held, detail=str(exc), degraded=True)


@router.post("/agents/{agent_id}/hold", response_model=HoldResponse)
def hold_agent(agent_id: str, request: Request) -> HoldResponse:
    """Pause an agent — set its hold flag the supervisor worker polls.

    LIVE-ONLY: relays into ``live_app._agents_api_state['holds']`` which
    ``_wait_while_held`` polls between super-steps. Degrades to a typed no-op
    when the live app / state is absent. Accepts ``__all__`` to hold every agent."""
    return _set_hold(request, agent_id, held=True)


@router.post("/agents/{agent_id}/release", response_model=HoldResponse)
def release_agent(agent_id: str, request: Request) -> HoldResponse:
    """Un-hold an agent (clear its hold flag). LIVE-ONLY; degrade-safe."""
    return _set_hold(request, agent_id, held=False)


# ── POST /agents/{agent_id}/interject ───────────────────────────────────────
@router.post("/agents/{agent_id}/interject", response_model=InterjectResponse)
def interject(
    agent_id: str, body: InterjectRequest, request: Request
) -> InterjectResponse:
    """Queue an operator interjection for the running supervisor.

    LIVE-ONLY: appends to ``live_app._agents_api_state['interjects']`` (under the
    state lock so a concurrent drain in ``_orch_control_provider`` can't lose it)
    — the supervisor delivers it as an operator message on its next super-step.
    Degrades to a typed no-op when the live app / state is absent."""
    ctx = request.app.state.ctx
    app = _live_app(ctx)
    text = (body.text or "").strip()
    if not text:
        return InterjectResponse(
            agent_id=agent_id, detail="empty interjection", degraded=True
        )
    if agent_id not in _AGENT_IDS and agent_id != "__all__":
        # Validate like /hold does (it has checked since day one). An unknown id
        # used to be accepted, queued, and then dropped by
        # ``_orch_control_provider``'s directed-route step — while the text still
        # went through carrying "(指向 <typo>)", so the operator read their own
        # instruction back and believed it had been routed. Fail loudly instead.
        return InterjectResponse(
            agent_id=agent_id,
            detail=(f"未知的智能体 id「{agent_id}」——插话未入队。"
                    f"可用：{', '.join(_AGENT_IDS)} 或 __all__"),
            degraded=True,
        )
    if app is None:
        return InterjectResponse(
            agent_id=agent_id, detail="no live app (standalone)", degraded=True
        )
    api_state = getattr(app, "_agents_api_state", None)
    if not isinstance(api_state, dict) or not isinstance(api_state.get("interjects"), list):
        return InterjectResponse(
            agent_id=agent_id, detail="operator-control state not wired", degraded=True
        )
    # No live run → the queue has no consumer; a queued interjection would sit
    # in a black hole while the operator believes it was delivered (2026-07-10
    # #90: the STS run had silently died, the 15:42 interjection vanished, and
    # nothing told the operator). Say so instead, with the actionable next step.
    _task = api_state.get("task")
    if not (isinstance(_task, dict) and _task.get("active")):
        return InterjectResponse(
            agent_id=agent_id,
            detail=("当前没有正在运行的任务——插话不会被送达。"
                    "请直接在群聊输入框发送新指令（同一会话可点『继续』恢复）。"),
            degraded=True,
        )
    # STAMP THE TASK (2026-07-28). The queue is a process-level singleton and its
    # entries carried no run identity, while run-task never cleaned it up. So an
    # interjection queued into a window where nobody would consume it (the run's
    # final LLM call, an abort, an exception — `active` is still True but no
    # super-step is coming) survived to be drained by the FIRST hop of the NEXT
    # task. A stale instruction carrying "@instrument_control" then became a
    # deterministic hard route on a brand-new experiment — the one path in that
    # queue that touches hardware.
    _task_id = str(_task.get("id") or "")
    try:
        lock = api_state.get("lock")
        entry = {"id": "", "agent_id": agent_id, "text": text,
                 "t": _time.time(), "task_id": _task_id}
        if lock is not None:
            with lock:
                sa_id = f"sa_{int(_time.time() * 1000)}_{len(api_state['interjects'])}"
                entry["id"] = sa_id
                api_state["interjects"].append(entry)
        else:  # pragma: no cover - lock always present on a live state
            sa_id = f"sa_{int(_time.time() * 1000)}_{len(api_state['interjects'])}"
            entry["id"] = sa_id
            api_state["interjects"].append(entry)
        logger.info("agents_control: operator interjection → %s: %s", agent_id, text[:80])
        # Best-effort: land the interjection in the durable group transcript too,
        # so a replay shows WHEN the operator cut in (it was invisible before).
        try:
            conv_id = str(_task.get("conversation_id") or "")
            conv_store = getattr(app, "_conv_store", None)
            if conv_id and conv_store is not None:
                conv_store.append_message(
                    conv_id, kind="operator", agent_id=agent_id, role="user",
                    text=f"[插话 → {agent_id}] {text}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("agents_control: interject transcript append failed: %s", exc)
        return InterjectResponse(
            ok=True, agent_id=agent_id, system_addendum_id=sa_id, degraded=False
        )
    except Exception as exc:
        logger.warning("agents_control: interject failed (%s): %s", agent_id, exc)
        return InterjectResponse(agent_id=agent_id, detail=str(exc), degraded=True)
