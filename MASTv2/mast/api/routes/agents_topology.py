"""Agents *topology* routes — parity rebuild Wave A (core-exists-no-api gap).

The Gradio→TS SPA rewrite dropped the rich Agents inspector
(``gui/static/agents/agents-ui.jsx``) whose data feed was the live
``/agents/snapshot`` + ``/artifacts`` Starlette routes mounted on the old
``MASTApp``. The LOGIC still exists in the core (the orchestrator stream
buckets per-agent threads/handoffs/transcript into ``MASTApp``'s in-memory
agents state, and ``MASTApp._snapshot_pending_interrupts`` exposes the live
HITL gate). This module re-exposes that data as typed endpoints so the new
frontend can render the topology / interrupt host / artifact matrix again.

Endpoints:
  GET /api/agents/snapshot              topology + threads + handoff_events + per-agent status
  GET /api/agents/{agent_id}/interrupts pending HITL approvals for an agent (or roster)
  GET /api/artifacts                    workspace artifact topology (produced + edits)
  GET /api/artifacts/permissions        artifact R/W matrix (best-effort from typed schema)

ALREADY api-exists (routes/agents.py — frontend just renders): the per-agent
tool catalog (``GET /agents/tools``), the model+thinking table
(``GET /agents/models``), and the full conversation surface — list / create /
rename / delete conversations, ``GET .../messages``, ``POST .../chat/abort``.
The SUP run_task / task_status / task_abort / transcript live behind the
chat-stream seam (routes/chat_stream.py) — out of scope here.

GRACEFUL DEGRADATION (house rule): this API boots STANDALONE. The static
registry/model/thinking comes from the kept ``mast.webui.agents_api`` backend
(works without a live runtime). The LIVE session state (threads / handoffs /
holds / interrupts / artifacts) lives on the running ``MASTApp`` instance — in
mounted mode integration wires read-only accessors onto ``ctx``
(``agents_snapshot`` / ``agents_interrupts`` / ``agents_artifacts``). When any
is absent or raises, the handler returns a valid empty/degraded response
(``degraded=True``) — NEVER 500. Heavy backends are lazy-imported INSIDE
handlers in try/except. NO business logic / safety here — the core owns it.
"""

from __future__ import annotations

import logging
import time as _time

from fastapi import APIRouter, Request

from mast.api.schemas_agents_topology import (
    AcknowledgeParkResponse,
    ActiveTask,
    AgentNode,
    AgentsCapabilities,
    AgentsSnapshotResponse,
    ArtifactEntry,
    ArtifactGroup,
    ArtifactPermission,
    ArtifactPermissionsResponse,
    ArtifactsResponse,
    BufferEvent,
    GoalCheckView,
    HandoffEvent,
    InterruptsResponse,
    PendingActivation,
    PendingActivationsResponse,
    PendingInterrupt,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents_topology"])

# Canonical roster (mirrors orchestrator._AGENT_NAMES + supervisor + the two
# non-graph side agents the inspector surfaces). Kept local so a missing agents
# runtime can never crash this read-only module on import.
_SUP_ID = "_supervisor"
_PIPELINE = (
    # 2026-08-21：research_director（科研策划，Campaign 层）。加在这里是因为
    # 同一个响应的产物权限那一半**已经**在报它（它派生自
    # ``artifacts.derive_flow()``），roster 不跟上就会出现「流程图里有这个节点，
    # 但 pipeline 里没有」的自相矛盾 —— 前端 ``agentDef()`` 对未知 id 有兜底，
    # 所以那种情况不会崩，只会静静地渲染成一个没有名字和颜色的灰节点。
    # 前端 ``components/agents/registry.tsx`` 的配色/中文名/图标还没补。
    "research_director",
    "literature",
    "experiment_design",
    "instrument_control",
    "data_processing",
    "paper_writing",
    "paper_review",
)
_AGENT_IDS = (_SUP_ID, *_PIPELINE, "buffer_summarizer")

# ── best-effort live hooks (wired at integration in mounted mode) ────────


def _agents_snapshot(ctx):
    """Read-only accessor → the live ``MASTApp`` agents snapshot dict (the exact
    payload the old ``/agents/snapshot`` returned). None in standalone dev /
    when no orchestrator session has run. Never raises."""
    hook = getattr(ctx, "agents_snapshot", None)
    if not callable(hook):
        return None
    try:
        snap = hook()
        return snap if isinstance(snap, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("agents_snapshot hook failed: %s", exc)
        return None


def _agents_interrupts(ctx):
    """Read-only accessor → the live pending-interrupts list
    (``MASTApp._snapshot_pending_interrupts``). None when no live gate. Never
    raises."""
    hook = getattr(ctx, "agents_interrupts", None)
    if not callable(hook):
        return None
    try:
        rows = hook()
        return rows if isinstance(rows, list) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("agents_interrupts hook failed: %s", exc)
        return None


def _agents_artifacts(ctx):
    """Read-only accessor → the live artifacts dict ``{produced, edits}`` (the
    old ``/artifacts`` payload). None in standalone dev. Never raises."""
    hook = getattr(ctx, "agents_artifacts", None)
    if not callable(hook):
        return None
    try:
        arts = hook()
        return arts if isinstance(arts, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("agents_artifacts hook failed: %s", exc)
        return None


def _waited_human(seconds: float) -> str:
    """"3 分钟" / "2 天" for a park row.

    Delegates to ``_shared.activation.humanize_age`` so the duration the UI shows and
    the duration the AGENT is told in its wake question are produced by one function.
    Two formatters would eventually disagree, and an operator comparing the panel with
    the transcript would have no way to tell which was right.
    """
    try:
        from mast.agents._shared.activation import humanize_age

        return humanize_age(seconds)
    except Exception:  # noqa: BLE001
        return ""


def _static_models() -> tuple[dict, dict]:
    """(models, thinking) from the kept ``webui.agents_api`` backend — works
    standalone (reads the AGENT_MODEL registry, no live runtime). ({}, {}) on
    any failure."""
    try:
        from mast.webui.agents_api import _resolve_agent_models

        models, thinking = _resolve_agent_models()
        return (models or {}), (thinking or {})
    except Exception as exc:  # noqa: BLE001
        logger.debug("static agent models unavailable: %s", exc)
        return {}, {}


# ── GET /api/agents/snapshot ───────────────────────────────────────────


@router.get("/agents/snapshot", response_model=AgentsSnapshotResponse)
def get_agents_snapshot(request: Request) -> AgentsSnapshotResponse:
    """Topology snapshot: per-agent registry+status, threads index, handoff-event
    timeline, buffer ticker, holds, interject/interrupt counts, active task, and
    honest capability flags.

    Static registry/model/thinking always resolve (kept ``webui.agents_api``);
    the live session fields overlay them when an agents runtime is wired onto
    ``ctx`` (``ctx.agents_snapshot``). Degrades to a registry-only snapshot
    (``degraded=True``) when no live runtime — never 500."""
    ctx = request.app.state.ctx

    models, thinking = _static_models()
    snap = _agents_snapshot(ctx)
    degraded = snap is None
    snap = snap or {}

    # Live snapshot can carry its own (more current) model/thinking — prefer it.
    models = {**models, **(snap.get("models") or {})}
    thinking = {**thinking, **(snap.get("thinking") or {})}
    holds = snap.get("holds") or {}
    threads_index = snap.get("threads_index") or {}
    active_agent_id = snap.get("active_agent_id")
    # Parallel fan-out: SEVERAL agents can be live at once, so "active" is a set.
    # Fall back to the single id when the core predates active_agents, so an
    # older snapshot still lights up its one agent.
    active_agents = [str(a) for a in (snap.get("active_agents") or []) if a]
    active_set = set(active_agents) or ({active_agent_id} if active_agent_id else set())

    agents = [
        AgentNode(
            id=aid,
            model=models.get(aid),
            thinking=thinking.get(aid),
            active=(aid in active_set),
            held=bool(holds.get(aid)),
            thread_count=int(threads_index.get(aid, 0) or 0),
        )
        for aid in _AGENT_IDS
    ]

    handoff_events = [
        HandoffEvent(t=e.get("t"), kind=e.get("kind"), text=str(e.get("text", "")),
                     targets=[str(t) for t in (e.get("targets") or [])])
        for e in (snap.get("handoff_events") or [])
        if isinstance(e, dict)
    ]
    buffer_events = [
        BufferEvent(t=e.get("t"), kind=e.get("kind"), text=str(e.get("text", "")))
        for e in (snap.get("buffer_events") or [])
        if isinstance(e, dict)
    ]

    caps_raw = snap.get("capabilities") or {}
    capabilities = AgentsCapabilities(
        interject=bool(caps_raw.get("interject")),
        hold=bool(caps_raw.get("hold")),
        model_switch_live=bool(caps_raw.get("model_switch_live")),
        buffer_summarizer_active=bool(caps_raw.get("buffer_summarizer_active")),
        abort=bool(caps_raw.get("abort")),
        tool_visibility=bool(caps_raw.get("tool_visibility")),
    )

    at = snap.get("active_task")
    active_task = (
        ActiveTask(
            id=at.get("id"),
            description=str(at.get("description", "")),
            active=bool(at.get("active")),
            final_text=str(at.get("final_text", "")),
            error=at.get("error"),
        )
        if isinstance(at, dict)
        else None
    )

    pending = snap.get("pending_interrupts") or []

    return AgentsSnapshotResponse(
        agents=agents,
        pipeline=list(_PIPELINE),
        models=models,
        thinking=thinking,
        holds={k: bool(v) for k, v in holds.items()},
        threads_index={k: int(v or 0) for k, v in threads_index.items()},
        handoff_events=handoff_events,
        buffer_events=buffer_events,
        interject_count=int(snap.get("interject_count", 0) or 0),
        pending_interrupt_count=len(pending) if isinstance(pending, list) else 0,
        interrupt_gating=bool(snap.get("interrupt_gating")),
        session_active=bool(snap.get("session_active")),
        active_agent_id=active_agent_id,
        active_agents=sorted(active_set),
        artifacts_count=int(snap.get("artifacts_count", 0) or 0),
        backend=str(snap.get("backend", "none")) if snap else "none",
        capabilities=capabilities,
        active_task=active_task,
        server_time=snap.get("server_time"),
        degraded=degraded,
    )


# ── GET /api/agents/{agent_id}/interrupts ──────────────────────────────


def _to_interrupt(row: dict) -> PendingInterrupt:
    """Map a live pending-interrupt dict → the typed model, stashing any extra
    keys in ``extra`` so nothing is silently dropped."""
    known = {
        "event_id",
        "kind",
        "agent_id",
        "skill",
        "params",
        "rationale",
        "allowed_decisions",
        "routes",
        "thread_id",
        "ask",
    }
    extra = {k: v for k, v in row.items() if k not in known}
    _ask = row.get("ask")
    return PendingInterrupt(
        event_id=str(row.get("event_id", "")),
        kind=row.get("kind"),
        agent_id=row.get("agent_id"),
        skill=row.get("skill"),
        params=row.get("params") or {},
        rationale=row.get("rationale"),
        allowed_decisions=[str(d) for d in (row.get("allowed_decisions") or [])],
        routes=[str(r) for r in (row.get("routes") or [])],
        thread_id=row.get("thread_id"),
        ask=_ask if isinstance(_ask, dict) else None,
        extra=extra,
    )


@router.get("/agents/{agent_id}/interrupts", response_model=InterruptsResponse)
def get_agent_interrupts(agent_id: str, request: Request) -> InterruptsResponse:
    """Pending HITL approvals awaiting the operator for *agent_id* (DANGEROUS
    skill gates + workflow-human nodes).

    Pass the roster sentinel ``__all__`` (or ``_supervisor``) to get every
    pending interrupt regardless of owner. Sourced from the live gate
    (``ctx.agents_interrupts`` → ``MASTApp._snapshot_pending_interrupts``).
    Degrades to an empty list when no live gate is wired — never 500."""
    ctx = request.app.state.ctx
    rows = _agents_interrupts(ctx)
    if rows is None:
        return InterruptsResponse(agent_id=agent_id, degraded=True)

    try:
        all_for = agent_id in ("__all__", _SUP_ID)
        interrupts = [
            _to_interrupt(r)
            for r in rows
            if isinstance(r, dict)
            and (all_for or r.get("agent_id") == agent_id)
        ]
        return InterruptsResponse(
            agent_id=agent_id,
            interrupts=interrupts,
            count=len(interrupts),
            interrupt_gating=True,
            degraded=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("interrupts read failed for %s: %s", agent_id, exc)
        return InterruptsResponse(agent_id=agent_id, degraded=True)


# ── GET /api/artifacts ─────────────────────────────────────────────────
#
# (The hand-written {agent: artifact} table and the hand-written flow list that
# used to live here are GONE. Both are now derived from the agents' real tool
# lists in mast.agents._shared.artifacts — a table maintained by hand is a table
# that drifts, and this one had drifted into pure fiction.)


@router.get("/artifacts", response_model=ArtifactsResponse)
def get_artifacts(request: Request) -> ArtifactsResponse:
    """The artifacts that ACTUALLY EXIST — enumerated from the real stores.

    Rewritten 2026-07-11. This used to read ``live_app._agents_api_state["task"]
    ["artifacts"]``, a dict that was initialised to ``{}`` in one place and then
    **never written by anything, anywhere in the tree**. So "produced artifacts"
    could only ever be empty — and a mock in the test-suite fed it fake data, so
    the suite stayed green while the feature was dead.

    It now lists what is on disk: the drafts and reviews the paper agents saved,
    the figures, the plans, and the scans the instrument saved (via
    ``core.scan_registry``). Every row is a file you can open. Empty means
    nothing has been produced yet — which is a true statement, not a broken one.

    Also returns ``groups`` — the per-CLASS status of all nine artifacts
    (2026-07-28, ). The class rows are the answer to "has this been
    produced yet?", and they must come from the backend: the UI used to derive
    them by subtracting produced ids from class ids, two id spaces that never
    intersect, so every class read as 待产出 forever — on a machine where six of
    the nine stores had content in them.
    """
    try:
        from mast.agents._shared.artifacts import class_status, list_existing

        rows = list_existing()
        groups = [
            ArtifactGroup(
                artifact_id=s.artifact.id,
                label=s.artifact.label,
                kind=s.artifact.kind,
                store=s.artifact.store,
                editable=s.artifact.editable,
                count=s.count,
                known=s.known,
                produced=s.produced,
                detail=s.detail,
                high_volume=s.artifact.high_volume,
            )
            # Share the one directory walk — class_status() would otherwise redo it.
            for s in class_status(rows)
        ]
        out = [
            ArtifactEntry(
                # id addresses THIS FILE ("draft:Au111_report_v002"); kind is its
                # CLASS ("draft"). They were both the class, so five drafts shared
                # one id and the editor could not tell them apart — it would open
                # whichever the backend happened to resolve.
                id=r["doc_id"],
                kind=r["artifact_id"],
                producer=None,
                edited=False,
                preview=r["name"],
                path=r["path"],
                bytes=int(r["bytes"]),
                modified_at=r["modified_at"],
                editable=bool(r["editable"]),
            )
            for r in rows
        ]
        return ArtifactsResponse(
            artifacts=out, groups=groups, count=len(out),
            session_active=bool(_agents_artifacts(request.app.state.ctx)),
            degraded=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("artifacts listing failed: %s", exc)
        return ArtifactsResponse(degraded=True)


# ── GET /api/artifacts/permissions ─────────────────────────────────────


@router.get("/artifacts/permissions", response_model=ArtifactPermissionsResponse)
def get_artifact_permissions(request: Request) -> ArtifactPermissionsResponse:
    """The artifact DATA-FLOW graph: who writes each artifact, who reads it.

    DERIVED, not declared (2026-07-11, ). The edges come from
    ``agents._shared.artifacts.derive_flow()``, which asks each agent which tools
    it actually holds and maps those tools to the artifacts they touch. Give an
    agent a tool and its edge appears here; take the tool away and the edge goes.
    There is no second table to keep in sync — which is precisely how the old
    hand-written matrix rotted into fiction (it claimed paper_review read the
    .sxm scan files, and named MASTState slots that are dead).

    Facts this derivation surfaces rather than hides:
      * experiment_plan now has TWO writers (2026-08-14): experiment_design
        drafts one, instrument_control advances it. This line used to say the
        design agent had no tool that could persist a plan — a real
        misallocation, shown rather than papered over. It was fixed (XD now
        holds ``create_plan`` via the single ``DESIGN_TOOL_NAMES`` constant that
        both runtime and this derivation read), and because the graph is derived
        rather than declared, the edge appeared here on its own. Approval is
        still nobody's tool: ``approve_plan`` is in no agent's surface;
      * several artifacts have MULTIPLE writers (figures, the experiment DB, the
        shared memory) — the case the single-writer model could not express;
      * the vision buffer has NO agent writer at all ("agents never write the
        buffer" is a standing invariant, and writers=[] can finally say so).

    Static — no live runtime needed, so it never degrades on an idle system."""
    try:
        from mast.agents._shared.artifacts import derive_flow

        permissions = [
            ArtifactPermission(
                artifact_id=f.artifact.id,
                label=f.artifact.label,
                kind=f.artifact.kind,
                store=f.artifact.store,
                writers=list(f.writers),
                readers=list(f.readers),
                multi_writer=f.multi_writer,
            )
            for f in derive_flow()
        ]
        return ArtifactPermissionsResponse(
            permissions=permissions,
            count=len(permissions),
            degraded=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("artifact flow build failed: %s", exc)
        return ArtifactPermissionsResponse(degraded=True)


# ── Pending activations (parked agents) — wakeup scheduling W3, 2026-07-30 ───
#
# WHY THIS ENDPOINT IS NOT OPTIONAL. The design that introduced parking says of
# itself: "这套机制最像一个漂亮的静默死亡通道" — its most likely failure is an agent
# that waits forever while the system looks idle and healthy. This repo has already
# paid for "nothing happened" and "it hung" being indistinguishable (the 2026-07-28
# fail-silent work; 621 events in the diagnostics ledger). So visibility is a
# PRECONDITION of the feature, not a follow-up: if this endpoint and its UI row are
# not there, the parking mechanism should not be switched on.
#
# Reads the durable board directly rather than a live MASTApp accessor, and that is
# deliberate: the whole point of a park is to outlive its run, so the case that
# matters most is exactly the one where no run — and possibly no runtime — exists.

@router.get("/agents/pending-activations", response_model=PendingActivationsResponse)
def pending_activations(request: Request) -> PendingActivationsResponse:
    """Every parked agent: what it waits for, how long, how often it has declined,
    and how much time is left before the wait escalates to the operator.

    Sweeps expired parks first. That is not a side effect snuck into a GET — it is
    what keeps the deadline honest when nothing else is running: with the whole
    system idle (the main case for parking) no other code path is awake to notice a
    deadline passing, and a deadline nobody evaluates is not a deadline.
    """
    try:
        from mast.agents._shared.artifact_channel import field_label, producer_of
        from mast.core.park_board import board

        b = board()
        b.sweep_expired()
        now = _time.time()
        rows: list[PendingActivation] = []
        # ``include_terminal=False`` 留的是 waiting/woken/expired。``done_by_goal``
        # 是终态，但**面板必须看得见它**：一个在等的 agent 某天悄悄不见了，正是
        # 这块面板存在的理由的反面（「什么都没发生」和「它挂了」不能长得一样）。
        # 它的 ``needs_attention`` 是 False，所以不会去抢红色那一栏。
        _rows_src = b.list_parks(include_terminal=False) + [
            r for r in b.list_parks(status="done_by_goal") ]
        for r in _rows_src:
            waiting = list(r.get("waiting_for") or [])
            deadline = float(r.get("deadline_at") or 0.0)
            created = float(r.get("created_at") or 0.0)
            expired_unack = (r.get("status") == "expired"
                             and not float(r.get("acknowledged_at") or 0))
            rows.append(PendingActivation(
                park_id=str(r.get("park_id") or ""),
                agent=str(r.get("agent") or ""),
                experiment_id=str(r.get("experiment_id") or ""),
                waiting_for=waiting,
                waiting_for_labels=[field_label(f) for f in waiting],
                blocked_by=(producer_of(waiting[0]) if waiting else ""),
                reason=str(r.get("reason") or ""),
                instruction=str(r.get("instruction") or ""),
                hard=bool(r.get("hard")),
                status=str(r.get("status") or "waiting"),
                created_at=created,
                deadline_at=deadline,
                seconds_left=(deadline - now) if deadline else 0.0,
                waited_human=_waited_human(now - created if created else 0.0),
                declines=int(r.get("declines") or 0),
                woken_run_id=str(r.get("woken_run_id") or ""),
                needs_attention=expired_unack,
                campaign_id=str(r.get("campaign_id") or ""),
                goal_check=GoalCheckView(**(r.get("goal_check") or {})),
            ))
        return PendingActivationsResponse(
            parks=rows, count=len(rows),
            attention_count=sum(1 for r in rows if r.needs_attention),
            degraded=False,
        )
    except Exception as exc:  # noqa: BLE001 — a status view never 500s
        logger.warning("pending activations read failed: %s", exc)
        return PendingActivationsResponse(degraded=True)


@router.post("/agents/pending-activations/{park_id}/acknowledge",
             response_model=AcknowledgeParkResponse)
def acknowledge_pending_activation(park_id: str,
                                   request: Request) -> AcknowledgeParkResponse:
    """Operator has seen an EXPIRED park; stop the row demanding attention.

    Acknowledging does not resume the agent and does not delete the row. Both
    omissions are deliberate: auto-resuming a wait the agent declined would override
    a decision it was asked to make, and deleting the record would erase the evidence
    that a wait timed out — which is the most useful thing about it.
    """
    try:
        from mast.core.park_board import board

        rec = board().acknowledge(park_id)
        if rec.get("error"):
            return AcknowledgeParkResponse(ok=False, park_id=park_id,
                                           message=str(rec["error"]))
        return AcknowledgeParkResponse(ok=True, park_id=park_id, message="已确认")
    except Exception as exc:  # noqa: BLE001
        logger.warning("acknowledge park %s failed: %s", park_id, exc)
        return AcknowledgeParkResponse(ok=False, park_id=park_id, message=str(exc))
