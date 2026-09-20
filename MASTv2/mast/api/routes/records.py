"""Domain D — experiment detail, records drill-down, and operator feedback.

This is the read+write contract for the Records/Experiments detail seam. It is
ADDITIVE to the existing ``GET /api/experiments`` list (routes/experiments.py):
here we add the detail drill-down (samples/actions/map_markers/feedback), the
experiment CRUD lifecycle (create/end/add-sample), the richer v2 campaigns +
action drill-down, and the operator-feedback write.

GRACEFUL DEGRADATION is mandatory (house rule 2): the API must boot standalone
with no live core wired. Every endpoint checks ``ctx`` for the subsystem it
needs and, if absent or if any call raises, returns a valid empty/degraded
response (``degraded=True``) — never a 500. Heavy core modules
(gui.records_api, logging.experiment_log) are lazy-imported INSIDE handlers,
wrapped in try/except, exactly like routes/skills.py imports builder_api.

WRITE endpoints (house rule 4) are defined for contract completeness and also
degrade safely: with no live core they return a typed result with
``ok=false, degraded=true``. NO business logic or safety checks live here — the
handlers only call into the core; real wiring to live singletons + the safety
passthrough is integrated later at integration.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request

from mast.api.schemas_records import (
    ActionDetail,
    ActionSummary,
    CampaignStats,
    CampaignSummary,
    CampaignsResponse,
    CreateExperimentRequest,
    CreateExperimentResult,
    CreateSampleRequest,
    CreateSampleResult,
    EndExperimentRequest,
    EndExperimentResult,
    ExperimentDetail,
    FeedbackEntry,
    FeedbackList,
    FeedbackResolveRequest,
    FeedbackRequest,
    FeedbackResult,
    MapMarker,
    SampleSummary,
    TipInService,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["records"])


# ── helpers ────────────────────────────────────────────────────────────


#: Cap on the serialized `data` payload surfaced per action. An action's result
#: can be a whole spectrum; a timeline of a few hundred of those would be
#: megabytes of JSON for a table the operator mostly scans. Over the cap we send
#: a summary and set data_truncated — the full payload is in the record, and the
#: export path (records_export.timeline) carries it uncapped.
_ACTION_DATA_MAX_CHARS = 4000


def _action_payload(action) -> dict:
    """Surface what the skill returned: `data`, `artifact_path`, call count.

    These were empty for the whole 2026-07-27 session () and are
    populated now — but this route flattened them away, so the fix was invisible
    to the operator, who is the person the record exists for.

    Never raises: a malformed payload must degrade to an empty dict, not take
    the whole timeline down with it.
    """
    import json as _json

    out: dict = {"data": {}, "artifact_path": None,
                 "nanonis_calls": 0, "data_truncated": False}
    try:
        calls = getattr(action, "nanonis_calls", None)
        out["nanonis_calls"] = len(calls) if isinstance(calls, (list, tuple)) else 0
        ap = getattr(action, "artifact_path", None)
        out["artifact_path"] = str(ap) if ap else None
        raw = getattr(action, "data", None)
        if isinstance(raw, dict) and raw:
            try:
                # Round-trip through JSON so what leaves here is guaranteed
                # serializable. A skill result can hold anything (a Path, a numpy
                # scalar, a client handle); handing that straight to Pydantic
                # would 500 the WHOLE timeline over one bad action — the record
                # exists precisely for the runs that went wrong.
                blob = _json.dumps(raw, ensure_ascii=False, default=str)
                size = len(blob)
                safe = _json.loads(blob)
            except Exception:  # noqa: BLE001 — unserializable payload
                size = _ACTION_DATA_MAX_CHARS + 1
                safe = {}
            if size <= _ACTION_DATA_MAX_CHARS:
                out["data"] = safe if isinstance(safe, dict) else {}
            else:
                out["data"] = {
                    "_summary": f"{len(raw)} 个字段，约 {size} 字符（超出内联上限）",
                    "_keys": sorted(str(k) for k in raw)[:40],
                }
                out["data_truncated"] = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("action payload surface failed: %s", exc)
    return out


def _action_success(action) -> bool | None:
    """Best-effort success flag off an ActionRecord (mast.core.types)."""
    result = getattr(action, "result", None)
    if result is None:
        return None
    return bool(getattr(result, "success", False))


def _action_error(action) -> str | None:
    result = getattr(action, "result", None)
    if result is None:
        return None
    err = getattr(result, "error", "") or ""
    return err or None


# ── GET /api/experiments/{experiment_id} ───────────────────────────────


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment_detail(experiment_id: str, request: Request) -> ExperimentDetail:
    """Full experiment detail: header + samples + actions + map markers +
    operator feedback. Mirrors gui.experiment_viewer.build_experiment_detail_html
    over the v1 ExperimentStorage."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return ExperimentDetail(id=experiment_id, degraded=True)

    try:
        exp = storage.get_experiment(experiment_id)
    except Exception as exc:
        logger.warning("experiment detail lookup failed: %s", exc)
        return ExperimentDetail(id=experiment_id, degraded=True)

    if not exp:
        # Storage is wired but the id is unknown — empty, not broken, not 500.
        return ExperimentDetail(id=experiment_id, found=False, degraded=False)

    # Samples (+ per-sample action counts, like the detail HTML builder).
    samples: list[SampleSummary] = []
    try:
        for s in storage.get_samples(experiment_id) or []:
            sid = s.get("id")
            sa_count = 0
            try:
                sa_count = len(storage.get_actions(experiment_id, sample_id=sid) or [])
            except Exception:
                sa_count = 0
            samples.append(
                SampleSummary(
                    id=str(sid),
                    experiment_id=s.get("experiment_id"),
                    name=s.get("name"),
                    description=s.get("description"),
                    status=s.get("status"),
                    start_time=s.get("start_time"),
                    end_time=s.get("end_time"),
                    sample_type=s.get("sample_type"),
                    sample_subtype=s.get("sample_subtype"),
                    action_count=sa_count,
                )
            )
    except Exception as exc:
        logger.warning("experiment samples lookup failed: %s", exc)

    # Actions (flattened ActionRecord timeline).
    actions: list[ActionSummary] = []
    try:
        for a in storage.get_actions(experiment_id) or []:
            actions.append(
                ActionSummary(
                    id=str(getattr(a, "id", "")),
                    experiment_id=getattr(a, "experiment_id", None) or None,
                    sample_id=getattr(a, "sample_id", None) or None,
                    timestamp=getattr(a, "timestamp", None),
                    skill_name=getattr(a, "skill_name", None),
                    skill_version=getattr(a, "skill_version", None) or None,
                    parameters=getattr(a, "parameters", None) or {},
                    success=_action_success(a),
                    error=_action_error(a),
                    duration_s=float(getattr(a, "duration_s", 0.0) or 0.0),
                    context=getattr(a, "context", None) or None,
                    approval_source=getattr(a, "approval_source", None) or None,
                    **_action_payload(a),
                )
            )
    except Exception as exc:
        logger.warning("experiment actions lookup failed: %s", exc)

    # Map markers (this map IS the experiment record's map).
    map_markers: list[MapMarker] = []
    try:
        for m in storage.get_markers(experiment_id=experiment_id) or []:
            map_markers.append(
                MapMarker(
                    id=int(m.get("id")),
                    experiment_id=m.get("experiment_id"),
                    sample_id=m.get("sample_id"),
                    kind=m.get("kind", "move"),
                    skill_name=m.get("skill_name", "") or "",
                    x_m=m.get("x_m"),
                    y_m=m.get("y_m"),
                    w_m=m.get("w_m"),
                    h_m=m.get("h_m"),
                    angle_deg=float(m.get("angle_deg", 0.0) or 0.0),
                    label=m.get("label", "") or "",
                    status=m.get("status", "done"),
                    source=m.get("source", "skill"),
                    timestamp=m.get("timestamp"),
                    meta=m.get("meta") or {},
                )
            )
    except Exception as exc:
        logger.warning("experiment markers lookup failed: %s", exc)

    # Operator feedback recorded alongside this experiment record.
    feedback: list[FeedbackEntry] = []
    try:
        for f in storage.get_feedback(experiment_id=experiment_id) or []:
            feedback.append(
                FeedbackEntry(
                    id=int(f.get("id")),
                    experiment_id=f.get("experiment_id"),
                    sample_id=f.get("sample_id"),
                    conversation_id=f.get("conversation_id"),
                    kind=f.get("kind", "rating"),
                    rating=f.get("rating", "") or "",
                    comment=f.get("comment", "") or "",
                    agent=f.get("agent", "") or "",
                    timestamp=f.get("timestamp"),
                    meta=f.get("meta") or {},
                )
            )
    except Exception as exc:
        logger.warning("experiment feedback lookup failed: %s", exc)

    # 本实验期间在役的针尖。**展示层聚合** —— tips 表仍是针尖的唯一
    # 真源，这里一个字都不写，只是把两张表本来就答得出来的问题答出来。
    # 与上面每一块一样单独 try：查不到针尖不该让整个实验详情变成 degraded。
    tips: list[TipInService] = []
    try:
        for t in storage.tips_in_service_during(experiment_id) or []:
            tips.append(
                TipInService(
                    id=str(t.get("id") or ""),
                    tip_index=t.get("tip_index"),
                    name=str(t.get("name") or ""),
                    material=str(t.get("material") or ""),
                    fabrication=str(t.get("fabrication") or ""),
                    form=str(t.get("form") or ""),
                    installed_at=t.get("installed_at"),
                    removed_at=t.get("removed_at"),
                    in_service_now=not t.get("removed_at"),
                    overlap_exact=bool(t.get("overlap_exact", True)),
                )
            )
    except Exception as exc:
        logger.warning("experiment tips lookup failed: %s", exc)

    return ExperimentDetail(
        id=experiment_id,
        name=exp.get("name"),
        goal_text=exp.get("goal_text"),
        status=exp.get("status"),
        start_time=exp.get("start_time"),
        end_time=exp.get("end_time"),
        notes=exp.get("notes"),
        samples=samples,
        actions=actions,
        map_markers=map_markers,
        feedback=feedback,
        tips=tips,
        found=True,
        degraded=False,
    )


# ── POST /api/experiments (create; id=thread_id) ───────────────────────


@router.post("/experiments", response_model=CreateExperimentResult)
def create_experiment(body: CreateExperimentRequest, request: Request) -> CreateExperimentResult:
    """Start a new experiment session. The live ExperimentLog assigns the id
    (id=thread_id contract). Degrades safely with no core wired."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return CreateExperimentResult(ok=False, degraded=True)

    try:
        # Prefer the high-level lifecycle (lazy-imported heavy core): it tracks
        # the active experiment/sample and clears the scan-map plan overlay.
        from mast.logging.experiment_log import get_active_log  # type: ignore

        log = get_active_log()
        if log is not None:
            new_id = log.start_experiment(body.name, body.goal)
            return CreateExperimentResult(ok=True, id=str(new_id), degraded=False)

        # No active ExperimentLog — fall back to the storage primitive.
        new_id = storage.create_experiment(body.name, body.goal)
        return CreateExperimentResult(ok=True, id=str(new_id), degraded=False)
    except Exception as exc:
        logger.warning("create experiment failed: %s", exc)
        return CreateExperimentResult(ok=False, degraded=True)


# ── POST /api/experiments/{experiment_id}/end ──────────────────────────


@router.post("/experiments/{experiment_id}/end", response_model=EndExperimentResult)
def end_experiment(
    experiment_id: str, body: EndExperimentRequest, request: Request
) -> EndExperimentResult:
    """End an experiment (auto-ends its active sample first via ExperimentLog).
    Degrades safely with no core wired."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return EndExperimentResult(ok=False, id=experiment_id, degraded=True)

    try:
        from mast.logging.experiment_log import get_active_log  # type: ignore

        log = get_active_log()
        # Only the high-level log knows the active id; it ends sample-then-exp.
        if log is not None and log.current_experiment_id == experiment_id:
            log.end_experiment(body.status)
        else:
            storage.end_experiment(experiment_id, body.status)
        return EndExperimentResult(
            ok=True, id=experiment_id, status=body.status, degraded=False
        )
    except Exception as exc:
        logger.warning("end experiment failed: %s", exc)
        return EndExperimentResult(ok=False, id=experiment_id, degraded=True)


# ── POST /api/experiments/{experiment_id}/samples ──────────────────────


@router.post("/experiments/{experiment_id}/samples", response_model=CreateSampleResult)
def create_sample(
    experiment_id: str, body: CreateSampleRequest, request: Request
) -> CreateSampleResult:
    """Start a new sample under an experiment. Degrades safely with no core
    wired."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return CreateSampleResult(ok=False, experiment_id=experiment_id, degraded=True)

    try:
        from mast.logging.experiment_log import get_active_log  # type: ignore

        log = get_active_log()
        if log is not None and log.current_experiment_id == experiment_id:
            new_id = log.start_sample(
                body.name,
                body.description,
                sample_type=body.sample_type,
                sample_subtype=body.sample_subtype,
            )
        else:
            new_id = storage.create_sample(
                experiment_id,
                body.name,
                body.description,
                sample_type=body.sample_type,
                sample_subtype=body.sample_subtype,
            )
        return CreateSampleResult(
            ok=True, id=str(new_id), experiment_id=experiment_id, degraded=False
        )
    except Exception as exc:
        logger.warning("create sample failed: %s", exc)
        return CreateSampleResult(ok=False, experiment_id=experiment_id, degraded=True)


# ── POST /api/experiments/{experiment_id}/samples/{sample_id}/end ──────
# Restores the old 设置/右栏 sample End button (storage.end_sample exists; only
# the route was missing). Reuses the experiment end-result shape.
@router.post("/experiments/{experiment_id}/samples/{sample_id}/end",
             response_model=EndExperimentResult)
def end_sample(
    experiment_id: str, sample_id: str, body: EndExperimentRequest, request: Request
) -> EndExperimentResult:
    """End a sample (sets end_time + status). Degrades safely with no core wired."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return EndExperimentResult(ok=False, id=sample_id, degraded=True)
    try:
        from mast.logging.experiment_log import get_active_log  # type: ignore

        log = get_active_log()
        # Use the high-level log ONLY when the URL's sample IS the log's current
        # active sample — otherwise log.end_sample() would end whatever sample is
        # active, NOT the one the URL named. For any other
        # sample, end it by id directly so the correct row is closed.
        if (log is not None
                and log.current_experiment_id == experiment_id
                and log.current_sample_id == sample_id):
            log.end_sample(body.status)
        else:
            storage.end_sample(sample_id, body.status)
        return EndExperimentResult(
            ok=True, id=sample_id, status=body.status, degraded=False
        )
    except Exception as exc:
        logger.warning("end sample failed: %s", exc)
        return EndExperimentResult(ok=False, id=sample_id, degraded=True)


# ── GET /api/records/campaigns ─────────────────────────────────────────


@router.get("/records/campaigns", response_model=CampaignsResponse)
def list_campaigns(
    request: Request,
    status: str | None = Query(default=None),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    agent: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> CampaignsResponse:
    """Campaigns drill-down (the richer v2 ExperimentStoreV2 payload shaped by
    gui.records_api). Filters (status/from/to/agent) + pagination are applied
    over the built payload. Degrades to an empty page when the v2 store is
    absent/empty."""
    try:
        # Lazy-import the heavy v2 records adapter, exactly like skills.py.
        from mast.webui.records_api import build_records_payload  # type: ignore

        payload = build_records_payload()
    except Exception as exc:
        logger.warning("records campaigns build failed: %s", exc)
        return CampaignsResponse(page=page, page_size=page_size, degraded=True)

    raw = (payload or {}).get("campaigns") or []
    if not raw:
        # Empty / unwired v2 store → empty-but-not-broken page.
        return CampaignsResponse(page=page, page_size=page_size, degraded=True)

    # Filters (presentation-only; the store remains the source of truth).
    def _keep(c: dict) -> bool:
        if status and (c.get("status") or "") != status:
            return False
        created = c.get("created_at") or ""
        if from_ and created and created < from_:
            return False
        if to and created and created > to:
            return False
        if agent and (c.get("created_by") or "") != agent:
            return False
        return True

    filtered = [c for c in raw if _keep(c)]
    total = len(filtered)
    start = (page - 1) * page_size
    window = filtered[start : start + page_size]

    campaigns: list[CampaignSummary] = []
    for c in window:
        stats = c.get("stats") or {}
        campaigns.append(
            CampaignSummary(
                id=str(c.get("id")),
                title=c.get("title"),
                hypothesis=c.get("hypothesis"),
                hypothesis_kind=c.get("hypothesis_kind"),
                status=c.get("status"),
                created_at=c.get("created_at"),
                created_by=c.get("created_by", "") or "",
                parent_campaign_id=c.get("parent_campaign_id"),
                stats=CampaignStats(
                    experiments=int(stats.get("experiments", 0) or 0),
                    actions=int(stats.get("actions", 0) or 0),
                    observations=int(stats.get("observations", 0) or 0),
                    scans=int(stats.get("scans", 0) or 0),
                ),
            )
        )

    return CampaignsResponse(
        campaigns=campaigns,
        count=total,
        page=page,
        page_size=page_size,
        degraded=False,
    )


# ── GET /api/records/actions/{action_id} ───────────────────────────────


def _action_detail_from_v2(match: dict) -> ActionDetail:
    """Map a v2 action shape (records_api ``_actions`` row OR a freshly mapped
    raw store row) onto ActionDetail. Single mapping so the focus-payload hit
    and the cross-experiment fallback return an IDENTICAL shape."""
    return ActionDetail(
        id=str(match.get("id")),
        experiment_id=match.get("experiment_id"),
        parent_action_id=match.get("parent_action_id"),
        agent_id=match.get("agent_id"),
        action_type=match.get("action_type"),
        params=match.get("params") or {},
        hlc=match.get("hlc"),
        status=match.get("status"),
        duration_ms=match.get("duration_ms"),
        state_delta=match.get("state_delta"),
        error=match.get("error"),
        found=True,
        degraded=False,
    )


def _find_action_across_experiments(action_id: str) -> ActionDetail | None:
    """Resolve an action by id across ALL experiments via a direct v2-store
    lookup (ActionRepo.get), independent of the focus experiment.

    The focus payload only carries the focus experiment's actions, so an action
    from any other experiment's timeline is absent there. This goes straight to
    the store and maps the raw row with the SAME transforms records_api._actions
    uses (hlc_display + _loads). Returns None when the store is absent/empty or
    the id is unknown so the caller can degrade safely; never raises.
    """
    try:
        # Lazy-import the heavy v2 store/repos, exactly like records_api does.
        from mast.logging.v2.repos import build_repos  # type: ignore
        from mast.logging.v2.storage import open_store  # type: ignore
        from mast.webui.records_api import _loads, hlc_display  # type: ignore

        store = open_store()
        repos = build_repos(store)
        row = repos.actions.get(action_id)
    except Exception as exc:
        logger.warning("records action cross-experiment lookup failed: %s", exc)
        return None

    if not row:
        return None

    # Raw DB row → the v2 action shape records_api._actions emits.
    return _action_detail_from_v2(
        {
            "id": row.get("id"),
            "experiment_id": row.get("experiment_id"),
            "parent_action_id": row.get("parent_action_id"),
            "agent_id": row.get("agent_id"),
            "action_type": row.get("action_type"),
            "params": _loads(row.get("params_json"), {}),
            "hlc": hlc_display(row.get("hlc")),
            "status": row.get("status"),
            "duration_ms": row.get("duration_ms"),
            "state_delta": _loads(row.get("state_delta_json"), None),
            "error": row.get("error"),
        }
    )


@router.get("/records/actions/{action_id}", response_model=ActionDetail)
def get_action_detail(action_id: str, request: Request) -> ActionDetail:
    """Single action detail for the records drill-down. Prefers the richer v2
    payload (gui.records_api action shape); if the id is not in the
    currently-focused experiment's action subset, falls back to a direct v2
    store lookup so an action from ANY experiment's timeline still resolves.
    Degrades to not-found when no store is wired or the id is unknown."""
    payload: dict | None
    try:
        from mast.webui.records_api import build_records_payload  # type: ignore

        payload = build_records_payload()
    except Exception as exc:
        logger.warning("records action build failed: %s", exc)
        payload = None

    if payload is not None:
        raw_actions = (payload or {}).get("actions") or []
        match = next((a for a in raw_actions if str(a.get("id")) == action_id), None)
        if match is not None:
            return _action_detail_from_v2(match)

    # Not in the focus payload (or payload build failed) — the action may belong
    # to another experiment. Resolve it by id across ALL experiments.
    cross = _find_action_across_experiments(action_id)
    if cross is not None:
        return cross

    # Store unreachable entirely → degraded; reachable but id unknown → found=False.
    if payload is None:
        return ActionDetail(id=action_id, degraded=True)
    return ActionDetail(id=action_id, found=False, degraded=False)


# ── POST /api/feedback ─────────────────────────────────────────────────


@router.post("/feedback", response_model=FeedbackResult)
def post_feedback(body: FeedbackRequest, request: Request) -> FeedbackResult:
    """Record one operator-feedback row alongside the experiment record
    (mast.logging.storage.log_feedback). A rating tap and a free comment are
    independent. Degrades safely with no storage wired."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return FeedbackResult(ok=False, degraded=True)

    # Tie feedback to the ACTIVE experiment when the caller names none — the
    # floating feedback widget sends no experiment_id, which left every row
    # orphaned (experiment_id=None) and invisible in the per-experiment Records
    # drill-down (it filters by experiment_id) even while an experiment was
    # running (2026-07-06 feedback: saved feedback couldn't be found anywhere).
    exp_id = body.experiment_id
    sample_id = body.sample_id
    conv_id = body.conversation_id
    if not exp_id or not sample_id:
        try:
            from mast.logging.experiment_log import get_active_log  # type: ignore

            log = get_active_log()
            if log is not None:
                exp_id = exp_id or getattr(log, "current_experiment_id", None)
                # Same reasoning as experiment_id above, and the same evidence:
                # the floating widget sends no sample_id either, so EVERY row was
                # unscoped — 2026-07-27 forensics found sample_id empty on 36 of
                # 36 feedback rows. A comment about a bad scan that cannot be
                # traced to the sample it was about is much less useful later.
                sample_id = sample_id or getattr(log, "current_sample_id", None)
        except Exception:
            pass

    if not conv_id:
        # And the conversation. 28 of 36 rows had none; the 8 that DID were all
        # pointing at the same conversation — one created three days earlier with
        # zero messages in it — so the client-supplied value was not merely
        # missing but wrong. The live task slot knows which conversation is
        # actually on screen, so resolve it here rather than trusting the caller.
        try:
            live = getattr(request.app, "_live_app", None) or request.app
            st = getattr(live, "_agents_api_state", None)
            task = st.get("task") if isinstance(st, dict) else None
            if isinstance(task, dict) and task.get("active"):
                conv_id = str(task.get("conversation_id") or "") or None
        except Exception:
            pass

    try:
        new_id = storage.log_feedback(
            rating=body.rating,
            comment=body.comment,
            experiment_id=exp_id,
            sample_id=sample_id,
            conversation_id=conv_id,
            agent=body.agent,
            meta=body.meta,
        )
        kind = "rating" if body.rating else "comment"
        return FeedbackResult(ok=True, id=int(new_id), kind=kind, degraded=False)
    except Exception as exc:
        logger.warning("post feedback failed: %s", exc)
        return FeedbackResult(ok=False, degraded=True)


@router.get("/feedback", response_model=FeedbackList)
def list_feedback(request: Request, limit: int = 200) -> FeedbackList:
    """Global operator-feedback list (newest first) across ALL experiments —
    INCLUDING rows with no experiment_id. The per-experiment drill-down
    (get_experiment_detail) filters by experiment_id and so never surfaced the
    floating-widget feedback (which is unscoped — 2026-07-06: 31 saved rows were
    invisible in the UI). This is the single place the UI can list everything.
    Degrades to empty with no storage wired."""
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return FeedbackList(items=[], degraded=True)
    try:
        rows = storage.get_feedback(experiment_id=None, limit=limit) or []
        items = [
            FeedbackEntry(
                id=int(f.get("id")),
                experiment_id=f.get("experiment_id"),
                sample_id=f.get("sample_id"),
                conversation_id=f.get("conversation_id"),
                kind=f.get("kind", "rating"),
                rating=f.get("rating", "") or "",
                comment=f.get("comment", "") or "",
                agent=f.get("agent", "") or "",
                # The page the operator was on when they filed it. The frontend
                # sends it and the DB stores it, but this listing dropped it on
                # the floor — so the one endpoint that shows ALL feedback was the
                # one that could not say where any of it came from . The
                # per-experiment route already passes it through.
                meta=f.get("meta") or {},
                timestamp=f.get("timestamp"),
                resolved_at=f.get("resolved_at"),
                resolved_version=f.get("resolved_version") or "",
                resolved_note=f.get("resolved_note") or "",
            )
            for f in rows
        ]
        return FeedbackList(items=items, degraded=False)
    except Exception as exc:
        logger.warning("list feedback failed: %s", exc)
        return FeedbackList(items=[], degraded=True)


@router.post("/feedback/{feedback_id}/resolved", response_model=FeedbackResult)
def resolve_feedback(request: Request, feedback_id: int,
                     body: FeedbackResolveRequest) -> FeedbackResult:
    """Mark one feedback row processed (or un-mark it).

    The feedback list has no created-at anyone can see and had no processed
    flag, so old already-closed entries sat mixed in with new ones and every
    batch opened with an archaeology pass over git log / KNOWN_ISSUES / the
    machine-test checklists. One batch got that wrong in both directions: an
    item that had been fixed was re-reported, and an item assumed fixed turned
    out to be half-fixed.

    Never 500s, and never reports success it did not have: an id that does not
    exist comes back ``ok=False`` rather than a cheerful no-op — 「标过了」 and
    「那一行不在」 have to stay distinguishable, or the marker inherits exactly
    the ambiguity it was added to remove.
    """
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return FeedbackResult(ok=False, degraded=True)
    try:
        changed = storage.set_feedback_resolved(
            int(feedback_id), resolved=bool(body.resolved),
            version=body.version or "", note=body.note or "",
        )
        return FeedbackResult(ok=bool(changed), id=int(feedback_id))
    except Exception as exc:  # noqa: BLE001 — a marker must never take the page down
        logger.warning("resolve feedback %s failed: %s", feedback_id, exc)
        return FeedbackResult(ok=False, degraded=True)
