"""GET /api/experiments — the experiments list (Records tab top level).

Phase 2: degrades to empty in standalone dev (no ExperimentStorage wired).
Phase 3 wires the live storage. Full multi-level records drill-down
(campaigns → experiments → actions → observations) lands with the Records
migration in Phase 4.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.schemas import ExperimentsResponse, ExperimentSummary

logger = logging.getLogger(__name__)

router = APIRouter(tags=["experiments"])


@router.get("/experiments", response_model=ExperimentsResponse)
def list_experiments(request: Request) -> ExperimentsResponse:
    ctx = request.app.state.ctx
    storage = ctx.experiment_storage
    if storage is None:
        return ExperimentsResponse(degraded=True)

    try:
        rows = storage.list_experiments()  # best-effort; exact API wired in Phase 3
        experiments = [
            ExperimentSummary(
                id=str(r.get("id")),
                name=r.get("name"),
                # DB column is `goal_text` (storage.py); the old `goal` key never
                # existed → every experiment's goal came back null (review
                # 2026-07-03). Fall back to `goal` for any alternate row shape.
                goal=r.get("goal_text") if r.get("goal_text") is not None else r.get("goal"),
                status=r.get("status"),
                start_time=str(r.get("start_time")) if r.get("start_time") is not None else None,
                sample_name=r.get("sample_name"),
                sample_id=(str(r["sample_id"]) if r.get("sample_id") else None),
            )
            for r in (rows or [])
            if r.get("id") is not None
        ]
        return ExperimentsResponse(experiments=experiments, count=len(experiments), degraded=False)
    except Exception as exc:
        logger.warning("experiments list failed: %s", exc)
        return ExperimentsResponse(degraded=True)
