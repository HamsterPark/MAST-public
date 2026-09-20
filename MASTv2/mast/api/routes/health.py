"""GET /api/health — liveness + which optional backends are wired."""

from __future__ import annotations

from fastapi import APIRouter, Request

from mast.api.schemas import HealthResponse
from mast.api.version import get_version

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    ctx = request.app.state.ctx
    return HealthResponse(
        version=get_version(),
        skill_registry_wired=ctx.skill_registry is not None,
        settings_store_wired=ctx.settings_store is not None,
        experiment_storage_wired=ctx.experiment_storage is not None,
    )
