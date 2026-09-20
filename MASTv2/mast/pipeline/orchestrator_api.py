"""FastAPI scaffold for orchestrator agent communication — NOT YET WIRED.

STATUS (honest): this module is a *non-functional stub*. ``create_app`` builds
a FastAPI app whose handlers return correctly-typed placeholder responses
(``success=False`` / empty lists / 404s) but are NOT connected to any real
planner, executor, registry, or ExperimentStorage. The real HTTP surface for
v2 is the TypeScript SPA served by ``mast.api``; this REST API has not been adopted by
any caller yet (it is only re-exported from ``mast.pipeline`` for legacy
import compatibility).

Do not treat a 200 response from these endpoints as evidence that a mission /
skill actually ran — they don't. The endpoints exist so the schema contract
and auth/CORS middleware can be tested in isolation, and so the wiring can be
filled in later without changing the public surface.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mast.config import MASTConfig
from mast.pipeline.schemas import (
    MissionRequest,
    MissionResponse,
    SkillRequest,
    StatusResponse,
)


def _get_api_key(config: MASTConfig) -> str:
    """Resolve API key: config > environment variable > empty."""
    key = config.server.api_key
    if not key:
        key = os.environ.get("MAST_API_KEY", "")
    return key


def create_app(config: MASTConfig) -> FastAPI:
    """Create the FastAPI scaffold app — endpoints are PLACEHOLDERS.

    Every route below returns a correctly-typed but non-functional response:
    ``/mission`` and ``/skill`` always return ``success=False`` with
    ``"no executor configured"``, list endpoints return ``[]``, and the
    experiment endpoints are not backed by storage. The real executor /
    registry / planner / ExperimentStorage are meant to be injected later via
    ``app.state`` — until then this app does nothing useful beyond exercising
    the schema contract and the auth/CORS middleware. See the module docstring.
    """
    app = FastAPI(title="MAST API", version="0.1.0")

    # Attach config for use in handlers
    app.state.config = config
    app.state.active_experiment: str | None = None

    # ── CORS — default: localhost only ──────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1", "http://localhost"],
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    # ── API Key authentication middleware ───────────────────────
    api_key = _get_api_key(config)

    @app.middleware("http")
    async def verify_api_key(request: Request, call_next):
        # /health is always public
        if request.url.path == "/health":
            return await call_next(request)
        # If no API key configured, allow all (localhost-only use)
        if not api_key:
            return await call_next(request)
        # Check X-API-Key header
        provided = request.headers.get("X-API-Key", "")
        if provided != api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)

    # ── POST /mission ────────────────────────────────────────────

    @app.post("/mission", response_model=MissionResponse)
    async def execute_mission(req: MissionRequest) -> MissionResponse:
        """STUB: does NOT execute anything — always returns success=False.

        A real implementation would call planner + executor; this scaffold is
        not wired to either, so it returns a not-implemented response.
        """
        return MissionResponse(
            success=False,
            summary="Mission execution not yet implemented",
            error="no executor configured",
        )

    # ── POST /skill ──────────────────────────────────────────────

    @app.post("/skill")
    async def execute_skill(req: SkillRequest) -> dict:
        """STUB: does NOT execute the skill — always returns success=False.

        No executor is wired to this scaffold. (Security note kept for when it
        is: the server always controls approval_source and never trusts the
        client-supplied value.)
        """
        # Security: server always controls approval_source, never trust client
        effective_approval = "auto"
        return {
            "success": False,
            "skill_name": req.skill_name,
            "approval_source": effective_approval,
            "error": "no executor configured",
        }

    # ── GET /status ──────────────────────────────────────────────

    @app.get("/status", response_model=StatusResponse)
    async def get_status() -> StatusResponse:
        """STUB: always reports connected=False (no hardware wired here)."""
        return StatusResponse(
            connected=False,
            active_experiment=app.state.active_experiment,
        )

    # ── GET /skills ──────────────────────────────────────────────

    @app.get("/skills")
    async def list_skills() -> list[dict]:
        """STUB: no registry wired — always returns an empty list."""
        return []

    # ── Experiments ──────────────────────────────────────────────

    @app.get("/experiments")
    async def list_experiments() -> list[dict]:
        """STUB: no storage wired — always returns an empty list."""
        return []

    @app.get("/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str) -> dict:
        """Get experiment details."""
        raise HTTPException(status_code=404, detail="Experiment not found")

    @app.post("/experiment/start")
    async def start_experiment(name: str = "unnamed", goal: str = "") -> dict:
        """Start a new experiment."""
        # Placeholder — real implementation uses ExperimentStorage
        return {"experiment_id": "", "error": "storage not configured"}

    @app.post("/experiment/end")
    async def end_experiment() -> dict:
        """End the current experiment."""
        if app.state.active_experiment is None:
            raise HTTPException(status_code=400, detail="No active experiment")
        return {"ended": app.state.active_experiment}

    # ── GET /health ──────────────────────────────────────────────

    @app.get("/health")
    async def health_check() -> dict:
        """Health check."""
        return {"status": "ok", "version": "0.1.0"}

    return app


def start_server(config: MASTConfig, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the MAST API server using uvicorn."""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)
