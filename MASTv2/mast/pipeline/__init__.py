"""mast.pipeline — top-level entry points + orchestrator API (v1 parity).

main.py exposes `python -m mast`, which builds the graph. (The Gradio UI it
used to launch was removed in the TS rewrite.) MAST_ARCH is NOT used in v2 —
this package is v2 by construction; v1 was archived out of the repo
2026-06-01.

Also re-exports v1's orchestrator HTTP API (FastAPI) and pydantic schemas so
v1 scripts/integrations targeting `mast.pipeline.create_app` etc. continue
to work under v2 (feature-parity policy, 2026-05-18).
"""

from mast.pipeline.cloud_sync import CloudSync
from mast.pipeline.orchestrator_api import create_app, start_server
from mast.pipeline.schemas import (
    MissionRequest,
    MissionResponse,
    SkillRequest,
    StatusResponse,
)

__all__ = [
    "CloudSync",
    "create_app",
    "start_server",
    "MissionRequest",
    "MissionResponse",
    "SkillRequest",
    "StatusResponse",
]
