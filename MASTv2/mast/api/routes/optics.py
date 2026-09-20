"""Optical-bench endpoints for the 光学台 tab.

Thin relays over the existing optics skills (``mast/skills/builtins/optics_*``)
which drive the process-global instrument registry
(``mast.instruments.get_instrument_registry``). No hardware logic lives here.

House style (mirrors routes/qa.py, routes/signals.py):
  - handlers take ``request: Request`` but the optics skills reach hardware
    through the registry SINGLETON, so no ExecutionContext / ctx wiring is
    needed — the panel is manual operator control, not the autonomous agent
    path;
  - ``def`` handlers (FastAPI runs them in a threadpool) so a blocking serial
    move never stalls the event loop;
  - GRACEFUL DEGRADATION is mandatory: no bench / no hardware returns an empty
    inventory or an ``ok=false`` envelope — never a 500. Soft travel limits are
    enforced in the driver (Layer-0), so a bad target comes back as
    ``ok=false`` too;
  - the pump-probe DRY RUN is allowed here because that path touches only the
    delay-line registry (no Nanonis) — a real acquisition scan (Nanonis reads)
    stays on the skill/agent path and is not exposed as a manual endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_optics import (
    DelayLineInfo,
    DelayMoveRequest,
    DryRunRequest,
    HomeRequest,
    MoveRequest,
    OpticsActionResponse,
    OpticsAxis,
    OpticsDevice,
    OpticsDevicesResponse,
    StopRequest,
    WiggleRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["optics"])


def _run(skill_cls, params: dict) -> OpticsActionResponse:
    """Execute one optics skill and wrap its SkillResult in the envelope.

    The optics skills ignore ``context`` (they use the registry singleton), so
    ``execute(None, params)`` is the manual-operator path. A skill never raises
    on absent hardware — it returns ``success=False`` — but we still guard so a
    surprise never becomes a 500."""
    try:
        res = skill_cls().execute(None, params)
        return OpticsActionResponse(
            ok=bool(getattr(res, "success", False)),
            error=(getattr(res, "error", "") or None),
            summary=getattr(res, "summary", None),
            data=dict(getattr(res, "data", {}) or {}),
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never crash the API
        logger.warning("optics action %s failed: %s", skill_cls.__name__, exc)
        return OpticsActionResponse(ok=False, error=f"{type(exc).__name__}: {exc}")


# ── GET /api/optics/devices ───────────────────────────────────────────────
@router.get("/optics/devices", response_model=OpticsDevicesResponse)
def list_devices(request: Request) -> OpticsDevicesResponse:
    """Optical-bench inventory + the pump-probe delay-line binding.

    Never touches hardware beyond a config read; empty inventory when no bench
    is configured (``degraded`` stays False — an empty bench is a valid state,
    not a failure)."""
    try:
        from mast.skills.builtins.optics_stage import ListOpticalDevices

        res = ListOpticalDevices().execute(None, {})
        if not getattr(res, "success", False):
            return OpticsDevicesResponse(
                degraded=True, detail=getattr(res, "error", None) or "unavailable"
            )
        data = res.data or {}
        devices = [
            OpticsDevice(
                id=str(d.get("id", "")),
                name=str(d.get("name", d.get("id", ""))),
                type=str(d.get("type", "?")),
                enabled=bool(d.get("enabled", True)),
                connected=bool(d.get("connected", False)),
                axes=[
                    OpticsAxis(
                        name=str(a.get("name", "")),
                        unit=str(a.get("unit", "um")),
                        min_pos=a.get("min_pos"),
                        max_pos=a.get("max_pos"),
                        role=a.get("role"),
                    )
                    for a in (d.get("axes") or [])
                ],
            )
            for d in (data.get("devices") or [])
        ]
        dl_raw = data.get("delay_line")
        delay_line = None
        if dl_raw:
            delay_line = DelayLineInfo(
                device_id=str(dl_raw.get("device_id", "")),
                axis=str(dl_raw.get("axis", "")),
                ps_per_mm=float(dl_raw.get("ps_per_mm", 0.0) or 0.0),
                zero_offset_mm=float(dl_raw.get("zero_offset_mm", 0.0) or 0.0),
                delay_range_ps=[float(x) for x in (dl_raw.get("delay_range_ps") or [])],
            )
        return OpticsDevicesResponse(devices=devices, delay_line=delay_line)
    except Exception as exc:  # noqa: BLE001 — degrade, never crash
        logger.warning("optics devices relay failed: %s", exc)
        return OpticsDevicesResponse(degraded=True, detail=f"{type(exc).__name__}: {exc}")


# ── reads ─────────────────────────────────────────────────────────────────
@router.get("/optics/position", response_model=OpticsActionResponse)
def get_position(request: Request, device_id: str, axis: str) -> OpticsActionResponse:
    """Read one optical stage axis position."""
    from mast.skills.builtins.optics_stage import OpticalStageGetPos

    return _run(OpticalStageGetPos, {"device_id": device_id, "axis": axis})


@router.get("/optics/delay", response_model=OpticsActionResponse)
def get_delay(request: Request) -> OpticsActionResponse:
    """Read the current pump-probe optical delay (ps) + reachable range."""
    from mast.skills.builtins.optics_stage import DelayLineGetDelay

    return _run(DelayLineGetDelay, {})


# ── writes ────────────────────────────────────────────────────────────────
@router.post("/optics/move", response_model=OpticsActionResponse)
def move(request: Request, body: MoveRequest) -> OpticsActionResponse:
    """Move an optical stage axis (absolute, or relative delta). Soft travel
    limits are enforced in the driver — an out-of-range target is ``ok=false``."""
    from mast.skills.builtins.optics_stage import OpticalStageMove

    return _run(OpticalStageMove, {
        "device_id": body.device_id, "axis": body.axis,
        "position": body.position, "relative": body.relative, "wait": True,
    })


@router.post("/optics/wiggle", response_model=OpticsActionResponse)
def wiggle(request: Request, body: WiggleRequest) -> OpticsActionResponse:
    """Bring-up self-test: nudge an axis and report moved / direction / scale."""
    from mast.skills.builtins.optics_stage import OpticalStageWiggle

    return _run(OpticalStageWiggle, {
        "device_id": body.device_id, "axis": body.axis, "delta": body.delta,
    })


@router.post("/optics/home", response_model=OpticsActionResponse)
def home(request: Request, body: HomeRequest) -> OpticsActionResponse:
    """Find-zero / reference an axis (sweeps the travel — confirm no measurement
    depends on the current position)."""
    from mast.skills.builtins.optics_stage import HomeOpticalStage

    return _run(HomeOpticalStage, {"device_id": body.device_id, "axis": body.axis})


@router.post("/optics/stop", response_model=OpticsActionResponse)
def stop(request: Request, body: StopRequest) -> OpticsActionResponse:
    """Panic-stop one device, or the whole bench when device_id is omitted."""
    from mast.skills.builtins.optics_stage import StopOpticalStage

    params: dict[str, Any] = {}
    if body.device_id:
        params["device_id"] = body.device_id
    return _run(StopOpticalStage, params)


@router.post("/optics/delay/move", response_model=OpticsActionResponse)
def delay_move(request: Request, body: DelayMoveRequest) -> OpticsActionResponse:
    """Move the pump-probe delay line to an optical delay (ps)."""
    from mast.skills.builtins.optics_stage import DelayLineMoveTo

    return _run(DelayLineMoveTo, {"delay_ps": body.delay_ps, "wait": True})


@router.post("/optics/dry-run", response_model=OpticsActionResponse)
def dry_run(request: Request, body: DryRunRequest) -> OpticsActionResponse:
    """Test-drive the delay line through a full pump-probe sweep — steps every
    point and settles, but fires no laser trigger, reads no signal and saves no
    file. Verifies the whole range is reachable before a real measurement.
    (Only the delay-line registry is touched — no Nanonis, no ExecutionContext.)"""
    from mast.skills.builtins.optics_pump_probe import PumpProbeScan

    return _run(PumpProbeScan, {
        "delay_start_ps": body.delay_start_ps,
        "delay_stop_ps": body.delay_stop_ps,
        "points": body.points,
        "dry_run": True,
    })
