"""Pydantic models for the optical-bench endpoints (光学台 tab).

Single source of types for ``/api/optics/*``, exported via ``/openapi.json`` and
consumed by the frontend type generators (``npm run gen:api``). The shapes
mirror the optics skills that already exist and are merely RELAYED by the route:

* ``OpticsDevicesResponse`` ⇐ ``ListOpticalDevices`` (motion devices + the
  pump-probe delay-line binding);
* ``OpticsActionResponse`` ⇐ every write/read action (move / wiggle / delay /
  stop / dry-run) — a uniform ``{ok, error, summary, data}`` envelope carrying
  the skill's ``SkillResult`` so the panel can render without a bespoke model
  per action.

House rule: the optics registry is a process-global singleton that degrades to
an empty inventory with no hardware, so ``degraded`` lets the panel render a
"no bench configured / not connected" state without ever seeing a 500.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── inventory (GET /api/optics/devices) ───────────────────────────────────


class OpticsAxis(BaseModel):
    """One movable axis of an optical stage."""

    name: str
    unit: str = "um"
    min_pos: Optional[float] = None
    max_pos: Optional[float] = None
    role: Optional[str] = None


class OpticsDevice(BaseModel):
    """One configured optical-bench device (stage controller)."""

    id: str
    name: str
    type: str
    enabled: bool = True
    connected: bool = False
    axes: list[OpticsAxis] = Field(default_factory=list)


class DelayLineInfo(BaseModel):
    """The pump-probe delay-line binding, if one is configured."""

    device_id: str
    axis: str
    ps_per_mm: float
    zero_offset_mm: float
    delay_range_ps: list[float] = Field(default_factory=list)


class OpticsDevicesResponse(BaseModel):
    """Optical-bench inventory + delay-line binding for the panel overview."""

    devices: list[OpticsDevice] = Field(default_factory=list)
    delay_line: Optional[DelayLineInfo] = None
    degraded: bool = False
    detail: Optional[str] = None


# ── actions (uniform envelope) ────────────────────────────────────────────


class OpticsActionResponse(BaseModel):
    """Uniform result envelope for an optics action (relays a SkillResult).

    ``data`` carries the skill's result dict verbatim (position, on_target,
    scale_ratio, reached_delays_ps, …) — the panel reads the fields it needs.
    ``ok`` is the skill's success; ``error`` its message on failure (an
    out-of-range move or absent hardware is ``ok=false``, never an HTTP error).
    """

    ok: bool
    error: Optional[str] = None
    summary: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


# ── request bodies ────────────────────────────────────────────────────────


# NOTE: keep every request field REQUIRED (no server-side defaults). The frontend
# type generator (openapi-typescript) marks default-valued properties as required
# anyway, so a "with default" field would force the client to send it — worse than
# just requiring the essential ones and letting the skills' own defaults cover the
# tuning knobs (timeout / settle), which the manual panel does not expose.


class MoveRequest(BaseModel):
    device_id: str
    axis: str
    position: float
    relative: bool = False  # sent by the panel every call


class WiggleRequest(BaseModel):
    device_id: str
    axis: str
    delta: float  # sent by the panel every call


class HomeRequest(BaseModel):
    device_id: str
    axis: str


class StopRequest(BaseModel):
    device_id: Optional[str] = None


class DelayMoveRequest(BaseModel):
    delay_ps: float


class DryRunRequest(BaseModel):
    delay_start_ps: float
    delay_stop_ps: float
    points: int
