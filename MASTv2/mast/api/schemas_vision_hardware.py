"""Pydantic request/response models for Wave A — vision_hardware parity slice.

This is the SINGLE SOURCE OF TYPES for the four parity endpoints that re-expose
functionality whose logic still lives in the Python core but which the TS SPA
lost visibility of when the GUI was rewritten Gradio→TS:

* ``GET /api/hardware/live-readings`` — the Lab-Console live instrument readings
  (bias / current / z / setpoint …) read from the live ``InstrumentState.snapshot``
  off ``ctx.live_app._state`` (recoverable, not truly missing — see
  gui/status_panel.py ``get_status_html`` / ``build_compact_header_html``).
* ``GET /api/experimental/monitor/status`` — the long-term-monitor daemon status
  the old ``exp_capture._mon_tick`` rendered (running / channel / count / last
  value / csv path …).
* ``GET /api/vision/buffer`` — the full Vision-Buffer event list with kind/since
  filters, mirroring gui/vision_buffer.py ``_event_to_dict`` row shape, read from
  ``BufferService.get_event_history`` + ``get_stats``.
* ``GET /api/system/diagnostics`` — a categorized rollup of the dashboard
  ``run_system_check`` 6-item self-check (the existing ``/api/system/check``
  returns the raw rows; this adds the summary the dashboard header showed).

Every model carries ``degraded`` so a missing live subsystem yields a valid
empty response, never a 500. No business/safety logic lives in these shapes.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# GET /api/hardware/live-readings
# ─────────────────────────────────────────────────────────────────────


class LiveReadings(BaseModel):
    """Live instrument readings from ``InstrumentState.snapshot()``.

    Field names mirror ``mast.core.types.HardwareState`` (SI units). ``z_m`` is
    the snapshot's ``z_pos_m`` (the EventBus bridge already calls it ``z_m``).
    All optional — a fresh/offline snapshot leaves them ``None`` rather than
    fabricating values."""

    bias_v: Optional[float] = None
    current_a: Optional[float] = None
    z_m: Optional[float] = None
    setpoint_a: Optional[float] = None
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    z_controller_on: Optional[bool] = None
    z_controller_status: Optional[str] = None
    withdrawn: Optional[bool] = None
    scan_running: Optional[bool] = None
    timestamp: Optional[str] = None
    # True when the monitor link is down and these values are carried-forward
    # (not live) — the UI shows a "stale" indicator instead of pretending the
    # numbers are current.
    stale: bool = False


class LiveReadingsResponse(BaseModel):
    """The live instrument readings + recent sparkline history.

    ``degraded`` True when no live app / InstrumentState is wired (standalone
    dev) or the snapshot read raised — the response is then empty but valid.
    ``connected`` reflects whether the 'main' Nanonis role is up (best effort)."""

    readings: LiveReadings = Field(default_factory=LiveReadings)
    bias_history: list[float] = Field(default_factory=list)
    current_history: list[float] = Field(default_factory=list)
    z_history: list[float] = Field(default_factory=list)
    connected: bool = False
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# GET /api/experimental/monitor/status
# ─────────────────────────────────────────────────────────────────────


class MonitorStatusResponse(BaseModel):
    """Long-term-monitor daemon status (mirrors exp_capture ``_exp_monitor``
    state dict that ``_mon_tick`` rendered).

    ``degraded`` True only when no live app is wired; an idle / never-started
    monitor is a valid non-degraded ``running=False`` response (the live core is
    present, it simply isn't monitoring)."""

    running: bool = False
    channel: str = ""
    interval_s: float = 0.0
    count: int = 0
    last_value: Optional[float] = None
    unit: str = ""
    last_t: str = ""
    csv_path: str = ""
    error: str = ""
    started: bool = False
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/buffer
# ─────────────────────────────────────────────────────────────────────


class VisionBufferEvent(BaseModel):
    """One Vision-Buffer row (mirrors gui/vision_buffer.py ``_event_to_dict``).

    The wall-clock ``t_wall`` is derived from ``t_mono_ns`` exactly like the
    legacy panel, so the TS table can sort/format identically."""

    event_id: str = ""
    seqno: int = 0
    kind: str = ""
    severity: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    cause_ref: Optional[str] = None
    t_mono_ns: int = 0
    t_wall: float = 0.0


class VisionBufferStats(BaseModel):
    """The BufferService counter snapshot (``get_stats``) the stats bar showed."""

    events_published: int = 0
    events_dropped_oldest: int = 0
    events_fanout_failed: int = 0
    wal_event_writes: int = 0
    subscribers_active: int = 0


class VisionBufferResponse(BaseModel):
    """Filtered Vision-Buffer event list + counters.

    ``degraded`` True when no BufferService is wired (standalone dev) or the
    read raised — empty list, never a 500. ``count`` is the post-filter size;
    ``total`` is the unfiltered ring size, so the table can show ``shown/total``
    like the legacy stats bar."""

    events: list[VisionBufferEvent] = Field(default_factory=list)
    stats: VisionBufferStats = Field(default_factory=VisionBufferStats)
    count: int = 0
    total: int = 0
    kind: Optional[str] = None
    since: int = -1
    degraded: bool = False


# ─────────────────────────────────────────────────────────────────────
# GET /api/system/diagnostics
# ─────────────────────────────────────────────────────────────────────


class DiagnosticItem(BaseModel):
    """One self-check row (mirrors dashboard.run_system_check result dicts —
    same shape as schemas_admin.SystemCheckItem, duplicated here so this slice
    owns its own types per the house rule)."""

    name: str = ""
    status: Literal["ok", "warning", "error", "unavailable"] = "unavailable"
    detail: str = ""


class DiagnosticsSummary(BaseModel):
    """Per-status counts across the self-check rows (the dashboard header pill)."""

    ok: int = 0
    warning: int = 0
    error: int = 0
    unavailable: int = 0
    total: int = 0


class DiagnosticsResponse(BaseModel):
    """Categorized rollup of the dashboard 6-item self-check.

    ``healthy`` is True when no row is ``error`` (warnings/unavailable are
    tolerated — they mean "degraded but running", matching the dashboard's
    green/amber/red header logic). ``degraded`` True when no live app is wired
    (the check needs the connection pool / storage / monitor singletons)."""

    items: list[DiagnosticItem] = Field(default_factory=list)
    summary: DiagnosticsSummary = Field(default_factory=DiagnosticsSummary)
    healthy: bool = False
    count: int = 0
    degraded: bool = False


__all__ = [
    "LiveReadings",
    "LiveReadingsResponse",
    "MonitorStatusResponse",
    "VisionBufferEvent",
    "VisionBufferStats",
    "VisionBufferResponse",
    "DiagnosticItem",
    "DiagnosticsSummary",
    "DiagnosticsResponse",
]
