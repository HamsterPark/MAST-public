"""Wave A parity — vision_hardware domain read endpoints.

Re-exposes functionality whose LOGIC still lives in the Python core but which
the TS SPA lost visibility of when the GUI was rewritten Gradio→TS. Nothing here
re-implements behaviour: each handler RELAYS one kept backend call and shapes the
result into a typed response.

Endpoints
---------
* ``GET /api/hardware/live-readings`` — live instrument readings + sparkline
  history from ``ctx.live_app._state`` (``InstrumentState.snapshot`` / ``history``),
  the same source gui/status_panel.py read. Recoverable, not truly missing.
* ``GET /api/experimental/monitor/status`` — long-term-monitor daemon status off
  the live app's ``_exp_monitor`` dict (gui/exp_capture.py ``_mon_tick``).
* ``GET /api/vision/buffer?kind=&since=`` — full Vision-Buffer event list with
  kind/since filters from ``BufferService.get_event_history`` + ``get_stats``
  (gui/vision_buffer.py ``_event_to_dict`` / ``_render_stats_html``).
* ``GET /api/system/diagnostics`` — categorized rollup of the dashboard 6-item
  ``run_system_check`` self-check (the existing ``/api/system/check`` returns the
  raw rows; this adds the summary the dashboard header showed).

GRACEFUL DEGRADATION is the contract (house rule 2). This module imports with
NOTHING heavy present and every handler boots STANDALONE: it reads optional live
subsystems off ``request.app.state.ctx`` (the live MASTApp, its InstrumentState,
the BufferService) and, if any is absent OR any call raises, returns a valid
empty/degraded response with ``degraded=True`` — never a 500. Heavy core modules
are LAZY-imported INSIDE handlers in try/except. We NEVER import gradio.

The API layer holds NO business/safety logic — it only relays into the core.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Query, Request, Response

from mast.api.schemas_vision_hardware import (
    DiagnosticItem,
    DiagnosticsResponse,
    DiagnosticsSummary,
    LiveReadings,
    LiveReadingsResponse,
    MonitorStatusResponse,
    VisionBufferEvent,
    VisionBufferResponse,
    VisionBufferStats,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision_hardware"])


# ─────────────────────────────────────────────────────────────────────
# Small helpers — all defensive, never raise.
# ─────────────────────────────────────────────────────────────────────


def _get_app(ctx: Any) -> Any:
    """The live MASTApp, if integration wires one; else None (standalone).

    The bootstrap wires both ``ctx.app`` and ``ctx.live_app`` to the same
    CoreRuntime; accept either so this slice works whichever the integrator keeps."""
    if ctx is None:
        return None
    return getattr(ctx, "app", None) or getattr(ctx, "live_app", None)


def _get_buffer(ctx: Any) -> Any:
    """Best-effort fetch of a live BufferService from the context.

    Standalone dev wires nothing → returns None and the caller degrades. Tries
    the wired ``ctx.buffer_service`` / ``ctx.buffer`` first, then the lazy GUI
    helper, then the cached ``app._buffer`` attribute — mirrors routes/vision.py.
    Never imports gradio or constructs anything heavy on the standalone path."""
    if ctx is None:
        return None
    buf = getattr(ctx, "buffer_service", None) or getattr(ctx, "buffer", None)
    if buf is not None:
        return buf
    app = _get_app(ctx)
    if app is not None:
        try:
            from mast.core.runtime import _ensure_buffer_for_gui  # lazy, may be absent

            b = _ensure_buffer_for_gui(app)
            if b is not None:
                return b
        except Exception as exc:  # noqa: BLE001 — never crash on a missing helper
            logger.debug("vision_hardware: _ensure_buffer_for_gui unavailable: %s", exc)
        return getattr(app, "_buffer", None)
    return None


def _enum_value(v: Any, default: str = "") -> str:
    return str(getattr(v, "value", v) or default)


def _event_t_wall(ev: Any) -> float:
    """Approx wall-clock seconds for a VisionEvent (mirrors gui _event_to_dict)."""
    ns = int(getattr(ev, "t_mono_ns", 0) or 0)
    if not ns:
        return 0.0
    delta_s = (time.monotonic_ns() - ns) / 1e9
    return time.time() - delta_s


def _main_connected(app: Any) -> bool:
    """Best-effort 'is the main Nanonis role connected' — never raises."""
    pool = getattr(app, "_pool", None)
    if pool is None:
        return False
    try:
        pool.get("main")
        return True
    except Exception:  # noqa: BLE001
        return False


# ─────────────────────────────────────────────────────────────────────
# GET /api/hardware/live-readings
# ─────────────────────────────────────────────────────────────────────


@router.get("/hardware/live-readings", response_model=LiveReadingsResponse)
def get_live_readings(request: Request) -> LiveReadingsResponse:
    """Live instrument readings (bias / current / z / setpoint …) + sparkline
    history, read from the live ``InstrumentState.snapshot()`` — the same instant,
    cached source gui/status_panel.py read (no TCP round-trip on the request
    thread; the daemon keeps it fresh). Degrades to empty when no live app /
    state is wired."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return LiveReadingsResponse(degraded=True)

    state_holder = getattr(app, "_state", None)
    if state_holder is None:
        return LiveReadingsResponse(degraded=True, connected=_main_connected(app))

    try:
        hw = state_holder.snapshot()
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("live-readings snapshot failed: %s", exc)
        return LiveReadingsResponse(degraded=True, connected=_main_connected(app))

    readings = LiveReadings(
        bias_v=getattr(hw, "bias_v", None),
        current_a=getattr(hw, "current_a", None),
        z_m=getattr(hw, "z_pos_m", None),
        setpoint_a=getattr(hw, "setpoint_a", None),
        x_m=getattr(hw, "x_pos_m", None),
        y_m=getattr(hw, "y_pos_m", None),
        z_controller_on=getattr(hw, "z_controller_on", None),
        z_controller_status=getattr(hw, "z_controller_status", None),
        withdrawn=getattr(hw, "withdrawn", None),
        scan_running=getattr(hw, "scan_running", None),
        timestamp=getattr(hw, "timestamp", None),
        stale=bool(getattr(hw, "stale", False)),
    )

    # Sparkline rings (best-effort; the panel shows these next to each reading).
    def _hist(channel: str) -> list[float]:
        try:
            return [float(v) for v in (state_holder.history(channel) or [])]
        except Exception:  # noqa: BLE001
            return []

    return LiveReadingsResponse(
        readings=readings,
        bias_history=_hist("bias"),
        current_history=_hist("current"),
        z_history=_hist("z"),
        connected=_main_connected(app),
        degraded=False,
    )


# ─────────────────────────────────────────────────────────────────────
# GET /api/experimental/monitor/status
# ─────────────────────────────────────────────────────────────────────


@router.get("/experimental/monitor/status", response_model=MonitorStatusResponse)
def get_monitor_status(request: Request) -> MonitorStatusResponse:
    """Long-term-monitor daemon status — reads the live app's ``_exp_monitor``
    state dict that the legacy ``_mon_tick`` rendered (running / channel / count /
    last value / csv path / error). Acquires the monitor's lock if present so the
    snapshot is consistent. Degrades when no live app is wired; a never-started
    monitor is a valid non-degraded ``running=False`` (started=False) response."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return MonitorStatusResponse(degraded=True)

    st = getattr(app, "_exp_monitor", None)
    if not isinstance(st, dict):
        # Live core present but the monitor was never started — not an error.
        return MonitorStatusResponse(started=False, degraded=False)

    lock = st.get("lock")
    acquired = False
    if lock is not None:
        try:
            # BOUNDED. This is a sync endpoint the UI polls every 2 s, so it runs
            # on the shared anyio threadpool (40 slots). An unbounded acquire()
            # here means: monitor daemon holds the lock across some I/O → every
            # poll parks a worker → ~80 s of that and the pool is drained → EVERY
            # other sync endpoint (feedback, snapshot, topology…) queues behind
            # it. That is a whole-service stall caused by a status read
            # (analysis).
            # A status read is never worth blocking for: miss the lock, report
            # degraded, let the next poll try again 2 s later.
            acquired = lock.acquire(timeout=1.0)
        except Exception:  # noqa: BLE001
            acquired = False
    try:
        running = bool(st.get("running", False))
        channel = str(st.get("channel", "") or "")
        interval_s = float(st.get("interval_s", 0.0) or 0.0)
        count = int(st.get("count", 0) or 0)
        last_value = st.get("last_value")
        unit = str(st.get("unit", "") or "")
        last_t = str(st.get("last_t", "") or "")
        csv_path = str(st.get("csv_path", "") or "")
        error = str(st.get("error", "") or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("monitor status read failed: %s", exc)
        return MonitorStatusResponse(started=True, degraded=True)
    finally:
        if acquired and lock is not None:
            try:
                lock.release()
            except Exception:  # noqa: BLE001
                pass

    return MonitorStatusResponse(
        running=running,
        channel=channel,
        interval_s=interval_s,
        count=count,
        last_value=float(last_value) if isinstance(last_value, (int, float)) else None,
        unit=unit,
        last_t=last_t,
        csv_path=csv_path,
        error=error,
        started=True,
        # Missing the lock means these fields were read WITHOUT it — individually
        # atomic, but possibly from different instants. Say so rather than
        # presenting a maybe-torn snapshot as authoritative.
        degraded=(lock is not None and not acquired),
    )


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/buffer
# ─────────────────────────────────────────────────────────────────────


@router.get("/vision/buffer", response_model=VisionBufferResponse)
def get_vision_buffer(
    request: Request,
    kind: Optional[str] = Query(default=None, description="filter to one event kind"),
    since: int = Query(default=-1, description="only events with seqno > since"),
    limit: int = Query(default=500, ge=1, le=2000),
) -> VisionBufferResponse:
    """Full Vision-Buffer event list (newest last, chronological) with kind/since
    filters, for the Vision Buffer table. Reads ``BufferService.get_event_history``
    + ``get_stats`` — the read-only path the legacy panel used. Each row mirrors
    gui/vision_buffer.py ``_event_to_dict``. Degrades to an empty list when no
    BufferService is wired."""
    ctx = request.app.state.ctx
    buf = _get_buffer(ctx)
    if buf is None:
        return VisionBufferResponse(kind=kind, since=since, degraded=True)

    try:
        raw = buf.get_event_history(since_seqno=since, limit=limit)
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("vision/buffer get_event_history failed: %s", exc)
        return VisionBufferResponse(kind=kind, since=since, degraded=True)

    raw_list = list(raw or [])
    total = len(raw_list)

    events: list[VisionBufferEvent] = []
    try:
        for ev in raw_list:
            ev_kind = _enum_value(getattr(ev, "kind", None))
            if kind and ev_kind != kind:
                continue
            events.append(
                VisionBufferEvent(
                    event_id=str(getattr(ev, "event_id", "") or ""),
                    seqno=int(getattr(ev, "seqno", 0) or 0),
                    kind=ev_kind,
                    severity=_enum_value(getattr(ev, "severity", None)),
                    payload=dict(getattr(ev, "payload", {}) or {}),
                    cause_ref=getattr(ev, "cause_ref", None),
                    t_mono_ns=int(getattr(ev, "t_mono_ns", 0) or 0),
                    t_wall=_event_t_wall(ev),
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision/buffer shaping failed: %s", exc)
        return VisionBufferResponse(kind=kind, since=since, degraded=True)

    # Stats bar counters (best-effort — an empty dict is fine).
    try:
        s = dict(buf.get_stats() or {})
    except Exception:  # noqa: BLE001
        s = {}
    stats = VisionBufferStats(
        events_published=int(s.get("events_published", 0) or 0),
        events_dropped_oldest=int(s.get("events_dropped_oldest", 0) or 0),
        events_fanout_failed=int(s.get("events_fanout_failed", 0) or 0),
        wal_event_writes=int(s.get("wal_event_writes", 0) or 0),
        subscribers_active=int(s.get("subscribers_active", 0) or 0),
    )

    return VisionBufferResponse(
        events=events,
        stats=stats,
        count=len(events),
        total=total,
        kind=kind,
        since=since,
        degraded=False,
    )


# ─────────────────────────────────────────────────────────────────────
# GET /api/vision/event-frame/{event_id}
# ─────────────────────────────────────────────────────────────────────


@router.get("/vision/event-frame/{event_id}", response_class=Response,
            responses={200: {"content": {"image/png": {}}}, 404: {}})
def get_vision_event_frame(event_id: str, request: Request) -> Response:
    """The PNG of the exact frame a vision event judged, or 404.

    「这一条为什么不显示图像」/「同样无图像」. The 视觉缓冲
    panel rendered a 24×24 dashed box where the picture belongs — a stub that was
    never filled in. So the operator was told "反馈振荡/振铃（55 周期/行）——50%
    处" and given no way to look at the 50% frame and judge for themselves.

    A separate endpoint rather than base64 in the list response: a 500-row event
    list with thumbnails inlined would be tens of MB on a 3 s poll, while an
    ``<img src>`` is lazy, browser-cached, and costs nothing for rows nobody
    scrolls to.

    ``frame_path`` ONLY — the PNG ``scan_monitor`` persisted for that milestone.
    No borrowing from the newest .sxm: showing a different frame's picture next
    to this frame's verdict is the falsified history of #76/#78, and it is worse
    than the honest 404 the panel renders as 「该事件没有存图」.
    """
    ctx = request.app.state.ctx
    buf = _get_buffer(ctx)
    if buf is None:
        return Response(status_code=404)
    try:
        from pathlib import Path

        for ev in reversed(list(buf.get_event_history(since_seqno=-1,
                                                      limit=2000) or [])):
            if str(getattr(ev, "event_id", "") or "") != event_id:
                continue
            fp = (dict(getattr(ev, "payload", {}) or {})).get("frame_path")
            if not fp:
                return Response(status_code=404)
            p = Path(str(fp))
            if not p.is_file():
                return Response(status_code=404)
            return Response(
                content=p.read_bytes(), media_type="image/png",
                # Immutable: a milestone PNG is written once and never rewritten
                # (that is the whole point of storing the real frame).
                headers={"Cache-Control": "public, max-age=86400, immutable"})
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("vision event-frame read failed for %s: %s", event_id, exc)
    return Response(status_code=404)


# ─────────────────────────────────────────────────────────────────────
# GET /api/system/diagnostics
# ─────────────────────────────────────────────────────────────────────


@router.get("/system/diagnostics", response_model=DiagnosticsResponse)
def get_system_diagnostics(request: Request) -> DiagnosticsResponse:
    """Categorized rollup of the dashboard 6-item self-check.

    The existing ``GET /api/system/check`` returns the raw rows; this adds the
    per-status summary + an aggregate ``healthy`` flag the dashboard header
    showed (green when no row is ``error``). Same backend
    (``mast.webui.dashboard.run_system_check``) — relayed, not re-run. Degrades to
    an empty diagnostics when no live app is wired (the check needs the
    connection pool / storage / monitor singletons)."""
    ctx = request.app.state.ctx
    app = _get_app(ctx)
    if app is None:
        return DiagnosticsResponse(degraded=True)

    try:
        from mast.webui.dashboard import run_system_check

        raw = run_system_check(app) or []
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("system diagnostics run failed: %s", exc)
        return DiagnosticsResponse(degraded=True)

    items: list[DiagnosticItem] = []
    counts = {"ok": 0, "warning": 0, "error": 0, "unavailable": 0}
    for r in raw:
        if not isinstance(r, dict):
            continue
        status = str(r.get("status", "unavailable"))
        if status not in counts:
            status = "unavailable"
        counts[status] += 1
        items.append(
            DiagnosticItem(
                name=str(r.get("name", "")),
                status=status,  # type: ignore[arg-type]
                detail=str(r.get("detail", "")),
            )
        )

    summary = DiagnosticsSummary(
        ok=counts["ok"],
        warning=counts["warning"],
        error=counts["error"],
        unavailable=counts["unavailable"],
        total=len(items),
    )
    return DiagnosticsResponse(
        items=items,
        summary=summary,
        healthy=counts["error"] == 0,
        count=len(items),
        degraded=False,
    )
