"""Wave A contract tests — vision_hardware parity endpoints.

Per the house test rule the router under test is NOT yet included in
mast.api.app (integration wires that); we mount it on a throwaway FastAPI
app with a fresh AppContext. We assert:

  * standalone (no live core wired) → every endpoint 200s and degrades (empty
    but valid, ``degraded=True``), never a 500;
  * with a fake live app / InstrumentState / BufferService wired onto the ctx,
    the live paths relay the backend data into the typed shape;
  * filters (vision/buffer kind+since) and the diagnostics summary categorize
    correctly.

No hardware, no gradio. The fakes mimic only the surface the handlers touch.
"""

from __future__ import annotations

import time
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.vision_hardware import router


# ── throwaway app ──────────────────────────────────────────────────────


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── fakes (mimic only the surface the handlers touch) ───────────────────


class _FakeState:
    """Mimics InstrumentState.snapshot()/history()."""

    def __init__(self, hw, hist=None):
        self._hw = hw
        self._hist = hist or {}

    def snapshot(self):
        return self._hw

    def history(self, channel):
        return self._hist.get(channel, [])


class _FakePool:
    def __init__(self, connected=True):
        self._connected = connected

    def get(self, role):
        if not self._connected:
            raise RuntimeError("not connected")
        return object()


class _FakeApp:
    """A stand-in for the live MASTApp / CoreRuntime."""

    def __init__(self, *, state=None, pool=None, exp_monitor=None, buffer=None):
        self._state = state
        self._pool = pool
        if exp_monitor is not None:
            self._exp_monitor = exp_monitor
        self._buffer = buffer


class _FakeEvent:
    def __init__(self, seqno, kind, severity="info", payload=None, cause_ref=None):
        self.event_id = f"ev-{seqno}"
        self.seqno = seqno
        self.kind = kind
        self.severity = severity
        self.payload = payload or {}
        self.cause_ref = cause_ref
        self.t_mono_ns = time.monotonic_ns()


class _FakeBuffer:
    def __init__(self, events, stats=None):
        self._events = events
        self._stats = stats or {}

    def get_event_history(self, since_seqno=-1, limit=100):
        matched = [e for e in self._events if e.seqno > since_seqno]
        return matched[-limit:]

    def get_stats(self):
        return dict(self._stats)


def _hw(**kw):
    base = dict(
        bias_v=None, current_a=None, z_pos_m=None, setpoint_a=None,
        x_pos_m=None, y_pos_m=None, z_controller_on=None,
        z_controller_status=None, withdrawn=None, scan_running=None,
        timestamp="2026-06-21T00:00:00",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


# ═══════════════════════════════════════════════════════════════════════
# GET /api/hardware/live-readings
# ═══════════════════════════════════════════════════════════════════════


def test_live_readings_degrades_unwired() -> None:
    r = _client().get("/api/hardware/live-readings")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["connected"] is False
    assert body["readings"]["bias_v"] is None
    assert body["bias_history"] == []


def test_live_readings_no_state_degrades() -> None:
    ctx = AppContext()
    ctx.app = _FakeApp(state=None, pool=_FakePool(connected=True))
    r = _client(ctx).get("/api/hardware/live-readings")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["connected"] is True  # main role probed even without state


def test_live_readings_live_path() -> None:
    hw = _hw(
        bias_v=0.5, current_a=1e-10, z_pos_m=2e-9, setpoint_a=5e-11,
        z_controller_on=True, z_controller_status="On", scan_running=False,
        withdrawn=False,
    )
    state = _FakeState(hw, hist={"bias": [0.4, 0.5], "current": [1e-10], "z": [2e-9, 2.1e-9]})
    ctx = AppContext()
    ctx.live_app = _FakeApp(state=state, pool=_FakePool(connected=True))
    r = _client(ctx).get("/api/hardware/live-readings")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["connected"] is True
    rd = body["readings"]
    assert rd["bias_v"] == 0.5
    assert rd["current_a"] == 1e-10
    assert rd["z_m"] == 2e-9  # z_pos_m mapped to z_m
    assert rd["setpoint_a"] == 5e-11
    assert rd["z_controller_on"] is True
    assert rd["z_controller_status"] == "On"
    assert body["bias_history"] == [0.4, 0.5]
    assert body["z_history"] == [2e-9, 2.1e-9]


def test_live_readings_snapshot_raises_degrades() -> None:
    class _Boom:
        def snapshot(self):
            raise RuntimeError("boom")

        def history(self, channel):
            return []

    ctx = AppContext()
    ctx.app = _FakeApp(state=_Boom(), pool=_FakePool(connected=False))
    r = _client(ctx).get("/api/hardware/live-readings")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ═══════════════════════════════════════════════════════════════════════
# GET /api/experimental/monitor/status
# ═══════════════════════════════════════════════════════════════════════


def test_monitor_status_degrades_unwired() -> None:
    r = _client().get("/api/experimental/monitor/status")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["running"] is False
    assert body["started"] is False


def test_monitor_status_never_started() -> None:
    ctx = AppContext()
    ctx.app = _FakeApp()  # live core, but no _exp_monitor attr
    r = _client(ctx).get("/api/experimental/monitor/status")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["started"] is False
    assert body["running"] is False


def test_monitor_status_running() -> None:
    import threading

    st = {
        "running": True,
        "count": 42,
        "last_value": 1.5e-10,
        "unit": "A",
        "last_t": "12:00:00",
        "error": "",
        "csv_path": "/tmp/mon.csv",
        "channel": "通道 0",
        "interval_s": 5.0,
        "lock": threading.Lock(),
    }
    ctx = AppContext()
    ctx.app = _FakeApp(exp_monitor=st)
    r = _client(ctx).get("/api/experimental/monitor/status")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["started"] is True
    assert body["running"] is True
    assert body["count"] == 42
    assert body["last_value"] == 1.5e-10
    assert body["unit"] == "A"
    assert body["csv_path"] == "/tmp/mon.csv"
    assert body["interval_s"] == 5.0


# ═══════════════════════════════════════════════════════════════════════
# GET /api/vision/buffer
# ═══════════════════════════════════════════════════════════════════════


def test_vision_buffer_degrades_unwired() -> None:
    r = _client().get("/api/vision/buffer")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["events"] == []
    assert body["count"] == 0
    assert body["total"] == 0


def test_vision_buffer_live_path_and_stats() -> None:
    events = [
        _FakeEvent(1, "scan_complete", "info", {"file_path": "/a.sxm"}),
        _FakeEvent(2, "tip_quality_drop", "warn", {"score": 0.3}),
        _FakeEvent(3, "e_stop", "critical", {}, cause_ref="tip_status#2"),
    ]
    stats = {
        "events_published": 3,
        "events_dropped_oldest": 1,
        "events_fanout_failed": 0,
        "wal_event_writes": 3,
        "subscribers_active": 2,
    }
    ctx = AppContext()
    ctx.wire(buffer_service=_FakeBuffer(events, stats))
    r = _client(ctx).get("/api/vision/buffer")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == 3
    assert body["total"] == 3
    assert body["events"][0]["kind"] == "scan_complete"
    assert body["events"][2]["severity"] == "critical"
    assert body["events"][2]["cause_ref"] == "tip_status#2"
    assert body["events"][0]["payload"]["file_path"] == "/a.sxm"
    assert body["stats"]["events_published"] == 3
    assert body["stats"]["events_dropped_oldest"] == 1
    assert body["stats"]["subscribers_active"] == 2
    # t_wall derived from t_mono_ns (recent → near now)
    assert body["events"][0]["t_wall"] > 0


def test_vision_buffer_kind_filter() -> None:
    events = [
        _FakeEvent(1, "scan_complete", "info"),
        _FakeEvent(2, "tip_quality_drop", "warn"),
        _FakeEvent(3, "scan_complete", "info"),
    ]
    ctx = AppContext()
    ctx.wire(buffer_service=_FakeBuffer(events))
    r = _client(ctx).get("/api/vision/buffer", params={"kind": "scan_complete"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2  # filtered
    assert body["total"] == 3  # unfiltered ring size
    assert all(e["kind"] == "scan_complete" for e in body["events"])
    assert body["kind"] == "scan_complete"


def test_vision_buffer_since_filter() -> None:
    events = [_FakeEvent(i, "scan_complete", "info") for i in range(1, 6)]
    ctx = AppContext()
    ctx.wire(buffer_service=_FakeBuffer(events))
    r = _client(ctx).get("/api/vision/buffer", params={"since": 3})
    assert r.status_code == 200
    body = r.json()
    seqs = [e["seqno"] for e in body["events"]]
    assert seqs == [4, 5]
    assert body["since"] == 3


def test_vision_buffer_backend_raises_degrades() -> None:
    class _BoomBuffer:
        def get_event_history(self, since_seqno=-1, limit=100):
            raise RuntimeError("boom")

        def get_stats(self):
            return {}

    ctx = AppContext()
    ctx.wire(buffer_service=_BoomBuffer())
    r = _client(ctx).get("/api/vision/buffer")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ═══════════════════════════════════════════════════════════════════════
# GET /api/system/diagnostics
# ═══════════════════════════════════════════════════════════════════════


def test_diagnostics_degrades_unwired() -> None:
    r = _client().get("/api/system/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["items"] == []
    assert body["healthy"] is False


def test_diagnostics_live_path_summary(monkeypatch) -> None:
    rows = [
        {"name": "Nanonis main", "status": "ok", "detail": "Connected"},
        {"name": "Data storage", "status": "ok", "detail": "ok"},
        {"name": "LLM API", "status": "warning", "detail": "no planner"},
        {"name": "Sensor: temp", "status": "unavailable", "detail": "placeholder"},
        {"name": "Sensor: vac", "status": "ok", "detail": "Healthy"},
        {"name": "Skill registry", "status": "ok", "detail": "10 skills"},
    ]

    import mast.webui.dashboard as dash

    monkeypatch.setattr(dash, "run_system_check", lambda app: rows)

    ctx = AppContext()
    ctx.app = _FakeApp()
    r = _client(ctx).get("/api/system/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == 6
    assert body["summary"]["ok"] == 4
    assert body["summary"]["warning"] == 1
    assert body["summary"]["unavailable"] == 1
    assert body["summary"]["error"] == 0
    assert body["summary"]["total"] == 6
    assert body["healthy"] is True  # no error rows


def test_diagnostics_unhealthy_when_error(monkeypatch) -> None:
    rows = [
        {"name": "Data storage", "status": "error", "detail": "db missing"},
        {"name": "Skill registry", "status": "ok", "detail": "10 skills"},
    ]
    import mast.webui.dashboard as dash

    monkeypatch.setattr(dash, "run_system_check", lambda app: rows)

    ctx = AppContext()
    ctx.app = _FakeApp()
    r = _client(ctx).get("/api/system/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert body["healthy"] is False
    assert body["summary"]["error"] == 1


def test_diagnostics_backend_raises_degrades(monkeypatch) -> None:
    import mast.webui.dashboard as dash

    def _boom(app):
        raise RuntimeError("boom")

    monkeypatch.setattr(dash, "run_system_check", _boom)
    ctx = AppContext()
    ctx.app = _FakeApp()
    r = _client(ctx).get("/api/system/diagnostics")
    assert r.status_code == 200
    assert r.json()["degraded"] is True
