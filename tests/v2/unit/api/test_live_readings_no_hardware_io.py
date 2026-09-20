"""要求:实时读数更快,做到准实时。

Answering that meant establishing WHERE the readings actually come from, because
the Nanonis TCP port is fragile by project rule and a faster UI must not turn
into a faster instrument poll by accident. These tests pin what was measured:

  1. ``GET /api/hardware/live-readings`` touches the cache ONLY. Zero
     ``safe_call``s reach the ConnectionPool no matter how often it is hit — so
     the UI poll rate is decoupled from instrument load, which is the entire
     licence for polling faster.
  2. ``InstrumentState.snapshot()`` is a pure cache read while ``refresh()``
     costs **10 monitor-port round-trips**. That ratio is the budget argument
     for leaving the ~1 s producer alone (the monitor port is shared with the
     tip-crash SafetyWatchdog at 0.5 s and ScanMonitor at 0.5 s).

If someone later "simplifies" the endpoint into calling ``refresh()``, the UI's
500 ms poll silently becomes 500 ms of hardware polling on the port that the
safety watchdog lives on. That is the regression these tests exist to stop.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.vision_hardware import router
from mast.core.state import InstrumentState
from mast.core.types import HardwareState


class _CountingPool:
    """ConnectionPool stand-in that records every attempted TCP transaction."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def safe_call(self, method_name: str, *args, role: str = "main"):
        self.calls.append((role, method_name))

        class _Rec:
            error = None
            return_value = None

        return _Rec()

    def get(self, role: str):  # _main_connected() probe
        return object()


def _client(state, pool) -> TestClient:
    live_app = type("_App", (), {"_state": state, "_pool": pool})()
    ctx = AppContext()
    ctx.live_app = live_app  # type: ignore[attr-defined]
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_endpoint_does_no_tcp_however_often_it_is_polled() -> None:
    """The licence for a faster UI poll: the request path never talks to the
    instrument, so 500 ms polling costs the Nanonis port exactly nothing."""
    pool = _CountingPool()
    state = InstrumentState(pool)
    state.apply_patch(bias_v=1.5, current_a=1e-9, z_pos_m=-2e-8, setpoint_a=1e-9)
    pool.calls.clear()  # apply_patch is cache-only too, but be explicit

    c = _client(state, pool)
    for _ in range(50):
        r = c.get("/api/hardware/live-readings")
        assert r.status_code == 200

    assert pool.calls == [], (
        f"live-readings performed {len(pool.calls)} hardware call(s) — the UI "
        "poll rate is now coupled to the fragile Nanonis port "
    )
    body = r.json()
    assert body["degraded"] is False
    assert body["readings"]["bias_v"] == 1.5


#: How many monitor-role round-trips one ``refresh()`` costs. **Pinned, not
#: derived** — this number is the whole basis of the "do not speed the producer
#: up" decision, so it must change only when somebody decides to change it.
#:
#: 2026-08-05: 10 → 11. The added read is ``LockIn_ModOnOffGet``, so the current
#: monitor can tell a 973 Hz modulation ripple from a misbehaving tip (it was
#: counting the ripple as ~128 jumps/second). The operator switches modulation
#: from the Nanonis panel, so there is no cheaper source: anything derived from
#: MAST's own calls is blind exactly when it matters. If this count ever needs to
#: come back down, that read is the first candidate — modulation state changes
#: rarely and could be polled at a fraction of 1 Hz.
_REFRESH_ROUND_TRIPS = 11


def test_snapshot_is_free_and_refresh_is_ten_monitor_round_trips() -> None:
    """The measurement the 'don't speed up the producer' decision rests on."""
    pool = _CountingPool()
    state = InstrumentState(pool)

    for _ in range(100):
        state.snapshot()
    assert pool.calls == [], "snapshot() must never touch the instrument"

    state.refresh()
    assert len(pool.calls) == _REFRESH_ROUND_TRIPS, [m for _, m in pool.calls]
    # ALL of them land on the monitor role — the port the tip-crash
    # SafetyWatchdog (0.5 s) and ScanMonitor (0.5 s) also share, serialised
    # behind one per-role lock. That contention is why the producer stays at 1 s.
    assert {role for role, _ in pool.calls} == {"monitor"}


def test_degraded_when_no_state_is_wired_still_no_calls() -> None:
    """Standalone dev / no live app must degrade without probing hardware."""
    pool = _CountingPool()
    live_app = type("_App", (), {"_state": None, "_pool": pool})()
    ctx = AppContext()
    ctx.live_app = live_app  # type: ignore[attr-defined]
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    c = TestClient(app)

    r = c.get("/api/hardware/live-readings")
    assert r.status_code == 200
    assert r.json()["degraded"] is True
    assert [m for _, m in pool.calls] == [], "degraded path must not poll hardware"


def test_stale_flag_survives_into_the_response() -> None:
    """The UI now updates 4x faster, so 'this is not live' has to be reportable
    from the same payload — the panel gates its warning on this flag."""
    pool = _CountingPool()
    hw = HardwareState()
    hw.bias_v = 0.5
    hw.stale = True

    class _Holder:
        def snapshot(self):
            return hw

        def history(self, channel):
            return []

    c = _client(_Holder(), pool)
    body = c.get("/api/hardware/live-readings").json()
    assert body["readings"]["stale"] is True
    assert pool.calls == []


# ── the cross-layer coupling that caused #27 in the first place ─────────────

_REPO = Path(__file__).resolve().parents[4]


def _producer_interval_s() -> float:
    src = (_REPO / "MASTv2" / "mast" / "core" / "runtime.py").read_text(encoding="utf-8")
    m = re.search(r"start_background_refresh\(interval_s=([\d.]+)\)", src)
    if not m:  # pragma: no cover - the call moved / became configurable
        pytest.skip("producer interval is no longer a literal in runtime.py — "
                    "re-derive the UI poll rate against whatever replaced it")
    return float(m.group(1))


def _ui_poll_ms() -> int:
    src = (_REPO / "frontend" / "src" / "lib" / "pollRates.ts").read_text(encoding="utf-8")
    m = re.search(r"LIVE_READINGS_POLL_MS\s*=\s*([\d_]+)", src)
    assert m, "LIVE_READINGS_POLL_MS not found in frontend/src/lib/pollRates.ts"
    return int(m.group(1).replace("_", ""))


def test_ui_does_not_under_sample_the_producer() -> None:
    """THE #27 bug: the UI polled every 2000 ms against a ~1000 ms producer, so
    it discarded every other reading the instrument had already given us and
    could show a value ~3 s old. The client must sample at least as often as the
    cache is refreshed."""
    poll_ms = _ui_poll_ms()
    producer_ms = _producer_interval_s() * 1000
    assert poll_ms <= producer_ms, (
        f"UI polls every {poll_ms} ms but the cache refreshes every "
        f"{producer_ms:.0f} ms — readings are being thrown away "
    )


def test_ui_poll_rate_has_a_sane_floor() -> None:
    """Faster than this buys nothing: the endpoint can only ever hand back
    whatever the ~1 s producer last wrote, and every extra request is a real
    round-trip over the operator's Tailscale link."""
    assert _ui_poll_ms() >= 250, (
        "polling below 250 ms cannot make a reading fresher — the ceiling is "
        "the producer's refresh interval, not the poll rate "
    )
