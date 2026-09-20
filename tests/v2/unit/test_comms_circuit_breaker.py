"""Comms circuit breaker (⑬, field trace).

Two layers:
  1. CommsCircuitBreaker state machine — deterministic (injected clock).
  2. ConnectionPool.safe_call wiring — consecutive TCP timeouts trip the
     breaker, after which further calls short-circuit WITHOUT touching the
     socket (the whole point: judge the link down once, don't stall every tool
     ~5 s in turn, and don't re-hammer the fragile port).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_comms_circuit_breaker.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.config import NanonisConfig
from mast.core.comms_health import CLOSED, HALF_OPEN, OPEN, CommsCircuitBreaker
from mast.core.connection import ConnectionPool


# ── 1. state machine ─────────────────────────────────────────────────────

class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _breaker(clock=None, **kw):
    kw.setdefault("fail_threshold", 3)
    kw.setdefault("open_cooldown_s", 20.0)
    kw.setdefault("streak_window_s", 30.0)
    return CommsCircuitBreaker(clock=clock or _Clock(), **kw)


def test_closed_allows_until_threshold():
    b = _breaker()
    assert b.state() == CLOSED and b.allow()
    b.record_failure("t1")
    b.record_failure("t2")
    assert b.state() == CLOSED and b.allow()  # 2 < 3, still closed


def test_third_consecutive_failure_opens():
    clk = _Clock()
    b = _breaker(clk)
    for i in range(3):
        b.record_failure(f"timeout {i}")
    assert b.state() == OPEN
    assert b.is_open()
    assert b.allow() is False  # short-circuit while inside cooldown
    assert 0 < b.cooldown_remaining_s() <= 20.0


def test_cooldown_releases_single_probe_then_holds():
    clk = _Clock()
    b = _breaker(clk)
    for _ in range(3):
        b.record_failure("x")
    clk.t = 25.0  # past the 20 s cooldown
    assert b.state() == HALF_OPEN
    assert b.allow() is True         # exactly ONE probe released
    assert b.allow() is False        # probe outstanding — everyone else held


def test_probe_success_closes_breaker():
    clk = _Clock()
    b = _breaker(clk)
    for _ in range(3):
        b.record_failure("x")
    clk.t = 25.0
    assert b.allow() is True         # probe
    b.record_success()               # probe succeeded → Nanonis recovered
    assert b.state() == CLOSED
    assert b.allow() is True


def test_probe_failure_reopens_with_fresh_cooldown():
    clk = _Clock()
    b = _breaker(clk)
    for _ in range(3):
        b.record_failure("x")
    clk.t = 25.0
    assert b.allow() is True         # probe
    b.record_failure("still down")   # probe failed → re-open
    assert b.is_open()
    assert b.allow() is False
    assert b.cooldown_remaining_s() > 19.0  # full cooldown re-armed from t=25


def test_app_error_success_resets_streak():
    # A completed round-trip that returned a Nanonis app-error string is a
    # SUCCESS for the link — it must reset the streak.
    b = _breaker()
    b.record_failure("t1")
    b.record_failure("t2")
    b.record_success()
    b.record_failure("t3")
    assert b.state() == CLOSED  # streak restarted at 1, not 3


def test_stale_failures_do_not_count_as_consecutive():
    clk = _Clock()
    b = _breaker(clk)
    b.record_failure("t1")
    b.record_failure("t2")        # streak = 2 at t=0
    clk.t = 40.0                  # > 30 s window since last failure
    b.record_failure("t3")        # too old to be consecutive → streak resets to 1
    assert b.state() == CLOSED


# ── 2. ConnectionPool.safe_call wiring ───────────────────────────────────

class _TimingOutNanonis:
    """A fake whose every call raises TimeoutError (the field-trace symptom)."""

    def __init__(self) -> None:
        self.calls = 0

    def Bias_Get(self):
        self.calls += 1
        raise TimeoutError("timed out")

    def close(self):
        pass


def _pool_with_timeout(clock=None):
    pool = ConnectionPool(NanonisConfig())
    fake = _TimingOutNanonis()
    pool._connections["main"] = (None, fake)
    # Reconnect always fails (nothing is listening) — force it deterministically
    # so the test doesn't depend on a real socket.connect to localhost.
    pool._reconnect_role = lambda role: False  # type: ignore[assignment]
    if clock is not None:
        pool._breaker = CommsCircuitBreaker(
            fail_threshold=3, open_cooldown_s=5.0, streak_window_s=30.0,
            clock=clock)
    return pool, fake


def test_pool_opens_after_three_timeouts_and_short_circuits():
    pool, fake = _pool_with_timeout()
    # First three calls each hit the socket and time out.
    for _ in range(3):
        rec = pool.safe_call("Bias_Get")
        assert rec.error  # a TCP failure
    assert fake.calls == 3
    assert pool.comms_healthy() is False

    # The FOURTH call must short-circuit WITHOUT touching the fake.
    rec = pool.safe_call("Bias_Get")
    assert "comms_circuit_open" in rec.error
    assert fake.calls == 3  # <-- the socket was NOT hit again


def test_pool_recovers_after_cooldown_probe_succeeds():
    clk = _Clock()
    pool, fake = _pool_with_timeout(clock=clk)
    for _ in range(3):
        pool.safe_call("Bias_Get")
    assert pool.comms_healthy() is False

    # Swap in a healthy connection and advance past the cooldown → one probe.
    class _Good:
        def Bias_Get(self):
            return ("", b"", [0.0])

        def close(self):
            pass

    pool._connections["main"] = (None, _Good())
    clk.t = 10.0  # past the 5 s cooldown
    rec = pool.safe_call("Bias_Get")
    assert not rec.error
    assert pool.comms_healthy() is True  # breaker closed on the successful probe


def test_closed_pool_does_not_consult_breaker():
    pool, _ = _pool_with_timeout()
    pool._closed = True
    rec = pool.safe_call("Bias_Get")
    assert "closed" in rec.error.lower()
