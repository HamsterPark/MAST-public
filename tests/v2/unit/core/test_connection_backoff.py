"""Regression test for the reconnect-storm GUI freeze (v2.0.8 hotfix).

When Nanonis is offline, the SafetyWatchdog daemon (0.5 s tick) +
GUI Timer (5 s tick) used to trigger ``ConnectionPool._reconnect_role``
hundreds of times per minute. Each attempt did a real
``socket.connect`` (10-50 ms on Windows even on connection-refused)
plus an unthrottled ``logger.error`` + ``logger.warning``. Combined
with ``InstrumentState.refresh`` issuing 8 monitor-role safe_call's
on every 5 s GUI tick, the main UI thread froze.

The fix lives in ``connection.py``:
  • exponential backoff (0, 0.5, 1, 2, 4, 8, 16 s) keyed by failure
    count — within the window the function returns False without
    calling ``socket.connect``.
  • throttled log: only the first failure + one log per 30 s window
    are emitted at ERROR/WARNING; the rest go to DEBUG.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_connection_backoff.py -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from unittest.mock import patch

import pytest

import mast.core.connection as _conn
from mast.core.connection import ConnectionPool, _RECONNECT_BACKOFF_S
from mast.config import NanonisConfig


def _cfg() -> NanonisConfig:
    # Use a quasi-impossible port so socket.connect refuses fast on Windows.
    return NanonisConfig(host="127.0.0.1", port_main=1, port_monitor=1,
                         port_data=1, port_emergency=1, timeout_s=0.05)


# ──────────────────────────────────────────────────────────────────────


def test_first_reconnect_attempts_socket_and_logs_error(caplog):
    """First failure: socket.connect IS called, ERROR is logged."""
    pool = ConnectionPool(_cfg())
    sock_calls: list = []

    def _fake_connect(_self, addr):
        sock_calls.append(addr)
        raise ConnectionRefusedError("refused")

    with patch("socket.socket.connect", _fake_connect):
        with caplog.at_level("ERROR", logger="mast.core.connection"):
            ok = pool._reconnect_role("main")
    assert ok is False
    assert len(sock_calls) == 1
    # Failure count bumped, error logged
    assert pool._reconnect_state["main"]["failures"] == 1
    assert any("Reconnect failed" in r.message and "failure #1" in r.message
               for r in caplog.records)


def test_second_reconnect_within_backoff_skips_socket(caplog):
    """Second failure within the 0.5 s backoff: no socket.connect, no ERROR."""
    pool = ConnectionPool(_cfg())
    sock_calls: list = []

    def _fake_connect(_self, addr):
        sock_calls.append(addr)
        raise ConnectionRefusedError("refused")

    with patch("socket.socket.connect", _fake_connect):
        pool._reconnect_role("main")    # first failure → sock_calls=[1]
        caplog.clear()                  # discard first-call records
        with caplog.at_level("ERROR", logger="mast.core.connection"):
            ok = pool._reconnect_role("main")  # within backoff
    assert ok is False
    # CRITICAL assertion: no second socket.connect during backoff
    assert len(sock_calls) == 1, (
        f"Expected backoff to suppress second connect, got {len(sock_calls)}"
    )
    # No ERROR record from the suppressed second attempt
    error_records_after_clear = [r for r in caplog.records if r.levelname == "ERROR"]
    assert not error_records_after_clear, (
        f"Expected silent backoff window, got: "
        f"{[r.message for r in error_records_after_clear]}"
    )


def test_backoff_table_grows_exponentially():
    """Backoff window expands by ≈2× per consecutive failure."""
    # Index 0 = no backoff (first attempt always runs)
    assert _RECONNECT_BACKOFF_S[0] == 0.0
    # Subsequent: 0.5, 1, 2, 4, 8, 16, capped at 16
    assert _RECONNECT_BACKOFF_S[1] == 0.5
    assert _RECONNECT_BACKOFF_S[2] == 1.0
    assert _RECONNECT_BACKOFF_S[3] == 2.0
    assert _RECONNECT_BACKOFF_S[-1] == 16.0
    # Cap holds for very high failure counts (last entry repeats)
    assert _RECONNECT_BACKOFF_S[-1] == _RECONNECT_BACKOFF_S[-2]


def test_log_throttle_silences_repeated_failures(caplog):
    """100 fast failures emit ≤ 2 ERROR records (1 first + later throttled)."""
    pool = ConnectionPool(_cfg())

    def _fake_connect(_self, addr):
        raise ConnectionRefusedError("refused")

    with patch("socket.socket.connect", _fake_connect):
        with caplog.at_level("ERROR", logger="mast.core.connection"):
            # First call attempts and logs
            pool._reconnect_role("main")
            # 100 fast follow-ups — all should be suppressed by backoff
            # (none should reach socket.connect; none should log ERROR)
            for _ in range(100):
                pool._reconnect_role("main")
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_records) <= 2, (
        f"Expected ≤2 ERROR records under throttle, got {len(error_records)}: "
        f"{[r.message for r in error_records]}"
    )


def test_safe_call_warning_is_throttled(caplog):
    """safe_call's 'Connection lost' WARNING also follows the throttle."""
    pool = ConnectionPool(_cfg())

    def _fake_connect(_self, addr):
        raise ConnectionRefusedError("refused")

    with patch("socket.socket.connect", _fake_connect):
        with caplog.at_level("WARNING", logger="mast.core.connection"):
            for _ in range(50):
                pool.safe_call("Util_VersionGet", role="main")
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    # First call logs once; the rest are silenced by throttle
    assert len(warning_records) <= 2, (
        f"Expected ≤2 WARNINGs from 50 safe_call's, got "
        f"{len(warning_records)}: {[r.message for r in warning_records]}"
    )


def test_successful_reconnect_resets_failure_count():
    """A successful reconnect clears the backoff counter."""
    pool = ConnectionPool(_cfg())
    # Force one failure to seed the counter
    with patch("socket.socket.connect",
               lambda _self, _addr: (_ for _ in ()).throw(ConnectionRefusedError("refused"))):
        pool._reconnect_role("main")
    assert pool._reconnect_state["main"]["failures"] == 1
    # Push the last_fail_at outside the backoff window so the next attempt
    # actually runs (the prod path waits 0.5 s for real; tests must not).
    pool._reconnect_state["main"]["last_fail_at"] = 0.0

    # Now pretend the next connect succeeds
    class _FakeNanonis:
        def close(self): pass

    def _ok_connect(_self, _addr):
        return None

    with patch("socket.socket.connect", _ok_connect), \
         patch.object(_conn, "Nanonis", lambda _sock: _FakeNanonis()):
        ok = pool._reconnect_role("main")
    assert ok is True
    # Counter reset for next outage
    assert pool._reconnect_state["main"]["failures"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
