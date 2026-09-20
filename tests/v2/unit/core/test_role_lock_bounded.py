"""The per-role Nanonis lock must be BOUNDED, and the emergency path must never
queue behind a stall (审计 致命三).

The failure this pins:

    sse-pump thread wedges inside recv() on role="main"
      → every other role="main" caller blocks on an unbounded lock acquire
      → each of those holds an anyio threadpool token
      → the default CapacityLimiter(40) drains
      → all 173 sync endpoints under mast/api/ stall
      → and the E-STOP's retract fallback, `role="main"`, queues behind the
        exact stall it exists to rescue the instrument from.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_role_lock_bounded.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
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

import threading
import time

import pytest


# ─────────────────────────────────────────────────────────────────────
# Fixtures — a pool wired to a fake, blockable "Nanonis"
# ─────────────────────────────────────────────────────────────────────

class _WedgedNano:
    """A Nanonis client whose call never returns until released."""

    def __init__(self, gate: threading.Event):
        self._gate = gate
        self.closed = False

    def Util_VersionGet(self):
        self._gate.wait(timeout=30)
        return ["", b"", [1]]

    def ZCtrl_Withdraw(self, *a):
        return ["", b"", []]

    def close(self):
        self.closed = True
        self._gate.set()  # a real close() makes the blocked recv raise


class _FakeSock:
    def __init__(self, gate: threading.Event):
        self._gate = gate
        self.closed = False

    def close(self):
        self.closed = True
        self._gate.set()


@pytest.fixture
def pool():
    from mast.config import NanonisConfig
    from mast.core.connection import ConnectionPool

    return ConnectionPool(NanonisConfig(host="127.0.0.1", timeout_s=1.0))


def _install(pool, role, gate):
    """Put a fake connection into the pool for *role* without any TCP."""
    nn = _WedgedNano(gate)
    sock = _FakeSock(gate)
    with pool._struct_lock:
        pool._connections[role] = (sock, nn)
    return sock, nn


def _hold_role(pool, role, gate):
    """Start a thread that occupies *role*'s lock until *gate* is set."""
    started = threading.Event()

    def _worker():
        started.set()
        pool.safe_call("Util_VersionGet", role=role)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    started.wait(timeout=2)
    time.sleep(0.05)  # let it get inside the lock
    return th


# ─────────────────────────────────────────────────────────────────────
# The bound itself
# ─────────────────────────────────────────────────────────────────────

def test_safe_call_refuses_instead_of_queueing_forever(pool):
    from mast.core.connection import is_lock_busy

    gate = threading.Event()
    _install(pool, "main", gate)
    th = _hold_role(pool, "main", gate)
    try:
        t0 = time.perf_counter()
        rec = pool.safe_call("Util_VersionGet", role="main", lock_timeout_s=0.3)
        waited = time.perf_counter() - t0

        assert rec.error, "a busy role must report, not hang"
        assert is_lock_busy(rec), rec.error
        assert 0.25 <= waited < 3.0, f"waited {waited:.2f}s — bound not honoured"
        # It must NOT have touched the socket.
        assert rec.return_value is None
    finally:
        gate.set()
        th.join(timeout=5)


def test_a_free_role_is_unaffected(pool):
    gate = threading.Event()
    gate.set()  # nothing blocks
    _install(pool, "monitor", gate)

    rec = pool.safe_call("Util_VersionGet", role="monitor")
    assert not rec.error, rec.error


def test_different_roles_do_not_block_each_other(pool):
    """The 4-port design is the point: a wedged main must leave emergency free."""
    from mast.core.connection import is_lock_busy

    main_gate = threading.Event()
    emerg_gate = threading.Event()
    emerg_gate.set()
    _install(pool, "main", main_gate)
    _install(pool, "emergency", emerg_gate)
    th = _hold_role(pool, "main", main_gate)
    try:
        rec = pool.safe_call("ZCtrl_Withdraw", 1, -1, role="emergency",
                             lock_timeout_s=0.5)
        assert not is_lock_busy(rec)
        assert not rec.error, rec.error
    finally:
        main_gate.set()
        th.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────
# break_role / urgent_call — the last resort
# ─────────────────────────────────────────────────────────────────────

def test_break_role_closes_the_socket_without_the_role_lock(pool):
    gate = threading.Event()
    sock, nn = _install(pool, "main", gate)
    th = _hold_role(pool, "main", gate)
    try:
        # The role lock is held by the wedged caller. break_role must still work
        # — taking the role lock here would be the deadlock, not the fix.
        t0 = time.perf_counter()
        assert pool.break_role("main", "test") is True
        assert time.perf_counter() - t0 < 2.0
        assert sock.closed and nn.closed
        assert "main" not in pool._connections
    finally:
        gate.set()
        th.join(timeout=5)


def test_break_role_on_an_absent_role_is_a_no_op(pool):
    assert pool.break_role("emergency", "nothing there") is False


def test_urgent_call_unsticks_a_wedged_role(pool):
    """The E-STOP's `role="main"` fallback must get through."""
    gate = threading.Event()
    _install(pool, "main", gate)
    th = _hold_role(pool, "main", gate)
    try:
        t0 = time.perf_counter()
        rec = pool.urgent_call("ZCtrl_Withdraw", 1, -1, role="main",
                               lock_timeout_s=0.3)
        waited = time.perf_counter() - t0
        from mast.core.connection import is_lock_busy

        # The point is not that the retract succeeds here (nothing real is
        # listening) — it is that the emergency path got PAST the wedged lock,
        # fast, instead of parking behind it.
        assert waited < 5.0, f"urgent_call took {waited:.1f}s"
        assert not is_lock_busy(rec), (
            "urgent_call must escalate past a busy lock, not report it")
        assert pool._connections.get("main") is not None or rec.error, (
            "the wedged socket was neither replaced nor reported")
    finally:
        gate.set()
        th.join(timeout=5)


def test_urgent_call_is_a_plain_call_when_nothing_is_stuck(pool):
    gate = threading.Event()
    gate.set()
    _install(pool, "emergency", gate)

    rec = pool.urgent_call("ZCtrl_Withdraw", 1, -1, role="emergency")
    assert not rec.error, rec.error


# ─────────────────────────────────────────────────────────────────────
# Teardown must not hang either
# ─────────────────────────────────────────────────────────────────────

def test_close_all_does_not_block_forever_on_a_wedged_role(pool, monkeypatch):
    import mast.core.connection as conn_mod

    monkeypatch.setattr(conn_mod, "_CLOSE_LOCK_TIMEOUT_S", 0.3)
    gate = threading.Event()
    sock, nn = _install(pool, "main", gate)
    th = _hold_role(pool, "main", gate)
    try:
        t0 = time.perf_counter()
        pool.close_all()
        waited = time.perf_counter() - t0
        assert waited < 3.0, (
            f"close_all blocked {waited:.1f}s — a hung shutdown is what makes "
            "the launcher fall through to TerminateProcess")
        assert sock.closed
    finally:
        gate.set()
        th.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────
# The emergency callers are actually wired to urgent_call
# ─────────────────────────────────────────────────────────────────────

def test_emergency_stop_uses_the_urgent_path():
    """Pin the wiring: MASTApp.emergency_stop and the watchdog SafeRetract must
    go through urgent_call, or the bound above buys the E-STOP nothing."""
    import ast

    root = Path(_MASTV2_ROOT) / "mast" / "core"
    for fname, anchor in (("runtime.py", "def emergency_stop"),
                          ("executor.py", "def on_anomaly")):
        src = (root / fname).read_text(encoding="utf-8", errors="replace")
        assert "urgent_call" in src, f"{fname} still on the queueing path"
        assert anchor in src
        ast.parse(src)  # and it still parses

    # And the three E-STOP verbs are still LITERAL at their call sites, so the
    # abort-policy checker keeps seeing them.
    import re
    src = (root / "runtime.py").read_text(encoding="utf-8", errors="replace")
    verbs = set(re.findall(r'_urgent\(\s*"([A-Za-z0-9_]+)"', src))
    assert {"AutoApproach_OnOffSet", "Motor_StopMove", "Scan_Action",
            "ZCtrl_Withdraw"} <= verbs, verbs


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
