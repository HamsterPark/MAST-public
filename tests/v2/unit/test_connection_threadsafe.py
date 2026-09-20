"""ConnectionPool serialises same-role calls.

Two threads on the same role must not interleave on one Nanonis socket; a
per-role lock enforces that, while different roles still run concurrently.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_connection_threadsafe.py -x -v
"""
from __future__ import annotations

import sys
import threading
import time
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
from mast.core.connection import ConnectionPool


class _FakeNanonis:
    """Records if two threads are inside a method at once (per instance)."""
    def __init__(self, tracker, role):
        self._tracker = tracker
        self._role = role

    def Probe(self):
        t = self._tracker
        with t["enter_lock"]:
            t["inside"][self._role] = t["inside"].get(self._role, 0) + 1
            if t["inside"][self._role] > 1:
                t["overlap"][self._role] = True
        time.sleep(0.01)  # widen the window for a race to show
        with t["enter_lock"]:
            t["inside"][self._role] -= 1
        return ("",)  # nanonis-style (no error)

    def close(self):
        pass


def _pool_with(roles):
    pool = ConnectionPool(NanonisConfig())
    tracker = {"enter_lock": threading.Lock(), "inside": {}, "overlap": {}}
    for r in roles:
        # inject a fake connection directly (bypass real TCP)
        pool._connections[r] = (None, _FakeNanonis(tracker, r))
    return pool, tracker


def test_same_role_calls_are_serialised():
    pool, tracker = _pool_with(["monitor"])
    threads = [threading.Thread(target=lambda: pool.safe_call("Probe", role="monitor"))
               for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker["overlap"].get("monitor") is not True  # never two-at-once


def test_different_roles_run_concurrently():
    # main + monitor each have their own fake; they SHOULD be able to overlap.
    pool, tracker = _pool_with(["main", "monitor"])
    barrier = threading.Barrier(2)

    def _call(role):
        barrier.wait()
        pool.safe_call("Probe", role=role)

    ta = threading.Thread(target=_call, args=("main",))
    tb = threading.Thread(target=_call, args=("monitor",))
    ta.start(); tb.start(); ta.join(); tb.join()
    # no assertion on overlap (timing-dependent) — the point is no deadlock and
    # both complete; per-role locks are distinct so cross-role never blocks.
    assert "main" in tracker["inside"] and "monitor" in tracker["inside"]


def test_get_and_close_threadsafe_smoke():
    pool, _ = _pool_with(["main"])
    # concurrent get + close_all must not raise / corrupt
    errs = []

    def _spam_get():
        for _ in range(200):
            try:
                pool.get("main")
            except ConnectionError:
                pass
            except Exception as e:  # pragma: no cover
                errs.append(e)

    th = [threading.Thread(target=_spam_get) for _ in range(4)]
    for t in th:
        t.start()
    pool.close_all()
    for t in th:
        t.join()
    assert not errs


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
