"""Acquisition cadence against a scope that refills in REAL time.

Every other test hands the pump a scripted t0 sequence it can consume as fast as
it likes. That hides the property this file exists for: on real hardware the
buffer only refills every ``n·dt``, so polling faster just burns round trips on
the data role — a port the scan monitor also uses.

Three regressions are pinned here, all of them measured rather than assumed:

* a naive "sleep a fixed slice after a duplicate" loop ran at 45 calls/s
  (≈45% duty) because it probed four times into every 51 ms refill window;
* rendering an alert's evidence PNG inline stalled acquisition for up to a
  second — twenty buffers, lost at exactly the moment something was going wrong;
* the first segment used to cost ~800 ms of lazy imports.

These are wall-clock tests, so the thresholds are deliberately loose: they are
there to catch an order-of-magnitude regression, not to measure performance.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
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

import numpy as np
import pytest

from mast.core.types import NanonisCallRecord
from mast.monitoring.pump import Osci1TPump

DT = 1.0 / 20000.0
NPTS = 1024
TRACE_S = NPTS * DT                 # 51.2 ms — one buffer refill
FLOOR_HZ = 1.0 / TRACE_S            # ~19.5 fetches/s is the physical minimum


class RefillingScope:
    """A scope whose buffer advances with wall time, like the real one.

    Polling faster than ``TRACE_S`` returns the same ``t0`` again — which is what
    the pump has to notice and back off from.
    """

    def __init__(self):
        self.start = time.monotonic()
        self.calls = 0
        self.duplicates = 0
        self.served: set[int] = set()
        self._last = None

    def safe_call(self, method, *a, role="main", **k):
        self.calls += 1
        if method == "Signals_NamesGet":
            return NanonisCallRecord(method=method, return_value=(
                "", b"", [(2,), (2,), ("Z (m)", "Current (A)")]))
        if method == "Osci1T_TimebaseGet":
            return NanonisCallRecord(method=method, return_value=(
                "", b"", [(0,), (1,), [(DT,)]]))
        if method == "Util_RTFreqGet":
            return NanonisCallRecord(method=method,
                                     return_value=("", b"", [(20000.0,)]))
        if method == "Osci1T_DataGet":
            idx = int((time.monotonic() - self.start) / TRACE_S)
            t0 = idx * TRACE_S
            if t0 == self._last:
                self.duplicates += 1
            else:
                self.served.add(idx)
            self._last = t0
            return NanonisCallRecord(method=method, return_value=(
                "", b"", [t0, (DT,), (NPTS,), [(1e-10,)] * NPTS]))
        return NanonisCallRecord(method=method)


def _pump_for(seconds: float, segment_s: float = 1.0):
    scope = RefillingScope()
    pump = Osci1TPump(lambda: scope, segment_s=segment_s)
    stop = threading.Event()
    segments: list = []

    def run():
        for seg in pump.pump_segments(stop):
            segments.append(seg)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(seconds)
    stop.set()
    t.join(timeout=3.0)
    return scope, segments


#: One acquisition run shared by the cadence assertions — this is a wall-clock
#: test and running it three times would cost three times as long for nothing.
_RUN_SECONDS = 6.0


@pytest.fixture(scope="module")
def run():
    scope, segments = _pump_for(_RUN_SECONDS)
    return scope, segments


def test_poll_rate_stays_near_the_physical_floor(run):
    """One fetch per refill is the minimum; four is a bug that costs the scan
    monitor its share of the data role."""
    scope, _ = run
    rate = scope.calls / _RUN_SECONDS

    assert rate < 2.0 * FLOOR_HZ, (
        f"{rate:.1f} calls/s against a {FLOOR_HZ:.1f}/s refill rate — the pump "
        "is probing blindly instead of waiting out the refill window")
    duplicate_frac = scope.duplicates / max(1, scope.calls)
    assert duplicate_frac < 0.35, (
        f"{duplicate_frac:.0%} of fetches returned an already-seen buffer")


def test_most_buffers_are_captured(run):
    """Coverage is not 100% — extracting features pauses the pump between
    segments — but a large drop means the cadence has broken."""
    scope, segments = run
    expected = int(_RUN_SECONDS / TRACE_S)
    coverage = len(scope.served) / expected

    assert coverage > 0.80, f"only {coverage:.0%} of oscilloscope buffers captured"
    assert segments, "no segments were produced at all"


def test_lost_buffers_are_accounted_for_not_hidden(run):
    """Whatever is missed must show up as gap_s, because a spectrum computed
    across a silently-concatenated hole has a step discontinuity in it."""
    scope, segments = run
    expected = int(_RUN_SECONDS / TRACE_S)
    missed = expected - len(scope.served)
    reported_gap = sum(s.gap_s for s in segments)

    if missed > 2:
        assert reported_gap > 0, "buffers were lost but no gap was recorded"
        # each miss is one trace; allow slack for segment boundaries
        assert reported_gap >= 0.5 * missed * TRACE_S


def test_feature_extraction_is_fast_enough_not_to_stall_the_pump():
    """The per-segment work has to be small next to the segment itself.

    A per-sample Python loop in the RTN detector once cost 278 ms — five
    oscilloscope buffers — and it ran on every segment.
    """
    from mast.monitoring import features as F

    fs = 20000.0
    rng = np.random.default_rng(0)
    quiet = 100e-12 + rng.normal(0, 2e-12, int(fs))
    # a genuine two-level trace, which takes the expensive branch
    telegraph = quiet + (rng.random(int(fs)) > 0.5) * 20e-12

    for name, y in (("quiet", quiet), ("telegraph", telegraph)):
        F.compute_segment_features([y], fs)          # warm the lazy imports
        t = time.perf_counter()
        F.compute_segment_features([y], fs)
        elapsed = time.perf_counter() - t
        assert elapsed < 0.15, (
            f"{name} segment took {elapsed * 1000:.0f} ms of a 1000 ms segment")
