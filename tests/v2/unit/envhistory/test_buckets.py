"""Incremental aggregation: does a bucket say what actually happened?

The two failure modes that matter here are both silent:

  * a reading the sensor never took getting into the statistics (``read_all``
    writes ``value=0.0`` when a driver raises, and a 0 in a 77 K series is
    indistinguishable from a cryostat failure at a glance);
  * a bucket boundary that stops advancing, which looks exactly like "the
    recorder is fine, nothing is changing".

Both get explicit coverage. The statistics themselves are checked against
numpy rather than against hand-computed constants — a test that repeats the
implementation's arithmetic is satisfied by the implementation's bugs.
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

import numpy as np

from mast.envhistory.buckets import BucketAccumulator, bucket_start

T0 = 1_800_000_000.0
DT = 60.0


def _feed(acc, values, *, start=T0, step=2.0, unit="K", status="ok", dt=DT,
          admit=True):
    out = []
    for i, v in enumerate(values):
        b = acc.add(start + i * step, v, unit, status, dt_s=dt, admit=admit)
        if b is not None:
            out.append(b)
    return out


def test_bucket_start_floors_to_the_grid():
    assert bucket_start(T0 + 5.0, 60.0) == T0
    assert bucket_start(T0 + 59.9, 60.0) == T0
    assert bucket_start(T0 + 60.0, 60.0) == T0 + 60.0


def test_statistics_match_numpy():
    acc = BucketAccumulator("temperature")
    vals = [77.0 + 0.013 * i for i in range(30)]
    _feed(acc, vals)
    b = acc.flush()
    assert b is not None
    assert b.n == 30
    # Tolerance, not equality: Welford accumulates incrementally while numpy
    # sums pairwise, so they agree to ~1e-13 rather than bit-for-bit. Demanding
    # equality would be testing IEEE addition order, not the aggregation.
    assert abs(b.mean - float(np.mean(vals))) < 1e-10
    assert b.min == min(vals)
    assert b.max == max(vals)
    # ddof=0 on purpose — documented in buckets.py so this comparison is exact.
    assert abs(b.std - float(np.std(vals))) < 1e-12
    assert b.last == vals[-1]
    assert b.unit == "K"


def test_error_placeholder_value_never_enters_the_statistics():
    """``read_all`` substitutes ``SensorReading(value=0.0, status="error")``
    when a driver raises. That 0.0 is not a reading."""
    acc = BucketAccumulator("temperature")
    acc.add(T0, 77.1, "K", "ok", dt_s=DT)
    acc.add(T0 + 2, 0.0, "", "error", dt_s=DT)
    acc.add(T0 + 4, 77.3, "K", "ok", dt_s=DT)
    b = acc.flush()
    assert b.n == 2
    assert b.n_excluded == 1
    assert b.min == 77.1          # NOT 0.0
    assert b.mean == (77.1 + 77.3) / 2
    # …but the fault is still on the record.
    assert b.worst_status == "error"


def test_unavailable_is_excluded_but_kept_in_worst_status():
    acc = BucketAccumulator("helium_level")
    acc.add(T0, 0.0, "%", "unavailable", dt_s=DT)
    b = acc.flush()
    assert b.n == 0
    assert b.n_excluded == 1
    assert b.mean is None and b.min is None
    assert b.worst_status == "unavailable"


def test_quiet_gate_excludes_without_hiding_the_status():
    acc = BucketAccumulator("tunnel_current")
    acc.add(T0, 2e-11, "A", "ok", dt_s=DT, admit=True)
    acc.add(T0 + 2, 5e-9, "A", "ok", dt_s=DT, admit=False)   # scanning
    b = acc.flush()
    assert b.n == 1
    assert b.n_excluded == 1
    assert b.max == 2e-11           # the scan's current never enters the trend


def test_alarm_survives_even_when_every_reading_was_excluded():
    acc = BucketAccumulator("vacuum")
    acc.add(T0, 0.0, "Pa", "alarm", dt_s=DT)
    b = acc.flush()
    assert b.worst_status == "alarm"


def test_crossing_a_boundary_emits_exactly_one_bucket():
    acc = BucketAccumulator("temperature")
    done = _feed(acc, [77.0] * 45)   # 45 × 2 s = 90 s → crosses one boundary
    assert len(done) == 1
    assert done[0].bucket_ts == T0
    assert done[0].n == 30           # 60 s / 2 s


def test_clock_running_backwards_still_advances():
    """A backward system-time jump must not freeze the recorder.

    An implementation that only rolls forward would judge every later reading
    "belongs to a past bucket" and drop it — the visible symptom is a series
    that quietly stops, which is exactly what this project already shipped once
    in the scan-map layer.
    """
    acc = BucketAccumulator("temperature")
    acc.add(T0 + 120, 77.0, "K", "ok", dt_s=DT)
    done = acc.add(T0, 77.5, "K", "ok", dt_s=DT)     # clock jumped back 2 min
    assert done is not None
    assert done.bucket_ts == T0 + 120
    b = acc.flush()
    assert b.bucket_ts == T0                        # and it kept accumulating
    assert b.n == 1


def test_changing_the_bucket_width_flushes_the_open_bucket():
    acc = BucketAccumulator("temperature")
    acc.add(T0, 77.0, "K", "ok", dt_s=60.0)
    done = acc.add(T0 + 2, 77.1, "K", "ok", dt_s=600.0)
    assert done is not None
    assert done.dt_s == 60.0 and done.n == 1
    b = acc.flush()
    assert b.dt_s == 600.0 and b.n == 1


def test_scope_uses_the_majority_not_the_last_reading():
    """A sample switch at second 58 must not reattribute the whole minute."""
    acc = BucketAccumulator("temperature")
    for i in range(29):
        acc.add(T0 + i * 2, 77.0, "K", "ok", dt_s=DT,
                experiment_id="E1", sample_id="S01")
    acc.add(T0 + 58, 77.0, "K", "ok", dt_s=DT, experiment_id="E1", sample_id="S02")
    b = acc.flush()
    assert b.sample_id == "S01"


def test_flush_on_an_untouched_accumulator_is_none():
    assert BucketAccumulator("x").flush() is None


def test_non_finite_values_are_excluded():
    acc = BucketAccumulator("x")
    acc.add(T0, float("nan"), "", "ok", dt_s=DT)
    acc.add(T0 + 2, float("inf"), "", "ok", dt_s=DT)
    acc.add(T0 + 4, 1.0, "", "ok", dt_s=DT)
    b = acc.flush()
    assert b.n == 1 and b.n_excluded == 2
    assert b.mean == 1.0


def test_to_row_matches_the_ddl_column_order():
    from mast.envhistory.store import _BUCKET_COLS
    acc = BucketAccumulator("temperature")
    acc.add(T0, 77.0, "K", "ok", dt_s=DT, experiment_id="E", sample_id="S")
    row = acc.flush().to_row()
    assert len(row) == len(_BUCKET_COLS)
    assert row[0] == "temperature"
    assert row[_BUCKET_COLS.index("unit")] == "K"
    assert row[_BUCKET_COLS.index("experiment_id")] == "E"
