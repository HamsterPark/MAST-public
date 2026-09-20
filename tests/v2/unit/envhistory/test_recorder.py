"""The recorder: the segment hook, the sweep trigger, and the promise that
none of it can bite the acquisition path.

The hook lives inside ``CurrentMonitorService._on_segment`` — a code path that
was commissioned days ago and is watching the tip. So the test that matters most
is the one where the recorder throws: the segment must still land in the
monitor's store, untouched.
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

import time

import numpy as np
import pytest

from mast.envhistory.recorder import EnvHistoryRecorder, get_recorder, set_recorder
from mast.envhistory.store import EnvHistoryStore
from mast.envhistory.thresholds import EnvHistoryThresholds

FS = 20000.0


class _Segment:
    """Same shape as mast.monitoring.pump.Segment (only what the hook reads)."""

    def __init__(self, *, runs=None, gap_s=0.0, discontinuity=False, dur=1.0):
        n = int(FS * dur)
        rng = np.random.default_rng(7)
        self.runs = runs if runs is not None else [rng.normal(0, 2e-11, n)]
        self.fs_hz = FS
        self.t_start = 1000.0
        self.t_end = 1000.0 + dur
        self.gap_s = gap_s
        self.discontinuity = discontinuity
        self.n_samples = n


QUIET = {"ctx_zctrl_on": True, "ctx_scanning": False, "ctx_skill": "",
         "ctx_stale": False, "ctx_bias_v": 0.5}
BUSY = dict(QUIET, ctx_skill="TipShaping")


@pytest.fixture()
def store(tmp_path):
    s = EnvHistoryStore(tmp_path / "eh.sqlite")
    yield s
    s.close()


def _rec(store, **kw):
    th = kw.pop("th", EnvHistoryThresholds(eh_spectrum_interval_s=300.0,
                                           eh_spectrum_min_segments=2,
                                           eh_spectrum_accum_every_s=0.0))
    r = EnvHistoryRecorder(store_getter=lambda: store,
                           thresholds_getter=lambda: th, **kw)
    set_recorder(r)
    return r


def test_quiet_segments_accumulate_and_emit_a_spectrum(store):
    r = _rec(store)
    for _ in range(4):
        r.on_current_segment(_Segment(), {}, QUIET)
    # Force the window open by pretending the accumulation started long ago.
    r._spec_current._t_first -= 10_000.0            # noqa: SLF001 — clock seam
    r.on_current_segment(_Segment(), {}, QUIET)
    rows = store.spectra_query(channel="current")
    assert len(rows) == 1
    assert rows[0]["n_segments"] >= 2
    assert rows[0]["unit"] == "A^2/Hz"


def test_busy_segments_are_not_accumulated(store):
    r = _rec(store)
    for _ in range(6):
        r.on_current_segment(_Segment(), {}, BUSY)
    assert r._spec_current.n_accum == 0              # noqa: SLF001
    assert store.spectra_query() == []


def test_mostly_gap_segments_are_rejected(store):
    """RoleBusy back-off can book seconds of hole into a one-second segment."""
    r = _rec(store)
    r.on_current_segment(_Segment(gap_s=0.9), {}, QUIET)
    assert r._spec_current.n_accum == 0              # noqa: SLF001


def test_discontinuous_segments_are_rejected(store):
    """Somebody re-configured the scope; the fs label may not match the data."""
    r = _rec(store)
    r.on_current_segment(_Segment(discontinuity=True), {}, QUIET)
    assert r._spec_current.n_accum == 0              # noqa: SLF001


def test_spectra_disabled_means_no_work(store):
    r = _rec(store, th=EnvHistoryThresholds(eh_spectra_enabled=0.0))
    r.on_current_segment(_Segment(), {}, QUIET)
    assert r._spec_current.n_accum == 0              # noqa: SLF001


def test_master_switch_disables_spectra_too(store):
    r = _rec(store, th=EnvHistoryThresholds(eh_enabled=0.0,
                                            eh_spectra_enabled=1.0))
    r.on_current_segment(_Segment(), {}, QUIET)
    assert r._spec_current.n_accum == 0              # noqa: SLF001


def test_subsampling_limits_the_fft_rate(store):
    r = _rec(store, th=EnvHistoryThresholds(eh_spectrum_accum_every_s=3600.0))
    for _ in range(5):
        r.on_current_segment(_Segment(), {}, QUIET)
    assert r._spec_current.n_accum == 1              # noqa: SLF001


def test_the_hook_never_raises(store):
    """Anything malformed arriving from the acquisition path is swallowed."""
    r = _rec(store)
    r.on_current_segment(None, None, QUIET)
    r.on_current_segment(object(), {}, QUIET)
    r.on_current_segment(_Segment(runs=[]), {}, QUIET)
    assert store.spectra_query() == []


def test_insufficient_quiet_is_not_reported_on_a_fresh_system(store):
    """The flag means "the window has been open far too long and the instrument
    is never idle" — not "no snapshot has ever been written", which is the
    normal state of a machine that just booted."""
    r = _rec(store)
    r.on_current_segment(_Segment(), {}, QUIET)
    assert r.status()["spectra"]["current"]["insufficient_quiet"] is False


def test_insufficient_quiet_is_reported_once_the_window_drags(store):
    r = _rec(store, th=EnvHistoryThresholds(eh_spectrum_interval_s=300.0,
                                            eh_spectrum_min_segments=50,
                                            eh_spectrum_accum_every_s=0.0))
    r.on_current_segment(_Segment(), {}, QUIET)
    r._spec_current._t_first -= 10_000.0             # noqa: SLF001 — clock seam
    r.on_current_segment(_Segment(), {}, QUIET)
    assert r.status()["spectra"]["current"]["insufficient_quiet"] is True


def test_scope_is_attached_to_the_spectrum(store):
    r = _rec(store, scope_provider=lambda: ("/exp", "S01", "E1", "SID1"))
    for _ in range(3):
        r.on_current_segment(_Segment(), {}, QUIET)
    r._spec_current._t_first -= 10_000.0             # noqa: SLF001
    r.on_current_segment(_Segment(), {}, QUIET)
    row = store.spectra_query()[0]
    assert row["experiment_id"] == "E1" and row["sample_id"] == "SID1"


# ── the sweep trigger ───────────────────────────────────────────────────────

class _Storage:
    def __init__(self):
        self.calls = []

    def prune_environment_log(self, cutoff, **kw):
        self.calls.append(cutoff)
        return 7


def test_tick_runs_the_sweep_once_per_interval(store):
    st = _Storage()
    r = _rec(store, storage_getter=lambda: st,
             th=EnvHistoryThresholds(eh_sweep_interval_s=1e9))
    r.on_tick(time.time(), QUIET, "quiet")
    _join(r)
    assert len(st.calls) == 1
    r.on_tick(time.time(), QUIET, "quiet")           # interval not elapsed
    _join(r)
    assert len(st.calls) == 1
    assert r.status()["sweep"]["rows_pruned"] == 7


def test_sweep_is_skipped_when_recording_is_off(store):
    st = _Storage()
    r = _rec(store, storage_getter=lambda: st,
             th=EnvHistoryThresholds(eh_enabled=0.0))
    r.on_tick(time.time(), QUIET, "quiet")
    _join(r)
    assert st.calls == []


def test_a_storage_that_explodes_does_not_kill_the_tick(store):
    class _Boom:
        def prune_environment_log(self, cutoff, **kw):
            raise RuntimeError("db locked")

    r = _rec(store, storage_getter=lambda: _Boom())
    r.on_tick(time.time(), QUIET, "quiet")           # must not raise
    _join(r)
    assert r.status()["sweep"].get("error") is True


def test_z_burst_does_not_start_when_disabled(store):
    calls = []
    r = _rec(store, pool_getter=lambda: _FakePool(calls))
    r.on_tick(time.time(), QUIET, "quiet")
    _join(r)
    assert calls == []                               # eh_z_enabled defaults off


def test_z_burst_is_skipped_while_the_instrument_is_busy(store):
    calls = []
    r = _rec(store, pool_getter=lambda: _FakePool(calls),
             th=EnvHistoryThresholds(eh_z_enabled=1.0))
    r.on_tick(time.time(), BUSY, "active")
    _join(r)
    assert calls == []
    assert "不空闲" in r.status()["spectra"]["z"]["last_result"].get("skipped", "")


def test_z_burst_is_skipped_when_the_breaker_is_open(store):
    calls = []
    r = _rec(store, pool_getter=lambda: _FakePool(calls, healthy=False),
             th=EnvHistoryThresholds(eh_z_enabled=1.0))
    r.on_tick(time.time(), QUIET, "quiet")
    _join(r)
    assert calls == []                               # ZERO TCP into an open breaker
    assert "熔断" in r.status()["spectra"]["z"]["last_result"].get("skipped", "")


def test_status_shape(store):
    r = _rec(store)
    st = r.status()
    assert set(st) >= {"enabled", "spectra_enabled", "z_enabled", "sink",
                       "spectra", "sweep", "store"}
    assert set(st["spectra"]) == {"current", "z"}


def test_stop_flushes_and_is_idempotent(store):
    r = _rec(store)
    r.sink.write("temperature", 77.0, "K", "ok")
    r.stop()
    r.stop()
    assert store.series("temperature")["points"]


def test_singleton_round_trip(store):
    r = _rec(store)
    assert get_recorder() is r
    set_recorder(None)
    assert get_recorder() is None


class _FakePool:
    def __init__(self, calls, healthy=True):
        self.calls = calls
        self._healthy = healthy

    def comms_healthy(self):
        return self._healthy

    def safe_call(self, verb, *a, **kw):
        self.calls.append(verb)
        raise AssertionError("burst must not have started")


def _join(r, timeout=5.0):
    for attr in ("_sweep_thread", "_burst_thread"):
        t = getattr(r, attr, None)
        if t is not None:
            t.join(timeout=timeout)
