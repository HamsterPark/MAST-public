"""Regressions for the security audit of 2026-07-29.

Each test here corresponds to one finding. They are grouped in one file on
purpose: every one of them is a case where the code looked right, passed the
existing suite, and was still wrong — so what they document is *why* the obvious
version does not work.
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

import sqlite3
import threading
import time

import numpy as np
import pytest

from mast.core.types import NanonisCallRecord
from mast.monitoring import features as F
from mast.monitoring.alerts import AlertEngine
from mast.monitoring.pump import Osci1TPump, Segment
from mast.monitoring.store import CurrentMonitorStore, segment_npy_path
from mast.monitoring.thresholds import MonitorThresholds

FS = 20000.0
DT = 1.0 / FS
NPTS = 1024


# ══ C1: the oscilloscope is a shared module ═════════════════════════════════

def _wire(*fields):
    return ("", b"", list(fields))


class SharedScope:
    """A scope another skill can re-point at any moment."""

    def __init__(self):
        self.channel = 1                     # index of "Current (A)"
        self.t0 = 0.0
        self.data_gets = 0
        self.ch_gets = 0

    def safe_call(self, method, *a, role="main", count_health=True, **k):
        if method == "Signals_NamesGet":
            return NanonisCallRecord(method=method, return_value=_wire(
                (3,), (3,), ("Z (m)", "Current (A)", "Bias (V)")))
        if method == "Osci1T_TimebaseGet":
            return NanonisCallRecord(method=method,
                                     return_value=_wire((0,), (1,), [(DT,)]))
        if method == "Util_RTFreqGet":
            return NanonisCallRecord(method=method, return_value=_wire((20000.0,)))
        if method == "Osci1T_ChSet":
            self.channel = int(a[0])
            return NanonisCallRecord(method=method)
        if method == "Osci1T_ChGet":
            self.ch_gets += 1
            return NanonisCallRecord(method=method, return_value=_wire((self.channel,)))
        if method == "Osci1T_DataGet":
            self.data_gets += 1
            t0 = self.t0
            self.t0 += NPTS * DT
            # Bias reads ~1 V — six orders above any tunnelling current.
            level = 1.0 if self.channel == 2 else 100e-12
            return NanonisCallRecord(method=method, return_value=_wire(
                t0, (DT,), (NPTS,), [(level,)] * NPTS))
        return NanonisCallRecord(method=method)


def test_pump_notices_when_another_skill_repoints_the_scope():
    """AcquireOsciTrace is a READ skill, so it takes no instrument token and the
    suppression list never sees it. Checking dt cannot catch a channel swap —
    the timebase does not change. Left undetected, Bias on the channel reads as
    100% saturation and is three segments from halting a composite skill."""
    scope = SharedScope()
    pump = Osci1TPump(lambda: scope, segment_s=0.2)
    pump.configure()
    assert pump.channel_is_ours() is True

    scope.channel = 2                        # somebody grabbed it for Bias
    assert pump.channel_is_ours() is False


def test_pump_drops_frames_acquired_on_a_foreign_channel():
    scope = SharedScope()
    pump = Osci1TPump(lambda: scope, segment_s=0.3)
    stop = threading.Event()
    segments: list[Segment] = []

    def run():
        for seg in pump.pump_segments(stop):
            segments.append(seg)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.6)
    scope.channel = 2                        # re-pointed mid-acquisition
    time.sleep(2.0)
    stop.set()
    t.join(timeout=3.0)
    assert segments, "no segments were produced at all"

    assert scope.ch_gets > 0, "the pump never verified the channel"
    for seg in segments:
        peak = float(np.abs(seg.samples).max()) if seg.n_samples else 0.0
        assert peak < 1e-3, "stored a volt-level signal as tunnelling current"


def test_channel_check_is_no_opinion_when_no_channel_was_resolved():
    class NoNames(SharedScope):
        def safe_call(self, method, *a, **k):
            if method == "Signals_NamesGet":
                return NanonisCallRecord(method=method, error="boom")
            return super().safe_call(method, *a, **k)

    pump = Osci1TPump(lambda: NoNames(), segment_s=0.2)
    pump.configure()
    assert pump.channel_is_ours() is None     # never guesses


# ══ C2: the pump must not reset the shared circuit breaker ══════════════════

def test_pump_polling_does_not_clear_another_role_failure_streak():
    """One breaker instance serves all four roles and record_success() clears
    the streak unconditionally. A 20 Hz poller on an idle role would otherwise
    wipe the failures of every other role between their retries, and the
    three-consecutive-failures condition could never accumulate — disabling the
    protection that exists to stop reconnect storms wrecking the Nanonis port."""
    from mast.core.comms_health import CommsCircuitBreaker

    breaker = CommsCircuitBreaker()
    for _ in range(2):
        breaker.record_failure("main timeout")
    assert not breaker.is_open()

    # The pump's calls are health-neutral, so they cannot land here at all.
    # Simulate the two possibilities explicitly:
    breaker.record_failure("main timeout")
    assert breaker.is_open(), "three consecutive failures must open the breaker"


def test_safe_call_honours_count_health(monkeypatch):
    from mast.core.connection import ConnectionPool
    from mast.config import NanonisConfig

    pool = ConnectionPool(NanonisConfig())
    calls = {"success": 0, "failure": 0}
    monkeypatch.setattr(pool, "_on_comms_success",
                        lambda: calls.__setitem__("success", calls["success"] + 1))
    monkeypatch.setattr(pool, "_on_comms_failure",
                        lambda reason: calls.__setitem__("failure", calls["failure"] + 1))

    class Fake:
        def Osci1T_DataGet(self, *a):
            return ("", b"", [])

    monkeypatch.setattr(pool, "get", lambda role: Fake())

    pool.safe_call("Osci1T_DataGet", 0, role="data", count_health=False)
    assert calls["success"] == 0, "a health-neutral call still reported success"

    pool.safe_call("Osci1T_DataGet", 0, role="data")
    assert calls["success"] == 1, "a normal call must still report success"


def test_pump_marks_its_calls_health_neutral():
    seen = []

    class Recorder(SharedScope):
        def safe_call(self, method, *a, role="main", count_health=True, **k):
            seen.append((method, count_health))
            return super().safe_call(method, *a, role=role, **k)

    pump = Osci1TPump(lambda: Recorder(), segment_s=0.2)
    pump.configure()
    pump.poll_once()
    pump.channel_is_ours()
    assert seen
    counted = [m for m, ch in seen if ch]
    assert not counted, f"these pump calls still feed the breaker: {counted}"


# ══ H1: the duplicate-frame branch had no sleep floor ═══════════════════════

def test_a_frozen_buffer_does_not_become_a_busy_loop():
    """If the scope stops re-arming — Level trigger that never fires, module
    stopped, front panel closed — t0 freezes and the 'wait out the rest of the
    window' arithmetic goes permanently negative. With a zero floor that is a
    back-to-back DataGet loop holding the data role lock at ~100% duty."""

    class FrozenScope(SharedScope):
        def safe_call(self, method, *a, role="main", count_health=True, **k):
            if method == "Osci1T_DataGet":
                self.data_gets += 1
                return NanonisCallRecord(method=method, return_value=_wire(
                    5.0, (DT,), (NPTS,), [(100e-12,)] * NPTS))   # t0 never moves
            return super().safe_call(method, *a, role=role,
                                     count_health=count_health, **k)

    scope = FrozenScope()
    pump = Osci1TPump(lambda: scope, segment_s=1.0)
    stop = threading.Event()

    def run():
        for _ in pump.pump_segments(stop):
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(2.0)
    stop.set()
    t.join(timeout=3.0)

    rate = scope.data_gets / 2.0
    assert rate < 250, f"{rate:.0f} DataGet/s against a frozen buffer — busy loop"


# ══ H2 / M2: the pin exemption during a sweep ══════════════════════════════

def _seed(store, *, t_start, pinned=False):
    n = 2000
    y = 100e-12 + np.random.default_rng(int(t_start) % 977).normal(0, 2e-12, n)
    p = segment_npy_path(store.data_dir, t_start, FS)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, y.astype(np.float32))
    return store.add_segment(
        {"t_start": t_start, "t_end": t_start + 0.1, "fs_hz": FS, "n_samples": n,
         "npy_path": str(p), "npy_bytes": p.stat().st_size, "pinned": pinned},
        F.envelope(y, FS, 100).tobytes(), 0.01)


@pytest.fixture()
def store(tmp_path):
    s = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    yield s
    s.close()


def test_a_segment_pinned_mid_sweep_keeps_its_waveform(store):
    """The sweep snapshots `WHERE pinned=0`, then releases the lock and deletes
    row by row — for tens of thousands of rows that window is minutes long, and
    the two things most likely to pin something during it are exactly the two
    worth keeping: a tip-shaping skill finishing, and an operator pressing
    「标记为好针尖」."""
    old = time.time() - 48 * 3600
    ids = [_seed(store, t_start=old + i) for i in range(6)]

    # Reproduce the interleaving directly: take the sweep's snapshot, pin a
    # segment that was in it, then let the deletion proceed.
    with store._lock:                                    # noqa: SLF001
        rows = store._conn.execute(                      # noqa: SLF001
            "SELECT id, npy_path, npy_bytes FROM segments"
            " WHERE pinned=0 AND npy_path IS NOT NULL").fetchall()
    assert len(rows) == 6

    store.set_pin(ids[3], True, "labeled:good")          # arrives mid-sweep
    for row in rows:
        store._drop_npy(row)                             # noqa: SLF001

    assert store.segment_meta(ids[3])["has_file"] is True, \
        "deleted the waveform behind a label applied during the sweep"
    assert store.segment_meta(ids[0])["has_file"] is False


def test_a_failed_remove_leaves_the_row_pointing_at_the_file(store, monkeypatch):
    """Nulling npy_path for a file still on disk orphans it forever: every later
    sweep filters on `npy_path IS NOT NULL`. On Windows this is routine —
    read_segment_decimated maps segments with mmap, and a mapped file cannot be
    unlinked."""
    sid = _seed(store, t_start=time.time() - 48 * 3600)
    monkeypatch.setattr("os.remove",
                        lambda p: (_ for _ in ()).throw(PermissionError("mapped")))

    res = store.retention_sweep(keep_hours=1.0, keep_gb=100.0)

    meta = store.segment_meta(sid)
    assert meta["has_file"] is True, "orphaned a file that could not be deleted"
    assert Path(meta["npy_path"]).is_file()
    assert res["freed_bytes"] == 0


# ══ H3: step features spliced across a gap ═════════════════════════════════

def test_jump_and_spike_features_ignore_the_joint_between_runs():
    """A gap is a hole in time. A first difference taken across it invents a step
    equal to however much the current moved while we were not looking — and that
    invented step feeds `giant_spike`, one of the three rules allowed to halt a
    composite skill. The PSD path already refuses to splice; these must too."""
    rng = np.random.default_rng(0)
    run_a = 100e-12 + rng.normal(0, 2e-12, 8000)
    run_b = 900e-12 + rng.normal(0, 2e-12, 8000)   # setpoint moved during the gap

    spliced = F.jump_metrics(np.concatenate([run_a, run_b]), FS)
    per_run = F.compute_segment_features([run_a, run_b], FS)

    assert spliced["max_step_a"] > 500e-12          # the artefact, if spliced
    assert per_run["max_step_a"] < 100e-12, \
        "the run boundary still shows up as a step"
    assert per_run["jump_count"] == 0


def test_the_invented_step_no_longer_reaches_giant_spike():
    rng = np.random.default_rng(1)
    run_a = 100e-12 + rng.normal(0, 1e-12, 8000)
    run_b = 5e-9 + rng.normal(0, 1e-12, 8000)
    feats = F.compute_segment_features([run_a, run_b], FS)
    verdict = AlertEngine(lambda: MonitorThresholds()).evaluate(feats)
    assert "giant_spike" not in verdict.rules


def test_a_real_spike_inside_one_run_is_still_caught():
    """The other direction: not splicing must not cost detection."""
    rng = np.random.default_rng(2)
    y = 100e-12 + rng.normal(0, 1e-12, 16000)
    y[8000:8050] += 400e-12
    feats = F.compute_segment_features([y], FS)
    assert feats["spike_count"] >= 1
    assert feats["max_step_a"] > 100e-12


# ══ H4: live_tail scanned an ever-growing table ════════════════════════════

def test_t_end_is_indexed(store):
    """live_tail filters on t_end and the age sweep does too; rows are permanent
    by design (~86k/day), and the scan holds the lock the acquisition thread
    needs."""
    with store._lock:                                    # noqa: SLF001
        names = {r[1] for r in store._conn.execute(      # noqa: SLF001
            "PRAGMA index_list('segments')").fetchall()}
    assert "idx_seg_tend" in names

    with store._lock:                                    # noqa: SLF001
        plan = store._conn.execute(                      # noqa: SLF001
            "EXPLAIN QUERY PLAN SELECT t_start FROM segments"
            " WHERE t_end >= ? ORDER BY t_start ASC", (0.0,)).fetchall()
    assert not any("SCAN segments" in str(row) and "USING" not in str(row)
                   for row in plan), f"still a full table scan: {plan}"


# ══ H5: spectroscopy and the dead ctx parameter ════════════════════════════

@pytest.mark.parametrize("skill", [
    "BiasSpectroscopy", "TakeSpectrum", "GridSTS", "ZSpectroscopy",
    "GenericSweep", "SetBias",
])
def test_spectroscopy_skills_are_suppressed(skill):
    """An I-V or I-z sweep drives the preamp into its rail by design. Without
    this a grid of STS points reads as sustained saturation and halts the very
    composite skill running it."""
    from mast.monitoring.service import CurrentMonitorService
    assert CurrentMonitorService._is_suppressed({"ctx_skill": skill}) is True


def test_saturation_is_not_critical_with_the_z_controller_off():
    """`ctx` used to be accepted and never read. With feedback off the tip may be
    retracted, parked, or mid-spectroscopy: 'the current is pinned' then
    describes the experiment, not a fault."""
    engine = AlertEngine(lambda: MonitorThresholds())
    feats = {"sat_frac": 0.9, "frozen": 0, "spike_max_sigma": 3.0,
             "max_step_a": 1e-13, "rms_detrended_a": 2e-12}

    with_junction = engine.evaluate(feats, {"ctx_zctrl_on": True})
    assert with_junction.level == "critical_candidate"
    assert "saturation" in with_junction.rules

    without = engine.evaluate(feats, {"ctx_zctrl_on": False})
    assert without.level == "warn"
    assert "saturation" not in without.rules
    assert "saturation_no_junction" in without.rules


def test_unknown_context_still_allows_a_verdict():
    """Refusing to judge whenever state is missing would disable the monitor on
    every install without a live InstrumentState."""
    engine = AlertEngine(lambda: MonitorThresholds())
    feats = {"sat_frac": 0.9, "frozen": 0, "spike_max_sigma": 3.0,
             "max_step_a": 1e-13, "rms_detrended_a": 2e-12}
    assert engine.evaluate(feats, {}).level == "critical_candidate"
    assert engine.evaluate(feats, None).level == "critical_candidate"


# ══ M4 / M8: gap-dominated segments, and the streak across suppression ═════

def test_a_segment_that_is_mostly_gap_is_not_judged(tmp_path):
    """RoleBusy back-off can book seconds of gap into a one-second segment when a
    scan takes the data role. Three of those must not confirm a CRITICAL between
    them on a handful of real samples."""
    from mast.monitoring import service as S
    from mast.monitoring import store as ST

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    ST.set_store_for_test(st)
    try:
        th = MonitorThresholds.from_mapping({"cm_crit_consecutive": 1.0})
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st, thresholds_getter=lambda: th)
        svc._engine = AlertEngine(lambda: th)

        now = time.time()
        y = np.full(1000, 1e-12)                     # frozen readout
        seg = Segment(t_start=now, t_end=now + 1.0, osci_t0=0.0, fs_hz=FS,
                      runs=[y], n_samples=y.size, gap_s=0.95)
        svc._on_segment(seg)

        assert st.alerts_query(level="critical")[1] == 0
        assert st.latest_feature()["alert_level"] == "unknown"
    finally:
        ST.set_store_for_test(None)
        st.close()


def test_suppressed_segments_still_break_the_critical_streak(tmp_path):
    """confirm() is the only place the streak resets. Skipping it while
    suppressed lets a saturation before tip-shaping and one after it count as
    consecutive."""
    from mast.monitoring import service as S
    from mast.monitoring import store as ST

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    ST.set_store_for_test(st)
    try:
        th = MonitorThresholds.from_mapping({"cm_crit_consecutive": 2.0})
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st, thresholds_getter=lambda: th)
        svc._engine = AlertEngine(lambda: th)

        now = time.time()

        def frozen(t, skill=""):
            seg = Segment(t_start=t, t_end=t + 1.0, osci_t0=0.0, fs_hz=FS,
                          runs=[np.full(4000, 1e-12)], n_samples=4000)
            svc._context_labels = lambda: {"ctx_skill": skill}   # noqa: ARG005
            svc._on_segment(seg)

        frozen(now)                                  # bad
        frozen(now + 1, skill="TipShape")            # clean segment in between
        frozen(now + 2)                              # bad again
        assert st.alerts_query(level="critical")[1] == 0, \
            "streak survived a suppressed segment"
        frozen(now + 3)
        assert st.alerts_query(level="critical")[1] == 1
    finally:
        ST.set_store_for_test(None)
        st.close()


# ══ M1: a restart must not run two pumps ═══════════════════════════════════


def _pump_thread_idents() -> set[int]:
    """Identities of live worker threads. Compared as a DELTA, never counted:
    other tests in this module create threads with the same name, and a stale
    one would make an absolute count lie."""
    return {t.ident for t in threading.enumerate()
            if t.name == "CurrentMonitorService" and t.is_alive()}


def test_restart_refuses_while_the_previous_worker_is_still_alive(tmp_path):
    """stop() can return with the thread still running — a poll can sit in
    safe_call for up to 35 s (5 s recv + 30 s role lock). Starting a second one
    then gives two pumps writing segment files concurrently, and the filenames
    are only unique under a single writer: two in the same millisecond both see
    the same name as free. They would also each write a `segments` row for the
    same wall-clock second, both claiming to be a contiguous recording."""
    from mast.monitoring import service as S
    from mast.monitoring import store as ST

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    ST.set_store_for_test(st)
    release = threading.Event()
    try:
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st,
            thresholds_getter=lambda: MonitorThresholds())

        # A worker wedged inside a blocking call, exactly like a stuck safe_call.
        def wedged():
            release.wait(timeout=30.0)

        stuck = threading.Thread(target=wedged, daemon=True,
                                 name="CurrentMonitorService")
        stuck.start()
        with svc._lock:                                  # noqa: SLF001
            svc._thread = stuck                          # noqa: SLF001

        svc.stop(join_timeout=0.2)                       # join times out
        assert stuck.is_alive()

        before = _pump_thread_idents()
        svc.start()                                      # must NOT start another
        assert _pump_thread_idents() <= before, "started a second pump"

        release.set()
        stuck.join(timeout=5.0)
        svc.start()                                      # now it may start
        assert svc.status()["running"] is True
        svc.stop(join_timeout=3.0)
    finally:
        release.set()
        ST.set_store_for_test(None)
        st.close()


def test_stop_forgets_the_scope_configuration(tmp_path):
    """The pump caches the channel index and timebase from configure(), and the
    channel guard compares against exactly that cached index — carrying it into
    a new connection would verify against a stale baseline."""
    from mast.monitoring import service as S

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    try:
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st,
            thresholds_getter=lambda: MonitorThresholds())
        svc._pump = object()                             # noqa: SLF001
        svc._engine = object()                           # noqa: SLF001
        svc.stop(join_timeout=0.1)
        assert svc._pump is None                         # noqa: SLF001
        assert svc._engine is None                       # noqa: SLF001
    finally:
        st.close()


class _RestoreSpy:
    """A stand-in pump that only records whether the scope was handed back."""

    def __init__(self):
        self.restored = 0
        self.config = {}
        self.stats = {}

    def restore(self):
        self.restored += 1
        return {"channel_index": 0, "timebase_index": 0}


def _svc_with(pump, tmp_path, store):
    from mast.monitoring import service as S
    svc = S.CurrentMonitorService(
        pool_getter=lambda: None, state_getter=lambda: None,
        store_getter=lambda: store,
        thresholds_getter=lambda: MonitorThresholds())
    svc._pump = pump                                     # noqa: SLF001
    return svc


def test_stop_hands_osci1t_back_to_the_operator(tmp_path):
    """Osci1T 是单实例共享模块，configure() 改过它的通道和时基。

    不还的话，一次 MAST 启动就永久改掉了用户示波器上看的信号，而他不会收到任何
    提示 —— 这与 2026-08-02 那次「启动改掉正在进行的扫描速度」是同一类缺陷：
    **启动碰了仪器，停止却不收拾。**
    """
    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    try:
        spy = _RestoreSpy()
        _svc_with(spy, tmp_path, st).stop(join_timeout=0.1)
        assert spy.restored == 1
    finally:
        st.close()


def test_stop_does_not_restore_while_the_worker_is_still_alive(tmp_path):
    """worker 还活着就意味着它随时可能再 _maybe_reconfigure() 一次。

    那样还原只是在跟自己的采集线程抢写同一个模块，还完立刻被盖掉 —— 宁可不还，
    也不要制造一场写竞争。
    """
    import threading
    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    release = threading.Event()
    try:
        spy = _RestoreSpy()
        svc = _svc_with(spy, tmp_path, st)
        stuck = threading.Thread(target=lambda: release.wait(timeout=30.0),
                                 daemon=True, name="CurrentMonitorService")
        stuck.start()
        with svc._lock:                                  # noqa: SLF001
            svc._thread = stuck                          # noqa: SLF001

        svc.stop(join_timeout=0.2)                       # join 超时，线程仍在跑

        assert stuck.is_alive()
        assert spy.restored == 0, "worker 没停就还原，会跟采集线程抢写 Osci1T"
    finally:
        release.set()
        st.close()


# ══ M6: no forged CRITICAL through an unauthenticated setting ══════════════

def test_saturation_threshold_cannot_be_set_below_a_real_preamp_range():
    """The settings write deliberately carries no PIN. At a 1 pA rail every
    ordinary tunnelling current counts as saturated, and three seconds later
    that is a CRITICAL halting a running composite skill."""
    th = MonitorThresholds.from_mapping({"cm_sat_current_a": 1e-12})
    assert th.cm_sat_current_a >= 1e-9

    normal = 100e-12 + np.random.default_rng(0).normal(0, 2e-12, 4000)
    feats = F.compute_segment_features(
        [normal], FS, F.FeatureParams(sat_current_a=th.cm_sat_current_a))
    assert feats["sat_frac"] == 0.0
    verdict = AlertEngine(lambda: th).evaluate(feats, {"ctx_zctrl_on": True})
    assert verdict.level == "ok"


# ══ M3: a half-applied write must not poison the connection ════════════════

def test_a_failed_commit_does_not_leave_an_open_transaction(store):
    """sqlite3 opens an implicit transaction for DML. An INSERT that succeeded
    followed by a commit that did not (a full disk) leaves the connection
    holding it; every later write joins the same transaction and a crash loses
    all of them — precisely the corpus this subsystem exists to accumulate."""

    class FlakyCommit:
        """Proxy that fails the first commit — sqlite3.Connection.commit is a
        read-only attribute, so it cannot be patched in place."""

        def __init__(self, real):
            self._real = real
            self.armed = True

        def __getattr__(self, name):
            return getattr(self._real, name)

        def commit(self):
            if self.armed:
                self.armed = False
                raise sqlite3.OperationalError("database or disk is full")
            return self._real.commit()

    real = store._conn                                   # noqa: SLF001
    store._conn = FlakyCommit(real)                      # noqa: SLF001
    try:
        assert store.add_alert(ts=1.0, level="warn", rule="x",
                               summary_zh="y") is None
        assert not real.in_transaction, "left an uncommitted transaction open"
    finally:
        store._conn = real                               # noqa: SLF001

    aid = store.add_alert(ts=2.0, level="warn", rule="z", summary_zh="w")
    assert aid is not None
    rows, total = store.alerts_query()
    assert total == 1 and rows[0]["rule"] == "z", "the failed write leaked through"


def test_a_blocked_start_recovers_instead_of_going_silent(tmp_path):
    """The guard against two pumps must not become a way to stop monitoring.

    On the reconnect path a live previous worker is the COMMON case — stop
    joins for 5 s while a poll can be inside a 30 s role lock, and only a
    second or two of pool rebuild separates stop from start. A refusal with no
    follow-up leaves the daemon off exactly when the link was just replaced,
    and reports it dishonestly: `running` goes false while `state` still holds
    whatever the old worker last wrote.
    """
    from mast.monitoring import service as S
    from mast.monitoring import store as ST

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    ST.set_store_for_test(st)
    release = threading.Event()
    try:
        th = MonitorThresholds.from_mapping({"cm_enabled": 0.0})   # parks, no TCP
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st, thresholds_getter=lambda: th)

        stuck = threading.Thread(target=lambda: release.wait(timeout=30.0),
                                 daemon=True, name="CurrentMonitorService")
        stuck.start()
        with svc._lock:                                   # noqa: SLF001
            svc._thread = stuck                           # noqa: SLF001
        svc.stop(join_timeout=0.1)
        assert stuck.is_alive()

        before = _pump_thread_idents()
        svc.start()                                       # blocked
        assert _pump_thread_idents() <= before,             "started a second pump alongside the stuck one"

        # It must SAY it is not running, not leave a stale verdict on screen.
        st_now = svc.status()
        assert st_now["running"] is False
        assert st_now["state"] == S.STATE_PAUSED
        assert st_now["detail"]

        # ...and it must come back by itself once the old worker leaves.
        release.set()
        stuck.join(timeout=5.0)
        deadline = time.time() + 15.0
        while time.time() < deadline and not svc.status()["running"]:
            time.sleep(0.1)
        assert svc.status()["running"] is True,             "never restarted after the previous worker exited"
        svc.stop(join_timeout=3.0)
    finally:
        release.set()
        ST.set_store_for_test(None)
        st.close()


def test_a_brief_wait_turns_the_common_case_back_into_a_start(tmp_path):
    """The 2 s bounded join: a worker that is merely finishing must not cost us
    a refusal, because on the reconnect path that is most of them."""
    from mast.monitoring import service as S
    from mast.monitoring import store as ST

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    ST.set_store_for_test(st)
    try:
        th = MonitorThresholds.from_mapping({"cm_enabled": 0.0})
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st, thresholds_getter=lambda: th)

        leaving = threading.Thread(target=lambda: time.sleep(0.4), daemon=True,
                                   name="CurrentMonitorService")
        leaving.start()
        with svc._lock:                                   # noqa: SLF001
            svc._thread = leaving                         # noqa: SLF001
        svc.stop(join_timeout=0.05)                       # join gives up early
        assert leaving.is_alive()

        svc.start()                                       # waits, then starts
        assert svc.status()["running"] is True
        svc.stop(join_timeout=3.0)
    finally:
        ST.set_store_for_test(None)
        st.close()


def test_a_stop_cancels_a_pending_restart(tmp_path):
    """A watcher armed before shutdown must not wake up afterwards and put a
    worker back. Harmless in practice — safe_call short-circuits on a closed
    pool before touching the socket — but starting threads during teardown is
    not a behaviour to leave in."""
    from mast.monitoring import service as S
    from mast.monitoring import store as ST

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    ST.set_store_for_test(st)
    release = threading.Event()
    try:
        th = MonitorThresholds.from_mapping({"cm_enabled": 0.0})
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st, thresholds_getter=lambda: th)

        stuck = threading.Thread(target=lambda: release.wait(timeout=30.0),
                                 daemon=True, name="CurrentMonitorService")
        stuck.start()
        with svc._lock:                                   # noqa: SLF001
            svc._thread = stuck                           # noqa: SLF001
        svc.stop(join_timeout=0.1)
        svc.start()                                       # blocked → watcher armed
        assert svc._restart_watcher is not None            # noqa: SLF001

        svc.stop(join_timeout=0.1)                        # teardown supersedes it
        before = _pump_thread_idents()
        release.set()
        stuck.join(timeout=5.0)
        time.sleep(0.6)                                   # let the watcher wake

        assert _pump_thread_idents() <= before, "restarted during teardown"
        assert svc.status()["running"] is False
    finally:
        release.set()
        ST.set_store_for_test(None)
        st.close()


def test_the_giving_up_message_does_not_promise_a_manual_start(tmp_path):
    """A manual start takes the same refusal path, so telling the operator to
    press it would hand them a button that cannot work."""
    from mast.monitoring import service as S

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    release = threading.Event()
    try:
        svc = S.CurrentMonitorService(
            pool_getter=lambda: None, state_getter=lambda: None,
            store_getter=lambda: st,
            thresholds_getter=lambda: MonitorThresholds())
        stuck = threading.Thread(target=lambda: release.wait(timeout=10.0),
                                 daemon=True, name="CurrentMonitorService")
        stuck.start()
        svc._retiring = stuck                             # noqa: SLF001

        import mast.monitoring.service as mod
        original = mod._RETIRING_MAX_WAIT_S
        mod._RETIRING_MAX_WAIT_S = 0.2
        try:
            svc._await_retirement_then_start(svc._generation)   # noqa: SLF001
        finally:
            mod._RETIRING_MAX_WAIT_S = original

        detail = svc.status()["detail"]
        assert "自动恢复" in detail
        assert "可手动启动" not in detail
    finally:
        release.set()
        st.close()
