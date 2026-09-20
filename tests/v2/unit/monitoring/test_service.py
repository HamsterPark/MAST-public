"""Daemon behaviour: lifecycle, the no-TCP parking states, suppression and telemetry.

The pump is stubbed with a plain generator, so these tests exercise the service
logic without any wire format. What they pin down is the set of behaviours that
protect the rest of the system: never poll a dead or circuit-broken pool, never
crash the runtime, never alert during tip shaping, and never lose the segments
that shaping produced.
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

from mast.monitoring import service as S
from mast.monitoring.pump import PumpPaused, PumpUnavailable, Segment
from mast.monitoring.store import CurrentMonitorStore, set_store_for_test
from mast.monitoring.thresholds import MonitorThresholds

FS = 20000.0


class FakeState:
    def __init__(self, **kw):
        self._snap = type("Snap", (), {
            "scan_running": kw.get("scanning", False),
            "bias_v": kw.get("bias_v", -1.0),
            "setpoint_a": kw.get("setpoint_a", 100e-12),
            "z_pos_m": kw.get("z_m", -1e-9),
            "z_controller_on": kw.get("zctrl", True),
            "stale": kw.get("stale", False),
        })()

    def snapshot(self):
        return self._snap


class FakePool:
    def __init__(self, healthy=True):
        self._healthy = healthy
        self.calls = 0

    def comms_healthy(self):
        return self._healthy

    def safe_call(self, *a, **kw):
        self.calls += 1
        raise AssertionError("service must not touch TCP in a parked state")


def _segment(t_start: float, *, amp=2e-12, mean=100e-12, dur=1.0,
             seed=0, gap=0.0) -> Segment:
    n = int(FS * dur)
    y = mean + np.random.default_rng(seed).normal(0, amp, n)
    return Segment(t_start=t_start, t_end=t_start + dur, osci_t0=t_start,
                   fs_hz=FS, runs=[y], n_samples=n, gap_s=gap,
                   channel_name="Current (A)")


class StubPump:
    """Yields scripted segments, then optionally raises.

    ``idle_ticks`` —— 每让出一段空闲就调一次 ``on_idle``，模拟真泵在等缓冲刷新时
    给辅助通道的搭车机会（见 pump.pump_segments 与 service._pump_idle）。
    默认 0 = 老行为，一次都不让。
    """

    def __init__(self, segments, raises=None, idle_ticks=0, idle_budget_s=0.04):
        self._segments = list(segments)
        self._raises = raises
        self._idle_ticks = int(idle_ticks)
        self._idle_budget_s = float(idle_budget_s)
        self.config = {"fs_hz": FS, "channel_name": "Current (A)", "n_buffer": 1024}

    def pump_segments(self, stop, on_idle=None):
        for seg in self._segments:
            if stop.is_set():
                return
            for _ in range(self._idle_ticks):
                if on_idle is not None:
                    on_idle(self._idle_budget_s)
            yield seg
        if self._raises is not None:
            raise self._raises


@pytest.fixture()
def store(tmp_path):
    s = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    set_store_for_test(s)
    yield s
    set_store_for_test(None)
    s.close()


@pytest.fixture()
def bus_events(monkeypatch):
    """Capture everything published to the event bus."""
    seen: list[dict] = []

    class FakeBus:
        def publish_current_monitor(self, **kw):
            seen.append(kw)

        def subscribe_with_id(self, cb):
            return cb

        def unsubscribe(self, h):
            pass

    monkeypatch.setattr("mast.core.events.EventBus.get", staticmethod(lambda: FakeBus()))
    return seen


def _service(store, *, pool=None, state=None, th=None, pump=None,
             skill="") -> S.CurrentMonitorService:
    thresholds = th or MonitorThresholds()
    svc = S.CurrentMonitorService(
        pool_getter=lambda: pool,
        state_getter=lambda: state,
        store_getter=lambda: store,
        thresholds_getter=lambda: thresholds,
    )
    if pump is not None:
        svc._pump = pump
        from mast.monitoring.alerts import AlertEngine
        svc._engine = AlertEngine(lambda: thresholds)
    return svc


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_start_is_idempotent_and_stop_joins(store, bus_events):
    th = MonitorThresholds.from_mapping({"cm_enabled": 0.0})
    svc = _service(store, th=th)
    svc.start()
    first = svc._thread
    svc.start()
    assert svc._thread is first                # not a second thread
    assert svc.status()["running"] is True
    svc.stop(join_timeout=3.0)
    assert svc.status()["running"] is False


def test_disabled_in_settings_parks_without_touching_tcp(store, bus_events):
    pool = FakePool()
    th = MonitorThresholds.from_mapping({"cm_enabled": 0.0})
    svc = _service(store, pool=pool, th=th)
    svc.start()
    time.sleep(0.2)
    svc.stop(join_timeout=3.0)
    assert svc.status()["state"] == S.STATE_DISABLED
    assert pool.calls == 0


def test_open_circuit_breaker_parks_with_zero_tcp(store, bus_events):
    """Retrying into an open breaker is exactly what the breaker prevents."""
    pool = FakePool(healthy=False)
    svc = _service(store, pool=pool)
    svc.start()
    time.sleep(0.2)
    svc.stop(join_timeout=3.0)
    assert svc.status()["state"] == S.STATE_COMMS_DOWN
    assert pool.calls == 0


def test_no_pool_parks(store, bus_events):
    svc = _service(store, pool=None)
    svc.start()
    time.sleep(0.2)
    svc.stop(join_timeout=3.0)
    assert svc.status()["state"] == S.STATE_NO_POOL


def test_missing_scope_module_goes_quiet_and_reports_a_retry(store, bus_events):
    """The bundled simulator does not load Osci1T — 'unavailable' is a normal
    state, not an error, and must not spam."""
    svc = _service(store, pool=FakePool(),
                   pump=StubPump([], raises=PumpUnavailable("NeedModule")))
    svc._pump_until_interrupted(MonitorThresholds())
    st = svc.status()
    assert st["state"] == S.STATE_UNAVAILABLE
    assert st["retry_in_s"] >= 30
    assert "Osci1T" in st["detail"]


def test_paused_pump_reports_and_recovers(store, bus_events):
    svc = _service(store, pool=FakePool(), pump=StubPump([], raises=PumpPaused("busy")))
    svc._pump_until_interrupted(MonitorThresholds())
    assert svc.status()["state"] == S.STATE_PAUSED


def test_loop_survives_an_internal_error(store, bus_events, monkeypatch):
    svc = _service(store, pool=FakePool())
    monkeypatch.setattr(svc, "_pump_until_interrupted",
                        lambda th: (_ for _ in ()).throw(RuntimeError("boom")))
    svc.start()
    time.sleep(0.2)
    running = svc.status()["running"]
    svc.stop(join_timeout=3.0)
    assert running is True                     # thread did not die on the error


def test_start_service_never_raises(monkeypatch):
    monkeypatch.setattr(S.CurrentMonitorService, "start",
                        lambda self: (_ for _ in ()).throw(RuntimeError("nope")))
    S.set_service_for_test(None)
    assert S.start_service(object()) is None   # boot continues
    S.set_service_for_test(None)


def test_stop_service_without_a_service_is_a_noop():
    S.set_service_for_test(None)
    S.stop_service()                           # no raise


# ── segment handling ─────────────────────────────────────────────────────────

def test_segment_is_stored_with_features_and_context(store, bus_events):
    svc = _service(store, pool=FakePool(), state=FakeState(scanning=True, bias_v=-1.5),
                   pump=StubPump([]))
    svc._on_segment(_segment(time.time()))

    row = store.latest_feature()
    assert row is not None
    assert row["rms_detrended_a"] == pytest.approx(2e-12, rel=0.2)
    assert row["ctx_scanning"] == 1
    assert row["ctx_bias_v"] == pytest.approx(-1.5)
    assert row["alert_level"] == "ok"
    meta = store.segment_meta(row["segment_id"])
    assert meta["has_file"] is True and meta["channel_name"] == "Current (A)"


def test_raw_samples_are_written_as_float32(store, bus_events):
    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_segment(_segment(time.time()))
    meta = store.segment_meta(store.latest_feature()["segment_id"])
    arr = np.load(meta["npy_path"])
    assert arr.dtype == np.float32
    assert arr.size == int(FS)


def test_a_full_disk_loses_samples_but_not_the_record(store, bus_events, monkeypatch):
    """Losing a waveform is an inconvenience; losing the row that says the
    segment existed, and its features, is the thing worth protecting."""
    monkeypatch.setattr("numpy.save",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_segment(_segment(time.time()))

    row = store.latest_feature()
    assert row is not None
    assert row["rms_detrended_a"] is not None
    assert store.segment_meta(row["segment_id"])["has_file"] is False


def test_telemetry_is_one_scalar_event_per_segment(store, bus_events):
    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_segment(_segment(time.time()))

    seg_events = [e for e in bus_events if e.get("kind") == "segment"]
    assert len(seg_events) == 1
    ev = seg_events[0]
    for key in ("seg_id", "ts", "fs_hz", "rms_pa", "mean_na", "level"):
        assert key in ev
    assert all(not isinstance(v, (list, dict, np.ndarray)) for v in ev.values()), \
        "arrays must never ride the event bus"


def test_status_events_only_fire_on_transition(store, bus_events):
    svc = _service(store, pool=FakePool())
    svc._set_state(S.STATE_RUNNING, "")
    svc._set_state(S.STATE_RUNNING, "")
    svc._set_state(S.STATE_PAUSED, "x")
    kinds = [e for e in bus_events if e.get("kind") == "status"]
    assert len(kinds) == 2       # not one per poll — the replay buffer is only 100


# ── suppression ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("skill", ["TipShape", "conditiontip_01", "AutoApproach",
                                   "BiasPulse", "WithdrawTip", "MotorMove"])
def test_tip_work_suppresses_alerts_and_pins_the_window(store, bus_events,
                                                        monkeypatch, skill):
    """During shaping the current is SUPPOSED to look violent. Record and pin
    it — it is the best training data there is — but do not cry wolf."""
    monkeypatch.setattr("mast.core.instrument_lock.instrument_lock",
                        lambda: type("L", (), {"snapshot": lambda self: {"skill": skill}})())
    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    # a segment that would otherwise be a critical candidate
    seg = _segment(time.time())
    seg.runs[0][:] = 1e-12                    # frozen readout (in range)
    svc._on_segment(seg)

    row = store.latest_feature()
    assert row["alert_level"] == "suppressed"   # distinguishable from a real 'ok'
    assert row["ctx_skill"] == skill
    assert store.segment_meta(row["segment_id"])["pinned"] == 1
    assert not [e for e in bus_events if e.get("kind") == "alert"]


def test_unrelated_skill_does_not_suppress(store, bus_events, monkeypatch):
    monkeypatch.setattr("mast.core.instrument_lock.instrument_lock",
                        lambda: type("L", (), {"snapshot": lambda self: {"skill": "StartScan"}})())
    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_segment(_segment(time.time()))
    assert store.latest_feature()["alert_level"] == "ok"


# ── suppression: scanning (v6.1.2) ───────────────────────────────────────────


def _jumpy_segment(t_start: float) -> Segment:
    """A tip crossing terraces: quiet noise with abrupt level changes at the
    edges — a random WALK of levels, not a square wave.

    The distinction matters and cost a rewrite. A ±50 pA square wave does raise
    jump_rate, but it is by construction a two-level signal, so it also scores
    rtn_score 0.98 and rms 50 pA — and those two rules are deliberately NOT
    suppressed, so the segment came back `warn` and the test proved nothing
    about jump_burst. These constants isolate the one feature under test:
    jump 10.0 Hz (thr 5), rms 12.9 pA (thr 20), RTN 0.00 (thr 0.7),
    spike 2.5 σ (thr 8), line 0.41 (thr 10).
    """
    seg = _segment(t_start)
    y = seg.runs[0]
    rng = np.random.default_rng(4)
    edges = np.sort(rng.choice(np.arange(50, y.size - 50), size=40,
                               replace=False))
    level, prev = 0.0, 0
    for i in edges:
        y[prev:i] += level
        level = float(np.clip(level + 14e-12 * rng.choice([-1.0, 1.0]),
                              -42e-12, 42e-12))
        prev = i
    y[prev:] += level
    return seg


def test_a_jumpy_scanning_segment_is_recorded_but_raises_no_alert(store,
                                                                  bus_events):
    """The field failure, end to end: one 34-minute frame produced ≈50 alerts
    because the tip crossing topography reads as a current fault."""
    svc = _service(store, pool=FakePool(), state=FakeState(scanning=True),
                   pump=StubPump([]))
    svc._on_segment(_jumpy_segment(time.time()))

    row = store.latest_feature()
    assert row["ctx_scanning"] == 1
    assert row["alert_level"] == "suppressed"
    assert not [e for e in bus_events if e.get("kind") == "alert"]


def test_the_suppressed_rule_names_are_written_down(store, bus_events):
    """'照记不报': a stored `suppressed` row that cannot say suppressed for WHAT
    forces the next person calibrating this rig to guess."""
    import json

    svc = _service(store, pool=FakePool(), state=FakeState(scanning=True),
                   pump=StubPump([]))
    svc._on_segment(_jumpy_segment(time.time()))

    extra = json.loads(store.latest_feature()["extra_json"] or "{}")
    assert extra.get("suppressed_by") == "scanning"
    assert "jump_burst" in extra.get("suppressed_rules", [])


def test_the_same_segment_at_rest_does_alert(store, bus_events):
    """The control. Without it 'no alert while scanning' could equally mean the
    features never fired at all."""
    svc = _service(store, pool=FakePool(), state=FakeState(scanning=False),
                   pump=StubPump([]))
    svc._on_segment(_jumpy_segment(time.time()))

    assert store.latest_feature()["alert_level"] == "warn"
    rules = {e.get("rule") for e in bus_events if e.get("kind") == "alert"}
    assert "jump_burst" in rules


def test_the_segment_event_carries_the_suppression_reason(store, bus_events):
    """`suppressed` used to mean exactly one thing (a tip-shaping skill), and
    the UI banner hard-codes that sentence. It now has a second cause, so the
    payload has to say which one — as a STRING, not a list (scalars only)."""
    svc = _service(store, pool=FakePool(), state=FakeState(scanning=True),
                   pump=StubPump([]))
    svc._on_segment(_jumpy_segment(time.time()))

    ev = next(e for e in bus_events if e.get("kind") == "segment")
    assert ev["level"] == "suppressed"
    assert ev["suppressed_by"] == "scanning"
    assert isinstance(ev["suppressed_rules"], str)
    assert "jump_burst" in ev["suppressed_rules"]


def test_a_scanning_segment_does_not_suppress_a_critical(store, bus_events,
                                                         monkeypatch):
    """The three CRITICAL rules describe the instrument; a scan does not make
    a railed preamp acceptable."""
    monkeypatch.setattr("mast.buffer.active.get_active_buffer", lambda: None)
    th = MonitorThresholds.from_mapping({"cm_crit_consecutive": 1.0})
    svc = _service(store, pool=FakePool(), state=FakeState(scanning=True),
                   th=th, pump=StubPump([]))
    seg = _segment(time.time())
    seg.runs[0][:] = 1e-12                     # frozen readout
    svc._on_segment(seg)

    rows, total = store.alerts_query(level="critical")
    assert total == 1 and rows[0]["rule"] == "freeze"


def test_suppression_matcher_is_case_insensitive_substring():
    assert S.CurrentMonitorService._is_suppressed({"ctx_skill": "TIPSHAPE_v2"}) is True
    assert S.CurrentMonitorService._is_suppressed({"ctx_skill": "StartScan"}) is False
    assert S.CurrentMonitorService._is_suppressed({"ctx_skill": ""}) is False


# ── alerting ─────────────────────────────────────────────────────────────────

def test_confirmed_critical_alerts_pins_and_publishes(store, bus_events, monkeypatch):
    monkeypatch.setattr("mast.buffer.active.get_active_buffer", lambda: None)
    th = MonitorThresholds.from_mapping({"cm_crit_consecutive": 2.0,
                                         "cm_evidence_min_interval_s": 10.0})
    svc = _service(store, pool=FakePool(), th=th, pump=StubPump([]))

    now = time.time()
    for i in range(2):
        seg = _segment(now + i)
        seg.runs[0][:] = 1e-12                 # frozen (in range, so not saturation)
        svc._on_segment(seg)

    rows, total = store.alerts_query(level="critical")
    assert total == 1 and rows[0]["rule"] == "freeze"
    assert store.latest_feature()["alert_level"] == "critical"
    assert store.segment_meta(store.latest_feature()["segment_id"])["pinned"] == 1
    assert [e for e in bus_events if e.get("kind") == "alert"
            and e.get("level") == "critical"]

    # The evidence image is attached asynchronously — rendering it inline was
    # measured at up to ~1 s, i.e. twenty oscilloscope buffers lost at exactly
    # the moment something is going wrong. The alert lands first; the picture
    # follows.
    aid = rows[0]["id"]
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if store.alerts_query(level="critical")[0][0]["evidence_available"]:
            break
        time.sleep(0.1)
    assert store.alerts_query(level="critical")[0][0]["evidence_available"] == 1
    assert store.alert_evidence_png(aid)


def test_a_single_bad_segment_does_not_escalate(store, bus_events, monkeypatch):
    monkeypatch.setattr("mast.buffer.active.get_active_buffer", lambda: None)
    th = MonitorThresholds.from_mapping({"cm_crit_consecutive": 3.0})
    svc = _service(store, pool=FakePool(), th=th, pump=StubPump([]))
    seg = _segment(time.time())
    seg.runs[0][:] = 1e-12
    svc._on_segment(seg)
    assert store.alerts_query(level="critical")[1] == 0
    assert store.latest_feature()["alert_level"] == "warn"


def test_alerts_can_be_switched_off_while_still_recording(store, bus_events):
    th = MonitorThresholds.from_mapping({"cm_alerts_enabled": 0.0,
                                         "cm_crit_consecutive": 1.0})
    svc = _service(store, pool=FakePool(), th=th, pump=StubPump([]))
    seg = _segment(time.time())
    seg.runs[0][:] = 1e-12
    svc._on_segment(seg)
    assert store.alerts_query()[1] == 0
    assert store.latest_feature() is not None      # still recorded


def test_warn_alert_is_recorded_without_evidence(store, bus_events):
    th = MonitorThresholds.from_mapping({"cm_rms_warn_a": 1e-13})
    svc = _service(store, pool=FakePool(), th=th, pump=StubPump([]))
    svc._on_segment(_segment(time.time()))
    rows, total = store.alerts_query(level="warn")
    assert total >= 1
    assert rows[0]["evidence_available"] == 0      # cheap: WARNs do not render


# ── skill-step tail pin ──────────────────────────────────────────────────────

def test_skill_step_event_pins_the_trailing_window(store, bus_events):
    """SKILL_STEP fires when a skill FINISHES, so by then the interesting
    segments are already written — this makes sure they survive the sweep."""
    from mast.core.events import Event, EventType

    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_segment(_segment(time.time()))
    sid = store.latest_feature()["segment_id"]
    assert store.segment_meta(sid)["pinned"] == 0

    svc._on_bus_event(1, Event(type=EventType.SKILL_STEP,
                               data={"skill": "TipShape", "step": 1, "success": True}))
    assert store.segment_meta(sid)["pinned"] == 1


def test_skill_step_for_an_unrelated_skill_pins_nothing(store, bus_events):
    from mast.core.events import Event, EventType

    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_segment(_segment(time.time()))
    sid = store.latest_feature()["segment_id"]
    svc._on_bus_event(1, Event(type=EventType.SKILL_STEP,
                               data={"skill": "GetBias", "step": 1, "success": True}))
    assert store.segment_meta(sid)["pinned"] == 0


def test_bus_callback_never_raises(store, bus_events):
    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_bus_event(1, object())             # not an Event at all
    svc._on_bus_event(2, None)


# ── status ───────────────────────────────────────────────────────────────────

def test_status_is_json_safe_and_zero_tcp(store, bus_events):
    import json

    pool = FakePool()
    svc = _service(store, pool=pool, pump=StubPump([]))
    st = svc.status()
    json.dumps(st)                              # must be serialisable as-is
    assert pool.calls == 0
    assert st["strategy"] == "osci1t"
    assert st["fs_hz"] == FS
    assert st["connected"] is True


def test_status_takes_the_strategy_from_the_pump_not_a_literal(store, bus_events):
    """``strategy`` 必须由**泵实例**派生。

    在 2026-08-09 之前这一行是写死的 ``"osci1t" if cfg else None`` —— 字段看起来
    是动态的，而任何时候都只会是那一个值。于是「我们到底切到 2T 了吗」这个问题
    在 API 上根本问不出来，而它恰恰是这次切换唯一能远程验收的东西。
    """
    class TwoTracePump(StubPump):
        STRATEGY = "osci2t"

    svc = _service(store, pool=FakePool(), pump=TwoTracePump([]))
    assert svc.status()["strategy"] == "osci2t"

    svc2 = _service(store, pool=FakePool(), pump=StubPump([]))
    assert svc2.status()["strategy"] == "osci1t"     # 没声明 → 退回 1T


def test_status_strategy_is_none_before_the_pump_is_configured(store, bus_events):
    """没配起来就没有事实。空 config = None，不是「osci1t」。"""
    class Unconfigured(StubPump):
        STRATEGY = "osci2t"

    pump = Unconfigured([])
    pump.config = {}
    svc = _service(store, pool=FakePool(), pump=pump)
    assert svc.status()["strategy"] is None


def test_a_downgraded_strategy_says_so_in_the_detail(store, bus_events):
    """要 2T、模块没开、退回 1T —— 采集照常，所以**只有这句话**能说明
    「为什么往返还是那么多」。一次静默降级和一次故障一样难查。"""
    pump = StubPump([_segment(time.time())])
    pump.fallback_note = "(要的是 osci2t,退回 osci1t:Osci2T 模块未加载)"
    svc = _service(store, pool=FakePool(), pump=pump)
    svc._pump_until_interrupted(MonitorThresholds())
    st = svc.status()
    assert st["state"] == S.STATE_RUNNING
    assert "osci2t" in st["detail"] and "osci1t" in st["detail"]


def test_status_counts_segments_and_gaps(store, bus_events):
    svc = _service(store, pool=FakePool(), pump=StubPump([]))
    svc._on_segment(_segment(time.time(), gap=0.25))
    svc._on_segment(_segment(time.time() + 1, gap=0.5))
    st = svc.status()
    assert st["segments_done"] == 2
    assert st["gaps_total_s"] == pytest.approx(0.75)
    assert st["last_segment_ts"] is not None


# ── reconnect / shutdown ordering ────────────────────────────────────────────

class ClosablePool:
    """A pool that fails loudly if anyone calls it after close_all().

    That is the whole hazard this ordering exists to prevent: a daemon still
    polling a pool being torn down spins on errors and races the rebuild for the
    same fragile Nanonis port.
    """

    def __init__(self, tag: str):
        self.tag = tag
        self.closed = False
        self.calls = 0
        self.used_after_close = 0

    def comms_healthy(self):
        return not self.closed

    def safe_call(self, method, *a, role="main", **k):
        self.calls += 1
        if self.closed:
            self.used_after_close += 1
        from mast.core.types import NanonisCallRecord
        return NanonisCallRecord(method=method, error="NeedModule: no scope here")

    def close_all(self):
        self.closed = True


def test_reconnect_ordering_stops_before_the_pool_closes(store, bus_events):
    """Mirrors CoreRuntime.reconnect(): stop the daemon, close the old pool,
    swap in the new one, start again."""
    holder = {"pool": ClosablePool("A")}

    class FakeRuntime:
        _state = None

        @property
        def _pool(self):
            return holder["pool"]

    S.set_service_for_test(None)
    app = FakeRuntime()
    svc = S.start_service(app)
    assert svc is not None
    time.sleep(0.6)
    old = holder["pool"]
    assert old.calls > 0, "the daemon never reached the pool at all"

    S.stop_service()
    old.close_all()
    holder["pool"] = ClosablePool("B")
    S.start_service(app)
    time.sleep(0.8)

    try:
        live = [t for t in threading.enumerate()
                if t.name == "CurrentMonitorService" and t.is_alive()]
        assert len(live) == 1, f"{len(live)} monitor daemons running at once"
        assert old.used_after_close == 0, "polled a pool that was already closed"
        assert holder["pool"].calls > 0, "did not pick up the new pool"
    finally:
        S.stop_service()
        S.set_service_for_test(None)

    time.sleep(0.3)
    assert not [t for t in threading.enumerate()
                if t.name == "CurrentMonitorService" and t.is_alive()]


# ── 辅助通道搭车采样（→ #30，2026-08-06） ────────────────────

class _AuxSpy:
    """假采样器。记下每次被问到时的 seg_id 与 budget。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def maybe_sample(self, ctx=None, segment_id=None, suppressed=False,
                     now=None, budget_s=None):
        self.calls.append((segment_id, budget_s, bool(suppressed)))
        return None                      # 不落库，这一组只关心「什么时候被问」

    def snapshot(self):
        return None


def test_aux_is_sampled_in_the_pumps_idle_not_only_at_segment_boundaries(store):
    """采样机会必须多于「每段一次」—— 否则 5 Hz 无从谈起。

    机会封顶在段边界上时，一秒只有 0.75 次（1.024 s 采集 + 0.31 s 特征提取），
    调小 cm_aux_interval_s 一点用都没有：那是节流上限，不是时钟。
    """
    spy = _AuxSpy()
    pump = StubPump([_segment(time.time())], raises=PumpPaused("done"),
                    idle_ticks=6)
    svc = _service(store, pump=pump)
    svc._aux = spy
    svc._pump_until_interrupted(MonitorThresholds())
    # 6 次搭车 + 1 次段边界
    assert len(spy.calls) >= 7, spy.calls
    budgets = [b for _sid, b, _s in spy.calls if b is not None]
    assert len(budgets) == 6, "搭车那几次必须带 budget"


def test_the_segment_boundary_sample_still_gets_no_budget(store):
    """段边界那条路径本来就有 0.31 s 空闲，不该被搭车的几十毫秒预算限制住。"""
    spy = _AuxSpy()
    pump = StubPump([_segment(time.time())], raises=PumpPaused("done"),
                    idle_ticks=2)
    svc = _service(store, pump=pump)
    svc._aux = spy
    svc._pump_until_interrupted(MonitorThresholds())
    assert spy.calls[-1][1] is None, spy.calls[-1]


def test_an_idle_sample_is_tagged_with_the_last_stored_segment(store):
    """搭车时下一段还没成形、更没有 id，所以贴的是**上一段**。

    方向反了，用途没变 —— aux 行仍然贴着它旁边的那段电流。这条钉的是
    「贴的是已落库的那个 id」，而不是 0 或 None 这种读起来像「没有段落」的值。
    """
    spy = _AuxSpy()
    pump = StubPump([_segment(time.time()), _segment(time.time() + 1.0)],
                    raises=PumpPaused("done"), idle_ticks=2)
    svc = _service(store, pump=pump)
    svc._aux = spy
    svc._pump_until_interrupted(MonitorThresholds())
    idle = [(sid, b) for sid, b, _s in spy.calls if b is not None]
    # 第一段之前还没有任何段落落库 → None；第二段之前贴的是第一段的 id
    assert idle[0][0] is None
    assert idle[-1][0] == 1, idle


def test_aux_disabled_means_the_idle_slot_costs_nothing(store):
    """关掉辅助通道之后，让出来的空闲不该产生任何工作。"""
    spy = _AuxSpy()
    th = MonitorThresholds.from_mapping({"cm_aux_enabled": 0.0})
    pump = StubPump([_segment(time.time())], raises=PumpPaused("done"),
                    idle_ticks=5)
    svc = _service(store, th=th, pump=pump)
    svc._aux = spy
    svc._pump_until_interrupted(th)
    assert [b for _sid, b, _s in spy.calls if b is not None] == []


def test_a_throwing_aux_sampler_never_stops_the_pump(store):
    """辅助通道绝不反噬电流采集——搭车路径上也一样。"""
    class Boom(_AuxSpy):
        def maybe_sample(self, *a, **kw):
            raise RuntimeError("aux exploded")

    pump = StubPump([_segment(time.time())], raises=PumpPaused("done"),
                    idle_ticks=3)
    svc = _service(store, pump=pump)
    svc._aux = Boom()
    svc._pump_until_interrupted(MonitorThresholds())      # 不抛就是通过
    assert svc._segments_done == 1
