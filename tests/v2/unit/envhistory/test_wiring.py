"""The assembled chain: sensors → recorder → EnvironmentMonitor → buckets.

Every piece has its own unit tests; this file checks that they are actually
CONNECTED. That is the failure this project keeps meeting — a subsystem that is
implemented, tested and simply never wired, which looks from the outside exactly
like a feature that does not work.

``EnvironmentMonitor.read_all()`` deliberately does NOT feed the sinks — that
happens in ``_loop`` — so :func:`_cycle` mirrors one iteration of the loop body.
:func:`test_the_background_loop_actually_feeds_the_sinks` starts the real thread
and pins that equivalence, so the mirror cannot silently drift away from the
loop it stands in for. Everything else uses the mirror, because a test that
races a timer is a test that fails on a loaded CI box.
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

import pytest

from mast.core.types import SensorReading
from mast.environment.base import EnvironmentSensor
from mast.environment.monitor import EnvironmentMonitor
from mast.environment.nanonis_env import InstrumentStateSensor
from mast.envhistory.recorder import EnvHistoryRecorder
from mast.envhistory.sink import QUIET_SERIES
from mast.envhistory.store import EnvHistoryStore
from mast.envhistory.thresholds import EnvHistoryThresholds


class _Thermo(EnvironmentSensor):
    def __init__(self, value=77.1, status="ok"):
        self.value, self.status = value, status

    def name(self):
        return "temperature"

    def read(self):
        return SensorReading(value=self.value, unit="K", status=self.status)


class _Snap:
    def __init__(self, scanning=False, skill=""):
        self.current_a = 2.0e-11
        self.stale = False
        self.scan_running = scanning
        self.z_controller_on = True
        self.bias_v = 0.5
        self.setpoint_a = 2e-11
        self.z_pos_m = 1e-9
        self._skill = skill


class _State:
    def __init__(self, snap):
        self._snap = snap

    def snapshot(self):
        return self._snap


def _cycle(monitor, sink) -> None:
    """One iteration of ``EnvironmentMonitor._loop``'s body: read every sensor,
    then hand each reading to the sink. Pinned by the thread test below."""
    for name, reading in monitor.read_all().items():
        sink.write(name, reading.value, reading.unit, reading.status)


@pytest.fixture()
def rig(tmp_path):
    """Assemble the chain the way core.runtime does."""
    store = EnvHistoryStore(tmp_path / "eh.sqlite")
    snap = _Snap()
    state = _State(snap)
    th = EnvHistoryThresholds(eh_bucket_s=60.0)

    sensors = [_Thermo(), InstrumentStateSensor(lambda: state)]
    rec = EnvHistoryRecorder(store_getter=lambda: store,
                             thresholds_getter=lambda: th,
                             state_getter=lambda: state)
    rec.sink.set_gated_sensors([
        s.name() for s in sensors if getattr(s, "quiet_gated", False)
    ])
    monitor = EnvironmentMonitor(sensors=sensors, storage=None, interval_s=2.0,
                                 sinks=[rec.sink])
    yield monitor, rec, store, snap
    rec.stop()
    store.close()


def test_readings_flow_all_the_way_into_buckets(rig):
    monitor, rec, store, _snap = rig
    for _ in range(3):
        _cycle(monitor, rec.sink)
    assert rec.sink.flush() >= 1

    temp = store.series("temperature")
    assert temp["unit"] == "K"
    assert temp["points"][0]["n"] == 3
    assert abs(temp["points"][0]["mean"] - 77.1) < 1e-9

    cur = store.series("tunnel_current")
    assert cur["unit"] == "A"
    assert cur["points"][0]["n"] == 3           # idle → the quiet gate admits


def test_the_gated_series_is_the_mirrored_current(rig):
    _monitor, rec, _store, _snap = rig
    assert rec.sink.gated_sensors == frozenset({"tunnel_current"})


def test_a_scan_stops_the_current_series_but_not_the_thermometer(rig):
    monitor, rec, store, snap = rig
    _cycle(monitor, rec.sink)           # idle
    snap.scan_running = True
    for _ in range(2):
        # The sink caches the context for a second, so force a re-read; the real
        # loop gets this for free from its 2 s period.
        rec.sink._ctx_at = 0.0          # noqa: SLF001 — clock seam
        _cycle(monitor, rec.sink)       # during a scan the current is topography
    rec.sink.flush()

    cur = store.series("tunnel_current")["points"][0]
    temp = store.series("temperature")["points"][0]
    assert cur["n"] == 1 and cur["n_excluded"] == 2
    assert temp["n"] == 3 and temp["n_excluded"] == 0


def test_the_quiet_duty_cycle_explains_the_hole(rig):
    monitor, rec, store, snap = rig
    _cycle(monitor, rec.sink)
    snap.scan_running = True
    rec.sink._ctx_at = 0.0              # noqa: SLF001 — clock seam
    _cycle(monitor, rec.sink)
    rec.sink.flush()
    q = store.series(QUIET_SERIES)["points"][0]
    # 0 < duty < 1: the instrument was idle for part of the bucket.
    assert 0.0 < q["mean"] < 1.0


def test_a_sensor_that_raises_does_not_poison_the_series(rig):
    """read_all substitutes value=0.0/status=error; a 0 must not reach the trend."""
    monitor, rec, store, _snap = rig

    class _Broken(_Thermo):
        def read(self):
            raise RuntimeError("serial port gone")

    _cycle(monitor, rec.sink)
    monitor.replace_sensors([_Broken(), *[s for s in monitor._sensors.values()  # noqa: SLF001
                                          if s.name() != "temperature"]])
    _cycle(monitor, rec.sink)
    rec.sink.flush()
    p = store.series("temperature")["points"][0]
    assert p["n"] == 1 and p["n_excluded"] == 1
    assert p["min"] == 77.1                       # NOT 0.0
    assert p["worst_status"] == "error"


def test_disabling_the_recorder_stops_recording_immediately(tmp_path):
    """No restart: the sink reads the live holder on every write."""
    store = EnvHistoryStore(tmp_path / "eh.sqlite")
    state = _State(_Snap())
    th = {"v": EnvHistoryThresholds(eh_bucket_s=60.0)}
    rec = EnvHistoryRecorder(store_getter=lambda: store,
                             thresholds_getter=lambda: th["v"],
                             state_getter=lambda: state)
    monitor = EnvironmentMonitor(sensors=[_Thermo()], storage=None,
                                 sinks=[rec.sink])
    _cycle(monitor, rec.sink)
    th["v"] = EnvHistoryThresholds(eh_enabled=0.0)
    _cycle(monitor, rec.sink)
    rec.sink.flush()
    assert store.series("temperature")["points"][0]["n"] == 1
    rec.stop()
    store.close()


def test_the_csv_sink_and_the_history_sink_coexist(tmp_path):
    """runtime injects both; neither may break the other."""
    from mast.environment.csv_sink import EnvironmentCsvSink

    store = EnvHistoryStore(tmp_path / "eh.sqlite")
    exp = tmp_path / "exp"
    csv = EnvironmentCsvSink(lambda: (exp, "S01__x", "E1", "SID1"))
    rec = EnvHistoryRecorder(store_getter=lambda: store,
                             thresholds_getter=lambda: EnvHistoryThresholds(),
                             state_getter=lambda: _State(_Snap()))
    monitor = EnvironmentMonitor(sensors=[_Thermo()], storage=None,
                                 sinks=[csv, rec.sink])
    for sink in (csv, rec.sink):
        _cycle(monitor, sink)
    rec.sink.flush()
    csv.close()

    assert store.series("temperature")["points"][0]["n"] == 1
    written = list((exp / "env").glob("temperature_*.csv"))
    assert written and "77.1" in written[0].read_text(encoding="utf-8")
    rec.stop()
    store.close()


def test_a_broken_history_sink_cannot_stop_the_csv_sink(tmp_path):
    """The monitor's first job is alarms; history is strictly secondary."""
    from mast.environment.csv_sink import EnvironmentCsvSink

    class _Boom:
        def write(self, *a, **kw):
            raise RuntimeError("history exploded")

    exp = tmp_path / "exp"
    csv = EnvironmentCsvSink(lambda: (exp, None, "E1", None))
    monitor = EnvironmentMonitor(sensors=[_Thermo()], storage=None,
                                 sinks=[_Boom(), csv])
    _run_loop_until(monitor, lambda: csv.written >= 1)
    csv.close()
    assert list((exp / "env").glob("temperature_*.csv"))


def test_the_background_loop_actually_feeds_the_sinks(rig):
    """Pins the equivalence :func:`_cycle` assumes.

    ``read_all()`` does not touch the sinks — only ``_loop`` does. If that ever
    changes (or the sinks stop being called at all), every mirror-based test
    above would keep passing while nothing was recorded on the real machine.
    """
    monitor, rec, store, _snap = rig
    monitor._interval = 0.02                     # noqa: SLF001 — speed, not logic
    _run_loop_until(monitor, lambda: rec.sink.stats()["written"] >= 3)
    assert rec.sink.flush() >= 1
    assert store.series("temperature")["points"]


def _run_loop_until(monitor, done, timeout=10.0):
    """Start the real background loop until *done*, then stop it."""
    import time as _t
    monitor._interval = min(getattr(monitor, "_interval", 2.0), 0.02)  # noqa: SLF001
    monitor.start()
    try:
        deadline = _t.monotonic() + timeout
        while _t.monotonic() < deadline and not done():
            _t.sleep(0.02)
        assert done(), "the background loop never fed the sinks"
    finally:
        monitor.stop()


# ── the runtime's own assembly methods ──────────────────────────────────────
#
# These two are what core.runtime calls at setup, and nothing else covers them.
# They only ever touch `self` through getattr, so a minimal stand-in exercises
# the real code — building a whole CoreRuntime here would drag in a connection
# pool, a checkpointer and an atexit handler for no extra assurance.

class _FakeRuntime:
    def __init__(self, state=None, pool=None, storage=None):
        self._state = state
        self._pool = pool
        self._storage = storage

    def _env_csv_scope(self):
        return None


def _runtime_sensors(fake):
    from mast.core.runtime import CoreRuntime
    return CoreRuntime._build_nanonis_env_sensors(fake)          # noqa: SLF001


def test_runtime_builds_the_mirrored_current_sensor():
    from mast.envhistory.thresholds import set_env_history_thresholds

    set_env_history_thresholds(None)                    # defaults: recording on
    sensors = _runtime_sensors(_FakeRuntime(state=_State(_Snap())))
    assert [s.name() for s in sensors] == ["tunnel_current"]
    assert sensors[0].quiet_gated is True
    assert sensors[0].counts_as_real is True
    r = sensors[0].read()
    assert r.status == "ok" and r.unit == "A"


def test_runtime_builds_nothing_while_recording_is_off():
    """The sensor EXISTING is what adds a 2 s row to environment_log, and
    turning recording off also turns off the retention sweep that justifies it."""
    from mast.envhistory.thresholds import EnvHistoryThresholds, set_env_history_thresholds

    set_env_history_thresholds(EnvHistoryThresholds(eh_enabled=0.0))
    try:
        assert _runtime_sensors(_FakeRuntime(state=_State(_Snap()))) == []
    finally:
        set_env_history_thresholds(None)


def test_runtime_builds_nothing_before_instrument_state_exists():
    """Early boot / no Nanonis — an empty list, not a crash."""
    from mast.envhistory.thresholds import set_env_history_thresholds

    set_env_history_thresholds(None)
    assert _runtime_sensors(_FakeRuntime(state=None)) == []


def test_runtime_derives_the_gate_list_from_the_sensor_attribute(tmp_path):
    """Not a hard-coded name table — renaming a sensor must not silently
    disable its gate."""
    from mast.core.runtime import CoreRuntime
    from mast.envhistory.recorder import get_recorder, set_recorder
    from mast.envhistory.store import set_store_for_test
    from mast.envhistory.thresholds import set_env_history_thresholds

    set_env_history_thresholds(None)
    store = EnvHistoryStore(tmp_path / "eh.sqlite")
    set_store_for_test(store)
    fake = _FakeRuntime(state=_State(_Snap()))
    sensors = [_Thermo(), *_runtime_sensors(fake)]
    rec = CoreRuntime._build_env_history_recorder(fake, sensors)   # noqa: SLF001
    try:
        assert rec is not None
        assert rec.sink.gated_sensors == frozenset({"tunnel_current"})
        assert get_recorder() is rec                    # registered as the singleton
        assert {"enabled", "sink", "spectra", "sweep", "store"} <= set(rec.status())
    finally:
        rec.stop()
        set_recorder(None)
        set_store_for_test(None)
        store.close()
