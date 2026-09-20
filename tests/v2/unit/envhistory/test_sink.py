"""The sink that rides the environment monitor's 2-second heartbeat.

Its one hard requirement: **it must never take the monitoring loop down.** That
loop's first job is noticing a vacuum or temperature excursion and escalating
it; a history write failing has to be a non-event. So the failure paths get as
much coverage as the happy one.
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

from mast.envhistory.sink import QUIET_SERIES, EnvHistorySink
from mast.envhistory.store import EnvHistoryStore
from mast.envhistory.thresholds import EnvHistoryThresholds


class _Snap:
    def __init__(self, scanning=False, zctrl=True, stale=False):
        self.scan_running = scanning
        self.z_controller_on = zctrl
        self.stale = stale
        self.bias_v = 0.5
        self.setpoint_a = 2e-11
        self.z_pos_m = 1e-9


class _State:
    def __init__(self, snap=None):
        self._snap = snap or _Snap()

    def snapshot(self):
        return self._snap


class _BoomStore:
    """A store whose writes always fail — stands in for a full disk."""

    def upsert_buckets(self, rows):
        return 0

    def add_spectrum(self, **kw):
        return None

    def storage_stats(self):
        return {}


@pytest.fixture()
def store(tmp_path):
    s = EnvHistoryStore(tmp_path / "eh.sqlite")
    yield s
    s.close()


def _sink(store, *, th=None, state=None, scope=None, on_tick=None):
    return EnvHistorySink(
        store_getter=lambda: store,
        thresholds_getter=lambda: th or EnvHistoryThresholds(eh_bucket_s=60.0),
        state_getter=(lambda: state) if state is not None else None,
        scope_provider=scope,
        on_tick=on_tick,
    )


def test_readings_accumulate_and_flush_into_buckets(store):
    s = _sink(store, state=_State())
    for _ in range(5):
        s.write("temperature", 77.1, "K", "ok")
    assert s.flush() >= 1
    d = store.series("temperature")
    assert d["points"] and d["points"][0]["n"] == 5


def test_disabled_recorder_writes_nothing(store):
    s = _sink(store, th=EnvHistoryThresholds(eh_enabled=0.0), state=_State())
    for _ in range(5):
        s.write("temperature", 77.1, "K", "ok")
    assert s.flush() == 0
    assert store.series("temperature")["points"] == []


def test_quiet_gate_applies_only_to_gated_series(store):
    """A scan makes the tunnelling current topography, but the cryostat is
    still at 77 K — gating temperature too would just throw data away."""
    s = _sink(store, state=_State(_Snap(scanning=True)))
    s.set_gated_sensors(["tunnel_current"])
    s.write("tunnel_current", 5e-9, "A", "ok")
    s.write("temperature", 77.1, "K", "ok")
    s.flush()
    cur = store.series("tunnel_current")["points"][0]
    tmp = store.series("temperature")["points"][0]
    assert cur["n"] == 0 and cur["n_excluded"] == 1
    assert tmp["n"] == 1 and tmp["n_excluded"] == 0


def test_gated_series_records_while_the_instrument_is_idle(store):
    s = _sink(store, state=_State(_Snap(scanning=False)))
    s.set_gated_sensors(["tunnel_current"])
    s.write("tunnel_current", 2e-11, "A", "ok")
    s.flush()
    assert store.series("tunnel_current")["points"][0]["n"] == 1


def test_empty_gate_list_means_gate_nothing(store):
    """`None` keeps the default; `[]` must mean "no gated series" — treating an
    empty list as "unset" is the silent-no-op shape this repo has been bitten by."""
    s = _sink(store, state=_State(_Snap(scanning=True)))
    s.set_gated_sensors([])
    assert s.gated_sensors == frozenset()
    s.write("tunnel_current", 5e-9, "A", "ok")
    s.flush()
    assert store.series("tunnel_current")["points"][0]["n"] == 1


def test_none_gate_restores_the_default(store):
    s = _sink(store, state=_State())
    s.set_gated_sensors(["a", "b"])
    s.set_gated_sensors(None)
    assert "tunnel_current" in s.gated_sensors


def test_synthetic_quiet_series_is_recorded(store):
    s = _sink(store, state=_State(_Snap(scanning=True)))
    s.write("temperature", 77.1, "K", "ok")
    s.flush()
    d = store.series(QUIET_SERIES)
    assert d["points"] and d["points"][0]["mean"] == 0.0   # busy → duty 0


def test_write_never_raises_when_the_store_fails(store):
    s = _sink(_BoomStore(), state=_State())
    for _ in range(3):
        s.write("temperature", 77.1, "K", "ok")     # must not raise
    s.flush()
    assert s.stats()["failed"] >= 1
    assert s.stats()["disabled"] is True


def test_write_never_raises_when_the_scope_provider_explodes(store):
    def boom():
        raise RuntimeError("no experiment")
    s = _sink(store, state=_State(), scope=boom)
    s.write("temperature", 77.1, "K", "ok")         # must not raise
    s.flush()
    assert store.series("temperature")["points"][0]["n"] == 1


def test_flush_failure_is_counted_not_swallowed(store):
    """flush() runs at shutdown and on every experiment/sample switch. A store
    that rejects the write there must still be reported — losing a batch of
    buckets silently looks identical to "the instrument had nothing to say"."""
    s = _sink(_BoomStore(), state=_State())
    s.write("temperature", 77.1, "K", "ok")
    assert s.flush() == 0
    assert s.stats()["failed"] >= 1
    assert s.stats()["disabled"] is True


def test_self_disable_expires(store, monkeypatch):
    import mast.envhistory.sink as sink_mod
    monkeypatch.setattr(sink_mod, "_DISABLE_S", 0.0)
    s = _sink(_BoomStore(), state=_State())
    s.write("temperature", 77.1, "K", "ok")
    s.flush()
    assert s.stats()["failed"] == 1
    s.write("temperature", 77.2, "K", "ok")         # window already elapsed
    s.flush()
    assert s.stats()["failed"] == 2                 # tried again, not stuck off


def test_scope_ids_are_attached(store):
    s = _sink(store, state=_State(),
              scope=lambda: ("/exp", "S01__x", "E1", "SID1"))
    s.write("temperature", 77.1, "K", "ok")
    s.flush()
    with store._lock:                                # noqa: SLF001 — schema check
        row = store._conn.execute(
            "SELECT experiment_id, sample_id FROM env_buckets"
            " WHERE sensor='temperature'").fetchone()
    assert row["experiment_id"] == "E1"
    assert row["sample_id"] == "SID1"


def test_on_tick_fires_once_per_heartbeat_not_per_sensor(store):
    calls = []
    s = _sink(store, state=_State(), on_tick=lambda *a: calls.append(a))
    for name in ("temperature", "vacuum", "helium_level"):
        s.write(name, 1.0, "", "ok")
    assert len(calls) == 1        # one context refresh serves the whole cycle


def test_a_failing_on_tick_does_not_break_the_write(store):
    def boom(*_a):
        raise RuntimeError("sweep exploded")
    s = _sink(store, state=_State(), on_tick=boom)
    s.write("temperature", 77.1, "K", "ok")          # must not raise
    s.flush()
    assert store.series("temperature")["points"][0]["n"] == 1


def test_stats_reports_what_the_status_endpoint_needs(store):
    s = _sink(store, state=_State())
    s.write("temperature", 77.1, "K", "ok")
    st = s.stats()
    assert st["written"] == 1
    assert "temperature" in st["sensors"]
    assert st["quietness"] in ("quiet", "active", "no_tunnel", "unknown")
