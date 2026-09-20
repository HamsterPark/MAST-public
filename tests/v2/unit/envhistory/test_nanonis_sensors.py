"""Nanonis-backed environment sensors.

Two rules are asserted rather than assumed, because breaking either is invisible
in normal operation and expensive on the instrument:

  * with the comms breaker OPEN the sensor issues **zero** TCP calls — retrying
    into an open breaker is exactly what the breaker exists to prevent;
  * every call goes out on ``role="data"`` with ``count_health=False`` — the
    breaker is one instance shared by four roles, and a steadily-succeeding
    background poller on it resets everyone else's failure streak, which
    silently disables the global breaker.
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

import types

from mast.environment.nanonis_env import InstrumentStateSensor, NanonisSignalSensor


class _Snap:
    def __init__(self, current_a=2e-11, stale=False):
        self.current_a = current_a
        self.stale = stale
        self.bias_v = 0.5


class _State:
    def __init__(self, snap):
        self._snap = snap

    def snapshot(self):
        return self._snap


class _Rec:
    def __init__(self, error="", value=None):
        self.error = error
        self.method = "x"
        self.return_value = ["", b"", [value]] if value is not None else ["", b"", []]


class _Pool:
    def __init__(self, healthy=True, rec=None):
        self._healthy = healthy
        self._rec = rec or _Rec(value=1.5)
        self.calls = []

    def comms_healthy(self):
        return self._healthy

    def safe_call(self, verb, *args, **kw):
        self.calls.append((verb, args, kw))
        if verb == "Signals_NamesGet":
            return types.SimpleNamespace(
                error="", method=verb,
                return_value=["", b"", [["Current (A)", "Bias (V)", "B field (T)"]]])
        return self._rec


# ── InstrumentStateSensor: the zero-TCP mirror ──────────────────────────────

def test_state_sensor_mirrors_the_cached_current():
    s = InstrumentStateSensor(lambda: _State(_Snap(current_a=2e-11)))
    r = s.read()
    assert r.status == "ok"
    assert r.value == 2e-11
    assert r.unit == "A"
    assert s.name() == "tunnel_current"


def test_state_sensor_is_gated_and_counts_as_real():
    s = InstrumentStateSensor(lambda: _State(_Snap()))
    assert s.quiet_gated is True
    assert s.counts_as_real is True


def test_stale_snapshot_reports_unavailable_not_the_last_value():
    """A flat line continuing after TCP died is worse than a gap."""
    s = InstrumentStateSensor(lambda: _State(_Snap(current_a=2e-11, stale=True)))
    assert s.read().status == "unavailable"


def test_missing_state_reports_unavailable():
    assert InstrumentStateSensor(lambda: None).read().status == "unavailable"


def test_state_sensor_survives_a_getter_that_raises():
    def boom():
        raise RuntimeError("no runtime")
    assert InstrumentStateSensor(boom).read().status == "unavailable"


def test_missing_attribute_reports_unavailable():
    s = InstrumentStateSensor(lambda: _State(_Snap(current_a=None)))
    assert s.read().status == "unavailable"


# ── NanonisSignalSensor: the analogue-input path ────────────────────────────

def test_signal_sensor_resolves_by_name_and_reads():
    pool = _Pool()
    s = NanonisSignalSensor(lambda: pool, name="b_field",
                            signal_name="b field", unit="T")
    r = s.read()
    assert r.status == "ok" and r.value == 1.5 and r.unit == "T"
    verbs = [c[0] for c in pool.calls]
    assert verbs == ["Signals_NamesGet", "Signals_ValGet"]
    # index 2 in the stub name list
    assert pool.calls[-1][1][0] == 2


def test_signal_sensor_uses_the_data_role_and_does_not_feed_the_breaker():
    pool = _Pool()
    NanonisSignalSensor(lambda: pool, name="b_field", signal_name="b field").read()
    for verb, _args, kw in pool.calls:
        assert kw.get("role") == "data", verb
        assert kw.get("count_health") is False, verb


def test_open_breaker_means_zero_tcp():
    pool = _Pool(healthy=False)
    r = NanonisSignalSensor(lambda: pool, name="b_field", signal_name="b field").read()
    assert r.status == "unavailable"
    assert pool.calls == []          # not one packet


def test_role_busy_is_unavailable_not_an_error():
    """RoleBusy is a normal event — a scan holds the data role for a while."""
    pool = _Pool(rec=_Rec(error="RoleBusy: data held by ScanFrame"))
    s = NanonisSignalSensor(lambda: pool, name="b_field", signal_index=3)
    assert s.read().status == "unavailable"


def test_a_real_error_is_an_error():
    pool = _Pool(rec=_Rec(error="NeedModule: Signals"))
    s = NanonisSignalSensor(lambda: pool, name="b_field", signal_index=3)
    assert s.read().status == "error"


def test_index_resolution_is_cached():
    pool = _Pool()
    s = NanonisSignalSensor(lambda: pool, name="b_field", signal_name="b field")
    s.read()
    s.read()
    assert [c[0] for c in pool.calls].count("Signals_NamesGet") == 1


def test_unresolvable_name_backs_off_instead_of_re_asking_every_cycle():
    pool = _Pool()
    s = NanonisSignalSensor(lambda: pool, name="lhe", signal_name="no such signal")
    assert s.read().status == "unavailable"
    assert s.read().status == "unavailable"
    # The 128-string name reply is not cheap; asking twice a second for a signal
    # that does not exist is pure waste.
    assert [c[0] for c in pool.calls].count("Signals_NamesGet") == 1


def test_explicit_index_skips_name_resolution():
    pool = _Pool()
    s = NanonisSignalSensor(lambda: pool, name="b_field", signal_index=7)
    s.read()
    assert [c[0] for c in pool.calls] == ["Signals_ValGet"]


def test_is_healthy_does_not_issue_extra_tcp():
    pool = _Pool()
    s = NanonisSignalSensor(lambda: pool, name="b_field", signal_index=7)
    s.read()
    n = len(pool.calls)
    assert s.is_healthy() is True
    assert len(pool.calls) == n


def test_no_pool_is_unavailable():
    s = NanonisSignalSensor(lambda: None, name="b_field", signal_index=1)
    assert s.read().status == "unavailable"


def test_constructor_requires_a_way_to_find_the_signal():
    import pytest
    with pytest.raises(ValueError):
        NanonisSignalSensor(lambda: None, name="x")


# ── the has_real_sensors gate ───────────────────────────────────────────────

def test_nanonis_sensors_count_as_real_hardware():
    """A machine with only a Nanonis controller — no serial gauges — must still
    start the archive loop, or nothing at all gets recorded."""
    from mast.environment.autodetect import has_real_sensors
    from mast.environment.placeholders import HeliumLevelSensor

    assert has_real_sensors([HeliumLevelSensor()]) is False
    assert has_real_sensors([
        HeliumLevelSensor(),
        InstrumentStateSensor(lambda: _State(_Snap())),
    ]) is True


def test_reserved_names_suppress_the_matching_placeholder():
    """Otherwise the dedup renames the REAL sensor to `helium_level#2` and the
    panel shows the placeholder's permanent N/A under the expected name."""
    from mast.environment.autodetect import build_environment_sensors

    names = {s.name() for s in build_environment_sensors(cfg={"autodetect": False,
                                                              "sensors": []})}
    assert "helium_level" in names
    kept = {s.name() for s in build_environment_sensors(
        cfg={"autodetect": False, "sensors": []},
        reserved_names=["helium_level"])}
    assert "helium_level" not in kept
    assert "noise_level" in kept          # unrelated placeholders stay
