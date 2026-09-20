"""Threshold holder: clamping, tolerance of persisted junk, and the knob catalog."""
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

from mast.monitoring.thresholds import (
    BOOL_KEYS, EDITABLE_KEYS, FIELD_BOUNDS, MonitorThresholds,
    get_monitor_thresholds, knob_catalog, set_monitor_thresholds,
)


@pytest.fixture(autouse=True)
def _reset():
    yield
    set_monitor_thresholds(None)


def test_defaults_are_monitoring_on_alerts_on():
    t = MonitorThresholds()
    assert t.enabled is True
    assert t.alerts_enabled is True
    assert t.crit_consecutive == 3


def test_partial_mapping_keeps_other_defaults():
    t = MonitorThresholds.from_mapping({"cm_segment_s": 2.0})
    assert t.cm_segment_s == 2.0
    assert t.cm_keep_hours == MonitorThresholds().cm_keep_hours


def test_out_of_range_values_are_clamped_not_rejected():
    t = MonitorThresholds.from_mapping({"cm_segment_s": 999.0, "cm_keep_gb": 0.0})
    assert t.cm_segment_s == FIELD_BOUNDS["cm_segment_s"][1]
    assert t.cm_keep_gb == FIELD_BOUNDS["cm_keep_gb"][0]


def test_unknown_and_non_numeric_keys_are_ignored():
    t = MonitorThresholds.from_mapping(
        {"nonsense": 1.0, "cm_segment_s": "two", "cm_keep_hours": 12.0})
    assert t.cm_keep_hours == 12.0
    assert t.cm_segment_s == MonitorThresholds().cm_segment_s


def test_json_booleans_work_for_the_switch_knobs():
    """The UI sends JSON true/false for the 0/1 knobs."""
    t = MonitorThresholds.from_mapping({"cm_enabled": False, "cm_alerts_enabled": True})
    assert t.enabled is False
    assert t.alerts_enabled is True


def test_set_and_get_round_trip_then_reset():
    set_monitor_thresholds({"cm_rms_warn_a": 5e-12})
    assert get_monitor_thresholds().cm_rms_warn_a == pytest.approx(5e-12)
    set_monitor_thresholds(None)
    assert get_monitor_thresholds().cm_rms_warn_a == MonitorThresholds().cm_rms_warn_a


def test_mapping_round_trips_through_to_mapping():
    original = MonitorThresholds.from_mapping({"cm_keep_gb": 8.0, "cm_enabled": 0.0})
    assert MonitorThresholds.from_mapping(original.to_mapping()) == original


def test_knob_catalog_covers_every_editable_key_with_bounds():
    cat = {k["key"]: k for k in knob_catalog()}
    assert set(cat) == set(EDITABLE_KEYS)
    for key, knob in cat.items():
        lo, hi = FIELD_BOUNDS[key]
        assert knob["min"] == lo and knob["max"] == hi
        assert lo <= knob["default"] <= hi
        assert knob["label_zh"] and knob["label_zh"] != key   # a real Chinese label
        assert knob["is_bool"] == (key in BOOL_KEYS)


def test_knob_catalog_reflects_the_live_value():
    set_monitor_thresholds({"cm_keep_hours": 6.0})
    cat = {k["key"]: k for k in knob_catalog()}
    assert cat["cm_keep_hours"]["value"] == 6.0
    assert cat["cm_keep_hours"]["default"] == MonitorThresholds().cm_keep_hours


def test_every_editable_key_exists_on_the_dataclass():
    """A key in the UI list with no field behind it would silently do nothing."""
    fields = set(MonitorThresholds().to_mapping())
    assert set(EDITABLE_KEYS) <= fields
