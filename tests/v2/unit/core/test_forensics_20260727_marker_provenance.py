"""Map markers distinguish requested positions, scan centres and tip readbacks.

An independent rectangular frame and an off-centre synthetic tip position provide
distinct coordinates for each provenance path. Repeated measurements may share
a position while remaining separate events; stale readbacks carry their timestamp."""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.types import HardwareState  # noqa: E402
from mast.logging.storage import ExperimentStorage  # noqa: E402

# Independently chosen coordinates: tip offset (+40, -30) nm inside the frame.
FRAME_XY = (400e-9, -200e-9)
TIP_XY = (440e-9, -230e-9)


def _rt(storage, eid, state):
    el = SimpleNamespace(current_experiment_id=eid, current_sample_id=None)
    return SimpleNamespace(_storage=storage, _experiment_log=el,
                           _state=SimpleNamespace(snapshot=lambda: state))


def _synthetic_state(**kw):
    """A synthetic 120 by 80 nm frame containing an off-centre tip."""
    return HardwareState(
        scan_center_x_m=FRAME_XY[0], scan_center_y_m=FRAME_XY[1],
        scan_width_m=120e-9, scan_height_m=80e-9,
        x_pos_m=TIP_XY[0], y_pos_m=TIP_XY[1], **kw)


def _marker(storage, eid, payload, state):
    CoreRuntime._record_map_marker(_rt(storage, eid, state), payload)
    return storage.get_markers(experiment_id=eid)


def test_positionless_sts_is_stamped_as_a_readback(tmp_path):
    """AcquireSTS carried no position — its marker is where the tip HAPPENED to
    be, and the record now says so."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    ms = _marker(st, eid, {"skill": "AcquireSTS", "success": True,
                           "params": {"save_basename": ""}}, _synthetic_state())
    assert len(ms) == 1
    assert ms[0]["x_m"] == pytest.approx(TIP_XY[0])
    assert ms[0]["y_m"] == pytest.approx(TIP_XY[1])
    assert ms[0]["meta"]["pos_src"] == "tip_readback"


def test_a_scan_marker_is_stamped_as_the_frame_not_the_tip(tmp_path):
    """A scan marker uses the frame centre, even when the tip is elsewhere."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    ms = _marker(st, eid, {"skill": "StopScan", "success": True, "params": {}},
                 _synthetic_state())
    assert ms[0]["meta"]["pos_src"] == "scan_frame"
    assert ms[0]["x_m"] == pytest.approx(FRAME_XY[0])
    # …and it is NOT where the tip was
    assert abs(ms[0]["x_m"] - TIP_XY[0]) == pytest.approx(40e-9)


def test_an_explicitly_aimed_skill_is_stamped_as_commanded(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    ms = _marker(st, eid, {"skill": "BiasSpectr", "success": True,
                           "params": {"x_m": 1.0e-6, "y_m": 1.2e-6}},
                 _synthetic_state())
    assert ms[0]["meta"]["pos_src"] == "param"
    assert ms[0]["x_m"] == pytest.approx(1.0e-6)


def test_a_stale_snapshot_marks_the_position_as_stale(tmp_path):
    """`HardwareState.stale` means every value was carried forward from an older
    read. A marker placed from one is a position that may be minutes old."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    ms = _marker(st, eid, {"skill": "AcquireSTS", "success": True, "params": {}},
                 _synthetic_state(stale=True, timestamp="2000-01-01T00:00:00"))
    assert ms[0]["meta"]["pos_src"] == "tip_readback"
    assert ms[0]["meta"]["pos_stale"] is True
    assert ms[0]["meta"]["pos_as_of"] == "2000-01-01T00:00:00"


def test_a_live_snapshot_is_not_marked_stale(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    ms = _marker(st, eid, {"skill": "AcquireSTS", "success": True, "params": {}},
                 _synthetic_state())
    assert "pos_stale" not in ms[0]["meta"]


def test_per_region_markers_are_stamped_as_region_records(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    ms = _marker(st, eid, {
        "skill": "BatchRegionsScan", "success": True, "params": {},
        "data": {"regions": [{"index": 1, "label": "P1", "success": True,
                              "center_x_m": 1e-6, "center_y_m": 1.4e-6,
                              "width_m": 1e-7, "height_m": 1e-7}]}},
        _synthetic_state())
    assert ms[0]["meta"]["pos_src"] == "region_record"


def test_repeated_spectra_without_motion_share_the_position(tmp_path):
    """Repeated acquisition without movement records separate events at the same synthetic position."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    state = _synthetic_state()
    for skill in ("ConfigureSTS", "AcquireSTS", "ConfigureSTS", "AcquireSTS"):
        CoreRuntime._record_map_marker(_rt(st, eid, state), {
            "skill": skill, "success": True, "params": {}})
    ms = st.get_markers(experiment_id=eid)
    assert len(ms) == 4
    assert len({(m["x_m"], m["y_m"]) for m in ms}) == 1
    assert all(m["meta"]["pos_src"] == "tip_readback" for m in ms)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
