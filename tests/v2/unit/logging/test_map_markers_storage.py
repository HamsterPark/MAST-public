"""map_markers: spatial record of every positioned operation, persisted into the
experiment record (此地图就是实验记录的地图).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/logging/test_map_markers_storage.py -x -v
"""
from __future__ import annotations

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

from mast.logging.storage import ExperimentStorage


def _store(tmp_path):
    return ExperimentStorage(str(tmp_path / "exp.db"))


def test_log_and_get_marker_roundtrip(tmp_path):
    st = _store(tmp_path)
    exp = st.create_experiment("Run A")
    rid = st.log_marker(kind="scan", x_m=1e-7, y_m=2e-7, w_m=1e-7, h_m=1e-7,
                        angle_deg=0.0, label="扫图", skill_name="FullScan",
                        experiment_id=exp, meta={"params": {"bias_v": 1.0}})
    assert isinstance(rid, int)
    rows = st.get_markers(exp)
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "scan" and r["x_m"] == 1e-7 and r["w_m"] == 1e-7
    assert r["skill_name"] == "FullScan"
    assert r["meta"]["params"]["bias_v"] == 1.0
    assert r["source"] == "skill" and r["status"] == "done"


def test_marker_scoping_by_sample(tmp_path):
    st = _store(tmp_path)
    exp = st.create_experiment("Run A")
    s1 = st.create_sample(exp, "sampleA")
    s2 = st.create_sample(exp, "sampleB")
    st.log_marker(kind="sts", x_m=0.0, y_m=0.0, experiment_id=exp, sample_id=s1)
    st.log_marker(kind="pulse", x_m=1e-8, y_m=0.0, experiment_id=exp, sample_id=s2)
    # whole experiment → both
    assert len(st.get_markers(exp)) == 2
    # 换样品 → fresh canvas: only that sample's markers
    only_a = st.get_markers(exp, sample_id=s1)
    assert len(only_a) == 1 and only_a[0]["kind"] == "sts"
    only_b = st.get_markers(exp, sample_id=s2)
    assert len(only_b) == 1 and only_b[0]["kind"] == "pulse"


def test_markers_chronological(tmp_path):
    st = _store(tmp_path)
    exp = st.create_experiment("Run A")
    for k in ("scan", "sts", "pulse", "move"):
        st.log_marker(kind=k, x_m=0.0, y_m=0.0, experiment_id=exp)
    rows = st.get_markers(exp)
    assert [r["kind"] for r in rows] == ["scan", "sts", "pulse", "move"]


def test_manual_marker_source(tmp_path):
    st = _store(tmp_path)
    exp = st.create_experiment("Run A")
    st.log_marker(kind="manual", x_m=3e-8, y_m=3e-8, label="手动移动针尖",
                  source="manual", experiment_id=exp)
    rows = st.get_markers(exp)
    assert rows[0]["source"] == "manual" and rows[0]["kind"] == "manual"


def test_null_xy_marker_persists(tmp_path):
    # A marker with unknown xy still persists (NULLs) and round-trips.
    st = _store(tmp_path)
    exp = st.create_experiment("Run A")
    st.log_marker(kind="move", x_m=None, y_m=None, experiment_id=exp)
    rows = st.get_markers(exp)
    assert rows and rows[0]["x_m"] is None


def test_scope_change_clears_plan_overlay(tmp_path):
    # 换样品/换实验 → the planning route overlay is cleared so the
    # previous scope's ghost route doesn't leak onto the new sample's map. This
    # covers BOTH the GUI button and the agent start_sample tool (both go through
    # ExperimentLog.start_*).
    from mast.io.exp_map import MapMarker
    from mast.io.plan_overlay import get_plan_overlay
    from mast.logging.experiment_log import ExperimentLog

    el = ExperimentLog(_store(tmp_path))
    el.start_experiment("Run A")
    get_plan_overlay().set_plan([MapMarker(kind="scan", x_m=1e-7, y_m=0.0)])
    assert not get_plan_overlay().is_empty()
    el.start_sample("sampleA")               # 换样品 → fresh canvas
    assert get_plan_overlay().is_empty()
    # also cleared on a brand-new experiment
    get_plan_overlay().set_plan([MapMarker(kind="sts", x_m=0.0, y_m=0.0)])
    el.start_experiment("Run B")
    assert get_plan_overlay().is_empty()
