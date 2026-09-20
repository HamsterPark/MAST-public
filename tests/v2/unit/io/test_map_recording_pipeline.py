"""End-to-end scan-map pipeline verification with SIMULATED experiment data.

For every positioned operation type — 扫图 / 修针尖 / 做谱 / 电脉冲 / 移动 (skill
markers) and 手动 (manual state-diff markers) — drive the REAL recording path
(``CoreRuntime._record_map_marker`` / ``detect_manual_state_changes`` →
``ExperimentStorage.log_marker``), then RETRIEVE (``get_markers``) and RENDER
(``render_map_figure``), asserting each op is recorded with the right kind /
coords AND actually drawn. No hardware: ``_state=None`` so marker placement falls
back to the skills' explicit params.

This is the pipeline behind the operator's ask "验证所有记录是否都能正确记录和显示".

Run: .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/io/test_map_recording_pipeline.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.io import exp_map as M  # noqa: E402
from mast.logging.storage import ExperimentStorage  # noqa: E402

EXP, SAMP = "exp-sim-1", "samp-sim-1"

# (skill_name, params, expected marker kind). Covers every positioned op type.
SIM_OPS = [
    ("StartScan",        {"x_m": 1e-8, "y_m": 2e-8, "width_m": 5e-8, "height_m": 5e-8}, "scan"),
    ("FullScan",         {"x_m": -6e-8, "y_m": -4e-8, "width_m": 3e-8, "height_m": 3e-8}, "scan"),
    ("ConditionTip",     {"x_m": 3e-8, "y_m": 1e-8}, "tip_shape"),
    ("ShapeTipOnSurface", {"x_m": -2e-8, "y_m": 4e-8}, "tip_shape"),
    ("BiasSpectroscopy", {"x_m": 2e-8, "y_m": 3e-8}, "sts"),
    ("GridSTS",          {"x_m": -1e-8, "y_m": 1.5e-8}, "sts"),
    ("TipPulse",         {"x_m": 4e-8, "y_m": 2e-8}, "pulse"),
    ("MoveTip",          {"x_m": 6e-8, "y_m": -3e-8}, "move"),
]


@pytest.fixture()
def rt(tmp_path):
    r = CoreRuntime.__new__(CoreRuntime)  # skip setup(); wire only what recording needs
    r._storage = ExperimentStorage(str(tmp_path / "exp.db"))
    r._experiment_log = SimpleNamespace(current_experiment_id=EXP, current_sample_id=SAMP)
    r._state = None  # no live snapshot → marker_from_skill uses explicit params
    return r


# ── classification: every simulated op maps to the intended kind ─────────────

def test_classification_of_every_sim_op():
    for skill, _params, kind in SIM_OPS:
        assert M.classify_skill(skill) == kind, f"{skill} → {M.classify_skill(skill)} (want {kind})"


# ── record → retrieve: every op persists a marker with the right kind + xy ───

def test_all_ops_record_and_retrieve(rt):
    for skill, params, _kind in SIM_OPS:
        rt._record_map_marker({"skill": skill, "params": params, "success": True})

    rows = rt._storage.get_markers(EXP, SAMP)
    markers = M.markers_from_rows(rows)
    kinds = [m.kind for m in markers]

    # every positioned kind is present
    for want in ("scan", "tip_shape", "sts", "pulse", "move"):
        assert want in kinds, f"kind {want!r} not recorded — got {kinds}"

    # count matches the number of simulated ops (all placed)
    assert len(markers) == len(SIM_OPS), f"recorded {len(markers)} of {len(SIM_OPS)}"

    # coords round-trip correctly, and scans carry a footprint
    by_skill = {r["skill_name"]: r for r in rows}
    scan = by_skill["StartScan"]
    assert abs(scan["x_m"] - 1e-8) < 1e-15 and abs(scan["y_m"] - 2e-8) < 1e-15
    assert scan["w_m"] and scan["h_m"] and scan["kind"] == "scan"
    sts = by_skill["BiasSpectroscopy"]
    assert abs(sts["x_m"] - 2e-8) < 1e-15 and sts["kind"] == "sts"


def test_failed_op_records_failed_status(rt):
    rt._record_map_marker({"skill": "StartScan",
                           "params": {"x_m": 0.0, "y_m": 0.0, "width_m": 4e-8, "height_m": 4e-8},
                           "success": False})
    rows = rt._storage.get_markers(EXP, SAMP)
    assert rows and rows[-1]["status"] == "failed"


def test_non_positioned_skill_records_nothing(rt):
    # A skill that isn't a positioned operation (e.g. GetBias) must NOT leave a marker.
    rt._record_map_marker({"skill": "GetBias", "params": {}, "success": True})
    assert rt._storage.get_markers(EXP, SAMP) == []


# ── manual activity (state-diff daemon path) also records ────────────────────

def test_manual_state_changes_record_and_retrieve(rt):
    # Simulate a manual scan-frame reconfigure + a manual bias change (settled),
    # exactly as _map_state_tick would feed detect_manual_state_changes.
    base = {"bias": 0.1, "setpoint": 1e-9, "zstatus": "On", "scanning": False,
            "frame": (0.0, 0.0, 5e-8, 5e-8, 0.0), "tip": (0.0, 0.0)}
    cur = {"bias": 0.5, "setpoint": 1e-9, "zstatus": "On", "scanning": False,
           "frame": (2e-8, 2e-8, 4e-8, 4e-8, 0.0), "tip": (2e-8, 2e-8)}
    markers, _nb = M.detect_manual_state_changes(base, cur, cur, tip_xy=(2e-8, 2e-8))
    assert markers, "no manual markers detected"
    for m in markers:
        rt._storage.log_marker(
            kind=m.kind, x_m=m.x_m, y_m=m.y_m, w_m=m.w_m, h_m=m.h_m,
            angle_deg=m.angle_deg, label=m.label, skill_name="", status="done",
            source="manual", experiment_id=EXP, sample_id=SAMP, meta=m.meta or {})
    rows = rt._storage.get_markers(EXP, SAMP)
    assert any(r["source"] == "manual" for r in rows)
    assert any("手动" in (r["label"] or "") for r in rows)


# ── render: the retrieved markers actually DRAW (display verification) ───────

def test_full_map_renders_all_kinds(rt):
    for skill, params, _kind in SIM_OPS:
        rt._record_map_marker({"skill": skill, "params": params, "success": True})
    markers = M.markers_from_rows(rt._storage.get_markers(EXP, SAMP))

    # a live current frame + tip, like get_scan_map overlays from the state
    frame = M.MapMarker(kind="frame", x_m=1e-8, y_m=1e-8, w_m=5e-8, h_m=5e-8)
    fig = M.render_map_figure(markers, frame=frame, tip=(1e-8, 1e-8), sample_label="Si(111)")
    assert fig is not None
    ax = fig.axes[0]
    # scan footprints → Rectangle patches (2 scans + the live frame = ≥3)
    from matplotlib.patches import Rectangle
    n_rect = sum(1 for p in ax.patches if isinstance(p, Rectangle))
    assert n_rect >= 3, f"expected ≥3 footprint rects, got {n_rect}"
    # point ops (sts/pulse/tip_shape/move) → scatter collections drawn
    assert len(ax.collections) >= 3, f"expected point-op scatters, got {len(ax.collections)}"
    # title reflects the op count (non-frame markers)
    assert "个操作" in ax.get_title()


def test_rotated_scan_footprint_renders(rt):
    # A rotated scan must still render (regression for the footprint rotation path).
    rt._state = None
    rt._record_map_marker({"skill": "StartScan",
                           "params": {"x_m": 1e-8, "y_m": 1e-8, "width_m": 5e-8,
                                      "height_m": 3e-8, "angle_deg": 30.0},
                           "success": True})
    markers = M.markers_from_rows(rt._storage.get_markers(EXP, SAMP))
    fig = M.render_map_figure(markers)
    assert fig is not None and fig.axes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
