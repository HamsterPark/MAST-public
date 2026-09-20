"""Unit tests for the live experiment map (mast.io.exp_map + plan_overlay).

Covers skill→marker classification, position extraction (params / live frame /
tip, nm→m), extent computation, figure rendering (empty + populated), PNG save,
marker row round-trip, and the planning overlay (set / advance / clear).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/io/test_exp_map.py -x -v
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

from mast.core.types import HardwareState
from mast.io import exp_map as M
from mast.io.plan_overlay import PlanOverlay


def _state(**kw):
    s = HardwareState()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# ── classify_skill ──────────────────────────────────────────────────────────

def test_classify_skill_kinds():
    assert M.classify_skill("FullScan") == "scan"
    assert M.classify_skill("StartScan") == "scan"
    assert M.classify_skill("BiasSpectroscopy") == "sts"
    assert M.classify_skill("GridSTS") == "sts"
    assert M.classify_skill("TipPulse") == "pulse"
    assert M.classify_skill("ShapeTipOnSurface") == "tip_shape"
    assert M.classify_skill("ConditionTip_DQN") == "tip_shape"
    assert M.classify_skill("FolMe_XYPosSet") == "move"
    # Non-positioned / explicitly irrelevant → None
    assert M.classify_skill("PreScanCheck") is None
    assert M.classify_skill("AssessImageQuality") is None
    assert M.classify_skill("SetBias") is None
    assert M.classify_skill("") is None


# ── marker_from_skill ────────────────────────────────────────────────────────

def test_marker_scan_uses_live_frame():
    st = _state(scan_center_x_m=1e-7, scan_center_y_m=2e-7,
                scan_width_m=1e-7, scan_height_m=5e-8, scan_angle_deg=0.0)
    m = M.marker_from_skill("FullScan", {}, st)
    assert m is not None and m.kind == "scan"
    assert m.x_m == 1e-7 and m.y_m == 2e-7
    assert m.has_footprint and m.w_m == 1e-7 and m.h_m == 5e-8


def test_marker_point_prefers_params_then_tip():
    st = _state(x_pos_m=9e-9, y_pos_m=9e-9)
    # explicit params win
    m = M.marker_from_skill("TipPulse", {"x": 3e-8, "y": -1e-8}, st)
    assert m.kind == "pulse" and m.x_m == 3e-8 and m.y_m == -1e-8
    # no params → fall back to live tip
    m2 = M.marker_from_skill("TipPulse", {}, st)
    assert m2.x_m == 9e-9 and m2.y_m == 9e-9


def test_marker_nm_param_converted_to_m():
    m = M.marker_from_skill("BiasSpectroscopy", {"x_nm": 50.0, "y_nm": -20.0}, None)
    assert m is not None
    assert abs(m.x_m - 50e-9) < 1e-18 and abs(m.y_m - (-20e-9)) < 1e-18


def test_marker_none_when_no_position():
    # move skill but no params and no state → cannot place
    assert M.marker_from_skill("FolMe_XYPosSet", {}, None) is None
    # non-positioned skill → None
    assert M.marker_from_skill("SetBias", {"bias_v": 1.0}, _state(x_pos_m=1e-9, y_pos_m=1e-9)) is None


def test_failed_status_propagates():
    st = _state(scan_center_x_m=0.0, scan_center_y_m=0.0,
                scan_width_m=1e-7, scan_height_m=1e-7)
    m = M.marker_from_skill("FullScan", {}, st, status="failed")
    assert m.status == "failed"


# ── frame / tip from state ───────────────────────────────────────────────────

def test_frame_and_tip_from_state():
    st = _state(scan_center_x_m=1e-7, scan_center_y_m=0.0,
                scan_width_m=2e-7, scan_height_m=2e-7,
                x_pos_m=5e-8, y_pos_m=5e-8)
    fr = M.frame_marker_from_state(st)
    assert fr.kind == "frame" and fr.has_footprint
    assert M.tip_xy_from_state(st) == (5e-8, 5e-8)
    # missing frame → None
    assert M.frame_marker_from_state(_state()) is None
    assert M.tip_xy_from_state(_state()) is None


# ── extent ───────────────────────────────────────────────────────────────────

def test_extent_none_when_empty():
    assert M.compute_extent([]) is None


def test_extent_covers_everything():
    markers = [M.MapMarker(kind="pulse", x_m=-1e-7, y_m=-1e-7),
               M.MapMarker(kind="sts", x_m=1e-7, y_m=1e-7)]
    ext = M.compute_extent(markers)
    assert ext is not None
    x_min, x_max, y_min, y_max = ext
    assert x_min <= -1e-7 and x_max >= 1e-7
    assert y_min <= -1e-7 and y_max >= 1e-7


def test_extent_degenerate_single_point():
    ext = M.compute_extent([M.MapMarker(kind="move", x_m=1e-8, y_m=1e-8)])
    assert ext is not None
    x_min, x_max, _, _ = ext
    assert x_max > x_min  # a window, not a zero-width box


def test_extent_degenerate_uses_fixed_span_not_magnitude():
    # A lone marker far out (500 µm) must get a ~100 nm window, not zoom out to
    # its coordinate magnitude ().
    ext = M.compute_extent([M.MapMarker(kind="move", x_m=5e-4, y_m=5e-4)])
    x_min, x_max, _, _ = ext
    assert (x_max - x_min) < 1e-6   # ~100 nm, not ~500 µm


def test_extent_filters_non_finite():
    import math as _m
    # inf/nan coords must be dropped, not poison the whole extent ().
    markers = [M.MapMarker(kind="sts", x_m=float("inf"), y_m=0.0),
               M.MapMarker(kind="sts", x_m=float("nan"), y_m=0.0),
               M.MapMarker(kind="pulse", x_m=1e-8, y_m=1e-8)]
    ext = M.compute_extent(markers)
    assert ext is not None and all(_m.isfinite(v) for v in ext)
    # all-non-finite → None (nothing placeable)
    bad = [M.MapMarker(kind="sts", x_m=float("inf"), y_m=float("nan"))]
    assert M.compute_extent(bad) is None


def test_render_non_finite_no_crash():
    # The whole pipeline must survive a non-finite coordinate (offloaded render
    # would otherwise raise matplotlib ValueError in a daemon thread).
    markers = [M.MapMarker(kind="sts", x_m=float("inf"), y_m=0.0),
               M.MapMarker(kind="pulse", x_m=1e-8, y_m=1e-8)]
    fig = M.render_map_figure(markers)
    assert fig is not None


def test_coerce_float_rejects_non_finite():
    # via marker_from_skill: an inf param must not produce a placed marker.
    m = M.marker_from_skill("TipPulse", {"x": float("inf"), "y": 1e-8}, None)
    assert m is None  # x unparseable → no position


# ── render ───────────────────────────────────────────────────────────────────

def test_render_empty_returns_figure():
    fig = M.render_map_figure([])
    assert fig is not None
    assert len(fig.axes) == 1


def test_render_populated_no_crash():
    markers = [
        M.MapMarker(kind="scan", x_m=0.0, y_m=0.0, w_m=1e-7, h_m=1e-7),
        M.MapMarker(kind="sts", x_m=2e-8, y_m=2e-8),
        M.MapMarker(kind="pulse", x_m=-3e-8, y_m=1e-8),
        M.MapMarker(kind="tip_shape", x_m=1e-8, y_m=-2e-8, status="failed"),
        M.MapMarker(kind="manual", x_m=0.0, y_m=0.0, w_m=5e-8, h_m=5e-8),
        M.MapMarker(kind="move", x_m=1e-9, y_m=1e-9),
    ]
    frame = M.MapMarker(kind="frame", x_m=0.0, y_m=0.0, w_m=1.2e-7, h_m=1.2e-7)
    plan = [M.MapMarker(kind="scan", x_m=3e-7, y_m=0.0, status="planned"),
            M.MapMarker(kind="sts", x_m=4e-7, y_m=1e-7, status="planned")]
    fig = M.render_map_figure(markers, frame=frame, tip=(2e-8, 2e-8),
                              plan=plan, sample_label="Au(111)")
    assert fig is not None and len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "X (nm)"


def test_render_rotated_footprint_no_crash():
    m = M.MapMarker(kind="scan", x_m=0.0, y_m=0.0, w_m=1e-7, h_m=1e-7,
                    angle_deg=30.0)
    fig = M.render_map_figure([m])
    assert fig is not None


def test_save_map_png(tmp_path, monkeypatch):
    import mast._runtime_paths as rp
    monkeypatch.setattr(rp, "project_root", lambda: tmp_path)
    fig = M.render_map_figure([M.MapMarker(kind="sts", x_m=0.0, y_m=0.0)])
    path = M.save_map_png(fig, label="sampleA")
    assert Path(path).is_file() and path.endswith(".png")


# ── row round-trip + summary ──────────────────────────────────────────────────

def test_marker_from_row_roundtrip():
    row = {"kind": "pulse", "x_m": 1e-8, "y_m": 2e-8, "w_m": None, "h_m": None,
           "angle_deg": 0.0, "label": "电脉冲", "skill_name": "TipPulse",
           "status": "done", "source": "skill", "timestamp": "t", "meta": {"a": 1}}
    m = M.marker_from_row(row)
    assert m.kind == "pulse" and m.x_m == 1e-8 and m.meta == {"a": 1}
    assert M.markers_from_rows([row, row]) and len(M.markers_from_rows([row])) == 1


def test_summarize_markers():
    out = M.summarize_markers([
        M.MapMarker(kind="sts", x_m=0.0, y_m=0.0),
        M.MapMarker(kind="sts", x_m=0.0, y_m=0.0),
        M.MapMarker(kind="frame", x_m=0.0, y_m=0.0, w_m=1e-7, h_m=1e-7),
    ])
    assert "STS" in out and "×2" in out
    assert M.summarize_markers([]) == "尚无带位置的操作记录。"


# ── plan overlay ──────────────────────────────────────────────────────────────

def test_plan_overlay_set_and_clear():
    ov = PlanOverlay()
    assert ov.is_empty()
    ov.set_plan([M.MapMarker(kind="scan", x_m=1e-7, y_m=0.0)], title="T")
    snap = ov.snapshot()
    assert len(snap) == 1 and snap[0].status == "planned" and snap[0].source == "plan"
    assert ov.title == "T"
    ov.clear()
    assert ov.is_empty() and ov.title == ""


def test_plan_overlay_does_not_mutate_input():
    # set_plan must copy, not mutate, the caller's markers ().
    ov = PlanOverlay()
    src = M.MapMarker(kind="scan", x_m=1e-7, y_m=0.0, status="done", source="skill")
    ov.set_plan([src])
    assert src.status == "done" and src.source == "skill"   # untouched
    assert ov.snapshot()[0].status == "planned"             # copy mutated


def test_plan_overlay_skips_positionless():
    ov = PlanOverlay()
    ov.set_plan([M.MapMarker(kind="scan", x_m=None, y_m=None),
                 M.MapMarker(kind="sts", x_m=1e-8, y_m=1e-8)])
    assert len(ov.snapshot()) == 1


def test_plan_overlay_advance_consumes_nearest():
    ov = PlanOverlay()
    ov.set_plan([
        M.MapMarker(kind="scan", x_m=0.0, y_m=0.0),
        M.MapMarker(kind="sts", x_m=1e-6, y_m=0.0),
    ])
    # land right on the first step → consumed
    assert ov.advance(1e-10, 0.0) is True
    assert len(ov.snapshot()) == 1
    # far from any remaining step → not consumed
    assert ov.advance(5e-6, 5e-6, tol_m=1e-9) is False
    assert len(ov.snapshot()) == 1


def test_plan_overlay_advance_empty():
    assert PlanOverlay().advance(0.0, 0.0) is False


# ── manual-activity detection (state diff + spectrum files) ───────────────────

def _trk(**kw):
    return M.snapshot_track(_state(**kw))


def test_snapshot_track_excludes_noisy_fields():
    t = M.snapshot_track(_state(bias_v=1.0, setpoint_a=5e-11, current_a=3e-12,
                                z_pos_m=1e-9, x_pos_m=2e-8, y_pos_m=2e-8,
                                scan_center_x_m=0.0, scan_center_y_m=0.0,
                                scan_width_m=1e-7, scan_height_m=1e-7))
    assert t["bias"] == 1.0 and t["setpoint"] == 5e-11
    assert t["tip"] == (2e-8, 2e-8) and t["frame"][2] == 1e-7
    assert "current" not in t and "z" not in t  # noisy fields not tracked


def test_detect_bias_change_debounced():
    base = _trk(bias_v=0.0, x_pos_m=1e-8, y_pos_m=1e-8)
    cur = _trk(bias_v=1.5, x_pos_m=1e-8, y_pos_m=1e-8)
    # NOT settled (prev != cur) → no marker yet (debounce suppresses sweeps)
    markers, nb = M.detect_manual_state_changes(base, cur, _trk(bias_v=0.7), tip_xy=(1e-8, 1e-8))
    assert not markers and nb["bias"] == 0.0
    # settled (prev == cur) → emit once, advance baseline
    markers, nb = M.detect_manual_state_changes(base, cur, cur, tip_xy=(1e-8, 1e-8))
    assert any("偏压" in m.label for m in markers)
    assert nb["bias"] == 1.5
    assert markers[0].x_m == 1e-8 and markers[0].source == "manual"


def test_detect_setpoint_and_zstatus():
    base = _trk(setpoint_a=1e-10, z_controller_status="On")
    cur = _trk(setpoint_a=5e-11, z_controller_status="Off")
    markers, _ = M.detect_manual_state_changes(base, cur, cur, tip_xy=(0.0, 0.0))
    labels = [m.label for m in markers]
    assert any("设定点" in s for s in labels)
    assert any("Z 反馈" in s for s in labels)


def test_detect_scan_start_and_midscan():
    base = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
                scan_height_m=1e-7, scan_running=False)
    cur = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
               scan_height_m=1e-7, scan_running=True)
    markers, nb = M.detect_manual_state_changes(base, cur, base)
    assert any("手动开始扫描" in m.label and m.has_footprint for m in markers)
    assert nb["scanning"] is True
    # mid-scan (already scanning) → nothing, frame NOT synced (pre-scan kept)
    markers2, nb2 = M.detect_manual_state_changes(nb, cur, cur)
    assert markers2 == []


def test_scan_start_suppressed_for_system_scan():
    """an agent/system scan (a scan-vision monitor is live) must
    NOT be logged as 手动开始扫描. suppress_scan_start drops the marker but still
    advances the baseline so the edge isn't re-detected next tick."""
    base = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
                scan_height_m=1e-7, scan_running=False)
    cur = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
               scan_height_m=1e-7, scan_running=True)
    # System scan → suppressed, baseline still advances.
    markers, nb = M.detect_manual_state_changes(base, cur, base,
                                                suppress_scan_start=True)
    assert not any("手动开始扫描" in m.label for m in markers)
    assert nb["scanning"] is True
    # Genuine manual scan (no monitor) → the marker IS emitted (unchanged).
    markers2, _ = M.detect_manual_state_changes(base, cur, base,
                                                suppress_scan_start=False)
    assert any("手动开始扫描" in m.label for m in markers2)


def test_detect_scan_stop_catches_manual_reframe():
    # Operator reconfigures the frame during/after a scan → must be logged on the
    # scan-stop tick (the scan itself never moves the frame). (review fix)
    pre = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
               scan_height_m=1e-7, scan_running=True)  # baseline: was scanning at F0
    cur = _trk(scan_center_x_m=8e-8, scan_center_y_m=0.0, scan_width_m=1e-7,
               scan_height_m=1e-7, scan_running=False)  # stopped, frame moved to F1
    markers, nb = M.detect_manual_state_changes(pre, cur, cur)
    assert any("扫描框" in m.label for m in markers)
    assert nb["scanning"] is False and nb["frame"][0] == 8e-8


def test_detect_scan_stop_no_reframe_no_marker():
    # Scan stops with the SAME frame (normal case) → no spurious manual marker.
    pre = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
               scan_height_m=1e-7, scan_running=True)
    cur = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
               scan_height_m=1e-7, scan_running=False)
    markers, _ = M.detect_manual_state_changes(pre, cur, cur)
    assert markers == []


def test_detect_frame_then_no_double_tip():
    # a frame move must not ALSO log a tip move in the same diff
    base = _trk(scan_center_x_m=0.0, scan_center_y_m=0.0, scan_width_m=1e-7,
                scan_height_m=1e-7, x_pos_m=0.0, y_pos_m=0.0)
    cur = _trk(scan_center_x_m=5e-8, scan_center_y_m=0.0, scan_width_m=1e-7,
               scan_height_m=1e-7, x_pos_m=4e-8, y_pos_m=0.0)
    markers, _ = M.detect_manual_state_changes(base, cur, cur)
    frame_ms = [m for m in markers if "扫描框" in m.label]
    tip_ms = [m for m in markers if "移动针尖" in m.label]
    assert len(frame_ms) == 1 and len(tip_ms) == 0


def test_extract_dat_position(tmp_path):
    p = tmp_path / "spec.dat"
    p.write_text("X (m)\t3.000000E-8\nY (m)\t-1.000000E-8\n[DATA]\n"
                 "Bias (V)\tCurrent (A)\n0.0\t1e-12\n", encoding="utf-8")
    pos = M.extract_dat_position(str(p))
    assert pos is not None
    assert abs(pos[0] - 3e-8) < 1e-18 and abs(pos[1] - (-1e-8)) < 1e-18


def test_manual_marker_from_spectrum_file(tmp_path):
    p = tmp_path / "Bias-Spectroscopy005.dat"
    p.write_text("X (m)\t1.0E-8\nY (m)\t2.0E-8\n[DATA]\nBias (V)\tCurrent (A)\n"
                 "0.0\t1e-12\n", encoding="utf-8")
    m = M.manual_marker_from_spectrum_file(str(p))
    assert m is not None and m.kind == "manual" and m.source == "manual"
    assert m.x_m == 1e-8 and m.y_m == 2e-8 and "谱" in m.label
    # unknown extension → None
    assert M.manual_marker_from_spectrum_file(str(tmp_path / "x.png")) is None


def test_extract_dat_position_rejects_implausible(tmp_path):
    # a mis-parsed header value (light-years away) must be rejected, not plotted
    p = tmp_path / "bad.dat"
    p.write_text("X (m)\t1.0E+6\nY (m)\t0.0\n[DATA]\nBias (V)\tCurrent (A)\n"
                 "0.0\t1e-12\n", encoding="utf-8")
    assert M.extract_dat_position(str(p)) is None


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "nanonis"


def test_extract_3ds_bbox_real_fixture():
    # Exercises the numpy path on a REAL .3ds (would catch a missing-import
    # NameError). Returns a plausible bbox or None — never raises.
    f = _FIXTURES / "grid_3x3_bias.3ds"
    if not f.is_file():
        import pytest
        pytest.skip("no .3ds fixture")
    bbox = M.extract_3ds_bbox(str(f))
    if bbox is not None:
        cx, cy, w, h = bbox
        assert all(abs(v) < 1e-3 for v in (cx, cy)) and w > 0 and h > 0
    # full path through manual_marker_from_spectrum_file must not raise either
    m = M.manual_marker_from_spectrum_file(str(f))
    assert m is None or (m.kind == "manual" and "网格谱" in m.label)


def test_manual_marker_from_real_dat_fixture():
    f = _FIXTURES / "bias_spectroscopy_200pt.dat"
    if not f.is_file():
        import pytest
        pytest.skip("no .dat fixture")
    m = M.manual_marker_from_spectrum_file(str(f))
    # the real fixture may or may not carry X/Y header keys; just must not raise
    assert m is None or (m.kind == "manual" and m.has_xy)


# ── snapshot_track hardening (2026-07-01 scan-map rework) ────────────────────
# A non-numeric hardware value (a mock sentinel in tests / a bad TCP parse in
# production) must be coerced to None so the downstream `abs(a-b) <= tol` diff
# never raises "'<=' not supported between X and float" — that used to be raised
# and logged 'map state tick error' every single watcher tick.

def test_snapshot_track_coerces_numeric_floats():
    st = _state(bias_v=0.5, setpoint_a=1e-9, x_pos_m=1e-8, y_pos_m=2e-8,
                scan_center_x_m=1e-8, scan_center_y_m=2e-8,
                scan_width_m=5e-8, scan_height_m=5e-8, scan_angle_deg=30.0,
                scan_running=False)
    tr = M.snapshot_track(st)
    assert tr["bias"] == 0.5 and tr["setpoint"] == 1e-9
    assert tr["tip"] == (1e-8, 2e-8)
    assert tr["frame"][:4] == (1e-8, 2e-8, 5e-8, 5e-8) and tr["frame"][4] == 30.0
    assert tr["scanning"] is False


def test_snapshot_track_bad_types_coerce_to_none():
    # Genuinely non-numeric values (a bad TCP parse: string / list / opaque obj)
    # coerce to None rather than crashing the downstream diff.
    class _NoFloat:
        pass
    st = _state(bias_v="n/a", setpoint_a=_NoFloat(), x_pos_m=[1, 2], y_pos_m="x",
                scan_center_x_m=object(), scan_width_m="bad", scan_running="maybe",
                z_controller_status=object())
    tr = M.snapshot_track(st)  # must NOT raise
    assert tr["bias"] is None and tr["setpoint"] is None
    assert tr["tip"] is None and tr["frame"] is None
    assert tr["scanning"] is None and tr["zstatus"] is None


def test_detect_manual_changes_never_crashes_on_mock_state():
    """Full watcher path with MagicMock hardware on BOTH sides (mock sentinels are
    float-convertible → 1.0, so the coerced diff runs the `abs(a-b) <= tol` path
    without the raw-MagicMock TypeError that used to log 'map state tick error'
    every tick)."""
    from unittest.mock import MagicMock
    mk = lambda: _state(bias_v=MagicMock(), setpoint_a=MagicMock(),  # noqa: E731
                        x_pos_m=MagicMock(), y_pos_m=MagicMock(),
                        scan_running=MagicMock(), z_controller_status=MagicMock())
    cur = M.snapshot_track(mk())   # must NOT raise
    base = M.snapshot_track(mk())
    markers, _nb = M.detect_manual_state_changes(base, cur, base)  # exercises <=, no TypeError
    assert isinstance(markers, list)


# 只读自检不得产生表示真实动作的地图标记。
# 技能名称包含 scan 或 tipcondition 不足以证明执行了扫描或整形，
# 分类需依据明确的动作语义，避免虚构的覆盖或损伤区域。

import pytest

from mast.io.exp_map import classify_skill

# 实机全表跑出来的真实误伤，按「为什么它不该留标记」分组
_MUST_NOT_MARK = [
    # 只读自检 —— 本次事故的直接肇事者
    "ScanIntelSelfCheck", "TipConditioningSelfCheck",
    "CoarseMotionSelfCheck", "TipForgeSelfCheck",
    # getter：读一次扫描框不是"在这里扫过一张图"
    "GetScanFrame", "GetScanSpeed", "GetScanBuffer", "GetScanXYPosition",
    "GetTipShaperConfig", "GetSpectroscopyStatus", "GetSTSTiming",
    "ListScanMarkers",

    # 扫描停止动作可携带画框中心，不能仅凭 Stop 前缀排除地图记录。

    "ListSignalChannels", "list_fetch_requests",
    # snake_case = 经 tool_skills 桥接的 agent 工具，根本不碰仪器
    "plot_scan", "load_scan", "glob_scans", "mosaic_scans", "diff_scans",
    "fit_sts_peaks", "unmix_spectra", "list_scan_dir", "get_latest_scan_file",
    # 离线分析 / 存取 / 画标记
    "PredictSpectrumFromTopo", "UnmixSpectra", "AutoCrop_UnscannedRegion",
    "DiffScans_ChangeDetect", "DrawScanMarker", "EraseScanMarkers",
    "ScanBackgroundDelete",
    # 光学台（TERS）是另一台仪器，不该出现在 STM 表面地图上
    "OpticalStageScan", "PumpProbeScan", "ConfigureProbeScanner",
    "StopProbeScanner", "PulseProbeBias", "DelayLineMoveTo",
]

# field 证据说它们**是**真实记录点 —— 排除规则不许碰
_MUST_STAY_MARKED_FIELD_EVIDENCE = [
    "StopScan",        # 扫描停止时记录画框中心
    "ConfigureSTS",    # 取证 markers 44/46
]

# 真会在表面上留下事件的，必须**继续**被标记 —— 排除规则收得太紧同样是缺陷
_MUST_MARK = [
    ("ScanAt", "scan"), ("FullScan", "scan"), ("StartScan", "scan"),
    ("ExecuteScanPlan", "scan"), ("BatchRegionsScan", "scan"),
    ("SurveySurface_TileScan", "scan"), ("ScanAssessRescan", "scan"),
    ("TipShape", "tip_shape"), ("ShapeTipOnSurface", "tip_shape"),
    ("ConditionTip", "tip_shape"), ("PokeConditionTip", "tip_shape"),
    ("MakeSpectroscopyTip", "tip_shape"), ("MakeAtomicResolutionTip", "tip_shape"),
    ("TipPulse", "pulse"), ("BiasPulse", "pulse"),
    ("AcquireSTS", "sts"), ("GridSTS", "sts"), ("LineProfileSTS", "sts"),
    ("ApproachTip", "approach"), ("AutoApproach", "approach"),
    ("MoveToXY", "move"),
]


@pytest.mark.parametrize("name", _MUST_NOT_MARK)
def test_readonly_skills_never_produce_a_marker(name):
    assert classify_skill(name) is None, (
        f"{name} 会往实验地图里写一个没发生过的事件。地图是实验记录，"
        "也是修针挑干净位置、扫图算覆盖预算的依据。")


@pytest.mark.parametrize("name,kind", _MUST_MARK)
def test_real_surface_events_still_produce_their_marker(name, kind):
    assert classify_skill(name) == kind, (
        f"{name} 不再产生 {kind} 标记 —— 排除规则收得太紧，"
        "真实发生的事件从记录里消失同样是缺陷。")


def test_the_two_hazards_this_file_already_documented_still_hold():
    """回归：文件里已经写过注释的两处子串陷阱，别在改排除规则时弄坏。"""
    assert classify_skill("StopAutoApproach") is None
    assert classify_skill("GetAutoApproachStatus") is None
    assert classify_skill("MakeSpectroscopyTip") == "tip_shape"   # 不是 sts
    assert classify_skill("PreScanCheck") is None


@pytest.mark.parametrize("name", _MUST_STAY_MARKED_FIELD_EVIDENCE)
def test_field_evidenced_recording_points_are_never_excluded(name):
    """停止扫描和配置动作也可能产生有效地图标记，不应仅凭动词前缀全部排除。"""
    assert classify_skill(name) is not None, (
        f"{name} 按接口契约会写标记，不该被排除 —— 漏记比误记危险")
