"""Unit tests for spatial-mosaic assembly (mast.io.mosaic).

Covers header→footprint parsing, channel picking, canvas assembly (placement,
extent, resolution, max-canvas clamp), the read→build→render→save chain (with a
fake read_sxm), and graceful handling of empty/bad input.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/io/test_mosaic.py -x -v
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

from datetime import datetime

import numpy as np
import pytest

from mast.io import mosaic as mz


def _scan(cx, cy, w, h, val, n=8):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle": 0.0,
            "data": np.full((n, n), float(val)), "nx": n, "ny": n,
            "channel": "Z", "path": f"scan_{val}.sxm"}


# ── header parsing ───────────────────────────────────────────────────────────

def test_parse_xy_meta():
    meta = mz.parse_xy_meta({
        "scan_offset": "1.0e-7 2.0e-7",
        "scan_range": "5.0e-8 5.0e-8",
        "scan_angle": "0.0",
    })
    assert meta["cx"] == pytest.approx(1e-7)
    assert meta["cy"] == pytest.approx(2e-7)
    assert meta["w"] == pytest.approx(5e-8)
    assert meta["angle"] == 0.0


def test_parse_xy_meta_missing_returns_none():
    assert mz.parse_xy_meta({"scan_range": "5e-8 5e-8"}) is None
    assert mz.parse_xy_meta({}) is None
    # zero size rejected
    assert mz.parse_xy_meta({"scan_offset": "0 0", "scan_range": "0 0"}) is None


def test_pick_channel_exact_then_substring_then_first():
    chans = {"Z": {"forward": np.ones((2, 2))},
             "Current": {"forward": np.zeros((2, 2))}}
    assert mz._pick_channel(chans, "Z", "forward")[0] == "Z"
    assert mz._pick_channel(chans, "curr", "forward")[0] == "Current"
    # unknown → first
    assert mz._pick_channel(chans, "nope", "forward")[0] in chans


# ── canvas assembly ──────────────────────────────────────────────────────────

def test_build_mosaic_two_scans_side_by_side():
    s1 = _scan(0.0, 0.0, 1e-7, 1e-7, 1.0)
    s2 = _scan(2e-7, 0.0, 1e-7, 1e-7, 2.0)   # to the right, non-overlapping
    m = mz.build_mosaic([s1, s2])
    assert m["placed"] == 2
    assert m["image"] is not None
    ext = m["extent_m"]
    assert ext[0] == pytest.approx(-5e-8)    # left edge of s1
    assert ext[1] == pytest.approx(2.5e-7)   # right edge of s2
    w, h = m["canvas_px"]
    assert w > h                              # wider than tall
    assert m["image"].shape == (h, w, 3)
    assert m["image"].dtype == np.uint8


def test_build_mosaic_empty_returns_error():
    m = mz.build_mosaic([])
    assert m["image"] is None
    assert m["error"]


def test_build_mosaic_clamps_canvas():
    s1 = _scan(0.0, 0.0, 1e-7, 1e-7, 1.0)
    s2 = _scan(1e-4, 0.0, 1e-7, 1e-7, 2.0)   # 100 µm away → huge span
    m = mz.build_mosaic([s1, s2], max_canvas_px=256)
    w, h = m["canvas_px"]
    assert w <= 256 and h <= 256
    assert m["placed"] == 2


def test_build_mosaic_angle_warning():
    s = _scan(0.0, 0.0, 1e-7, 1e-7, 1.0)
    s["angle"] = 30.0
    m = mz.build_mosaic([s])
    assert m["angle_warning"] is True


# ── read → build → render → save ─────────────────────────────────────────────

_FAKE_SXM = {
    "header": {
        "scan_offset": "1.0e-7 2.0e-7",
        "scan_range": "5.0e-8 5.0e-8",
        "scan_angle": "0.0",
        "scan_dir": "down",
        "scan_pixels": [4, 4],
    },
    "channels": {
        "Z": {"forward": np.arange(16, dtype=float).reshape(4, 4),
              "backward": np.zeros((4, 4))},
        "Current": {"forward": np.ones((4, 4))},
    },
}


def test_load_scan_for_mosaic(monkeypatch):
    import mast.io.nanonis_files as nf
    monkeypatch.setattr(nf, "read_sxm", lambda p: _FAKE_SXM)
    s = mz.load_scan_for_mosaic("anything.sxm", channel="Z")
    assert s is not None
    assert s["cx"] == pytest.approx(1e-7)
    assert s["w"] == pytest.approx(5e-8)
    assert s["nx"] == 4 and s["ny"] == 4
    assert s["channel"] == "Z"
    assert s["data"].shape == (4, 4)


def test_mosaic_from_paths_skips_unreadable(monkeypatch):
    import mast.io.nanonis_files as nf

    def fake(p):
        if "bad" in str(p):
            raise ValueError("corrupt")
        return _FAKE_SXM
    monkeypatch.setattr(nf, "read_sxm", fake)
    m = mz.mosaic_from_paths(["good1.sxm", "bad.sxm", "good2.sxm"], channel="Z")
    assert m["placed"] >= 1
    assert m["n_input"] == 2          # only the two readable ones loaded
    assert len(m["scans_meta"]) == 2


def test_render_mosaic_figure_returns_figure():
    from matplotlib.figure import Figure
    m = mz.build_mosaic([_scan(0, 0, 1e-7, 1e-7, 1.0)])
    assert isinstance(mz.render_mosaic_figure(m), Figure)
    # error mosaic still renders without raising
    assert isinstance(mz.render_mosaic_figure({"image": None, "error": "x"}), Figure)


def test_save_mosaic_writes_png_npy_json(tmp_path, monkeypatch):
    import mast._runtime_paths as rp
    monkeypatch.setattr(rp, "project_root", lambda: tmp_path)
    m = mz.build_mosaic([_scan(0, 0, 1e-7, 1e-7, 1.0)])
    paths = mz.save_mosaic(m, label="sampleA", now=datetime(2026, 6, 19, 9, 0, 0))
    assert Path(paths["png"]).exists()
    assert Path(paths["npy"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["png"]).parent == tmp_path / "artifacts" / "mosaics"
    assert "sampleA" in Path(paths["png"]).name


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ── 角度读不到 ⇒ 仍要提示（v6.1.3，弱实例）──────────────────────────────
#
# `angle_warning` 判的是 `abs(angle) > 1.0`。角度解析失败兜底成 0.0 时，
# `abs(0.0) > 1.0` 为假 —— **一帧真的转过的图会被当成轴对齐悄悄摆上画布，
# 而提示不触发**。兜底值又一次落在「没什么可担心的」那一侧。
#
# 修法不是把 angle 改成 None（六个调用方都当它是数），而是把「这是猜的」
# 传下去，让判提示的那一侧自己决定。


def test_unparseable_angle_is_flagged_unknown():
    meta = mz.parse_xy_meta({"scan_offset": "0 0", "scan_range": "1e-7 1e-7",
                          "scan_angle": "not a number"})
    assert meta is not None
    assert meta["angle"] == 0.0          # 摆放仍用轴对齐这个合理近似
    assert meta["angle_known"] is False  # 但不许假装读到了


def test_a_missing_angle_key_is_also_unknown():
    """header 里根本没有 SCAN_ANGLE —— 是「不知道」，不是「等于 0」。"""
    meta = mz.parse_xy_meta({"scan_offset": "0 0", "scan_range": "1e-7 1e-7"})
    assert meta["angle_known"] is False


def test_a_real_angle_is_known():
    meta = mz.parse_xy_meta({"scan_offset": "0 0", "scan_range": "1e-7 1e-7",
                          "scan_angle": "30.0"})
    assert meta["angle"] == 30.0 and meta["angle_known"] is True


def test_unknown_angle_raises_the_placement_warning():
    """对照组在下一条：角度已知且为 0 时**不该**提示。"""
    scans = [{"data": np.ones((4, 4)), "nx": 4, "ny": 4, "cx": 0.0, "cy": 0.0,
              "w": 1e-7, "h": 1e-7, "angle": 0.0, "angle_known": False,
              "channel": "Z", "path": "a.sxm"}]
    out = mz.build_mosaic(scans)
    assert out["angle_warning"] is True


def test_a_known_zero_angle_does_not_warn():
    scans = [{"data": np.ones((4, 4)), "nx": 4, "ny": 4, "cx": 0.0, "cy": 0.0,
              "w": 1e-7, "h": 1e-7, "angle": 0.0, "angle_known": True,
              "channel": "Z", "path": "a.sxm"}]
    out = mz.build_mosaic(scans)
    assert out["angle_warning"] is False
