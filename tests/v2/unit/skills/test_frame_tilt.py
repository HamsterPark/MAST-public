"""AnalyzeFrameTilt 只读入口：计算表面起伏并支持目标仪器独立验证阈值。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mast.skills.builtins.frame_tilt import AnalyzeFrameTilt

NOISE_M = 15.4e-12
STEP_M = 240e-12


def _make_sxm(path: Path, frame: np.ndarray, *, offset=(0.0, 0.0),
              size_m=(1e-7, 1e-7), angle_deg=0.0) -> Path:
    ny, nx = frame.shape
    header = (
        ":SCAN_PIXELS:\n" f"{nx} {ny}\n"
        ":SCAN_OFFSET:\n" f"{offset[0]:.6E} {offset[1]:.6E}\n"
        ":SCAN_RANGE:\n" f"{size_m[0]:.6E} {size_m[1]:.6E}\n"
        ":SCAN_ANGLE:\n" f"{angle_deg:.6E}\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n:SCANIT_END:\n"
    )
    data = np.asarray(frame, dtype=">f4")
    path.write_bytes(header.encode() + b"\x1a\x04" + data.tobytes() + data.tobytes())
    return path


def _noise(n=256, sigma=NOISE_M, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, size=(n, n))


def _tilted(angle_deg, *, n=256, frame_m=1e-7, seed=1, axis="x"):
    m_per_px = frame_m / n
    gy, gx = np.mgrid[:n, :n].astype(np.float64)
    slope = math.tan(math.radians(angle_deg))
    ramp = (gx if axis == "x" else gy) * m_per_px * slope
    return ramp + _noise(n, seed=seed)


def _run(path, **params):
    params.setdefault("scan_path", str(path))
    return AnalyzeFrameTilt().execute(None, params)


# ── 基本 ─────────────────────────────────────────────────────────────────────

def test_recovers_a_synthetic_tilt(tmp_path):
    p = _make_sxm(tmp_path / "t.sxm", _tilted(0.5, axis="x"))
    res = _run(p)
    assert res.success, res.error
    assert res.data["tilt"]["valid"] is True
    assert res.data["tilt"]["tilt_fast_deg"] == pytest.approx(0.5, abs=0.03)


def test_touches_no_hardware(tmp_path):
    """只读文件 —— 传 context=None 都必须能跑通。"""
    p = _make_sxm(tmp_path / "t.sxm", _noise())
    assert AnalyzeFrameTilt().execute(None, {"scan_path": str(p)}).success


def test_reports_geometry_including_rotation(tmp_path):
    p = _make_sxm(tmp_path / "t.sxm", _noise(), size_m=(2e-7, 2e-7),
                  angle_deg=30.0)
    g = _run(p).data["geometry"]
    assert g["width_m"] == pytest.approx(2e-7)
    assert g["angle_deg"] == pytest.approx(30.0)
    assert g["nm_per_px"] == pytest.approx(200.0 / 256)


# ── surface_rms:AutoTilt 的输入 ─────────────────────────────────────────────

def test_surface_rms_tracks_the_real_height_spread(tmp_path):
    """这是喂给 AutoTilt 的那个数 —— 它必须反映表面自身的起伏尺度。"""
    quiet = _run(_make_sxm(tmp_path / "a.sxm", _noise(sigma=15.4e-12)))
    rough = _run(_make_sxm(tmp_path / "b.sxm", _noise(sigma=154e-12, seed=2)))
    assert quiet.data["surface_rms_m"] == pytest.approx(15.4e-12, rel=0.2)
    assert rough.data["surface_rms_m"] == pytest.approx(154e-12, rel=0.2)


def test_steps_count_towards_surface_rms_but_not_local_texture(tmp_path):
    """两个口径回答的是不同问题,不能混用。

    台阶是表面形貌的一部分 → 进 surface_rms(AutoTilt 的分母);
    局部纹理对台阶免疫 → 用它当分母会让触发阈小一个数量级,几乎每帧都要求调平。
    """
    img = _noise(sigma=15.4e-12, seed=3)
    img[:, 128:] += STEP_M
    d = _run(_make_sxm(tmp_path / "s.sxm", img)).data
    assert d["surface_rms_m"] > 2.5 * d["local_texture_rms_m"]
    # 去趋势的合成台阶用来验证算法偏差。
    assert 40e-12 < d["surface_rms_m"] < 130e-12


def test_surface_rms_has_the_tilt_removed(tmp_path):
    """**必须**扣掉倾斜再算 —— 否则倾斜本身会被算进「形貌」,分母随倾斜一起变大,
    「斜坡淹没形貌」这条判据就永远不成立(自我抵消)。"""
    flat = _run(_make_sxm(tmp_path / "a.sxm", _noise(seed=8))).data
    steep = _run(_make_sxm(tmp_path / "b.sxm", _tilted(2.0, seed=8))).data
    assert steep["surface_rms_m"] == pytest.approx(flat["surface_rms_m"], rel=0.2)


def test_noise_floor_is_reported_for_recalibration(tmp_path):
    """噪声输入由固定种子生成，与仪器标定无关。"""
    p = _make_sxm(tmp_path / "n.sxm", _noise(sigma=50e-12, seed=4))
    assert _run(p).data["noise_floor_m"] == pytest.approx(50e-12, rel=0.3)


# ── 台阶主导 ─────────────────────────────────────────────────────────────────

def test_step_ratios_are_exposed_for_recalibration(tmp_path):
    """多尺度比值、逐尺度明细、单尺度对照都要给出来 —— 重标定要看整张表。"""
    img = _noise(seed=5)
    img[:, 128:] += STEP_M
    step = _run(_make_sxm(tmp_path / "s.sxm", img)).data["step"]
    assert step["ratio_multiscale"] > 1.4
    assert isinstance(step["ratio_by_tile"], dict) and step["ratio_by_tile"]
    assert "ratio_single_tile32" in step


def test_a_step_dominated_frame_refuses_to_report_a_tilt(tmp_path):
    """跨台阶拟合测的是包络不是失配角 —— 给个数比不给更糟。"""
    img = _noise(seed=6)
    img[:, 128:] += STEP_M
    d = _run(_make_sxm(tmp_path / "s.sxm", img)).data
    assert d["tilt"]["valid"] is False
    assert d["tilt"]["invalid_reason"] == "step_dense"


def test_check_steps_off_lets_you_inspect_the_raw_fit(tmp_path):
    img = _noise(seed=7)
    img[:, 128:] += STEP_M
    d = _run(_make_sxm(tmp_path / "s.sxm", img), check_steps=False).data
    assert d["tilt"]["valid"] is True      # 否决关掉了


# ── 慢扫轴诚实标注 ───────────────────────────────────────────────────────────

def test_slow_axis_is_always_flagged_untrusted(tmp_path):
    """一帧要扫半个多小时 —— 慢轴上的「倾斜」混着热漂移。"""
    p = _make_sxm(tmp_path / "t.sxm", _tilted(0.5))
    assert _run(p).data["tilt"]["slow_axis_trusted"] is False


def test_summary_says_which_axis_to_believe(tmp_path):
    p = _make_sxm(tmp_path / "t.sxm", _tilted(0.5))
    s = _run(p).summary
    assert "快扫轴" in s and "不可信" in s


# ── 失败路径 ─────────────────────────────────────────────────────────────────

def test_missing_file_is_reported(tmp_path):
    res = _run(tmp_path / "nope.sxm")
    assert not res.success and "文件不存在" in res.error


def test_a_frame_without_geometry_refuses_rather_than_guessing(tmp_path):
    """没有物理尺寸就换算不出角度 —— 编一个数字比不给更糟。"""
    p = tmp_path / "bad.sxm"
    header = (":SCAN_PIXELS:\n64 64\n"
              ":DATA_INFO:\n\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
              "\t14\tZ\tm\tboth\t1.0\t0.0\n\n:SCANIT_END:\n")
    frame = np.asarray(_noise(64), dtype=">f4")
    p.write_bytes(header.encode() + b"\x1a\x04" + frame.tobytes() * 2)
    res = _run(p)
    assert not res.success
    assert "几何" in res.error


def test_skill_is_analysis_and_auto(tmp_path):
    from mast.core.types import SafetyLevel, SkillCategory
    meta = AnalyzeFrameTilt().metadata()
    assert meta.category == SkillCategory.ANALYSIS
    assert meta.safety_level == SafetyLevel.AUTO
    assert meta.preconditions == []


def test_registered_and_frozen_safe():
    from mast.core.registry import SkillRegistry
    reg = SkillRegistry()
    reg.discover("mast.skills.builtins")
    assert reg.has("AnalyzeFrameTilt")
    import mast.skills.builtins as pkg
    assert "AnalyzeFrameTilt" in (pkg.__all__ or ())
