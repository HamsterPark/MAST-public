# -*- coding: utf-8 -*-
"""三个新分析技能的**外壳**：``AssessScanTexture`` / ``MeasureLatticeCell`` /
``AssessFeedbackTracking``。

判据本体在 ``tests/v2/unit/vision/test_frame_texture.py`` 等三份里；这里只测外壳
才管得到的五件事：

1. **真 .sxm 读得动**，且 IO 走的是 ``sxm_oriented_frames``（不自己扒 channels）；
2. **未标定的阈值没有 default** —— 有 default 时 pydantic 会替调用方填上，
   于是「这台机器没标定过」永远看不见（本仓已为这个形状付过五次学费）；
3. **三态**：``success=True`` 只要文件读得动，判决在 ``data.verdict``；
   「判不了」与「判出来不好」是两句不同的话；
4. **扫描顺序的映射**：定向之后 row 0 恒为帧顶，而它是先扫还是后扫取决于
   ``:SCAN_DIR:``；纯函数只报几何，外壳负责翻成扫描顺序；
5. 文件不存在这类**没做成**的事才用 ``success=False``。
"""
from __future__ import annotations

import numpy as np
import pytest

from mast.core.types import SafetyLevel, SkillCategory
from mast.skills.builtins.feedback_tracking import AssessFeedbackTracking
from mast.skills.builtins.frame_drift_skill import MeasureFrameDrift
from mast.skills.builtins.lattice_cell_skill import MeasureLatticeCell
from mast.skills.builtins.scan_texture import AssessScanTexture

SXM = "tests/v2/fixtures/nanonis/scan_topography.sxm"


@pytest.fixture()
def sxm(request) -> str:
    p = request.config.rootpath / SXM
    assert p.exists(), f"缺 fixture: {p}"
    return str(p)


ALL = (AssessScanTexture, MeasureLatticeCell, AssessFeedbackTracking,
       MeasureFrameDrift)


# ── 1. metadata 与注册 ───────────────────────────────────────────────────

@pytest.mark.parametrize("cls", ALL)
def test_metadata_is_read_only_analysis(cls):
    m = cls().metadata()
    assert m.category == SkillCategory.ANALYSIS
    assert m.safety_level == SafetyLevel.AUTO, (
        "只读分析不该要人确认；要人确认的是会动硬件的那些")
    assert m.name and m.version and m.description


@pytest.mark.parametrize("cls", ALL)
def test_registered_in_builtins(cls):
    import mast.skills.builtins as B
    assert cls.__name__ in B.__all__
    assert getattr(B, cls.__name__) is cls


def test_discovered_by_the_registry():
    """注册表按包扫描，所以新模块**自动**可见 —— 但这条要真的验一次，
    「加进 __all__」与「agent 用得上」是两件事。"""
    from mast.core.registry import SkillRegistry
    reg = SkillRegistry()
    reg.discover()
    names = {getattr(m, "name", m) for m in reg.list_skills()}
    for cls in ALL:
        assert cls().metadata().name in names, cls.__name__


# ── 2. 未标定的阈值不许有 default ───────────────────────────────────────

def test_good_ratio_has_no_default():
    """0.6 只在一个样品、一根针尖、一夜上标过。给它 default 会让
    「本机没标定」这件事永远看不见。"""
    spec = {p.name: p for p in AssessScanTexture().metadata().parameters}
    assert "good_ratio" in spec
    assert spec["good_ratio"].default is None
    assert spec["good_ratio"].required is False


def test_kappa_has_no_default():
    """κ 只有 MeasureBarrierHeight 量得出。编一个默认值会让保真度看起来
    有依据，而它对 κ 是线性敏感的。"""
    spec = {p.name: p for p in AssessFeedbackTracking().metadata().parameters}
    assert spec["kappa_per_nm"].default is None
    assert spec["kappa_per_nm"].required is False


def test_without_good_ratio_the_measurement_still_comes_out(sxm, monkeypatch):
    """没门槛时不给「好块占比」，但逐块比值与中位数照给 —— 它们是测量，
    不需要门槛。给不出数和拒绝下结论是两件事。"""
    r = AssessScanTexture().execute(None, {"scan_path": sxm})
    assert r.success
    assert r.data.get("tile_good_fraction") is None


def test_without_kappa_fidelity_is_none_but_lag_ratio_is_given(sxm):
    r = AssessFeedbackTracking().execute(None, {"scan_path": sxm})
    assert r.success and r.data["verdict"] == "measured"
    assert r.data["fidelity"] is None
    assert r.data["lag_ratio"] is not None
    assert "MeasureBarrierHeight" in r.data["fidelity_note"]


# ── 3. 三态 ─────────────────────────────────────────────────────────────

def test_a_frame_too_coarse_for_atoms_is_undetermined_not_a_failure(sxm):
    """100 nm / 256 px = 0.39 nm/px：0.4 nm 的周期只占一个像素。

    ⚠️ 这一条是**回归钉**：不设任何尺度门时 ``measure_cell`` 会在这张纯形貌帧上
    **成功返回一个原胞** —— 噪声里凑得出峰。判不了必须说判不了。

    门按**每周期像素数**判（``_MIN_PX_PER_PERIOD``），不按绝对的 nm/px：
    见 ``test_a_coarse_pixel_frame_with_a_long_period_is_still_measurable``。
    """
    r = AssessScanTexture().execute(None, {"scan_path": sxm})
    assert r.success is True, "读得动文件就不是技能失败"
    assert r.data["verdict"] == "undetermined"
    assert r.data["reason"] in ("too_few_pixels_per_period", "too_few_peaks",
                                "too_few_refined_peaks"), r.data["reason"]
    # 条纹幅值不依赖晶格方向，所以它照样给。
    assert r.data["streak_pm"] is not None and r.data["streak_pm"] > 0


def test_a_coarse_pixel_frame_with_a_long_period_is_still_measurable():
    """尺度门必须按**每周期像素数**判，不能按绝对 nm/px。

    ``atomic_phase.scale_gate`` 的 0.02/0.05 nm/px 是在 **0.25 nm** 的晶格上标的
    （= 12.5 / 5 px 每周期）。直接套到别的体系上会算错：2026-09-03/04 的 WO₂I₂
    （a ≈ 0.40/0.36 nm）在 58.6 pm/px 上扫了 **71 帧**，每周期 6.2-6.8 px、
    FFT 上两组基矢干干净净，而那道绝对门判它 ``off``。

    这里造一个同样形状的合成帧：像素尺度 0.06 nm/px（绝对门会拒），
    周期 0.4 nm（每周期 6.7 px，够）。必须量得出来。
    """
    from mast.vision.atomic_phase import scale_gate
    from mast.vision.lattice_cell import measure_cell
    n, nmpp = 512, 0.06
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    x, y = xx * nmpp, yy * nmpp
    img = (np.cos(2 * np.pi * x / 0.40) + np.cos(2 * np.pi * y / 0.36)
           + 0.35 * np.cos(2 * np.pi * (x / 0.40 + y / 0.36))) * 1e-12
    assert scale_gate(nmpp) == "off", "前提：绝对门确实会拒这个像素尺度"
    r = measure_cell(img, nmpp)
    assert r.ok, r.reason
    assert r.a1_nm == pytest.approx(0.40, abs=0.01)
    assert r.a2_nm == pytest.approx(0.36, abs=0.01)
    assert any("scale_gate" in w for w in r.warnings), (
        "两道门不一致时要说出来，而不是安静地各判各的")


def test_lattice_skill_reports_undetermined_with_per_frame_reasons(sxm):
    r = MeasureLatticeCell().execute(None, {"scan_paths": sxm})
    assert r.success is True
    assert r.data["verdict"] == "undetermined"
    assert r.data["n_frames"] == 0


def test_drift_needs_two_frames_and_refuses_across_different_frames(sxm):
    """同一个 .sxm 给两遍 ⇒ 位移必然是零，但**扫描框相同**这一关要先过。

    中心或尺寸不同的两帧之间的位移里混着「你把框挪了多少」，那不是漂移。
    """
    r = MeasureFrameDrift().execute(None, {"scan_paths": sxm})
    assert r.success is False and "两个" in r.error

    r = MeasureFrameDrift().execute(None, {"scan_paths": "%s,%s" % (sxm, sxm)})
    assert r.success is True
    assert r.data["verdict"] == "measured"
    assert r.data["dy_median_nm"] == pytest.approx(0.0, abs=1e-6)
    assert r.data["pairs"][0]["confidence"] == pytest.approx(1.0, abs=1e-3)


def test_missing_file_is_a_skill_failure(sxm):
    for cls, key, val in ((AssessScanTexture, "scan_path", "no_such_file.sxm"),
                          (AssessFeedbackTracking, "scan_path", "no_such_file.sxm"),
                          # 漂移技能先过「至少两个路径」那一关，所以给两个。
                          (MeasureFrameDrift, "scan_paths", "a.sxm,b.sxm"),
                          (MeasureLatticeCell, "scan_paths", "no_such_file.sxm")):
        r = cls().execute(None, {key: val})
        assert r.success is False, cls.__name__
        assert "不存在" in r.error, (cls.__name__, r.error)


def test_empty_paths_is_a_failure_not_an_empty_result():
    r = MeasureLatticeCell().execute(None, {"scan_paths": ""})
    assert r.success is False and "scan_paths" in r.error


def test_some_readable_some_not_is_a_partial_result_not_a_failure(sxm):
    """一批里有读不到的，不该拖垮整批 —— 但它必须出现在 ``rejected`` 里，
    而不是被安静地丢掉（「读不到」永远不能折叠成「没有」）。"""
    r = MeasureLatticeCell().execute(
        None, {"scan_paths": "%s,no_such_file.sxm" % sxm})
    assert r.success is True
    assert any("不存在" in x["reason"] for x in r.data["rejected"])


# ── 4. 工作点从文件头取，不让调用方手填 ─────────────────────────────────

def test_working_point_comes_from_the_header(sxm):
    """线速度与 Z 增益由文件头给 —— 这正是 lag_ratio 能自检的前提：
    图像给一个数、文件头给另一个数，两者独立。"""
    r = AssessFeedbackTracking().execute(None, {"scan_path": sxm})
    d = r.data
    assert d["speed_m_s"] is not None and d["speed_m_s"] > 0
    # 这张老 fixture 的头里没有 z-controller 段，于是自检做不了 —— 但那要
    # 如实报成 None，不能凑一个数出来。
    assert d["i_gain_m_s"] is None
    assert d["expected_lag_ratio"] is None
    assert d["ratio_disagreement"] is None


def test_missing_current_channel_is_named(tmp_path, sxm):
    """没有 Current 通道时要说清楚下一步（把它加进采集通道），
    而不是只说「失败」。"""
    r = AssessFeedbackTracking().execute(None, {"scan_path": sxm,
                                                "direction": "backward"})
    assert r.success is True
    # 这张 fixture 正反扫都有；换个不存在的通道来走那条路
    from mast.skills.builtins._sxm_frame import load_frame
    fr = load_frame(sxm, "NoSuchChannel")
    assert fr.error and "没有通道" in fr.error


# ── 5. 几何 → 扫描顺序的映射由外壳做 ────────────────────────────────────

def test_scan_order_mapping_uses_scan_dir():
    """``up`` 帧的第一行是帧**底**，所以 first_scanned 取几何下沿。

    纯函数只报 top/bottom（它看不到文件头），映射在技能层。搞反的话
    「针尖在这一帧里变了」会指向错误的一端。
    """
    from mast.vision.frame_texture import tile_lattice_map
    n, nmpp = 512, 0.0195
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # 上半（帧顶）有晶格，下半是噪声
    rng = np.random.default_rng(2)
    lat = 10.0 * np.cos(2 * np.pi * xx * nmpp / 0.4)
    img = np.vstack([lat[: n // 2], rng.normal(0, 10.0, (n // 2, n))]) * 1e-12
    t = tile_lattice_map(img, nmpp, [(0.0, 0.4)], tile_nm=4.0)
    assert t.ok and t.top_band_median > t.bottom_band_median

    # 外壳的映射：down 帧 row 0 是先扫，up 帧 row 0 是后扫。
    for scan_dir, expect_first_is_top in (("down", True), ("up", False)):
        first, last = ((t.bottom_band_median, t.top_band_median)
                       if scan_dir.startswith("up")
                       else (t.top_band_median, t.bottom_band_median))
        assert (first == t.top_band_median) is expect_first_is_top, scan_dir
