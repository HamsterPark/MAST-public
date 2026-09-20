"""既有扫描 composite 的收编:默认参数走档位表(单一真源)。

在这之前,``FullScan`` / ``SurveySurface`` / ``BatchRegionsScan`` 各自带着一个
写死的 ``line_time_s = 0.1``。那个常数对 1 µm 的巡查图和 5 nm 的原子分辨图给的
是同一个速度,而它们之间差着一两个数量级。

收编之后,「没有显式指定时用什么参数」在全系统只有一个来源:用户的按尺度
档位表。显式指定仍然最高优先(XD 定的)。
"""

from __future__ import annotations

import pytest

from mast.core import scan_policy
from mast.skills.composite.batch_regions_scan import BatchRegionsScan
from mast.skills.composite.full_scan import FullScan
from mast.skills.composite.survey_surface import SurveySurface_TileScan


@pytest.fixture(autouse=True)
def _clean_policy():
    scan_policy.set_policy(None)
    yield
    scan_policy.set_policy(None)


def _step(steps, skill, contains=None):
    for s in steps:
        if s.skill_name == skill and (contains is None or contains in s.step_id):
            return s
    raise AssertionError(f"没有找到 {skill} 步骤")


# ── FullScan ─────────────────────────────────────────────────────────────────

def test_full_scan_takes_the_line_time_from_the_tier_for_its_size():
    steps = FullScan().plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 1e-6, "height_m": 1e-6,          # survey 档:0.5 s
    })
    assert _step(steps, "SetScanSpeed").params["fwd_line_time"] == 0.5


def test_full_scan_uses_a_different_tier_for_a_small_frame():
    steps = FullScan().plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        # 5 nm ⇒ **atomic_verify** 档:0.30 s。
        # 2026-08-14 之前这里是 atomic 档的 1.2 s;新档 atomic_verify(≤5 nm /
        # 512 px / 0.30 s)插在 slow 与 atomic 之间,于是 2–5 nm 改落新档。
        # 那不是变慢或变快的问题 —— 512 px 配 0.30 s 让**每像素驻留时间**
        # 保持在用户配方的 586 µs(256 px 配 0.15 s 的同一个数),同时把
        # nm/px 压进原子判据的满权重档。加像素而不同比加线时,等于把驻留砍半。
        "width_m": 5e-9, "height_m": 5e-9,
    })
    assert _step(steps, "SetScanSpeed").params["fwd_line_time"] == 0.30


def test_full_scan_honours_an_explicit_line_time():
    """XD 定的最高优先 —— 档位表不能盖过用户这一次点名的值。"""
    steps = FullScan().plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 1e-6, "height_m": 1e-6, "line_time_s": 0.03,
    })
    assert _step(steps, "SetScanSpeed").params["fwd_line_time"] == 0.03


def test_full_scan_follows_an_edited_tier_table():
    scan_policy.set_policy([
        {"name": "mine", "upper_size_m": None, "pixels": 256, "line_time_s": 7.0},
    ])
    steps = FullScan().plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 1e-6, "height_m": 1e-6,
    })
    assert _step(steps, "SetScanSpeed").params["fwd_line_time"] == 7.0


def test_full_scan_line_time_default_is_none_so_omission_is_detectable():
    """default 若写成 0.1,pydantic 会把它物化进参数,「没传」与「显式 0.1」
    就永远分不开 —— 档位表也就永远轮不到。"""
    spec = {p.name: p for p in FullScan().metadata().parameters}["line_time_s"]
    assert spec.default is None


# ── SurveySurface ────────────────────────────────────────────────────────────

def test_survey_uses_the_tier_of_the_TILE_not_the_whole_survey():
    """实际扫的是一块一块的 tile —— 该用哪个档由 tile 尺寸决定,
    不是由整片 survey 的尺寸决定。"""
    # grid 被强制在 2..8 之间,所以 total/tile 要落在这个比例内
    steps = SurveySurface_TileScan().plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "total_size_m": 3e-7,      # 整片 300 nm(若按它查档 → roi 0.8 s)
        "tile_size_m": 5e-8,       # 每块 50 nm → highres 档 1.0 s
    })
    assert _step(steps, "SetScanSpeed").params["fwd_line_time"] == 1.0


def test_survey_honours_an_explicit_line_time():
    steps = SurveySurface_TileScan().plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "total_size_m": 2e-7, "tile_size_m": 5e-8, "line_time_s": 0.25,
    })
    assert _step(steps, "SetScanSpeed").params["fwd_line_time"] == 0.25


def test_survey_line_time_default_is_none():
    spec = {p.name: p for p in SurveySurface_TileScan().metadata().parameters}["line_time_s"]
    assert spec.default is None


# ── BatchRegionsScan ─────────────────────────────────────────────────────────

def _regions_json(sizes):
    import json
    return json.dumps([
        {"center_x_m": i * 1e-7, "center_y_m": 0.0,
         "width_m": s, "height_m": s, "label": f"r{i}"}
        for i, s in enumerate(sizes)
    ])


def test_batch_resolves_the_line_time_per_region():
    """一个批次里的区域尺寸可以差很多 —— 给它们同一个速度对大多数都是错的。"""
    steps = BatchRegionsScan().plan({
        "regions": _regions_json([1e-6, 5e-8, 5e-9]),
    })
    speeds = [s.params["fwd_line_time"] for s in steps
              if s.skill_name == "SetScanSpeed"]
    # survey / highres / atomic_verify（5 nm 那一个 2026-08-14 起落新档，见上）
    assert speeds == [0.5, 1.0, 0.30]


def test_batch_honours_an_explicit_line_time_for_every_region():
    steps = BatchRegionsScan().plan({
        "regions": _regions_json([1e-6, 5e-9]),
        "line_time_s": 0.2,
    })
    speeds = [s.params["fwd_line_time"] for s in steps
              if s.skill_name == "SetScanSpeed"]
    assert speeds == [0.2, 0.2]


def test_batch_line_time_default_is_none():
    spec = {p.name: p
            for p in BatchRegionsScan().metadata().parameters}["line_time_s"]
    assert spec.default is None


def test_batch_scan_speed_matches_the_region_size_and_line_time():
    """速度是由「区域尺寸 / 每线时间」导出的 —— 两者都换档后要保持一致。"""
    steps = BatchRegionsScan().plan({"regions": _regions_json([1e-6])})
    speed_step = _step(steps, "SetScanSpeed")
    assert speed_step.params["fwd_speed"] == pytest.approx(1e-6 / 0.5)
