"""地图汇总必须为无法归入扫描、谱或破坏的事件提供独立计数。

覆盖率当前随实时扫描框决定的栅格尺度变化；测试同时记录这种几何依赖，
防止未知事件被当作没有事件，也使未来栅格解耦时能明确更新契约。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.io.coarse_map import CoarseMapConfig, build_coarse_map, derive_sites  # noqa: E402
from mast.io.exp_map import MapMarker  # noqa: E402
from mast.io.map_analysis import (  # noqa: E402
    AnalysisConfig,
    coverage_stats,
    rasterize,
)


def row(kind, epoch=0, **kw):
    d = {"kind": kind, "coord_epoch": epoch, "timestamp": "2000-01-01T00:00:00",
         "x_m": 0.0, "y_m": 0.0, "meta": {}}
    d.update(kw)
    return d


# ── 站点摘要:归不了类的行必须有自己的列 ──────────────────────────────────

def test_a_busy_site_does_not_read_as_an_untouched_one():
    """新增手动事件必须改变站点摘要；无法进一步分类的事件应进入 other 计数。"""
    quiet = derive_sites([row("scan"), row("scan")], CoarseMapConfig())
    busy = derive_sites([row("scan"), row("scan")]
                        + [row("manual") for _ in range(11)], CoarseMapConfig())

    assert quiet[0].summary["scans"] == busy[0].summary["scans"] == 2
    assert busy[0].summary != quiet[0].summary, (
        "11 条事件没让摘要动一下 —— 账本把数不清的记成了没有")
    assert busy[0].summary["other"] == 11


def test_the_site_summary_adds_up_to_the_rows_it_was_given():
    """三列 + other = 这个站点收到的全部行(粗动边界除外)。

    「加得起来」是这条测试的全部内容:只要有一类行谁都不认领,它就必须出现在
    ``other`` 里,而不是从总数里蒸发。"""
    rows = ([row("scan")] * 3 + [row("sts")] * 2 + [row("pulse")] * 4
            + [row("manual")] * 7 + [row("move")] * 5)
    s = derive_sites(rows, CoarseMapConfig())[0].summary

    counted = s["scans"] + s["sts"] + s["damage"] + s["other"]
    assert counted == len(rows), f"{len(rows)} 行进去,只数出 {counted} 行:{s}"


def test_the_boundary_row_is_not_counted_as_work_at_the_site():
    """``coarse_move`` 是**离开**,不是在这里干的活。

    没有这一条,「把没归类的都算进 other」会让每个站点凭空多一件事。"""
    sites = derive_sites([row("scan"), row("coarse_move"), row("scan", epoch=1)],
                         CoarseMapConfig())
    assert sites[0].summary["other"] == 0, sites[0].summary
    assert sites[0].summary["scans"] == 1


def test_the_relocation_reason_mentions_the_unclassified_events():
    """换位建议里那句「当前站点已用 X 扫图 / Y 破坏」也不许漏掉它们。

    那句话是用户决定要不要换区时读的唯一一行摘要。"""
    rows = [row("scan")] + [row("manual") for _ in range(11)]
    cmap = build_coarse_map(rows, CoarseMapConfig())
    assert cmap.suggestion is not None
    assert "11 条未归类事件" in cmap.suggestion.reason, cmap.suggestion.reason


def test_a_clean_site_does_not_grow_a_noise_clause():
    """反过来:没有未归类事件时不许平白加那半句。

    没有这一条,把那半句写成无条件的也能让上面那条变绿。"""
    cmap = build_coarse_map([row("scan"), row("scan")], CoarseMapConfig())
    assert cmap.suggestion is not None
    assert "未归类" not in cmap.suggestion.reason, cmap.suggestion.reason


# ── 覆盖率:它跟着实时扫描框变,与标记无关(机制钉住,尚未修) ──────────────

def _scans(n=24, size=120e-9, seed=7):
    import random
    rnd = random.Random(seed)
    return [MapMarker(kind="scan", w_m=size, h_m=size, status="done",
                      x_m=rnd.uniform(-1e-6, 1e-6), y_m=rnd.uniform(-1e-6, 1e-6))
            for _ in range(n)]


def _coverage(markers, frame_nm):
    cfg = AnalysisConfig(piezo_half_range_m=1.5e-6, frame_size_m=frame_nm * 1e-9)
    covered, blocked = rasterize(markers, cfg)
    return coverage_stats(covered, blocked)[0], covered.shape[0]


def test_coverage_moves_when_only_the_live_scan_frame_moved():
    """同一组合成标记只改变当前扫描框，覆盖率栅格仍会改变。
    该测试记录栅格与实时视野耦合的既有行为；解耦后应一并更新此回归。"""
    markers = _scans()
    cov_100, grid_100 = _coverage(markers, 120)
    cov_50, grid_50 = _coverage(markers, 60)

    assert grid_100 != grid_50, "栅格没变的话这条测试就没有在测它以为在测的东西"
    assert cov_100 != cov_50, (
        "如果这里相等了,说明栅格已经与实时扫描框解耦 —— 缺陷已修,请删掉本测试")


def test_the_grid_is_reported_so_two_readings_can_be_told_apart():
    """两个覆盖率能不能比较,读的人必须看得见依据。"""
    from mast.io.map_analysis import analyze_map

    rows = [row("scan", w_m=1.2e-7, h_m=1.2e-7, x_m=0.0, y_m=0.0)]
    a = analyze_map(rows, AnalysisConfig(piezo_half_range_m=1.5e-6,
                                         frame_size_m=120e-9))
    b = analyze_map(rows, AnalysisConfig(piezo_half_range_m=1.5e-6,
                                         frame_size_m=60e-9))

    assert a.grid_cells > 0 and a.grid_cell_m > 0
    assert a.grid_cells != b.grid_cells, (
        "两次分析的栅格不同却报同一个 grid_cells —— 那就没人能发现它们不可比")


def test_coverage_plus_usable_unscanned_equals_one_only_without_damage():
    """无损伤标记时，覆盖率与可用未扫描比例应合计为一；添加损伤后不应仍合计为一。"""
    clean = _scans(n=5)
    cov, _ = _coverage(clean, 120)
    cfg = AnalysisConfig(piezo_half_range_m=1.5e-6, frame_size_m=120e-9)
    covered, blocked = rasterize(clean, cfg)
    _, _, uu = coverage_stats(covered, blocked)
    assert cov + uu == pytest.approx(1.0, abs=1e-12)

    dirty = clean + [MapMarker(kind="pulse", x_m=0.0, y_m=0.0)]
    covered, blocked = rasterize(dirty, cfg)
    c2, _, uu2 = coverage_stats(covered, blocked)
    assert c2 + uu2 < 0.999, "有破坏时两者不该再加成 1"
