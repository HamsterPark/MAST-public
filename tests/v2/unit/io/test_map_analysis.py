"""Scan-map analysis — the judgements the agent is not allowed to eyeball.

Pure-function tests on synthetic marker rows. Every threshold is passed through
``AnalysisConfig``, so no test here patches a private constant to state its
premises (and none of them touch a database).
"""
from __future__ import annotations

import math

import pytest

from mast.io import map_analysis as MA
from mast.io.exp_map import markers_from_rows
from mast.io.map_analysis import (
    AnalysisConfig,
    analyze_map,
    build_avoid_circles,
    candidate_positions,
    coarse_move_advice,
    coverage_stats,
    current_epoch_of,
    epoch_series,
    filter_epoch,
    markers_near,
    pick_next_position,
    rasterize,
    sts_coverage,
    survey_progress,
)

# ── helpers ────────────────────────────────────────────────────────────────

def row(kind, x_nm=None, y_nm=None, w_nm=None, h_nm=None, angle=0.0,
        status="done", meta=None, label="", epoch=0, ts="2026-07-30T10:00:00"):
    """One map_markers row as ``get_markers`` would return it (nm → m)."""
    return {
        "kind": kind,
        "x_m": None if x_nm is None else x_nm * 1e-9,
        "y_m": None if y_nm is None else y_nm * 1e-9,
        "w_m": None if w_nm is None else w_nm * 1e-9,
        "h_m": None if h_nm is None else h_nm * 1e-9,
        "angle_deg": angle,
        "status": status,
        "source": "skill",
        "label": label,
        "skill_name": "",
        "timestamp": ts,
        "meta": meta or {},
        "coord_epoch": epoch,
    }


def cfg(**kw):
    """Small, fast analysis universe unless a test says otherwise."""
    base = dict(piezo_half_range_m=500e-9, grid_cell_m=10e-9,
                frame_size_m=100e-9, edge_margin_frac=0.0)
    base.update(kw)
    return AnalysisConfig(**base)


# ── coverage rasterisation ─────────────────────────────────────────────────

def test_coverage_of_one_axis_aligned_scan_matches_area_ratio():
    c = cfg()
    # A 200×200 nm frame inside a 1000×1000 nm universe = 4% of the area.
    covered, blocked = rasterize(
        markers_from_rows([row("scan", 0, 0, 200, 200)]), c)
    frac, usable, unscanned = coverage_stats(covered, blocked)
    assert frac == pytest.approx(0.04, abs=0.005)
    assert usable == pytest.approx(1.0)          # nothing damaged
    assert unscanned == pytest.approx(0.96, abs=0.005)


def test_rotated_footprint_is_not_its_bounding_box():
    """A 45°-rotated square covers its true area, not the circumscribed box.

    Claiming coverage we do not have is the error that makes a survey skip real
    surface, so the rotated case gets tested against the honest number."""
    c = cfg()
    straight, _ = rasterize(markers_from_rows([row("scan", 0, 0, 200, 200)]), c)
    rotated, _ = rasterize(
        markers_from_rows([row("scan", 0, 0, 200, 200, angle=45.0)]), c)
    # Same area (within rasterisation error), NOT the 2× bounding box.
    assert int(rotated.sum()) == pytest.approx(int(straight.sum()), rel=0.05)


def test_coverage_is_accurate_at_the_real_piezo_scale():
    """Regression: the grid must resolve a scan frame.

    Coverage is decided by whether a cell centre lands inside a footprint, so a
    frame only a few cells wide gains or loses a whole row of cells depending on
    where it happens to sit. At the default 25 nm cell a 100 nm frame was 4 cells
    across and three frames read 0.44% instead of 0.33% — a 33% overstatement of
    a number that decides when to abandon the area."""
    c = AnalysisConfig(piezo_half_range_m=1.5e-6, frame_size_m=100e-9)
    rows = [row("scan", 0, 0, 100, 100), row("scan", 120, 0, 100, 100),
            row("scan", 0, 120, 100, 100)]
    covered, blocked = rasterize(markers_from_rows(rows), c)
    frac, _, _ = coverage_stats(covered, blocked)
    exact = 3 * (100e-9 ** 2) / ((3.0e-6) ** 2)
    assert frac == pytest.approx(exact, rel=0.05)


def test_failed_scan_does_not_count_as_covered():
    c = cfg()
    covered, _ = rasterize(
        markers_from_rows([row("scan", 0, 0, 200, 200, status="failed")]), c)
    assert covered.sum() == 0


# ── avoidance semantics ───────────────────────────────────────────────────

def test_each_damage_kind_uses_its_own_radius():
    c = cfg(tip_shape_r_m=30e-9, pulse_r_m=150e-9, crash_r_m=150e-9,
            approach_r_m=200e-9)
    circles = build_avoid_circles(markers_from_rows([
        row("tip_shape", 0, 0), row("pulse", 200, 0),
        row("crash", -200, 0), row("approach", 0, 200),
    ]), c)
    got = {x.kind: x.radius_m for x in circles}
    assert got == {"tip_shape": 30e-9, "pulse": 150e-9,
                   "crash": 150e-9, "approach": 200e-9}


@pytest.mark.parametrize("damages,expected", [(True, 1), (False, 0)])
def test_approach_avoidance_follows_the_rig_capability(damages, expected):
    """A rig whose approach does not touch the surface should not pay for it —
    and "unknown" resolves to True upstream, so only an explicit no disables it."""
    circles = build_avoid_circles(
        markers_from_rows([row("approach", 0, 0)]),
        cfg(approach_damages=damages))
    assert len(circles) == expected


def test_operator_placed_keepout_becomes_an_avoid_circle():
    circles = build_avoid_circles(
        markers_from_rows([row("manual", 100, 0,
                               meta={"avoid_radius_m": 80e-9})]), cfg())
    assert [(x.kind, x.radius_m) for x in circles] == [("manual_avoid", 80e-9)]


def test_scanned_area_does_not_reduce_usable_area():
    """已扫 ≠ 不可用. Only damage subtracts from the surface budget: returning to
    a clean, already-imaged spot to take spectra is ordinary work."""
    c = cfg()
    covered, blocked = rasterize(
        markers_from_rows([row("scan", 0, 0, 400, 400)]), c)
    frac, usable, unscanned = coverage_stats(covered, blocked)
    assert frac > 0.1
    assert usable == pytest.approx(1.0)      # untouched by the scan
    assert unscanned < 1.0 - 0.1             # but no longer "new"


def test_damage_reduces_usable_area():
    c = cfg(tip_shape_r_m=200e-9)
    covered, blocked = rasterize(markers_from_rows([row("tip_shape", 0, 0)]), c)
    _, usable, _ = coverage_stats(covered, blocked)
    expected = 1.0 - math.pi * 200e-9 ** 2 / (1000e-9 ** 2)
    assert usable == pytest.approx(expected, abs=0.01)


# ── candidate route ───────────────────────────────────────────────────────

def test_candidate_sequence_is_deterministic():
    """The route is a pure function of the configuration — that is what lets an
    interrupted survey resume without anyone storing a cursor."""
    c = cfg()
    assert candidate_positions(c) == candidate_positions(c)


def test_center_first_starts_at_the_piezo_centre():
    """Where the scan tube creeps least."""
    first = candidate_positions(cfg(strategy="center_first"))[0]
    assert first[0] == pytest.approx(0.0)
    assert first[1] == pytest.approx(0.0)
    assert first[2] == 0


def test_perimeter_inward_starts_on_the_outer_ring_and_works_in():
    c = cfg(strategy="perimeter_inward")
    pts = candidate_positions(c)
    radii_by_ring: dict[int, float] = {}
    for x, y, ring in pts:
        radii_by_ring.setdefault(ring, math.hypot(x, y))
    rings = sorted(radii_by_ring)
    assert rings[0] == 0
    # Outer ring comes first, and each subsequent ring is strictly closer in.
    assert all(radii_by_ring[a] > radii_by_ring[b]
               for a, b in zip(rings, rings[1:]))
    # The first point emitted belongs to the outermost ring.
    assert pts[0][2] == 0


def test_candidate_count_is_capped_and_truncation_is_reported():
    """A tiny frame in a large range implies tens of thousands of positions;
    the search is bounded and says so rather than pretending."""
    c = AnalysisConfig(piezo_half_range_m=1.5e-6, grid_cell_m=50e-9,
                       frame_size_m=5e-9, max_candidates=300)
    assert len(candidate_positions(c)) == 300
    res = analyze_map([], c)
    assert res.route_truncated is True


# ── position selection ────────────────────────────────────────────────────

def test_next_position_avoids_a_damaged_centre():
    """The centre is preferred — unless something ruined it."""
    c = cfg(strategy="center_first", tip_shape_r_m=120e-9)
    pos, _ = pick_next_position(markers_from_rows([row("tip_shape", 0, 0)]), c)
    assert pos is not None
    assert math.hypot(pos.x_m, pos.y_m) > 0
    # The frame must clear the keep-out disc entirely.
    assert (max(abs(0.0 - pos.x_m) - c.frame_size_m / 2, 0.0) ** 2
            + max(abs(0.0 - pos.y_m) - c.frame_size_m / 2, 0.0) ** 2) > 120e-9 ** 2
    assert "避让区" in pos.reason


def test_next_position_resumes_after_already_scanned_points():
    """Interrupted survey: the record alone decides where to pick up."""
    c = cfg(strategy="center_first")
    route = candidate_positions(c)
    # Fully cover the first three candidates.
    scanned = [row("scan", x * 1e9, y * 1e9, 100, 100)
               for x, y, _ in route[:3]]
    pos, _ = pick_next_position(markers_from_rows(scanned), c)
    assert pos is not None
    done = {(round(x, 12), round(y, 12)) for x, y, _ in route[:3]}
    assert (round(pos.x_m, 12), round(pos.y_m, 12)) not in done


def test_next_position_is_none_when_the_surface_is_spent():
    c = cfg(strategy="center_first", crash_r_m=2000e-9)   # one disc covers all
    pos, left = pick_next_position(markers_from_rows([row("crash", 0, 0)]), c)
    assert pos is None
    assert left == 0


def test_next_position_frame_stays_inside_the_piezo_range():
    c = cfg(strategy="perimeter_inward", edge_margin_frac=0.0)
    pos, _ = pick_next_position([], c)
    assert pos is not None
    half = c.frame_size_m / 2
    assert abs(pos.x_m) + half <= c.piezo_half_range_m + 1e-18
    assert abs(pos.y_m) + half <= c.piezo_half_range_m + 1e-18


# ── coarse-move advice ────────────────────────────────────────────────────

def test_no_coarse_advice_on_a_rig_that_cannot_relocate():
    """"Just move somewhere else" is not an available recovery here, and saying
    it would be worse than useless."""
    adv = coarse_move_advice(cfg(has_xy_coarse_motion=False),
                             next_position=None, usable_unscanned_frac=0.0,
                             candidates_left=0)
    assert adv.suggest is False
    assert "没有 XY 粗动" in adv.reasons[0]


def test_coarse_advice_when_candidates_are_exhausted():
    adv = coarse_move_advice(cfg(), next_position=None,
                             usable_unscanned_frac=0.9, candidates_left=0)
    assert adv.suggest is True
    assert any("没有可用的候选位置" in r for r in adv.reasons)


def test_coarse_advice_when_usable_unscanned_area_runs_low():
    c = cfg(min_usable_unscanned_frac=0.15)
    adv = coarse_move_advice(c, next_position=None, usable_unscanned_frac=0.05,
                             candidates_left=0)
    assert adv.suggest is True
    assert any("可用且未扫的面积" in r for r in adv.reasons)


def test_center_first_flags_a_ruined_centre_zone():
    """Centre-first exists because the tube creeps least there; once the centre
    is wrecked the strategy has lost its reason to exist."""
    c = cfg(strategy="center_first", crash_r_m=300e-9, center_zone_frac=0.25)
    _, blocked = rasterize(markers_from_rows([row("crash", 0, 0)]), c)
    adv = coarse_move_advice(c, next_position=None, usable_unscanned_frac=0.9,
                             candidates_left=3, blocked=blocked)
    assert any("中心区" in r for r in adv.reasons)


# ── coordinate generations ────────────────────────────────────────────────

def test_epoch_series_counts_coarse_moves_in_order():
    rows = [row("scan"), row("coarse_move"), row("scan"), row("coarse_move"),
            row("sts")]
    assert epoch_series(rows) == [0, 0, 1, 1, 2]
    assert current_epoch_of(rows) == 2


def test_analysis_ignores_markers_from_a_dead_coordinate_system():
    """After a lateral coarse move the old numbers address different surface.
    Counting them would mark fresh surface as already-scanned — and worse, would
    keep the tip away from a patch that was never damaged."""
    c = cfg(crash_r_m=300e-9)
    rows = [
        row("crash", 0, 0, epoch=0),          # ruined the OLD area
        row("scan", 0, 0, 400, 400, epoch=0),
        row("coarse_move", 0, 0, epoch=0),    # ← boundary
    ]
    res = analyze_map(rows, c)
    assert res.current_epoch == 1
    assert res.markers_total == 3
    assert res.markers_current_epoch == 0
    assert res.avoid_circles == []
    assert res.coverage_frac == pytest.approx(0.0)
    # The centre is fresh surface again.
    assert res.next_position is not None
    assert (res.next_position.x_m, res.next_position.y_m) == pytest.approx((0.0, 0.0))


def test_filter_epoch_derives_generation_when_the_column_is_missing():
    """Works on a database whose rows predate the column."""
    rows = [row("scan"), row("coarse_move"), row("sts")]
    for r in rows:
        del r["coord_epoch"]
    assert len(filter_epoch(rows, 0)) == 2
    assert filter_epoch(rows, 1)[0]["kind"] == "sts"


# ── queries ───────────────────────────────────────────────────────────────

def test_sts_coverage_reports_the_true_total_when_truncated():
    c = cfg(max_sts_points=2)
    pts, total = sts_coverage(
        markers_from_rows([row("sts", i * 10, 0) for i in range(5)]), c)
    assert len(pts) == 2
    assert total == 5           # never mistake a display cap for the real count


def test_markers_near_sorts_by_distance_and_flags_dead_coordinates():
    rows = [row("tip_shape", 0, 0, epoch=0), row("pulse", 50, 0, epoch=1),
            row("crash", 5, 0, epoch=1)]
    hits = markers_near(markers_from_rows(rows), 0.0, 0.0, 100e-9,
                        current_epoch=1)
    assert [h["kind"] for h in hits] == ["tip_shape", "crash", "pulse"]
    # History is searched across generations, but a stale coordinate says so.
    assert hits[0]["stale_coords"] is True
    assert hits[1]["stale_coords"] is False


def test_survey_progress_reports_no_completion_percentage():
    """The plan overlay consumes steps as they are reached, so the original
    total is unrecoverable; inventing a denominator would make the number move
    for reasons unrelated to progress."""
    plan = markers_from_rows([row("plan", 0, 0), row("plan", 300, 0)])
    scanned = markers_from_rows([row("scan", 0, 0, 100, 100)])
    out = survey_progress(plan, scanned, tol_m=50e-9)
    assert out == {"pending_plan_steps": 2, "pending_already_scanned": 1}
    assert not any("percent" in k or "pct" in k for k in out)


# ── top level ─────────────────────────────────────────────────────────────

def test_analyze_map_on_an_empty_record_recommends_the_start_of_the_route():
    res = analyze_map([], cfg(strategy="center_first"))
    assert res.current_epoch == 0
    assert res.coverage_frac == 0.0
    assert res.usable_frac == pytest.approx(1.0)
    assert res.next_position is not None
    assert res.coarse_advice.suggest is False


def test_analyze_map_counts_damage_by_kind():
    res = analyze_map([row("tip_shape", 0, 0), row("tip_shape", 100, 0),
                       row("crash", -200, 0)], cfg())
    assert res.damage_counts == {"tip_shape": 2, "crash": 1}


def test_analyze_map_tolerates_markers_without_coordinates():
    """A coarse move whose tip position could not be read still bounds a
    generation — the count comes from the row existing, not from its xy."""
    rows = [row("scan", 0, 0, 100, 100), row("coarse_move")]
    res = analyze_map(rows, cfg())
    assert res.current_epoch == 1
    assert res.markers_current_epoch == 0


def test_analyze_map_is_fast_enough_for_a_button():
    """Full-size universe, a realistic pile of markers, interactive budget.

    Asks for the upcoming route too, because that is what the endpoint does —
    a budget that only holds when the extra walk is skipped is not a budget."""
    import time
    rows = [row("scan", (i % 20) * 100 - 1000, (i // 20) * 100 - 1000, 100, 100)
            for i in range(200)]
    rows += [row("tip_shape", (i % 10) * 90 - 400, 400) for i in range(30)]
    c = AnalysisConfig(piezo_half_range_m=1.5e-6, grid_cell_m=25e-9,
                       frame_size_m=100e-9)
    t0 = time.perf_counter()
    analyze_map(rows, c, upcoming_count=8)
    assert (time.perf_counter() - t0) < 1.0


# ── upcoming route (what the map draws ahead of the tip) ───────────────────

def test_upcoming_is_empty_unless_asked_for():
    """The agent tools want a decision, not a route; every position they do not
    ask for is context they do not pay for."""
    assert analyze_map([], cfg()).upcoming == []


def test_upcoming_starts_at_the_recommended_position():
    """The route the map draws must begin where the recommendation points, or
    the operator is looking at two different plans."""
    res = analyze_map([row("tip_shape", 0, 0)], cfg(strategy="center_first"),
                      upcoming_count=5)
    assert res.next_position is not None
    assert len(res.upcoming) == 5
    assert res.upcoming[0].x_m == pytest.approx(res.next_position.x_m)
    assert res.upcoming[0].y_m == pytest.approx(res.next_position.y_m)


def test_upcoming_frames_do_not_overlap_each_other():
    """Drawn as a route, so it has to be executable: N distinct patches of
    surface, not one recommendation restated N times."""
    c = cfg(strategy="center_first")
    res = analyze_map([], c, upcoming_count=6)
    pts = [(p.x_m, p.y_m) for p in res.upcoming]
    assert len(pts) == 6
    for i, (x, y) in enumerate(pts):
        for (px, py) in pts[i + 1:]:
            assert abs(x - px) >= c.frame_size_m or abs(y - py) >= c.frame_size_m


def test_upcoming_respects_damage_and_coverage():
    """Same filters as the single recommendation — a route is not a licence to
    plan through a keep-out zone or over ground already imaged."""
    c = cfg(strategy="center_first", crash_r_m=200e-9)
    route = candidate_positions(c)
    scanned = [row("scan", x * 1e9, y * 1e9, 100, 100) for x, y, _ in route[:4]]
    res = analyze_map([row("crash", 0, 0), *scanned], c, upcoming_count=6)
    done = {(round(x, 12), round(y, 12)) for x, y, _ in route[:4]}
    half = c.frame_size_m / 2
    for p in res.upcoming:
        assert (round(p.x_m, 12), round(p.y_m, 12)) not in done
        gap = math.hypot(max(abs(p.x_m) - half, 0.0), max(abs(p.y_m) - half, 0.0))
        assert gap > 200e-9


def test_upcoming_is_short_when_the_surface_is_nearly_spent():
    """Fewer than asked for is the honest answer, not something to pad."""
    c = cfg(strategy="center_first", piezo_half_range_m=160e-9,
            frame_size_m=100e-9)
    res = analyze_map([], c, upcoming_count=8)
    assert 0 < len(res.upcoming) < 8


def test_pick_next_positions_is_exported():
    """It is the batch API other modules import; leaving it out of ``__all__``
    made ``from ... import *`` silently miss it."""
    import mast.io.map_analysis as ma
    assert "pick_next_positions" in ma.__all__


# ── nearest_clean_from：点动作的就近换位 ────────────────────────────────────
#
# 与 pick_next_position 是两个问题。那个走的是从原点出发的固定路线（第 N 张图
# 该放哪）；这个回答的是「我刚在这儿打了一发，最近的还能打的地方在哪」——打完
# 一发就把针尖甩回压电中心，等于每次都重新激起一遍蠕变。

def _near(markers_rows, x_nm, y_nm, *, spot_nm=30.0, count=8, c=None, **kw):
    from mast.io.map_analysis import nearest_clean_from
    conf = c or cfg()
    return nearest_clean_from(markers_from_rows(markers_rows), conf,
                              x_nm * 1e-9, y_nm * 1e-9,
                              spot_r_m=spot_nm * 1e-9, count=count, **kw)


def test_nearest_returns_spots_in_distance_order():
    got = _near([], 0, 0)
    assert got, "空白表面上必须给得出候选点"
    d = [s.distance_m for s in got]
    assert d == sorted(d)


def test_current_spot_is_offered_until_the_caller_says_it_is_spent():
    """站着的地方若还干净，它就是最近的落点 —— 第一发不必先白走一趟。

    「打一次换一个地方」由调用方把打过的点放进 ``exclude`` 表达，而不是让函数
    假设当前位置一定脏：那样每一轮的第一发都要多一次无谓的移动和一次蠕变。"""
    assert _near([], 0, 0)[0].distance_m == 0.0
    after = _near([], 0, 0, exclude=[(0.0, 0.0)])
    assert after and after[0].distance_m > 0.0


def test_nearest_is_actually_near():
    """最近的候选应该在一个作用直径的量级上，而不是几百 nm 外。"""
    got = _near([], 400, -250, spot_nm=30.0)
    assert got[0].distance_m <= 4.0 * 30e-9


def test_nearest_avoids_a_previous_pulse():
    """脉冲避让圆（默认 150 nm）罩住的地方一个都不能给。"""
    c = cfg(pulse_r_m=150e-9)
    got = _near([row("pulse", 0, 0)], 0, 0, spot_nm=30.0, c=c, count=12)
    for s in got:
        assert math.hypot(s.x_m, s.y_m) > 150e-9 + 30e-9 - 1e-15


def test_nearest_avoids_this_run_s_own_spots():
    """本轮刚用过的点由调用方传进来 —— marker 可能还没落库，读到旧的就会两发打同一处。"""
    used = [(0.0, 0.0), (60e-9, 0.0)]
    got = _near([], 0, 0, spot_nm=30.0, exclude=used, count=12)
    for s in got:
        for px, py in used:
            assert math.hypot(s.x_m - px, s.y_m - py) >= 2 * 30e-9 - 1e-15


def test_nearest_keeps_the_whole_spot_inside_the_piezo_range():
    c = cfg(piezo_half_range_m=200e-9)
    got = _near([], 190, 190, spot_nm=30.0, c=c, count=12)
    for s in got:
        assert abs(s.x_m) <= 200e-9 - 30e-9 + 1e-15
        assert abs(s.y_m) <= 200e-9 - 30e-9 + 1e-15


def test_nearest_returns_empty_when_everything_nearby_is_ruined():
    """没地方就是没地方 —— 返回空表，让调用方去建议粗动，而不是硬塞一个坏点。"""
    c = cfg(piezo_half_range_m=120e-9, crash_r_m=400e-9)
    assert _near([row("crash", 0, 0)], 0, 0, spot_nm=30.0, c=c) == []


def test_scanned_area_is_not_a_reason_to_skip():
    """已扫 ≠ 不可用：只有破坏才消耗表面（模块的核心语义之一）。"""
    scanned = [row("scan", 0, 0, 500, 500)]
    assert _near(scanned, 0, 0, spot_nm=30.0)


def test_max_distance_caps_the_search():
    got = _near([], 0, 0, spot_nm=30.0, count=32, max_distance_m=100e-9)
    assert got and all(s.distance_m <= 100e-9 for s in got)


def test_nearest_is_stateless_across_calls():
    """同样的历史 + 同样的位置 → 同样的答案。断点续走不需要存游标。"""
    rows = [row("pulse", 120, 0), row("tip_shape", -80, 40)]
    assert [(s.x_m, s.y_m) for s in _near(rows, 0, 0)] == \
           [(s.x_m, s.y_m) for s in _near(rows, 0, 0)]


def test_nearest_clean_from_is_exported():
    import mast.io.map_analysis as ma
    assert "nearest_clean_from" in ma.__all__ and "CleanSpot" in ma.__all__


# ── 更分散的计划扫描范围 ──────────────────────────────────────


def test_spacing_factor_actually_disperses_the_candidates():
    """这个旋钮一直存在，只是从来没人设过它（map_scope 不传 → 永远 1.2）。

    1.2 的意思是「相邻帧留一条缝、不重叠」，也就是**尽可能挨着排** ——
    正是要求的不够分散。这条钉住调大它真的会把候选点拉开，
    否则「已暴露」只是暴露了一个不起作用的字段。
    """
    tight = MA.candidate_positions(MA.AnalysisConfig(
        piezo_half_range_m=1.5e-6, frame_size_m=100e-9, point_spacing_factor=1.2))
    spread = MA.candidate_positions(MA.AnalysisConfig(
        piezo_half_range_m=1.5e-6, frame_size_m=100e-9, point_spacing_factor=3.0))

    def nearest_gap(pts):
        # 相邻候选之间的最小间距 —— 「分散」就是这个数变大。
        best = float("inf")
        for i in range(1, min(len(pts), 40)):
            ax, ay = pts[i][0], pts[i][1]   # (x, y, ring)
            for j in range(i):
                bx, by = pts[j][0], pts[j][1]
                d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                best = min(best, d)
        return best

    assert nearest_gap(spread) > nearest_gap(tight) * 1.8
    # 两种策略都受它影响 —— 它是「分散程度」这一个旋钮，不是第三种模式。
    per_tight = MA.candidate_positions(MA.AnalysisConfig(
        piezo_half_range_m=1.5e-6, frame_size_m=100e-9,
        strategy="perimeter_inward", point_spacing_factor=1.2, ring_width_factor=1.2))
    per_spread = MA.candidate_positions(MA.AnalysisConfig(
        piezo_half_range_m=1.5e-6, frame_size_m=100e-9,
        strategy="perimeter_inward", point_spacing_factor=3.0, ring_width_factor=3.0))
    assert nearest_gap(per_spread) > nearest_gap(per_tight) * 1.5


def test_spacing_only_along_the_ring_does_not_disperse_anything():
    """这就是 map_scope 为什么把两个因子绑在同一个旋钮上。

    只拉开环**上**的点距而不拉开环**间**距，最近邻距离由环间距决定，一点没变 ——
    「更分散」于是变成一句看起来做了事、实际什么都没做的配置。
    实测：只调 point_spacing_factor 时最小间距 1.21e-7 → 1.20e-7（反而略小）。
    """
    def nearest_gap(pts):
        best = float("inf")
        for i in range(1, min(len(pts), 40)):
            ax, ay = pts[i][0], pts[i][1]
            for j in range(i):
                bx, by = pts[j][0], pts[j][1]
                best = min(best, ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
        return best

    base = dict(piezo_half_range_m=1.5e-6, frame_size_m=100e-9,
                strategy="perimeter_inward")
    only_points = MA.candidate_positions(MA.AnalysisConfig(
        **base, point_spacing_factor=3.0, ring_width_factor=1.2))
    both = MA.candidate_positions(MA.AnalysisConfig(
        **base, point_spacing_factor=3.0, ring_width_factor=3.0))
    assert nearest_gap(both) > nearest_gap(only_points) * 1.8


def test_the_profile_key_reaches_the_analysis_config():
    """两侧对账：设置里那个键必须真的落到规划器上。

    只加 profile 字段而不在 map_scope 里传，表现是**静默无效** ——
    用户把它调到 3，计划出来的点一个都没动。
    """
    from mast.core import instrument_profile as ip
    from mast.core.map_scope import analysis_config

    before = dict(ip.get_profile())
    try:
        ip.set_profile({**before, "scan_spacing_factor": 4.0})
        cfg = analysis_config()
        assert cfg.point_spacing_factor == 4.0
        assert cfg.ring_width_factor == 4.0
    finally:
        ip.set_profile(before)


def test_a_sub_unit_spacing_is_refused_rather_than_honoured():
    """小于 1 会让相邻帧重叠 —— 那不是「更分散」的反面，是把同一块地方扫两遍。"""
    from mast.core import instrument_profile as ip
    from mast.core.map_scope import analysis_config

    before = dict(ip.get_profile())
    try:
        ip.set_profile({**before, "scan_spacing_factor": 0.3})
        assert analysis_config().point_spacing_factor >= 1.0
    finally:
        ip.set_profile(before)
