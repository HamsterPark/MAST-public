"""多图规划器(mast.core.scan_planner)。

要求的第三个场景:「如果 llm 一下子要扫若干张图片呢?这时候就要有规划脚本
skill 了」。规划器的价值一半在**它拒绝的时候** —— 算不出来就给带 code 的拒绝 +
可行替代,绝不把「不确定」吞成一份看起来正常的计划。
"""

from __future__ import annotations

import math

import pytest

from mast.core import scan_policy
from mast.core.scan_planner import (
    BIG_MOVE_FRAMES,
    MAX_FRAMES,
    PlanReject,
    ScanPlan,
    expand_series,
    order_by_travel,
    order_series_monotonic,
    plan_batch,
)


@pytest.fixture(autouse=True)
def _clean_policy():
    scan_policy.set_policy(None)
    yield
    scan_policy.set_policy(None)


def _positions(n, step=2e-7):
    return [(i * step, 0.0) for i in range(n)]


# ── 展开 series:绝不发明数值 ────────────────────────────────────────────────

def test_expand_explicit_values():
    assert expand_series({"values": [-1.0, 0.5, 2.0]}) == [-1.0, 0.5, 2.0]


def test_expand_linear_range():
    got = expand_series({"start": 0.0, "stop": 1.0, "n": 5})
    assert got == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_expand_log_range():
    got = expand_series({"start": 1e-11, "stop": 1e-9, "n": 3, "spacing": "log"})
    assert got == pytest.approx([1e-11, 1e-10, 1e-9], rel=1e-9)


def test_expand_refuses_incomplete_specs():
    """不发明序列 —— 这是用户要测的物理量,不是我们能猜的。"""
    for bad in ({}, {"start": 0.0}, {"start": 0.0, "stop": 1.0},
                {"values": []}, {"values": ["a"]}, None, "1,2,3"):
        assert expand_series(bad) is None


def test_expand_refuses_nonpositive_log():
    assert expand_series({"start": -1.0, "stop": 1.0, "n": 3,
                          "spacing": "log"}) is None


def test_expand_single_point():
    assert expand_series({"start": 2.0, "stop": 9.0, "n": 1}) == [2.0]


# ── 顺序优化 ─────────────────────────────────────────────────────────────────

def test_travel_order_is_nearest_neighbour():
    pts = [(0.0, 10.0), (0.0, 1.0), (0.0, 5.0)]
    assert order_by_travel(pts, (0.0, 0.0)) == [1, 2, 0]


def test_travel_order_visits_everything_once():
    pts = [(i * 1e-7, (i % 3) * 1e-7) for i in range(9)]
    order = order_by_travel(pts, (0.0, 0.0))
    assert sorted(order) == list(range(9))


def test_series_order_starts_from_the_nearer_end():
    """来回跳会反复激起结的回滞与充放电瞬态 —— 从最近的一端单调走。"""
    vals = [-1.0, -0.5, 0.5, 1.0]
    assert [vals[i] for i in order_series_monotonic(vals, current=0.9)] == \
        [1.0, 0.5, -0.5, -1.0]
    assert [vals[i] for i in order_series_monotonic(vals, current=-0.9)] == \
        [-1.0, -0.5, 0.5, 1.0]


def test_series_order_without_a_current_value_is_ascending():
    vals = [0.5, -1.0, 1.0]
    assert [vals[i] for i in order_series_monotonic(vals, None)] == [-1.0, 0.5, 1.0]


# ── survey ───────────────────────────────────────────────────────────────────

def test_survey_lays_out_the_requested_number_of_frames():
    plan = plan_batch({"kind": "survey", "n_images": 3, "size_m": 1e-7},
                      survey_positions=_positions(5))
    assert isinstance(plan, ScanPlan)
    assert len(plan.frames) == 3
    assert all(f.size_m == 1e-7 for f in plan.frames)


def test_survey_resolves_parameters_per_frame_from_the_tier_table():
    plan = plan_batch({"kind": "survey", "n_images": 2, "size_m": 1e-6},
                      survey_positions=_positions(4))
    for f in plan.frames:
        assert f.resolved.tier_name == "survey"
        assert f.resolved.configure_scan["line_time_s"] == 0.5
        assert f.resolved.set_scan_buffer["pixels"] == 256


def test_survey_with_no_positions_is_a_typed_rejection():
    rej = plan_batch({"kind": "survey", "n_images": 3, "size_m": 1e-7},
                     survey_positions=[])
    assert isinstance(rej, PlanReject)
    assert rej.code == "no_uncovered_area"
    assert rej.alternatives


def test_survey_with_too_few_positions_reports_the_gap_and_offers_partial():
    """少给的那几张不能悄悄吞掉 —— 用户要知道缺口。"""
    rej = plan_batch({"kind": "survey", "n_images": 5, "size_m": 1e-7},
                     survey_positions=_positions(2))
    assert isinstance(rej, PlanReject)
    assert rej.code == "candidates_exhausted"
    assert len(rej.partial) == 2
    assert "2" in rej.detail and "5" in rej.detail


def test_survey_orders_frames_to_reduce_travel():
    far, near = (1e-6, 0.0), (1e-7, 0.0)
    plan = plan_batch({"kind": "survey", "n_images": 2, "size_m": 1e-7},
                      tip_xy=(0.0, 0.0), survey_positions=[far, near])
    assert plan.frames[0].center_x_m == pytest.approx(1e-7)


def test_survey_needs_size():
    rej = plan_batch({"kind": "survey", "n_images": 2},
                     survey_positions=_positions(3))
    assert isinstance(rej, PlanReject) and rej.code == "bad_intent"


# ── zoomin ───────────────────────────────────────────────────────────────────

def _cands(n):
    return [{"x_m": i * 3e-8, "y_m": 0.0, "feature": "defect",
             "reason": "分割候选"} for i in range(n)]


def test_zoomin_plans_from_the_supplied_candidates():
    plan = plan_batch(
        {"kind": "zoomin", "n_images": 2,
         "target": {"feature": "defect", "final_size_m": 1e-8}},
        zoom_candidates=_cands(4))
    assert isinstance(plan, ScanPlan)
    assert len(plan.frames) == 2
    assert all(f.size_m == 1e-8 for f in plan.frames)
    assert all(f.resolved.tier_name == "atomic" for f in plan.frames)


def test_zoomin_without_candidates_is_a_typed_rejection():
    rej = plan_batch(
        {"kind": "zoomin", "target": {"feature": "defect", "final_size_m": 1e-8}},
        zoom_candidates=[])
    assert isinstance(rej, PlanReject)
    assert rej.code == "candidates_exhausted"


def test_zoomin_needs_a_final_size():
    rej = plan_batch({"kind": "zoomin", "target": {"feature": "defect"}},
                     zoom_candidates=_cands(2))
    assert isinstance(rej, PlanReject) and rej.code == "bad_intent"


# ── 特征词表:诚实拒绝超出分割语义的请求 ─────────────────────────────────────

def test_a_specific_defect_type_is_honestly_refused():
    """分割给的是几何/异常语义 —— DEFECT = 「异常凸凹」,不等于任何特定缺陷类型。

    假装能做才是真正的坏结果:会扫回一堆看着像但其实不是的东西。
    """
    rej = plan_batch(
        {"kind": "zoomin",
         "target": {"feature": "topological_defect", "final_size_m": 1e-8}},
        zoom_candidates=_cands(3))
    assert isinstance(rej, PlanReject)
    assert rej.code == "needs_semantic_vision"
    assert any("user_marked" in a for a in rej.alternatives)


@pytest.mark.parametrize("feature", ["terrace", "step_edge", "defect",
                                     "contamination", "user_marked"])
def test_the_four_segmentation_classes_plus_user_marked_are_accepted(feature):
    plan = plan_batch(
        {"kind": "zoomin", "target": {"feature": feature, "final_size_m": 1e-8}},
        zoom_candidates=_cands(2))
    assert isinstance(plan, ScanPlan)


# ── bias_series ──────────────────────────────────────────────────────────────

def test_bias_series_expands_and_orders_monotonically():
    plan = plan_batch({
        "kind": "bias_series", "size_m": 1e-7,
        "series": {"param": "bias_v", "values": [-1.0, -0.5, 0.5, 1.0]},
        "overrides": {"bias_v": 0.9},
    })
    assert isinstance(plan, ScanPlan)
    biases = [f.resolved.set_bias["bias_v"] for f in plan.frames]
    assert biases == [1.0, 0.5, -0.5, -1.0]


def test_bias_series_without_values_refuses_to_invent_them():
    rej = plan_batch({"kind": "bias_series", "size_m": 1e-7,
                      "series": {"param": "bias_v"}})
    assert isinstance(rej, PlanReject)
    assert rej.code == "series_without_values"
    assert any("values" in a for a in rej.alternatives)


def test_bias_series_frames_share_one_position():
    plan = plan_batch({
        "kind": "bias_series", "size_m": 1e-7,
        "target": {"coords": {"x_m": 3e-7, "y_m": -1e-7}},
        "series": {"param": "bias_v", "values": [1.0, 2.0]},
    })
    assert {(f.center_x_m, f.center_y_m) for f in plan.frames} == {(3e-7, -1e-7)}


def test_param_series_can_vary_the_size_and_each_frame_gets_its_own_tier():
    """变尺寸时每帧都该按**自己的**尺寸查档 —— 这正是「一个速度走天下」的反面。"""
    plan = plan_batch({
        "kind": "param_series", "size_m": 1e-7,
        "series": {"param": "size_m", "values": [1e-6, 1e-7, 5e-9]},
    })
    tiers = [f.resolved.tier_name for f in plan.frames]
    # 2026-08-14:5 nm 这一帧改落 `atomic_verify`(新增的 512 px / 0.30 s 验收档
    # —— 旧 `atomic` 档的 256 px 在 5.12 nm 以上过不了原子判据的尺度门)。
    # 本用例钉的是「每帧按**自己的**尺寸查档」,那件事没变。
    assert tiers == ["atomic_verify", "highres", "survey"]     # 升序单调


def test_series_param_must_be_from_the_closed_set():
    rej = plan_batch({"kind": "param_series", "size_m": 1e-7,
                      "series": {"param": "temperature", "values": [1, 2]}})
    assert isinstance(rej, PlanReject) and rej.code == "bad_intent"


def test_series_longer_than_the_batch_cap_is_refused():
    rej = plan_batch({
        "kind": "bias_series", "size_m": 1e-7,
        "series": {"param": "bias_v", "start": 0.0, "stop": 1.0,
                   "n": MAX_FRAMES + 1}})
    assert isinstance(rej, PlanReject)


# ── repeat ───────────────────────────────────────────────────────────────────

def test_repeat_plans_the_same_footprint_n_times():
    plan = plan_batch({"kind": "repeat", "n_images": 3, "size_m": 1e-7,
                       "target": {"coords": {"x_m": 1e-7, "y_m": 2e-7}}})
    assert len(plan.frames) == 3
    assert {(f.center_x_m, f.center_y_m) for f in plan.frames} == {(1e-7, 2e-7)}


# ── regions_explicit ─────────────────────────────────────────────────────────

def test_explicit_regions_are_ordered_and_resolved():
    plan = plan_batch({"kind": "regions_explicit", "regions": [
        {"center_x_m": 1e-6, "center_y_m": 0.0, "size_m": 1e-6, "label": "far"},
        {"center_x_m": 1e-7, "center_y_m": 0.0, "size_m": 5e-9, "label": "near"},
    ]})
    assert [f.label for f in plan.frames] == ["near", "far"]
    # 2026-08-14:5 nm ⇒ `atomic_verify`(见上一条注释)。
    assert plan.frames[0].resolved.tier_name == "atomic_verify"
    assert plan.frames[1].resolved.tier_name == "survey"


def test_malformed_region_is_refused():
    rej = plan_batch({"kind": "regions_explicit",
                      "regions": [{"center_x_m": 0.0}]})
    assert isinstance(rej, PlanReject) and rej.code == "bad_intent"


# ── 大跳变与参数变更标注 ─────────────────────────────────────────────────────

def test_a_big_jump_is_flagged_for_settling():
    """每次大位移都会重新激起压电蠕变,让接下来一两帧带上漂移。"""
    plan = plan_batch({"kind": "regions_explicit", "regions": [
        {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 1e-8},
        {"center_x_m": 1e-6, "center_y_m": 0.0, "size_m": 1e-8},
    ]}, tip_xy=(0.0, 0.0))
    assert plan.frames[0].needs_settle is False
    assert plan.frames[1].needs_settle is True
    assert plan.frames[1].move_distance_m > BIG_MOVE_FRAMES * 1e-8


def test_small_moves_are_not_flagged():
    plan = plan_batch({"kind": "regions_explicit", "regions": [
        {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 1e-6},
        {"center_x_m": 1e-6, "center_y_m": 0.0, "size_m": 1e-6},
    ]}, tip_xy=(0.0, 0.0))
    assert all(f.needs_settle is False for f in plan.frames)


def test_changed_params_lists_only_what_actually_differs():
    """下发时只动变化了的项 —— 每次重设都是一次可以不发生的瞬态。"""
    plan = plan_batch({
        "kind": "bias_series", "size_m": 1e-7,
        "series": {"param": "bias_v", "values": [1.0, 2.0]},
    })
    assert "bias_v" in plan.frames[1].changed_params
    assert "line_time_s" not in plan.frames[1].changed_params


# ── 预算 ─────────────────────────────────────────────────────────────────────

def test_a_plan_that_does_not_fit_the_time_budget_is_refused_with_a_smaller_count():
    rej = plan_batch({
        "kind": "survey", "n_images": 4, "size_m": 1e-6,
        "constraints": {"max_total_minutes": 5},
    }, survey_positions=_positions(6))
    assert isinstance(rej, PlanReject)
    assert rej.code == "budget_exceeded"
    assert rej.partial and any("张" in a for a in rej.alternatives)


def test_a_plan_within_budget_is_accepted():
    plan = plan_batch({
        "kind": "survey", "n_images": 2, "size_m": 1e-6,
        "constraints": {"max_total_minutes": 60},
    }, survey_positions=_positions(4))
    assert isinstance(plan, ScanPlan)
    assert plan.total_estimated_s == pytest.approx(2 * 256 * 0.5 * 2)


# ── 序列化 ───────────────────────────────────────────────────────────────────

def test_plan_serialises_everything_the_operator_needs():
    plan = plan_batch({"kind": "survey", "n_images": 2, "size_m": 1e-7},
                      survey_positions=_positions(3))
    d = plan.as_dict()
    assert d["n_frames"] == 2
    assert d["total_estimated_min"] > 0
    f0 = d["frames"][0]
    for key in ("center_x_m", "size_m", "tier", "line_time_s", "pixels",
                "estimated_s", "needs_settle"):
        assert key in f0


def test_frames_expose_a_map_footprint():
    plan = plan_batch({"kind": "survey", "n_images": 1, "size_m": 1e-7},
                      survey_positions=_positions(1))
    fp = plan.frames[0].footprint()
    assert fp["kind"] == "scan"
    assert fp["w_m"] == 1e-7 and fp["h_m"] == 1e-7


def test_reject_serialises_its_code_and_alternatives():
    rej = plan_batch({"kind": "survey", "n_images": 1, "size_m": 1e-7},
                     survey_positions=[])
    d = rej.as_dict()
    assert d["code"] == "no_uncovered_area"
    assert isinstance(d["alternatives"], list)


# ── 确定性 ───────────────────────────────────────────────────────────────────

def test_the_same_intent_gives_the_same_plan():
    """同一个请求两次要给同样的计划 —— 模型不保证这一点,脚本必须保证。"""
    intent = {"kind": "survey", "n_images": 3, "size_m": 1e-7}
    a = plan_batch(intent, survey_positions=_positions(5))
    b = plan_batch(intent, survey_positions=_positions(5))
    assert [(f.center_x_m, f.center_y_m, f.size_m) for f in a.frames] == \
        [(f.center_x_m, f.center_y_m, f.size_m) for f in b.frames]
