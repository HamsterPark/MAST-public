"""跨畴界线谱几何 —— S4 STS 设计 §5.5。纯函数,无 mock、无硬件、无文件。

这一组测试守的是三件不同的事:

* **算得对**:轴归一化、间距、点数、``|s|`` 升序。
* **算不出来时拒绝**:零长度轴不除零、跨站点没有米坐标就不布点。
* **名字说实话**:字段叫 ``axis_*`` 不叫 ``normal_*``。

其中最重要的一条不是「算得对」,是 :func:`test_fine_window_is_widened_to_uncertainty`
和它的变异验证:精细窗窄于定位不确定度时,密采点可能整片落在畴界同一侧 —— 一晚上
跑完、每条谱都合格、结论是空的,**没有任何单点判据会报警**。防线只有那个 ``max``。
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from mast.core.sts_line_plan import (
    CROSS_SITE_FORM,
    CROSS_SITE_HINT,
    LINE_STS,
    LinePlanRefused,
    LinePoint,
    MEASURED_FOLME_SPEED_M_S,
    STSLineSpec,
    estimate_line_duration,
    format_duration_note,
    looks_like_cross_site,
    plan_from_geometry,
    plan_line_across_wall,
    plan_line_detail,
    resolve_fine_half_width_m,
    resolve_line_geometry,
    resolve_line_spec,
)

NM = 1e-9


def spec(**kw) -> STSLineSpec:
    """一张够小、够好数的表:精细 2 nm / 粗 5 nm / 窗 ±3 nm / 半长 20 nm。"""
    base = dict(fine_spacing_nm=2.0, coarse_spacing_nm=5.0,
                fine_half_width_nm=3.0, line_half_length_nm=20.0,
                max_points=64)
    base.update(kw)
    return STSLineSpec(**base)


# ── 轴 ──────────────────────────────────────────────────────────────────────


def test_axis_is_normalised_length_does_not_change_the_points():
    """轴向量长度只定方向,不定步长。给 1 和给 1000 必须得到同一批点。"""
    a = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    b = plan_line_detail((0.0, 0.0), (1000.0, 0.0), 0.0, spec())
    assert [p.axis_s_m for p in a.points] == [p.axis_s_m for p in b.points]
    assert [p.x_m for p in a.points] == pytest.approx([p.x_m for p in b.points])
    assert a.axis_unit_x == pytest.approx(1.0)
    assert a.axis_unit_y == pytest.approx(0.0)


def test_points_lie_on_the_axis_through_the_origin():
    """点必须在过原点、沿轴的直线上 —— 45° 轴上两个分量各差 s/√2。"""
    ox, oy = 100e-9, -50e-9
    plan = plan_line_detail((ox, oy), (1.0, 1.0), 0.0, spec())
    for p in plan.points:
        assert p.x_m == pytest.approx(ox + p.axis_s_m / math.sqrt(2.0))
        assert p.y_m == pytest.approx(oy + p.axis_s_m / math.sqrt(2.0))
    assert plan.axis_angle_deg == pytest.approx(45.0)


def test_zero_length_axis_is_refused_not_divided_by_zero():
    """``p_hi == p_lo`` ⇒ 方向无从定义。**拒绝**,不是除零,也不是随手挑一个方向。"""
    with pytest.raises(LinePlanRefused) as exc:
        plan_line_detail((0.0, 0.0), (0.0, 0.0), 1e-9, spec())
    assert exc.value.code == "zero_length_axis"

    # 经 bracket 形态进来的同一件事(lo 与 hi 是同一个点)也要拒绝。
    geom = resolve_line_geometry("bracket", lo_xy=(1e-9, 2e-9), hi_xy=(1e-9, 2e-9))
    with pytest.raises(LinePlanRefused) as exc2:
        plan_from_geometry(geom, spec())
    assert exc2.value.code == "zero_length_axis"


def test_non_finite_inputs_are_refused():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(LinePlanRefused):
            plan_line_detail((0.0, 0.0), (bad, 0.0), 0.0, spec())
        with pytest.raises(LinePlanRefused):
            plan_line_detail((bad, 0.0), (1.0, 0.0), 0.0, spec())


# ── 命名钉子(D22 / 陷阱 23)────────────────────────────────────────────────


def test_fields_are_named_axis_not_normal():
    """一条很傻的测试,挡的是下游拿轴当法向去算「畴界宽度」的系统性误差。

    上游产出的是 bracket 轴(搜索线方向),与畴界真法向差一个未知夹角 θ;拿它算
    宽度会系统性偏大 1/cos θ,**而且不会有任何东西报警**。名字必须说实话。
    """
    def names_a_direction_normal(key: str) -> bool:
        """禁的是**拿 normal 当那个方向的名字**(``normal_x`` / ``normal_s_m`` …);
        ``axis_is_not_the_wall_normal`` 这种否认句反而是要有的。"""
        return key.startswith("normal") or "normal_" in key

    point_names = {f.name for f in fields(LinePoint)}
    assert "axis_s_m" in point_names
    assert not [n for n in point_names if names_a_direction_normal(n)]

    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    summary = plan.summary_dict()
    assert {"axis_unit_x", "axis_unit_y", "axis_angle_deg",
            "axis_length_m"} <= set(summary)
    assert not [k for k in summary if names_a_direction_normal(k)]
    # 并且明说一句:这不是法向。
    assert summary["axis_is_not_the_wall_normal"] is True
    for row in plan.positions():
        assert "axis_s_m" in row
        assert not [k for k in row if names_a_direction_normal(k)]


# ── 精细窗的 max(D22 / 陷阱 24)—— 本模块最贵的一条 ──────────────────────


def test_fine_window_is_widened_to_uncertainty():
    """``fine_half_width_nm=3`` + ``uncertainty_m=5 nm`` ⇒ 精细窗被撑到 5 nm。

    畴界在 bracket 内的位置未知,只知道它在 5 nm 之内。窗口只开 ±3 nm 时,密采的
    点可能整片落在畴界同一侧 —— 而这条线的全部价值就在于跨过它。
    """
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 5 * NM, spec(fine_half_width_nm=3.0))
    assert plan.fine_half_width_m == pytest.approx(5 * NM)
    assert plan.fine_half_width_source == "uncertainty"
    assert "fine_window_widened_by_uncertainty" in plan.warnings

    # 可观测后果(不是只看报出来的那个数):精细区必须真的铺到 ±4 nm。
    # 窗口若只有 3 nm,±4 nm 处不会有精细点。
    fine_s = sorted(p.axis_s_m for p in plan.points if p.zone == "fine")
    assert fine_s == pytest.approx([-4 * NM, -2 * NM, 0.0, 2 * NM, 4 * NM])


def test_spec_wins_when_it_is_wider_than_the_uncertainty():
    """反向:spec 比不确定度宽时用 spec,并且**不**挂「被撑大」的警告。"""
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 1 * NM, spec(fine_half_width_nm=6.0))
    assert plan.fine_half_width_m == pytest.approx(6 * NM)
    assert plan.fine_half_width_source == "spec"
    assert "fine_window_widened_by_uncertainty" not in plan.warnings


def test_mutation_dropping_the_max_makes_the_window_too_narrow():
    """变异验证:把 ``max`` 换成「直接取 spec 值」,上面那条必须红。

    先证明变异真的动了手(窗口确实变回 3 nm、±4 nm 的精细点确实消失),再证明
    原实现不是这样 —— 否则这条测试只是在自说自话。
    """
    sp = spec(fine_half_width_nm=3.0)
    unc = 5 * NM

    def mutated_fine_half(_spec, _unc):        # 变异体:忘了那个 max
        return float(_spec.fine_half_width_nm) * 1e-9

    assert mutated_fine_half(sp, unc) == pytest.approx(3 * NM)
    assert resolve_fine_half_width_m(sp, unc) == pytest.approx(5 * NM)

    # 变异体下密采区只到 ±2 nm ⇒ 一旦畴界落在 bracket 的 ±5 nm 里偏外的位置,
    # 整片密采点都在同一侧。用同样的布点规则复算一遍,钉住这个差别。
    def fine_positions(half_width_m: float) -> list[float]:
        step = sp.fine_spacing_nm * 1e-9
        k_max = int(math.floor((half_width_m + step * 1e-9) / step))
        return [k * step for k in range(-k_max, k_max + 1)]

    assert fine_positions(mutated_fine_half(sp, unc)) == pytest.approx(
        [-2 * NM, 0.0, 2 * NM])
    real = plan_line_detail((0.0, 0.0), (1.0, 0.0), unc, sp)
    assert sorted(p.axis_s_m for p in real.points if p.zone == "fine") == \
        pytest.approx(fine_positions(resolve_fine_half_width_m(sp, unc)))
    assert len([p for p in real.points if p.zone == "fine"]) == 5


def test_bracket_span_becomes_the_uncertainty():
    """bracket 形态下不确定度 = ``|p_hi − p_lo|``,由几何自己算,调用方不必记得。"""
    geom = resolve_line_geometry("bracket", lo_xy=(0.0, 0.0), hi_xy=(8 * NM, 0.0))
    assert geom.uncertainty_m == pytest.approx(8 * NM)
    assert geom.origin_x_m == pytest.approx(4 * NM)
    plan = plan_from_geometry(geom, spec(fine_half_width_nm=3.0))
    assert plan.fine_half_width_m == pytest.approx(8 * NM)
    assert plan.fine_half_width_source == "uncertainty"


def test_uncertainty_may_be_widened_but_never_shrunk():
    """显式的不确定度只允许**放大** bracket 跨度 —— 缩小等于声称比上游更确定。"""
    wide = resolve_line_geometry("bracket", lo_xy=(0.0, 0.0), hi_xy=(4 * NM, 0.0),
                                 uncertainty_m=9 * NM)
    assert wide.uncertainty_m == pytest.approx(9 * NM)
    narrow = resolve_line_geometry("bracket", lo_xy=(0.0, 0.0), hi_xy=(4 * NM, 0.0),
                                   uncertainty_m=1 * NM)
    assert narrow.uncertainty_m == pytest.approx(4 * NM)


# ── 疏密、点数、顺序 ────────────────────────────────────────────────────────


def test_dense_near_the_wall_sparse_far_from_it():
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    s = sorted(p.axis_s_m for p in plan.points)
    # 精细窗 ±3 nm、点距 2 nm ⇒ 0, ±2;粗区锚在窗边 3 nm、点距 5 nm ⇒ ±8, ±13, ±18
    assert s == pytest.approx([-18 * NM, -13 * NM, -8 * NM, -2 * NM, 0.0,
                               2 * NM, 8 * NM, 13 * NM, 18 * NM])
    zones = {round(p.axis_s_m * 1e9): p.zone for p in plan.points}
    assert zones[0] == "fine" and zones[2] == "fine"
    assert zones[8] == "coarse" and zones[18] == "coarse"


def test_acquisition_order_is_ascending_abs_s_and_alternates_sides():
    """D23:离墙最近的先测,让漂移损伤物理上最不重要的远端点。"""
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    by_order = sorted(plan.points, key=lambda p: p.order)
    assert [p.order for p in by_order] == list(range(len(by_order)))
    keys = [abs(p.axis_s_m) for p in by_order]
    assert keys == sorted(keys), "采集顺序必须是 |s| 升序"
    # 同一距离的两侧交替(0, −d, +d, −2d, +2d …),漂移不会全砸在一侧。
    assert [round(p.axis_s_m * 1e9) for p in by_order[:5]] == [0, -2, 2, -8, 8]


def test_returned_list_is_in_spatial_order_and_order_field_carries_the_schedule():
    """签名返回的表按空间排(画图用),采集顺序在 ``order`` 字段里。"""
    pts = plan_line_across_wall((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    assert [p.index for p in pts] == list(range(len(pts)))
    assert [p.axis_s_m for p in pts] == sorted(p.axis_s_m for p in pts)
    assert sorted(p.order for p in pts) == list(range(len(pts)))


def test_positions_helper_hands_the_engine_acquisition_order():
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    rows = plan.positions()
    assert [r["order"] for r in rows] == list(range(len(rows)))
    assert [abs(r["axis_s_m"]) for r in rows] == sorted(
        abs(r["axis_s_m"]) for r in rows)


def test_point_budget_drops_the_far_points_and_says_how_many():
    """超预算时丢**最远**的粗点(D23 同一条理由),并如实报出丢了几个。"""
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0,
                            spec(line_half_length_nm=100.0, max_points=9))
    assert plan.point_count == 9
    assert plan.dropped_point_count > 0
    assert "points_truncated_by_budget" in plan.warnings
    # 留下的必须是离墙最近的那些。
    assert max(abs(p.axis_s_m) for p in plan.points) < 100 * NM


def test_fine_zone_alone_over_budget_is_refused_not_silently_thinned():
    """精细区本身就超预算 ⇒ 拒绝。**不缩小精细窗** —— 那正是这条线的意义。"""
    with pytest.raises(LinePlanRefused) as exc:
        plan_line_detail((0.0, 0.0), (1.0, 0.0), 20 * NM,
                         spec(fine_spacing_nm=0.5, max_points=16))
    assert exc.value.code == "fine_zone_over_budget"
    assert "不要" in exc.value.message      # 明说别去缩窗


def test_line_shorter_than_the_fine_window_warns_instead_of_pretending():
    """线没伸出精细窗 ⇒ 两端可能都还在不确定度带里,挂警告说出来。"""
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 30 * NM,
                            spec(fine_spacing_nm=5.0, line_half_length_nm=10.0))
    assert "line_does_not_extend_past_uncertainty" in plan.warnings
    assert all(p.zone == "fine" for p in plan.points)


def test_coarse_spacing_below_fine_spacing_is_refused():
    """「近墙密远墙疏」反过来时不悄悄纠正 —— 调用方会以为跑的是自己配的那套。"""
    with pytest.raises(LinePlanRefused) as exc:
        plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0,
                         spec(fine_spacing_nm=5.0, coarse_spacing_nm=1.0))
    assert exc.value.code == "bad_spec"


# ── 输入形态闭集与跨站点拒绝(D25 / 陷阱 25)──────────────────────────────


def test_cross_site_input_is_refused_and_says_what_to_do():
    """只有粗动步数、没有米坐标 ⇒ 拒绝,并说得出「收进单个站点」。"""
    with pytest.raises(LinePlanRefused) as exc:
        resolve_line_geometry(CROSS_SITE_FORM)
    assert exc.value.code == "cross_site_no_metres"
    assert CROSS_SITE_HINT in exc.value.message
    assert "收进单个站点" in exc.value.message


def test_steps_only_markers_are_recognised_as_cross_site():
    assert looks_like_cross_site({"uncertainty_steps": 3}, x_m=1e-9, y_m=0.0)
    assert looks_like_cross_site({"role": "cross_site"}, x_m=1e-9, y_m=0.0)
    # 没有米坐标本身就够了 —— 宁可多问一句,也不拿不存在的坐标去布点。
    assert looks_like_cross_site({"bracket_id": "b1"}, x_m=None, y_m=0.0)
    assert not looks_like_cross_site({"bracket_id": "b1"}, x_m=1e-9, y_m=0.0)


def test_unknown_input_form_is_refused():
    with pytest.raises(LinePlanRefused) as exc:
        resolve_line_geometry("whatever", origin_xy=(0.0, 0.0), axis_deg=0.0,
                              uncertainty_m=0.0)
    assert exc.value.code == "unknown_input_form"


def test_explicit_geometry_must_state_its_uncertainty():
    """留空不许当 0:那会让精细窗退回 spec 值,正好踩中陷阱 24。"""
    with pytest.raises(LinePlanRefused) as exc:
        resolve_line_geometry("explicit", origin_xy=(0.0, 0.0), axis_deg=0.0)
    assert exc.value.code == "missing_uncertainty"
    # 显式写 0 是一句话,允许。
    geom = resolve_line_geometry("explicit", origin_xy=(0.0, 0.0), axis_deg=0.0,
                                 uncertainty_m=0.0)
    assert geom.uncertainty_m == 0.0


def test_mixed_frame_without_an_axis_is_refused_not_guessed():
    """mixed 帧没有 bracket 也没给角度 ⇒ 拒绝。**不拿帧角顶替**。"""
    with pytest.raises(LinePlanRefused) as exc:
        resolve_line_geometry("mixed_frame", center_xy=(0.0, 0.0),
                              frame_size_m=5 * NM)
    assert exc.value.code == "missing_axis"


def test_mixed_frame_uses_the_frame_size_as_its_uncertainty():
    geom = resolve_line_geometry("mixed_frame", center_xy=(1e-9, 2e-9),
                                 frame_size_m=5 * NM, axis_deg=90.0)
    assert geom.uncertainty_m == pytest.approx(5 * NM)
    assert geom.axis_x == pytest.approx(0.0, abs=1e-12)
    assert geom.axis_y == pytest.approx(1.0)
    plan = plan_from_geometry(geom, spec(fine_half_width_nm=1.0))
    assert plan.form == "mixed_frame"
    assert plan.fine_half_width_m == pytest.approx(5 * NM)


def test_mixed_frame_without_a_frame_size_is_refused():
    with pytest.raises(LinePlanRefused) as exc:
        resolve_line_geometry("mixed_frame", center_xy=(0.0, 0.0), axis_deg=0.0)
    assert exc.value.code == "missing_uncertainty"


# ── 流程表 ──────────────────────────────────────────────────────────────────


def test_spec_overrides_reject_out_of_range_instead_of_clamping():
    sp = resolve_line_spec({"fine_spacing_nm": 1e6, "coarse_spacing_nm": 7.0})
    assert sp.fine_spacing_nm == LINE_STS.fine_spacing_nm     # 越界的被丢掉
    assert sp.coarse_spacing_nm == 7.0
    assert resolve_line_spec({"fine_spacing_nm": None}).fine_spacing_nm == \
        LINE_STS.fine_spacing_nm
    assert resolve_line_spec(None) is LINE_STS


def test_spec_is_frozen():
    with pytest.raises(Exception):
        LINE_STS.fine_spacing_nm = 3.0     # type: ignore[misc]


# ── 时间账(D24 / 陷阱 16)──────────────────────────────────────────────────


def test_duration_uses_an_unconfigured_nominal_speed_and_discloses_it():
    """公开版名义速度只参与预算，并在有/无采谱耗时两种摘要中说明局限。"""
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    budget = estimate_line_duration(plan.points, acquire_s_per_point=20.0)
    assert budget["speed_m_s"] == MEASURED_FOLME_SPEED_M_S == 5e-9
    assert budget["speed_source"] == "nominal_unconfigured"
    assert budget["n_points"] == plan.point_count
    assert budget["total_s"] > budget["move_total_s"] > 0.0
    for report in (budget, estimate_line_duration(plan.points)):
        note = format_duration_note(report)
        assert "未标定名义值" in note and "目标仪器核验" in note
    assert "已知部分 ≥" not in format_duration_note(estimate_line_duration(plan.points))


def test_explicit_speed_overrides_the_nominal_budget_without_changing_move_overhead():
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    slow = estimate_line_duration(plan.points, acquire_s_per_point=20.0, speed_m_s=2e-9)
    fast = estimate_line_duration(plan.points, acquire_s_per_point=20.0, speed_m_s=4e-9)
    for report in (slow, fast):
        assert report["speed_source"] == "caller"
        assert "名义值" not in format_duration_note(report)
    assert slow["speed_m_s"] == 2e-9 and fast["speed_m_s"] == 4e-9
    assert slow["move_total_s"] - fast["move_total_s"] == pytest.approx(
        slow["travel_m"] * (1 / 2e-9 - 1 / 4e-9))


def test_unknown_acquisition_time_is_not_quietly_zero():
    """采谱时间未标定 ⇒ ``total_s`` 是 ``None`` 并点名,不是一个偏小的数。"""
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0, spec())
    budget = estimate_line_duration(plan.points)
    assert budget["total_s"] is None
    assert budget["unknown"] == ["acquire_s_per_point"]
    assert budget["known_lower_bound_s"] > 0.0
    note = format_duration_note(budget)
    assert "未知" in note


def test_nearest_first_order_costs_travel_and_the_budget_says_so():
    """D23 的顺序买的是抗漂移,卖的是路程。两个数都报出来,别让人以为免费。"""
    plan = plan_line_detail((0.0, 0.0), (1.0, 0.0), 0.0,
                            spec(line_half_length_nm=50.0))
    budget = estimate_line_duration(plan.points)
    assert budget["travel_m"] > budget["travel_m_if_spatial_order"]
    assert budget["move_total_s"] > budget["move_total_s_if_spatial_order"]


def test_start_leg_is_counted_only_when_the_start_is_known():
    plan = plan_line_detail((100 * NM, 0.0), (1.0, 0.0), 0.0, spec())
    without = estimate_line_duration(plan.points)
    with_start = estimate_line_duration(plan.points, start_xy=(0.0, 0.0))
    assert without["start_leg_counted"] is False
    assert with_start["start_leg_counted"] is True
    assert with_start["travel_m"] > without["travel_m"]
    assert with_start["n_moves"] == without["n_moves"] + 1
