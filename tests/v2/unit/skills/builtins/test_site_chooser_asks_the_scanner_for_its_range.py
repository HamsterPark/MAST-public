"""选点范围必须由配置和仪器回读共同约束。

通过独立合成范围验证：回读较小时收紧，较大时仍遵守配置；未知值明确
标注回退。选点、中心区和边界守卫采用一致的几何定义。
"""
from __future__ import annotations

import pytest


class _Rec:
    """Nanonis 的**真实线格式**:``('', b'…', [值...])`` —— 数值在第三个元素里面。

    第一版夹具返回 ``(x, y)``(顶层就是数字),于是那个只扫顶层的解码在测试里
    是绿的、在真机上一个数都取不到。夹具的形状必须照真的写,否则它会替被测代码
    圆谎。"""

    def __init__(self, value=(), error=""):
        self.return_value = ("", b"", list(value)) if value else ("", b"", [])
        self.error = error
        self.method = ""
        self.args = ()


class _Ctx:
    """只回答 Piezo_RangeGet 和 FolMe_XYPosGet 的假上下文。"""

    def __init__(self, *, tip=(0.0, 0.0), piezo_full_m=None, piezo_error=""):
        self.tip = tip
        self.piezo_full_m = piezo_full_m
        self.piezo_error = piezo_error
        self.state = None
        self.asked: list = []

    def safe_call(self, method, *args, role="main"):
        self.asked.append(method)
        if method == "Piezo_RangeGet":
            if self.piezo_error:
                return _Rec(error=self.piezo_error)
            f = self.piezo_full_m
            return _Rec(value=[f, f, 4e-7])
        if method == "FolMe_XYPosGet":
            return _Rec(value=[self.tip[0], self.tip[1]])
        return _Rec()


def _run(ctx, **params):
    from mast.skills.builtins.clean_spot import FindCleanSpot

    return FindCleanSpot().execute(ctx, dict(params))


def test_it_actually_asks_the_scanner():
    """光有代码不算 —— 得真的发出那一问。"""
    ctx = _Ctx(piezo_full_m=2.4e-6)
    _run(ctx, purpose="tip_shape")
    assert "Piezo_RangeGet" in ctx.asked, "选点器没有向仪器问压电范围"


def test_a_smaller_real_range_wins_over_the_config():
    """仪器说的比配置小 ⇒ **收紧**,并且在结果里说是谁说的。"""
    ctx = _Ctx(piezo_full_m=2.4e-6)      # 合成半程 1200 nm
    res = _run(ctx, purpose="tip_shape")
    assert res.success, res.error
    half = res.data["piezo_half_range_m"]
    assert half == pytest.approx(1.2e-6, rel=1e-6), (
        f"用的还是配置里的 1.5 µm?拿到 {half * 1e9:.1f} nm")
    assert "instrument" in res.data["piezo_range_source"]


def test_no_candidate_may_sit_beyond_the_real_limit():
    """所有候选坐标均须位于合成回读范围内。"""
    SYNTHETIC_HALF = 1.2e-6
    ctx = _Ctx(tip=(-5e-7, 1e-6), piezo_full_m=2.0 * SYNTHETIC_HALF)
    res = _run(ctx, purpose="tip_shape", count=24)
    for s in (res.data.get("candidates") or []):
        assert abs(s["x_m"]) <= SYNTHETIC_HALF, f"候选 x={s['x_m'] * 1e9:.1f} nm 越界"
        assert abs(s["y_m"]) <= SYNTHETIC_HALF, f"候选 y={s['y_m'] * 1e9:.1f} nm 越界"


def test_a_bigger_real_range_does_not_widen_the_config():
    """仪器说的更大 ⇒ **不放宽**。有人可能有意把工作区收小,那是他的决定。"""
    ctx = _Ctx(piezo_full_m=1.0e-5)          # 半程 5 µm,远大于配置
    res = _run(ctx, purpose="tip_shape")
    assert res.data["piezo_half_range_m"] <= 1.5e-6 + 1e-12
    assert "沿用配置" in res.data["piezo_range_source"]


def test_an_unreadable_range_says_so_instead_of_pretending():
    """读不到 ⇒ 沿用配置,但**必须说出来**。

    「读不到」伪装成「已核对」正是这次事故能潜伏这么久的原因。
    """
    ctx = _Ctx(piezo_error="Piezo module not available")
    res = _run(ctx, purpose="tip_shape")
    assert res.data["piezo_range_source"] == "config"
    assert res.data["piezo_half_range_m"] == pytest.approx(1.5e-6)


def test_the_centre_zone_is_applied():
    """中心区约束必须作用于所有候选坐标；用装得下的净空半径验证。"""
    from mast.io.map_analysis import AnalysisConfig, nearest_clean_from

    cfg = AnalysisConfig(
        piezo_half_range_m=1.2e-6, pulse_r_m=500e-9, tip_shape_r_m=30e-9,
        frame_size_m=100e-9, center_zone_side_m=500e-9, has_xy_coarse_motion=True)
    assert cfg.effective_half_range_m == pytest.approx(250e-9), "中心区语义变了"

    # 用 **tip_shape**(30 nm 净空)验中心区,不是 pulse(500 nm)。
    # 2026-08-17 起,可用区半程装不下一个落点半径时会**退回压电范围**
    # (见 test_a_zone_too_small_to_hold_one_spot_is_released)——
    # 而 ±250 nm 装不下 500 nm 的脉冲净空,拿 pulse 来验中心区,
    # 验到的是那条退路,不是中心区本身。
    spots = nearest_clean_from([], cfg, 0.0, 0.0, spot_r_m=30e-9, count=12,
                               frame_m=0.0)
    assert spots, "中心区里连一个 30 nm 的落点都放不下?"
    far = [s for s in spots if abs(s.x_m) > 250e-9 or abs(s.y_m) > 250e-9]
    assert not far, (
        f"有 {len(far)} 个落点跑到了中心区外 —— 选点器又在读硬压电边界了?"
        f"(第一个: {far[0].x_m * 1e9:.0f}, {far[0].y_m * 1e9:.0f} nm)")


def test_a_zone_too_small_to_hold_one_spot_is_released():
    """中心区装不下单个操作所需净空时，应退回硬件工作范围，避免无解约束。"""
    from mast.io.map_analysis import AnalysisConfig, nearest_clean_from

    cfg = AnalysisConfig(
        piezo_half_range_m=1.2e-6, pulse_r_m=500e-9, tip_shape_r_m=30e-9,
        frame_size_m=100e-9, center_zone_side_m=500e-9, has_xy_coarse_motion=True)
    assert cfg.effective_half_range_m < 500e-9, "前提没了:这个区本来就装得下"

    spots = nearest_clean_from([], cfg, 600e-9, -300e-9, spot_r_m=500e-9,
                               count=12, frame_m=0.0)
    assert len(spots) >= 9, (
        f"无解的可用区没有退回压电范围,只给了 {len(spots)} 个落点 —— "
        "调用方会读到「表面用完」,然后粗动换区,然后再次无解")
    assert any(abs(s.x_m) > 250e-9 for s in spots), "退路没生效"


def test_the_chooser_and_the_out_of_zone_guard_agree_on_where_the_zone_is():
    """选点器与区外守卫必须使用同一可用区定义，返回坐标应始终位于该区内。"""
    from mast.io.map_analysis import AnalysisConfig, nearest_clean_from

    cfg = AnalysisConfig(
        piezo_half_range_m=1.2e-6, pulse_r_m=500e-9, tip_shape_r_m=30e-9,
        frame_size_m=100e-9, center_zone_side_m=500e-9, has_xy_coarse_motion=True)

    # 守卫用的那个数(clean_spot.py 里 ``reach = cfg.effective_half_range_m``)。
    guard_reach = float(cfg.effective_half_range_m)
    # 选点器发出来的点,必须全在守卫认的那个圈里 —— 否则它会一边发点、
    # 一边说站在那个点上的针尖「在区外」。
    # 用 tip_shape 的 30 nm:pulse 那一档会触发「区太小」退路,验的就不是这件事了。
    for s in nearest_clean_from([], cfg, 0.0, 0.0, spot_r_m=30e-9, count=24,
                                frame_m=0.0):
        assert abs(s.x_m) <= guard_reach and abs(s.y_m) <= guard_reach, (
            f"选点器发了 ({s.x_m * 1e9:.0f}, {s.y_m * 1e9:.0f}) nm,"
            f"而守卫认的可用区只有 ±{guard_reach * 1e9:.0f} nm —— 两个定义又岔开了")


def test_the_margin_subtracts_the_frame_not_the_damage_radius():
    """边距扣除应保证目标扫描帧完整位于范围内，而不是减去表面损伤半径。
    无粗动、无中心区约束的合成配置用于隔离这一几何性质。"""
    from mast.io.map_analysis import AnalysisConfig, nearest_clean_from

    SYNTHETIC_HALF = 1.2e-6
    cfg = AnalysisConfig(
        piezo_half_range_m=SYNTHETIC_HALF, tip_shape_r_m=30e-9, pulse_r_m=500e-9,
        frame_size_m=100e-9, center_zone_side_m=None, has_xy_coarse_motion=False)

    pulse_spots = nearest_clean_from([], cfg, 0.0, 0.0, spot_r_m=500e-9, count=40)
    assert len(pulse_spots) >= 9, (
        f"脉冲落点只剩 {len(pulse_spots)} 个 —— 减错了边距?"
        "(减 r_spot 会让 1200-500=700 < 网格步长 1000,只剩原点)")

    for s in nearest_clean_from([], cfg, 0.0, 0.0, spot_r_m=30e-9, count=60):
        assert abs(s.x_m) <= SYNTHETIC_HALF and abs(s.y_m) <= SYNTHETIC_HALF


def test_one_operation_fills_a_zone_that_can_hold_exactly_one():
    """中心区恰好只有一个可用网格点时，用掉该点后不再返回候选；
    没有中心区约束的配置仍应有其他可用位置。"""
    from mast.io.map_analysis import AnalysisConfig, nearest_clean_from

    HALF = 1.2e-6
    # 净空 200 nm ⇒ 格距 400 nm;区半程 250 nm ⇒ 只有区心一个格点,而且装得下。
    cfg = AnalysisConfig(
        piezo_half_range_m=HALF, tip_shape_r_m=200e-9, pulse_r_m=500e-9,
        frame_size_m=100e-9, center_zone_side_m=500e-9, has_xy_coarse_motion=True)
    first = nearest_clean_from([], cfg, 0.0, 0.0, spot_r_m=200e-9, count=8,
                               frame_m=0.0)
    assert len(first) == 1 and first[0].x_m == 0.0, (
        f"区里本该只有区心一个落点,实际 {len(first)} 个")
    assert nearest_clean_from([], cfg, 0.0, 0.0, spot_r_m=200e-9, count=8,
                              exclude=[(0.0, 0.0)], frame_m=0.0) == [], (
        "用掉唯一那个之后区里还有地方 —— 「一个就满」这条逻辑散了")

    # 没有粗动的机器上中心区不设,一次操作**绝不许**把整片表面用完 ——
    # 那种机器换不了区,「表面用完」等于这一跑到此为止。
    fixed = AnalysisConfig(
        piezo_half_range_m=HALF, tip_shape_r_m=200e-9, pulse_r_m=500e-9,
        frame_size_m=100e-9, center_zone_side_m=None, has_xy_coarse_motion=False)
    assert nearest_clean_from([], fixed, 0.0, 0.0, spot_r_m=200e-9, count=8,
                              exclude=[(0.0, 0.0)], frame_m=0.0)