"""原子相验收档位的尺度、驻留时间与估时契约。

配置视野与像素数必须通过尺度门；像素数增加时，线时长应保持每像素驻留时间。
测试验证档位边界、可选参数留空、运行配方与扫描策略的一致性。"""

from __future__ import annotations

import pytest

from mast.core import scan_policy
from mast.core.special_tip_workflow import AtomicTipWorkflow
from mast.vision.atomic_phase import SCALE_FULL_NMPP, plan_scale

TIER = "atomic_verify"
NM = 1e-9


@pytest.fixture(autouse=True)
def _clean_holder():
    scan_policy.set_policy(None)
    yield
    scan_policy.set_policy(None)


def _tier() -> dict:
    t = scan_policy.get_tier_by_name(TIER)
    assert t is not None, f"出厂档 {TIER} 不见了"
    return t


# ── 档位存在且结构合法 ────────────────────────────────────────────────────────

def test_tier_exists_and_sits_between_slow_and_atomic():
    """插在 slow(2 nm) 与 atomic(10 nm) 之间 —— 顺序错了结构校验会拒。"""
    names = scan_policy.tier_names()
    assert TIER in names
    assert names.index("slow") < names.index(TIER) < names.index("atomic")
    assert len(names) == 6, f"档数变了: {names}"


def test_factory_table_still_passes_structure_validation():
    """六档表自己必须过得了 fail-closed 校验(严格递增 + 兜底档在末位)。"""
    assert scan_policy.sanitize(scan_policy.factory_tiers())


def test_optional_fields_stay_none():
    """出厂一律不下发 setpoint / PI —— 那两个没有普适安全值。"""
    t = _tier()
    assert t["setpoint_a"] is None
    assert t["p_gain"] is None
    assert t["time_constant_s"] is None


# ── 每个数的派生依据 ──────────────────────────────────────────────────────────

def test_the_frame_actually_passes_the_scale_gate_with_room_to_spare():
    """档位最大视野必须完整通过尺度门，并保留明确的采样余量。"""
    t = _tier()
    nmpp, scale, problem = plan_scale(t["upper_size_m"], t["pixels"])
    assert scale == "full", f"{TIER} 的最大帧 {nmpp:.5f} nm/px 过不了尺度门"
    assert problem == ""
    assert nmpp == pytest.approx(5.0 / 512)
    # 余量:比满权重门槛低一半以上
    assert nmpp <= SCALE_FULL_NMPP / 2


def test_pixel_dwell_is_conserved_from_the_operator_recipe():
    """像素数和线时长应同步变化，保持与运行配方相同的每像素驻留时间。"""
    recipe = AtomicTipWorkflow()
    recipe_dwell = recipe.eval_line_time_s / recipe.eval_pixels
    t = _tier()
    dwell = t["line_time_s"] / t["pixels"]

    assert recipe_dwell == pytest.approx(586e-6, abs=1e-6)
    assert dwell == pytest.approx(recipe_dwell, rel=1e-9), (
        f"{TIER} 每像素驻留 {dwell * 1e6:.0f} µs ≠ 配方的 "
        f"{recipe_dwell * 1e6:.0f} µs —— 加像素没有同比加线时")


def test_field_of_view_comes_from_the_operator_recipe():
    """档位视野与运行配方共享配置来源。"""
    assert _tier()["upper_size_m"] == pytest.approx(AtomicTipWorkflow().eval_frame_nm * NM)


def test_pixels_and_line_time_are_the_recipe_doubled():
    recipe = AtomicTipWorkflow()
    t = _tier()
    assert t["pixels"] == 2 * recipe.eval_pixels
    assert t["line_time_s"] == pytest.approx(2 * recipe.eval_line_time_s)


def test_tip_lateral_speed_is_far_below_the_scraping_incident():
    """横向速度由视野宽度与线时长之比决定；单独比较 line_time 无法说明扫描速度。"""
    t = _tier()
    speed_nm_s = (t["upper_size_m"] / NM) / t["line_time_s"]
    assert speed_nm_s == pytest.approx(16.7, abs=0.1)
    assert speed_nm_s < 50.0, "预扫描档位表定的安全线"


def test_frame_time_is_about_five_minutes_not_thirty():
    """依据像素数、线时长和扫描方向计算帧时，防止配置估时与实际执行参数不一致。"""
    t = _tier()
    est = scan_policy.estimate_scan_seconds(t["pixels"], t["line_time_s"])
    assert est == pytest.approx(307.2, abs=1.0)
    assert est <= 12 * 60, "自动查表选得到的档不该让人等这么久"


# ── 双边动作:查表副作用 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("size_nm", [2.001, 3.0, 4.0, 5.0])
def test_adding_atomic_verify_moved_the_2_to_5_nm_lookup(size_nm):
    """新增档位后，尺寸自动查表也必须选择该档位。
    测试同时核验选择结果和 resolver 行为，防止只新增名称而遗漏自动派发。"""
    tier = scan_policy.get_tier_for_size(size_nm * NM)
    assert tier["name"] == TIER
    assert tier["pixels"] == 512
    assert tier["line_time_s"] == pytest.approx(0.30)


@pytest.mark.parametrize("size_nm,expect", [
    (2.0, "slow"),          # 下边界不动:正好 2 nm 仍属 slow
    (5.001, "atomic"),      # 上边界之外仍是 atomic
    (10.0, "atomic"),
    (50.0, "highres"),      # 常用档一个都没动
    (100.0, "highres"),
])
def test_the_other_tiers_did_not_move(size_nm, expect):
    """新增档只吃 2-5 nm 这一段,别的尺度必须原样 —— 改边界会影响所有既有调用方。"""
    assert scan_policy.get_tier_for_size(size_nm * NM)["name"] == expect


def test_atomic_tier_still_cannot_judge_its_own_top_end():
    """`atomic` 档的上半段**依然**判不出原子相 —— 这一次没有偷偷修好它。

    10 nm / 256 px = 0.0391 nm/px ⇒ 过渡带。新增 `atomic_verify` 是**绕开**这个
    问题(给验收一条能过门的路),不是修它。写在这里免得下一个人以为已经解决了:
    要在 5.12-10 nm 上判原子相,仍然只能显式换档或加像素。
    """
    t = scan_policy.get_tier_by_name("atomic")
    nmpp, scale, problem = plan_scale(t["upper_size_m"], t["pixels"])
    assert scale == "reduced"
    assert nmpp == pytest.approx(10.0 / 256)
    assert problem, "过渡带必须说出来"


def test_tier_is_selectable_by_name_for_any_size():
    """S2 显式走 purpose=atomic_verify,不依赖自动查表(get_tier_by_name 绕过尺寸)。"""
    assert scan_policy.get_tier_by_name("ATOMIC_VERIFY")["name"] == TIER
    assert scan_policy.get_tier_by_name("  atomic_verify ")["name"] == TIER
