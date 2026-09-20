"""扫描参数档位表(mast.core.scan_policy)。

重点在**结构 fail-closed**:一张半张的表比没有表更危险 —— 用户以为自己设了
6 档,系统按 4 档跑,而这个差异在任何界面上都看不出来。
"""

from __future__ import annotations

import pytest

from mast.core import scan_policy
from mast.core.scan_policy import PolicyRejected


@pytest.fixture(autouse=True)
def _clean_holder():
    """每个用例前后都把进程级 holder 清空 —— 档位表是全局状态,用例之间不能串。"""
    scan_policy.set_policy(None)
    yield
    scan_policy.set_policy(None)


# ── 出厂表 ────────────────────────────────────────────────────────────────────

def test_factory_table_is_used_when_operator_never_set_anything():
    assert scan_policy.is_customised() is False
    assert scan_policy.get_stored_policy() == []
    tiers = scan_policy.get_policy()
    assert [t["name"] for t in tiers] == ["slow", "atomic_verify", "atomic", "highres", "roi", "survey"]
    assert all(t["source"] == "factory" for t in tiers)


def test_factory_table_is_structurally_valid():
    """出厂表自己必须过得了校验 —— 否则第一次保存就会莫名其妙失败。"""
    assert scan_policy.sanitize(scan_policy.factory_tiers())


def test_factory_tiers_returns_a_deep_copy():
    a = scan_policy.factory_tiers()
    a[0]["pixels"] = 9999
    assert scan_policy.factory_tiers()[0]["pixels"] == 512


def test_factory_optional_fields_are_all_none():
    """setpoint / PI 出厂一律空 = 不下发(空即 no-op)。

    出厂给一个 setpoint 等于替所有仪器决定了隧道结电阻;给一个 P 增益等于替所有
    针尖决定了反馈响应。两者都没有普适安全值。
    """
    for tier in scan_policy.factory_tiers():
        assert tier["setpoint_a"] is None
        assert tier["p_gain"] is None
        assert tier["time_constant_s"] is None


# ── 查表语义 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size_nm,expect", [
    (0.5, "slow"),
    (2.0, "slow"),          # 新边界(2026-08-09):正好 2 nm 属 slow
    # 2026-08-14:2-5 nm 这一段从 atomic 改判 atomic_verify(新增档,见下方
    # test_adding_atomic_verify_moved_the_2_to_5_nm_lookup)。加档位是双边动作,
    # 这一行就是那个「另一边」。
    (2.001, "atomic_verify"),
    (5.0, "atomic_verify"),  # 正好 5 nm 属 atomic_verify(闭上界)
    (5.001, "atomic"),
    (10.0, "atomic"),       # 边界值落在小的那一档(闭上界)
    (10.001, "highres"),
    (100.0, "highres"),     # 钉死:正好 100 nm 属 highres
    (100.001, "roi"),
    (500.0, "roi"),
    (500.001, "survey"),
    (1000.0, "survey"),
    (5000.0, "survey"),
])
def test_tier_lookup_closed_upper_bound(size_nm, expect):
    assert scan_policy.get_tier_for_size(size_nm * 1e-9)["name"] == expect


def test_boundary_survives_nm_to_m_float_conversion():
    """界面上输 100 nm、边界也是 100 nm,必须落进 100 nm 那一档。

    ``100.0 * 1e-9`` 在双精度里是 1.0000000000000001e-07,比边界大一个最低位。
    没有相对容差时它会静默落进下一档,用错速度和像素 —— 而用户看到的两个
    数字一模一样,这种 bug 根本无从查起。
    """
    assert 100.0 * 1e-9 > 1e-7          # 前提:这个浮点陷阱确实存在
    assert scan_policy.get_tier_for_size(100.0 * 1e-9)["name"] == "highres"
    assert scan_policy.get_tier_for_size(500.0 * 1e-9)["name"] == "roi"
    assert scan_policy.get_tier_for_size(10.0 * 1e-9)["name"] == "atomic"


def test_boundary_tolerance_is_far_below_physical_significance():
    """容差只吸收浮点噪声,不能吞掉真实的尺寸差异。

    比边界大 1‰ 的尺寸必须落进下一档 —— 100 nm 与 100.1 nm 是用户真的能
    分辨、也真的在意的差别。
    """
    assert scan_policy.get_tier_for_size(1e-7 * 1.001)["name"] == "roi"


def test_tier_lookup_survives_garbage_size():
    """NaN / None / 字符串不能让查表抛异常 —— 查表在扫描路径上。"""
    for bad in (float("nan"), None, "abc"):
        assert scan_policy.get_tier_for_size(bad)["name"] == "slow"


def test_tier_lookup_huge_size_falls_to_open_tier():
    assert scan_policy.get_tier_for_size(1.0)["name"] == "survey"


def test_get_tier_by_name_is_case_insensitive():
    assert scan_policy.get_tier_by_name("ROI")["name"] == "roi"
    assert scan_policy.get_tier_by_name("  highres ")["name"] == "highres"
    assert scan_policy.get_tier_by_name("nope") is None
    assert scan_policy.get_tier_by_name("") is None


def test_tier_names_matches_active_table():
    assert scan_policy.tier_names() == ["slow", "atomic_verify", "atomic", "highres", "roi", "survey"]


# ── 可变档数(既定的需求) ───────────────────────────────────────────────

def test_operator_can_define_a_different_number_of_tiers():
    """档数可变是明确需求 —— 固定 4 档已被否决。"""
    scan_policy.set_policy([
        {"name": "tiny", "upper_size_m": 5e-9, "pixels": 1024, "line_time_s": 6.0},
        {"name": "mid", "upper_size_m": 2e-7, "pixels": 512, "line_time_s": 1.5},
        {"name": "big", "upper_size_m": None, "pixels": 128, "line_time_s": 0.3},
    ])
    assert scan_policy.is_customised() is True
    assert scan_policy.tier_names() == ["tiny", "mid", "big"]
    assert scan_policy.get_tier_for_size(3e-9)["pixels"] == 1024
    assert scan_policy.get_tier_for_size(1e-7)["line_time_s"] == 1.5
    assert scan_policy.get_tier_for_size(1e-3)["pixels"] == 128


def test_single_open_tier_is_legal():
    """「我只扫一个尺度」的极简用法必须能表达。"""
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0},
    ])
    assert scan_policy.get_tier_for_size(1e-9)["name"] == "only"
    assert scan_policy.get_tier_for_size(1e-6)["name"] == "only"


def test_eight_tiers_ok_nine_rejected():
    def mk(n):
        return [
            {"name": f"t{i}", "upper_size_m": (i + 1) * 1e-9,
             "pixels": 256, "line_time_s": 1.0}
            for i in range(n - 1)
        ] + [{"name": "open", "upper_size_m": None,
              "pixels": 256, "line_time_s": 1.0}]

    assert len(scan_policy.sanitize(mk(8))) == 8
    with pytest.raises(PolicyRejected, match="档位数量"):
        scan_policy.sanitize(mk(9))


# ── 结构 fail-closed ─────────────────────────────────────────────────────────

def test_reject_table_without_open_tier():
    """没有兜底档 = 超过最大边界的尺寸查不到参数。"""
    with pytest.raises(PolicyRejected, match="兜底档"):
        scan_policy.sanitize([
            {"name": "a", "upper_size_m": 1e-8, "pixels": 512, "line_time_s": 2.0},
            {"name": "b", "upper_size_m": 1e-7, "pixels": 512, "line_time_s": 1.0},
        ])


def test_reject_table_with_two_open_tiers():
    with pytest.raises(PolicyRejected, match="恰好有 1 个兜底档"):
        scan_policy.sanitize([
            {"name": "a", "upper_size_m": None, "pixels": 512, "line_time_s": 2.0},
            {"name": "b", "upper_size_m": None, "pixels": 512, "line_time_s": 1.0},
        ])


def test_reject_open_tier_not_last():
    """兜底档在中间 = 它后面的档是静默死代码。"""
    with pytest.raises(PolicyRejected, match="最后一档"):
        scan_policy.sanitize([
            {"name": "open", "upper_size_m": None, "pixels": 512, "line_time_s": 2.0},
            {"name": "b", "upper_size_m": 1e-7, "pixels": 512, "line_time_s": 1.0},
        ])


def test_reject_non_monotonic_bounds():
    """区间重叠时,查表结果取决于遍历顺序 —— 那不是表,是巧合。"""
    with pytest.raises(PolicyRejected, match="严格递增"):
        scan_policy.sanitize([
            {"name": "a", "upper_size_m": 1e-7, "pixels": 512, "line_time_s": 2.0},
            {"name": "b", "upper_size_m": 1e-8, "pixels": 512, "line_time_s": 1.0},
            {"name": "c", "upper_size_m": None, "pixels": 256, "line_time_s": 0.5},
        ])


def test_reject_equal_bounds():
    with pytest.raises(PolicyRejected, match="严格递增"):
        scan_policy.sanitize([
            {"name": "a", "upper_size_m": 1e-8, "pixels": 512, "line_time_s": 2.0},
            {"name": "b", "upper_size_m": 1e-8, "pixels": 512, "line_time_s": 1.0},
            {"name": "c", "upper_size_m": None, "pixels": 256, "line_time_s": 0.5},
        ])


def test_reject_bound_outside_hardware_range():
    with pytest.raises(PolicyRejected, match="超出硬件可扫范围"):
        scan_policy.sanitize([
            {"name": "huge", "upper_size_m": 1.0, "pixels": 256, "line_time_s": 1.0},
            {"name": "open", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0},
        ])


def test_reject_non_numeric_bound():
    with pytest.raises(PolicyRejected, match="不是数值"):
        scan_policy.sanitize([
            {"name": "a", "upper_size_m": "大概这么大", "pixels": 256,
             "line_time_s": 1.0},
            {"name": "open", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0},
        ])


def test_reject_non_dict_tier():
    with pytest.raises(PolicyRejected, match="不是对象"):
        scan_policy.sanitize([["not", "a", "dict"]])


def test_reject_non_list_payload():
    with pytest.raises(PolicyRejected, match="必须是列表"):
        scan_policy.sanitize({"tiers": "atomic,highres"})


def test_a_rejected_write_leaves_the_previous_table_intact():
    """一次坏 POST 不能把用户已经调好的表打成半张。"""
    good = [
        {"name": "mine", "upper_size_m": 1e-7, "pixels": 1024, "line_time_s": 3.0},
        {"name": "rest", "upper_size_m": None, "pixels": 256, "line_time_s": 0.5},
    ]
    scan_policy.set_policy(good)
    with pytest.raises(PolicyRejected):
        scan_policy.set_policy([
            {"name": "broken", "upper_size_m": 1e-7, "pixels": 256,
             "line_time_s": 1.0},
        ])
    assert scan_policy.tier_names() == ["mine", "rest"]
    assert scan_policy.get_tier_for_size(1e-8)["pixels"] == 1024


# ── 数值卫生(结构之外是宽容的) ───────────────────────────────────────────────

def test_numbers_are_clamped_not_rejected():
    """手滑打多个 0 不该让整张表失败 —— 数值层面修剪,结构层面才拒绝。"""
    tiers = scan_policy.sanitize([
        {"name": "a", "upper_size_m": 1e-8, "pixels": 999999, "line_time_s": 1e9},
        {"name": "open", "upper_size_m": None, "pixels": 1, "line_time_s": 0.0},
    ])
    assert tiers[0]["pixels"] == 4096
    assert tiers[0]["line_time_s"] == 600.0
    assert tiers[1]["pixels"] == 16
    assert tiers[1]["line_time_s"] == 1e-4


def test_unknown_keys_are_dropped():
    tiers = scan_policy.sanitize([
        {"name": "a", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0,
         "rm_minus_rf": True, "注入": "<script>"},
    ])
    assert "rm_minus_rf" not in tiers[0]
    assert "注入" not in tiers[0]


def test_nan_field_becomes_none_not_crash():
    tiers = scan_policy.sanitize([
        {"name": "a", "upper_size_m": None, "pixels": float("nan"),
         "line_time_s": float("inf"), "setpoint_a": float("nan")},
    ])
    # 必需字段回填出厂值,可选字段留空
    assert tiers[0]["pixels"] == 256
    assert tiers[0]["line_time_s"] == 0.5
    assert tiers[0]["setpoint_a"] is None


def test_missing_name_gets_a_positional_placeholder():
    tiers = scan_policy.sanitize([
        {"upper_size_m": 1e-8, "pixels": 512, "line_time_s": 2.0},
        {"upper_size_m": None, "pixels": 256, "line_time_s": 0.5},
    ])
    assert tiers[0]["name"] == "tier1"
    assert tiers[1]["name"] == "tier2"


# ── 字段级 fallback:按尺度对齐,不是按下标 ────────────────────────────────────

def test_required_field_falls_back_to_the_factory_tier_covering_the_same_scale():
    """用户加一档「300nm 专用」只填像素时,每线时间该来自出厂 roi 档
    (覆盖 300nm 的那一档),而不是来自「第 N 档」这种与物理无关的位置。"""
    tiers = scan_policy.sanitize([
        {"name": "custom300", "upper_size_m": 3e-7, "pixels": 1024},
        {"name": "open", "upper_size_m": None, "pixels": 256, "line_time_s": 0.5},
    ])
    assert tiers[0]["pixels"] == 1024
    assert tiers[0]["line_time_s"] == 0.8        # 出厂 roi 档的值
    assert tiers[0]["_factory_filled"] == ["line_time_s"]


def test_open_tier_fallback_uses_the_factory_open_tier():
    tiers = scan_policy.sanitize([
        {"name": "open", "upper_size_m": None, "pixels": 128},
    ])
    assert tiers[0]["line_time_s"] == 0.5        # 出厂 survey 档
    assert tiers[0]["_factory_filled"] == ["line_time_s"]


def test_fully_specified_tier_records_no_factory_fill():
    tiers = scan_policy.sanitize([
        {"name": "open", "upper_size_m": None, "pixels": 128, "line_time_s": 0.2},
    ])
    assert tiers[0]["_factory_filled"] == []


# ── 存取 round-trip ──────────────────────────────────────────────────────────

def test_set_policy_accepts_both_bare_list_and_wrapped_dict():
    payload = [
        {"name": "one", "upper_size_m": None, "pixels": 300, "line_time_s": 1.0},
    ]
    scan_policy.set_policy(payload)
    assert scan_policy.tier_names() == ["one"]
    scan_policy.set_policy(None)
    scan_policy.set_policy({"tiers": payload})
    assert scan_policy.tier_names() == ["one"]


def test_empty_payload_clears_back_to_factory():
    scan_policy.set_policy([
        {"name": "x", "upper_size_m": None, "pixels": 64, "line_time_s": 1.0},
    ])
    assert scan_policy.is_customised()
    scan_policy.set_policy([])
    assert not scan_policy.is_customised()
    assert scan_policy.tier_names() == ["slow", "atomic_verify", "atomic", "highres", "roi", "survey"]


def test_get_policy_returns_copies_not_live_refs():
    tiers = scan_policy.get_policy()
    tiers[0]["pixels"] = 7
    assert scan_policy.get_policy()[0]["pixels"] == 512


def test_custom_table_is_marked_operator_sourced():
    scan_policy.set_policy([
        {"name": "x", "upper_size_m": None, "pixels": 64, "line_time_s": 1.0},
    ])
    assert all(t["source"] == "operator" for t in scan_policy.get_policy())


# ── 估时与渲染 ────────────────────────────────────────────────────────────────

def test_estimate_scan_seconds_round_trip_factor():
    assert scan_policy.estimate_scan_seconds(256, 0.5) == pytest.approx(256.0)
    assert scan_policy.estimate_scan_seconds(512, 2.0) == pytest.approx(2048.0)


def test_estimate_scan_seconds_is_zero_for_garbage():
    for px, lt in ((0, 1.0), (256, 0.0), (None, 1.0), (256, None), ("x", "y")):
        assert scan_policy.estimate_scan_seconds(px, lt) == 0.0


def test_format_policy_block_shows_bounds_and_source():
    block = scan_policy.format_policy_block()
    assert "atomic" in block and "≤ 10 nm" in block
    assert "更大" in block              # 兜底档
    assert "[出厂]" in block


def test_format_policy_block_shows_si_and_human_units_for_setpoint():
    """人类可读单位必须紧跟 SI 数值，避免量纲误读。"""
    scan_policy.set_policy([
        {"name": "x", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0,
         "setpoint_a": 1e-10},
    ])
    block = scan_policy.format_policy_block()
    assert "1e-10 A" in block and "100 pA" in block


# 默认档位适合迭代检查，慢速档位服务更细致的采集。
# 测试显式固定当前选择与参数，避免以提高像素为由悄悄延长默认等待时间。

def test_the_everyday_tier_finishes_in_about_eight_minutes():
    """最常用的那档(highres,≤100 nm)必须在 ~8 分钟内出图。

    上限 12 分钟不是随便定的:超过它,「找区域 → 判针尖 → 试参数」这个循环
    一小时跑不完 5 轮,而那正是这张表要服务的工作方式。
    """
    tier = scan_policy.get_tier_for_size(6e-8)          # 独立构造的 60 nm 档内值
    assert tier["name"] == "highres"
    est = scan_policy.estimate_scan_seconds(tier["pixels"], tier["line_time_s"])
    assert 6 * 60 <= est <= 12 * 60, f"highres 一帧 {est / 60:.1f} min"


def test_no_default_tier_is_a_half_hour_frame():
    """按尺寸自动选到的档,没有一个该让人等半小时 —— 那是 slow 档的活。"""
    slow = scan_policy.get_tier_by_name("slow")
    for tier in scan_policy.get_policy():
        est = scan_policy.estimate_scan_seconds(
            tier["pixels"], tier["line_time_s"])
        if tier["name"] == slow["name"]:
            continue
        assert est <= 12 * 60, f"{tier['name']} 一帧 {est / 60:.1f} min,太久了"


def test_the_slow_tier_exists_and_is_actually_slow():
    """显式要成图的那一档:~30 分钟,像素翻倍。

    它同时是 ≤2 nm 的自动档(那个尺度上你就是在看原子),也能按名字选给任意尺寸
    —— get_tier_by_name 绕过尺寸查表,不需要新机制。
    """
    slow = scan_policy.get_tier_by_name("slow")
    assert slow is not None, "慢速档不见了"
    est = scan_policy.estimate_scan_seconds(slow["pixels"], slow["line_time_s"])
    assert 25 * 60 <= est <= 35 * 60, f"slow 一帧 {est / 60:.1f} min"
    assert slow["pixels"] > scan_policy.get_tier_for_size(5e-8)["pixels"], \
        "慢速档的意义是拿时间换分辨率;像素不比默认档高就没意义了"
    assert scan_policy.get_tier_for_size(1e-9)["name"] == "slow"


def test_省时来自像素而不是把线时压短():
    """这次省时的主力是 512→256 像素,**不是**缩短每线时间。

    区别是承重的:线时才是压针尖的那个量(反馈跟不上 → 正反扫重影),
    而行数减半在**每像素驻留时间翻倍**的同时把帧时减半。把这两件事捆在一起,
    正是旧表 34 分钟的由来。要推翻:见 scan_policy.py 里那段「要推翻这次改动
    需要回答什么」。

    ⚠️ 2026-08-14 放宽了一次,**换成一条更强的规则**:原来是「非 slow 档一律
    ≤256 px」,现在是「超过 256 px 的档必须用 line_time 把像素**买下来**」。
    起因是 `atomic_verify`(5 nm / 512 px / 0.30 s):在 5 nm 这个尺度上 256 px
    算出来是 0.0195 nm/px,余量只有 2.4%,用户一句「扫 5.2 nm」就翻出满权重档
    —— 那一档**判不出原子相**。所以那里的 512 px 不是「看起来更好」,是尺度门要的。

    新规则拦得住旧规则要拦的那件事,而且更贴近根因:把 `atomic` 改成 512 px 却
    留着 1.2 s,每像素驻留从 4.69 ms 掉到 2.34 ms —— `nm/px` 变好看了,每个采样点
    携带的信息反而变少,尺度门是被骗过去的。那正是 `pixels ≤ 256` 当初真正在防的。
    """
    for tier in scan_policy.get_policy():
        if tier["name"] == "slow":
            continue
        if tier["pixels"] > 256:
            # 加像素必须同比加 line_time:每像素驻留不得低于用户配方的
            # 0.15 s / 256 px = 586 µs。只动 pixels 不动 line_time ⇒ 这里红。
            recipe_dwell_s = 0.15 / 256
            dwell = tier["line_time_s"] / tier["pixels"]
            assert dwell >= recipe_dwell_s * (1 - 1e-9), (
                f"{tier['name']} 用了 {tier['pixels']} 像素却只给 "
                f"{tier['line_time_s']} s/线 —— 每像素驻留 {dwell * 1e6:.0f} µs "
                f"低于用户配方的 {recipe_dwell_s * 1e6:.0f} µs。"
                f"提像素不提线时 = 仪器侧的零信息注入。")
        # 每像素驻留 = line_time / pixels,必须还在采样能力之内(Signals 周期
        # 0.5 ms;低于它等于让 Nanonis 在一个采样点里跑完几个像素)。
        dwell_s = tier["line_time_s"] / tier["pixels"]
        assert dwell_s >= 5e-4, (
            f"{tier['name']} 每像素只有 {dwell_s * 1e6:.0f} µs,低于采样周期 500 µs")
