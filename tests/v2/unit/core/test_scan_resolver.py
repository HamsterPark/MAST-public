"""意图→参数解析层(mast.core.scan_resolver)。

这是「LLM 不填数字」的落点,所以测试的重点是**优先级链的每一格**:五个来源
× 每个字段,每格至少一例。链错一格的后果是用户的偏好被静默忽略,而这在
界面上完全看不出来。
"""

from __future__ import annotations

import pytest

from mast.core import scan_policy, scan_resolver
from mast.core.scan_resolver import (
    SOURCE_DEFAULT,
    SOURCE_EXPLICIT,
    SOURCE_KEEP,
    SOURCE_PREFS,
    SOURCE_PREFS_DERIVED,
    SOURCE_TIER_FACTORY,
    SOURCE_TIER_OPERATOR,
    ScanIntent,
    resolve_scan,
)


@pytest.fixture(autouse=True)
def _clean_holder():
    scan_policy.set_policy(None)
    yield
    scan_policy.set_policy(None)


def _intent(size_nm=100.0, **kw):
    kw.setdefault("center_x_m", 0.0)
    kw.setdefault("center_y_m", 0.0)
    return ScanIntent(size_m=size_nm * 1e-9, **kw)


def _resolve(intent, prefs=None, **kw):
    """默认传空 prefs —— 不显式指定就会读进程 holder,那是测试污染的经典入口。"""
    return resolve_scan(intent, prefs=prefs or {}, **kw)


# ── 基本形状 ──────────────────────────────────────────────────────────────────

def test_resolves_a_complete_parameter_set_from_size_alone():
    """意图只有「扫哪、多大」,出来的是一整套可执行参数 —— 这就是全部要点。"""
    res = _resolve(_intent(100.0))
    assert res.tier_name == "highres"
    assert res.configure_scan["width_m"] == pytest.approx(1e-7)
    assert res.configure_scan["height_m"] == pytest.approx(1e-7)
    assert res.configure_scan["line_time_s"] == 1.0
    assert res.set_scan_buffer == {"pixels": 256, "lines": 256}
    assert res.estimated_scan_s == pytest.approx(256 * 1.0 * 2)   # ≈8.5 min


def test_optional_hardware_writes_are_none_by_default():
    """空即 no-op:出厂没给 setpoint/bias/PI,就一个都不下发。

    「不下发」不等于「下发 0」—— 后者会把 setpoint 设成 0 安培。
    """
    res = _resolve(_intent())
    assert res.set_setpoint is None
    assert res.set_bias is None
    assert res.set_zctrl_gain is None


def test_angle_absent_means_keep_current_not_zero():
    """不给角度 = 保持硬件现值。

    ConfigureScan 在 2026-07-03 修过这个坑:传 0 会让每次 recenter 都把画面
    转回 0°。所以 keep-current 的实现必须是「键不出现」,不是「键=0」。
    """
    res = _resolve(_intent())
    assert "angle_deg" not in res.configure_scan
    assert res.trace["angle_deg"]["source"] == SOURCE_KEEP


def test_invalid_size_raises_rather_than_inventing_one():
    """不替用户发明扫描尺寸(既有产品决策)。

    0 / 负数 / NaN 不是「太小的尺寸」,是上游出错的信号。静默 clamp 成 0.1 nm
    会扫出一张荒谬的图却什么都不说 —— 那是本项目视为致命的 fail-silent。
    """
    for bad in (0.0, -1e-9, None, float("nan"), "big"):
        with pytest.raises(ValueError, match="size_m"):
            _resolve(ScanIntent(center_x_m=0.0, center_y_m=0.0, size_m=bad))


def test_positive_but_out_of_range_size_is_clamped_not_rejected():
    """正数但超范围是「一个人真打出来的数字」—— 属参数卫生,修剪并声明。"""
    res = _resolve(ScanIntent(center_x_m=0.0, center_y_m=0.0, size_m=1e-3))
    assert res.configure_scan["width_m"] == pytest.approx(1e-5)
    assert any("size_m" in w and "修剪" in w for w in res.warnings)


def test_invalid_center_raises():
    with pytest.raises(ValueError, match="扫描中心"):
        _resolve(ScanIntent(center_x_m=float("nan"), center_y_m=0.0, size_m=1e-7))


# ── 优先级链:pixels ─────────────────────────────────────────────────────────

def test_pixels_explicit_wins_over_everything():
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 128, "line_time_s": 1.0},
    ])
    res = _resolve(_intent(explicit={"pixels": 1024}), prefs={"scan_lines": 256})
    assert res.set_scan_buffer["pixels"] == 1024
    assert res.trace["pixels"]["source"] == SOURCE_EXPLICIT


def test_pixels_operator_tier_beats_prefs():
    """用户按尺度设过的值,比一个全局标量更精确 —— 尺度化的赢。"""
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 128, "line_time_s": 1.0},
    ])
    res = _resolve(_intent(), prefs={"scan_lines": 1024})
    assert res.set_scan_buffer["pixels"] == 128
    assert res.trace["pixels"]["source"] == SOURCE_TIER_OPERATOR
    assert res.trace["pixels"]["tier"] == "only"


def test_pixels_prefs_beats_factory_tier():
    res = _resolve(_intent(), prefs={"scan_lines": 1024})
    assert res.set_scan_buffer["pixels"] == 1024
    assert res.trace["pixels"]["source"] == SOURCE_PREFS


def test_pixels_falls_back_to_factory_tier():
    res = _resolve(_intent(100.0))
    assert res.set_scan_buffer["pixels"] == 256
    assert res.trace["pixels"]["source"] == SOURCE_TIER_FACTORY


def test_pixels_factory_filled_field_in_operator_table_is_traced_as_factory():
    """用户定制了表,不代表每个数字都是他填的 —— trace 要说实话。"""
    scan_policy.set_policy([
        {"name": "mine", "upper_size_m": None, "line_time_s": 3.0},   # 没填 pixels
    ])
    res = _resolve(_intent())
    assert res.trace["pixels"]["source"] == SOURCE_TIER_FACTORY
    assert res.trace["line_time_s"]["source"] == SOURCE_TIER_OPERATOR


# ── 优先级链:line_time_s ────────────────────────────────────────────────────

def test_line_time_explicit_wins():
    res = _resolve(_intent(explicit={"line_time_s": 0.25}),
                   prefs={"line_time_s": 9.0})
    assert res.configure_scan["line_time_s"] == 0.25
    assert res.trace["line_time_s"]["source"] == SOURCE_EXPLICIT


def test_line_time_operator_tier_beats_prefs():
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256, "line_time_s": 3.0},
    ])
    res = _resolve(_intent(), prefs={"line_time_s": 9.0})
    assert res.configure_scan["line_time_s"] == 3.0
    assert res.trace["line_time_s"]["source"] == SOURCE_TIER_OPERATOR


def test_line_time_prefs_beats_factory():
    res = _resolve(_intent(), prefs={"line_time_s": 7.0})
    assert res.configure_scan["line_time_s"] == 7.0
    assert res.trace["line_time_s"]["source"] == SOURCE_PREFS


def test_line_time_derived_from_pref_speed_when_no_line_time():
    """用户设的是「扫描速度」而不是「每线时间」时,偏好照样要被兑现。

    表里只存 line_time(单一真源),所以速度要按帧宽折算过去。
    """
    res = _resolve(_intent(100.0), prefs={"scan_speed_nm_s": 50.0})
    # 100 nm 帧 / 50 nm/s = 2.0 s 每线
    assert res.configure_scan["line_time_s"] == pytest.approx(2.0)
    assert res.trace["line_time_s"]["source"] == SOURCE_PREFS_DERIVED


def test_explicit_line_time_beats_derived_speed():
    res = _resolve(_intent(100.0, explicit={"line_time_s": 0.5}),
                   prefs={"scan_speed_nm_s": 50.0})
    assert res.configure_scan["line_time_s"] == 0.5


def test_pref_line_time_beats_pref_speed():
    res = _resolve(_intent(100.0),
                   prefs={"line_time_s": 1.5, "scan_speed_nm_s": 50.0})
    assert res.configure_scan["line_time_s"] == 1.5
    assert res.trace["line_time_s"]["source"] == SOURCE_PREFS


# ── 优先级链:setpoint / bias ────────────────────────────────────────────────

def test_setpoint_chain_explicit_tier_prefs_keep():
    # explicit
    res = _resolve(_intent(explicit={"setpoint_a": 5e-11}),
                   prefs={"setpoint_pa": 200.0})
    assert res.set_setpoint == {"setpoint_a": 5e-11}
    assert res.trace["setpoint_a"]["source"] == SOURCE_EXPLICIT

    # 档位表
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256,
         "line_time_s": 1.0, "setpoint_a": 3e-11},
    ])
    res = _resolve(_intent(), prefs={"setpoint_pa": 200.0})
    assert res.set_setpoint == {"setpoint_a": 3e-11}
    assert res.trace["setpoint_a"]["source"] == SOURCE_TIER_OPERATOR

    # prefs(单位换算 pA → A)
    scan_policy.set_policy(None)
    res = _resolve(_intent(), prefs={"setpoint_pa": 200.0})
    assert res.set_setpoint["setpoint_a"] == pytest.approx(2e-10)
    assert res.trace["setpoint_a"]["source"] == SOURCE_PREFS

    # keep-current
    res = _resolve(_intent())
    assert res.set_setpoint is None
    assert res.trace["setpoint_a"]["source"] == SOURCE_KEEP


def test_bias_never_comes_from_the_tier_table():
    """bias 决定探测的电子态,不是尺度的函数 —— 表里根本没有它的位置。

    知识库里 L1 那个 "0.5-2 V" 是巡查建议,不是「1 µm 的图就该用 1 V」。
    """
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0,
         "bias_v": 2.0},        # 就算硬塞进去也必须被丢掉
    ])
    res = _resolve(_intent())
    assert res.set_bias is None
    assert res.trace["bias_v"]["source"] == SOURCE_KEEP


def test_bias_chain_is_explicit_then_prefs_then_keep():
    res = _resolve(_intent(explicit={"bias_v": -1.5}), prefs={"bias_v": 0.5})
    assert res.set_bias == {"bias_v": -1.5}
    assert res.trace["bias_v"]["source"] == SOURCE_EXPLICIT

    res = _resolve(_intent(), prefs={"bias_v": 0.5})
    assert res.set_bias == {"bias_v": 0.5}
    assert res.trace["bias_v"]["source"] == SOURCE_PREFS

    res = _resolve(_intent())
    assert res.set_bias is None


# ── PI 增益 ──────────────────────────────────────────────────────────────────

def test_pi_gains_only_come_from_the_tier_table_and_derive_i_from_p_over_t():
    """SetZCtrlGain 三个参数全必填,积分增益 I = P/T 不是独立自由度。"""
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0,
         "p_gain": 1.5e-11, "time_constant_s": 2e-4},
    ])
    res = _resolve(_intent())
    assert res.set_zctrl_gain["p_gain"] == 1.5e-11
    assert res.set_zctrl_gain["time_constant_s"] == 2e-4
    assert res.set_zctrl_gain["i_gain"] == pytest.approx(1.5e-11 / 2e-4)


def test_half_specified_pi_config_is_not_sent_and_says_what_is_missing():
    """只填一半的 PI 配置是不可执行的 —— 与其送一个必然被拒的调用,不如说清楚。"""
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0,
         "p_gain": 1.5e-11},
    ])
    res = _resolve(_intent())
    assert res.set_zctrl_gain is None
    assert any("time_constant_s" in w for w in res.warnings)


def test_zero_time_constant_does_not_divide_by_zero():
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0,
         "p_gain": 1.5e-11, "time_constant_s": 0.0},
    ])
    res = _resolve(_intent())
    assert res.set_zctrl_gain is None
    assert any("时间常数为 0" in w for w in res.warnings)


# ── purpose ──────────────────────────────────────────────────────────────────

def test_purpose_forces_a_tier_and_warns():
    """「在这 50 nm 快扫一眼」—— 尺寸是精扫的,参数要粗扫的。"""
    res = _resolve(_intent(50.0, purpose="survey"))
    assert res.tier_name == "survey"
    assert res.set_scan_buffer["pixels"] == 256
    assert any("强制换档" in w for w in res.warnings)


def test_purpose_matching_the_auto_tier_produces_no_warning():
    res = _resolve(_intent(50.0, purpose="highres"))
    assert res.tier_name == "highres"
    assert not any("强制换档" in w for w in res.warnings)


def test_unknown_purpose_falls_back_to_size_and_warns():
    res = _resolve(_intent(50.0, purpose="超高清"))
    assert res.tier_name == "highres"
    assert any("不是已知档名" in w for w in res.warnings)


def test_purpose_auto_and_blank_both_mean_by_size():
    for p in ("auto", "", "  AUTO "):
        assert _resolve(_intent(50.0, purpose=p)).tier_name == "highres"


# ── 参数卫生(clamp)与组合约束 ───────────────────────────────────────────────

def test_out_of_range_explicit_is_clamped_with_a_warning_not_rejected():
    """这一层是参数卫生:手滑的数字修剪掉并声明。真正该拒的越界归 SafetyGate。"""
    res = _resolve(_intent(explicit={"pixels": 99999}))
    assert res.set_scan_buffer["pixels"] == 4096
    assert any("pixels" in w and "修剪" in w for w in res.warnings)


def test_tip_speed_limit_slows_the_line_time():
    """像素、每线时间、帧宽单独看都合法,乘起来才知道针尖扫得多快。

    这一类组合约束正是模型必然漏掉的东西。
    """
    res = _resolve(_intent(1000.0, explicit={"line_time_s": 0.01}),
                   v_tip_max_m_s=1e-6)
    # 1 µm / 0.01 s = 100 µm/s,远超 1 µm/s 上限 → 放慢到 1.0 s
    assert res.configure_scan["line_time_s"] == pytest.approx(1.0)
    assert any("针尖横向速度" in w for w in res.warnings)
    assert "受针尖速度上限限制" in res.trace["line_time_s"]["human"]


def test_tip_speed_limit_does_not_fire_when_within_bounds():
    res = _resolve(_intent(100.0), v_tip_max_m_s=1e-6)
    assert not any("针尖横向速度" in w for w in res.warnings)


# ── trace 完整性 ─────────────────────────────────────────────────────────────

def test_every_resolved_parameter_appears_in_the_trace():
    """用户要能一眼看出「这个数字哪来的」。漏一个字段就是一个查不出来的谜。"""
    res = _resolve(_intent())
    for key in ("size_m", "center_x_m", "center_y_m", "pixels", "line_time_s",
                "angle_deg", "channels", "setpoint_a", "bias_v",
                "p_gain", "time_constant_s"):
        assert key in res.trace, f"trace 缺字段 {key}"
        assert "source" in res.trace[key]


def test_trace_sources_are_all_from_the_known_enum():
    known = {SOURCE_EXPLICIT, SOURCE_TIER_OPERATOR, SOURCE_TIER_FACTORY,
             SOURCE_PREFS, SOURCE_PREFS_DERIVED, SOURCE_DEFAULT, SOURCE_KEEP}
    res = _resolve(_intent(), prefs={"scan_lines": 256, "bias_v": 1.0})
    for key, rec in res.trace.items():
        assert rec["source"] in known, f"{key} 的来源 {rec['source']} 不在枚举里"


def test_summary_lines_pair_human_units_with_si():
    """人类单位必须紧跟 SI 值 —— 2026-07-27 坐标事故的教训。"""
    res = _resolve(_intent(100.0, explicit={"bias_v": 1.5, "setpoint_a": 1e-10}))
    text = "\n".join(res.summary_lines())
    assert "1e-07" in text and "100 nm" in text          # 尺寸
    assert "1e-10" in text and "100 pA" in text          # setpoint


def test_channels_default_and_override():
    assert _resolve(_intent()).configure_scan["channels"] == "Z,Current"
    assert _resolve(_intent()).trace["channels"]["source"] == SOURCE_DEFAULT
    res = _resolve(_intent(explicit={"channels": "Z,Current,LI Demod 1 X"}))
    assert res.configure_scan["channels"] == "Z,Current,LI Demod 1 X"
    assert res.trace["channels"]["source"] == SOURCE_EXPLICIT


# ── 纯函数性质 ────────────────────────────────────────────────────────────────

def test_resolver_is_pure_same_input_same_output():
    a = _resolve(_intent(250.0, explicit={"bias_v": 1.0}))
    b = _resolve(_intent(250.0, explicit={"bias_v": 1.0}))
    assert a.configure_scan == b.configure_scan
    assert a.set_scan_buffer == b.set_scan_buffer
    assert a.tier_name == b.tier_name


def test_resolver_does_not_mutate_the_intent():
    intent = _intent(explicit={"pixels": 99999})
    _resolve(intent)
    assert intent.explicit == {"pixels": 99999}


# ── 预览端点(设置界面用) ─────────────────────────────────────────────────────

def test_preview_returns_the_numbers_the_operator_needs_to_see():
    out = scan_resolver.preview(2e-7)
    assert out["tier_name"] == "roi"
    assert out["pixels"] == 256
    assert out["line_time_s"] == 0.8
    assert out["estimated_scan_s"] > 0
    assert isinstance(out["summary"], list) and out["summary"]


def test_preview_reflects_an_edited_table_immediately():
    """改表即见效果 —— 这是预览存在的唯一理由。"""
    before = scan_resolver.preview(2e-7)["pixels"]
    scan_policy.set_policy([
        {"name": "custom", "upper_size_m": None, "pixels": 64, "line_time_s": 0.2},
    ])
    after = scan_resolver.preview(2e-7)
    assert before == 256 and after["pixels"] == 64
    assert after["tier_name"] == "custom"
