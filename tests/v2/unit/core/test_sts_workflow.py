"""谱学条件表 —— S4 STS 设计 D19 / §3.1。

纯函数、零 IO,所以这份测试没有一个 mock。它盯的是四件写错了**零报错**的事:

1. 出厂表里凭空出现了一组稳定条件 —— 整批谱会正常落盘、正常被判据吃掉,而它们
   测的是一个没人给过的假设。
2. 越界的覆写被**夹紧**或被**丢弃** —— 两种兜底都合理得让人看不出兜底发生了:
   调用方以为自己设的是 X,实际跑的是别的数,而结果里没有任何字段会说出这件事。
3. 认不出来的字段名被安静地忽略 —— 「我明明设了它」和「它根本没收到」长得一样。
4. ``ConfigureLockIn`` 的幅度/频率漏给 —— 省略的那个会被报成 0.0,
   而一个报出来的 0.0 与「没动」在结果里长得一模一样。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \
      tests/v2/unit/core/test_sts_workflow.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.core import sts_workflow as W  # noqa: E402
from mast.core.sts_workflow import (  # noqa: E402
    CONDITIONS,
    DEFAULT_CONDITION,
    REJECT_NOT_A_NUMBER,
    REJECT_OUT_OF_BOUNDS,
    REJECT_UNKNOWN_FIELD,
    SAMPLE_FACT_FIELDS,
    STSConditionSpec,
    crosses_zero,
    list_conditions,
    resolve_condition,
    resolve_series,
)

#: 一组填齐了的条件 —— 每条用例自己声明它的数字,不靠出厂表碰巧有值。
FULL = dict(stab_bias_v=-1.0, stab_setpoint_a=100e-12, start_v=-1.0, end_v=1.0)


def _full_spec(**kw) -> STSConditionSpec:
    return STSConditionSpec(**{**FULL, **kw})


# ═══════════════════════════════════════════════════════════════════════
# 1. 出厂表里**没有**样品事实 —— 而且刻意没有
# ═══════════════════════════════════════════════════════════════════════

def test_the_factory_group_is_deliberately_uncalibrated():
    """出厂的 ``default`` 组四个样品事实全是 ``None``，用不了。

    这不是「还没做完」，是「这个问题现在没有答案」：稳定偏压该取多少取决于这块
    样品的能隙、表面态、针尖状态。编一组数出来，流程会照跑不误、每条谱都「成功」。
    同一条纪律在 ``scan_prep_thresholds``（造 profile = 伪造标定）与
    ``domain_reference``（没参照系就永远不出 label）里各写过一遍。
    """
    spec = CONDITIONS[DEFAULT_CONDITION]
    assert spec.missing_fields == SAMPLE_FACT_FIELDS
    assert spec.calibrated is False
    res = resolve_condition()
    assert res.usable is False
    for name in SAMPLE_FACT_FIELDS:
        assert name in res.refusal_detail(), f"拒绝没点名 {name}"


def test_none_is_not_zero_for_a_stabilisation_bias():
    """``None`` 是「未标定」，0 V 是一个**真实答案** —— 而且是个危险的答案。

    恒流反馈下 |V| 越小针尖被推得越近。把「没给」折叠成 0.0，得到的不是一个保守
    默认值，是一次把针尖往样品里推的设置。
    """
    assert STSConditionSpec().stab_bias_v is None
    zeroed = _full_spec(stab_bias_v=0.0)
    assert zeroed.calibrated is True, "显式的 0 是给过了，不是没给"
    assert "未标定" in STSConditionSpec().describe()
    assert "未标定" not in zeroed.describe()


def test_the_instrument_side_fields_do_have_factory_values():
    """点数/调制频率/整定时间**有**出厂值 —— 它们是仪器与判据那一侧的事实。

    界不是本模块发明的：400 点与 ``ConfigureSTS`` 的 2..10000 同界，调制幅度上限
    与 ``ConfigureLockIn.amplitude_v`` 同界。两个真源的界迟早会不一致，而不一致的
    那一天没有人会发现。
    """
    spec = STSConditionSpec()
    assert spec.num_points == 400 and spec.mod_freq_hz == 713.0
    assert spec.settle_s == 1.0 and spec.z_offset_m == 0.0
    assert W._BOUNDS["num_points"] == (2, 10000)
    assert W._BOUNDS["mod_amp_v"][1] == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 2. 越界 = **拒绝**，不夹紧、不丢弃
# ═══════════════════════════════════════════════════════════════════════

def _codes(res):
    return {r["field"]: r["code"] for r in res.rejections}


def test_an_out_of_bounds_override_is_rejected_not_clamped_and_not_dropped():
    """三种处置里只有一种是对的。

    * **夹紧**：调用方以为自己设的是 12 V，实际跑 10 V，没有字段说出这件事；
    * **丢弃**（``special_tip_workflow`` 的做法）：连「你设过一个越界值」都不说；
    * **拒绝**：说出来，由消费方停下。这里的覆写来自一次**活的调用**，不是流程
      作者写在代码里的一个 dict —— 那才是丢弃成立的场合。
    """
    res = resolve_condition(DEFAULT_CONDITION, {**FULL, "stab_bias_v": 12.0})
    assert _codes(res) == {"stab_bias_v": REJECT_OUT_OF_BOUNDS}
    assert res.usable is False
    # 关键：它**没有**被夹到 10.0，也没有被悄悄换回出厂值。
    assert res.spec.stab_bias_v != 10.0
    assert "不夹紧" in res.refusal_detail()


def test_an_unknown_field_is_rejected_not_silently_ignored():
    """认不出来的键**不会被安静地丢掉**。

    「我明明设了它」和「它根本没收到」长得一模一样 —— 而这条链上没有任何东西会
    在事后指出差别。加一个可编辑的键永远是双边动作，本仓记过四次。
    """
    res = resolve_condition(DEFAULT_CONDITION, {**FULL, "stab_bais_v": -1.0})
    assert _codes(res) == {"stab_bais_v": REJECT_UNKNOWN_FIELD}
    assert "已知字段" in res.refusal_detail()


def test_a_non_number_is_rejected():
    res = resolve_condition(DEFAULT_CONDITION, {**FULL, "num_points": "四百"})
    assert _codes(res) == {"num_points": REJECT_NOT_A_NUMBER}
    for bad in (float("nan"), float("inf")):
        r = resolve_condition(DEFAULT_CONDITION, {**FULL, "settle_s": bad})
        assert _codes(r) == {"settle_s": REJECT_NOT_A_NUMBER}, bad


def test_none_means_not_given_not_set_to_empty():
    """``None`` 一律当「没给」—— 与 ``special_tip_workflow`` 一致。"""
    res = resolve_condition(DEFAULT_CONDITION, {**FULL, "settle_s": None})
    assert res.rejections == ()
    assert res.spec.settle_s == STSConditionSpec().settle_s


def test_an_int_field_is_rounded_not_truncated():
    res = resolve_condition(DEFAULT_CONDITION, {**FULL, "num_points": 255.6})
    assert res.spec.num_points == 256 and isinstance(res.spec.num_points, int)


# ═══════════════════════════════════════════════════════════════════════
# 3. 组名解析
# ═══════════════════════════════════════════════════════════════════════

def test_an_unknown_group_name_is_refused_with_the_list_of_known_ones():
    """「不知道有哪些」是这一类拒绝里最没用的回答。"""
    res = resolve_condition("wide_survey_v2")
    assert res.spec is None and res.usable is False
    assert "wide_survey_v2" in res.problem
    assert DEFAULT_CONDITION in res.problem
    assert res.known_names == list_conditions()


def test_an_unknown_group_is_not_fuzzy_matched_to_a_similar_one():
    """就近匹配会让一次**跑错条件**的运行看上去完全正常。"""
    res = resolve_condition("defaul")           # 差一个字母
    assert res.spec is None, "被匹配到 default 了 —— 那批谱的条件从此没人说得清"


def test_an_empty_name_falls_back_to_the_declared_default_constant():
    """空组名走 :data:`DEFAULT_CONDITION`，而它与技能参数默认值是**同一个常量**。

    两处各写一遍字面量 ``"default"``，改名那天只会改到一处。
    """
    assert resolve_condition("").name == DEFAULT_CONDITION
    assert resolve_condition(None).name == DEFAULT_CONDITION


def test_the_table_can_be_swapped_for_a_test_without_touching_the_real_one():
    """``table=`` 是给测试与 campaign spec 的口，不改全局那张表。"""
    tbl = {"x": _full_spec(label="x")}
    assert resolve_condition("x", table=tbl).usable is True
    assert resolve_condition("x").spec is None, "测试用的表漏进了全局"


# ═══════════════════════════════════════════════════════════════════════
# 4. 翻译成子技能的参数
# ═══════════════════════════════════════════════════════════════════════

def test_the_lockin_config_always_carries_both_numbers():
    """幅度与频率**永远都在** —— 省略哪个，``ConfigureLockIn`` 就把它报成 0.0。

    那个 0.0 与「没动」在结果里长得一模一样。相位一个字都不给：本机的
    调制器根本没有相位字段，那次写入会被固件无条件拒绝。
    """
    cfg = _full_spec(mod_on=False).lockin_config()
    assert set(cfg) == {"mod_on", "amplitude_v", "frequency_hz"}
    assert cfg["mod_on"] is False
    assert cfg["amplitude_v"] > 0, "关调制不等于把幅度报成 0"
    assert "phase_deg" not in cfg


def test_the_sweep_config_sends_the_z_offset_explicitly():
    """``ConfigureSTS`` 省略 z_offset 时按 0 处理 —— 显式的 0 与「没给」要分得开。"""
    cfg = _full_spec().sweep_config()
    assert cfg == {"start_v": -1.0, "end_v": 1.0, "num_points": 400,
                   "z_offset_m": 0.0}


def test_bias_settle_params_are_for_bias_settle_change_not_setbias():
    """键名是 ``BiasSettleChange`` 的，不是 ``SetBias`` 的。"""
    p = _full_spec(settle_s=2.0).bias_settle_params()
    assert p == {"bias_v": -1.0, "settle_s": 2.0}


def test_a_zero_width_window_is_called_out():
    """零宽窗口不是一次扫描 —— 它会正常落盘、正常被判据吃掉，只是一个点都没扫过。"""
    spec = _full_spec(start_v=0.5, end_v=0.5)
    assert spec.window_problem()
    assert any("零宽" in p for p in spec.problems())
    assert spec.calibrated is True, "字段是齐的，坏的是窗口本身 —— 两件事"


def test_missing_fields_do_not_get_double_reported_as_a_window_problem():
    """缺字段由 ``missing_fields`` 负责，不在窗口那条里重复报一遍。"""
    assert STSConditionSpec().window_problem() == ""
    assert len(STSConditionSpec().problems()) == 1


# ═══════════════════════════════════════════════════════════════════════
# 5. MLS 透传
# ═══════════════════════════════════════════════════════════════════════

def test_mls_segments_are_transposed_and_passed_through_untouched():
    """段内结构继续由 Nanonis 的 MLS 负责 —— S4 不建第二套分段抽象（D18）。

    等长与偏压边界的校验在 ``SetSTSMLSVals`` 里（那是它的归属地）。在这里再写一份
    就有了两个真源，而两份校验迟早会不一致。
    """
    segs = ({"bias_start_v": -1.0, "bias_end_v": 0.0, "initial_settling_s": 0.1,
             "settling_s": 0.01, "integration_s": 0.02, "steps": 100,
             "lockin_run": 1},
            {"bias_start_v": 0.0, "bias_end_v": 1.0, "initial_settling_s": 0.1,
             "settling_s": 0.01, "integration_s": 0.05, "steps": 300,
             "lockin_run": 0})
    arrays = _full_spec(mls_segments=segs).mls_arrays()
    assert arrays["bias_start_v"] == [-1.0, 0.0]
    assert arrays["steps"] == [100, 300]
    assert arrays["lockin_run"] == [1, 0]
    assert all(len(v) == 2 for v in arrays.values())


def test_no_mls_segments_means_no_mls_call():
    assert _full_spec().mls_arrays() is None
    assert _full_spec(mls_segments=()).mls_arrays() is None


# ═══════════════════════════════════════════════════════════════════════
# 6. 序列
# ═══════════════════════════════════════════════════════════════════════

def test_one_bad_name_does_not_take_down_the_rest_of_the_series():
    """逐条独立解析：一个坏组名不该废掉整条序列。"""
    tbl = {"a": _full_spec(label="a"), "b": _full_spec(label="b")}
    out = resolve_series(["a", "nope", "b"], table=tbl)
    assert [r.usable for r in out] == [True, False, True]


def test_crossing_zero_is_reported_but_reading_nothing_is_not():
    """穿零要指出来（D20）；而**读不到**不是「不穿零」。

    恒流反馈下偏压穿零会把针尖推进样品，它是宽偏压序列的必经之路 —— 要紧的是走
    ``BiasSettleChange`` 并把这一步**说出来**，事后才能把一次针尖变化与它对上。
    """
    lo, hi = _full_spec(stab_bias_v=-1.0), _full_spec(stab_bias_v=1.0)
    assert crosses_zero(lo, hi) is True and crosses_zero(hi, lo) is True
    assert crosses_zero(lo, _full_spec(stab_bias_v=-0.2)) is False
    # 一端未标定 ⇒ 说不出它穿没穿 —— 报 False，但那是「不知道」不是「没穿」，
    # 而未标定的条件本来就会被消费方拒绝，走不到下发。
    assert crosses_zero(lo, STSConditionSpec()) is False


def test_zero_itself_is_not_a_crossing():
    """0 V 两侧才算穿零；停在 0 上是另一件事（而它本身就危险）。"""
    assert crosses_zero(_full_spec(stab_bias_v=0.0),
                        _full_spec(stab_bias_v=1.0)) is False


# ═══════════════════════════════════════════════════════════════════════
# 7. 变异验证 —— 先证明变异已应用，再证明测试红了
# ═══════════════════════════════════════════════════════════════════════

def test_mutation_clamping_instead_of_rejecting_turns_its_test_red():
    """把越界**夹紧**（而不是拒绝）—— 也就是最诱人的那种兜底。"""
    orig = W._coerce

    def _clamping(field, value):
        val, bad = orig(field, value)
        if bad is not None and bad["code"] == REJECT_OUT_OF_BOUNDS:
            lo, hi = W._BOUNDS[field]
            return max(lo, min(hi, float(value))), None
        return val, bad

    W._coerce = _clamping
    try:
        probe = resolve_condition(DEFAULT_CONDITION, {**FULL, "stab_bias_v": 12.0})
        assert probe.spec.stab_bias_v == 10.0 and probe.rejections == ()  # 变异已应用
        with pytest.raises(AssertionError):
            test_an_out_of_bounds_override_is_rejected_not_clamped_and_not_dropped()
    finally:
        W._coerce = orig


def test_mutation_dropping_unknown_fields_turns_its_test_red():
    """把认不出来的键**安静地丢掉** —— ``special_tip_workflow`` 今天的做法。"""
    orig = resolve_condition

    def _dropping(name=None, overrides=None, *, table=None):
        res = orig(name, overrides, table=table)
        keep = tuple(r for r in res.rejections
                     if r["code"] != REJECT_UNKNOWN_FIELD)
        return type(res)(name=res.name, spec=res.spec, rejections=keep,
                         problem=res.problem, known_names=res.known_names)

    W.resolve_condition = _dropping
    try:
        assert W.resolve_condition(
            DEFAULT_CONDITION, {**FULL, "nope": 1}).rejections == ()  # 变异已应用
        globals()["resolve_condition"] = _dropping
        with pytest.raises(AssertionError):
            test_an_unknown_field_is_rejected_not_silently_ignored()
    finally:
        W.resolve_condition = orig
        globals()["resolve_condition"] = orig


def test_mutation_giving_the_factory_group_a_stabilisation_turns_its_test_red():
    """给出厂组编一组稳定条件 —— 整批谱会正常落盘，测的是没人给过的假设。"""
    orig = CONDITIONS[DEFAULT_CONDITION]
    CONDITIONS[DEFAULT_CONDITION] = _full_spec(label=DEFAULT_CONDITION)
    try:
        assert resolve_condition().usable is True          # 变异已应用
        with pytest.raises(AssertionError):
            test_the_factory_group_is_deliberately_uncalibrated()
    finally:
        CONDITIONS[DEFAULT_CONDITION] = orig


def test_mutation_omitting_the_lockin_amplitude_turns_its_test_red():
    """漏给幅度 —— ``ConfigureLockIn`` 会把它报成 0.0，而那与「没动」长得一样。"""
    orig = STSConditionSpec.lockin_config

    def _partial(self):
        cfg = orig(self)
        if not self.mod_on:
            cfg.pop("amplitude_v", None)     # 「反正关了」
        return cfg

    STSConditionSpec.lockin_config = _partial
    try:
        assert "amplitude_v" not in _full_spec(mod_on=False).lockin_config()
        with pytest.raises(AssertionError):
            test_the_lockin_config_always_carries_both_numbers()
    finally:
        STSConditionSpec.lockin_config = orig
