# -*- coding: utf-8 -*-
"""ClassifyUnexplainedCurrent：结合零偏压底噪和偏压依赖区分电流来源。

独立幂律夹具取 I(V)=0.5 pA×V⁸，零偏压读数单独模拟底噪；
测试覆盖场发射、串扰、普通结电流、饱和及不能静态复现的瞬态。
"""
from __future__ import annotations

import pytest

from mast.skills.builtins import ClassifyUnexplainedCurrent
from mast.skills.builtins.current_origin import (
    _ABS_FLOOR_A,
    _N_FIELD_EMISSION,
    classify,
    effective_exponent,
    noise_floor_from_zero,
)

#: 独立合成幂律与零偏压底噪。
SYNTHETIC = [(2.0, 128e-12), (1.0, 0.5e-12), (0.0, -0.02e-12)]


# ── 指数拟合 ───────────────────────────────────────────────────────────
def test_synthetic_power_law_comes_out_as_field_emission():
    """解析幂律输入应恢复已知指数，并进入高指数分类。"""
    i_zero = SYNTHETIC[-1][1]
    floor = noise_floor_from_zero(i_zero)
    n, used = effective_exponent(SYNTHETIC, floor)
    assert len(used) == 2, "1.00 V 的 0.5 pA 必须算数据不是噪声"
    assert n == pytest.approx(8.0, abs=1e-10)
    verdict, msg = classify(i_zero, n, 128e-12)
    assert verdict == "field_emission"
    assert "场发射" in msg


def test_hardcoded_1pa_floor_would_discard_a_synthetic_point():
    """固定 1 pA 下限会剔除合成幂律中的较小有效点；自适应底噪应保留它。"""
    n, used = effective_exponent(SYNTHETIC, 1e-12)
    assert len(used) == 1 and n is None


def test_floor_is_calibrated_from_the_zero_bias_reading():
    assert noise_floor_from_zero(-0.02e-12) == pytest.approx(3 * 0.02e-12)
    assert noise_floor_from_zero(0.0) == _ABS_FLOOR_A
    assert noise_floor_from_zero(None) == _ABS_FLOOR_A, "读不到时退回绝对下限，不是 0"


def test_exponent_needs_two_points():
    assert effective_exponent([(1.0, 5e-11)], _ABS_FLOOR_A)[0] is None
    assert effective_exponent([], _ABS_FLOOR_A)[0] is None


def test_linear_current_gives_exponent_one():
    n, _ = effective_exponent([(0.5, 25e-12), (1.0, 50e-12), (2.0, 100e-12)],
                              _ABS_FLOOR_A)
    assert n == pytest.approx(1.0, abs=0.05)


# ── 判别表 ─────────────────────────────────────────────────────────────
def test_bias_independent_current_is_not_a_junction_current():
    """串扰/偏置长这样：0 V 下就有，且随偏压几乎不变。"""
    pts = [(2.0, 20.1e-12), (1.0, 20.0e-12), (0.0, 19.9e-12)]
    floor = noise_floor_from_zero(19.9e-12)
    n, _ = effective_exponent(pts, floor)
    verdict, msg = classify(19.9e-12, n, 20.1e-12)
    assert verdict == "not_a_junction_current"
    assert "串扰" in msg


def test_ordinary_tunnelling_is_called_a_junction_current():
    floor = noise_floor_from_zero(0.01e-12)
    n, _ = effective_exponent([(1.0, 50e-12), (0.5, 24e-12)], floor)
    verdict, _ = classify(0.01e-12, n, 50e-12)
    assert verdict == "junction_current"


def test_nothing_measurable_is_not_called_field_emission():
    verdict, _ = classify(0.0, None, 1e-15)
    assert verdict == "no_measurable_current"


def test_unreadable_zero_point_refuses_rather_than_guessing():
    """0 V 读不到就判不了 —— 那一点正是判别点。"""
    verdict, msg = classify(None, 9.0, 100e-12)
    assert verdict == "undetermined"
    assert "0 V" in msg


def test_offset_plus_real_dependence_is_mixed_not_either_one():
    floor = noise_floor_from_zero(10e-12)
    verdict, _ = classify(10e-12, 5.0, 900e-12)   # 90 倍 ⇒ 不是「与偏压无关」
    assert verdict == "mixed"


def test_field_emission_threshold_is_above_ordinary_band_curvature():
    """卡 2 会把 1–2 V 上正常的能带超线性判成场发射。"""
    assert _N_FIELD_EMISSION >= 3.0
    assert classify(0.0, 2.5, 50e-12)[0] == "junction_current"


# ── execute：桩 ────────────────────────────────────────────────────────
class _Ctx:
    """currents: dict{bias -> A} 或 callable(bias)->A。"""

    def __init__(self, currents, bias0=2.0, fb_off_ok=True, fb_on=True,
                 fb_readable=True):
        self.currents = currents
        self.bias = bias0
        self.bias0 = bias0
        self.fb_off_ok = fb_off_ok
        self.fb_on = fb_on
        self.fb_readable = fb_readable
        self.calls: list[tuple] = []

    def run(self, name, params=None):
        params = params or {}
        self.calls.append((name, dict(params)))
        if name == "SetBias":
            self.bias = float(params["bias_v"])
            return type("R", (), {"success": True, "data": {}})()
        if name == "GetBias":
            # 真机走 reply_scalar；这里直接给数（GetCurrent 那条元组坑在别处已钉）
            return type("R", (), {"success": True,
                                  "data": {"bias_v": self.bias0}})()
        if name == "GetCurrent":
            c = self.currents
            v = c(self.bias) if callable(c) else c.get(round(self.bias, 6))
            return type("R", (), {"success": True, "data": {"current_a": v}})()
        if name == "GetZControllerState":
            return type("R", (), {
                "success": True,
                "data": {"controller_on": self.fb_on if self.fb_readable else None}})()
        if name == "ZControllerOnOff":
            want = bool(params.get("enable"))
            if not want and not self.fb_off_ok:
                return type("R", (), {"success": True,
                                      "data": {"z_controller_on": True,
                                               "verified": True}})()
            self.fb_on = want
            return type("R", (), {"success": True,
                                  "data": {"z_controller_on": want,
                                           "verified": True}})()
        return type("R", (), {"success": True, "data": {}})()

    def names(self):
        return [n for n, _ in self.calls]


def _run(ctx, **kw):
    sk = ClassifyUnexplainedCurrent()
    sk._settle_s = 0.0
    return sk.execute(ctx, kw)


SYNTHETIC_MAP = {2.0: 128e-12, 1.0: 0.5e-12, 0.5: 0.001953125e-12, 0.0: -0.02e-12}


def test_end_to_end_on_the_synthetic_power_law():
    res = _run(_Ctx(SYNTHETIC_MAP))
    assert res.success is True
    assert res.data["verdict"] == "field_emission"
    assert res.data["exponent_n"] > _N_FIELD_EMISSION


# ── ★ 饱和必须排在判别之前 ─────────────────────────────────────────────
def test_saturation_is_checked_before_the_table_and_does_not_read_as_crosstalk():
    """饱和时每个偏压读数都一样 —— 落进判别表就成了「与偏压无关 ⇒ 串扰」，正好判反。"""
    ctx = _Ctx(lambda b: 1.1e-8)
    res = _run(ctx)
    assert res.data["verdict"] == "amplifier_saturated"
    assert res.data["next_skill"] == "RecoverTipFromSaturation"
    # 关键：不能已经开始扫偏压
    assert "SetBias" not in ctx.names()
    assert "ZControllerOnOff" not in ctx.names()


# ── 0 V 必须在 ─────────────────────────────────────────────────────────
def test_zero_bias_is_added_when_the_caller_forgot_it():
    ctx = _Ctx(SYNTHETIC_MAP)
    res = _run(ctx, test_biases_v="2.0,1.0")
    assert res.data["zero_bias_added"] is True
    assert any(abs(p["bias_v"]) < 1e-9 for p in res.data["points"])


def test_zero_bias_not_flagged_when_already_present():
    res = _run(_Ctx(SYNTHETIC_MAP), test_biases_v="2.0,1.0,0.0")
    assert res.data["zero_bias_added"] is False


# ── 反馈 ───────────────────────────────────────────────────────────────
def test_feedback_is_off_before_the_first_bias_is_set():
    """反馈开着时 Z 去追 setpoint，I(V) 描述的是反馈环不是结。"""
    ctx = _Ctx(SYNTHETIC_MAP)
    _run(ctx)
    names = ctx.names()
    assert names.index("ZControllerOnOff") < names.index("SetBias")


def test_refuses_when_feedback_cannot_be_turned_off():
    ctx = _Ctx(SYNTHETIC_MAP, fb_off_ok=False)
    res = _run(ctx)
    assert res.success is False
    assert "反馈" in (res.error or "")
    assert "SetBias" not in ctx.names(), "关不掉反馈就不该开始扫偏压"


def test_feedback_and_bias_are_restored():
    ctx = _Ctx(SYNTHETIC_MAP)
    res = _run(ctx)
    assert ctx.fb_on is True, "反馈必须还原"
    assert ctx.bias == pytest.approx(2.0), "偏压必须还原到工作点"
    assert res.data["bias_restored_to"] == pytest.approx(2.0)


def test_bias_is_restored_even_when_the_sweep_blows_up():
    """诊断把偏压留在 0 V 上比不做诊断更糟。"""

    class Boom(_Ctx):
        def run(self, name, params=None):
            if name == "GetCurrent" and self.bias == 1.0:
                raise RuntimeError("链路断了")
            return super().run(name, params)

    ctx = Boom(SYNTHETIC_MAP)
    with pytest.raises(RuntimeError):
        _run(ctx)
    assert ctx.bias == pytest.approx(2.0)
    assert ctx.fb_on is True


# ── 静态复现 / transient ───────────────────────────────────────────────
def test_current_that_does_not_reproduce_statically_is_called_transient():
    """串扰假设说的是「动的时候」—— 静态 I(V) 根本看不见它。

    这不是「没有」，是「没在这儿」。
    """
    quiet = {2.0: 0.02e-12, 1.0: 0.01e-12, 0.5: 0.01e-12, 0.0: -0.01e-12}
    res = _run(_Ctx(quiet), observed_current_a=2e-11)
    assert res.data["verdict"] == "transient"
    assert res.data["reproduced"] is False
    assert "进行中重测" in res.data["message"]
    assert res.data["static_verdict"] is not None


def test_current_that_does_reproduce_keeps_the_static_verdict():
    """反面：静态复现得出来时不该判 transient，否则上一条会恒真。"""
    res = _run(_Ctx(SYNTHETIC_MAP), observed_current_a=100e-12)
    assert res.data["reproduced"] is True
    assert res.data["verdict"] == "field_emission"
    assert "static_verdict" not in res.data


def test_observed_current_is_optional():
    res = _run(_Ctx(SYNTHETIC_MAP))
    assert "reproduced" not in res.data
    assert res.data["verdict"] == "field_emission"


# ── 读不到 ≠ 没有 ──────────────────────────────────────────────────────
def test_unreadable_currents_do_not_become_zero():
    res = _run(_Ctx(lambda b: None))
    assert res.data["verdict"] == "undetermined"
    assert res.data["i_zero_a"] is None


def test_bad_bias_string_is_rejected():
    res = _run(_Ctx(SYNTHETIC_MAP), test_biases_v="2.0,abc")
    assert res.success is False and "解析" in (res.error or "")


# ── 还原的对象是「进来时的样子」，不是「一般该是的样子」 ────────────────
def test_feedback_that_was_off_stays_off():
    """用户本来关着反馈（手动操作／已退针）时，不能替他开回来 ——
    开反馈会驱动 Z 去够 setpoint。"""
    ctx = _Ctx(SYNTHETIC_MAP, fb_on=False)
    res = _run(ctx)
    assert res.success is True
    assert ctx.fb_on is False, "进来时是关的，出去时也必须是关的"
    assert res.data["feedback_was_on"] is False


def test_refuses_when_the_initial_feedback_state_cannot_be_read():
    """读不到初态就无从还原 —— 猜错的代价是一根针尖。"""
    ctx = _Ctx(SYNTHETIC_MAP, fb_readable=False)
    res = _run(ctx)
    assert res.success is False
    assert "还原" in (res.error or "")
    assert "SetBias" not in ctx.names(), "读不到初态就不该开始动"
