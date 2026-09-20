# -*- coding: utf-8 -*-
"""AcquireBiasSeries 的合成测试：修改偏压时保持结阻。"""
from __future__ import annotations

import pytest

from mast.skills.builtins import AcquireBiasSeries
from mast.skills.builtins.bias_series import (
    _MAX_SETPOINT_A,
    _MIN_SETPOINT_A,
    setpoints_for,
)


# ── 恒结阻 ─────────────────────────────────────────────────────────────
def test_setpoint_tracks_bias_at_constant_resistance():
    """结阻不变 ⇒ setpoint 与 |bias| 成正比。这是整个技能的立身之本。"""
    plan = setpoints_for([2.0, 1.0, 0.5], 1e11)
    r = [abs(b) / s for b, s, _ in plan]
    assert r == pytest.approx([1e11] * 3, rel=1e-9)


def test_reproduces_the_table_used_on_the_rig():
    """08-26 手工用的那张表：100 GΩ 下 2V→20pA、1.5V→15pA、1V→10pA。"""
    got = {b: round(s * 1e12, 1) for b, s, _ in setpoints_for([2.0, 1.5, 1.0], 1e11)}
    assert got == {2.0: 20.0, 1.5: 15.0, 1.0: 10.0}


def test_the_condition_that_broke_the_rig_is_a_low_resistance_one():
    """出事的 1.0 V / 50 pA 就是 20 GΩ —— 比安全值近了 5 倍。
    这条把「事故条件」钉成一个数，免得以后有人以为问题出在偏压本身。"""
    (_, sp, _), = setpoints_for([1.0], 2e10)
    assert sp == pytest.approx(50e-12)


@pytest.mark.parametrize("bias", [2.0, -2.0, 0.5, -0.5])
def test_sign_of_bias_does_not_change_setpoint(bias):
    """setpoint 是电流大小，与偏压符号无关 —— 负偏压那半边必须给出同样的距离。"""
    (_, sp, _), = setpoints_for([bias], 1e11)
    (_, sp_pos, _), = setpoints_for([abs(bias)], 1e11)
    assert sp == pytest.approx(sp_pos)


# ── 夹紧必须如实报告 ────────────────────────────────────────────────────
def test_clamping_is_reported_not_silent():
    """夹过的那几档结阻不再等于目标值。**标记必须跟着数据走**，
    只写日志的话调用方会拿它们跟其余档横向比 —— 而它们的针尖距离不同。"""
    plan = setpoints_for([0.01, 2.0, 100.0], 1e11)
    flags = [c for _, _, c in plan]
    assert flags[0] is True and flags[2] is True and flags[1] is False


def test_clamps_stay_inside_the_envelope():
    for _, sp, _ in setpoints_for([0.001, 0.01, 1.0, 50.0, 500.0], 1e11):
        assert _MIN_SETPOINT_A <= sp <= _MAX_SETPOINT_A


# ── 参数校验 ───────────────────────────────────────────────────────────
def _errs(**kw):
    p = {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 20e-9,
         "biases_v": "2,1,-1,-2"}
    p.update(kw)
    return AcquireBiasSeries().validate_params(p)


def test_zero_bias_is_rejected():
    """0 V 在恒结阻下算出 0 电流，而且 0 偏压本来就没有隧穿。"""
    assert any("0 V" in e for e in _errs(biases_v="2,1,0,-1"))


def test_single_bias_is_rejected():
    assert any("至少要 2 档" in e for e in _errs(biases_v="1.0"))


def test_unparsable_bias_list_is_rejected():
    assert any("解析不了" in e for e in _errs(biases_v="2,abc,1"))


def test_a_normal_list_passes():
    assert _errs() == []


# ── 序列结构：首尾对照 ──────────────────────────────────────────────────
class _Ctx:
    """记录每一次 ScanAt 的参数；AssessFrameTrust 按脚本返回逐行 MAD。"""

    def __init__(self, mads=None):
        self.scans = []
        self.mads = list(mads or [])
        self.i = 0

    def run(self, name, params=None):
        params = params or {}
        if name == "ScanAt":
            self.scans.append(params)
            return type("R", (), {"success": True,
                                  "data": {"scan_path": "f%d.sxm" % len(self.scans)}})()
        if name == "AssessFrameTrust":
            mad = self.mads[self.i] if self.i < len(self.mads) else 12.0
            self.i += 1
            return type("R", (), {"success": True,
                                  "data": {"row_jump_mad_pm": mad, "tip_verdict": "stable",
                                           "rms_pm": 100.0}})()
        return type("R", (), {"success": True, "data": {}})()


def _run(ctx, **kw):
    p = {"center_x_m": 1e-7, "center_y_m": 2e-7, "size_m": 20e-9,
         "biases_v": "2,1,-1,-2", "junction_r_ohm": 1e11}
    p.update(kw)
    return AcquireBiasSeries().execute(ctx, p)


def test_series_is_bracketed_by_two_reference_frames():
    ctx = _Ctx()
    res = _run(ctx)
    tags = [f["tag"] for f in res.data["frames"]]
    assert tags[0] == "pre_drift" and tags[-1] == "post_drift"
    assert tags.count("series") == 4


def test_reference_frames_use_the_same_condition():
    """首尾两帧必须条件相同，否则它们对照不了任何东西。"""
    ctx = _Ctx()
    _run(ctx)
    first, last = ctx.scans[0], ctx.scans[-1]
    assert first["bias_v"] == last["bias_v"]
    assert first["setpoint_a"] == pytest.approx(last["setpoint_a"])


def test_every_frame_stays_at_the_same_spot():
    ctx = _Ctx()
    _run(ctx)
    assert {(s["center_x_m"], s["center_y_m"]) for s in ctx.scans} == {(1e-7, 2e-7)}


def test_all_series_frames_share_one_resistance():
    ctx = _Ctx()
    _run(ctx)
    rs = [abs(s["bias_v"]) / s["setpoint_a"] for s in ctx.scans]
    assert rs == pytest.approx([1e11] * len(rs), rel=1e-9)


# ── 漂移对照的判读 ──────────────────────────────────────────────────────
def test_drift_check_flags_a_tip_that_got_worse():
    ctx = _Ctx(mads=[12.0] + [12.0] * 4 + [900.0])
    res = _run(ctx)
    d = res.data["drift_check"]
    assert d["comparable"] is False and "变差" in d["note"]


def test_drift_check_flags_a_tip_that_got_better():
    """08-26 真实发生的那种：首 158 → 尾 20 pm。
    不标出来的话，这段变化会被读成偏压的效应。"""
    ctx = _Ctx(mads=[158.0] + [60.0] * 4 + [20.0])
    res = _run(ctx)
    d = res.data["drift_check"]
    assert d["comparable"] is False and "变稳" in d["note"]


def test_drift_check_passes_a_steady_tip():
    ctx = _Ctx(mads=[12.0] * 6)
    d = _run(_Ctx(mads=[12.0] * 6)).data["drift_check"]
    assert d["comparable"] is True and "可以横向比" in d["note"]


# ── execute 不许假设 validate_params 跑过 ───────────────────────────────
@pytest.mark.parametrize("bad", ["0", "", "1.0", "0,0"])
def test_bad_bias_list_returns_a_result_not_an_exception(bad):
    """同 barrier_map：坏参数要**返回** SkillResult，不能让 execute 炸。"""
    res = AcquireBiasSeries().execute(
        _Ctx(), {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 20e-9, "biases_v": bad})
    assert res.success is False
    assert "至少要 2 档" in (res.error or "") or "解析不了" in (res.error or "")


def test_zero_volts_are_dropped_from_a_mixed_list():
    """混着 0 V 的序列：把 0 剔掉后仍有 ≥2 档就照常跑，不是整条拒绝。"""
    ctx = _Ctx()
    res = AcquireBiasSeries().execute(
        ctx, {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": 20e-9,
              "biases_v": "2,0,-2", "junction_r_ohm": 1e11})
    assert res.success
    assert [p["bias_v"] for p in res.data["plan"]] == [2.0, -2.0]
