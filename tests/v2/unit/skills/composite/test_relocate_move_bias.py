# -*- coding: utf-8 -*-
"""横移前降低偏压，并保持独立电流保护；测试验证顺序、恢复和读取失败行为。"""
from __future__ import annotations

import pytest

from mast.skills.composite.relocate_coarse_xy import (
    _MOVE_BIAS_V,
    _MOVE_DANGER_CURRENT_A,
    RelocateCoarseXY,
)


class _Ctx:
    """记录 Bias_Set 序列；Bias_Get 替身必须使用协议三段信封，确保解析路径被真实执行。"""

    def __init__(self, bias_now=2.0, get_fails=False):
        self.bias_now = bias_now
        self.get_fails = get_fails
        self.sets: list[float] = []
        self.calls: list[str] = []

    def safe_call(self, verb, *args):
        self.calls.append(verb)
        if verb == "Bias_Get":
            if self.get_fails:
                return type("R", (), {"error": "boom", "return_value": None})()
            # 使用协议信封形状的合成回包。
            return type("R", (), {"error": "",
                                  "return_value": (0, 0, [self.bias_now])})()
        if verb == "Bias_Set":
            self.sets.append(float(args[0]))
            self.bias_now = float(args[0])
        return type("R", (), {"error": "", "return_value": None})()

    def run(self, name, params=None):
        self.calls.append(name)
        return type("R", (), {"success": True, "data": {}})()


def _fresh():
    sk = RelocateCoarseXY()
    sk._call_log = []
    sk._panic_failures = []
    sk._bias_restore = None
    return sk


# ── 降压 ───────────────────────────────────────────────────────────────
def test_move_bias_and_current_guard_keep_their_configuration_contract():
    """验证低偏压配置与独立电流保护阈值，不声称该偏压对所有仪器都避免场发射。"""
    assert _MOVE_BIAS_V <= 1.0
    assert _MOVE_DANGER_CURRENT_A == pytest.approx(1e-11)


def test_clear_phase_lowers_the_bias():
    sk = _fresh()
    ctx = _Ctx(bias_now=2.0)
    sk._lower_bias_for_move(ctx)
    assert _MOVE_BIAS_V in ctx.sets, "清障阶段没有把偏压降下来"
    assert sk._bias_restore == pytest.approx(2.0), "没有记住原值，就无法归还"


def test_bias_is_lowered_before_anything_reads_current():
    """顺序要紧：清障阶梯自己也在读电流，降压必须排在它前面。

    读源码而不是跑 _phase_clear —— 后者要拉起退针阶梯、温度、qPlus 一整套，
    那些与本条无关，把它们桩全反而让这条测试变脆。
    """
    import inspect

    from mast.skills.composite import relocate_coarse_xy as mod

    src = inspect.getsource(mod.RelocateCoarseXY._phase_clear)
    i_lower = src.index("_lower_bias_for_move")
    for later in ("AutoApproach_OnOffSet", "Motor_StopMove", "Scan_Action"):
        assert i_lower < src.index(later), f"降压排在了 {later} 之后"


@pytest.mark.parametrize("bias_now", [0.5, 0.2, 0.0, -0.3])
def test_already_low_bias_is_left_alone(bias_now):
    """本来就在安全值以下就不碰 —— 「谁改谁恢复」的前提是只改该改的。"""
    sk = _fresh()
    ctx = _Ctx(bias_now=bias_now)
    sk._lower_bias_for_move(ctx)
    assert ctx.sets == []
    assert sk._bias_restore is None


def test_negative_imaging_bias_is_also_lowered():
    """负偏压一样会场发射 —— 判据是绝对值。"""
    sk = _fresh()
    ctx = _Ctx(bias_now=-2.0)
    sk._lower_bias_for_move(ctx)
    assert _MOVE_BIAS_V in ctx.sets
    assert sk._bias_restore == pytest.approx(-2.0)


def test_unreadable_bias_does_not_explode_the_phase():
    """读不到偏压时不该让整条流程炸 —— 但也不该假装记住了原值。"""
    sk = _fresh()
    ctx = _Ctx(get_fails=True)
    sk._lower_bias_for_move(ctx)
    assert sk._bias_restore is None


# ── 归还 ───────────────────────────────────────────────────────────────
def test_restore_puts_the_original_bias_back():
    sk = _fresh()
    ctx = _Ctx(bias_now=2.0)
    sk._lower_bias_for_move(ctx)
    ctx.sets.clear()
    sk._restore_bias(ctx)
    assert ctx.sets == [pytest.approx(2.0)]


def test_restore_is_idempotent():
    """三条路径都会调它（正常收尾 / panic / reapproach=False），
    重复调用不该把偏压来回设。"""
    sk = _fresh()
    ctx = _Ctx(bias_now=2.0)
    sk._lower_bias_for_move(ctx)
    ctx.sets.clear()
    sk._restore_bias(ctx)
    sk._restore_bias(ctx)
    sk._restore_bias(ctx)
    assert len(ctx.sets) == 1


def test_restore_without_a_prior_lowering_does_nothing():
    sk = _fresh()
    ctx = _Ctx(bias_now=0.3)
    sk._restore_bias(ctx)
    assert ctx.sets == []


def test_panic_restores_the_bias():
    """一次失败的粗动不能把调用方留在 0.5 V 上 —— 而它以为还在成像偏压。
    这正是 08-26 追了半夜的「工作点被上一个 skill 改掉且不改回来」。"""
    sk = _fresh()
    ctx = _Ctx(bias_now=2.0)
    sk._lower_bias_for_move(ctx)
    ctx.sets.clear()
    sk._panic(ctx)
    assert pytest.approx(2.0) in ctx.sets


def test_reapproach_restores_bias_before_approaching():
    """进针要在调用方原本的工作点上完成，否则「进好的针」是在一个
    没人要求过的偏压下建立的。"""
    sk = _fresh()
    ctx = _Ctx(bias_now=2.0)
    sk._lower_bias_for_move(ctx)
    ctx.calls.clear(); ctx.sets.clear()
    sk._phase_reapproach(ctx, {})
    assert "Bias_Set" in ctx.calls and "ApproachTip" in ctx.calls
    assert ctx.calls.index("Bias_Set") < ctx.calls.index("ApproachTip")
