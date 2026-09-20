# -*- coding: utf-8 -*-
"""RecoverTipFromSaturation：先检查仪器占用，再执行有界恢复阶梯。

未下发的操作与已执行但无效果的操作必须区分；每级操作后都复核电流，
恢复后立即停止，不应无条件跑完整个粗动阶梯。
"""
from __future__ import annotations

import pytest

from mast.skills.builtins import RecoverTipFromSaturation
from mast.skills.builtins.saturation_recovery import (
    _LADDER,
    _RECOVERED_A,
    _SATURATION_A,
    is_saturated,
)

BUSY = ("仪器正被占用：技能直调 API 正在执行 ScanAt，已持有 123s。"
        "'MotorMove' 未执行——同一台 Nanonis 不能被两条链路同时驱动。")


class _Ctx:
    """currents 是一个 callable(cumulative_steps) -> A，模拟「退得越远电流越小」。"""

    def __init__(self, currents, busy=False, motor_ok=True):
        self.currents = currents
        self.busy = busy
        self.motor_ok = motor_ok
        self.cum = 0
        self.calls: list[tuple] = []

    def run(self, name, params=None):
        params = params or {}
        self.calls.append((name, params))
        if name == "MotorMove":
            if self.busy:
                return type("R", (), {"success": False, "error": BUSY, "data": {}})()
            if params.get("steps", 0) and self.motor_ok:
                self.cum += int(params["steps"])
            return type("R", (), {"success": self.motor_ok,
                                  "error": "" if self.motor_ok else "motor dead",
                                  "data": {}})()
        if name == "GetCurrent":
            v = self.currents(self.cum) if callable(self.currents) else self.currents
            return type("R", (), {"success": True, "data": {"current_a": v}})()
        return type("R", (), {"success": True, "data": {}})()

    def motor_steps(self):
        return [p.get("steps") for n, p in self.calls if n == "MotorMove" and p.get("steps")]


def _run(ctx, **kw):
    sk = RecoverTipFromSaturation()
    sk._settle_s = 0.0
    return sk.execute(ctx, kw)


# ── 饱和的识别 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("cur,want", [
    (1.1e-8, True),      # 独立构造的超量程电流
    (-1.1e-8, True),
    (9.6e-9, True),
    (1e-9, False), (5e-11, False), (0.0, False),
])
def test_saturation_detection(cur, want):
    assert is_saturated(cur) is want


def test_unreadable_is_none_not_false():
    """读不到 ≠ 没饱和。当成没饱和会让调用方以为可以继续，而针尖可能还压着。"""
    assert is_saturated(None) is None


# ── ★ 第一条：查锁 ─────────────────────────────────────────────────────
def test_lock_held_refuses_before_touching_anything():
    ctx = _Ctx(lambda c: 1.1e-8, busy=True)
    res = _run(ctx)
    assert res.success is False
    assert res.data["blocked_by_lock"] is True
    assert "占用" in (res.error or "")
    # 关键：不能在锁被别人拿着的时候去撤针 —— 那些命令到不了仪器
    assert "WithdrawTip" not in [n for n, _ in ctx.calls]
    assert "SetBias" not in [n for n, _ in ctx.calls]


def test_not_saturated_does_nothing():
    ctx = _Ctx(lambda c: 3e-11)
    res = _run(ctx)
    assert res.data["outcome"] == "not_saturated"
    assert ctx.motor_steps() == []
    assert "WithdrawTip" not in [n for n, _ in ctx.calls]


def test_unreadable_current_refuses_rather_than_assuming_recovered():
    ctx = _Ctx(lambda c: None)
    res = _run(ctx)
    assert res.success is False
    assert "判不了" in (res.error or "")


# ── 阶梯 ───────────────────────────────────────────────────────────────
def test_piezo_withdraw_alone_can_be_enough():
    """压电退针就够时不该动粗动 —— 退过头要多花几分钟才进得回来。"""
    seq = iter([1.1e-8] * 3 + [5e-12] * 8)

    class C(_Ctx):
        def run(self, name, params=None):
            if name == "GetCurrent":
                self.calls.append((name, params or {}))
                return type("R", (), {"success": True,
                                      "data": {"current_a": next(seq, 5e-12)}})()
            return super().run(name, params)

    ctx = C(lambda c: 5e-12)
    res = _run(ctx)
    assert res.data["outcome"] == "recovered"
    assert res.data["by"] == "piezo_withdraw"
    assert ctx.motor_steps() == []


def test_coarse_ladder_is_monotonically_increasing():
    """逐级加大：先试小的，退多了要多花几分钟重新进针。"""
    assert list(_LADDER) == sorted(_LADDER)
    assert _LADDER[0] < _LADDER[-1]


def test_recovers_after_enough_coarse_steps():
    """合成电流在累计步数达到设定值后脱离饱和，应逐级推进并及时停止。"""
    ctx = _Ctx(lambda c: 5e-12 if c >= 160 else 1.1e-8)
    res = _run(ctx)
    assert res.data["outcome"] == "recovered"
    assert res.data["by"] == "coarse_z"
    assert res.data["coarse_steps_used"] >= 160
    # 逐级而不是一次退到底
    assert len(ctx.motor_steps()) > 1


def test_stops_as_soon_as_it_clears():
    """脱离就停 —— 不该把阶梯跑完。"""
    ctx = _Ctx(lambda c: 5e-12 if c >= 20 else 1.1e-8)
    _run(ctx)
    assert sum(s for s in ctx.motor_steps() if s) == _LADDER[0]


def test_budget_is_respected_and_failure_is_honest():
    ctx = _Ctx(lambda c: 1.1e-8)      # 永远不脱离
    res = _run(ctx, max_coarse_steps=100)
    assert res.success is False
    assert res.data["outcome"] == "not_recovered"
    assert res.data["coarse_steps_used"] <= 100
    assert "方向" in (res.error or ""), "未恢复时应提示检查 z-retract 方向配置"


def test_bias_is_lowered_before_reading_for_the_verdict():
    """带成像偏压时几十 nm 就场发射，读数没法用来判断。"""
    ctx = _Ctx(lambda c: 5e-12 if c >= 20 else 1.1e-8)
    _run(ctx)
    names = [n for n, _ in ctx.calls]
    assert "SetBias" in names
    assert names.index("SetBias") < names.index("WithdrawTip")


def test_recovered_message_says_reapproach_is_needed():
    """脱离饱和 = 针尖远离样品，调用方必须知道要重新进针。"""
    ctx = _Ctx(lambda c: 5e-12 if c >= 20 else 1.1e-8)
    res = _run(ctx)
    assert "重新进针" in res.data["message"]


def test_recovered_threshold_is_below_saturation_by_orders():
    assert _RECOVERED_A < _SATURATION_A / 10
