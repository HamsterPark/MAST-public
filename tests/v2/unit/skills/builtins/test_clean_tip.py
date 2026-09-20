# -*- coding: utf-8 -*-
"""CleanTipUntilBarrier：先测基线，再依据目标和显著改善条件决定是否操作。

已达标时不应继续扎针或脉冲；变差、停滞或读不到时应报告结果并停止。
"""
from __future__ import annotations

import pytest

from mast.skills.builtins import CleanTipUntilBarrier
from mast.skills.builtins.clean_tip import _LADDER, _MAX_STALE


class _Ctx:
    """按脚本吐 φ 序列；记录每一次动作。"""

    def __init__(self, phis):
        self.phis = list(phis)
        self.i = 0
        self.actions = []

    def run(self, name, params=None):
        params = params or {}
        if name == "MeasureBarrierHeight":
            v = self.phis[min(self.i, len(self.phis) - 1)]
            self.i += 1
            return type("R", (), {"success": True,
                                  "data": {"phi_ev": v,
                                           "verdict": "clean" if (v or 0) >= 3 else "contaminated"}})()
        if name in ("TipShape", "BiasPulse"):
            self.actions.append((name, params.get("tip_lift_m") or params.get("bias_v")))
        return type("R", (), {"success": True, "data": {"bias_v": -1.0}})()


def _run(ctx, **kw):
    p = {"target_phi_ev": 3.0, "max_steps": 7}
    p.update(kw)
    return CleanTipUntilBarrier().execute(ctx, p)


# ── 最重要的一条 ────────────────────────────────────────────────────────
def test_already_clean_does_absolutely_nothing():
    """基线已超过目标值时，正确动作是不执行任何整形步骤。"""
    ctx = _Ctx([3.5])
    res = _run(ctx)
    assert res.data["outcome"] == "already_clean"
    assert res.data["steps"] == []
    assert ctx.actions == [], "已达标却动了针尖 —— 这正是要防的事故"


@pytest.mark.parametrize("phi", [3.0, 3.5, 4.2])
def test_at_or_above_target_never_touches_the_tip(phi):
    ctx = _Ctx([phi])
    _run(ctx)
    assert ctx.actions == []


def test_baseline_unmeasurable_refuses_to_act():
    """基线判不了就**不动针尖** —— 在没有判据时动手正是这个技能存在的理由。"""
    ctx = _Ctx([None])
    res = _run(ctx)
    assert res.success is False
    assert ctx.actions == []
    assert "判不了" in (res.error or "")


# ── 阶梯与护栏 ─────────────────────────────────────────────────────────
def test_stops_as_soon_as_target_is_reached():
    ctx = _Ctx([1.0, 1.2, 3.4])          # 基线 + 两步
    res = _run(ctx)
    assert res.data["outcome"] == "reached"
    assert len(ctx.actions) == 2


def test_gentle_pokes_come_before_pulses():
    """2 nm 以内的扎针比脉冲温和得多 —— 阶梯必须先扎后脉冲。"""
    kinds = [k for k, _ in _LADDER]
    assert kinds[0] == "poke"
    assert kinds.index("pulse") > max(i for i, k in enumerate(kinds) if k == "poke" and i < kinds.index("pulse"))
    assert all(abs(a) <= 0.8 for k, a in _LADDER if k == "poke"), "扎针深度超过配置的 0.8 nm 上限"


def test_every_pulse_is_followed_by_a_stabilising_poke():
    """脉冲之后应执行稳定步骤。直接调用 _pulse 验证动作顺序，避免整个阶梯提前停止导致断言空过。"""
    ctx = _Ctx([1.0])
    CleanTipUntilBarrier()._pulse(ctx, 3.0)
    assert [a[0] for a in ctx.actions] == ["BiasPulse", "TipShape"], \
        "脉冲之后没有紧跟一次扎针稳定"


def test_pulse_and_poke_both_reach_the_rig_in_a_full_run():
    """集成侧：一路小幅改善（不触发 stale）时阶梯会走到脉冲档，
    且每一发脉冲后面都跟着扎针。"""
    ctx = _Ctx([1.0, 1.3, 1.7, 2.2, 2.9, 2.95, 2.96, 2.97])
    _run(ctx, max_steps=7)
    kinds = [a[0] for a in ctx.actions]
    assert "BiasPulse" in kinds, "这一轮没走到脉冲档，本用例失去意义"
    for i, k in enumerate(kinds):
        if k == "BiasPulse":
            assert i + 1 < len(kinds) and kinds[i + 1] == "TipShape"


def test_stops_after_consecutive_no_improvement():
    ctx = _Ctx([1.0] + [1.0] * 10)
    res = _run(ctx)
    assert res.data["outcome"] == "not_reached"
    assert len(ctx.actions) <= _MAX_STALE + 1


def test_stops_immediately_when_it_gets_worse():
    """「递进加深」在本仓栽过：只写了「变好就停」，没写「变差也要停」。"""
    ctx = _Ctx([1.0, 2.0, 0.5])          # 先变好，再明显变差
    res = _run(ctx)
    assert any(s.get("stopped", "").startswith("变差") for s in res.data["steps"])


def test_reports_the_best_state_not_the_last():
    """交出的必须是过程中最好的那个，而不是最后那个。"""
    ctx = _Ctx([1.0, 2.5, 1.1, 1.0, 1.0])
    res = _run(ctx)
    assert res.data["best_phi_ev"] == pytest.approx(2.5)
    assert res.data["best_at"] == "step1"
    assert res.data["final_phi_ev"] != res.data["best_phi_ev"]


def test_unmeasurable_step_is_not_counted_as_no_improvement():
    """「没测到」≠「没变好」—— 两者驱动的下一步不同。"""
    ctx = _Ctx([1.0, None, None, None, None, 3.5])
    res = _run(ctx)
    notes = [s.get("note", "") for s in res.data["steps"]]
    assert any("判不了" in n for n in notes)
    assert res.data["outcome"] == "reached"     # 没被 stale 提前掐断


def test_bias_is_read_back_after_every_step():
    """整形流程可能改变工作点，每一步之后都应回读当前状态。"""
    ctx = _Ctx([1.0, 1.05, 1.05, 1.05, 1.05])
    res = _run(ctx)
    assert all("bias_after_v" in s for s in res.data["steps"])


def test_noise_sized_gains_do_not_count_as_improvement():
    """改善幅度必须超过配置的显著变化门槛，避免将微小起伏当成进展。"""
    ctx = _Ctx([1.0, 1.05, 1.08, 1.10, 1.10])    # 每步涨几个百分点 = 噪声
    res = _run(ctx)
    assert res.data["outcome"] == "not_reached"
    assert len(ctx.actions) <= _MAX_STALE + 1
