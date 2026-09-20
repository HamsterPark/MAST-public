# -*- coding: utf-8 -*-
"""WaitForThermalSettle：稳定性以温度变化率为判据，并可叠加绝对温度门槛。

采用解析斜坡、常数和渐近平稳的合成序列，检验低温仍在变化、稳定但高于
给定上限、读不到温度及超时等不同状态。
"""
from __future__ import annotations

import pytest

from mast.skills.builtins import WaitForThermalSettle
from mast.skills.builtins.thermal_settle import (
    _DEFAULT_RATE_K_PER_MIN,
    _MIN_SAMPLES,
    rate_k_per_min,
)


# ── 速率计算 ───────────────────────────────────────────────────────────
def test_rate_needs_enough_points():
    """两点连线算出的「速率」全是噪声 —— 点不够要返回 None，不是 0。"""
    assert rate_k_per_min([(0.0, 10.0), (10.0, 9.0)]) is None
    assert rate_k_per_min([]) is None


def test_rate_matches_a_known_slope():
    pts = [(t * 60.0, 10.0 - 0.5 * t) for t in range(_MIN_SAMPLES)]   # -0.5 K/min
    assert rate_k_per_min(pts) == pytest.approx(-0.5, rel=1e-6)


def test_none_samples_are_dropped():
    pts = [(0.0, 5.0), (60.0, None), (120.0, 4.0), (180.0, 3.0),
           (240.0, 2.0), (300.0, 1.0)]
    assert rate_k_per_min(pts) is not None


# ── 核心：低于阈值 ≠ 稳定 ───────────────────────────────────────────────
class _Ctx:
    """按脚本吐温度序列。

    ⚠ 接受 **callable**（无限序列）或有限列表。
    第一版只支持列表，序列用完后一直返回最后一个值 —— 于是温度「变成常数」，
    技能正确地判它稳了，而我把这当成了技能的 bug。**测试桩的行为要和它
    要模拟的现实一致**：真实的降温不会在第 40 个采样后突然定住。
    """

    def __init__(self, temps):
        self.fn = temps if callable(temps) else None
        self.temps = None if callable(temps) else list(temps)
        self.i = 0
        self.calls = 0

    def run(self, name, params=None):
        self.calls += 1
        if name == "GetTemperature":
            v = self.fn(self.i) if self.fn else self.temps[min(self.i, len(self.temps) - 1)]
            self.i += 1
            return type("R", (), {"success": True, "data": {"value_k": v}})()
        return type("R", (), {"success": True, "data": {}})()


def _run(ctx, **kw):
    sk = WaitForThermalSettle()
    sk._poll_s = 0.0                      # 测试里不真等
    p = {"max_rate_k_per_min": 0.03, "timeout_s": 300.0, "window": 6}
    p.update(kw)
    return sk.execute(ctx, p)


def test_below_threshold_but_still_falling_is_not_settled():
    """**这是整个技能的理由。** 温度一直在 5 K 以下，但一直在快速下降 ——
    不能因为「已经低于 6 K」就说稳了。"""
    res = _run(_Ctx(lambda k: 5.0 - 0.2 * k), max_temp_k=6.0, timeout_s=1.0)
    assert res.success is False
    assert res.data["settled"] is False


def test_flat_temperature_settles():
    res = _run(_Ctx([4.0] * 20), max_temp_k=6.0)
    assert res.success is True
    assert res.data["settled"] is True
    assert res.data["temperature_k"] == pytest.approx(4.0)


def test_synthetic_cooldown_curve_settles_only_at_the_end():
    """合成序列前段为快速线性下降，末段渐近平稳，分别落在变化率门槛两侧。"""
    fast = [40.0 - 2.0 * i for i in range(8)]
    assert abs(rate_k_per_min([(i * 12.0, v) for i, v in enumerate(fast)])) > 1.0
    tail = [6.0 + 0.02 / (i + 1) for i in range(6)]
    assert abs(rate_k_per_min([(i * 25.0, v) for i, v in enumerate(tail)])) < 0.1


def test_rate_only_mode_ignores_absolute_temperature():
    """留空 max_temp_k = 只看速率 —— 在 77 K 上工作时不该要求它降到 5 K。"""
    res = _run(_Ctx([77.2] * 20))          # 没给 max_temp_k
    assert res.success is True
    assert res.data["temperature_k"] == pytest.approx(77.2)


def test_absolute_gate_still_applies_when_given():
    """给了 max_temp_k 就是**两条并且**：温度平了但仍高于上限 ⇒ 不算稳。"""
    res = _run(_Ctx([77.2] * 20), max_temp_k=6.0, timeout_s=3.0)
    assert res.success is False


# ── 读不到 ≠ 稳定 ──────────────────────────────────────────────────────
class _DeadCtx:
    def run(self, name, params=None):
        return type("R", (), {"success": False, "data": {}})()


def test_unreadable_temperature_is_a_failure_not_a_pass():
    """**读不到与稳定是两件事。** 把前者当后者会让调用方在未知热状态下开工。"""
    sk = WaitForThermalSettle()
    sk._poll_s = 0.0
    res = sk.execute(_DeadCtx(), {"timeout_s": 30.0})
    assert res.success is False
    assert "读不到" in (res.error or "")


def test_timeout_reports_not_settled_rather_than_pretending():
    res = _run(_Ctx(lambda k: 10.0 - 0.05 * k), timeout_s=1.0)
    assert res.success is False
    assert res.data["settled"] is False


def test_default_rate_rejects_the_synthetic_linear_slope():
    """默认变化率门槛应能拒绝测试中已知的 0.5 K/min 线性变化。"""
    assert 0 < _DEFAULT_RATE_K_PER_MIN < 0.5
