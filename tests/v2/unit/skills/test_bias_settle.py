"""BiasSettleChange —— 改偏压的安全通道。

这条通道存在的全部意义是把一颗**领域炸弹**封进代码里:恒流反馈下,偏压趋近 0
时隧道电流也趋近 0,反馈唯一的反应是把针尖一直往表面推。``SetBias(bias_v=-1.0)``
从 +1 V 调过去是一次完全合法的调用 —— 参数在范围内、安全门不拦、日志只留一行
「成功」—— 而针尖已经扎进样品了。
"""

from __future__ import annotations

import pytest

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.bias_settle import (
    ZERO_DEADBAND_V,
    BiasSettleChange,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("mast.skills.composite.bias_settle.time.sleep",
                        lambda *_a, **_k: None)


class BiasCtx:
    def __init__(self, bias=1.0, feedback_on=True, *, bias_read_error=False,
                 sub_fail=None):
        self.bias = bias
        self.feedback_on = feedback_on
        self.bias_read_error = bias_read_error
        self.sub_fail = sub_fail
        self.calls: list[tuple[str, tuple]] = []
        self.runs: list[tuple[str, dict]] = []

    def safe_call(self, method, *args, **kwargs):
        self.calls.append((method, args))
        if method == "Bias_Get":
            if self.bias_read_error:
                return NanonisCallRecord(method=method, args=args,
                                         error="link down")
            return NanonisCallRecord(method=method, args=args,
                                     return_value=["", b"", [self.bias]])
        if method == "ZCtrl_OnOffGet":
            if self.feedback_on is None:
                return NanonisCallRecord(method=method, args=args,
                                         error="unknown")
            return NanonisCallRecord(
                method=method, args=args,
                return_value=["", b"", [1 if self.feedback_on else 0]])
        return NanonisCallRecord(method=method, args=args)

    def run(self, skill_name, params, version=None):
        self.runs.append((skill_name, dict(params)))
        if self.sub_fail:
            return SkillResult(skill_name=skill_name, success=False,
                               error=self.sub_fail)
        self.bias = params.get("bias_v", params.get("bias_v_end", self.bias))
        return SkillResult(skill_name=skill_name, success=True, data={})

    def check_abort(self):
        return False


def _run(ctx, **params):
    return BiasSettleChange().execute(ctx, params)


# ── 穿零保护(核心) ─────────────────────────────────────────────────────────

def test_sign_change_goes_through_a_ramp_not_a_direct_set():
    """+1 V → −1 V 中途必经 0 —— 直接设过去就是把针尖推向表面。"""
    ctx = BiasCtx(bias=1.0)
    res = _run(ctx, bias_v=-1.0)
    assert res.success
    assert res.data["crossed_zero"] is True
    assert res.data["strategy"] == "ramp_through_zero"
    assert [name for name, _ in ctx.runs] == ["SetBiasRamp"]
    assert not any(name == "SetBias" for name, _ in ctx.runs)


def test_zero_crossing_uses_the_faster_slew():
    """死区里停留的每一毫秒反馈都在往下推,所以穿零要比常规斜坡快。"""
    from mast.skills.composite.bias_settle import (
        CROSS_ZERO_SLEW_V_PER_S,
        DEFAULT_SLEW_V_PER_S,
    )
    cross = _run(BiasCtx(bias=1.0), bias_v=-1.0)
    same_sign = _run(BiasCtx(bias=1.0), bias_v=3.0)
    assert cross.data["slew_rate_v_per_s"] == CROSS_ZERO_SLEW_V_PER_S
    assert same_sign.data["slew_rate_v_per_s"] == DEFAULT_SLEW_V_PER_S
    assert CROSS_ZERO_SLEW_V_PER_S > DEFAULT_SLEW_V_PER_S


@pytest.mark.parametrize("start,target", [
    (1.0, -1.0), (-0.5, 0.5), (2.0, -0.1), (-3.0, 0.2),
])
def test_every_sign_change_is_detected(start, target):
    res = _run(BiasCtx(bias=start), bias_v=target)
    assert res.data["crossed_zero"] is True


@pytest.mark.parametrize("start,target", [
    (1.0, 2.0), (-1.0, -2.0), (0.5, 0.6), (-0.2, -3.0),
])
def test_same_sign_changes_are_not_treated_as_crossings(start, target):
    res = _run(BiasCtx(bias=start), bias_v=target)
    assert res.data["crossed_zero"] is False


# ── 死区 ─────────────────────────────────────────────────────────────────────

def test_parking_in_the_dead_band_is_refused_while_feedback_is_on():
    """低偏压死区是个**不能停**的地方 —— 反馈在那里维持不住电流。"""
    ctx = BiasCtx(bias=1.0, feedback_on=True)
    res = _run(ctx, bias_v=0.01)
    assert not res.success
    assert "死区" in res.error
    assert ctx.runs == [], "被拒绝了却还是改了偏压"


def test_dead_band_is_allowed_when_feedback_is_off():
    """STS 常常需要关反馈后把偏压扫到 0 附近 —— 那是合法的。"""
    ctx = BiasCtx(bias=1.0, feedback_on=False)
    res = _run(ctx, bias_v=0.01)
    assert res.success


def test_dead_band_can_be_opted_into_explicitly():
    ctx = BiasCtx(bias=1.0, feedback_on=True)
    res = _run(ctx, bias_v=0.01, allow_stop_in_deadband=True)
    assert res.success


def test_unknown_feedback_state_is_treated_as_on():
    """读不到反馈状态时按「开着」处理 —— 猜错的代价不对称:
    多拒一次只是麻烦,少拒一次是撞针。"""
    ctx = BiasCtx(bias=1.0, feedback_on=None)
    res = _run(ctx, bias_v=0.01)
    assert not res.success


def test_a_target_just_outside_the_dead_band_is_fine():
    ctx = BiasCtx(bias=1.0, feedback_on=True)
    res = _run(ctx, bias_v=ZERO_DEADBAND_V * 1.1)
    assert res.success


# ── 幅度与稳定 ───────────────────────────────────────────────────────────────

def test_large_same_sign_change_uses_a_ramp():
    ctx = BiasCtx(bias=1.0)
    res = _run(ctx, bias_v=3.0)
    assert res.data["strategy"] == "ramp"
    assert [name for name, _ in ctx.runs] == ["SetBiasRamp"]


def test_small_change_is_set_directly():
    ctx = BiasCtx(bias=1.0)
    res = _run(ctx, bias_v=1.1)
    assert res.data["strategy"] == "direct"
    assert [name for name, _ in ctx.runs] == ["SetBias"]


def test_small_change_still_waits_for_the_feedback_transient():
    from mast.skills.composite.bias_settle import SMALL_CHANGE_SETTLE_S
    res = _run(BiasCtx(bias=1.0), bias_v=1.1)
    assert res.data["settle_s"] == SMALL_CHANGE_SETTLE_S


def test_large_change_waits_longer():
    from mast.skills.composite.bias_settle import (
        DEFAULT_SETTLE_S,
        SMALL_CHANGE_SETTLE_S,
    )
    res = _run(BiasCtx(bias=1.0), bias_v=3.0)
    assert res.data["settle_s"] == DEFAULT_SETTLE_S
    assert DEFAULT_SETTLE_S > SMALL_CHANGE_SETTLE_S


def test_explicit_settle_is_honoured():
    res = _run(BiasCtx(bias=1.0), bias_v=1.1, settle_s=7.5)
    assert res.data["settle_s"] == 7.5


# ── 失败路径 ─────────────────────────────────────────────────────────────────

def test_unreadable_start_bias_refuses_to_act():
    """不知道起点就判断不了会不会穿零 —— 这时候动手是在赌。"""
    ctx = BiasCtx(bias_read_error=True)
    res = _run(ctx, bias_v=-1.0)
    assert not res.success
    assert "读不到当前偏压" in res.error
    assert ctx.runs == []


def test_sub_skill_failure_is_reported_not_swallowed():
    ctx = BiasCtx(bias=1.0, sub_fail="bias out of range")
    res = _run(ctx, bias_v=-1.0)
    assert not res.success
    assert "bias out of range" in res.error


def test_result_records_the_route_it_took():
    """用户要能看出这次到底走了哪条路 —— 「成功」两个字不够。"""
    res = _run(BiasCtx(bias=1.0), bias_v=-1.0)
    for key in ("bias_v_start", "bias_v", "delta_v", "crossed_zero",
                "strategy", "settle_s", "feedback_on"):
        assert key in res.data
