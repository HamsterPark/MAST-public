# -*- coding: utf-8 -*-
"""四段合成曲线：基线、压入、反馈关闭时回基线、反馈恢复后的新平衡。

电流通道用于辨认第四段；缺少该段时不得将采集不足判断为没有变化。
数据由下面的分段常数生成器构造，不使用真实采集曲线或试验参数。
"""
from __future__ import annotations

import pytest

from mast.io import z_trace


def _poke_trace(*, depth_pm=450.0, final_pm=0.0, seg4_s=1.0,
                setpoint_pa=125.0, sat_pa=6000.0, dt=0.005,
                pre_s=0.3, plunge_s=0.6, seg3_s=0.3,
                current_returns=True):
    """构造四段常数序列；参数为独立选取的合成工作点。"""
    z, t, c, ct = [], [], [], []
    now = 0.0

    def push(z_pm, i_pa, dur):
        nonlocal now
        n = max(1, int(dur / dt))
        for _ in range(n):
            z.append(z_pm * 1e-12)
            t.append(now)
            c.append(i_pa * 1e-12)
            ct.append(now)
            now += dt

    push(0.0, setpoint_pa, pre_s)                       # 1 基线
    push(-depth_pm, sat_pa, plunge_s)                   # 2 压入
    push(0.0, sat_pa, seg3_s)                           # 3 抬回原位、反馈仍 OFF
    if seg4_s > 0:
        push(final_pm, setpoint_pa if current_returns else sat_pa, seg4_s)
    return z, t, c, ct, pre_s


# ── 定位函数本身 ──────────────────────────────────────────────────────────

def test_feedback_restore_is_found_from_the_current():
    z, t, c, ct, ev = _poke_trace(final_pm=360.0)
    t4, why = z_trace.feedback_restored_t(c, ct, event_start_t=ev)
    assert why == "current"
    assert t4 == pytest.approx(0.3 + 0.6 + 0.3, abs=0.02)


def test_no_return_when_the_capture_ends_while_the_current_is_still_railed():
    """合成采集在电流回落前结束，必须报告尚无反馈恢复证据。"""
    z, t, c, ct, ev = _poke_trace(seg4_s=0.0)
    t4, why = z_trace.feedback_restored_t(c, ct, event_start_t=ev)
    assert t4 is None and why == "no_return"


def test_no_press_when_the_current_never_rises():
    """电流没升上去 ⇒ 可能根本没压到表面。**这与 no_return 是两件事。**"""
    z, t, c, ct, ev = _poke_trace(depth_pm=0.0, sat_pa=125.0, final_pm=0.0)
    t4, why = z_trace.feedback_restored_t(c, ct, event_start_t=ev)
    assert t4 is None and why == "no_press"


def test_the_baseline_itself_never_counts_as_the_restore():
    """基线段的电流本来就等于 setpoint —— t4 不许落在扎针之前。"""
    z, t, c, ct, ev = _poke_trace(final_pm=100.0)
    t4, _ = z_trace.feedback_restored_t(c, ct, event_start_t=ev)
    assert t4 > ev


# ── 判定：读第四段 ────────────────────────────────────────────────────────

def test_the_verdict_reads_segment_four_not_segment_three():
    """合成第三段为零、第四段为 +360 pm；判定必须读取第四段。"""
    z, t, c, ct, ev = _poke_trace(final_pm=360.0, seg4_s=1.0)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9, current_s=c, current_t=ct)
    assert out["feedback_segment_source"] == "current"
    assert out["direction"] == "up"
    assert out["delta_m"] * 1e12 == pytest.approx(360.0, abs=5.0)


def test_without_the_current_the_old_tail_window_would_have_said_no_change():
    """同一条曲线，不给电流 ⇒ 尾窗 ⇒ 判成「没变」。**负例先证明自己会红。**

    没有这一条，上一条就只是「新代码给出了某个数」，说明不了它修好了什么。
    尾窗 0.2 s 落在第三段（那里 Δz=0），于是同一次扎针被判成没扎上。
    """
    z, t, c, ct, ev = _poke_trace(final_pm=360.0, seg4_s=0.05)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9)
    assert out["direction"] == "none"
    assert out["feedback_segment_source"] == "no_current"


def test_a_capture_that_ended_before_feedback_is_insufficient_not_no_change():
    """反馈尚未恢复时只能报告采集不足，不能建议用更大压入补偿缺失数据。"""
    z, t, c, ct, ev = _poke_trace(final_pm=360.0, seg4_s=0.0)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9, current_s=c, current_t=ct)
    assert out["direction"] == "insufficient_data"
    assert out["reason"] == "feedback_segment_not_captured"
    assert "加大扎入深度" in out["advice"]
    assert "post_roll_s" in out["advice"]


def test_a_too_short_segment_four_is_also_insufficient():
    z, t, c, ct, ev = _poke_trace(final_pm=360.0, seg4_s=0.10)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9, current_s=c, current_t=ct)
    assert out["direction"] == "insufficient_data"
    assert out["reason"] == "feedback_segment_too_short"
    assert out["feedback_segment_s"] == pytest.approx(0.10, abs=0.02)


def test_no_press_still_gets_a_verdict():
    """没压到表面时**不能**降级成「判不了」—— 那时「没扎上」就是实话。"""
    z, t, c, ct, ev = _poke_trace(depth_pm=0.0, sat_pa=125.0, final_pm=0.0)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9, current_s=c, current_t=ct)
    assert out["direction"] == "none"
    assert out["feedback_segment_source"] == "no_press"


def test_a_real_no_change_is_still_reported_as_no_change():
    """第四段录到了、Z 确实没变 ⇒ 「没扎上」。修的是错判，不是把这一态删掉。"""
    z, t, c, ct, ev = _poke_trace(final_pm=2.0, seg4_s=1.0)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9, current_s=c, current_t=ct)
    assert out["direction"] == "none"
    assert out["feedback_segment_source"] == "current"


def test_a_pit_reads_as_down():
    z, t, c, ct, ev = _poke_trace(final_pm=-150.0, seg4_s=1.0)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9, current_s=c, current_t=ct)
    assert out["direction"] == "down"


# ── 向后兼容 ──────────────────────────────────────────────────────────────

def test_omitting_the_current_keeps_the_old_behaviour_exactly():
    """老调用方（不传电流）行为一字不变 —— 只是多了一个 ``no_current`` 标记。"""
    z, t, c, ct, ev = _poke_trace(final_pm=360.0, seg4_s=1.0)
    a = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0, tol_abs_m=0.02e-9)
    assert a["feedback_segment_source"] == "no_current"
    assert a["feedback_segment_s"] is None
    assert a["feedback_restored_t"] is None
    # 尾窗取到的是第四段的末尾（这条曲线第四段有 1 s），所以方向仍是 up ——
    # **不是**因为它读对了段，而是因为这条合成曲线的第四段够长。
    assert a["direction"] == "up"


def test_the_current_channel_may_be_shorter_than_z_without_crashing():
    """两个通道的采样点数在真机上并不相等（各自打各自的时间戳）。"""
    z, t, c, ct, ev = _poke_trace(final_pm=360.0, seg4_s=1.0)
    out = z_trace.step_verdict(z, t, ev, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9,
                               current_s=c[:len(c) // 2], current_t=ct[:len(ct) // 2])
    assert out["direction"] in ("up", "none", "insufficient_data")
