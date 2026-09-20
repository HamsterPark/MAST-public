# -*- coding: utf-8 -*-
"""慢扰动分析应区分真实振荡与有限帧长造成的谱泄漏。

用解析周期、纯曲率及不同帧长构造正反对照：真实周期在秒轴上稳定，
泄漏尺度随帧长改变，不能把相近但未分辨的谱峰当成跨帧长一致证据。"""
from __future__ import annotations

import numpy as np
import pytest

from mast.vision.slow_drift import (
    analyse_slow_drift,
    combine_frames,
    row_time_series,
)

NM = 1e-9
PM = 1e-12


def frame_with_oscillation(period_s, amp_m, line_time_s, n_rows,
                           *, n_cols=64, tilt_m=0.0, noise_m=0.0, seed=0,
                           phase=0.0):
    """造一帧：慢轴上带一个已知周期的起伏。

    横向（快轴）给一点结构，好确认行均值确实把它平掉了。
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(n_rows, dtype=float)
    t = rows * line_time_s
    slow = amp_m * np.sin(2 * np.pi * t / period_s + phase) + tilt_m * (t / max(t[-1], 1e-9))
    img = slow[:, None] * np.ones((1, n_cols))
    img = img + 5e-12 * np.cos(2 * np.pi * np.arange(n_cols) / 7.0)[None, :]
    if noise_m:
        img = img + rng.normal(0, noise_m, img.shape)
    return img


# ── 基本功 ──────────────────────────────────────────────────────────────────

def test_row_mean_removes_the_fast_axis_structure_and_the_tilt():
    """行均值只该留下慢轴方向的共模，线性倾斜要被扣掉。"""
    img = frame_with_oscillation(60.0, 10 * PM, 2.0, 256, tilt_m=500 * PM)
    t, y, slope = row_time_series(img, 2.0)
    assert t is not None
    assert abs(float(np.mean(y))) < 1e-14      # 趋势扣干净
    assert slope == pytest.approx(500 * PM / (255 * 2.0), rel=0.05)
    assert float(np.std(y)) == pytest.approx(10 * PM / np.sqrt(2), rel=0.15)


def test_finds_an_injected_period():
    img = frame_with_oscillation(60.0, 10 * PM, 2.0, 256, noise_m=1 * PM)
    r = analyse_slow_drift(img, 2.0)
    assert r.ok, r.reason
    assert r.components
    top = r.components[0]
    assert top.period_s == pytest.approx(60.0, rel=0.10)
    assert top.amplitude_m == pytest.approx(10 * PM, rel=0.30)


def test_the_analysable_band_is_bounded_by_the_frame_length():
    """一帧只能回答「至少能装下两三个周期」的那些频率。

    第一版这条测试拿 32 行的帧去期待 ``frame_too_short`` —— 但 32 行 x 2 s
    = 62 s 的帧完全能分析 25 s 以上的周期，它只是分析不了 300 s 的那条。
    「帧太短」不是帧的属性，是**帧与所问频率的关系**。所以这里改成检验
    下限本身：注入的周期超出下限时，它不会被报成一条成分。
    """
    img = frame_with_oscillation(300.0, 10 * PM, 2.0, 32)
    r = analyse_slow_drift(img, 2.0)
    if r.ok:
        f_min = r.detail["f_min_hz"]
        assert f_min == pytest.approx(2.5 / r.span_s, rel=1e-6)
        # 300 s 远低于下限，不该有任何成分声称自己是它
        assert all(c.period_s < 1.0 / f_min * 1.05 for c in r.components)
    else:
        assert r.reason == "frame_too_short"


# ── 泄漏：本模块的核心风险 ──────────────────────────────────────────────────

def test_spectral_leakage_is_flagged_when_all_frames_are_the_same_length():
    """全是同一种帧长时，**分不开真周期与泄漏** —— 必须说出来而不是下结论。

    一条毫无周期性、只有残余曲率的帧也会在 span/k 上堆出峰。这条测试用纯
    二次曲面（零周期成分）作输入：任何被报出来的"周期"都只能是伪影。
    """
    n, lt = 256, 2.0
    rows = np.arange(n, dtype=float)
    curved = 300 * PM * (rows / n) ** 2
    img = curved[:, None] * np.ones((1, 64))

    r = analyse_slow_drift(img, lt, scan_angle_deg=0.0, label="a")
    g = combine_frames([r])
    assert g["single_frame_length"] is True
    assert any("分不开" in w for w in g["warnings"])
    # 报出来的成分必须被标成可疑，而不是当结论
    if g["groups"]:
        assert any(d["leakage_suspect"] for d in g["groups"])


def test_leakage_ratio_uses_only_the_spans_that_component_came_from():
    """泄漏判据只能使用该成分实际出现过的帧长；增加另一种帧长不能改变已有成分的证据来源。"""
    long_lt, short_lt = 2.2, 2.2
    long_n, short_n = 256, 128
    # 长帧：只有一条 span/3 的假成分（用纯曲率造）
    rows = np.arange(long_n, dtype=float)
    long_img = (300 * PM * (rows / long_n) ** 2)[:, None] * np.ones((1, 64))
    # 短帧：一条真实的 60 s 振荡
    short_img = frame_with_oscillation(60.0, 8 * PM, short_lt, short_n, noise_m=0.5 * PM)

    rl = analyse_slow_drift(long_img, long_lt, scan_angle_deg=0.0, label="long")
    rs = analyse_slow_drift(short_img, short_lt, scan_angle_deg=90.0, label="short")
    g = combine_frames([rl, rs])
    assert g["single_frame_length"] is False

    for d in g["groups"]:
        # 每条成分的比值列表，长度必须等于它自己出现过的帧长数
        assert len(d["frame_span_ratio"]) == d["n_distinct_spans"]
        assert d["n_distinct_spans"] <= 2


def test_a_component_seen_at_two_frame_lengths_is_marked_span_invariant():
    """同一周期出现在两种帧长上、且不随帧长变 ⇒ 真周期最强的证据。"""
    img_a = frame_with_oscillation(60.0, 8 * PM, 2.2, 256, noise_m=0.5 * PM, seed=1)
    img_b = frame_with_oscillation(60.0, 8 * PM, 2.2, 128, noise_m=0.5 * PM, seed=2)
    ra = analyse_slow_drift(img_a, 2.2, scan_angle_deg=0.0, label="a")
    rb = analyse_slow_drift(img_b, 2.2, scan_angle_deg=90.0, label="b")
    g = combine_frames([ra, rb])
    inv = [d for d in g["groups"] if d.get("span_invariant")]
    assert inv, "两种帧长上的同一周期没有被标成 span_invariant"
    assert inv[0]["period_s"] == pytest.approx(60.0, rel=0.15)
    assert inv[0]["n_distinct_spans"] == 2
    assert g["n_span_invariant"] >= 1


def test_two_different_leakages_are_not_merged_into_one_fake_truth():
    """两个不同帧长的谱泄漏可能在频率分辨率内重合。
    归并应使用频率分辨率，且不能将未分辨的重合提升为跨帧长不变的证据。"""
    class _C:
        def __init__(self, period, span, ang, amp=1e-12):
            self.period_s = period
            self.freq_hz = 1.0 / period
            self.amplitude_m = amp
            self.over_floor = 10.0
            self.scan_angle_deg = ang
            self.span_s = span
            self.label = "x"

    class _R:
        ok = True
        residual_rms_m = 1e-12
        trend_nm_per_h = 0.0

        def __init__(self, comps, span):
            self.components = comps
            self.span_s = span

    ra = _R([_C(120.0, 600.0, 0.0)], 600.0)     # 600/5
    rb = _R([_C(100.0, 300.0, 90.0)], 300.0)    # 300/3
    g = combine_frames([ra, rb])
    # 合并本身是**可以接受**的：300 s 的帧分辨率只有 1/300 Hz，这两条在
    # 频率上确实分不开。不可接受的是因此宣布「周期不随帧长变」—— 那是把
    # 「分辨率不够所以看起来一样」当成了「测出来一样」。
    for d in g["groups"]:
        assert not d.get("span_invariant"), (
            "120.0 s（=600/5）与 100.0 s（=300/3）之间散布 20%%，"
            "不该被宣布为跨帧长不变：%s" % d.get("period_by_span_s"))
    assert g.get("n_span_invariant", 0) == 0


# ── 跨扫描角：区分时间性扰动与样品结构 ──────────────────────────────────────

def test_same_period_at_two_scan_angles_is_time_locked():
    a = frame_with_oscillation(45.0, 6 * PM, 2.0, 256, noise_m=0.5 * PM, seed=3)
    b = frame_with_oscillation(45.0, 6 * PM, 2.0, 256, noise_m=0.5 * PM, seed=4)
    g = combine_frames([
        analyse_slow_drift(a, 2.0, scan_angle_deg=0.0, label="a"),
        analyse_slow_drift(b, 2.0, scan_angle_deg=60.0, label="b"),
    ])
    top = g["groups"][0]
    assert top["time_locked"] is True
    assert top["n_distinct_angles"] == 2


def test_one_angle_only_is_not_time_locked():
    """同一个扫描角下的重复只是重复 —— 它区分不了样品结构。"""
    a = frame_with_oscillation(45.0, 6 * PM, 2.0, 256, noise_m=0.5 * PM, seed=5)
    b = frame_with_oscillation(45.0, 6 * PM, 2.0, 256, noise_m=0.5 * PM, seed=6)
    g = combine_frames([
        analyse_slow_drift(a, 2.0, scan_angle_deg=30.0, label="a"),
        analyse_slow_drift(b, 2.0, scan_angle_deg=30.0, label="b"),
    ])
    assert g["groups"][0]["time_locked"] is False


# ── n_frames 数的是帧，不是成员────────────────────────

def test_two_components_from_one_frame_do_not_count_as_two_frames():
    """同一帧的多个成分合并后，n_members 记录成分数，n_frames 仍只记录唯一帧数。"""
    from mast.vision.slow_drift import (
        DriftComponent, SlowDriftResult, combine_frames as _combine,
    )

    def _c(period_s: float, amp_m: float) -> DriftComponent:
        return DriftComponent(freq_hz=1.0 / period_s, period_s=period_s,
                              amplitude_m=amp_m, over_floor=10.0,
                              label="same.sxm", scan_angle_deg=0.0, span_s=600.0)

    # 一帧，两个周期相近到会被并组的成分（Δf = 7e-5 Hz，容差 ~1.5×2/600）
    res = SlowDriftResult(ok=True, n_frames=1, fs_hz=0.5, span_s=600.0,
                          residual_rms_m=1e-12, trend_nm_per_h=0.0)
    res.components = [_c(170.0, 2.0e-12), _c(168.0, 2.4e-12)]

    out = _combine([res])
    groups = out.get("groups") or []
    assert groups, "两个成分一个也没出来"
    merged = [g for g in groups if g["n_members"] >= 2]
    assert merged, "这两个成分本该被并成一组（频率差远小于 1/span 容差）"
    g = merged[0]
    assert g["n_frames"] == 1, (
        "同一帧的两个成分被数成了两帧 —— 读的人会当成「两帧互相印证」")
    assert g["frames"] == ["same.sxm"], "帧名列表里出现了重复"
    assert g["n_distinct_angles"] == 1
    assert g["time_locked"] is False, "一个角度不该判成 time_locked"
