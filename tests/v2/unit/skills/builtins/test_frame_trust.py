# -*- coding: utf-8 -*-
"""AssessFrameTrust：用合成地形和逐行偏置验证稳定性统计。

MAD 应对缓慢坡度和局部台阶保持稳健，同时识别持续的行间高度抖动。
"""
from __future__ import annotations

import numpy as np
import pytest

from mast.skills.builtins.frame_trust import (
    _ROW_GOOD_PM,
    _ROW_USABLE_PM,
    row_big_jumps,
    row_jump_mad_pm,
    row_jump_sigma_pm,
)


def _flat(n=256, noise_pm=5.0, seed=0):
    return np.random.default_rng(seed).normal(0, noise_pm, (n, n))


def _tip_unstable(base, wobble_pm=300.0, seed=1):
    """针尖持续不稳：**每一行**的高度都在随机抬落。"""
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, wobble_pm, (base.shape[0], 1))


# ── 核心主张：MAD 与地形无关 ────────────────────────────────────────────
@pytest.mark.parametrize("terrain,label", [
    (lambda yy, xx: 4000.0 * xx / 255.0, "沿快轴斜坡 4 nm"),
    (lambda yy, xx: 4000.0 * yy / 255.0, "沿慢轴斜坡 4 nm"),
    (lambda yy, xx: 2500.0 * (yy > 128), "沿慢轴台阶 2.5 nm"),
    (lambda yy, xx: 1500.0 * np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / 900.0), "团簇"),
    (lambda yy, xx: 900.0 * np.sin(2 * np.pi * xx / 40.0), "沿快轴周期起伏"),
])
def test_mad_is_blind_to_terrain(terrain, label):
    """地形怎么变，逐行 MAD 都必须留在「稳」的那一档里。"""
    base = _flat()
    yy, xx = np.mgrid[0:256, 0:256]
    z = base + terrain(yy, xx)
    assert np.nanstd(z) / np.nanstd(base) > 3.0, "地形没造出足够 RMS 差，测试无分辨力"
    assert row_jump_mad_pm(z) <= _ROW_GOOD_PM, (
        "%s 把 MAD 抬到 %.1f pm —— 判据被地形带跑了" % (label, row_jump_mad_pm(z)))


def test_sigma_is_fooled_by_a_real_step_but_mad_is_not():
    """少数大台阶会主导标准差，而中位数绝对偏差应保持稳健。"""
    base = _flat()
    yy, _ = np.mgrid[0:256, 0:256]
    stepped = base + 2500.0 * (yy > 128)
    assert row_jump_sigma_pm(stepped) > 10 * row_jump_sigma_pm(base)   # σ 被骗
    assert row_jump_mad_pm(stepped) == pytest.approx(                  # MAD 没有
        row_jump_mad_pm(base), rel=0.5)


# ── 针尖持续不稳必须被抓到 ──────────────────────────────────────────────
def test_persistent_tip_wobble_is_caught():
    base = _flat()
    assert row_jump_mad_pm(base) <= _ROW_GOOD_PM
    assert row_jump_mad_pm(_tip_unstable(base)) > _ROW_USABLE_PM


@pytest.mark.parametrize("wobble,band", [
    # 抖动→MAD 的实测映射：10→13, 30→40, 80→107, 200→267, 400→534
    (0.0, "stable"), (10.0, "stable"),
    (80.0, "usable_coarse"), (120.0, "usable_coarse"),
    (300.0, "unstable"), (400.0, "unstable"),
])
def test_wobble_amplitude_maps_to_bands(wobble, band):
    z = _flat(noise_pm=3.0)
    if wobble:
        z = _tip_unstable(z, wobble_pm=wobble)
    mad = row_jump_mad_pm(z)
    got = ("stable" if mad <= _ROW_GOOD_PM
           else "usable_coarse" if mad <= _ROW_USABLE_PM else "unstable")
    assert got == band


# ── 大跳变：报出来，但不据它下针尖结论 ───────────────────────────────────
def test_big_jumps_counts_a_real_step_too():
    """**它分不开针尖跳和真台阶** —— 这条测试就是把这个局限钉住，
    免得以后有人拿 big_jumps 直接当针尖判据。"""
    base = _flat()
    yy, _ = np.mgrid[0:256, 0:256]
    stepped = base + 2500.0 * (yy > 128)
    n_big, n_rows = row_big_jumps(stepped)
    assert n_big >= 1 and n_rows == 255
    # 而针尖是稳的 —— 两个结论必须能同时成立
    assert row_jump_mad_pm(stepped) <= _ROW_GOOD_PM


def test_clean_frame_has_no_big_jumps():
    n_big, _ = row_big_jumps(_flat())
    assert n_big == 0


def test_absolute_floor_stops_tiny_ripples_counting_as_jumps():
    """大跳变判定是「5×MAD **并且** ≥50 pm」两条并且。

    只用倍数的话，一帧越干净越容易报跳变：噪声 0.5 pm 的帧上 5×MAD 才 ~3 pm，
    任何 20 pm 的真实起伏都会被数成「大跳变」。绝对下限就是防这个的。
    """
    # 必须是**行间突变**而不是平滑起伏：正弦的逐行差分只有幅度的 2π/周期，
    # 造不出要测的那个形状（第一版就是这么写的，变异照样绿）。
    base = _flat(noise_pm=0.5, seed=3)
    small = base.copy()
    small[100:, :] += 20.0        # 一个 20 pm 的行间突变，远小于 50 pm 下限
    n_big, _ = row_big_jumps(small)
    assert n_big == 0, "20 pm 的微小行间突变被数成了大跳变 —— 绝对下限没起作用"
    # 而真正超过下限的跳变仍要被抓到
    stepped = base.copy()
    stepped[128:, :] += 900.0
    n_big2, _ = row_big_jumps(stepped)
    assert n_big2 >= 1


# ── 独立合成数值覆盖稳定、粗扫可用和不稳定三档 ──────────────────────────
@pytest.mark.parametrize("mad_pm,want", [
    (10.0, "stable"), (25.0, "stable"),
    (100.0, "usable_coarse"), (150.0, "usable_coarse"),
    (400.0, "unstable"), (800.0, "unstable"), (2000.0, "unstable"),
])
def test_synthetic_values_land_in_the_right_band(mad_pm, want):
    got = ("stable" if mad_pm <= _ROW_GOOD_PM
           else "usable_coarse" if mad_pm <= _ROW_USABLE_PM else "unstable")
    assert got == want


def test_thresholds_are_positive_and_ordered():
    """两条门槛必须为正且有序，形成非空的中间分类区间。"""
    assert 0 < _ROW_GOOD_PM < _ROW_USABLE_PM


# ── 读不出来时说「判不了」 ──────────────────────────────────────────────
def test_too_few_rows_returns_none_not_a_number():
    assert row_jump_mad_pm(np.zeros((2, 64))) is None
    assert row_big_jumps(np.zeros((2, 64))) == (None, None)


def test_all_nan_rows_are_dropped_not_counted_as_zero():
    """全 NaN 的行是「读不到」，不能当成 0 高度参与差分 —— 那会凭空造出巨大跳变。

    ⚠ 帧的基线必须**明显偏离 0**，否则 NaN→0 恰好等于基线，这条测试就没有分辨力
    （第一版就是这样：合成基线在 0 附近，把剔除逻辑关掉测试照样绿）。
    真实的 .sxm 去斜之后基线也未必在 0 —— 未扫完的行是 NaN，正是这个形状。
    """
    z = _flat() + 3000.0          # 基线抬到 3 nm：NaN→0 会造出 3 nm 的假跳变
    z[100:110, :] = np.nan
    mad = row_jump_mad_pm(z)
    assert mad is not None and mad <= _ROW_GOOD_PM
    n_big, _ = row_big_jumps(z)
    assert n_big == 0, "NaN 行被当成 0 高度，凭空造出了大跳变"
