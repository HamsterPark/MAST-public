# -*- coding: utf-8 -*-
"""``vision.frame_drift``：帧间亚像素位移，与上下扫回程差的分离。

合成上把位移**放进去再量出来**（``scipy.ndimage.shift``），所以「量得准」是真检验。
最要紧的两条是行为条：判不了要说判不了（上游两个整像素实现把它折叠成了 0.0），
以及先筛后统计会把方法的失败伪装成一个可信的小数。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import shift as ndshift

from mast.vision.frame_drift import (
    pair_displacement,
    separate_hysteresis,
)


def speckle(n=256, seed=0, feature_nm=0.06, nmpp=0.02):
    """一片非周期的「地形」：随机点被高斯抹开 —— 配准靠的正是这种内容。

    ``feature_nm`` 默认 0.06 nm ≈ 3 px，用于构造具有可配准局部特征的随机场。
    **别调得太大**：σ=50 px 的图上位移只有特征的十分之一，任何配准方法都测不出来，
    那时测的是方法的下限而不是被测对象（开发时第一版就是 1.0 nm = 50 px，
    于是「量得准」那条测试在一个所有方法都失效的场景里失败）。
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1.0, (n, n))
    return gaussian_filter(img, max(0.5, feature_nm / nmpp)) * 1e-10


# ── 1. 位移量得准 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dy_px,dx_px", [(0.0, 0.0), (3.0, -2.0), (-1.5, 0.5)])
def test_recovers_the_shift_that_was_put_in(dy_px, dx_px):
    a = speckle()
    b = ndshift(a, (dy_px, dx_px), order=3, mode="nearest")
    r = pair_displacement(a, b, nm_per_px=0.02, smooth_nm=None)
    assert r.ok, (r.reason, r.warnings)
    # 约定：返回的是把 b 对回 a 所需的位移，符号与 phase_cross_correlation 一致。
    assert abs(r.dy_px) == pytest.approx(abs(dy_px), abs=0.3)
    assert abs(r.dx_px) == pytest.approx(abs(dx_px), abs=0.3)
    assert r.dy_nm == pytest.approx(r.dy_px * 0.02, rel=1e-9)


def test_confidence_is_high_for_a_pure_shift_and_low_for_unrelated_frames():
    a = speckle(seed=1)
    b = ndshift(a, (2.0, 1.0), order=3, mode="nearest")
    other = speckle(seed=99)
    assert pair_displacement(a, b, nm_per_px=0.02, smooth_nm=None).confidence > 0.9
    r = pair_displacement(a, other, nm_per_px=0.02, smooth_nm=None)
    assert (not r.ok) or r.confidence < 0.5


# ── 2. 判不了要说判不了 ─────────────────────────────────────────────────

def test_unrelated_frames_are_refused_not_reported_as_zero_drift():
    """上游两个整像素实现的每一条异常路径都 ``return 0.0, 0.0`` ——
    「测不出来」与「没有漂移」被折叠成同一个数。这里必须分开。"""
    r = pair_displacement(speckle(seed=1), speckle(seed=2), nm_per_px=0.02,
                          smooth_nm=None, min_confidence=0.6)
    assert not r.ok
    assert r.reason in ("low_confidence", "anti_correlated", "shift_too_large")
    assert r.dx_nm is None and r.dy_nm is None
    assert r.warnings


def test_anti_correlated_alignment_is_refused():
    """没有共同特征的帧可能产生相关峰；对齐后反相关应拒绝被报告为可信位移。"""
    a = speckle(seed=3)
    r = pair_displacement(a, -a, nm_per_px=0.02, smooth_nm=None,
                          min_confidence=0.0)
    assert not r.ok and r.reason == "anti_correlated"
    assert r.confidence < 0


def test_a_flat_frame_is_refused():
    """相位相关在没有共同特征的两张图上照样给出一个峰。"""
    r = pair_displacement(np.zeros((64, 64)), np.zeros((64, 64)), nm_per_px=0.02)
    assert not r.ok and r.reason == "flat_frame"


def test_huge_shift_is_refused_because_the_peak_wraps():
    a = speckle(n=128)
    b = ndshift(a, (50.0, 0.0), order=1, mode="wrap")
    r = pair_displacement(a, b, nm_per_px=0.02, smooth_nm=None)
    assert not r.ok
    assert r.reason in ("shift_too_large", "low_confidence"), r.reason


def test_shape_and_scale_problems_are_named():
    assert pair_displacement(np.zeros((8, 8)), np.zeros((9, 9)),
                             nm_per_px=0.02).reason == "shape_mismatch"
    a = speckle(n=64)
    assert pair_displacement(a, a, nm_per_px=0).reason == "unknown_pixel_size"
    bad = a.copy(); bad[0, 0] = np.nan
    assert pair_displacement(a, bad, nm_per_px=0.02).reason == "non_finite"


# ── 3. 低通不是美化 ─────────────────────────────────────────────────────

def _lattice_plus_terrain(terrain_gain, n=256, nmpp=0.02, seed=5):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    lattice = 20.0 * np.cos(2 * np.pi * xx * nmpp / 0.4) * 1e-12
    return lattice + speckle(n=n, seed=seed, nmpp=nmpp) * terrain_gain


def test_registration_works_when_there_is_non_periodic_terrain_to_lock_onto():
    """有可配准的非周期内容时，低通之后位移量得准。"""
    nmpp = 0.02
    a = _lattice_plus_terrain(terrain_gain=1.0, nmpp=nmpp)
    b = ndshift(a, (3.0, 0.0), order=3, mode="nearest")
    r = pair_displacement(a, b, nm_per_px=nmpp, smooth_nm=0.06)
    assert r.ok, (r.reason, r.warnings)
    assert abs(r.dy_px) == pytest.approx(3.0, abs=0.5), r


def test_a_lattice_dominated_frame_does_not_yield_a_confident_displacement():
    """纯周期晶格与自身在多个格矢处相关，缺少非周期地形时平移无法唯一确定。
    测试要求算法在证据不足时弃权，不能把某个等价相关峰解释为可靠位移。"""
    nmpp = 0.02
    a = _lattice_plus_terrain(terrain_gain=0.02, nmpp=nmpp)
    b = ndshift(a, (3.0, 0.0), order=3, mode="nearest")
    r = pair_displacement(a, b, nm_per_px=nmpp, smooth_nm=0.5,
                          min_confidence=0.5)
    assert (not r.ok) or abs(r.dy_px) < 1.0, (
        "晶格主导时要么拒答，要么锁在格矢上 —— 不能报出一个像模像样的漂移：%r" % (r,))


# ── 4. 回程差与净漂移 ───────────────────────────────────────────────────

def test_alternating_hysteresis_is_separated_from_net_drift():
    """一上一下时 Δy = 净漂移 ± 回程差。两帧滑动平均消掉交替项。"""
    net, hyst = 0.05, 0.30
    seq = [net + (hyst if i % 2 == 0 else -hyst) for i in range(9)]
    out = separate_hysteresis(seq)
    assert out["net_drift_per_frame_nm"] == pytest.approx(net, abs=1e-9)
    assert out["hysteresis_nm"] == pytest.approx(0.35, abs=1e-9)  # |net±hyst| 的中位


def test_unmeasured_pairs_do_not_count_as_zero():
    """``None`` 是「这一对没量出来」，不是「位移为零」——
    把它当 0 会把净漂移往零拉，而那正是上游实现的毛病。"""
    seq = [0.5, None, 0.5, 0.5, None]
    out = separate_hysteresis(seq)
    assert out["n_unmeasured"] == 2
    assert out["n_pairs"] == 3
    assert out["net_drift_per_frame_nm"] == pytest.approx(0.5)


def test_all_unmeasured_gives_none_not_zero():
    out = separate_hysteresis([None, None, None])
    assert out["net_drift_per_frame_nm"] is None
    assert out["hysteresis_nm"] is None
    assert out["n_pairs"] == 0


# ── 5. STM 帧的慢轴不是周期的；约定要钉死 ──────────────────────────────

def _stm_like(n=128, nmpp=0.3125, seed=11):
    """独立合成非周期地形、逐行偏置与慢轴斜坡。循环配准可能错误对齐边界，
    测试要求两个方向的已知平移都能恢复。"""
    rng = np.random.default_rng(seed)
    terrain = speckle(n=n, seed=seed, feature_nm=0.6, nmpp=nmpp)
    rows = rng.normal(0, 0.15, (n, 1)) * terrain.std()
    creep = np.linspace(0, 3.0, n)[:, None] * terrain.std() * np.ones((1, n))
    return terrain + rows + creep


def test_a_row_shift_is_measured_on_an_stm_like_frame_and_the_sign_is_pinned():
    a = _stm_like()
    b = ndshift(a, (10.0, 0.0), order=1, mode="nearest")     # content moves +10 rows
    r = pair_displacement(a, b, nm_per_px=0.3125, smooth_nm=0.5)
    assert r.ok, (r.reason, r.warnings)
    # convention: the shift that registers ``second`` onto ``first`` = −(feature motion)
    assert r.dy_px == pytest.approx(-10.0, abs=0.5), r
    assert abs(r.dx_px) < 0.5, r
    c = ndshift(a, (0.0, -6.0), order=1, mode="nearest")     # content moves −6 columns
    r2 = pair_displacement(a, c, nm_per_px=0.3125, smooth_nm=0.5)
    assert r2.ok and r2.dx_px == pytest.approx(6.0, abs=0.5) and abs(r2.dy_px) < 0.5, r2
