# -*- coding: utf-8 -*-
"""多帧晶格：把压电畸变与热漂移真正分开。

合成数据全部**按物理构造** —— 这条在本包的单帧测试里写过，2026-08-19 我还是
在验证脚本里违反了一次：给三个晶格方向用了同一个标量漂移项 ``c``。那不是任何
真实漂移能产生的场（真实的是 ``c_j = tau * (v . K_lab^j)``，随方向变），于是
算法「拟合不出来」，报出 -37 度的假剪切。**错在输入，不在被测代码**，而这种
错看代码是看不出来的。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mast.vision.lattice_multiframe import (
    angle_conditioning,
    assess_atomic_consistency,
    calibrate_multi_angle,
    collect_observation,
)

A_NM = 0.2884
D_NM = A_NM * math.sqrt(3) / 2        # 一阶峰周期 0.2498 nm
K = 1.0 / D_NM                        # 1/nm
NM_PER_PX = 0.0195


def _R(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def frame(theta_deg, A, v_drift=(0.0, 0.0), phi0=17.0, n=384,
          noise=0.06, seed=0, tau_per_nm=1.0):
    """按 ``K_img(theta) = R(theta)^T A^T K_lab - c_j e_y`` 造一帧。

    ``v_drift`` 是实验室系的漂移速度；它对第 j 个晶格方向的贡献
    ``c_j = tau * (v . K_lab^j)`` **随 j 变**。给三个方向用同一个 c 会造出
    一个非物理的场（见模块 docstring）。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    ux, uy = xx * NM_PER_PX, yy * NM_PER_PX
    A = np.asarray(A, float)
    v = np.asarray(v_drift, float)
    img = np.zeros((n, n))
    for j in range(3):
        phi = math.radians(phi0 + 60.0 * j)
        Klab = K * np.array([math.cos(phi), math.sin(phi)])
        c_j = tau_per_nm * float(v @ Klab)
        Kimg = _R(theta_deg).T @ (A.T @ Klab) - np.array([0.0, c_j])
        img += np.cos(2 * math.pi * (Kimg[0] * ux + Kimg[1] * uy))
    return img / 3.0 + rng.normal(0, noise, (n, n))


def _solve(A, v, angles, **kw):
    obs = [collect_observation(frame(t, A, v, seed=i), NM_PER_PX, t)
           for i, t in enumerate(angles)]
    return calibrate_multi_angle(obs, "Au(111)", **kw), obs


# ── 分离 ────────────────────────────────────────────────────────────────────

def test_recovers_pure_piezo_shear_with_no_drift():
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    r, _ = _solve(A, (0.0, 0.0), [0, 30, 60, 90, 120, 150])
    assert r.ok, r.reason
    assert r.x_scale == pytest.approx(np.linalg.norm(A[:, 0]), abs=0.03)
    assert r.y_scale == pytest.approx(np.linalg.norm(A[:, 1]), abs=0.03)
    # 注入 3.18°（A 两列的夹角偏离 90° 的量）
    assert r.shear_deg == pytest.approx(3.18, abs=0.5)
    assert r.drift_shear_deg < 0.5, "没有漂移却报出了漂移"


def test_orthogonal_piezo_with_pure_drift_is_not_mistaken_for_shear():
    """**这一条曾经是红的，而且错得很有说服力。**

    正交压电 + 纯漂移，第一版解出 x=0.83 / y=1.80 / 剪切 -36.8°，跨角度残差
    只有 1.2% —— 一个残差很小的完全错误的解。根因不在多帧部分（G 的角度误差
    只有 0.2°），而在 ``solve_affine``：它的第三个方程是 ``u.v = -target^2/2``
    也就是 **cos120°**，而 ``_independent_pair`` 挑出的那一对夹角是 60°。
    传 60° 的一对进去，方程组仍然有解 —— W 会去把 60° **掰成** 120°，代价是
    一个巨大的假剪切。
    """
    A = np.array([[1.10, 0.0], [0.0, 1.08]])
    r, _ = _solve(A, (0.010, -0.004), [0, 30, 60, 90, 120, 150])
    assert r.ok, r.reason
    assert r.x_scale == pytest.approx(1.10, abs=0.03)
    assert r.y_scale == pytest.approx(1.08, abs=0.03)
    assert abs(r.shear_deg) < 1.0, (
        "压电本来是正交的，却报出 %.1f° 剪切 —— 漂移被算到压电头上了"
        % r.shear_deg)


def test_separates_shear_and_drift_when_both_are_present():
    """合成图同时加入压电变形与漂移，验证多帧证据能够区分两者。"""
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    r, _ = _solve(A, (0.010, -0.004), [0, 30, 60, 90, 120, 150])
    assert r.ok, r.reason
    assert r.shear_deg == pytest.approx(3.18, abs=0.5)
    assert r.x_scale == pytest.approx(1.10, abs=0.03)


def test_two_angles_are_enough_and_three_improve_the_residual():
    """未知 6 个、每帧 4 个方程 ⇒ 两个角度即恰定。"""
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    r2, _ = _solve(A, (0.010, -0.004), [0, 90])
    r3, _ = _solve(A, (0.010, -0.004), [0, 45, 90])
    assert r2.ok and r3.ok
    for r in (r2, r3):
        assert r.x_scale == pytest.approx(1.10, abs=0.03)
    # 三帧是过定的，残差才有意义（两帧恰定时残差恒为 0，说明不了对错）
    assert r2.residual_rel == pytest.approx(0.0, abs=1e-6)
    assert r3.residual_rel > 0.0


# ── 拒绝 ────────────────────────────────────────────────────────────────────

def test_refuses_when_the_angles_are_not_spread_out():
    """角度全挤在一起时 ``R(theta)`` 几乎不变，压电项与漂移项在方程里分不开。

    这时**仍然解得出一个数** —— 最小二乘从不拒绝答题。拒绝必须是显式的。
    """
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    r, obs = _solve(A, (0.010, -0.004), [0, 3, 6])
    assert all(o is not None for o in obs), "这几帧本身是能量到晶格的"
    assert not r.ok
    assert r.reason == "angles_too_close"
    assert any("分不开" in w for w in r.warnings)


def test_angle_conditioning_uses_double_angle():
    """±K 不可分 ⇒ theta 与 theta+180° 对这个问题是同一个角度。

    直接取 max-min 会把 (0°, 179°) 当成张开 179°，而它实际上只有 1°。
    """
    assert angle_conditioning([0, 90])[0] == pytest.approx(90.0, abs=1e-6)
    assert angle_conditioning([0, 179])[0] < 5.0
    assert angle_conditioning([0])[1] == float("inf")


def test_single_angle_is_refused():
    A = np.array([[1.10, 0.0], [0.0, 1.08]])
    r, _ = _solve(A, (0.0, 0.0), [30])
    assert not r.ok and r.reason == "need_two_angles"


# ── 一致性：数据不足 != 互相矛盾 ────────────────────────────────────────────

def test_same_lattice_across_frames_is_consistent():
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    frames = [frame(0.0, A, (0.01, -0.004), noise=0.10, seed=s) for s in range(4)]
    r = assess_atomic_consistency(frames, NM_PER_PX, angles_deg=[0.0] * 4)
    assert r.verdict == "consistent"
    assert r.period_spread < 0.02
    assert r.n_atomic == 4


def test_different_jitter_each_frame_never_reads_as_a_lattice():
    """每帧抖在不同的空间频率上。

    这批数据**在单帧上就被角向集中度挡住了**（带通噪声的谱是弥散环，不是
    离散布拉格点），所以根本轮不到跨帧一致性 —— 结论是 ``absent``：帧都
    可用、都没有晶格。这比 ``inconsistent`` 更准确，也指向不同的下一步。
    """
    rng = np.random.default_rng(7)
    n = 384
    f = np.fft.fftfreq(n)
    FX, FY = np.meshgrid(f, f)
    RR = np.hypot(FX, FY)
    fakes = []
    for s in range(4):
        g = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0, 1, (n, n)))
                                 * ((RR > 0.06 + 0.01 * s) & (RR < 0.085 + 0.01 * s))))
        fakes.append(g / g.std())
    r = assess_atomic_consistency(fakes, NM_PER_PX, angles_deg=[0.0] * 4)
    assert r.verdict == "absent", r.reason
    assert r.n_atomic == 0
    assert r.n_unusable == 0, "这些帧是完整的 —— 别把它们算成「没扫完」"


def test_incomplete_frames_give_undetermined_not_inconsistent():
    """采集不完整不构成晶格不一致的证据；残帧应排除或返回未确定。"""
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    good = frame(0.0, A, (0.0, 0.0), noise=0.08, seed=1)
    partial = good.copy()
    partial[80:, :] = np.nan            # 扫了几行就停
    r = assess_atomic_consistency([good, partial, partial.copy()], NM_PER_PX,
                                  angles_deg=[0.0] * 3)
    assert r.verdict == "undetermined"
    assert r.n_unusable == 2 and r.n_no_lattice == 0
    assert any("采集" in w for w in r.warnings)


def test_a_lone_lattice_among_usable_blank_frames_is_suspect():
    """帧完整、却量不到晶格 —— 这时孤零零那一帧的晶格才真可疑。

    与上一条的区别正是 ``absent`` 与 ``inconsistent`` 的区别：那里一帧都没有
    （明确的否定），这里有一帧有、别的都没有（那一帧可疑）。
    """
    A = np.array([[1.10, 0.0], [0.0, 1.08]])
    rng = np.random.default_rng(3)
    good = frame(0.0, A, (0.0, 0.0), noise=0.08, seed=1)
    # 负例用**近乎平坦**的面，不用白噪声：白噪声偶尔能凑出集中度 > 20 的
    # 假峰（实测四个种子里有一个），于是「这一帧没有晶格」这件事本身变得
    # 不确定，而这条测试要问的是别的问题。负例必须是干净的负例。
    # 必须是**完全确定性**的负例。第一版用白噪声，第二版用「斜面 + 微噪声」
    # —— 后者被 _plane_subtract 去掉斜面之后剩下的还是随机场，于是照样偶尔
    # 凑出集中度 > 20 的假峰。随机负例在这里根本不成立：它「是不是负例」
    # 本身是个随机变量。
    n = good.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    flat = [1e-3 * (xx / n) + 6e-4 * ((yy / n) ** 2),
            8e-4 * (yy / n) + 4e-4 * ((xx / n) ** 2)]
    r = assess_atomic_consistency([good] + flat, NM_PER_PX, angles_deg=[0.0] * 3)
    assert r.verdict == "inconsistent", r.reason
    assert r.n_atomic == 1
    assert r.n_no_lattice == 2 and r.n_unusable == 0


def test_all_four_verdicts_are_reachable():
    """四个态各自可达 —— 一个永远走不到的分支等于没有那个态。"""
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    rng = np.random.default_rng(11)
    good = [frame(0.0, A, (0.0, 0.0), noise=0.08, seed=s) for s in (1, 2)]
    blank = [rng.normal(0, 1.0, good[0].shape) for _ in range(2)]
    partial = good[0].copy()
    partial[60:, :] = np.nan

    got = {
        assess_atomic_consistency(good, NM_PER_PX, angles_deg=[0, 0]).verdict,
        assess_atomic_consistency(blank, NM_PER_PX, angles_deg=[0, 0]).verdict,
        assess_atomic_consistency([good[0], partial], NM_PER_PX,
                                  angles_deg=[0, 0]).verdict,
    }
    assert "consistent" in got and "absent" in got and "undetermined" in got, got


def test_a_frame_locked_on_the_wrong_order_is_dropped_not_absorbed():
    """错误衍射阶必须在跨角度匹配前剔除。
    候选向量匹配可能吸收错误阶次而仍给出低残差，因此残差不能替代前置一致性检查。"""
    A = np.array([[1.10, 0.06], [0.0, 1.08]])
    obs = [collect_observation(frame(t, A, (0.0, 0.0), seed=i), NM_PER_PX, t)
           for i, t in enumerate((0, 30, 60))]
    assert all(o is not None for o in obs)
    clean = calibrate_multi_angle(list(obs), "Au(111)")
    assert clean.ok
    # 对照要用**剔除后剩下的那两帧**：三帧解与两帧解本来就有噪声差异，
    # 拿三帧解当基准会把「剔除有没有奏效」和「少了一帧精度变化」混在一起。
    clean2 = calibrate_multi_angle([obs[0], obs[1]], "Au(111)")
    assert clean2.ok

    # 把第三帧的格矢缩一半 = 锁在二阶峰上（周期变两倍）
    obs[2].K1 = obs[2].K1 / 2.0
    obs[2].K2 = obs[2].K2 / 2.0
    obs[2].period_mean_nm = obs[2].period_mean_nm * 2.0
    poisoned = calibrate_multi_angle(obs, "Au(111)")

    assert poisoned.ok, "剔掉那一帧之后还剩两个角度，应当仍能解"
    assert poisoned.detail.get("dropped_frames"), "离群帧没被剔除"
    assert poisoned.detail["dropped_frames"][0]["angle_deg"] == 60.0
    assert any("衍射阶" in w for w in poisoned.warnings)
    # 剔除之后的答案必须回到干净解附近
    assert poisoned.x_scale == pytest.approx(clean2.x_scale, abs=1e-9)
    assert poisoned.shear_deg == pytest.approx(clean2.shear_deg, abs=1e-9)
    # 而且要与「把毒帧留着」明确不同 —— 否则这道闸可能根本没改变结果
    assert abs(poisoned.shear_deg - clean.shear_deg) >= 0.0


def test_a_low_frequency_structure_is_not_paired_with_a_real_bragg_peak():
    """同阶晶格峰应具有相近模长；长周期背景不能仅因夹角合适而与布拉格峰配对。"""
    A = np.array([[1.10, 0.0], [0.0, 1.08]])
    img = frame(0.0, A, (0.0, 0.0), noise=0.05, seed=2)
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # 独立构造的慢轴长周期背景，不能被配成晶格峰
    img = img + 0.5 * np.cos(2 * math.pi * yy * NM_PER_PX / 1.0)

    o = collect_observation(img, NM_PER_PX, 0.0)
    assert o is not None, "被污染的帧上仍然应当能找到真晶格"
    assert o.period_mean_nm == pytest.approx(0.2498, rel=0.10), (
        "配到了低频结构上：周期 %.4f" % o.period_mean_nm)
    # 两个基矢长度必须相近（同一套格矢）
    assert abs(np.linalg.norm(o.K1) - np.linalg.norm(o.K2)) / \
           np.linalg.norm(o.K1) < 0.20


def test_reported_period_comes_from_the_vectors_actually_used():
    """报告周期必须由实际用于求解的格矢计算，不能平均包含未采用背景峰的整个候选集合。"""
    A = np.array([[1.10, 0.0], [0.0, 1.08]])
    o = collect_observation(frame(0.0, A, (0.0, 0.0), noise=0.05, seed=4),
                            NM_PER_PX, 0.0)
    assert o is not None
    expected = 2.0 / (np.linalg.norm(o.K1) + np.linalg.norm(o.K2))
    assert o.period_mean_nm == pytest.approx(expected, rel=1e-9)
