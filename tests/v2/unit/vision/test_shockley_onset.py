"""肖克利表面态 onset 判据(mast.vision.spectroscopy.assess_shockley_onset)。

**第一节就是纯高斯白噪声的误报率** —— 项目铁律:此前已有四个「噪声/质量」判据
在这一条上被实测推翻。这个判据的误报后果很具体:它一旦在噪声上说「看到表面态
了」,配方就会宣布针尖合格并拿它去做谱,而实验数据从那一刻起全是废的。

合成一律**物理**:能量轴覆盖参考表面态(−0.8..0.3 V),onset 用常见表面态参考值
(Au(111) −0.49 V / Ag(111) −0.065 V),台阶宽度用热展宽 + lock-in 调制展宽算出来
的仪器分辨极限,而不是随手挑一个「看起来像」的数。
"""

from __future__ import annotations

import numpy as np
import pytest

from mast.vision import spectroscopy as S


# ── 物理合成 ──────────────────────────────────────────────────────────────────

AU_ONSET_V = -0.49        # 知识库 clean_metal.CONSTANTS["Au(111)"]
AG_ONSET_V = -0.065       # Ag(111):极浅,对展宽最敏感
SWEEP = (-0.8, 0.3)       # 知识库 sts_params 的 Au(111) 扫描窗口
N_PTS = 500
MOD_VRMS = 0.005          # 5 mV rms —— 4 K 常用
T_K = 4.2


def _axis(n=N_PTS):
    return np.linspace(SWEEP[0], SWEEP[1], n)


def _spectrum(onset_v, *, height=1.0, background=1.0, slope=0.0,
              noise=0.05, seed=0, mod=MOD_VRMS, temp=T_K, width_scale=1.0):
    """一条带肖克利台阶的合成 dI/dV 谱。

    台阶宽度取仪器展宽(热 + 调制),因为真实谱里就是它主导 —— 用一个更陡的台阶
    合成等于给判据送一道它在真机上永远遇不到的送分题。
    """
    v = _axis()
    w = S.broadening_floor_v(temp, mod) / 4.394 * width_scale   # 10-90 → logistic w
    g = S._logistic_step(v, background, slope, height, onset_v, w)
    if noise:
        g = g + np.random.default_rng(seed).normal(0.0, noise, v.size)
    return v, g


def _assess(v, g, *, expected=AU_ONSET_V, **kw):
    kw.setdefault("lockin_mod_vrms", MOD_VRMS)
    kw.setdefault("temperature_k", T_K)
    return S.assess_shockley_onset(v, g, expected_onset_v=expected, **kw)


# ── 1. 纯白噪声:零误报(硬门) ────────────────────────────────────────────────

def test_pure_white_noise_never_passes():
    """300 个种子的纯高斯白噪声,一个都不许通过。

    这是整个判据存在的前提。噪声里「找到」一个表面态,配方就会拿一根坏针尖去做
    一整轮谱学实验,而且报告上写着「已验证金属性」。
    """
    rng = np.random.default_rng(20260802)
    v = _axis()
    false_positives = []
    for seed in range(300):
        g = rng.normal(0.0, 1.0, v.size)
        res = _assess(v, g)
        if res.passed:
            false_positives.append((seed, res.onset_v, res.delta_bic,
                                    res.step_sigma_ratio))
    assert not false_positives, (
        f"纯白噪声上出现 {len(false_positives)} 次误报（应为 0）：{false_positives[:5]}")


def test_white_noise_on_a_sloped_background_never_passes():
    """带线性背景的白噪声也不许通过 —— dI/dV 背景本来就是斜的。"""
    rng = np.random.default_rng(11)
    v = _axis()
    for seed in range(120):
        g = 1.0 + 0.8 * v + rng.normal(0.0, 0.08, v.size)
        assert not _assess(v, g).passed, f"斜背景白噪声误报（seed={seed}）"


# ── 2. 真台阶:必须认出来 ────────────────────────────────────────────────────

def test_clean_step_at_expected_energy_passes():
    v, g = _spectrum(AU_ONSET_V, noise=0.05, seed=1)
    res = _assess(v, g)
    assert res.passed, f"干净的表面态台阶没被认出来：{res.reasons}"
    assert abs(res.onset_v - AU_ONSET_V) < 0.010, (
        f"onset 位置误差 {abs(res.onset_v - AU_ONSET_V) * 1000:.1f} mV，"
        f"应远小于 20 mV 容差")
    assert res.delta_bic > 50.0
    assert res.step_sigma_ratio > 5.0


@pytest.mark.parametrize("noise", [0.02, 0.05, 0.10, 0.15])
def test_step_survives_a_range_of_noise_levels(noise):
    """信噪比从 50 到 6.7 都该认得出 —— 真机上 lock-in 谱没有那么干净。"""
    v, g = _spectrum(AU_ONSET_V, noise=noise, seed=7)
    res = _assess(v, g)
    assert res.passed, f"噪声 {noise} 下漏检：{res.reasons}"


def test_onset_position_is_recovered_across_the_tolerance_band():
    """onset 在容差内平移时,判据要跟着它走而不是锁死在期望值上。"""
    for offset in (-0.015, -0.005, 0.005, 0.015):
        v, g = _spectrum(AU_ONSET_V + offset, noise=0.04, seed=3)
        res = _assess(v, g)
        assert res.passed, f"偏移 {offset * 1000:+.0f} mV 时漏检：{res.reasons}"
        assert abs(res.onset_v - (AU_ONSET_V + offset)) < 0.010


# ── 3. 该拒的都要拒 ─────────────────────────────────────────────────────────

def test_pure_linear_ramp_is_rejected_as_no_step():
    """线性斜坡不是台阶。纯线性模型更简约,ΔBIC 判据就是为这一条设的。"""
    v = _axis()
    g = 1.0 + 2.0 * v + np.random.default_rng(5).normal(0.0, 0.02, v.size)
    res = _assess(v, g)
    assert not res.passed
    assert "no_step" in res.reasons or "low_amplitude" in res.reasons


def test_step_at_the_wrong_energy_is_rejected():
    """台阶存在但位置差 150 mV —— 那不是这个面的表面态。"""
    v, g = _spectrum(AU_ONSET_V + 0.15, noise=0.04, seed=2)
    res = _assess(v, g)
    assert not res.passed
    assert "onset_out_of_window" in res.reasons


def test_tiny_step_below_the_noise_is_rejected():
    """台阶高度只有噪声的两倍 —— 拟合得出来,但不该当成证据。"""
    v, g = _spectrum(AU_ONSET_V, height=0.10, noise=0.05, seed=4)
    res = _assess(v, g)
    assert not res.passed
    assert "low_amplitude" in res.reasons


def test_a_step_far_broader_than_the_instrument_limit_is_rejected():
    """展宽到 20 倍仪器极限的「台阶」是一段缓慢抬升,不是 onset。"""
    v, g = _spectrum(AU_ONSET_V, noise=0.02, seed=6, width_scale=20.0)
    res = _assess(v, g)
    assert not res.passed
    assert "too_wide" in res.reasons


def test_single_spike_is_not_fitted_as_an_infinitely_sharp_step():
    """一个 tip switch 的尖峰不许被拟合成宽度→0 的完美台阶。

    宽度下限是物理先验(台阶不可能比 kT + 调制展宽更陡),没有它这条测试会红。
    """
    v = _axis()
    g = 1.0 + np.random.default_rng(9).normal(0.0, 0.02, v.size)
    g[np.argmin(np.abs(v - AU_ONSET_V))] += 3.0      # 单点尖峰
    res = _assess(v, g)
    assert not res.passed, f"单点尖峰被当成了表面态：{res}"


def test_too_few_points_reports_insufficient_data():
    res = _assess(np.linspace(-0.6, -0.4, 8), np.zeros(8))
    assert not res.passed
    assert "insufficient_data" in res.reasons


def test_window_entirely_on_one_side_of_the_onset_is_insufficient():
    """窗口只覆盖 onset 一侧时,边缘斜率不能冒充台阶。"""
    v = np.linspace(-0.45, 0.3, 300)
    g = 1.0 + 2.0 * v
    res = _assess(v, g)
    assert not res.passed
    assert "insufficient_data" in res.reasons


# ── 4. 倒置谱 = 接线问题,不是针尖问题 ──────────────────────────────────────

def test_inverted_spectrum_is_flagged_as_wiring_not_tip():
    """lock-in 相位差 180° 时谱整体倒置。

    判据不自动翻转(那会把「相位设错了」悄悄变成「针尖不合格」),但要在 warnings
    里点名 —— 否则用户会去修一根其实没问题的针尖。
    """
    v, g = _spectrum(AU_ONSET_V, noise=0.04, seed=8)
    res = _assess(v, -g)
    assert not res.passed
    assert "didv_may_be_inverted" in res.warnings, (
        f"倒置谱没被识别出来：reasons={res.reasons} warnings={res.warnings}")


# ── 5. 展宽:Ag(111) 是最难的一个 ───────────────────────────────────────────

def test_broadening_floor_matches_hand_calculation():
    # 4.2 K + 5 mV rms:热展宽 1.27 mV、调制展宽 12.5 mV → 方和根 ≈ 12.6 mV
    assert S.broadening_floor_v(4.2, 0.005) == pytest.approx(0.01256, abs=2e-4)
    # 77 K + 10 mV rms:热展宽已达 23 mV,与 Ag(111) 的 onset 同量级
    assert S.broadening_floor_v(77.0, 0.010) == pytest.approx(0.0341, abs=5e-4)
    assert S.broadening_floor_v(0.0, 0.0) == 0.0


def test_ag111_at_77k_warns_that_broadening_rivals_the_onset():
    """Ag(111) 的 −65 mV 在 77 K + 10 mV 调制下,展宽与 onset 深度同量级。

    结论仍可能成立,但用户必须知道分辨力已经很勉强 —— 这正是知识库说 Ag(111)
    是「最灵敏的质量指标」的另一面。
    """
    v = np.linspace(-0.3, 0.2, 400)
    w = S.broadening_floor_v(77.0, 0.010) / 4.394
    g = S._logistic_step(v, 1.0, 0.0, 1.0, AG_ONSET_V, w)
    res = S.assess_shockley_onset(
        v, g, expected_onset_v=AG_ONSET_V, tol_v=0.020,
        lockin_mod_vrms=0.010, temperature_k=77.0, width_max_v=0.040)
    assert "broadening_comparable_to_onset" in res.warnings


def test_ag111_at_4k_is_a_normal_pass():
    """同一个 Ag(111) 在 4 K + 2 mV 调制下就是一次普通的通过。"""
    v = np.linspace(-0.3, 0.2, 400)
    w = S.broadening_floor_v(4.2, 0.002) / 4.394
    g = S._logistic_step(v, 1.0, 0.0, 1.0, AG_ONSET_V, w)
    g = g + np.random.default_rng(12).normal(0.0, 0.03, v.size)
    res = S.assess_shockley_onset(
        v, g, expected_onset_v=AG_ONSET_V, tol_v=0.020,
        lockin_mod_vrms=0.002, temperature_k=4.2)
    assert res.passed, res.reasons


# ── 6. 纯函数纪律 ───────────────────────────────────────────────────────────

def test_result_is_frozen_and_never_raises_on_garbage():
    """判据不许抛 —— 上游是一条可能什么都没测到的谱。"""
    for bad in ([], [np.nan] * 40, [1.0] * 40):
        res = S.assess_shockley_onset(bad, bad, expected_onset_v=AU_ONSET_V)
        assert not res.passed
    v, g = _spectrum(AU_ONSET_V, seed=1)
    res = _assess(v, g)
    with pytest.raises(Exception):
        res.passed = False          # type: ignore[misc]  frozen


def test_thresholds_are_all_parameters_not_globals():
    """把幅度门抬到荒谬值,同一条谱必须落选 —— 阈值确实是参数。"""
    v, g = _spectrum(AU_ONSET_V, noise=0.05, seed=1)
    assert _assess(v, g).passed
    assert not _assess(v, g, amp_sigma_k=1e6).passed


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
