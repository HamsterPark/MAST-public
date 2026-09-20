"""原子相判据(mast.vision.atomic_phase)。

**第一节是纯噪声与无晶格表面的误报率** —— 与肖克利 onset 同一条铁律。这个判据
说「有原子相」意味着配方宣布针尖锻好了、停止扰动;在噪声上说这句话,后面整批数据
都建立在一根其实不行的针尖上。

合成一律**物理**:晶格周期取 Au(111) 的原子行间距(√3/2 × 0.288 = 0.249 nm),
起伏取金属表面原子分辨的真实量级(~10 pm),噪声取 ~2 pm,视野与像素取配方真正
会用的 5 nm / 256 px(= 0.0195 nm/px,刚好落在满权重尺度档)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mast.vision import atomic_phase as A


# ── 物理合成 ──────────────────────────────────────────────────────────────────

AU_NN_NM = 0.288                      # Au(111) 最近邻距离
ROW_SPACING_NM = AU_NN_NM * math.sqrt(3.0) / 2.0     # 0.2494 nm —— 原子行间距
FRAME_NM = 5.0
PIXELS = 256
NMPP = FRAME_NM / PIXELS              # 0.01953 nm/px —— 满权重档
CORRUGATION_M = 10e-12                # 10 pm
NOISE_M = 2e-12                       # 2 pm


def _hex_lattice(*, period_nm=ROW_SPACING_NM, n=PIXELS, nmpp=NMPP,
                 amp=CORRUGATION_M, noise=NOISE_M, seed=0, shear_px_per_row=0.0,
                 tilt_rad=0.0):
    """一帧六角晶格的高度图(米)。

    三组波矢相隔 60° —— 这是 (111) 面的实际对称性,不是随手画的正弦条纹。
    ``shear_px_per_row`` 模拟慢轴漂移:每往下一行,整行沿快扫方向平移一点。
    """
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    if shear_px_per_row:
        x = x + y * float(shear_px_per_row)
    xs, ys = x * nmpp, y * nmpp
    k = 2.0 * math.pi / float(period_nm)
    h = np.zeros((n, n), dtype=np.float64)
    for deg in (0.0, 60.0, 120.0):
        th = math.radians(deg) + tilt_rad
        h += np.cos(k * (xs * math.cos(th) + ys * math.sin(th)))
    h = h / 3.0 * float(amp)
    if noise:
        h = h + np.random.default_rng(seed).normal(0.0, float(noise), h.shape)
    return h


def _noise_frame(*, n=PIXELS, sigma=NOISE_M, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, (n, n))


def _stepped_frame(*, n=PIXELS, step_m=240e-12, noise=NOISE_M, seed=0):
    """有台阶、没有原子分辨的表面 —— 针尖钝但表面很好的典型帧。"""
    y, _x = np.mgrid[0:n, 0:n].astype(np.float64)
    h = np.where(y > n * 0.55, step_m, 0.0)
    return h + np.random.default_rng(seed).normal(0.0, noise, h.shape)


def _quasi_periodic_noise(*, seed=0, period_nm=ROW_SPACING_NM, n=PIXELS,
                          nmpp=NMPP, frac=0.15, amp=10e-12):
    """带通白噪声 —— 针尖抖动/反馈振铃造出的**准周期**条纹。

    这是整个判据最难的对照组:它在 FFT 的原子带里produce 一个合格的谱峰(实测 30
    个种子 30 个都过 SNR 门),自相关角向最大值也与真晶格重叠。峰强度类判据一个
    都拦不住它 —— 只有「离散布拉格点 vs 弥散环」分得开。
    """
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.normal(0.0, 1.0, (n, n)))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    fr = np.hypot(fy, fx)
    f0 = nmpp / float(period_nm)          # 目标周期对应的空间频率(周期/像素)
    band = np.exp(-((fr - f0) ** 2) / (2.0 * (frac * f0) ** 2))
    return np.real(np.fft.ifft2(F * band)) * float(amp)


# ── 1. 误报:噪声与无晶格表面 ────────────────────────────────────────────────

def test_pure_noise_never_passes():
    """40 个种子的纯高斯白噪声,一个都不许通过。"""
    bad = [s for s in range(40)
           if A.assess_atomic_phase(_noise_frame(seed=s), nm_per_px=NMPP).passed]
    assert not bad, f"纯噪声上出现误报（种子 {bad}）"


def test_stepped_surface_without_atoms_never_passes():
    """台阶清晰但没有原子相 —— 这正是「针尖还不够好」的典型帧。"""
    for s in range(10):
        res = A.assess_atomic_phase(_stepped_frame(seed=s), nm_per_px=NMPP)
        assert not res.passed, f"台阶帧被判成原子相（seed={s}）：{res}"


def test_row_noise_alone_does_not_count_as_a_lattice():
    """逐行随机偏置(常见的扫描线噪声)会在慢轴方向造出周期性,但那不是晶格。"""
    rng = np.random.default_rng(3)
    h = np.repeat(rng.normal(0.0, 8e-12, (PIXELS, 1)), PIXELS, axis=1)
    h = h + rng.normal(0.0, NOISE_M, h.shape)
    assert not A.assess_atomic_phase(h, nm_per_px=NMPP).passed


def test_quasi_periodic_tip_ringing_never_passes():
    """**最难的对照组**:带通噪声在原子带里有合格谱峰,但不是晶格。

    实测(见模块 docstring):30 个种子全部通过带内 SNR 门、自相关角向最大值
    0.13-0.28 与 6 pm 噪声下的真晶格 0.23 完全重叠。峰强度类判据一个都拦不住,
    只有角向集中度分得开。这条测试红了就说明判据退回到「FFT 里有峰就算数」。
    """
    passed = [s for s in range(30)
              if A.assess_atomic_phase(_quasi_periodic_noise(seed=s),
                                       nm_per_px=NMPP).passed]
    assert not passed, f"准周期抖动被判成原子相（种子 {passed}）"


def test_angular_concentration_separates_lattice_from_ringing():
    """判据的分离度必须留有余量 —— 这是阈值 20 的实测依据。

    真晶格(含 10 pm 噪声这种起伏与噪声 1:1 的极限情况)与准周期抖动之间实测差
    一个数量级以上。余量塌了要么是判据被改坏了,要么是合成变得不物理。
    """
    from mast.vision.seg_scale_adaptive import DEFAULTS, detect_texture, flatten_robust

    def _conc(frame):
        flat = flatten_robust(frame)
        p = dict(DEFAULTS)
        p["atomic_band_nm"] = A.ATOMIC_BAND_NM
        p["lat_snr"] = 4.0
        tex = detect_texture(flat, NMPP, p)
        atomic = tex.get("atomic")
        assert atomic, "对照组必须先有谱峰，否则这条测试没在测它想测的东西"
        return A.angular_concentration(flat, float(atomic[0]))

    worst_lattice = min(_conc(_hex_lattice(noise=10e-12, seed=s))
                        for s in range(5))
    best_ringing = max(_conc(_quasi_periodic_noise(seed=s)) for s in range(10))
    assert worst_lattice > A.DEFAULT_CONCENTRATION_MIN > best_ringing, (
        f"阈值 {A.DEFAULT_CONCENTRATION_MIN} 没有夹在两组之间："
        f"最差晶格 {worst_lattice:.1f}，最强抖动 {best_ringing:.1f}")
    assert worst_lattice > best_ringing * 5.0, (
        f"分离度只有 {worst_lattice / best_ringing:.1f} 倍，余量不足")


# ── 2. 真晶格:必须认出来 ────────────────────────────────────────────────────

def test_clean_hex_lattice_passes():
    res = A.assess_atomic_phase(_hex_lattice(seed=1), nm_per_px=NMPP,
                                expected_a_nm=ROW_SPACING_NM)
    assert res.passed, f"干净的六角晶格没被认出来：{res.reasons}"
    assert res.scale == "full"
    assert res.period_fast_axis_nm is not None
    assert res.period_fast_axis_nm == pytest.approx(ROW_SPACING_NM, rel=0.20)
    assert res.angular_concentration >= A.DEFAULT_CONCENTRATION_MIN
    assert res.slow_axis_trusted is False       # 帧法恒为 False


@pytest.mark.parametrize("noise_pm", [1.0, 2.0, 4.0, 6.0])
def test_lattice_survives_realistic_noise(noise_pm):
    """用解析晶格与独立噪声验证低幅度信号的识别能力。"""
    res = A.assess_atomic_phase(
        _hex_lattice(amp=12e-12, noise=noise_pm * 1e-12, seed=5), nm_per_px=NMPP)
    assert res.passed, f"噪声 {noise_pm} pm 下漏检：{res.reasons}"


# ── 3. 慢轴漂移:判定不受影响,晶格常数只报快轴 ─────────────────────────────

def test_drift_shear_does_not_break_detection():
    """慢轴漂移把方晶格剪成菱形。判「有没有原子相」必须照样成立。"""
    res = A.assess_atomic_phase(
        _hex_lattice(seed=2, shear_px_per_row=0.15), nm_per_px=NMPP)
    assert res.passed, f"漂移剪切下漏检：{res.reasons}"


def test_fast_axis_period_is_immune_to_slow_axis_drift():
    """逐行 1D 谱对慢轴漂移完全免疫 —— 这是它存在的唯一理由。

    同一块晶格,加不加剪切,快轴周期必须一致(而径向周期会被拉偏)。
    """
    clean = A.fast_axis_period_nm(_hex_lattice(seed=4, noise=0.0), NMPP)[0]
    sheared = A.fast_axis_period_nm(
        _hex_lattice(seed=4, noise=0.0, shear_px_per_row=0.30), NMPP)[0]
    assert clean is not None and sheared is not None
    assert sheared == pytest.approx(clean, rel=0.05), (
        f"快轴周期被慢轴漂移改变了：{clean:.4f} → {sheared:.4f} nm")


def test_fast_axis_reports_the_projected_period_for_an_oblique_lattice():
    """晶格与快扫方向成角度时,快轴测到的是投影 a/|cosθ| —— 只会更大。

    所以与已知晶格常数比对必须「下界严、上界松」。
    """
    res = A.assess_atomic_phase(
        _hex_lattice(seed=6, tilt_rad=math.radians(20.0)), nm_per_px=NMPP,
        expected_a_nm=ROW_SPACING_NM)
    assert res.period_fast_axis_nm is not None
    assert res.period_fast_axis_nm >= ROW_SPACING_NM * 0.85


# ── 4. 尺度门:判不了要说判不了 ─────────────────────────────────────────────

def test_coarse_pixels_refuse_to_judge_rather_than_report_absence():
    """0.06 nm/px 上晶格物理上不可分辨 —— 结论必须是「判不了」。

    区别很实在:「没有原子相」会让配方接着扰动针尖;「这一帧判不了」应该让它换个
    更小的视野再看。
    """
    res = A.assess_atomic_phase(_hex_lattice(seed=1), nm_per_px=0.06)
    assert not res.passed
    assert res.scale == "off"
    assert res.reasons == ("scale_gate",)


def test_transition_band_is_not_a_positive_verdict_by_default():
    """0.02..0.05 nm/px 是过渡带:证据强度撑不住一次针尖验收。"""
    res = A.assess_atomic_phase(_hex_lattice(seed=1), nm_per_px=0.03)
    assert res.scale == "reduced"
    assert not res.passed
    assert "scale_reduced" in res.reasons
    # 显式放行时才给正面结论,并且仍然留 warning。
    res2 = A.assess_atomic_phase(_hex_lattice(seed=1), nm_per_px=0.03,
                                 allow_reduced_scale=True)
    assert "scale_reduced" in res2.warnings


def test_unknown_pixel_size_refuses_to_judge():
    res = A.assess_atomic_phase(_hex_lattice(seed=1), nm_per_px=None)
    assert not res.passed
    assert res.reasons == ("unknown_pixel_size",)
    assert A.scale_gate(None) is None
    assert A.scale_gate(0.0) is None


def test_scale_gate_boundaries():
    assert A.scale_gate(0.0195) == "full"
    assert A.scale_gate(0.02) == "reduced"      # 边界含在过渡带
    assert A.scale_gate(0.05) == "reduced"
    assert A.scale_gate(0.051) == "off"


# ── 5. 与已知晶格常数比对 ───────────────────────────────────────────────────

def test_period_far_below_the_known_lattice_is_rejected():
    """投影不会让周期变小 —— 明显更小的周期不是这个晶格。"""
    res = A.assess_atomic_phase(
        _hex_lattice(period_nm=0.19, seed=7), nm_per_px=NMPP,
        expected_a_nm=0.40)
    assert not res.passed
    assert "period_below_lattice" in res.reasons


# ── 6. 纯函数纪律 ───────────────────────────────────────────────────────────

def test_never_raises_on_garbage():
    for bad in (np.zeros((4, 4)), np.full((64, 64), np.nan), np.zeros((64, 64))):
        res = A.assess_atomic_phase(bad, nm_per_px=NMPP)
        assert not res.passed


def test_result_is_frozen():
    res = A.assess_atomic_phase(_hex_lattice(seed=1), nm_per_px=NMPP)
    with pytest.raises(Exception):
        res.passed = False           # type: ignore[misc]


def test_thresholds_are_parameters_not_globals():
    frame = _hex_lattice(seed=1)
    assert A.assess_atomic_phase(frame, nm_per_px=NMPP).passed
    assert not A.assess_atomic_phase(frame, nm_per_px=NMPP,
                                     sharpness_min=1e6).passed
    assert not A.assess_atomic_phase(frame, nm_per_px=NMPP,
                                     concentration_min=1e9).passed


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ── 未扫的行不许参与判读────────────────────────────

def test_a_half_scanned_frame_is_judged_on_what_was_actually_scanned():
    """部分采集帧只在已经扫描的行上判读。

    以解析晶格构造完整图，再分别注入 NaN 与全零未采集区域；内部裁行的结果
    必须与调用方手工裁行完全一致，避免把未采集背景当作降低成像质量的证据。
    """
    rs = np.random.RandomState(7)
    n, nmpp = 256, 5.0 / 256
    yy, xx = np.mgrid[0:n, 0:n]
    a_px = 0.25 / nmpp
    lat = sum(np.cos(2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / a_px)
              for th in (0.0, np.pi / 3, 2 * np.pi / 3)) * 2e-11
    full = lat + rs.randn(n, n) * 2e-12 - 1.5e-7

    whole = A.assess_atomic_phase(full, nm_per_px=nmpp)
    assert whole.passed, whole.reasons

    # 判据：**函数内部裁行，必须等于我手工裁好再喂进去**。
    # 不拿「和整帧比」当判据 —— 行数少了，慢轴方向的 FFT 峰本来就变宽，
    # 角向集中度可能改变。那是数据变少的物理后果，不是缺陷；
    # 拿它当判据只会逼出一条凭手感调出来的容差带。
    hand_cropped = A.assess_atomic_phase(full[110:], nm_per_px=nmpp)
    for tag, blank in (("NaN（.sxm 那一路）", np.nan),
                       ("全零（活体缓冲那一路）", 0.0)):
        half = full.copy()
        half[:110] = blank
        got = A.assess_atomic_phase(half, nm_per_px=nmpp)
        assert got.passed, f"{tag}：半张帧被判成没有晶格 —— {got.reasons}"
        assert got.angular_concentration == pytest.approx(
            hand_cropped.angular_concentration, rel=1e-6), (
            f"{tag}：函数内部裁出来的结果（{got.angular_concentration:.1f}）"
            f"与手工裁好再喂（{hand_cropped.angular_concentration:.1f}）不一致 —— "
            f"未扫的那片还在参与判读")


def test_a_frame_with_only_a_few_scanned_rows_says_it_cannot_tell():
    """扫出来的太少 ⇒ 说「判不了」，**不说「没有」**。"""
    img = np.full((256, 256), np.nan)
    img[:8] = np.random.RandomState(1).randn(8, 256) * 1e-11 - 1.5e-7
    got = A.assess_atomic_phase(img, nm_per_px=5.0 / 256)
    assert not got.passed
    assert "insufficient_data" in got.reasons, got.reasons


def test_an_entirely_flat_frame_is_still_dead_flat_not_unscanned():
    """整帧全零是**测出来是零**（反馈关掉/没接上），不是「没测到」。

    零只在**有非零行作对照**时才说明「没写过」。把这两者合并，
    一次「反馈没接上」就会被报成「什么都没扫到」——
    而这两句话要人做的事完全不同。
    """
    got = A.assess_atomic_phase(np.zeros((256, 256)), nm_per_px=5.0 / 256)
    assert "dead_flat" in got.reasons, got.reasons


# ── 帧内针尖突变：判据只**报告**两半，不改判定 ──────────────────────────

def _lattice_frame(n=256, nmpp=5.0 / 256, seed=7, amp=2e-11, noise=2e-12):
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    a_px = 0.25 / nmpp
    lat = sum(np.cos(2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / a_px)
              for th in (0.0, np.pi / 3, 2 * np.pi / 3)) * amp
    return lat + rs.randn(n, n) * noise - 1.5e-7


def test_the_two_halves_are_reported_so_a_caller_can_see_a_mid_frame_change():
    """按扫描顺序分别报告前后半帧的集中度。

    解析晶格与独立噪声拼成两半，模拟扫描中途状态变化。整帧统计可能掩盖
    混合状态，因此必须提供分段数据；此处只报告，认证决定由调用方作出。
    半帧与整帧具有不同谱宽，不能直接复用整帧绝对集中度下限。
    """
    nmpp = 5.0 / 256
    rs = np.random.RandomState(11)
    f = _lattice_frame(nmpp=nmpp)
    f[128:] = rs.randn(128, 256) * 8e-12 - 1.5e-7          # 后半只剩噪声

    r = A.assess_atomic_phase(f, nm_per_px=nmpp)
    assert r.half_concentrations is not None
    first, second = r.half_concentrations
    assert first > 0 and second >= 0
    assert second / first < 0.04, (first, second)           # 断裂：比值极小


def test_a_uniform_lattice_has_two_comparable_halves():
    """均匀晶格的两半**互相可比** —— 这是上一条的反证。

    比值判据用的是「后/前」，不是绝对值：半帧效应对两半是**同等**作用的，
    所以它不影响比值。实测均匀晶格（1–6 pm 噪声）比值 0.93–1.13。
    """
    nmpp = 5.0 / 256
    r = A.assess_atomic_phase(_lattice_frame(nmpp=nmpp, noise=4e-12), nm_per_px=nmpp)
    first, second = r.half_concentrations
    assert 0.2 < second / first < 5.0, (first, second)


def test_reporting_the_halves_never_changes_the_verdict():
    """**这一条钉的是「不改判定」**：同一帧，开/关半帧计算，``passed`` 与
    ``reasons`` 必须一模一样。

    第一版把它做成了判定，当场打红 4 条既有测试，还让这个文档承诺
    「不抛异常」的纯函数抛了 ``TypeError``。
    """
    nmpp = 5.0 / 256
    rs = np.random.RandomState(11)
    f = _lattice_frame(nmpp=nmpp)
    f[128:] = rs.randn(128, 256) * 8e-12 - 1.5e-7
    on = A.assess_atomic_phase(f, nm_per_px=nmpp)
    off = A.assess_atomic_phase(f, nm_per_px=nmpp, _check_halves=False)
    assert on.passed == off.passed
    assert on.reasons == off.reasons
    assert off.half_concentrations is None


def test_a_frame_too_short_to_split_is_not_judged_on_halves():
    """行太少就**不切** —— 切出来的半帧慢轴只有十几行，判据本来就不该说话。"""
    nmpp = 5.0 / 256
    r = A.assess_atomic_phase(_lattice_frame(nmpp=nmpp)[:24], nm_per_px=nmpp)
    assert r.half_concentrations is None, "24 行还去切两半"


def test_a_frame_too_thin_for_a_reliable_half_fft_is_not_split():
    """半帧计算须满足最小物理视野，而非只满足最少行数。

    把解析晶格裁成窄条，验证几何条件不足时不报告两半统计；不能把无法测量
    产生的两个零值解释为扫描中途针尖突变。完整合成帧仍应提供两半结果。
    """
    nmpp = 5.0 / 256
    full = _lattice_frame(nmpp=nmpp)                       # 256 行，半帧 2.5 nm
    r_full = A.assess_atomic_phase(full, nm_per_px=nmpp)
    assert r_full.half_concentrations is not None, "整帧本来就该切"

    narrow = full[:150]                                    # 半帧 75 行 ≈ 1.46 nm
    r = A.assess_atomic_phase(narrow, nm_per_px=nmpp)
    assert r.half_concentrations is None, (
        "半帧只有 %.2f nm 还去切 —— 两半会回 0.0，下游读成「针尖变了」"
        % (narrow.shape[0] / 2 * nmpp))
    assert r.passed, r.reasons                             # 窄归窄，晶格还在


# ── 视野里装不下足够多的周期 = 判不了，不是「没有」───────────────────────────
#
# 实测边界（2026-08-23，合成完美晶格 + 二分）：**4.70–4.75 个周期**，在
# 128/256/384 三种像素数、0.235/0.288/0.400 三种晶格常数、噪声 0→2× 信号幅度下
# 全都一样 —— ``seg_scale_adaptive`` 把可搜周期上限压到 ``min(H, W)/4`` 带来的
# **纯几何**限制，不是统计阈值。
#
# 可见周期不足时应报告无法判读，避免让调用方把几何欠采样误当作针尖不合格。

def _perfect_lattice_frame(n_px: int, frame_nm: float, a_nm: float = 0.288):
    """一张数学上完美的六角晶格。回 (高度数组, nm/px)。"""
    nmpp = frame_nm / n_px
    y, x = np.mgrid[0:n_px, 0:n_px].astype(float) * nmpp
    g = 2 * math.pi / a_nm
    z = sum(np.cos(g * (x * math.cos(t) + y * math.sin(t)))
            for t in (0.0, math.pi / 3, 2 * math.pi / 3))
    return z * 20e-12, nmpp


def test_a_frame_too_small_for_the_lattice_says_undecidable_not_absent():
    """完美晶格 + 装不下的视野 ⇒ 出局词属于「判不了」，不属于「没有」。"""
    z, nmpp = _perfect_lattice_frame(128, 1.0)                    # ≈ 3.5 个 Au 周期
    r = A.assess_atomic_phase(z, nm_per_px=nmpp, expected_a_nm=0.288)
    assert not r.passed
    assert "too_few_periods" in r.reasons
    assert "not_a_lattice" not in r.reasons, (
        "这是一张数学上完美的晶格 —— 说「不是晶格」就是把「没往那里看」"
        "报成了「看过了，没有」")


def test_a_frame_with_room_to_spare_is_judged_normally():
    """反证：同一晶格给足视野必须照常判得出来 —— 闸门不许顺手吞掉好帧。"""
    z, nmpp = _perfect_lattice_frame(256, 4.0)
    r = A.assess_atomic_phase(z, nm_per_px=nmpp, expected_a_nm=0.288)
    assert r.passed, r.reasons
    assert "too_few_periods" not in r.reasons


def test_without_an_expected_period_the_gate_falls_back_to_the_band_floor():
    """调用方没说期望周期时，只在**连最小的可能晶格都装不下**时才开口。

    否则「2 nm 视野里找 0.8 nm 周期」这类正当调用会被一起判成判不了。
    """
    small, nmpp_s = _perfect_lattice_frame(128, 0.6)              # < 5 × 0.18 nm
    assert "too_few_periods" in A.assess_atomic_phase(small, nm_per_px=nmpp_s).reasons
    ok, nmpp_o = _perfect_lattice_frame(256, 2.0)                 # 装得下 0.18 nm 的 11 个周期
    assert "too_few_periods" not in A.assess_atomic_phase(ok, nm_per_px=nmpp_o).reasons


@pytest.mark.parametrize("n_px", [128, 256, 384])
@pytest.mark.parametrize("a_nm", [0.235, 0.288, 0.400])
def test_the_ability_boundary_is_in_periods_not_pixels(n_px: int, a_nm: float):
    """判得了与否只跟**周期数**有关，跟像素数无关 —— 这条一红就是几何约定被改了。

    4 个周期判不了、6 个周期判得了，三种像素数 × 三种晶格常数一致。
    """
    below, nmpp_b = _perfect_lattice_frame(n_px, 4.0 * a_nm, a_nm)
    above, nmpp_a = _perfect_lattice_frame(n_px, 6.0 * a_nm, a_nm)
    assert "too_few_periods" in A.assess_atomic_phase(
        below, nm_per_px=nmpp_b, expected_a_nm=a_nm).reasons
    r = A.assess_atomic_phase(above, nm_per_px=nmpp_a, expected_a_nm=a_nm)
    assert r.passed, r.reasons


def test_the_gate_does_not_fire_on_a_frame_that_merely_lacks_a_lattice():
    """一张视野够大、但**真的**没有晶格的帧，必须仍然报「没有」。

    闸门只该管「装不下」，不该把「装得下但没有」也说成判不了 —— 那会让
    修针流程在真该扰动针尖的时候按兵不动。
    """
    # 「判不了」那一组（与 ``ALL_REASONS`` 里的下半段同源）。断言问的是
    # **属不属于这一组**，不点名具体哪个词 —— 纯噪声落在哪一条判据上
    # （谱峰 / 集中度 / 锐度 / 快轴）不是本测试要钉的东西。
    undecidable = {"too_few_periods", "scale_gate", "scale_reduced",
                   "unknown_pixel_size", "insufficient_data", "dead_flat",
                   "dependency_unavailable"}
    rng = np.random.default_rng(7)
    flat = rng.standard_normal((256, 256)) * 5e-12      # 4 nm 视野的纯噪声
    r = A.assess_atomic_phase(flat, nm_per_px=4.0 / 256, expected_a_nm=0.288)
    assert not r.passed
    assert r.reasons, "没有晶格的帧必须给出理由"
    assert not (set(r.reasons) & undecidable), (
        "视野足够大、只是没有晶格 —— 报「判不了」会让修针流程在真该扰动"
        "针尖的时候按兵不动。实际给的是 %s" % (r.reasons,))


# 角向集中度不能证明多个峰属于同一个二维晶格。
# 弥散条纹也能在少数方向集中能量；配对还应核验一阶峰半径的一致性。
# 合成对角条纹用来隔离这一失效机制。

def _diagonal_streak(n=256, frame_nm=5.0, seed=0):
    """沿同一方向叠加不同周期余弦、大尺度鼓包和独立噪声。
    谱峰共线但半径不同，用作非晶格的解析反例。"""
    rng = np.random.default_rng(seed)
    nmpp = frame_nm / n
    y, x = np.mgrid[0:n, 0:n].astype(float) * nmpp
    u = (x + y) / math.sqrt(2.0)                    # 只沿对角一个方向
    z = np.zeros_like(u)
    for per in (0.9, 0.55, 0.35):
        z += np.cos(2 * math.pi * u / per) / per
    z += 3.0 * np.exp(-(((x - 1.5) ** 2 + (y - 3.5) ** 2) / 2.0))   # 大鼓包
    z += 0.15 * rng.standard_normal(z.shape)
    return z * 3e-10, nmpp                          # 独立设定的合成高度尺度


def test_a_diagonal_streak_is_not_a_lattice():
    """条纹骗得过角向集中度，但骗不过「峰在不在同一个半径上」。"""
    z, nmpp = _diagonal_streak()
    r = A.assess_atomic_phase(z, nm_per_px=nmpp)
    assert not r.passed, (
        "一条弥散对角条纹被判成了原子分辨（conc %.1f）—— "
        "不满足离散晶格峰的几何约束" % r.angular_concentration)
    assert "peaks_not_one_lattice" in r.reasons, r.reasons


def test_a_real_lattice_is_not_rejected_by_the_new_check():
    """反证：真晶格的一阶峰同半径，这道闸门碰都不该碰它。

    没有这一条，上面那条可以靠「拒绝一切」通过。
    """
    z, nmpp = _perfect_lattice_frame(256, 4.0)
    r = A.assess_atomic_phase(z, nm_per_px=nmpp, expected_a_nm=0.288)
    assert r.passed, r.reasons
    assert "peaks_not_one_lattice" not in r.reasons


def test_the_new_word_is_evidence_of_absence_not_of_ignorance():
    """``peaks_not_one_lattice`` 属「**没有**」那一组，不属「判不了」。

    分错组的代价是相反的下一步：判不了要改成像条件，没有才轮到动针尖。
    """
    from mast.skills.composite.verify_atomic_resolution import (
        REASON_VERDICT, VERDICT_ABSENT)

    assert REASON_VERDICT["peaks_not_one_lattice"] == (VERDICT_ABSENT, None)

# 绝对起伏下限不能覆盖不同衬度与噪声的晶格。
# 测试以低幅值解析晶格和噪声作对照，防止选定的绝对 pm 下限误伤有效信号。
#
# 一维波浪横带是角向集中度的已知盲区：能量集中在一个方向也会给出高分，
# 却不证明具有二维晶格。因此需要独立的逐行谱或其他结构证据交叉核验。
def _wavy_bands(*, n=128, nmpp=0.016, period_nm=0.22, amp_m=2e-12,
                wander=1.5, noise=0.05, seed=0):
    """沿慢轴的正弦条带叠加沿快轴的随机相位游走。
    wander 控制弯曲程度；此合成结构用于核验高角向集中度的一维模式仍被交叉证据拒绝。"""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    T = period_nm / nmpp
    ph = np.cumsum(rng.standard_normal(n)) * wander / np.sqrt(n)
    ph = ph - ph.mean()
    img = np.sin(2 * np.pi * y / T + ph[None, :])
    img = img + noise * rng.standard_normal((n, n))
    return img * amp_m


def test_straight_bands_are_not_atomic_resolution():
    """笔直横带 —— 基线:这一档判据本来就拦得住,拦不住说明合成写坏了。"""
    r = A.assess_atomic_phase(_wavy_bands(wander=0.0, seed=3), nm_per_px=0.016)
    assert not r.passed, "笔直横带被判成原子分辨:%s" % (r.reasons,)


@pytest.mark.parametrize("wander,seed", [(1.5, 3), (2.5, 3)])
def test_wavy_bands_are_rejected_by_the_fast_axis_disagreement(wander, seed):
    """波浪横带:二维谱与逐行谱在说**两个不同的周期** ⇒ 判否。

    这一条守的是 2026-08-24 把 ``radial_fast_axis_disagree`` 从 ``warns``
    挪进 ``reasons`` 那次改动。判据 4 的用意是「逐行谱才是可信的那个方向」,
    而「逐行谱什么也没看到」与「逐行谱看到的是另一个周期」是**同一种失败**——
    原来只有前者判否。
    """
    img = _wavy_bands(wander=wander, seed=seed)
    r = A.assess_atomic_phase(img, nm_per_px=0.016)
    assert r.angular_concentration > 500.0, (
        "这条用例的价值在于它 conc 很高却不是晶格;实际只有 %.1f,"
        "合成退化了就守不住东西了" % r.angular_concentration)
    assert not r.passed, "波浪横带(conc %.0f)被判成原子分辨" % r.angular_concentration
    assert "radial_fast_axis_disagree" in r.reasons, (
        "拦是拦住了,但不是被这一条拦的:%s —— 换一条拦住不算数,"
        "本用例钉的是这一条判据活着" % (r.reasons,))


@pytest.mark.xfail(strict=True, reason=(
    "已知缺口：相位游走足够大时，逐行谱与二维谱的相对差可能落入容差内，"
    "现有交叉核验仍可能放行一维条纹。仅提高集中度或要求更多峰也可能误伤"
    "缺少某些可见方向的晶格。保留 xfail，修复后通过 xpass 提示更新此回归。"))
def test_heavily_wandering_bands_are_still_a_false_positive():
    img = _wavy_bands(wander=4.0, seed=3)
    r = A.assess_atomic_phase(img, nm_per_px=0.016)
    assert not r.passed, "波浪横带 conc %.0f 仍被判成原子分辨" % r.angular_concentration


def test_the_disagreement_word_is_a_reason_not_a_warning():
    """**归属本身就是被守的东西。**

    它当了很久的告警:判据看见了、记下来了、然后放行 —— 生产方接好了而消费方
    不存在。这条用例钉的是「它在 reasons 里」,所以有人把它挪回 warns 时会红。
    """
    assert "radial_fast_axis_disagree" in A.ALL_REASONS
    import re
    src = A.assess_atomic_phase.__doc__ or ""
    del src
    import inspect
    body = inspect.getsource(A.assess_atomic_phase)
    assert re.search(r'reasons\.append\(\s*"radial_fast_axis_disagree"',
                     body), "它不在 reasons.append 里"
    assert not re.search(r'warns\.append\(\s*"radial_fast_axis_disagree"',
                         body), "它又回到 warns 里去了"
