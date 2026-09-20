"""倾斜估计与台阶主导判据的独立合成回归。

用解析倾斜面、阶梯、二次曲面与高斯噪声检查误报、几何单位换算和检测边界。
高度单位为米，默认噪声幅度为独立设定的 20 pm，不复用站点测量。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mast.vision import tilt as T


# ── 物理合成 ──────────────────────────────────────────────────────────────────

NOISE_M = 20e-12         # 独立合成高斯噪声幅度，不代表仪器或表面测量
STEP_M = 240e-12         # 240 pm —— 金属单原子台阶
FRAME_M = 100e-9         # 100 nm 视野
N_PX = 256


def _noise(n=N_PX, sigma=NOISE_M, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n, n))


def _tilted(angle_deg, *, n=N_PX, frame_m=FRAME_M, sigma=NOISE_M, seed=1,
            axis="x"):
    """一个真实倾斜 angle_deg 的平面 + 高斯噪声。

    斜率是无量纲的 tan(θ);像素坐标乘以 m/px 才是物理距离。
    """
    m_per_px = frame_m / n
    gy, gx = np.mgrid[:n, :n].astype(np.float64)
    slope = math.tan(math.radians(angle_deg))
    ramp = (gx if axis == "x" else gy) * m_per_px * slope
    return ramp + _noise(n, sigma, seed)


def _stepped(n_steps, *, n=N_PX, sigma=NOISE_M, seed=2, angle_deg=20.0):
    """n_steps 个等间距原子台阶,整体绕帧法线转了 angle_deg。

    转角是必须的:完全平行于快扫轴(水平)的台阶对分割器不可见(逐行中位数
    差分会把纯 y 向阶梯吸收掉),那是另一个用例专门测的场景。
    """
    gy, gx = np.mgrid[:n, :n].astype(np.float64)
    th = math.radians(angle_deg)
    proj = gx * math.sin(th) + gy * math.cos(th)
    terrace_width = n / max(n_steps, 1)
    height = np.floor(proj / terrace_width) * STEP_M
    return height + _noise(n, sigma, seed)


# ══════════════════════════════════════════════════════════════════════════
#  一、纯高斯白噪声的误报率(铁律)
# ══════════════════════════════════════════════════════════════════════════

def test_pure_gaussian_noise_does_not_trigger_step_dominance():
    """纯噪声上「台阶主导」的误报率必须为 0。

    误报的后果是调平永远被跳过 —— 而且理由是「这里台阶太密不好判断」,听起来
    完全合理,没有人会去质疑它。
    """
    false_hits = 0
    for seed in range(20):
        verdict = T.assess_steps(_noise(seed=seed), use_segmentation=False)
        if verdict.step_dominated:
            false_hits += 1
    assert false_hits == 0, f"20 次纯噪声里有 {false_hits} 次误报台阶主导"


def test_pure_gaussian_noise_dominance_ratio_stays_at_the_measured_floor():
    """纯噪声的结构比值应接近 1，且不随帧尺寸与噪声幅度改变。
    同时要求与生产阈值之间留出明确余量，防止把无结构噪声误判为台阶。
    """
    worst = 0.0
    for n in (128, 256, 512):
        for sigma in (5e-12, NOISE_M, 100e-12):
            for seed in range(5):
                img = T.detrend_quadratic(_noise(n=n, sigma=sigma, seed=seed))
                worst = max(worst, T.step_dominance_multiscale(img)[0])
    assert worst < 1.15, f"纯噪声的多尺度比值最大 {worst:.4f},超出合成噪声允许范围"
    # 阈值必须留出真实余量,否则一点噪声就会变成"有台阶"
    assert T.STRUCTURE_RATIO_THRESHOLD > worst * 1.2


def test_pure_gaussian_noise_yields_near_zero_tilt():
    """纯噪声上不该「测出」倾斜 —— 否则系统会去调一个不存在的倾斜。"""
    for seed in range(10):
        est = T.estimate_tilt(_noise(seed=seed), width_m=FRAME_M,
                              height_m=FRAME_M, check_steps=False)
        assert est.valid
        # 100 nm 帧上 0.01° 对应 17 pm 的高度差,已在噪声量级以下
        assert est.slope_mag_deg < 0.01, (
            f"seed={seed} 在纯噪声上测出 {est.slope_mag_deg:.4f}°")


def test_noise_floor_recovers_the_injected_sigma():
    """噪声底估计要能还原注入的噪声幅度(否则 RANSAC 阈值就是错的)。"""
    for sigma in (5e-12, NOISE_M, 100e-12):
        got = T.noise_floor(_noise(sigma=sigma, seed=7))
        assert 0.7 * sigma < got < 1.4 * sigma, (
            f"注入 σ={sigma:.3g},估出 {got:.3g}")


def test_noise_floor_is_immune_to_a_step():
    """台阶在行内差分里是巨大的离群值 —— MAD 必须不为所动。"""
    flat = _noise(sigma=NOISE_M, seed=3)
    stepped = flat.copy()
    stepped[:, 128:] += STEP_M          # 一道垂直台阶(跨快扫轴)
    assert T.noise_floor(stepped) < 2.0 * T.noise_floor(flat)


# ══════════════════════════════════════════════════════════════════════════
#  二、倾斜回收精度
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("angle", [0.05, 0.2, 0.5, 1.0, 2.0])
def test_recovers_a_synthetic_tilt_along_the_fast_axis(angle):
    est = T.estimate_tilt(_tilted(angle, axis="x"), width_m=FRAME_M,
                          height_m=FRAME_M, check_steps=False)
    assert est.valid
    assert est.tilt_fast_deg == pytest.approx(angle, rel=0.05, abs=0.01)
    assert est.slope_mag_deg == pytest.approx(angle, rel=0.05, abs=0.01)


@pytest.mark.parametrize("angle", [0.2, 1.0])
def test_recovers_a_synthetic_tilt_along_the_slow_axis(angle):
    est = T.estimate_tilt(_tilted(angle, axis="y"), width_m=FRAME_M,
                          height_m=FRAME_M, check_steps=False)
    assert est.valid
    assert est.tilt_slow_deg == pytest.approx(angle, rel=0.05, abs=0.01)


def test_pixel_size_conversion_is_actually_applied():
    """系数是 z/像素,不除以像素物理尺寸就是个没有物理意义的数字。

    同一个高度图配不同的物理帧宽,必须给出不同的角度 —— 这一条钉的正是既有
    ransac_plane_subtract 停下来的地方。
    """
    img = _tilted(1.0, frame_m=FRAME_M, axis="x")
    a = T.estimate_tilt(img, width_m=FRAME_M, height_m=FRAME_M,
                        check_steps=False)
    b = T.estimate_tilt(img, width_m=FRAME_M * 2, height_m=FRAME_M * 2,
                        check_steps=False)
    # 帧宽翻倍 = 同样的高度差摊在两倍距离上 = 斜率减半
    assert math.tan(math.radians(b.tilt_fast_deg)) == pytest.approx(
        math.tan(math.radians(a.tilt_fast_deg)) / 2, rel=0.05)


def test_z_span_is_the_trigger_quantity():
    """统一判据是「这一帧吃掉多少 Z 量程」= L·tan(θ)。

    同样 0.3°,1 µm 帧吃 5.2 nm,10 nm 帧只吃 52 pm —— 粗扫敏感、精扫宽容
    自动成立。
    """
    big = T.z_span_for_frame(0.3, math.hypot(1e-6, 1e-6))
    small = T.z_span_for_frame(0.3, math.hypot(1e-8, 1e-8))
    assert big == pytest.approx(7.4e-9, rel=0.05)
    assert small == pytest.approx(7.4e-11, rel=0.05)
    assert big / small == pytest.approx(100.0, rel=0.01)


def test_estimate_reports_z_span_consistent_with_the_helper():
    est = T.estimate_tilt(_tilted(0.5, axis="x"), width_m=FRAME_M,
                          height_m=FRAME_M, check_steps=False)
    expect = T.z_span_for_frame(est.slope_mag_deg, math.hypot(FRAME_M, FRAME_M))
    assert est.z_span_m == pytest.approx(expect, rel=1e-6)


# ══════════════════════════════════════════════════════════════════════════
#  三、慢扫轴不可信(用户的领域修正)
# ══════════════════════════════════════════════════════════════════════════

def test_frame_based_estimate_always_flags_the_slow_axis_as_untrusted():
    """一帧 512 线 × 2 s/线要扫 34 分钟:图像顶部与底部相隔半小时,这段时间的
    热漂移会原样表现为慢轴上的视在倾斜,与真实倾斜无法区分。

    帧法必须**如实标注**这一点,不能假装两个方向一样可信。
    """
    est = T.estimate_tilt(_tilted(0.5), width_m=FRAME_M, height_m=FRAME_M,
                          check_steps=False)
    assert est.valid
    assert est.slow_axis_trusted is False


def test_thermal_drift_along_the_slow_axis_masquerades_as_tilt():
    """把漂移显式地演一遍:一个真正水平的表面 + 沿慢轴的线性 z 漂移,帧法会把
    它读成倾斜。这正是慢轴标志存在的理由。"""
    n = N_PX
    drift_total = 300e-12          # 半小时漂 300 pm,真机上很温和
    gy, _gx = np.mgrid[:n, :n].astype(np.float64)
    flat_but_drifting = gy / n * drift_total + _noise(seed=11)
    est = T.estimate_tilt(flat_but_drifting, width_m=FRAME_M, height_m=FRAME_M,
                          check_steps=False)
    assert est.valid
    # 快轴老实说没有倾斜……
    assert abs(est.tilt_fast_deg) < 0.01
    # ……慢轴却报出一个纯属漂移的"倾斜"
    assert abs(est.tilt_slow_deg) > 0.1
    assert est.slow_axis_trusted is False


# ══════════════════════════════════════════════════════════════════════════
#  四、台阶否决:两个判据的盲区互补
# ══════════════════════════════════════════════════════════════════════════

def test_single_step_is_caught_by_dominance():
    img = _noise(seed=4)
    img[:, 128:] += STEP_M
    verdict = T.assess_steps(img, use_segmentation=False)
    assert verdict.step_dominated
    assert verdict.ratio_multiscale > T.STRUCTURE_RATIO_THRESHOLD


def test_dense_steps_are_caught_only_by_the_multiscale_sweep():
    """单一 32 px 分块在密集台阶下塌回 1.0,与纯平表面无法区分 —— 而且是往
    「看起来很干净」的方向失效。多尺度扫描正是为这个盲区加的。"""
    dense = _stepped(32)             # 台面宽 8 px,远小于 32
    flat = T.detrend_quadratic(dense)
    single = T.structure_dominance(flat, tile=32)
    multi, by_tile = T.step_dominance_multiscale(flat)

    assert single < T.STRUCTURE_RATIO_THRESHOLD, (
        f"前提不成立:单尺度已经抓到了({single:.2f})")
    assert multi >= T.STRUCTURE_RATIO_THRESHOLD, (
        f"多尺度也没抓到密集台阶(max={multi:.2f}, by_tile={by_tile})")


@pytest.mark.parametrize("n_steps", [2, 4, 8, 16, 24, 32])
def test_steps_down_to_the_measured_limit_are_caught(n_steps):
    """所列合成台阶覆盖至 8 px 台面宽度，均应通过结构主导判据。
    """
    verdict = T.assess_steps(_stepped(n_steps), use_segmentation=False)
    assert verdict.step_dominated, (
        f"{n_steps} 个台阶(台面 {N_PX / n_steps:.1f}px)没被判为台阶主导 "
        f"(ratio={verdict.ratio_multiscale:.2f}, {verdict.ratio_by_tile})")


def test_terraces_narrower_than_the_hard_limit_are_honestly_invisible():
    """台面窄于约 6 px 时本判据与纯噪声无法区分 —— 这是物理极限,不是 bug。

    分块必须装得下若干像素才估得出局部 σ,而那个尺寸已经跨过台阶了。这个区间
    由分割器那一路负责,也正是两个判据必须取「或」的原因之一。
    把它钉成用例,是为了让将来有人"顺手把阈值调低点让它也能抓到"时立刻看到
    代价:阈值降到 1.06 附近,纯噪声就会开始报台阶。
    """
    ratio, _ = T.step_dominance_multiscale(T.detrend_quadratic(_stepped(64)))
    assert ratio < T.STRUCTURE_RATIO_THRESHOLD
    assert ratio < 1.15, "台面 4px 的比值居然离开了噪声底,标定需要重做"


def test_piezo_bow_cannot_be_separated_from_dense_steps_without_quadratic_detrend():
    """用独立合成曲率构造反例：一阶去趋势不能区分曲面与密集台阶。
    二阶去趋势须消除曲率误报，同时保留台阶检测，原有分离断言不变。
    """
    n = N_PX
    gy, gx = np.mgrid[:n, :n].astype(np.float64)
    bow = ((gx - n / 2) ** 2 + (gy - n / 2) ** 2) / (n / 2) ** 2 * 80e-12

    worst_bow_1st = max(
        T.step_dominance_multiscale(T.plane_subtract(bow + _noise(seed=s)))[0]
        for s in range(10))
    dense_1st = T.step_dominance_multiscale(
        T.plane_subtract(_stepped(32)))[0]
    assert worst_bow_1st > dense_1st, (
        "前提变了:一阶下弯曲不再高于密集台阶,这条用例的理由需要重写")

    # 二阶去趋势后:弯曲被压回噪声底,台阶信号基本不动
    worst_bow_2nd = max(
        T.step_dominance_multiscale(T.detrend_quadratic(bow + _noise(seed=s)))[0]
        for s in range(10))
    dense_2nd = T.step_dominance_multiscale(T.detrend_quadratic(_stepped(32)))[0]
    assert worst_bow_2nd < 1.15
    assert dense_2nd > T.STRUCTURE_RATIO_THRESHOLD


def test_strong_curvature_is_still_not_mistaken_for_steps():
    """4 倍强曲率也必须被二次面吃掉。"""
    n = N_PX
    gy, gx = np.mgrid[:n, :n].astype(np.float64)
    bow = ((gx - n / 2) ** 2 + (gy - n / 2) ** 2) / (n / 2) ** 2 * 200e-12
    verdict = T.assess_steps(bow + _noise(seed=8), use_segmentation=False)
    assert not verdict.step_dominated


def test_steps_parallel_to_the_fast_axis_are_still_caught():
    """完全平行于快扫轴的台阶对分割器不可见(逐行中位数差分把纯 y 向阶梯整个
    吸收)。主导比不做行对齐,正好补上这个盲区。"""
    horizontal = _stepped(8, angle_deg=0.0)
    verdict = T.assess_steps(horizontal, use_segmentation=False)
    assert verdict.step_dominated
    assert verdict.triggered_by == "dominance"


def test_gentle_curvature_is_not_mistaken_for_steps():
    """缓曲率(压电非线性)不是台阶 —— 判成台阶会让调平在本可以调的地方跳过。"""
    n = N_PX
    gy, gx = np.mgrid[:n, :n].astype(np.float64)
    bow = ((gx - n / 2) ** 2 + (gy - n / 2) ** 2) / (n / 2) ** 2 * 50e-12
    verdict = T.assess_steps(bow + _noise(seed=5), use_segmentation=False)
    assert not verdict.step_dominated


def test_a_pure_tilt_is_not_mistaken_for_steps():
    """倾斜本身不能被判成台阶 —— 否则「该调平」的场景永远进不去。"""
    verdict = T.assess_steps(_tilted(1.0), use_segmentation=False)
    assert not verdict.step_dominated


def test_step_dominated_frame_refuses_to_report_a_tilt():
    """台阶主导时应拒绝给出倾斜角，避免让结构高度差驱动硬件调平。"""
    est = T.estimate_tilt(_stepped(4), width_m=FRAME_M, height_m=FRAME_M)
    assert not est.valid
    assert est.invalid_reason == "step_dense"
    assert est.step is not None and est.step.step_dominated


def test_segmentation_failure_degrades_to_the_other_criterion(monkeypatch):
    """分割器坏掉不该让调平整个失效 —— 判据是「或」,少一路只会更宽松。"""
    monkeypatch.setattr(
        T, "_segmentation_step_signal",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    img = _noise(seed=6)
    img[:, 128:] += STEP_M
    with pytest.raises(RuntimeError):
        T._segmentation_step_signal(img, None)      # 前提:桩确实会抛
    verdict = T.assess_steps(img, use_segmentation=False)
    assert verdict.step_dominated


# ══════════════════════════════════════════════════════════════════════════
#  五、无效输入:全枚举、不抛
# ══════════════════════════════════════════════════════════════════════════

def test_frame_too_small_is_reported_not_guessed():
    est = T.estimate_tilt(_noise(n=32), width_m=1e-8, height_m=1e-8)
    assert not est.valid and est.invalid_reason == "frame_too_small"


def test_too_many_nan_is_reported():
    """实时扫描中未采集的行是 NaN —— 扫到一半的帧不能当完整帧拟合。"""
    img = _tilted(1.0)
    img[100:, :] = np.nan
    est = T.estimate_tilt(img, width_m=FRAME_M, height_m=FRAME_M)
    assert not est.valid and est.invalid_reason == "too_many_nan"


def test_a_few_nan_rows_are_tolerated():
    img = _tilted(1.0)
    img[250:, :] = np.nan          # < 20%
    est = T.estimate_tilt(img, width_m=FRAME_M, height_m=FRAME_M,
                          check_steps=False)
    assert est.valid


def test_missing_geometry_is_reported():
    est = T.estimate_tilt(_noise(), width_m=0.0, height_m=FRAME_M)
    assert not est.valid and est.invalid_reason == "geometry_missing"


def test_non_2d_input_is_reported():
    est = T.estimate_tilt(np.zeros(10), width_m=FRAME_M, height_m=FRAME_M)
    assert not est.valid and est.invalid_reason == "frame_not_2d"


def test_constant_frame_does_not_crash():
    """常数面的噪声底是 0 —— RANSAC 的内点判据不能因此一个点都找不到。"""
    est = T.estimate_tilt(np.zeros((N_PX, N_PX)), width_m=FRAME_M,
                          height_m=FRAME_M, check_steps=False)
    assert est.valid
    assert est.slope_mag_deg == pytest.approx(0.0, abs=1e-9)


def test_invalid_estimate_still_serialises():
    est = T.estimate_tilt(np.zeros(10), width_m=FRAME_M, height_m=FRAME_M)
    d = est.as_dict()
    assert d["valid"] is False and d["invalid_reason"] == "frame_not_2d"


# ══════════════════════════════════════════════════════════════════════════
#  六、旋转
# ══════════════════════════════════════════════════════════════════════════

def test_zero_angle_does_not_apply_a_rotation():
    est = T.estimate_tilt(_tilted(1.0), width_m=FRAME_M, height_m=FRAME_M,
                          scan_angle_deg=0.0, check_steps=False)
    assert est.rotation_applied is False
    assert est.tilt_x_deg == pytest.approx(est.tilt_fast_deg)


def test_ninety_degree_rotation_swaps_the_axes():
    est = T.estimate_tilt(_tilted(1.0, axis="x"), width_m=FRAME_M,
                          height_m=FRAME_M, scan_angle_deg=90.0,
                          check_steps=False)
    assert est.rotation_applied is True
    assert abs(est.tilt_x_deg) < 0.05
    assert est.tilt_y_deg == pytest.approx(1.0, rel=0.05, abs=0.02)


def test_rotation_preserves_the_slope_magnitude():
    """旋转只换坐标系,不改变倾斜的大小。"""
    img = _tilted(1.0, axis="x")
    mags = [
        T.estimate_tilt(img, width_m=FRAME_M, height_m=FRAME_M,
                        scan_angle_deg=ang, check_steps=False).slope_mag_deg
        for ang in (0.0, 30.0, 45.0, 90.0, 180.0)
    ]
    assert max(mags) - min(mags) < 1e-6
