"""扫描图自动预处理(:mod:`mast.vision.scan_prep`)。

设计文档:``docs/v2/design/scan_prep_auto_flatten.md``。

合成数据一律**物理**:高度以米为单位;噪声 2-5 pm 并且**是空间相关的**(有限针尖
半径 + 有限反馈带宽 —— 真实 STM 图不是逐像素独立的白噪声,这一点对第 3 节的判据
至关重要);台阶 240 pm(金属单原子台阶);晶格取 Au(111) 的原子行间距 0.2494 nm
与 10 pm 起伏;逐行漂移用随机游走(z 反馈慢漂的实际形状);每一帧都带真实量级的
样品倾斜(400 pm/帧),因为**扣平面这一步在真实数据上主要是在扣样品倾斜**,不带倾斜
的合成帧会让最小二乘平面转去拟合台阶本身。

**本文件最重要的一节是第 3 节「被否决的三条判据」。** 它们看起来都很合理,都在真实
数据上被算过、被推翻过,而且都是后人最容易「顺手加回来」的东西。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage

from mast.vision import scan_prep as SP
from mast.vision.scan_prep_thresholds import (
    DEFAULT_PROFILE,
    PROFILES,
    ScanPrepThresholds,
    available_profiles,
    knob_catalog,
    resolve,
)

# ── 物理常数 ────────────────────────────────────────────────────────────────

AU_NN_NM = 0.288
ROW_SPACING_NM = AU_NN_NM * math.sqrt(3.0) / 2.0      # 0.2494 nm
PIXELS = 256
ATOMIC_NMPP = 5.0 / PIXELS                            # 0.0195 nm/px —— 满权重尺度档
MESO_NMPP = 20.0 / PIXELS                             # 0.078 nm/px
NOISE_M = 5e-12
STEP_M = 240e-12                                      # 金属单原子台阶
TILT_M = 400e-12                                      # 一帧上的样品倾斜
CORRUGATION_M = 10e-12

TH = resolve()


# ── 合成 ────────────────────────────────────────────────────────────────────

def _white(n=PIXELS, sigma=NOISE_M, seed=0):
    """**纯**高斯白噪声 —— 误报率对照组,故意不做空间平滑。"""
    return np.random.default_rng(seed).normal(0.0, sigma, (n, n))


def _noise(n=PIXELS, sigma=NOISE_M, seed=0, smooth=1.2):
    """空间相关的 z 噪声。真实图上相邻像素是相关的(针尖半径 + 反馈带宽),
    这正是中值滤波高通会失效的原因 —— 见 :func:`test_rejected_median_filter_...`。"""
    a = np.random.default_rng(seed).normal(0.0, 1.0, (n, n))
    if smooth:
        a = ndimage.gaussian_filter(a, smooth)
        a = a / (float(a.std()) + 1e-30)
    return a * sigma


def _tilted(n=PIXELS, seed=1):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    return TILT_M * (x + 0.3 * y) / n + _white(n, seed=seed)


def _bowed(n=PIXELS, curv_m=400e-12, seed=2):
    """压电弯曲 / 蠕变:一张碗形的面。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    u, v = (x - n / 2) / (n / 2), (y - n / 2) / (n / 2)
    return curv_m * (u ** 2 + 0.7 * v ** 2) + _white(n, seed=seed)


def _row_drift(n=PIXELS, step_pm=25.0, seed=3):
    """z 反馈慢漂:逐行随机游走偏置。这是 ``line_gain`` 该点着的那种帧。"""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, step_pm * 1e-12, n))[:, None] + _white(n, seed=seed + 100)


#: 台阶边在图上被抹开几个像素 —— 有限的针尖半径。**像素级锐利的台阶是非物理的**,
#: 而且会在 FFT 里产生高频旁瓣，因此合成台阶使用平滑边缘以控制这项混杂。
_EDGE_PX = 2.0


def _terrace_band(n=PIXELS, step_m=STEP_M, seed=4, drift_pm=0.0, smooth=1.2,
                  with_truth=False):
    """**真台阶**:一条竖着贯穿画面的高 terrace 带 —— **每一行**都跨过它两次。

    为什么用「带」而不是一条斜边:一条把画面切成两半的斜边会被最小二乘平面吸收掉
    (平面转过去拟合台阶本身),高度直方图随之糊成一个峰。真实帧上的台阶总是叠在一个
    远大于台阶高度的样品倾斜上,平面主要在扣那个倾斜 —— 这里用「带 + 沿慢轴的倾斜」
    把这个真实情形复现出来。

    ``with_truth`` 时返回 ``(带噪声的图, 无噪声无倾斜的台阶场)``,后者是「理想平场
    应该还原出什么」的真值。
    """
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    u = x - 0.15 * y                                   # 台阶边略微倾斜,不是纯竖线
    mask = ndimage.gaussian_filter(
        ((u > 0.24 * n) & (u < 0.70 * n)).astype(np.float64), _EDGE_PX)
    truth = step_m * mask
    h = truth + TILT_M * (y / n)                       # 倾斜沿慢轴,与台阶方向正交
    if drift_pm:
        h = h + np.cumsum(np.random.default_rng(seed + 7)
                          .normal(0.0, drift_pm * 1e-12, n))[:, None]
    img = h + _noise(n, seed=seed, smooth=smooth)
    return (img, truth) if with_truth else img


def _row_band(n=PIXELS, offset_m=STEP_M, seed=5, smooth=1.2):
    """**伪台阶**:一段行整体抬高 —— 针尖突变 / z 跳变的形状,严格按行切。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    mask = ndimage.gaussian_filter(
        ((y > 0.35 * n) & (y < 0.72 * n)).astype(np.float64), _EDGE_PX)
    h = offset_m * mask + TILT_M * (x / n)             # 倾斜沿快轴,与分层方向正交
    return h + _noise(n, seed=seed, smooth=smooth)


def _hex_lattice(n=PIXELS, nmpp=ATOMIC_NMPP, period_nm=ROW_SPACING_NM,
                 amp=CORRUGATION_M, seed=6, tilt_deg=25.0):
    """六角晶格(三组波矢相隔 60°)。整体转 25°,故意避开两条扫描轴。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xs, ys = x * nmpp, y * nmpp
    k = 2.0 * math.pi / float(period_nm)
    h = np.zeros((n, n))
    for deg in (0.0, 60.0, 120.0):
        th = math.radians(deg + tilt_deg)
        h += np.cos(k * (xs * math.cos(th) + ys * math.sin(th)))
    return h / 3.0 * amp + _noise(n, sigma=2e-12, seed=seed, smooth=0.6)


def _scan_line_stripes(n=PIXELS, nmpp=ATOMIC_NMPP, period_nm=0.4,
                       amp=20e-12, seed=7):
    """**沿扫描轴**的条纹伪影:扫描线噪声 / 逐行平场留下的脊,不是晶格。"""
    y, _x = np.mgrid[0:n, 0:n].astype(np.float64)
    k = 2.0 * math.pi / (float(period_nm) / float(nmpp))
    return amp * np.cos(k * y) + _noise(n, sigma=2e-12, seed=seed, smooth=0.6)


def _measure(img, **kw):
    kw.setdefault("nm_per_px", MESO_NMPP)
    kw.setdefault("thresholds", TH)
    return SP.measure_frame(img, **kw)


def _amplitude_pm(a):
    v = a[np.isfinite(a)]
    return float(np.percentile(v, 99.5) - np.percentile(v, 0.5)) * 1e12


# ═══════════════════════════════════════════════════════════════════════
# 1. 纯白噪声上的误报 —— 判据必须先在这里干净
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", range(10))
def test_line_gain_and_bow_gain_stay_at_one_on_pure_white_noise(seed):
    """纯白噪声上两个「增益」判据都不许触发。

    这不是运气:一行 256 个点上做一阶拟合只吃掉 2 个自由度,方差比 = 1/(1−2/256),
    即 ``line_gain ≈ 1.004``;二阶曲面多 3 个自由度,``bow_gain ≈ 1.00002``。判据的
    阈值 1.30 / 1.15 离它们极远 —— **过拟合点不着这两条判据**。哪天有人想把阈值
    调到 1.05,这条测试会红,而那正是该红的时候。
    """
    m = _measure(_white(seed=seed))
    assert m.line_gain < 1.02, f"白噪声上 line_gain 竟然是 {m.line_gain}"
    assert m.bow_gain < 1.02, f"白噪声上 bow_gain 竟然是 {m.bow_gain}"
    plan = SP.plan_for(m, TH)
    assert plan.method == "plane"
    assert not plan.step_like
    assert not plan.fine_structure


def test_pure_noise_is_flagged_as_a_noisy_frame():
    m = _measure(_white(seed=42))
    assert abs(m.rowcorr_median) < 0.1
    assert "噪声帧" in " ".join(SP.plan_for(m, TH).notes)


# ═══════════════════════════════════════════════════════════════════════
# 2. 平场方式的选择
# ═══════════════════════════════════════════════════════════════════════

def test_tilted_flat_surface_only_needs_a_plane():
    assert SP.plan_for(_measure(_tilted()), TH).method == "plane"


def test_bowed_surface_switches_to_a_second_order_baseline():
    m = _measure(_bowed())
    assert m.bow_gain > TH.bow_gain
    assert SP.plan_for(m, TH).method == "poly2"


def test_row_drift_switches_to_line_flattening():
    m = _measure(_row_drift())
    assert m.line_gain > TH.line_gain
    assert SP.plan_for(m, TH).method == "line"


def test_a_real_step_with_row_drift_gets_the_protective_masked_line():
    m = _measure(_terrace_band(drift_pm=60.0))
    plan = SP.plan_for(m, TH)
    assert plan.step_like, "贯穿画面的 terrace 带必须判为真台阶"
    assert plan.method == "masked_line"
    assert plan.clip == TH.clip_step, "有台阶时色阶要放宽,保住台阶高度"


@pytest.mark.parametrize("drift_pm", [0.0, 25.0, 60.0])
def test_masked_line_reconstructs_the_true_surface_better_than_the_alternatives(drift_pm):
    """masked_line 存在的理由,拿**真值**说一遍。

    合成帧有已知的无噪声台阶场,理想的平场应该把它还原出来(差一个常数)。比较三种
    处理相对真值的残差 RMS —— 这比看「还剩多少台阶高度」硬,因为它同时惩罚
    「台阶被削掉」和「每行被带出一条假斜率」。
    """
    img, truth = _terrace_band(drift_pm=drift_pm, with_truth=True)
    m = _measure(img)
    ref = truth - truth.mean()

    def _residual_pm(a):
        d = a - ref
        return float(np.sqrt(np.mean((d - d.mean()) ** 2))) * 1e12

    plane = _residual_pm(SP.poly_subtract(img, 1))
    naive = _residual_pm(SP.line_subtract(img, 1))
    protected = _residual_pm(SP.apply_flatten(img, "masked_line", m))
    assert protected < 0.5 * naive, (
        f"masked_line 残差 {protected:.1f} pm 没有明显好过朴素逐行平场 "
        f"{naive:.1f} pm")
    assert protected < plane, (
        f"masked_line 残差 {protected:.1f} pm 竟然不如只扣平面 {plane:.1f} pm")
    assert protected < 0.1 * STEP_M * 1e12, (
        f"masked_line 的重建残差 {protected:.1f} pm 大到台阶高度的 10% 以上")


def test_explicit_override_skips_the_automatic_choice():
    m = _measure(_row_drift())
    plan = SP.plan_for(m, TH, override="plane")
    assert plan.method == "plane"
    assert "由调用方指定" in " ".join(plan.why)
    with pytest.raises(ValueError):
        SP.plan_for(m, TH, override="nonsense")


# ═══════════════════════════════════════════════════════════════════════
# 3. 被否决的三条判据 —— 每一条都留一个钉子
# ═══════════════════════════════════════════════════════════════════════

def test_rejected_peak_count_alone_calls_a_row_split_a_step():
    """只数直方图峰会把合成行向分层当成台阶；行纯度必须把两者分开。
    测试保留相同的台阶与分层夹具，并对各自输出和保护性处理作独立断言。
    """
    step = _measure(_terrace_band())
    split = _measure(_row_band())

    # 前提:只数峰的话,两者都会被判成「有台阶」——这正是被否决的判据。
    for m, what in ((step, "真台阶"), (split, "行向分层")):
        peaks_only = (m.n_peaks >= TH.step_peaks and m.sep_over_rough > TH.step_sep)
        assert peaks_only, (
            f"{what} 帧本该让「只数峰」这条判据触发(否则这条测试没在测东西):"
            f"peaks={m.n_peaks} sep/rough={m.sep_over_rough:.1f}")

    # 加上 row_purity 才分得开。
    assert step.row_purity < TH.step_purity < split.row_purity, (
        f"行纯度没把两者分开:真台阶 {step.row_purity:.2f},"
        f"行向分层 {split.row_purity:.2f}")
    assert SP.plan_for(step, TH).step_like is True
    assert SP.plan_for(split, TH).step_like is False
    assert "行向分层" in " ".join(SP.plan_for(split, TH).notes)


def test_rejected_median_filter_highpass_collapses_the_mad():
    """中值高通会在空间相关的合成图上产生精确零，压低 MAD 并夸大分离度。
    测试分别约束精确零比例、MAD 的下降和下游分离度偏差。
    """
    base = SP.poly_subtract(_terrace_band(smooth=1.8), 1)
    mean_hp = base - ndimage.uniform_filter(base, (1, 9))
    med_hp = base - ndimage.median_filter(base, (1, 9))

    zero_frac = float(np.mean(med_hp == 0.0))
    assert zero_frac > 0.25, (
        f"中值滤波高通只有 {zero_frac:.1%} 的精确零 —— 这条测试的前提没成立,"
        "先检查合成噪声是不是还带着空间相关性")
    assert float(np.mean(mean_hp == 0.0)) < 0.01, "均值滤波高通不该产生精确零"

    mad_mean, mad_med = SP._mad(mean_hp), SP._mad(med_hp)
    assert mad_med < 0.55 * mad_mean, (
        f"中值滤波高通的 MAD({mad_med:.3g}) 没有相对均值滤波({mad_mean:.3g}) 塌下去")

    # 真正的危害:下游那个判据被顶上去。
    sep = float(SP.height_levels(base)["separation"])
    assert sep / mad_med > 1.7 * (sep / mad_mean), (
        "中值滤波高通没有把 sep_over_rough 顶高 —— 这条测试没在测危害本身")


def test_rejected_row_median_jump_and_hf_fraction_do_not_separate():
    """在合成台阶与噪声帧上，两项候选行统计量的取值区间重叠。
    测试要求当前采用的 line_gain 与行相关仍能区分这两类输入。
    """
    def _jump_ratio(img):
        pl = SP.poly_subtract(img, 1)
        med = np.nanmedian(pl, axis=1)
        d = np.abs(np.diff(med[np.isfinite(med)]))
        return float(np.percentile(d, 99) / (np.median(d) + 1e-30))

    def _hf_fraction(img):
        pl = SP.poly_subtract(img, 1)
        med = np.nanmedian(pl, axis=1)
        med = med[np.isfinite(med)]
        sp = np.abs(np.fft.rfft(med - med.mean())) ** 2
        return float(sp[len(sp) // 2:].sum() / (sp.sum() + 1e-30))

    step = [_terrace_band(seed=s) for s in range(4)]
    noise = [_noise(seed=s) for s in range(4)]

    def _overlaps(a, b):
        return max(min(a), min(b)) <= min(max(a), max(b))

    for fn, label in ((_jump_ratio, "行中位数跳变比"),
                      (_hf_fraction, "行中位数谱高频占比")):
        a = [fn(i) for i in step]
        b = [fn(i) for i in noise]
        assert _overlaps(a, b), (
            f"{label} 居然把「有台阶」{sorted(round(v, 3) for v in a)} 与「纯噪声」"
            f"{sorted(round(v, 3) for v in b)} 分开了 —— 如果这在更多数据上也成立,"
            "才值得重新考虑这条被否决的判据")

    # 留下来的两个量分得开。
    assert _measure(_row_drift()).line_gain > TH.line_gain
    assert _measure(_tilted()).line_gain < TH.line_gain
    assert _measure(_white()).rowcorr_median < TH.rowcorr_poor
    assert _measure(_hex_lattice(), nm_per_px=ATOMIC_NMPP).rowcorr_median > TH.rowcorr_poor


# ═══════════════════════════════════════════════════════════════════════
# 4. 精细周期结构:只决定色阶,永远不产出「有晶格」的结论
# ═══════════════════════════════════════════════════════════════════════

def test_axis_guard_rejects_stripes_running_along_a_scan_axis():
    """扫描轴条纹不得触发精细结构色阶，合成斜晶格必须触发。
    保留对两种合成输入的独立 SNR 断言，不因默认 profile 未标定而跳过。
    """
    stripes = SP.fine_periodic_peak(_scan_line_stripes(), ATOMIC_NMPP, TH)
    lattice = SP.fine_periodic_peak(_hex_lattice(), ATOMIC_NMPP, TH)
    assert stripes["snr"] < TH.fine_periodic_snr, (
        f"扫描轴条纹拿到了 SNR {stripes['snr']:.1f} —— 轴向死区失效了")
    assert lattice["snr"] > TH.fine_periodic_snr, (
        f"真晶格只拿到 SNR {lattice['snr']:.1f}")


def test_a_real_lattice_tightens_the_colour_scale_and_atomic_phase_confirms_it():
    m = _measure(_hex_lattice(), nm_per_px=ATOMIC_NMPP)
    plan = SP.plan_for(m, TH)
    assert plan.fine_structure
    assert plan.clip == TH.clip_lattice
    assert m.atomic is not None and m.atomic["passed"], f"atomic_phase 没通过:{m.atomic}"
    assert "原子相判据通过" in " ".join(plan.notes)


def test_scale_gate_says_cannot_judge_not_no_lattice():
    """采样尺度不足必须返回无法判断，不能把分辨率不足当成没有原子相。"""
    coarse = 0.09                                # 远在 0.05 nm/px 尺度门之外
    m = _measure(_hex_lattice(nmpp=coarse), nm_per_px=coarse)
    assert m.atomic is not None and not m.atomic["passed"]
    assert "scale_gate" in m.atomic["reasons"]
    notes = " ".join(SP.plan_for(m, TH).notes)
    assert "判不了" in notes
    assert "没有原子相" not in notes


def test_the_module_never_claims_a_lattice_on_snr_alone():
    """SNR 高但 atomic_phase 没通过时,措辞必须让 atomic_phase 说了算。

    这是整个收编里最关键的一条边界:峰强度分不开针尖抖动造出的准周期条纹
    (``mast.vision.atomic_phase`` 实测 30 个种子 30 个都能给出 SNR 15-38 的合格谱峰)。
    """
    import dataclasses

    m = _measure(_hex_lattice(), nm_per_px=ATOMIC_NMPP)
    assert m.fine_periodic_snr > TH.fine_periodic_snr, "前提:这一帧的 SNR 要高"
    forced = dataclasses.replace(m, atomic={
        "passed": False, "scale": "full", "reasons": ["not_a_lattice"],
        "warnings": [], "period_fast_axis_nm": None, "period_radial_nm": None,
        "angular_concentration": 2.0, "fft_sharpness": 9.0, "snr": 20.0})
    notes = " ".join(SP.plan_for(forced, TH).notes)
    assert "只用于色阶" in notes
    assert "atomic_phase 的结论为准" in notes
    assert "原子相判据通过" not in notes


def test_metrics_carry_no_field_that_reads_as_a_lattice_claim():
    """字段名也是接口:``FrameMetrics`` 里不许出现听起来像「有晶格」的量。"""
    import dataclasses

    names = {f.name for f in dataclasses.fields(SP.FrameMetrics)}
    for bad in ("atom_snr", "has_lattice", "lattice_period_nm", "atom_period_nm"):
        assert bad not in names, f"{bad} 会被读成一个结论 —— 用 fine_periodic_* 命名"


# ═══════════════════════════════════════════════════════════════════════
# 5. 未扫完的帧:NaN 不许崩,更不许制造出一个假的针尖突变
# ═══════════════════════════════════════════════════════════════════════

def test_unfinished_scan_does_not_crash_and_is_annotated():
    img = _row_drift()
    cut = int(PIXELS * 0.6)
    img[cut:, :] = np.nan
    m = _measure(img)
    assert 0.35 < m.nan_frac < 0.45
    assert m.dead_rows == PIXELS - cut
    assert m.analysis_rows == (0, cut)
    assert "扫描未完成" in " ".join(SP.plan_for(m, TH).notes)


def test_filling_dead_rows_manufactures_a_tip_change_this_is_the_trap():
    """未扫描区域的常数填充会制造边界突变；合成缺失行夹具覆盖该陷阱。"""
    from mast.vision.tip_change import detect_tip_change

    cut = int(PIXELS * 0.45)
    img = _terrace_band(drift_pm=10.0, seed=21)
    img[cut:, :] = np.nan
    naive = SP.line_subtract(img, 1)
    naive = np.nan_to_num(naive, nan=float(np.nanmedian(naive)))

    trap = detect_tip_change(naive, nm_per_px=MESO_NMPP)
    assert trap.changed and trap.change_row is not None, (
        "这条测试的前提没成立:填充边界本该被误报成突变")
    assert abs(trap.change_row - cut) < 15, (
        f"误报的位置({trap.change_row})本该就在填充边界({cut})上")


@pytest.mark.parametrize("cut_frac", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
def test_dead_rows_do_not_manufacture_a_tip_change(cut_frac):
    """本模块的做法:转发之前先把分析限制到真正采到的那一段行。"""
    cut = int(PIXELS * cut_frac)
    img = _terrace_band(drift_pm=10.0, seed=21)
    img[cut:, :] = np.nan

    m = _measure(img)
    assert m.analysis_rows == (0, cut)
    assert m.tip_change is not None and not m.tip_change["changed"], (
        f"限制行段之后仍然误报:{m.tip_change}")


def test_all_nan_frame_is_survivable():
    m = _measure(np.full((64, 64), np.nan))
    assert m.nan_frac == pytest.approx(1.0)
    SP.plan_for(m, TH)                      # 不许抛


def test_acquired_row_span_picks_the_longest_run():
    img = np.full((100, 20), np.nan)
    img[5:15] = 0.0
    img[40:80] = 0.0
    assert SP.acquired_row_span(img) == (40, 80)


# ═══════════════════════════════════════════════════════════════════════
# 6. 批次一致性
# ═══════════════════════════════════════════════════════════════════════

def _plan(method, *, step_like=False):
    return SP.FlattenPlan(method=method, step_like=step_like)


def test_batch_majority_vote_harmonises_a_group():
    items = [("g", _plan("line")), ("g", _plan("line")), ("g", _plan("line")),
             ("g", _plan("plane"))]
    out = SP.harmonise_batch(items, TH)
    assert [p.method for p in out] == ["line"] * 4
    assert "批次一致性" in " ".join(out[3].why)


def test_batch_vote_exempts_frames_with_a_real_step():
    """有真台阶的帧要的是保护性处理,被多数票剥掉就等于把台阶平掉。"""
    items = [("g", _plan("line")), ("g", _plan("line")), ("g", _plan("line")),
             ("g", _plan("masked_line", step_like=True))]
    out = SP.harmonise_batch(items, TH)
    assert out[3].method == "masked_line"
    assert "批次一致性" not in " ".join(out[3].why)


def test_batch_vote_needs_enough_frames():
    items = [("g", _plan("line")), ("g", _plan("plane"))]
    assert [p.method for p in SP.harmonise_batch(items, TH)] == ["line", "plane"]


def test_batch_groups_are_independent():
    items = [("a", _plan("line")), ("a", _plan("line")), ("a", _plan("line")),
             ("b", _plan("plane")), ("b", _plan("plane")), ("b", _plan("plane")),
             ("b", _plan("line"))]
    out = SP.harmonise_batch(items, TH)
    assert [p.method for p in out[:3]] == ["line"] * 3
    assert [p.method for p in out[3:]] == ["plane"] * 4


# ═══════════════════════════════════════════════════════════════════════
# 7. 阈值 profile:按样品体系切换、可重标定
# ═══════════════════════════════════════════════════════════════════════

def test_only_one_builtin_profile_and_it_states_where_it_came_from():
    """A runnable example must not masquerade as a commissioned profile."""
    assert DEFAULT_PROFILE == "generic-uncommissioned"
    assert list(PROFILES) == [DEFAULT_PROFILE]
    th = PROFILES[DEFAULT_PROFILE]
    assert th.name == DEFAULT_PROFILE
    assert "未标定" in th.provenance and "uncommissioned" in th.provenance
    assert "合成示例" in th.provenance
    assert "未经任何样品或仪器验证" in th.provenance
    assert "scan_prep_commission" in th.provenance
    assert th.corrugation_high_pm is None and th.corrugation_ref_scan_nm is None


def test_provenance_travels_into_every_plan():
    """阈值的出身必须跟着结论走 —— 读的人不可能忘记这套数是哪来的。"""
    plan = SP.plan_for(_measure(_white()), resolve())
    assert plan.profile == DEFAULT_PROFILE
    assert "未标定" in plan.provenance
    assert plan.provenance == PROFILES[DEFAULT_PROFILE].provenance
    assert plan.to_dict()["threshold_provenance"] == plan.provenance


def test_unknown_profile_falls_back_loudly_not_silently():
    th = resolve("no-such-sample")
    assert th.name == DEFAULT_PROFILE
    assert "no-such-sample" in th.provenance and "回落" in th.provenance


def test_overrides_are_clamped_to_the_field_bounds():
    """``line_gain`` 按构造 ≥1,设成 0.5 等于「永远触发」——必须被夹住。"""
    th = resolve(overrides={"line_gain": 0.5, "step_purity": 99.0, "nonsense": 1.0})
    assert th.line_gain == 1.0
    assert th.step_purity == 1.0
    assert not hasattr(th, "nonsense")


def test_thresholds_can_be_swapped_per_sample_system():
    """换一套阈值,同一帧的结论就该跟着换 —— 这就是「可按样品体系切换」的含义。"""
    m = _measure(_terrace_band())
    assert SP.plan_for(m, TH).step_like is True
    strict = ScanPrepThresholds.from_mapping({"step_sep": 200.0}, base=TH)
    assert SP.plan_for(m, strict).step_like is False


def test_knob_catalog_marks_what_can_and_cannot_be_calibrated_from_a_distribution():
    cat = {k["key"]: k for k in knob_catalog()}
    assert cat["step_sep"]["calibratable_from_distribution"] is True
    for key in ("line_gain", "bow_gain", "step_purity"):
        assert cat[key]["calibratable_from_distribution"] is False
        assert cat[key]["max"] > cat[key]["min"] >= 0.0
        assert cat[key]["label_zh"] and cat[key]["hint_zh"]


def test_available_profiles_lists_provenance():
    assert DEFAULT_PROFILE in available_profiles()


# ═══════════════════════════════════════════════════════════════════════
# 8. 结构完整性
# ═══════════════════════════════════════════════════════════════════════

def test_metrics_dict_is_json_friendly():
    """``SkillResult.data`` 会进 ToolMessage —— 不许夹带 ndarray。"""
    import json

    d = _measure(_hex_lattice(), nm_per_px=ATOMIC_NMPP).to_dict()
    json.dumps(d)                                   # 不许抛
    assert not any(k.startswith("_") for k in d), "内部量不该外露"
    assert not any(isinstance(v, np.ndarray) for v in d.values())


def test_poly_subtract_is_nan_safe_where_the_old_helper_is_not():
    """``mast.data.processors.plane_subtract`` 遇到 NaN 会把整幅图算成 NaN。"""
    from mast.data.processors import plane_subtract

    img = _tilted()
    img[10:20, :] = np.nan
    assert np.isnan(plane_subtract(img)).all(), (
        "这条测试的前提没成立:既有 plane_subtract 本该被 NaN 传染")
    out = SP.poly_subtract(img, 1)
    assert np.isfinite(out[np.isfinite(img)]).all()
    assert abs(float(np.nanmean(out))) < 1e-13


def test_line_subtract_interpolates_coefficients_for_starved_rows():
    """可用像素太少的行不许拿垃圾拟合 —— 否则一条荒谬的斜率会被减到整行上。"""
    img = _row_drift()
    img[50, 5:] = np.nan                            # 只剩 5 个像素的一行
    out = SP.line_subtract(img, 1)
    finite = out[50][np.isfinite(out[50])]
    assert finite.size == 5
    assert float(np.abs(finite).max()) < 1e-9, "被垃圾拟合甩飞了"
