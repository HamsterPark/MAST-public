"""Spectroscopic tip probes — network-free, non-image.

The autonomous system already records I(z) approach curves and I(V) tunnelling
spectra; both are strong, physics-grounded tip-quality signals that are
*independent* of the imaging path (a useful cross-check on the vision heads).

  * I(z): a clean tip tunnels with a single exponential I ∝ exp(−2κz). The decay
    gives the apparent barrier height (≈ work function, ~4-5 eV for a clean metal
    tip); a poor log-linear fit or sudden current steps flag a blunt / unstable /
    contaminated tip.
  * I(V): a stable tip gives a smooth, spike-free, roughly antisymmetric curve;
    sudden jumps = tip switching during the sweep.
  * dI/dV Shockley onset: on a noble-metal (111) face the surface state appears
    as a STEP in dI/dV at a known energy. Seeing it at the right place is the
    operator's test for a METALLIC tip — see :func:`assess_shockley_onset`.

See docs/v2/benchmarks/vision_v25_diagnostic/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mast.vision.module import IvResult, IzResult

# κ[Å⁻¹] ≈ K_PHI · sqrt(φ[eV]) for a vacuum tunnel barrier.
_K_PHI = 0.5123

#: Boltzmann constant over elementary charge, V/K.
_KB_OVER_E = 8.617333262e-5


def assess_iz(z_nm: npt.ArrayLike, current: npt.ArrayLike) -> IzResult:
    """Assess an I(z) approach curve. ``z_nm`` in nanometres (tip-sample
    distance, any monotonic direction), ``current`` in any linear unit."""
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    I = np.abs(np.asarray(current, dtype=np.float64).ravel())
    if z.size != I.size or z.size < 5:
        return IzResult(is_clean_exponential=False, fit_r2=0.0)

    floor = max(1e-30, 1e-4 * float(np.nanmax(I)))
    ok = np.isfinite(z) & np.isfinite(I) & (I > floor)
    if ok.sum() < 5:
        return IzResult(is_clean_exponential=False, fit_r2=0.0)
    zz, ll = z[ok], np.log(I[ok])
    order = np.argsort(zz)
    zz, ll = zz[order], ll[order]

    slope, intercept = np.polyfit(zz, ll, 1)
    pred = slope * zz + intercept
    ss_res = float(np.sum((ll - pred) ** 2))
    ss_tot = float(np.sum((ll - ll.mean()) ** 2)) + 1e-12
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))

    decay_per_nm = float(abs(slope))              # = 2κ (per nm)
    kappa_per_A = decay_per_nm / 2.0 / 10.0       # nm⁻¹ → Å⁻¹
    barrier = float((kappa_per_A / _K_PHI) ** 2) if kappa_per_A > 0 else None

    # sudden steps in log-current = tip jumps during the ramp
    d = np.diff(ll)
    mad = float(np.median(np.abs(d - np.median(d)))) + 1e-9
    n_jumps = int(np.sum(np.abs(d - np.median(d)) > 8.0 * mad))

    clean = bool(r2 > 0.90 and barrier is not None and 0.5 <= barrier <= 8.0 and n_jumps == 0)
    return IzResult(
        is_clean_exponential=clean,
        barrier_ev=(barrier if barrier is not None else None),
        fit_r2=r2,
        decay_per_nm=decay_per_nm,
        n_jumps=n_jumps,
    )


def assess_iv(bias_v: npt.ArrayLike, current: npt.ArrayLike) -> IvResult:
    """Assess an I(V) tunnelling spectrum. ``bias_v`` in volts, ``current`` any
    linear unit. Detects tip switching (spikes/jumps) + curve smoothness/symmetry
    and reports a near-zero-conductance gap width if present."""
    V = np.asarray(bias_v, dtype=np.float64).ravel()
    I = np.asarray(current, dtype=np.float64).ravel()
    if V.size != I.size or V.size < 7:
        return IvResult(is_stable=False, smoothness=0.0, symmetry=0.0)
    order = np.argsort(V)
    V, I = V[order], I[order]
    Is = I / (np.max(np.abs(I)) + 1e-30)

    d1 = np.diff(Is)
    d2 = np.diff(d1)
    smoothness = float(np.clip(1.0 - np.std(d2) / (np.std(d1) + 1e-9), 0.0, 1.0))

    # spikes = sudden jumps in the current (tip switches)
    mad = float(np.median(np.abs(d1 - np.median(d1)))) + 1e-9
    n_spikes = int(np.sum(np.abs(d1 - np.median(d1)) > 8.0 * mad))

    # antisymmetry: I(V) ≈ −I(−V)  →  correlate I with −flip(I)
    flip = -Is[::-1]
    if np.std(Is) > 1e-9 and np.std(flip) > 1e-9:
        symmetry = float(np.clip(np.corrcoef(Is, flip)[0, 1], 0.0, 1.0))
    else:
        symmetry = 0.0

    # near-zero-conductance gap around V=0
    thr = 0.03 * float(np.max(np.abs(I)) + 1e-30)
    near0 = np.abs(I) < thr
    gap_ev = float(V[near0].max() - V[near0].min()) if near0.sum() >= 3 else None

    is_stable = bool(n_spikes == 0 and smoothness > 0.5)
    return IvResult(
        is_stable=is_stable,
        smoothness=smoothness,
        symmetry=symmetry,
        n_spikes=n_spikes,
        gap_ev=gap_ev,
    )


# ─────────────────────────────────────────────────────────────────────────
# Shockley surface-state onset — 「这根针尖是金属性的吗」
# ─────────────────────────────────────────────────────────────────────────
#
# 用 lock-in STS 评估金属表面的肖克利表面态 onset。
# 位置、幅度、展宽和模型比较共同决定是否有足够证据。
#
# 物理上它不是峰是**台阶**:二维表面态的态密度在 onset 之上是常数,所以 dI/dV 从
# 背景抬起一个台阶,拐点就是 onset 能量。Au(111) −0.49 V、Cu(111) −0.44 V、
# Ag(111) −0.065 V(知识库 knowledge/clean_metal.py 的 CONSTANTS 表)。
#
# ## 为什么不用现成的找峰
#
# ``agents.data_processing.tools.fit_sts_peaks`` 是 ``scipy.signal.find_peaks``,
# 找的是极大值。台阶没有极大值 —— 在一条单调抬升的曲线上,find_peaks 要么什么都
# 不报,要么报噪声。判据必须与被测物理量的形状一致。
#
# ## 三条防线(缺一条就会把噪声读成好针尖)
#
# 判据的失败模式必须是「说判不了」,不能是「说达标」。所以除了位置对不对,还要:
#
#   * **幅度** —— 台阶高度要显著高于残差噪声(k·σ_MAD),否则拟合出来的是噪声;
#   * **宽度下限** —— 台阶不可能比仪器展宽还陡。热展宽 3.5kT/e 与 lock-in 调制
#     展宽 2.5·V_rms 求方和根,得到 ``width_floor_v``;拟合被约束在它之上,
#     免得一个 tip switch 的尖峰被拟合成「无限陡的完美台阶」;
#   * **ΔBIC** —— 台阶模型 vs 纯线性模型的贝叶斯信息量之差。这一条专治两类
#     假阳性:纯白噪声(台阶模型多 3 个自由度,BIC 惩罚吃掉它蹭到的那点 RSS)与
#     线性斜坡(线性模型本来就更简约)。合成实测:300 个白噪声种子零误报。
#
# 对纯高斯白噪声验证误报是这三项判据的共同前提。


@dataclass(frozen=True)
class ShockleyOnsetResult:
    """一条 dI/dV 谱里有没有肖克利表面态台阶,以及它在不在该在的地方。

    ``passed`` 是唯一的结论;不通过时 ``reasons`` 说明差在哪一条,``warnings``
    放的是「结论仍然成立但你该知道」的事(例如展宽已经与 onset 深度同量级)。

    **``passed=False`` 不等于「针尖坏了」**:谱扫的窗口不对、lock-in 相位反了、
    表面本来就不是 (111) 面,都会走到这里。reasons 是给人看的,不是给流程当
    「再扎一次」的唯一依据。
    """

    passed: bool
    onset_v: float | None = None
    onset_err_v: float | None = None
    expected_onset_v: float = 0.0
    tol_v: float = 0.0
    step_height: float | None = None
    step_sigma_ratio: float | None = None      # 台阶高 / 残差噪声
    step_frac_of_range: float | None = None    # 台阶高 / 谱的动态范围
    width_v: float | None = None
    width_floor_v: float = 0.0                 # 仪器展宽下限(热 + 调制)
    background_slope: float | None = None
    r2: float | None = None
    delta_bic: float | None = None
    n_points_fit: int = 0
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def broadening_floor_v(temperature_k: float, lockin_mod_vrms: float) -> float:
    """一个台阶在这台机器上最陡能有多陡(伏)。

    热展宽用 3.5·kT/e(费米函数 10-90 上升宽度约 3.5 kT),lock-in 调制展宽用
    2.5·V_rms(正弦调制的等效方窗宽度约 2.83·V_rms,取略保守值)。两者独立,
    求方和根。
    """
    t = max(0.0, float(temperature_k or 0.0))
    v = max(0.0, float(lockin_mod_vrms or 0.0))
    thermal = 3.5 * _KB_OVER_E * t
    modul = 2.5 * v
    return float(math.hypot(thermal, modul))


def _logistic_step(v, a, b, h, v0, w):
    """线性背景 + 一个上升台阶。``w`` 是逻辑斯蒂宽度(≈ 10-90 宽度 / 4.4)。"""
    # clip 掉指数的溢出:拟合器会试探很远的 v0/w 组合。
    z = np.clip((v - v0) / max(w, 1e-9), -60.0, 60.0)
    return a + b * v + h / (1.0 + np.exp(-z))


def _bic(rss: float, n: int, k: int) -> float:
    """高斯残差下的 BIC。``rss`` 为 0 时给一个极小值兜底(完美拟合)。"""
    rss = max(float(rss), 1e-300)
    return n * math.log(rss / n) + k * math.log(max(n, 2))


def assess_shockley_onset(
    bias_v: npt.ArrayLike,
    didv: npt.ArrayLike,
    *,
    expected_onset_v: float,
    tol_v: float = 0.020,
    lockin_mod_vrms: float = 0.005,
    temperature_k: float = 4.2,
    fit_window_v: float = 0.15,
    width_max_v: float = 0.040,
    amp_sigma_k: float = 5.0,
    r2_floor: float = 0.85,
    delta_bic_min: float = 10.0,
    min_points: int = 15,
) -> ShockleyOnsetResult:
    """在 ``expected_onset_v`` 附近找肖克利表面态台阶,判它在不在容差内。

    纯函数:输入两条等长数组(``bias_v`` 伏、``didv`` 任意线性单位)与一组显式阈值,
    输出一个 frozen 结果。不读配置、不碰硬件、不抛异常。

    判据(五条全过才 ``passed``):

    1. ``|onset − expected| ≤ tol_v``;
    2. 台阶高度 ≥ ``amp_sigma_k`` × 残差噪声,且 ≥ 谱动态范围的 20%;
    3. 台阶宽度 ≤ ``max(width_max_v, 2 × width_floor_v)`` —— 上限是「别把一整段
       缓慢抬升叫做台阶」,而下限由拟合约束在仪器展宽处(见模块注释);
    4. 窗内 ``r² ≥ r2_floor``;
    5. ``ΔBIC(线性 → 台阶) ≥ delta_bic_min``。

    ``didv`` 整体倒置(lock-in 相位差 180°)时不自动翻转 —— 那会把「相位设错了」
    悄悄变成「针尖不合格」。拟合约束台阶为上升,倒置谱会以 ``no_step`` 落选,并在
    ``warnings`` 里点名 ``didv_may_be_inverted``。
    """
    reasons: list[str] = []
    warns: list[str] = []
    expected = float(expected_onset_v)
    tol = abs(float(tol_v))
    w_floor = broadening_floor_v(temperature_k, lockin_mod_vrms)

    def _fail(*why: str) -> ShockleyOnsetResult:
        return ShockleyOnsetResult(
            passed=False, expected_onset_v=expected, tol_v=tol,
            width_floor_v=w_floor, reasons=tuple(why), warnings=tuple(warns))

    v_all = np.asarray(bias_v, dtype=np.float64).ravel()
    g_all = np.asarray(didv, dtype=np.float64).ravel()
    if v_all.size != g_all.size or v_all.size < min_points:
        return _fail("insufficient_data")
    ok = np.isfinite(v_all) & np.isfinite(g_all)
    v_all, g_all = v_all[ok], g_all[ok]
    if v_all.size < min_points:
        return _fail("insufficient_data")
    order = np.argsort(v_all)
    v_all, g_all = v_all[order], g_all[order]

    # 窗口化:全谱背景是弯的,窗外的弯曲会把线性背景项带偏。窗至少要能容下容差。
    half = max(float(fit_window_v), tol * 2.0, 4.0 * w_floor)
    sel = (v_all >= expected - half) & (v_all <= expected + half)
    v, g = v_all[sel], g_all[sel]
    if v.size < min_points:
        return _fail("insufficient_data")
    # 期望位置附近两侧都要有数据,否则「台阶」可能只是窗边缘的斜率。
    if not (np.any(v < expected - tol) and np.any(v > expected + tol)):
        return _fail("insufficient_data")

    span = float(np.percentile(g, 95.0) - np.percentile(g, 5.0))
    if not np.isfinite(span) or span <= 0.0:
        return _fail("no_step")

    n = int(v.size)
    # ── 参考模型:纯线性背景 ──
    lin_coef, *_ = np.linalg.lstsq(np.c_[v, np.ones_like(v)], g, rcond=None)
    lin_pred = lin_coef[0] * v + lin_coef[1]
    rss_lin = float(np.sum((g - lin_pred) ** 2))

    # ── seed:平滑后导数的极值位置。只作初值与交叉验证,不作结论 ──
    try:
        from scipy import ndimage as _ndi
        # 平滑尺度取仪器展宽对应的点数(至少 1 点)。
        dv = float(np.median(np.diff(v))) or 1e-6
        sig_px = max(1.0, w_floor / max(abs(dv), 1e-12) / 2.0)
        gs = _ndi.gaussian_filter1d(g, min(sig_px, max(1.0, n / 8.0)))
    except Exception:  # noqa: BLE001 — scipy 缺席时退回未平滑
        gs = g
    dg = np.gradient(gs, v)
    v0_seed = float(v[int(np.argmax(dg))])

    # ── 台阶模型 ──
    try:
        from scipy.optimize import curve_fit
    except Exception:  # noqa: BLE001
        return _fail("scipy_unavailable")

    p0 = [float(np.percentile(g, 5.0)), float(lin_coef[0]), span, v0_seed,
          max(w_floor, 1e-4)]
    # w 的下限是物理先验:台阶不可能比 kT + 调制展宽更陡。没有它,一个 tip switch
    # 的单点尖峰会被拟合成宽度→0 的「完美台阶」,r² 还很高。
    w_lo = max(w_floor * 0.5, 1e-5)
    w_hi = max(float(half), w_lo * 2.0)
    bounds = (
        [-np.inf, -np.inf, 0.0, float(v.min()), w_lo],
        [np.inf, np.inf, 10.0 * span, float(v.max()), w_hi],
    )
    try:
        popt, pcov = curve_fit(_logistic_step, v, g, p0=p0, bounds=bounds,
                               maxfev=20000)
    except Exception:  # noqa: BLE001 — 拟合不收敛就是「判不了」
        return _fail("fit_failed")

    a, b, h, v0, w = (float(x) for x in popt)
    pred = _logistic_step(v, *popt)
    resid = g - pred
    rss = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2)) + 1e-300
    r2 = float(np.clip(1.0 - rss / ss_tot, 0.0, 1.0))
    sigma = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
    sigma = max(sigma, 1e-300)
    ratio = float(h / sigma)
    frac = float(h / span) if span > 0 else 0.0
    d_bic = _bic(rss_lin, n, 2) - _bic(rss, n, 5)
    try:
        onset_err = float(np.sqrt(abs(pcov[3][3])))
        if not np.isfinite(onset_err):
            onset_err = None
    except Exception:  # noqa: BLE001
        onset_err = None

    # 10-90 宽度:逻辑斯蒂的 10→90 跨度是 ln(81)·w ≈ 4.394·w。判据比的是它。
    width_1090 = 4.394 * w

    # ── 逐条判据 ──
    if abs(v0 - expected) > tol:
        reasons.append("onset_out_of_window")
    if ratio < amp_sigma_k or frac < 0.20:
        reasons.append("low_amplitude")
    if width_1090 > max(float(width_max_v), 2.0 * w_floor):
        reasons.append("too_wide")
    if r2 < float(r2_floor):
        reasons.append("poor_fit")
    if d_bic < float(delta_bic_min):
        reasons.append("no_step")

    # ── 说明性 warning(不改结论) ──
    if abs(v0 - v0_seed) > max(tol, 2.0 * w_floor):
        warns.append("fit_derivative_disagree")
    if w_floor >= abs(expected) * 0.5 and abs(expected) > 0:
        # Ag(111) 的 −65 mV 在 77 K + 10 mV 调制下就是这种情况:展宽与 onset 深度
        # 同量级,位置判据本身的分辨力已经很勉强。
        warns.append("broadening_comparable_to_onset")
    if "no_step" in reasons or "low_amplitude" in reasons:
        # 上升台阶找不到时,看看反过来是不是有个下降台阶 —— 那多半是 lock-in
        # 相位差了 180°,是接线问题不是针尖问题。
        try:
            popt_i, _ = curve_fit(_logistic_step, v, -g, p0=p0, bounds=bounds,
                                  maxfev=20000)
            resid_i = (-g) - _logistic_step(v, *popt_i)
            sig_i = max(float(np.median(np.abs(
                resid_i - np.median(resid_i)))) * 1.4826, 1e-300)
            if (float(popt_i[2]) / sig_i >= amp_sigma_k
                    and abs(float(popt_i[3]) - expected) <= tol):
                warns.append("didv_may_be_inverted")
        except Exception:  # noqa: BLE001 — 诊断性尝试，失败就不说
            pass

    return ShockleyOnsetResult(
        passed=not reasons,
        onset_v=v0,
        onset_err_v=onset_err,
        expected_onset_v=expected,
        tol_v=tol,
        step_height=h,
        step_sigma_ratio=ratio,
        step_frac_of_range=frac,
        width_v=width_1090,
        width_floor_v=w_floor,
        background_slope=b,
        r2=r2,
        delta_bic=float(d_bic),
        n_points_fit=n,
        reasons=tuple(reasons),
        warnings=tuple(warns),
    )


# ─────────────────────────────────────────────────────────────────────────
# 谱数据质量 —— 「这一条谱值不值得留」
# ─────────────────────────────────────────────────────────────────────────
#
# 判据来自 S4 STS 设计（D2-D6）。它与上面的 assess_iv/assess_iz 是**消费关系**:
# 那两个函数的签名与语义一个字都不改(它们还有别的消费者),这里只决定**哪几条
# 能当闸**。
#
# ## 为什么不能直接拿 assess_iv 当谱质量闸
#
# assess_iv 的 spike 判据拿**全曲线**的 MAD(d1) 当噪声尺度。带隙式曲线有一整段
# 落在隙内近乎水平,MAD 被这一段主导,于是指数上升段每一点都「超过 8 MAD」。
# 合成实测(见 tests/v2/unit/vision/test_spectrum_quality.py,数值逐条钉死):
#
#   * 一条**无噪声、教科书式**的带隙 I(V)(隙 0.6 V,400 点):n_spikes=53、
#     is_stable=False —— 好谱被判成坏谱;
#   * 同一条曲线叠一次**真实的 15% tip switch**:n_spikes 53 → 53,**差值为 0**
#     —— 不是保守,是没有分辨力(switch 落在隙内,那里电流本来就是 0);
#   * 而金属式(sinh)曲线上它工作得很好:干净 spikes=0,阶跃即触发。
#
# ⇒ assess_iv 是一个**被正确实现、但被用在错误问题上**的判据。所以这里按
#   `spectral_family` 分闸:族无关的三条(饱和/SNR/正反扫迟滞)永远能当闸,
#   n_spikes/symmetry 只在 metallic 上当闸,unknown ⇒ **只有族无关的三条**,
#   绝不默认按 metallic 处理。
#
# ## 为什么金属族当闸的是 n_spikes 而不是 is_stable
#
# `is_stable = n_spikes==0 and smoothness>0.5`,而 smoothness 随点数单调漂移
# (同一条解析曲线 n=40→0.788、n=1000→0.991),再叠 0.1% 满量程噪声后 400 点的
# **干净** sinh 曲线 smoothness 只有 0.125 ⇒ is_stable=False。拿 is_stable 当闸
# 等于把一个未归一化的点数依赖阈值从后门放进来,与同一张表里「smoothness 点数
# 归一后才当闸」自相矛盾。归一化形式待真机数据定(设计 O5),在此之前
# `min_smoothness` 保持 None。
#
# ## 未标定 ⇒ unrated,永远不产 bad(这一条由结构强制,不靠自觉)
#
# 每个可当闸的阈值都是 `float | None`,None = 未标定 ⇒ 该子判据不进
# `gated_criteria`,**没有任何代码路径能让它产出一条失败原因**。教训来源:一个
# 判不了的判据被当成「不合格」,会让「合格」在结构上不可达。

#: 出口四态,闭集。「判不了」永远不折叠成「不合格」。
SPECTRUM_VERDICTS: tuple[str, ...] = ("keep", "keep_flagged", "discard", "unrated")

#: 衬底族。`unknown` **不是** metallic 的同义词。
SPECTRAL_FAMILIES: tuple[str, ...] = ("metallic", "gapped", "unknown")

#: 子判据名,闭集。结果里的 gated_criteria / ungated_criteria 只用这些词。
SPECTRUM_CRITERIA: tuple[str, ...] = (
    "saturation", "snr", "hysteresis",          # 族无关
    "smoothness", "iv_stability", "symmetry",   # I(V),族相关
    "iz_exponential",                           # I(z)
)

#: `reasons` 闭集词表。
#: 三条与设计 §3.2 的词表相比新增,理由各写在用它的地方:
#:   ``asymmetric_iv``  —— symmetry 一旦被标定成闸,失败必须说得出原因;
#:   ``no_z_column``    —— I(z) 缺的是 Z 列,拿 no_bias_column 说它会误导;
#:   ``substrate_unknown`` 是**保留词**:本期 spectral_family 走显式参数,不做
#:   衬底推断接线(设计 O1 未决),所以没有任何代码路径会产出它。
SPECTRUM_REASONS: tuple[str, ...] = (
    "insufficient_points", "no_bias_column", "no_z_column", "no_current_column",
    "no_backward_column", "low_snr", "saturated", "hysteresis_exceeded",
    "unstable_iv", "asymmetric_iv", "poor_iz_fit", "barrier_out_of_range",
    "all_criteria_uncalibrated", "substrate_unknown", "kind_undetermined",
)

#: `warnings` 闭集词表 —— 结论仍成立但你该知道。
SPECTRUM_WARNINGS: tuple[str, ...] = (
    "didv_polarity_suspect", "gap_ev_is_not_a_gap_measurement",
    "saturation_near_limit", "hysteresis_reported_only",
    "kind_disagrees_with_header", "smoothness_point_count_dependent",
)

#: 会把 keep 降级成 keep_flagged 的那几条 warning —— 「有一件需要人看的事」。
#: 另外两条(gap_ev 不是带隙、smoothness 依赖点数)是**常驻告诫**不是事件,
#: 每条谱都会带上,让它们降级等于所有谱都 keep_flagged,那个字段就没用了。
_FLAGGING_WARNINGS: frozenset[str] = frozenset({
    "didv_polarity_suspect", "saturation_near_limit",
    "hysteresis_reported_only", "kind_disagrees_with_header",
})

#: 每个族里**能当闸**的子判据(设计 D3 那张表逐字)。族无关的三条不在这里,
#: 它们对所有族都能当闸。
_FAMILY_GATEABLE: dict[str, frozenset[str]] = {
    "metallic": frozenset({"iv_stability", "symmetry"}),
    "gapped": frozenset(),      # 见模块注释:这两条在带隙曲线上是错的判据
    "unknown": frozenset(),     # 「不知道」不是 metallic
}


def saturation_frac(current: npt.ArrayLike, *, eps: float = 1e-3,
                    flat_rel: float = 1e-3, min_run: int = 3) -> float | None:
    """贴轨点占比 —— 前置放大器有没有被打饱和。

    判据是**重复的极值**而不是「接近最大值」:饱和会把连续多点钉在同一个数上,
    而「接近最大值」在指数曲线上永远有几个点(扫到窗口边缘的那几点),拿它当判据
    会把每一条正常的 I(V) 都判成饱和。

    一段算「饱和」要同时满足:``|I| ≥ (1−eps)·max|I|``、相邻点取值相同(差
    ≤ ``flat_rel·max|I|``)、且这样的连续点数 ≥ ``min_run``。

    返回落在饱和段里的点数占比;点数不足 ⇒ ``None``(判不了,不是 0)。
    """
    I = np.asarray(current, dtype=np.float64).ravel()
    I = I[np.isfinite(I)]
    n = int(I.size)
    if n < max(int(min_run), 2):
        return None
    A = np.abs(I)
    mx = float(np.max(A))
    if not np.isfinite(mx) or mx <= 0.0:
        return 0.0
    near = A >= (1.0 - float(eps)) * mx
    flat = np.abs(np.diff(I)) <= float(flat_rel) * mx
    count = 0
    i = 0
    while i < n:
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and near[j + 1] and flat[j]:
            j += 1
        if (j - i + 1) >= int(min_run):
            count += j - i + 1
        i = j + 1
    return float(count) / float(n)


def spectrum_snr(current: npt.ArrayLike) -> float | None:
    """谱的信噪比 = 动态范围 / 逐点噪声。

    ``span`` 用 5-95 分位跨度(与 :func:`assess_shockley_onset` 同款,避免单点极值
    主导);噪声用**二阶差分**的 MAD×1.4826/√6 估计 —— 二阶而不是一阶,因为一阶
    差分里还含曲线本身的斜率,拿它当噪声会把一条陡的好谱说成噪声大。系数 √6 是
    白噪声经 (1,−2,1) 卷积后方差放大 6 倍的还原。

    噪声在浮点精度以下(严格线性/常数输入)或动态范围为 0 ⇒ ``None``(量不出来,
    不是「信噪比为 0」)。信噪比需结合曲线动态范围和噪声估计解释。
    """
    I = np.asarray(current, dtype=np.float64).ravel()
    I = I[np.isfinite(I)]
    if I.size < 5:
        return None
    span = float(np.percentile(I, 95.0) - np.percentile(I, 5.0))
    if not np.isfinite(span) or span <= 0.0:
        return None
    d2 = np.diff(I, 2)
    if d2.size < 3:
        return None
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    sigma = mad * 1.4826 / math.sqrt(6.0)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return None
    return float(span / sigma)


@dataclass(frozen=True)
class HysteresisResult:
    """正反扫比对的**三个**数,以及为什么可能一个都没有。

    三个数各回答一个不同的问题,少报一个就会被读错:

    * ``outlier_points`` —— 「两个方向之间**发生了什么**吗」。这是全套判据里
      **唯一**对「扫描过程中针尖变了」有分辨力的一条:所有单方向统计量对此结构
      性地瞎(它们看不到「本来该是什么样」)。
    * ``median_frac`` —— 「这里到底**有没有信号**」。两条独立白噪声的中位差就是
      噪声本身的量级(实测 0.297),而超阈点数是 0。**只看点数会把一堆噪声判成
      「正反扫一致」。**
    * ``max_frac`` —— 「变了**多少**」。

    ``available=False`` 表示这一族没算(缺反扫列 / 点数不足) ⇒ 三个数全 None。
    **那不是「没有迟滞」。**
    """

    available: bool
    median_frac: float | None = None
    max_frac: float | None = None
    outlier_points: int | None = None
    outlier_frac: float | None = None
    n_points: int = 0
    reason: str = ""


def assess_hysteresis(forward: npt.ArrayLike, backward: npt.ArrayLike, *,
                      sigma_k: float = 8.0) -> HysteresisResult:
    """正扫 vs 反扫逐点比对。两列共用同一根偏压轴、逐行对齐,可以**直接相减**。

    这与二维图的 trace/retrace 配准不是同一件事:那里要留 ±12% 平移窗吸收压电
    迟滞,因为快轴是「扫」出来的;这里偏压轴是「命令」出来的,同一行就是同一个
    偏压。

    ``sigma_k`` 沿用本模块既有的 8×MAD 约定(assess_iv/assess_iz 同款)。
    """
    F = np.asarray(forward, dtype=np.float64).ravel()
    B = np.asarray(backward, dtype=np.float64).ravel()
    if F.size != B.size:
        return HysteresisResult(available=False, reason="length_mismatch")
    ok = np.isfinite(F) & np.isfinite(B)
    F, B = F[ok], B[ok]
    n = int(F.size)
    if n < 5:
        return HysteresisResult(available=False, n_points=n,
                                reason="insufficient_points")

    d = np.abs(F - B)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sig = mad * 1.4826
    # 严格 `>`:两列完全相同时 d 恒为 0、sig 也是 0,`0 > 0` 为假 ⇒ 0 个离群点。
    # 而「大部分点相同、少数几点跳变」时 sig=0 会让那几点如实计入 —— 这正是要的。
    n_out = int(np.sum(d > float(sigma_k) * sig))
    span = float(np.percentile(F, 95.0) - np.percentile(F, 5.0))
    if not np.isfinite(span) or span <= 0.0:
        # 正扫本身没有动态范围 ⇒ 两个比值没有分母。点数照报。
        return HysteresisResult(available=True, outlier_points=n_out,
                                outlier_frac=n_out / n, n_points=n,
                                reason="no_forward_span")
    return HysteresisResult(
        available=True,
        median_frac=med / span,
        max_frac=float(np.max(d)) / span,
        outlier_points=n_out,
        outlier_frac=n_out / n,
        n_points=n,
    )


def resolve_spectrum_kind(
    bias_v: npt.ArrayLike | None = None,
    z_m: npt.ArrayLike | None = None,
    *,
    bias_sweep_floor_v: float = 1e-3,
    z_sweep_floor_m: float = 1e-11,
) -> tuple[str, str]:
    """``(kind, evidence)`` —— 看**哪一列在扫**,而不是看头里写了什么。

    ``kind`` ∈ ``{"iv", "iz", ""}``,空串 = 判不了。``evidence`` 是一句给人看的
    依据(哪一列扫了多大范围)。

    两个门槛是**物理下限**不是标定阈值:1 mV 以下谈不上 I(V),10 pm 以下谈不上
    I(z)。两条都在扫或都不在扫 ⇒ 判不了,让调用方显式给 kind —— 不猜。

    为什么不信头:文件头的 ``Experiment`` 字段是仪器软件上一次的设置留下的,
    「字段标签会说谎」。头只作交叉检验。
    """
    def _span(arr) -> float | None:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size < 2:
            return None
        return float(np.max(a) - np.min(a))

    sb, sz = _span(bias_v), _span(z_m)
    bias_sweeping = sb is not None and sb >= float(bias_sweep_floor_v)
    z_sweeping = sz is not None and sz >= float(z_sweep_floor_m)
    parts = []
    if sb is not None:
        parts.append(f"偏压列跨度 {sb:.4g} V")
    if sz is not None:
        parts.append(f"Z 列跨度 {sz * 1e9:.4g} nm")
    ev = "；".join(parts) if parts else "既无偏压列也无 Z 列"

    if bias_sweeping and not z_sweeping:
        return "iv", f"{ev} ⇒ 在扫的是偏压"
    if z_sweeping and not bias_sweeping:
        return "iz", f"{ev} ⇒ 在扫的是 Z"
    if bias_sweeping and z_sweeping:
        return "", f"{ev} ⇒ 两条都在扫，判不了，请显式给 kind"
    return "", f"{ev} ⇒ 没有一条在扫，判不了，请显式给 kind"


@dataclass(frozen=True)
class SpectrumQualityResult:
    """一条谱值不值得留,以及**这个结论是靠哪几条判据做出来的**。

    ``verdict`` 是唯一的结论,四态闭集(:data:`SPECTRUM_VERDICTS`)。

    ``gated_criteria`` / ``ungated_criteria`` 不是装饰:没有这两个字段,
    ``verdict`` 就是一句没有依据的话 —— ``gated_criteria`` 为空时的 ``keep``
    只能读作「采到了」,绝不能读作「合格」,而这两句话下游必须能分辨。

    **``discard`` 不等于「针尖坏了」**:扫的窗口不对、lock-in 相位反了、表面不是
    那个面,都会落到这里。``reasons`` 是给人看的。
    """

    verdict: str
    kind: str = ""
    spectral_family: str = "unknown"
    gated_criteria: tuple[str, ...] = ()
    ungated_criteria: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    n_points: int = 0
    #: 因为非有限值被丢掉的行数。``read_dat`` 会把截断的末行 NaN 补齐,那些行
    #: 判不了 —— 但丢了多少必须说出来,不能只剩一个变小的 n_points。
    n_points_dropped: int = 0
    # 三条新判据
    spectrum_snr: float | None = None
    saturation_frac: float | None = None
    hysteresis_median_frac: float | None = None
    hysteresis_max_frac: float | None = None
    hysteresis_outlier_points: int | None = None
    hysteresis_outlier_frac: float | None = None
    # assess_iv 原样透传
    iv_smoothness: float | None = None
    iv_symmetry: float | None = None
    iv_n_spikes: int | None = None
    iv_is_stable: bool | None = None
    iv_gap_ev: float | None = None
    # assess_iz 原样透传
    iz_fit_r2: float | None = None
    iz_barrier_ev: float | None = None
    iz_decay_per_nm: float | None = None
    iz_n_jumps: int | None = None
    iz_is_clean_exponential: bool | None = None


def assess_spectrum_quality(
    *,
    kind: str,
    current: npt.ArrayLike,
    bias_v: npt.ArrayLike | None = None,
    z_nm: npt.ArrayLike | None = None,
    current_bwd: npt.ArrayLike | None = None,
    didv: npt.ArrayLike | None = None,
    spectral_family: str = "unknown",
    min_snr: float | None = None,
    max_saturation_frac: float | None = None,
    max_hysteresis_outlier_frac: float | None = None,
    min_smoothness: float | None = None,
    min_symmetry: float | None = None,
    require_backward: bool = False,
    min_points: int = 7,
    extra_warnings: tuple[str, ...] = (),
) -> SpectrumQualityResult:
    """一条谱值不值得留。纯函数:不读配置、不碰硬件、不抛异常。

    ``kind`` ∈ ``{"iv","iz"}`` 由调用方解析好(内容判据见
    :func:`resolve_spectrum_kind`)。``spectral_family`` 决定哪些子判据能当闸
    (见模块注释与 :data:`_FAMILY_GATEABLE`)。

    **阈值为 ``None`` = 未标定 ⇒ 该子判据只报数、永远不产出失败原因。**
    所有可当闸项都未标定 ⇒ ``verdict="unrated"``、``reasons`` 含
    ``all_criteria_uncalibrated`` —— 「判不了」既不计产量也不计不合格。

    ``extra_warnings`` 给外壳注入它才看得到的告警(例如头里的 Experiment 字段与
    数据不一致)。它必须在**定 verdict 之前**进来,否则外壳事后追加一条会让
    verdict 与 warnings 互相矛盾。
    """
    family = str(spectral_family or "unknown").strip().lower()
    if family not in SPECTRAL_FAMILIES:
        family = "unknown"
    kind = str(kind or "").strip().lower()
    reasons: list[str] = []
    warns: list[str] = [w for w in extra_warnings if w in SPECTRUM_WARNINGS]

    I = np.asarray(current, dtype=np.float64).ravel() if current is not None \
        else np.empty(0)
    n = int(I.size)
    dropped = 0

    def _out(verdict: str, *, gated=(), ungated=(), **kw) -> SpectrumQualityResult:
        return SpectrumQualityResult(
            verdict=verdict, kind=kind, spectral_family=family,
            gated_criteria=tuple(gated), ungated_criteria=tuple(ungated),
            reasons=tuple(reasons), warnings=tuple(warns), n_points=n,
            n_points_dropped=dropped, **kw)

    # ── 结构性「读不动」:先于任何判据,且只报这一条 ──
    if kind not in ("iv", "iz"):
        reasons.append("kind_undetermined")
        return _out("unrated")
    if n < int(min_points):
        reasons.append("insufficient_points")
        return _out("unrated")
    if kind == "iv" and bias_v is None:
        reasons.append("no_bias_column")
        return _out("unrated")
    if kind == "iz" and z_nm is None:
        reasons.append("no_z_column")
        return _out("unrated")

    # ── 非有限值先丢掉,逐列同步丢,别让它们进下游 ──
    # ``read_dat`` 会把**截断的末行**用 NaN 补齐,而
    # assess_iv 不过滤 NaN:一个 NaN 会让 smoothness 变成 NaN,pydantic 的
    # `0 ≤ x ≤ 1` 约束当场抛 ValidationError。纯函数承诺「不抛异常」,所以这道
    # 过滤必须在这里,而不是去改 assess_iv 的语义(它还有别的消费者)。
    axis_raw = bias_v if kind == "iv" else z_nm
    axis = np.asarray(axis_raw, dtype=np.float64).ravel()
    if axis.size != n:
        reasons.append("insufficient_points")
        return _out("unrated")
    keep = np.isfinite(axis) & np.isfinite(I)
    if not bool(np.all(keep)):
        dropped = int(n - keep.sum())

        def _mask(a):
            if a is None:
                return None
            arr = np.asarray(a, dtype=np.float64).ravel()
            return arr[keep] if arr.size == n else arr

        axis, I = axis[keep], I[keep]
        current_bwd, didv = _mask(current_bwd), _mask(didv)
        n = int(I.size)
        if n < int(min_points):
            reasons.append("insufficient_points")
            return _out("unrated")

    metrics: dict = {}
    gated: list[str] = []
    ungated: list[str] = []
    fails: list[str] = []

    def _judge(name: str, *, ok: bool | None, threshold_set: bool,
               family_allows: bool = True, reason: str = "") -> None:
        """一个子判据的**唯一**入口:算不出 / 族不允许 / 阈值未标定 ⇒ 只报数。"""
        if ok is None or not family_allows or not threshold_set:
            ungated.append(name)
            return
        gated.append(name)
        if not ok and reason:
            fails.append(reason)

    # ── 族无关三条 ───────────────────────────────────────────────
    sat = saturation_frac(I)
    metrics["saturation_frac"] = sat
    _judge("saturation",
           ok=(None if sat is None else sat <= float(max_saturation_frac or 0.0)),
           threshold_set=max_saturation_frac is not None,
           reason="saturated")
    if sat:
        # 检出任何一段贴轨都值得人看一眼,不需要一个阈值才能说这句话。
        warns.append("saturation_near_limit")

    snr = spectrum_snr(I)
    metrics["spectrum_snr"] = snr
    _judge("snr",
           ok=(None if snr is None else snr >= float(min_snr or 0.0)),
           threshold_set=min_snr is not None,
           reason="low_snr")

    hys = (assess_hysteresis(I, current_bwd) if current_bwd is not None
           else HysteresisResult(available=False, reason="no_backward_column"))
    metrics.update(
        hysteresis_median_frac=hys.median_frac,
        hysteresis_max_frac=hys.max_frac,
        hysteresis_outlier_points=hys.outlier_points,
        hysteresis_outlier_frac=hys.outlier_frac,
    )
    if not hys.available:
        # 缺反扫列是一条**要说出来的事实**,不是「通过」。设计 D5 逐字。
        reasons.append("no_backward_column")
    _judge("hysteresis",
           ok=(None if hys.outlier_frac is None
               else hys.outlier_frac <= float(max_hysteresis_outlier_frac or 0.0)),
           threshold_set=max_hysteresis_outlier_frac is not None,
           reason="hysteresis_exceeded")
    if hys.available and max_hysteresis_outlier_frac is None:
        # 全套里唯一对「扫描中途针尖变了」有分辨力的一条没有当闸 —— 这批数据的
        # 结论里缺的正是这一块,得让人知道。
        warns.append("hysteresis_reported_only")

    # ── I(V) 族相关 ─────────────────────────────────────────────
    if kind == "iv":
        iv = assess_iv(axis, I)
        metrics.update(iv_smoothness=iv.smoothness, iv_symmetry=iv.symmetry,
                       iv_n_spikes=iv.n_spikes, iv_is_stable=iv.is_stable,
                       iv_gap_ev=iv.gap_ev)
        if iv.gap_ev is not None:
            # 两个方向都错(无隙曲线报出隙、真隙报大三倍)。它是粗糙代理,
            # 每次上报都要带着这句话,绝不能以「带隙」之名传下去。
            warns.append("gap_ev_is_not_a_gap_measurement")
        _judge("iv_stability", ok=(iv.n_spikes == 0), threshold_set=True,
               family_allows="iv_stability" in _FAMILY_GATEABLE[family],
               reason="unstable_iv")
        _judge("symmetry",
               ok=(iv.symmetry >= float(min_symmetry or 0.0)),
               threshold_set=min_symmetry is not None,
               family_allows="symmetry" in _FAMILY_GATEABLE[family],
               reason="asymmetric_iv")
        _judge("smoothness",
               ok=(iv.smoothness >= float(min_smoothness or 0.0)),
               threshold_set=min_smoothness is not None,
               reason="unstable_iv")
        if min_smoothness is not None:
            # 同一条解析曲线 n=40→0.788、n=1000→0.991。写死的阈值只对某一个
            # num_points 成立,而 num_points 是流程的自由参数。
            warns.append("smoothness_point_count_dependent")
    else:
        iz = assess_iz(axis, I)
        metrics.update(iz_fit_r2=iz.fit_r2, iz_barrier_ev=iz.barrier_ev,
                       iz_decay_per_nm=iz.decay_per_nm, iz_n_jumps=iz.n_jumps,
                       iz_is_clean_exponential=iz.is_clean_exponential)
        # I(z) 的物理形状不随材料带隙变(隧穿衰减是真空势垒的事) ⇒ 族无关,
        # 且阈值(r²>0.90、势垒 0.5-8 eV、跳变 0)是物理先验不是样品标定值。
        _judge("iz_exponential", ok=iz.is_clean_exponential, threshold_set=True,
               reason=("barrier_out_of_range"
                       if (iz.barrier_ev is None
                           or not 0.5 <= iz.barrier_ev <= 8.0)
                       else "poor_iz_fit"))

    # ── dI/dV 极性:报警,**绝不自动翻转** ──
    # 隧穿电导物理上非负。5-95 分位带整体落在 0 以下 ⇒ lock-in 相位差 180°,
    # 那是接线问题不是数据质量问题。自动取反会把「相位设错了」悄悄变成一条
    # 看起来正常的谱,错误就此永久消失。
    if didv is not None:
        g = np.asarray(didv, dtype=np.float64).ravel()
        g = g[np.isfinite(g)]
        if g.size >= 5 and float(np.percentile(g, 95.0)) < 0.0:
            warns.append("didv_polarity_suspect")

    reasons.extend(fails)

    # ── 出口四态 ────────────────────────────────────────────────
    if require_backward and not hys.available:
        verdict = "unrated"
    elif not gated:
        reasons.append("all_criteria_uncalibrated")
        verdict = "unrated"
    elif fails:
        verdict = "discard"
    elif any(w in _FLAGGING_WARNINGS for w in warns):
        verdict = "keep_flagged"
    else:
        verdict = "keep"

    # 去重但保序 —— 两个子判据可以映到同一个原因词(smoothness 与 iv_stability
    # 都是 unstable_iv),下游按闭集词表读,同一个词出现两次没有意义。
    def _uniq(seq):
        seen: set[str] = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    warns = _uniq(warns)
    reasons = _uniq(reasons)
    return _out(verdict, gated=gated, ungated=ungated, **metrics)


__all__ = [
    "assess_iz",
    "assess_iv",
    "assess_shockley_onset",
    "assess_hysteresis",
    "assess_spectrum_quality",
    "broadening_floor_v",
    "resolve_spectrum_kind",
    "saturation_frac",
    "spectrum_snr",
    "HysteresisResult",
    "ShockleyOnsetResult",
    "SpectrumQualityResult",
    "SPECTRAL_FAMILIES",
    "SPECTRUM_CRITERIA",
    "SPECTRUM_REASONS",
    "SPECTRUM_VERDICTS",
    "SPECTRUM_WARNINGS",
]
