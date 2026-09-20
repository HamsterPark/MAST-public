"""Cheap, interpretable classical tip-quality signals — network-free.

These complement (and cross-check) the learned heads and repair their documented
blind spots — most importantly, a featureless / pure-noise frame scores *low*
here, instead of the deployed model's "good, Q≈71". All are O(one FFT) or O(N).

  * FFT Bragg sharpness / resolution — a sharp tip on a periodic surface gives a
    tall, tight Bragg peak; a blunt / multi tip smears it. Peak prominence ↑ =
    sharper; the peak frequency gives the finest resolved period.
  * forward−backward instability — trace and retrace of a *stable* tip agree; a
    tip that changed / feedback that rang during the scan makes them disagree.
    This is exactly the E3 channel the model consumes as input but never reports.
  * terrace z-noise / flatness — RMS roughness of the flattest region.

See ``docs/v2/benchmarks/vision_v25_diagnostic/``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mast.vision.module import TipMetricsResult


def _to_2d(a: npt.NDArray) -> npt.NDArray[np.float32]:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.mean(axis=-1)
    elif a.ndim != 2:
        raise ValueError(f"expected 2-D, got {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float32)


def _split_fwd_bwd(image, bwd):
    """Return (fwd2d, bwd2d|None) from the flexible input forms."""
    a = np.asarray(image)
    if bwd is None and a.ndim == 3 and a.shape[0] == 2:      # (2,H,W) trace/retrace
        return _to_2d(a[0]), _to_2d(a[1])
    if a.ndim == 3 and a.shape[0] in (1, 2):
        a = a[0]
    return _to_2d(a), (None if bwd is None else _to_2d(bwd))


def _detrend(h):
    h = h - np.median(h, axis=1, keepdims=True)
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]].astype(np.float32)
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(h.size, np.float32)]
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    return (h.ravel() - A @ coef).reshape(h.shape).astype(np.float32)


def _fft_sharpness(hn, nm_per_px):
    """(sharpness prominence, resolved-period nm | None, has_lattice)."""
    N = hn.shape[0]
    F = np.fft.fftshift(np.fft.fft2(hn))
    mag = np.abs(F)
    cy, cx = N // 2, hn.shape[1] // 2
    yy, xx = np.ogrid[:N, :hn.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    ring = r > 3.0                                   # ignore DC / very low freq
    if not ring.any():
        return 0.0, None, False
    vals = mag[ring]
    med = float(np.median(vals)) + 1e-9
    peak = float(vals.max())
    sharp = peak / med                               # prominence over background
    has_lat = sharp > 8.0
    # frequency (radius in px) of the strongest peak → resolved period
    j = int(np.argmax(mag * ring))
    py, px = np.unravel_index(j, mag.shape)
    rad = float(np.hypot(py - cy, px - cx))
    res_nm = (float(N / rad * nm_per_px) if (rad > 0 and nm_per_px and nm_per_px > 0) else None)
    return float(sharp), (res_nm if has_lat else None), has_lat


def _fwd_bwd_instability(fwd, bwd, max_shift_frac: float = 0.12):
    """∈[0,1]: 1 − max normalised cross-correlation over lateral shifts. 0 =
    trace/retrace agree (stable), 1 = uncorrelated (unstable / tip ringing).

    Allowing a lateral shift absorbs the **piezo-hysteresis fast-axis offset**
    between trace and retrace (measured ~6-7 px, dy≈0 on real Createc hardware) —
    a zero-shift per-pixel metric SATURATES on real data (correlation ~0.29 →
    every frame flagged), even for a perfectly stable tip. Verified on 7,287 real
    scans: correlation 0.29 → 0.58 once the offset is allowed (Agent-B,
    2026-07-23 cross-agent finding). Synthetic data lacks this offset, so the
    old metric only looked fine on the bench.
    """
    a = _detrend(fwd); b = _detrend(bwd)
    a = a - a.mean(); b = b - b.mean()
    na = float(np.sqrt(np.sum(a * a))); nb = float(np.sqrt(np.sum(b * b)))
    # 范数守卫仅用于避免除零，不能对带单位的形貌幅度另设绝对判定阈值。
    # 归一化互相关应对单位换算与采样尺寸保持一致；极小值与一维相关实现一致。
    # 尺度不变性由 test_tip_metrics_scale_invariance.py 验证。
    if na < 1e-30 or nb < 1e-30:
        return 1.0
    xc = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))) / (na * nb)
    H, W = a.shape
    cy, cx = H // 2, W // 2
    ry = max(2, int(max_shift_frac * H)); rx = max(2, int(max_shift_frac * W))
    win = xc[cy - ry:cy + ry + 1, cx - rx:cx + rx + 1]
    max_ncc = float(win.max()) if win.size else float(xc.max())
    return float(np.clip(1.0 - max_ncc, 0.0, 1.0))


def _terrace_noise(hn):
    """(z_noise RMS on the flattest region, flat fraction)."""
    from scipy import ndimage as ndi
    gy, gx = np.gradient(ndi.gaussian_filter(hn, 1.0))
    gmag = np.hypot(gy, gx)
    flat = gmag < np.percentile(gmag, 40.0)
    flatness = float(flat.mean())
    resid = hn - ndi.gaussian_filter(hn, 2.0)
    z_noise = float(np.sqrt(np.mean(resid[flat] ** 2))) if flat.any() else float(hn.std())
    return z_noise, flatness


def _edge_resolution(h_std_units, nm_per_px):
    """Sharpest-step 10-90 rise width in px (≈ lateral resolution). A sharp tip
    resolves a step edge in few px; a blunt/multi tip smears it. Proxy:
    dominant step height / peak gradient. None when there is no clear step."""
    from scipy import ndimage as ndi
    sm = ndi.gaussian_filter(h_std_units, 1.0)
    g = np.hypot(*np.gradient(sm))
    gmax = float(np.percentile(g, 99.9))
    gmed = float(np.median(g)) + 1e-9
    step = float(np.percentile(sm, 97.0) - np.percentile(sm, 3.0))
    # a real step edge is a COHERENT gradient outlier (gmax ≫ typical gradient);
    # random noise has uniform gradient → no edge to measure resolution from.
    if gmax < 1e-6 or step < 0.3 or (gmax / gmed) < 6.0:
        return None, None
    width_px = float(np.clip(0.8 * step / gmax, 0.5, h_std_units.shape[0] / 2))
    width_nm = (width_px * nm_per_px) if (nm_per_px and nm_per_px > 0) else None
    return width_px, width_nm


def _terrace_levels(h_std_units):
    """Number of distinct terrace levels = significant modes in the height
    histogram (clean stepped surface → multi-modal; noise → one broad mode)."""
    from scipy import ndimage as ndi
    sm = ndi.gaussian_filter(h_std_units, 2.0)
    hist, _ = np.histogram(sm, bins=32)
    hist = ndi.gaussian_filter1d(hist.astype(np.float32), 2.0)
    thr = 0.20 * hist.max()
    peaks = 0
    for i in range(2, len(hist) - 2):
        if (hist[i] > thr and hist[i] > hist[i - 1] and hist[i] >= hist[i + 1]
                and hist[i] > hist[i - 2] and hist[i] > hist[i + 2]):
            peaks += 1
    return int(max(1, peaks))


# Bragg-family sharpness scale gate (physics-truth validation 2026-07-27):
# below 0.02 nm/px the criteria carry full weight; 0.02-0.05 is a transition
# band (reduced weight); above 0.05 the lattice is unresolvable and the
# criteria are physically inapplicable (measured rho +0.37 → +0.05).
# `has_lattice` CANNOT serve as this gate — 97 % of meso frames trigger it too.
_SHARPNESS_FULL_NMPP = 0.02
_SHARPNESS_OFF_NMPP = 0.05


def _sharpness_scale(nm_per_px: float | None) -> str | None:
    if nm_per_px is None or nm_per_px <= 0:
        return None
    if nm_per_px < _SHARPNESS_FULL_NMPP:
        return "full"
    if nm_per_px <= _SHARPNESS_OFF_NMPP:
        return "reduced"
    return "off"


def assess_tip_classical(image, bwd=None, nm_per_px: float | None = None,
                         template_bank=None) -> TipMetricsResult:
    """Compute the cheap classical tip-quality signals. Pass a (2,H,W) trace/
    retrace pair (or a separate ``bwd``) to enable fwd-bwd instability.

    ``template_bank`` (optional, from :mod:`mast.vision.barker_quality`)
    enables the Barker CCR criterion; feature circularity is computed always.
    These support tip-state TIER judgement only — see :class:`TipMetricsResult`."""
    fwd, bwd2 = _split_fwd_bwd(image, bwd)
    h = _detrend(fwd)
    std = float(h.std())
    if std < 1e-9:
        return TipMetricsResult(fft_sharpness=0.0, z_noise=0.0, flatness=1.0, has_lattice=False,
                                sharpness_scale=_sharpness_scale(nm_per_px))
    hn = h / std
    sharp, res_nm, has_lat = _fft_sharpness(hn, nm_per_px)
    z_noise, flatness = _terrace_noise(hn)
    edge_px, edge_nm = _edge_resolution(hn, nm_per_px)
    n_levels = _terrace_levels(hn)
    instab = None
    if bwd2 is not None and bwd2.shape == fwd.shape:
        instab = _fwd_bwd_instability(fwd, bwd2)
    # Barker-style criteria (guarded — must never break the basic metrics)
    ccr = None
    ccr_n = 0
    circ = None
    circ_n = 0
    try:
        from mast.vision.barker_quality import ccr_score, circularity_score
        circ, circ_n = circularity_score(fwd)
        if template_bank:
            ccr, ccr_n = ccr_score(fwd, template_bank)
    except Exception:  # noqa: BLE001
        pass
    return TipMetricsResult(
        fft_sharpness=float(sharp),
        resolution_nm=res_nm,
        fwd_bwd_instability=instab,
        z_noise=float(z_noise),
        flatness=float(flatness),
        has_lattice=bool(has_lat),
        edge_resolution_px=edge_px,
        edge_resolution_nm=edge_nm,
        n_terrace_levels=n_levels,
        sharpness_scale=_sharpness_scale(nm_per_px),
        barker_ccr=ccr,
        barker_ccr_n=int(ccr_n),
        circularity_dev=circ,
        circularity_n=int(circ_n),
    )


def trace_retrace_correlation(fwd, bwd, max_shift_frac: float = 0.12) -> float:
    """一维正反扫描线的一致性 ∈[-1, 1]。1 = 走出同一条形貌,0 = 不相关,负 = 反相。

    这是 :func:`_fwd_bwd_instability` 的**一维姊妹**(那个吃二维帧、用 FFT 互相关;
    这个吃单条线、用 ``np.correlate``)。两者共用同一条去趋势 :func:`_detrend`、
    同一条「允许横向平移」的理由、同一个 ``max_shift_frac`` 默认值。放在一起是
    有意的:同一个物理概念的两种维度,不该住在两个文件里各自漂移。

    ## 为什么必须**先去趋势**(它同时去掉了直流)

    在它之前,``PreScanCheck`` 与 ``CheckLineQuality`` 都在**绝对高度**上算余弦
    相似度(两处逐位同算法,实测确认)。带一个正常的 Z 工作点偏置(~1 nm)时,
    那个数被直流项统治,于是它回答的不是「这两条线走出同一条形貌吗」,而是
    「这两条线的均值差不多吗」。

    实测(合成,1 nm 偏置 + 10 pm 起伏,判据阈值 0.80):

        两条**完全独立**的噪声线   → 0.9999   **通过**
        完全反相(最坏的针尖行为)   → 0.9609   **通过**
        死平废帧(两个一样的平面)   → 1.0000   **通过**

    也就是说那道阈值是**装饰性的**:验证相在它最该失败的方向上不可能失败。
    减掉均值之后同一批分别是 0.013 / −1.0(死平那张由帧有效性守卫在更前面拦掉)。

    ## 为什么**允许横向平移**

    正反扫可能受压电迟滞影响而发生快轴偏移。若强制零平移，位置差会被误读成
    形貌不一致；因此在 max_shift_frac 限定的范围内寻找最佳对应关系。

    读不出信号(任一侧去趋势后能量为零)返回 ``0.0`` —— **那是「不相关」,不是
    「一致」**。而「这一帧根本不该被判」由
    :func:`mast.vision.frame_validity.judge_frame` 在更前面拦掉,不是这里的职责。
    """
    a0 = np.asarray(fwd, dtype=np.float64)
    b0 = np.asarray(bwd, dtype=np.float64)
    # 二维帧走**已经在 7,287 张真实扫描上验证过**的那一份(FFT 互相关 + 同样的
    # 平移窗口)。同一个概念不写第三份实现:一维用 np.correlate,二维转发给
    # ``_fwd_bwd_instability``,两条共用 ``_detrend`` 与 ``max_shift_frac``。
    if a0.ndim == 2 and b0.ndim == 2 and min(a0.shape) > 1 and min(b0.shape) > 1:
        h = min(a0.shape[0], b0.shape[0])
        w = min(a0.shape[1], b0.shape[1])
        return float(np.clip(
            1.0 - _fwd_bwd_instability(a0[:h, :w], b0[:h, :w],
                                       max_shift_frac=max_shift_frac),
            -1.0, 1.0))
    a = a0.ravel()
    b = b0.ravel()
    n = min(a.size, b.size)
    if n < 4:
        return 0.0
    a, b = a[:n], b[:n]
    # 去直流由 _detrend 完成 —— 它拟合的平面**带常数项**(``np.ones`` 那一列),
    # 所以残差按构造均值为零(实测:常数线与台阶线的残差均值恰好 0.0,
    # 噪声线 1e-19,相对 RMS 1.3e-8)。
    #
    # 这里原本还写着 ``a = a - a.mean()`` —— **一行读起来在承重、实际是 no-op 的
    # 代码**。变异验证当场逮到:删掉它,所有测试仍然绿。留着它比删掉更坏:
    # 下一个人会以为去直流靠的是它,于是换掉 _detrend 时不会意识到自己动了什么。
    a = _detrend(a.reshape(1, -1)).ravel()
    b = _detrend(b.reshape(1, -1)).ravel()
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    corr = np.correlate(a, b, mode="full") / (na * nb)
    lag0 = n - 1
    r = max(2, int(max_shift_frac * n))
    win = corr[max(0, lag0 - r):lag0 + r + 1]
    return float(np.clip(win.max() if win.size else corr[lag0], -1.0, 1.0))


__all__ = ["assess_tip_classical", "trace_retrace_correlation"]
