"""Image quality metrics for STM topography."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import convolve2d, find_peaks

from .processors import plane_subtract


def fft_quality_score(image: np.ndarray) -> float:
    """原子分辨质量分（0..1），越大越好，可作为闭环优化的目标函数。

    使用原子带上功率谱的角向集中度：按角度分 bin，取最大 bin / 中位 bin。
    离散布拉格点的方向集中度高，弥散环较低。径向环平均会抹掉六角方向信息，
    而径向峰强度的自适应归一化可能把纯噪声的局部极大抬高，因此不采用它们。

    集中度通过 log10 映射到 0..1：1 对应 0，10⁴ 对应 1；对数映射保留
    跨数量级变化的分辨力。具体有效性仍由尺度门及调用方的适用条件约束。
    """
    from mast.vision.atomic_phase import angular_concentration

    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16:
        return 0.0
    finite = np.isfinite(h)
    if not finite.any():
        return 0.0
    h = np.where(finite, h, float(np.mean(h[finite])))
    h = plane_subtract(h)

    # **先定主周期，再在那一个周期上算集中度。**
    #
    # 第一版是在一整段周期上扫、取集中度最大的那个 —— 那等于对噪声取上包络：
    # 合成的带通抖动因此也被推到满分 1.000，与真晶格完全分不开。取最大值把
    # 一个统计量变成了极值统计，而极值对噪声的敏感度高得多。
    n = min(h.shape)
    win = np.outer(np.hanning(h.shape[0]), np.hanning(h.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(h * win))) ** 2
    cy, cx = h.shape[0] // 2, h.shape[1] // 2
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    rq = rr.astype(int)
    lo = max(3, int(n * 0.02))          # 掐掉 DC 与最低频的背景山
    hi = int(min(cx, cy))
    if hi <= lo + 2:
        return 0.0
    prof = np.bincount(rq.ravel(), weights=F.ravel(),
                       minlength=hi + 1)[:hi + 1].astype(float)
    cnt = np.bincount(rq.ravel(), minlength=hi + 1)[:hi + 1].astype(float)
    prof = np.divide(prof, np.maximum(cnt, 1.0))
    r_pk = int(np.argmax(prof[lo:hi]) + lo)
    if r_pk <= 0:
        return 0.0
    period_px = float(n) / r_pk
    if not (3.0 <= period_px <= n / 3.0):
        return 0.0
    try:
        best = float(angular_concentration(h, period_px))
    except Exception:  # noqa: BLE001 — 目标函数不该因为一个坏周期就炸
        return 0.0
    if not np.isfinite(best) or best <= 1.0:
        return 0.0
    # 1 → 0，1e4 → 1。跨四个数量级，线性映射会让真晶格全挤在顶端。
    return float(np.clip(np.log10(best) / 4.0, 0.0, 1.0))


def rms_roughness(image: np.ndarray) -> float:
    """RMS surface roughness after plane subtraction."""
    leveled = plane_subtract(image)
    return float(np.sqrt(np.mean(leveled ** 2)))


def noise_estimate(image: np.ndarray) -> float:
    """Estimate noise level using median absolute deviation of Laplacian.

    Uses the robust MAD estimator on the Laplacian-filtered image,
    which responds primarily to high-frequency noise.
    """
    # Discrete Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    laplacian = convolve2d(image.astype(np.float64), kernel, mode="valid")
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    # MAD to sigma conversion (for Gaussian noise)
    sigma = mad * 1.4826
    return float(sigma)
