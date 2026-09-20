"""Image processing functions for STM topography data."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from skimage.registration import phase_cross_correlation


def plane_subtract(image: np.ndarray) -> np.ndarray:
    """Subtract least-squares fitted plane (z = ax + by + c) from 2D image."""
    ny, nx = image.shape
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(nx * ny)])
    coeffs, *_ = np.linalg.lstsq(A, image.ravel(), rcond=None)
    plane = (coeffs[0] * X + coeffs[1] * Y + coeffs[2])
    return image - plane


def line_by_line_level(image: np.ndarray, method: str = "median") -> np.ndarray:
    """Line-by-line leveling.

    Methods: 'median' — subtract row median, 'mean' — subtract row mean,
    'poly1' — subtract linear fit per row, 'poly2' — subtract quadratic fit per row.
    """
    result = image.astype(np.float64, copy=True)
    ny, nx = result.shape

    if method == "median":
        result -= np.median(result, axis=1, keepdims=True)
    elif method == "mean":
        result -= np.mean(result, axis=1, keepdims=True)
    elif method in ("poly1", "poly2"):
        x = np.arange(nx, dtype=np.float64)
        deg = 1 if method == "poly1" else 2
        for i in range(ny):
            coeffs = np.polyfit(x, result[i], deg)
            result[i] -= np.polyval(coeffs, x)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return result


def fft2d(image: np.ndarray) -> np.ndarray:
    """Compute 2D FFT magnitude (log scale, DC-shifted to center)."""
    ft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.abs(ft)
    magnitude[magnitude == 0] = 1e-20
    return np.log(magnitude)


def fft_filter(
    image: np.ndarray,
    filter_type: str = "low",
    cutoff: float = 0.5,
) -> np.ndarray:
    """Apply frequency-domain filter.

    filter_type: 'low', 'high', or 'band'.
    cutoff: fraction of Nyquist (scalar for low/high, (low, high) tuple for band).
    """
    ny, nx = image.shape
    ft = np.fft.fftshift(np.fft.fft2(image))

    cy, cx = ny // 2, nx // 2
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = min(cx, cy)

    if filter_type == "low":
        mask = (r <= cutoff * r_max).astype(np.float64)
    elif filter_type == "high":
        mask = (r >= cutoff * r_max).astype(np.float64)
    elif filter_type == "band":
        lo, hi = cutoff  # type: ignore[misc]
        mask = ((r >= lo * r_max) & (r <= hi * r_max)).astype(np.float64)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type!r}")

    filtered = np.fft.ifft2(np.fft.ifftshift(ft * mask))
    return np.real(filtered)


def drift_estimate(image1: np.ndarray, image2: np.ndarray) -> tuple[float, float]:
    """Estimate drift between two images using phase cross-correlation.

    Returns (dy, dx) shift in pixels.

    ⚠️ **Not usable on STM topography as it stands** (found 2026-09-03, behaviour
    left unchanged because that is a deliberate call, not a drive-by edit).

    It takes ``phase_cross_correlation``'s default ``normalization='phase'``,
    which divides every frequency by its own magnitude. On a broadband image
    that is the point; on a **band-limited** one — and topography above the
    atomic scale is smooth, so it is band-limited — the near-empty high
    frequencies get amplified into pure numerical noise, which then dominates.
    Measured on a 128² random field shifted by exactly (5, −3) px::

        gaussian blur σ=0 px    phase (−5.0, +3.0)    normalization=None (−5.0, +3.0)
        gaussian blur σ=3 px    phase (+0.1, −0.1)    normalization=None (−4.7, +2.7)
        gaussian blur σ=10 px   phase ( 0.0,  0.0)    normalization=None (−3.2, +0.1)

    So on a smooth frame it returns a confident **zero shift** — indistinguishable
    from "no drift". Pass ``normalization=None`` for that data;
    :func:`mast.vision.frame_drift.pair_displacement` does, and it is the only
    production caller of this function today.
    """
    shift, _, _ = phase_cross_correlation(image1, image2, upsample_factor=10)
    return (float(shift[0]), float(shift[1]))
