"""Head C Level 0 — classical terrace detector.

Pure scipy + numpy. No torch, no GPU, no learnable parameters. The VIGIL
architecture document (§3.1) explicitly elects this route for L0 because:

  * "terrace vs not-terrace" is geometric, not semantic — local height
    variance + connectivity already separates the two without any training
    set.
  * Classical methods are surface-agnostic from day one (Barker et al.
    2024 ACS Nano on deterministic SPM tip-state classification).
  * Wall-time on CPU is < 15 ms / 512² so the L0 mask is cheap enough to
    serve as a geometric prior for every L1/L2 inference downstream.

The output mask is a sidecar — it does NOT get written into the v0.4 mask
codec's low byte. ``mask_codec.write_surface()`` enforces that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def detect_terrace(
    image_pm: npt.NDArray[np.float32],
    *,
    window: int = 9,
    var_percentile: float = 30.0,
    min_area_frac: float = 0.005,
    closing_iters: int = 2,
) -> npt.NDArray[np.bool_]:
    """Return a bool mask, True where the surface is locally flat (terrace).

    Args:
        image_pm:        2D float32 STM topography in picometers.
        window:          square-window side for local-variance estimation.
        var_percentile:  pixels below this percentile of variance count as
                         terrace candidates.
        min_area_frac:   minimum component size (fraction of total pixels)
                         to keep after thresholding.
        closing_iters:   binary closing iterations to fill in atomic-scale
                         roughness within terraces.

    Returns:
        bool mask of the same shape, True = terrace.
    """
    if image_pm.ndim != 2:
        raise ValueError(f"detect_terrace expects 2D, got {image_pm.shape}")
    if image_pm.dtype != np.float32:
        # accept other floats but warn the caller — VIGIL HDF5 shards are float32
        image_pm = image_pm.astype(np.float32)

    # Lazy scipy / skimage imports keep this module importable in environments
    # without the SciPy stack installed (CI smoke runs).
    from scipy import ndimage  # noqa: PLC0415

    H, W = image_pm.shape

    # Robust z-score (MAD ≈ 1.4826 σ for Gaussian noise) — unit-agnostic
    med = float(np.median(image_pm))
    mad = float(np.median(np.abs(image_pm - med))) * 1.4826 + 1e-6
    z = (image_pm - med) / mad

    # Local variance via box filter — O(N) regardless of window size
    mean = ndimage.uniform_filter(z, size=window)
    sq = ndimage.uniform_filter(z * z, size=window)
    var = np.maximum(sq - mean * mean, 0.0)

    thr = float(np.percentile(var, var_percentile))
    candidate = var < thr

    # Reject components smaller than min_area_frac of the image
    min_area = max(1, int(min_area_frac * H * W))
    labels, n = ndimage.label(candidate)
    if n > 0:
        sizes = ndimage.sum(candidate, labels, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_area
        candidate = keep[labels]

    # Fill atomic-scale roughness inside terraces
    if closing_iters > 0:
        candidate = ndimage.binary_closing(candidate, iterations=int(closing_iters))

    return candidate.astype(bool)


def terrace_iou(pred: npt.NDArray[np.bool_], gt: npt.NDArray[np.bool_]) -> float:
    """Binary IoU between two bool masks. Returns 0.0 if both are empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 0.0
    return inter / union


__all__ = ["detect_terrace", "terrace_iou"]
