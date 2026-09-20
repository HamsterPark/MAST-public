"""Classical, network-free STM segmentation — terrace / step / defect / contam.

:func:`segment_classical` now delegates to the **scale-adaptive** segmenter
(:mod:`mast.vision.seg_scale_adaptive`, adopted 2026-07-27), which on VIGIL
physics ground truth beats the Bragg-subtraction pipeline kept below across
every decision-oriented metric (C1 defects: presence F1 0.874 vs 0.817,
object F1 0.106 vs 0.063, count MAE 4.0 vs 9.1). The Bragg pipeline remains
as ``_segment_bragg_legacy`` — the fail-safe fallback and a comparison
baseline. Both never hallucinate "contamination" across clean atomic lattices
the way the deployed learned C head does
(``docs/v2/benchmarks/vision_v25_diagnostic/``).

Use :func:`mast.vision.seg_scale_adaptive.summarize_segmentation` for the
presence / object-count / area-fraction readout that autonomy should consume —
pixel-level masks of sparse targets are the wrong lens (measured: a usable
detector reads pixel IoU 0.000 but presence F1 0.817).

Class order matches :data:`VIGILBackend._L1_CLASSES` so the mask codes are
interchangeable with the learned Head-C output.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# 0=terrace 1=step 2=defect 3=contamination — identical order to Head-C L1.
CLASSES: list[str] = ["TERRACE", "STEP", "DEFECT", "CONTAMINATION"]


def _to_2d(image: npt.NDArray) -> npt.NDArray[np.float32]:
    """Adapt a VisionModule image to a single 2-D height map (float32).

    Accepts (H,W) · (2,H,W) trace/retrace (→ forward) · (1,H,W) · (H,W,3) (→ mean).
    """
    a = np.asarray(image)
    if a.ndim == 3:
        if a.shape[0] in (1, 2):
            a = a[0]
        elif a.shape[-1] == 3:
            a = a.mean(axis=-1)
        else:
            raise ValueError(f"cannot interpret 3-D image of shape {a.shape}")
    elif a.ndim != 2:
        raise ValueError(f"image must be 2-D or 3-D, got {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float32)


def _detrend(h: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    h = h - np.median(h, axis=1, keepdims=True)          # per-row line offset
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]].astype(np.float32)
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(h.size, np.float32)]
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)  # global plane
    return (h.ravel() - A @ coef).reshape(h.shape).astype(np.float32)


def _remove_small(mask: npt.NDArray[np.bool_], min_px: int) -> npt.NDArray[np.bool_]:
    """Drop connected components smaller than ``min_px`` px (scipy — stable API,
    avoids the skimage.morphology.remove_small_objects param churn)."""
    from scipy import ndimage as ndi

    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lbl]


def _bragg_lattice(hn: npt.NDArray[np.float32]) -> npt.NDArray[np.float32] | None:
    """Reconstruct the periodic (atomic-lattice) part from its sharp FFT peaks.

    Clean lattice → reconstruction ≈ image (residual ~0); defects/adsorbates are
    aperiodic → survive in the residual. Returns None when not clearly periodic.
    """
    F = np.fft.fft2(hn)
    mag = np.abs(F).copy()
    mag[0, 0] = 0.0
    keep = mag >= (mag.mean() + 4.0 * mag.std())
    if int(keep.sum()) < 6:
        return None
    keep[0, 0] = True  # keep the DC / mean term
    return np.real(np.fft.ifft2(F * keep)).astype(np.float32)


def segment_classical(
    image: npt.NDArray,
    nm_per_px: float | None = None,
) -> npt.NDArray[np.uint8]:
    """Return a 4-class label map (0=terrace 1=step 2=defect 3=contam).

    Delegates to the validated scale-adaptive segmenter; the Bragg pipeline is
    the fail-safe fallback. ``nm_per_px`` (scan_size_nm / pixels) physicalises
    every kernel/gate; when unknown a neutral default is used (structure
    detection degrades gracefully)."""
    try:
        from mast.vision.seg_scale_adaptive import segment_scale_adaptive
        seg, _info = segment_scale_adaptive(_to_2d(image), nm_per_px=nm_per_px)
        return seg.astype(np.uint8)
    except Exception as exc:  # noqa: BLE001 — segmentation must never crash a monitor
        logger.warning("scale-adaptive segmentation failed (%s); Bragg fallback", exc)
        return _segment_bragg_legacy(image, nm_per_px)


def _segment_bragg_legacy(
    image: npt.NDArray,
    nm_per_px: float | None = None,
) -> npt.NDArray[np.uint8]:
    """The pre-2026-07-27 Bragg-subtraction segmenter (fallback / baseline)."""
    from scipy import ndimage as ndi
    from skimage.filters import sobel
    from skimage.morphology import skeletonize

    h = _detrend(_to_2d(image))
    std = float(h.std())
    if std < 1e-9:                       # flat image → all terrace
        return np.zeros(h.shape, np.uint8)
    hn = h / std

    # atomic spacing in px (~0.30 nm); clamp so meso (C2) images stay sane
    nmpp = float(nm_per_px) if (nm_per_px and nm_per_px > 0) else 0.05
    lat_px = float(np.clip(0.30 / nmpp, 2.0, 10.0))
    coarse = ndi.gaussian_filter(hn, float(np.clip(1.6 * lat_px, 3.0, 18.0)))

    seg = np.zeros(h.shape, np.uint8)

    # STEP — gradient ridges of the coarse (atom-free) topography
    g = sobel(coarse)
    step = _remove_small(g > np.percentile(g, 97.0), 30)
    if step.any():
        step = ndi.binary_dilation(skeletonize(step))     # thin ridge → 1-px band

    # DEFECT — aperiodic residual (Bragg subtraction; meso → high-pass)
    lat = _bragg_lattice(hn)
    resid = hn - lat if lat is not None else hn - coarse
    ra = np.abs(resid)
    med = float(np.median(ra))
    mad = float(np.median(np.abs(ra - med))) + 1e-6
    defect = _remove_small(ra > (med + 6.0 * mad), 3)

    # CONTAM — extended disordered patches (residual energy blobs, not points)
    le = ndi.uniform_filter(ra, 9)
    contam = _remove_small((le > np.percentile(le, 98.5)) & ~step, 80)

    # FFT lattice subtraction rings at the non-periodic image border — clear a
    # thin frame of spurious defect/contam (real steps at the border are kept).
    bw = 4
    frame = np.zeros(h.shape, bool)
    frame[:bw] = frame[-bw:] = frame[:, :bw] = frame[:, -bw:] = True
    defect &= ~frame
    contam &= ~frame

    seg[contam] = 3
    seg[defect] = 2
    seg[step] = 1
    return seg


def class_counts(seg: npt.NDArray[np.uint8]) -> dict[str, int]:
    """{class_name: pixel_count} for a 4-class label map."""
    return {name: int((seg == i).sum()) for i, name in enumerate(CLASSES)}


def segment_terraces(image, nm_per_px: float | None = None, max_levels: int = 5) -> npt.NDArray[np.uint8]:
    """Label the distinct terrace LEVELS (0,1,2,…) by height — a per-terrace map
    for navigation, not just terrace-vs-not. Multi-Otsu on the atom-blurred
    height; a flat frame → all level 0."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_multiotsu

    h = _detrend(_to_2d(image))
    std = float(h.std())
    if std < 1e-9:
        return np.zeros(h.shape, np.uint8)
    hn = h / std
    nmpp = float(nm_per_px) if (nm_per_px and nm_per_px > 0) else 0.05
    lat_px = float(np.clip(0.30 / nmpp, 2.0, 10.0))
    # median filter preserves step edges (terraces stay sharp) while removing the
    # atomic corrugation — unlike a heavy gaussian, which merges the level peaks.
    coarse = ndi.median_filter(hn, size=int(round(lat_px)) * 2 + 1)

    # estimate the true number of terrace levels from the height histogram, so we
    # don't force multi-Otsu to split a flat/2-level frame into spurious bands.
    from scipy.signal import find_peaks
    hist, _ = np.histogram(coarse, bins=64)
    hist = ndi.gaussian_filter1d(hist.astype(np.float32), 1.0)
    # pad both ends so terrace levels sitting at the histogram EDGES (the lowest /
    # highest plateau) register as peaks (find_peaks ignores boundary maxima).
    padded = np.concatenate([[0.0], hist, [0.0]])
    peaks, _ = find_peaks(padded, prominence=0.08 * hist.max(), distance=3)
    k = int(np.clip(len(peaks), 1, max_levels))
    if k < 2:
        return np.zeros(h.shape, np.uint8)
    try:
        thr = threshold_multiotsu(coarse, classes=k)
        return np.digitize(coarse, thr).astype(np.uint8)
    except Exception:
        return np.zeros(h.shape, np.uint8)


def detect_defects(image, nm_per_px: float | None = None) -> tuple[npt.NDArray[np.bool_], int]:
    """Locate aperiodic point defects / adsorbates by FFT lattice subtraction
    (the periodic lattice cancels; anomalies survive). Returns (defect_mask,
    n_defects). On a non-periodic frame, falls back to a high-pass residual."""
    from scipy import ndimage as ndi

    h = _detrend(_to_2d(image))
    std = float(h.std())
    if std < 1e-9:
        return np.zeros(h.shape, bool), 0
    hn = h / std
    lat = _bragg_lattice(hn)
    resid = hn - lat if lat is not None else hn - ndi.gaussian_filter(hn, 3.0)
    ra = np.abs(resid)
    med = float(np.median(ra))
    mad = float(np.median(np.abs(ra - med))) + 1e-6
    mask = _remove_small(ra > (med + 6.0 * mad), 2)
    bw = 4
    mask[:bw] = mask[-bw:] = mask[:, :bw] = mask[:, -bw:] = False   # kill FFT ring
    _, n = ndi.label(mask)
    return mask, int(n)


def segment_classical_result(image, nm_per_px: float | None = None, level: int = 0):
    """Classical 4-class segmentation packaged as a :class:`SegmentationResult`
    (shared by the backend's level-0 / level-1-fallback and VisionModule)."""
    from mast.vision.module import SegmentationResult
    from mast.vision.seg_utils import encode_rle

    seg = segment_classical(image, nm_per_px).astype(np.uint8)
    return SegmentationResult(
        mask_rle=encode_rle(seg),
        shape=tuple(seg.shape),
        class_counts=class_counts(seg),
        level=level,  # type: ignore[arg-type]
        classes=list(CLASSES),
        tipflag_stability_rle=b"",
        tipflag_transition_rle=b"",
    )


__all__ = ["segment_classical", "segment_classical_result", "segment_terraces",
           "detect_defects", "class_counts", "CLASSES", "_segment_bragg_legacy"]
