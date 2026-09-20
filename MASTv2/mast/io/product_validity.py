"""Product-validity gate for composite skills + analysis tools.

A composite that "ran every step / saved a file" is NOT the same as a composite
that produced a USABLE product. In one real case, a scan finished
(``timed_out: False``) and was saved, yet the frame was all-NaN / crashed
(fft 0.0, rms NaN) and the run still reported success. This module gives the
composite layer — and any analysis tool — a cheap, positive-evidence-only check
on the actual array, so "跑完/存盘" can be downgraded to a *degraded* result when
the product is garbage.

Pure numpy — no hardware, no checkpoint state. Thresholds are deliberately
conservative: an array is only ever called INVALID on unmistakable evidence
(empty / all-NaN / dead-flat), never for a merely noisy-but-real scan. That
asymmetry is the point — a false "degraded" costs the operator a real result, so
the gate stays silent unless the product is plainly unusable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# An array is treated as "all NaN" once at least this fraction of its samples are
# non-finite. Not exactly 1.0 so a stray finite pixel in an otherwise-dead frame
# (a hot pixel on a crashed scan) still reads as invalid.
_NAN_FRACTION_INVALID = 0.999


def assess_array(arr) -> dict:
    """Judge whether a numeric array is a usable product.

    Returns a plain dict ``{valid, reason, size, nan_fraction, variance,
    constant}``. ``valid`` is False only for: an empty array, an all-NaN/inf
    array, or a dead-flat array (zero peak-to-peak over >1 sample — a tip crash
    or open feedback loop). Everything else is valid.
    """
    a = np.asarray(arr)
    size = int(a.size)
    if size == 0:
        return {"valid": False, "reason": "empty array (0 samples)", "size": 0,
                "nan_fraction": 1.0, "variance": 0.0, "constant": True}
    af = a.astype(np.float64, copy=False)
    finite_mask = np.isfinite(af)
    n_finite = int(np.count_nonzero(finite_mask))
    nan_fraction = 1.0 - (n_finite / size)
    if n_finite == 0:
        return {"valid": False, "reason": "all samples are NaN/inf", "size": size,
                "nan_fraction": 1.0, "variance": 0.0, "constant": True}
    finite = af[finite_mask]
    variance = float(np.var(finite))
    ptp = float(np.ptp(finite))
    constant = ptp <= 0.0
    if nan_fraction >= _NAN_FRACTION_INVALID:
        return {"valid": False,
                "reason": f"{nan_fraction:.1%} of samples are NaN/inf",
                "size": size, "nan_fraction": nan_fraction,
                "variance": variance, "constant": constant}
    if constant and size > 1:
        return {"valid": False,
                "reason": ("dead-flat product: zero variance across "
                           f"{size} samples (tip crashed / feedback open?)"),
                "size": size, "nan_fraction": nan_fraction,
                "variance": 0.0, "constant": True}
    return {"valid": True, "reason": "", "size": size,
            "nan_fraction": nan_fraction, "variance": variance, "constant": False}


# Extensions whose bytes we can load + assess. A product path with any other
# extension (a .png thumbnail, a .json sidecar) is not a scan array and is left
# for the caller to judge.
LOADABLE_PRODUCT_EXT = (".sxm", ".npy", ".npz", ".sm4", ".dat", ".txt", ".csv", ".3ds")


def assess_scan_file(path) -> dict:
    """Load a saved scan through the canonical loader and assess its image.

    Returns :func:`assess_array`'s verdict plus ``{path, loaded}``. A file that
    is missing or unreadable is reported ``valid: False`` with ``loaded: False``
    — a product that "was saved" but cannot be opened is exactly the silent
    failure this gate exists to surface.
    """
    p = Path(str(path))
    if not p.exists():
        return {"valid": False, "reason": f"product file does not exist: {p}",
                "path": str(p), "loaded": False}
    try:
        from mast.data import load_image_2d
        arr = load_image_2d(p)
    except Exception as exc:  # noqa: BLE001 — unreadable product = invalid product
        return {"valid": False,
                "reason": f"product unreadable: {type(exc).__name__}: {exc}",
                "path": str(p), "loaded": False}
    out = assess_array(arr)
    out["path"] = str(p)
    out["loaded"] = True
    return out


__all__ = ["assess_array", "assess_scan_file", "LOADABLE_PRODUCT_EXT"]
