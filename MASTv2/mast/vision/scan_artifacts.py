"""Scan-artifact detection — network-free.

Three cheap, orthogonal checks an autonomous scanner wants on every frame:

  * feedback oscillation / ringing — periodic ripples COHERENT along a scan axis
    show up in the 2-D FFT as a strong peak ON a frequency axis (ky≈0 or kx≈0);
    a real 2-D lattice puts its peaks OFF the axes, so the two are separable.
  * thermal drift — trace and retrace of a drifting scan are spatially offset;
    the fwd↔bwd cross-correlation peak offset measures it (needs both channels).
  * bad scan-lines / spikes — rows whose statistics are robust outliers, and
    isolated pixel spikes (|x−local median| ≫ MAD).

See docs/v2/benchmarks/vision_v25_diagnostic/.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mast.vision.module import ScanArtifactsResult


def _split_fwd_bwd(image, bwd):
    a = np.asarray(image)
    if bwd is None and a.ndim == 3 and a.shape[0] == 2:
        return _to_2d(a[0]), _to_2d(a[1])
    if a.ndim == 3 and a.shape[0] in (1, 2):
        a = a[0]
    return _to_2d(a), (None if bwd is None else _to_2d(bwd))


def _to_2d(a: npt.NDArray) -> npt.NDArray[np.float32]:
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.mean(axis=-1)
    elif a.ndim != 2:
        raise ValueError(f"expected 2-D, got {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float32)


def _detrend_rows(h):
    return h - np.median(h, axis=1, keepdims=True)


def _oscillation(h):
    """(severity, cycles_per_line): ratio of the strongest ON-axis peak to the
    strongest OFF-axis peak. Feedback ringing is a coherent streak → a dominant
    on-axis peak (severity ≫ 1); an isotropic feature field or a 2-D lattice puts
    comparable/larger energy off-axis (severity ≲ 1), so neither false-triggers."""
    f = h - h.mean()
    F = np.fft.fftshift(np.abs(np.fft.fft2(f)))
    cy, cx = F.shape[0] // 2, F.shape[1] // 2
    yy, xx = np.ogrid[:F.shape[0], :F.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    on_axis = ((np.abs(yy - cy) <= 1) | (np.abs(xx - cx) <= 1)) & (rr > 3)
    off_axis = (~on_axis) & (rr > 3)
    if not on_axis.any() or not off_axis.any():
        return 0.0, None
    on_peak = float(F[on_axis].max())
    off_peak = float(F[off_axis].max()) + 1e-9
    severity = min(on_peak / off_peak, 1e4)
    idx = np.where(on_axis)
    j = int(np.argmax(F[on_axis]))
    py, px = idx[0][j], idx[1][j]
    cyc = float(max(abs(px - cx), abs(py - cy)))
    return float(severity), (cyc if cyc > 0 else None)


def _plane_detrend(h):
    """Remove a smooth tilt (global plane) but KEEP per-row offsets — so a spiked
    scan-line still stands out (per-row median subtraction would erase it)."""
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]].astype(np.float32)
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(h.size, np.float32)]
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    return (h.ravel() - A @ coef).reshape(h.shape).astype(np.float32)


def _drift_px(fwd, bwd):
    """fwd↔bwd registration offset via cross-correlation peak (thermal drift).
    Searched only within a central window (drift within one frame is small). Real
    drift makes the offset peak taller than the zero-lag value (features realign
    off-centre); on a periodic lattice a lattice-vector shift ties the zero-lag
    peak (shift-by-a-period == identity) → NOT drift, so we require the offset
    peak to clearly beat zero-lag before reporting it."""
    a = _detrend_rows(fwd); b = _detrend_rows(bwd)
    a = a / (a.std() + 1e-9); b = b / (b.std() + 1e-9)
    Fa = np.fft.fft2(a); Fb = np.fft.fft2(b)
    xc = np.fft.fftshift(np.real(np.fft.ifft2(Fa * np.conj(Fb))))
    cy, cx = xc.shape[0] // 2, xc.shape[1] // 2
    zero_lag = float(xc[cy, cx])
    rmax = max(4, int(0.15 * max(xc.shape)))
    yy, xx = np.ogrid[:xc.shape[0], :xc.shape[1]]
    win = np.hypot(yy - cy, xx - cx) <= rmax
    masked = np.where(win, xc, -np.inf)
    py, px = np.unravel_index(int(np.argmax(masked)), xc.shape)
    peak = float(xc[py, px])
    # offset peak must beat zero-lag by a margin — else it's a lattice alias, not drift
    if peak <= zero_lag * 1.08:
        return 0.0
    return float(np.hypot(py - cy, px - cx))


def _bad_rows(h):
    """h should be plane-detrended (tilt removed, row offsets kept)."""
    rm = h.mean(axis=1); rs = h.std(axis=1)
    frac = 0.0
    for v in (rm, rs):
        med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-9
        frac = max(frac, float((np.abs(v - med) > 6.0 * mad).mean()))
    return frac


def _spike_frac(h):
    """Fraction of ISOLATED single-pixel outliers (glitches). Real features
    (adsorbates, defects) are multi-pixel blobs, so components larger than 2 px
    are excluded — only true point spikes count."""
    from scipy import ndimage as ndi
    resid = np.abs(h - ndi.median_filter(h, size=3))
    mad = np.median(resid) + 1e-9
    hot = resid > 8.0 * mad
    if not hot.any():
        return 0.0
    lbl, n = ndi.label(hot)
    sizes = np.bincount(lbl.ravel())
    isolated = np.isin(lbl, np.where(sizes <= 2)[0]) & hot   # ≤2-px specks only
    return float(isolated.sum()) / h.size


def detect_scan_artifacts(
    image: npt.NDArray, bwd=None, oscillation_threshold: float = 3.0
) -> ScanArtifactsResult:
    """Detect feedback oscillation, drift (needs trace+retrace) and bad rows/spikes."""
    fwd, bwd2 = _split_fwd_bwd(image, bwd)
    h = _detrend_rows(fwd)
    if float(h.std()) < 1e-9:
        return ScanArtifactsResult(has_artifact=False)

    sev, cyc = _oscillation(h)
    osc = bool(sev > oscillation_threshold)
    drift = _drift_px(fwd, bwd2) if (bwd2 is not None and bwd2.shape == fwd.shape) else None
    bad_row = _bad_rows(_plane_detrend(fwd))     # tilt removed, row offsets kept
    spike = _spike_frac(h / (h.std() + 1e-9))

    has = bool(osc or bad_row > 0.05 or spike > 0.02 or (drift is not None and drift > 3.0))
    return ScanArtifactsResult(
        has_artifact=has,
        oscillation=osc,
        oscillation_severity=float(sev),
        oscillation_cycles_per_line=cyc,
        drift_px=drift,
        bad_row_frac=float(bad_row),
        spike_frac=float(spike),
    )


__all__ = ["detect_scan_artifacts"]
