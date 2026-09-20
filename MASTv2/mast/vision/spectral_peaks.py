"""Peaks in a one-dimensional spectrum, with a noise scale the data itself sets.

A confined electronic state shows up as a peak in dI/dV — the quantum-corral resonances, an
adsorbate's own level, a vibrational threshold. Finding them means deciding what counts as a
peak, and the honest way to do that is to measure the noise on this spectrum rather than to
carry a threshold in from somewhere else. The scale comes from the second difference of the
raw trace (``var(Δ²y) = 6σ²`` for independent noise), and the bar a candidate has to clear is
``σ·√(2 ln n)`` — the largest excursion noise alone would produce *somewhere* in a trace this
long. A flat multiple of σ finds a "peak" in every long stretch of pure noise.

Each surviving peak is refined by a Lorentzian on a linear background, which gives a position
better than the sample spacing and a width that means something. When that fit does not
converge the peak keeps its three-point parabolic position and says so.

Pure functions, explicit thresholds, no exceptions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

#: floor on the prominence bar, in noise sigmas (the √(2 ln n) term usually sets it)
DEFAULT_PROMINENCE_SIGMA = 3.0
#: a "peak" wider than this is a background feature, not a level
DEFAULT_MAX_FWHM_V = 0.30
#: fewer points than this and neither the noise nor a width is measurable
MIN_POINTS = 15


@dataclass(frozen=True)
class Peak:
    energy_ev: float
    energy_err_ev: float | None
    fwhm_ev: float | None
    amplitude: float
    prominence_sigma: float
    fit: str                       # lorentzian | parabolic | grid
    fit_r2: float | None = None


@dataclass(frozen=True)
class PeaksResult:
    verdict: str                   # peaks / none / undecidable
    peaks: tuple[Peak, ...] = ()
    noise_sigma: float = 0.0
    energy_range_ev: tuple[float, float] | None = None
    n_points: int = 0
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "peaks"

    @property
    def energies_mev(self) -> tuple[float, ...]:
        return tuple(p.energy_ev * 1e3 for p in self.peaks)


def _savgol(y: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    from scipy.signal import savgol_filter

    # 0 = choose one: a five-point cubic nearly interpolates its own points and smooths nothing
    w = int(window) if window else max(5, y.size // 40)
    if w % 2 == 0:
        w += 1
    w = max(5, min(w, (y.size // 2) * 2 - 1))
    if w < 5 or w >= y.size:
        return y.copy()
    return savgol_filter(y, w, 3)


def _lorentzian(x, amp, x0, gamma, c, slope):
    return c + slope * (x - x0) + amp * (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)


def find_peaks_1d(x: npt.ArrayLike, y: npt.ArrayLike, *,
                  prominence_sigma: float = DEFAULT_PROMINENCE_SIGMA,
                  max_peaks: int = 8, smooth_points: int = 0,
                  max_fwhm_ev: float = DEFAULT_MAX_FWHM_V,
                  polarity: str = "positive",
                  refine: str = "lorentzian") -> PeaksResult:
    """Peaks of ``y(x)``, sorted by ``x``, with the noise scale taken from the data."""
    from scipy import optimize, signal

    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size:
        return PeaksResult("undecidable", reasons=("shape_mismatch",))
    good = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[good], ya[good]
    if xa.size < MIN_POINTS:
        return PeaksResult("undecidable", n_points=int(xa.size), reasons=("too_few_points",))
    order = np.argsort(xa)
    xa, ya = xa[order], ya[order]
    rng = (float(xa[0]), float(xa[-1]))
    if polarity == "negative":
        ya = -ya
    smooth = _savgol(ya, smooth_points)
    # the noise scale comes from the second difference of the RAW trace, not from the residual
    # of the smoother: a short Savitzky–Golay window nearly interpolates its own points, so the
    # residual underestimates the noise and every wiggle becomes a peak. For iid noise,
    # var(Δ²y) = 6σ².
    d2 = np.diff(ya, 2)
    sigma = 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / math.sqrt(6.0)
    if not (sigma > 0):
        return PeaksResult("undecidable", n_points=int(xa.size), energy_range_ev=rng,
                           reasons=("no_noise_scale",),
                           notes={"hint": "a spectrum with no scatter has no detectable peaks"})
    # the bar is the largest excursion noise alone would produce somewhere in a trace this
    # long, σ·√(2 ln n) — using a flat 3σ finds a "peak" in every long stretch of pure noise
    bar = sigma * max(float(prominence_sigma), math.sqrt(2.0 * math.log(max(xa.size, 3))))
    idx, props = signal.find_peaks(smooth, prominence=bar, width=0)
    if idx.size == 0:
        both = ("peaks" if polarity == "both" else "none")
        return PeaksResult("none" if both == "none" else "none", noise_sigma=sigma,
                           n_points=int(xa.size), energy_range_ev=rng)
    peaks: list[Peak] = []
    dx = float(np.median(np.diff(xa)))
    for n_peak, (i, prom) in enumerate(zip(idx, props["prominences"])):
        # fit each peak over its OWN half-width, cut at the midpoint to either neighbour:
        # the prominence bases of two overlapping peaks span both, and a Lorentzian fitted
        # over that window merges them into one broad feature that is not there
        half = max(3, int(round(props["widths"][n_peak])))
        lo = max(0, i - 2 * half)
        hi = min(xa.size, i + 2 * half + 1)
        if n_peak > 0:
            lo = max(lo, (i + int(idx[n_peak - 1])) // 2)
        if n_peak + 1 < idx.size:
            hi = min(hi, (i + int(idx[n_peak + 1])) // 2 + 1)
        e0, err, fwhm, fit, r2 = float(xa[i]), None, None, "grid", None
        if refine in ("lorentzian", "parabolic") and hi - lo >= 7:
            if refine == "lorentzian":
                p0 = [float(prom), float(xa[i]), max(4 * dx, 0.01), float(np.min(ya[lo:hi])), 0.0]
                try:
                    popt, pcov = optimize.curve_fit(_lorentzian, xa[lo:hi], ya[lo:hi], p0=p0,
                                                    maxfev=8000)
                    e0, fwhm, fit = float(popt[1]), abs(float(popt[2])), "lorentzian"
                    if np.all(np.isfinite(pcov)):
                        err = float(np.sqrt(abs(pcov[1, 1])))
                    pred = _lorentzian(xa[lo:hi], *popt)
                    ss = float(((ya[lo:hi] - pred) ** 2).sum())
                    tot = float(((ya[lo:hi] - ya[lo:hi].mean()) ** 2).sum())
                    r2 = 1.0 - ss / tot if tot > 0 else None
                except (RuntimeError, ValueError):
                    refine_here = "parabolic"
                else:
                    refine_here = None
            else:
                refine_here = "parabolic"
            if refine_here == "parabolic" and 0 < i < xa.size - 1:
                a, b, c = smooth[i - 1], smooth[i], smooth[i + 1]
                den = a - 2 * b + c
                shift = 0.5 * (a - c) / den if abs(den) > 1e-30 else 0.0
                e0, fit = float(xa[i] + shift * dx), "parabolic"
        if fwhm is not None and fwhm > float(max_fwhm_ev):
            continue
        peaks.append(Peak(energy_ev=e0, energy_err_ev=err, fwhm_ev=fwhm,
                          amplitude=float(smooth[i]), prominence_sigma=float(prom / sigma),
                          fit=fit, fit_r2=r2))
    if not peaks:
        return PeaksResult("none", noise_sigma=sigma, n_points=int(xa.size), energy_range_ev=rng,
                           warnings=("all_candidates_too_broad",))
    peaks.sort(key=lambda p: p.energy_ev)
    # two candidates closer than half the NARROWER linewidth are one peak the refinement split
    # in two. Using the wider one would swallow a genuine neighbour sitting on its shoulder,
    # which is exactly the pair a corral spectrum has.
    merged: list[Peak] = []
    for p in peaks:
        if merged:
            gap = p.energy_ev - merged[-1].energy_ev
            widths = [w for w in (p.fwhm_ev, merged[-1].fwhm_ev) if w]
            width = max(0.5 * min(widths) if widths else 0.0, 3 * dx)
            if gap < width:
                if p.prominence_sigma > merged[-1].prominence_sigma:
                    merged[-1] = p
                continue
        merged.append(p)
    return PeaksResult("peaks", peaks=tuple(merged[: int(max_peaks)]), noise_sigma=sigma,
                       n_points=int(xa.size), energy_range_ev=rng)
