"""Surface-state dispersion from standing-wave spectroscopy.

Hasegawa & Avouris (PRL 71, 1071) and Crommie, Lutz & Eigler (Nature 363, 524) both did the
same thing in 1993: take dI/dV at a series of distances from a step or an adatom, watch the
ripple wavelength change with energy, and read the two-dimensional free-electron dispersion
off it. This module is that reading.

For each energy:

1. normalise every spectrum by the one furthest from the scatterer, so what is left is the
   interference term rather than the tip's and the surface's own spectral shape;
2. take the dI/dV as a function of distance at that energy;
3. get a starting wavevector from the dominant period of that trace, then fit the physical
   model — ``c − r·J₀(2kd)·e^{−2d/L}`` at a step, a decaying cosine around a point scatterer —
   for ``k(E)``.

Then fit ``E = E₀ + ħ²k²/2m*``: the intercept is the band bottom, the slope gives the
effective mass. ``ħ²/2mₑ = 0.0381 eV·nm²``.

Pure functions. Every threshold is an argument; nothing is read from disk or from config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

#: ħ²/2mₑ in eV·nm², the constant that turns a wavevector into an energy
HBAR2_OVER_2M_EV_NM2 = 0.0381
#: the closest a spectrum may be to the scatterer and still be described by the far field
DEFAULT_MIN_DISTANCE_NM = 1.0
#: below this many energies with a usable wavevector there is no dispersion to fit
MIN_ENERGIES = 4
#: a straight line in k² this poor is not a dispersion
MIN_R2 = 0.80


@dataclass(frozen=True)
class DispersionResult:
    """dispersion / no_standing_wave / undecidable, plus what was fitted."""

    verdict: str
    e0_ev: float | None = None
    e0_err_ev: float | None = None
    m_eff: float | None = None
    m_eff_err: float | None = None
    r2: float | None = None
    n_spectra: int = 0
    n_energies_used: int = 0
    k_table: tuple[tuple[float, float, float], ...] = ()     # (E [eV], k [1/nm], sigma_k)
    distance_range_nm: tuple[float, float] | None = None
    scatterer_kind: str = ""
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "dispersion"

    @property
    def e0_mev(self) -> float | None:
        return None if self.e0_ev is None else self.e0_ev * 1e3


#: the wavevector search range, 1/nm. The lower end is a half-wavelength longer than any
#: frame; the upper end is finer than any tip resolves.
K_MIN, K_MAX = 0.2, 15.0
#: a fitted k within this fraction of its bound is the optimiser on the fence, not a number
BOUND_MARGIN = 0.02
#: the fitted oscillation has to stand this far above the residual to count as one
MIN_AMPLITUDE_OVER_RESIDUAL = 1.5
#: the periodogram peak has to stand this far above its own median for the trace to be ripple
MIN_PERIODOGRAM_SIGNIFICANCE = 3.0
#: and the model has to explain this much of the trace's variance
MIN_ENERGY_R2 = 0.35


def _nyquist_k(d_nm: npt.NDArray[np.float64]) -> float:
    """The largest wavevector these sample positions can carry.

    The ripple repeats every pi/k, so two samples per repeat needs pi/k >= 2*spacing. Searching
    past it does not find a shorter wavelength, it finds whichever alias of the noise happens
    to line up — and with hundreds of candidate k that always happens somewhere.
    """
    s = np.diff(np.sort(np.asarray(d_nm, float)))
    step = float(np.median(s[s > 0])) if np.any(s > 0) else 0.0
    return K_MAX if step <= 0 else min(K_MAX, math.pi / (2.0 * step))


def _dominant_k(d_nm: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
                kind: str = "step", k_max: float | None = None) -> tuple[float, float] | None:
    """A first guess at k, by correlating the trace against the model at every k.

    Not an FFT. The spectra are not evenly spaced — a line usually has them packed near the
    scatterer and sparse far away — and regridding onto a uniform axis dilutes exactly the
    close-in ripple the wavevector lives in. Correlating against the basis at each candidate k
    uses the samples where they are, which is what a Lomb periodogram does for the same reason.
    """
    from scipy import special

    if d_nm.size < 8:
        return None
    yy = np.asarray(y, float) - float(np.mean(y))
    s = float(np.std(yy))
    if not (s > 0):
        return None
    yy = yy / s
    hi = float(k_max if k_max is not None else K_MAX)
    if hi <= K_MIN:
        return None
    ks = np.linspace(K_MIN, hi, 900)
    arg = 2.0 * ks[:, None] * np.asarray(d_nm, float)[None, :]
    basis = special.j0(arg) if kind == "step" else np.cos(arg)
    basis = basis - basis.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(basis, axis=1)
    score = np.abs(basis @ yy) / np.maximum(norm, 1e-12)
    i = int(np.argmax(score))
    med = float(np.median(score))
    # how far the best k stands above the rest of the search. A trace with no ripple in it has
    # a flat periodogram, and every energy below the band bottom is such a trace: without this
    # they all contribute a wavevector fitted to noise, and the band bottom is the intercept
    # those wavevectors decide.
    sig = float(score[i]) / max(med, 1e-12)
    return float(ks[i]), sig


def _fit_k(d_nm, y, k0: float, kind: str,
           k_max: float | None = None) -> tuple[float, float] | None:
    from scipy import optimize, special

    def step(x, amp, k, lam, c):
        return c + amp * special.j0(2 * k * x) * np.exp(-2 * x / lam)

    def point(x, amp, k, lam, c):
        return c + amp * np.cos(2 * k * x) / np.sqrt(np.maximum(k * x, 1e-6)) * np.exp(-2 * x / lam)

    model = step if kind == "step" else point
    k_lo = K_MIN
    k_hi = float(k_max if k_max is not None else K_MAX)
    if k_hi <= k_lo:
        return None
    p0 = [float(np.ptp(y)) / 2 or 0.1, float(k0), 60.0, float(np.mean(y))]
    try:
        popt, pcov = optimize.curve_fit(model, d_nm, y, p0=p0,
                                        bounds=([-10, k_lo, 2.0, -50], [10, k_hi, 1000.0, 50]),
                                        maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    k = float(popt[1])
    # a wavevector sitting on a bound is not a measurement, it is the optimiser giving up.
    # Below the band bottom there is no standing wave at all, and every one of those energies
    # would otherwise enter the table pinned at k_lo and drag the intercept — which is the
    # band bottom, the number being measured.
    if not (k_lo * (1.0 + BOUND_MARGIN) < k < k_hi * (1.0 - BOUND_MARGIN)):
        return None
    # and the model has to actually describe the trace: the oscillation it fitted must stand
    # above what is left over after fitting it
    fitted = model(d_nm, *popt)
    resid = float(np.sqrt(np.mean((y - fitted) ** 2)))
    amp = abs(float(popt[0]))
    if resid > 0 and amp < MIN_AMPLITUDE_OVER_RESIDUAL * resid:
        return None
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    if r2 < MIN_ENERGY_R2:
        return None
    err = float(np.sqrt(abs(pcov[1, 1]))) if np.all(np.isfinite(pcov)) else float("nan")
    return k, (err if math.isfinite(err) and err > 0 else 0.05 * k)


def fit_dispersion(energies_ev: npt.ArrayLike, distances_nm: npt.ArrayLike,
                   didv: npt.ArrayLike, *, scatterer_kind: str = "step",
                   min_distance_nm: float = DEFAULT_MIN_DISTANCE_NM,
                   normalize: bool = True,
                   min_energies: int = MIN_ENERGIES,
                   min_r2: float = MIN_R2) -> DispersionResult:
    """``didv`` is (n_energies, n_distances) on a common energy grid.

    Returns the band bottom and the effective mass, or says why it could not."""
    warns: list[str] = []
    e = np.asarray(energies_ev, dtype=float).ravel()
    d = np.asarray(distances_nm, dtype=float).ravel()
    g = np.asarray(didv, dtype=float)
    if g.ndim != 2 or g.shape != (e.size, d.size):
        return DispersionResult("undecidable", reasons=("shape_mismatch",),
                                notes={"expected": (e.size, d.size), "got": tuple(g.shape)})
    keep = d >= float(min_distance_nm)
    if keep.sum() < 6:
        return DispersionResult("undecidable", n_spectra=int(d.size),
                                reasons=("too_few_distances",))
    d = d[keep]
    g = g[:, keep]
    order = np.argsort(d)
    d, g = d[order], g[:, order]
    if float(d[-1] - d[0]) < 3.0:
        return DispersionResult("undecidable", n_spectra=int(d.size),
                                distance_range_nm=(float(d[0]), float(d[-1])),
                                reasons=("distance_span_too_short",))
    if normalize:
        far = g[:, -1][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(np.abs(far) > 1e-30, g / far, g)
        g = np.nan_to_num(g, nan=1.0, posinf=1.0, neginf=1.0)

    k_nyq = _nyquist_k(d)
    table: list[tuple[float, float, float]] = []
    for i, energy in enumerate(e):
        y = g[i]
        if not np.isfinite(y).all() or float(np.ptp(y)) <= 0:
            continue
        first = _dominant_k(d, y, scatterer_kind, k_nyq)
        if first is None or first[1] < MIN_PERIODOGRAM_SIGNIFICANCE:
            continue
        got = _fit_k(d, y, first[0], scatterer_kind, k_nyq)
        if got is None:
            continue
        table.append((float(energy), got[0], got[1]))
    if len(table) < int(min_energies):
        return DispersionResult("no_standing_wave" if len(table) else "undecidable",
                                n_spectra=int(d.size), n_energies_used=len(table),
                                distance_range_nm=(float(d[0]), float(d[-1])),
                                scatterer_kind=scatterer_kind,
                                k_table=tuple(table),
                                reasons=("too_few_energies_with_a_wavevector",))

    ee = np.array([t[0] for t in table])
    kk = np.array([t[1] for t in table])
    sk = np.array([t[2] for t in table])
    w = 1.0 / np.maximum(sk, 1e-6) ** 2
    x = kk ** 2
    # weighted least squares of E = E0 + s·k²
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * ee).sum() / sw
    sxx = (w * (x - mx) ** 2).sum()
    sxy = (w * (x - mx) * (ee - my)).sum()
    if sxx <= 0:
        return DispersionResult("undecidable", n_spectra=int(d.size), n_energies_used=len(table),
                                k_table=tuple(table), reasons=("degenerate_fit",))
    slope = sxy / sxx
    intercept = my - slope * mx
    pred = intercept + slope * x
    ss_res = float(((ee - pred) ** 2).sum())
    ss_tot = float(((ee - ee.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if slope <= 0:
        return DispersionResult("no_standing_wave", n_spectra=int(d.size),
                                n_energies_used=len(table), k_table=tuple(table), r2=r2,
                                scatterer_kind=scatterer_kind,
                                reasons=("negative_dispersion",))
    m_eff = HBAR2_OVER_2M_EV_NM2 / slope
    resid_var = ss_res / max(len(table) - 2, 1)
    slope_err = math.sqrt(resid_var / sxx) if sxx > 0 else float("nan")
    e0_err = math.sqrt(resid_var * (1.0 / sw + mx ** 2 / sxx))
    m_err = m_eff * (slope_err / slope) if slope > 0 and math.isfinite(slope_err) else None
    verdict = "dispersion" if r2 >= float(min_r2) else "undecidable"
    if verdict != "dispersion":
        warns.append("poor_linearity")
    return DispersionResult(verdict, e0_ev=float(intercept), e0_err_ev=float(e0_err),
                            m_eff=float(m_eff), m_eff_err=(float(m_err) if m_err else None),
                            r2=float(r2), n_spectra=int(d.size), n_energies_used=len(table),
                            k_table=tuple(table),
                            distance_range_nm=(float(d[0]), float(d[-1])),
                            scatterer_kind=scatterer_kind, warnings=tuple(warns))


def distance_to_line(x_m, y_m, *, edge_x_m: float, edge_y_m: float, edge_angle_deg: float):
    """Signed perpendicular distance (nm) from points to a straight step edge."""
    th = math.radians(float(edge_angle_deg))
    nx, ny = -math.sin(th), math.cos(th)
    dx = np.asarray(x_m, float) - float(edge_x_m)
    dy = np.asarray(y_m, float) - float(edge_y_m)
    return (dx * nx + dy * ny) * 1e9


def distance_to_point(x_m, y_m, *, scatterer_x_m: float, scatterer_y_m: float):
    """Radial distance (nm) from points to a point scatterer. Inputs are metres."""
    return np.hypot(np.asarray(x_m, float) - float(scatterer_x_m),
                    np.asarray(y_m, float) - float(scatterer_y_m)) * 1e9
