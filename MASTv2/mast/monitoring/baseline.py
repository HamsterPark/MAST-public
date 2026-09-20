"""Current-noise baseline: a reference curve for the monitor.

Current-dependent noise can make one absolute RMS threshold unsuitable across
working points. The sigma model uses ``sigma^2(I) = c + d*I^2``; the white-floor
model additionally includes the fixed shot-noise term ``2eI``. Their adequacy must
be checked on the target instrument rather than assumed from a shipped example.

The baseline supplies a denominator, not an additional alarm threshold. Width
and spectrum statistics reuse the same implementations as the live monitor.
Condition changes remain visible: an unavailable or mismatched reference must
not silently become a statement that the instrument is healthy.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Sequence

import numpy as np

from mast.monitoring.features import _as_1d, _detrend_linear

logger = logging.getLogger(__name__)

#: Elementary charge — shot noise 2eI is a KNOWN quantity, never a fitted one.
#: Fitting it would spend a degree of freedom on a term physics already fixes.
Q_E = 1.602176634e-19

#: Beyond this multiple of the calibrated current range the model is not
#: extrapolated — the verdict becomes ``unjudged``. Chosen rather than derived:
#: the measured span (17 pA … 1 nA) already covers normal operation, so anything
#: outside 2x it is a working point the baseline genuinely says nothing about.
EXTRAPOLATION_LIMIT = 2.0

#: Histogram resolution for the stored width distribution.
_HIST_BINS = 400


# ── models ──────────────────────────────────────────────────────────────────

@dataclass
class SigmaModel:
    """``sigma^2(I) = c + d*I^2`` — the reference curve the monitor judges against."""

    c_a2: float                  #: A^2, the current-independent term
    d: float                     #: dimensionless, the multiplicative term
    r2: float                    #: fit quality in log space
    i_lo_a: float                #: calibrated range, low
    i_hi_a: float                #: calibrated range, high
    n_points: int

    def expected(self, i_a: float) -> Optional[float]:
        """Expected sigma in amps at this current, or None if unusable."""
        try:
            v = self.c_a2 + self.d * float(i_a) ** 2
        except (TypeError, ValueError):
            return None
        return math.sqrt(v) if v > 0 else None

    def in_range(self, i_a: float) -> bool:
        try:
            i = abs(float(i_a))
        except (TypeError, ValueError):
            return False
        return self.i_lo_a <= i <= self.i_hi_a

    def extrapolation_factor(self, i_a: float) -> float:
        """How far outside the calibrated range, as a multiple. 1.0 = inside."""
        try:
            i = abs(float(i_a))
        except (TypeError, ValueError):
            return float("inf")
        if i < self.i_lo_a and i > 0:
            return self.i_lo_a / i
        if i > self.i_hi_a and self.i_hi_a > 0:
            return i / self.i_hi_a
        return 1.0

    @property
    def additive_sigma_a(self) -> float:
        """The sigma this machine shows as I -> 0: the preamp floor."""
        return math.sqrt(self.c_a2) if self.c_a2 > 0 else 0.0

    @property
    def relative_floor(self) -> float:
        """The sigma/I ratio the multiplicative term alone would give."""
        return math.sqrt(self.d) if self.d > 0 else 0.0

    @property
    def crossover_a(self) -> Optional[float]:
        """Current at which the two terms are equal. Below it, raising the
        current improves signal-to-noise; above it, it stops helping."""
        if self.d <= 0 or self.c_a2 <= 0:
            return None
        return math.sqrt(self.c_a2 / self.d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(additive_sigma_a=self.additive_sigma_a,
                 relative_floor=self.relative_floor,
                 crossover_a=self.crossover_a)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["SigmaModel"]:
        if not d:
            return None
        try:
            return cls(c_a2=float(d["c_a2"]), d=float(d["d"]),
                       r2=float(d.get("r2") or 0.0),
                       i_lo_a=float(d["i_lo_a"]), i_hi_a=float(d["i_hi_a"]),
                       n_points=int(d.get("n_points") or 0))
        except (KeyError, TypeError, ValueError):
            logger.debug("SigmaModel.from_dict: unusable payload", exc_info=True)
            return None


@dataclass
class WhiteModel:
    """``PSD_white(I) = amp^2 + 2eI + b*I^2`` — the same split, on the floor.

    Shot noise enters as a KNOWN term, not a fitted one. Separating the three
    is what says whether a machine is preamp-limited or vibration-limited, and
    at which current the answer changes.
    """

    amp_a2_per_hz: float         #: preamp + feedback resistor, A^2/Hz
    b: float                     #: multiplicative, 1/Hz
    r2: float
    i_lo_a: float
    i_hi_a: float
    n_points: int

    @property
    def amp_a_per_rthz(self) -> float:
        return math.sqrt(self.amp_a2_per_hz) if self.amp_a2_per_hz > 0 else 0.0

    @property
    def crossover_a(self) -> Optional[float]:
        if self.b <= 0 or self.amp_a2_per_hz <= 0:
            return None
        return math.sqrt(self.amp_a2_per_hz / self.b)

    def expected(self, i_a: float) -> Optional[float]:
        try:
            i = abs(float(i_a))
        except (TypeError, ValueError):
            return None
        v = self.amp_a2_per_hz + 2.0 * Q_E * i + self.b * i * i
        return v if v > 0 else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(amp_a_per_rthz=self.amp_a_per_rthz, crossover_a=self.crossover_a)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["WhiteModel"]:
        if not d:
            return None
        try:
            return cls(amp_a2_per_hz=float(d["amp_a2_per_hz"]), b=float(d["b"]),
                       r2=float(d.get("r2") or 0.0),
                       i_lo_a=float(d["i_lo_a"]), i_hi_a=float(d["i_hi_a"]),
                       n_points=int(d.get("n_points") or 0))
        except (KeyError, TypeError, ValueError):
            return None


# ── fitting ─────────────────────────────────────────────────────────────────

def _log_fit(currents, values, model_fn, seed, *, known=None):
    """Least squares in log space for values spanning several orders of magnitude.
    
    Relative residuals prevent large observations from overwhelming the small
    observations that constrain the additive term. Known physical terms remain fixed.
    """
    I = np.asarray(currents, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(I) & np.isfinite(y) & (I > 0) & (y > 0)
    I, y = I[ok], y[ok]
    if I.size < 3:
        return None, I, y
    k = np.zeros_like(I) if known is None else np.asarray(known, float)[ok]

    def resid(p):
        return np.log(model_fn(np.exp(p), I, k)) - np.log(y)

    try:
        from scipy.optimize import least_squares
        r = least_squares(resid, np.log(np.asarray(seed, dtype=np.float64)))
        params = np.exp(r.x)
    except Exception:  # noqa: BLE001 — scipy absent or the fit did not converge
        logger.debug("log fit unavailable, falling back to a coarse grid",
                     exc_info=True)
        params = _grid_fit(model_fn, I, y, k, seed)
        if params is None:
            return None, I, y
    pred = model_fn(params, I, k)
    ly = np.log(y)
    ss = 1.0 - ((np.log(pred) - ly) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-300)
    return (params, float(ss)), I, y


def _grid_fit(model_fn, I, y, k, seed):
    """Coarse log-grid search — the no-scipy path. Never as good as the real
    fit, but it must not be the reason a baseline cannot be built at all."""
    best, best_err = None, float("inf")
    s = np.asarray(seed, dtype=np.float64)
    for f0 in np.logspace(-3, 3, 25):
        for f1 in np.logspace(-3, 3, 25):
            p = np.array([s[0] * f0, s[1] * f1])
            try:
                err = float(((np.log(model_fn(p, I, k)) - np.log(y)) ** 2).sum())
            except (ValueError, FloatingPointError):
                continue
            if math.isfinite(err) and err < best_err:
                best, best_err = p, err
    return best


def fit_sigma_model(currents_a: Sequence[float],
                    sigmas_a: Sequence[float]) -> Optional[SigmaModel]:
    """Fit ``sigma^2 = c + d*I^2`` over a current sweep."""
    res, I, y = _log_fit(currents_a, sigmas_a,
                         lambda p, i, k: np.sqrt(p[0] + p[1] * i * i),
                         seed=(np.nanmin(np.asarray(sigmas_a, float)) ** 2
                               if len(sigmas_a) else 1e-26,
                               1e-6))
    if res is None:
        return None
    (c, d), r2 = res
    return SigmaModel(c_a2=float(c), d=float(d), r2=r2,
                      i_lo_a=float(np.min(I)), i_hi_a=float(np.max(I)),
                      n_points=int(I.size))


def fit_white_model(currents_a: Sequence[float],
                    white_a2_per_hz: Sequence[float]) -> Optional[WhiteModel]:
    """Fit ``PSD_white = amp^2 + 2eI + b*I^2``, with 2eI held at its known value."""
    I0 = np.asarray(currents_a, dtype=np.float64)
    shot = 2.0 * Q_E * np.abs(I0)
    res, I, y = _log_fit(currents_a, white_a2_per_hz,
                         lambda p, i, k: p[0] + k + p[1] * i * i,
                         seed=(float(np.nanmin(np.asarray(white_a2_per_hz, float)))
                               if len(white_a2_per_hz) else 1e-28, 1e-6),
                         known=shot)
    if res is None:
        return None
    (amp2, b), r2 = res
    return WhiteModel(amp_a2_per_hz=float(amp2), b=float(b), r2=r2,
                      i_lo_a=float(np.min(I)), i_hi_a=float(np.max(I)),
                      n_points=int(I.size))


# ── width statistics (same detrend as features.detrended_rms) ───────────────

def width_stats(y, *, per_segment: bool = False) -> dict:
    """Every width estimator on one sample, in amps.

    ``sigma_a`` is bit-for-bit :func:`features.detrended_rms`'s
    ``rms_detrended_a`` when handed the same segment — same linear detrend
    (``features._detrend_linear``), same ``numpy.std``. That equality is the
    point: the baseline has to be comparable with the column the live monitor
    writes, or it is measuring the gap between two implementations.

    The robust estimators sit next to the plain one because the RATIO between
    them is the tail diagnostic. For a Gaussian all three agree; a dominant
    single tone flattens the core, a two-state RTN makes it bimodal, and spikes
    push ``sigma`` past both robust estimates. One number hides all of that.
    """
    arr = _as_1d(y)
    if arr.size < 3:
        return {"n": int(arr.size)}
    resid, _ = _detrend_linear(arr)
    p = np.percentile(resid, [0.1, 1, 25, 50, 75, 99, 99.9])
    med = float(p[3])
    mad = float(np.median(np.abs(resid - med)))
    sd = float(resid.std())
    iqr = float(p[4] - p[2])
    out = {
        "n": int(arr.size),
        "mean_a": float(arr.mean()),
        "sigma_a": sd,
        "sigma_iqr_a": iqr / 1.349,
        "sigma_mad_a": mad * 1.4826,
        "iqr_a": iqr,
        "p99_p1_a": float(p[5] - p[1]),
        "ptp_a": float(resid.max() - resid.min()),
    }
    if sd > 0:
        z = resid / sd
        out["kurtosis"] = float(np.mean(z ** 4) - 3.0)
        out["skewness"] = float(np.mean(z ** 3))
        out["iqr_over_sigma"] = out["sigma_iqr_a"] / sd
        out["mad_over_sigma"] = out["sigma_mad_a"] / sd
    fw, _hist = fwhm(resid)
    out["fwhm_a"] = fw
    if sd > 0 and fw and math.isfinite(fw):
        # 2.3548 for a Gaussian. Reported rather than judged: it is a shape
        # descriptor, and no threshold on it has been calibrated on the instrument.
        out["fwhm_over_sigma"] = fw / sd
    return out


def fwhm(x, bins: int = _HIST_BINS):
    """Full width at half maximum of the smoothed value histogram.

    Returns ``(width, (edges, smoothed_counts))``; the histogram is handed back
    because the caller stores it — recomputing it later from data that has since
    been swept off disk is not possible.
    """
    arr = _as_1d(x)
    if arr.size < 32:
        return float("nan"), None
    lo, hi = np.percentile(arr, [0.05, 99.95])
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return float("nan"), None
    h, edges = np.histogram(arr, bins=bins, range=(float(lo), float(hi)))
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    k /= k.sum()
    hs = np.convolve(h.astype(np.float64), k, mode="same")
    if hs.max() <= 0:
        return float("nan"), (edges, hs)
    above = np.where(hs >= 0.5 * hs.max())[0]
    if above.size < 2:
        return float("nan"), (edges, hs)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    return float(ctr[above[-1]] - ctr[above[0]]), (edges, hs)


# ── spectral lines ──────────────────────────────────────────────────────────

def peak_metrics(freqs, psd, lo_hz: float, hi_hz: float,
                 i_a: float | None = None) -> dict:
    """One line's height against the broadband floor either side of it.

    The floor is a MEDIAN of the neighbouring bands, not a mean: those bands
    routinely contain other lines, and the mean of a band holding a 30x line
    measures the line, not the floor.
    """
    f = np.asarray(freqs, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    inb = (f >= lo_hz) & (f <= hi_hz)
    w = hi_hz - lo_hz
    side = (((f >= lo_hz - 4 * w) & (f < lo_hz - w))
            | ((f > hi_hz + w) & (f <= hi_hz + 4 * w)))
    if inb.sum() < 1 or side.sum() < 3:
        return {}
    floor = float(np.median(p[side]))
    pk = float(p[inb].max())
    excess = float(np.trapezoid(np.maximum(p[inb] - floor, 0.0), f[inb]))
    rms = math.sqrt(max(excess, 0.0))
    out = {
        "f_peak_hz": float(f[inb][int(np.argmax(p[inb]))]),
        "a_per_rthz": math.sqrt(pk) if pk > 0 else 0.0,
        "over_floor": (pk / floor) if floor > 0 else None,
        "rms_a": rms,
    }
    if i_a:
        out["rel_rms"] = rms / abs(float(i_a))
    return out


def robust_white_floor(freqs, psd, lo_hz: float = 250.0,
                       hi_hz: float = 950.0) -> Optional[float]:
    """Median PSD over a line-free-ish band, in A^2/Hz. Median for the same
    reason as above — the band is not actually line-free."""
    f = np.asarray(freqs, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    m = (f >= lo_hz) & (f <= hi_hz)
    return float(np.median(p[m])) if m.sum() else None


def scaling_exponent(xs, ys) -> tuple[float, float]:
    """log-log slope and R^2 — the exponent that tells mechanisms apart.

    ``a_I`` (amplitude vs current, bias fixed) and ``a_V`` (vs bias, current
    held by feedback) together identify the mechanism; neither alone does.
    Distance modulation gives (+1, 0), voltage pickup (+1, -1), additive
    current noise (0, 0), shot noise (+0.5, 0). The setpoint sweep on its own
    cannot separate the first two — that is why the bias sweep exists.
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    c = np.polyfit(lx, ly, 1)
    r = ly - np.polyval(c, lx)
    r2 = 1.0 - (r ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-300)
    return float(c[0]), float(r2)


#: (name, expected a_I, expected a_V). Used only to LABEL a measured pair —
#: the exponents are the evidence, the label is a convenience.
MECHANISMS: tuple[tuple[str, float, float], ...] = (
    ("distance_modulation", 1.0, 0.0),
    ("voltage_pickup", 1.0, -1.0),
    ("additive_preamp", 0.0, 0.0),
    ("shot_noise", 0.5, 0.0),
)

#: Sum of |da_I| + |da_V| beyond which no label is claimed. A pair that sits
#: between two mechanisms is genuinely between them — usually because the line
#: has two contributions — and saying so is more useful than picking the nearer.
_MECH_TOL = 0.7


def classify_mechanism(a_i: float, a_v: float) -> tuple[Optional[str], float]:
    """Nearest mechanism to a measured (a_I, a_V), or None if none is near."""
    if not (math.isfinite(a_i) and math.isfinite(a_v)):
        return None, float("inf")
    best, best_d = None, float("inf")
    for name, pi, pv in MECHANISMS:
        d = abs(a_i - pi) + abs(a_v - pv)
        if d < best_d:
            best, best_d = name, d
    return (best if best_d <= _MECH_TOL else None), best_d


# ── live comparison ─────────────────────────────────────────────────────────

@dataclass
class BaselineVerdict:
    """The answer to "how does right now compare with the reference"."""

    judged: bool = False
    reason: str = ""
    sigma_a: Optional[float] = None
    expected_a: Optional[float] = None
    ratio: Optional[float] = None
    in_range: bool = False
    extrapolation: Optional[float] = None
    baseline_id: Optional[int] = None
    conditions_match: Optional[bool] = None
    condition_diff: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compare(sigma_a: float | None, i_a: float | None,
            model: SigmaModel | None, *,
            baseline_id: int | None = None,
            conditions_match: bool | None = None,
            condition_diff: Sequence[str] = ()) -> BaselineVerdict:
    """Compare a live segment against the baseline curve.

    Returns ``judged=False`` with a stated reason whenever the comparison cannot
    be made — a missing baseline, an unreadable current, a working point too far
    outside the calibrated range. It never returns a ratio it does not believe,
    because a plausible-looking 1.0 is indistinguishable from "fine" and this
    subsystem has paid for that confusion before.
    """
    v = BaselineVerdict(baseline_id=baseline_id,
                        conditions_match=conditions_match,
                        condition_diff=list(condition_diff))
    if model is None:
        v.reason = "没有可用基线 —— 先跑一次 CharacteriseCurrentNoise"
        return v
    if sigma_a is None or not math.isfinite(float(sigma_a)):
        v.reason = "本段的噪声 RMS 读不到"
        return v
    if i_a is None or not math.isfinite(float(i_a)) or abs(float(i_a)) <= 0:
        v.reason = "本段的电流读不到,无从选取基线上的对应点"
        return v
    v.sigma_a = float(sigma_a)
    v.in_range = model.in_range(i_a)
    v.extrapolation = model.extrapolation_factor(i_a)
    if v.extrapolation > EXTRAPOLATION_LIMIT:
        v.reason = ("电流 %.3g A 距基线标定区间 [%.3g, %.3g] A 太远(%.1f 倍),"
                    "不外推" % (abs(float(i_a)), model.i_lo_a, model.i_hi_a,
                                v.extrapolation))
        return v
    exp = model.expected(i_a)
    if not exp or exp <= 0:
        v.reason = "基线模型在这个电流上给不出正数"
        return v
    v.expected_a = exp
    v.ratio = float(sigma_a) / exp
    v.judged = True
    return v


# ── condition snapshot ──────────────────────────────────────────────────────

# ── Z channel, and Z-vs-current jointly ────────────────────────────────────

#: Minimum number of synchronous segment pairs before a coherence is reported.
#:
#: **A single segment gives coherence identically 1.0 at every frequency** —
#: |Pxy|^2 = Pxx*Pyy holds exactly when each spectrum is one periodogram, with
#: no averaging to break the equality. That is not a measurement, it is an
#: algebraic identity, and it looks exactly like a perfect result. Eight pairs
#: put the bias at roughly 1/8 and make the number mean something.
_MIN_COHERENCE_SEGMENTS = 8

#: Coherence a bin must clear before it may contribute to kappa.
#: 0.9, not the 0.5 used for "is this band mechanical at all" —
#: |H| is biased upward by whatever fraction of the current-side
#: power is NOT the Z motion, and at 0.5 that fraction is half.
_KAPPA_MIN_COHERENCE = 0.9


def cross_spectra(x_runs: Sequence, y_runs: Sequence, fs_hz: float):
    """Welch-averaged auto- and cross-spectra of two synchronous channels.

    Returns ``(freqs, Pxx, Pyy, Pxy)`` with ``Pxy`` complex, or ``None`` when
    there is not enough synchronous data. Same window and per-segment detrend as
    :func:`features.psd_of_runs`, so the auto-spectra here and the PSD stored
    elsewhere are the same quantity.
    """
    xs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in (x_runs or [])]
    ys = [np.asarray(a, dtype=np.float64).reshape(-1) for a in (y_runs or [])]
    pairs = [(a, b) for a, b in zip(xs, ys) if a.size == b.size and a.size >= 64]
    if len(pairs) < 2 or fs_hz <= 0:
        return None
    n = min(a.size for a, _ in pairs)
    w = np.hanning(n)
    norm = fs_hz * (w ** 2).sum()
    Pxx = Pyy = Pxy = None
    for a, b in pairs:
        a, b = a[:n], b[:n]
        fa = np.fft.rfft(_detrend_linear(a)[0] * w)
        fb = np.fft.rfft(_detrend_linear(b)[0] * w)
        pxx = (np.abs(fa) ** 2) * 2.0 / norm
        pyy = (np.abs(fb) ** 2) * 2.0 / norm
        pxy = (np.conj(fa) * fb) * 2.0 / norm
        Pxx = pxx if Pxx is None else Pxx + pxx
        Pyy = pyy if Pyy is None else Pyy + pyy
        Pxy = pxy if Pxy is None else Pxy + pxy
    k = len(pairs)
    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)
    return freqs, Pxx / k, Pyy / k, Pxy / k, k


def z_current_coupling(z_runs: Sequence, i_runs: Sequence, fs_hz: float,
                       i_mean_a: float | None = None) -> dict:
    """How much of the current noise is the gap actually moving.

    Two derived quantities, and they answer different questions:

    * **coherence** gamma^2(f) — is this line the same physical event in both
      channels? A mechanical vibration moves Z and modulates I, so it is
      coherent; electrical pickup on the current side is not. This is the
      independent check on the mechanism attribution the two sweeps produce.
    * **transfer function** |H| = |Pzi| / Pzz, in A/m — how much current a metre
      of gap motion produces. For a tunnel junction dI/dz = -2*kappa*I, so
      ``kappa = |H| / (2 I)`` falls out **without running an I-z curve**. It is
      measured on whatever the building is already shaking at, which is why it
      costs nothing beyond the burst that was taken anyway.

    ``kappa`` is reported over the band where coherence is high enough for the
    ratio to mean anything; where it is not, no number is given rather than an
    average dominated by frequencies at which the two channels are unrelated.
    """
    out: dict = {"available": False}
    cs = cross_spectra(z_runs, i_runs, fs_hz)
    if cs is None:
        out["detail"] = "同步段不足，算不了 Z-电流耦合"
        return out
    freqs, Pzz, Pii, Pzi, k = cs
    out["n_pairs"] = k
    if k < _MIN_COHERENCE_SEGMENTS:
        # Report the spectra, refuse the coherence: with too few averages the
        # number is biased towards 1 and would read as "perfectly coupled".
        out["detail"] = (f"只有 {k} 对同步段（需要 ≥{_MIN_COHERENCE_SEGMENTS}）"
                         "——段数太少时相干性会偏向 1，那是平均不足的假象，不是耦合")
        out["available"] = False
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma2 = (np.abs(Pzi) ** 2) / (Pzz * Pii)
        H = np.abs(Pzi) / Pzz
    gamma2 = np.clip(np.nan_to_num(gamma2, nan=0.0, posinf=0.0), 0.0, 1.0)

    #: Bias floor of the estimator: E[gamma^2] ~ 1/k for uncorrelated channels.
    #: Anything at or below this is indistinguishable from no coupling at all.
    floor = 1.0 / k
    out.update(available=True, freqs_hz=freqs.tolist(),
               coherence=gamma2.tolist(), coherence_floor=float(floor),
               transfer_a_per_m=H.tolist())

    band = (freqs >= 3.0) & (freqs <= min(1000.0, freqs[-1]))
    strong = band & (gamma2 > max(0.5, 4.0 * floor))
    out["coherent_fraction"] = float(strong.sum() / max(1, band.sum()))

    # kappa comes from a STRICTER band than "coherent". |H| = |Pzi|/Pzz is only
    # an unbiased transfer estimate where the two channels really are the same
    # event; at gamma^2 = 0.5 half the current-side power is something else, and
    # that half biases |H| upward. Measured on a synthetic junction with kappa
    # injected at 12/nm, the 0.5 band returned 14.3/nm (+19 %) while the 0.9
    # band returned the right answer. The looser band is still reported as
    # ``coherent_fraction`` — it answers a different question ("how much of the
    # spectrum is mechanical at all").
    kappa_sel = band & (gamma2 > _KAPPA_MIN_COHERENCE)
    if kappa_sel.sum() >= 3 and i_mean_a:
        # Power-weighted, not a plain median: the estimate is a ratio of
        # spectra, so the bins carrying the most Z motion are the ones that
        # determine it best.
        kappa = (float(np.abs(Pzi)[kappa_sel].sum() / Pzz[kappa_sel].sum())
                 / (2.0 * abs(float(i_mean_a))))
        strong = kappa_sel
        out["kappa_per_m"] = kappa
        out["kappa_per_nm"] = kappa * 1e-9
        # Apparent barrier height from kappa = sqrt(2 m phi)/hbar.
        hbar, m_e, q = 1.054571817e-34, 9.1093837015e-31, 1.602176634e-19
        out["apparent_barrier_ev"] = ((hbar * kappa) ** 2) / (2 * m_e) / q
        out["kappa_band_hz"] = [float(freqs[strong][0]), float(freqs[strong][-1])]
        out["kappa_n_bins"] = int(strong.sum())
    else:
        out["kappa_per_m"] = None
        out["detail"] = (
            f"没有足够的频点相干性超过 {_KAPPA_MIN_COHERENCE} —— "
            "这一段里电流噪声与 Z 的运动不是同一件事，不给传递函数。"
            "（相干频段占比另报，那问的是另一个问题）")
    return out


def z_stats(z_runs: Sequence, fs_hz: float):
    """(summary, freqs, psd) —— 谱数组单独给出，调用方落盘；
    数据不可用时是 ``(summary, None, None)``。

    Width and spectrum summary for the Z channel.

    Z is RECORDED, never judged. The current-side detectors were each validated
    against pure Gaussian noise before being trusted (four of them were rewritten
    when the real instrument disagreed); carrying those thresholds over to a
    displacement signal would be a guess, and the way a guess fails here is a
    false alarm at three in the morning that stops a running experiment.

    One thing the Z channel genuinely cannot use from the current side: white
    noise is the wrong null hypothesis. Piezo creep and thermal drift are a
    random walk (1/f^2), whose range grows as sqrt(N) — so any fixed threshold
    on span is eventually crossed by drift alone. The first difference is the
    quantity that IS iid for a random walk, and it is what gets reported.
    """
    runs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in (z_runs or [])]
    runs = [a for a in runs if a.size >= 64]
    if not runs or fs_hz <= 0:
        return {"available": False, "detail": "没有可用的 Z 数据"}, None, None
    from mast.monitoring.features import psd_of_runs

    allz = np.concatenate(runs)
    w = width_stats(allz)
    freqs, psd = psd_of_runs(runs, fs_hz)
    out = {
        "available": True, "n_runs": len(runs), "fs_hz": float(fs_hz),
        "z_mean_m": float(allz.mean()),
        "sigma_m": w.get("sigma_a"), "fwhm_m": w.get("fwhm_a"),
        "ptp_m": w.get("ptp_a"),
        "sigma_mad_m": w.get("sigma_mad_a"),
        "kurtosis": w.get("kurtosis"),
        "white_floor_m2hz": robust_white_floor(freqs, psd),
        # 一阶差分：随机游走的 Δ 是 iid，span 不是。
        "step_rms_m": float(np.median([np.std(np.diff(a)) for a in runs])),
        "lines": {},
    }
    # l_example_800hz is an unconfigured example analysis band; verify it for the target.
    for nm, lo, hi in (("vib_6hz", 4.0, 8.0), ("l_50hz", 48.5, 51.5),
                       ("l_100hz", 98.5, 101.5), ("l_example_800hz", 795.0, 805.0)):
        m = peak_metrics(freqs, psd, lo, hi)
        if m:
            out["lines"][nm] = m
    return out, freqs, psd


def polarity_check(points: Sequence[dict]) -> Optional[dict]:
    """同一 |V| 上的正负偏压对比 —— 「偏压符号影响噪声吗」的直接回答。

    物理上不该有影响：距离调制的 dz 是力学量，电压耦合的 dI = dV/R 只依赖 |V|。
    但那是**推理**，而这台机器实际用过负偏压。成对比较把它变成一条被测量的结论；
    真出现不对称，那本身就是关于结的信息（整流、态密度不对称）。

    只比 |V| 配得上的点对。配不上就不比 —— 拿两个不同工作点的 sigma 相除得到的
    比值看起来完全正常，而它什么也不说明。
    """
    bias = [(k, p) for k, p in enumerate(points or [])
            if p.get("sweep") == "bias" and p.get("bias_v") and p.get("sigma_a")]
    pairs = []
    for kc, c in bias:
        if float(c["bias_v"]) >= 0:
            continue
        # 每个负点只与**时间上最近**的那个同 |V| 正点配对。
        #
        # 不同间隔的重复性可能不同；全配会混入长期漂移，故只与序列中最近点比较。
        cands = [(abs(kc - ka), ka, a) for ka, a in bias
                 if float(a["bias_v"]) > 0
                 and abs(float(a["bias_v"]) - abs(float(c["bias_v"]))) <= 1e-6]
        if not cands:
            continue
        gap, ka, a = min(cands, key=lambda t: t[0])
        pairs.append({
            "abs_bias_v": float(a["bias_v"]),
            "sigma_pos_a": a["sigma_a"], "sigma_neg_a": c["sigma_a"],
            "ratio_neg_over_pos": c["sigma_a"] / a["sigma_a"],
            "i_pos_a": a.get("i_measured_a"), "i_neg_a": c.get("i_measured_a"),
            #: 配对的两点在采集序列里隔了几个工况点。它决定这个比值该拿哪个
            #: 时间尺度的重复性去比 —— 所以报出来，不是内部细节。
            "points_apart": int(gap),
        })
    if not pairs:
        return None
    r = np.array([q["ratio_neg_over_pos"] for q in pairs], dtype=float)
    return {
        "n_pairs": len(pairs), "pairs": pairs,
        "ratio_mean": float(r.mean()),
        "ratio_max_dev": float(np.max(np.abs(r - 1.0))),
        "note": ("与 1.0 的偏离要拿 repeatability.relative_sd 去比 —— "
                 "小于它的差异是重复性，不是极性"),
    }



def bias_magnitude_check(points: Sequence[dict]) -> Optional[dict]:
    """噪声随**偏压幅度**怎么变 —— 正负分开拟合。

    ``polarity_check`` 回答的是「符号有没有影响」；这里回答「|V| 变大时噪声往
    哪走」。符号效应可能随幅度改变，只报一个汇总极性比值会隐藏这种关系。

    为什么用**相对**噪声 σ/|I| 而不是 σ：整条偏压扫描的 setpoint 是固定的，
    所以 |I| 基本不变，两者只差一个常数；但相对量能直接和 σ 模型的
    ``relative_floor`` 比，而绝对量不能。

    为什么正负**分开**拟合：合在一起拟合 |V| 的话，一个「正的往上、负的往下」
    的真实结构会互相抵消成一条平坦的线 —— 一个既不描述正偏压也不描述负偏压的
    结论，而且它看起来毫无问题。

    在对数-对数上取斜率（σ_rel ∝ |V|^k）：
      * k ≈ 0   —— 与偏压无关（距离调制主导时该是这样）
      * k < 0   —— 相对噪声随 |V| 下降（电压耦合的签名：dI = dV/R，
                   R ∝ V/I 固定时 dI/I ∝ dV/V）
      * k > 0   —— 随 |V| 上升，那既不是纯距离也不是纯电压耦合，值得单独看
    """
    rows = [p for p in (points or [])
            if p.get("sweep") == "bias" and p.get("bias_v")
            and p.get("sigma_a") and p.get("i_measured_a")]
    if len(rows) < 4:
        return None

    def _fit(sub):
        if len(sub) < 3:
            return None
        v = np.array([abs(float(p["bias_v"])) for p in sub], dtype=float)
        rel = np.array([abs(float(p["sigma_a"]) / float(p["i_measured_a"]))
                        for p in sub], dtype=float)
        ok = (v > 0) & (rel > 0)
        if ok.sum() < 3 or len(set(np.round(v[ok], 12))) < 3:
            return None
        lv, lr = np.log10(v[ok]), np.log10(rel[ok])
        k, b = np.polyfit(lv, lr, 1)
        pred = k * lv + b
        ss_res = float(np.sum((lr - pred) ** 2))
        ss_tot = float(np.sum((lr - np.mean(lr)) ** 2))
        return {
            "n": int(ok.sum()),
            "slope_log10": float(k),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "abs_bias_v": [float(x) for x in v[ok]],
            "sigma_rel": [float(x) for x in rel[ok]],
            "rel_at_min_v": float(rel[ok][int(np.argmin(v[ok]))]),
            "rel_at_max_v": float(rel[ok][int(np.argmax(v[ok]))]),
        }

    pos = _fit([p for p in rows if float(p["bias_v"]) > 0])
    neg = _fit([p for p in rows if float(p["bias_v"]) < 0])
    if pos is None and neg is None:
        return None

    out: dict = {"positive": pos, "negative": neg, "n_points": len(rows)}
    if pos and neg:
        out["slope_difference"] = float(pos["slope_log10"] - neg["slope_log10"])
        # 两支斜率差得多 ⇒ 极性的影响本身随幅度变，只报一个极性比值不够
        out["polarity_effect_is_bias_dependent"] = bool(
            abs(out["slope_difference"]) > 0.3)
    out["note"] = (
        "斜率是 log10(σ/|I|) 对 log10(|V|) 的。≈0 与偏压无关（距离调制）；"
        "<0 是电压耦合的签名；正负两支要分开看 —— 合起来拟合会让相反的趋势"
        "互相抵消成一条平坦的假线。要判显著，拿它跟 repeatability.relative_sd 比。")
    return out

def build_models(points: Sequence[dict]) -> dict:
    """Turn a list of measured working points into the cross-point derivatives.

    One implementation, two callers: the acquisition skill and the importer that
    brings an externally measured sweep in. Two copies of this would be two sets
    of coefficients from the same data, and the difference would be invisible
    until someone compared a stored baseline against a freshly measured one.

    Each point is a dict with at least ``i_measured_a``, ``sigma_a``, ``sweep``
    (``"setpoint"`` or ``"bias"``), optionally ``white_a2hz``, ``bias_v`` and a
    ``lines`` sub-dict of :func:`peak_metrics` outputs.

    Mechanism attribution needs BOTH sweeps. With only the setpoint sweep the
    lines come back carrying ``a_i`` and a null ``mechanism`` — that is the
    honest result, because a_I alone cannot tell a vibration from pickup on the
    bias line, and printing a guess there would be worse than printing nothing.
    """
    pts = [p for p in (points or []) if p.get("i_measured_a") and p.get("sigma_a")]
    sset = [p for p in pts if p.get("sweep") == "setpoint"]
    sbias = [p for p in pts if p.get("sweep") == "bias"]
    bias_magnitude = bias_magnitude_check(pts)

    sigma_model = white_model = None
    if len(sset) >= 3:
        sigma_model = fit_sigma_model([p["i_measured_a"] for p in sset],
                                      [p["sigma_a"] for p in sset])
        wf = [(p["i_measured_a"], p["white_a2hz"]) for p in sset
              if p.get("white_a2hz")]
        if len(wf) >= 3:
            white_model = fit_white_model([x for x, _ in wf], [y for _, y in wf])

    names: set = set()
    for p in pts:
        names.update((p.get("lines") or {}).keys())

    lines: dict = {}
    for nm in sorted(names):
        def _amp(rows, by_bias):
            xs, ys = [], []
            for p in rows:
                m = (p.get("lines") or {}).get(nm)
                if not (m and m.get("a_per_rthz")):
                    continue
                x = abs(p.get("bias_v") or 0.0) if by_bias else p["i_measured_a"]
                if x:
                    xs.append(x)
                    ys.append(m["a_per_rthz"])
            return xs, ys

        xi, yi = _amp(sset, False)
        xv, yv = _amp(sbias, True)
        a_i, r2i = scaling_exponent(xi, yi) if len(xi) >= 3 else (float("nan"),) * 2
        a_v, r2v = scaling_exponent(xv, yv) if len(xv) >= 3 else (float("nan"),) * 2
        mech, dist = classify_mechanism(a_i, a_v)
        ref = next((p for p in sset if (p.get("lines") or {}).get(nm)), None)
        rm = ((ref or {}).get("lines") or {}).get(nm) or {}
        lines[nm] = {
            "f_peak_hz": rm.get("f_peak_hz"), "a_per_rthz": rm.get("a_per_rthz"),
            "over_floor": rm.get("over_floor"), "rel_rms": rm.get("rel_rms"),
            "a_i": a_i if math.isfinite(a_i) else None,
            "r2_i": r2i if math.isfinite(r2i) else None,
            "a_v": a_v if math.isfinite(a_v) else None,
            "r2_v": r2v if math.isfinite(r2v) else None,
            "mechanism": mech,
            "mechanism_distance": dist if math.isfinite(dist) else None,
        }

    #: Repeatability is measured on points that share a working point, whatever
    #: sweep they came from — that is the only honest estimate of "how well does
    #: re-measuring reproduce", and it is what the ratio threshold has to clear.
    rep = None
    groups: dict = {}
    for p in pts:
        # 按 (电流量级, 偏压) 分组。同一组里的多个点就是「同一工作点测了几次」,
        # 无论它们来自哪一趟扫描 —— repeat 点正是为此存在的。
        key = (round(math.log10(abs(p["i_measured_a"])), 2),
               round(float(p.get("bias_v") or 0.0), 4))
        groups.setdefault(key, []).append(p["sigma_a"])
    best = max(groups.values(), key=len) if groups else []
    if len(best) >= 2:
        arr = np.asarray(best, dtype=np.float64)
        rep = {"n": int(arr.size), "sigma_mean_a": float(arr.mean()),
               "sigma_sd_a": float(arr.std(ddof=1)),
               "relative_sd": float(arr.std(ddof=1) / arr.mean()) if arr.mean() else None}

    # 极性结论并进 repeatability 而不是另开一个字段：判断「负/正差了 3% 算不算数」
    # 唯一的尺度就是同一工作点的重复散布。分开存的后果是每个读它的人各自去把两者
    # 凑到一起 —— 而调用方已经漏过一次（技能合并了它，导入那条路没有）。
    pol = polarity_check(pts)
    if pol:
        # 显著性在这里算完，不留给读的人自己算。
        #
        # 差异要除以**三点平均的标准误**，不是与单次重复散布直接比大小 ——
        # 后者是第一次读这批数据时容易犯的错，它把一个 1.6 sigma 的结果读成了
        # 「超过重复性」。标准误 = rsd * sqrt(2) / sqrt(n_pairs)：sqrt(2) 因为
        # 比的是两次测量之差，sqrt(n) 因为取了 n 对的平均。
        rsd = (rep or {}).get("relative_sd")
        if rsd and pol["n_pairs"]:
            se = float(rsd) * math.sqrt(2.0) / math.sqrt(pol["n_pairs"])
            dev = abs(1.0 - pol["ratio_mean"])
            pol["std_error"] = se
            pol["sigma"] = dev / se if se > 0 else None
            pol["significant"] = bool(se > 0 and dev / se >= 2.0)
            pol["scale_note"] = (
                "标准误用的是**跨整场表征**的重复散布（%.1f%%）。配对点若只隔一两个"
                "工况，真实的短时散布更小、显著性更高 —— 这个数是保守的下界。"
                % (float(rsd) * 100))
        rep = {**(rep or {}), "polarity": pol}
    return {"sigma_model": sigma_model, "white_model": white_model,
            "lines": lines, "repeatability": rep, "polarity": pol,
            # 偏压**幅度**依赖 —— 与 polarity（符号依赖）是两个问题。
            # 2026-08-19 之前只算了后者，前者靠人手算，于是它不会进基线、
            # 也不会被下一次比较引用。
            "bias_magnitude": bias_magnitude,
            "n_setpoint": len(sset), "n_bias": len(sbias)}


#: Fields whose change makes a baseline questionable, and the label to report.
#: Deliberately does NOT include bias or setpoint — the whole point of the
#: model is that it covers a range of those.
CONDITION_KEYS: tuple[tuple[str, str], ...] = (
    ("tip_id", "针尖"),
    ("sample_id", "样品"),
    ("scan_position", "扫描位"),
    ("magnet_on", "磁场"),
    ("pump_stm", "STM 腔分子泵"),
    ("pump_mbe", "MBE 腔分子泵"),
    ("fs_hz", "采样率"),
)


def condition_diff(stored: dict | None, current: dict | None) -> list[str]:
    """Which recorded conditions differ. An unknown on EITHER side is reported
    as unknown rather than as a match — see the module docstring."""
    out: list[str] = []
    s, c = stored or {}, current or {}
    for key, label in CONDITION_KEYS:
        a, b = s.get(key), c.get(key)
        if a is None or b is None:
            if a != b:
                out.append("%s(未知)" % label)
            continue
        if isinstance(a, float) or isinstance(b, float):
            try:
                if abs(float(a) - float(b)) > max(1e-9, abs(float(a)) * 1e-3):
                    out.append("%s %s→%s" % (label, a, b))
            except (TypeError, ValueError):
                out.append(label)
            continue
        if a != b:
            out.append("%s %s→%s" % (label, a, b))
    return out


__all__ = [
    "SigmaModel", "WhiteModel", "BaselineVerdict",
    "fit_sigma_model", "fit_white_model", "width_stats", "fwhm",
    "peak_metrics", "robust_white_floor", "scaling_exponent",
    "classify_mechanism", "MECHANISMS", "compare", "build_models",
    "polarity_check", "cross_spectra", "z_current_coupling", "z_stats",
    "condition_diff", "CONDITION_KEYS", "Q_E", "EXTRAPOLATION_LIMIT",
]
