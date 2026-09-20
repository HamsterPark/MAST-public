"""Δf(z) → F(z), U(z): the Sader–Jarvis inversion, and how far to trust it.

A frequency-modulation AFM records a frequency shift, not a force. Sader and Jarvis (Appl.
Phys. Lett. 84, 1801, 2004) gave the closed-form inversion that turns one into the other at
any oscillation amplitude:

    F(z) = 2k ∫_z^∞ [ (1 + √A/(8√(π(t−z)))) Ω(t) − A^{3/2}/√(2(t−z)) Ω′(t) ] dt,  Ω = Δf/f₀

Both correction terms diverge as ``t → z``; the first interval is integrated analytically
against ``1/√(t−z)`` and the rest by trapezoid, which is what keeps the result stable at the
first few points.

Two things are reported alongside the numbers, because a force curve that looks fine can still
be meaningless:

* **forward residual** — the inverted force is pushed back through the forward integral and
  compared with the measured Δf. A large residual means the inversion did not describe the
  data, whatever the curve looks like.
* **well-posedness** — Sader et al. (Nat. Nanotechnol. 13, 1088, 2018) showed that force laws
  varying faster than the amplitude cannot be recovered from Δf at all. The ratio of amplitude
  to the measured decay length is reported with a three-way label; **the thresholds here are
  not calibrated**, so this is advice to repeat the measurement at another amplitude, never a
  verdict on the numbers.

Pure functions: arrays and thresholds in, a frozen result out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

EV_J = 1.602176634e-19
#: A/λ below this is comfortably small-amplitude; above the second the inversion is suspect.
#: Uncalibrated: they label the situation, they do not decide the answer.
WELL_POSED_MAX = 0.3
CAUTION_MAX = 3.0
#: a forward residual worse than this means the inversion did not describe the data
MAX_FORWARD_RESIDUAL = 0.10


@dataclass(frozen=True)
class ForceInversionResult:
    verdict: str                          # well / no_well / undecidable
    f_min_n: float | None = None
    z_f_min_m: float | None = None
    z_df_min_m: float | None = None
    z_offset_fmin_minus_dfmin_m: float | None = None
    e_bind_ev: float | None = None
    decay_length_m: float | None = None
    forward_residual: float | None = None
    amplitude_over_decay_length: float | None = None
    well_posedness: str = ""
    background_used: bool = False
    n_points: int = 0
    z_m: tuple[float, ...] = ()
    force_n: tuple[float, ...] = ()
    energy_ev: tuple[float, ...] = ()
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "well"

    @property
    def f_min_pn(self) -> float | None:
        return None if self.f_min_n is None else self.f_min_n * 1e12

    @property
    def e_bind_mev(self) -> float | None:
        return None if self.e_bind_ev is None else self.e_bind_ev * 1e3

    @property
    def decay_length_pm(self) -> float | None:
        return None if self.decay_length_m is None else self.decay_length_m * 1e12


def sader_jarvis(z: npt.ArrayLike, df: npt.ArrayLike, *, f0_hz: float, k_n_per_m: float,
                 amplitude_m: float) -> npt.NDArray[np.float64]:
    """Force (N) at each ``z`` (m, increasing away from the surface) from Δf (Hz)."""
    z = np.asarray(z, dtype=float)
    omega = np.asarray(df, dtype=float) / float(f0_hz)
    d_omega = np.gradient(omega, z)
    a = float(amplitude_m)
    out = np.zeros_like(z)
    n = z.size
    c1 = math.sqrt(a) / (8 * math.sqrt(math.pi))       # weight of the Ω/√(t−z) correction
    c2 = a ** 1.5 / math.sqrt(2.0)                     # weight of the Ω′/√(t−z) correction
    for i in range(n - 1):
        t = z[i + 1:]
        dt = t - z[i]
        # the integral starts at z_i, where both corrections diverge as 1/√(t−z_i). The first
        # interval is done analytically against that weight (∫₀^h dt/√t = 2√h) and the rest,
        # which is regular, by trapezoid.
        h = float(dt[0])
        head = float(omega[i]) * h \
            + 2 * math.sqrt(h) * (c1 * float(omega[i]) - c2 * float(d_omega[i]))
        integ = (1.0 + c1 / np.sqrt(dt)) * omega[i + 1:] - c2 * d_omega[i + 1:] / np.sqrt(dt)
        rest = float(np.trapezoid(integ, t)) if t.size > 1 else 0.0
        out[i] = 2.0 * float(k_n_per_m) * (head + rest)
    out[-1] = out[-2] if n > 1 else 0.0
    return out


def forward_df(z: npt.ArrayLike, force, *, f0_hz: float, k_n_per_m: float,
               amplitude_m: float, n_nodes: int = 64) -> npt.NDArray[np.float64]:
    """Δf a given force law would produce — the check that closes the loop."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    a = float(amplitude_m)
    if a < 1e-13:
        h = 0.5e-12
        grad = (force(z + h) - force(z - h)) / (2 * h)
        return -(float(f0_hz) / (2 * float(k_n_per_m))) * grad
    j = np.arange(1, n_nodes + 1)
    u = np.cos((2 * j - 1) * np.pi / (2 * n_nodes))
    zz = z[:, None] + a * (1.0 + u)[None, :]
    return -(float(f0_hz) / (float(k_n_per_m) * a * n_nodes)) * (force(zz) @ u)


def _decay_length(z: npt.NDArray[np.float64], f: npt.NDArray[np.float64],
                  i_min: int) -> float | None:
    """Exponential decay length of the attractive tail beyond the force minimum."""
    tail = slice(i_min + 1, min(i_min + 1 + max(20, (f.size - i_min) // 2), f.size))
    zz, ff = z[tail], -f[tail]
    ok = ff > 0
    if ok.sum() < 6:
        return None
    coef = np.polyfit(zz[ok], np.log(ff[ok]), 1)
    return float(-1.0 / coef[0]) if coef[0] < 0 else None


def invert_force_curve(z_m: npt.ArrayLike, df_hz: npt.ArrayLike, *, f0_hz: float,
                       k_n_per_m: float, amplitude_m: float,
                       background_df_hz: npt.ArrayLike | None = None,
                       smooth_points: int = 0) -> ForceInversionResult:
    """The whole measurement: subtract the background, invert, and say how far to trust it."""
    warns: list[str] = []
    z = np.asarray(z_m, dtype=float).ravel()
    df = np.asarray(df_hz, dtype=float).ravel()
    if z.size != df.size:
        return ForceInversionResult("undecidable", reasons=("shape_mismatch",))
    good = np.isfinite(z) & np.isfinite(df)
    z, df = z[good], df[good]
    if z.size < 20:
        return ForceInversionResult("undecidable", n_points=int(z.size),
                                    reasons=("too_few_points",))
    order = np.argsort(z)
    z, df = z[order], df[order]
    background_used = False
    if background_df_hz is not None:
        bg = np.asarray(background_df_hz, dtype=float).ravel()
        if bg.size == z.size:
            df = df - bg
            background_used = True
        else:
            warns.append("background_length_mismatch")
    if smooth_points and smooth_points >= 5:
        from scipy.signal import savgol_filter

        w = int(smooth_points) | 1
        if w < z.size:
            df = savgol_filter(df, w, 3)
    i_df = int(np.argmin(df))
    if i_df in (0, z.size - 1):
        warns.append("df_min_at_edge")
    f = sader_jarvis(z, df, f0_hz=f0_hz, k_n_per_m=k_n_per_m, amplitude_m=amplitude_m)
    # the last stretch carries the truncation error of the semi-infinite integral
    tail = max(4, z.size // 10)
    core = slice(1, z.size - tail)
    i_f = int(np.argmin(f[core])) + core.start
    f_min = float(f[i_f])
    # U(z) = −∫_z^∞ F: referenced to zero far away
    u = -np.concatenate([[0.0], np.cumsum(np.diff(z) * 0.5 * (f[:-1] + f[1:]))])
    u = u - u[-1]
    e_bind = float(-np.min(u[core]))
    lam = _decay_length(z, f, i_f)
    ratio = (float(amplitude_m) / lam) if lam and lam > 0 else None
    if ratio is None:
        posed = "unknown"
    elif ratio < WELL_POSED_MAX:
        posed = "small_amplitude"
    elif ratio < CAUTION_MAX:
        posed = "caution"
    else:
        posed = "large_amplitude"

    interp = np.interp
    back = forward_df(z, lambda zz: interp(np.asarray(zz).ravel(), z, f).reshape(np.shape(zz)),
                      f0_hz=f0_hz, k_n_per_m=k_n_per_m, amplitude_m=amplitude_m)
    scale = float(np.max(np.abs(df))) or 1.0
    residual = float(np.sqrt(np.mean((back[core] - df[core]) ** 2))) / scale

    # "is there a well" compares the depth against the scatter far from the surface, where
    # there is no short-range force left to find
    far = f[max(core.start, core.stop - max(10, z.size // 5)):core.stop]
    scatter = float(np.std(far - np.mean(far))) if far.size > 3 else 0.0
    reasons: list[str] = []
    verdict = "well"
    if "df_min_at_edge" in warns:
        # the sweep never went past the turning point, so the well was not bracketed and the
        # depth that comes out is a lower bound on something, not a measurement of the well
        verdict, reasons = "undecidable", ["minimum_not_bracketed"]
    elif f_min >= 0:
        verdict, reasons = "no_well", ["no_attractive_minimum"]
    elif abs(f_min) < 3 * scatter:
        verdict, reasons = "no_well", ["minimum_within_noise"]
    elif posed in ("caution", "large_amplitude") and residual > MAX_FORWARD_RESIDUAL:
        verdict, reasons = "undecidable", ["inversion_does_not_describe_the_data"]
    if posed == "large_amplitude":
        warns.append("amplitude_exceeds_the_force_decay_length")
    return ForceInversionResult(
        verdict, f_min_n=f_min, z_f_min_m=float(z[i_f]), z_df_min_m=float(z[i_df]),
        z_offset_fmin_minus_dfmin_m=float(z[i_f] - z[i_df]),
        e_bind_ev=e_bind / EV_J, decay_length_m=lam, forward_residual=residual,
        amplitude_over_decay_length=ratio, well_posedness=posed,
        background_used=background_used, n_points=int(z.size),
        z_m=tuple(float(v) for v in z), force_n=tuple(float(v) for v in f),
        energy_ev=tuple(float(v) / EV_J for v in u),
        reasons=tuple(reasons), warnings=tuple(warns))
