"""Per-segment feature extraction for the tunnelling-current monitor.

Pure functions over a numpy array of amps — no hardware, no store, no config.
Every returned key is a column in ``store._SCHEMA``'s ``features`` table, so the
names here ARE the storage contract; renaming one means a schema migration.

Two conventions worth stating once:

* ``rms_a`` is the true RMS (DC included, ≈ |mean| for a tunnelling current);
  ``rms_detrended_a`` is the AC noise amplitude after removing a linear trend.
  The second one is the tip-health number — the first is kept because a drifting
  mean with a quiet AC part looks identical to a quiet mean on ``rms_a`` alone.
* PSD is averaged per contiguous run and never across a gap. A dropped frame
  between two runs is a discontinuity, and splicing across it would manufacture
  broadband power that the tip never produced.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Frequency-band edges (Hz) whose integrated power lands in its own column.
#: A decade-ish collapse of the 21-band diagnostic catalogue in
#: ``mast.knowledge.stm_noise`` — enough to separate 1/f, mains, mechanical and
#: white regimes without exploding the schema. The mains band (45-65) is split
#: out so ``line_ratio`` and the neighbours don't contaminate each other.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("band_0p1_1_a2", 0.1, 1.0),
    ("band_1_10_a2", 1.0, 10.0),
    ("band_10_45_a2", 10.0, 45.0),
    ("band_45_65_a2", 45.0, 65.0),
    ("band_65_200_a2", 65.0, 200.0),
    ("band_200_1k_a2", 200.0, 1000.0),
    ("band_1k_5k_a2", 1000.0, 5000.0),
    ("band_5k_nyq_a2", 5000.0, float("inf")),
)

#: Shortest run that gets an FFT. Below this the frequency resolution is so
#: coarse that the low bands are a single bin of noise.
_MIN_PSD_RUN = 256

#: 1/f fit window. Wide on purpose: measured on synthetic flicker noise, a
#: 1-40 Hz fit of a single 1 s record scatters with σ≈0.38 (individual segments
#: land anywhere from -1.6 to +0.4), because that window holds only ~40 raw
#: bins and each periodogram bin has ~100% variance. Extending to a tenth of
#: Nyquist brings ~25× more bins and drops σ to ≈0.06 around a correct -0.98.
#: Mains harmonics now fall inside the window, which is why the fit bins by
#: median rather than mean — a line spike perturbs one bin's median hardly at
#: all.
_INVF_LO_HZ = 1.0
_INVF_HI_CAP_HZ = 1000.0
_INVF_HI_NYQ_FRAC = 0.1

#: RTN gates. ``_RTN_DEPTH_FLOOR`` is the one that decides — see
#: :func:`_valley_depth` for why Ashman's D alone flags white noise. D is kept
#: as a secondary check (a deep valley between two levels that nearly touch is
#: still not a telegraph); its floor sits above the ~2.65 a split Gaussian
#: produces, so it can only ever tighten the verdict, never create one.
_RTN_DEPTH_FLOOR = 0.30
_RTN_D_FLOOR = 2.8


@dataclass(frozen=True)
class FeatureParams:
    """Knobs the feature layer needs. Populated from ``thresholds`` by the
    service; defaults here keep the module standalone-testable.

    ``jump_k`` is 8 from measurement, not from taste. The detection rate on pure
    Gaussian noise falls off a cliff between 5 and 7: at k=5 a clean 1 s segment
    at 20 kHz reports ~6 jumps/s (peaking at 10), which would put every quiet
    segment permanently over any sane rate threshold; at k=7 it is exactly zero
    while twenty injected 40 pA steps are still all found. 8 keeps that margin.
    The rate scales with the sample count, so this floor has to be re-checked if
    the segment length or sampling rate changes by an order of magnitude.
    """

    jump_k: float = 8.0
    spike_k: float = 8.0
    sat_current_a: float = 90e-9
    line_hz: float = 50.0
    env_buckets_per_s: int = 100


# ── small numeric helpers ───────────────────────────────────────────────────


def _as_1d(y) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _robust_sigma(x: np.ndarray) -> float:
    """1.4826·MAD — the spread of the bulk, immune to the very outliers we are
    trying to count."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Odd-length running median. scipy if present, sliding-window otherwise —
    this module must stay importable on a bare numpy install."""
    if x.size < k or k < 3:
        return x
    try:
        from scipy.signal import medfilt
        return np.asarray(medfilt(x, kernel_size=k), dtype=np.float64)
    except Exception:  # noqa: BLE001 — numpy fallback below
        pad = k // 2
        padded = np.pad(x, pad, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(padded, k)
        return np.median(win, axis=-1)


def _detrend_linear(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove a least-squares line; return (residual, slope per sample)."""
    n = x.size
    if n < 3:
        return x - (x.mean() if n else 0.0), 0.0
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept), float(slope)


def _count_runs(mask: np.ndarray) -> int:
    """Number of contiguous True stretches — one burst counts once, not once
    per sample."""
    if mask.size == 0 or not mask.any():
        return 0
    return int(np.count_nonzero(mask[1:] & ~mask[:-1]) + (1 if mask[0] else 0))


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


# ── feature groups ──────────────────────────────────────────────────────────


def basic_stats(y) -> dict:
    """Amplitude summary in amps. ``rms_a`` includes DC (see module docstring)."""
    arr = _as_1d(y)
    if arr.size == 0:
        return {k: None for k in
                ("mean_a", "median_a", "min_a", "max_a", "ptp_a", "rms_a")}
    return {
        "mean_a": float(arr.mean()),
        "median_a": float(np.median(arr)),
        "min_a": float(arr.min()),
        "max_a": float(arr.max()),
        "ptp_a": float(arr.max() - arr.min()),
        "rms_a": float(np.sqrt(np.mean(arr ** 2))),
    }


def detrended_rms(y, fs_hz: float) -> dict:
    """AC noise amplitude after removing a linear drift, plus that drift.

    A slow setpoint ramp or thermal drift would otherwise inflate the RMS and
    read as a noisy tip.
    """
    arr = _as_1d(y)
    if arr.size < 3 or fs_hz <= 0:
        return {"rms_detrended_a": None, "slope_a_per_s": None}
    resid, slope_per_sample = _detrend_linear(arr)
    return {
        "rms_detrended_a": float(resid.std()),
        "slope_a_per_s": float(slope_per_sample * fs_hz),
    }


def moments(y) -> dict:
    """Fisher kurtosis (0 = Gaussian) and skewness. Heavy tails from spikes or
    a two-state population both show up here before any dedicated detector."""
    arr = _as_1d(y)
    if arr.size < 4:
        return {"kurtosis": None, "skewness": None}
    resid, _ = _detrend_linear(arr)
    sigma = float(resid.std())
    if sigma <= 0:
        return {"kurtosis": 0.0, "skewness": 0.0}
    z = resid / sigma
    return {
        "kurtosis": float(np.mean(z ** 4) - 3.0),
        "skewness": float(np.mean(z ** 3)),
    }


def jump_metrics(y, fs_hz: float, k: float = 8.0) -> dict:
    """Abrupt steps: first differences beyond ``median(|Δ|) + k·1.4826·MAD``.

    Vectorised port of ``tip_shaper_readback._detect_jumps`` — same threshold,
    so a jump flagged during shaping and a jump flagged by the monitor mean the
    same thing.
    """
    arr = _as_1d(y)
    if arr.size < 3 or fs_hz <= 0:
        return {"jump_count": None, "jump_rate_hz": None, "max_step_a": None}
    diffs = np.diff(arr)
    absd = np.abs(diffs)
    med = float(np.median(absd))
    sigma = _robust_sigma(absd)
    if sigma <= 0:
        sigma = med if med > 0 else 1e-30
    thr = med + k * sigma
    mask = absd > thr
    dur_s = arr.size / fs_hz
    return {
        "jump_count": int(np.count_nonzero(mask)),
        "jump_rate_hz": float(np.count_nonzero(mask) / dur_s) if dur_s > 0 else 0.0,
        "max_step_a": float(absd.max()),
    }


def spike_metrics(y, fs_hz: float, k: float = 8.0) -> dict:
    """Transients measured against the robust centre of the segment.

    The baseline is ``detrend → subtract median``, and the scale is the MAD.
    A running median is deliberately NOT used: a real discharge lasts a
    millisecond or more, so any median window short enough to be cheap simply
    follows the spike and subtracts it away. The MAD tolerates far more
    contamination than a spike train ever produces (a few percent of samples),
    so the scale still reflects the quiet background.

    ``spike_max_sigma`` is the headline number: how far the worst excursion sits
    outside the bulk noise. Counting is per burst, not per sample, so one 3 ms
    discharge is one spike.
    """
    arr = _as_1d(y)
    if arr.size < 8:
        return {"spike_count": None, "spike_max_sigma": None}
    resid, _ = _detrend_linear(arr)
    resid = resid - float(np.median(resid))
    sigma = _robust_sigma(resid)
    if sigma <= 0:
        return {"spike_count": 0, "spike_max_sigma": 0.0}
    z = np.abs(resid) / sigma
    return {
        "spike_count": _count_runs(z > k),
        "spike_max_sigma": float(z.max()),
    }


def saturation_metrics(y, sat_current_a: float) -> dict:
    """How much of the segment sits at (or beyond) the preamp rail.

    ``railed_frac`` is the LONGEST constant stretch, not the total: a genuine
    rail hit is one continuous flat-top, whereas scattered repeats of the same
    value are just quantisation.
    """
    arr = _as_1d(y)
    if arr.size == 0 or sat_current_a <= 0:
        return {"sat_frac": None, "railed_frac": None}
    over = np.abs(arr) >= sat_current_a
    flat = np.zeros(arr.size, dtype=bool)
    flat[1:] = arr[1:] == arr[:-1]
    return {
        "sat_frac": float(np.count_nonzero(over) / arr.size),
        "railed_frac": float(_longest_run(flat) / arr.size),
    }


def freeze_metrics(y) -> dict:
    """A dead readout: the ADC (or the whole TCP path) handing back one value.

    Strict equality on min/max is deliberate. A real current, however quiet,
    always dithers in the last bits; an exactly constant trace means nobody is
    measuring. Same detector as ``CaptureSignalBuffer``'s frozen-reading check.
    """
    arr = _as_1d(y)
    if arr.size < 8:
        return {"frozen": None, "unique_frac": None}
    return {
        "frozen": 1 if float(arr.max()) == float(arr.min()) else 0,
        "unique_frac": float(np.unique(arr).size / arr.size),
    }


def _psd_of_runs(runs: Sequence[np.ndarray], fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch-style average of per-run one-sided PSDs. Returns (freqs, psd).

    Runs are truncated to the shortest usable length so every periodogram lands
    on the same frequency grid — averaging is then a plain mean, with no
    resampling to blur the bins.
    """
    from mast.io.signal_fft import compute_fft

    usable = [_as_1d(r) for r in runs]
    usable = [r for r in usable if r.size >= _MIN_PSD_RUN]
    if not usable or fs_hz <= 0:
        return np.empty(0), np.empty(0)
    n = min(int(r.size) for r in usable)
    freqs: np.ndarray | None = None
    acc: np.ndarray | None = None
    count = 0
    for r in usable:
        out = compute_fft(
            {"samples": r[:n].tolist(), "fs_hz": fs_hz, "unit": "A"},
            window="hann", detrend=True, output="power",
        )
        if not out:
            continue
        vals = np.asarray(out["spectrum"], dtype=np.float64)
        if freqs is None:
            freqs = np.asarray(out["freqs_hz"], dtype=np.float64)
            acc = vals
        elif vals.shape == acc.shape:
            acc = acc + vals
        else:
            continue
        count += 1
    if freqs is None or acc is None or count == 0:
        return np.empty(0), np.empty(0)
    return freqs, acc / count


def psd_features(runs: Sequence[np.ndarray], fs_hz: float,
                 line_hz: float = 50.0) -> dict:
    """Band powers, mains contamination and the 1/f slope.

    ``line_ratio`` compares the mains window against the median of its
    neighbours rather than against an absolute level, so it stays meaningful
    whatever the overall noise floor is.
    """
    empty = {name: None for name, _, _ in BANDS}
    empty.update({"line_power_a2": None, "line_ratio": None,
                  "inv_f_slope": None, "inv_f_r2": None,
                  "white_floor_a2hz": None})
    freqs, psd = _psd_of_runs(runs, fs_hz)
    if freqs.size < 4:
        return empty

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    nyq = float(freqs[-1])
    out: dict = {}

    for name, lo, hi in BANDS:
        hi_eff = min(hi, nyq) if math.isfinite(hi) else nyq
        if lo >= hi_eff:
            out[name] = None
            continue
        sel = (freqs >= lo) & (freqs < hi_eff)
        out[name] = float(psd[sel].sum() * df) if sel.any() else 0.0

    # Mains: mean power DENSITY in a ±2 Hz window against the same quantity in
    # the surrounding sidebands.
    #
    # Not peak-over-median, which was the first attempt: periodogram bins are
    # exponentially distributed, so the maximum of five bins over the median of
    # seventeen has a heavy tail — measured p99 ≈ 12 on traces with no mains
    # component at all, which put 5% of perfectly healthy segments over a
    # threshold of 10. Averaging within each band instead drops the noise-only
    # p99 to ≈3 while a 1 pA line still reads ≈20: same sensitivity, far less
    # variance, because a real mains line puts its energy inside the window
    # either way.
    half = 2.0
    win = (freqs >= line_hz - half) & (freqs <= line_hz + half)
    side = (((freqs >= line_hz - 5 * half) & (freqs < line_hz - half)) |
            ((freqs > line_hz + half) & (freqs <= line_hz + 5 * half)))
    if win.any():
        out["line_power_a2"] = float(psd[win].sum() * df)
        n_side = int(np.count_nonzero(side))
        base = (float(psd[side].sum()) / n_side) if n_side else 0.0
        band = float(psd[win].sum()) / int(np.count_nonzero(win))
        out["line_ratio"] = float(band / base) if base > 0 else None
    else:
        out["line_power_a2"] = None
        out["line_ratio"] = None

    # 1/f: log-log slope. Flicker noise sits near -1; a tip that has picked
    # something up often steepens.
    #
    # Fitted on log-spaced bin MEDIANS weighted by how many raw bins each one
    # covers — not on the raw bins. On a linear frequency grid the decade
    # 1-10 Hz contributes ten points against the ~900 in 100-1000 Hz, so an
    # unbinned fit is dominated by the top of the window; log binning equalises
    # the leverage, the median resists mains harmonics inside the window, and
    # the √count weights keep the sparse low-frequency bins from dictating the
    # slope. See _INVF_* for the measured variance this buys.
    lo = max(_INVF_LO_HZ, df)
    hi = min(_INVF_HI_CAP_HZ, nyq * _INVF_HI_NYQ_FRAC)
    fit = (freqs >= lo) & (freqs <= hi) & (psd > 0)
    if hi > lo * 2 and int(np.count_nonzero(fit)) >= 8:
        lx_all = np.log10(freqs[fit])
        ly_all = np.log10(psd[fit])
        n_bins = int(min(16, max(4, np.count_nonzero(fit) // 3)))
        edges = np.linspace(float(lx_all[0]), float(lx_all[-1]), n_bins + 1)
        idx = np.clip(np.digitize(lx_all, edges[1:-1]), 0, n_bins - 1)
        lx, ly, wt = [], [], []
        for b in range(n_bins):
            sel = idx == b
            count = int(np.count_nonzero(sel))
            if count >= 1:
                lx.append(float(lx_all[sel].mean()))
                ly.append(float(np.median(ly_all[sel])))
                wt.append(count)
        if len(lx) >= 4:
            lxa = np.asarray(lx)
            lya = np.asarray(ly)
            w = np.sqrt(np.asarray(wt, dtype=np.float64))
            slope, intercept = np.polyfit(lxa, lya, 1, w=w)
            pred = slope * lxa + intercept
            ss_res = float(np.sum(w * (lya - pred) ** 2))
            wmean = float(np.sum(w * lya) / np.sum(w))
            ss_tot = float(np.sum(w * (lya - wmean) ** 2))
            out["inv_f_slope"] = float(slope)
            out["inv_f_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
        else:
            out["inv_f_slope"] = None
            out["inv_f_r2"] = None
    else:
        out["inv_f_slope"] = None
        out["inv_f_r2"] = None

    # White floor: median density of the top decade, where 1/f has died out.
    top = freqs >= (nyq / 10.0)
    out["white_floor_a2hz"] = float(np.median(psd[top])) if top.any() else None
    return out


def _valley_depth(x: np.ndarray, mu_lo: float, mu_hi: float) -> float:
    """How empty the region between two candidate levels is. 0 = unimodal.

    This is the test that decides whether a telegraph exists at all, and it is
    deliberately not Ashman's D. Splitting ANY distribution into two clusters
    "succeeds" — for a plain Gaussian, k-means lands the two centres at ±0.8σ
    and Ashman's D comes out around 2.65, comfortably past the textbook
    bimodality threshold of 2. Feeding white noise to that criterion reports a
    busy telegraph on every quiet segment.

    A real two-state signal instead leaves the space BETWEEN the levels empty.
    So: take the smoothed histogram, compare its minimum inside [mu_lo, mu_hi]
    against the smaller of the two densities at the level positions. For a
    unimodal distribution the interior minimum sits at an endpoint (the middle
    is the mode) and the ratio is ~1, giving depth ~0. For separated states the
    valley approaches zero and the depth approaches 1.
    """
    if mu_hi <= mu_lo or x.size < 64:
        return 0.0
    bins = int(min(80, max(24, x.size // 200)))
    hist, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([0.25, 0.5, 0.25])          # tame per-bin counting noise
    sm = np.convolve(hist.astype(np.float64), kernel, mode="same")
    inside = (centers >= mu_lo) & (centers <= mu_hi)
    if int(np.count_nonzero(inside)) < 3:
        return 0.0
    p_lo = float(sm[int(np.argmin(np.abs(centers - mu_lo)))])
    p_hi = float(sm[int(np.argmin(np.abs(centers - mu_hi)))])
    peak = min(p_lo, p_hi)
    if peak <= 0:
        return 0.0
    valley = float(sm[inside].min())
    return float(max(0.0, 1.0 - valley / peak))


def _two_means(x: np.ndarray, iters: int = 25) -> tuple[float, float, float, float, float]:
    """1-D Lloyd's algorithm. Returns (mu_lo, mu_hi, sd_lo, sd_hi, frac_hi)."""
    lo_seed, hi_seed = np.percentile(x, [15, 85])
    if hi_seed <= lo_seed:
        return float(x.mean()), float(x.mean()), float(x.std()), float(x.std()), 0.0
    mu_lo, mu_hi = float(lo_seed), float(hi_seed)
    hi_mask = x > (mu_lo + mu_hi) / 2.0
    for _ in range(iters):
        new_hi = x > (mu_lo + mu_hi) / 2.0
        if not new_hi.any() or new_hi.all():
            break
        hi_mask = new_hi
        mu_lo = float(x[~hi_mask].mean())
        mu_hi = float(x[hi_mask].mean())
    if not hi_mask.any() or hi_mask.all():
        return float(x.mean()), float(x.mean()), float(x.std()), float(x.std()), 0.0
    return (mu_lo, mu_hi,
            float(x[~hi_mask].std()), float(x[hi_mask].std()),
            float(np.count_nonzero(hi_mask) / x.size))


def rtn_metrics(y, fs_hz: float) -> dict:
    """Random-telegraph (two-level) switching — the classic unstable-apex tell.

    Split the samples into two clusters, score the separation with Ashman's D,
    and only then count transitions. The gate matters: fitting two Gaussians to
    one always "succeeds", so without the D floor pure white noise would report
    a busy telegraph. ``rtn_score`` therefore stays ≈0 for a unimodal trace,
    which the synthetic tests pin down.

    **Operating range** (measured on synthetic traces, 25 pA level spacing
    against varying 1/f background, 20 samples each):

        spacing / noise    detected     false alarms
              ≥ 8              95%           0%
                5              85%           0%
                3              20%           0%
                2               0%           0%

    So this is reliable while the two levels are separated by roughly five times
    the background, and below three it reports nothing — not because the
    detector fails, but because the two populations genuinely overlap in the
    histogram by then. False alarms stay at zero throughout, which is the
    trade this detector is tuned for: a monitor that invents telegraph noise is
    worse than one that misses a marginal case, since five other features
    (detrended RMS, peak-to-peak, 1/f slope, jump rate, low-band power) separate
    a bad tip from a good one with far more margin anyway.
    """
    zero = {"rtn_score": 0.0, "rtn_gap_a": 0.0, "rtn_rate_hz": 0.0,
            "rtn_dwell_hi_ms": 0.0, "rtn_dwell_lo_ms": 0.0, "rtn_transitions": 0}
    none = {k: None for k in zero}
    arr = _as_1d(y)
    if arr.size < 64 or fs_hz <= 0:
        return none
    resid, _ = _detrend_linear(arr)
    x = _median_filter(resid, 5)
    if float(x.std()) <= 0:
        return zero

    mu_lo, mu_hi, sd_lo, sd_hi, frac_hi = _two_means(x)
    gap = abs(mu_hi - mu_lo)
    denom = math.sqrt(sd_lo ** 2 + sd_hi ** 2)
    d = (math.sqrt(2.0) * gap / denom) if denom > 0 else 0.0
    depth = _valley_depth(x, mu_lo, mu_hi)
    if (depth < _RTN_DEPTH_FLOOR or d < _RTN_D_FLOOR
            or frac_hi <= 0.02 or frac_hi >= 0.98):
        return zero

    # Schmitt trigger at the midpoint: without hysteresis, noise around the
    # threshold would be counted as thousands of transitions.
    #
    # Vectorised rather than looped. The state at each sample is "which
    # threshold did we most recently cross" — so run a running maximum of the
    # indices where each threshold was crossed and compare them. A per-sample
    # Python loop over a 1 s segment at 20 kHz measured 278 ms, which is
    # ~28× everything else in this module combined and long enough to stall
    # acquisition for several oscilloscope buffers.
    mid = 0.5 * (mu_lo + mu_hi)
    hyst = gap / 4.0
    hi_thr, lo_thr = mid + hyst, mid - hyst
    n = x.size
    idx = np.arange(n)
    last_hi = np.maximum.accumulate(np.where(x > hi_thr, idx, -1))
    last_lo = np.maximum.accumulate(np.where(x < lo_thr, idx, -1))
    state = last_hi > last_lo
    # Before either threshold is first crossed neither index is set; seed those
    # samples from where the trace started.
    unseeded = (last_hi < 0) & (last_lo < 0)
    if unseeded.any():
        state[unseeded] = bool(x[0] > mid)

    changes = np.flatnonzero(np.diff(state)) + 1
    transitions = int(changes.size)
    bounds = np.concatenate(([0], changes, [n]))
    run_lengths = np.diff(bounds)
    run_states = state[bounds[:-1]]
    dwell_hi = run_lengths[run_states]
    dwell_lo = run_lengths[~run_states]

    dur_s = arr.size / fs_hz
    to_ms = 1000.0 / fs_hz
    return {
        "rtn_score": float(1.0 / (1.0 + math.exp(-(depth - 0.5) / 0.12))),
        "rtn_gap_a": float(gap),
        "rtn_rate_hz": float(transitions / dur_s) if dur_s > 0 else 0.0,
        "rtn_dwell_hi_ms": float(dwell_hi.mean() * to_ms) if dwell_hi.size else 0.0,
        "rtn_dwell_lo_ms": float(dwell_lo.mean() * to_ms) if dwell_lo.size else 0.0,
        "rtn_transitions": int(transitions),
    }


def envelope(y, fs_hz: float, buckets_per_s: int = 100) -> np.ndarray:
    """(2, K) float32 min/max envelope — row 0 min, row 1 max.

    This is what survives the retention sweep after the raw ``.npy`` is deleted,
    and what the live chart draws. Min/max (not mean) because a spike that
    averages away is exactly the event worth keeping.
    """
    arr = _as_1d(y)
    if arr.size == 0 or fs_hz <= 0 or buckets_per_s <= 0:
        return np.zeros((2, 0), dtype=np.float32)
    per_bucket = max(1, int(round(fs_hz / buckets_per_s)))
    k = max(1, arr.size // per_bucket)
    trimmed = arr[:k * per_bucket].reshape(k, per_bucket)
    out = np.empty((2, k), dtype=np.float32)
    out[0] = trimmed.min(axis=1)
    out[1] = trimmed.max(axis=1)
    return out


def _per_run_jumps(runs: Sequence[np.ndarray], fs_hz: float, k: float) -> dict:
    """:func:`jump_metrics` over each contiguous run, combined.

    Counts add up, the largest step is the largest step seen INSIDE any run, and
    the rate is over the total time actually sampled — none of which can see the
    joint between two runs.
    """
    parts = [jump_metrics(r, fs_hz, k) for r in runs if r.size >= 3]
    parts = [p for p in parts if p.get("jump_count") is not None]
    if not parts:
        return {"jump_count": None, "jump_rate_hz": None, "max_step_a": None}
    total = sum(int(p["jump_count"]) for p in parts)
    n_samples = sum(int(r.size) for r in runs if r.size >= 3)
    dur = n_samples / fs_hz if fs_hz > 0 else 0.0
    return {
        "jump_count": total,
        "jump_rate_hz": float(total / dur) if dur > 0 else 0.0,
        "max_step_a": float(max(p["max_step_a"] for p in parts)),
    }


def _per_run_spikes(runs: Sequence[np.ndarray], fs_hz: float, k: float) -> dict:
    """:func:`spike_metrics` over each contiguous run, combined."""
    parts = [spike_metrics(r, fs_hz, k) for r in runs if r.size >= 8]
    parts = [p for p in parts if p.get("spike_count") is not None]
    if not parts:
        return {"spike_count": None, "spike_max_sigma": None}
    return {
        "spike_count": sum(int(p["spike_count"]) for p in parts),
        "spike_max_sigma": float(max(p["spike_max_sigma"] for p in parts)),
    }


def compute_segment_features(runs: Iterable[np.ndarray], fs_hz: float,
                             params: FeatureParams | None = None) -> dict:
    """Every feature for one segment, flattened to the ``features`` columns.

    Amplitude/shape features see the concatenated samples (a gap does not
    invalidate a variance); PSD sees the runs separately (a gap does invalidate
    a spectrum).
    """
    p = params or FeatureParams()
    run_list = [_as_1d(r) for r in runs]
    run_list = [r for r in run_list if r.size]
    if not run_list:
        return {}
    flat = np.concatenate(run_list) if len(run_list) > 1 else run_list[0]

    feats: dict = {"fs_hz": float(fs_hz)}
    feats.update(basic_stats(flat))
    feats.update(detrended_rms(flat, fs_hz))
    feats.update(moments(flat))
    # Step-like features go per run, for the same reason the PSD does: the joint
    # between two runs is a hole in time, and a first difference taken across it
    # manufactures a step equal to however much the current moved while we were
    # not looking (a bias change, a setpoint ramp, an approach advancing). That
    # invented step feeds `giant_spike`, which is one of the three rules allowed
    # to halt a running composite skill.
    feats.update(_per_run_jumps(run_list, fs_hz, p.jump_k))
    feats.update(_per_run_spikes(run_list, fs_hz, p.spike_k))
    feats.update(saturation_metrics(flat, p.sat_current_a))
    feats.update(freeze_metrics(flat))
    try:
        feats.update(psd_features(run_list, fs_hz, p.line_hz))
    except Exception:  # noqa: BLE001 — a bad spectrum must not lose the segment
        logger.debug("psd_features failed (swallowed)", exc_info=True)
        feats.update({name: None for name, _, _ in BANDS})
        feats.update({"line_power_a2": None, "line_ratio": None,
                      "inv_f_slope": None, "inv_f_r2": None,
                      "white_floor_a2hz": None})
    try:
        feats.update(rtn_metrics(flat, fs_hz))
    except Exception:  # noqa: BLE001
        logger.debug("rtn_metrics failed (swallowed)", exc_info=True)
        feats.update({"rtn_score": None, "rtn_gap_a": None, "rtn_rate_hz": None,
                      "rtn_dwell_hi_ms": None, "rtn_dwell_lo_ms": None,
                      "rtn_transitions": None})
    return feats


#: Column order the store writes and the exporter reads.
FEATURE_COLUMNS: tuple[str, ...] = (
    "mean_a", "median_a", "min_a", "max_a", "ptp_a",
    "rms_a", "rms_detrended_a", "slope_a_per_s", "kurtosis", "skewness",
    "jump_count", "jump_rate_hz", "max_step_a",
    "spike_count", "spike_max_sigma",
    *[name for name, _, _ in BANDS],
    "line_power_a2", "line_ratio", "inv_f_slope", "inv_f_r2", "white_floor_a2hz",
    "rtn_score", "rtn_gap_a", "rtn_rate_hz",
    "rtn_dwell_hi_ms", "rtn_dwell_lo_ms", "rtn_transitions",
    "sat_frac", "railed_frac", "frozen", "unique_frac",
)

#: Public alias for the per-run Welch average. The environment-history recorder
#: needs the SPECTRUM itself, not the band powers :func:`psd_features` reduces it
#: to — and it must be the same estimator, computed per contiguous run and never
#: spliced across a gap, or the two views of "the noise floor" would disagree.
psd_of_runs = _psd_of_runs

__all__ = [
    "BANDS", "FEATURE_COLUMNS", "FeatureParams",
    "basic_stats", "detrended_rms", "moments", "jump_metrics", "spike_metrics",
    "saturation_metrics", "freeze_metrics", "psd_features", "psd_of_runs",
    "rtn_metrics", "envelope", "compute_segment_features",
]
