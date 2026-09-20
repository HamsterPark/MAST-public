"""Mid-scan tip-change detection v2 — lag-k differenced, null-calibrated.

An STM tip very often changes *during* a scan (apex picks up/drops an atom):
every row below the event images with a different apex. For an autonomous
system this is critical — the scan below the change is untrustworthy and the
scan should be aborted early (half-frame loss instead of full-frame loss).

Why v2 (physics-truth validation, ``docs/v2/benchmarks/vigil_truth_validation/``):
the old single-split max-t detector had its null hypothesis wrong. Per-row
feature series on *no-change* frames are not stationary white noise but slow
trends + autocorrelated noise (thermal drift, creep, feedback settling); a
length-H linear trend grows the pooled t like √H, so at 512 rows an invisible
trend beats any fixed threshold — the test was really detecting "is there a
trend". Measured: AUC 0.510 (= random) at controlled FPR; its old "90 %
detections" were bought by an 83 % false-positive rate.

v2 moves the slow trend INTO the null hypothesis:

    per-row channels → lag-k difference (k ≥ max transition width) →
    running-median detrend → MAD normalise → per-channel |z| →
    null-library calibration (log-domain z vs no-change frames) → best channel

Channels (all O(N) or O(N·FFT_row)): ``dc`` (deplaned row median — the
dominant z-offset signature, discarded entirely by the old detector), ``rms``,
``hf`` (high-band fraction), ``ncc`` (adjacent-row correlation), ``tr``
(trace/retrace mismatch, when a retrace is given), ``bragg_amp``/``bragg_ph``
(per-row lattice demodulation, when a usable Bragg peak exists).

Validated on VIGIL physical ground truth (C1 n_pos=695 / C2 n_pos=1693):
AUC 0.668/0.624 over all labelled events, **0.971/0.878 on the visible
subset** (events whose noise-free rendering actually contains a break —
36 %/19 % of labels; the rest have median effect 2 pm, physically
undetectable by any image detector). At FPR 5 %: 94.5 % of visible C1 events.

The per-frame LOD (~6×MAD of the lag-k dc differences) states, in input z
units, the smallest row-DC jump THIS frame could have shown. Single-atom
steps are ~200+ pm and typical contamination z-offsets tens of pm, so on a
typical frame the events that matter are within reach — and "no change
detected" becomes falsifiable: sensitive down to `lod`, nothing seen.

The strictly-causal :func:`cusum_online` scores the same dc channel with a
trailing null (detection delay ≈ 0 rows on VIGIL C1) for a future streaming
deployment; note its feature extraction is still whole-frame (deplane).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter

from mast.vision.module import TipChangeResult

EPS = 1e-12

# Pre-registered hyper-parameters (priors, NOT tuned on the validation data):
#   K        ≥ the largest ground-truth transition width (10 rows)
#   TREND_WIN ≈ 8-10× the transition width; slower drifts join the null
#   med3     3-row median on the combined statistic (events are ≥K-wide plateaus)
K = 12
TREND_WIN = 101

# Null-library calibration (log-domain median/MAD of per-channel no-change
# peaks), measured on the VIGIL physics corpora (500 calibration frames each,
# calibration/eval split by frame-index parity; scratchpad rebuild_calib.py of
# 2026-07-27). The channel peaks being calibrated are already frame-internal
# MAD-normalised |z| maxima — dimensionless extreme-value statistics — so the
# table transfers across data sources far better than any raw amplitude would.
# Re-calibration on the instrument: collect no-change frames, recompute
# log-med/log-MAD per channel, swap the table.
_CALIB: dict[str, dict[str, tuple[float, float]]] = {
    # atomic scale (VIGIL C1, 3-15 nm scans, nm/px 0.006-0.03)
    "vigil-c1": {
        "dc": (1.1318, 0.1451), "rms": (1.2592, 0.2728), "hf": (1.2843, 0.3029),
        "ncc": (1.6631, 0.4862), "tr": (1.5435, 0.5288),
        "bragg_amp": (1.3505, 0.5143), "bragg_ph": (1.8646, 0.9359),
    },
    # meso scale (VIGIL C2, 10-279 nm scans, nm/px 0.02-0.54)
    "vigil-c2": {
        "dc": (1.2755, 0.2872), "rms": (1.4174, 0.4823), "hf": (1.3901, 0.3569),
        "ncc": (2.0462, 0.6350), "tr": (1.6728, 0.6919),
        "bragg_amp": (1.2354, 0.4033), "bragg_ph": (2.0259, 0.8307),
    },
}

# Default thresholds on the calibrated z, per null table — ≈ the measured
# FPR 1 % operating point (C1 tau 6.70, C2 tau 9.77). A CRITICAL "abort the
# scan" alert must be stingy with false alarms; the measured FPR gradient
# across scale (2-4×) is why the meso threshold is higher.
_DEFAULT_TAU = {"vigil-c1": 7.0, "vigil-c2": 10.0}

# nm/px above which the meso null table + threshold apply (C1 tops out at
# 0.0293, C2 starts at 0.0197; 0.03 is the natural boundary).
_MESO_NMPP = 0.03


# ────────────────────────────────────────────────────────────────────
# input adaptation & core statistics
# ────────────────────────────────────────────────────────────────────

def _to_pair(image: npt.NDArray) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """(trace, retrace|None) from (H,W) · (1,H,W) · (2,H,W) · (H,W,3)."""
    a = np.asarray(image)
    if a.ndim == 3:
        if a.shape[0] == 2:
            return (np.ascontiguousarray(a[0], dtype=np.float64),
                    np.ascontiguousarray(a[1], dtype=np.float64))
        if a.shape[0] == 1:
            a = a[0]
        elif a.shape[-1] == 3:
            a = a.mean(axis=-1)
        else:
            raise ValueError(f"cannot interpret 3-D image of shape {a.shape}")
    if a.ndim != 2:
        raise ValueError(f"image must be 2-D or 3-D, got {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float64), None


def deplane(h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Least-squares plane removal (global tilt — deliberately not per-row)."""
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(h.size)]
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    return h - (A @ coef).reshape(H, W)


def _robust_z(d: npt.NDArray[np.float64], trend_win: int,
              mad_floor: float = 0.0) -> npt.NDArray[np.float64]:
    """Detrend a lag-k difference series with a running median, MAD-normalise.

    ``mad_floor`` is a NUMERICAL guard only (float quantisation on noise-free
    synthetic input) — physical frames always carry a continuous noise floor,
    and it must never bind on them. A heavier "bimodal" quantile guard was
    tried and REMOVED here (2026-07-27): the dc channel of real frames is
    intrinsically two-component (flat-terrace rows at pm noise + topography /
    event rows at hundreds of pm, measured q90/MAD up to 136×), so any guard
    keyed on that shape also crushes genuine events — it cost 22 points of
    visible-event recall on the VIGIL re-verification. Pathological
    noise-free periodic synthetics can still nudge one channel over
    threshold; that is an accepted, documented edge (real instruments do not
    produce them, and mid-scan alerts are deduped per scan)."""
    if trend_win >= 3 and len(d) > trend_win:
        d = d - median_filter(d, size=trend_win, mode="reflect")
    med = np.median(d)
    mad = max(float(np.median(np.abs(d - med))) * 1.4826, mad_floor) + EPS
    return (d - med) / mad


def _lagk_z(f: npt.NDArray[np.float64], k: int, trend_win: int) -> npt.NDArray[np.float64]:
    """|robust z| of the lag-k difference, re-centred to event rows (length H).

    d[i] = f[i+k] − f[i] responds to an event at row r for i in (r−k, r): a
    step becomes a width-k plateau centred at r − k/2 in i-space, i.e. at r
    after the +k//2 shift below. Edge margin is only k//2 rows."""
    H = len(f)
    d = f[k:] - f[:-k]
    # Floor relative to the series' own robust amplitude — a NUMERICAL guard
    # only (float32 quantisation on noise-free synthetic input, ratio ~1e-7):
    # physical frames run topography/noise ratios up to ~1e3, so the floor
    # must sit far below that or it eats real sensitivity (measured on VIGIL).
    fm = np.median(f)
    floor = 1e-5 * (np.median(np.abs(f - fm)) * 1.4826)
    z = np.abs(_robust_z(d, trend_win, mad_floor=floor))
    zfull = np.zeros(H)
    zfull[k // 2: k // 2 + len(z)] = z
    return zfull


def row_channels(
    trace: npt.NDArray,
    retrace: npt.NDArray | None = None,
    *,
    bragg_min_prom: float = 8.0,
    bragg_min_fx: int = 4,
) -> tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.complex128] | None, dict]:
    """({name: length-H series}, bragg complex row series | None, debug)."""
    h = deplane(np.ascontiguousarray(np.asarray(trace), dtype=np.float64))
    H, W = h.shape
    rm = np.median(h, axis=1)
    hz = h - rm[:, None]                      # row DC removed AFTER keeping dc

    ch: dict[str, npt.NDArray[np.float64]] = {}
    ch["dc"] = rm
    ch["rms"] = np.median(np.abs(hz), axis=1) * 1.4826
    F = np.abs(np.fft.rfft(hz, axis=1))
    nb = F.shape[1]
    ch["hf"] = F[:, nb // 2:].sum(axis=1) / (F.sum(axis=1) + EPS)

    # adjacent-row NCC (pad to length H by repeating the first value)
    a, b = hz[:-1], hz[1:]
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + EPS
    ncc = num / den
    ch["ncc"] = np.concatenate([[ncc[0]], ncc])

    # trace/retrace per-row max normalised cross-correlation (lag-tolerant,
    # absorbs the piezo-hysteresis fast-axis offset)
    if retrace is not None:
        r = deplane(np.ascontiguousarray(np.asarray(retrace), dtype=np.float64))
        rz = r - np.median(r, axis=1, keepdims=True)
        Fa = np.fft.rfft(hz, axis=1)
        Fb = np.fft.rfft(rz, axis=1)
        xc = np.fft.irfft(Fa * np.conj(Fb), n=W, axis=1)
        na = np.sqrt((hz * hz).sum(axis=1) * (rz * rz).sum(axis=1)) + EPS
        ch["tr"] = 1.0 - xc.max(axis=1) / na

    # Bragg per-row demodulation (needs a lattice peak with usable |fx|)
    zrow = None
    dbg: dict = {"bragg": None}
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    F2 = np.fft.fft2(hz * win)
    mag = np.abs(F2)
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    rr = np.hypot(fy[:, None], fx[None, :])
    ring = rr >= 5.0
    if ring.any():
        medv = np.median(mag[ring]) + EPS
        cand = np.where(ring & (np.abs(fx)[None, :] >= bragg_min_fx), mag, 0.0)
        j = int(np.argmax(cand))
        py, px = np.unravel_index(j, mag.shape)
        prom = float(mag[py, px] / medv)
        if prom >= bragg_min_prom:
            fxx, fyy = fx[px], fy[py]
            x = np.arange(W)
            i = np.arange(H)
            dem = hz * np.exp(-2j * np.pi * fxx * x / W)[None, :]
            zrow = dem.mean(axis=1) * np.exp(-2j * np.pi * fyy * i / H)
            ch["bragg_amp"] = np.abs(zrow)
            dbg["bragg"] = {"fx": float(fxx), "fy": float(fyy), "prom": prom}
    return ch, zrow, dbg


def _run_centroid(v: npt.NDArray[np.float64]) -> int:
    """Centroid of the contiguous ≥0.6·max run around the peak."""
    imax = int(np.argmax(v))
    thr = 0.6 * v[imax]
    lo = imax
    while lo > 0 and v[lo - 1] >= thr:
        lo -= 1
    hi = imax
    while hi < len(v) - 1 and v[hi + 1] >= thr:
        hi += 1
    w = v[lo:hi + 1]
    rows = np.arange(lo, hi + 1)
    return int(round(float((rows * w).sum() / (w.sum() + EPS))))


def _pick_calib(nm_per_px: float | None) -> str:
    if nm_per_px is not None and nm_per_px > _MESO_NMPP:
        return "vigil-c2"
    return "vigil-c1"


def lod_dc(trace_or_dc: npt.NDArray, k: int = K) -> float | None:
    """Per-frame detection limit: the minimum row-DC jump (~6×MAD of the lag-k
    dc differences, input z units) this frame could have shown. Truth-free —
    computable on any frame, synthetic or real. None if the frame is too short."""
    a = np.asarray(trace_or_dc)
    f = np.median(deplane(np.ascontiguousarray(a, dtype=np.float64)), axis=1) if a.ndim == 2 else a.astype(np.float64)
    if len(f) < k + 8:
        return None
    d = f[k:] - f[:-k]
    return float(6.0 * np.median(np.abs(d - np.median(d))) * 1.4826)


def detect_tip_change(
    image: npt.NDArray,
    threshold: float | None = None,
    margin_frac: float | None = None,   # legacy arg, ignored (edge margin is k//2)
    *,
    nm_per_px: float | None = None,
    phase_gate_frac: float = 0.3,
) -> TipChangeResult:
    """Detect an abrupt mid-scan change in the row statistics (a tip change).

    Accepts (H,W) or a (2,H,W) trace/retrace pair (enables the ``tr`` channel).
    ``threshold`` is on the null-calibrated z; None picks the ≈FPR-1 % default
    for the scale (7.0 atomic / 10.0 meso, chosen by ``nm_per_px``)."""
    del margin_frac  # v1 relic — v2's edge margin is k//2 rows by construction
    trace, retrace = _to_pair(image)
    H = trace.shape[0]
    calib_name = _pick_calib(nm_per_px)
    tau = float(threshold) if threshold is not None else _DEFAULT_TAU[calib_name]

    if H < 2 * K or float(trace.std()) < 1e-15:
        return TipChangeResult(changed=False, change_row=None, score=0.0,
                               threshold=tau, calib=calib_name)

    ch, zrow, _dbg = row_channels(trace, retrace)

    Zs: dict[str, npt.NDArray[np.float64]] = {n: _lagk_z(f, K, TREND_WIN) for n, f in ch.items()}
    # Bragg phase channel: wrap-safe lag-k phase difference via complex product,
    # gated where the demodulation amplitude is too weak to carry phase.
    if zrow is not None:
        amp = np.abs(zrow)
        d = np.angle(zrow[K:] * np.conj(zrow[:-K]))
        gate = np.minimum(amp[K:], amp[:-K]) < phase_gate_frac * (np.median(amp) + EPS)
        d = d.copy()
        d[gate] = np.median(d[~gate]) if (~gate).any() else 0.0
        # phase differences live in radians — 1e-4 rad is far below any real
        # lattice-phase jitter, floor only binds on noise-free synthetic input
        z = np.abs(_robust_z(d, TREND_WIN, mad_floor=1e-4))
        zfull = np.zeros(H)
        zfull[K // 2: K // 2 + len(z)] = z
        Zs["bragg_ph"] = zfull

    # Null-library calibration: per-channel log-domain z of the frame peak vs
    # the no-change distribution; the frame score is the best channel.
    # Peaks are RAW maxima — the calibration table was built on raw maxima;
    # med3 smoothing is only used for row localisation below.
    table = _CALIB[calib_name]
    peaks = {n: float(z.max()) for n, z in Zs.items()}
    channel_scores: dict[str, float] = {}
    best_name, best = "", -np.inf
    for n, p in peaks.items():
        if n not in table:
            continue
        med, mad = table[n]
        s = (float(np.log(max(p, EPS))) - med) / mad
        channel_scores[n] = round(s, 3)
        if s > best:
            best, best_name = s, n
    if not np.isfinite(best):
        return TipChangeResult(changed=False, change_row=None, score=0.0,
                               threshold=tau, calib=calib_name)

    changed = bool(best > tau)
    change_row: int | None = None
    if changed:
        change_row = _run_centroid(median_filter(Zs[best_name], size=3, mode="nearest"))

    return TipChangeResult(
        changed=changed,
        change_row=change_row,
        score=float(best),
        threshold=tau,
        lod=lod_dc(ch["dc"]),
        channel_scores=channel_scores,
        calib=calib_name,
    )


# ────────────────────────────────────────────────────────────────────
# strictly causal online path (future streaming deployment)
# ────────────────────────────────────────────────────────────────────

def cusum_online(
    trace: npt.NDArray,
    *,
    k: int = K,
    kref: float = 0.75,
    trail: int = 101,
    burn_in: int = 40,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Strictly causal two-sided CUSUM on the ``dc`` channel only.

    dc (plane-removed row median) is empirically the dominant signal channel on
    VIGIL events (z-offset mode) while being the cleanest channel on no-change
    frames. Detection delay on VIGIL C1: median −6 rows (≈ synchronous with the
    event), p90 +2.4 — "stop mid-scan" is real. Returns (frame score, per-row
    statistic T); T[i] uses only rows ≤ i (trailing-window median/MAD null).
    NOTE: deplane is a whole-frame op — the *statistic* is causal, the feature
    extraction is not yet; a streaming deployment needs a running plane fit."""
    t2d, _ = _to_pair(trace)
    f = np.median(deplane(t2d), axis=1)
    H = len(f)
    T = np.zeros(H)
    if H < k + burn_in // 2:
        return 0.0, T
    d = f[k:] - f[:-k]
    n = len(d)
    nb = max(min(burn_in, n), 1)
    # same numerical MAD floor as the offline path (see _robust_z)
    fm = np.median(f)
    floor = 1e-5 * (np.median(np.abs(f - fm)) * 1.4826)
    med0 = np.median(d[:nb])
    mad0 = max(np.median(np.abs(d[:nb] - med0)) * 1.4826, floor) + EPS
    sp = sm = 0.0
    for i in range(n):
        if i < burn_in:
            med, mad = med0, mad0
        else:
            hist = d[max(0, i - trail):i]
            med = np.median(hist)
            mad = max(np.median(np.abs(hist - med)) * 1.4826, floor) + EPS
        z = (d[i] - med) / mad
        sp = max(0.0, sp + z - kref)
        sm = max(0.0, sm - z - kref)
        T[i + k // 2] = max(sp, sm)
    return float(T.max()), T


__all__ = ["detect_tip_change", "cusum_online", "row_channels", "lod_dc", "deplane"]
