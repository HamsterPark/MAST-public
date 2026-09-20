# -*- coding: utf-8 -*-
"""Scale-adaptive classical STM segmentation — terrace / step / defect / contam.

Production adoption of the validated prototype
``docs/v2/benchmarks/vigil_truth_validation/seg_scale_adaptive.py``
(2026-07-27; code kept line-identical apart from packaging so the validated
numbers keep applying — do NOT retune here without re-running that harness).

Measured on VIGIL physics ground truth (150-frame eval pools, tune/eval frames
strictly separated) against the previous Bragg-subtraction segmenter:

  C1 atomic defects:  presence F1 0.874 (vs 0.817) · object F1 0.106 (vs 0.063)
                      · count MAE 4.0 (vs 9.1) · block-presence IoU 0.121 (vs 0.061)
  C2 meso defects:    object F1 0.060 (vs 0.041) · count MAE 18.4 (vs 29.3)
  C2 step edges:      object F1 0.064 (vs 0.040)

Known failure modes (as measured, not hidden): fully-covered ordered arrays
carry an unresolvable DEFECT-vs-CONTAM class ambiguity (island array vs
molecular domain is geometrically indistinguishable in a single uncalibrated
frame); speckle-level scattered contamination is missed; nothing above 279 nm
scan size has physics ground truth yet.

Method (no training, deterministic, scipy/skimage only):
  1. every length is physical: kernels / structuring elements / area gates are
     nm ÷ nm_per_px — no per-corpus branches;
  2. FFT band-split texture measurement: an atomic-band peak (0.18–0.8 nm)
     blurs the lattice away before anything else (atoms are not defects); a
     super-atomic peak (0.8–4 nm) that also passes the radial-autocorrelation
     validity check r(T)/r(0) ≥ 0.35 marks an ordered nano-array (measured
     separation: real molecular domains +0.91 / island arrays +0.84 vs Au
     herringbone +0.05 / 1-f noise −0.05);
  3. islands/pits come from the LAYER STRUCTURE itself (KDE layering with
     valley-depth merge): when layering succeeds an island IS a layer and
     in-layer residuals erase it — residuals only catch layering failures;
  4. steps: layer-boundary bands ∪ Gwyddion "Step" quantile-difference filter;
  5. z is treated as uncalibrated (arbitrary units) — all thresholds are MAD
     adaptive. A perfectly fast-axis-parallel step can be absorbed by the
     row-alignment (real scans always have a finite step angle).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

CLASSES = ["TERRACE", "STEP", "DEFECT", "CONTAMINATION"]  # order == classical_seg

# ── tuned on the tune pool (idx%5==0), frozen for eval — and for production ──
DEFAULTS = dict(
    atomic_band_nm=(0.18, 0.80),  # atomic lattice period band
    array_band_nm=(0.80, 4.00),   # super-atomic (molecular/island array) band
    lat_snr=4.0,                  # radial-spectrum peak SNR gate
    peak_rel_prom=0.35,           # "strong peak" = prom ≥ this × strongest
    base_smooth_nm=0.30,          # base smoothing when no texture (nm)
    sigma_max=14.0,               # smoothing σ ceiling (px)
    peak_prom=0.04,               # KDE peak prominence (fraction of max)
    peak_sep_sig=4.0,             # min layer-peak separation = this × σ_n
    valley_rel=0.55,              # valley > this × lower peak → merge (fake layer)
    step_band_nm=0.30,            # layer-boundary band half-width (nm)
    step_band_max=2,              # band half-width ceiling (px)
    stepq_r_nm=0.30,              # Step quantile filter neighbourhood radius (nm)
    stepq_k=7.0,                  # Step filter threshold = med + k×MAD
    stepq_min_nm=2.0,             # Step filter min component length (nm)
    reg_k=5.0,                    # anomalous-region gate = k × σ_n
    reg_min_nm2=0.04,             # anomalous-region min area (nm²)
    field_min_n=6,                # island field: min compact blobs
    field_max_each=0.06,          # island field: single-blob area ceiling
    field_min_total=0.15,         # island field: total area floor
    field_extent=0.35,            # island field: bbox fill-ratio floor
    border_max_frac=0.06,         # big-blob area boundary (small = island/pit)
    bnd_conv=3.5,                 # big-blob boundary convolution ≥ this → island patch
    core_shrink=0.6,              # island/pit mask shrink to |offset| core
    env_sigma_T=1.5,              # envelope smoothing = this × texture period
    tex_ratio=2.5,                # texture contam: envelope hi/lo mode ratio floor
    tex_frac=(0.03, 0.97),        # texture contam: hi-side fraction window
    array_ac_min=0.35,            # array validity: radial AC r(T)/r(0) floor
    contam_min_nm2=1.5,           # contamination min area (nm²)
    tophat_r_nm=(0.15, 0.55),     # small-defect top-hat radii (nm, two scales)
    tophat_k=5.0,                 # top-hat response threshold = med + k×MAD
    tophat_abs_sig=3.0,           # top-hat absolute gate = this × σ_n
    def_min_nm2=0.03,             # small-defect min area (nm²)
    base_min_frac=0.10,           # base layer = lowest layer with ≥ this fraction
)

_FALLBACK_NMPP = 0.05             # neutral default when nm/px unknown


# ── robust levelling (Gwyddion median-of-differences + iterative poly) ──────
def align_rows_mediandiff(h: np.ndarray) -> np.ndarray:
    d = np.median(np.diff(h, axis=0), axis=1)
    return h - np.concatenate([[0.0], np.cumsum(d)])[:, None]


_V: dict = {}


def _vander(shape, order=2):
    key = (shape, order)
    if key not in _V:
        H, W = shape
        yy, xx = np.mgrid[0:H, 0:W]
        x = (xx / W - 0.5).ravel()
        y = (yy / H - 0.5).ravel()
        cols = [np.ones_like(x)]
        for o in range(1, order + 1):
            for i in range(o + 1):
                cols.append(x ** (o - i) * y ** i)
        _V[key] = np.column_stack(cols)
    return _V[key]


def level_iterative(h, order=2, iters=3, clip=2.5):
    h = np.asarray(h, np.float64)
    A = _vander(h.shape, order)
    m = np.ones(h.size, bool)
    coef = np.zeros(A.shape[1])
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[m], h.ravel()[m], rcond=None)
        r = h.ravel() - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        m = np.abs(r - np.median(r)) < clip * s
        if m.sum() < h.size * 0.2:
            break
    return (h.ravel() - A @ coef).reshape(h.shape)


def flatten_robust(h):
    return level_iterative(align_rows_mediandiff(np.asarray(h, np.float64)))


# ── FFT radial spectrum: per-band strong peaks ──────────────────────────────
def _radial_profile(h):
    H, W = h.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    F = np.abs(np.fft.rfft2((h - h.mean()) * win)) ** 2
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    return np.hypot(fy, fx).ravel(), F.ravel()


def _band_peak(fr, P, t_lo, t_hi, rel_prom):
    """Find a peak inside period band [t_lo, t_hi] px; returns (T_px, snr).
    T takes the SMALLEST period among strong peaks (never lock onto a moiré /
    superstructure and over-inflate the smoothing kernel)."""
    if t_hi <= t_lo * 1.1:
        return None, 0.0
    nb = 160
    bins = np.linspace(1.0 / t_hi, 1.0 / t_lo, nb + 1)
    idx = np.digitize(fr, bins) - 1
    ok = (idx >= 0) & (idx < nb)
    prof = np.bincount(idx[ok], weights=P[ok], minlength=nb)
    n = np.bincount(idx[ok], minlength=nb) + 1e-9
    prof = ndi.gaussian_filter1d(prof / n, 2.0)
    base = np.median(prof) + 1e-30
    pk, props = find_peaks(prof, prominence=base * 0.5)
    if len(pk) == 0:
        return None, 0.0
    strong = props["prominences"] >= rel_prom * props["prominences"].max()
    cand = pk[strong]
    best = cand.max()                       # max frequency = min period
    f0 = 0.5 * (bins[best] + bins[best + 1])
    return float(1.0 / f0), float(prof[best] / base)


def detect_texture(h, nmpp, P):
    """dict(atomic=(T,snr)|None, array=(T,snr)|None); bands defined in nm."""
    fr, Pw = _radial_profile(h)
    H, W = h.shape
    out = {"atomic": None, "array": None}
    a_lo = max(2.5, P["atomic_band_nm"][0] / nmpp)
    a_hi = min(min(H, W) / 4.0, P["atomic_band_nm"][1] / nmpp)
    T, snr = _band_peak(fr, Pw, a_lo, a_hi, P["peak_rel_prom"])
    if T is not None and snr >= P["lat_snr"] and T >= 3.0:
        out["atomic"] = (T, snr)
    r_lo = max(2.5, P["array_band_nm"][0] / nmpp)
    r_hi = min(min(H, W) / 4.0, P["array_band_nm"][1] / nmpp)
    T, snr = _band_peak(fr, Pw, r_lo, r_hi, P["peak_rel_prom"])
    if T is not None and snr >= P["lat_snr"] and T >= 3.0:
        out["array"] = (T, snr)
    return out


def bandpass_envelope(h, T_px, sigma_T=1.5):
    """Gaussian envelope of |bandpass(h; f0±35%)| — local texture amplitude."""
    H, W = h.shape
    F = np.fft.fft2(h - h.mean())
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    fr = np.hypot(fy, fx)
    f0 = 1.0 / T_px
    keep = (fr > 0.65 * f0) & (fr < 1.35 * f0)
    bp = np.real(np.fft.ifft2(F * keep))
    return ndi.gaussian_filter(np.abs(bp), min(sigma_T * T_px, 30.0))


def radial_ac_ratio(h, T_px):
    """Radial autocorrelation r(T)/r(0) — validity test for ordered arrays."""
    x = h - h.mean()
    F = np.fft.fft2(x)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2))) / x.size
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.mgrid[0:ac.shape[0], 0:ac.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    sel = (rr > 0.85 * T_px) & (rr < 1.15 * T_px)
    if not sel.any():
        return 0.0
    return float(ac[sel].mean() / (ac[cy, cx] + 1e-30))


# ── KDE layering (with valley-depth merge) ──────────────────────────────────
def _hist_modes(sel, sig_n, prom=0.04, min_sep=None, valley_rel=0.55, max_levels=8):
    """Histogram peaks of a 1-D sample + valley-depth merge; ascending peaks."""
    sel = np.asarray(sel, np.float64).ravel()
    lo, hi = np.percentile(sel, [0.5, 99.5])
    if hi <= lo:
        return [float(np.median(sel))]
    cnt, edges = np.histogram(sel, bins=np.linspace(lo, hi, 257))
    cnt = ndi.gaussian_filter1d(cnt.astype(float), 2.0)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    binw = ctr[1] - ctr[0]
    sep_bins = max(6, int(round((min_sep or 4.0 * sig_n) / binw)))
    pk, props = find_peaks(np.concatenate([[0.0], cnt, [0.0]]),
                           prominence=cnt.max() * prom, distance=sep_bins)
    pk = np.clip(pk - 1, 0, len(ctr) - 1)
    if len(pk) == 0:
        return [float(np.median(sel))]
    order = np.argsort(props["prominences"])[::-1][:max_levels]
    pk = np.sort(pk[order])
    # valley merge: a shallow valley between peaks = one tilted layer, not two
    merged = True
    while merged and len(pk) > 1:
        merged = False
        for j in range(len(pk) - 1):
            valley = cnt[pk[j]:pk[j + 1] + 1].min()
            if valley > valley_rel * min(cnt[pk[j]], cnt[pk[j + 1]]):
                keep = pk[j] if cnt[pk[j]] >= cnt[pk[j + 1]] else pk[j + 1]
                pk = np.delete(pk, j + 1 if keep == pk[j] else j)
                merged = True
                break
    return [float(x) for x in ctr[pk]]


def kde_layers(coarse, sig_n, prom=0.04, min_sep=None, valley_rel=0.55,
               max_levels=8, low_grad_pct=60):
    gy, gx = np.gradient(coarse)
    g = np.hypot(gy, gx)
    sel = coarse[g < np.percentile(g, low_grad_pct)]
    if sel.size < 100:
        sel = coarse.ravel()
    peaks = _hist_modes(sel, sig_n, prom, min_sep, valley_rel, max_levels)
    if len(peaks) == 1:
        return peaks, np.zeros(coarse.shape, np.int32)
    peaks_a = np.array(peaks)
    bounds = 0.5 * (peaks_a[1:] + peaks_a[:-1])
    lab = np.digitize(coarse, bounds).astype(np.int32)
    lab = ndi.median_filter(lab, size=5)
    return peaks, lab


def residual_in_layer(coarse, lab):
    r = coarse - np.median(coarse)
    for v in np.unique(lab):
        m = lab == v
        if m.sum() >= 20:
            r[m] = coarse[m] - np.median(coarse[m])
    return r


# ── Gwyddion "Step" quantile-difference filter ─────────────────────────────
def quantile_step_filter(coarse, r_px: int):
    """sqrt(q_{2/3} − q_{1/3}) over a disk — Gwyddion "Step" edge detection."""
    from skimage.morphology import disk
    fp = disk(max(2, int(r_px)))
    hi = ndi.percentile_filter(coarse, 200.0 / 3.0, footprint=fp)
    lo = ndi.percentile_filter(coarse, 100.0 / 3.0, footprint=fp)
    return np.sqrt(np.maximum(hi - lo, 0.0))


# ── small helpers ───────────────────────────────────────────────────────────
def _remove_small(mask, min_px):
    lb, n = ndi.label(mask)
    if n == 0:
        return mask
    sz = np.bincount(lb.ravel())
    keep = sz >= max(1, int(min_px))
    keep[0] = False
    return keep[lb]


def _remove_streaks(mask, ratio=8, thin=3):
    """Kill scan-line streaks: components with bbox aspect ≥ ratio, ≤ thin px."""
    lb, n = ndi.label(mask)
    if n == 0:
        return mask
    out = mask.copy()
    for k, sl in enumerate(ndi.find_objects(lb), start=1):
        hh = sl[0].stop - sl[0].start
        ww = sl[1].stop - sl[1].start
        if (ww >= ratio * hh and hh <= thin) or (hh >= ratio * ww and ww <= thin):
            out[sl][lb[sl] == k] = False
    return out


def _boundary_band(lab, w_px):
    e = np.zeros(lab.shape, bool)
    e[:-1, :] |= lab[:-1, :] != lab[1:, :]
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    if not e.any():
        return e
    return ndi.binary_dilation(e, iterations=max(1, int(w_px)))


def _spans_frame(sl, shape):
    return ((sl[0].start == 0 and sl[0].stop == shape[0]) or
            (sl[1].start == 0 and sl[1].stop == shape[1]))


def _touches_border(cc_sl, cc, shape):
    sl0, sl1 = cc_sl
    if sl0.start == 0 and cc[0, :].any():
        return True
    if sl0.stop == shape[0] and cc[-1, :].any():
        return True
    if sl1.start == 0 and cc[:, 0].any():
        return True
    if sl1.stop == shape[1] and cc[:, -1].any():
        return True
    return False


def _disk(r):
    from skimage.morphology import disk
    try:
        return disk(int(r), decomposition="sequence")
    except TypeError:
        return disk(int(r))


# ── main entry ──────────────────────────────────────────────────────────────
def segment_scale_adaptive(raw, nm_per_px: float | None = None, params: dict | None = None,
                           debug: bool = False):
    """Return (seg uint8 0..3, info dict). z in arbitrary units; nm_per_px
    falls back to the neutral 0.05. debug=True adds per-path masks to info."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import white_tophat, black_tophat

    P = dict(DEFAULTS)
    if params:
        P.update(params)
    nmpp = float(nm_per_px) if (nm_per_px and nm_per_px > 0) else _FALLBACK_NMPP

    h = flatten_robust(raw)
    if float(h.std()) < 1e-12:
        return np.zeros(h.shape, np.uint8), dict(flat=True)

    # 1) texture: atomic lattice / super-atomic array (molecular domain, islands)
    tex = detect_texture(h, nmpp, P)
    atomic, array_tex = tex["atomic"], tex["array"]
    # with an atomic peak, a super-atomic peak is a moiré/reconstruction (no
    # array detection); without one, the array must ALSO pass the radial-AC
    # validity check (herringbone / 1-f noise collapse on ac(T))
    ac = radial_ac_ratio(h, array_tex[0]) if array_tex is not None else 0.0
    use_array = (array_tex is not None) and (atomic is None) and (ac >= P["array_ac_min"])
    if atomic is not None:
        T = atomic[0]
    elif use_array:
        T = array_tex[0]
    else:
        T = None
    sigma = (float(np.clip(0.6 * T, 1.0, P["sigma_max"])) if T is not None
             else float(np.clip(P["base_smooth_nm"] / nmpp, 1.0, 6.0)))
    coarse = ndi.gaussian_filter(h, sigma)
    fine = h - coarse
    sig_n = 1.4826 * np.median(np.abs(fine - np.median(fine))) + 1e-12

    # 2) layering + in-layer residual
    peaks, lab = kde_layers(coarse, sig_n, prom=P["peak_prom"],
                            min_sep=P["peak_sep_sig"] * sig_n,
                            valley_rel=P["valley_rel"])
    resid = residual_in_layer(coarse, lab)

    # 3) anomalous regions → topology rules
    # candidates = non-base-layer components ∪ in-layer residual anomalies.
    # KEY: when layering SUCCEEDS an island is itself a "layer" and the
    # in-layer residual erases it — islands/pits must come from the layer
    # structure first; residuals only catch layering failures (no KDE peaks).
    fracs = [float((lab == v).mean()) for v in range(len(peaks))] if len(peaks) > 1 else [1.0]
    base_layer = 0
    for v, fr in enumerate(fracs):
        if fr >= P["base_min_frac"]:
            base_layer = v
            break
    dev = coarse - (peaks[base_layer] if peaks else float(np.median(coarse)))
    reg_min_px = max(9, P["reg_min_nm2"] / nmpp ** 2)
    reg = np.abs(resid) > P["reg_k"] * sig_n
    if len(peaks) > 1:
        reg |= lab != base_layer
    reg = _remove_small(reg, reg_min_px)
    defect = np.zeros(h.shape, bool)
    lb, n = ndi.label(reg)
    regions = []
    npx = h.size
    if n:
        comps = []
        for k, sl in enumerate(ndi.find_objects(lb), start=1):
            cc = lb[sl] == k
            a = int(cc.sum())
            bbox_a = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
            perim = int((cc & ~ndi.binary_erosion(cc)).sum())
            comps.append(dict(k=k, sl=sl, cc=cc, area=a,
                              extent=a / max(bbox_a, 1),
                              conv=perim / (2.0 * np.sqrt(np.pi * a) + 1e-12),
                              offs=float(np.median(dev[sl][cc])),
                              spans=_spans_frame(sl, h.shape),
                              border=_touches_border(sl, cc, h.shape)))
        # island-field mode: many compact blobs of decent total area → DEFECT
        compact = [c for c in comps
                   if not c["spans"] and c["area"] <= P["field_max_each"] * npx
                   and c["extent"] >= P["field_extent"]]
        field = (len(compact) >= P["field_min_n"] and
                 sum(c["area"] for c in compact) >= P["field_min_total"] * npx)
        big = P["border_max_frac"] * npx
        for c in comps:
            if c["area"] > big:
                # big blob: dendritic curly boundary = unfilled island/pit
                # sheet; straight boundary = terrace
                if c["conv"] >= P["bnd_conv"]:
                    kind = "island" if c["offs"] > 0 else "pit"
                else:
                    kind = "terrace"
            elif c["spans"]:
                kind = "terrace"                     # frame-spanning strip
            elif field and c in compact:
                kind = "island" if c["offs"] > 0 else "pit"
            elif c["border"] and c["extent"] < P["field_extent"]:
                kind = "terrace"                     # drift-torn border shreds
            else:
                kind = "island" if c["offs"] > 0 else "pit"
            if kind != "terrace":
                # shrink to the offset core so the blur halo doesn't inflate
                core = np.abs(dev[c["sl"]]) > P["core_shrink"] * abs(c["offs"])
                defect[c["sl"]][c["cc"] & core] = True
            regions.append(dict(area_px=c["area"], offset=round(c["offs"], 1),
                                extent=round(c["extent"], 2), conv=round(c["conv"], 2),
                                spans=c["spans"], border=c["border"], kind=kind))
    defect_reg = defect.copy()

    # 4) STEP: boundary bands ∪ quantile filter (suppressed near defects,
    #    de-streaked)
    wq = int(np.clip(round(P["step_band_nm"] / nmpp), 1, P["step_band_max"]))
    step_band = _boundary_band(lab, wq) if len(peaks) > 1 else np.zeros(h.shape, bool)
    sq = quantile_step_filter(coarse, int(np.clip(round(P["stepq_r_nm"] / nmpp), 2, 6)))
    med, mad = np.median(sq), 1.4826 * np.median(np.abs(sq - np.median(sq))) + 1e-12
    sq_min_px = max(20, (P["stepq_min_nm"] / nmpp) * (2 * wq + 1) * 0.5)
    step_q = sq > med + P["stepq_k"] * mad
    if defect_reg.any():
        # suppress only around the defect BODY (ring-shaped fake steps grow on
        # the blur halo, width ~σ); never suppress the whole reg — terrace
        # residual sheets neighbour REAL steps
        step_q &= ~ndi.binary_dilation(defect_reg,
                                       iterations=max(wq + 2, int(round(sigma))))
    step_q = _remove_small(_remove_streaks(step_q), sq_min_px)
    step = step_band | step_q

    # 5) small defects: top-hat on the coarse in-layer residual (lattice gone)
    resp = np.zeros_like(resid)
    for r_nm in P["tophat_r_nm"]:
        se = _disk(np.clip(round(r_nm / nmpp), 2, 32))
        resp = np.maximum(resp, np.maximum(white_tophat(resid, se),
                                           black_tophat(resid, se)))
    tophat = np.zeros(h.shape, bool)
    v = resp[resp > 0]
    if v.size > 50:
        rm = np.median(v)
        rmad = 1.4826 * np.median(np.abs(v - rm)) + 1e-12
        gate = max(rm + P["tophat_k"] * rmad, P["tophat_abs_sig"] * sig_n)
        tophat = _remove_small(_remove_streaks(resp > gate),
                               max(4, P["def_min_nm2"] / nmpp ** 2))
        defect |= tophat & ~step

    # 6) contamination = texture evidence
    contam = np.zeros(h.shape, bool)
    contam_min_px = max(25, P["contam_min_nm2"] / nmpp ** 2)
    # envelope reference period = LARGEST detected period — moiré / row
    # reconstructions must be suppressed too, or a textured terrace becomes
    # "contamination" via the envelope bimodality (TiO2 rows vs flat terrace)
    T_env = max([p[0] for p in (atomic, array_tex) if p is not None] or [3.0])
    sig_env = float(np.clip(0.6 * T_env, 1.0, P["sigma_max"]))
    fine_env = h - ndi.gaussian_filter(h, sig_env) if sig_env > sigma else fine
    env = ndi.gaussian_filter(np.abs(fine_env), min(P["env_sigma_T"] * T_env, 30.0))
    tex_stats = {"ac": round(ac, 2)}
    if use_array:
        # 6a) ordered super-atomic array (AC-validated): bandpass envelope maps
        #     its coverage. Coarse-height bimodality inside (island tops /
        #     trench floors, ≥6σ_n apart) → island array → DEFECT; unimodal
        #     (dense continuous film) → molecular domain → CONTAM. Physics: an
        #     unfilled island layer must expose substrate → bimodal heights.
        env0 = bandpass_envelope(h, array_tex[0], P["env_sigma_T"])
        # quantile-anchored threshold (Otsu on a heavy-tailed envelope lands
        # inside the main mode and marks only scraps of a full-cover domain)
        thr = 0.45 * float(np.percentile(env0, 92))
        amask = env0 > thr
        lo_med = float(np.median(env0[~amask])) if (~amask).any() else 0.0
        hi_med = float(np.median(env0[amask])) if amask.any() else 0.0
        ratio0 = hi_med / (lo_med + 1e-30)
        tex_stats["array_ratio"] = round(ratio0, 2)
        if ratio0 < 2.0 or amask.mean() > 0.9:
            amask = np.ones(h.shape, bool)     # uniform full cover → whole frame
        amask = _remove_small(amask, contam_min_px)
        # island patches already went to DEFECT via §3 curvature; only the
        # non-defect ordered-texture area becomes CONTAM here
        contam |= amask & ~step & ~defect
    else:
        # 6b) no array: envelope Otsu bimodality → high-texture patches =
        #     amorphous contamination; detected defects + halos excluded
        try:
            thr = threshold_otsu(env)
        except Exception:
            thr = None
        if thr is not None:
            hi = env > thr
            hf = float(hi.mean())
            lo_med = float(np.median(env[~hi])) if (~hi).any() else 0.0
            hi_med = float(np.median(env[hi])) if hi.any() else 0.0
            ratio = hi_med / (lo_med + 1e-30)
            tex_stats["env_ratio"] = round(ratio, 2)
            tex_stats["env_hi_frac"] = round(hf, 3)
            if P["tex_frac"][0] <= hf <= P["tex_frac"][1] and ratio >= P["tex_ratio"]:
                hi &= ~ndi.binary_dilation(defect, iterations=max(4, int(round(sigma))))
                contam |= _remove_small(hi, contam_min_px) & ~step

    seg = np.zeros(h.shape, np.uint8)
    seg[contam] = 3
    seg[defect & ~contam] = 2
    seg[step] = 1
    info = dict(atomic=None if atomic is None else (round(atomic[0], 1), round(atomic[1], 1)),
                array=None if array_tex is None else (round(array_tex[0], 1), round(array_tex[1], 1)),
                use_array=use_array, sigma=round(sigma, 2),
                sig_n=round(float(sig_n), 3), n_layers=len(peaks),
                n_regions=len(regions), tex=tex_stats, regions=regions[:40])
    if debug:
        info["masks"] = dict(step_band=step_band, step_q=step_q, reg=reg,
                             defect_reg=defect_reg, tophat=tophat, contam=contam,
                             coarse=coarse, resid=resid, env=env, lab=lab)
    return seg, info


# ── downstream-decision summary (the metrics agents actually consume) ───────

def summarize_segmentation(seg: np.ndarray, nm_per_px: float | None = None,
                           present_px: int = 50, obj_min_px: int = 9) -> dict:
    """Decision-oriented readout of a 4-class label map.

    Pixel IoU is the wrong lens for sparse targets (ground-truth defects are
    ~1.5 % of pixels → IoU 0.0x even when the density estimate is fine);
    autonomy consumes presence / counts / coverage. Object counting merges
    fragments with a physically-sized closing (defects 0.15 nm, contamination
    0.5 nm) exactly as the validated eval harness does.

    Returns {class: {present, count, area_frac}} for STEP/DEFECT/CONTAMINATION
    plus TERRACE area_frac."""
    nmpp = float(nm_per_px) if (nm_per_px and nm_per_px > 0) else _FALLBACK_NMPP
    close_px = {1: 1,
                2: int(np.clip(round(0.15 / nmpp), 1, 5)),
                3: int(np.clip(round(0.50 / nmpp), 1, 10))}
    out: dict = {}
    npx = seg.size
    for c, name in enumerate(CLASSES):
        m = seg == c
        entry: dict = {"area_frac": round(float(m.mean()), 4)}
        if c in (1, 2, 3):
            entry["present"] = bool(m.sum() >= present_px)
            cnt = 0
            if m.any():
                mm = ndi.binary_closing(m, np.ones((3, 3), bool),
                                        iterations=close_px[c])
                lb, nlab = ndi.label(mm)
                if nlab:
                    sz = np.bincount(lb.ravel())
                    cnt = int((sz[1:] >= obj_min_px).sum())
            entry["count"] = cnt
        out[name] = entry
    out["_npx"] = npx
    return out


__all__ = ["segment_scale_adaptive", "summarize_segmentation", "flatten_robust",
           "detect_texture", "bandpass_envelope", "quantile_step_filter",
           "kde_layers", "CLASSES", "DEFAULTS"]
