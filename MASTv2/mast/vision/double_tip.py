"""Algorithmic double-/multi-tip detection — network-free.

A double tip images the surface twice, shifted:  I = I_true ∗ (δ + a·δ_d).
That convolution *echo* puts a localised off-centre peak at the tip-separation
vector ``d`` in the autocorrelation of the **lattice-subtracted residual**
(subtracting the periodic lattice stops its own autocorr peaks from masking the
ghost; the two scan axes are notched to reject scan-line correlations).

On clean feature-bearing frames this is exact — AUROC 1.0 and 100 % recovery of
``d`` on synthetic ghosts; on textured / streaky frames pre-existing structure
competes, so treat the vector as advisory there. Full study:
``docs/v2/benchmarks/vision_v25_diagnostic/``.

This is a cheap, interpretable guard to run *next to* the learned N (multi-apex)
head — it needs no model and returns the physical separation, which the head
cannot.

────────────────────────────────────────────────────────────────────────────────
REAL-FRAME AUDIT 2026-08-11 — what this file may and may not claim
────────────────────────────────────────────────────────────────────────────────
Measured on this instrument's own frames (Au(111)/mica, ``_0110`` is the
operator's textbook multi tip; ``_0104/_0106/_0108/_0086/_0064/_0060`` are the
frames he reads as usable):

1. **It was answering 0.0 on every real frame.** ``std < 1e-9`` was an *absolute
   metres* guard, and a real STM frame's z-std is 1e-10 m (Au(111) steps are
   236 pm). All seven normal frames tripped it and returned
   ``is_double=False, score=0.0`` — a fake "clean". The only two frames that got
   past it were the two the operator had already binned (``_0102`` z-span
   75.6 nm, ``_0092`` 79.8 nm), i.e. the guard was an *inverted* filter: it
   admitted only the garbage. The guard is now dimensionless (see ``_prepare``).

2. **Even fed a normalised frame it does not rank multi tip above clean.**
   Autocorr scores: _0110 (multi tip) 0.118, versus clean _0104 0.181,
   _0086 0.180, _0108 0.175, _0064 0.155. The only frame over the 0.18 default
   threshold was a *clean* one. AUROC ≈ 0.33 — below chance on this surface.

3. **Why: the echo is linear in CURRENT, not in z.** With N apexes the feedback
   holds Σ_i a_i·exp(2κ·z_true(r−d_i)) constant, so
   ``z_meas = (1/2κ)·ln Σ_i a_i·e^{2κ z_true(r−d_i)}`` — a *soft-max*, not a
   convolution. Autocorrelation and cepstrum are both pure functions of the
   power spectrum and detect a **linear** echo; on a step/terrace surface the
   soft-max instead splits each 236 pm step into partial steps at fixed offsets.
   The old test-suite hid this because it built its ghosts linearly
   (``(1-a)·img + a·shift(img)``) — implementation and test shared one false
   premise, so the tests could not fail.

4. **Surface periodicity outscores any real ghost here.** On these frames the
   strongest replica peak is the *terrace repeat* (11–13 nm on _0104/_0086),
   not a tip separation. A synthetic multi tip injected with the correct
   soft-max model at d=(5,9) px — verified landed, Δmax 1384 pm, 39 % of pixels
   moved >20 pm — did **not** become the top peak in any autocorrelation
   variant. On a dense staircase a single frame cannot separate "the same step
   drawn k times" from "k genuinely different steps".

⇒ Consequences encoded below: the verdict is **tri-state**; ``single_tip`` is
only claimed inside the validated regime (aperiodic features on a flat
background); on morphology-dominated frames the answer is ``undetermined`` with
a reason, never a reassuring "clean". The evidence that *does* separate a tip
ghost from surface structure is **cross-frame agreement** — an apex separation
is the same vector in nm at every scan size and position, a terrace repeat is
not — so see :func:`agree_across_frames`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mast.vision.module import DoubleTipResult, ReplicaCandidate


def _to_2d(image: npt.NDArray) -> npt.NDArray[np.float32]:
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


def _residual(f: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Content the ghost lives on. Subtract ANY dominant periodic lattice (even a
    simple square/hex one, ≥3 sharp Bragg peaks) so a clean defect-free lattice
    does NOT masquerade as a ghost — the replica must come from *aperiodic*
    features (defects/steps/adsorbates). Non-periodic frames are kept (mean-sub)."""
    F = np.fft.fft2(f)
    mag = np.abs(F).copy()
    mag[0, 0] = 0.0
    keep = mag >= (mag.mean() + 5.0 * mag.std())
    if int(keep.sum()) >= 3:
        return (f - np.real(np.fft.ifft2(F * keep))).astype(np.float32)
    return f


def _autocorr(f: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    F = np.fft.fft2(f)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    m = ac.max()
    return (ac / m).astype(np.float32) if m > 0 else ac.astype(np.float32)


def _cepstrum(f: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Power-cepstrum: the FFT-satellite / power-spectrum-fringe signature of a
    convolution echo (|1 + a·e^{-ik·d}|² modulates the spectrum by cos(k·d) → a
    peak at quefrency d). Alternative to the autocorrelation replica peak."""
    logmag = np.log1p(np.abs(np.fft.fft2(f)))
    cep = np.fft.fftshift(np.abs(np.fft.ifft2(logmag)) ** 2)
    m = cep.max()
    return (cep / m).astype(np.float32) if m > 0 else cep.astype(np.float32)


def _radial_baseline_subtract(M: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Remove the isotropic central-shoulder falloff (subtract the azimuthal mean
    at each radius) so a localised anisotropic ghost peak survives."""
    cy, cx = M.shape[0] // 2, M.shape[1] // 2
    yy, xx = np.ogrid[:M.shape[0], :M.shape[1]]
    r = np.hypot(yy - cy, xx - cx).astype(int)
    tot = np.bincount(r.ravel(), M.ravel())
    cnt = np.bincount(r.ravel())
    radial = tot / np.maximum(cnt, 1)
    return (M - radial[r]).astype(np.float32)


def _score_map(M: npt.NDArray[np.float32], rmin: int) -> tuple[float, tuple[int, int]]:
    """Strongest off-centre, off-axis replica in the radial-baseline-subtracted
    map → (peak, (dy, dx)). Scan-line axes are notched (a ghost is off-axis)."""
    Mres = _radial_baseline_subtract(M)
    cy, cx = M.shape[0] // 2, M.shape[1] // 2
    rmax = M.shape[0] // 2 - 2
    yy, xx = np.ogrid[:M.shape[0], :M.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    offaxis = (np.abs(yy - cy) > 3) & (np.abs(xx - cx) > 3)
    ann = (rr >= rmin) & (rr <= rmax) & offaxis
    if not ann.any():
        return 0.0, (0, 0)
    idx = np.where(ann)
    jj = int(np.argmax(Mres[ann]))
    return float(Mres[ann].max()), (int(idx[0][jj] - cy), int(idx[1][jj] - cx))


def _pixel_noise(f: npt.NDArray[np.float32]) -> float:
    """Pixel-to-pixel noise from the along-row second difference (robust, and —
    unlike an absolute metres constant — dimensionless once divided into std)."""
    if f.shape[1] < 3:
        return 0.0
    d2 = f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    return 1.4826 * mad / np.sqrt(6.0)


def _morphology_ratio(f: npt.NDArray[np.float32]) -> float:
    """large-scale morphology RMS ÷ local roughness RMS.

    On a step/terrace frame the terraces dominate (this instrument's ``_0110``:
    plane 203.9 pm vs roughness 14.2 pm → 14×). That is the regime where the
    strongest autocorrelation replica is the *terrace repeat*, not a tip
    separation, so a single frame cannot decide — see the module header, point 4.
    """
    from scipy import ndimage as ndi

    sigma = max(2.0, min(f.shape) / 16.0)
    low = ndi.gaussian_filter(f.astype(np.float64), sigma)
    high = f - low
    hs = float(high.std())
    return float(low.std()) / hs if hs > 0 else float("inf")


def _peak_significance(M: npt.NDArray[np.float32], rmin: int,
                       peak: float) -> float:
    """Peak height in MADs above the *same-annulus* null.

    The raw score is not comparable between frames (it depends on how much
    structure the frame has); this is. Measured: clean frames on this instrument
    reach raw 0.18 — the old default threshold — purely from terrace structure.
    """
    Mres = _radial_baseline_subtract(M)
    cy, cx = M.shape[0] // 2, M.shape[1] // 2
    yy, xx = np.ogrid[:M.shape[0], :M.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    ann = (rr >= rmin) & (rr <= M.shape[0] // 2 - 2)
    if not ann.any():
        return 0.0
    v = Mres[ann]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    return (peak - med) / mad if mad > 0 else 0.0


def _lag_significance(M: npt.NDArray[np.float32], rmin: int,
                      vec: tuple[int, int], tol: float = 3.0) -> float:
    """Significance of the replica **at a given lag** (best within ``tol`` of it).

    Tests presence, not dominance. Asking each quadrant for its *argmax* instead
    made a quadrant whose ghost was real but sub-dominant vote against — and let
    a quadrant whose argmax was pure noise vote at all.
    """
    Mres = _radial_baseline_subtract(M)
    cy, cx = M.shape[0] // 2, M.shape[1] // 2
    yy, xx = np.ogrid[:M.shape[0], :M.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    ann = (rr >= rmin) & (rr <= M.shape[0] // 2 - 2)
    if not ann.any():
        return 0.0
    v = Mres[ann]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    if mad <= 0:
        return 0.0
    best = -np.inf
    for sy, sx in ((vec[0], vec[1]), (-vec[0], -vec[1])):   # centrosymmetric twins
        near = ann & (np.hypot(yy - (cy + sy), xx - (cx + sx)) <= tol)
        if near.any():
            best = max(best, float(Mres[near].max()))
    return (best - med) / mad if np.isfinite(best) else 0.0


def _replica_map(f: npt.NDArray[np.float32], method: str
                 ) -> tuple[npt.NDArray[np.float32], str]:
    r = _residual(f)
    if method in ("cepstrum", "fft"):
        return _cepstrum(r), "cepstrum"
    if method == "combined":
        a, c = _autocorr(r), _cepstrum(r)
        return (a, "autocorr") if a.max() >= c.max() else (c, "cepstrum")
    return _autocorr(r), "autocorr"


def _fold(v: tuple[int, int]) -> tuple[int, int]:
    """d and −d are the SAME separation: an autocorrelation is centrosymmetric,
    so which of the twin peaks argmax lands on is arbitrary. Comparing unfolded
    vectors silently scores every twin as a disagreement (this cost the tile
    test 4/4 → 0/4 before it was folded)."""
    dy, dx = v
    return (-dy, -dx) if (dx < 0 or (dx == 0 and dy < 0)) else (dy, dx)


def _tile_agreement(f: npt.NDArray[np.float32], vec: tuple[int, int],
                    method: str, rmin: int, sig_floor: float,
                    tol: float = 3.0) -> tuple[int, int]:
    """→ (quadrants agreeing on ``vec``, quadrants entitled to an opinion).

    A tip's apex separation is a property of the tip, so it is the same in every
    part of the frame. NOTE this does **not** discriminate against a regular
    staircase, whose terrace repeat is also frame-wide — that is what
    :func:`agree_across_frames` is for.

    A quadrant that found nothing significant is **abstaining, not dissenting**.
    Counting it as dissent made sparse-feature frames (features land in 2 of 4
    quadrants) unreportable no matter how strong the ghost was.
    """
    ny, nx = f.shape
    hy, hx = ny // 2, nx // 2
    if min(hy, hx) < 4 * rmin:
        return 0, 0
    # HALVES, not quadrants: the test only needs "different parts of the frame",
    # and a quadrant of a 160 px frame holds ~3 features — too few for the lag
    # statistic to have any power, so a real ghost scored 1/4 and was thrown out.
    tiles = (f[:hy, :], f[hy:, :], f[:, :hx], f[:, hx:])
    agree = eligible = 0
    for sub in tiles:
        sub = sub - float(sub.mean())
        s = float(sub.std())
        if s <= 0:
            continue
        eligible += 1
        M, _ = _replica_map((sub / s).astype(np.float32), method)
        if _lag_significance(M, rmin, vec, tol) >= sig_floor:
            agree += 1
    return agree, eligible


def detect_double_tip(
    image: npt.NDArray,
    nm_per_px: float | None = None,
    threshold: float = 0.18,
    rmin: int = 6,
    method: str = "autocorr",
    significance_min: float = 6.0,
    morphology_max: float = 1.5,
    aperiodic_min: float = 0.04,
    row_jump_max: float = 1.5,
) -> DoubleTipResult:
    """Detect a double/multi tip and recover the apex-separation vector.

    ⚠️ Read ``result.verdict`` (``multi_tip`` / ``single_tip`` / ``undetermined``),
    not ``is_double`` — see :class:`DoubleTipResult` and this module's header.
    ``single_tip`` is only claimed inside the validated regime; on a
    morphology-dominated step/terrace frame this returns ``undetermined`` plus
    the candidate vectors, which :func:`agree_across_frames` can then settle.

    ``method``: ``"autocorr"`` (default; replica peak) · ``"cepstrum"`` (a.k.a.
    ``"fft"``; the power-spectrum-fringe / satellite signature) · ``"combined"``
    (stronger of the two). Both are pure functions of the power spectrum, so
    neither can see a soft-max (step-splitting) ghost — header point 3.
    """
    def _out(verdict, reason, *, peak=0.0, vec=(0, 0), used=method, sig=0.0,
             cands=()):
        dy, dx = vec
        sep = (float(np.hypot(dy, dx) * nm_per_px)
               if (nm_per_px and nm_per_px > 0) else None)
        return DoubleTipResult(
            is_double=(verdict == "multi_tip"), score=max(0.0, peak),
            threshold=float(threshold), separation_px=(abs(dy), abs(dx)),
            separation_nm=sep, method=used, verdict=verdict, reason=reason,
            significance=float(sig), candidates=tuple(cands))

    f0 = _to_2d(image)
    if not np.isfinite(f0).all():
        f0 = np.nan_to_num(f0, nan=float(np.nanmedian(f0[np.isfinite(f0)]))
                           if np.isfinite(f0).any() else 0.0)
    f = f0 - float(f0.mean())
    std = float(f.std())
    if std <= 0:
        return _out("undetermined", "flat_frame_no_structure")

    # ── the guard that used to silence this detector ────────────────────────
    # WAS: ``if std < 1e-9: return is_double=False`` — absolute METRES, so every
    # real STM frame (z-std ~1e-10 m) got a fake "clean". Now dimensionless:
    # structure is measured against the frame's OWN pixel noise.
    f = (f / std).astype(np.float32)
    noise = _pixel_noise(f)
    snr = (1.0 / noise) if noise > 0 else float("inf")
    if snr < 2.0:
        return _out("undetermined", f"no_structure_above_noise (snr {snr:.1f})")

    # ── could a replica have shown up at all? (falsifiable negative) ────────
    # "No ghost seen" is only worth saying if a ghost WOULD have been seen. The
    # repo already names this quantity: herringbone reports aperiodic_fraction
    # next to the double-tip score for exactly this reason. Measured floor on
    # this instrument: _0102 (tip changed mid-scan) 0.023 — nothing aperiodic to
    # ghost, so its old "not double" carried no information at all.
    aper = float(np.var(_residual(f)))
    if aper < aperiodic_min:
        return _out("undetermined",
                    f"no_aperiodic_content (aperiodic fraction {aper:.3f} < "
                    f"{aperiodic_min:.3f}); a ghost would not have been visible "
                    f"on this frame, so 'no ghost' would say nothing")

    # Row-offset streaks: when row-to-row DC jumps are as big as the whole
    # frame's structure the image is scan lines, not topography (_0092: 2.34 vs
    # ≤0.53 on every frame the operator kept).
    row_jump = float(np.percentile(np.abs(np.diff(np.median(f, axis=1))), 95))
    if row_jump > row_jump_max:
        return _out("undetermined",
                    f"row_offset_dominated (p95 row-to-row jump {row_jump:.2f} × "
                    f"frame std > {row_jump_max:.2f}); streaks, not topography")

    M, used = _replica_map(f, method)
    peak, vec = _score_map(M, rmin)
    sig = _peak_significance(M, rmin, peak)
    tiles, tiles_elig = _tile_agreement(f, vec, method, rmin,
                                        sig_floor=significance_min / 2.0)
    frame_wide = tiles >= 2 and tiles >= 0.6 * max(tiles_elig, 1)
    dy, dx = vec
    cands = (ReplicaCandidate(
        dy_px=int(dy), dx_px=int(dx),
        separation_nm=(float(np.hypot(dy, dx) * nm_per_px)
                       if (nm_per_px and nm_per_px > 0) else None),
        score=max(0.0, peak), significance=float(sig), tiles_agreeing=int(tiles)),)

    morph = _morphology_ratio(f)
    if morph > morphology_max:
        # Terraces/steps dominate ⇒ the replica statistic is not valid here (its
        # top peak is the terrace repeat). Hand over to the operator's criterion,
        # which asks a different question — do the secondary steps share ONE
        # offset — and needs no linear-echo assumption.
        sp = detect_step_splitting(image, nm_per_px) if nm_per_px else {
            "verdict": "undetermined", "reason": "unknown_pixel_size"}
        if sp.get("verdict") == "multi_tip":
            dpx = int(round(float(sp["offset_nm"]) / float(nm_per_px)))
            cand = ReplicaCandidate(dy_px=0, dx_px=dpx,
                                    separation_nm=float(sp["offset_nm"]),
                                    score=max(0.0, peak), significance=float(sig))
            return _out("multi_tip", f"step-splitting: {sp['reason']}",
                        peak=peak, vec=(0, dpx), used="step_splitting", sig=sig,
                        cands=(cand,))
        return _out("undetermined",
                    f"morphology_dominated (large-scale/roughness {morph:.1f} > "
                    f"{morphology_max:.1f}); replica statistic invalid here. "
                    f"Step-splitting says: {sp.get('reason', '?')}. "
                    f"Cross-check with agree_across_frames()",
                    peak=peak, vec=vec, used=used, sig=sig, cands=cands)

    if sig >= significance_min and frame_wide:
        return _out("multi_tip",
                    f"replica {sig:.1f} MAD above null and frame-wide "
                    f"({tiles}/{tiles_elig} quadrants that had an opinion agree)",
                    peak=peak, vec=vec, used=used, sig=sig, cands=cands)

    why = ("peak not significant" if sig < significance_min
           else "peak significant but not frame-wide")
    return _out("single_tip",
                f"no frame-wide replica — {why} ({sig:.1f} MAD vs "
                f"{significance_min:.1f}, {tiles}/{tiles_elig} quadrants agree); "
                f"frame had aperiodic content {aper:.3f} so a ghost would have shown",
                peak=peak, vec=vec, used=used, sig=sig, cands=cands)


def _row_substeps(row: npt.NDArray, k: float = 4.0) -> list[tuple[float, float]]:
    """One scan line's sub-steps → [(position_px, signed height), ...].

    The only threshold is ``k`` × the row's OWN derivative noise. There is
    deliberately no absolute height floor: an absolute constant in physical units
    is the exact bug this module was built to remove (an absolute 1e-9 m guard
    silenced the whole detector), and it reappeared here the moment a caller
    passed heights in nm instead of pm — 0 sub-steps found, reported as "no
    splitting". Both statistics this function feeds (gap ratio, height ratio)
    are dimensionless, so nothing needs the units.

    The height is the **integrated derivative over a contiguous above-threshold
    run**, not a difference of plateau medians. On a dense staircase the terraces
    are a few pixels wide, so there are no plateaus to take medians of — that is
    exactly why the earlier "per-row gradient spike + ±6 px plateau" measurement
    produced a monotonically decaying jump histogram on EVERY frame, clean or
    not, and could not separate anything.
    """
    x = np.arange(row.size)
    r = row - np.polyval(np.polyfit(x, row, 1), x)
    d = np.diff(r)
    sigma = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    if not np.isfinite(sigma) or sigma <= 0:
        return []
    thr = k * sigma
    out: list[tuple[float, float]] = []
    i = 0
    while i < d.size:
        if abs(d[i]) > thr:
            s = np.sign(d[i])
            j = i
            while j < d.size and abs(d[j]) > thr and np.sign(d[j]) == s:
                j += 1
            h = float(d[i:j].sum())
            w = np.abs(d[i:j])
            out.append((float((np.arange(i, j) * w).sum() / w.sum()), h))
            i = j
        else:
            i += 1
    return out


def detect_step_splitting(
    image: npt.NDArray,
    nm_per_px: float,
    max_gap_nm: float = 25.0,

    tightness_max: float = 0.35,
    distinct_max: float = 0.6,
    min_pairs: int = 100,
) -> dict:
    """⭐ The operator's criterion: **do the secondary steps share one offset?**

    This is the primary multi-tip test on step/terrace surfaces, and it does not
    depend on the linear-echo assumption that autocorrelation and cepstrum need
    (see the module header, point 3 — constant-current imaging is a soft-max, so
    that assumption is wrong on exactly these frames). It needs only "the same
    tip is the same tip across the whole frame".

    PHYSICS. With apexes at separation ``d`` and weights a₁,a₂, a down-step of
    height H acquires an intermediate plateau of

        width  = |d|                                        (a TIP property)
        height = (1/2κ)·ln[(a₁+a₂e^{2κH})/(a₁+a₂)]          (a TIP property)

    so **every** true step is split the same way. Therefore:

    * the split pair is ASYMMETRIC in height (ratio ρ = small/large, one value
      frame-wide) and its internal gap is ONE value frame-wide;
    * two genuinely adjacent steps are SYMMETRIC (ρ≈1, both full quanta) and
      their gap is the terrace width, which varies across the frame.

    So the signature is a population of asymmetric pairs whose gap is both
    **tight** and **distinctly smaller than the terrace repeat**. That last
    clause is load-bearing: without it the test happily "finds" the terrace
    repeat itself and calls it a tip separation.

    ⚠️ Lateral DRAG also puts sub-structure on step edges. It is separated by
    running trace and retrace separately: a tip ghost is the same geometry in
    both, drag reverses with the scan direction. Pass each and compare
    ``offset_nm`` — see ``agree_across_frames`` for the cross-frame version.
    """
    f = _to_2d(image)
    if not np.isfinite(f).all():
        f = np.nan_to_num(f, nan=float(np.nanmedian(f[np.isfinite(f)]))
                          if np.isfinite(f).any() else 0.0)
    if not (nm_per_px and nm_per_px > 0):
        return {"verdict": "undetermined", "reason": "unknown_pixel_size"}

    # A true step and its secondary are a MAJOR edge followed closely by a MINOR
    # one of the same sign. The terrace repeat is the spacing between successive
    # MAJOR edges. Splitting the two populations by "is this edge one of the big
    # ones in its own row" — rather than by the pair's height ratio — matters:
    # when the split is strongly asymmetric (a₂/a₁ small) nearly every pair is
    # asymmetric, so the "symmetric" leftovers are noise pairs and the terrace
    # estimate built from them came out 4× too small, which then vetoed a
    # correctly-recovered offset.
    offsets: list[float] = []
    ratios: list[float] = []
    terraces: list[float] = []
    for row in f:
        e = _row_substeps(row)
        if len(e) < 3:
            continue
        hs = np.abs(np.asarray([h for _, h in e]))
        major = hs >= np.median(hs)
        last_major: tuple[float, float] | None = None
        for i, (p, h) in enumerate(e):
            if major[i]:
                if last_major is not None and np.sign(h) == np.sign(last_major[1]):
                    g = (p - last_major[0]) * nm_per_px
                    if 0 < g <= max_gap_nm:
                        terraces.append(g)
                last_major = (p, h)
            elif last_major is not None and np.sign(h) == np.sign(last_major[1]):
                g = (p - last_major[0]) * nm_per_px
                if 0 < g <= max_gap_nm:
                    offsets.append(g)
                    ratios.append(abs(h) / abs(last_major[1]))
    O = np.asarray(offsets)
    T = np.asarray(terraces)
    if O.size < min_pairs or T.size < 10:
        return {"verdict": "undetermined",
                "reason": f"too few major/minor sub-step pairs "
                          f"({O.size} < {min_pairs}); either the frame has few "
                          f"steps or the pixel size cannot resolve a split",
                "n_pairs": int(O.size)}

    med = float(np.median(O))
    tight = 1.4826 * float(np.median(np.abs(O - med))) / med if med > 0 else 9e9
    terrace = float(np.median(T))
    distinct = med / terrace if terrace > 0 else 9e9
    out = {"n_pairs": int(O.size), "n_asymmetric": int(O.size),
           "offset_nm": med, "tightness": tight, "terrace_nm": terrace,
           "distinctness": distinct, "split_ratio": float(np.median(ratios))}

    if tight <= tightness_max and distinct <= distinct_max:
        out["verdict"] = "multi_tip"
        out["reason"] = (f"secondary steps share one offset: {med:.2f} nm "
                         f"(spread {tight:.2f}), distinctly below the terrace "
                         f"repeat {terrace:.2f} nm; split ratio "
                         f"{out['split_ratio']:.2f}")
        return out
    out["verdict"] = "no_splitting"
    why = []
    if tight > tightness_max:
        why.append(f"offsets are not shared (spread {tight:.2f} > {tightness_max})")
    if distinct > distinct_max:
        why.append(f"the offset found ({med:.2f} nm) is not distinct from the "
                   f"terrace repeat ({terrace:.2f} nm) — it IS the terrace repeat")
    out["reason"] = "; ".join(why)
    return out


def agree_across_frames(
    frames: "list[tuple[DoubleTipResult, float, float]]",
    tol_nm: float = 2.0,
    min_frames: int = 3,
    min_fraction: float = 0.6,
) -> dict:
    """Settle what one frame cannot: does the SAME displacement recur everywhere?

    ``frames`` is ``[(result, nm_per_px, scan_angle_deg), ...]``. Candidate
    vectors are de-rotated into **sample** coordinates by ``scan_angle_deg``
    before comparison, so a 90°-rotated frame can be mixed in directly.

    The physics: a tip's apex separation is a property of the tip, so in sample
    coordinates it is the same vector (in nm) at every scan size, every scan
    position and every scan angle. A terrace repeat is a property of the local
    surface and is none of those things. That difference — not the strength of
    any single peak — is what separates a multi tip from a dense staircase.

    Rotating the scan additionally separates both from *scan-frame* artefacts
    (creep, feedback ringing, line noise), which stay pinned to the scan axes
    while everything real rotates with the sample.
    """
    pts: list[tuple[float, float, int]] = []
    for i, (res, nmpp, ang) in enumerate(frames):
        if not (nmpp and nmpp > 0):
            continue
        th = np.radians(-float(ang))
        c, s = np.cos(th), np.sin(th)
        for cand in res.candidates:
            vy, vx = cand.dy_px * nmpp, cand.dx_px * nmpp
            # de-rotate into sample coords; sign-fold (d and -d are one vector)
            ry, rx = c * vy - s * vx, s * vy + c * vx
            if (rx < 0) or (rx == 0 and ry < 0):
                ry, rx = -ry, -rx
            pts.append((ry, rx, i))

    best: dict = {"agrees": False, "n_frames": 0, "vector_nm": None,
                  "separation_nm": None, "reason": "no candidates"}
    if not pts:
        return best
    n_in = len({i for _, _, i in pts})
    for ay, ax, _ in pts:
        near = [(y, x, i) for y, x, i in pts if np.hypot(y - ay, x - ax) <= tol_nm]
        got = {i for _, _, i in near}
        if len(got) > best["n_frames"]:
            my = float(np.mean([y for y, _, _ in near]))
            mx = float(np.mean([x for _, x, _ in near]))
            # A MAJORITY must agree, not merely ``min_frames`` of them: with one
            # candidate per frame and a 2 nm tolerance, 2 of 6 unrelated clean
            # frames coincided by chance in the real-frame audit — that pair
            # alone would have been a false "multi tip".
            best = {"agrees": (len(got) >= min_frames
                               and len(got) >= min_fraction * n_in),
                    "n_frames": len(got), "vector_nm": (my, mx),
                    "separation_nm": float(np.hypot(my, mx)),
                    "reason": (f"{len(got)}/{n_in} frames put a replica within "
                               f"{tol_nm:g} nm of ({my:.1f}, {mx:.1f}) nm")}
    if n_in < min_frames:
        best["agrees"] = False
        best["reason"] = f"only {n_in} frame(s) with candidates; need {min_frames}"
    return best


__all__ = ["detect_double_tip", "agree_across_frames"]

# 大尺度台阶图上的多针尖检测。
#
# 单原子台阶高度提供物理参考，但高度直方图不能覆盖全部多针尖形态：
# 多个顶点还可能横向复制台阶边缘而不产生可分辨的半高能级。
# 因此以非周期形貌的自相关复制峰为主，高度分布只作上下文。
# 只有孤立团簇、缺少台阶或非周期内容的小视野不适合这一路判据。
#
# Au(111) 单原子台阶高度（米），为物理参考而非可调阈值。
AU111_STEP_M = 235.5e-12

#: 相邻台面高度差离整数倍台阶多远算「劈开了」。
#: 0.25 = 四分之一个台阶 —— 再小就会把测量噪声与蠕变读成劈裂。
STEP_SPLIT_TOL = 0.25


def rig_step_m() -> float:
    """**这台仪器读出来的**单原子台阶高度(米)。

    不是物理值 —— 「我们的仪器没有校准,好像是 z 差 15%,
    不用管,就假设我们的金是差 15% 对的。」判据量的是仪器读数,尺子就得用
    仪器的刻度。没设过就退回物理值(那时判据会整体偏,报文里会说)。
    """
    try:
        from mast.core.instrument_profile import get_config

        v = float(get_config("au_step_pm", AU111_STEP_M * 1e12))
        return v * 1e-12 if 50.0 <= v <= 500.0 else AU111_STEP_M
    except Exception:  # noqa: BLE001
        return AU111_STEP_M

# 台阶图判据的最小视野（米），用于限制应用范围。
# 视野需包含可比对的台阶边缘；单团簇的不同高度不能当作台面能级。
# 达到视野门仍不证明存在台阶，后续还需非周期内容检查。
STEP_SPLIT_MIN_FRAME_M = 100e-9

# 大图上判多针尖的自相关 replica score 阈值。
# 该分数描述同一非周期形貌重复出现的强度；其可用范围依赖
# 针尖顶点间隔、扫描器与输入形貌，使用前须独立验证阈值。
DOUBLE_TIP_SCORE_ON_STEPS = 0.16


def step_splitting(image, *, step_m: "float | None" = None,
                   frame_m: "float | None" = None,
                   score_threshold: float = DOUBLE_TIP_SCORE_ON_STEPS,
                   min_levels: int = 3) -> dict:
    """大尺度图上的多针尖判读，返回 single / split / undecidable。

    主判据是非周期形貌的自相关复制峰。高度直方图仅报告台面上下文，
    不能独自判定多针尖，因为横向复制不一定改变台阶高度差。

    视野小于 STEP_SPLIT_MIN_FRAME_M，或自相关报告 no_aperiodic_content
    时拒判。morphology_dominated 不单独拒判：有台阶的大图本来就可能
    以形貌为主。阈值适用性须由调用方针对目标成像条件验证。
    """
    import numpy as np

    out: dict = {"verdict": "undecidable", "score": None,
                 "score_threshold": float(score_threshold),
                 "levels_pm": [], "gaps_au": [], "deviations": [],
                 "frame_m": frame_m}
    try:
        if step_m is None:
            step_m = rig_step_m()
        out["step_height_pm"] = float(step_m) * 1e12

        f = _to_2d(np.asarray(image))
        if f.size == 0:
            out["reason"] = "空图"
            return out
        if frame_m is not None and float(frame_m) < STEP_SPLIT_MIN_FRAME_M:
            out["reason"] = (f"这张图只有 {float(frame_m) * 1e9:.0f} nm,"
                             f"视野不足以包含台阶(需 {STEP_SPLIT_MIN_FRAME_M * 1e9:.0f} nm 以上)")
            return out
        if frame_m is None:
            out["frame_unknown"] = True

        try:
            out.update(_terrace_context(f, float(step_m) * 1e12, int(min_levels)))
        except Exception:  # noqa: BLE001
            pass

        r = detect_double_tip(f)
        score = float(getattr(r, "score", 0.0) or 0.0)
        out["score"] = round(score, 4)
        out["replica_reason"] = str(getattr(r, "reason", "") or "")[:120]
        sep = getattr(r, "separation_px", None)
        if sep:
            out["separation_px"] = list(sep)

        if "no_aperiodic_content" in out["replica_reason"]:
            out["reason"] = "图上没有可比对的结构(自相关无非周期成分)—— 判不了"
            return out

        out["verdict"] = "split" if score >= float(score_threshold) else "single"
        cmp_ = "≥" if score >= float(score_threshold) else "<"
        out["reason"] = (
            f"自相关 replica score {score:.3f} {cmp_} 阈值 {score_threshold:.2f}"
            + (f";台面能级 {len(out['levels_pm'])} 个" if out.get("levels_pm") else ""))
        return out
    except Exception as exc:  # noqa: BLE001 — 判据坏掉不许弄坏实验
        out["reason"] = f"多针尖判据没跑成:{exc}"
        return out


def _terrace_context(f, step_pm: float, min_levels: int) -> dict:
    """高度直方图给的上下文:有几个台面、相邻差是几个台阶。**不下判决。**"""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    from mast.data.processors import plane_subtract

    lv = np.asarray(plane_subtract(f), dtype=float) * 1e12
    v = lv[np.isfinite(lv)]
    if v.size < 500:
        return {}
    lo, hi = float(np.percentile(v, 0.3)), float(np.percentile(v, 99.7))
    if hi - lo < (min_levels - 1) * step_pm:
        return {"terrace_note": f"整幅落差 {hi - lo:.0f} pm,放不下 {min_levels - 1} 个台阶"}
    bin_pm = step_pm / 16.0
    nb = max(16, int((hi - lo) / bin_pm))
    h, edges = np.histogram(v, bins=nb, range=(lo, hi))
    c = 0.5 * (edges[1:] + edges[:-1])
    h = gaussian_filter1d(h.astype(float), sigma=(step_pm / 6.0) / bin_pm)
    peaks = [i for i in range(1, len(h) - 1)
             if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] >= 0.10 * h.max()]
    levels: list[float] = []
    for i in peaks:
        if levels and c[i] - levels[-1] < 0.5 * step_pm:
            continue
        levels.append(float(c[i]))
    if len(levels) < 2:
        return {"levels_pm": [round(x, 1) for x in levels]}
    gaps = np.diff(levels) / step_pm
    dev = np.abs(gaps - np.round(gaps))
    return {"levels_pm": [round(x, 1) for x in levels],
            "gaps_au": [round(float(g), 2) for g in gaps],
            "deviations": [round(float(d), 2) for d in dev]}


