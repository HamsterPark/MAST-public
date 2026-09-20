# -*- coding: utf-8 -*-
"""Barker-style deterministic tip-quality criteria (ACS Nano 2024 line,
automated) + feature circularity — network-free.

Production adoption of ``docs/v2/benchmarks/vigil_truth_validation/
proto_barker_quality.py`` (validated 2026-07-27 on VIGIL physics truth,
C1 n=1000 eval frames):

  GOOD/BAD tip AUC:  Barker bootstrap CCR **0.638** · multi-level
  circularity 0.617 · baseline fft_sharpness 0.586; rank fusion of
  (fft_sharpness + CCR) gives the best continuous association with the true
  tip radius (Spearman −0.453 vs −0.436 baseline).

WORDING DISCIPLINE (measured, not stylistic): stratified by tip class these
criteria support a tip-STATE-TIER judgement (clean/contaminated/changed —
between-class medians 574 vs 65) but NOT a continuous radius measurement
(within-class Spearman −0.215, and +0.270 with the SIGN FLIPPED on
contaminated tips). Never report them as "measured tip radius".

Barker's original method needs a human-picked "good tip reference feature".
MAST has no human in the loop, so the template bank is BOOTSTRAPPED across
frames: significant feature patches are collected from frames already judged
good, k diverse medoids become the bank, and a new frame's CCR is the median
over its features of the best NCC against the bank. Physical meaning follows
Barker: a good tip images point features as the same compact shape; a blunt /
multi / contaminated tip distorts them → CCR drops.

The bank is per-instrument-and-sample STATE. This module provides the
mechanism (build/save/load/score); automatic accumulation into the bank is
deliberately NOT wired anywhere yet — feeding it from an unvalidated auto
"good" judgement would poison the reference. Circularity needs no bank.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

PATCH = 24          # template/feature patch side (atomic-scale features)


# ── shared preprocessing (line-identical to the validated prototype) ────────

def flatten(img: npt.NDArray) -> npt.NDArray[np.float64]:
    """STM topography flattening: row-median align + global 2nd-order poly."""
    h = np.asarray(img, np.float64)
    h = h - np.median(h, axis=1, keepdims=True)
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    yy = yy / H - 0.5
    xx = xx / W - 0.5
    A = np.stack([np.ones_like(xx), xx, yy, xx * yy, xx ** 2, yy ** 2], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    return (h.ravel() - A @ coef).reshape(H, W)


def _ring_stats(A: npt.NDArray[np.float64]):
    """Per-radius median/MAD around the image centre (lookup tables)."""
    H, W = A.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - cy, xx - cx)
    ri = r.astype(int)
    nr = ri.max() + 1
    cnt = np.maximum(np.bincount(ri.ravel(), minlength=nr), 1)
    med = np.bincount(ri.ravel(), A.ravel(), minlength=nr) / cnt
    dev = np.abs(A - med[ri])
    mad = np.bincount(ri.ravel(), dev.ravel(), minlength=nr) / cnt
    return ri, med, 1.4826 * mad + 1e-12


def defeature(h: npt.NDArray[np.float64], zmin: float = 8.0,
              axis_damp: float = 0.2) -> npt.NDArray[np.float64]:
    """De-lattice: notch FFT ring-z>zmin bins (Bragg peaks + 2 px halo) back to
    the ring median; damp the scan-axis cross. Returns the "feature image" —
    only aperiodic content (defects/adsorbates/steps/noise) survives."""
    f = h - h.mean()
    Fs = np.fft.fftshift(np.fft.fft2(f))
    A = np.abs(Fs)
    H, W = A.shape
    cy, cx = H // 2, W // 2
    ri, med, mad = _ring_stats(A)
    z = (A - med[ri]) / mad[ri]
    pk = ndi.binary_dilation(z > zmin, iterations=2)
    Fs = np.where(pk, Fs * (med[ri] / (A + 1e-30)), Fs)
    yy, xx = np.ogrid[:H, :W]
    ax = ((np.abs(yy - cy) <= 1) | (np.abs(xx - cx) <= 1)) & (np.hypot(yy - cy, xx - cx) > 4)
    Fs = np.where(ax, Fs * axis_damp, Fs)
    return np.real(np.fft.ifft2(np.fft.ifftshift(Fs)))


# ── feature extraction ──────────────────────────────────────────────────────

def feature_patches(img: npt.NDArray, max_n: int = 30,
                    snr_min: float = 5.0, min_dist: int = 14):
    """Significant local extrema patches after flatten→defeature (both signs).
    Returns [(patch(PATCH×PATCH, sign-corrected, L2-normalised), y, x, amp)]."""
    h = flatten(np.asarray(img, np.float64))
    if float(h.std()) < 1e-12:
        return []
    g = ndi.gaussian_filter(defeature(h), 1.0)
    a = np.abs(g)
    mad = np.median(np.abs(g - np.median(g))) * 1.4826 + 1e-12
    mx = ndi.maximum_filter(a, size=9)
    ys, xs = np.where((a >= mx) & (a > snr_min * mad))
    if ys.size == 0:
        return []
    order = np.argsort(-a[ys, xs])
    out = []
    half = PATCH // 2
    H, W = g.shape
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        if y < half or x < half or y >= H - half or x >= W - half:
            continue
        if any((y - py) ** 2 + (x - px) ** 2 < min_dist ** 2 for _, py, px, _ in out):
            continue
        p = g[y - half:y + half, x - half:x + half].copy()
        if g[y, x] < 0:
            p = -p                              # concave features flipped convex
        s = float(np.sqrt((p * p).sum()))
        if s < 1e-12:
            continue
        out.append((p / s, y, x, float(a[y, x])))
        if len(out) >= max_n:
            break
    return out


def _ncc(a: npt.NDArray, b: npt.NDArray) -> float:
    x = a - a.mean()
    y = b - b.mean()
    nx, ny = np.sqrt((x * x).sum()), np.sqrt((y * y).sum())
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return float((x * y).sum() / (nx * ny))


# ── template bank ───────────────────────────────────────────────────────────

def build_template_bank(patch_lists: list[list], k: int = 5,
                        diversity_ncc: float = 0.6) -> list[npt.NDArray]:
    """Greedy k-medoid selection from GOOD-frame feature patches. First = most
    typical (highest mean similarity); each next must have NCC < diversity_ncc
    against every chosen template."""
    pool = [p for pl in patch_lists for (p, *_rest) in pl]
    if len(pool) < 3:
        return pool
    n = len(pool)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = _ncc(pool[i], pool[j])
    chosen: list[int] = []
    mean_sim = S.mean(axis=1)
    for _ in range(k):
        best, best_v = None, -9.0
        for i in range(n):
            if i in chosen or any(S[i, j] >= diversity_ncc for j in chosen):
                continue
            if mean_sim[i] > best_v:
                best, best_v = i, float(mean_sim[i])
        if best is None:
            break
        chosen.append(best)
    return [pool[i] for i in chosen]


def save_bank(bank: list[npt.NDArray], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, *[np.asarray(t, np.float32) for t in bank])


def load_bank(path: str | Path) -> list[npt.NDArray]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with np.load(p) as z:
            return [z[k].astype(np.float64) for k in z.files]
    except Exception as exc:  # noqa: BLE001
        logger.warning("barker bank load failed (%s): %s", path, exc)
        return []


# ── frame scores ────────────────────────────────────────────────────────────

def ccr_score(img: npt.NDArray, bank: list[npt.NDArray]) -> tuple[float | None, int]:
    """Frame CCR: median over features of the best template NCC.
    (None, 0) = no bank or no usable features."""
    if not bank:
        return None, 0
    feats = feature_patches(img)
    if not feats:
        return None, 0
    vals = [max(_ncc(p, t) for t in bank) for (p, *_r) in feats]
    return float(np.median(vals)), len(vals)


def circularity_score(img: npt.NDArray, n_levels: int = 4) -> tuple[float | None, int]:
    """Frame feature circularity: per-feature mean of contour-radius std/mean
    over several height levels; frame score = median. Smaller = rounder = a
    better tip. Bank-free and scale-free. (None, 0) = no features."""
    feats = feature_patches(img)
    if not feats:
        return None, 0
    devs = []
    half = PATCH // 2
    for p, *_r in feats:
        pk = float(p[half, half])
        if pk <= 0:
            continue
        vals = []
        for lev in np.linspace(0.35, 0.8, n_levels):
            m = p >= lev * pk
            lab, _ = ndi.label(m)
            m = lab == lab[half, half]
            if m.sum() < 5:
                continue
            ys, xs = np.where(m)
            cy, cx = ys.mean(), xs.mean()
            edge = m & ~ndi.binary_erosion(m)
            ey, ex = np.where(edge)
            if ey.size < 6:
                continue
            r = np.hypot(ey - cy, ex - cx)
            if r.mean() > 1e-9:
                vals.append(float(r.std() / r.mean()))
        if vals:
            devs.append(float(np.mean(vals)))
    if not devs:
        return None, 0
    return float(np.median(devs)), len(devs)


__all__ = ["feature_patches", "build_template_bank", "ccr_score",
           "circularity_score", "save_bank", "load_bank", "flatten",
           "defeature", "PATCH"]
