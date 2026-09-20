"""Transparent classical tip good/bad verdict — network-free detector ensemble.

Fuses the cheap classical signals (FFT sharpness / resolution, forward-backward
instability, terrace noise; mid-scan tip change v2) into an *interpretable*
good/bad decision — a drop-in alternative to the deployed model's opaque coarse
label, which on synthetic data barely beat "always bad" and called pure noise
"good".

The rules are deliberately simple and each attaches a human-readable reason:
  BAD if any of —
    * the tip changed mid-scan (v2 null-calibrated z above the per-scale
      threshold; rows below the change were imaged by a different apex),
    * trace and retrace disagree strongly (unstable tip / ringing feedback),
    * NO resolved surface at all (no lattice AND low FFT sharpness) — this is
      the "pure noise → good" failure the learned model has.
  else GOOD, with a confidence built from how comfortably the signals clear
  their thresholds.

Double-tip is still REPORTED (``is_double`` field) but is NO LONGER a BAD rule:
physics-truth validation showed the echo detector is not usable as an alarm —
AUC 0.442 on VIGIL labels (defective ground truth) and 0.625 even on physically
correct injected ghosts, missing ~4 of 5 at FPR 5 %; the small-separation ghost
overlaps its own source (a principled resolution floor, not a tuning problem).
See ``docs/v2/benchmarks/vigil_truth_validation/README.md``.

Thresholds have sensible defaults and can be overridden per instrument (mirrors
mast.vision.thresholds; ``change_threshold=0`` means auto-by-scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mast.vision.module import TipQualityResult


@dataclass(frozen=True)
class TipQualityThresholds:
    instability_max: float = 0.50     # fwd-bwd instability above this → unstable
    sharpness_min: float = 8.0        # below this AND no lattice → no surface
    double_threshold: float = 0.18    # double-tip replica score (reported only)
    change_threshold: float = 0.0     # mid-scan change calibrated z; 0 = auto by scale


def assess_tip_quality_classical(
    image,
    bwd=None,
    nm_per_px: float | None = None,
    thresholds: TipQualityThresholds | None = None,
) -> TipQualityResult:
    """Fused, interpretable classical good/bad verdict. Pass a (2,H,W) trace/
    retrace pair (or ``bwd``) to enable the instability signal."""
    from mast.vision.double_tip import detect_double_tip
    from mast.vision.tip_change import detect_tip_change
    from mast.vision.tip_metrics import assess_tip_classical

    if thresholds is not None:
        th = thresholds
    else:
        # per-instrument live-read holder (retunable from 设置; no reload)
        from mast.vision.classical_thresholds import get_classical_thresholds
        a = get_classical_thresholds()
        th = TipQualityThresholds(
            instability_max=a.tq_instability_max, sharpness_min=a.tq_sharpness_min,
            double_threshold=a.tq_double_threshold, change_threshold=a.tq_change_threshold)
    a = np.asarray(image)
    fwd = a[0] if (a.ndim == 3 and a.shape[0] == 2 and bwd is None) else a
    pair = a if (a.ndim == 3 and a.shape[0] == 2 and bwd is None) else (
        np.stack([np.asarray(fwd), np.asarray(bwd)]) if (bwd is not None and np.asarray(bwd).shape == np.asarray(fwd).shape)
        else fwd)

    m = assess_tip_classical(image, bwd=bwd, nm_per_px=nm_per_px)
    dt = detect_double_tip(fwd, nm_per_px=nm_per_px, threshold=th.double_threshold)
    tc = detect_tip_change(
        pair,
        threshold=(th.change_threshold if th.change_threshold > 0 else None),
        nm_per_px=nm_per_px)

    reasons: list[str] = []
    if tc.changed:
        reasons.append(f"tip changed mid-scan at row {tc.change_row} (z={tc.score:.1f})")
    if m.fwd_bwd_instability is not None and m.fwd_bwd_instability > th.instability_max:
        reasons.append(f"unstable trace/retrace (instability {m.fwd_bwd_instability:.2f})")
    # The "no resolved surface" rule is Bragg-family evidence — at meso scale
    # (sharpness_scale == "off") a big featureless terrace is a perfectly good
    # survey frame, not a bad tip, so the rule is gated off there.
    no_surface = (m.sharpness_scale != "off"
                  and (not m.has_lattice)
                  and (m.fft_sharpness is not None
                       and m.fft_sharpness < th.sharpness_min))
    if no_surface:
        reasons.append(f"no resolved surface (FFT sharpness {m.fft_sharpness:.1f}, no lattice)")

    label = "bad" if reasons else "good"

    # confidence: for BAD, grows with how many/how hard the rules fired; for GOOD,
    # grows with sharpness margin + stability + presence of a resolved lattice.
    if label == "bad":
        conf = float(min(1.0, 0.6 + 0.15 * len(reasons)))
    else:
        s_margin = 0.0
        if m.fft_sharpness is not None:
            s_margin = np.clip((m.fft_sharpness - th.sharpness_min) / (40.0), 0, 1)
        stab = 1.0 - (m.fwd_bwd_instability or 0.0)
        conf = float(np.clip(0.45 + 0.35 * s_margin + 0.20 * stab, 0.0, 1.0))
        if m.has_lattice:
            conf = float(min(1.0, conf + 0.05))

    return TipQualityResult(
        label=label,
        confidence=conf,
        reasons=reasons,
        is_double=bool(dt.is_double),
        tip_changed=bool(tc.changed),
        fwd_bwd_instability=m.fwd_bwd_instability,
        fft_sharpness=m.fft_sharpness,
        has_lattice=bool(m.has_lattice),
        z_noise=m.z_noise,
        flatness=m.flatness,
    )


__all__ = ["assess_tip_quality_classical", "TipQualityThresholds"]
