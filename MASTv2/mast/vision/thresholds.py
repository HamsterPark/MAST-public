"""Tunable VIGIL v2.5 tip-quality discrimination thresholds.

The vision *model weights* are fixed; what "good tip" means is a set of
thresholds that used to be hard-coded inside :meth:`VIGILBackend._fuse_quality`
(coarse good/bad) and :meth:`VIGILBackend.assess_tip_fine` (morphology / usable).
Operators found the built-in cut too strict — too many usable tips were labelled
``bad`` / not-usable and triggered needless tip reconditioning — so those magic
numbers now live here, in ONE place, and are adjustable from 设置 (Settings).

Design (mirrors the ``orchestrator_recursion_limit`` live-read pattern):

  * A process-level holder (:func:`get_thresholds` / :func:`set_thresholds`) keeps
    the ACTIVE snapshot. The vision layer reads it lazily on every assessment;
    the API/runtime layer writes it (startup hydration + each POST /api/settings).
    So a threshold change takes effect on the very next assessment — **no model
    reload**. The vision layer never imports settings; the wiring is one-way.
  * The snapshot is an immutable :class:`VisionThresholds`. The writer swaps the
    reference under a lock; the GPU-thread reader returns the reference lock-free
    (a CPython attribute read is atomic — it never sees a half-updated object).

The pure decision functions :func:`fuse_coarse` / :func:`decide_fine` are the
single source of truth: :class:`VIGILBackend` delegates to them, and the
``tools/vision_threshold_demo.py`` gallery script calls the same functions, so
the demo images always reflect exactly what the deployed model would decide.

**Loosened shipped defaults (2026-07-10)**: measuring the deployed model over 796
real sf09 STM frames showed the OLD cuts accepted only ~3% as "good" / ~4% as
"usable" — pathologically strict (too many usable tips rejected → needless
reconditioning). So the shipped defaults below are intentionally loosened
(good-cut 0.35, N/K/T tolerance 0.70, Q floor 62, S floor 0.45; ~44% good). The
q-norm anchors (55/90) and fusion weights (0.55/0.30/0.15) are UNCHANGED. The
*pure functions* still reproduce the old behaviour bit-for-bit when called with
the OLD values (the ``_STRICT`` set in tests/v2/vision/test_thresholds.py) — only
the shipped default constants moved. Operators re-tune any valve from 设置.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, fields, replace

# ── The 6 valves surfaced in the 设置 UI (per-dimension + overall cut) ─────────
# Q=quality, N=multi-apex, K=contamination, T=instability, S=axis-ratio geometry.
# C (4-class segmentation) has no pass/fail scalar and is deliberately absent.
# The remaining 5 fields (q_lo/q_hi normalisation + the 3 fusion weights) are the
# "expert" tier — kept configurable for completeness but NOT shown in the UI.
EDITABLE_KEYS: tuple[str, ...] = (
    "coarse_good_threshold",  # 总门槛 overall good/bad cut
    "m0_quality_min",         # Q usable-quality floor
    "multi_apex_p_max",       # N multi-apex tolerance
    "contam_p_max",           # K contamination tolerance
    "instability_p_max",      # T instability tolerance
    "m0_axis_ratio_min",      # S apex axis-ratio floor
)

# Defensive clamp ranges (a bad POST can't make the model reject/accept
# everything). Defaults all sit inside their range so clamping is a no-op on the
# shipped config and the zero-regression invariant holds.
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "coarse_q_lo": (0.0, 100.0),
    "coarse_q_hi": (0.0, 100.0),
    "coarse_w_clean": (0.0, 1.0),
    "coarse_w_quality": (0.0, 1.0),
    "coarse_w_axis": (0.0, 1.0),
    "coarse_good_threshold": (0.0, 1.0),
    "multi_apex_p_max": (0.0, 1.0),
    "instability_p_max": (0.0, 1.0),
    "contam_p_max": (0.0, 1.0),
    "m0_quality_min": (0.0, 100.0),
    "m0_axis_ratio_min": (0.0, 1.0),
}


@dataclass(frozen=True)
class VisionThresholds:
    """Immutable snapshot of the tip-quality discrimination thresholds.

    The 6 UI valves (:data:`EDITABLE_KEYS`) default to the loosened 2026-07-10
    values (see the module docstring); the 5 expert fields (q-norm anchors +
    fusion weights) keep their original values. Lower ``*_min`` /
    ``coarse_good_threshold`` and raise ``*_p_max`` to be more lenient (fewer
    tips rejected). The OLD strict cuts were 0.5 / 0.5 / 0.5 / 0.5 / 72 / 0.6.
    """

    # ── coarse good/bad fusion (_fuse_quality) — expert anchors/weights fixed ─
    coarse_q_lo: float = 55.0          # q_norm lower anchor: q≤lo → quality 0
    coarse_q_hi: float = 90.0          # q_norm upper anchor: q≥hi → quality 1
    coarse_w_clean: float = 0.55       # weight of clean=(1-n)(1-k)(1-t)
    coarse_w_quality: float = 0.30     # weight of q_norm
    coarse_w_axis: float = 0.15        # weight of axis_ratio
    coarse_good_threshold: float = 0.35  # fused good≥this → "good" (was 0.50)

    # ── fine morphology / usability (assess_tip_fine) — loosened valves ──────
    multi_apex_p_max: float = 0.70     # n_p>this → multi_tip (M2); was 0.50
    instability_p_max: float = 0.70    # t_p>this → perturbation (M3); was 0.50
    contam_p_max: float = 0.70         # k_p>this → contaminated (M3); was 0.50
    m0_quality_min: float = 62.0       # q≥this (and axis) → M0; was 72.0
    m0_axis_ratio_min: float = 0.45    # ar≥this (and quality) → M0; was 0.60

    def to_mapping(self) -> dict[str, float]:
        """Full config as a JSON-safe ``{str: float}`` dict."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, m: dict | None) -> "VisionThresholds":
        """Build from a (partial) mapping — tolerant of persisted settings.

        Unknown keys are ignored and missing keys fall back to the default, so a
        stored ``vision_thresholds`` from an older/newer build never raises. Each
        recognised numeric value is clamped to :data:`FIELD_BOUNDS`.
        """
        if not m:
            return cls()
        known = {f.name for f in fields(cls)}
        clean: dict[str, float] = {}
        for k, v in m.items():
            if k in known and isinstance(v, (int, float)) and not isinstance(v, bool):
                lo, hi = FIELD_BOUNDS.get(k, (float("-inf"), float("inf")))
                clean[k] = float(min(hi, max(lo, float(v))))
        return replace(cls(), **clean)


# ── process-level active snapshot ────────────────────────────────────────────
_LOCK = threading.Lock()
_ACTIVE = VisionThresholds()  # defaults == historical hard-coded values


def get_thresholds() -> VisionThresholds:
    """Return the active immutable snapshot (lock-free atomic reference read)."""
    return _ACTIVE


def set_thresholds(m: "dict | VisionThresholds | None") -> VisionThresholds:
    """Swap the active snapshot. Accepts a mapping (from settings) or an instance.

    ``None`` / empty resets to defaults. Returns the new active snapshot.
    """
    global _ACTIVE
    new = m if isinstance(m, VisionThresholds) else VisionThresholds.from_mapping(m)
    with _LOCK:
        _ACTIVE = new
    return new


# ── pure decision functions (single source of truth) ─────────────────────────
def fuse_coarse(r: dict, th: VisionThresholds) -> tuple[str, float]:
    """Fuse the 6-head signals into a good/bad label + confidence.

    ``r`` is the raw head dict from ``v25.infer`` (keys ``n_p/k_p/t_p/
    s_axis_ratio/q_score``). N (multi-apex) is the strongest killer, so the
    "all bad-signals clean" product dominates; Q (soft-ordinal quality) and S
    (apex roundness) blend in. With default ``th`` this is bit-identical to the
    old ``VIGILBackend._fuse_quality``.
    """
    n = float(r["n_p"]); k = float(r["k_p"]); t = float(r["t_p"])
    ar = float(r["s_axis_ratio"]); q = float(r["q_score"])
    clean = (1.0 - n) * (1.0 - k) * (1.0 - t)          # all heads must be clean
    span = max(1e-6, th.coarse_q_hi - th.coarse_q_lo)   # default 90-55 = 35
    q_norm = min(1.0, max(0.0, (q - th.coarse_q_lo) / span))
    good = (th.coarse_w_clean * clean
            + th.coarse_w_quality * q_norm
            + th.coarse_w_axis * ar)
    good = min(1.0, max(0.0, good))
    label = "good" if good >= th.coarse_good_threshold else "bad"
    return label, (good if label == "good" else 1.0 - good)


def decide_fine(r: dict, th: VisionThresholds) -> dict:
    """Map the 6 heads onto the fine morphology / usability decision.

    Returns ``{morph, multi_tip, perturbation, contaminated, is_usable}``. With
    default ``th`` this reproduces the old ``VIGILBackend.assess_tip_fine`` logic
    (the caller still derives ``n_tips`` from ``n_p``).
    """
    n_p = float(r["n_p"]); t_p = float(r["t_p"]); k_p = float(r["k_p"])
    q = float(r["q_score"]); ar = float(r["s_axis_ratio"])
    multi_tip = bool(n_p > th.multi_apex_p_max)
    perturbation = bool(t_p > th.instability_p_max)
    contaminated = bool(k_p > th.contam_p_max)
    if multi_tip:
        morph = "M2"
    elif contaminated or perturbation:
        morph = "M3"
    elif q >= th.m0_quality_min and ar >= th.m0_axis_ratio_min:
        morph = "M0"
    else:
        morph = "M1"
    is_usable = morph in ("M0", "M1") and not (perturbation or multi_tip or contaminated)
    return {
        "morph": morph,
        "multi_tip": multi_tip,
        "perturbation": perturbation,
        "contaminated": contaminated,
        "is_usable": is_usable,
    }
