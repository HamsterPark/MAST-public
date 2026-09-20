"""VisionModule — singleton, GPU-thread-owned, sync (NOT async).

Layered architecture:
  Phase 1 (current): legacy wrapper paths via _legacy_wrapper.py — calls v1
    ResNet18 / AttentionUNet / DuelingDQN checkpoints when MAST_LEGACY_VISION=1
    is set (default). Same Pydantic API; fine-tip downgrades to "unknown".

  Phase 9: VIGIL DINOv3-L/16 + 5 heads (Heads A coarse / B fine 4-sub /
    C hierarchical L0+L1+L2 / D partial). MAST does not train the heads —
    weights are produced by the VIGIL project (see
    MAST-reference/compass_artifact_wf-7560c2c7…_text_markdown.md) and
    pulled in as artifacts/mast_vision_v2.pt. Active when
    MAST_VISION_BACKEND=vigil OR MAST_LEGACY_VISION=0.

Calling pattern from any LangGraph agent (which is async):
    result = await asyncio.to_thread(VisionModule.get().assess_tip_coarse, image)

Never call the methods directly inside `async def` — they're sync GPU calls
that would block the event loop. buffer_no_blocking 钩子（不随仓）
catches some cases but cannot detect this.
"""
from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Result types — frozen, JSON-serializable, suitable for LangGraph state
#
# Backwards compatibility note: the VIGIL extension fields are all
# Optional with defaults, so legacy backend output continues to validate.
# ─────────────────────────────────────────────────────────────────────

class TipCoarseResult(BaseModel):
    """Head A output. 2-class GOOD/BAD with optional scale-aware confidence.

    Legacy backend fills label/confidence; VIGIL backend additionally fills
    `scan_size_nm` (the conditioning input, echoed back) and `embedding_sha`
    (deterministic hash of the [CLS] vector for debugging).
    """
    model_config = ConfigDict(frozen=True)
    label: Literal["good", "bad"]
    confidence: float = Field(ge=0, le=1)
    embedding_sha: str | None = None
    scan_size_nm: float | None = Field(default=None, ge=0)
    # M12 Head Q (continuous sharpness). `tip_radius_nm` = R_tip estimate
    # (= scan_size_nm × 10**sharpness_log10); `sharpness_log10` = the raw
    # log10(R_tip/scan_size) regression output. None on legacy/mock backends.
    tip_radius_nm: float | None = Field(default=None, ge=0)
    sharpness_log10: float | None = None
    # VIGIL v2.5 (ssl_sf09c1) per-head rich signals — None on legacy/mock/M12.
    # `quality_score` = soft-ordinal Q (~60-90, ↑ better); `multi_apex_prob` =
    # P(apex≥2) (the strongest quality signal); `contam_prob` = tip contamination;
    # `instability_prob` = tip unstable during scan; `axis_ratio` = apex roundness
    # ∈(0,1] (1=round); `asym_prob` = apex asymmetry. The `label`/`confidence`
    # are a fusion of these (see vigil_backend._fuse_quality).
    quality_score: float | None = None
    multi_apex_prob: float | None = Field(default=None, ge=0, le=1)
    contam_prob: float | None = Field(default=None, ge=0, le=1)
    instability_prob: float | None = Field(default=None, ge=0, le=1)
    axis_ratio: float | None = Field(default=None, ge=0, le=1)
    asym_prob: float | None = Field(default=None, ge=0, le=1)
    # SAFE-mode override audit trail — see module-level `_safe_override_active`.
    # None whenever the verdict is the model's own; a dict of what the model
    # actually said when SAFE rewrote it to "good".
    safe_mode_raw: dict | None = None


class TipFineResult(BaseModel):
    """Head B output. Four orthogonal sub-heads as per VIGIL §2.2:
        - morph: M0 / M1 / M2 / M3 (4-way CE)
        - switching / drift / perturbation: independent binary BCE

    Pre-VIGIL (legacy or VIGIL T0): label='unknown', top2=[],
        morph/switching/drift/perturbation all None.
    Post-VIGIL T1: label = morph code, sub-head fields populated.
    """
    model_config = ConfigDict(frozen=True)
    label: str       # 'M0' / 'M1' / 'M2' / 'M3' / 'unknown'
    top2: list[tuple[str, float]] = Field(default_factory=list)
    is_usable: bool = True
    # VIGIL Head B sub-head outputs (T1)
    morph: Literal["M0", "M1", "M2", "M3", "unknown"] | None = None
    switching: bool | None = None
    drift: bool | None = None
    perturbation: bool | None = None
    # M12 Head B multi-tip sub-head: `multi_tip` (binary single/multi) and
    # `n_tips` (continuous estimate). None on legacy/mock backends.
    multi_tip: bool | None = None
    n_tips: float | None = Field(default=None, ge=0)
    #: SAFE-mode override audit trail (see :class:`TipCoarseResult`).
    safe_mode_raw: dict | None = None


class SegmentationResult(BaseModel):
    """Head C output. Hierarchical: L0 (2-class terrace), L1 (4-class
    terrace/step/defect/contamination), L2 (27 + TipFlag bits).

    `level` is None for legacy 7-class output; 0/1/2 for VIGIL.
    `mask_rle` is the primary surface mask. `tipflag_*_rle` carry
    spatial TipFlag bits 9 (stability) and 11 (transition zone) when
    Head C L2 produces them; both are empty bytes for L0/L1.
    """
    model_config = ConfigDict(frozen=True)
    mask_rle: bytes
    shape: tuple[int, int]
    class_counts: dict[str, int] = Field(default_factory=dict)
    # VIGIL Head C extensions
    level: Literal[0, 1, 2] | None = None
    classes: list[str] = Field(default_factory=list)  # ordered class names for the mask
    tipflag_stability_rle: bytes = b""
    tipflag_transition_rle: bytes = b""


class PartialAssessmentResult(BaseModel):
    """Head D output. Self-consistency probability at partial scan completion.

    `quality_pred` is the Head A-equivalent on the partial image.
    `self_consistency` (Head D) is the model's predicted KL-distance to its
    own full-image prediction — a calibrated early-stop signal. Legacy
    backend leaves it None.
    """
    model_config = ConfigDict(frozen=True)
    quality_pred: float = Field(ge=0, le=1)
    coarse_label: Literal["good", "bad", "unknown"]
    frac_acquired: float = Field(ge=0, le=1)
    self_consistency: float | None = Field(default=None, ge=0, le=1)


class ReplicaCandidate(BaseModel):
    """One candidate apex-separation vector: "the same feature reappears here".

    A multi tip draws every feature ``n_apex`` times, so the honest output is a
    LIST of displacements, not a boolean — the operator's question is "how many
    times is each step drawn, and how far apart", which one bool cannot answer.
    """
    model_config = ConfigDict(frozen=True)
    dy_px: int
    dx_px: int
    separation_nm: float | None = None
    score: float = Field(ge=0)
    significance: float = 0.0     # peak height in MADs above the same-annulus null
    tiles_agreeing: int = 0       # of 4 quadrants that independently found it


class DoubleTipResult(BaseModel):
    """Algorithmic double-/multi-tip detection. Backend-independent, no network.

    ⚠️ TRI-STATE. ``verdict`` is ``multi_tip`` / ``single_tip`` / ``undetermined``
    and it is the field to read. ``is_double`` is kept only for the four legacy
    call sites and is ``verdict == "multi_tip"`` — so it is **False both when the
    tip is clean and when nothing could be decided**, which is exactly the
    confusion that made this detector silently answer "clean" on a frame the
    operator reads as a textbook multi tip (real-frame audit 2026-08-11, see
    ``reason``). New callers must branch on ``verdict``.

    ``single_tip`` is only ever claimed inside the regime this method was actually
    validated in (isolated aperiodic features on a flat background). On
    morphology-dominated step/terrace frames the method cannot exclude a multi
    tip, so it returns ``undetermined`` rather than a reassuring lie.

    `score` is the raw off-centre replica strength; `significance` is that peak in
    MADs above the same-annulus null — the raw score is NOT comparable between
    frames, the significance is. `candidates` carries the recovered displacement
    vectors. See docs/v2/benchmarks/vision_v25_diagnostic/.
    """
    model_config = ConfigDict(frozen=True)
    is_double: bool
    score: float = Field(ge=0)
    threshold: float = Field(ge=0)
    separation_px: tuple[int, int]
    separation_nm: float | None = None
    method: str = "autocorr"
    verdict: Literal["multi_tip", "single_tip", "undetermined"] = "undetermined"
    reason: str | None = None
    significance: float = 0.0
    candidates: tuple[ReplicaCandidate, ...] = ()


class TipMetricsResult(BaseModel):
    """Cheap, interpretable classical tip-quality signals — no network.

    Complements (and cross-checks) the learned heads, and fixes their documented
    blind spots (e.g. a featureless / pure-noise frame scores low here instead of
    the model's "good"). All fields are None when not computable for the input.
      * `fft_sharpness`   — strongest Bragg-peak prominence (lattice contrast) →
                            lateral resolution / tip sharpness; ↑ = sharper.
      * `resolution_nm`   — finest resolved period from the FFT; ↓ = sharper.
      * `fwd_bwd_instability` ∈[0,1] — normalised |forward − backward| energy;
                            ↑ = tip/feedback unstable during the scan (needs both
                            trace + retrace channels).
      * `z_noise`         — RMS roughness of the flattest terrace region.
      * `flatness`        — fraction of the frame that is flat terrace.

    WORDING DISCIPLINE (physics-truth validation 2026-07-27,
    ``docs/v2/benchmarks/vigil_truth_validation/``): sharpness-family signals
    support a tip-STATE-TIER judgement (clean / contaminated / changed), NOT a
    continuous radius measurement (within-class Spearman is weak and sign-flips
    on contaminated tips). `sharpness_scale` gates the Bragg-family
    interpretation by scale: "full" (< 0.02 nm/px), "reduced" (0.02–0.05),
    "off" (> 0.05 — atomically unresolved, the criterion is physically
    inapplicable; measured rho collapses +0.37 → +0.05); None = scale unknown.
    NOTE `has_lattice` cannot serve as this gate — 97 % of meso frames also
    trigger it (island arrays / moiré give equally sharp FFT peaks).

    `barker_ccr` (needs a bootstrapped good-tip template bank; None without
    one) and `circularity_dev` (bank-free; smaller = rounder features =
    better tip) are the Barker-style criteria — best-in-class for the
    GOOD/BAD tier call (AUC 0.638 / 0.617 vs 0.586 for fft_sharpness)."""
    model_config = ConfigDict(frozen=True)
    fft_sharpness: float | None = None
    resolution_nm: float | None = None
    fwd_bwd_instability: float | None = Field(default=None, ge=0, le=1)
    z_noise: float | None = Field(default=None, ge=0)
    flatness: float | None = Field(default=None, ge=0, le=1)
    has_lattice: bool = False
    # step-edge resolution (10-90% rise of the sharpest edge) ↓ = sharper tip
    edge_resolution_px: float | None = Field(default=None, ge=0)
    edge_resolution_nm: float | None = Field(default=None, ge=0)
    # distinct terrace levels from the height histogram (multi-modal = clean steps)
    n_terrace_levels: int | None = Field(default=None, ge=0)
    # scale validity of the Bragg-family sharpness interpretation (see above)
    sharpness_scale: Literal["full", "reduced", "off"] | None = None
    # Barker-style criteria (tier judgement, not radius measurement)
    barker_ccr: float | None = None
    barker_ccr_n: int = Field(default=0, ge=0)
    circularity_dev: float | None = Field(default=None, ge=0)
    circularity_n: int = Field(default=0, ge=0)


class TipChangeResult(BaseModel):
    """Mid-scan tip-change detection — the apex changed *during* the scan, so the
    rows below `change_row` were imaged by a different tip.

    v2 (2026-07-27, ``docs/v2/benchmarks/vigil_truth_validation/``): `score` is a
    null-calibrated z (per-channel lag-k differenced row statistics, detrended,
    MAD-normalised, best channel vs a no-change null library). May be slightly
    negative on ultra-clean frames (below the null median). The old max-t
    baseline was AUC 0.510 with its 90 % detections bought by an 83 % FPR; v2 at
    the same controlled FPR detects 94.5 % of the *visible* events.

    `lod` is this frame's own detection limit — the minimum row-DC jump
    (~6×MAD of the lag-k dc differences, in INPUT z units) the detector could
    have seen. It turns a negative into a statement: "no change detected AND
    this frame was sensitive to jumps ≥ lod". None when not computable.
    `channel_scores` are the per-channel calibrated peaks; `calib` names the
    null table used ("vigil-c1" atomic / "vigil-c2" meso, picked by nm/px)."""
    model_config = ConfigDict(frozen=True)
    changed: bool
    change_row: int | None = None
    score: float
    threshold: float = Field(ge=0)
    method: str = "v2cal-lagk"
    lod: float | None = Field(default=None, ge=0)
    channel_scores: dict[str, float] = Field(default_factory=dict)
    calib: str | None = None


class TipQualityResult(BaseModel):
    """Transparent classical good/bad tip verdict — a fused, *interpretable*
    alternative to the deployed model's opaque coarse decision. Fixes its blind
    spots by construction: pure noise / no resolved surface → bad; a double tip,
    a mid-scan change, or an unstable trace/retrace → bad, each with a stated
    reason. `reasons` lists exactly which rules fired. Network-free.
    See docs/v2/benchmarks/vision_v25_diagnostic/."""
    model_config = ConfigDict(frozen=True)
    label: Literal["good", "bad"]
    confidence: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    # transparent sub-signals
    is_double: bool = False
    tip_changed: bool = False
    fwd_bwd_instability: float | None = None
    fft_sharpness: float | None = None
    has_lattice: bool = False
    z_noise: float | None = None
    flatness: float | None = None
    #: SAFE-mode override audit trail (see :class:`TipCoarseResult`).
    safe_mode_raw: dict | None = None


class ScanArtifactsResult(BaseModel):
    """Network-free scan-artifact detection — feedback oscillation/ringing,
    thermal drift, and bad scan-lines / spikes. All cheap 1-D/FFT statistics.
    `has_artifact` is the OR of the individual flags. See
    docs/v2/benchmarks/vision_v25_diagnostic/."""
    model_config = ConfigDict(frozen=True)
    has_artifact: bool = False
    oscillation: bool = False
    oscillation_severity: float = Field(default=0.0, ge=0)   # peak prominence
    oscillation_cycles_per_line: float | None = None
    drift_px: float | None = Field(default=None, ge=0)       # fwd↔bwd registration shift
    bad_row_frac: float = Field(default=0.0, ge=0, le=1)
    spike_frac: float = Field(default=0.0, ge=0, le=1)


class IzResult(BaseModel):
    """Tip probe from an I(z) approach curve — a good tip tunnels with a clean
    exponential decay I∝exp(−2κz). `barrier_ev` is the apparent barrier height
    (≈ work function, ~4-5 eV for a clean metal tip; a low/again-noisy value
    signals a blunt or contaminated tip); `fit_r2` is the log-linear fit quality;
    `n_jumps` counts sudden current steps (tip instability during the ramp).
    Network-free, non-image. See docs/v2/benchmarks/vision_v25_diagnostic/."""
    model_config = ConfigDict(frozen=True)
    is_clean_exponential: bool
    barrier_ev: float | None = None
    fit_r2: float = Field(ge=0, le=1)
    decay_per_nm: float | None = None
    n_jumps: int = Field(default=0, ge=0)


class IvResult(BaseModel):
    """Tip probe from an I(V) tunnelling spectrum — a stable tip gives a smooth,
    spike-free, roughly antisymmetric curve. `n_spikes` counts sudden jumps
    (tip switches during the sweep), `symmetry`/`smoothness` ∈[0,1] (↑ better),
    `gap_ev` is the near-zero-conductance gap width if present. Network-free,
    non-image. See docs/v2/benchmarks/vision_v25_diagnostic/."""
    model_config = ConfigDict(frozen=True)
    is_stable: bool
    smoothness: float = Field(ge=0, le=1)
    symmetry: float = Field(ge=0, le=1)
    n_spikes: int = Field(default=0, ge=0)
    gap_ev: float | None = None


class UnifiedTipAssessment(BaseModel):
    """One-call comprehensive tip/scan verdict — fuses every network-free
    classical detector (transparent good/bad + double-tip + mid-scan-change +
    scan artifacts + optional I(z)/I(V)) with the learned ``stm_quality_v1``
    scorer. `overall` is the fused verdict; every sub-result is attached for
    transparency and the reasons are merged. The learned + classical channels are
    complementary — the learned head is the strongest real-data quality signal
    but is single-channel (blind to trace/retrace instability, which the classical
    side covers). See docs/v2/benchmarks/vision_v25_diagnostic/."""
    model_config = ConfigDict(frozen=True)
    overall: Literal["good", "usable", "bad"]
    confidence: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    # learned scorer (stm_quality_v1) — None when its weights/backbone are absent
    dl_quality_score: float | None = None
    dl_quality_tier: str | None = None
    # network-free classical channels (always present)
    classical: TipQualityResult
    artifacts: ScanArtifactsResult
    # optional spectroscopic probes (only when I(z)/I(V) curves are supplied)
    iz: IzResult | None = None
    iv: IvResult | None = None
    #: SAFE-mode override audit trail (see :class:`TipCoarseResult`). Carries the
    #: overall/reasons this assessment WOULD have had; the sub-results attached
    #: above keep their own raw numbers either way.
    safe_mode_raw: dict | None = None


# ─────────────────────────────────────────────────────────────────────
# SAFE-mode verdict override
# ─────────────────────────────────────────────────────────────────────

def _safe_override_active() -> bool:
    """True when the operator selected SAFE and tip verdicts must read "good".

    SAFE's contract is "the tip is fine, keep running experiments". Before
    2026-08-01 that was enforced only by a belief block in the agent's prompt
    while these methods kept returning "bad" — so the model was arguing with its
    own tool results, and a bad verdict still halted the running experiment
    through the buffer's CRITICAL path. Overriding here removes the incentive at
    its source instead of asking the model to ignore it.

    Scope: the tip's *quality/morphology verdict* only. Scan artifacts (drift,
    bad rows, oscillation), raw numeric metrics, and every physical-safety
    signal stay truthful — see the callers, which keep those fields untouched.

    Unbound holder (tests, headless, offline tools) → False, i.e. no override.
    """
    try:
        from mast.core.operating_mode import safe_mode_active
        return safe_mode_active()
    except Exception:  # noqa: BLE001 — never let this break an assessment
        return False


# ─────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────

class VisionModule:
    """GPU-resident vision singleton. Loaded once per process, thread-safe.

    Phase 1: legacy implementations under _legacy_wrapper.py
    Phase 9: DINOv3 ViT-B/16 + 4 heads (filled in this file's body)
    """

    _instance: "VisionModule | None" = None
    _instance_lock = threading.Lock()
    _call_lock = threading.Lock()  # serialize GPU calls

    def __init__(
        self,
        legacy_artifacts_dir: str = "artifacts/legacy",
        vigil_checkpoint_path: str = "MASTv2/artifacts/mast_vision_v25.pt",
        backend: str | None = None,
    ):
        """Build the singleton.

        Args:
            legacy_artifacts_dir: legacy v1 .pth root (used when backend == 'legacy').
            vigil_checkpoint_path: path to the VIGIL v2.5 (ssl_sf09c1) DINOv3-vits16
                + LoRA + 6-head checkpoint.
            backend: 'legacy', 'vigil', or 'mock'. None → resolve from env.
                Resolution order (first match wins):
                  1. MAST_VISION_BACKEND={legacy,vigil,mock}
                  2. MAST_LEGACY_VISION=1 (legacy) or =0 (vigil)
                  3. default → 'vigil' (Phase 9: M12 is the authoritative model)

                The default is 'vigil' ONLY when the M12 checkpoint is actually
                present (bundled / dev); if it is absent and the backend was
                auto-resolved (not explicitly requested), the default quietly
                downgrades to 'legacy' so an unbundled environment still works.

                If the requested backend fails to initialise (torch/timm/peft
                missing, checkpoint or backbone cache absent, etc.), falls back
                to MockBackend so the pipeline stays callable. Construction is
                cheap — the 1.2 GB backbone loads lazily on first inference.
        """
        self._legacy_dir = legacy_artifacts_dir
        self._vigil_path = vigil_checkpoint_path
        self._backend_name = self._resolve_backend(backend)
        # Authoritative-when-available: if we DEFAULTED to vigil but the M12
        # checkpoint isn't there, fall back to legacy before instantiating.
        # An explicit request (backend='vigil' OR MAST_VISION_BACKEND=vigil) is
        # always honoured — only the bare fall-through default downgrades.
        explicitly_vigil = (backend == "vigil") or (
            os.environ.get("MAST_VISION_BACKEND", "").strip().lower() == "vigil"
        )
        if not explicitly_vigil and self._backend_name == "vigil" and not self._vigil_ckpt_present():
            import logging
            logging.getLogger(__name__).info(
                "VisionModule: M12 checkpoint not found at %s; default backend "
                "downgraded vigil → legacy. Set MAST_VISION_BACKEND=vigil to force.",
                self._vigil_path,
            )
            self._backend_name = "legacy"
        self._backend = self._instantiate_backend(self._backend_name)
        self._warmup()

    def _vigil_ckpt_present(self) -> bool:
        """Cheap check (stat only) — is the vision ckpt on disk?

        MUST use the SAME multi-root resolver as VIGILBackend (project_root +
        frozen exe-dir + _MEIPASS). Otherwise the frozen-install case — where
        run_service sets MAST2_PROJECT_ROOT → a SEPARATE data dir (the Inno
        DataDirPage default {sd}\\MAST-data) that does NOT hold the artifacts,
        while the build copies them next to the exe — falsely reports the ckpt
        missing → silently downgrades vigil→legacy→mock, and the launcher's
        vision warm never completes → 3-min "加载超时" (user-reported)."""
        try:
            from pathlib import Path

            from mast.vision.vigil_backend import VIGILBackend
            p = Path(self._vigil_path)
            resolved = p if p.is_absolute() else VIGILBackend._resolve_resource(p)
            return resolved.is_file()
        except Exception:
            return False

    def _instantiate_backend(self, name: str):
        """Try to build *name* backend; fall back to MockBackend on any error."""
        try:
            if name == "legacy":
                from mast.vision._legacy_wrapper import LegacyBackend
                return LegacyBackend(self._legacy_dir)
            if name == "vigil":
                from mast.vision.vigil_backend import VIGILBackend
                return VIGILBackend(self._vigil_path)
            if name == "mock":
                from mast.vision._mock_backend import MockBackend
                return MockBackend(reason="MAST_VISION_BACKEND=mock")
            raise ValueError(f"unknown backend '{name}'")
        except Exception as exc:
            import logging
            from mast.vision._mock_backend import MockBackend
            logging.getLogger(__name__).warning(
                "VisionModule backend '%s' failed to initialise (%s: %s); "
                "falling back to MockBackend",
                name, type(exc).__name__, exc,
            )
            self._backend_name = "mock"
            return MockBackend(
                reason=f"fallback from {name} ({type(exc).__name__})"
            )

    @staticmethod
    def _resolve_backend(explicit: str | None) -> str:
        if explicit is not None:
            if explicit not in ("legacy", "vigil", "mock"):
                raise ValueError(
                    f"backend must be 'legacy', 'vigil' or 'mock', got {explicit!r}"
                )
            return explicit
        env = os.environ.get("MAST_VISION_BACKEND", "").strip().lower()
        if env in ("legacy", "vigil", "mock"):
            return env
        # Phase 9 default: 'vigil' (M12 authoritative). MAST_LEGACY_VISION=1
        # forces the v1 legacy path. The caller (__init__) downgrades vigil →
        # legacy when the M12 checkpoint is absent, so this default is safe.
        legacy_flag = os.environ.get("MAST_LEGACY_VISION", "0").strip()
        return "legacy" if legacy_flag == "1" else "vigil"

    @classmethod
    def get(cls, **kw) -> "VisionModule":
        """Get-or-create the singleton. Thread-safe."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(**kw)
            return cls._instance

    @classmethod
    def prewarm_async(cls) -> None:
        """Load the vision model in a background daemon thread (idempotent).

        The M12 backbone cold-loads in ~30 s (1.2 GB from disk). If that happens
        lazily at the first scan, the scan-progress vision monitor can't fire
        its 12.5 %…100 % milestones until the load finishes — for a short scan
        the whole acquisition is over by then. Pre-warming at GUI/service
        startup means the model is resident before the operator's first scan, so
        the per-progress vision fires promptly. Fail-safe: any error is swallowed
        (vision just stays lazy). No-op once the singleton is loaded."""
        def _w() -> None:
            try:
                vm = cls.get()
                be = getattr(vm, "_backend", None)
                if be is not None and hasattr(be, "preload"):
                    be.preload()  # force the actual weight load
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).info(
                    "vision prewarm skipped (%s: %s)", type(exc).__name__, exc)
        threading.Thread(target=_w, name="vision-prewarm", daemon=True).start()

    def _warmup(self) -> None:
        """One-shot inference on dummy data to JIT compile. Phase 1: stub."""
        pass

    # ── Sync inference API. Always wrap with asyncio.to_thread inside agents. ──

    def assess_tip_coarse(self, image: "np.ndarray") -> TipCoarseResult:
        """Head 1 — binary tip quality.

        Phase 1: legacy wrapper calls v1 ResNet18 / VGG4 from artifacts/legacy/.
        Phase 9: DINOv3 cls-token → coarse_head → softmax.

        In SAFE mode a "bad" label is rewritten to "good" (raw kept in
        ``safe_mode_raw``). This is the single most load-bearing override in the
        system: this method is the only path that produces a ``TipStatus``
        (``scan_monitor._publish_coarse``), so every downstream consumer — the
        buffer's CRITICAL edge, the halt hook, the HITL gate, ``full_scan``'s
        vision note, the agent's ``read_latest_tip_status`` — follows from here.
        The per-head signals (quality_score, multi_apex_prob, …) are left alone.
        """
        with self._call_lock:
            res = self._backend.assess_tip_coarse(image)
        if res.label == "bad" and _safe_override_active():
            return res.model_copy(update={
                "label": "good",
                # Invert rather than pin to 1.0: an honest "we are not measuring
                # this right now" reading, and >0.5 so score-threshold consumers
                # (paper skills default to 0.5/0.8) do not re-derive "bad".
                "confidence": max(1.0 - float(res.confidence), 0.51),
                "safe_mode_raw": {"label": res.label, "confidence": float(res.confidence)},
            })
        return res

    def assess_tip_fine(self, image: "np.ndarray") -> TipFineResult:
        """Head 2 — 8-class tip-failure taxonomy.

        Phase 1: returns "unknown" (legacy is binary).
        Phase 9: DINOv3 cls-token → fine_head → top-2 probabilities.

        In SAFE mode ``is_usable`` is forced True (raw kept in ``safe_mode_raw``);
        the morphology code and sub-head probabilities stay truthful — SAFE says
        "do not repair", not "pretend the apex has a different shape".
        """
        with self._call_lock:
            res = self._backend.assess_tip_fine(image)
        if not res.is_usable and _safe_override_active():
            return res.model_copy(update={
                "is_usable": True,
                "safe_mode_raw": {"is_usable": False, "label": res.label},
            })
        return res

    def segment(
        self,
        image: "np.ndarray",
        classes: list[str] | None = None,
        *,
        level: int | None = None,
        tile: int | None = None,
    ) -> SegmentationResult:
        """Head 3 — semantic segmentation.

        Phase 1: legacy AttentionUNet output mapped to 7-class subset.
        Phase 9 (M12): ``level`` selects the hierarchy —
            0 = terrace mask (classical CV, no network),
            1 = 4-class terrace/step/defect/contamination (default),
            2 = downgrades to L1 (M12 has no 27-class head).
        ``tile`` enables high-resolution sliding-window tiling for large scans
        (None = auto-tile when the image is large; 0 = never; N = tile size).
        Legacy / mock backends ignore ``level`` / ``tile`` (fixed taxonomies).
        """
        with self._call_lock:
            return self._backend.segment(image, classes, level=level, tile=tile)

    def partial_assess(self, scan_lines: "np.ndarray", n_available: int) -> PartialAssessmentResult:
        """Head 4 — assess partial scan during acquisition.

        Phase 1: legacy uses last n_available lines as if they were a full scan.
        Phase 9: DINOv3 with bool_masked_pos for unacquired patches.

        SAFE mode rewrites ``coarse_label`` like the other tip verdicts. No
        production caller today, but ``buffer_summarizer`` already renders this
        label into Chinese for the agent — leaving the one facade method without
        the override is how a hole gets opened later by an unrelated change.
        """
        with self._call_lock:
            res = self._backend.partial_assess(scan_lines, n_available)
        if res.coarse_label == "bad" and _safe_override_active():
            return res.model_copy(update={"coarse_label": "good"})
        return res

    def set_scan_size_nm(self, scan_size_nm: float) -> None:
        """Tell the backend the physical scan size for the next inference.

        The M12 backend conditions Head Q (sharpness) and Head B (tip state) on
        log10(scan_size_nm). Callers that know the scan size (from Nanonis
        ScanFrameSet metadata) should set it before assess_tip_*; absent it the
        backend uses a neutral default. No-op for legacy / mock backends, which
        do not condition on scale.
        """
        setter = getattr(self._backend, "set_scan_size_nm", None)
        if callable(setter):
            with self._call_lock:
                setter(float(scan_size_nm))

    # ── Classical, network-free analyses (backend-independent) ──────────────

    def detect_double_tip(
        self, image: "np.ndarray", nm_per_px: float | None = None, threshold: float = 0.18
    ) -> DoubleTipResult:
        """Algorithmic double-/multi-tip detection (convolution-echo → autocorr
        replica). No network; works with any backend. Most reliable on clean
        feature-bearing frames — see :class:`DoubleTipResult`."""
        from mast.vision.double_tip import detect_double_tip as _detect
        return _detect(image, nm_per_px=nm_per_px, threshold=threshold)

    def assess_tip_classical(
        self, image: "np.ndarray", bwd: "np.ndarray | None" = None, nm_per_px: float | None = None
    ) -> TipMetricsResult:
        """Cheap classical tip-quality signals (FFT sharpness/resolution, fwd-bwd
        instability, terrace noise/flatness). No network; complements — and
        cross-checks — the learned heads. Pass ``bwd`` (retrace) to enable the
        fwd-bwd instability signal, or a (2,H,W) trace/retrace pair as ``image``."""
        from mast.vision.tip_metrics import assess_tip_classical as _assess
        return _assess(image, bwd=bwd, nm_per_px=nm_per_px)

    def segment_classical(
        self, image: "np.ndarray", nm_per_px: float | None = None
    ) -> SegmentationResult:
        """Network-free 4-class segmentation (terrace/step/defect/contamination).
        Backend-independent; on synthetic GT it matches/beats the learned Head C
        and never hallucinates contamination on clean lattices. Same class order
        as :meth:`segment`. Pass ``nm_per_px`` (scan_size_nm / pixels) to set the
        atomic length scale."""
        from mast.vision.classical_seg import segment_classical_result
        return segment_classical_result(image, nm_per_px, level=0)

    def detect_tip_change(
        self, image: "np.ndarray", threshold: float | None = None,
        nm_per_px: float | None = None,
    ) -> TipChangeResult:
        """Detect a tip change *during* the scan (abrupt row-statistics jump).
        Network-free; critical for autonomous scanning — rows below the change
        were imaged by a different apex. v2: null-calibrated lag-k detector;
        ``threshold`` is on the calibrated z (None = auto by scale via
        ``nm_per_px``); accepts a (2,H,W) trace/retrace pair. Also reports the
        frame's own detection limit (`lod`). See :class:`TipChangeResult`."""
        from mast.vision.tip_change import detect_tip_change as _detect
        return _detect(image, threshold=threshold, nm_per_px=nm_per_px)

    def assess_tip_quality(
        self, image: "np.ndarray", bwd: "np.ndarray | None" = None, nm_per_px: float | None = None
    ) -> TipQualityResult:
        """Transparent classical good/bad tip verdict (fuses FFT sharpness,
        fwd-bwd instability, double-tip and mid-scan-change detectors) — an
        interpretable, network-free alternative to the deployed coarse label that
        fixes its blind spots by construction. See :class:`TipQualityResult`.

        In SAFE mode a "bad" verdict is rewritten to "good" with the reasons
        cleared (raw kept in ``safe_mode_raw``) — the reasons are literally the
        tip-repair argument, so leaving them would put the case for repairing the
        tip back into the agent's context. The transparent sub-signals
        (fft_sharpness, fwd_bwd_instability, is_double, tip_changed, …) are left
        untouched: they are measurements, not the verdict."""
        from mast.vision.tip_quality import assess_tip_quality_classical as _assess
        res = _assess(image, bwd=bwd, nm_per_px=nm_per_px)
        if res.label == "bad" and _safe_override_active():
            return res.model_copy(update={
                "label": "good",
                "reasons": [],
                "confidence": max(1.0 - float(res.confidence), 0.51),
                "safe_mode_raw": {"label": res.label, "reasons": list(res.reasons),
                                  "confidence": float(res.confidence)},
            })
        return res

    def assess_quality(self, image: "np.ndarray", scan_size_nm: float | None = None):
        """Learned frame-quality score (``stm_quality_v1``) — frozen DINOv3 +
        a ridge head trained on REAL human labels.

        Additive and independent of :meth:`assess_tip_coarse`: on a locked
        group-disjoint fold of the real gold set this scores Spearman 0.684 /
        AUROC 0.946 where the deployed v2.5 head scores 0.196 / 0.628. Pass
        ``scan_size_nm`` (Nanonis reports it) to apply the scale de-bias.

        Single-channel, so it is blind to trace-vs-retrace instability — pair it
        with :meth:`assess_tip_quality`, which covers exactly that. Raises
        :class:`~mast.vision.quality_model.STMQualityModelMissing` when the head
        or backbone cache is absent. Returns
        :class:`~mast.vision.quality_model.QualityScoreResult`."""
        from mast.vision.quality_model import get_scorer
        with self._call_lock:
            return get_scorer().score(image, scan_size_nm=scan_size_nm)

    def detect_scan_artifacts(
        self, image: "np.ndarray", bwd: "np.ndarray | None" = None
    ) -> ScanArtifactsResult:
        """Detect feedback oscillation/ringing, thermal drift and bad scan-lines /
        spikes (network-free). Pass a (2,H,W) trace/retrace pair (or ``bwd``) to
        enable the drift signal. See :class:`ScanArtifactsResult`."""
        from mast.vision.scan_artifacts import detect_scan_artifacts as _detect
        return _detect(image, bwd=bwd)

    def assess_iz(self, z_nm: "np.ndarray", current: "np.ndarray") -> IzResult:
        """Tip probe from an I(z) approach curve (clean exponential + barrier
        height + tip-jump detection). Network-free, non-image, independent of the
        imaging path — a cross-check on the vision heads. See :class:`IzResult`."""
        from mast.vision.spectroscopy import assess_iz as _iz
        return _iz(z_nm, current)

    def assess_iv(self, bias_v: "np.ndarray", current: "np.ndarray") -> IvResult:
        """Tip probe from an I(V) tunnelling spectrum (smoothness / symmetry / tip
        switching / gap). Network-free, non-image. See :class:`IvResult`."""
        from mast.vision.spectroscopy import assess_iv as _iv
        return _iv(bias_v, current)

    def assess(
        self,
        image: "np.ndarray",
        bwd: "np.ndarray | None" = None,
        scan_size_nm: float | None = None,
        iz: "tuple | None" = None,
        iv: "tuple | None" = None,
        use_learned: bool = True,
    ) -> UnifiedTipAssessment:
        """Comprehensive one-call tip/scan assessment — the single entry point.

        Fuses the network-free classical detectors (:meth:`assess_tip_quality` =
        FFT sharpness + fwd-bwd instability + double-tip + mid-scan-change,
        :meth:`detect_scan_artifacts`, and — when curves are supplied via ``iz`` /
        ``iv`` — the spectroscopic probes) with the learned ``stm_quality_v1``
        scorer (:meth:`assess_quality`) into one ``overall`` verdict + confidence +
        merged reasons, with every sub-result attached.

        ``bwd`` (retrace) enables the instability + drift signals. ``scan_size_nm``
        (from Nanonis) sets the atomic length scale + the learned scale de-bias.
        ``iz`` = ``(z_nm, current)``; ``iv`` = ``(bias_v, current)``.
        ``use_learned=False`` skips the (optional, torch-backed) learned scorer for
        a fully network-free assessment. See :class:`UnifiedTipAssessment`."""
        import numpy as np  # noqa: PLC0415

        a = np.asarray(image)
        H = int(a.shape[-2]) if a.ndim >= 2 else 0
        nm_per_px = (float(scan_size_nm) / H) if (scan_size_nm and H) else None

        classical = self.assess_tip_quality(image, bwd=bwd, nm_per_px=nm_per_px)
        artifacts = self.detect_scan_artifacts(image, bwd=bwd)
        izr = self.assess_iz(*iz) if iz is not None else None
        ivr = self.assess_iv(*iv) if iv is not None else None

        dl_score: float | None = None
        dl_tier: str | None = None
        if use_learned:
            try:
                q = self.assess_quality(image, scan_size_nm=scan_size_nm)
                dl_score, dl_tier = float(q.score), q.tier
            except Exception:  # noqa: BLE001 — scorer is optional (weights/backbone may be absent)
                pass

        # Artifact reasons are scan problems (retune feedback, re-scan) — kept in
        # SAFE. Tip reasons argue for repairing the apex — dropped in SAFE.
        artifact_reasons: list[str] = []
        if artifacts.oscillation:
            artifact_reasons.append("feedback oscillation / ringing")
        if artifacts.drift_px is not None and artifacts.drift_px > 3.0:
            artifact_reasons.append(f"thermal drift ~{artifacts.drift_px:.0f}px (trace vs retrace)")
        if artifacts.bad_row_frac > 0.05:
            artifact_reasons.append(f"bad scan-lines ({artifacts.bad_row_frac:.0%})")
        if artifacts.spike_frac > 0.02:
            artifact_reasons.append(f"pixel spikes ({artifacts.spike_frac:.0%})")
        if artifacts.has_artifact and not artifact_reasons:
            # has_artifact is the OR of the detector's own flags, and the list
            # above is a hand-written mirror of it. When they disagree the
            # verdict comes out "bad" with NO stated reason — for a reader (and
            # for the model) that is the least answerable kind of bad news, and
            # in SAFE it is the one remaining way to imply "repair the tip"
            # without saying anything falsifiable. Say what fired instead.
            artifact_reasons.append("scan artifact detected (see artifacts sub-result)")

        # Spectroscopic probes + the learned tier are also tip-quality channels.
        probe_reasons: list[str] = []
        if izr is not None and not izr.is_clean_exponential:
            probe_reasons.append("I(z) not a clean exponential (blunt/unstable tip)")
        if ivr is not None and not ivr.is_stable:
            probe_reasons.append("I(V) unstable / tip switching")
        dl_bad = dl_tier == "bad"
        dl_excellent = dl_tier == "excellent"
        if dl_bad:
            probe_reasons.append(f"learned quality low (tier=bad, score={dl_score:.2f})")

        safe = _safe_override_active()

        if safe:
            # SAFE: only scan artifacts can still pull the verdict down. The I(z)/
            # I(V) probes and the learned tier are tip-quality channels, so they
            # follow the same rule as the classical verdict; their raw sub-results
            # stay attached (izr/ivr/dl_score) for anyone who asks.
            reasons = artifact_reasons
            hard_bad = artifacts.has_artifact
            overall = "bad" if hard_bad else ("usable" if reasons else "good")
        else:
            # Order preserved from before the SAFE split: classical, then scan
            # artifacts, then the spectroscopic/learned probes.
            reasons = list(classical.reasons) + artifact_reasons + probe_reasons
            # BAD if the classical verdict is bad, a scan artifact fired, or the learned
            # scorer says bad; GOOD only if nothing fired AND (no learned score OR it is
            # excellent); USABLE in between (minor concerns / only-marginal learned).
            hard_bad = classical.label == "bad" or artifacts.has_artifact or dl_bad
            if hard_bad:
                overall = "bad"
            elif reasons or (dl_score is not None and not dl_excellent):
                overall = "usable"
            else:
                overall = "good"

        if overall == "bad":
            conf = min(1.0, 0.6 + 0.1 * len(reasons))
        elif overall == "good":
            conf = min(1.0, classical.confidence + (0.1 if dl_excellent else 0.0))
        else:
            conf = max(0.4, 0.7 * classical.confidence)

        safe_raw = None
        if safe and (probe_reasons or classical.safe_mode_raw is not None):
            safe_raw = {"tip_reasons": list(classical.reasons) + probe_reasons,
                        "classical": classical.safe_mode_raw,
                        "dl_quality_tier": dl_tier}

        return UnifiedTipAssessment(
            overall=overall, confidence=float(conf), reasons=reasons,
            dl_quality_score=dl_score, dl_quality_tier=dl_tier,
            classical=classical, artifacts=artifacts, iz=izr, iv=ivr,
            safe_mode_raw=safe_raw,
        )


__all__ = [
    "VisionModule",
    "TipCoarseResult",
    "TipFineResult",
    "DoubleTipResult",
    "ReplicaCandidate",
    "TipMetricsResult",
    "TipChangeResult",
    "TipQualityResult",
    "ScanArtifactsResult",
    "IzResult",
    "IvResult",
    "UnifiedTipAssessment",
    "SegmentationResult",
    "PartialAssessmentResult",
]
