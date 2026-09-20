"""mast.vision — ms-scale vision module, GPU-thread-owned singleton.

Phase 1 (current): module.py skeleton + _legacy_wrapper.py calls v1
                   ResNet18/UNet/DQN checkpoints (MAST_LEGACY_VISION=1 default).
Phase 9 (VIGIL): vigil_backend.py loads DINOv3-L/16 + LoRA + 5 heads from
                 artifacts/mast_vision_v2.pt. MAST does NOT train these
                 heads — see MAST-reference/compass_artifact_wf-7560c2c7…
                 and ``vigil_backend.VIGILBackend.__doc__`` for the
                 expected checkpoint schema.

Public surface:
    VisionModule              — singleton with backend selection
    Result types              — TipCoarseResult / TipFineResult /
                                SegmentationResult / PartialAssessmentResult /
                                DoubleTipResult / TipMetricsResult /
                                TipChangeResult / TipQualityResult
    mask_codec                — v0.4 uint16 codec (encode / decode /
                                write_surface / set_tipflag)
    image_processor           — STMImageProcessor (float32 pm → DINOv3 input)
    classical_seg             — network-free 4-class segmenter (backs Head C
                                level 0 + the level-1 model-down fallback;
                                beats the learned Head C on synthetic GT)
    double_tip                — algorithmic double-/multi-tip detection
                                (convolution-echo autocorrelation / cepstrum)
    tip_change                — mid-scan tip-change detection (row change-point)
    tip_metrics               — cheap FFT sharpness / fwd-bwd instability /
                                terrace-noise tip-quality signals
    tip_quality               — transparent classical good/bad verdict (fuses the
                                detectors above; interpretable, network-free)
    terrace_l0                — legacy binary terrace detector (standalone util)
    scan_prep                 — 「这一帧该怎么处理」:量出 line_gain / bow_gain /
                                row_purity 等,选平场方式与色阶,并说清楚为什么。
                                它**不产出**「这一帧上有什么」的结论 —— 那些一律转发
                                atomic_phase / tip_change / tip_metrics /
                                scan_artifacts(见 docs/v2/design/
                                scan_prep_auto_flatten.md §2 的实测对照)。
                                阈值按样品体系分 profile(scan_prep_thresholds),
                                换体系先跑 scan_prep_commission 看分布。
    VIGILBackend              — DINOv3 backend (weights pulled in externally)

The classical_seg / double_tip / tip_metrics tools are network-free and
backend-independent — see docs/v2/benchmarks/vision_v25_diagnostic/ for the
study that motivated them.

ALL model forward calls wrapped in torch.inference_mode() + threading.Lock.
P99 latency target: 30 ms at 224² on RTX 4060 (Phase 9 gate).
"""

from mast.vision.module import (
    DoubleTipResult,
    IvResult,
    IzResult,
    PartialAssessmentResult,
    ReplicaCandidate,
    UnifiedTipAssessment,
    ScanArtifactsResult,
    SegmentationResult,
    TipChangeResult,
    TipCoarseResult,
    TipFineResult,
    TipMetricsResult,
    TipQualityResult,
    VisionModule,
)

__all__ = [
    "VisionModule",
    "TipCoarseResult",
    "TipFineResult",
    "SegmentationResult",
    "PartialAssessmentResult",
    "DoubleTipResult",
    "ReplicaCandidate",
    "TipMetricsResult",
    "TipChangeResult",
    "TipQualityResult",
    "ScanArtifactsResult",
    "IzResult",
    "IvResult",
    "UnifiedTipAssessment",
]
