"""Mock vision backend — returns deterministic placeholder results without torch.

Active when MAST_VISION_BACKEND=mock OR when the legacy backend cannot find
its v1 checkpoints / cannot import torch. This is the right fallback for the
MAST distribution while DINOv3 weights are still being trained — the
pipeline stays callable, every head returns a "best-effort" answer, and the
operator clearly sees the unknown-confidence flag.

Fail-safe stance:
  A mock that has NO model must not claim the tip is 'good'. Reporting
  'good' at confidence 0.5 silently sails past any downstream gate that
  only blocks on label=='bad' (or that treats 0.5 as "lean good"), so a
  completely un-assessed tip would be approved for tunnelling. That is the
  opposite of safe.

  Instead the mock now degrades pessimistically:
    - assess_tip_coarse → label='bad', confidence=0.5
        (low confidence so confidence-gated callers still pause, but the
         label itself never auto-approves the tip)
    - assess_tip_fine  → is_usable=False, label='unknown'
    - segment          → empty mask (no false defects injected; segmentation
         is not a safety gate)
    - partial_assess   → quality_pred=0.0, coarse_label='unknown'
         (do not encourage an early stop on the strength of a mock)

Calling pattern stays identical to LegacyBackend / VIGILBackend — drop in
without touching any agent / skill code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .module import (
    PartialAssessmentResult,
    SegmentationResult,
    TipCoarseResult,
    TipFineResult,
)

logger = logging.getLogger(__name__)


class MockBackend:
    """Torch-free placeholder. Logs once at construction; no model loading."""

    def __init__(self, reason: str = "MAST_VISION_BACKEND=mock") -> None:
        self._reason = reason
        logger.info("VisionModule using MockBackend (%s)", reason)

    def assess_tip_coarse(self, image):  # noqa: ANN001
        # Fail-safe: a model-less mock must NOT report 'good'. See module docstring.
        return TipCoarseResult(label="bad", confidence=0.5, embedding_sha=None)

    def assess_tip_fine(self, image):  # noqa: ANN001
        # Fail-safe: no model → not usable.
        return TipFineResult(label="unknown", top2=[], is_usable=False)

    def segment(self, image, classes=None, *, level=None, tile=None):  # noqa: ANN001
        try:
            shape = (int(image.shape[0]), int(image.shape[1]))  # type: ignore[attr-defined]
        except Exception:
            shape = (256, 256)
        return SegmentationResult(
            mask_rle=b"",
            shape=shape,
            class_counts={"TERRACE": shape[0] * shape[1]},
        )

    def partial_assess(self, scan_lines, n_available):  # noqa: ANN001
        try:
            total = int(scan_lines.shape[0])  # type: ignore[attr-defined]
        except Exception:
            total = 1
        n_avail = max(0, min(int(n_available), total))
        frac = n_avail / total if total > 0 else 0.0
        return PartialAssessmentResult(
            # Fail-safe: do not encourage an early stop on a mock's say-so.
            quality_pred=0.0,
            coarse_label="unknown",
            frac_acquired=frac,
            self_consistency=None,
        )


__all__ = ["MockBackend"]
