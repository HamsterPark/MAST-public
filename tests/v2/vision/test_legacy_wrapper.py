"""LegacyBackend tests — verify real v1 .pth checkpoint loading + inference.

Skipped automatically when v1's checkpoints aren't present (CI without
artifacts). When run on the dev box with `models/tip_classifier_vgg4/` and
`models/defect_segmenter_attn_v2/` populated, it exercises the real torch
forward pass on a synthetic STM image.
"""
from __future__ import annotations

import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np
import pytest

from mast.vision._legacy_wrapper import LegacyBackend, _rle_decode, _rle_encode
from mast.vision.module import (
    PartialAssessmentResult,
    SegmentationResult,
    TipCoarseResult,
    TipFineResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
HAVE_V1_MODELS = (MODELS_DIR / "tip_classifier_vgg4" / "metadata.json").exists()


# ─────────────────────────────────────────────────────────────────────
# RLE encoding (no torch needed)
# ─────────────────────────────────────────────────────────────────────

class TestRLEEncoding:
    """Unified multi-class value-run codec (mast.vision.seg_utils). Tests are
    round-trip (encode→decode == identity), so they are format-agnostic."""

    def test_all_zeros(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        out = _rle_decode(_rle_encode(mask), mask.shape)
        assert np.array_equal(out, mask)

    def test_all_ones(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        out = _rle_decode(_rle_encode(mask), mask.shape)
        assert np.array_equal(out, mask)

    def test_alternating_pattern(self):
        mask = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
        out = _rle_decode(_rle_encode(mask), mask.shape)
        assert np.array_equal(out, mask)

    def test_multiclass_round_trip(self):
        """The whole point of Phase 9: 4 classes survive the round-trip."""
        rng = np.random.default_rng(0)
        mask = rng.integers(0, 4, size=(16, 24), dtype=np.uint8)
        out = _rle_decode(_rle_encode(mask), mask.shape)
        assert np.array_equal(out, mask)
        assert set(np.unique(out).tolist()) <= {0, 1, 2, 3}


# ─────────────────────────────────────────────────────────────────────
# LegacyBackend (graceful when no v1 checkpoint present)
# ─────────────────────────────────────────────────────────────────────

class TestLegacyBackendGraceful:
    """Behavior when v1 model dir is missing.

    MAST2 v2.0.0 update: LegacyBackend.__init__ now fails fast with
    FileNotFoundError if no .pth files are present so VisionModule can
    fall back to MockBackend at the wrapper level. The wrapper-level
    fallback is verified in tests/v2/vision/test_mock_backend.py.
    """

    def test_missing_checkpoint_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No legacy vision checkpoints"):
            LegacyBackend(artifacts_dir=tmp_path)

    def test_missing_artifacts_dir_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="No legacy vision checkpoints"):
            LegacyBackend(artifacts_dir=missing)


# ─────────────────────────────────────────────────────────────────────
# Real v1 model loading (skipped if not present)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not HAVE_V1_MODELS, reason="v1 checkpoints not available on this machine"
)
class TestLegacyBackendRealModels:
    """Exercise actual torch forward pass on synthetic STM image."""

    @pytest.fixture(scope="class")
    def backend(self):
        # Use real models/ directory
        b = LegacyBackend(artifacts_dir=MODELS_DIR)
        return b

    def test_loads_tip_classifier(self, backend):
        backend._ensure_loaded()
        assert backend._tip_model is not None

    def test_loads_segmenter(self, backend):
        backend._ensure_loaded()
        # Segmenter may or may not load depending on which subdir is configured
        # Just verify no exception thrown
        assert backend._models_loaded

    def test_assess_tip_coarse_returns_valid_result(self, backend):
        # Synthetic STM image: random noise + low-frequency bias
        img = np.random.rand(64, 64).astype(np.float32)
        result = backend.assess_tip_coarse(img)
        assert isinstance(result, TipCoarseResult)
        assert result.label in ("good", "bad")
        assert 0 <= result.confidence <= 1

    def test_assess_tip_fine_downgrades_to_unknown(self, backend):
        img = np.random.rand(64, 64).astype(np.float32)
        result = backend.assess_tip_fine(img)
        assert isinstance(result, TipFineResult)
        # v1 has no 8-class taxonomy → must downgrade
        assert result.label == "unknown"

    def test_segment_returns_mask_with_expected_shape(self, backend):
        img = np.random.rand(128, 128).astype(np.float32)
        result = backend.segment(img)
        assert isinstance(result, SegmentationResult)
        assert len(result.shape) == 2
        # mask_rle decodes to a class map covering exactly H*W pixels
        decoded = _rle_decode(result.mask_rle, result.shape)
        assert decoded.shape == result.shape

    def test_partial_assess_at_30pct(self, backend):
        scan = np.random.rand(256, 256).astype(np.float32)
        result = backend.partial_assess(scan, n_available=int(256 * 0.3))
        assert isinstance(result, PartialAssessmentResult)
        assert 0.25 <= result.frac_acquired <= 0.35


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
