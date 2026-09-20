"""Real M12 inference test — loads the actual DINOv3-ViT-L/16 + LoRA + 3 heads
and runs every VisionModule head on a synthesized STM image.

This is the end-to-end proof that the vendored VIGIL closure
(``mast.vision._vigil``) + the M12 checkpoint + the bundled backbone cache
produce valid, well-typed results with the LoRA actually loaded (no silent
key-mismatch → no silently-wrong model).

Gated OFF by default — loading the 1.2 GB backbone on CPU takes ~30 s and needs
the weights present. Enable with::

    MAST_TEST_M12_REAL=1 .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/vision/test_vigil_backend_m12_real.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast._runtime_paths import project_root  # noqa: E402

_ROOT = project_root()
_CKPT = _ROOT / "MASTv2" / "artifacts" / "mast_vision_m12.pt"
_CACHE = _ROOT / "MASTv2" / "artifacts" / "vision_backbone"

_ENABLED = os.environ.get("MAST_TEST_M12_REAL", "").strip() == "1"
_WEIGHTS = _CKPT.is_file() and _CACHE.is_dir()

pytestmark = pytest.mark.skipif(
    not (_ENABLED and _WEIGHTS),
    reason="set MAST_TEST_M12_REAL=1 and provide M12 ckpt + backbone cache to run",
)


@pytest.fixture(scope="module")
def backend():
    from mast.vision.vigil_backend import VIGILBackend

    be = VIGILBackend(checkpoint_path=str(_CKPT), backbone_cache_dir=str(_CACHE),
                      device="cpu")
    be.preload()  # forces load + the LoRA-key-match safety assertion
    assert be.is_loaded()
    return be


def _synthetic(seed: int = 7, h: int = 256, w: int = 256):
    rng = np.random.default_rng(seed)
    terr = (np.mgrid[0:h, 0:w][1] // 85).astype(np.float32) * 30.0
    fwd = terr + rng.normal(0, 2.0, (h, w)).astype(np.float32)
    bwd = terr + rng.normal(0, 2.0, (h, w)).astype(np.float32)
    return np.stack([fwd, bwd])  # (2, H, W) → fwd/bwd channels


def test_loads_with_lora_matched(backend):
    assert backend._model.backbone_name == "dinov3-vitl16"
    assert backend._model.lora_rank == 8
    assert backend._model.embed_dim == 1024


def test_assess_tip_coarse_real(backend):
    from mast.vision.module import TipCoarseResult

    backend.set_scan_size_nm(10.0)
    r = backend.assess_tip_coarse(_synthetic())
    assert isinstance(r, TipCoarseResult)
    assert r.label in ("good", "bad")
    assert 0.0 <= r.confidence <= 1.0
    assert r.scan_size_nm == 10.0
    assert r.tip_radius_nm is not None and r.tip_radius_nm >= 0.0
    assert r.sharpness_log10 is not None
    assert r.embedding_sha and len(r.embedding_sha) == 12


def test_assess_tip_fine_real(backend):
    from mast.vision.module import TipFineResult

    backend.set_scan_size_nm(10.0)
    r = backend.assess_tip_fine(_synthetic())
    assert isinstance(r, TipFineResult)
    assert r.morph in ("M0", "M1", "M2", "M3")
    assert r.label == r.morph
    assert len(r.top2) == 2
    for name, p in r.top2:
        assert name in ("M0", "M1", "M2", "M3")
        assert 0.0 <= p <= 1.0
    assert isinstance(r.switching, bool)
    assert isinstance(r.drift, bool)
    assert isinstance(r.perturbation, bool)
    assert isinstance(r.multi_tip, bool)
    assert r.n_tips is not None and r.n_tips >= 0.0


def test_segment_l1_real(backend):
    from mast.vision._legacy_wrapper import _rle_decode
    from mast.vision.module import SegmentationResult

    r = backend.segment(_synthetic())
    assert isinstance(r, SegmentationResult)
    assert r.level == 1
    assert r.classes == ["TERRACE", "STEP", "DEFECT", "CONTAMINATION"]
    assert sum(r.class_counts.values()) == r.shape[0] * r.shape[1]
    mask = _rle_decode(r.mask_rle, r.shape)
    assert mask.shape == r.shape
    assert set(np.unique(mask).tolist()) <= {0, 1, 2, 3}


def test_segment_l0_real_classical_cv(backend):
    r = backend.segment(_synthetic(), level=0)
    assert r.level == 0
    assert "TERRACE" in r.class_counts


def test_segment_level2_downgrades_to_l1_real(backend):
    r = backend.segment(_synthetic(), level=2)
    assert r.level == 1  # M12 has no L2 → downgraded


def test_partial_assess_real(backend):
    backend.set_scan_size_nm(10.0)
    img = _synthetic()[0]  # single channel (H, W)
    r = backend.partial_assess(img, n_available=img.shape[0] // 2)
    assert r.coarse_label in ("good", "bad")
    assert 0.0 <= r.quality_pred <= 1.0
    assert abs(r.frac_acquired - 0.5) < 0.01
    assert r.self_consistency is None  # M12 has no Head D


def test_segment_tiled_large_real(backend):
    """Every L1 path reports the scan's NATIVE resolution. Tiling preserves it
    by construction; the standard 256-resize path now nearest-neighbour upsamples
    its 256×256 model output back to the scan grid so downstream coordinate
    scaling (RegionMap → GUI overlay) maps mask pixels onto the right scan coords."""
    from mast.vision._legacy_wrapper import _rle_decode
    from mast.vision.module import SegmentationResult

    big = _synthetic(h=640, w=640)  # > AUTO_TILE_THRESHOLD (512) → auto-tiles
    r = backend.segment(big)        # auto
    assert isinstance(r, SegmentationResult)
    assert r.level == 1
    assert r.shape == (640, 640)    # native resolution preserved (NOT 256)
    m = _rle_decode(r.mask_rle, r.shape)
    assert m.shape == (640, 640)
    assert set(np.unique(m).tolist()) <= {0, 1, 2, 3}
    # forced tile size also yields native resolution
    assert backend.segment(big, tile=256).shape == (640, 640)
    # tile=0 forces the standard (non-tiled) resize path; it must STILL report
    # the scan's native (640, 640) — the 256×256 model output is upsampled back
    # (fix: the mask used to leak the 256 model-input size to the caller).
    assert backend.segment(big, tile=0).shape == (640, 640)


def test_determinism_same_input_same_sha(backend):
    backend.set_scan_size_nm(10.0)
    a = backend.assess_tip_coarse(_synthetic(seed=42))
    b = backend.assess_tip_coarse(_synthetic(seed=42))
    assert a.embedding_sha == b.embedding_sha
    assert a.label == b.label


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
