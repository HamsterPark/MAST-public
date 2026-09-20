"""Scale-adaptive segmentation — mast.vision.seg_scale_adaptive.

The heavy validation lives in docs/v2/benchmarks/vigil_truth_validation/
(150-frame VIGIL eval pools, tune/eval separated); these tests pin the
production packaging: API shape, frozen tuning, the decision summary, and the
behavioural core (no hallucination on clean lattices — the reason this
segmenter exists).
"""
from __future__ import annotations

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
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np  # noqa: E402

from mast.vision.seg_scale_adaptive import (  # noqa: E402
    CLASSES,
    DEFAULTS,
    segment_scale_adaptive,
    summarize_segmentation,
)


def test_classes_match_classical_seg_order():
    from mast.vision.classical_seg import CLASSES as C2
    assert CLASSES == C2 == ["TERRACE", "STEP", "DEFECT", "CONTAMINATION"]


def test_frozen_tuning_pins():
    """The parameter set was tuned on the VIGIL tune pool and frozen — the
    validated eval numbers only apply to THESE values. Changing any of them
    requires re-running the benchmark harness (see module docstring)."""
    assert DEFAULTS["atomic_band_nm"] == (0.18, 0.80)
    assert DEFAULTS["array_band_nm"] == (0.80, 4.00)
    assert DEFAULTS["array_ac_min"] == 0.35
    assert DEFAULTS["reg_k"] == 5.0
    assert DEFAULTS["base_min_frac"] == 0.10


def test_returns_seg_and_info():
    rng = np.random.RandomState(0)
    seg, info = segment_scale_adaptive(rng.randn(128, 128), nm_per_px=0.1)
    assert seg.shape == (128, 128) and seg.dtype == np.uint8
    assert set(np.unique(seg).tolist()) <= {0, 1, 2, 3}
    assert "n_layers" in info and "sigma" in info


def test_flat_frame_graceful():
    seg, info = segment_scale_adaptive(np.zeros((64, 64)), nm_per_px=0.1)
    assert (seg == 0).all()
    assert info.get("flat") is True


def test_pure_noise_no_hallucinated_structure():
    """1/f-free white noise must stay ~terrace: no invented contamination."""
    rng = np.random.RandomState(7)
    seg, _ = segment_scale_adaptive(rng.randn(256, 256), nm_per_px=0.05)
    assert (seg == 3).mean() < 0.05


def test_debug_masks_exposed():
    rng = np.random.RandomState(0)
    _seg, info = segment_scale_adaptive(rng.randn(128, 128), nm_per_px=0.1, debug=True)
    assert "masks" in info and "coarse" in info["masks"]


# ── decision summary (the readout autonomy consumes) ──

def test_summarize_counts_islands_exactly():
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:128, 0:128]
    img = (0.05 * rng.randn(128, 128)).astype(np.float32)
    for cy, cx in ((44, 44), (90, 70)):
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2)).astype(np.float32)
    seg, _ = segment_scale_adaptive(img, nm_per_px=0.2)
    s = summarize_segmentation(seg, 0.2)
    assert s["DEFECT"]["present"] is True
    assert s["DEFECT"]["count"] == 2
    assert 0.0 < s["DEFECT"]["area_frac"] < 0.2
    assert s["TERRACE"]["area_frac"] > 0.7


def test_summarize_empty_classes():
    s = summarize_segmentation(np.zeros((128, 128), np.uint8), 0.1)
    for name in ("STEP", "DEFECT", "CONTAMINATION"):
        assert s[name]["present"] is False
        assert s[name]["count"] == 0
        assert s[name]["area_frac"] == 0.0
    assert s["TERRACE"]["area_frac"] == 1.0


def test_summarize_merges_fragments_physically():
    """Ground truth (and some detectors) shred one physical object into
    dozens of 1-2 px fragments; the physically-sized closing must count ONE
    object, not dozens (the validated eval-harness semantics)."""
    seg = np.zeros((128, 128), np.uint8)
    rng = np.random.RandomState(1)
    for _ in range(40):                       # fragment cloud within ~1 nm
        y, x = 60 + rng.randint(-4, 5), 60 + rng.randint(-4, 5)
        seg[y, x] = 2
    s = summarize_segmentation(seg, nm_per_px=0.05)
    assert s["DEFECT"]["count"] <= 2          # merged, not ~40
