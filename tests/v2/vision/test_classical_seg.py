"""Classical (network-free) 4-class STM segmenter — mast.vision.classical_seg.

Validates the segmenter that, on VIGIL synthetic GT, beats the learned Head C
(C1 0.29 vs 0.20, C2 0.27 vs 0.03) because it does not hallucinate contamination
on clean lattices. See docs/v2/benchmarks/vision_v25_diagnostic/.
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

from mast.vision.classical_seg import (  # noqa: E402
    CLASSES,
    class_counts,
    segment_classical,
    segment_classical_result,
)


def _lattice(n=128, period=8, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
    return (img + 0.03 * rng.randn(n, n)).astype(np.float32)


def test_classes_order_matches_head_c():
    assert CLASSES == ["TERRACE", "STEP", "DEFECT", "CONTAMINATION"]


def test_clean_lattice_is_mostly_terrace():
    """The whole point: a clean atomic lattice must NOT be called contamination."""
    seg = segment_classical(_lattice(), nm_per_px=0.03)
    assert seg.shape == (128, 128)
    assert set(np.unique(seg).tolist()) <= {0, 1, 2, 3}
    frac_terrace = (seg == 0).mean()
    frac_contam = (seg == 3).mean()
    assert frac_terrace > 0.85            # dominated by terrace
    assert frac_contam < 0.05             # ~no hallucinated contamination


def test_flat_image_all_terrace():
    seg = segment_classical(np.ones((64, 64), np.float32) * 3.14)
    assert (seg == 0).all()


def _lattice_sum(n=128, period=7.3, seed=0, noise=0.03, amp=0.2):
    """Physically-shaped lattice: SUM of two rotated wave-vectors (each makes
    its own Bragg peak — a sin·sin PRODUCT only has sum/difference
    frequencies, which no real lattice image shows)."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    th = 0.3
    x2 = xx * np.cos(th) + yy * np.sin(th)
    y2 = -xx * np.sin(th) + yy * np.cos(th)
    img = amp * (0.5 * np.sin(x2 * 2 * np.pi / period + 0.7)
                 + 0.5 * np.sin(y2 * 2 * np.pi / (period * 1.13) + 1.1))
    return (img + noise * rng.randn(n, n)).astype(np.float32)


def test_finds_an_adsorbate_on_lattice():
    """Gaussian adsorbate at a physical amplitude ratio (~15× the lattice
    corrugation, ~0.3 nm wide at 0.03 nm/px) on a resolved lattice."""
    yy, xx = np.mgrid[0:128, 0:128]
    img = _lattice_sum()
    img += 3.0 * np.exp(-((yy - 44) ** 2 + (xx - 44) ** 2) / (2 * 5.0 ** 2)).astype(np.float32)
    seg = segment_classical(img, nm_per_px=0.03)
    assert (seg[36:52, 36:52] == 2).sum() > 0


def test_finds_meso_islands_and_counts_them():
    """Two islands on a flat meso terrace → both found, object count exact."""
    from mast.vision.seg_scale_adaptive import summarize_segmentation

    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:128, 0:128]
    img = (0.05 * rng.randn(128, 128)).astype(np.float32)
    for cy, cx in ((44, 44), (90, 70)):
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2)).astype(np.float32)
    seg = segment_classical(img, nm_per_px=0.2)
    assert (seg[36:52, 36:52] == 2).sum() > 0
    assert (seg[82:98, 62:78] == 2).sum() > 0
    s = summarize_segmentation(seg, 0.2)
    assert s["DEFECT"]["present"] is True
    assert s["DEFECT"]["count"] == 2


def test_finds_a_step():
    """SLANTED step (real steps always cross the fast axis at an angle; a
    perfectly axis-parallel step is absorbed by robust row alignment — a
    synthetic artefact, documented in seg_scale_adaptive)."""
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    img = (((xx + 0.35 * yy) > 76) * 5.0
           + 0.05 * np.random.RandomState(1).randn(n, n)).astype(np.float32)
    seg = segment_classical(img, nm_per_px=0.5)
    assert (seg == 1).sum() > 0            # a STEP ridge was found


def test_accepts_trace_retrace_pair():
    pair = np.stack([_lattice(seed=1), _lattice(seed=2)])   # (2, H, W)
    seg = segment_classical(pair, nm_per_px=0.03)
    assert seg.shape == (128, 128)


def test_class_counts_sum_to_pixels():
    seg = segment_classical(_lattice(), nm_per_px=0.03)
    cc = class_counts(seg)
    assert set(cc) == set(CLASSES)
    assert sum(cc.values()) == seg.size


def test_segment_classical_result_is_valid_segmentation_result():
    from mast.vision.module import SegmentationResult
    from mast.vision.seg_utils import decode_rle

    r = segment_classical_result(_lattice(), nm_per_px=0.03, level=0)
    assert isinstance(r, SegmentationResult)
    assert r.level == 0
    assert r.classes == CLASSES
    assert sum(r.class_counts.values()) == r.shape[0] * r.shape[1]
    mask = decode_rle(r.mask_rle, r.shape)
    assert mask.shape == r.shape
    assert set(np.unique(mask).tolist()) <= {0, 1, 2, 3}


def test_vision_module_segment_classical_delegates():
    from mast.vision.module import VisionModule

    vm = VisionModule(backend="mock")     # no torch / model load
    r = vm.segment_classical(_lattice(), nm_per_px=0.03)
    assert r.level == 0 and r.classes == CLASSES
    assert (np.array(list(r.class_counts.values())).sum()) == r.shape[0] * r.shape[1]


# ── segment_terraces (per-terrace-level map) ─────────────────────────────────
def test_segment_terraces_flat_is_one_level():
    from mast.vision.classical_seg import segment_terraces
    lv = segment_terraces((np.ones((128, 128)) + 0.01 * np.random.RandomState(0).randn(128, 128)).astype(np.float32))
    assert len(set(lv.ravel().tolist())) == 1     # a flat frame is one terrace


def test_segment_terraces_staircase_multi_level():
    from mast.vision.classical_seg import segment_terraces
    n = 128
    xx = np.mgrid[0:n, 0:n][1]
    stair = ((xx // 43) * 5.0 + 0.05 * np.random.RandomState(0).randn(n, n)).astype(np.float32)
    lv = segment_terraces(stair, nm_per_px=0.5)
    n_lv = len(set(lv.ravel().tolist()))
    assert 2 <= n_lv <= 5                          # multiple terraces (count approximate)
    assert lv.dtype == np.uint8 and lv.shape == (n, n)


# ── detect_defects (lattice-fit residual) ────────────────────────────────────
def test_detect_defects_finds_planted_defects():
    from mast.vision.classical_seg import detect_defects
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    lat = (np.sin(xx * 2 * np.pi / 8) * np.sin(yy * 2 * np.pi / 8)).astype(np.float32)
    for y, x in [(30, 30), (60, 80), (90, 40)]:
        lat[y - 2:y + 2, x - 2:x + 2] += 4.0
    mask, n_def = detect_defects(lat + 0.03 * np.random.RandomState(0).randn(n, n), nm_per_px=0.03)
    assert mask.dtype == np.bool_ and mask.shape == (n, n)
    assert n_def >= 3                              # the 3 planted defects (+maybe noise)


def test_detect_defects_clean_lattice_none():
    from mast.vision.classical_seg import detect_defects
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    clean = (np.sin(xx * 2 * np.pi / 8) * np.sin(yy * 2 * np.pi / 8)
             + 0.03 * np.random.RandomState(1).randn(n, n)).astype(np.float32)
    _, n_def = detect_defects(clean, nm_per_px=0.03)
    assert n_def <= 2                              # a clean lattice has ~no defects
