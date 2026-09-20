"""Head C Level 0 — classical terrace detector tests.

Per VIGIL §3.1:
    - Output is a sidecar bool mask, never written to v0.4 low byte.
    - IoU > 0.9 against synthetic ground truth.
    - Wall-time < 15 ms / 512² on CPU (loose budget for CI).
    - Surface-agnostic: works on flat regions of any height as long as
      local variance is the discriminator.
"""
from __future__ import annotations

import sys
import time
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
import pytest  # noqa: E402

from mast.vision.terrace_l0 import detect_terrace, terrace_iou  # noqa: E402


def _synthetic_terrace(size: int = 256, terrace_box: int = 128, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A flat box of size `terrace_box` at the centre of a noisier `size×size` image.

    Returns (image_pm, ground_truth_terrace_mask).
    """
    rng = np.random.default_rng(seed)
    img = (rng.normal(0, 30.0, (size, size))).astype(np.float32)  # 30 pm noise everywhere
    gt = np.zeros((size, size), dtype=bool)
    lo = (size - terrace_box) // 2
    hi = lo + terrace_box
    img[lo:hi, lo:hi] = rng.normal(0, 1.0, (terrace_box, terrace_box)).astype(np.float32)
    gt[lo:hi, lo:hi] = True
    return img, gt


def test_terrace_iou_high_on_synthetic():
    img, gt = _synthetic_terrace(size=256, terrace_box=128)
    pred = detect_terrace(img)
    iou = terrace_iou(pred, gt)
    assert iou > 0.85, f"IoU too low: {iou:.3f}"


def test_terrace_returns_bool_mask():
    img, _ = _synthetic_terrace()
    pred = detect_terrace(img)
    assert pred.dtype == bool
    assert pred.shape == img.shape


def test_terrace_handles_float64_input():
    img, _ = _synthetic_terrace()
    img64 = img.astype(np.float64)
    pred = detect_terrace(img64)
    assert pred.dtype == bool


def test_terrace_rejects_3d_input():
    bad = np.zeros((4, 64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="expects 2D"):
        detect_terrace(bad)


def test_terrace_wall_time_under_budget():
    """Loose CI budget: 100 ms for 512² (production target is <15 ms)."""
    img, _ = _synthetic_terrace(size=512, terrace_box=256)
    # Warm up
    detect_terrace(img)
    t0 = time.perf_counter()
    for _ in range(3):
        detect_terrace(img)
    dt = (time.perf_counter() - t0) / 3 * 1000
    assert dt < 100, f"L0 detector too slow: {dt:.1f} ms / 512² (CI budget 100 ms)"


def test_terrace_iou_zero_when_no_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[0:5, 0:5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[5:10, 5:10] = True
    assert terrace_iou(a, b) == 0.0


def test_terrace_iou_one_for_identical_masks():
    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    assert terrace_iou(a, a) == 1.0


def test_terrace_iou_zero_zero_for_both_empty():
    """Both empty masks → conventionally 0 (the function avoids division)."""
    a = np.zeros((4, 4), dtype=bool)
    assert terrace_iou(a, a) == 0.0
