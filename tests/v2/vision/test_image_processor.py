"""STMImageProcessor smoke tests.

Verifies the float32 pm → DINOv3 input pipeline:
    - Output shape (B, 3, target, target)
    - Output dtype float32
    - No NaN / inf
    - Batch dim auto-added for 2D input
    - target_size must be a multiple of patch_size
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

import pytest  # noqa: E402
import torch  # noqa: E402

from mast.vision.image_processor import STMImageProcessor  # noqa: E402


def test_processor_outputs_correct_shape_2d():
    p = STMImageProcessor(target_size=224, patch_size=16)
    x = torch.randn(64, 64) * 100  # 100 pm noise
    out = p(x)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == torch.float32


def test_processor_outputs_correct_shape_batch():
    p = STMImageProcessor(target_size=512, patch_size=16)
    x = torch.randn(4, 256, 256) * 50
    out = p(x)
    assert out.shape == (4, 3, 512, 512)


def test_processor_no_nan_inf_on_random_input():
    p = STMImageProcessor(target_size=512, patch_size=16)
    x = torch.randn(2, 256, 256) * 100
    out = p(x)
    assert torch.isfinite(out).all().item()


def test_processor_no_nan_inf_on_constant_input():
    """Constant image has zero MAD; the +1e-6 epsilon keeps division finite."""
    p = STMImageProcessor(target_size=224, patch_size=16)
    x = torch.full((1, 64, 64), 42.0, dtype=torch.float32)
    out = p(x)
    assert torch.isfinite(out).all().item()


def test_processor_no_nan_inf_on_extreme_outliers():
    p = STMImageProcessor(target_size=224, patch_size=16)
    x = torch.zeros(1, 64, 64)
    x[0, 0, 0] = 1e9  # one extreme outlier
    out = p(x)
    assert torch.isfinite(out).all().item()


def test_processor_target_size_must_match_patch_size():
    with pytest.raises(ValueError, match="multiple of patch_size"):
        STMImageProcessor(target_size=200, patch_size=16)


def test_processor_default_dino_stats():
    p = STMImageProcessor(target_size=224, patch_size=16)
    assert p.DINO_MEAN == (0.485, 0.456, 0.406)
    assert p.DINO_STD == (0.229, 0.224, 0.225)


def test_processor_rejects_4d_input():
    p = STMImageProcessor(target_size=224, patch_size=16)
    x = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError, match="image_pm must be"):
        p(x)
