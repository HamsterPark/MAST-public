"""Fast structural tests for the vendored VIGIL closure (mast.vision._vigil).

These exercise the head modules + channel/scale processors WITHOUT the 1.2 GB
DINOv3 backbone or the M12 checkpoint (random-init weights, tiny tensors), so
they run in the normal suite and guard the vendoring (imports resolve, forward
shapes are right, the double-log scale embedding + register slicing behave).
The full real-weights path is in test_vigil_backend_m12_real.py (gated).
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

torch = pytest.importorskip("torch")  # noqa: E402


def test_vendored_package_imports():
    """The whole closure must import with no dangling vigil.* references."""
    from mast.vision._vigil import m12  # noqa: F401
    from mast.vision._vigil.backbone import dinov_loader, lora_config, wrapper  # noqa: F401
    from mast.vision._vigil.data import multichannel_processor, stm_image_processor  # noqa: F401
    from mast.vision._vigil.heads import (  # noqa: F401
        head_b_v2, head_c_l1_common, head_c_l1_dino, head_q_sharpness,
    )


def test_compose_channels_e3_is_fwd_bwd_diff():
    from mast.vision._vigil.data.multichannel_processor import compose_channels

    fwd = torch.ones(1, 8, 8)
    bwd = torch.zeros(1, 8, 8)
    out = compose_channels(fwd, bwd, mode="E3")
    assert out.shape == (1, 3, 8, 8)
    assert torch.allclose(out[:, 0], fwd)            # ch0 = fwd
    assert torch.allclose(out[:, 1], bwd)            # ch1 = bwd
    assert torch.allclose(out[:, 2], fwd - bwd)      # ch2 = fwd − bwd


def test_scale_embedding_double_log_shape_and_finite():
    """ScaleEmbedding is fed log10(scan) by m12.infer and logs again — verify
    it stays finite for the realistic [1e-3, 1e3] nm range (the verified path)."""
    from mast.vision._vigil.heads.head_q_sharpness import ScaleEmbedding

    emb = ScaleEmbedding(dim=64).eval()
    for scan in (0.001, 1.0, 5.0, 10.0, 100.0, 1000.0):
        log_scan = torch.log10(torch.tensor([scan]).clamp(min=1e-3))
        out = emb(log_scan)
        assert out.shape == (1, 64)
        assert torch.isfinite(out).all()


def test_head_q_and_b_forward_shapes():
    from mast.vision._vigil.heads.head_b_v2 import HeadBv2
    from mast.vision._vigil.heads.head_q_sharpness import HeadQSharpness

    cls_with_scale = torch.randn(2, 1024 + 64)
    q = HeadQSharpness(in_dim=1024 + 64).eval()(cls_with_scale)
    assert q.shape == (2,)
    b = HeadBv2(in_dim=1024 + 64).eval()(cls_with_scale)
    assert b.morph_coarse_logits.shape == (2, 3)   # M0 / M1 / M2M3
    assert b.morph_logits.shape == (2, 4)          # M0..M3
    probs = b.as_probs()
    assert probs["morph_coarse_probs"].shape == (2, 3)
    assert torch.allclose(probs["morph_coarse_probs"].sum(-1), torch.ones(2), atol=1e-5)


def test_head_c_l1_segmentation_shape():
    from mast.vision._vigil.heads.head_c_l1_dino import HeadCLevel1DINO

    head = HeadCLevel1DINO(embed_dim=1024, num_classes=4).eval()
    tokens = torch.randn(1, 16 * 16, 1024)         # 256 patch tokens
    logits = head(tokens, hw=(16, 16), target_hw=(256, 256))
    assert logits.shape == (1, 4, 256, 256)
    assert logits.argmax(1).max().item() <= 3


def test_stm_processor_normalize_per_image_range():
    from mast.vision._vigil.data.stm_image_processor import STMImageProcessor

    proc = STMImageProcessor(target_size=64)
    img = torch.randn(1, 32, 32) * 50.0  # pm
    out = proc.normalize_per_image(img)
    assert out.shape == (1, 32, 32)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0  # tanh-squashed [0,1]


def test_to_dino_input_resizes_nonsquare_to_square_target():
    """Regression: to_dino_input must resize a NON-square image whose width
    already equals target_size. Checking only shape[-1] (width) skipped the
    resize → the backbone got a (target, H≠target) tensor and crashed."""
    from mast.vision._vigil.data.stm_image_processor import STMImageProcessor

    proc = STMImageProcessor(target_size=256)
    # Width already 256 but height 128 — the buggy width-only check skipped resize.
    img = torch.rand(1, 3, 128, 256)
    out = proc.to_dino_input(img)
    assert out.shape == (1, 3, 256, 256)
    # And the symmetric case (height matches, width differs).
    img2 = torch.rand(1, 3, 256, 200)
    assert proc.to_dino_input(img2).shape == (1, 3, 256, 256)
    # Already-square-at-target is a no-op resize (still correct shape).
    img3 = torch.rand(2, 3, 256, 256)
    assert proc.to_dino_input(img3).shape == (2, 3, 256, 256)


def test_tile_starts_covers_full_length_with_overlap():
    from mast.vision._vigil.m12 import _tile_starts
    # small ≤ tile → single window at 0
    assert _tile_starts(200, 256, 224) == [0]
    # 640 with tile 256 stride 224 → 0,224 then clamp last to 384 (=640-256)
    starts = _tile_starts(640, 256, 224)
    assert starts[0] == 0 and starts[-1] == 640 - 256
    # every pixel is covered by at least one [s, s+tile) window
    covered = set()
    for s in starts:
        covered.update(range(s, s + 256))
    assert covered.issuperset(range(640))


def test_amp_ctx_is_noop_on_cpu():
    import torch
    from contextlib import nullcontext
    from mast.vision._vigil.m12 import _amp_ctx
    ctx = _amp_ctx("cpu", torch)
    assert isinstance(ctx, type(nullcontext()))


def test_get_head_c_loss_fn_is_training_stripped():
    """The training-only dispatcher must raise (it needs vigil.training.losses,
    out of the inference vendoring closure) rather than ImportError."""
    from mast.vision._vigil.heads.head_c_l1_common import get_head_c_loss_fn

    with pytest.raises(NotImplementedError):
        get_head_c_loss_fn("focal_ce")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
