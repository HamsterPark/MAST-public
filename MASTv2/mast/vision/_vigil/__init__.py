"""Vendored VIGIL inference closure for the MAST M12 vision backend (Phase 9).

These modules are a faithful copy of the VIGIL DINO v2.3 inference path
(``D:/…/VIGIL/src/vigil``), trimmed to the forward-only surface MAST needs.
Deviations from upstream are limited to:

  * intra-package imports rewritten to relative (``from .x import y``);
  * ``backbone.dinov_loader.load_backbone`` loads the dinov3 family via timm
    directly (M12 was trained on the timm backbone; the HF repo is never
    bundled) — byte-identical to upstream's HF-fails→timm-fallback path,
    minus the heavy ``transformers`` hard-dependency;
  * ``heads.head_c_l1_common.get_head_c_loss_fn`` reduced to a raising stub
    (training-only; needs ``vigil.training.losses`` which is out of closure).

MAST does NOT train these heads. Weights are produced by the VIGIL project
and shipped as ``artifacts/mast_vision_m12.pt`` (LoRA + 3 heads, ~41 MB) +
the DINOv3-ViT-L/16 timm backbone cache (~1.2 GB, bundled offline).

The public entry point is :mod:`mast.vision._vigil.m12`.
"""
