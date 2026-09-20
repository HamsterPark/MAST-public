"""VIGIL v2.5 forward glue — batch-input build + backbone run (S0/shared path).

VENDORED into MAST (Phase 9 / VIGIL v2.5). Two standalone functions extracted
from the VIGIL training/eval scripts, with imports rewritten to be self-contained
inside ``mast.vision._vigil``:

  * ``build_batch_input`` — extracted from
    ``D:/…/VIGIL/scripts/eval_l0l3.py::_build_batch_input``
    (double-channel normalize → compose → DINO input). E3 = fwd/bwd differential.

  * ``run_backbone`` — extracted from
    ``D:/…/VIGIL/scripts/train_stage_b_v3_orthoheads.py::_run_backbone``,
    **simplified to ONLY the ``bb_share=="S0"``, ``per_head_lora=False``, no-T_a
    path** — that is all the ssl_sf09c1 config needs (S0 / shared / last-layer).
    The S1 dual-backbone, per-head-type adapter-switching, and T_a tiling
    branches are intentionally omitted (out of MAST inference closure); a guard
    raises ``NotImplementedError`` if those configs are requested, rather than
    silently producing wrong features.

The legacy underscore names (``_build_batch_input`` / ``_run_backbone``) are kept
as aliases so callers porting the upstream inference loop need no rename.

BUGFIX(v2.4) carried from upstream: ScaleEmbedding internally applies
``log10(scan)``; pass the **raw** scan size (nm), do not pre-log it.
"""
from __future__ import annotations

import torch

from .data.multichannel_processor import compose_channels, plane_flatten


def build_batch_input(batch, proc, device, mode, rot_k=0):
    """双通道 normalize → compose → DINO input(与 driver.build_batch_input 一致)。

    Args:
      batch: dict,含 ``fwd_pm`` / ``bwd_pm`` (B,H,W) float32 picometer。
             单图推理时令 ``bwd_pm == fwd_pm`` 即可走 E3(差分通道恒 0)。
      proc:  STMImageProcessor(target_size / do_resize / 可选 .flatten 属性)。
      device: torch device。
      mode:  输入通道轴模式 E1|E2|E3|E4(ssl_sf09c1 = E3 fwd-bwd 差分)。
      rot_k: rot90 次数(增强用;推理 0)。

    Returns:
      images: (B, 3, H, W) ImageNet-normalized DINO 输入。
    """
    fwd = batch["fwd_pm"].to(device)
    bwd = batch["bwd_pm"].to(device)
    if rot_k:
        fwd = torch.rot90(fwd, k=rot_k, dims=(-2, -1))
        bwd = torch.rot90(bwd, k=rot_k, dims=(-2, -1))
    if getattr(proc, "flatten", False):
        fwd = plane_flatten(fwd)
        bwd = plane_flatten(bwd)
    fwd_n = proc.normalize_per_image(fwd)
    bwd_n = proc.normalize_per_image(bwd)
    images = compose_channels(fwd_n, bwd_n, mode=mode)
    return proc.to_dino_input(images).to(device)


def run_backbone(wrapper, images, tap_layers, scale_emb, scan, per_head_lora,
                 bb_share="S0", spatial_meta=None):
    """backbone forward → (feats, cls_with_scale)。**仅 S0 + shared + last-layer 路径。**

    上游 ``_run_backbone`` 有三路径(S0/shared、S0/per_head_type、S1 双主干)× 可选 T-a
    tiling;MAST 推理只需 ssl_sf09c1 的 **S0 + shared + 整图** 路径,故本函数只实现:

        feats = wrapper(images)                         # 1 个 backbone, 1 次 forward
        cls_with_scale = cat([feats["cls"], scale_emb(scan)], dim=-1)
        return feats, cls_with_scale

    其余配置(``bb_share=='S1'`` / ``per_head_lora`` / ``spatial_meta['mode']=='T_a'``)
    不在本 closure,直接抛 ``NotImplementedError`` 而非静默出错。

    NOTE: MAST 的 ``DINOBackboneWrapper.forward`` 不接 ``tap_layers`` 形参(只服务
    last-layer)。本函数的 ``tap_layers`` 形参仅保留上游签名;**必须为 None**(layer_tap
    == 'last' → resolve_tap_layers 返回 None),否则抛错——因为分层接出(l6/l9/last4)需要
    上游带 ``forward_intermediates`` 的 wrapper,不在本 closure。

    BUGFIX(v2.4): ScaleEmbedding 内部已 log10(scan);传原始 scan,不外部再 log10。
    """
    if bb_share != "S0":
        raise NotImplementedError(
            f"run_backbone vendored subset supports bb_share='S0' only, got {bb_share!r} "
            "(S1 dual-backbone is out of the MAST inference closure)."
        )
    if per_head_lora:
        raise NotImplementedError(
            "run_backbone vendored subset supports shared LoRA only (per_head_lora=False); "
            "per_head_type adapter switching is out of the MAST inference closure."
        )
    if spatial_meta is not None and spatial_meta.get("mode") == "T_a":
        raise NotImplementedError(
            "run_backbone vendored subset does not support T_a tiling "
            "(spatial_meta mode 'T_a' is out of the MAST inference closure)."
        )
    if tap_layers is not None:
        raise NotImplementedError(
            f"run_backbone vendored subset supports last-layer only (tap_layers=None), "
            f"got {tap_layers!r}. Layer-tap (l6/l9/last4) needs a wrapper with "
            "forward_intermediates, which is out of the MAST inference closure."
        )
    feats = wrapper(images)
    cls_with_scale = torch.cat([feats["cls"], scale_emb(scan)], dim=-1)
    return feats, cls_with_scale


# Aliases matching the upstream private names (callers porting the inference loop).
_build_batch_input = build_batch_input
_run_backbone = run_backbone
