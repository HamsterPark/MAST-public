"""DINOv2/v3 骨干统一加载器 —— VIGIL DINO v2.2 PR-6.

支持 7 个 backbone(dinov2 base/large/reg-large、dinov3 S/B/L、nv-dinov2-L);
默认冻结骨干(LoRA-only 调);patch_embed 的 in_channels 可配(Stage A=1/B=3)。

实现取自 Handbook v2.2 §6.2;偏离(见 WORKLOG):
  D2.  logger 用标准库 logging。
  D15. ``transformers`` 延迟到函数内 import —— v2.2 在模块顶层
       ``from transformers import AutoModel``,若 transformers 未装会导致
       ``import vigil.backbone.dinov_loader`` 直接失败,连不需要下载的
       registry 查询/测试都跑不了。改为在 load_backbone 内 import。
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from .wrapper import DINOBackboneWrapper

logger = logging.getLogger(__name__)


# 7 个已知 backbone 的元数据注册表
BACKBONE_REGISTRY: dict[str, dict] = {
    "dinov2-base": {
        "family": "dinov2", "hf_id": "facebook/dinov2-base",
        "patch_size": 14, "embed_dim": 768, "num_layers": 12,
        "native_resolution": 224,
    },
    "dinov2-large": {
        "family": "dinov2", "hf_id": "facebook/dinov2-large",
        "patch_size": 14, "embed_dim": 1024, "num_layers": 24,
        "native_resolution": 224,
    },
    "dinov2-with-registers-large": {
        "family": "dinov2", "hf_id": "facebook/dinov2-with-registers-large",
        "patch_size": 14, "embed_dim": 1024, "num_layers": 24,
        "native_resolution": 224, "num_registers": 4,
    },
    "dinov3-vits16": {
        "family": "dinov3", "hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "timm_id": "vit_small_patch16_dinov3.lvd1689m",
        "patch_size": 16, "embed_dim": 384, "num_layers": 12,
        "native_resolution": 256,
    },
    "dinov3-vitb16": {
        "family": "dinov3", "hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "timm_id": "vit_base_patch16_dinov3.lvd1689m",
        "patch_size": 16, "embed_dim": 768, "num_layers": 12,
        "native_resolution": 256,
    },
    "dinov3-vitl16": {
        "family": "dinov3", "hf_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "timm_id": "vit_large_patch16_dinov3.lvd1689m",
        "patch_size": 16, "embed_dim": 1024, "num_layers": 24,
        "native_resolution": 256,
    },
    "nv-dinov2-vitl14": {
        "family": "nv-dinov2", "hf_id": "nvcr.io/nvidia/tao/nv-dinov2-vitl14",
        "patch_size": 14, "embed_dim": 1024, "num_layers": 24,
        "native_resolution": 224,
    },
}


def get_backbone_info(name: str) -> dict:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name]


def load_backbone(
    name: str,
    in_channels: int = 3,
    freeze: bool = True,
    cache_dir: str | Path | None = None,
) -> DINOBackboneWrapper:
    """加载任一注册 backbone,可选修改 in_channels。"""
    info = get_backbone_info(name)
    used_timm = False
    if info["family"] == "dinov3":
        # MAST deviation (Phase 9): M12 was trained on the *timm* backbone.
        # The original VIGIL loader tried the HF repo (facebook/dinov3-*,
        # needs trust_remote_code + Hub access) first and fell back to timm
        # on failure. In MAST the HF repo is never available (we bundle only
        # the timm weights cache) so the HF attempt always failed → timm.
        # We therefore load timm DIRECTLY: byte-identical weights to the
        # verified fallback path, minus the doomed HF call AND the heavy
        # ``transformers`` hard-dependency in the frozen build. The bundled
        # weights are handed over as an explicit FILE path (below) — the old
        # "HF_HOME + HF_HUB_OFFLINE gets timm to the bundle" claim was wrong
        # and cost two releases (see backbone_cache module docstring).
        import timm

        from mast.vision._vigil.backbone_cache import timm_pretrained_kwargs

        timm_id = info.get("timm_id")
        if not timm_id:
            raise ValueError(f"dinov3 backbone {name!r} has no timm_id in registry")
        # Hand timm the bundled weight FILE directly. Going through
        # huggingface_hub instead is what broke the frozen build: its cache
        # path is a module-level constant frozen at import time, so the
        # HF_HOME we set at load time could not retarget it and timm looked in
        # an empty user cache -> LocalEntryNotFoundError -> MockBackend
        # (; see backbone_cache module docstring).
        backbone = timm.create_model(
            timm_id, pretrained=True, num_classes=0, in_chans=in_channels,
            **timm_pretrained_kwargs(timm_id, cache_dir),
        )
        used_timm = True
    elif info["family"] == "nv-dinov2":
        backbone = _load_nv_dinov2(info["hf_id"], cache_dir)
    else:  # dinov2 family — HF only (M12 does not use this path)
        from transformers import AutoModel  # D15: deferred import

        backbone = AutoModel.from_pretrained(
            info["hf_id"], cache_dir=cache_dir, trust_remote_code=False,
        )

    # timm 路径在 create_model 时已通过 in_chans 处理通道,无需 _modify
    if in_channels != 3 and not used_timm:
        _modify_patch_embed_channels(backbone, info, in_channels)

    if freeze:
        for p in backbone.parameters():
            p.requires_grad_(False)
        logger.info(
            "Backbone %s loaded and frozen (%d params)",
            name, sum(p.numel() for p in backbone.parameters()),
        )
    return DINOBackboneWrapper(backbone, info)


def _load_nv_dinov2(hf_id: str, cache_dir: str | Path | None) -> nn.Module:
    """NV-DINOv2(TAO toolkit 格式)加载 —— v2.2 仍是 STUB。

    生产前需:NGC CLI 认证 -> 下载 .tao -> 转 plain state_dict。
    当前回退到 dinov2-large 并告警。
    """
    from transformers import AutoModel  # D15: deferred import

    logger.warning(
        "NV-DINOv2 loader is a STUB; falling back to facebook/dinov2-large. "
        "TODO: NGC download + state_dict conversion (HPC stage)."
    )
    return AutoModel.from_pretrained("facebook/dinov2-large", cache_dir=cache_dir)


def _find_patch_embed(backbone: nn.Module) -> nn.Module | None:
    """按已知路径定位 patch_embedding 子模块。"""
    candidates = [("embeddings", "patch_embeddings"), ("patch_embed",)]
    for path in candidates:
        m = backbone
        try:
            for attr in path:
                m = getattr(m, attr)
            return m
        except AttributeError:
            continue
    return None


def _modify_patch_embed_channels(
    backbone: nn.Module, info: dict, in_channels: int
) -> None:
    """就地修改 patch_embed.proj 以适配非 3 通道输入。"""
    patch_embed = _find_patch_embed(backbone)
    if patch_embed is None:
        raise RuntimeError("Could not locate patch_embed in backbone")

    old_conv = (
        patch_embed.projection
        if hasattr(patch_embed, "projection")
        else patch_embed.proj
    )
    old_weight = old_conv.weight  # (D, 3, K, K)
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
        elif in_channels < 3:
            new_conv.weight.copy_(old_weight[:, :in_channels])
        else:
            new_conv.weight[:, :3].copy_(old_weight)
            nn.init.normal_(new_conv.weight[:, 3:], std=0.01)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    if hasattr(patch_embed, "projection"):
        patch_embed.projection = new_conv
    else:
        patch_embed.proj = new_conv
