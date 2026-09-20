"""DINO 骨干统一接口 wrapper —— VIGIL DINO v2.2 PR-6.

把 DINOv2 / DINOv3 / NV-DINOv2 的 forward 统一成 ``{cls, tokens, hw}``。
实现取自 Handbook v2.2 §6.2。
"""
from __future__ import annotations

from torch import Tensor, nn


class DINOBackboneWrapper(nn.Module):
    """跨 DINOv2 / DINOv3 / NV-DINOv2 的统一接口。

    forward(images) 返回:
      cls:    (B, embed_dim)               CLS token
      tokens: (B, N_patches, embed_dim)    patch tokens(去除 CLS / registers)
      hw:     (h_p, w_p)                   patch 网格尺寸
    """

    def __init__(self, backbone: nn.Module, info: dict) -> None:
        super().__init__()
        self.backbone = backbone
        self.info = info
        self.embed_dim = info["embed_dim"]
        self.patch_size = info["patch_size"]
        self.family = info["family"]
        self.num_registers = info.get("num_registers", 0)

    def forward(self, images: Tensor) -> dict:
        """images: (B, C, H, W);H、W 必须是 patch_size 的整数倍。"""
        _b, _c, h, w = images.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"H={h}, W={w} must be multiples of patch_size={self.patch_size}"
            )

        # D-PR6-3: HF AutoModel 走 pixel_values,timm 走 forward_features。
        if hasattr(self.backbone, "forward_features") and not hasattr(self.backbone, "config"):
            # timm model (fallback for dinov3 when HF unreachable)
            feats = self.backbone.forward_features(images)  # (B, 1+N_reg+N_patch, D)
            cls = feats[:, 0]
            tokens = feats[:, 1 + self.num_registers :]
        else:
            outputs = self.backbone(pixel_values=images, output_hidden_states=False)
            hidden = outputs.last_hidden_state  # (B, 1 + N_reg + N_patch, embed_dim)
            cls = hidden[:, 0]
            tokens = hidden[:, 1 + self.num_registers :]
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        return {"cls": cls, "tokens": tokens, "hw": (h_p, w_p)}

    def get_patch_grid(self, resolution: int) -> tuple[int, int]:
        p = resolution // self.patch_size
        return p, p
