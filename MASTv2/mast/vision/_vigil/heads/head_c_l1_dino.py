"""Head C Level 1 — DINO patch-token route.

取自 Handbook v2.2 §11.2 (PR-11),逐字落地。DINO 路线:输入 = backbone
patch tokens (B, N, D);1×1 conv 降维 -> bilinear upsample 到目标分辨率 ->
3×3 refine -> class head。T0 默认 DINO 路线(参数少,与骨干共享语义)。

本模块**只接收 patch token 张量**,不 import backbone 代码 —— token 由调用方
(trainer / inference) 从 backbone 取出后传入,Head 与 backbone 解耦。

偏离 (相对 v2.2 §11.2 正文):
  D-l1d-1. 修正规则 A:v2.2 把 U-Net / DINO / common 三个文件写在同一
           ```python 代码块。本文件只保留 `HeadCLevel1DINO`,并补上对
           `head_c_l1_common` 的显式 import (`HeadCLevel1Base` /
           `NUM_L1_CLASSES`)—— 拆文件后必须显式导入。
  D-l1d-2. 修正规则 C:torch 顶层 import,保持原样。
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .head_c_l1_common import NUM_L1_CLASSES, HeadCLevel1Base


class HeadCLevel1DINO(HeadCLevel1Base):
    """DINO-token-based segmentation head.

    Input: backbone patch tokens (B, N, D)
    Output: (B, C, H, W) logits
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = NUM_L1_CLASSES,
        upsample_factor: int = 16,  # patch_size for v3-L/16
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.upsample_factor = upsample_factor

        # 1x1 conv to reduce dim, then upsample, then 3x3 refine, then class head
        self.reduce = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, hidden_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(
        self,
        tokens: Tensor,
        hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> Tensor:
        """tokens: (B, N, D); hw: (h_p, w_p); target_hw: (H, W) target resolution."""
        B, N, D = tokens.shape
        h_p, w_p = hw
        assert N == h_p * w_p, f"N={N} != h_p*w_p={h_p}*{w_p}"

        # Reshape to (B, D, h_p, w_p)
        x = tokens.transpose(1, 2).contiguous().reshape(B, D, h_p, w_p)

        # Reduce + norm
        x = self.norm(self.reduce(x))

        # Upsample to target resolution
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

        # Refine + head
        x = self.refine(x)
        return self.head(x)  # (B, C, H, W)

    def predict(self, features: dict) -> Tensor:
        return self.forward(features["tokens"], features["hw"], features["target_hw"])
