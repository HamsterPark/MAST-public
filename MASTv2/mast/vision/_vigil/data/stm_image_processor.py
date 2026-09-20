"""STM 图像预处理器 —— VIGIL DINO v2.2 PR-4.

把 HDF5 shard 的 float32 picometer 高度数据归一化为 DINO 期望的输入分布。
绕过 HuggingFace AutoImageProcessor 的 uint8 量化路径,逐图鲁棒归一化
(median + MAD)+ tanh squash。实现取自 Handbook v2.2 §4.2,逐字落地。

合成 float32 raw 走 ``normalize_per_image``;SF09 uint8 走
``normalize_uint8_normalized`` —— 两条路径归一化到同一目标分布,使 SSL
不会把"域"信息从对比度统计里读出来。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class STMImageProcessor:
    """float32 pm -> DINO-compatible input tensor.

    步骤:逐图鲁棒归一化(减 median,除 MAD×1.4826)-> ±5σ clip ->
    tanh squash 到 [0,1] -> ImageNet mean/std -> 可选 resize。

    不做通道组装(那是 PR-3 的职责);输入为单通道。
    """

    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)

    def __init__(self, target_size: int = 512, do_resize: bool = True) -> None:
        self.target_size = target_size
        self.do_resize = do_resize
        self._mean = torch.tensor(self.DINO_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(self.DINO_STD).view(1, 3, 1, 1)

    @torch.no_grad()
    def normalize_per_image(self, image_pm: Tensor) -> Tensor:
        """合成路径:单通道 float32 picometer 图 -> 逐图鲁棒归一化 + tanh squash.

        Args:
          image_pm: (B, H, W) or (H, W), float32, 单位 picometer
        Returns:
          (B, H, W),逐图 ~N(0,1) 后 tanh-squash 到 [0, 1]
        """
        if image_pm.dim() == 2:
            image_pm = image_pm.unsqueeze(0)
        x = image_pm.float()

        med = x.flatten(1).median(dim=1).values.view(-1, 1, 1)
        mad = (x - med).abs().flatten(1).median(dim=1).values.view(-1, 1, 1)
        sigma_robust = mad * 1.4826 + 1e-6
        x = (x - med) / sigma_robust

        x = x.clamp(-5.0, 5.0)
        x = torch.tanh(x * 0.3) * 0.5 + 0.5
        return x  # (B, H, W) in [0, 1]

    @torch.no_grad()
    def normalize_uint8_normalized(self, image_01: Tensor) -> Tensor:
        """SF09 uint8 路径:输入 [0,1](uint8/255)的真实图,做与合成路径相同的
        逐图鲁棒归一化,落到同一目标分布。

        Args:
          image_01: (B, H, W) or (H, W), float32, 值域 [0, 1]
        Returns:
          (B, H, W) in [0, 1],与 normalize_per_image 同一目标分布。
        """
        if image_01.dim() == 2:
            image_01 = image_01.unsqueeze(0)
        x = image_01.float()

        med = x.flatten(1).median(dim=1).values.view(-1, 1, 1)
        mad = (x - med).abs().flatten(1).median(dim=1).values.view(-1, 1, 1)
        sigma_robust = mad * 1.4826 + 1e-6
        x = (x - med) / sigma_robust
        x = x.clamp(-5.0, 5.0)
        x = torch.tanh(x * 0.3) * 0.5 + 0.5
        return x

    @torch.no_grad()
    def to_dino_input(self, image_3ch: Tensor) -> Tensor:
        """3 通道 [0,1] 图 -> ImageNet 归一化 + 可选 resize.

        Args:
          image_3ch: (B, 3, H, W) in [0, 1]
        Returns:
          (B, 3, target_size, target_size) ImageNet-normalized float32
        """
        # Resize whenever EITHER spatial dim differs from target_size. Checking
        # only the width (shape[-1]) let a non-square scan whose width already
        # equals target_size through un-resized — the DINO backbone then sees a
        # non-(target,target) tensor and crashes (patch grid mismatch). Pass an
        # explicit (H, W) target so the output is always square target_size.
        if self.do_resize and (
            image_3ch.shape[-1] != self.target_size
            or image_3ch.shape[-2] != self.target_size
        ):
            image_3ch = F.interpolate(
                image_3ch,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = self._mean.to(image_3ch.device)
        std = self._std.to(image_3ch.device)
        return (image_3ch - mean) / std
