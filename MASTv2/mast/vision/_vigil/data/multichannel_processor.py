"""多通道输入处理 —— VIGIL DINO v2.2 PR-3.

支持 4 种输入配置 E1-E4、1ch→3ch patch_embed 权重扩展、真实图像适配、
通道 2/3 渐进激活 curriculum。实现取自 Handbook v2.2 §3.2,逐字落地。

| 模式 | ch1 | ch2 | ch3 |
|------|-----|-----|-----|
| E1   | fwd | fwd | fwd |
| E2   | fwd | bwd | mean(fwd,bwd) |
| E3   | fwd | bwd | fwd - bwd |
| E4   | fwd | bwd | |f-b| / (|f|+|b|) |

VIGIL shard 存 2 物理通道(trace/retrace),diff 由本模块现算 —— 与 v2.2 块 B
"images_raw (N,2,H,W),diff loader 现算" 一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

InputMode = Literal["E1", "E2", "E3", "E4"]


def plane_flatten(img: Tensor) -> Tensor:
    """v2.4 (Q2): per-image least-squares plane subtraction (removes spatial tilt).

    img: (B, H, W) single channel. Fits z = a·x + b·y + c per image (shared grid →
    one normal-equation solve), subtracts the plane. MUST be applied **per channel
    separately** (caller passes fwd / bwd independently) — never on a merged channel
    (see 既有教训). per_image normalize alone cannot remove
    a spatial slope; this does. v2.4 A/B: default OFF (tilt is small, may be null).
    """
    B, H, W = img.shape
    dev, dt = img.device, img.dtype
    ys = torch.linspace(-1.0, 1.0, H, device=dev, dtype=dt)
    xs = torch.linspace(-1.0, 1.0, W, device=dev, dtype=dt)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    A = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.ones(H * W, device=dev, dtype=dt)], dim=1)  # (HW,3)
    z = img.reshape(B, H * W, 1)
    AtA = A.t() @ A                                   # (3,3) shared across batch
    Atz = torch.einsum("nk,bnc->bkc", A, z)           # (B,3,1)
    coeffs = torch.linalg.solve(AtA, Atz)             # (B,3,1)
    plane = torch.einsum("nk,bkc->bnc", A, coeffs).reshape(B, H, W)
    return img - plane


def compose_channels(
    fwd: Tensor,
    bwd: Tensor | None,
    mode: InputMode,
    eps: float = 1e-3,
) -> Tensor:
    """Compose 3-channel input from fwd/bwd according to mode.

    Args:
      fwd: (B, H, W) or (H, W), float32 (normalized pm)
      bwd: (B, H, W) or (H, W); can be None if mode == E1
      mode: E1 | E2 | E3 | E4
      eps: numerical stability for E4 normalization

    Returns:
      (B, 3, H, W) tensor.
    """
    if fwd.dim() == 2:
        fwd = fwd.unsqueeze(0)  # (1, H, W)

    if mode == "E1":
        return fwd.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()

    if bwd is None:
        raise ValueError(f"Mode {mode} requires bwd channel")
    if bwd.dim() == 2:
        bwd = bwd.unsqueeze(0)

    if mode == "E2":
        ch3 = (fwd + bwd) / 2.0
    elif mode == "E3":
        ch3 = fwd - bwd
    elif mode == "E4":
        ch3 = (fwd - bwd).abs() / (fwd.abs() + bwd.abs() + eps)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.stack([fwd, bwd, ch3], dim=1)  # (B, 3, H, W)


def expand_patch_embed_weight(
    w_1ch: Tensor,
    mode: InputMode,
    init_std: float = 0.01,
) -> Tensor:
    """Expand a single-channel patch_embed weight to 3 channels.

    Stage A trains with in_channels=1 (matching real PNG); Stage B needs
    in_channels=3 (fwd, bwd, diff/derived).

    Args:
      w_1ch: (embed_dim, 1, kernel_h, kernel_w) Stage A weight
      mode: target Stage B input mode
      init_std: std for random init of channels 2/3 (E3/E4 only)

    Returns:
      (embed_dim, 3, kernel_h, kernel_w) Stage B weight
    """
    if w_1ch.dim() != 4 or w_1ch.shape[1] != 1:
        raise ValueError(f"Expected weight shape (D, 1, K, K), got {tuple(w_1ch.shape)}")

    if mode == "E1":
        # All three channels identical, divided by 3 to preserve scale
        return w_1ch.repeat(1, 3, 1, 1) / 3.0

    if mode == "E2":
        # Channel 0 = fwd weight (Stage A); channels 1, 2 zero-init
        zeros = torch.zeros_like(w_1ch.repeat(1, 2, 1, 1))
        return torch.cat([w_1ch, zeros], dim=1)

    if mode in ("E3", "E4"):
        # Channel 0 = fwd; channels 1, 2 small random init
        extra = torch.zeros_like(w_1ch.repeat(1, 2, 1, 1))
        nn.init.normal_(extra, mean=0.0, std=init_std)
        return torch.cat([w_1ch, extra], dim=1)

    raise ValueError(f"Unknown mode: {mode}")


def adapt_real_image(real_image: Tensor, mode: InputMode) -> Tensor:
    """Adapt a single-channel real STM image to 3-channel input.

    Real images are forward-only. Stage C inference runs on a model trained
    with (fwd, bwd, diff) inputs.

    Strategy:
      E1: replicate; E2/E3/E4: ch1=fwd, ch2=fwd (bwd proxy), ch3=zero.
    """
    if real_image.dim() == 2:
        real_image = real_image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif real_image.dim() == 3:
        if real_image.shape[0] == 1:
            real_image = real_image.unsqueeze(0)  # (1, 1, H, W)
        else:
            real_image = real_image.unsqueeze(1)  # (B, 1, H, W)

    fwd = real_image  # (B, 1, H, W)

    if mode == "E1":
        return fwd.expand(-1, 3, -1, -1).contiguous()

    bwd_proxy = fwd.clone()
    ch3 = fwd.clone() if mode == "E2" else torch.zeros_like(fwd)
    return torch.cat([fwd, bwd_proxy, ch3], dim=1)


@dataclass
class PatchEmbedCurriculum:
    """Curriculum warmup: progressively activate channels 2, 3 over N steps.

    Channel 0 (fwd) always at full weight. Channels 1, 2 ramp 0->1 linearly,
    protecting Stage A patch_embed weights during early Stage B training.
    """

    total_steps: int = 5000
    step: int = 0

    def update(self) -> None:
        """Call once per training step."""
        self.step += 1

    @property
    def progress(self) -> float:
        return min(1.0, self.step / max(1, self.total_steps))

    def channel_weights(self, n_channels: int = 3) -> Tensor:
        """Per-channel multiplier; shape (n_channels,)."""
        p = self.progress
        w = torch.ones(n_channels)
        if n_channels > 1:
            w[1:] = p
        return w

    def apply(self, image: Tensor) -> Tensor:
        """Apply per-channel scaling to (B, C, H, W) image."""
        w = self.channel_weights(image.shape[1]).to(image.device)
        return image * w.view(1, -1, 1, 1)


class DiffZeroDrop:
    """Stage B augmentation: randomly zero out the diff channel.

    Simulates real-image deployment (diff unavailable). Forces the model to
    learn a fallback path that ignores ch3.
    """

    def __init__(self, p: float = 0.30) -> None:
        self.p = p

    def __call__(self, image: Tensor, mode: InputMode) -> Tensor:
        """image: (B, 3, H, W), mode: E3 or E4."""
        if mode not in ("E3", "E4"):
            return image  # E1, E2: diff doesn't carry distinct info
        if torch.rand(1).item() > self.p:
            return image
        out = image.clone()
        out[:, 2] = 0.0
        return out


class TraceRetraceSwap:
    """Stage B augmentation: swap fwd/bwd channels with probability p.

    Enforces directional symmetry. For E3, also negate ch3
    (fwd - bwd -> bwd - fwd) to stay consistent.
    """

    def __init__(self, p: float = 0.20) -> None:
        self.p = p

    def __call__(self, image: Tensor, mode: InputMode) -> Tensor:
        """image: (B, 3, H, W)."""
        if mode == "E1":
            return image
        if torch.rand(1).item() > self.p:
            return image
        out = image.clone()
        out[:, 0], out[:, 1] = image[:, 1].clone(), image[:, 0].clone()
        if mode == "E3":
            out[:, 2] = -image[:, 2]
        # E4: ch3 = |f-b|/(|f|+|b|) is swap-symmetric, no negation needed
        return out
