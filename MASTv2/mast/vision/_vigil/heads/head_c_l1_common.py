"""Shared utilities for Head C Level 1 (both segmentation routes).

取自 Handbook v2.2 §11.2 (PR-11),逐字落地。Head C L1 学 4 类 per-pixel 分割
(terrace=0, step=1, defect=2, contamination=3)。T0 做双路线消融:U-Net 路线
(`head_c_l1_unet.py`) 与 DINO patch-token 路线 (`head_c_l1_dino.py`),两者共享
本文件的 `HeadCLevel1Base` 抽象接口与 `focal_cross_entropy` 损失。

偏离 (相对 v2.2 §11.2 正文):
  D-l1c-1. 修正规则 A:v2.2 把三个文件 (`head_c_l1_common.py` /
           `head_c_l1_unet.py` / `head_c_l1_dino.py`) 写在同一 ```python
           代码块,用 `# src/vigil/...` 注释分隔。本文件只保留 common 部分,
           U-Net / DINO 路线拆到各自独立文件。
  D-l1c-2. ignore_index 哨兵值 99 与 `mask_codec._L1_MAPPING` 的保留 sentinel
           一致 (mask_codec.remap_to_level1 把保留 surface code 15 映射为 99)。
           本文件未改该常量,仅在此说明跨模块约定来源。
  D-l1c-3. v2.2 §11.2 顶层 `import torch`。按修正规则 C,torch 允许顶层 import。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NUM_L1_CLASSES = 4  # terrace, step, defect, contamination

# Sentinel for masked-out / reserved pixels; matches mask_codec.remap_to_level1.
L1_IGNORE_INDEX = 99


class HeadCLevel1Base(nn.Module):
    """Shared interface for both L1 routes."""

    def predict(self, features: dict | Tensor) -> Tensor:
        """Returns (B, NUM_L1_CLASSES, H, W) logits."""
        raise NotImplementedError

    def loss(
        self,
        logits: Tensor,
        target: Tensor,
        class_weights: Tensor | None = None,
        ignore_index: int = L1_IGNORE_INDEX,
    ) -> Tensor:
        """Compute focal cross-entropy with class balancing."""
        return focal_cross_entropy(
            logits,
            target,
            class_weights=class_weights,
            gamma=2.0,
            ignore_index=ignore_index,
        )


def focal_cross_entropy(
    logits: Tensor,
    target: Tensor,
    class_weights: Tensor | None = None,
    gamma: float = 2.0,
    ignore_index: int = L1_IGNORE_INDEX,
) -> Tensor:
    """Focal cross-entropy loss for per-pixel segmentation.

    Args:
      logits: (B, C, H, W)
      target: (B, H, W) long, with ignore_index = 99 for masked-out regions
      class_weights: (C,) optional per-class weight
      gamma: focal exponent
    """
    log_probs = F.log_softmax(logits, dim=1)  # (B, C, H, W)
    probs = log_probs.exp()

    # Gather log-prob and prob for true class
    valid = (target != ignore_index)
    target_safe = target.clone()
    target_safe[~valid] = 0

    log_p_t = log_probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)  # (B, H, W)
    p_t = probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)

    focal_term = (1 - p_t) ** gamma
    loss_per_pixel = -focal_term * log_p_t

    if class_weights is not None:
        w_t = class_weights.to(target.device)[target_safe]
        loss_per_pixel = loss_per_pixel * w_t

    # Apply valid mask + reduce
    loss_per_pixel = loss_per_pixel * valid.float()
    return loss_per_pixel.sum() / valid.sum().clamp(min=1.0)


def compute_class_weights(
    target_batch: Tensor,
    num_classes: int = NUM_L1_CLASSES,
    beta: float = 0.999,
    method: str = "cui_2019_effective_number",
) -> Tensor:
    """v2.3 (块 D): per-class weights with 4 strategies.

    Args:
        target_batch: (B, H, W) long
        num_classes: 4 for Head C-L1
        beta: only used for cui_2019_effective_number
        method: 'inverse_freq' | 'cui_2019_effective_number' (default) |
                'sqrt_inverse' | 'uniform'

    Returns:
        (num_classes,) tensor normalized so weights.sum() == num_classes.
    """
    counts = torch.bincount(
        target_batch.flatten()[target_batch.flatten() != L1_IGNORE_INDEX].long(),
        minlength=num_classes,
    ).float().clamp(min=1.0)

    if method == "inverse_freq":
        weights = 1.0 / counts
    elif method == "cui_2019_effective_number":
        effective_num = 1.0 - beta ** counts
        weights = (1.0 - beta) / (effective_num + 1e-8)
    elif method == "sqrt_inverse":
        weights = 1.0 / torch.sqrt(counts)
    elif method == "uniform":
        weights = torch.ones(num_classes, device=counts.device)
    else:
        raise ValueError(
            f"Unknown class_weights method: {method!r}. Valid: "
            "inverse_freq, cui_2019_effective_number, sqrt_inverse, uniform"
        )

    return weights / weights.sum() * num_classes


# ─── v2.3 块 D: Conditional supervision + loss dispatcher ──────────────────


SUPERVISION_CRITERIA = (
    "all",
    "M0_only",
    "M0_AND_single",
    "M0_AND_single_AND_stable",        # v2.3 default
    "M0M1_AND_single_AND_stable",
    "M0M1M2_AND_single_AND_stable",
)


def head_c_supervision_mask(batch: dict, criteria: str) -> Tensor:
    """v2.3 (块 D): boolean mask selecting samples eligible for Head C-L1 loss.

    BAD/multi/unstable tip mask carries noisy geometric info that contaminates
    backbone gradient — only supervise on subsets where mask is physically meaningful.
    """
    alpha1 = batch["alpha1_type"]
    n_tips = batch["n_tips"]
    if "is_unstable_label" in batch:
        stable = ~batch["is_unstable_label"].bool()
    else:
        stable = torch.ones_like(alpha1, dtype=torch.bool)

    if criteria == "all":
        return torch.ones_like(stable, dtype=torch.bool)
    if criteria == "M0_only":
        return alpha1 == 0
    if criteria == "M0_AND_single":
        return (alpha1 == 0) & (n_tips == 1)
    if criteria == "M0_AND_single_AND_stable":
        return (alpha1 == 0) & (n_tips == 1) & stable
    if criteria == "M0M1_AND_single_AND_stable":
        return (alpha1 <= 1) & (n_tips == 1) & stable
    if criteria == "M0M1M2_AND_single_AND_stable":
        return (alpha1 <= 2) & (n_tips == 1) & stable
    raise ValueError(
        f"Unknown supervision criteria: {criteria!r}. Valid: {SUPERVISION_CRITERIA}"
    )


def get_head_c_loss_fn(loss_type: str, focal_gamma: float = 2.0):
    """v2.3 (块 D): Head C-L1 loss dispatcher — TRAINING ONLY.

    The original VIGIL implementation pulled boundary/tversky/combo losses
    from ``vigil.training.losses``, which is outside the inference vendoring
    closure (MAST only ships the forward path). This stub keeps the symbol
    present so the public surface matches upstream, but raises if invoked —
    MAST never trains these heads (weights come from the VIGIL project).
    """
    raise NotImplementedError(
        "get_head_c_loss_fn is training-only and was stripped from the MAST "
        "inference vendoring (it needs vigil.training.losses). MAST does not "
        "train Head C-L1 — load pre-trained weights from the VIGIL project."
    )
