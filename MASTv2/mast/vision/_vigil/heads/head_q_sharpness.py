"""Head Q: continuous tip sharpness regression.

取自 Handbook v2.2 §9 (PR-9: Head Q 锐度回归)。

Head Q 学连续 sharpness 量 ``log10(R_tip_nm / scan_size_nm)``,范围约 [-3, 0],
是 v2 plan 的主任务头之一(替代 v1 的 Head A)。Loss 三项:
  - Smooth L1 (Huber): 主回归信号,对 outlier robust
  - Pairwise ranking:  鲁棒于 sim-to-real 的绝对值漂移
  - Quantile (pinball at 0.5): 鲁棒于尾部 label noise

本模块是独立 nn.Module,输入是 ``cls_with_scale`` 张量 (B, 1088),不 import backbone。

偏离 Handbook v2.2:
  - §9.2 用 ``from scipy.stats import spearmanr, kendalltau`` 写在 ``head_q_metrics``
    函数体内 —— 保持函数内延迟 import(scipy 是相对重的依赖),与施工规则 C 一致。
    文档原文已是函数内 import,此处仅明确保留该形式,未改动。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ScaleEmbedding(nn.Module):
    """Encode log10(scan_size_nm) as a 64-d sinusoidal-style embedding.

    Shared between Head Q and (future) Head A.
    """

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.proj = nn.Linear(1, dim)

    def forward(self, scan_size_nm: Tensor) -> Tensor:
        # scan_size_nm: (B,)
        x = torch.log10(scan_size_nm.clamp(min=0.5)).unsqueeze(-1)  # (B, 1)
        h = self.proj(x)
        return torch.cat(
            [
                torch.sin(h[..., : self.dim // 2]),
                torch.cos(h[..., self.dim // 2 :]),
            ],
            dim=-1,
        )  # (B, dim)


class HeadQSharpness(nn.Module):
    """Predict log10(R_tip / scan_size) from [CLS] ‖ scale_emb.

    Output range: [-3, 0]
      log10(R/L) = -3  ->  R/L = 0.001 (extremely sharp)
      log10(R/L) =  0  ->  R/L = 1.0   (extremely blunt)
    """

    def __init__(
        self,
        in_dim: int = 1088,
        hidden: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = [256, 64]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cls_with_scale: Tensor) -> Tensor:
        """cls_with_scale: (B, in_dim) = [CLS] (1024) ‖ scale_emb (64).

        Returns: (B,) predicted log10(R/L).
        """
        return self.mlp(cls_with_scale).squeeze(-1)


def head_q_loss(
    pred: Tensor,
    target: Tensor,
    lambda_huber: float = 1.0,
    lambda_rank: float = 0.2,
    lambda_quantile: float = 0.1,
    huber_beta: float = 0.1,
    rank_margin: float = 0.1,
    sample_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute three-component Head Q loss.

    Args:
      pred: (B,) predicted log10(R/L)
      target: (B,) ground-truth log10(R/L)
      lambda_*: relative weights
      huber_beta: Huber loss transition point
      rank_margin: margin for pairwise ranking loss
      sample_weight: optional (B,) per-sample weights (v2.4 Q1 soft-gating: down-weight
        unstable tips whose log(R/scan) label is ill-posed). None → all 1.0 (identical to
        the unweighted loss). Weights point-wise huber/quantile and, for the pairwise rank
        term, each pair by min(w_i, w_j).

    Returns:
      Dict with 'total', 'huber', 'rank', 'quantile' losses
    """
    if sample_weight is None:
        sample_weight = torch.ones_like(pred)
    w = sample_weight
    wsum = w.sum().clamp(min=1e-6)

    # 1. Smooth L1 (Huber) — per-sample, weighted
    L_huber = (F.smooth_l1_loss(pred, target, beta=huber_beta, reduction="none") * w).sum() / wsum

    # 2. Pairwise ranking
    # For each pair (i, j), if target_i < target_j, enforce pred_i < pred_j - margin
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)  # (B, B)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    sign = torch.sign(diff_target)
    # When sign = +1, want diff_pred > margin; loss = relu(margin - diff_pred)
    # When sign = -1, want diff_pred < -margin; loss = relu(margin + diff_pred)
    # Combined: loss = relu(margin - sign * diff_pred)
    L_rank_mat = F.relu(rank_margin - sign * diff_pred)
    # Mask self-pairs (sign == 0); weight each pair by min(w_i, w_j)
    pair_w = torch.min(w.unsqueeze(1), w.unsqueeze(0))
    valid_mask = (sign != 0).float() * pair_w
    L_rank = (L_rank_mat * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    # 3. Quantile (pinball at 0.5) — per-sample, weighted
    residual = target - pred
    q_ps = 0.5 * residual.clamp(min=0) + 0.5 * (-residual).clamp(min=0)
    L_quantile = (q_ps * w).sum() / wsum

    total = (
        lambda_huber * L_huber
        + lambda_rank * L_rank
        + lambda_quantile * L_quantile
    )

    return {
        "total": total,
        "huber": L_huber.detach(),
        "rank": L_rank.detach(),
        "quantile": L_quantile.detach(),
    }


def head_q_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Evaluation metrics for Head Q on a batch."""
    # 延迟 import scipy(重依赖,施工规则 C)。
    from scipy.stats import kendalltau, spearmanr

    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()

    mae = float(abs(p - t).mean())
    rmse = float(((p - t) ** 2).mean() ** 0.5)

    # Spearman ρ / Kendall τ
    rho, _ = spearmanr(p, t)
    tau, _ = kendalltau(p, t)

    return {
        "q_mae": mae,
        "q_rmse": rmse,
        "q_spearman": float(rho) if rho is not None else 0.0,
        "q_kendall": float(tau) if tau is not None else 0.0,
    }
