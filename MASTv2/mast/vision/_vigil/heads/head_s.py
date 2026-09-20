"""Head S: apex 轴比回归 + 对称性二元分类。

目标:
  - axis_ratio (float, minor/major ∈ (0, 1]): CLS 向量‖scale 向量估计
    针尖横截面椭圆度 ——ratio=1 表示圆对称,越小越椭圆。
  - is_asymmetric (bool): 二元 logit,判断针尖是否不对称。

输入:
  CLS‖scale 拼接向量 (B, in_dim)。in_dim = embed_dim + 64,
  例如 vits16 → 384+64=448,vitl16 → 1024+64=1088。
  in_dim 通过构造参数传入,绝不硬编码。

标签 (来自 batch dict):
  batch["axis_ratio"]       float tensor (B,), ∈ (0, 1]
  batch["is_asymmetric"]    bool tensor  (B,)
  batch["theta_deg"]        float tensor (B,), 暂未使用,预留接口

监督域:
  single ∧ stable 的样本。具体 mask 由上层 driver 构造后以
  supervision_mask (B,) bool 形式传入 head_s_loss。

损失:
  L = huber_w * SmoothL1(axis_ratio_pred, axis_ratio_gt)
    + bce_w   * BCEWithLogits(asym_logit, is_asymmetric.float())
  默认 huber_w=1.0, bce_w=0.5。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HeadS(nn.Module):
    """Predict apex axis_ratio and is_asymmetric from [CLS] ‖ scale_emb.

    Architecture:
      共享 MLP trunk (in_dim → hidden → hidden//2) → 两条独立输出头:
        - axis_ratio_head: Linear → Sigmoid (约束输出到 (0, 1])
        - asym_head: Linear → raw logit (BCE 损失)

    Args:
      in_dim:  输入维度 = embed_dim + 64。vits16→448, vitl16→1088。
               由调用方传入,绝不硬编码。
      hidden:  MLP 隐层宽度 (默认 256)。
      dropout: Dropout 比例。
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mid = hidden // 2

        # 共享 trunk: in_dim → hidden → hidden//2
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # axis_ratio 输出头: sigmoid 约束到 (0, 1]
        # 注意: sigmoid 输出 ∈ (0,1),当 gt=1.0 时 loss≈0 仍可训练
        self.axis_ratio_head = nn.Linear(mid, 1)

        # is_asymmetric 输出头: raw logit, 由 BCEWithLogitsLoss 处理
        self.asym_head = nn.Linear(mid, 1)

    def forward(self, feats: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
          feats: (B, in_dim)

        Returns:
          dict with:
            "axis_ratio":  (B,) float, sigmoid-constrained ∈ (0, 1)
            "asym_logit":  (B,) float, raw logit for BCEWithLogits
        """
        h = self.trunk(feats)                              # (B, mid)
        axis_ratio = torch.sigmoid(self.axis_ratio_head(h)).squeeze(-1)  # (B,)
        asym_logit = self.asym_head(h).squeeze(-1)        # (B,)
        return {
            "axis_ratio": axis_ratio,
            "asym_logit": asym_logit,
        }


def head_s_loss(
    pred: dict[str, Tensor],
    batch: dict[str, Tensor],
    sample_weight: Tensor | None = None,
    supervision_mask: Tensor | None = None,
    huber_w: float = 1.0,
    bce_w: float = 0.5,
    huber_beta: float = 0.05,
) -> tuple[Tensor, dict]:
    """Compute Head S loss (Huber axis_ratio + BCE is_asymmetric).

    Args:
      pred:              forward() 输出 dict。
      batch:             含 "axis_ratio" (B,) float、"is_asymmetric" (B,) bool。
      sample_weight:     (B,) float per-sample 权重,None → 全 1.0。
      supervision_mask:  (B,) bool,True 样本参与损失。
                         None → 使用全部样本。
                         空 mask (全 False) → 返回 0 标量 + {"n": 0}。
      huber_w:           Huber 损失权重 (默认 1.0)。
      bce_w:             BCE 损失权重 (默认 0.5)。
      huber_beta:        SmoothL1 折点 (默认 0.05)。

    Returns:
      (loss_scalar, log_dict)
        log_dict keys: "huber", "bce", "n"
    """
    axis_ratio_pred = pred["axis_ratio"]       # (B,)
    asym_logit = pred["asym_logit"]            # (B,)

    B = axis_ratio_pred.shape[0]
    device = axis_ratio_pred.device

    # 确定有效 mask
    if supervision_mask is None:
        mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        mask = supervision_mask.to(device=device, dtype=torch.bool)

    n_valid = int(mask.sum().item())

    # 空 mask → 返回 0 标量 (仍挂在计算图上,梯度为 0)
    if n_valid == 0:
        zero = (axis_ratio_pred.sum() + asym_logit.sum()) * 0.0
        return zero, {"huber": 0.0, "bce": 0.0, "n": 0}

    # 取有效样本
    ar_pred = axis_ratio_pred[mask]            # (n,)
    al_pred = asym_logit[mask]                 # (n,)
    ar_gt = batch["axis_ratio"].to(device=device, dtype=ar_pred.dtype)[mask]  # (n,)
    asym_gt = batch["is_asymmetric"].to(device=device)[mask].float()          # (n,)

    # 权重
    if sample_weight is not None:
        w = sample_weight.to(device=device, dtype=ar_pred.dtype)[mask]  # (n,)
        wsum = w.sum().clamp(min=1e-6)
    else:
        w = None
        wsum = float(n_valid)

    # 1. Huber (SmoothL1) on axis_ratio
    huber_per = F.smooth_l1_loss(ar_pred, ar_gt, beta=huber_beta, reduction="none")  # (n,)
    if w is not None:
        L_huber = (huber_per * w).sum() / wsum
    else:
        L_huber = huber_per.mean()

    # 2. BCEWithLogits on is_asymmetric
    bce_per = F.binary_cross_entropy_with_logits(al_pred, asym_gt, reduction="none")  # (n,)
    if w is not None:
        L_bce = (bce_per * w).sum() / wsum
    else:
        L_bce = bce_per.mean()

    loss = huber_w * L_huber + bce_w * L_bce

    log_dict = {
        "huber": float(L_huber.detach().item()),
        "bce": float(L_bce.detach().item()),
        "n": n_valid,
    }
    return loss, log_dict


def head_s_metrics(
    pred: dict[str, Tensor],
    batch: dict[str, Tensor],
    mask: Tensor | None = None,
) -> dict[str, float]:
    """Evaluation metrics for Head S.

    Computes:
      - s_axis_mae:  mean absolute error on axis_ratio
      - s_asym_f1:   F1 score for is_asymmetric (threshold 0.5 on sigmoid(asym_logit))
      - s_asym_acc:  accuracy for is_asymmetric

    Args:
      pred:   forward() 输出 dict。
      batch:  含 "axis_ratio"、"is_asymmetric"。
      mask:   (B,) bool,None → 全部样本参与。

    Returns:
      dict[str, float]
    """
    ar_pred = pred["axis_ratio"].detach().cpu()    # (B,)
    al_pred = pred["asym_logit"].detach().cpu()    # (B,)
    ar_gt = batch["axis_ratio"].detach().cpu().float()
    asym_gt = batch["is_asymmetric"].detach().cpu().bool()

    if mask is not None:
        m = mask.cpu().bool()
        ar_pred = ar_pred[m]
        al_pred = al_pred[m]
        ar_gt = ar_gt[m]
        asym_gt = asym_gt[m]

    n = ar_pred.shape[0]
    if n == 0:
        return {"s_axis_mae": float("nan"), "s_asym_f1": float("nan"), "s_asym_acc": float("nan")}

    # axis_ratio MAE
    mae = float((ar_pred - ar_gt).abs().mean().item())

    # is_asymmetric 二元预测 (sigmoid > 0.5 等价于 logit > 0)
    asym_pred_bool = al_pred > 0.0  # (n,) bool

    tp = int((asym_pred_bool & asym_gt).sum().item())
    fp = int((asym_pred_bool & ~asym_gt).sum().item())
    fn = int((~asym_pred_bool & asym_gt).sum().item())
    correct = int((asym_pred_bool == asym_gt).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = correct / n

    return {
        "s_axis_mae": mae,
        "s_asym_f1": f1,
        "s_asym_acc": acc,
    }
