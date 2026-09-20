"""Head K: 针尖污染图级二元分类 (tip contamination, image-level binary).

目标
----
预测当前扫描图像是否存在针尖污染 (tip contamination)。
污染标志来自 mask 高字节 bit 10 (TipFlag axis-α v2 TIP_CONTAMINATION)，
在已有 DataLoader 中以 ``batch["tipflag_contam_image"]`` (bool) 暴露。

输入
----
``feats``: (B, in_dim) = CLS‖scale_emb，in_dim = embed_dim + 64
  e.g. vits16 -> embed_dim=384, in_dim=448

标签
----
``batch["tipflag_contam_image"]``: bool Tensor (B,)，True = 污染，False = 干净。
  等同 M2 污染标注 (vigil_v2.2 M2 ContamHead label)。

监督域
------
supervision = all（不过滤 instability 样本，与 PR-25.3 结论一致）

损失
----
BCEWithLogitsLoss。支持 pos_weight 处理类不平衡（污染样本通常远少于干净样本）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HeadK(nn.Module):
    """Predict image-level tip-contamination logit from [CLS] ‖ scale_emb.

    Architecture: Linear(in_dim, hidden) → GELU → Dropout → Linear(hidden, 1)
    Output: (B,) raw logit; apply sigmoid for probability.
    """

    def __init__(
        self,
        in_dim: int = 448,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: Tensor) -> Tensor:
        """Forward pass.

        Args:
            feats: (B, in_dim) concatenated [CLS] ‖ scale_emb.

        Returns:
            (B,) raw logits (un-sigmoided).
        """
        return self.mlp(feats).squeeze(-1)


def head_k_loss(
    pred: Tensor,
    batch: dict,
    sample_weight: Tensor | None = None,
    supervision_mask: Tensor | None = None,
    pos_weight: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """Compute BCEWithLogits loss for Head K (tip contamination).

    Args:
        pred: (B,) raw logits from HeadK.forward().
        batch: dict with key ``tipflag_contam_image`` (bool or {0,1} Tensor, shape (B,)).
        sample_weight: optional (B,) per-sample weights. None → all 1.0.
        supervision_mask: optional (B,) bool Tensor. Only True samples contribute
            to the loss. None → all samples used.
            If the filtered set is empty, returns (0.0_tensor, {"bce": 0.0, "n": 0, "n_pos": 0}).
        pos_weight: optional scalar or (1,) Tensor passed to BCEWithLogitsLoss
            to up-weight positive (contaminated) samples. Recommended value ≈
            (n_neg / n_pos) when the dataset is imbalanced.

    Returns:
        (loss_tensor, log_dict)
        log_dict keys: ``bce`` (float), ``n`` (int), ``n_pos`` (int).
    """
    target = batch["tipflag_contam_image"].float().to(pred.device)

    # Apply supervision mask
    if supervision_mask is not None:
        mask = supervision_mask.bool().to(pred.device)
        pred = pred[mask]
        target = target[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]

    n = pred.shape[0]
    if n == 0:
        zero = pred.sum() * 0.0
        return zero, {"bce": 0.0, "n": 0, "n_pos": 0}

    n_pos = int(target.sum().item())

    if sample_weight is not None:
        w = sample_weight.to(pred.device)
        # Weighted BCE: compute element-wise then scale
        bce_reduction = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none", pos_weight=pos_weight
        )
        wsum = w.sum().clamp(min=1e-6)
        loss = (bce_reduction * w).sum() / wsum
    else:
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="mean", pos_weight=pos_weight
        )

    log_dict: dict = {
        "bce": float(loss.detach().item()),
        "n": n,
        "n_pos": n_pos,
    }
    return loss, log_dict


def head_k_metrics(
    pred: Tensor,
    batch: dict,
    mask: Tensor | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics for Head K.

    Metrics returned:
      ``k_f1``      – binary F1 (threshold 0.5 on sigmoid)
      ``k_acc``     – accuracy
      ``k_auroc``   – AUROC (requires sklearn; skipped if unavailable)
      ``k_pos_rate``– fraction of positive predictions

    Args:
        pred: (B,) raw logits.
        batch: dict with ``tipflag_contam_image``.
        mask: optional (B,) bool Tensor to select a subset.

    Returns:
        Dict of metric name → float.
    """
    target = batch["tipflag_contam_image"].bool().to(pred.device)

    if mask is not None:
        m = mask.bool().to(pred.device)
        pred = pred[m]
        target = target[m]

    p_np = pred.detach().cpu().float().numpy()
    t_np = target.detach().cpu().numpy().astype(int)

    prob = 1.0 / (1.0 + __import__("numpy").exp(-p_np))  # sigmoid
    pred_bin = (prob >= 0.5).astype(int)

    n = len(t_np)
    if n == 0:
        return {"k_f1": 0.0, "k_acc": 0.0, "k_pos_rate": 0.0}

    # Accuracy
    acc = float((pred_bin == t_np).mean())

    # F1
    tp = int(((pred_bin == 1) & (t_np == 1)).sum())
    fp = int(((pred_bin == 1) & (t_np == 0)).sum())
    fn = int(((pred_bin == 0) & (t_np == 1)).sum())
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0

    pos_rate = float(pred_bin.mean())

    metrics: dict[str, float] = {
        "k_f1": f1,
        "k_acc": acc,
        "k_pos_rate": pos_rate,
    }

    # AUROC — 延迟 import sklearn（重依赖，施工规则 C）
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415

        if len(set(t_np)) >= 2:
            auroc = float(roc_auc_score(t_np, prob))
        else:
            auroc = float("nan")
        metrics["k_auroc"] = auroc
    except ImportError:
        pass  # sklearn 不可用时静默跳过

    return metrics
