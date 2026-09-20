"""Head N: apex 数目预测 (n_sub_apex + 1)。

目标
----
预测 STM 针尖的 apex 数目 n_apex = n_sub_apex + 1（1=单尖, 2=双尖, ≥3=多尖）。
该量由 C2 生成器写入 batch["n_apex"]（float），并预先折叠为序数三类
batch["n_class_ord"]（0→1个, 1→2个, 2→≥3个, float32 存储, 取 .long() 使用）。

rev3 §1 简化(v2.5 PR N,默认)
-----------------------------
默认改为**二元** {1, ≥2}：``mode="binary"``，out_dim=1，BCEWithLogits on
``(n_apex >= 2)``。这里用 ``n_apex = n_sub_apex + 1``(与 supervision 的 single
定义一致),**不用** 仅 M0 才有的 ``alpha1_params.n_tips``。保留 ordinal/regression
模式作对照。

输入
----
``cls_with_scale`` 张量 (B, in_dim)，由 backbone [CLS] token ‖ ScaleEmbedding(64) 拼接而成。
对 vits16：embed_dim=384，in_dim=384+64=448。
对 vitl16：embed_dim=1024，in_dim=1024+64=1088。
**in_dim 必须由调用方参数化传入，本模块不硬编码任何骨干维度。**

标签
----
- ``batch["n_apex"]``     : float，回归目标(mode="regression")；二元阈值
                            (n_apex>=2) 也由它派生(mode="binary")。
- ``batch["n_class_ord"]``: float32(0/1/2 序数)，分类目标(mode="ordinal"，需 .long())。

监督域
------
仅在 stable=True 的样本上监督（由 driver 传入 supervision_mask）。

损失
----
- mode="binary"    → 二元 {1, ≥2} BCEWithLogits on (n_apex>=2)（默认）
- mode="ordinal"   → 三类序数交叉熵 (CE)
- mode="regression"→ Huber (Smooth L1) 回归
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HeadN(nn.Module):
    """预测 apex 数目：二元 {1,≥2} BCE / 序数三类 CE / 连续 Huber 回归。

    MLP 结构: Linear(in_dim, hidden) → GELU → Dropout → Linear(hidden, out)
      mode="binary"    : out=1（{1, ≥2} 二元 logit；默认）
      mode="ordinal"   : out=3（{1, 2, ≥3} 三类 logits）
      mode="regression": out=1（回归 n_apex）

    Args:
        in_dim  : 输入维度，= embed_dim + 64 (ScaleEmbedding)。vits16 → 448，vitl16 → 1088。
        hidden  : MLP 隐藏层维度，默认 256。
        dropout : Dropout 概率，默认 0.1。
        mode    : "binary"（默认）/ "ordinal" / "regression"。
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        dropout: float = 0.1,
        mode: str = "binary",
    ) -> None:
        super().__init__()
        if mode not in ("binary", "ordinal", "regression"):
            raise ValueError(
                f"HeadN mode must be 'binary' | 'ordinal' | 'regression', got {mode!r}"
            )
        self.mode = mode
        out_dim = 3 if mode == "ordinal" else 1  # binary/regression 都 out=1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feats: Tensor) -> Tensor:
        """前向推理。

        Args:
            feats: (B, in_dim) — cls_with_scale 张量

        Returns:
            mode="binary"    → (B,)   {1,≥2} 单 logit
            mode="ordinal"   → (B, 3) logits（未经 softmax）
            mode="regression"→ (B,)   预测 n_apex
        """
        out = self.mlp(feats)
        if self.mode in ("binary", "regression"):
            return out.squeeze(-1)  # (B,)
        return out  # (B, 3)


def _infer_n_mode(pred: Tensor, mode: str | None) -> str:
    """确定 Head N 当前 mode。显式 mode 优先;否则按 pred 维度推断。

    pred (B,3) → ordinal;pred (B,) 维度无法区分 binary/regression,
    此时若未显式给 mode 则默认 'regression'(保持向后兼容旧调用)。
    """
    if mode is not None:
        return mode
    if pred.dim() == 2 and pred.shape[-1] == 3:
        return "ordinal"
    return "regression"


def head_n_loss(
    pred: Tensor,
    batch: dict,
    sample_weight: Tensor | None = None,
    supervision_mask: Tensor | None = None,
    mode: str | None = None,
) -> tuple[Tensor, dict]:
    """计算 Head N 损失。

    Args:
        pred             : HeadN 输出。
                           binary    → (B,)   {1,≥2} 单 logit
                           ordinal   → (B, 3) logits
                           regression→ (B,)
        batch            : 含键 ``n_apex`` (float) 和 ``n_class_ord`` (float32 需 .long())。
        sample_weight    : (B,) float，逐样本权重；None → 全 1.0。
        supervision_mask : (B,) bool；只在 True 样本上计算损失。
                           全 False 或 None（且 pred 无法提供有效样本）→ 返回
                           零梯度安全损失 + {"n": 0}。
        mode             : 显式 mode("binary"/"ordinal"/"regression")。None → 按 pred
                           维度推断(注:(B,) 无法区分 binary/regression,会落到 regression,
                           故 binary 调用方**必须**显式传 mode="binary")。

    Returns:
        (loss, logdict)
        logdict 含损失分量及有效样本数 "n"。
    """
    B = pred.shape[0]
    device = pred.device
    mode = _infer_n_mode(pred, mode)

    # --- 构造有效样本掩码 ---
    if supervision_mask is None:
        mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        mask = supervision_mask.to(device=device, dtype=torch.bool)

    n_valid = int(mask.sum().item())

    # 全无有效样本 → 零梯度安全损失
    if n_valid == 0:
        zero = pred.sum() * 0.0  # requires_grad=True（pred 来自 nn.Module）
        return zero, {"n_loss": zero.detach(), "n": 0}

    # --- 切出有效样本 ---
    pred_m = pred[mask]  # (n_valid, 3) 或 (n_valid,)

    if sample_weight is None:
        w = torch.ones(n_valid, device=device, dtype=torch.float32)
    else:
        w = sample_weight.to(device=device, dtype=torch.float32)[mask]

    wsum = w.sum().clamp(min=1e-6)

    if mode == "ordinal":
        # 序数三类交叉熵
        labels = batch["n_class_ord"]
        if not isinstance(labels, Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.to(device=device).long()[mask]  # (n_valid,)

        ce_per = F.cross_entropy(pred_m, labels, reduction="none")  # (n_valid,)
        loss = (ce_per * w).sum() / wsum
        logdict: dict = {"n_loss": loss.detach(), "n": n_valid}
    elif mode == "binary":
        # 二元 {1, ≥2}: BCEWithLogits on (n_apex >= 2)
        n_apex = batch["n_apex"]
        if not isinstance(n_apex, Tensor):
            n_apex = torch.tensor(n_apex, dtype=torch.float32)
        target = (n_apex.to(device=device, dtype=torch.float32) >= 2.0).float()[mask]  # (n_valid,)

        bce_per = F.binary_cross_entropy_with_logits(pred_m, target, reduction="none")
        loss = (bce_per * w).sum() / wsum
        logdict = {"n_loss": loss.detach(), "n": n_valid}
    else:
        # Huber (Smooth L1) 回归
        targets = batch["n_apex"]
        if not isinstance(targets, Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)[mask]  # (n_valid,)

        huber_per = F.smooth_l1_loss(pred_m, targets, reduction="none")  # (n_valid,)
        loss = (huber_per * w).sum() / wsum
        logdict = {"n_loss": loss.detach(), "n": n_valid}

    return loss, logdict


def head_n_metrics(
    pred: Tensor,
    batch: dict,
    mask: Tensor | None = None,
    mode: str | None = None,
) -> dict[str, float]:
    """评估指标。

    Args:
        pred : HeadN 输出（binary→(B,), ordinal→(B,3), regression→(B,)）。
        batch: 含 ``n_apex`` / ``n_class_ord``。
        mask : (B,) bool，可选子集。None → 全部样本。
        mode : 显式 mode；None → 按 pred 维度推断((B,) 默认 regression,binary 须显式)。

    Returns:
        binary    → {"n_f1": float, "n_acc": float}
        ordinal   → {"n_macro_f1": float, "n_acc": float}
        regression→ {"n_mae": float}
    """
    B = pred.shape[0]
    device = pred.device
    mode = _infer_n_mode(pred, mode)

    if mask is None:
        mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)

    n_valid = int(mask.sum().item())

    if mode == "ordinal":
        from sklearn.metrics import f1_score  # 延迟 import（重依赖，施工规则 C）

        if n_valid == 0:
            return {"n_macro_f1": 0.0, "n_acc": 0.0}
        labels = batch["n_class_ord"]
        if not isinstance(labels, Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        labels_np = labels.to(device=device).long()[mask].detach().cpu().numpy()
        pred_np = pred[mask].argmax(dim=-1).detach().cpu().numpy()

        macro_f1 = float(
            f1_score(labels_np, pred_np, average="macro", zero_division=0)
        )
        acc = float((pred_np == labels_np).mean())
        return {"n_macro_f1": macro_f1, "n_acc": acc}
    elif mode == "binary":
        if n_valid == 0:
            return {"n_f1": 0.0, "n_acc": 0.0}
        n_apex = batch["n_apex"]
        if not isinstance(n_apex, Tensor):
            n_apex = torch.tensor(n_apex, dtype=torch.float32)
        target = (n_apex.to(device=device, dtype=torch.float32) >= 2.0).bool()[mask]
        pred_pos = (pred[mask] > 0).bool()

        tp = float((pred_pos & target).sum().item())
        fp = float((pred_pos & ~target).sum().item())
        fn = float((~pred_pos & target).sum().item())
        tn = float((~pred_pos & ~target).sum().item())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        acc = ((tp + tn) / (tp + fp + fn + tn)) if n_valid > 0 else 0.0
        return {"n_f1": f1, "n_acc": acc}
    else:
        if n_valid == 0:
            return {"n_mae": 0.0}
        targets = batch["n_apex"]
        if not isinstance(targets, Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)
        targets_np = targets.to(device=device, dtype=torch.float32)[mask].detach().cpu().numpy()
        pred_np = pred[mask].detach().cpu().numpy()

        mae = float(abs(pred_np - targets_np).mean())
        return {"n_mae": mae}
