"""Head B v2: image-level tip state classification.

取自 Handbook v2.2 §9b (PR-9b: Head B v2 — 形态学 + 多针尖 + 稳定性五子头)。

8 个独立训练的子头,共享 cls_with_scale (1088-d) 输入:

细粒度 (monitoring,不卡 T0 gate):
  - morph (4-way CE):          α₁ type {M0, M1, M2, M3}
  - multitip (Smooth L1):      log2(n_tips) ∈ [0, 3]
  - switching (binary BCE):    α₂ has_switching
  - drift (binary BCE):        α₂ has_drift
  - perturbation (binary BCE): α₂ has_perturbation

粗粒度 (v2.2 新增,T0 hard gate 挂这里;独立分类器,非细粒度 logits 聚合):
  - morph_coarse (3-way CE):       M0 / M1 / M2M3_bad (M2 与 M3 合并)
  - multitip_coarse (binary BCE):  single / multi
  - stability_coarse (binary BCE): stable / unstable (OR 合并)

本模块是独立 nn.Module,输入是 ``cls_with_scale`` 张量 (B, 1088),不 import backbone。
粗粒度标签由 loader (PR-4 shard_loader) 现算,不增 shard metadata 字段。

偏离 Handbook v2.2:
  - §9b.3 ``head_b_metrics`` / ``_coarse_morph_macro_f1`` / ``_coarse_morph_acc``
    在函数体内 ``import numpy as np`` / ``from sklearn.metrics import ...`` ——
    sklearn 是重依赖,保留函数内延迟 import(施工规则 C);numpy 本可顶层 import,
    但文档原文把 ``np`` 也写在函数内,为减少与文档的字面差异此处保留函数内 import。
  - 其余代码逐字对齐文档,无逻辑改动。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# Enum codes matching shard metadata
ALPHA1_M0 = 0  # s-wave (clean default tip)
ALPHA1_M1 = 1  # p-wave / CO functionalized
ALPHA1_M2 = 2  # contaminated apex
ALPHA1_M3 = 3  # uncontrolled asymmetric adsorbate
NUM_MORPH_CLASSES = 4
NUM_MORPH_COARSE_CLASSES = 3  # v2.2: M0 / M1 / M2M3_bad

LOG2_N_TIPS_MAX = 3.0  # n_tips=8 -> log2=3.0
LOG2_N_TIPS_MIN = 0.0  # n_tips=1 -> log2=0.0

# v2.2: coarse morph mapping. M0->0, M1->1, M2 or M3->2 (bad).
MORPH_COARSE_MAP = torch.tensor([0, 1, 2, 2], dtype=torch.long)


@dataclass(frozen=True)
class HeadBPredictions:
    """Forward output of Head B v2.

    v2.2: includes 3 coarse-grained fallback heads (the T0 hard-gate signals).
    """

    # -- Fine-grained (monitoring; v2.1 unchanged) --
    morph_logits: Tensor        # (B, 4)   — M0/M1/M2/M3
    log2_n_tips: Tensor         # (B,)     — regression
    switching_logit: Tensor     # (B,)     — binary
    drift_logit: Tensor         # (B,)     — binary
    perturbation_logit: Tensor  # (B,)     — binary
    # -- Coarse-grained (T0 gate; v2.2 new) --
    morph_coarse_logits: Tensor     # (B, 3) — M0/M1/M2M3_bad
    multitip_coarse_logit: Tensor   # (B,)   — binary single/multi
    stability_coarse_logit: Tensor  # (B,)   — binary stable/unstable

    def as_probs(self) -> dict[str, Tensor]:
        """Convert logits to probabilities for inference."""
        return {
            # Fine
            "morph_probs": F.softmax(self.morph_logits, dim=-1),  # (B, 4)
            "n_tips_estimate": (
                2 ** self.log2_n_tips.clamp(LOG2_N_TIPS_MIN, LOG2_N_TIPS_MAX)
            ),  # (B,)
            "is_multi_tip_fine": (self.log2_n_tips > 0.5),  # (B,) bool
            "switching_prob": torch.sigmoid(self.switching_logit),
            "drift_prob": torch.sigmoid(self.drift_logit),
            "perturbation_prob": torch.sigmoid(self.perturbation_logit),
            # Coarse (v2.2)
            "morph_coarse_probs": F.softmax(
                self.morph_coarse_logits, dim=-1
            ),  # (B, 3)
            "multitip_coarse_prob": torch.sigmoid(self.multitip_coarse_logit),
            "stability_coarse_prob": torch.sigmoid(self.stability_coarse_logit),
        }


class HeadBv2(nn.Module):
    """8 sub-heads for image-level tip state classification.

    v2.2: 5 fine + 3 coarse fallback heads. Coarse heads are independently
    trained (not aggregations of fine logits).
    """

    def __init__(
        self,
        in_dim: int = 1088,
        morph_hidden: int = 256,
        multitip_hidden: int = 128,
        stability_hidden: int = 64,
        # v2.2: coarse heads use smaller MLPs (simpler task, less capacity needed)
        morph_coarse_hidden: int = 128,
        multitip_coarse_hidden: int = 64,
        stability_coarse_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        def _mlp(out_dim: int, hidden: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )

        # Fine-grained (v2.1)
        self.morph = _mlp(NUM_MORPH_CLASSES, morph_hidden)
        self.multitip = _mlp(1, multitip_hidden)
        self.switching = _mlp(1, stability_hidden)
        self.drift = _mlp(1, stability_hidden)
        self.perturbation = _mlp(1, stability_hidden)
        # -- v2.2: coarse fallback heads --
        self.morph_coarse = _mlp(NUM_MORPH_COARSE_CLASSES, morph_coarse_hidden)
        self.multitip_coarse = _mlp(1, multitip_coarse_hidden)
        self.stability_coarse = _mlp(1, stability_coarse_hidden)

    def forward(self, cls_with_scale: Tensor) -> HeadBPredictions:
        """cls_with_scale: (B, in_dim)."""
        return HeadBPredictions(
            # Fine
            morph_logits=self.morph(cls_with_scale),
            log2_n_tips=self.multitip(cls_with_scale).squeeze(-1),
            switching_logit=self.switching(cls_with_scale).squeeze(-1),
            drift_logit=self.drift(cls_with_scale).squeeze(-1),
            perturbation_logit=self.perturbation(cls_with_scale).squeeze(-1),
            # Coarse (v2.2)
            morph_coarse_logits=self.morph_coarse(cls_with_scale),
            multitip_coarse_logit=self.multitip_coarse(cls_with_scale).squeeze(-1),
            stability_coarse_logit=self.stability_coarse(cls_with_scale).squeeze(
                -1
            ),
        )


# -----------------------------------------------------------------
# Losses
# -----------------------------------------------------------------


def _class_balanced_weights(counts: Tensor, beta: float = 0.999) -> Tensor:
    """Cui et al. 2019 effective number reweighting.

    Args:
      counts: (C,) per-class sample counts (float)
    Returns:
      (C,) normalized weights summing to C (so per-sample loss has unit avg scale)
    """
    effective_num = 1.0 - beta ** counts.clamp(min=1.0)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    return weights / weights.sum() * counts.numel()


def head_b_loss(
    pred: HeadBPredictions,
    target: dict[str, Tensor],
    class_counts_morph: Tensor | None = None,
    pos_counts: dict[str, Tensor] | None = None,
    # v2.2: coarse-head supports
    class_counts_morph_coarse: Tensor | None = None,
    pos_counts_coarse: dict[str, Tensor] | None = None,
    focal_gamma: float = 2.0,
    multitip_huber_beta: float = 0.2,
) -> dict[str, Tensor]:
    """Compute all 8 sub-head losses (5 fine + 3 coarse).

    Args:
      pred: HeadBPredictions
      target: dict with keys:
        Fine (v2.1):
        - "alpha1_type": (B,) long ∈ {0, 1, 2, 3}
        - "log2_n_tips": (B,) float
        - "has_switching", "has_drift", "has_perturbation": (B,) bool/float
        Coarse (v2.2, supplied by loader; PR-4):
        - "morph_coarse_label": (B,) long ∈ {0, 1, 2}  (M0 / M1 / M2M3_bad)
        - "is_multi_tip_label": (B,) bool
        - "is_unstable_label": (B,) bool
      class_counts_morph: (4,) running counts of α₁ types
      pos_counts: dict {"switching", "drift", "perturbation"} -> [neg, pos] counts
      class_counts_morph_coarse: (3,) running counts (v2.2)
      pos_counts_coarse: dict {"multitip_coarse", "stability_coarse"} -> [neg, pos] (v2.2)
      focal_gamma: focal modulation exponent
      multitip_huber_beta: Smooth L1 transition (log2 domain, so small beta is fine)

    Returns:
      Dict with "total" + 8 sub-loss tensors (all LIVE for uncertainty weighting):
        Fine: "morph", "multitip", "switching", "drift", "perturbation"
        Coarse: "morph_coarse", "multitip_coarse", "stability_coarse"
    """
    device = pred.morph_logits.device

    # 1. Morphology — class-balanced focal CE
    if class_counts_morph is None:
        class_counts_morph = torch.ones(NUM_MORPH_CLASSES, device=device)
    morph_w = _class_balanced_weights(class_counts_morph.to(device))
    log_probs = F.log_softmax(pred.morph_logits, dim=-1)
    probs = log_probs.exp()
    target_morph = target["alpha1_type"].long().to(device)
    log_p_t = log_probs.gather(1, target_morph.unsqueeze(1)).squeeze(1)
    p_t = probs.gather(1, target_morph.unsqueeze(1)).squeeze(1)
    focal_term = (1 - p_t) ** focal_gamma
    w_t = morph_w[target_morph]
    L_morph = (-focal_term * log_p_t * w_t).mean()

    # 2. Multi-tip — Smooth L1 in log2 domain
    target_log2 = (
        target["log2_n_tips"]
        .float()
        .to(device)
        .clamp(LOG2_N_TIPS_MIN, LOG2_N_TIPS_MAX)
    )
    L_multitip = F.smooth_l1_loss(
        pred.log2_n_tips.clamp(LOG2_N_TIPS_MIN - 0.5, LOG2_N_TIPS_MAX + 0.5),
        target_log2,
        beta=multitip_huber_beta,
    )

    # 3-5. Stability triplet — class-balanced focal BCE per task
    def _binary_focal_bce(
        logit: Tensor, target_bin: Tensor, neg_count: float, pos_count: float
    ) -> Tensor:
        # Effective number reweighting
        eff_pos = 1.0 - 0.999 ** max(pos_count, 1.0)
        eff_neg = 1.0 - 0.999 ** max(neg_count, 1.0)
        w_pos = (1 - 0.999) / (eff_pos + 1e-8)
        w_neg = (1 - 0.999) / (eff_neg + 1e-8)
        ratio = w_pos / max(w_neg, 1e-8)
        pos_w = torch.tensor([ratio], device=logit.device)
        bce = F.binary_cross_entropy_with_logits(
            logit,
            target_bin.float().to(logit.device),
            pos_weight=pos_w,
            reduction="none",
        )
        with torch.no_grad():
            p = torch.sigmoid(logit)
            p_t = torch.where(target_bin.bool().to(logit.device), p, 1 - p)
        focal_mod = (1 - p_t) ** focal_gamma
        return (bce * focal_mod).mean()

    def _counts(key: str) -> tuple[float, float]:
        if pos_counts is None or key not in pos_counts:
            return 1.0, 1.0
        c = pos_counts[key]
        return float(c[0].item()), float(c[1].item())

    n_neg_s, n_pos_s = _counts("switching")
    n_neg_d, n_pos_d = _counts("drift")
    n_neg_p, n_pos_p = _counts("perturbation")

    L_switching = _binary_focal_bce(
        pred.switching_logit, target["has_switching"], n_neg_s, n_pos_s
    )
    L_drift = _binary_focal_bce(
        pred.drift_logit, target["has_drift"], n_neg_d, n_pos_d
    )
    L_perturbation = _binary_focal_bce(
        pred.perturbation_logit, target["has_perturbation"], n_neg_p, n_pos_p
    )

    # -- v2.2: coarse-head losses --

    # 6. Morph coarse — 3-way class-balanced focal CE
    if class_counts_morph_coarse is None:
        class_counts_morph_coarse = torch.ones(
            NUM_MORPH_COARSE_CLASSES, device=device
        )
    morph_coarse_w = _class_balanced_weights(class_counts_morph_coarse.to(device))
    log_probs_c = F.log_softmax(pred.morph_coarse_logits, dim=-1)
    probs_c = log_probs_c.exp()
    target_morph_coarse = target["morph_coarse_label"].long().to(device)
    log_p_t_c = log_probs_c.gather(
        1, target_morph_coarse.unsqueeze(1)
    ).squeeze(1)
    p_t_c = probs_c.gather(1, target_morph_coarse.unsqueeze(1)).squeeze(1)
    focal_term_c = (1 - p_t_c) ** focal_gamma
    w_t_c = morph_coarse_w[target_morph_coarse]
    L_morph_coarse = (-focal_term_c * log_p_t_c * w_t_c).mean()

    # 7-8. Coarse binary heads
    def _coarse_counts(key: str) -> tuple[float, float]:
        if pos_counts_coarse is None or key not in pos_counts_coarse:
            return 1.0, 1.0
        c = pos_counts_coarse[key]
        return float(c[0].item()), float(c[1].item())

    n_neg_mt, n_pos_mt = _coarse_counts("multitip_coarse")
    n_neg_st, n_pos_st = _coarse_counts("stability_coarse")

    L_multitip_coarse = _binary_focal_bce(
        pred.multitip_coarse_logit,
        target["is_multi_tip_label"],
        n_neg_mt,
        n_pos_mt,
    )
    L_stability_coarse = _binary_focal_bce(
        pred.stability_coarse_logit,
        target["is_unstable_label"],
        n_neg_st,
        n_pos_st,
    )

    total = (
        L_morph
        + L_multitip
        + L_switching
        + L_drift
        + L_perturbation
        + L_morph_coarse
        + L_multitip_coarse
        + L_stability_coarse
    )

    return {
        "total": total,
        # v2.1 sub-losses (LIVE for uncertainty weighting)
        "morph": L_morph,
        "multitip": L_multitip,
        "switching": L_switching,
        "drift": L_drift,
        "perturbation": L_perturbation,
        # -- v2.2: coarse-head losses (LIVE) --
        "morph_coarse": L_morph_coarse,
        "multitip_coarse": L_multitip_coarse,
        "stability_coarse": L_stability_coarse,
    }


# -----------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------


def head_b_metrics(
    pred: HeadBPredictions, target: dict[str, Tensor]
) -> dict[str, float]:
    """Compute evaluation metrics for Head B v2."""
    # 延迟 import 重依赖(施工规则 C)。
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    # Morph: macro-F1
    morph_pred = pred.morph_logits.argmax(dim=-1).cpu().numpy()
    morph_target = target["alpha1_type"].cpu().numpy()
    try:
        morph_macro_f1 = f1_score(
            morph_target, morph_pred, average="macro", zero_division=0
        )
    except ValueError:
        morph_macro_f1 = 0.0
    morph_acc = (morph_pred == morph_target).mean()

    # Multi-tip: MAE in log2 domain + binary AUROC for "is_multi_tip"
    multitip_pred = pred.log2_n_tips.cpu().numpy()
    multitip_target = target["log2_n_tips"].cpu().numpy()
    multitip_mae = float(np.abs(multitip_pred - multitip_target).mean())
    is_multi_target = (multitip_target > 0.5).astype(int)
    try:
        multitip_auroc = roc_auc_score(is_multi_target, multitip_pred)
    except ValueError:
        multitip_auroc = 0.5

    # Stability: AUROC each
    def _auroc(logit: Tensor, target_bin: Tensor) -> float:
        try:
            return float(
                roc_auc_score(
                    target_bin.cpu().numpy().astype(int),
                    torch.sigmoid(logit).cpu().numpy(),
                )
            )
        except ValueError:
            return 0.5

    return {
        # Fine (v2.1)
        "b_morph_macro_f1": float(morph_macro_f1),
        "b_morph_acc": float(morph_acc),
        "b_multitip_mae": multitip_mae,
        "b_multitip_auroc": float(multitip_auroc),
        "b_switching_auroc": _auroc(
            pred.switching_logit, target["has_switching"]
        ),
        "b_drift_auroc": _auroc(pred.drift_logit, target["has_drift"]),
        "b_perturbation_auroc": _auroc(
            pred.perturbation_logit, target["has_perturbation"]
        ),
        # -- v2.2: coarse-head metrics (T0 gate signals) --
        "b_morph_coarse_macro_f1": _coarse_morph_macro_f1(pred, target),
        "b_morph_coarse_acc": _coarse_morph_acc(pred, target),
        "b_multitip_coarse_auroc": _auroc(
            pred.multitip_coarse_logit, target["is_multi_tip_label"]
        ),
        "b_stability_coarse_auroc": _auroc(
            pred.stability_coarse_logit, target["is_unstable_label"]
        ),
    }


def _coarse_morph_macro_f1(
    pred: HeadBPredictions, target: dict[str, Tensor]
) -> float:
    """3-way macro-F1 for morph_coarse head."""
    from sklearn.metrics import f1_score

    p = pred.morph_coarse_logits.argmax(dim=-1).cpu().numpy()
    t = target["morph_coarse_label"].cpu().numpy()
    try:
        return float(f1_score(t, p, average="macro", zero_division=0))
    except ValueError:
        return 0.0


def _coarse_morph_acc(
    pred: HeadBPredictions, target: dict[str, Tensor]
) -> float:
    p = pred.morph_coarse_logits.argmax(dim=-1).cpu().numpy()
    t = target["morph_coarse_label"].cpu().numpy()
    return float((p == t).mean())


# -----------------------------------------------------------------
# Running counts helper (for class-balanced weighting at training time)
# -----------------------------------------------------------------


class HeadBCountTracker:
    """Track running counts of α₁ types and stability flags for class-balanced loss.

    Updated each batch in trainer; counts passed into head_b_loss.
    v2.2: also tracks coarse-head class counts.
    """

    def __init__(self, ema_decay: float = 0.99) -> None:
        # Fine
        self.morph_counts = torch.zeros(NUM_MORPH_CLASSES)
        self.switching_counts = torch.zeros(2)  # [neg, pos]
        self.drift_counts = torch.zeros(2)
        self.perturbation_counts = torch.zeros(2)
        # -- v2.2: coarse-head counts --
        self.morph_coarse_counts = torch.zeros(NUM_MORPH_COARSE_CLASSES)
        self.multitip_coarse_counts = torch.zeros(2)
        self.stability_coarse_counts = torch.zeros(2)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update(self, target: dict[str, Tensor]) -> None:
        # Morph (fine)
        alpha1 = target["alpha1_type"].long().cpu()
        batch_morph = torch.bincount(
            alpha1, minlength=NUM_MORPH_CLASSES
        ).float()
        self.morph_counts = (
            self.ema_decay * self.morph_counts
            + (1 - self.ema_decay) * batch_morph
        )
        # Stability triplet (fine)
        for key, attr in [
            ("has_switching", "switching_counts"),
            ("has_drift", "drift_counts"),
            ("has_perturbation", "perturbation_counts"),
        ]:
            flag = target[key].bool().cpu()
            batch_counts = torch.tensor(
                [
                    (~flag).sum().item(),  # negatives
                    flag.sum().item(),     # positives
                ],
                dtype=torch.float32,
            )
            cur = getattr(self, attr)
            setattr(
                self,
                attr,
                self.ema_decay * cur + (1 - self.ema_decay) * batch_counts,
            )
        # -- v2.2: coarse-head counts --
        if "morph_coarse_label" in target:
            mc = target["morph_coarse_label"].long().cpu()
            batch_mc = torch.bincount(
                mc, minlength=NUM_MORPH_COARSE_CLASSES
            ).float()
            self.morph_coarse_counts = (
                self.ema_decay * self.morph_coarse_counts
                + (1 - self.ema_decay) * batch_mc
            )
        for key, attr in [
            ("is_multi_tip_label", "multitip_coarse_counts"),
            ("is_unstable_label", "stability_coarse_counts"),
        ]:
            if key not in target:
                continue
            flag = target[key].bool().cpu()
            batch_counts = torch.tensor(
                [
                    (~flag).sum().item(),
                    flag.sum().item(),
                ],
                dtype=torch.float32,
            )
            cur = getattr(self, attr)
            setattr(
                self,
                attr,
                self.ema_decay * cur + (1 - self.ema_decay) * batch_counts,
            )

    def pos_counts(self) -> dict[str, Tensor]:
        """Fine-grained stability triplet pos counts."""
        return {
            "switching": self.switching_counts,
            "drift": self.drift_counts,
            "perturbation": self.perturbation_counts,
        }

    def pos_counts_coarse(self) -> dict[str, Tensor]:
        """v2.2: coarse binary head pos counts."""
        return {
            "multitip_coarse": self.multitip_coarse_counts,
            "stability_coarse": self.stability_coarse_counts,
        }
