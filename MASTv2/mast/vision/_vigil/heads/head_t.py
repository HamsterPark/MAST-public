"""Head T —— 图级稳定性二分类头(VIGIL DINO v2.5 PR T 简化)。

目标
----
判断**整图扫描过程中是否发生过针尖不稳定 / 样品-针尖相互作用扰动**(tip-fault):

  image-level unstable = ``has_switching ∨ has_perturbation``

  注意:
    - **不含 drift**(drift 是漂移,属于几何形变,归鲁棒性,不算 tip-fault)。
    - no-op switching/drift 已在 loader 用 ``is_noop_*`` 过滤(见 shard_loader),
      故 batch 直接给的 ``has_switching`` / ``has_perturbation`` 已经是"可见事件"。
    - loader 已把这两者 OR 成 ``tipfault_image``(bool, B);本头优先用它,
      没有时回退到 ``has_switching ∨ has_perturbation``。

rev3 §1 简化(相对 v2.4)
------------------------
  - **删** 逐行(per-row)分支:不再预测 ``tipfault_row_mask`` / ``tipfault_rows_valid``,
    去掉 ``row_mlp`` 与 ``_downsample_row_mask``。
  - **删** switching / perturbation 双通道:图级输出从 2 logits 收成**单 logit**。
  - **mean-pool → learned attention-pool**:不再对 token 取简单均值,而是用一个小
    attention(``scores = Linear(D,1)(tokens)`` → softmax over N → 加权和)聚合。

输入
----
backbone patch tokens ``feats["tokens"]`` (B, N, D) + ``feats["hw"]`` (H', W'),
**不拼 scale**,所以 ``token_dim == embed_dim``(vits16 = 384,NOT 384+64)。

监督域
------
Head T 是"不稳定检测器",**必须见所有样本**(包括稳定的负样本),否则学不到
"什么叫稳定"。因此**不做 supervision_mask 过滤**(为统一接口仍保留 mask 参数,
但内部按全 True 处理)。

损失
----
``loss = BCEWithLogits(单 logit, 图级 unstable label)``,逐样本可加权。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _t_image_label(batch: dict, device, dtype) -> Tensor:
    """图级 unstable label (B,) float:优先 ``tipfault_image``,否则 has_switching∨has_perturbation。"""
    if "tipfault_image" in batch and batch["tipfault_image"] is not None:
        v = batch["tipfault_image"]
        v = v if torch.is_tensor(v) else torch.as_tensor(v)
        return v.to(device=device, dtype=dtype).reshape(-1)
    has_sw = batch["has_switching"]
    has_pt = batch["has_perturbation"]
    has_sw = has_sw if torch.is_tensor(has_sw) else torch.as_tensor(has_sw)
    has_pt = has_pt if torch.is_tensor(has_pt) else torch.as_tensor(has_pt)
    lab = has_sw.to(device=device).bool() | has_pt.to(device=device).bool()
    return lab.to(dtype=dtype).reshape(-1)


class HeadT(nn.Module):
    """图级稳定性头:learned attention-pool → 单 logit binary {stable, unstable}。

    Args:
      token_dim: patch token 维度 = backbone embed_dim(vits16=384)。**不含 scale**。
      hidden:    分类 MLP 的隐藏维度。
      dropout:   隐藏层后的 dropout 概率。
    """

    def __init__(self, token_dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.hidden = hidden

        # learned attention-pool: token (B,N,D) -> scores (B,N,1) -> softmax over N -> 加权和 (B,D)
        self.attn = nn.Linear(token_dim, 1)

        # 图级分类: (B, D) -> Linear -> GELU -> Dropout -> Linear -> (B, 1)
        self.img_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: dict) -> dict:
        """feats: backbone dict,用 ``tokens`` (B,N,D) 与 ``hw`` (H',W')。

        Returns dict:
          img_logit: (B,)     图级 unstable logit(单 logit)
          attn:      (B, N)    attention 权重(softmax 后,供可视化)
          hw:        (H', W')  patch 网格尺寸(透传)
        """
        tokens: Tensor = feats["tokens"]  # (B, N, D)
        h_p, w_p = feats["hw"]
        B, N, D = tokens.shape
        assert D == self.token_dim, f"token_dim mismatch: D={D} != {self.token_dim}"

        # learned attention-pool over N
        scores = self.attn(tokens).squeeze(-1)  # (B, N)
        attn = F.softmax(scores, dim=1)          # (B, N)
        pooled = torch.einsum("bn,bnd->bd", attn, tokens)  # (B, D)

        img_logit = self.img_mlp(pooled).squeeze(-1)  # (B,)
        return {"img_logit": img_logit, "attn": attn, "hw": (h_p, w_p)}


def head_t_loss(
    pred: dict,
    batch: dict,
    sample_weight: Tensor | None = None,
    supervision_mask: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """Head T 损失: 图级单 logit BCEWithLogits。

    Args:
      pred: HeadT.forward 输出 dict(img_logit / attn / hw)。
      batch: 含 ``tipfault_image``(bool,B)或 ``has_switching`` / ``has_perturbation``。
      sample_weight: 可选 (B,) 逐样本权重(为统一接口保留;None → 全 1)。
      supervision_mask: Head T **不过滤**(传 None 或全 True 均可);为统一接口保留。

    Returns:
      (loss, logdict)。空 batch → (0.0*logit.sum(), {"n":0})。
    """
    img_logit: Tensor = pred["img_logit"]  # (B,)
    device = img_logit.device
    B = img_logit.shape[0]

    # 空 batch 保护: 返回一个仍连着计算图的 0(让 .backward() 不报错)。
    if B == 0:
        zero = 0.0 * img_logit.sum()
        return zero, {"n": 0, "n_pos": 0}

    if sample_weight is None:
        sample_weight = torch.ones(B, device=device)
    w = sample_weight.to(device=device, dtype=img_logit.dtype)

    target = _t_image_label(batch, device, img_logit.dtype)  # (B,)
    wsum = w.sum().clamp(min=1e-6)
    bce = (
        F.binary_cross_entropy_with_logits(img_logit, target, reduction="none") * w
    ).sum() / wsum

    logdict = {
        "n": B,
        "img": float(bce.detach()),
        "n_pos": int((target > 0.5).sum().item()),
    }
    return bce, logdict


def head_t_metrics(pred: dict, batch: dict, mask: Tensor | None = None) -> dict[str, float]:
    """Head T 评估指标: 图级 t_f1 + t_acc(可选 t_auroc)。

    F1/acc 用 logit>0(即 prob>0.5)作为正预测阈值。
    ``mask`` (B,) bool 可选:v2.5 §2.4 T eval 在 fully-scanned 域(排除部分扫描增强样本);
    None → 全样本。
    """
    img_logit: Tensor = pred["img_logit"]  # (B,)
    device = img_logit.device
    target = _t_image_label(batch, device, torch.float32).bool()  # (B,)
    pred_pos = img_logit > 0
    if mask is not None:
        m = mask.to(device=device, dtype=torch.bool).reshape(-1)
        target = target[m]
        pred_pos = pred_pos[m]
        img_logit = img_logit[m]

    tp = float((pred_pos & target).sum().item())
    fp = float((pred_pos & ~target).sum().item())
    fn = float((~pred_pos & target).sum().item())
    tn = float((~pred_pos & ~target).sum().item())

    denom = 2 * tp + fp + fn
    t_f1 = (2 * tp / denom) if denom > 0 else 0.0
    total = tp + fp + fn + tn
    t_acc = ((tp + tn) / total) if total > 0 else 0.0

    out = {"t_f1": t_f1, "t_acc": t_acc}

    # 可选 AUROC(需两类都出现且 sklearn 可用;否则跳过)
    try:
        if target.any() and (~target).any():
            from sklearn.metrics import roc_auc_score

            probs = torch.sigmoid(img_logit).detach().cpu().numpy()
            y = target.detach().cpu().numpy()
            out["t_auroc"] = float(roc_auc_score(y, probs))
    except Exception:
        pass

    return out
