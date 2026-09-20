"""Head Q v2.5: soft-ordinal-regression(rev3 §1.5,头号).

把 v2.4 的纯标量 Huber 回归(`head_q_sharpness.HeadQSharpness`)换成**非均匀效用箱上的
软序数分类 + 期望读出连续分 + pairwise ranking**。一次满足:大类准(usable-vs-unusable)、
60-90 带内准(期望读出)、58/62 与 0→40 便宜(SORD 软标签 + 序结构)、resolution 分配正确
(非均匀箱)、sim-to-real 稳(ranking)、并附带不确定度(期望分布方差/熵 → MAST gating / Head D)。

**形式(form,§1.5(5) 5 臂可切换,同一模块同口径,满足铁律 §7-5)**:
  - ``soft_ordinal``  【anchor】: K logits → softmax → SORD 软标签 CE + 期望读出 + ranking
  - ``corn_hard``    : CORN K-1 条件 logit(秩一致)→ 硬累积读出(无期望)+ ranking
  - ``flat_cls``     : K logits → 硬 one-hot CE(无序结构,纯粗分类对照)
  - ``uniform10``    : 同 soft_ordinal 但 10 个**等宽**箱(预期输,用数字证)
  (``reg_huber`` 纯回归对照仍走旧 ``head_q_sharpness.head_q_loss``,不在本模块)

**分数空间**:头在 score∈[0,100] 上工作(越锐分越高)。driver/multihead 先用 ``v25_prereq.z_to_score``
把 ``z=log10(R/scan)`` 映成 score_true 再喂本模块的 loss。箱切点(edges)与箱中心(centers)由
``QBinning`` 持有,**结构锁定=非均匀 5 箱**;实际 z 切点由 W1-a(``v25_prereq``)据分布定后注入。

标签来源 / 输入 / 监督域不变(见 §1/§2):输入 ``cls_with_scale``(CLS‖64-d ScaleEmbedding);
监督域 single∧stable∧clean;sample_weight 支持 Q1 软门控。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .head_q_sharpness import ScaleEmbedding  # 复用,保持 cls_with_scale 契约一致

# 默认非均匀 5 箱(score 空间;结构锁定 rev3 §1.5(2)):[<60][60-70][70-80][80-90][90+]
# edges = 箱之间的 score 切点;centers = 每箱代表分(用于期望读出)。
# <60 一粗箱 center=40(0→40 不在乎);60-90 细 3 箱;90+ 顶箱。
_DEFAULT_EDGES = [60.0, 70.0, 80.0, 90.0]
# centers 仅供期望读出(Σ p·center);软标签用 CDF-over-edges 不依赖中心。
# <60 粗箱读出中心取 45(代表不可用区,<60 → usable 决策正确;blunt 读出可偏低)。
_DEFAULT_CENTERS = [45.0, 65.0, 75.0, 85.0, 95.0]


@dataclass
class QBinning:
    """非均匀效用箱配置(score 空间)。

    edges: 升序箱切点(len = n_bins-1)。centers: 每箱代表分(len = n_bins)。
    sigma: SORD 软标签宽度(score 单位;σ 大→近邻更容错,σ 小→近 one-hot,§1.5(5) 轻消融旋钮)。
    usable_cut: usable/unusable 硬决策边界(~60,§9 硬门)。
    """

    edges: list[float] = field(default_factory=lambda: list(_DEFAULT_EDGES))
    centers: list[float] = field(default_factory=lambda: list(_DEFAULT_CENTERS))
    sigma: float = 8.0
    usable_cut: float = 60.0

    def __post_init__(self) -> None:
        assert len(self.centers) == len(self.edges) + 1, "centers 须比 edges 多 1"
        assert all(self.edges[i] < self.edges[i + 1] for i in range(len(self.edges) - 1)), "edges 须升序"

    @property
    def n_bins(self) -> int:
        return len(self.centers)

    def centers_t(self, device, dtype=torch.float32) -> Tensor:
        return torch.tensor(self.centers, device=device, dtype=dtype)

    def edges_full_t(self, device, dtype=torch.float32, big: float = 1e4) -> Tensor:
        """完整箱边界 (K+1,) = [-big, *edges, +big],供 CDF 软直方图软标签用。"""
        return torch.tensor([-big, *self.edges, big], device=device, dtype=dtype)

    def assign(self, score: Tensor) -> Tensor:
        """score (B,) → 箱索引 (B,) long。score < edges[0] → 0;>= edges[-1] → n_bins-1。"""
        e = torch.as_tensor(self.edges, device=score.device, dtype=score.dtype)
        return torch.bucketize(score, e, right=False)

    @classmethod
    def uniform10(cls) -> "QBinning":
        """等宽十分类挑战者(§1.5(5) 臂 E,预期输):0-10,…,90-100。"""
        edges = [10.0 * i for i in range(1, 10)]
        centers = [10.0 * i + 5.0 for i in range(10)]
        return cls(edges=edges, centers=centers, sigma=8.0, usable_cut=60.0)

    @classmethod
    def fine6(cls) -> "QBinning":
        """6 箱挑战者(§1.5(5)):<60 拆 0-30/30-60 + 60-70/70-80/80-90/90+。"""
        return cls(
            edges=[30.0, 60.0, 70.0, 80.0, 90.0],
            centers=[15.0, 45.0, 65.0, 75.0, 85.0, 95.0],
            sigma=8.0, usable_cut=60.0,
        )

    @classmethod
    def coarse4(cls) -> "QBinning":
        """4 箱挑战者(§1.5(5)):<60/60-75/75-90/90+。"""
        return cls(edges=[60.0, 75.0, 90.0], centers=[40.0, 67.5, 82.5, 95.0],
                   sigma=8.0, usable_cut=60.0)


_FORMS = ("soft_ordinal", "corn_hard", "flat_cls", "uniform10")


class HeadQSoftOrdinal(nn.Module):
    """从 cls_with_scale 预测 score 的序数分布。

    form='corn_hard' 时输出 K-1 个 CORN 条件 logit;其余输出 K 个 logit。
    """

    def __init__(
        self,
        in_dim: int = 448,
        binning: QBinning | None = None,
        form: str = "soft_ordinal",
        hidden: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert form in _FORMS, f"form 须 ∈ {_FORMS}"
        self.binning = binning or QBinning()
        self.form = form
        K = self.binning.n_bins
        out_dim = (K - 1) if form == "corn_hard" else K
        if hidden is None:
            hidden = [256, 64]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cls_with_scale: Tensor) -> Tensor:
        """返回原始 logits:(B, K) 或 corn 的 (B, K-1)。"""
        return self.mlp(cls_with_scale)


# ---------------- 软标签 / 读出 / 损失 ----------------

def sord_soft_label(score_true: Tensor, edges_full: Tensor, sigma: float) -> Tensor:
    """SORD 软序数标签 —— **高斯 CDF 软直方图**(对非均匀箱宽正确,对齐箱边界)。

    `t_k = Φ((edge_{k+1} - s)/σ) - Φ((edge_k - s)/σ)`,即"以 score_true 为中心、σ 为宽的高斯
    落在第 k 箱(score 区间)内的概率质量"。比"到箱中心的距离"更正确:① 宽粗箱(<60)按其
    **真实区间宽度**自动吃到该吃的质量,不会因中心偏低把 score=55(不可用)误判到 60-70 可用箱;
    ② 软目标天然跨 ~60 边界(59↔61 软),不罚边界歧义;③ 0→40 都落 <60 箱,代价小。
    σ 是轻消融旋钮(§1.5(5)):σ 大→更跨箱容错,σ 小→近 one-hot。

    score_true (B,), edges_full (K+1,)=[-big,*edges,+big] → (B,K) 归一化软分布。
    """
    s = score_true.unsqueeze(1)                                   # (B,1)
    e = edges_full.unsqueeze(0)                                   # (1,K+1)
    cdf = 0.5 * (1.0 + torch.erf((e - s) / (max(sigma, 1e-6) * math.sqrt(2.0))))  # (B,K+1)
    t = (cdf[:, 1:] - cdf[:, :-1]).clamp(min=1e-8)                # (B,K) 每箱质量
    return t / t.sum(dim=1, keepdim=True)


def q_readout(logits: Tensor, binning: QBinning, form: str) -> tuple[Tensor, Tensor, Tensor]:
    """从 logits 算 (score_hat, p_bins, uncertainty)。

    soft_ordinal/flat_cls/uniform10: p=softmax(logits), score_hat=Σ p·center(期望读出)。
    corn_hard: 条件 sigmoid 链 → 累积 P(y>k) → 硬 rank=Σ[P>0.5] → score_hat=center[rank](无期望)。
    uncertainty = 分布的方差(score 空间);corn_hard 用 1-max(P) 近似。
    """
    centers = binning.centers_t(logits.device, logits.dtype)  # (K,)
    if form == "corn_hard":
        cond = torch.sigmoid(logits)                        # (B, K-1) P(y>k | y>k-1)
        cum = torch.cumprod(cond, dim=1)                    # (B, K-1) P(y>k)
        rank = (cum > 0.5).sum(dim=1)                       # (B,) ∈ [0, K-1]
        score_hat = centers[rank]
        # 软 bin 概率(仅用于不确定度):p_k = P(y>k-1) - P(y>k)
        ones = torch.ones(logits.shape[0], 1, device=logits.device, dtype=logits.dtype)
        p_gt = torch.cat([ones, cum], dim=1)                # P(y>-1)=1 ... P(y>K-2)
        p_bins = torch.cat([p_gt[:, :-1] - p_gt[:, 1:], p_gt[:, -1:]], dim=1)  # (B,K)
        p_bins = p_bins.clamp(min=0)
        p_bins = p_bins / p_bins.sum(dim=1, keepdim=True).clamp(min=1e-6)
        unc = 1.0 - p_bins.max(dim=1).values
        return score_hat, p_bins, unc
    p = torch.softmax(logits, dim=1)                         # (B,K)
    score_hat = (p * centers.unsqueeze(0)).sum(dim=1)        # 期望读出
    var = (p * (centers.unsqueeze(0) - score_hat.unsqueeze(1)) ** 2).sum(dim=1)
    return score_hat, p, var


def _pairwise_rank_loss(pred_score: Tensor, target_score: Tensor, margin: float, w: Tensor) -> Tensor:
    """pairwise ranking(carry v2.4):谁更锐谁分高。margin 在 score 单位。"""
    diff_t = target_score.unsqueeze(1) - target_score.unsqueeze(0)
    diff_p = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
    sign = torch.sign(diff_t)
    L = F.relu(margin - sign * diff_p)
    pair_w = torch.min(w.unsqueeze(1), w.unsqueeze(0))
    valid = (sign != 0).float() * pair_w
    return (L * valid).sum() / valid.sum().clamp(min=1.0)


def head_q_soft_ordinal_loss(
    logits: Tensor,
    score_true: Tensor,
    binning: QBinning,
    form: str = "soft_ordinal",
    lambda_rank: float = 0.2,
    rank_margin: float = 2.0,
    lambda_var: float = 0.0,
    sample_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    """复合损失,主次分明(rev3 §1.5(3))。

    主项:
      soft_ordinal/uniform10: SORD 软标签 CE(-Σ t_k log p_k)。
      corn_hard: CORN 条件 BCE 链(秩一致;无期望)。
      flat_cls: 硬 one-hot CE(无序结构)。
    + pairwise ranking(保留,作用在 score_hat;corn 作用在硬 rank-score)。
    + 可选 mean-variance 正则(lambda_var>0:收紧期望分布)。
    sample_weight: (B,) Q1 软门控;None→全 1。
    """
    if sample_weight is None:
        sample_weight = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
    w = sample_weight
    wsum = w.sum().clamp(min=1e-6)
    centers = binning.centers_t(logits.device, logits.dtype)

    if form in ("soft_ordinal", "uniform10"):
        p_log = torch.log_softmax(logits, dim=1)
        edges_full = binning.edges_full_t(logits.device, logits.dtype)
        t = sord_soft_label(score_true, edges_full, binning.sigma)  # (B,K) CDF 软直方图
        main = (-(t * p_log).sum(dim=1) * w).sum() / wsum
    elif form == "corn_hard":
        # CORN: 第 k 个二分类目标 = [score_true 落在 > 第 k 个 edge](即真 rank > k)
        rank_true = binning.assign(score_true)                    # (B,) ∈ [0,K-1]
        K = binning.n_bins
        # target_k = 1 if rank_true > k else 0, k=0..K-2
        ks = torch.arange(K - 1, device=logits.device)
        tgt = (rank_true.unsqueeze(1) > ks.unsqueeze(0)).float()  # (B, K-1)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").mean(dim=1)
        main = (bce * w).sum() / wsum
    elif form == "flat_cls":
        rank_true = binning.assign(score_true)
        ce = F.cross_entropy(logits, rank_true, reduction="none")
        main = (ce * w).sum() / wsum
    else:
        raise ValueError(form)

    score_hat, p_bins, unc = q_readout(logits.detach(), binning, form)  # detach: rank 项不回灌主分类梯度路径之外
    # ranking 用可微 score_hat(对 soft/flat 重算可微期望;corn 用硬 score)
    if form in ("soft_ordinal", "uniform10", "flat_cls"):
        p = torch.softmax(logits, dim=1)
        score_hat_diff = (p * centers.unsqueeze(0)).sum(dim=1)
    else:
        score_hat_diff = score_hat  # corn 硬读出不可微 → ranking 退化为常数(仍报)
    L_rank = _pairwise_rank_loss(score_hat_diff, score_true, rank_margin, w)

    total = main + lambda_rank * L_rank
    out = {"total": total, "main": main.detach(), "rank": L_rank.detach()}

    if lambda_var > 0 and form in ("soft_ordinal", "uniform10"):
        # mean-variance 式:期望贴标签 + 方差惩罚(收紧分布)
        p = torch.softmax(logits, dim=1)
        sh = (p * centers.unsqueeze(0)).sum(dim=1)
        var = (p * (centers.unsqueeze(0) - sh.unsqueeze(1)) ** 2).sum(dim=1)
        L_mv = ((sh - score_true) ** 2 * w).sum() / wsum + (var * w).sum() / wsum
        total = total + lambda_var * L_mv
        out["total"] = total
        out["mv"] = L_mv.detach()
    return out


# ---------------- metrics(rev3 §9 Q 验收) ----------------

def head_q_soft_ordinal_metrics(
    logits: Tensor, score_true: Tensor, binning: QBinning, form: str = "soft_ordinal",
) -> dict[str, float]:
    """Q 代理指标:usable-vs-unusable F1(硬门)+ within-usable macro-F1 + 60-90 带内 MAE
    + ~60 双向混淆 + Spearman/Kendall τ + blunt 尾 MAE(报不设门)。"""
    from scipy.stats import kendalltau, spearmanr

    score_hat, _, _ = q_readout(logits.detach(), binning, form)
    sh = score_hat.detach().cpu().numpy()
    st = score_true.detach().cpu().numpy()
    cut = binning.usable_cut

    pred_usable = sh >= cut
    true_usable = st >= cut

    def _f1(pred_pos, true_pos):
        tp = float((pred_pos & true_pos).sum())
        fp = float((pred_pos & ~true_pos).sum())
        fn = float((~pred_pos & true_pos).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    out: dict[str, float] = {}
    out["q_usable_f1"] = _f1(pred_usable, true_usable)  # 硬门
    # ~60 双向混淆率
    n_good = float(true_usable.sum())
    n_unus = float((~true_usable).sum())
    out["q_good_to_unusable"] = float((true_usable & ~pred_usable).sum()) / n_good if n_good > 0 else 0.0  # 毁好针(贵)
    out["q_unusable_to_good"] = float((~true_usable & pred_usable).sum()) / n_unus if n_unus > 0 else 0.0  # 浪费扫描
    # 60-90 带内 MAE(对期望读出,真值落 60-90 的样本)
    band = (st >= 60.0) & (st <= 90.0)
    out["q_band_mae"] = float(abs(sh[band] - st[band]).mean()) if band.any() else 0.0
    # blunt 尾 MAE(真值<60,报不设门)
    blunt = st < 60.0
    out["q_blunt_mae"] = float(abs(sh[blunt] - st[blunt]).mean()) if blunt.any() else 0.0
    # 全程序质量
    if len(sh) > 2:
        rho, _ = spearmanr(sh, st)
        tau, _ = kendalltau(sh, st)
        out["q_spearman"] = float(rho) if rho is not None else 0.0
        out["q_kendall"] = float(tau) if tau is not None else 0.0
    # within-usable macro-F1(真值≥60 的样本,按箱分类)
    import numpy as np
    if true_usable.sum() > 0:
        edges = np.array(binning.edges)
        rank_true = np.digitize(st, edges, right=False)
        rank_pred = np.digitize(sh, edges, right=False)
        # 只在 usable 箱(rank 对应 score>=usable_cut 的箱)上算
        usable_bin0 = int(np.searchsorted(edges, cut, side="left"))  # 第一个 usable 箱索引
        f1s = []
        for b in range(usable_bin0, binning.n_bins):
            tp_b = (rank_pred == b) & (rank_true == b)
            f1s.append(_f1(rank_pred == b, rank_true == b) if (rank_true == b).sum() > 0 else None)
        f1s = [x for x in f1s if x is not None]
        out["q_within_usable_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    out["q_mean_score_hat"] = float(sh.mean())
    return out
