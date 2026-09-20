"""v2.5 多头**构建**管理 (PR-D/G) —— 推理子集。

VENDORED into MAST from ``D:/…/VIGIL/src/vigil/training/multihead.py`` (v2.5).
Faithful copy of the head-**construction** + feature-routing surface, trimmed to
what the forward-only inference path needs; intra-package imports rewritten to
relative (``from vigil.heads.X import Y`` → ``from ..heads.X import Y``).

Kept (logic byte-identical to upstream):
  - ``resolve_tap_layers`` / ``tap_dim_mult`` / ``LAYER_TAP_CHOICES``
  - ``q_head_in_dim`` / ``q_head_input``
  - ``build_active_heads``
  - ``HeadLossCfg`` (``head_c_loss_fn`` kept as a plain field; inference passes None)
  - ``_patch_tokens`` / ``_last_layer_patch_tokens`` / ``_feats_for_t``
  - ``_make_q_binning`` / ``_Q_SOFT_FORMS``

Dropped (training-only, out of MAST inference closure — needs un-vendored
b/o/v/d heads, ``supervision``, ``eval.metrics``, ``v25_prereq``):
  - ``compute_active_losses`` / ``compute_active_metrics``
  - ``build_target_b`` / ``_q_self_consistency_target`` / ``_q_target_z``
  - ``parse_heads`` / ``_labels_to_device`` / ``_logf``

把 v2.3 的耦合头 {q, b, c} 推广到可配正交头集。本子集只覆盖 ssl_sf09c1 的
6 头 {q, c, t, n, s, k}(S0/shared/last-layer)。

输入分发:
  - CLS‖scale 头: q n s k   (吃 cls_with_scale, (B, embed_dim+scale_dim))
  - patch-token 头: c t      (吃 backbone feats dict 的 tokens/hw)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..heads.head_c_l1_dino import HeadCLevel1DINO
from ..heads.head_k import HeadK
from ..heads.head_n import HeadN
from ..heads.head_q_sharpness import HeadQSharpness
from ..heads.head_q_soft_ordinal import (
    HeadQSoftOrdinal,
    QBinning,
    q_readout,  # noqa: F401 (re-exported for inference callers)
)
from ..heads.head_s import HeadS
from ..heads.head_t import HeadT

# v2.5 §2.4: Head D 是 CLS 头(吃 cls_with_scale,与 Q 同输入)。本推理子集不含 b/o/v/d。
CLS_HEADS = {"q", "b", "n", "s", "k", "o", "v", "d"}
PATCH_HEADS = {"c", "t"}
ALL_HEADS = CLS_HEADS | PATCH_HEADS


# v2.5 PR ARCH (§4.4) 层接出 layer-tapping:dense 头(C/T)从可选层接出,CLS 头用 last。
# 映射到 wrapper.forward 的 tap_layers(vits16/vitb16/vitl16: 见 num_layers)。
#   last  → None      (=现行为, 只用最后层, dense 头维度 = embed_dim)
#   l6    → [n//2-1]   (中间偏前层, 单层, 维度 = embed_dim)
#   l9    → [3*n//4-1] (中间偏后层, 单层, 维度 = embed_dim)
#   last4 → 末 4 层    (concat, 维度 = 4*embed_dim, **wrapper 不投影**, C/T 头吃 4D)
LAYER_TAP_CHOICES = ("last", "l6", "l9", "last4")


def resolve_tap_layers(layer_tap: str, num_layers: int) -> list[int] | None:
    """layer-tap 名 → wrapper.forward 的 tap_layers(0-based block idx)。

    返回 None 表示走最后层(默认, 与历史等价, 不增加前向开销)。
    vits16/vitb16 num_layers=12: l6=[5], l9=[8], last4=[8,9,10,11]。
    vitl16 num_layers=24: l6=[11], l9=[17], last4=[20,21,22,23]。
    """
    if layer_tap == "last":
        return None
    if layer_tap == "l6":
        return [max(0, num_layers // 2 - 1)]
    if layer_tap == "l9":
        return [max(0, (3 * num_layers) // 4 - 1)]
    if layer_tap == "last4":
        return list(range(max(0, num_layers - 4), num_layers))
    raise ValueError(f"unknown layer_tap {layer_tap!r}; valid={LAYER_TAP_CHOICES}")


def tap_dim_mult(layer_tap: str) -> int:
    """dense 头吃的 token 维度相对 embed_dim 的倍数(last4 concat=4, 其余=1)。"""
    return 4 if layer_tap == "last4" else 1


_Q_SOFT_FORMS = ("soft_ordinal", "corn_hard", "flat_cls", "uniform10")


def _make_q_binning(form: str, bins: int, sigma: float) -> QBinning:
    """按 (form, bins) 选 QBinning;sigma 注入。uniform10 form 强制等宽10箱(忽略 bins)。"""
    if form == "uniform10":
        b = QBinning.uniform10()
    elif bins == 4:
        b = QBinning.coarse4()
    elif bins == 6:
        b = QBinning.fine6()
    elif bins == 10:
        b = QBinning.uniform10()
    else:  # 5 (默认) 或其它 → 锁定 5 非均匀箱
        b = QBinning()
    b.sigma = float(sigma)
    return b


def q_head_in_dim(head_q_input: str, embed_dim: int, scale_dim: int) -> int:
    """Head Q 的实际输入维(rev3 §1.5(5) cls_patch pooled-patch 输入)。

    head_q_input=='cls'(默认/anchor): in_dim = embed_dim + scale_dim(CLS‖scale,历史行为)。
    head_q_input=='cls_patch': in_dim = embed_dim + scale_dim + embed_dim
        (CLS‖scale‖mean-pooled-patch;后接的 mean-pool patch token 维 = embed_dim,
        与 dense 头消费的 patch token 同源,见 ``q_head_input`` / ``_patch_tokens``)。
    **仅 Q 头按此加宽;其它 CLS 头(n/s/k/d/...)恒用 embed_dim+scale_dim,不受影响。**
    """
    base = embed_dim + scale_dim
    return base + embed_dim if head_q_input == "cls_patch" else base


def q_head_input(
    head_q_input: str, cls_with_scale: torch.Tensor, feats: dict,
) -> torch.Tensor:
    """构造 Head Q 的前向输入张量(rev3 §1.5(5))。

    head_q_input=='cls'(默认/anchor): 原样返回 ``cls_with_scale``(B, embed_dim+scale_dim)
        —— **逐字节等价历史行为**(回归安全)。
    head_q_input=='cls_patch': 返回 cat([cls_with_scale, pooled_patch], dim=-1),其中
        pooled_patch = patch token 在 (空间)token 维上的均值 → (B, embed_dim)。patch token
        取自 ``_last_layer_patch_tokens(feats)``(已剥 register,与 C/T dense 头同源)。
    """
    if head_q_input != "cls_patch":
        return cls_with_scale
    tok, _hw = _last_layer_patch_tokens(feats)    # (B, N, embed_dim) register 已剥
    pooled = tok.mean(dim=1)                        # (B, embed_dim) 空间均值 = mean-pooled patch
    return torch.cat([cls_with_scale, pooled], dim=-1)  # (B, embed+scale+embed)


def build_active_heads(
    names: list[str], embed_dim: int, scale_dim: int, num_seg: int,
    head_n_mode: str = "binary",
    head_q_form: str = "soft_ordinal",
    head_q_bins: int = 5,
    head_q_sord_sigma: float = 8.0,
    layer_tap: str = "last",
    head_q_input: str = "cls",
) -> nn.ModuleDict:
    """按 names 构建头 ModuleDict。CLS 头 in_dim=embed_dim+scale_dim; patch 头吃 tokens。

    Head Q (v2.5 PR Q):head_q_form ∈ {soft_ordinal,corn_hard,flat_cls,uniform10} → HeadQSoftOrdinal
    (in_dim=实际 embed_dim+scale_dim,**不写死**;binning 按 head_q_bins/form 选);
    head_q_form=='reg_huber' → 保留旧 HeadQSharpness 作纯回归对照。

    v2.5 PR Q §1.5(5) cls_patch:head_q_input=='cls_patch' 时**仅 Q 头**加宽 in_dim 到
    embed_dim+scale_dim+embed_dim(CLS‖scale‖mean-pooled-patch,见 ``q_head_in_dim``);
    其它 CLS 头不变。前向时由 ``q_head_input`` 拼接 pooled-patch,与此 in_dim 一致。

    v2.5 PR ARCH (§4.4):dense 头(C/T)吃层接出 token,其输入维 = embed_dim*tap_dim_mult
    (last4 concat → 4*embed_dim,wrapper 不投影;HeadCLevel1DINO 的 1×1 conv reduce 与
    HeadT 的 token_dim 直接吃该维度)。CLS 头不受 layer_tap 影响(恒 last)。

    NOTE (MAST inference subset): 仅实现 ssl_sf09c1 用到的 6 头 {q, c, t, n, s, k};
    上游的 b/o/v/d 头不在本推理 closure(碰到这些名字会被静默跳过 → ModuleDict 不含该键,
    与 ckpt head_names 对齐时只会出现 {q,c,t,n,s,k},不触发跳过)。
    """
    in_cls = embed_dim + scale_dim
    in_q = q_head_in_dim(head_q_input, embed_dim, scale_dim)  # 仅 Q 头可能加宽(cls_patch)
    dense_dim = embed_dim * tap_dim_mult(layer_tap)  # C/T 头的 token 输入维
    heads = nn.ModuleDict()
    for n in names:
        if n == "q":
            if head_q_form in _Q_SOFT_FORMS:
                binning = _make_q_binning(head_q_form, head_q_bins, head_q_sord_sigma)
                heads[n] = HeadQSoftOrdinal(in_dim=in_q, binning=binning, form=head_q_form)
            else:  # reg_huber 纯回归对照
                heads[n] = HeadQSharpness(in_dim=in_q)
        elif n == "c":
            heads[n] = HeadCLevel1DINO(embed_dim=dense_dim, num_classes=num_seg)
        elif n == "n":
            heads[n] = HeadN(in_dim=in_cls, mode=head_n_mode)
        elif n == "s":
            heads[n] = HeadS(in_dim=in_cls)
        elif n == "k":
            heads[n] = HeadK(in_dim=in_cls)
        elif n == "t":
            heads[n] = HeadT(token_dim=dense_dim)
        # b/o/v/d: out of MAST inference closure (see docstring) — skipped.
    return heads


@dataclass
class HeadLossCfg:
    """推理期沿用的配置载体(只读取若干字段供 forward 路由,不在 MAST 内算 loss)。

    NOTE: ``head_c_loss_fn`` 在上游由 ``get_head_c_loss_fn(...)`` 填充;MAST 的该函数是
    training-only raising stub,故 **inference 一律传 None**(forward 不需要 C loss)。
    """

    head_c_loss_fn: object = None        # 上游 = get_head_c_loss_fn(...) 的返回;MAST 推理传 None
    supervision_criteria: str = "M0M1_AND_single_AND_stable"
    class_weights_method: str = "cui_2019_effective_number"
    num_seg: int = 4
    morph_hq_threshold: float = 0.3398   # 上游默认 MORPH_HQ_THRESHOLD_DEFAULT(W1-a top-60%)
    head_b_focal_gamma: float = 2.0
    head_q_pinball_tau: float = 0.5
    head_q_unstable_weight: float = 1.0
    # v2.5 PR Q: soft-ordinal Head Q
    head_q_form: str = "soft_ordinal"    # anchor; ∈ {soft_ordinal,corn_hard,flat_cls,uniform10,reg_huber}
    head_q_bins: int = 5                 # 5→QBinning(); 4→coarse4; 6→fine6; 10→uniform10
    head_q_sord_sigma: float = 8.0       # SORD 软标签宽度 (score 单位)
    head_q_input: str = "cls"            # cls | cls_patch
    head_q_lambda_rank: float = 0.2
    head_q_lambda_var: float = 0.0
    head_q_target_abs: bool = False


def _patch_tokens(feats: dict):
    """dense 头(C/T)用的 patch token。

    v2.5 PR ARCH:优先用层接出 ``tapped_tokens``(layer_tap≠last 时为中间层 / last4
    concat, 维度可能 = k*embed_dim);无该 key 时回退最后层 ``tokens``。register 剥离
    对 tapped 同样处理(取末尾 h_p*w_p 个)。
    """
    tok = feats.get("tapped_tokens")
    if tok is None:
        tok = feats["tokens"]
    hw = feats["hw"]
    n = hw[0] * hw[1]
    if tok.shape[1] > n:  # timm DINOv3 含 register tokens 在 patch 前
        tok = tok[:, tok.shape[1] - n:]
    return tok, hw


def _last_layer_patch_tokens(feats: dict):
    """全局头(Q cls_patch)用的 patch token = **最后层** tokens(embed_dim),register 已剥。
    **永不用 tapped_tokens**(那是 dense C/T 头的多层 concat, k*embed_dim);否则 cls_patch +
    layer_tap=last4 时 Q 头 in_dim 不匹配崩溃(q_head_in_dim 按 embed_dim 算)。Q 是全局头,
    其 pooled-patch 应固定来自最后层(=q_form winner 的定义),不随 dense 头 layer-tap 变。
    tap=last 时 feats 无 tapped_tokens → 与 _patch_tokens 逐字节等价(回归安全)。"""
    tok = feats["tokens"]
    hw = feats["hw"]
    n = hw[0] * hw[1]
    if tok.shape[1] > n:  # register 剥离(同 _patch_tokens)
        tok = tok[:, tok.shape[1] - n:]
    return tok, hw


def _feats_for_t(feats: dict) -> dict:
    """给 Head T 传 register 已剥离的 feats (N == h_p*w_p), tokens = 层接出 token。"""
    tok, hw = _patch_tokens(feats)
    return {"tokens": tok, "hw": hw, "cls": feats.get("cls")}
