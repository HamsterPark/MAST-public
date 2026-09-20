"""LoRA 注入与 patch_embed 解冻 —— VIGIL DINO v2.2 PR-6.

实现取自 Handbook v2.2 §6.2;偏离(见 WORKLOG):
  D2.  logger 用标准库 logging。
  D15. ``peft`` 延迟到函数内 import,使 ``import vigil.backbone.lora_config``
       不强制要求 peft 已装。
"""
from __future__ import annotations

import logging

from torch import nn

logger = logging.getLogger(__name__)


# --- v2.4 PR-B: LoRA target 阶梯 (vendored from VIGIL for v2.5 ssl_sf09c1) -----
#
# 实测 timm `vit_small_patch16_dinov3.lvd1689m` 模块名:
#   blocks.{i}.attn.qkv     融合 QKV Linear(384 -> 1152)
#   blocks.{i}.attn.proj    attention output projection (o-proj)
#   blocks.{i}.mlp.fc1/fc2  MLP 两层 Linear
# HF DINOv2/v3(AutoModel)对应:attention 拆 query/key/value + 输出投影 dense。
#
# PR-B 阶梯(LoRA 部分到 mlp 为止):
#   'qkv'        只挂 QKV
#   'qkv_o'      QKV + output projection
#   'qkv_o_mlp'  QKV + o-proj + MLP 两层
# 更高的 'patch_embed' / 'ln' 档不走 LoRA(conv 适配器输入维与通道数绑定无法干净扩展,
# LayerNorm 无权重矩阵不适合低秩分解,两者直接全解冻 base 权重)。
LORA_TARGET_PRESETS: dict[str, dict[str, list[str]]] = {
    # timm 分支(融合 qkv)
    "timm": {
        "qkv": ["qkv"],
        "qkv_o": ["qkv", "attn.proj"],
        "qkv_o_mlp": ["qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    },
    # HF 分支(拆分 q/k/v + dense)
    "hf": {
        "qkv": ["query", "key", "value"],
        "qkv_o": ["query", "key", "value", "dense"],
        "qkv_o_mlp": ["query", "key", "value", "dense", "fc1", "fc2"],
    },
}


def resolve_lora_targets(preset: str, is_timm: bool) -> list[str]:
    """把 PR-B 阶梯 preset 名解析成 attach_lora 用的 target_modules 列表。

    Args:
        preset: 'qkv' / 'qkv_o' / 'qkv_o_mlp' 之一。'patch_embed' / 'ln' 档不走
            LoRA(见 LORA_TARGET_PRESETS 注释),但为方便 driver 把它们映射到最大
            LoRA 集合(到 mlp 为止),非 LoRA 的解冻由 unfreeze_* 另行处理。
        is_timm: True 走 timm 融合命名,False 走 HF 拆分命名。

    Returns:
        target_modules 字符串列表(peft LoraConfig 子串匹配)。
    """
    table = LORA_TARGET_PRESETS["timm" if is_timm else "hf"]
    if preset in table:
        return list(table[preset])
    if preset in ("patch_embed", "ln", "layernorm", "qkv_o_mlp_pe", "qkv_o_mlp_pe_ln"):
        return list(table["qkv_o_mlp"])
    raise ValueError(
        f"Unknown LoRA target preset '{preset}'. "
        f"Available: {list(table.keys())} (+ patch_embed/ln 档复用 qkv_o_mlp)"
    )


def attach_lora(
    backbone: nn.Module,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """把 LoRA 注入 attention QKV 投影,返回 PEFT 包裹后的 model。"""
    from peft import LoraConfig, get_peft_model  # D15: deferred import

    if target_modules is None:
        target_modules = ["query", "key", "value"]

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(backbone, lora_cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA attached: rank=%d alpha=%d target=%s trainable=%.2fM/%.2fM (%.2f%%)",
        rank, alpha, target_modules,
        n_train / 1e6, n_total / 1e6, 100.0 * n_train / max(1, n_total),
    )
    return model


def unfreeze_patch_embed(model: nn.Module) -> None:
    """解冻 patch_embed.proj/projection,供 Stage A->B 通道适配。"""
    from .dinov_loader import _find_patch_embed

    base = model.base_model if hasattr(model, "base_model") else model
    if hasattr(base, "model"):
        base = base.model  # peft 又包了一层
    pe = _find_patch_embed(base)
    if pe is None:
        logger.warning("Could not locate patch_embed to unfreeze")
        return
    conv = pe.projection if hasattr(pe, "projection") else pe.proj
    conv.weight.requires_grad_(True)
    if conv.bias is not None:
        conv.bias.requires_grad_(True)
    logger.info(
        "Unfroze patch_embed: %d params",
        sum(p.numel() for p in conv.parameters() if p.requires_grad),
    )
