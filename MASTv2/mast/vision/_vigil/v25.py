"""VIGIL v2.5 ``ssl_sf09c1`` — DINOv3-vits16 + LoRA(qkv_o_mlp, r8) + 6 heads.

The current production STM-vision model (supersedes M12). A single frozen
DINOv3-ViT-S/16 timm backbone + a shared LoRA adapter, with 6 analysis heads:

  q  quality   — soft-ordinal continuous score (~60-90, ↑ better)
  c  segment   — 4-class terrace/step/defect/contam
  t  instability— P(tip unstable during scan) (↑ worse)
  n  multi-apex — P(apex>=2) (↑ worse; the STRONGEST single quality signal)
  s  geometry  — apex axis_ratio∈(0,1] (1=round) + asym_logit
  k  contam    — P(tip contamination) (↑ worse)

The ckpt (``ssl_sf09c1_s42/ckpt_final.pt``, ~29 MB) carries ONLY the LoRA + head
+ scale-embedding weights — the frozen ``dinov3-vits16`` backbone is loaded from
the bundled timm/HF cache (``MASTv2/artifacts/vision_backbone``) so inference is
fully offline. This module is the verified load+forward path (proven by
``tools/_smoke_vigil_v25.py``), packaged behind a small ``load_model`` /
``infer`` API that mirrors :mod:`mast.vision._vigil.m12` so the adapter
(:mod:`mast.vision.vigil_backend`) swaps backends with minimal change.

Integration source of truth: ``VIGIL/vigil_v2.5/reports/infer_sf09_grouped.py``
+ ``INTEGRATION_GUIDE_ssl_sf09c1.md`` (bb_share=S0, lora_scope=shared, E3 input,
soft-ordinal Q, ScaleEmbedding conditions every head on the physical scan nm).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SCALE_DIM = 64
sig = lambda x: __import__("torch").sigmoid(x)  # noqa: E731 (lazy torch)


def configure_backbone_cache(cache_dir: str | Path | None) -> Path | None:
    """Force timm/HF offline and locate the bundled cache. Returns its root.

    Thin delegate to :mod:`mast.vision._vigil.backbone_cache`. **The return
    value is the part that matters**: callers thread it into
    ``timm_pretrained_kwargs`` so timm loads the weight FILE directly.

    Pinning ``HF_HOME`` — all this function used to do — CANNOT work on its
    own: ``huggingface_hub`` freezes its cache path into a module-level
    constant at import time, so by the time we set the variable it is already
    too late. That is why v5.5.0 and v5.5.1 both failed to fix the field issue
    even though their log line showed the cache being found (2026-07-27).
    """
    from mast.vision._vigil.backbone_cache import configure as _configure
    return _configure(cache_dir)


@dataclass
class V25Model:
    """A loaded ssl_sf09c1 model + everything ``infer`` needs."""

    wrapper: Any
    scale_emb: Any
    heads: Any
    cfg: Any
    tap_layers: Any
    head_names: list[str]
    proc: Any
    mode: str
    bb_share: str
    backbone_name: str
    lora_rank: int
    embed_dim: int
    device: str
    _q_soft_forms: Any = field(default=None)


def load_model(
    checkpoint_path: str | Path,
    device: str | None = None,
    backbone_cache_dir: str | Path | None = None,
) -> V25Model:
    """Build dinov3-vits16 + LoRA + 6 heads and load the ssl_sf09c1 ckpt.

    Mirrors :func:`mast.vision._vigil.m12.load_model`. Raises ImportError if
    torch/timm are missing; the backend catches that and falls back to Mock."""
    configure_backbone_cache(backbone_cache_dir)

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from mast.vision._vigil.backbone.dinov_loader import get_backbone_info, load_backbone
    from mast.vision._vigil.backbone.lora_config import attach_lora, resolve_lora_targets
    from mast.vision._vigil.data.stm_image_processor import STMImageProcessor
    from mast.vision._vigil.heads.head_q_sharpness import ScaleEmbedding
    from mast.vision._vigil.training.multihead import (
        HeadLossCfg,
        _Q_SOFT_FORMS,
        build_active_heads,
        resolve_tap_layers,
    )

    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    a = ck["args"]
    head_names = list(ck["head_names"])
    backbone_name = ck.get("backbone_name", a.get("backbone", "dinov3-vits16"))
    lora_rank = int(ck.get("lora_rank", a.get("lora_rank", 8)))

    info = get_backbone_info(backbone_name)
    embed_dim = int(ck.get("embed_dim", info["embed_dim"]))
    num_layers = int(info.get("num_layers", 12))
    bb_share = a.get("bb_share", "S0")
    if bb_share != "S0":
        raise NotImplementedError(
            f"v25 backend only supports bb_share=S0 (got {bb_share!r}); "
            "the ssl_sf09c1 production config is S0/shared."
        )

    # backbone (frozen timm dinov3-vits16) + LoRA(qkv_o_mlp, shared)
    base_preset = a.get("lora_target", "qkv").replace("_pe_ln", "").replace("_pe", "")
    wrapper = load_backbone(backbone_name, in_channels=3, freeze=True)
    is_timm = hasattr(wrapper.backbone, "forward_features") and not hasattr(
        wrapper.backbone, "config"
    )
    targets = resolve_lora_targets(base_preset, is_timm)
    wrapper.backbone = attach_lora(
        wrapper.backbone, rank=lora_rank, alpha=2 * lora_rank, target_modules=targets
    )
    wrapper.to(device)

    # LoRA + head + scale-emb weights (strict=False: ckpt has no frozen base)
    _missing, unexpected = wrapper.backbone.load_state_dict(ck["lora_state"], strict=False)
    n_lora_unexpected = sum(1 for k in unexpected if "lora_" in k)
    if n_lora_unexpected:
        raise RuntimeError(
            f"v25 LoRA adapter mismatch: {n_lora_unexpected} unexpected lora keys "
            "(backbone/lora_target/rank does not match the ckpt)"
        )

    scale_emb = ScaleEmbedding(dim=SCALE_DIM).to(device)
    scale_emb.load_state_dict(ck["scale_emb_state"])
    scale_emb.eval()

    layer_tap = a.get("layer_tap", "last")
    tap_layers = resolve_tap_layers(layer_tap, num_layers)
    num_seg = a.get("num_seg")
    num_seg = 4 if num_seg is None else int(num_seg)
    heads = build_active_heads(
        head_names, embed_dim, SCALE_DIM, num_seg=num_seg,
        head_n_mode=a.get("head_n_mode", "binary"),
        head_q_form=a.get("head_q_form", "soft_ordinal"),
        head_q_bins=int(a.get("head_q_bins", 5)),
        head_q_sord_sigma=float(a.get("head_q_sord_sigma", 8.0)),
        layer_tap=layer_tap,
        head_q_input=a.get("head_q_input", "cls"),
    ).to(device)
    for nm in head_names:
        if nm in ck.get("heads_state", {}):
            heads[nm].load_state_dict(ck["heads_state"][nm])
        heads[nm].eval()

    cfg = HeadLossCfg(
        head_c_loss_fn=None,  # inference: the C-loss is training-only
        head_q_form=a.get("head_q_form", "soft_ordinal"),
        head_q_bins=int(a.get("head_q_bins", 5)),
        head_q_sord_sigma=float(a.get("head_q_sord_sigma", 8.0)),
        head_q_input=a.get("head_q_input", "cls"),
    )

    proc = STMImageProcessor(
        target_size=int(a.get("size", 512)),
        do_resize=not bool(a.get("native_res", False)),
    )
    proc.flatten = bool(a.get("flatten", False))
    wrapper.eval()

    return V25Model(
        wrapper=wrapper, scale_emb=scale_emb, heads=heads, cfg=cfg,
        tap_layers=tap_layers, head_names=head_names, proc=proc,
        mode=a.get("input_mode", "E3"), bb_share=bb_share,
        backbone_name=backbone_name, lora_rank=lora_rank, embed_dim=embed_dim,
        device=device, _q_soft_forms=_Q_SOFT_FORMS,
    )


def _forward(model: V25Model, fwd_pm: np.ndarray, bwd_pm: np.ndarray, scan_size_nm: float):
    """Run the backbone once; return (feats, cls_with_scale)."""
    import torch

    from mast.vision._vigil.forward import build_batch_input, run_backbone

    fwd = np.asarray(fwd_pm, dtype=np.float32)
    bwd = np.asarray(bwd_pm, dtype=np.float32)
    gb = torch.from_numpy(np.stack([fwd, bwd])[0:1])  # (1,H,W) fwd (proc handles pair)
    batch = {
        "fwd_pm": torch.from_numpy(fwd[None]),
        "bwd_pm": torch.from_numpy(bwd[None]),
    }
    images = build_batch_input(batch, model.proc, model.device, model.mode)
    scan = torch.full((1,), float(scan_size_nm), device=model.device)
    feats, cls = run_backbone(
        model.wrapper, images, model.tap_layers, model.scale_emb, scan,
        per_head_lora=False, bb_share=model.bb_share,
    )
    return feats, cls, images, gb


def infer(model: V25Model, fwd_pm: np.ndarray, bwd_pm: np.ndarray, scan_size_nm: float) -> dict:
    """Run all 6 heads on a single (fwd, bwd) image pair.

    Returns a dict with the raw head signals (see module docstring) + the
    segmentation map (model-input resolution) + a CLS embedding sha. All probs
    in [0,1]; ``q_score`` is the soft-ordinal continuous quality (~60-90)."""
    import torch

    from mast.vision._vigil.heads.head_q_soft_ordinal import q_readout
    from mast.vision._vigil.training.multihead import (
        _feats_for_t,
        _patch_tokens,
        q_head_input,
    )

    heads = model.heads
    cfg = model.cfg
    with torch.no_grad():
        feats, cls, images, _gb = _forward(model, fwd_pm, bwd_pm, scan_size_nm)

        # Q — soft-ordinal continuous score
        qp = heads["q"](q_head_input(cfg.head_q_input, cls, feats))
        if cfg.head_q_form in model._q_soft_forms:
            q_score = q_readout(qp, heads["q"].binning, cfg.head_q_form)[0]
        else:
            q_score = qp
        q_score = float(q_score.reshape(-1)[0])

        # T — instability P
        t = heads["t"](_feats_for_t(feats))
        tl = t["img_logit"] if isinstance(t, dict) else t
        t_p = float(sig(tl).reshape(-1)[0])

        # N — multi-apex P(apex>=2)
        nl = heads["n"](cls)
        nl = nl if nl.dim() == 1 else nl[:, -1]
        n_p = float(sig(nl).reshape(-1)[0])

        # S — geometry
        s = heads["s"](cls)
        s_axis_ratio = float(s["axis_ratio"].reshape(-1)[0])
        s_asym_p = float(sig(s["asym_logit"]).reshape(-1)[0])

        # K — contamination P
        k_p = float(sig(heads["k"](cls)).reshape(-1)[0])

        # C — 4-class seg
        tok, hw = _patch_tokens(feats)
        seg_logits = heads["c"](tok, hw=hw, target_hw=images.shape[-2:])
        seg = seg_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        seg_names = ["terrace", "step", "defect", "contam"]
        seg_frac = {nm: float((seg == j).mean()) for j, nm in enumerate(seg_names)}

        cls_sha = hashlib.sha256(
            cls.detach().float().cpu().numpy().tobytes()
        ).hexdigest()[:16]

    return {
        "q_score": q_score,
        "t_p": t_p,
        "n_p": n_p,
        "s_axis_ratio": s_axis_ratio,
        "s_asym_p": s_asym_p,
        "k_p": k_p,
        "seg_map": seg,
        "seg_frac": seg_frac,
        "cls_sha": cls_sha,
    }


def segment(model: V25Model, fwd_pm: np.ndarray, bwd_pm: np.ndarray, scan_size_nm: float) -> np.ndarray:
    """Just the 4-class segmentation map (model-input resolution)."""
    return np.asarray(infer(model, fwd_pm, bwd_pm, scan_size_nm)["seg_map"], dtype=np.uint8)


__all__ = ["V25Model", "load_model", "infer", "segment", "configure_backbone_cache", "SCALE_DIM"]
