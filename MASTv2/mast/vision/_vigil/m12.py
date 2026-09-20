"""M12 production inference — faithful port of VIGIL ``infer_vigil_v23``.

This mirrors the *verified* load + forward path
(``VIGIL/vigil_v2.3/ckpt_production/infer_vigil_v23.py``, itself line-aligned
with ``scripts/eval_v23_candidate.py``). It loads the M12 checkpoint (LoRA r=8
+ 3 heads, ~41 MB) onto a frozen DINOv3-ViT-L/16 timm backbone and runs all
three heads on an STM image.

⚠️  DO NOT "simplify" two things below — they look wrong but are exactly what
    the weights were trained with; changing them silently corrupts outputs:

      1. The scale embedding is fed ``log10(scan)`` and ScaleEmbedding then
         takes ``log10`` again (an effective double-log). Verbatim from the
         verified path. Keep it.
      2. The patch tokens are tail-sliced (``pt[:, pt.shape[1]-n:]``) to drop
         the leading register tokens the registry doesn't declare. Keep it.

MAST does NOT train these heads. ``load_model`` aborts loudly if the LoRA keys
don't match (backbone variant mismatch → silently-wrong model), per the
upstream safety assertion.
"""
from __future__ import annotations

import hashlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backbone.dinov_loader import load_backbone
from .backbone.lora_config import attach_lora
from .data.multichannel_processor import compose_channels, plane_flatten
from .data.stm_image_processor import STMImageProcessor
from .heads.head_b_v2 import HeadBv2
from .heads.head_c_l1_dino import HeadCLevel1DINO
from .heads.head_q_sharpness import HeadQSharpness, ScaleEmbedding

logger = logging.getLogger(__name__)

SCALE_DIM = 64
L1_NAMES = {0: "terrace", 1: "step", 2: "defect", 3: "contamination"}
# Head B fine morphology (4-way) and coarse fallback (3-way).
MORPH_FINE = {0: "M0", 1: "M1", 2: "M2", 3: "M3"}
MORPH_COARSE = {0: "M0", 1: "M1", 2: "M2M3"}


def configure_backbone_cache(cache_dir: str | Path | None) -> Path | None:
    """Force timm/HF offline and locate the bundled cache. Returns its root.

    Thin delegate to :mod:`mast.vision._vigil.backbone_cache` — shared with the
    v25 path so both backends resolve weights identically. The returned root is
    what makes the load actually work (timm gets an explicit weight file);
    ``HF_HOME`` alone cannot retarget an already-imported ``huggingface_hub``.
    """
    from mast.vision._vigil.backbone_cache import configure as _configure
    return _configure(cache_dir)


@dataclass
class M12Model:
    """Loaded M12 model components (backbone wrapper + scale emb + 3 heads)."""

    wrapper: Any
    scale_emb: Any
    head_q: Any
    head_b: Any
    head_c: Any
    proc: Any
    device: Any
    ckpt_args: dict
    backbone_name: str
    lora_rank: int
    embed_dim: int


def load_model(
    ckpt_path: str | Path,
    device: str = "cpu",
    backbone_cache_dir: str | Path | None = None,
) -> M12Model:
    """Load the M12 checkpoint + frozen DINOv3 backbone. Verbatim load path."""
    import torch

    configure_backbone_cache(backbone_cache_dir)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    bname = state["backbone_name"]
    rank = int(state["lora_rank"])
    embed = int(state["embed_dim"])
    ckpt_args = dict(state.get("args", {}))
    logger.info(
        "M12 ckpt: backbone=%s lora_rank=%d embed_dim=%d flatten=%s input_mode=%s",
        bname, rank, embed, ckpt_args.get("flatten", False),
        ckpt_args.get("input_mode", "E3"),
    )

    wrapper = load_backbone(bname, in_channels=3, freeze=True)
    is_timm = hasattr(wrapper.backbone, "forward_features") and not hasattr(
        wrapper.backbone, "config"
    )
    wrapper.backbone = attach_lora(
        wrapper.backbone, rank=rank, alpha=2 * rank,
        target_modules=["qkv"] if is_timm else ["query", "key", "value"],
    )
    res = wrapper.backbone.load_state_dict(state["lora_state"], strict=False)
    # SAFETY (upstream): strict=False silently drops mismatched keys. If the
    # local backbone variant (timm vs HF) differs from training, NONE of the
    # LoRA loads → silently-wrong model. Abort if EVERY LoRA key is unexpected.
    n_lora = len(state["lora_state"])
    unexpected = set(res.unexpected_keys)
    n_unexpected = sum(1 for k in state["lora_state"] if k in unexpected)
    if n_lora > 0 and n_unexpected == n_lora:
        raise RuntimeError(
            f"LoRA key mismatch: all {n_lora} LoRA keys are 'unexpected' → "
            f"backbone variant differs from training (is_timm={is_timm}). "
            "Outputs would be wrong. M12 trained on the timm backbone."
        )
    logger.info("M12 LoRA loaded: %d/%d keys matched (is_timm=%s)",
                n_lora - n_unexpected, n_lora, is_timm)
    wrapper.to(device).eval()

    scale_emb = ScaleEmbedding(dim=SCALE_DIM).to(device)
    scale_emb.load_state_dict(state["scale_emb_state"]); scale_emb.eval()
    head_q = HeadQSharpness(in_dim=embed + SCALE_DIM).to(device)
    head_q.load_state_dict(state["head_q_state"]); head_q.eval()
    head_b = HeadBv2(in_dim=embed + SCALE_DIM).to(device)
    head_b.load_state_dict(state["head_b_state"]); head_b.eval()
    head_c = HeadCLevel1DINO(embed_dim=embed, num_classes=4).to(device)
    head_c.load_state_dict(state["head_c_state"]); head_c.eval()

    proc = STMImageProcessor(target_size=int(ckpt_args.get("size", 256)))
    proc.flatten = bool(ckpt_args.get("flatten", False))
    return M12Model(
        wrapper=wrapper, scale_emb=scale_emb, head_q=head_q, head_b=head_b,
        head_c=head_c, proc=proc, device=device, ckpt_args=ckpt_args,
        backbone_name=bname, lora_rank=rank, embed_dim=embed,
    )


def _build_input(fwd_pm, bwd_pm, proc, device, flatten):
    """Verbatim from infer_vigil_v23.build_input. Per-channel; never merge."""
    fwd = fwd_pm.to(device); bwd = bwd_pm.to(device)
    if flatten:
        fwd = plane_flatten(fwd); bwd = plane_flatten(bwd)
    fwd_n = proc.normalize_per_image(fwd)
    bwd_n = proc.normalize_per_image(bwd)
    images = compose_channels(fwd_n, bwd_n, mode="E3")
    return proc.to_dino_input(images).to(device)


# ── GPU memory-efficiency (GTX 1650 / 4 GB Turing) ───────────────────
# Two switches keep ViT-L + high-res segmentation inside a 4 GB card:
#   1. fp16 autocast on CUDA — halves weights + activations, faster. CPU stays
#      fp32 (no autocast).
#   2. Memory-efficient SDPA — never materialises the N² attention matrix.
#      We request EFFICIENT_ATTENTION (xformers-style, works on Turing) and fall
#      back to MATH; we DO NOT request FLASH (Ampere+ only — GTX 1650 lacks it).
#      timm's ViT uses F.scaled_dot_product_attention when fused-attn is on, so
#      this context governs it. No-op on torch builds without sdpa_kernel.

def _amp_ctx(device: Any, torch):
    if str(device).startswith("cuda"):
        try:
            return torch.autocast("cuda", dtype=torch.float16)
        except Exception:  # noqa: BLE001
            return nullcontext()
    return nullcontext()


def _sdpa_ctx(torch):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    except Exception:  # noqa: BLE001 — older torch / no SDPA control → no-op
        return nullcontext()


def infer(model: M12Model, fwd_pm, bwd_pm, scan_size_nm: float) -> dict:
    """Run all three heads. Returns a rich, JSON-friendly result dict.

    fwd_pm / bwd_pm: (H, W) float arrays in picometres (pass the same array
    twice for a single-channel image — E1-degrade; Head B is weaker then).
    scan_size_nm: physical scan edge length (Head Q's scale conditioning).
    """
    import torch

    device = model.device
    with torch.inference_mode(), _amp_ctx(device, torch), _sdpa_ctx(torch):
        fwd = torch.as_tensor(np.asarray(fwd_pm), dtype=torch.float32)[None]  # (1,H,W)
        bwd = torch.as_tensor(np.asarray(bwd_pm), dtype=torch.float32)[None]
        images = _build_input(fwd, bwd, model.proc, device, model.proc.flatten)
        out = model.wrapper(images)

        scan = torch.tensor([float(scan_size_nm)], device=device)
        # ⚠️ double-log is intentional (see module docstring). Do not change.
        emb = model.scale_emb(torch.log10(scan.clamp(min=1e-3)))
        cls = torch.cat([out["cls"], emb], dim=-1)

        # Head Q — log10(R_tip / scan_size) → R_tip_nm
        q = float(model.head_q(cls).flatten()[0])
        r_nm = float(scan_size_nm) * (10.0 ** q)

        # Head B — fine + coarse tip state (HeadBPredictions dataclass)
        b = model.head_b(cls)
        probs = b.as_probs()
        morph_coarse_probs = probs["morph_coarse_probs"][0].cpu().numpy()
        morph_fine_probs = probs["morph_probs"][0].cpu().numpy()
        morph_coarse_idx = int(morph_coarse_probs.argmax())
        morph_fine_idx = int(morph_fine_probs.argmax())
        switching = float(probs["switching_prob"][0].item())
        drift = float(probs["drift_prob"][0].item())
        perturbation = float(probs["perturbation_prob"][0].item())
        multitip_coarse = float(probs["multitip_coarse_prob"][0].item())
        stability_coarse = float(probs["stability_coarse_prob"][0].item())
        n_tips = float(probs["n_tips_estimate"][0].item())

        # Head C-L1 — 4-class per-pixel segmentation. Tail-slice register
        # tokens (registry doesn't declare them). ⚠️ keep the slice.
        pt = out["tokens"]
        n = out["hw"][0] * out["hw"][1]
        if pt.shape[1] > n:
            pt = pt[:, pt.shape[1] - n:]
        seg_logits = model.head_c(pt, hw=out["hw"], target_hw=images.shape[-2:])
        seg = seg_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # (H,W)

        cls_sha = hashlib.sha1(
            out["cls"][0].detach().to("cpu").to(torch.float32).numpy().tobytes()
        ).hexdigest()[:12]

    seg_fraction = {L1_NAMES[c]: float((seg == c).mean()) for c in range(4)}
    return {
        # Head Q
        "q_log10_R_over_scan": q,
        "R_tip_nm": r_nm,
        # Head B coarse (T0 gate signals)
        "morph_coarse_idx": morph_coarse_idx,
        "morph_coarse": MORPH_COARSE[morph_coarse_idx],
        "morph_coarse_probs": morph_coarse_probs.tolist(),
        # Head B fine
        "morph_fine_idx": morph_fine_idx,
        "morph_fine": MORPH_FINE[morph_fine_idx],
        "morph_fine_probs": morph_fine_probs.tolist(),
        "switching_prob": switching,
        "drift_prob": drift,
        "perturbation_prob": perturbation,
        "multitip_coarse_prob": multitip_coarse,
        "stability_coarse_prob": stability_coarse,
        "n_tips_estimate": n_tips,
        # Head C-L1
        "seg_map": seg,
        "seg_class_fraction": seg_fraction,
        # debug
        "cls_sha": cls_sha,
    }


# ── High-resolution segmentation via sliding-window tiling ───────────
# Decouples peak VRAM from the scan's pixel count: a huge scan is processed in
# overlapping `tile`×`tile` windows (each fits a 4 GB card) and the per-tile
# Head C-L1 logits are averaged on the overlaps. SPM features (defects / atoms /
# tip artifacts) are local, so tiling is near-lossless (only long-range global
# context is lost, which Head C-L1 barely uses). Per-channel normalisation is
# done ONCE on the full image (consistent contrast across tiles), then tiles are
# cut at native resolution — NO 256-resize, so fine detail is preserved.

_PATCH = 16  # dinov3-vitl16 patch size


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, max(1, stride)))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _seg_logits(model: M12Model, images_crop):
    """Backbone + Head C-L1 on one (1,3,h,w) ImageNet-normalised crop →
    (1, num_classes, h, w) logits. h,w must be multiples of the patch size."""
    out = model.wrapper(images_crop)
    pt = out["tokens"]
    n = out["hw"][0] * out["hw"][1]
    if pt.shape[1] > n:
        pt = pt[:, pt.shape[1] - n:]  # ⚠️ drop leading register tokens
    return model.head_c(pt, hw=out["hw"], target_hw=images_crop.shape[-2:])


def segment_large(
    model: M12Model, fwd_pm, bwd_pm, *, tile: int = 256, overlap: int = 32,
) -> np.ndarray:
    """Tiled high-resolution 4-class segmentation → (H, W) uint8 class map.

    Use for scans too large to run whole on a small GPU. ``tile`` is snapped to
    a multiple of the patch size; ``overlap`` (px) blends tile seams."""
    import torch

    device = model.device
    proc = model.proc
    tile = max(_PATCH, (int(tile) // _PATCH) * _PATCH)
    overlap = max(0, min(int(overlap), tile - _PATCH))
    stride = max(_PATCH, tile - overlap)

    fwd = torch.as_tensor(np.asarray(fwd_pm), dtype=torch.float32)[None]
    bwd = torch.as_tensor(np.asarray(bwd_pm), dtype=torch.float32)[None]
    with torch.inference_mode(), _amp_ctx(device, torch), _sdpa_ctx(torch):
        if proc.flatten:
            fwd = plane_flatten(fwd); bwd = plane_flatten(bwd)
        fwd_n = proc.normalize_per_image(fwd)
        bwd_n = proc.normalize_per_image(bwd)
        images = compose_channels(fwd_n, bwd_n, mode="E3")  # (1,3,H,W) ∈ [0,1]
        mean = proc._mean.to(device); std = proc._std.to(device)
        images = (images.to(device) - mean) / std  # ImageNet-norm, NO resize
        _, _, H, W = images.shape

        # Small enough to run whole — pad to a patch multiple, run once, crop.
        if H <= tile and W <= tile:
            ph = (_PATCH - H % _PATCH) % _PATCH
            pw = (_PATCH - W % _PATCH) % _PATCH
            padded = torch.nn.functional.pad(images, (0, pw, 0, ph)) if (ph or pw) else images
            logits = _seg_logits(model, padded)
            seg = logits.argmax(1)[0, :H, :W].to("cpu").numpy().astype(np.uint8)
            return seg

        nclass = 4
        accum = torch.zeros((nclass, H, W), device=device, dtype=torch.float32)
        weight = torch.zeros((1, H, W), device=device, dtype=torch.float32)
        for y in _tile_starts(H, tile, stride):
            for x in _tile_starts(W, tile, stride):
                crop = images[:, :, y:y + tile, x:x + tile]
                logits = _seg_logits(model, crop)  # (1,nclass,tile,tile)
                accum[:, y:y + tile, x:x + tile] += logits[0].float()
                weight[:, y:y + tile, x:x + tile] += 1.0
        weight = weight.clamp(min=1.0)
        return (accum / weight).argmax(0).to("cpu").numpy().astype(np.uint8)


__all__ = ["M12Model", "load_model", "infer", "segment_large",
           "configure_backbone_cache", "L1_NAMES", "MORPH_FINE",
           "MORPH_COARSE", "SCALE_DIM"]
