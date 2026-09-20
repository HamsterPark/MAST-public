"""Legacy wrapper — call v1 ResNet18 / AttentionUNet / DuelingDQN checkpoints
to satisfy the v2 VisionModule API while DINOv3 (Phase 9) is not yet trained.

Active when MAST_LEGACY_VISION=1 (default in Phase 1-8). Phase 9 installs the
DINOv3 path; Phase 10 deletes this module entirely after a 2-week stable
production window per plan §Phase 10.

On 2026-06-01 (when v1 was archived) this was switched to v2's OWN
`mast.training.models.architectures` (byte-identical to the v1 file that trained
the checkpoints), so the fallback no longer imports v1 source. Checkpoints moved
into `MASTv2/artifacts/legacy_models/`.

Inputs: numpy arrays (H, W) or (H, W, 3). Output: same Pydantic types as the
DINOv3 path so downstream code is unchanged.

Legacy checkpoints (each has metadata.json + .pth file), under
`MASTv2/artifacts/legacy_models/`:
    tip_classifier_vgg4/        → VGG4_1ch, num_classes=2 (binary good/bad)
    defect_segmenter_attn_v2/   → AttentionUNet_1ch, num_classes=1 (mask)
    surface_classifier/         → ResNet18_1ch, num_classes=2
    tip_classifier_v3/          → ResNet18_1ch, num_classes=2 (alt)
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch  # noqa: F401

from .module import (
    PartialAssessmentResult,
    SegmentationResult,
    TipCoarseResult,
    TipFineResult,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Dynamic load of v1 architectures (bypasses v1/v2 path conflict)
# ─────────────────────────────────────────────────────────────────────

_V1_ARCH_MODULE: Any = None
_V1_ARCH_LOCK = threading.Lock()


def _load_v1_arch_module() -> Any:
    """Return the legacy vision architectures module.

    Uses v2's OWN ``mast.training.models.architectures`` — which is byte-identical
    to the v1 ``mast/training/models/architectures.py`` that trained the legacy
    checkpoints, so state_dicts load cleanly. Previously this SourceFileLoader-
    imported v1's file by path; switched to the v2-native module on 2026-06-01
    when v1 was archived, so the fallback no longer depends on v1 source. Cached.
    """
    global _V1_ARCH_MODULE
    if _V1_ARCH_MODULE is not None:
        return _V1_ARCH_MODULE
    with _V1_ARCH_LOCK:
        if _V1_ARCH_MODULE is not None:
            return _V1_ARCH_MODULE
        try:
            from mast.training.models import architectures as _arch
        except ImportError as exc:  # 公开版：legacy 模型结构不随仓发布
            raise RuntimeError("legacy vision architectures are not shipped in this snapshot") from exc
        _V1_ARCH_MODULE = _arch
        logger.info("Loaded legacy architectures (mast.training.models.architectures)")
        return _V1_ARCH_MODULE


# ─────────────────────────────────────────────────────────────────────
# Image preprocessing (numpy → torch tensor)
# ─────────────────────────────────────────────────────────────────────

def _preprocess_grayscale(img: np.ndarray, target_size: int) -> Any:
    """numpy (H, W) or (H, W, 3) → torch tensor (1, 1, target_size, target_size).

    - 3-channel input gets averaged to 1-channel
    - Resized via numpy nearest-neighbor (avoid scipy dependency)
    - Normalized to [0, 1] from raw range
    """
    import torch

    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(axis=2)
    elif img.ndim != 2:
        raise ValueError(f"Expected 2D or 3-channel image, got shape {img.shape}")

    # Resize via simple stride-pick (good enough for low-frequency STM textures)
    h, w = img.shape
    if h != target_size or w != target_size:
        # Use cv2-free scipy-free resize: torch's interpolate is the cleanest
        t = torch.from_numpy(img.astype(np.float32))[None, None, :, :]
        t = torch.nn.functional.interpolate(
            t, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
    else:
        t = torch.from_numpy(img.astype(np.float32))[None, None, :, :]

    # Normalize to [0, 1]
    t_min = t.min()
    t_max = t.max()
    if (t_max - t_min) > 1e-8:
        t = (t - t_min) / (t_max - t_min)
    return t


# ─────────────────────────────────────────────────────────────────────
# LegacyBackend — singleton-managed by VisionModule
# ─────────────────────────────────────────────────────────────────────

class LegacyBackend:
    """Loads v1 .pth checkpoints lazily. Real torch import on first use.

    Defaults look at `models/` (the v1 production location) at the project root.
    Override via constructor for tests or custom layout.
    """

    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        tip_classifier_subdir: str = "tip_classifier_vgg4",
        segmenter_subdir: str = "defect_segmenter_attn_v2",
    ):
        # Fail-fast checks so VisionModule can fall back to MockBackend
        # without users discovering the problem mid-pipeline.
        try:
            import torch  # noqa: F401
        except Exception as e:  # pragma: no cover — depends on env
            raise ImportError(
                "torch is not installed; legacy vision backend unavailable"
            ) from e

        # Vendored legacy checkpoints live under MASTv2/artifacts/legacy_models
        # (moved out of the archived v1 root `models/` on 2026-06-01 so v2 is
        # self-contained). Relative overrides resolve against MASTv2/.
        _v2_root = Path(__file__).resolve().parents[2]   # …/MASTv2
        if artifacts_dir is None:
            self.artifacts_dir = _v2_root / "artifacts" / "legacy_models"
        else:
            self.artifacts_dir = Path(artifacts_dir)
            if not self.artifacts_dir.is_absolute():
                self.artifacts_dir = _v2_root / self.artifacts_dir
        self._project_root = _v2_root

        self._tip_subdir = tip_classifier_subdir
        self._seg_subdir = segmenter_subdir

        # Probe checkpoint presence — if no .pth files at all under the
        # configured roots, raise so VisionModule can switch to MockBackend.
        # Individual missing subdirs are still tolerated at runtime (the head
        # downgrades to 'unknown').
        if not self.artifacts_dir.exists() or not any(
            self.artifacts_dir.rglob("*.pth")
        ):
            raise FileNotFoundError(
                f"No legacy vision checkpoints (*.pth) under {self.artifacts_dir}"
            )

        self._tip_model: Any = None
        self._tip_input_size: int = 64
        self._segmenter: Any = None
        self._device: Any = None
        self._models_loaded: bool = False
        self._load_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._models_loaded:
            return
        with self._load_lock:
            if self._models_loaded:
                return
            try:
                import torch
            except ImportError as e:
                raise RuntimeError(f"torch import failed in legacy backend: {e}") from e

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("LegacyBackend loading v1 models on %s", self._device)
            arch = _load_v1_arch_module()

            # ── Tip classifier ──
            tip_dir = self.artifacts_dir / self._tip_subdir
            self._tip_model = self._load_one(
                arch, tip_dir, default_arch="vgg4_1ch", default_classes=2
            )

            # ── Segmenter ──
            seg_dir = self.artifacts_dir / self._seg_subdir
            self._segmenter = self._load_one(
                arch, seg_dir, default_arch="attention_unet_1ch", default_classes=1
            )

            self._models_loaded = True
            logger.info("LegacyBackend loaded: tip=%s, segmenter=%s",
                        bool(self._tip_model), bool(self._segmenter))

    def _load_one(
        self,
        arch_module: Any,
        model_dir: Path,
        default_arch: str,
        default_classes: int,
    ) -> Any:
        """Load one v1 checkpoint. Returns None if checkpoint missing (graceful)."""
        import torch

        if not model_dir.is_dir():
            logger.warning("LegacyBackend: %s missing — that head returns 'unknown'", model_dir)
            return None

        meta_path = model_dir / "metadata.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("metadata.json parse failed: %s", e)
        arch_name = meta.get("architecture", default_arch)
        num_classes = meta.get("num_classes", default_classes)
        state_file = meta.get("state_dict_file")

        # Pick state_dict file
        if state_file:
            ckpt_path = model_dir / state_file
        else:
            pth_files = list(model_dir.glob("*.pth"))
            if not pth_files:
                logger.warning("No .pth file in %s", model_dir)
                return None
            ckpt_path = pth_files[0]

        # Instantiate model class
        cls_name_map = {
            "vgg4_1ch": "VGG4_1ch",
            "resnet18_1ch": "ResNet18_1ch",
            "unet_1ch": "UNet_1ch",
            "attention_unet_1ch": "AttentionUNet_1ch",
        }
        cls_name = cls_name_map.get(arch_name)
        if cls_name is None or not hasattr(arch_module, cls_name):
            logger.warning("Unknown architecture '%s'; skipping %s", arch_name, model_dir)
            return None

        cls = getattr(arch_module, cls_name)
        if cls_name == "VGG4_1ch":
            model = cls(num_classes=num_classes, input_size=64)
            self._tip_input_size = 64
        elif cls_name == "ResNet18_1ch":
            model = cls(num_classes=num_classes, pretrained=False)
            self._tip_input_size = 224
        elif cls_name in ("UNet_1ch", "AttentionUNet_1ch"):
            model = cls(in_ch=1, out_ch=num_classes)
        else:
            model = cls()

        try:
            state = torch.load(str(ckpt_path), map_location=self._device, weights_only=True)
        except (TypeError, RuntimeError):
            # weights_only=True only available on torch 2.4+; fall back
            state = torch.load(str(ckpt_path), map_location=self._device)
        # Some checkpoints save the whole model dict; some save .state_dict() directly
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            logger.error("State load mismatch for %s: %s", ckpt_path, e)
            return None
        model.to(self._device)
        model.eval()
        logger.info("Loaded %s from %s", cls_name, ckpt_path.name)
        return model

    # ─────────────────────────────────────────────────────────────────
    # API methods (called by VisionModule under threading.Lock)
    # ─────────────────────────────────────────────────────────────────

    def assess_tip_coarse(self, image: np.ndarray) -> TipCoarseResult:
        self._ensure_loaded()
        if self._tip_model is None:
            # FAIL-SAFE: with no classifier we CANNOT judge
            # the tip. Returning "good" (as the old code did) is optimistic and
            # unsafe — it makes assess_tip_fine report is_usable=True, so a bad
            # tip proceeds to imaging/data. Match the mock backend's pessimistic
            # fallback: "bad" @ 0 confidence (is_usable=False → don't trust it).
            return TipCoarseResult(label="bad", confidence=0.0, embedding_sha=None)

        import torch

        x = _preprocess_grayscale(image, target_size=self._tip_input_size).to(self._device)
        with torch.inference_mode():
            logits = self._tip_model(x)
            probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        # DeepSPM dataset ordering: LABEL_MAP = {"good": 1, "bad": 0}
        # (mast/training/datasets/deepspm.py:40). The classifier is trained with
        # CrossEntropyLoss on those raw integer labels, so logit index 0 == BAD
        # and index 1 == GOOD. The previous mapping ["good", "bad"] was INVERTED
        # — bad tips were reported as good, defeating the downstream safety gate.
        # 审查 finding [39].
        labels = ["bad", "good"]
        return TipCoarseResult(
            label=labels[idx if idx < len(labels) else 0],
            confidence=float(probs[idx].item()),
            embedding_sha=None,
        )

    def assess_tip_fine(self, image: np.ndarray) -> TipFineResult:
        """v1 has no 8-class fine taxonomy — return 'unknown' downgrade.

        Phase 9 DINOv3 will return real fine-tip labels.
        """
        coarse = self.assess_tip_coarse(image)
        return TipFineResult(
            label="unknown",
            top2=[],
            is_usable=(coarse.label == "good"),
        )

    def segment(self, image: np.ndarray, classes: list[str] | None = None,
                *, level: int | None = None, tile: int | None = None) -> SegmentationResult:
        # `level` / `tile` are Phase-9 (M12) concepts; the legacy AttentionUNet
        # has a fixed taxonomy + path, so they're accepted for API parity and
        # ignored.
        self._ensure_loaded()
        h, w = image.shape[:2] if hasattr(image, "shape") else (256, 256)
        if self._segmenter is None:
            return SegmentationResult(
                mask_rle=b"",
                shape=(h, w),
                class_counts={"TERRACE": h * w},
            )

        import torch

        # AttentionUNet wants H/W divisible by 16 (4 down levels) — pad up
        target_h = ((max(h, 64) + 15) // 16) * 16
        target_w = ((max(w, 64) + 15) // 16) * 16
        x = _preprocess_grayscale(image, target_size=max(target_h, target_w)).to(self._device)
        with torch.inference_mode():
            seg_out = self._segmenter(x)
            # 1-channel output: defect probability (sigmoid)
            mask_prob = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
        mask_bin = (mask_prob > 0.5).astype(np.uint8)
        # _preprocess_grayscale forces a SQUARE (S, S) tensor, so mask_bin is
        # square regardless of the original aspect ratio. Resize the binary mask
        # back to the true input geometry (h, w) so the returned shape / RLE /
        # class_counts describe the actual scan — otherwise non-square scans get
        # a transposed/stretched mask and a shape that doesn't match the image.
        # finding #94 (2026-05-30).
        if mask_bin.shape != (h, w):
            mask_bin = _resize_mask_nn(mask_bin, (h, w))
        # Run-length-encode the binary mask
        rle = _rle_encode(mask_bin)

        # v1 model is binary — map to v2 7-class subset
        n_defect = int(mask_bin.sum())
        n_terrace = mask_bin.size - n_defect
        return SegmentationResult(
            mask_rle=rle,
            shape=(h, w),
            class_counts={
                "TERRACE": n_terrace,
                "POINT_DEFECT_BRIGHT": n_defect,  # v1 doesn't distinguish bright/dark
            },
        )

    def partial_assess(
        self, scan_lines: np.ndarray, n_available: int
    ) -> PartialAssessmentResult:
        """v1 has no partial-image head — pad missing rows with zeros, treat as full scan."""
        if not hasattr(scan_lines, "shape") or scan_lines.ndim < 2:
            return PartialAssessmentResult(
                quality_pred=0.5, coarse_label="unknown", frac_acquired=0.0
            )
        total_lines = scan_lines.shape[0]
        n_avail = max(0, min(n_available, total_lines))
        if n_avail == 0:
            return PartialAssessmentResult(
                quality_pred=0.5, coarse_label="unknown", frac_acquired=0.0
            )
        # Build a square image: keep the first n_avail rows, pad rest with zeros
        padded = np.zeros_like(scan_lines)
        padded[:n_avail, :] = scan_lines[:n_avail, :]
        coarse = self.assess_tip_coarse(padded)
        return PartialAssessmentResult(
            quality_pred=coarse.confidence if coarse.label == "good" else 1 - coarse.confidence,
            coarse_label=coarse.label,
            frac_acquired=min(max(n_avail / total_lines, 0.0), 1.0),
        )


# ─────────────────────────────────────────────────────────────────────
# Mask geometry helper — restore (h, w) after the square forward pass
# ─────────────────────────────────────────────────────────────────────

def _resize_mask_nn(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label/binary mask to ``out_hw``.

    Pure numpy (no torch / scipy / cv2) so it is importable and testable
    without the heavy deps. Nearest-neighbour preserves the integer class
    labels exactly — never produces fractional/intermediate codes the way
    bilinear would.
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    in_h, in_w = mask.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return mask
    if out_h <= 0 or out_w <= 0 or in_h <= 0 or in_w <= 0:
        return np.zeros((max(out_h, 0), max(out_w, 0)), dtype=mask.dtype)
    row_idx = (np.arange(out_h) * in_h // out_h).clip(0, in_h - 1)
    col_idx = (np.arange(out_w) * in_w // out_w).clip(0, in_w - 1)
    return mask[row_idx[:, None], col_idx[None, :]]


# ─────────────────────────────────────────────────────────────────────
# RLE helper (mask_rle in SegmentationResult is RLE-encoded bytes)
# ─────────────────────────────────────────────────────────────────────

# RLE codec moved to mast.vision.seg_utils (Phase 9 — multi-class value-run
# format). Re-exported here under the historical names so existing imports
# (`from mast.vision._legacy_wrapper import _rle_encode`) keep working. The
# binary masks the legacy backend produces are just the 0/1 special case.
from mast.vision.seg_utils import decode_rle as _rle_decode  # noqa: E402
from mast.vision.seg_utils import encode_rle as _rle_encode  # noqa: E402


__all__ = ["LegacyBackend"]
