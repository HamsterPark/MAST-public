"""stm_quality_v1 — learned STM frame-quality scorer trained on REAL human labels.

Why this exists (see docs/v2/benchmarks/vision_v25_diagnostic/ and the VIGIL run
log): the deployed v2.5 checkpoint was trained on *synthetic* data only and had
never seen a real human quality label. On a locked, group-disjoint test fold of
the sf09 gold set it scores Spearman **0.196** / AUROC **0.628**. Training a
ridge head on the 778 previously-unused REAL labels over **frozen** DINOv3
features scores **0.684 / 0.946** on the same frames — the representation was
never the problem, the supervision was.

This is a **parallel, additive** scorer. It does NOT replace
``artifacts/mast_vision_v25.pt`` or touch :class:`VIGILBackend`; the two answer
different questions and can be run side by side.

Pipeline
    raw height map
      → plane-flatten + robust percentile stretch   (modality match, see below)
      → frozen timm DINOv3 ViT-S/16 @448, CLS ⊕ patch-mean (768-d)
      → StandardScaler + Ridge  → continuous quality score
      → optional scale de-bias when the scan size is known
      → tier (bad / marginal / excellent) via calibrated thresholds

**Modality match**: real STM frames are always plane-levelled before they are
looked at; feeding the raw tilted height map into an 8-bit stretch crushes the
surface texture (measured: 6.7× less high-frequency power, radial-spectrum slope
0.23 away from real). Plane-flattening first closes that to 0.023.

Known limits (carried from the model card, do not paper over them):
  * Trained on presentation-exported grayscale frames, not raw instrument height
    maps, and those frames carry **no scan size** — so the learned score judges
    appearance. A weak scale confound (ρ=+0.13 with log scan size on real data)
    is removed by the de-bias polynomial when ``scan_size_nm`` is supplied.
  * Single-channel input → it is **blind to trace-vs-retrace instability**. That
    signal is covered by :mod:`mast.vision.tip_quality` / :mod:`mast.vision.tip_metrics`;
    the two families are weakly correlated, i.e. complementary. Use both.
  * One global quality tier only — no morphology / geometry / segmentation.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MASTv2/artifacts/stm_quality_v1_dino.joblib"
DEFAULT_BACKBONE_CACHE = "MASTv2/artifacts/vision_backbone"
_TIERS = ("bad", "marginal", "excellent")


class QualityScoreResult(BaseModel):
    """Learned quality score for one frame.

    Defined here rather than in :mod:`mast.vision.module` to keep this additive
    scorer self-contained (it is loaded lazily and is optional).

    ``score`` is continuous, higher = better; ``tier`` applies the calibrated
    cut-points. ``scale_corrected`` says whether the scan-size de-bias was
    applied (it needs ``scan_size_nm``).
    """
    model_config = ConfigDict(frozen=True)
    score: float
    tier: str
    scale_corrected: bool = False
    scan_size_nm: float | None = Field(default=None, ge=0)
    model_name: str = "stm_quality_v1"


# ── modality-matching render (validated in the VIGIL study) ──────────────────
def plane_subtract(h: np.ndarray) -> np.ndarray:
    """Least-squares plane removal — the STM 'plane level' step."""
    h = np.asarray(h, dtype=np.float32)
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]].astype(np.float32)
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(h.size, np.float32)]
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    return (h.ravel() - A @ coef).reshape(h.shape).astype(np.float32)


def render_like_real(image: np.ndarray, plo: float = 0.5, phi: float = 99.5) -> np.ndarray:
    """Height map → the 8-bit-valued grayscale the model was trained on.

    Accepts (H,W) · (2,H,W)/(1,H,W) trace-retrace (forward is used) · (H,W,3).
    """
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[0] in (1, 2):
            a = a[0]
        elif a.shape[-1] == 3:
            a = a.mean(axis=-1)
        else:
            raise ValueError(f"cannot interpret 3-D image of shape {a.shape}")
    elif a.ndim != 2:
        raise ValueError(f"image must be 2-D or 3-D, got {a.shape}")
    h = plane_subtract(a)
    lo, hi = np.percentile(h, [plo, phi])
    if hi - lo < 1e-9:
        return np.zeros(h.shape, np.float32)
    return (np.clip((h - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.float32)


def _resolve(relpath: str | Path) -> Path:
    """Locate a bundled artifact (mirrors VIGILBackend's resolution order)."""
    rel = Path(relpath)
    if rel.is_absolute():
        return rel
    here = Path(__file__).resolve()
    cands = [here.parents[3] / rel, here.parents[2] / rel, Path.cwd() / rel]
    if getattr(sys, "frozen", False):
        cands.append(Path(sys.executable).resolve().parent / rel)
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        cands.append(Path(meipass) / rel)
    for c in cands:
        if c.exists():
            return c
    return cands[0]


class STMQualityModelMissing(RuntimeError):
    """The joblib head or the DINOv3 backbone cache is not available."""


class STMQualityScorer:
    """Lazy, thread-safe scorer. Construct once (VisionModule holds one)."""

    def __init__(self, model_path: str | Path | None = None,
                 backbone_cache_dir: str | Path | None = None,
                 device: str | None = None) -> None:
        self._model_path = _resolve(
            model_path or os.environ.get("MAST_QUALITY_MODEL", "").strip() or DEFAULT_MODEL)
        self._cache_dir = _resolve(
            backbone_cache_dir or os.environ.get("MAST_VISION_BACKBONE_CACHE", "").strip()
            or DEFAULT_BACKBONE_CACHE)
        self._device_arg = device
        self._bundle = None
        self._net = None
        self._device = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        return self._model_path

    def is_loaded(self) -> bool:
        return self._bundle is not None and self._net is not None

    def _load(self):
        if self.is_loaded():
            return self._bundle, self._net
        with self._lock:
            if self.is_loaded():
                return self._bundle, self._net
            if not self._model_path.exists():
                raise STMQualityModelMissing(f"quality head not found: {self._model_path}")
            try:
                import joblib
                import timm
                import torch
            except ImportError as e:  # noqa: BLE001
                raise STMQualityModelMissing(f"missing dependency: {e}") from e

            # NEVER let timm reach the network — same policy as VIGILBackend.
            from mast.vision._vigil.backbone_cache import timm_pretrained_kwargs
            from mast.vision._vigil.v25 import configure_backbone_cache
            cache_root = configure_backbone_cache(self._cache_dir)

            bundle = joblib.load(self._model_path)
            pref = (self._device_arg or os.environ.get("MAST_VISION_DEVICE", "auto")).lower()
            dev = ("cuda" if (pref in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
            # Load the backbone weights from the bundled FILE — see
            # mast.vision._vigil.backbone_cache: HF_HOME alone cannot retarget
            # an already-imported huggingface_hub, which silently degraded this
            # scorer along with the main backend.
            net = timm.create_model(bundle["dino_model"], pretrained=True,
                                    num_classes=0, dynamic_img_size=True,
                                    **timm_pretrained_kwargs(bundle["dino_model"], cache_root),
                                    ).to(dev).eval()
            self._bundle, self._net, self._device = bundle, net, dev
            logger.info("stm_quality_v1 loaded on %s (head=%s)", dev, self._model_path.name)
            return self._bundle, self._net

    def preload(self) -> None:
        """Force the load now (so the first real call is fast)."""
        self._load()

    def _features(self, img: np.ndarray) -> np.ndarray:
        import torch
        bundle, net = self._load()
        size = int(bundle.get("dino_input", 448))
        dev = self._device
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        with torch.inference_mode():
            t = torch.from_numpy(img[None] / 255.0).to(dev).unsqueeze(1).repeat(1, 3, 1, 1)
            t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                                align_corners=False)
            o = net.forward_features((t - mean) / std)
            npre = getattr(net, "num_prefix_tokens", 1)
            return torch.cat([o[:, 0], o[:, npre:].mean(1)], 1).float().cpu().numpy()

    def score(self, image: np.ndarray, scan_size_nm: float | None = None) -> QualityScoreResult:
        """Score one frame. Supply ``scan_size_nm`` (Nanonis reports it) to apply
        the scale de-bias — without it the score keeps a weak scale confound."""
        bundle, _ = self._load()
        img = render_like_real(image)
        s = float(bundle["model"].predict(self._features(img))[0])

        corrected = False
        poly = bundle.get("scale_debias_poly")
        if poly and scan_size_nm and scan_size_nm > 0:
            s = s - float(np.polyval(np.asarray(poly, float), np.log10(scan_size_nm))) + 1.0
            corrected = True

        cuts = bundle.get("tier_thresholds", [0.72, 1.25])
        tier = _TIERS[int(np.clip(np.digitize(s, cuts), 0, 2))]
        return QualityScoreResult(score=s, tier=tier, scale_corrected=corrected,
                                  scan_size_nm=scan_size_nm)


_SINGLETON: STMQualityScorer | None = None
_SINGLETON_LOCK = threading.Lock()


def get_scorer() -> STMQualityScorer:
    """Process-wide scorer (the backbone is ~86 MB — load it once)."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = STMQualityScorer()
    return _SINGLETON


def score_quality(image: np.ndarray, scan_size_nm: float | None = None) -> QualityScoreResult:
    """Convenience wrapper around the process-wide scorer."""
    return get_scorer().score(image, scan_size_nm=scan_size_nm)


__all__ = ["QualityScoreResult", "STMQualityScorer", "STMQualityModelMissing",
           "get_scorer", "score_quality", "render_like_real", "plane_subtract"]
