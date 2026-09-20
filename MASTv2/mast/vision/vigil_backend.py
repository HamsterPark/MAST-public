"""VIGIL v2.5 vision backend — DINOv3-ViT-S/16 + LoRA(qkv_o_mlp, r=8) + 6 heads.

MAST does NOT train these heads. The VIGIL project produces the ssl_sf09c1
checkpoint (``MASTv2/artifacts/mast_vision_v25.pt``, LoRA + 6 heads + scale-emb,
~29 MB, NO backbone) and we bundle the frozen DINOv3-ViT-S/16 timm backbone
weights cache (``MASTv2/artifacts/vision_backbone``, ~165 MB) so inference runs
fully offline. (Supersedes the M12 DINOv3-ViT-L/16 + 3-head model.)

The forward path is the verified VIGIL ``infer_sf09_grouped`` path (S0/shared/
E3/soft-ordinal-Q), vendored under :mod:`mast.vision._vigil` and packaged behind
:mod:`mast.vision._vigil.v25`. This file is the thin adapter that:

  * resolves the checkpoint + backbone cache (dev / PyInstaller-frozen);
  * loads lazily on first inference (construction is cheap);
  * adapts a single ``np.ndarray`` image to the (fwd, bwd) channel pair the
    model expects (a single-channel image fills fwd == bwd → E3 diff 0);
  * maps the six heads onto the VisionModule Pydantic result types.

The 6 heads (real semantics, per INTEGRATION_GUIDE §4 — N/K columns are
historically mis-named): q=quality (soft-ordinal ~60-90, ↑good) · c=4-class seg ·
t=tip instability P (↑bad) · n=multi-apex P(apex≥2) (↑bad, STRONGEST signal) ·
s=apex geometry (axis_ratio∈(0,1] + asym) · k=tip contamination P (↑bad).

Head → API mapping:
  * good/bad coarse label = a fusion of all 6 heads (see ``_fuse_quality``);
    the raw per-head signals are surfaced on ``TipCoarseResult`` (quality_score /
    multi_apex_prob / contam_prob / instability_prob / axis_ratio / asym_prob).
  * ``TipFineResult``: multi_tip ← N, perturbation ← T, morph derived from the
    strongest defect signal; switching/drift not separately modelled by v2.5.
  * Head C (4-class terrace/step/defect/contamination) → ``SegmentationResult``
    (level=1). L0 is classical CV (``segment(level=0)``); v2.5 has no L2 so
    ``segment(level=2)`` downgrades to L1.

Head Q needs the physical scan size. The calling agent supplies it via
``set_scan_size_nm`` (from Nanonis ScanFrameSet metadata); absent that, a
neutral default (``DEFAULT_SCAN_SIZE_NM``) is used and ``R_tip_nm`` should be
read as scale-relative only.

Fail-safe: a missing checkpoint raises ``VIGILCheckpointMissing`` and missing
torch raises ``VIGILDependencyMissing`` — :class:`mast.vision.module.VisionModule`
catches both at construction and falls back to MockBackend so the pipeline
stays callable.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from mast.vision.module import (
    PartialAssessmentResult,
    SegmentationResult,
    TipCoarseResult,
    TipFineResult,
)
from mast.vision.thresholds import decide_fine, fuse_coarse, get_thresholds

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────


class VIGILCheckpointMissing(FileNotFoundError):
    """Raised when the M12 checkpoint cannot be found."""


class VIGILHeadNotInCheckpoint(KeyError):
    """Raised when a method needs a head the checkpoint does not carry."""

    def __init__(self, head_name: str) -> None:
        super().__init__(head_name)
        self.head_name = head_name


class VIGILDependencyMissing(ImportError):
    """Raised when torch / timm / peft aren't importable."""


# ─────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────


class VIGILBackend:
    """M12: frozen DINOv3-ViT-L/16 + LoRA + Head Q / B / C-L1.

    Construction is cheap; the checkpoint + backbone are loaded on the first
    inference call (or eagerly via :meth:`preload`).
    """

    DEFAULT_CKPT = "MASTv2/artifacts/mast_vision_v25.pt"
    DEFAULT_BACKBONE_CACHE = "MASTv2/artifacts/vision_backbone"
    # Head Q's scale conditioning when the operator did not set a scan size.
    # 10 nm is a neutral mid-range STM scan; with it R_tip is scale-relative.
    DEFAULT_SCAN_SIZE_NM = 10.0

    _L1_CLASSES = ["TERRACE", "STEP", "DEFECT", "CONTAMINATION"]
    _MORPH_FINE = ("M0", "M1", "M2", "M3")

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CKPT,
        device: str | None = None,
        backbone_cache_dir: str | Path | None = None,
    ) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            ckpt = self._resolve_resource(ckpt)
        self._ckpt_path = ckpt

        cache = (
            backbone_cache_dir
            or os.environ.get("MAST_VISION_BACKBONE_CACHE", "").strip()
            or self.DEFAULT_BACKBONE_CACHE
        )
        cache = Path(cache)
        if not cache.is_absolute():
            cache = self._resolve_resource(cache)
        self._cache_dir = cache

        self._device_arg = device
        self._device: Any = None
        self._model: Any = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._scan_size_nm: float | None = None
        # Set once the lazy load has been attempted and failed. We then route
        # inference to a torch-free MockBackend (the documented fail-safe) so
        # the pipeline stays callable AND the expensive (1.2 GB) load is not
        # re-attempted on every subsequent inference. The captured exception is
        # re-raised exactly once (the first inference) so the operator sees the
        # real cause; from then on inference returns Mock results, not throws.
        self._load_error: BaseException | None = None
        self._mock: Any = None

    # ── Resource resolution (dev / OTA-pushed / PyInstaller-frozen) ──

    @staticmethod
    def _resolve_resource(relpath: Path) -> Path:
        """Resolve a relative vision-artifact path against the first candidate
        root that actually contains it, covering both deployment forms:

          * dev / OTA install → ``project_root()`` (the user-data dir next to
            the exe; the OTA delta drops ``MASTv2/artifacts/...`` here).
          * PyInstaller-frozen bundle → the unpacked resource tree
            (``sys._MEIPASS``) in case a build bundles the artifacts inside
            ``_internal`` rather than next to the exe.

        If none of the candidates contains the artifact we still return the
        primary (``project_root()``) candidate so the eventual is_file() / is_dir()
        check reports a meaningful, real path — NOT a silently-wrong one that
        would let the backend think it loaded when it did not."""
        from mast._runtime_paths import project_root

        candidates: list[Path] = [project_root() / relpath]
        # CRITICAL (frozen): mast2_build.ps1 Step 4.6 copies the vision artifacts
        # NEXT TO THE EXE (<exe-dir>/MASTv2/artifacts), but run_service overrides
        # ``MAST2_PROJECT_ROOT`` → user_root, so project_root() may point at a
        # SEPARATE user-data dir that does NOT hold the artifacts. Without this
        # candidate the cache isn't found → configure_backbone_cache can't set
        # HF_HUB_OFFLINE → timm tries to DOWNLOAD vits16 → hangs > the launcher's
        # 3-min vision-load timeout (user-reported "视觉模型加载超时"). So always
        # also probe the exe directory when frozen.
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent / relpath)
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(Path(meipass) / relpath)
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    # ── Introspection ───────────────────────────────────────────────

    @property
    def checkpoint_path(self) -> Path:
        return self._ckpt_path

    @property
    def backbone_cache_dir(self) -> Path:
        return self._cache_dir

    def is_loaded(self) -> bool:
        return self._loaded

    def has_head(self, name: str) -> bool:
        """v25 (ssl_sf09c1) carries 6 heads: q/c/t/n/s/k (+ legacy aliases
        b/c_l1 satisfied by the fused tip assessment / C head)."""
        return name.lower() in ("q", "c", "t", "n", "s", "k", "b", "c_l1")

    def preload(self) -> None:
        """Force-load the checkpoint + backbone now (e.g. at process warmup)."""
        self._ensure_loaded()

    # ── Scan-size context (set per inference, not per construction) ──

    def set_scan_size_nm(self, scan_size_nm: float) -> None:
        """Set the scan size used by the next Head Q inference.

        VIGIL conditions Head Q on log10(scan_size_nm); the calling agent
        knows the scan size from the Nanonis ScanFrameSet metadata. Not
        thread-local — VisionModule serialises inference under its _call_lock.
        """
        self._scan_size_nm = float(scan_size_nm)

    def _scan_size(self) -> float:
        s = self._scan_size_nm
        if s is not None and s == s and s > 0:  # not None, not NaN, positive
            return s
        return self.DEFAULT_SCAN_SIZE_NM

    # ── Loading ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # A previous load attempt already failed: re-raise the SAME cached
        # exception cheaply (no second 1.2 GB load attempt). The public API
        # methods catch this and route to the Mock fallback after the first
        # raise — see _fallback_or_raise.
        if self._load_error is not None:
            raise self._load_error
        with self._load_lock:
            if self._loaded:
                return
            if self._load_error is not None:
                raise self._load_error
            try:
                self._do_load()
            except BaseException as exc:  # noqa: BLE001 — cache + surface once
                # Cache so we never re-attempt the expensive load, and build the
                # torch-free Mock the public methods fall back to (documented
                # VisionModule fail-safe). Re-raise this first time so the
                # operator sees the real cause (missing ckpt / torch / corrupt
                # weights); subsequent inferences degrade to Mock, not throw.
                self._load_error = exc
                self._ensure_mock()
                raise

    def _do_load(self) -> None:
        """The actual (expensive) M12 load. Raises on any failure."""
        if not self._ckpt_path.is_file():
            raise VIGILCheckpointMissing(
                f"M12 vision checkpoint not found at {self._ckpt_path}. "
                "MAST does not train these weights — pull them in from the "
                "VIGIL project, or set MAST_VISION_BACKEND=legacy / mock."
            )
        try:
            import torch
        except ImportError as e:
            raise VIGILDependencyMissing(
                "VIGIL M12 backend needs torch (+ timm, peft) installed"
            ) from e

        # Device: MAST_VISION_DEVICE env (auto / cuda / cpu), default "auto"
        # (= cuda if available, else cpu). A capable GPU (e.g. RTX 4060) cold-loads
        # vits16 in ~5 s. Set MAST_VISION_DEVICE=cpu to force CPU on a weak/old-
        # driver GPU where CUDA init is slow or unstable (vits16 on CPU is fine for
        # advisory vision, ~0.5-2 s/inference). A CUDA load failure (OOM / driver
        # mismatch) auto-falls-back to CPU below.
        if self._device_arg is not None:
            device = self._device_arg
        else:
            pref = os.environ.get("MAST_VISION_DEVICE", "auto").strip().lower()
            if pref == "cpu":
                device = "cpu"
            else:  # "auto" (default) or "cuda"
                device = "cuda" if torch.cuda.is_available() else "cpu"

        from mast.vision._vigil import v25

        cache = str(self._cache_dir) if self._cache_dir.is_dir() else None
        if cache is None:
            logger.warning(
                "DINOv3 backbone cache not found at %s; timm will use the "
                "ambient HF cache / attempt a download", self._cache_dir,
            )
        try:
            self._model = v25.load_model(
                self._ckpt_path, device=device, backbone_cache_dir=cache
            )
        except ImportError as e:  # timm / peft missing
            raise VIGILDependencyMissing(str(e)) from e
        except Exception as e:  # noqa: BLE001 — CUDA OOM / driver mismatch → CPU
            if device != "cpu":
                logger.warning(
                    "v25 vision load on %s failed (%s); retrying on CPU "
                    "(advisory vision must not hard-fail).", device, e,
                )
                device = "cpu"
                self._model = v25.load_model(
                    self._ckpt_path, device="cpu", backbone_cache_dir=cache
                )
            else:
                raise
        self._device = device
        self._loaded = True
        logger.info(
            "VIGIL v2.5 (ssl_sf09c1) backend loaded on %s (backbone=%s, lora_rank=%d, heads=%s)",
            device, self._model.backbone_name, self._model.lora_rank, self._model.head_names,
        )

    # ── Mock fallback (documented VisionModule fail-safe) ────────────

    def _ensure_mock(self) -> Any:
        if self._mock is None:
            from mast.vision._mock_backend import MockBackend
            self._mock = MockBackend(
                reason=f"VIGIL M12 load failed ({type(self._load_error).__name__})"
            )
            logger.warning(
                "VIGIL M12 backend falling back to MockBackend after load failure: "
                "%s. Vision is ADVISORY only and now returns fail-safe placeholders.",
                self._load_error,
            )
        return self._mock

    def _fallback_active(self) -> bool:
        """True once a load attempt has failed → public methods use the Mock."""
        return self._load_error is not None and self._mock is not None

    # ── Image adaptation ────────────────────────────────────────────

    @staticmethod
    def _to_fwd_bwd(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Adapt a VisionModule image to the (fwd_pm, bwd_pm) channel pair.

        * (H, W)        → fwd == bwd (E1-degrade; single-channel real image).
        * (2, H, W)     → trace / retrace.
        * (H, W, 3)     → channel-mean → fwd == bwd.
        * (1, H, W)     → squeeze → fwd == bwd.
        """
        a = np.asarray(image)
        if a.ndim == 2:
            f = a.astype(np.float32)
            return f, f
        if a.ndim == 3:
            if a.shape[0] == 2:
                return a[0].astype(np.float32), a[1].astype(np.float32)
            if a.shape[0] == 1:
                f = a[0].astype(np.float32)
                return f, f
            if a.shape[-1] == 3:
                f = a.mean(axis=-1).astype(np.float32)
                return f, f
        raise ValueError(
            f"image must be (H,W), (2,H,W), (1,H,W) or (H,W,3); got {a.shape}"
        )

    @staticmethod
    def _fuse_quality(r: dict) -> tuple[str, float]:
        """Fuse the 6-head signals into a good/bad label + confidence.

        Delegates to the tunable single-source-of-truth
        :func:`mast.vision.thresholds.fuse_coarse` with the process-active
        :class:`~mast.vision.thresholds.VisionThresholds` snapshot (adjustable
        from 设置). With default thresholds this is bit-identical to the
        historical hard-coded fusion: per INTEGRATION_GUIDE §7 the multi-apex
        head (N) is the single STRONGEST quality signal (a multi-apex tip is the
        #1 STM imaging killer), so the "all bad-signals clean" product dominates,
        then soft-ordinal quality (Q) and apex roundness (S axis_ratio) blend in.
        N/K/T are P(bad-condition); ar∈(0,1] with 1=round (good).
        """
        return fuse_coarse(r, get_thresholds())

    def _infer(self, image: np.ndarray) -> dict:
        self._ensure_loaded()
        from mast.vision._vigil import v25

        fwd, bwd = self._to_fwd_bwd(image)
        # Scan size is STICKY across the inferences of one assessment cycle: a
        # single milestone runs assess_tip_coarse + assess_tip_fine + segment
        # (and multiple independent segment calls), all at the SAME scan scale.
        # The old code reset _scan_size_nm=None in a finally after the FIRST
        # inference, so every later inference in the cycle silently fell back to
        # the neutral 10 nm default → wrong Head-Q scale. The
        # caller (VisionModule) sets the size once per scan from ScanFrameSet
        # metadata; it persists until the next set_scan_size_nm.
        return v25.infer(self._model, fwd, bwd, self._scan_size())

    # ── VisionModule API ────────────────────────────────────────────

    def _scan_size_is_explicit(self) -> bool:
        s = self._scan_size_nm
        return s is not None and s == s and s > 0

    def assess_tip_coarse(self, image: np.ndarray) -> TipCoarseResult:
        """6-head fusion → good/bad, + the rich per-head signals.

        label/confidence come from :meth:`_fuse_quality` (Q + S − N − K − T,
        N weighted strongest). The raw per-head signals are surfaced on the
        result (``quality_score`` / ``multi_apex_prob`` / ``contam_prob`` /
        ``instability_prob`` / ``axis_ratio`` / ``asym_prob``) so downstream
        consumers can apply their own thresholds. ``tip_radius_nm`` /
        ``sharpness_log10`` are None — the v2.5 model produces a quality score,
        not an R_tip estimate. ScaleEmbedding still conditions every head on the
        scan size; if the caller set one it is echoed back in ``scan_size_nm``.
        """
        if self._fallback_active():
            return self._mock.assess_tip_coarse(image)
        explicit = self._scan_size_is_explicit()
        scan_used = self._scan_size_nm if explicit else None
        r = self._infer(image)
        label, confidence = self._fuse_quality(r)
        return TipCoarseResult(
            label=label,
            confidence=min(1.0, max(0.0, confidence)),
            embedding_sha=r["cls_sha"],
            scan_size_nm=scan_used,
            tip_radius_nm=None,
            sharpness_log10=None,
            quality_score=float(r["q_score"]),
            multi_apex_prob=min(1.0, max(0.0, float(r["n_p"]))),
            contam_prob=min(1.0, max(0.0, float(r["k_p"]))),
            instability_prob=min(1.0, max(0.0, float(r["t_p"]))),
            axis_ratio=min(1.0, max(0.0, float(r["s_axis_ratio"]))),
            asym_prob=min(1.0, max(0.0, float(r["s_asym_p"]))),
        )

    def assess_tip_fine(self, image: np.ndarray) -> TipFineResult:
        """Map the 6 heads onto the fine tip morphology contract.

        v2.5 decomposes the old monolithic Head B into orthogonal heads:
          multi_tip   ← N  (P(apex≥2), the directly-modelled multi-tip signal)
          perturbation← T  (tip unstable during scan)
          switching/drift → None (not separately modelled by v2.5)
        morph is derived heuristically from the strongest defect signal:
          multi-apex → M2; contaminated/unstable → M3; clean+sharp → M0; else M1.
        n_tips estimate = 1 + P(apex≥2) (≈1 single, →2 as multi-apex grows).
        """
        if self._fallback_active():
            return self._mock.assess_tip_fine(image)
        r = self._infer(image)
        n_p = float(r["n_p"])
        # Morphology / usability cuts are tunable (设置) via the shared
        # single-source-of-truth decision fn; default thresholds reproduce the
        # historical n_p/t_p/k_p>0.5, q≥72, ar≥0.6 logic exactly.
        d = decide_fine(r, get_thresholds())
        morph = d["morph"]
        return TipFineResult(
            label=morph,
            top2=[(morph, 1.0)],
            is_usable=d["is_usable"],
            morph=morph,
            switching=None,
            drift=None,
            perturbation=d["perturbation"],
            multi_tip=d["multi_tip"],
            n_tips=1.0 + max(0.0, min(1.0, n_p)),
        )

    # Above this max(H,W), level-1 segmentation auto-tiles (sliding window) so
    # peak VRAM stays bounded on small GPUs (e.g. GTX 1650, 4 GB) — the standard
    # path otherwise resizes the whole image to 256, losing native detail.
    AUTO_TILE_THRESHOLD = 512
    TILE_SIZE = 256

    def segment(
        self,
        image: np.ndarray,
        classes: list[str] | None = None,
        *,
        level: int | None = None,
        tile: int | None = None,
    ) -> SegmentationResult:
        """Head C: segmentation.

        level 0 → terrace mask via classical CV (never loads the network).
        level 1 (default) → 4-class terrace/step/defect/contamination.
        level 2 → M12 has no 27-class head; downgrades to L1 (result.level == 1).

        ``tile`` controls high-resolution sliding-window tiling for large scans
        (keeps peak VRAM bounded — see segment_large):
          * None  → AUTO: tile (size TILE_SIZE) when max(H,W) > AUTO_TILE_THRESHOLD,
                    else the standard 256-resize path.
          * 0     → never tile (always the standard resize path).
          * N > 0 → force tiling with tile size N.
        """
        target = 1 if level is None else level
        if target == 0:
            return self._segment_l0(image)  # classical CV — never the network
        if target == 2:
            logger.warning(
                "M12 has no Head C-L2 (27-class); downgrading segment(level=2) to L1"
            )
            target = 1
        if target != 1:
            raise ValueError(f"level must be 0, 1, or 2; got {level}")

        # After a failed model load, L1 falls back to the network-free classical
        # 4-class segmenter (mast.vision.classical_seg) — strictly better than an
        # empty mask, and on synthetic GT it beats the learned Head C anyway.
        if self._fallback_active():
            return self._segment_classical_result(image, 1)

        tile_size = self._resolve_tile(image, tile)
        if tile_size:
            return self._segment_l1_tiled(image, tile_size)
        return self._segment_l1(image)

    def _resolve_tile(self, image: np.ndarray, tile: int | None) -> int:
        """Return the tile size to use (0 = no tiling) from the tile arg + size."""
        if tile is not None:
            return max(0, int(tile))
        try:
            hw = self._input_hw(image)
            if hw is not None and max(hw[0], hw[1]) > self.AUTO_TILE_THRESHOLD:
                return self.TILE_SIZE
        except Exception:  # noqa: BLE001
            pass
        return 0

    @staticmethod
    def _input_hw(image: np.ndarray) -> tuple[int, int] | None:
        """Original (H, W) of a VisionModule image (the geometry the caller
        wants the segmentation mask reported in). Returns None on odd shapes."""
        a = np.asarray(image)
        if a.ndim == 2:
            return int(a.shape[0]), int(a.shape[1])
        if a.ndim == 3:
            if a.shape[0] in (1, 2):          # (C, H, W) trace/retrace or single
                return int(a.shape[1]), int(a.shape[2])
            if a.shape[-1] == 3:              # (H, W, 3)
                return int(a.shape[0]), int(a.shape[1])
        return None

    def partial_assess(
        self, scan_lines: np.ndarray, n_available: int
    ) -> PartialAssessmentResult:
        """No Head D in M12 → assess the partial image with Head B coarse;
        ``self_consistency`` stays None (matches the legacy contract)."""
        if not hasattr(scan_lines, "shape") or scan_lines.ndim < 2:
            return PartialAssessmentResult(
                quality_pred=0.5, coarse_label="unknown", frac_acquired=0.0
            )
        total = int(scan_lines.shape[0])
        n_avail = int(max(0, min(n_available, total)))
        frac = n_avail / total if total else 0.0
        if n_avail == 0:
            return PartialAssessmentResult(
                quality_pred=0.5, coarse_label="unknown", frac_acquired=0.0
            )
        partial = np.zeros_like(scan_lines)
        partial[:n_avail] = scan_lines[:n_avail]
        coarse = self.assess_tip_coarse(partial)
        quality_pred = (
            coarse.confidence if coarse.label == "good" else 1 - coarse.confidence
        )
        return PartialAssessmentResult(
            quality_pred=min(1.0, max(0.0, quality_pred)),
            coarse_label=coarse.label,
            frac_acquired=frac,
            self_consistency=None,
        )

    # ── Head C helpers ──────────────────────────────────────────────

    def _seg_result(self, seg: np.ndarray) -> SegmentationResult:
        from mast.vision.seg_utils import encode_rle

        counts = {
            cls_name: int((seg == i).sum())
            for i, cls_name in enumerate(self._L1_CLASSES)
        }
        return SegmentationResult(
            mask_rle=encode_rle(seg),
            shape=tuple(seg.shape),  # type: ignore[arg-type]
            class_counts=counts,
            level=1,
            classes=list(self._L1_CLASSES),
            tipflag_stability_rle=b"",
            tipflag_transition_rle=b"",
        )

    def _segment_l1(self, image: np.ndarray) -> SegmentationResult:
        r = self._infer(image)
        seg = np.asarray(r["seg_map"])  # model-input resolution (256×256), 0..3
        # m12.infer resizes the image to the 256-px model input, so seg_map is
        # 256×256 — NOT the scan's native pixel grid. Reporting that shape lets
        # downstream coordinate scaling (RegionMap → GUI overlay, feature
        # locations) map mask pixels onto the wrong scan coordinates. Resize the
        # label map back to the scan's original (H, W) with nearest-neighbour
        # (exact class labels, no fractional codes) so the mask is reported in
        # the caller's resolution. Falls back to the raw map if the input shape
        # is unusual.
        hw = self._input_hw(image)
        if hw is not None and tuple(seg.shape) != hw:
            from mast.vision._legacy_wrapper import _resize_mask_nn
            seg = _resize_mask_nn(seg.astype(np.uint8), hw)
        return self._seg_result(seg.astype(np.uint8))

    def _segment_l1_tiled(self, image: np.ndarray, tile: int) -> SegmentationResult:
        """v2.5 processes at native resolution (native_res=True, up to size=512)
        so the M12 sliding-window tiling is not needed — defer to the standard
        L1 path. (Kept as a method so the AUTO/forced-tile arg still resolves to
        a working segmentation rather than erroring.)"""
        return self._segment_l1(image)

    def _nm_per_px(self, image: np.ndarray) -> float | None:
        """Physical pixel size (scan_size_nm / pixels) if the scan size is known
        — lets the classical segmenter/metrics pick the atomic length scale."""
        s = self._scan_size_nm
        if not s or s <= 0:
            return None
        hw = self._input_hw(image)
        n = hw[0] if hw is not None else int(np.asarray(image).shape[-1])
        return (float(s) / float(n)) if n else None

    def _segment_classical_result(self, image: np.ndarray, level: int) -> SegmentationResult:
        """Network-free 4-class classical segmentation (terrace/step/defect/contam
        — see :mod:`mast.vision.classical_seg`). Beats the learned Head C on
        synthetic GT and never hallucinates contamination on clean lattices. Used
        for level 0 and as the level-1 model-down fallback."""
        from mast.vision.classical_seg import segment_classical_result

        return segment_classical_result(image, self._nm_per_px(image), level)

    def _segment_l0(self, image: np.ndarray) -> SegmentationResult:
        """Head C level 0 — classical CV, never touches the network. Now the
        full 4-class terrace/step/defect/contam segmenter (was terrace-only)."""
        return self._segment_classical_result(image, 0)


__all__ = [
    "VIGILBackend",
    "VIGILCheckpointMissing",
    "VIGILHeadNotInCheckpoint",
    "VIGILDependencyMissing",
]
