"""AssessImageQuality — FFT quality + roughness + noise analysis (Phase 7 graph-shaped composite).

Migrated from v1 `mast/skills/composite/assess_quality.py` to the unified
``CompositeSkillGraph`` framework. The original analysis is a single-skill
pipeline (load → FFT → RMS → noise → SSIM → training sample) with no real
sub-skills, so this composite uses a private :class:`_PhaseCtx` wrapper that
turns each internal phase into a graph step. The graph framework still gives
us:

  • progress emission per phase (visible in the GUI live progress panel)
  • resume on re-invocation (skip phases already completed in
    ``composite_progress``)
  • per-phase checkpoint_flush hooks (SqliteSaver writes after each step)

The wrapper short-circuits ``ctx.run("_phase_*", params)`` and dispatches to
internal compute methods so the executor's standard machinery (per-step
emit / checkpoint / on_step_result) is exercised verbatim.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.agents._shared.skill_adapter import wrap_skill

logger = logging.getLogger(__name__)


# Synthetic phase identifiers — these are NOT registered in SkillRegistry;
# they are intercepted by :class:`_PhaseCtx` below.
_PHASE_LOAD = "_phase_load"
_PHASE_FFT = "_phase_fft"
_PHASE_ROUGHNESS = "_phase_roughness"
_PHASE_NOISE = "_phase_noise"
_PHASE_SSIM = "_phase_ssim"
_PHASE_TRAINING = "_phase_training_sample"


class _PhaseCtx:
    """Wraps a real ExecutionContext to handle synthetic ``_phase_*`` skills.

    All non-phase attribute access (``safe_call``, ``check_abort``,
    ``emit_progress``, ``get_progress``, ``checkpoint_flush``, ``state``,
    etc.) delegates unchanged to the underlying context. ``run()`` is the
    only intercept point.
    """

    def __init__(self, real_ctx, skill: "AssessImageQuality") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class AssessImageQuality(CompositeSkillGraph):
    """Evaluate the quality of the latest (or specified) scan image.

    Computes:
    - FFT quality score (periodic structure indicator, 0-1)
    - RMS roughness (m)
    - Noise estimate (sigma from MAD of Laplacian)
    - Optional: SSIM forward/backward comparison
    - Optional: training sample auto-collection

    Reuses ``mast.data.quality.{fft_quality_score, rms_roughness, noise_estimate}``.
    Based on the BO-for-AutoSTM FFT method.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessImageQuality",
            version="2.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "评估扫描图像质量:FFT 打分、RMS 粗糙度、噪声估计。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path",
                    type="str",
                    description=".sxm 文件路径(缺省:最近一张扫描图)",
                    required=False,
                    default="",
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=2,
            tags=["analysis", "quality", "fft", "composite"],
        )

    # ------------------------------------------------------------------
    # Plan: 4 mandatory phases + 2 optional phases.
    # ------------------------------------------------------------------

    def plan(self, params: dict) -> list[CompositeStep]:
        scan_path = params.get("scan_path", "")
        steps: list[CompositeStep] = [
            CompositeStep(
                step_id="load",
                skill_name=_PHASE_LOAD,
                params={"scan_path": scan_path},
                optional=False,
                checkpoint_after=False,
                tags=("io",),
            ),
            CompositeStep(
                step_id="fft",
                skill_name=_PHASE_FFT,
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("compute", "fft"),
            ),
            CompositeStep(
                step_id="roughness",
                skill_name=_PHASE_ROUGHNESS,
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("compute", "roughness"),
            ),
            CompositeStep(
                step_id="noise",
                skill_name=_PHASE_NOISE,
                params={},
                optional=False,
                checkpoint_after=True,   # core metrics done — flush
                tags=("compute", "noise"),
            ),
            CompositeStep(
                step_id="ssim",
                skill_name=_PHASE_SSIM,
                params={},
                optional=True,           # non-critical
                checkpoint_after=False,
                tags=("compute", "ssim"),
            ),
            CompositeStep(
                step_id="training_sample",
                skill_name=_PHASE_TRAINING,
                params={},
                optional=True,           # non-critical
                checkpoint_after=False,
                tags=("io", "training"),
            ),
        ]
        return steps

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_step_result(self, step: CompositeStep, sub_result: SkillResult) -> None:
        """Stash compute results into partial_data via executor.set_partial."""
        if not sub_result.success or not sub_result.data:
            return
        for key, value in sub_result.data.items():
            self._executor.set_partial(key, value)

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        """Build the final SkillResult.data — only the keys the v1 caller expects."""
        partial = progress.partial_data
        data: dict[str, Any] = {}
        # Mandatory keys (v1 contract)
        for key in ("fft_quality", "rms_roughness_m", "noise_sigma", "scan_path"):
            if key in partial:
                data[key] = partial[key]
        # Optional keys (only present when their phase succeeded)
        if "fwd_bwd_ssim" in partial:
            data["fwd_bwd_ssim"] = partial["fwd_bwd_ssim"]
        return data

    # ------------------------------------------------------------------
    # Driver: wrap context so the executor can dispatch _phase_* steps.
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        # Reset transient in-memory state. ``self._image`` is reconstructed
        # from ``progress.partial_data['scan_path']`` below when resuming so
        # downstream compute phases see the same numpy array even though
        # ``load`` is skipped.
        self._image = None

        wrapped = _PhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        self._executor = executor

        # Resume case: prior progress contains a scan_path → reload the image
        # so later compute phases have something to work with. If reload
        # fails we leave ``self._image = None`` and the next compute phase
        # will fail loudly (better than silently producing garbage).
        if executor.is_completed("load"):
            prior_path = executor.progress.partial_data.get("scan_path")
            if prior_path:
                try:
                    self._image = self._load_sxm_image(prior_path)
                except Exception:
                    logger.exception(
                        "AssessImageQuality resume: failed to reload %s",
                        prior_path,
                    )

        plan_iter = iter(self.plan(params))
        all_good = executor.run_plan(plan_iter)
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        if all_good:
            return self.ok(**data)
        return self.fail(
            executor.progress.aborted_reason or "AssessImageQuality aborted",
            **data,
        )

    # ------------------------------------------------------------------
    # Phase dispatch — invoked by _PhaseCtx.run()
    # ------------------------------------------------------------------

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        try:
            if skill_name == _PHASE_LOAD:
                return self._phase_load(params, real_ctx)
            if skill_name == _PHASE_FFT:
                return self._phase_fft()
            if skill_name == _PHASE_ROUGHNESS:
                return self._phase_roughness()
            if skill_name == _PHASE_NOISE:
                return self._phase_noise()
            if skill_name == _PHASE_SSIM:
                return self._phase_ssim()
            if skill_name == _PHASE_TRAINING:
                return self._phase_training_sample(real_ctx)
        except Exception as exc:
            logger.exception("AssessImageQuality phase %s raised", skill_name)
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return SkillResult(
            skill_name=skill_name,
            success=False,
            error=f"Unknown phase: {skill_name}",
        )

    # ----- individual phase implementations -----

    def _phase_load(self, params: dict, real_ctx) -> SkillResult:
        scan_path = params.get("scan_path", "")
        if not scan_path:
            scan_path = self._find_latest_scan(real_ctx)
        if not scan_path:
            return SkillResult(
                skill_name=_PHASE_LOAD,
                success=False,
                error="No scan file found. Provide scan_path or run a scan first.",
            )
        # Validate existence BEFORE loading: an LLM can
        # pass a hallucinated / stale scan_path (e.g. a previous Nanonis session
        # folder that no longer exists), which otherwise surfaces as a cryptic
        # WinError from the loader. Fail with a clear, actionable message so the
        # model self-corrects (run a scan first) instead of retrying the bad path.
        # Goes through _scan_file_exists (a stubbable I/O seam, like
        # _load_sxm_image) so tests that mock the loader can mock existence too.
        if not self._scan_file_exists(scan_path):
            return SkillResult(
                skill_name=_PHASE_LOAD,
                success=False,
                error=(f"Scan file does not exist: {scan_path}. "
                       "It may be a stale path from a previous session — run a "
                       "scan first, or pass an existing scan_path."),
            )
        try:
            self._image = self._load_sxm_image(scan_path)
        except Exception as exc:
            return SkillResult(
                skill_name=_PHASE_LOAD,
                success=False,
                error=f"Failed to load {scan_path}: {exc}",
            )
        return SkillResult(
            skill_name=_PHASE_LOAD,
            success=True,
            data={"scan_path": str(scan_path)},
        )

    def _phase_fft(self) -> SkillResult:
        if self._image is None:
            return SkillResult(
                skill_name=_PHASE_FFT, success=False,
                error="No image loaded",
            )
        from mast.data.quality import fft_quality_score
        score = fft_quality_score(self._image)
        return SkillResult(
            skill_name=_PHASE_FFT, success=True,
            data={"fft_quality": score},
        )

    def _phase_roughness(self) -> SkillResult:
        if self._image is None:
            return SkillResult(
                skill_name=_PHASE_ROUGHNESS, success=False,
                error="No image loaded",
            )
        from mast.data.quality import rms_roughness
        rough = rms_roughness(self._image)
        return SkillResult(
            skill_name=_PHASE_ROUGHNESS, success=True,
            data={"rms_roughness_m": rough},
        )

    def _phase_noise(self) -> SkillResult:
        if self._image is None:
            return SkillResult(
                skill_name=_PHASE_NOISE, success=False,
                error="No image loaded",
            )
        from mast.data.quality import noise_estimate
        sigma = noise_estimate(self._image)
        return SkillResult(
            skill_name=_PHASE_NOISE, success=True,
            data={"noise_sigma": sigma},
        )

    def _phase_ssim(self) -> SkillResult:
        scan_path = self._executor.progress.partial_data.get("scan_path", "")
        if not scan_path:
            return SkillResult(
                skill_name=_PHASE_SSIM, success=False,
                error="No scan_path available",
            )
        ssim = self._compute_fwd_bwd_ssim(scan_path)
        if ssim is None:
            return SkillResult(
                skill_name=_PHASE_SSIM, success=False,
                error="SSIM not available (single-channel scan?)",
            )
        return SkillResult(
            skill_name=_PHASE_SSIM, success=True,
            data={"fwd_bwd_ssim": ssim},
        )

    def _phase_training_sample(self, real_ctx) -> SkillResult:
        fft_score = self._executor.progress.partial_data.get("fft_quality")
        if self._image is None or fft_score is None:
            return SkillResult(
                skill_name=_PHASE_TRAINING, success=False,
                error="Missing image or fft_quality",
            )
        self._record_training_sample(real_ctx, self._image, fft_score)
        return SkillResult(
            skill_name=_PHASE_TRAINING, success=True,
            data={},
        )

    # ------------------------------------------------------------------
    # Internal helpers — verbatim from v1 except for the call signatures.
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fwd_bwd_ssim(scan_path: str) -> float | None:
        """Compute SSIM between forward and backward Z channels."""
        import numpy as np

        try:
            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
            parsed = read_sxm(scan_path)
        except Exception:
            return None

        # 2026-08-11 两处修正，都经 `sxm_oriented_frames`（几何归位的单一真源）：
        #
        # 1. **反扫要左右翻转**。.sxm 的反扫从右往左采，存下来是镜像；原来这里
        #    直接拿裸块喂 SSIM，量的是一张图和它自己的镜像。
        # 2. **正反扫必须来自同一个通道**。原来的循环里 `fwd` 和 `bwd` 是各自独立
        #    赋值的：Z 只有正扫时 `fwd` 取自 Z，下一轮 `bwd` 就可能取自 Current，
        #    然后 break —— 拿 Z 的正扫和 Current 的反扫算 SSIM，量纲都不同。
        #
        # ⚠️ 仍未修的一点，**刻意留着**：SSIM 是**零位移逐点**判据，而正反扫之间
        # 有压电迟滞造成的快轴偏移（属于正常范围）。所以这个
        # 数会系统性偏低。要判「正反扫重不重合」请用 `CheckLineQuality`
        # （`trace_retrace_correlation`，允许平移）；这里这个数只上报、不判定，
        # 换判据是另一件事，不混在这次修正里。
        channels = parsed.get("channels") or {}
        fwd = bwd = None
        for ch_name in ("Z", *channels.keys()):
            if not isinstance(channels.get(ch_name), dict):
                continue
            oriented = sxm_oriented_frames(parsed, ch_name)
            f, b = oriented.get("forward"), oriented.get("backward")
            if f is not None and b is not None:
                fwd = np.asarray(f, dtype=np.float64)
                bwd = np.asarray(b, dtype=np.float64)
                break

        if fwd is None or bwd is None or fwd.shape != bwd.shape:
            return None

        try:
            from skimage.metrics import structural_similarity
            return float(structural_similarity(fwd, bwd, data_range=np.ptp(fwd)))
        except ImportError:
            mu_f = np.mean(fwd)
            mu_b = np.mean(bwd)
            sig_f = np.std(fwd)
            sig_b = np.std(bwd)
            sig_fb = np.mean((fwd - mu_f) * (bwd - mu_b))
            c1 = (0.01 * np.ptp(fwd)) ** 2
            c2 = (0.03 * np.ptp(fwd)) ** 2
            ssim = ((2 * mu_f * mu_b + c1) * (2 * sig_fb + c2)) / \
                   ((mu_f ** 2 + mu_b ** 2 + c1) * (sig_f ** 2 + sig_b ** 2 + c2))
            return float(ssim)

    @staticmethod
    def _record_training_sample(context, image, fft_score: float) -> None:
        """Auto-collect training data: save image with quality label."""
        if not hasattr(context, "executor"):
            return
        planner = getattr(context.executor, "_planner", None)
        if planner is None:
            return
        exp_log = getattr(planner, "_experiment_log", None)
        if exp_log is None or not hasattr(exp_log, "record_training_sample"):
            return
        label = "good_scan" if fft_score > 0.3 else "bad_scan"
        exp_log.record_training_sample(image, label, "AssessImageQuality")

    @staticmethod
    def _find_latest_scan(context) -> str | None:
        """Find the most recent .sxm file the instrument actually wrote.

        The v2 ExecutionContext has NO ``_session_path`` / ``executor`` attributes
        (those were v1 planner internals), so the old lookup resolved to
        ``MASTv2/mast/experiments`` — a directory that does not exist — and every
        AssessImageQuality returned 0.0. That silently broke the ConditionTip
        closed loop: quality was always 0, so it blindly fired pulses until
        max_attempts and then failed (2026-07-03 review).

        Reuse the LIVE session-dir discovery shared with GetLatestScanFile /
        SaveScan: it resolves Nanonis's real save dir via Util_SessionPathGet off
        the context's pool (plus working-sessions fallbacks)."""
        try:
            from mast.skills.builtins.scan_extra import (
                _candidate_save_dirs, _find_latest_sxm,
            )
            latest = _find_latest_sxm(_candidate_save_dirs(context))
            if latest is not None:
                return str(latest)
        except Exception:  # pragma: no cover - fall through to legacy lookup
            pass
        # Legacy fallback: repo-root experiments/ + any GUI-captured session path.
        try:
            from mast.webui.scan_preview import get_latest_scan
            from mast._runtime_paths import project_root
            experiments_dir = str(project_root() / "experiments")
            session_path = (getattr(context, "session_path", None)
                            or getattr(context, "_session_path", None))
            return get_latest_scan(experiments_dir, session_path, None)
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _scan_file_exists(path: str) -> bool:
        """Stubbable existence check for a scan path (F2). Separated from the
        loader so tests that mock _load_sxm_image can mock existence too."""
        try:
            return Path(path).exists()
        except (OSError, ValueError):
            return False

    @staticmethod
    def _load_sxm_image(path: str):
        """Load the first forward Z channel from an .sxm file."""
        import numpy as np

        from mast.io.nanonis_files import read_sxm
        parsed = read_sxm(path)
        channels = parsed.get("channels") or {}
        for prefer in ("Z",):
            ch = channels.get(prefer)
            if isinstance(ch, dict) and ch.get("forward") is not None:
                return np.asarray(ch["forward"], dtype=np.float64)
        for name, ch in channels.items():
            if isinstance(ch, dict) and ch.get("forward") is not None:
                return np.asarray(ch["forward"], dtype=np.float64)
        for name, ch in channels.items():
            if isinstance(ch, dict):
                for key in ("backward",):
                    if ch.get(key) is not None:
                        return np.asarray(ch[key], dtype=np.float64)
        raise RuntimeError(f"Could not parse image data from {path}")


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(AssessImageQuality, context_provider)
