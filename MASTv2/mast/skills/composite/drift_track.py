"""TrackDrift_ReferenceScan — periodic reference scan drift tracking (Phase 7 graph migration).

Semi-dynamic plan: the *number* of yielded steps is small and known up
front (SetBias + FullScan, optionally followed by a drift-compensation
ConfigureScan), but the decision whether to yield the final step depends
on the cross-correlation between the just-acquired image and a stored
reference. Pure analysis (image grab + load + xcorr) happens between
yields, not as sub-skills, because v1 drift_track used direct
``safe_call("Scan_FrameDataGrab", ...)`` and numpy/scipy in-process.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

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


class TrackDrift_ReferenceScan(CompositeSkillGraph):
    """Track drift by comparing current scan to a reference.

    Takes a scan at the reference position, compares with stored
    reference using cross-correlation, and compensates the drift
    by adjusting the scan frame via SetDriftCompensation.

    Reference: DeepSPM drift tracking pattern.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TrackDrift_ReferenceScan",
            version="2.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "与参考扫描比对来跟踪样品漂移，并按测得的漂移补偿扫描框位置。"
            ),
            parameters=[
                ParameterSpec(
                    name="ref_x_m",
                    type="float",
                    description="参考扫描中心 X",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="ref_y_m",
                    type="float",
                    description="参考扫描中心 Y",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="ref_width_m",
                    type="float",
                    description="参考扫描宽度",
                    unit="m",
                    required=False,
                    default=20e-9,
                    min_value=1e-10,
                ),
                ParameterSpec(
                    name="bias_v",
                    type="float",
                    description="参考扫描用的偏压",
                    unit="V",
                    required=False,
                    default=-0.5,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="ref_image_path",
                    type="str",
                    description="参考图路径（.npy）。留空则现采一张新的参考图。",
                    required=False,
                    default="",
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=60.0,
            composition_level=3,
            tags=["composite", "drift", "tracking", "reference"],
        )

    # ------------------------------------------------------------------
    # Dynamic plan
    # ------------------------------------------------------------------

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        ref_x = float(params["ref_x_m"])
        ref_y = float(params["ref_y_m"])
        width = float(params.get("ref_width_m", 20e-9))
        bias = float(params.get("bias_v", -0.5))
        ref_path = params.get("ref_image_path", "") or ""

        # 1. Set bias for reference scan (mirror v1: failure was not fatal —
        #    used self.step(), not step_or_fail).
        yield CompositeStep(
            step_id="set_bias",
            skill_name="SetBias",
            params={"bias_v": bias},
            optional=True,
            checkpoint_after=False,
            tags=("setup",),
        )
        if executor.progress.aborted:
            return

        # 2. Reference-area scan (mandatory)
        yield CompositeStep(
            step_id="ref_scan",
            skill_name="FullScan",
            params={
                "center_x_m": ref_x,
                "center_y_m": ref_y,
                "width_m": width,
                "height_m": width,
                "line_time_s": 0.1,
            },
            optional=False,
            checkpoint_after=True,
            tags=("scan",),
        )
        if executor.progress.aborted:
            return

        # 3. Grab the current image (NOT a sub-skill — direct safe_call).
        current_image = self._grab_scan_image(self._context)

        # 4. If no reference image, this IS the reference (first call). SAVE the
        #    grabbed image and hand back its path — the old code discarded
        #    current_image and only said "call again with ref_image_path", but
        #    never produced one, so the workflow was a dead end (review
        #    2026-07-03).
        if not ref_path:
            if current_image is None:
                executor.set_partial("ref_image_path", None)
                executor.set_partial(
                    "message",
                    "Could not grab a reference image (Scan_FrameDataGrab "
                    "returned no usable 2-D data) — cannot start drift tracking.",
                )
                executor.abort(reason="reference scan grab failed")
                return
            try:
                import time as _t
                from mast._runtime_paths import project_root
                frames = project_root() / "experiments" / "frames"
                frames.mkdir(parents=True, exist_ok=True)
                ref_file = frames / f"drift_ref_{int(_t.time() * 1000)}.npy"
                np.save(ref_file, np.asarray(current_image, dtype=np.float64))
                ref_saved = str(ref_file)
            except Exception as exc:
                executor.set_partial("ref_image_path", None)
                executor.set_partial(
                    "message", f"Reference grabbed but could not be saved: {exc}")
                executor.abort(reason=f"failed to save reference image: {exc}")
                return
            executor.set_partial("ref_image_path", ref_saved)
            executor.set_partial("drift_x_m", 0.0)
            executor.set_partial("drift_y_m", 0.0)
            executor.set_partial("compensated", False)
            executor.set_partial(
                "message",
                f"Reference captured and saved to {ref_saved}. Call again with "
                f"ref_image_path='{ref_saved}' to track drift.",
            )
            return

        # 5. Load reference image.
        try:
            ref_image = np.load(ref_path).astype(np.float64)
        except Exception as exc:
            executor.abort(reason=f"Failed to load reference: {exc}")
            return

        # 6. Compute drift via cross-correlation.
        drift_x_m, drift_y_m = self._compute_drift(
            ref_image, current_image, width,
        )
        executor.set_partial("drift_x_m", drift_x_m)
        executor.set_partial("drift_y_m", drift_y_m)
        executor.set_partial("compensated", False)

        # 7. Apply compensation if drift is significant.
        if abs(drift_x_m) > 1e-12 or abs(drift_y_m) > 1e-12:
            yield CompositeStep(
                step_id="apply_compensation",
                skill_name="ConfigureScan",
                params={
                    "center_x_m": ref_x + drift_x_m,
                    "center_y_m": ref_y + drift_y_m,
                    "width_m": width,
                    "height_m": width,
                },
                # v1 used self.step() (not step_or_fail) so a failure
                # here didn't abort the composite — drift was still
                # reported, just compensated=False.
                optional=True,
                checkpoint_after=True,
                tags=("compensation",),
            )
            if executor.progress.aborted:
                return
            comp_result = executor.sub_results.get("apply_compensation")
            if comp_result is not None and getattr(
                    comp_result, "success", False):
                executor.set_partial("compensated", True)

    # ------------------------------------------------------------------
    # Aggregate + helpers
    # ------------------------------------------------------------------

    def aggregate(
        self, sub_results: dict, progress: CompositeProgress,
    ) -> dict:
        data = {
            "drift_x_m": float(progress.partial_data.get("drift_x_m", 0.0)),
            "drift_y_m": float(progress.partial_data.get("drift_y_m", 0.0)),
            "compensated": bool(
                progress.partial_data.get("compensated", False)),
        }
        # Surface the saved reference path so the caller can pass it on the next
        # call (first-call reference capture).
        if "ref_image_path" in progress.partial_data:
            data["ref_image_path"] = progress.partial_data.get("ref_image_path")
        msg = progress.partial_data.get("message")
        if msg:
            data["message"] = msg
        return data

    def run_composite(self, context, params: dict) -> SkillResult:
        # Stash context for helpers (image grab) that run between yields.
        self._context = context
        return self._graph_execute(context, params)

    def _grab_scan_image(self, context) -> np.ndarray | None:
        """Grab current scan frame data from hardware as a 2-D image.

        Uses the shared robust parse (the old np.asarray(body) raised
        "inhomogeneous shape" on every real scan) and returns a 2-D array —
        _compute_drift's scipy.signal.correlate2d needs 2-D, not the raveled 1-D
        the old code produced."""
        try:
            from mast.io.nanonis_files import parse_frame_grab
            rec = context.safe_call("Scan_FrameDataGrab", 0, 1)
            self._all_calls.append(rec)
            if rec.error or rec.return_value is None:
                return None
            return parse_frame_grab(rec.return_value, shape_2d=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _compute_drift(
        ref_image: np.ndarray,
        current_image: np.ndarray | None,
        scan_width_m: float,
    ) -> tuple[float, float]:
        """Compute drift between reference and current image via cross-correlation."""
        if current_image is None:
            return 0.0, 0.0

        try:
            from scipy.signal import correlate2d

            # Normalize images
            ref = ref_image - ref_image.mean()
            if current_image.size != ref_image.size:
                logger.warning(
                    "Image size mismatch: ref=%d, cur=%d",
                    ref_image.size, current_image.size,
                )
                return 0.0, 0.0
            cur = current_image.reshape(ref_image.shape) - current_image.mean()

            # Cross-correlation
            corr = correlate2d(ref, cur, mode="same")
            peak = np.unravel_index(np.argmax(corr), corr.shape)
            center = (ref.shape[0] // 2, ref.shape[1] // 2)

            # Convert pixel shift to meters
            dy_px = peak[0] - center[0]
            dx_px = peak[1] - center[1]
            pixel_size = scan_width_m / ref.shape[1] if ref.shape[1] > 0 else 1e-9
            return dx_px * pixel_size, dy_px * pixel_size
        except Exception as exc:
            logger.warning("Cross-correlation failed: %s", exc)
            return 0.0, 0.0


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(TrackDrift_ReferenceScan, context_provider)
