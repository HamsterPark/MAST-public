"""DemoScanAndSTS — one-shot demo workflow: scan + STS, no tip checks (Phase 7 graph-shaped composite).

Migrated from v1 ``mast/skills/composite/demo_scan_and_sts.py`` to the
unified ``CompositeSkillGraph`` framework (v0.3.11 demo button). Built for
live demos where the operator wants a guaranteed sequence without LLM
second-guessing. Skips AssessImageQuality, ConditionTip, TipShape,
PreScanCheck — runs ConfigureScan → SetScanSpeed → StartScan → WaitScanComplete
→ SaveScan → ConfigureSTS → (MoveToXY + AcquireSTS) × sts_count.

All parameters have safe hard-coded defaults so the LLM only needs to emit
one tool call. The migration replaces the v1 ``wait_scan_complete()`` helper
with the registered ``WaitScanComplete`` skill so every action shows up as a
real graph step (and thus is visible to progress / resume / checkpoint).
"""

from __future__ import annotations

import logging
import math
from typing import Iterator

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


class DemoScanAndSTS(CompositeSkillGraph):
    """Demo-mode one-shot: scan + multi-point STS, ignoring tip quality.

    Defaults: 50 nm x 50 nm scan at origin, then 5 STS spectra (center +
    4 corners of an inner 20 nm square). All defaults are safe; the
    skill runs end-to-end without any LLM-side decisions.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="DemoScanAndSTS",
            version="2.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "**仅供演示** —— 用预设参数做一次「扫描 + STS」。"
                "**跳过全部针尖质量检查**（不做 AssessImageQuality、"
                "ConditionTip、TipShape、PreScanCheck）。只用于现场演示，"
                "且用户已经知道针尖是好的。扫描存盘、STS 谱采完之后返回。"
            ),
            parameters=[
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description=(
                        "扫描中心 X，单位 METERS。默认 0。"
                        "常见 -1.5u 到 1.5u（+/-1.5 um）。"
                    ),
                    unit="m",
                    required=False,
                    default=0.0,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="扫描中心 Y，单位 METERS。默认 0。",
                    unit="m",
                    required=False,
                    default=0.0,
                    min_value=-1.5e-6,
                    max_value=1.5e-6,
                ),
                ParameterSpec(
                    name="scan_size_m",
                    type="float",
                    description=(
                        "方形扫描的边长，单位 METERS。默认 50n"
                        "（50 nm）。硬上限 10u（10 um）。"
                    ),
                    unit="m",
                    required=False,
                    default=50e-9,
                    min_value=1e-9,
                    max_value=1e-5,
                ),
                ParameterSpec(
                    name="line_time_s",
                    type="float",
                    description="正扫每行时间。默认 0.1。",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.01,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="sts_count",
                    type="int",
                    description=(
                        "要采几条 STS 谱。默认 5"
                        "（中心 + 内接正方形的 4 个角）。"
                    ),
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=25,
                ),
                ParameterSpec(
                    name="sts_start_v",
                    type="float",
                    description="STS 起始偏压，单位伏。默认 -1.0。",
                    unit="V",
                    required=False,
                    default=-1.0,
                    min_value=-5.0,
                    max_value=5.0,
                ),
                ParameterSpec(
                    name="sts_end_v",
                    type="float",
                    description="STS 终止偏压，单位伏。默认 1.0。",
                    unit="V",
                    required=False,
                    default=1.0,
                    min_value=-5.0,
                    max_value=5.0,
                ),
                ParameterSpec(
                    name="sts_num_points",
                    type="int",
                    description="STS 扫描点数。默认 100。",
                    required=False,
                    default=100,
                    min_value=20,
                    max_value=2000,
                ),
                ParameterSpec(
                    name="scan_timeout_s",
                    type="float",
                    description="等待扫描完成的上限。",
                    unit="s",
                    required=False,
                    default=180.0,
                    min_value=10.0,
                    max_value=1800.0,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=120.0,
            composition_level=3,
            tags=["demo", "scan", "sts", "composite"],
        )

    # ------------------------------------------------------------------
    # Plan: static. 6 fixed steps + (move + sts) per STS position.
    # ------------------------------------------------------------------

    def plan(self, params: dict) -> list[CompositeStep]:
        cx = params.get("center_x_m", 0.0)
        cy = params.get("center_y_m", 0.0)
        size = params.get("scan_size_m", 50e-9)
        line_time = params.get("line_time_s", 0.1)
        sts_count = int(params.get("sts_count", 5))
        v_start = params.get("sts_start_v", -1.0)
        v_end = params.get("sts_end_v", 1.0)
        v_pts = int(params.get("sts_num_points", 100))
        scan_timeout = params.get("scan_timeout_s", 180.0)

        speed = size / line_time if line_time > 0 else 200e-9
        positions = self._build_sts_positions(cx, cy, size, sts_count)

        steps: list[CompositeStep] = [
            CompositeStep(
                step_id="configure_scan",
                skill_name="ConfigureScan",
                params={
                    "center_x_m": cx,
                    "center_y_m": cy,
                    "width_m": size,
                    "height_m": size,
                },
                optional=False,
                checkpoint_after=False,
                tags=("setup", "scan"),
            ),
            CompositeStep(
                step_id="set_scan_speed",
                skill_name="SetScanSpeed",
                params={
                    "fwd_speed": speed,
                    "bwd_speed": speed,
                    "fwd_line_time": line_time,
                    "bwd_line_time": line_time,
                    "keep_const": 0,
                },
                optional=False,
                checkpoint_after=False,
                tags=("setup", "scan"),
            ),
            CompositeStep(
                step_id="start_scan",
                skill_name="StartScan",
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("scan",),
            ),
            CompositeStep(
                step_id="wait_scan",
                skill_name="WaitScanComplete",
                params={"timeout_ms": int(scan_timeout * 1000)},
                optional=False,
                checkpoint_after=True,    # mid-flow flush after scan finishes
                tags=("scan", "wait"),
            ),
            CompositeStep(
                step_id="save_scan",
                skill_name="SaveScan",
                params={"timeout_ms": 30000},
                optional=True,            # best-effort; non-fatal per v1
                checkpoint_after=True,
                tags=("scan", "save"),
            ),
            CompositeStep(
                step_id="configure_sts",
                skill_name="ConfigureSTS",
                params={
                    "start_v": v_start,
                    "end_v": v_end,
                    "num_points": v_pts,
                },
                optional=False,
                checkpoint_after=False,
                tags=("setup", "sts"),
            ),
        ]

        for idx, (x, y) in enumerate(positions, 1):
            steps.append(CompositeStep(
                step_id=f"move_{idx}",
                skill_name="MoveToXY",
                params={"x_m": x, "y_m": y, "wait": True},
                optional=True,            # per-point failure does not abort run
                checkpoint_after=False,
                tags=("sts", "move", f"point={idx}"),
            ))
            steps.append(CompositeStep(
                step_id=f"sts_{idx}",
                skill_name="AcquireSTS",
                params={},
                optional=True,
                checkpoint_after=True,    # flush per acquired spectrum
                tags=("sts", "acquire", f"point={idx}"),
            ))
        return steps

    # ------------------------------------------------------------------
    # Dynamic plan: skip ``sts_N`` when ``move_N`` failed so we do not
    # acquire spectra at the wrong (unmoved) position. The static
    # :meth:`plan` is still used for step-count assertions and resume
    # bookkeeping; :meth:`plan_dynamic` is what the executor actually
    # walks at runtime.
    # ------------------------------------------------------------------

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        all_steps = self.plan(params)
        for step in all_steps:
            if step.step_id.startswith("sts_"):
                idx = step.step_id.split("_", 1)[1]
                move_id = f"move_{idx}"
                # Skip the STS step iff the matching move failed in THIS
                # invocation (i.e. recorded in failed_steps). Resumed-skipped
                # moves are in completed_steps and should still gate STS.
                if move_id in executor.progress.failed_steps:
                    logger.warning(
                        "DemoScanAndSTS: skipping %s because %s failed",
                        step.step_id, move_id,
                    )
                    continue
            yield step

    @staticmethod
    def _build_sts_positions(
        cx: float, cy: float, size: float, sts_count: int,
    ) -> list[tuple[float, float]]:
        """Build the STS sample point list.

        For ``sts_count <= 5`` this preserves the historical v1 layout
        (center + the four corners of an inner square at offset = size/3).
        For ``sts_count > 5`` it lays out a *real* grid of distinct points
        across the inner region — previously the surplus points were padded
        with the center coordinate, so every "extra" spectrum re-measured the
        exact same spot and was passed off as an independent measurement.
        """
        offset = size / 3.0
        if sts_count <= 5:
            corners: list[tuple[float, float]] = [
                (cx, cy),
                (cx - offset, cy - offset),
                (cx + offset, cy - offset),
                (cx + offset, cy + offset),
                (cx - offset, cy + offset),
            ]
            return corners[:sts_count]

        # >5 points: generate a genuine grid covering the inner square
        # [cx-offset, cx+offset] x [cy-offset, cy+offset]. Choose the
        # smallest n x n grid that holds sts_count distinct points, then
        # take them in row-major order so the returned list is exactly
        # sts_count long with no duplicate coordinates.
        n = math.ceil(math.sqrt(sts_count))
        if n < 2:
            n = 2
        span = 2.0 * offset            # full extent of the inner square
        step = span / (n - 1)          # n >= 2 here, so no divide-by-zero
        positions: list[tuple[float, float]] = []
        for iy in range(n):
            for ix in range(n):
                x = cx - offset + ix * step
                y = cy - offset + iy * step
                positions.append((x, y))
                if len(positions) == sts_count:
                    return positions
        return positions[:sts_count]

    # ------------------------------------------------------------------
    # Hooks: stash key fields + track STS success/failure counts.
    # ------------------------------------------------------------------

    def on_step_result(self, step: CompositeStep, sub_result: SkillResult) -> None:
        # Count STS acquisitions for the final aggregate.
        if step.step_id.startswith("sts_"):
            done = int(self._executor.progress.partial_data.get("sts_succeeded", 0))
            self._executor.set_partial("sts_succeeded", done + 1)
        elif step.step_id == "save_scan":
            self._executor.set_partial("scan_saved", True)
        elif step.step_id == "wait_scan":
            # ── Neither flag was consumed here before v6.1.3 (KNOWN_ISSUES
            #    §2.24). A timed-out or interrupted frame flowed into SaveScan
            #    and the whole STS run as if it were a finished image.
            #
            # REPORTED, NOT FATAL — and that is a judgement, so here is the
            # reasoning rather than just the result:
            #
            #  * The STS points do NOT come from the image. _build_sts_positions
            #    derives them from (center, size) alone, so a truncated frame
            #    does not invalidate a single spectrum. Aborting would throw away
            #    good data to punish an unrelated step.
            #  * This file already has a precedent for exactly this shape:
            #    SaveScan is optional=True and its outcome is REPORTED as
            #    `scan_saved` rather than failing the run.
            #  * This skill exists for live demos — "a guaranteed sequence
            #    without LLM second-guessing" (module docstring). Turning the
            #    demo into a hard failure mid-presentation is the opposite of
            #    what it is for.
            #
            # What must NOT happen is the frame being passed off as complete.
            # `scan_completed` / `scan_outcome` land in the aggregate so the
            # result tells the truth about the image while the STS half stands.
            data = getattr(sub_result, "data", {}) or {}
            timed_out = bool(data.get("timed_out", False))
            stopped_early = bool(data.get("stopped_early", False))
            outcome = str(data.get("outcome") or
                          ("timed_out" if timed_out else
                           "stopped_early" if stopped_early else "completed"))
            self._executor.set_partial("scan_completed",
                                       not (timed_out or stopped_early))
            self._executor.set_partial("scan_outcome", outcome)
            self._executor.set_partial("scan_lines_done", data.get("lines_done"))
            self._executor.set_partial("scan_lines_total", data.get("lines_total"))
            if timed_out or stopped_early:
                logger.warning(
                    "DemoScanAndSTS: the scan did not finish (%s, %s/%s lines) — "
                    "continuing to STS (the points are geometric, not picked "
                    "from the image), but the frame is NOT a complete image",
                    outcome, data.get("lines_done"), data.get("lines_total"),
                )

    def on_step_failed(self, step: CompositeStep, msg: str) -> bool:
        if step.step_id.startswith(("sts_", "move_")):
            failed = int(self._executor.progress.partial_data.get("sts_failed", 0))
            self._executor.set_partial("sts_failed", failed + 1)
        elif step.step_id == "save_scan":
            self._executor.set_partial("scan_saved", False)
            logger.warning("SaveScan failed: %s - continuing to STS", msg)
        return step.optional  # default: continue iff optional

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        partial = progress.partial_data
        # Single source of truth: both counters are maintained live by the
        # hooks — on_step_result bumps sts_succeeded, on_step_failed bumps
        # sts_failed (covering both a failed AcquireSTS and a failed MoveToXY
        # whose STS got skipped). Re-deriving sts_failed from
        # (sts_total - sts_succeeded) discarded that accumulator and broke on
        # resume / on points still in flight, so we read it back directly.
        sts_succeeded = int(partial.get("sts_succeeded", 0))
        sts_failed = int(partial.get("sts_failed", 0))
        sts_total = int(partial.get("sts_total", 0))
        return {
            "scan_size_m": partial.get("scan_size_m"),
            "scan_center": partial.get("scan_center"),
            "scan_saved": bool(partial.get("scan_saved", False)),
            # Did the FRAME finish? Separate from scan_saved: a truncated frame
            # can be saved perfectly well, and then the .sxm on disk looks like
            # a complete image to everything downstream. Defaults to True only
            # when the wait step never ran (nothing to contradict); the hook
            # sets it explicitly on every real run.
            "scan_completed": bool(partial.get("scan_completed", True)),
            "scan_outcome": partial.get("scan_outcome"),
            "scan_lines_done": partial.get("scan_lines_done"),
            "scan_lines_total": partial.get("scan_lines_total"),
            "sts_total": sts_total,
            "sts_succeeded": sts_succeeded,
            "sts_failed": sts_failed,
        }

    # ------------------------------------------------------------------
    # Driver — overrides default to seed partial_data with config before
    # executor walks the plan (matches GridSTS pattern).
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        cx = params.get("center_x_m", 0.0)
        cy = params.get("center_y_m", 0.0)
        size = params.get("scan_size_m", 50e-9)
        sts_count = int(params.get("sts_count", 5))

        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        # Seed config so aggregate() can read it back.
        executor.set_partial("scan_size_m", size)
        executor.set_partial("scan_center", (cx, cy))
        executor.set_partial("sts_total", sts_count)
        # Accumulators that must survive resume.
        executor.set_partial_default("sts_succeeded", 0)
        executor.set_partial_default("sts_failed", 0)
        executor.set_partial_default("scan_saved", False)
        self._executor = executor

        logger.info(
            "DemoScanAndSTS: scan=%.1e m at (%.2e, %.2e), then %d STS",
            size, cx, cy, sts_count,
        )

        all_good = executor.run_plan(self.plan_dynamic(params, executor))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        if all_good:
            return self.ok(**data)
        return self.fail(
            executor.progress.aborted_reason or "DemoScanAndSTS aborted",
            **data,
        )


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(DemoScanAndSTS, context_provider)
