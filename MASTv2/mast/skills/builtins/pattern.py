"""Pattern skills — Nanonis built-in grid/line/cloud experiment module.

vendored from v1 mast/skills/builtins/pattern.py 2026-04-23. Zero behavioural changes.
8 skills: RunGridExperiment, OpenPatternExperiment, PausePatternExperiment,
          SetPatternLine, SetPatternCloud, GetPatternCloud, GetPatternProps.

Phase 7 migration (2026-05-19): RunGridExperiment is now a CompositeSkillGraph.
The original time-driven while loop is reshaped into:

  * setup_grid step — Pattern_GridSet
  * start_experiment step — Pattern_ExpStart
  * per-tick poll steps — one ``_phase_tick_<i>`` per 2-second poll
  * cleanup step — finalises partial data

Per-tick polls are NOT optional (a single Pattern_ExpStatusGet failure
must abort so the LLM can pause and recover). The composite exits the
poll loop early when status==0 (experiment finished) by stashing a
``done`` flag the dynamic plan reads between yields.
"""

from __future__ import annotations

import time
from typing import Any, Iterator, List

from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    abort_error_text,
)


# Synthetic phase identifiers — intercepted by _RunGridExperimentPhaseCtx.
_PHASE_SETUP = "_phase_setup_grid"
_PHASE_START = "_phase_start_experiment"
_PHASE_TICK_PREFIX = "_phase_tick_"
_PHASE_CLEANUP = "_phase_cleanup"


class _RunGridExperimentPhaseCtx:
    """Wraps the real ExecutionContext to dispatch ``_phase_*`` skill names."""

    def __init__(self, real_ctx, skill: "RunGridExperiment") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class RunGridExperiment(CompositeSkillGraph):
    """Run grid experiment using Nanonis Pattern module.

    Uses Nanonis built-in Pattern module for grid spectroscopy,
    which handles drift compensation and auto-save internally.
    """

    TICK_INTERVAL_S = 2.0  # one Pattern_ExpStatusGet poll every 2 seconds

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunGridExperiment",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "用 Nanonis Pattern 模块跑网格谱学。它会处理漂移补偿与自动存盘。"
            ),
            parameters=[
                ParameterSpec(
                    name="nx",
                    type="int",
                    description="X 方向的网格点数",
                    required=True,
                    min_value=1,
                    max_value=100,
                ),
                ParameterSpec(
                    name="ny",
                    type="int",
                    description="Y 方向的网格点数",
                    required=True,
                    min_value=1,
                    max_value=100,
                ),
                ParameterSpec(
                    name="center_x_m",
                    type="float",
                    description="网格中心 X（m）",
                    unit="m",
                    required=False,
                    default=0.0,
                ),
                ParameterSpec(
                    name="center_y_m",
                    type="float",
                    description="网格中心 Y（m）",
                    unit="m",
                    required=False,
                    default=0.0,
                ),
                ParameterSpec(
                    name="width_m",
                    type="float",
                    description=(
                        "网格宽度，单位**米**（SI），**不是**纳米。换算：100 nm → 100n。默认 100n（=100 nm）"
                        "。"
                    ),
                    unit="m",
                    required=False,
                    default=100e-9,
                    min_value=0.0,
                    max_value=1e-5,   # = config scan_size_max_m (10 µm); guards nm→m slips
                ),
                ParameterSpec(
                    name="height_m",
                    type="float",
                    description=(
                        "网格高度，单位**米**（SI），**不是**纳米。换算：100 nm → 100n。默认 100n（=100 nm）"
                        "。"
                    ),
                    unit="m",
                    required=False,
                    default=100e-9,
                    min_value=0.0,
                    max_value=1e-5,   # = config scan_size_max_m (10 µm); guards nm→m slips
                ),
                ParameterSpec(
                    name="angle_deg",
                    type="float",
                    description="网格转角（度）",
                    unit="deg",
                    required=False,
                    default=0.0,
                ),
                ParameterSpec(
                    name="wait_timeout_s",
                    type="float",
                    description="最长等待时间",
                    unit="s",
                    required=False,
                    default=3600.0,
                    min_value=10.0,
                    max_value=36000.0,
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=600.0,
            composition_level=2,
            tags=["pattern", "grid", "spectroscopy", "composite"],
        )

    # ------------------------------------------------------------------
    # Dynamic plan — setup + start + N polls + cleanup. The poll count is
    # capped by ``wait_timeout_s / TICK_INTERVAL_S`` so the UI has a number
    # and we never spin forever.
    # ------------------------------------------------------------------

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        timeout = float(params.get("wait_timeout_s", 3600.0))
        max_ticks = max(1, int(timeout / self.TICK_INTERVAL_S))
        executor.set_total_steps(3 + max_ticks)
        self._timeout_s = timeout

        # Mandatory setup
        yield CompositeStep(
            step_id="setup_grid",
            skill_name=_PHASE_SETUP,
            params=dict(params),
            optional=False,
            checkpoint_after=False,
            tags=("setup",),
        )
        if executor.progress.aborted:
            return

        # Mandatory start
        yield CompositeStep(
            step_id="start_experiment",
            skill_name=_PHASE_START,
            params={},
            optional=False,
            checkpoint_after=True,    # experiment is now running — flush
            tags=("start",),
        )
        if executor.progress.aborted:
            return

        # Per-tick polls. We pre-compute the cap (max_ticks) but exit early
        # once a tick sets the ``exp_done`` flag in partial_data.
        for i in range(max_ticks):
            if executor.progress.aborted:
                return
            if executor.progress.partial_data.get("exp_done"):
                break
            yield CompositeStep(
                step_id=f"tick_{i}",
                skill_name=f"{_PHASE_TICK_PREFIX}{i}",
                params={"index": i},
                optional=False,
                checkpoint_after=False,
                tags=("poll", f"i={i}"),
            )

        # Cleanup always runs (so SkillResult.data is well-formed).
        yield CompositeStep(
            step_id="cleanup",
            skill_name=_PHASE_CLEANUP,
            params={"nx": params["nx"], "ny": params["ny"]},
            optional=False,
            checkpoint_after=True,
            tags=("cleanup",),
        )

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name == _PHASE_SETUP:
            return self._phase_setup_grid(params, real_ctx)
        if skill_name == _PHASE_START:
            return self._phase_start_experiment(real_ctx)
        if skill_name.startswith(_PHASE_TICK_PREFIX):
            return self._phase_tick(params, real_ctx)
        if skill_name == _PHASE_CLEANUP:
            return self._phase_cleanup(params, real_ctx)
        return SkillResult(
            skill_name=skill_name,
            success=False,
            error=f"Unknown phase: {skill_name}",
        )

    def _phase_setup_grid(self, params: dict, real_ctx) -> SkillResult:
        nx = int(params["nx"])
        ny = int(params["ny"])
        cx = float(params.get("center_x_m", 0.0))
        cy = float(params.get("center_y_m", 0.0))
        w = float(params.get("width_m", 100e-9))
        h = float(params.get("height_m", 100e-9))
        angle = float(params.get("angle_deg", 0.0))

        # Pattern_GridSet(Set_active_pattern, nx, ny, Grid_Scan_frame,
        #                 Center_X_m, Center_Y_m, Width_m, Height_m, Angle_deg)
        # Grid_Scan_frame MUST be 0: when 1, Nanonis sizes the grid to the
        # current scan frame and IGNORES the explicit center/width/height/angle
        # we pass below — silently discarding the user's grid geometry. Pass 0
        # so the cx/cy/w/h/angle the user requested are actually applied.
        rec = real_ctx.safe_call(
            "Pattern_GridSet", 1, nx, ny, 0, cx, cy, w, h, angle,
        )
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_SETUP,
                success=False,
                error=f"Pattern_GridSet failed: {rec.error}",
            )
        self._executor.set_partial("nx", nx)
        self._executor.set_partial("ny", ny)
        return SkillResult(skill_name=_PHASE_SETUP, success=True, data={})

    def _phase_start_experiment(self, real_ctx) -> SkillResult:
        rec = real_ctx.safe_call("Pattern_ExpStart", 0)
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_START,
                success=False,
                error=f"Pattern_ExpStart failed: {rec.error}",
            )
        self._executor.set_partial("start_time", time.time())
        return SkillResult(skill_name=_PHASE_START, success=True, data={})

    def _stop_pattern_experiment(self, real_ctx) -> None:
        """Best-effort stop of the running grid/pattern experiment. Never raises.
        Falls back to Pattern_ExpPause if Stop is unavailable."""
        try:
            rec = real_ctx.safe_call("Pattern_ExpStop")
            self._call_log.append(rec)
            if getattr(rec, "error", ""):
                rec2 = real_ctx.safe_call("Pattern_ExpPause", 1)
                self._call_log.append(rec2)
        except Exception as exc:  # pragma: no cover - best-effort
            import logging
            logging.getLogger(__name__).warning(
                "Pattern experiment stop failed: %s", exc)

    def _phase_tick(self, params: dict, real_ctx) -> SkillResult:
        # Match the v1 cadence: sleep ~2s before reading status.
        time.sleep(self.TICK_INTERVAL_S)

        # Honour abort even if check_abort() isn't on the context.
        check_abort = getattr(real_ctx, "check_abort", None)
        if callable(check_abort) and check_abort():
            # STOP the hardware experiment — the old code only set flags, so the
            # grid pattern kept running unmonitored on the controller after the
            # composite bailed (审查; cf. WaitScanComplete which
            # stops the scan on abort).
            self._stop_pattern_experiment(real_ctx)
            self._executor.set_partial("exp_done", True)
            self._executor.set_partial("aborted", True)
            return SkillResult(
                skill_name=_PHASE_TICK_PREFIX,
                success=False,
                error="aborted by user — pattern experiment stopped",
            )

        # Timeout check (the dynamic plan also caps tick count but the user
        # may set a shorter timeout than max_ticks * TICK_INTERVAL).
        start_time = float(self._executor.progress.partial_data.get(
            "start_time", time.time()))
        elapsed = time.time() - start_time
        if elapsed >= self._timeout_s:
            self._stop_pattern_experiment(real_ctx)
            self._executor.set_partial("exp_done", True)
            self._executor.set_partial("timed_out", True)
            return SkillResult(
                skill_name=_PHASE_TICK_PREFIX,
                success=False,
                error=(f"Grid experiment timed out after {self._timeout_s}s "
                       "— pattern experiment stopped"),
            )

        rec = real_ctx.safe_call("Pattern_ExpStatusGet")
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_TICK_PREFIX,
                success=False,
                error=f"Pattern_ExpStatusGet failed: {rec.error}",
            )
        parsed = rec.return_value
        status = None
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            inner = parsed[2]
            if isinstance(inner, (list, tuple)) and len(inner) > 0:
                status = inner[0]
        # status==0 → experiment finished.
        if status == 0:
            self._executor.set_partial("exp_done", True)
            self._executor.set_partial("timed_out", False)
        return SkillResult(
            skill_name=_PHASE_TICK_PREFIX,
            success=True,
            data={"status": status},
        )

    def _phase_cleanup(self, params: dict, real_ctx) -> SkillResult:
        nx = int(params["nx"])
        ny = int(params["ny"])
        self._executor.set_partial("total_points", nx * ny)
        return SkillResult(
            skill_name=_PHASE_CLEANUP,
            success=True,
            data={"nx": nx, "ny": ny, "total_points": nx * ny},
        )

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        self._call_log: list[NanonisCallRecord] = []
        wrapped = _RunGridExperimentPhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        executor.set_partial_default("exp_done", False)
        executor.set_partial_default("aborted", False)
        executor.set_partial_default("timed_out", False)
        self._executor = executor

        all_good = executor.run_plan(self.plan_dynamic(params, executor))

        timed_out = bool(executor.progress.partial_data.get("timed_out", False))
        aborted = bool(executor.progress.partial_data.get("aborted", False))

        data: dict[str, Any] = {
            "nx": int(executor.progress.partial_data.get("nx",
                                                         params["nx"])),
            "ny": int(executor.progress.partial_data.get("ny",
                                                         params["ny"])),
            "total_points": int(executor.progress.partial_data.get(
                "total_points",
                int(params["nx"]) * int(params["ny"]))),
            "_progress": executor.progress.to_dict(),
        }

        if all_good and not timed_out and not aborted:
            return SkillResult(
                skill_name=self._skill_name(),
                success=True,
                data=data,
                nanonis_calls=list(self._call_log),
            )
        # Mirror v1: timeout returns success=False with explicit error.
        timeout = float(params.get("wait_timeout_s", 3600.0))
        if timed_out:
            error = f"Grid experiment timed out after {timeout}s"
        elif aborted:
            # Whatever actually stopped it (operator 中止, E_STOP, tip-quality
            # halt, a failed mandatory step) — never a blanket "you did this".
            error = abort_error_text(executor.progress)
        else:
            error = executor.progress.aborted_reason or "RunGridExperiment aborted"
        return SkillResult(
            skill_name=self._skill_name(),
            success=False,
            error=error,
            data=data,
            nanonis_calls=list(self._call_log),
        )


class OpenPatternExperiment(BaseSkill):
    """Open the selected grid experiment.

    This is required to configure the experiment and be able to start it.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="OpenPatternExperiment",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "打开选定的网格实验。配置或启动该实验之前必须先做这一步。"
            ),
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pattern", "experiment", "open", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Pattern_ExpOpen")
        if record.error:
            return SkillResult(
                skill_name="OpenPatternExperiment",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="OpenPatternExperiment",
            success=True,
            data={"opened": True},
            nanonis_calls=[record],
        )


class PausePatternExperiment(BaseSkill):
    """Pause or resume the selected grid experiment."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PausePatternExperiment",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="暂停或恢复当前正在跑的网格实验。",
            parameters=[
                ParameterSpec(
                    name="pause",
                    type="bool",
                    description="True 为暂停，False 为恢复",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pattern", "experiment", "pause", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        pause = params["pause"]
        pause_val = 1 if pause else 0
        record = context.safe_call("Pattern_ExpPause", pause_val)
        if record.error:
            return SkillResult(
                skill_name="PausePatternExperiment",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="PausePatternExperiment",
            success=True,
            data={"paused": pause},
            nanonis_calls=[record],
        )


class SetPatternLine(BaseSkill):
    """Set line pattern parameters for line spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPatternLine",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设定线图案的参数：点数与两个端点。",
            parameters=[
                ParameterSpec(
                    name="set_active",
                    type="bool",
                    description="True 表示把当前图案切换成 Line",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="num_points",
                    type="int",
                    description="沿这条线的点数",
                    required=True,
                    min_value=1,
                    max_value=10000,
                ),
                ParameterSpec(
                    name="use_scan_frame",
                    type="bool",
                    description="True 表示把这条线设成扫描帧的对角线",
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="p1_x_m",
                    type="float",
                    description="线上第 1 点的 X 坐标（m）",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="p1_y_m",
                    type="float",
                    description="线上第 1 点的 Y 坐标（m）",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="p2_x_m",
                    type="float",
                    description="线上第 2 点的 X 坐标（m）",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="p2_y_m",
                    type="float",
                    description="线上第 2 点的 Y 坐标（m）",
                    unit="m",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pattern", "line", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        set_active = 1 if params.get("set_active", True) else 0
        num_points = params["num_points"]
        scan_frame = 1 if params.get("use_scan_frame", False) else 0
        p1x = params["p1_x_m"]
        p1y = params["p1_y_m"]
        p2x = params["p2_x_m"]
        p2y = params["p2_y_m"]
        record = context.safe_call(
            "Pattern_LineSet", set_active, num_points, scan_frame,
            p1x, p1y, p2x, p2y,
        )
        if record.error:
            return SkillResult(
                skill_name="SetPatternLine",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPatternLine",
            success=True,
            data={
                "num_points": num_points,
                "p1": [p1x, p1y],
                "p2": [p2x, p2y],
            },
            nanonis_calls=[record],
        )


class SetPatternCloud(BaseSkill):
    """Configure a cloud of points for point spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPatternCloud",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="为点云图案谱学设定一组 XY 点。",
            parameters=[
                ParameterSpec(
                    name="set_active",
                    type="bool",
                    description="True 表示把当前图案切换成 Cloud",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="x_coords",
                    type="str",
                    description="X 坐标，以 JSON 浮点数列表给出（m）",
                    required=True,
                ),
                ParameterSpec(
                    name="y_coords",
                    type="str",
                    description="Y 坐标，以 JSON 浮点数列表给出（m）",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pattern", "cloud", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import json

        set_active = 1 if params.get("set_active", True) else 0
        try:
            x_coords: List[float] = json.loads(params["x_coords"])
            y_coords: List[float] = json.loads(params["y_coords"])
        except (json.JSONDecodeError, TypeError) as exc:
            return SkillResult(
                skill_name="SetPatternCloud",
                success=False,
                error=f"Invalid JSON in cloud coordinates: {exc}",
                nanonis_calls=[],
            )
        # VALID JSON IS NOT ENOUGH. `"5"` parses cleanly — to the integer 5, and the
        # len() below then raises TypeError, which reaches the agent as an unhandled
        # exception rather than a SkillResult: no error path, no retry, a dead turn.
        # A model writing x_coords="5" for a single point (forgetting the brackets) is
        # not an exotic input; it is the most likely mistake there is.
        for label, seq in (("x_coords", x_coords), ("y_coords", y_coords)):
            if not isinstance(seq, list):
                return SkillResult(
                    skill_name="SetPatternCloud",
                    success=False,
                    error=(f"{label} 必须是 JSON 数组，例如 \"[1e-9, 2e-9]\"；"
                           f"收到的是 {type(seq).__name__}（{params[label]!r}）——"
                           "单个点也要写成数组：\"[1e-9]\""),
                    nanonis_calls=[],
                )
        if len(x_coords) != len(y_coords):
            return SkillResult(
                skill_name="SetPatternCloud",
                success=False,
                error="x_coords and y_coords must have equal length",
                nanonis_calls=[],
            )
        record = context.safe_call(
            "Pattern_CloudSet", set_active, len(x_coords), x_coords, y_coords,
        )
        if record.error:
            return SkillResult(
                skill_name="SetPatternCloud",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPatternCloud",
            success=True,
            data={"num_points": len(x_coords)},
            nanonis_calls=[record],
        )


class GetPatternCloud(BaseSkill):
    """Read the current cloud-pattern configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPatternCloud",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读为点云图案谱学配置好的那组 XY 点。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pattern", "cloud", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Pattern_CloudGet")
        if record.error:
            return SkillResult(
                skill_name="GetPatternCloud",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                data = {
                    "num_points": int(vals[0]),
                    "x_coords": list(vals[1]) if hasattr(vals[1], '__iter__') else [vals[1]],
                    "y_coords": list(vals[2]) if hasattr(vals[2], '__iter__') else [vals[2]],
                }
        return SkillResult(
            skill_name="GetPatternCloud",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


class GetPatternProps(BaseSkill):
    """Read grid experiment configuration properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPatternProps",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读网格实验配置：可用的实验、选中的实验、外部 VI、测量前延时、保存通道。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pattern", "config", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Pattern_PropsGet")
        if record.error:
            return SkillResult(
                skill_name="GetPatternProps",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 9:
                data = {
                    "experiments_size": int(vals[0]),
                    "num_experiments": int(vals[1]),
                    "experiments": vals[2],
                    "selected_exp_size": int(vals[3]),
                    "selected_experiment": str(vals[4]),
                    "ext_vi_path_size": int(vals[5]),
                    "ext_vi_path": str(vals[6]),
                    "pre_measure_delay_s": float(vals[7]),
                    "save_scan_channels": bool(vals[8]),
                }
        return SkillResult(
            skill_name="GetPatternProps",
            success=True,
            data=data,
            nanonis_calls=[record],
        )
