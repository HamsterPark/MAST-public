"""Bias and current measurement skills.

vendored from v1 mast/skills/builtins/bias.py 2026-04-23. Zero behavioural
changes. 6 skills: GetBias, SetBias, GetCurrent, GetBiasCalibration,
SetBiasCalibration, SetBiasRange. Covers all three SafetyLevel tiers.

This is the canary set for Phase 4: if wrap_skill works on these (read +
parametric write + DANGEROUS), the remaining 124 builtin skills can be
batch-ported during the v2 skill migration.

Phase 7 migration (2026-05-19): `SetBiasRamp` was extracted from the
multi-step slew logic inside ``SetBias`` and reshaped as a graph-shaped
composite. Each ramp step becomes its own ``_phase_set_step_<i>``
synthetic step so per-step progress is visible to the GUI and the
LangGraph checkpointer can resume a long ramp across restarts. ``SetBias``
itself is unchanged (atomic v1 behaviour preserved).
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterator

import numpy as np

from mast.core.state import coerce_number, reply_scalar
from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)


# Synthetic phase identifiers for SetBiasRamp — intercepted by _BiasRampPhaseCtx.
_PHASE_GET_CURRENT = "_phase_get_current"
_PHASE_RAMP_STEP_PREFIX = "_phase_set_step_"


class GetBias(BaseSkill):
    """Read current bias voltage."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetBias",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前偏压（bias）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["bias", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Bias_Get")
        if record.error:
            return SkillResult(
                skill_name="GetBias",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list)
        # parsed_list[0] should be bias in V
        parsed = record.return_value
        bias_v = reply_scalar(parsed, field="bias_v")
        # 写入状态缓存前统一提取标量；单元素元组不是合法的 bias_v 缓存值。
        if bias_v is None:
            return SkillResult(
                skill_name="GetBias",
                success=False,
                error=("读到的偏压值不是一个数(回包解不出) —— 拒绝把它当成读数。"
                       "多半是 Nanonis 数值字段的元组包装没解开。"),
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GetBias",
            success=True,
            data={"bias_v": bias_v},
            nanonis_calls=[record],
        )


class SetBias(BaseSkill):
    """Set bias voltage, with optional slew rate limiting."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetBias",
            version="1.1.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置偏压。大幅改变电压时，用 slew_rate_v_per_s "
                "做渐进 ramp。"
            ),
            parameters=[
                ParameterSpec(
                    name="bias_v",
                    type="float",
                    description="目标偏压，单位伏特",
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="slew_rate_v_per_s",
                    type="float",
                    description=(
                        "偏压最大变化速率（V/s）。省略则瞬时切换。"
                        "大幅跳变电压时建议使用，以保护样品/针尖。"
                    ),
                    unit="V/s",
                    required=False,
                    default=None,
                    min_value=0.01,
                    max_value=100.0,
                ),
            ],
            preconditions=[],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["bias", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        target = params["bias_v"]
        slew = params.get("slew_rate_v_per_s")
        all_calls = []

        if slew and slew > 0:
            # Get current bias for ramping. The ramp start MUST be the real
            # present bias — if Bias_Get fails (or returns unparseable data) we
            # must NOT silently assume 0.0V: doing so would step from 0V to the
            # first ramp target, producing the exact large instantaneous jump
            # the slew rate is meant to prevent. Abort the ramp instead.
            get_rec = context.safe_call("Bias_Get")
            all_calls.append(get_rec)
            current_bias: float | None = None
            if not get_rec.error and get_rec.return_value is not None:
                # return_value is (error_string, raw_bytes, parsed_list);
                # parsed_list[0] is bias in V (Bias.Get ResponseTypes == ["f"]).
                parsed = get_rec.return_value
                if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
                    try:
                        current_bias = float(parsed[2][0])
                    except (TypeError, IndexError, ValueError):
                        current_bias = None
            if current_bias is None:
                return SkillResult(
                    skill_name="SetBias",
                    success=False,
                    error=(
                        "Cannot slew-ramp bias: failed to read the current "
                        "bias (Bias_Get error: "
                        f"{get_rec.error or 'unparseable response'}). Refusing "
                        "to ramp from an assumed 0.0V start, which would defeat "
                        "slew protection. Retry, or call SetBias without "
                        "slew_rate_v_per_s for an instant set."
                    ),
                    nanonis_calls=all_calls,
                )

            # Ramp: step every 100ms
            step_interval = 0.1  # seconds
            voltage_diff = abs(target - current_bias)
            if voltage_diff > 0.001:  # Only ramp if change > 1mV
                n_steps = max(1, int(voltage_diff / (slew * step_interval)))
                _abort = getattr(context, "check_abort", None)
                for i, v in enumerate(np.linspace(current_bias, target, n_steps + 1)[1:]):
                    # A slew is a LOOP of hardware writes with sleeps in it, so it
                    # has to be interruptible in its own right. (The post-abort
                    # safe_call gate would refuse the next Bias_Set anyway — the
                    # instrument is safe either way — but bailing here stops
                    # promptly and reports the abort as an abort instead of as a
                    # confusing "slew failed".) Stopping mid-ramp is safe: the bias
                    # is left between two values that both passed the bounds check.
                    if callable(_abort) and _abort():
                        return SkillResult(
                            skill_name="SetBias", success=False,
                            error=(f"aborted by operator mid-slew at {v:.4f} V "
                                   f"({i}/{n_steps} steps); bias left at a safe "
                                   f"intermediate value. Do not retry."),
                            data={"bias_v": float(v), "aborted": True},
                            nanonis_calls=all_calls,
                        )
                    rec = context.safe_call("Bias_Set", float(v))
                    all_calls.append(rec)
                    if rec.error:
                        return SkillResult(
                            skill_name="SetBias",
                            success=False,
                            error=f"Slew failed at {v:.4f}V: {rec.error}",
                            nanonis_calls=all_calls,
                        )
                    time.sleep(step_interval)
                return SkillResult(
                    skill_name="SetBias",
                    success=True,
                    data={"bias_v": target, "slew_rate_v_per_s": slew, "steps": n_steps},
                    nanonis_calls=all_calls,
                )
            # Fall through to instant set for small voltage changes

        # Instant set (no slew, or slew with < 1mV change)
        record = context.safe_call("Bias_Set", target)
        all_calls.append(record)
        if record.error:
            return SkillResult(
                skill_name="SetBias",
                success=False,
                error=record.error,
                nanonis_calls=all_calls,
            )
        return SkillResult(
            skill_name="SetBias",
            success=True,
            data={"bias_v": target},
            nanonis_calls=all_calls,
        )


class GetCurrent(BaseSkill):
    """Read tunneling current."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetCurrent",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取隧道电流。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["current", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Current_Get")
        if record.error:
            return SkillResult(
                skill_name="GetCurrent",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        current_a = reply_scalar(parsed, field="current_a")
        # 回包经统一解析器转成数值后才写入结果与状态缓存。
        # 数组元素可能是单元素元组；直接传出会破坏 float 转换、差值和阈值判断。
        # 接收侧的类型检查仍保留，不能替代此处对技能结果类型的保证。
        if current_a is None:
            return SkillResult(
                skill_name="GetCurrent",
                success=False,
                error=("读到的电流值不是一个数(回包解不出) —— 拒绝把它当成读数。"
                       "这多半是 Nanonis 数值字段的元组包装没解开,不是电流异常。"),
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GetCurrent",
            success=True,
            data={"current_a": current_a},
            nanonis_calls=[record],
        )


class GetBiasCalibration(BaseSkill):
    """Read bias calibration and offset."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetBiasCalibration",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取偏压的标定系数与偏移量。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["bias", "calibration", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Bias_CalibrGet")
        if record.error:
            return SkillResult(
                skill_name="GetBiasCalibration",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            calibration = vals[0] if isinstance(vals, (list, tuple)) and len(vals) > 0 else parsed
            offset = vals[1] if isinstance(vals, (list, tuple)) and len(vals) > 1 else 0.0
        else:
            calibration = parsed
            offset = 0.0
        return SkillResult(
            skill_name="GetBiasCalibration",
            success=True,
            data={"calibration": calibration, "offset": offset},
            nanonis_calls=[record],
        )


class SetBiasCalibration(BaseSkill):
    """Set bias calibration and offset. DANGEROUS: affects all bias measurements."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetBiasCalibration",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置偏压的标定系数与偏移量。会影响所有偏压测量。",
            parameters=[
                ParameterSpec(
                    name="calibration",
                    type="float",
                    description="偏压标定系数（V/V 倍率）",
                    required=True,
                    # Sanity ceiling (rig-tunable), NOT a precise calibration
                    # window. Nominal is ~1.0; bound to 1e-3 … 1e3 so an agent
                    # can't set an extreme multiplier that silently defeats the
                    # ±10 V bias bound (a 1e6 factor turns a "safe" 1 V request
                    # into 1 MV at the hardware). Rejects 0 and negatives.
                    min_value=1e-3,
                    max_value=1e3,
                ),
                ParameterSpec(
                    name="offset",
                    type="float",
                    description="偏压偏移量（V）",
                    unit="V",
                    required=True,
                    # Sanity ceiling (rig-tunable): a huge additive offset also
                    # defeats the ±10 V bound. ±100 V is generous-but-finite.
                    min_value=-100.0,
                    max_value=100.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["bias", "calibration", "write", "dangerous"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calibration = params["calibration"]
        offset = params["offset"]
        record = context.safe_call("Bias_CalibrSet", calibration, offset)
        if record.error:
            return SkillResult(
                skill_name="SetBiasCalibration",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetBiasCalibration",
            success=True,
            data={"calibration": calibration, "offset": offset},
            nanonis_calls=[record],
        )


class SetBiasRange(BaseSkill):
    """Select bias range by index."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetBiasRange",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="按索引选择偏压量程。",
            parameters=[
                ParameterSpec(
                    name="range_index",
                    type="int",
                    description="要选择的偏压量程的索引",
                    required=True,
                    min_value=0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["bias", "range", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        range_index = params["range_index"]
        record = context.safe_call("Bias_RangeSet", range_index)
        if record.error:
            return SkillResult(
                skill_name="SetBiasRange",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetBiasRange",
            success=True,
            data={"range_index": range_index},
            nanonis_calls=[record],
        )


# ──────────────────────────────────────────────────────────────────────────
# SetBiasRamp — graph-shaped composite (Phase 7 migration)
# ──────────────────────────────────────────────────────────────────────────


class _BiasRampPhaseCtx:
    """Wraps the real ExecutionContext to dispatch ``_phase_*`` skill names.

    All non-phase attribute access (``safe_call``, ``check_abort``,
    ``emit_progress``, ``get_progress``, ``checkpoint_flush``, ``state``,
    etc.) delegates unchanged to the underlying context.
    """

    def __init__(self, real_ctx, skill: "SetBiasRamp") -> None:
        self._ctx = real_ctx
        self._skill = skill

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        return getattr(self._ctx, name)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        if skill_name.startswith("_phase_"):
            return self._skill._run_phase(skill_name, params, self._ctx)
        return self._ctx.run(skill_name, params)


class SetBiasRamp(CompositeSkillGraph):
    """Ramp bias voltage from start to end in N small steps.

    Equivalent to v1's ``SetBias(bias_v=end, slew_rate_v_per_s=slew)`` but
    exposed as a graph-shaped composite: each ramp step is its own
    ``_phase_set_step_<i>`` synthetic step. Benefits:

      * GUI live-progress shows ramp completion ratio.
      * LangGraph checkpointer flushes after each step so a long ramp
        survives a kernel restart.
      * Per-step checkpoint_after=False (avoid flush storm in inner loop);
        only the final step flushes.

    Behavioural contract:
      * If ``bias_v_start`` is None, the ramp first issues a ``Bias_Get``
        (``_phase_get_current``) to seed the current bias.
      * ``slew_rate_v_per_s`` is the **max** ramp rate. With the
        100 ms per-step interval, ``n_steps = max(1, int(|Δv| / (slew * 0.1)))``.
      * If |Δv| < 1 mV, a single Bias_Set is issued (no ramp).
      * Any step that fails returns success=False with the partial steps
        recorded in nanonis_calls.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetBiasRamp",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把偏压从起始值分小步 ramp 到目标值"
                "（受 slew rate 限制）。大幅改变偏压时用它，"
                "以保护样品 / 针尖。"
            ),
            parameters=[
                ParameterSpec(
                    name="bias_v_end",
                    type="float",
                    description="目标偏压，单位伏特",
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="bias_v_start",
                    type="float",
                    description=(
                        "起始偏压。省略则取当前的 "
                        "Bias_Get 读数。"
                    ),
                    unit="V",
                    required=False,
                    default=None,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="slew_rate_v_per_s",
                    type="float",
                    description=(
                        "偏压最大变化速率（V/s）。ramp 会被切成 "
                        "N=max(1, |Δv| / (slew * step_interval)) 步。"
                    ),
                    unit="V/s",
                    required=False,
                    default=1.0,
                    min_value=0.01,
                    max_value=100.0,
                ),
                ParameterSpec(
                    name="step_interval_s",
                    type="float",
                    description="两步 ramp 之间的等待时长",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.001,
                    max_value=10.0,
                ),
            ],
            preconditions=[],
            estimated_duration_s=2.0,
            composition_level=1,
            tags=["bias", "ramp", "write", "composite"],
        )

    # ------------------------------------------------------------------
    # Plan helpers
    # ------------------------------------------------------------------

    def _compute_steps(
        self, start_v: float, end_v: float, slew: float, step_interval: float,
    ) -> list[float]:
        """Return the list of intermediate target voltages (excluding start)."""
        voltage_diff = abs(end_v - start_v)
        if voltage_diff <= 0.001:  # < 1 mV → single shot
            return [end_v]
        n_steps = max(1, int(voltage_diff / (slew * step_interval)))
        return [float(v) for v in np.linspace(start_v, end_v, n_steps + 1)[1:]]

    def plan(self, params: dict) -> list[CompositeStep]:
        """Static plan when ``bias_v_start`` is provided; else use plan_dynamic."""
        if params.get("bias_v_start") is None:
            # The dynamic path will run a get_current phase first.
            raise NotImplementedError("Use plan_dynamic when bias_v_start is None")
        start_v = float(params["bias_v_start"])
        end_v = float(params["bias_v_end"])
        slew = float(params.get("slew_rate_v_per_s", 1.0))
        step_interval = float(params.get("step_interval_s", 0.1))

        targets = self._compute_steps(start_v, end_v, slew, step_interval)
        last = len(targets) - 1
        return [
            CompositeStep(
                step_id=f"set_step_{i}",
                skill_name=f"{_PHASE_RAMP_STEP_PREFIX}{i}",
                params={
                    "target_v": v,
                    "is_last": i == last,
                    "step_interval_s": step_interval,
                },
                optional=False,           # one bad set aborts the ramp
                checkpoint_after=(i == last),  # only flush at the end
                tags=("ramp", f"step={i}"),
            )
            for i, v in enumerate(targets)
        ]

    def plan_dynamic(
        self, params: dict, executor: GraphExecutor,
    ) -> Iterator[CompositeStep]:
        # Phase 0: resolve starting voltage. Either user-provided or read from
        # the instrument.
        start_v_arg = params.get("bias_v_start")
        if start_v_arg is None:
            yield CompositeStep(
                step_id="get_current",
                skill_name=_PHASE_GET_CURRENT,
                params={},
                optional=False,
                checkpoint_after=False,
                tags=("setup",),
            )
            if executor.progress.aborted:
                return
            # ⚠️ **第二个 0.0,和上面那个各自独立** —— 只修 `_phase_get_current`
            # 等于没修:那一步失败时 `bias_v_start` 从来没被 set 过,这里的
            # `.get(..., 0.0)` 会再把假起点补回来。用 `None` 哨兵而不是数字,
            # 拿不到就**不排斜坡**(get_current 那步失败已经会中止,这是兜底的兜底)。
            start_raw = executor.progress.partial_data.get("bias_v_start")
            if start_raw is None:
                return
            start_v = float(start_raw)
        else:
            start_v = float(start_v_arg)
            executor.set_partial("bias_v_start", start_v)

        end_v = float(params["bias_v_end"])
        slew = float(params.get("slew_rate_v_per_s", 1.0))
        step_interval = float(params.get("step_interval_s", 0.1))

        targets = self._compute_steps(start_v, end_v, slew, step_interval)
        executor.set_partial("n_steps", len(targets))
        executor.set_partial("bias_v_end", end_v)
        executor.set_partial("slew_rate_v_per_s", slew)
        executor.set_total_steps(len(targets) + (1 if start_v_arg is None else 0))

        last = len(targets) - 1
        for i, v in enumerate(targets):
            if executor.progress.aborted:
                return
            yield CompositeStep(
                step_id=f"set_step_{i}",
                skill_name=f"{_PHASE_RAMP_STEP_PREFIX}{i}",
                params={
                    "target_v": float(v),
                    "is_last": i == last,
                    "step_interval_s": step_interval,
                },
                optional=False,
                checkpoint_after=(i == last),
                tags=("ramp", f"step={i}"),
            )

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------

    def _run_phase(self, skill_name: str, params: dict, real_ctx) -> SkillResult:
        if skill_name == _PHASE_GET_CURRENT:
            return self._phase_get_current(real_ctx)
        if skill_name.startswith(_PHASE_RAMP_STEP_PREFIX):
            return self._phase_set_step(params, real_ctx)
        return SkillResult(
            skill_name=skill_name,
            success=False,
            error=f"Unknown phase: {skill_name}",
        )

    def _phase_get_current(self, real_ctx) -> SkillResult:
        rec = real_ctx.safe_call("Bias_Get")
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_GET_CURRENT,
                success=False,
                error=f"Bias_Get failed: {rec.error}",
            )
        # ⚠️ **读不到就拒绝,不要编一个起点。** 这里以前兜底成 0.0,而这个值
        # 直接进 `_compute_steps(start_v, ...)` —— 斜坡的**起点**。真实偏压 1 V
        # 而起点被当成 0 时,第一步就把硬件从 1 V 拽到接近 0,**那正是
        # `slew_rate_v_per_s` 存在的意义所要防止的突变**(隧穿态下偏压骤降 →
        # 电流塌 → Z 反馈追设定点 → 把针尖往样品推)。
        #
        # 一个失败的读取变成一个假的测量值,而那个假值废掉了一条安全保护。
        # 同一个文件里的 `SetBias`(见上方 "Refusing to ramp from an assumed
        # 0.0V start, which would defeat slew protection")早就是这么拒绝的 ——
        # 那句话在这个文件里写着,而这个函数一直在做它拒绝的事。
        current_bias: float | None = None
        parsed = rec.return_value
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            try:
                current_bias = float(parsed[2][0])
            except (TypeError, IndexError, ValueError):
                current_bias = None
        if current_bias is None or not math.isfinite(current_bias):
            return SkillResult(
                skill_name=_PHASE_GET_CURRENT,
                success=False,
                error=(
                    "Cannot slew-ramp bias: Bias_Get returned an unparseable "
                    "response, so the current bias is unknown. Refusing to ramp "
                    "from an assumed 0.0V start, which would defeat slew "
                    "protection. Retry, or pass bias_v_start explicitly."
                ),
            )
        self._executor.set_partial("bias_v_start", current_bias)
        return SkillResult(
            skill_name=_PHASE_GET_CURRENT,
            success=True,
            data={"bias_v_start": current_bias},
        )

    def _phase_set_step(self, params: dict, real_ctx) -> SkillResult:
        target_v = float(params["target_v"])
        is_last = bool(params.get("is_last", False))
        step_interval = float(params.get("step_interval_s", 0.1))

        rec = real_ctx.safe_call("Bias_Set", target_v)
        self._call_log.append(rec)
        if rec.error:
            return SkillResult(
                skill_name=_PHASE_RAMP_STEP_PREFIX,
                success=False,
                error=f"Slew failed at {target_v:.4f}V: {rec.error}",
            )
        self._executor.set_partial("last_bias_v", target_v)
        # Sleep between steps (skip on the last step — caller doesn't need
        # post-ramp delay).
        if not is_last and step_interval > 0:
            time.sleep(step_interval)
        return SkillResult(
            skill_name=_PHASE_RAMP_STEP_PREFIX,
            success=True,
            data={"target_v": target_v},
        )

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        self._call_log: list[NanonisCallRecord] = []

        wrapped = _BiasRampPhaseCtx(context, self)
        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=wrapped,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        self._executor = executor

        plan_iter = self.plan_dynamic(params, executor)
        all_good = executor.run_plan(plan_iter)

        end_v = float(params["bias_v_end"])
        slew = float(params.get("slew_rate_v_per_s", 1.0))
        n_steps = int(executor.progress.partial_data.get("n_steps", 0))

        data: dict[str, Any] = {
            "bias_v": end_v,
            "slew_rate_v_per_s": slew,
            "steps": n_steps,
            "_progress": executor.progress.to_dict(),
        }

        if not all_good:
            return SkillResult(
                skill_name=self._skill_name(),
                success=False,
                error=executor.progress.aborted_reason or "ramp aborted",
                data=data,
                nanonis_calls=list(self._call_log),
            )
        return SkillResult(
            skill_name=self._skill_name(),
            success=True,
            data=data,
            nanonis_calls=list(self._call_log),
        )
