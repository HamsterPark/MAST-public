"""Sweep skills: Bias Sweep (via Generic Sweeper) and Lock-In Frequency Sweep.

vendored from v1 mast/skills/builtins/sweep.py 2026-04-23. Zero behavioural changes.
10 skills: ConfigureBiasSweep, AcquireBiasSweep, ConfigureLockInSweep,
           AcquireLockInSweep, GetLockInSweepLimits, GetLockInSweepProps,
           GetLockInSweepSignal, GenSwpAcqChsGet, GenSwpPropsGet,
           GenSwpStop, GenSwpSwpSignalGet.
"""

from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply, channel_ids_from_buffer, scalar_int
from mast.skills.base import BaseSkill

_log = logging.getLogger(__name__)


def _extract_sweep_rows(parsed) -> list[list[float]] | None:
    """Pull the 2D sweep data out of a *_Swp.Start three-tuple response.

    Both ``GenSwp.Start`` and ``LockInFreqSwp.Start`` declare ResponseTypes
    ``["i","i","*+c","i","i","2f"]``. nanonis_spm's parseGeneralResponse builds
    the Variables list (return_value[2]) as::

        [names_size, num_channels, channel_names, rows, cols, data_2d]

    where ``data_2d`` is an ``np.ndarray`` reshaped to ``(rows, cols)`` (the
    first row is the swept signal, each subsequent row a recorded channel).

    Returns a list of per-row python ``list[float]`` traces, or ``None`` if the
    response carries no decodable data array.
    """
    if not (isinstance(parsed, (list, tuple)) and len(parsed) > 2):
        return None
    variables = parsed[2]
    if not (isinstance(variables, (list, tuple)) and len(variables) >= 6):
        return None
    arr = variables[5]
    # 2D numpy array (the canonical real-hardware shape).
    if hasattr(arr, "ndim") and hasattr(arr, "tolist"):
        if getattr(arr, "ndim", 0) == 2:
            return [list(row) for row in arr.tolist()]
        if getattr(arr, "ndim", 0) == 1:
            return [list(arr.tolist())]
        return None
    # Plain nested list/tuple fallback (e.g. test fixtures): rows of floats.
    if isinstance(arr, (list, tuple)) and arr:
        if all(isinstance(r, (list, tuple)) or hasattr(r, "__iter__") for r in arr):
            return [list(r) for r in arr]
        return [list(arr)]
    return None


def _extract_channel_names(parsed) -> list[str] | None:
    """Pull the channel-names string array from a *_Swp.Start response.

    Channel names live at Variables[2] (the ``*+c`` slot), NOT at
    return_value[1] (which is the raw response bytes).
    """
    if not (isinstance(parsed, (list, tuple)) and len(parsed) > 2):
        return None
    variables = parsed[2]
    if not (isinstance(variables, (list, tuple)) and len(variables) >= 3):
        return None
    names = variables[2]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return None


class ConfigureBiasSweep(BaseSkill):
    """Configure bias sweep via Generic Sweeper (supports channel configuration)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureBiasSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 bias sweep 的上下限、步数与记录通道。",
            parameters=[
                ParameterSpec(
                    name="lower_v",
                    type="float",
                    description="bias 下限，单位伏特",
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="upper_v",
                    type="float",
                    description="bias 上限，单位伏特",
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="num_steps",
                    type="int",
                    description="扫描的步数",
                    required=True,
                    min_value=2,
                    max_value=10000,
                ),
                ParameterSpec(
                    name="period_ms",
                    type="float",
                    description="每一步的积分周期，单位毫秒",
                    unit="ms",
                    required=False,
                    default=4.0,
                    min_value=0.1,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["sweep", "bias", "configure", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        lower_v = params["lower_v"]
        upper_v = params["upper_v"]
        num_steps = params["num_steps"]
        # 审查 [#138]: the execute() fallback (20.0) diverged from
        # the ParameterSpec default (4.0). On the composite path params are
        # resolved from the spec/context WITHOUT the registry back-filling
        # metadata defaults, so an omitted period_ms silently took 20.0 here
        # instead of the declared 4.0 — a different integration time than the
        # GUI/LLM saw advertised. Align the fallback with the single source of
        # truth (the ParameterSpec default).
        period_ms = params.get("period_ms", 4.0)
        calls = []

        rec_open = context.safe_call("GenSwp_Open")
        calls.append(rec_open)
        if rec_open.error:
            return SkillResult(
                skill_name="ConfigureBiasSweep",
                success=False,
                error=rec_open.error,
                nanonis_calls=calls,
            )

        rec_sig = context.safe_call("GenSwp_SwpSignalSet", "Bias (V)")
        calls.append(rec_sig)
        if rec_sig.error:
            return SkillResult(
                skill_name="ConfigureBiasSweep",
                success=False,
                error=rec_sig.error,
                nanonis_calls=calls,
            )

        try:
            rec_meas = context.safe_call("Signals_MeasNamesGet")
            calls.append(rec_meas)
            if not rec_meas.error and isinstance(rec_meas.return_value, (list, tuple)):
                parsed = rec_meas.return_value[-1]
                name_list = None
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, list) and item and isinstance(item[0], str):
                            name_list = item
                            break
                if name_list and "Current (A)" in name_list:
                    idx = name_list.index("Current (A)")
                    rec_acq = context.safe_call(
                        "GenSwp_AcqChsSet", [idx], ["Current (A)"],
                    )
                    calls.append(rec_acq)
        except Exception as exc:
            _log.warning("Failed to set acquisition channels: %s", exc)

        rec_limits = context.safe_call("GenSwp_LimitsSet", lower_v, upper_v)
        calls.append(rec_limits)
        if rec_limits.error:
            return SkillResult(
                skill_name="ConfigureBiasSweep",
                success=False,
                error=rec_limits.error,
                nanonis_calls=calls,
            )

        rec_props = context.safe_call(
            "GenSwp_PropsSet", 100, 1e6, num_steps, period_ms, 1, 2, 10,
        )
        calls.append(rec_props)
        if rec_props.error:
            return SkillResult(
                skill_name="ConfigureBiasSweep",
                success=False,
                error=rec_props.error,
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="ConfigureBiasSweep",
            success=True,
            data={
                "lower_v": lower_v,
                "upper_v": upper_v,
                "num_steps": num_steps,
                "period_ms": period_ms,
            },
            nanonis_calls=calls,
        )


class AcquireBiasSweep(BaseSkill):
    """Acquire a bias sweep at the current position via Generic Sweeper."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireBiasSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="在当前针尖位置采一条 bias sweep。",
            preconditions=["z_controller_on"],
            estimated_duration_s=30.0,
            composition_level=1,
            tags=["sweep", "bias", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []

        rec_open = context.safe_call("GenSwp_Open")
        calls.append(rec_open)

        rec_sig = context.safe_call("GenSwp_SwpSignalSet", "Bias (V)")
        calls.append(rec_sig)

        try:
            rec_meas = context.safe_call("Signals_MeasNamesGet")
            calls.append(rec_meas)
            # Signals.MeasNamesGet returns (err, raw, [size, count, names_list]).
            # The OLD code took return_value[-1] = the WHOLE Variables list
            # [size, count, names] and searched IT for "Current (A)" — always
            # False, so the acquisition channel was NEVER set (dead code; review
            # 2026-07-03). Find the names list (last list-of-strings element).
            meas_names = None
            mv = rec_meas.return_value
            variables = mv[2] if (isinstance(mv, (list, tuple)) and len(mv) > 2) else mv
            if isinstance(variables, (list, tuple)):
                for item in reversed(variables):
                    if isinstance(item, list) and item and isinstance(item[0], str):
                        meas_names = item
                        break
            if not rec_meas.error and meas_names and "Current (A)" in meas_names:
                idx = meas_names.index("Current (A)")
                rec_acq = context.safe_call(
                    "GenSwp_AcqChsSet", [idx], ["Current (A)"],
                )
                calls.append(rec_acq)
        except Exception as exc:
            _log.warning("Failed to set acquisition channels: %s", exc)

        # NOTE: AcquireBiasSweep deliberately does NOT call GenSwp_PropsSet.
        # PropsSet's Initial-settling / Max-slew / Settling args are floats with
        # no "no change" sentinel, so the old PropsSet(0, 1e6, 0, 0, 1, 2, 0)
        # zeroed the 100/10 ms settling times ConfigureBiasSweep had just set and
        # opened the slew limit to 1e6 units/s on every acquire (review
        # 2026-07-03). Timing/autosave are configured via ConfigureBiasSweep;
        # Acquire only starts the sweep (Get data=1 returns the trace regardless).

        # GenSwp_Start(Get_data, Direction, Save_basename, Reset_signal, Z-Ctrl).
        # Z-Ctrl=1 → feedback is turned OFF during the sweep (constant-height
        # bias sweep): without this the Z-controller stayed active and, as the
        # sweep crossed 0 V, drove the tip into the surface. Reset_signal=1 →
        # the bias returns to its pre-sweep value afterwards instead of being
        # left parked at the sweep's end limit.
        record = context.safe_call("GenSwp_Start", 1, 0, "", 1, 1)
        calls.append(record)
        if record.error:
            return SkillResult(
                skill_name="AcquireBiasSweep",
                success=False,
                error=record.error,
                nanonis_calls=calls,
            )

        data: dict = {"acquisition_complete": True}
        # return_value is (error_string, raw_bytes, Variables). For
        # GenSwp.Start (ResponseTypes ["i","i","*+c","i","i","2f"]) the
        # Variables list is:
        #   [0] names size (int), [1] num channels (int),
        #   [2] channel names (1D array string),
        #   [3] data rows (int), [4] data columns (int),
        #   [5] data (2D array float32, shape (rows, cols)).
        # The first data row is the swept signal (bias), each subsequent row a
        # recorded channel. The OLD code took parsed[2][0]/[1] — i.e. the
        # names-size int and num-channels int — as "bias"/"current" arrays,
        # silently returning scalar header counts instead of the sweep traces.
        parsed = record.return_value
        rows = _extract_sweep_rows(parsed)
        if rows is not None:
            if len(rows) >= 1:
                data["bias"] = rows[0]
                data["num_points"] = len(rows[0])
            if len(rows) >= 2:
                data["current"] = rows[1]
            if len(rows) >= 3:
                data["dIdV"] = rows[2]
        names = _extract_channel_names(parsed)
        if names is not None:
            data["channel_names"] = names
        return SkillResult(
            skill_name="AcquireBiasSweep",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class ConfigureLockInSweep(BaseSkill):
    """Configure lock-in frequency sweep parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureLockInSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 lock-in 扫频的上下限与各项参数。",
            parameters=[
                ParameterSpec(
                    name="lower_hz",
                    type="float",
                    description="频率下限，单位 Hz",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="upper_hz",
                    type="float",
                    description="频率上限，单位 Hz",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="num_steps",
                    type="int",
                    description="扫描的步数",
                    required=True,
                    min_value=2,
                    max_value=10000,
                ),
                ParameterSpec(
                    name="integration_periods",
                    type="int",
                    description="每一步的积分周期数",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="settling_periods",
                    type="int",
                    description="每一步的建立（settling）周期数",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=60.0,
            composition_level=1,
            tags=["sweep", "lockin", "frequency", "configure", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        lower_hz = params["lower_hz"]
        upper_hz = params["upper_hz"]
        num_steps = params["num_steps"]
        # 审查 [#138]: execute() fallbacks (3 / 3) diverged from the
        # ParameterSpec defaults (1 / 1). On the composite path defaults are NOT
        # back-filled, so an omitted integration_periods/settling_periods took 3
        # here instead of the advertised 1. Align with the ParameterSpec default.
        int_periods = params.get("integration_periods", 1)
        set_periods = params.get("settling_periods", 1)
        calls = []

        rec_open = context.safe_call("LockInFreqSwp_Open")
        calls.append(rec_open)
        if rec_open.error:
            return SkillResult(
                skill_name="ConfigureLockInSweep",
                success=False,
                error=rec_open.error,
                nanonis_calls=calls,
            )

        rec_limits = context.safe_call("LockInFreqSwp_LimitsSet", lower_hz, upper_hz)
        calls.append(rec_limits)
        if rec_limits.error:
            return SkillResult(
                skill_name="ConfigureLockInSweep",
                success=False,
                error=rec_limits.error,
                nanonis_calls=calls,
            )

        rec_props = context.safe_call(
            "LockInFreqSwp_PropsSet",
            num_steps, int_periods, 0.0, set_periods, 0.0, 1, 2, "",
        )
        calls.append(rec_props)
        if rec_props.error:
            return SkillResult(
                skill_name="ConfigureLockInSweep",
                success=False,
                error=rec_props.error,
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="ConfigureLockInSweep",
            success=True,
            data={
                "lower_hz": lower_hz,
                "upper_hz": upper_hz,
                "num_steps": num_steps,
                "integration_periods": int_periods,
                "settling_periods": set_periods,
            },
            nanonis_calls=calls,
        )


class AcquireLockInSweep(BaseSkill):
    """Acquire a lock-in frequency sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireLockInSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="采一条 lock-in 扫频。",
            estimated_duration_s=60.0,
            composition_level=1,
            tags=["sweep", "lockin", "frequency", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []

        rec_open = context.safe_call("LockInFreqSwp_Open")
        calls.append(rec_open)

        rec_props = context.safe_call(
            "LockInFreqSwp_PropsSet", 0, 0, 0.0, 0, 0.0, 1, 2, "",
        )
        calls.append(rec_props)

        record = context.safe_call("LockInFreqSwp_Start", 1, 0)
        calls.append(record)
        if record.error:
            return SkillResult(
                skill_name="AcquireLockInSweep",
                success=False,
                error=record.error,
                nanonis_calls=calls,
            )

        data: dict = {"acquisition_complete": True}
        # return_value is (error_string, raw_bytes, Variables). For
        # LockInFreqSwp.Start (ResponseTypes ["i","i","*+c","i","i","2f"]) the
        # Variables list is identical in layout to GenSwp.Start:
        #   [2] channel names (1D array string),
        #   [5] data (2D array float32, shape (rows, cols)).
        # The first data row is the swept frequency, each subsequent row a
        # recorded channel. The OLD code took parsed[2][0]/[1] (the header
        # ints) as "frequency"/"amplitude".
        parsed = record.return_value
        rows = _extract_sweep_rows(parsed)
        if rows is not None:
            if len(rows) >= 1:
                data["frequency"] = rows[0]
                data["num_points"] = len(rows[0])
            if len(rows) >= 2:
                data["amplitude"] = rows[1]
            if len(rows) >= 3:
                data["phase"] = rows[2]
        names = _extract_channel_names(parsed)
        if names is not None:
            data["channel_names"] = names
        return SkillResult(
            skill_name="AcquireLockInSweep",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class GetLockInSweepLimits(BaseSkill):
    """Read the frequency limits of the lock-in frequency sweep module."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetLockInSweepLimits",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 lock-in 扫频的频率下限与上限。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["sweep", "lockin", "frequency", "limits", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("LockInFreqSwp_LimitsGet")
        if record.error:
            return SkillResult(
                skill_name="GetLockInSweepLimits", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {
                    "lower_hz": float(vals[0]),
                    "upper_hz": float(vals[1]),
                }
        return SkillResult(
            skill_name="GetLockInSweepLimits", success=True,
            data=data, nanonis_calls=[record],
        )


class GetLockInSweepProps(BaseSkill):
    """Read the configuration of the lock-in frequency sweep module."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetLockInSweepProps",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 lock-in 扫频的属性（步数、积分、建立）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["sweep", "lockin", "frequency", "props", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("LockInFreqSwp_PropsGet")
        if record.error:
            return SkillResult(
                skill_name="GetLockInSweepProps", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 7:
                data = {
                    "num_steps": int(vals[0]),
                    "integration_periods": int(vals[1]),
                    "min_integration_time_s": float(vals[2]),
                    "settling_periods": int(vals[3]),
                    "min_settling_time_s": float(vals[4]),
                    "autosave": int(vals[5]),
                    "save_dialog": int(vals[6]),
                }
                # ResponseTypes ["H","H","f","H","f","I","I","i","*-c"]:
                # vals[7] is the basename SIZE (int), vals[8] is the basename
                # STRING. The old code returned str(vals[7]) — the size int —
                # as the basename.
                if len(vals) >= 9:
                    data["basename"] = str(vals[8]) if vals[8] else ""
        return SkillResult(
            skill_name="GetLockInSweepProps", success=True,
            data=data, nanonis_calls=[record],
        )


class GetLockInSweepSignal(BaseSkill):
    """Read the sweep signal used in the lock-in frequency sweep module."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetLockInSweepSignal",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 lock-in 扫频所用的扫描信号索引。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["sweep", "lockin", "frequency", "signal", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("LockInFreqSwp_SignalGet")
        if record.error:
            return SkillResult(
                skill_name="GetLockInSweepSignal", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            idx = vals[0] if isinstance(vals, (list, tuple)) else vals
            data = {"sweep_signal_index": int(idx)}
        return SkillResult(
            skill_name="GetLockInSweepSignal", success=True,
            data=data, nanonis_calls=[record],
        )


class GenSwpAcqChsGet(BaseSkill):
    """Get acquisition channels for generic sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GenSwpAcqChsGet",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Generic Sweeper 记录的采集通道列表。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["sweep", "generic", "channels", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("GenSwp_AcqChsGet")
        if record.error:
            return SkillResult(
                skill_name="GenSwpAcqChsGet", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        # GenSwp.AcqChsGet 的 Variables 依次包含通道数量、通道索引数组、名称字节数、
        # 名称数量和名称数组。索引数组与 Scan.BufferGet 的前两个字段同构。
        # 元素提取共用 io.nanonis_files 的解析器，兼容单元素元组而不传播错误类型；
        # 协议分支由合成字节测试验证，不附带某次仪器会话的成功声明。
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {
                    "num_channels": scalar_int(vals[0]),
                    "channel_indexes": channel_ids_from_buffer(vals),
                }
                if len(vals) >= 5 and isinstance(vals[4], (list, tuple)):
                    data["channel_names"] = [str(x) for x in vals[4]]
        return SkillResult(
            skill_name="GenSwpAcqChsGet", success=True,
            data=data, nanonis_calls=[record],
        )


class GenSwpPropsGet(BaseSkill):
    """Get generic sweep properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GenSwpPropsGet",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Generic Sweeper 的配置属性。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["sweep", "generic", "props", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("GenSwp_PropsGet")
        if record.error:
            return SkillResult(
                skill_name="GenSwpPropsGet", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        # return_value is (error_string, raw_bytes, Variables). For
        # GenSwp.PropsGet (ResponseTypes ["f","f","i","H","I","I","f"]) the
        # Variables list is:
        #   [0] initial settling time (ms), [1] max slew rate,
        #   [2] num steps, [3] period (ms), [4] autosave,
        #   [5] save dialog, [6] settling time (ms).
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 4:
                data = {
                    "initial_settling_time_ms": float(vals[0]),
                    "max_slew_rate": float(vals[1]),
                    "num_steps": int(vals[2]),
                    "period_ms": float(vals[3]),
                }
                if len(vals) >= 7:
                    data["autosave"] = int(vals[4])
                    data["save_dialog"] = int(vals[5])
                    data["settling_time_ms"] = float(vals[6])
        return SkillResult(
            skill_name="GenSwpPropsGet", success=True,
            data=data, nanonis_calls=[record],
        )


class GenSwpStop(BaseSkill):
    """Stop generic sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GenSwpStop",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="停止当前正在运行的 Generic Sweeper 扫描。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["sweep", "generic", "stop", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("GenSwp_Stop")
        if record.error:
            return SkillResult(
                skill_name="GenSwpStop", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GenSwpStop", success=True,
            data={"stopped": True}, nanonis_calls=[record],
        )


class GenSwpSwpSignalGet(BaseSkill):
    """Get sweep signal configuration for generic sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GenSwpSwpSignalGet",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Generic Sweeper 的扫描信号名。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["sweep", "generic", "signal", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("GenSwp_SwpSignalGet")
        if record.error:
            return SkillResult(
                skill_name="GenSwpSwpSignalGet", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        # return_value is (error_string, raw_bytes, Variables). For
        # GenSwp.SwpSignalGet (ResponseTypes ["i","*-c"]) the Variables list is:
        #   [0] sweep channel name size (int),
        #   [1] sweep channel name (string).
        # NOTE: this method does NOT return a sweep direction — the old
        # parsed[1] "sweep_direction" int() cast was a double bug (wrong tuple
        # slot AND a non-existent field) that crashed on the real name string.
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {"channel_name": str(vals[1])}
            elif isinstance(vals, str):
                data = {"channel_name": vals}
        elif isinstance(parsed, str):
            data = {"channel_name": parsed}
        return SkillResult(
            skill_name="GenSwpSwpSignalGet", success=True,
            data=data, nanonis_calls=[record],
        )
