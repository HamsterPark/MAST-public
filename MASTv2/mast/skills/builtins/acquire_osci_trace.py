"""Pull one buffered trace from the Nanonis 1-Channel Oscilloscope.

The hardware oscilloscope module records at the Real-Time loop rate
(20 kHz on a typical V5e) into an on-board buffer, then exposes the whole
buffer via a single TCP call. End-to-end latency is therefore dominated
by the buffer length, not by the TCP round-trip.

Requires the Osci1T module to be loaded in the running Nanonis
configuration. The bundled simulator does NOT load it by default — the
skill returns a clear error in that case and the LLM can fall back to
``CaptureSignalBuffer`` for a polled-rate trace.
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
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class AcquireOsciTrace(BaseSkill):
    """Pull one buffered trace from Osci1T (1-channel oscilloscope)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireOsciTrace",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从 Nanonis 的 1 通道示波器（Osci1T）取一段缓冲时间序列。按硬件 RT 速率采样（V5e 上 "
                "~20 kHz）。返回 (t0, dt, y_array)。要求 Nanonis 里已加载 Osci1T 模块 —— "
                "默认自带的模拟器上没有。"
            ),
            parameters=[
                ParameterSpec(
                    name="data_to_get",
                    type="int",
                    description=(
                        "0 = 当前显示的缓冲区（最快，可能是陈旧的），1 = 等下一个触发，2 = 等 2 "
                        "个触发（最干净的快照）。"
                    ),
                    required=False, default=0,
                    min_value=0, max_value=2,
                ),
                ParameterSpec(
                    name="signal_index",
                    type="int",
                    description=(
                        "可选的 0–15 信号通道索引，采集前赋给 Osci1T。-1 = 保持现有。"
                    ),
                    required=False, default=-1,
                    min_value=-1, max_value=15,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=1,
            tags=["oscilloscope", "trace", "hardware", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        data_to_get = int(params.get("data_to_get", 0))
        signal_index = int(params.get("signal_index", -1))

        calls = []

        # Start the module (idempotent). If the module isn't loaded this
        # surfaces a recognisable error string we forward upward.
        rec_run = context.safe_call("Osci1T_Run")
        calls.append(rec_run)
        if rec_run.error and "NeedModule" in rec_run.error:
            return SkillResult(
                skill_name="AcquireOsciTrace", success=False,
                error=(
                    "Osci1T module is not loaded in this Nanonis "
                    "configuration. On the bundled simulator try the v2 "
                    "config that includes the Oscilloscope module, or fall "
                    "back to CaptureSignalBuffer for a polled-rate trace."
                ),
                nanonis_calls=calls,
            )

        if signal_index >= 0:
            rec = context.safe_call("Osci1T_ChSet", signal_index)
            calls.append(rec)

        rec = context.safe_call("Osci1T_DataGet", data_to_get)
        calls.append(rec)
        if rec.error:
            return SkillResult(
                skill_name="AcquireOsciTrace", success=False,
                error=f"Osci1T_DataGet failed: {rec.error}",
                nanonis_calls=calls,
            )

        d = _decoded(rec.return_value)
        # Layout: t0 (float64) + dt (float64) + n (int32) + n×float64
        if len(d) < 4:
            return SkillResult(
                skill_name="AcquireOsciTrace", success=False,
                error=f"Unexpected response shape: {len(d)} fields",
                nanonis_calls=calls,
            )
        try:
            t0 = float(d[0])
            dt = float(d[1])
            n = int(d[2])
            y_raw = d[3]
            samples = [
                float(v[0]) if isinstance(v, tuple) else float(v)
                for v in y_raw
            ]
        except (TypeError, ValueError, IndexError) as exc:
            return SkillResult(
                skill_name="AcquireOsciTrace", success=False,
                error=f"Decode error: {exc}", nanonis_calls=calls,
            )

        fs = 1.0 / dt if dt > 0 else 0.0
        return SkillResult(
            skill_name="AcquireOsciTrace", success=True,
            data={
                "t0_s": t0,
                "dt_s": dt,
                "n_samples": n,
                "duration_s": n * dt,
                "fs_hz": fs,
                "nyquist_hz": fs / 2.0,
                "samples": samples,
                "source": "nanonis_osci1t",
            },
            nanonis_calls=calls,
        )


def _unwrap(v):
    """decodeArray yields single-element tuples like (5e-5,); unwrap to float."""
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return float(v)


class GetOsciTimebases(BaseSkill):
    """List the Osci1T sample-rate timebases available on this controller."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetOsciTimebases",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "列出可用的 Oscilloscope-1-Channel（Osci1T）时基。每个时基就是每采样点的间隔 dt（s）；"
                "采样率 fs = 1/dt。可选时基集合取决于 RT 频率与 RT 过采样。要求 Nanonis 里已加载 "
                "Osci1T 模块。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["oscilloscope", "timebase", "samplerate", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # Ensure the module is running so the timebase list is populated.
        rec_run = context.safe_call("Osci1T_Run")
        if rec_run.error and "NeedModule" in rec_run.error:
            return SkillResult(
                skill_name="GetOsciTimebases", success=False,
                error=(
                    "Osci1T module is not loaded in this Nanonis "
                    "configuration — no hardware timebases to list. Use the "
                    "polled fast-path (CaptureSignalBuffer) and its poll_hz "
                    "knob to choose a sample rate instead."
                ),
                nanonis_calls=[rec_run],
            )
        rec = context.safe_call("Osci1T_TimebaseGet")
        if rec.error:
            return SkillResult(
                skill_name="GetOsciTimebases", success=False,
                error=f"Osci1T_TimebaseGet failed: {rec.error}",
                nanonis_calls=[rec_run, rec],
            )
        # Osci1T.TimebaseGet ResponseTypes = ["i", "i", "*f"] →
        # Variables = [current_index, n_timebases, timebases_array_seconds].
        d = _decoded(rec.return_value)
        if len(d) < 3 or not isinstance(d[2], (list, tuple)):
            return SkillResult(
                skill_name="GetOsciTimebases", success=False,
                error=f"Unexpected TimebaseGet shape: {rec.return_value!r}",
                nanonis_calls=[rec_run, rec],
            )
        try:
            current_index = int(d[0])
            dts = [_unwrap(v) for v in d[2]]
        except (TypeError, ValueError, IndexError) as exc:
            return SkillResult(
                skill_name="GetOsciTimebases", success=False,
                error=f"Decode error: {exc}", nanonis_calls=[rec_run, rec],
            )
        timebases = [
            {"index": i, "dt_s": dt, "fs_hz": (1.0 / dt if dt > 0 else 0.0)}
            for i, dt in enumerate(dts)
        ]
        return SkillResult(
            skill_name="GetOsciTimebases", success=True,
            data={
                "timebases": timebases,
                "n_timebases": len(timebases),
                "current_index": current_index,
            },
            nanonis_calls=[rec_run, rec],
        )


class SetOsciTimebase(BaseSkill):
    """Select the Osci1T timebase (sample rate) by index."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetOsciTimebase",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "按索引设置 Oscilloscope-1-Channel（Osci1T）时基。先用 GetOsciTimebases 拿到 "
                "index→sample-rate 的对应关系。纯配置 —— 不移动针尖，也不改任何 setpoint。"
            ),
            parameters=[
                ParameterSpec(
                    name="timebase_index",
                    type="int",
                    description=(
                        "在 GetOsciTimebases 返回的时基列表里的索引。索引越小 = 时基越快 / "
                        "采样率越高。"
                    ),
                    required=True, min_value=0,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["oscilloscope", "timebase", "samplerate", "configure"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["timebase_index"])
        rec_run = context.safe_call("Osci1T_Run")
        if rec_run.error and "NeedModule" in rec_run.error:
            return SkillResult(
                skill_name="SetOsciTimebase", success=False,
                error=(
                    "Osci1T module is not loaded in this Nanonis "
                    "configuration — cannot set a hardware timebase."
                ),
                nanonis_calls=[rec_run],
            )
        rec = context.safe_call("Osci1T_TimebaseSet", idx)
        if rec.error:
            return SkillResult(
                skill_name="SetOsciTimebase", success=False,
                error=f"Osci1T_TimebaseSet failed: {rec.error}",
                nanonis_calls=[rec_run, rec],
            )
        return SkillResult(
            skill_name="SetOsciTimebase", success=True,
            data={"timebase_index": idx},
            nanonis_calls=[rec_run, rec],
        )


def _decoded(rv):
    if isinstance(rv, tuple) and len(rv) >= 3 and isinstance(rv[2], list):
        return rv[2]
    if isinstance(rv, list):
        return rv
    return []
