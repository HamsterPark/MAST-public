"""Raw real-time time-series capture — high-rate polling of one signal.

The general-purpose "tap a signal for N seconds" skill. Picks the right
TCP endpoint for the requested channel and dumps a `(timestamps, values)`
trace into the result.

Supported channels:
  - "current" → ``Current_Get``      (fast path, ~0.5 ms RTT)
  - "z"       → ``ZCtrl_ZPosGet``    (fast path, ~0.4 ms RTT)
  - "bias"    → ``Bias_Get``         (fast path)
  - other strings interpreted as a 0–127 signal index → routed via
    ``Signals_ValGet``. Falls back to per-sample Tap (50 Hz max)
    on this path.

Output is JSON-safe (lists of floats), suitable for either the GUI's
Data tab or as input to ``MonitorCurrentFFT`` / a downstream FFT.
"""
from __future__ import annotations

import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

class CaptureSignalBuffer(BaseSkill):
    """Poll one signal at high rate for N seconds; return the trace."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CaptureSignalBuffer",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "高速率采集单个 Nanonis 信号的时间序列。要亚毫秒 RTT 就用 "
                "channel='current'/'z'/'bias'；要走通用的 Signals_ValGet "
                "路径就传信号索引字符串（'0'..'127'）（受 TCP Tap 限制，~50 Hz 封顶）。"
            ),
            parameters=[
                ParameterSpec(
                    name="channel",
                    type="str",
                    description=(
                        "快速路径用 'current' / 'z' / 'bias'；通用路径用数字信号索引 0-127。"
                    ),
                    required=False, default="current",
                ),
                ParameterSpec(
                    name="duration_s",
                    type="float",
                    description="采集窗口，单位秒。",
                    unit="s", required=False, default=1.0,
                    min_value=0.01, max_value=60.0,
                ),
                ParameterSpec(
                    name="poll_hz",
                    type="float",
                    description=(
                        "目标轮询速率。受 TCP RTT 封顶（快速路径最高 ~2 kHz， Signals_ValGet 上 ~50 "
                        "Hz）。"
                    ),
                    unit="Hz", required=False, default=1000.0,
                    min_value=1.0, max_value=4000.0,
                ),
                ParameterSpec(
                    name="include_samples",
                    type="bool",
                    description=(
                        "为 True（默认）时返回完整样本列表。为 False 时只返回 min/max/mean/std + "
                        "长度。"
                    ),
                    required=False, default=True,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["signal", "capture", "stream", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        channel = (params.get("channel") or "current").lower()
        duration = float(params.get("duration_s", 1.0))
        poll_hz = float(params.get("poll_hz", 1000.0))
        include_samples = bool(params.get("include_samples", True))

        # Resolve which TCP call to use per poll, as a THUNK holding a LITERAL verb.
        #
        # It used to resolve `method = _FAST_PATHS[channel]` and call
        # `safe_call(method, *args)`. The verb strings were in the source but not
        # inside a safe_call(...), and every safety tool in this repo finds Nanonis
        # calls by grepping exactly that: the abort-policy checker, the security
        # audit, the API-coverage census. A verb reached through a variable is
        # invisible to all three — the call happens and nothing that guards this
        # system can see it. Four literal branches cost eight lines and keep the
        # tooling honest.
        if channel == "current":
            poll = lambda: context.safe_call("Current_Get")            # noqa: E731
            unit, mode = "A", "fast_path"
        elif channel == "z":
            poll = lambda: context.safe_call("ZCtrl_ZPosGet")          # noqa: E731
            unit, mode = "m", "fast_path"
        elif channel == "bias":
            poll = lambda: context.safe_call("Bias_Get")               # noqa: E731
            unit, mode = "V", "fast_path"
        else:
            try:
                idx = int(channel)
            except ValueError:
                return SkillResult(
                    skill_name="CaptureSignalBuffer", success=False,
                    error=(
                        f"Unknown channel: {channel!r}. "
                        "Use 'current'/'z'/'bias' or a signal index 0-127."
                    ),
                )
            # wait_for_newest=0 → no double-Tap delay
            poll = lambda: context.safe_call("Signals_ValGet", idx, 0)  # noqa: E731
            unit, mode = "", "signals_valget"

        period = 1.0 / poll_hz
        samples: list[float] = []
        timestamps: list[float] = []
        records = []
        t0 = time.perf_counter()
        next_poll = t0
        while True:
            now = time.perf_counter()
            elapsed = now - t0
            if elapsed >= duration:
                break
            if hasattr(context, "check_abort") and context.check_abort():
                break
            if now < next_poll:
                time.sleep(min(next_poll - now, 0.001))
                continue
            rec = poll()
            records.append(rec)
            if rec.error:
                next_poll += period
                continue
            d = _decoded(rec.return_value)
            if d:
                v = d[0]
                if isinstance(v, tuple) and v:
                    v = v[0]
                samples.append(float(v))
                timestamps.append(elapsed)
            next_poll += period

        n = len(samples)
        if n == 0:
            return SkillResult(
                skill_name="CaptureSignalBuffer", success=False,
                error="No samples collected (TCP errors or aborted)",
                nanonis_calls=records,
            )

        actual_dur = (timestamps[-1] - timestamps[0]) if n > 1 else 1.0 / poll_hz
        actual_fs = (n - 1) / actual_dur if actual_dur > 0 else poll_hz
        smin = min(samples); smax = max(samples)
        smean = sum(samples) / n
        svar = sum((x - smean) ** 2 for x in samples) / n
        sstd = svar ** 0.5

        data: dict = {
            "channel": channel,
            "mode": mode,
            "unit": unit,
            "n_samples": n,
            "requested_duration_s": duration,
            "actual_duration_s": actual_dur,
            "actual_fs_hz": actual_fs,
            "min": smin, "max": smax, "mean": smean, "std": sstd,
        }
        # Repeated bit-identical samples may indicate a cached or frozen readout,
        # a disconnected channel, or a controller that is not running. Report the
        # constant-signal condition rather than presenting a plausible scalar alone.
        if n >= 8 and smax == smin:
            data["anomaly"] = (
                f"⚠ {n} 个采样点的值**完全相同**（{smin:.6g} {unit}，std=0）。"
                "真实物理信号不会逐位重复。最可能的原因：读数被缓存/冻结、"
                "通道未连接、或该通道当前并未在采集。"
                "**不要把这些数字当作测量结果使用**，先确认信号源。"
            )
        elif n >= 8 and smean != 0 and (smax - smin) / abs(smean) < 1e-9:
            data["anomaly"] = (
                f"⚠ {n} 个采样点的极差仅 {smax - smin:.3g} {unit}"
                f"（相对均值 {(smax - smin) / abs(smean):.1e}），近乎恒定。"
                "可能是读数被缓存或通道未在采集；请先确认信号源再使用这些数字。"
            )
        if include_samples:
            data["samples"] = samples
            data["timestamps_s"] = timestamps

        return SkillResult(
            skill_name="CaptureSignalBuffer", success=True,
            data=data, nanonis_calls=records,
        )


def _decoded(rv):
    if isinstance(rv, tuple) and len(rv) >= 3 and isinstance(rv[2], list):
        return rv[2]
    if isinstance(rv, list):
        return rv
    return []
