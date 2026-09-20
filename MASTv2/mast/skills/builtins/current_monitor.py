"""Real-time tunneling current monitoring for contact-detection workflows.

Used by ``ShapeTipOnSurface`` to detect the moment the tip touches the sample
during a progressive tip-shaper plunge:

  - poll ``Current_Get`` at ~poll_hz Hz for ``duration_s`` seconds
  - report min / max / mean / std of |I|
  - flag "contact" when |I| exceeds ``contact_threshold_a`` for at least
    ``min_contact_samples`` consecutive polls

This is a short, well-bounded read-only skill: no Nanonis state mutation.
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


class MonitorCurrent(BaseSkill):
    """Poll tunneling current for a bounded window; flag contact events."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MonitorCurrent",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在一个固定时间窗内高速采样隧道电流。"
                "返回 min/max/mean/std，外加一个 `contact_detected` 标志 —— "
                "当 |I| 连续 min_contact_samples 次轮询都高于 "
                "contact_threshold_a 时置位。"
            ),
            parameters=[
                ParameterSpec(
                    name="duration_s",
                    type="float",
                    description=(
                        "持续轮询多久，单位**秒**。常用 0.1–5。"
                        "示例：0.5（= 500 ms）✓；60（= 1 分钟）只用于"
                        "长时间观察。"
                    ),
                    unit="s",
                    required=False,
                    default=1.0,
                    min_value=0.01,
                    max_value=60.0,
                ),
                ParameterSpec(
                    name="poll_hz",
                    type="float",
                    description=(
                        "轮询速率，单位 Hz。Nanonis TCP 大致有 ~kHz 的能力，"
                        "但往返抖动使得 ≤200 Hz 才现实。"
                    ),
                    unit="Hz",
                    required=False,
                    default=100.0,
                    min_value=1.0,
                    max_value=500.0,
                ),
                ParameterSpec(
                    name="contact_threshold_a",
                    type="float",
                    description=(
                        "判定接触的电流绝对值阈值，单位**安培**。"
                        "STM 的 setpoint 在 pA–nA 量级；接触起始通常是 "
                        "10–100 nA。示例：50n（= 50 nA）✓；1"
                        "（= 1 安培 —— STM 上绝不会出现）✗。"
                    ),
                    unit="A",
                    required=False,
                    default=5e-8,  # 50 nA — comfortably above normal setpoints
                    min_value=1e-12,
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="min_contact_samples",
                    type="int",
                    description=(
                        "判定接触所需的连续超阈轮询次数。"
                        "用来滤掉尖峰。"
                    ),
                    required=False,
                    default=3,
                    min_value=1,
                    max_value=100,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["current", "monitor", "contact", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        duration = float(params.get("duration_s", 1.0))
        poll_hz = float(params.get("poll_hz", 100.0))
        threshold = float(params.get("contact_threshold_a", 5e-8))
        min_streak = int(params.get("min_contact_samples", 3))

        period = 1.0 / poll_hz
        samples: list[float] = []
        timestamps: list[float] = []
        call_records = []
        consecutive_over = 0
        contact_detected = False
        contact_at_s = -1.0

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
                # Sleep just enough to hit the next slot — but cap so abort
                # checks happen roughly every 10 ms.
                time.sleep(min(next_poll - now, 0.01))
                continue
            rec = context.safe_call("Current_Get")
            call_records.append(rec)
            if rec.error:
                # Soft-fail: skip this sample, continue.
                next_poll += period
                continue
            parsed = rec.return_value
            value = self._extract_current(parsed)
            if value is None:
                next_poll += period
                continue
            samples.append(value)
            timestamps.append(elapsed)

            if abs(value) >= threshold:
                consecutive_over += 1
                if consecutive_over >= min_streak and not contact_detected:
                    contact_detected = True
                    contact_at_s = elapsed
            else:
                consecutive_over = 0
            next_poll += period

        n = len(samples)
        if n == 0:
            return SkillResult(
                skill_name="MonitorCurrent",
                success=False,
                error="No samples collected (TCP errors or aborted before first poll)",
                nanonis_calls=call_records,
            )

        # Stats on absolute current (we care about magnitude for contact).
        abs_samples = [abs(s) for s in samples]
        mean = sum(abs_samples) / n
        var = sum((x - mean) ** 2 for x in abs_samples) / n
        std = var ** 0.5

        return SkillResult(
            skill_name="MonitorCurrent",
            success=True,
            data={
                "n_samples": n,
                "duration_s": duration,
                "actual_duration_s": time.perf_counter() - t0,
                "min_abs_a": min(abs_samples),
                "max_abs_a": max(abs_samples),
                "mean_abs_a": mean,
                "std_abs_a": std,
                "signed_min_a": min(samples),
                "signed_max_a": max(samples),
                "contact_detected": contact_detected,
                "contact_at_s": contact_at_s if contact_detected else None,
                "contact_threshold_a": threshold,
                "samples_a": samples,        # full trace — small, JSON-safe
                "timestamps_s": timestamps,
            },
            nanonis_calls=call_records,
        )

    @staticmethod
    def _extract_current(parsed) -> float | None:
        """Pull the current float out of the Nanonis wrapper tuple.

        Current_Get decode shape: (err_str, raw_bytes, [value]).
        Fall back to common alternatives if the wrapper layout differs.
        """
        if parsed is None:
            return None
        if isinstance(parsed, (int, float)):
            return float(parsed)
        if isinstance(parsed, (list, tuple)):
            # Last element first — that's where the decoded list lives.
            tail = parsed[-1]
            if isinstance(tail, (list, tuple)) and tail:
                first = tail[0]
                if isinstance(first, (int, float)):
                    return float(first)
            if isinstance(tail, (int, float)):
                return float(tail)
        return None
