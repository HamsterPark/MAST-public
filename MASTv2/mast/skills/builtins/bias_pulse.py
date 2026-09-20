"""Bias pulse skill.

vendored from v1 mast/skills/builtins/bias_pulse.py 2026-04-23. Zero behavioural changes.
1 skill: BiasPulse.
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.builtins._tip_xy import tip_xy_fields


class BiasPulse(BaseSkill):
    """Generate a hardware-timed bias pulse."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="BiasPulse",
            version="1.0.0",
            category=SkillCategory.WRITE,
            capabilities=frozenset({"bias_pulse"}),
            # AUTO (2026-06-11 safety re-scoping): the ONLY action that can
            # physically wreck the instrument is a coarse Z approach toward the
            # sample (pan-type stepper, open loop — see motor.py / _is_coarse_
            # sample_approach). A bias pulse is bounded by Nanonis's own bias
            # range plus the global ±10 V SafetyGate cap, so it cannot damage
            # hardware — it runs autonomously, no HITL gate.
            safety_level=SafetyLevel.AUTO,
            description="由硬件计时发出单个偏压脉冲。",
            parameters=[
                ParameterSpec(
                    name="width_s",
                    type="float",
                    description="脉冲宽度，单位秒",
                    unit="s",
                    required=True,
                    min_value=1e-6,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="bias_v",
                    type="float",
                    description="脉冲期间的偏压",
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="z_hold",
                    type="int",
                    description="Z controller 的保持方式：0=不变，1=保持，2=不保持",
                    required=False,
                    default=1,
                    allowed_values=[0, 1, 2],
                ),
                ParameterSpec(
                    name="absolute",
                    type="bool",
                    description="True=绝对偏压，False=相对当前值",
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["bias", "pulse", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        width_s = params["width_s"]
        bias_v = params["bias_v"]
        z_hold = params.get("z_hold", 1)
        absolute = params.get("absolute", True)
        abs_rel = 2 if absolute else 1  # 1=relative, 2=absolute
        # WHERE the pulse lands, read before firing it. A pulse modifies the
        # surface (that is usually the point of it), so the scan map needs its
        # actual coordinate, not the ≤1 s-old cached one. Best-effort.
        spot = tip_xy_fields(context)
        # Bias_Pulse(Wait_until_done, Bias_pulse_width_s, Bias_value_V,
        #            Z_Controller_on_hold, Pulse_absolute_relative)
        record = context.safe_call("Bias_Pulse", 1, width_s, bias_v, z_hold, abs_rel)
        if record.error:
            return SkillResult(
                skill_name="BiasPulse",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="BiasPulse",
            success=True,
            data={
                "width_s": width_s,
                "bias_v": bias_v,
                "z_hold": z_hold,
                "absolute": absolute,
                **spot,
            },
            nanonis_calls=[record],
        )
