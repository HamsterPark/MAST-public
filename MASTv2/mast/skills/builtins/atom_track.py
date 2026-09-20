"""Atom Tracking configuration skill.

vendored from v1 mast/skills/builtins/atom_track.py 2026-04-23. Zero behavioural changes.
4 skills: ConfigureAtomTrack, AtomTrackDriftComp, AtomTrackQuickCompStart, AtomTrackStatusGet.
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


class ConfigureAtomTrack(BaseSkill):
    """Configure and control Atom Tracking."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureAtomTrack",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 Atom Tracking 参数，并启用/禁用各项控制。",
            parameters=[
                ParameterSpec(
                    name="integral_gain",
                    type="float",
                    description="控制器积分增益",
                    required=True,
                ),
                ParameterSpec(
                    name="frequency_hz",
                    type="float",
                    description="调制频率",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="amplitude_m",
                    type="float",
                    description="调制幅度",
                    unit="m",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="phase_deg",
                    type="float",
                    description="调制相位",
                    unit="deg",
                    required=False,
                    default=0.0,
                    min_value=-360.0,
                    max_value=360.0,
                ),
                ParameterSpec(
                    name="switch_off_delay_s",
                    type="float",
                    description="关闭前的位置平均时间",
                    unit="s",
                    required=False,
                    default=0.5,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="enable_modulation",
                    type="bool",
                    description="启用调制",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="enable_controller",
                    type="bool",
                    description="启用控制器",
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["atomtrack", "tracking", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []
        # AtomTrack_PropsSet(I_gain, Freq_Hz, Amp_m, Phase_deg, Switch_off_delay_s)
        rec_props = context.safe_call(
            "AtomTrack_PropsSet",
            params["integral_gain"],
            params["frequency_hz"],
            params["amplitude_m"],
            params.get("phase_deg", 0.0),
            params.get("switch_off_delay_s", 0.5),
        )
        calls.append(rec_props)
        if rec_props.error:
            return SkillResult(
                skill_name="ConfigureAtomTrack",
                success=False,
                error=rec_props.error,
                nanonis_calls=calls,
            )

        # AtomTrack_CtrlSet(AT_control, Status) -- 0=Modulation, 1=Controller, 2=Drift.
        # ResponseTypes=[] so only .error is meaningful. A failed enable used to
        # be swallowed (the skill still reported success=True), leaving the
        # caller believing modulation/controller were on when they were not.
        modulation_enabled = bool(params.get("enable_modulation", True))
        controller_enabled = bool(params.get("enable_controller", True))
        if modulation_enabled:
            rec_mod = context.safe_call("AtomTrack_CtrlSet", 0, 1)
            calls.append(rec_mod)
            if rec_mod.error:
                return SkillResult(
                    skill_name="ConfigureAtomTrack",
                    success=False,
                    error=f"enable modulation failed: {rec_mod.error}",
                    nanonis_calls=calls,
                )
        if controller_enabled:
            rec_ctrl = context.safe_call("AtomTrack_CtrlSet", 1, 1)
            calls.append(rec_ctrl)
            if rec_ctrl.error:
                return SkillResult(
                    skill_name="ConfigureAtomTrack",
                    success=False,
                    error=f"enable controller failed: {rec_ctrl.error}",
                    nanonis_calls=calls,
                )

        return SkillResult(
            skill_name="ConfigureAtomTrack",
            success=True,
            data={
                "integral_gain": params["integral_gain"],
                "frequency_hz": params["frequency_hz"],
                "amplitude_m": params["amplitude_m"],
                "modulation_enabled": modulation_enabled,
                "controller_enabled": controller_enabled,
            },
            nanonis_calls=calls,
        )


# ---------------------------------------------------------------------------
# AtomTrack — DriftComp, QuickCompStart, StatusGet
# ---------------------------------------------------------------------------


class AtomTrackDriftComp(BaseSkill):
    """Apply drift measurement to the drift compensation."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AtomTrackDriftComp",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="把 Atom Tracking 测得的漂移应用到漂移补偿上。",
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["atomtrack", "drift", "compensation", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("AtomTrack_DriftComp")
        if record.error:
            return SkillResult(
                skill_name="AtomTrackDriftComp",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="AtomTrackDriftComp",
            success=True,
            data={"drift_compensation_applied": True},
            nanonis_calls=[record],
        )


class AtomTrackQuickCompStart(BaseSkill):
    """Start tilt or drift compensation via Atom Tracking."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AtomTrackQuickCompStart",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="通过 Atom Tracking 启动倾斜或漂移补偿。",
            parameters=[
                ParameterSpec(
                    name="compensation_type",
                    type="int",
                    description="0=倾斜补偿, 1=漂移补偿",
                    required=True,
                    min_value=0,
                    max_value=1,
                    allowed_values=[0, 1],
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["atomtrack", "compensation", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        comp_type = params["compensation_type"]
        record = context.safe_call("AtomTrack_QuickCompStart", comp_type)
        if record.error:
            return SkillResult(
                skill_name="AtomTrackQuickCompStart",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        label = "Tilt" if comp_type == 0 else "Drift"
        return SkillResult(
            skill_name="AtomTrackQuickCompStart",
            success=True,
            data={"compensation_type": label, "started": True},
            nanonis_calls=[record],
        )


class AtomTrackStatusGet(BaseSkill):
    """Get the status of an Atom Tracking control."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AtomTrackStatusGet",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取某一项 Atom Tracking 控制（调制、控制器或漂移）的开/关状态。",
            parameters=[
                ParameterSpec(
                    name="control",
                    type="int",
                    description="0=调制, 1=控制器, 2=漂移测量",
                    required=True,
                    min_value=0,
                    max_value=2,
                    allowed_values=[0, 1, 2],
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["atomtrack", "status", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        control = params["control"]
        record = context.safe_call("AtomTrack_StatusGet", control)
        if record.error:
            return SkillResult(
                skill_name="AtomTrackStatusGet",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # AtomTrack.StatusGet ResponseTypes=["H"] -> parsed[2][0] is the
        # Status (0=Off, 1=On). Reading parsed[0] (the empty error string)
        # made this silently report "Off" for every control on real hardware.
        parsed = record.return_value
        status = False
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                status = bool(vals[0])
            elif isinstance(vals, (int, float)):
                status = bool(vals)
        elif isinstance(parsed, int):
            status = bool(parsed)
        _CTRL_MAP = {0: "Modulation", 1: "Controller", 2: "Drift Measurement"}
        return SkillResult(
            skill_name="AtomTrackStatusGet",
            success=True,
            data={
                "control": _CTRL_MAP.get(control, str(control)),
                "status": status,
            },
            nanonis_calls=[record],
        )
