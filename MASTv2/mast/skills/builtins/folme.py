"""Follow-Me tip speed, oversampling, Point & Shoot, and stop skills.

vendored from v1 mast/skills/builtins/folme.py 2026-04-23. Zero behavioural changes.
8 skills: SetTipSpeed, GetTipSpeed, SetFolMeOversampling, StopFolMe,
          GetPointShootOnOff, SetPointShootOnOff, SetPointShootExperiment,
          GetPointShootProps.
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill


class SetTipSpeed(BaseSkill):
    """Set tip movement speed for Follow-Me mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetTipSpeed",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Follow-Me（XY 定位）模式下的针尖移动速度。",
            parameters=[
                ParameterSpec(
                    name="speed_m_s",
                    type="float",
                    description="表面移动速度，单位米每秒",
                    unit="m/s",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="custom_speed",
                    type="bool",
                    description="True=用自定义速度，False=用扫描速度",
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "speed", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        speed = params["speed_m_s"]
        custom = int(params.get("custom_speed", True))
        record = context.safe_call("FolMe_SpeedSet", speed, custom)
        if record.error:
            return SkillResult(
                skill_name="SetTipSpeed",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetTipSpeed",
            success=True,
            data={"speed_m_s": speed, "custom_speed": bool(custom)},
            nanonis_calls=[record],
        )


class GetTipSpeed(BaseSkill):
    """Read tip movement speed in Follow-Me mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetTipSpeed",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 Follow-Me 模式下的针尖表面速度与 custom-speed 标志。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "speed", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("FolMe_SpeedGet")
        if record.error:
            return SkillResult(
                skill_name="GetTipSpeed",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {
                    "speed_m_s": float(vals[0]),
                    "custom_speed": bool(vals[1]),
                }
        return SkillResult(
            skill_name="GetTipSpeed",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


class SetFolMeOversampling(BaseSkill):
    """Set oversampling for Follow-Me data acquisition."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetFolMeOversampling",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Follow-Me 模式下移动时所采数据的 oversampling。",
            parameters=[
                ParameterSpec(
                    name="oversampling",
                    type="int",
                    description="oversampling 取值",
                    required=True,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "oversampling", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        oversampling = params["oversampling"]
        record = context.safe_call("FolMe_OversamplSet", oversampling)
        if record.error:
            return SkillResult(
                skill_name="SetFolMeOversampling",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetFolMeOversampling",
            success=True,
            data={"oversampling": oversampling},
            nanonis_calls=[record],
        )


class StopFolMe(BaseSkill):
    """Stop tip movement in Follow-Me mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopFolMe",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="停止 Follow-Me 模式下的针尖移动。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "stop", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("FolMe_Stop")
        if record.error:
            return SkillResult(
                skill_name="StopFolMe",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="StopFolMe",
            success=True,
            data={"stopped": True},
            nanonis_calls=[record],
        )


class GetPointShootOnOff(BaseSkill):
    """Read Point & Shoot enabled status in Follow-Me mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPointShootOnOff",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 Follow-Me 模式下 Point & Shoot 是启用还是禁用。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "point_shoot", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("FolMe_PSOnOffGet")
        if record.error:
            return SkillResult(
                skill_name="GetPointShootOnOff",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 1:
                data = {"enabled": bool(vals[0])}
        return SkillResult(
            skill_name="GetPointShootOnOff",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


class SetPointShootOnOff(BaseSkill):
    """Enable or disable Point & Shoot in Follow-Me mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPointShootOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="启用或禁用 Follow-Me 模式下的 Point & Shoot。",
            parameters=[
                ParameterSpec(
                    name="enable",
                    type="bool",
                    description="True 为启用 Point & Shoot，False 为禁用",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "point_shoot", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        enable = params["enable"]
        status = 1 if enable else 0
        record = context.safe_call("FolMe_PSOnOffSet", status)
        if record.error:
            return SkillResult(
                skill_name="SetPointShootOnOff",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPointShootOnOff",
            success=True,
            data={"enabled": enable},
            nanonis_calls=[record],
        )


class SetPointShootExperiment(BaseSkill):
    """Select the Point & Shoot experiment in Follow-Me mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPointShootExperiment",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="选择 Follow-Me 模式下 Point & Shoot 要运行哪个 experiment。",
            parameters=[
                ParameterSpec(
                    name="experiment_index",
                    type="int",
                    description="要选择的 Point & Shoot experiment 的索引",
                    required=True,
                    min_value=0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "point_shoot", "experiment", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        exp_idx = params["experiment_index"]
        record = context.safe_call("FolMe_PSExpSet", exp_idx)
        if record.error:
            return SkillResult(
                skill_name="SetPointShootExperiment",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPointShootExperiment",
            success=True,
            data={"experiment_index": exp_idx},
            nanonis_calls=[record],
        )


class GetPointShootProps(BaseSkill):
    """Read Point & Shoot configuration in Follow-Me mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPointShootProps",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取 Point & Shoot 配置：auto-resume、basename、"
                "外部 VI 路径、测量前延时。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["folme", "point_shoot", "config", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("FolMe_PSPropsGet")
        if record.error:
            return SkillResult(
                skill_name="GetPointShootProps",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 7:
                data = {
                    "auto_resume": bool(vals[0]),
                    "use_own_basename": bool(vals[1]),
                    "basename_size": int(vals[2]),
                    "basename": str(vals[3]),
                    "ext_vi_path_size": int(vals[4]),
                    "ext_vi_path": str(vals[5]),
                    "pre_measure_delay_s": float(vals[6]),
                }
        return SkillResult(
            skill_name="GetPointShootProps",
            success=True,
            data=data,
            nanonis_calls=[record],
        )
