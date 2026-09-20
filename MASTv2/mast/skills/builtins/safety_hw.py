"""Hardware safety skills — Nanonis SafeTip module.

vendored from v1 mast/skills/builtins/safety_hw.py 2026-04-23. Zero behavioural changes.
4 skills: EnableSafeTip, GetSafeTipStatus, GetSafeTipProps, GetSafeTipSignal.
"""

from __future__ import annotations

from mast.skills.base import BaseSkill
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)


class EnableSafeTip(BaseSkill):
    """Enable/disable Nanonis hardware SafeTip protection."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="EnableSafeTip",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="启用或禁用 Nanonis 硬件 SafeTip 保护。",
            parameters=[
                ParameterSpec(
                    name="enable",
                    type="bool",
                    description=(
                        "True 为启用，False 为禁用。默认 True：不带参数、走默认的"
                        "调用会**启用** SafeTip（硬件针尖保护应当保持开启，除非"
                        "用户显式传 enable=False）。"
                    ),
                    # Default ON (safety fix): a bare/defaulted call enables
                    # SafeTip. Disabling requires an explicit enable=False (the
                    # disable path is intentionally NOT gated — see audit #2).
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["safety", "hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # Default ON: a missing/None ``enable`` ENABLES SafeTip (see metadata).
        enable = params.get("enable")
        if enable is None:
            enable = True
        record = context.safe_call("SafeTip_OnOffSet", int(enable))
        if record.error:
            return SkillResult(
                skill_name="EnableSafeTip",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="EnableSafeTip",
            success=True,
            data={"enabled": enable},
            nanonis_calls=[record],
        )


class GetSafeTipStatus(BaseSkill):
    """Read Nanonis SafeTip protection status."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSafeTipStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前 SafeTip 保护状态。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["safety", "hardware", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("SafeTip_OnOffGet")
        if record.error:
            return SkillResult(
                skill_name="GetSafeTipStatus",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        enabled = False
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            enabled = bool(parsed[2][0])
        return SkillResult(
            skill_name="GetSafeTipStatus",
            success=True,
            data={"enabled": enabled},
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# SafeTip — PropsGet, SignalGet
# ---------------------------------------------------------------------------


class GetSafeTipProps(BaseSkill):
    """Get SafeTip configuration properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSafeTipProps",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 SafeTip 配置：自动恢复、自动暂停扫描、阈值。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["safety", "hardware", "props", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("SafeTip_PropsGet")
        if record.error:
            return SkillResult(
                skill_name="GetSafeTipProps",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # SafeTip.PropsGet ResponseTypes = ["H", "H", "f"] ->
        # parsed[2] = [auto_recovery, auto_pause_scan, threshold].
        parsed = record.return_value
        auto_recovery = False
        auto_pause_scan = False
        threshold = 0.0
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            values = parsed[2]
            if isinstance(values, (list, tuple)) and len(values) >= 3:
                auto_recovery = bool(values[0])
                auto_pause_scan = bool(values[1])
                threshold = float(values[2])
        return SkillResult(
            skill_name="GetSafeTipProps",
            success=True,
            data={
                "auto_recovery": auto_recovery,
                "auto_pause_scan": auto_pause_scan,
                "threshold": threshold,
            },
            nanonis_calls=[record],
        )


class GetSafeTipSignal(BaseSkill):
    """Get current SafeTip signal value."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSafeTipSignal",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前 SafeTip 信号值。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["safety", "hardware", "signal", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("SafeTip_SignalGet")
        if record.error:
            return SkillResult(
                skill_name="GetSafeTipSignal",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # SafeTip.SignalGet ResponseTypes = ["f"] -> parsed[2][0] = signal value.
        parsed = record.return_value
        signal_value = 0.0
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            values = parsed[2]
            if isinstance(values, (list, tuple)) and len(values) > 0:
                signal_value = float(values[0])
            elif isinstance(values, (int, float)):
                signal_value = float(values)
        elif isinstance(parsed, (int, float)):
            signal_value = float(parsed)
        return SkillResult(
            skill_name="GetSafeTipSignal",
            success=True,
            data={"signal_value": signal_value},
            nanonis_calls=[record],
        )
