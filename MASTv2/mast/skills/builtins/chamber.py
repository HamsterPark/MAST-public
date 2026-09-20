"""Chamber pressure — the reading, and what the coarse-motion interlock makes of it.

Until now the vacuum gauge fed a dashboard and nothing else: no skill read it, no
agent tool exposed it, and no decision depended on it. That is fine for a number
you look at and wrong for a number that decides whether driving a few hundred
volts into a piezo starts an arc.

This module is READ-ONLY and adds no capability. It exists so the agent (and the
operator, through the same code path) can ask *why* a coarse move was refused,
and so an autonomous run can wait for pressure to fall instead of retrying a
refusal it does not understand.

The verdict itself lives in :mod:`mast.core.vacuum_interlock` — one computation,
several audiences, exactly like the scan-map analysis.
"""

from __future__ import annotations

from mast.core.types import (
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill


class GetChamberPressure(BaseSkill):
    """Read the chamber pressure and the coarse-motion interlock verdict."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetChamberPressure",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取腔体压强(Pa)并给出**粗动互锁裁决**:现在能不能动粗动马达,以及理由。\n"
                "在中间真空区(约 0.1–1000 Pa = 1e-3–10 mbar,Paschen 极小值附近)给粗动"
                "压电加几百伏会打火击穿叠堆 —— 抽气和放气途中正好穿过这个区间。\n"
                "**allow=false 时不要重试粗动**:这是硬闸门,不是建议。"
                "读不到真空计也会是 false(读不到 ≠ 真空好);那种情况需要用户在界面上"
                "签署一次「当前气压安全」,你不能替他签。"
            ),
            parameters=[],
            estimated_duration_s=0.2,
            composition_level=0,
            tags=["vacuum", "pressure", "read", "safety", "coarse"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from mast.core import vacuum_interlock as vac

        verdict = vac.check()
        sample = vac.current_sample()
        data = verdict.as_dict()
        data["corona_danger_zone_pa"] = list(vac.CORONA_ZONE_PA)
        if sample is not None:
            data["sensor"] = sample.sensor_name
            data["sensor_class"] = sample.sensor_class
            data["raw_value"] = sample.value
            data["raw_unit"] = sample.unit
            data["sensor_status"] = sample.status
        att = vac.get_attestation()
        if att is not None:
            data["attestation"] = {
                "reason": att.reason, "label": att.label(),
                "signed_by": att.signed_by,
                "expired": att.expired(),
                "remaining_h": round(att.remaining_s() / 3600.0, 2),
            }
        # The skill SUCCEEDS whenever the check ran — the verdict is in the data.
        # A refusal is an answer, not a failure, and reporting it as an error
        # would make an ordinary "still pumping down" look like a broken tool.
        return SkillResult(skill_name="GetChamberPressure", success=True, data=data)


__all__ = ["GetChamberPressure"]
