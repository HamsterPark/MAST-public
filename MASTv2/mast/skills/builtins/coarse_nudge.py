# -*- coding: utf-8 -*-
"""StepCoarseXY：通过已有粗动流程执行少量步数的位置调整。

委托 RelocateCoarseXY(allow_revisit=True)，仅显式放开重复访问的效率约束。
清障、真空互锁、驱动读回、降偏压、电流检查、里程表与重新进针仍使用同一实现。
常规换区使用 RelocateCoarseXY；本入口用于步长标定或微调，受小步数上限约束。
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

#: 超过这个步数就不该走这条路了 —— 那是换区，用 RelocateCoarseXY。
#: 不是安全上限（真正的上限在 RelocateCoarseXY 的 axis_step_budget），
#: 是**用途上限**：让「挪一下」和「换个地方」在调用点就分得开。
_NUDGE_MAX_STEPS = 60


class StepCoarseXY(BaseSkill):
    """Nudge the coarse stage a few lateral steps (revisiting allowed)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StepCoarseXY",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把粗动台横向**挪几步**（可以只挪一步）——"
                "标定步长、微调位置用。"
                "\n\n转调 `RelocateCoarseXY(allow_revisit=True)`，**一层薄壳，"
                "不重写任何东西**：清障阶梯、真空互锁、驱动电压读回、降偏压到 "
                "0.5 V、电流归零证明、里程表记录、重新进针，全是那一份实现在跑。"
                "\n\n它存在的理由：`MotorMove` 在自主路径上被硬闸挡着"
                "（「有防护的路才是自主路径」），而 `RelocateCoarseXY` 的落点复核"
                "要求离已访问站点 ≥200 步 —— **「挪 2 步」两边都过不去**。"
                "那 200 步是效率约束（别重复用同一片表面），不是安全约束，"
                "所以这里明说地跳过它，安全检查一条不跳。"
                "\n\n**常规换区不要用它**，用 `RelocateCoarseXY`：重复用同一片"
                "表面是真的浪费。超过 %d 步会被拒 —— 那是换区不是挪一下。"
                % _NUDGE_MAX_STEPS
            ),
            parameters=[
                ParameterSpec(
                    name="axis", type="str", required=True,
                    description="横向轴：'x' 或 'y'", allowed_values=["x", "y"]),
                ParameterSpec(
                    name="direction", type="str", required=True,
                    description="方向：'+' 或 '-'", allowed_values=["+", "-"]),
                ParameterSpec(
                    name="steps", type="int", required=False, default=1,
                    min_value=1, max_value=_NUDGE_MAX_STEPS,
                    description=("走几步。默认 1。超过 %d 步请改用 "
                                 "RelocateCoarseXY。" % _NUDGE_MAX_STEPS)),
                ParameterSpec(
                    name="reapproach", type="bool", required=False, default=True,
                    description="走完是否自动重新进针。"),
                ParameterSpec(
                    name="prewithdraw_steps", type="int", required=False,
                    default=None, min_value=0, max_value=100_000,
                    description=("横移前的粗动 Z 退针步数，用于清障。留空采用 instrument_profile 的配置；每步位移必须按当前硬件标定，不能由步数直接假定固定距离。"
                                 )),
                ParameterSpec(
                    name="dry_run", type="bool", required=False, default=False,
                    description="跑完整条相位但不发出任何移动命令。"),
            ],
            estimated_duration_s=180.0,
            composition_level=4,
            tags=["coarse", "motor", "nudge", "calibration",
                  "粗动", "挪一步", "步长标定"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        steps = int(params.get("steps") or 1)
        if steps > _NUDGE_MAX_STEPS:
            return SkillResult(
                skill_name="StepCoarseXY", success=False,
                error=("%d 步不是「挪一下」，那是换区 —— 用 RelocateCoarseXY。"
                       "这条路明说地跳过了「别重复访问」的效率约束，"
                       "拿它做大距离移动会把那条规则整个架空。" % steps))

        inner = {"axis": params.get("axis"),
                 "direction": params.get("direction"),
                 "steps": steps,
                 "allow_revisit": True,
                 "reapproach": bool(params.get("reapproach", True)),
                 "dry_run": bool(params.get("dry_run", False))}
        if params.get("prewithdraw_steps") is not None:
            inner["prewithdraw_steps"] = int(params["prewithdraw_steps"])

        res = context.run("RelocateCoarseXY", inner)
        data = dict(getattr(res, "data", None) or {})
        data["delegated_to"] = "RelocateCoarseXY"
        data["allow_revisit"] = True
        data["requested_steps"] = steps
        return SkillResult(
            skill_name="StepCoarseXY",
            success=bool(getattr(res, "success", False)),
            error=getattr(res, "error", None),
            data=data)
