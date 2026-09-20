# -*- coding: utf-8 -*-
"""CleanTipUntilBarrier：每一步动作后用势垒重新验证结果。

先测基线；已经达标时返回 already_clean，不执行修针动作。
执行配置的动作阶梯并逐步复测，连续无改善时停止，变差时立即停止。
最终报告过程中最好的状态，而非默认把最后状态当成最好状态。

动作顺序与参数属于工作流设置，需要按实际条件验证，不能作为通用物理标定。
脉冲后按既定流程执行稳定步骤，再进行下一次测量。

本技能不更换针尖、不执行粗动、不寻找平区。势垒不可读时报告无法判定，
不把缺少测量混同于无改善；前者应先排查测量条件。
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

#: 默认目标势垒（eV）。3.0 是 `MeasureBarrierHeight` 判 clean 的线。
_DEFAULT_TARGET_EV = 3.0

# 动作阶梯规定顺序与参数；它是工作流策略，不代表跨样品验证过的有效性排序。
# 每步均须以新的测量结果决定是否继续。
_LADDER = (
    ("poke", 0.5),
    ("poke", 0.5),
    ("poke", 0.8),
    ("pulse", 3.0),
    ("pulse", -3.0),
    ("pulse", 4.0),
    ("pulse", -4.0),
)

#: 连续多少步没改善就停。
_MAX_STALE = 3

# 改善比率需要与测量重复性区分。此默认值是工作流判据，
# 应结合当前测量不确定度验证，不能把噪声内波动当成确定改善。
_IMPROVE_FRAC = 0.15


class CleanTipUntilBarrier(BaseSkill):
    """Condition the tip, verifying with the tunnelling barrier at every step."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CleanTipUntilBarrier",
            version="1.0.0",
            category=SkillCategory.WRITE,
            capabilities=frozenset({"tip_shaping", "bias_pulse"}),
            # 会打脉冲、会扎针 —— 与 PokeConditionTip / PrepareNobleTip 同级。
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在贵金属基底上按配置动作阶梯修针，每一步用 I–Z 势垒验证结果。\n\n先测基线；已达标时返回 already_clean，不执行动作。先执行扎针阶段，再执行脉冲阶段；每发脉冲后按流程补充稳定步骤。动作参数应按实际条件验证。\n\n连续 %d 步无改善就停，变差立即停止，最终报告过程中最好的状态。势垒不可读时报告无法判定，不能把缺少测量当成无改善。"
                % _MAX_STALE
            ),
            parameters=[
                ParameterSpec(
                    name="target_phi_ev", type="float", unit="eV",
                    description=("目标势垒。默认 %.1f eV（= MeasureBarrierHeight 判 clean 的线）。"
                                 "真空隧穿约 4–5 eV。" % _DEFAULT_TARGET_EV),
                    required=False, default=_DEFAULT_TARGET_EV,
                    min_value=0.5, max_value=6.0),
                ParameterSpec(
                    name="max_steps", type="int",
                    description="最多做几步修针动作。到点仍未达标就如实返回未达标。",
                    required=False, default=7, min_value=1, max_value=20),
                ParameterSpec(
                    name="bias_v", type="float", unit="V",
                    description="测势垒用的偏压。留空 = 用当前偏压。",
                    required=False, min_value=-10.0, max_value=10.0),
            ],
            estimated_duration_s=2400.0,
            rollback_skill="WithdrawTip",
            composition_level=2,
            tags=["tip", "conditioning", "barrier", "修针", "有判据"],
        )

    # ── 动作 ──────────────────────────────────────────────────────────
    @staticmethod
    def _poke(context, depth_nm):
        return context.run("TipShape", {
            "tip_lift_m": -abs(depth_nm) * 1e-9,
            "lift_height_m": abs(depth_nm) * 1e-9,
            "lift_time_1_s": 0.1, "lift_time_2_s": 0.1,
            "bias_lift_v": 0.02, "change_bias": False,
            "bias_settling_s": 0.5, "end_wait_s": 0.2,
            "restore_feedback": True})

    def _pulse(self, context, volts):
        res = context.run("BiasPulse", {"bias_v": float(volts), "width_s": 0.05,
                                        "z_hold": True, "absolute": True})
        # 脉冲后的稳定步骤是既定工作流的一部分。
        # 是否继续由后续测量决定，不预设动作本身一定改善针尖。
        self._poke(context, 0.5)
        return res

    def _measure(self, context, bias_v):
        p = {}
        if bias_v is not None:
            p["bias_v"] = float(bias_v)
        res = context.run("MeasureBarrierHeight", p)
        d = getattr(res, "data", None) or {}
        return d.get("phi_ev"), d.get("verdict")

    # ── 主流程 ────────────────────────────────────────────────────────
    def execute(self, context, params: dict) -> SkillResult:
        target = float(params.get("target_phi_ev") or _DEFAULT_TARGET_EV)
        max_steps = int(params.get("max_steps") or 7)
        bias_v = params.get("bias_v")

        phi0, verdict0 = self._measure(context, bias_v)
        base = {"phi_ev": phi0, "verdict": verdict0}
        if phi0 is None:
            return SkillResult(
                skill_name="CleanTipUntilBarrier", success=False,
                error=("基线势垒**判不了** —— 在没有判据的情况下动针尖，正是这个技能"
                       "存在的理由所要避免的。先把 I–Z 量出来再说。"),
                data={"baseline": base, "steps": []})

        # ★ 已达标就一步都不做。这是本技能最重要的一条。
        if phi0 >= target:
            return SkillResult(
                skill_name="CleanTipUntilBarrier", success=True,
                data={"outcome": "already_clean", "baseline": base,
                      "phi_ev": phi0, "target_phi_ev": target, "steps": [],
                      "message": ("基线 φ=%.2f eV 已达标（≥%.2f）—— **一步都没做**。"
                                  "针尖不需要修；在没有判据支持时动针尖，"
                                  "只会把好针尖弄坏。" % (phi0, target))})

        best_phi, best_at = phi0, "baseline"
        steps, stale = [], 0
        for i in range(min(max_steps, len(_LADDER))):
            kind, arg = _LADDER[i]
            act = self._pulse(context, arg) if kind == "pulse" else self._poke(context, arg)
            ok = bool(getattr(act, "success", False))
            # 动作可能改变工作点，每步之后回读并记录实际偏压。
            br = context.run("GetBias", {})
            bias_after = (getattr(br, "data", None) or {}).get("bias_v")
            phi, verdict = self._measure(context, bias_v)
            row = {"step": i + 1, "action": kind, "arg": arg, "action_ok": ok,
                   "bias_after_v": bias_after, "phi_ev": phi, "verdict": verdict}
            steps.append(row)
            if not ok:
                row["stopped"] = "动作失败"
                break
            if phi is None:
                # 「没测到」≠「没变好」：不计入 stale，也不当成变差
                row["note"] = "势垒判不了 —— 不计入改善判断"
                continue
            if phi > best_phi * (1.0 + _IMPROVE_FRAC):
                best_phi, best_at, stale = phi, "step%d" % (i + 1), 0
                row["improved"] = True
            else:
                stale += 1
            if phi >= target:
                row["stopped"] = "达标"
                break
            if phi < best_phi * (1.0 - _IMPROVE_FRAC) and best_at != "baseline":
                row["stopped"] = "变差 —— 立刻停"
                break
            if stale >= _MAX_STALE:
                row["stopped"] = "连续 %d 步没改善" % _MAX_STALE
                break

        reached = best_phi >= target
        return SkillResult(
            skill_name="CleanTipUntilBarrier", success=True,
            data={
                "outcome": "reached" if reached else "not_reached",
                "baseline": base, "target_phi_ev": target,
                "best_phi_ev": best_phi, "best_at": best_at,
                "final_phi_ev": steps[-1].get("phi_ev") if steps else phi0,
                "n_steps": len(steps), "steps": steps,
                "message": (
                    "φ %.2f → %.2f eV（最好出现在 %s），%s目标 %.2f。"
                    "**交出的是过程中最好的那个状态，不是最后那个。**"
                    % (phi0, best_phi, best_at,
                       "已达到" if reached else "未达到", target)),
            })
