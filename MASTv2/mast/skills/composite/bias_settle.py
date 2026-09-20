"""BiasSettleChange —— 改偏压的安全通道(穿零保护 + 稳定等待)。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

## 为什么需要它:恒流反馈下穿零是一颗领域炸弹

Z 反馈是**恒流**的:它调节针尖高度,让隧道电流等于设定点。而隧道电流大致正比
于偏压 —— 偏压趋近 0,电流也趋近 0。反馈看到「电流不够」,唯一的反应是**把针尖
往表面推**,而且会一直推到量程尽头。于是:

    bias 从 +1 V 直接设到 −1 V
      → 中途经过 0 V
      → 电流塌到 0
      → 反馈全力推进
      → 针尖扎进样品

这不是罕见边角情况,是教科书级的常识,任何做过 STS 的人都知道。但它在代码里
**看不出来**:``SetBias(bias_v=-1.0)`` 是一次完全合法的调用,参数在范围内,
安全门不会拦(全局边界是 ±10 V),日志里也只会留下一行「设置偏压成功」。

所以这条通道存在的意义是:**让「改偏压」这个动作自带领域知识**,而不是指望
每次调用它的人(或模型)都记得。

## 做法

  * 符号变化 → 用 ``SetBiasRamp`` 快速穿过零点,并且**不在 |V| < 阈值 的死区
    里停留**(死区默认 ±50 mV);
  * 大幅变化 → 斜坡 + 稳定等待(偏压跳变会让反馈产生瞬态);
  * 小幅变化 → 直接设,但仍然等一个稳定时间。

死区宽度与穿零时的 slew 是**真机验证项**:``SetBiasRamp`` 在 0 附近的实际行为
(会不会停在某一步上)必须在真机上确认过才能收紧这里的参数。
"""

from __future__ import annotations

import logging
import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 穿零死区半宽(V)。|V| 小于它的时候隧道电流已经小到反馈无法维持,
#: 绝不能在这个区间里停留。
ZERO_DEADBAND_V = 0.05

#: 穿零时的斜坡速率(V/s)。要足够快地跨过死区,又不能快到让反馈完全跟丢。
CROSS_ZERO_SLEW_V_PER_S = 2.0

#: 超过这个变化幅度就走斜坡(而不是一步设过去)。
RAMP_THRESHOLD_V = 0.5

#: 常规斜坡速率(V/s)。
DEFAULT_SLEW_V_PER_S = 1.0

#: 偏压改变后的默认稳定等待(s)。
DEFAULT_SETTLE_S = 2.0
#: 小幅变化的稳定等待(s)。
SMALL_CHANGE_SETTLE_S = 0.5


class BiasSettleChange(BaseSkill):
    """Change the sample bias safely (zero-crossing protected) and let it settle."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="BiasSettleChange",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "走安全路径改样品偏压：跨零时**不在低偏压死区里逗留**（在那里"
                "恒流反馈会把针尖往表面上压），大幅改动用 ramp 而不是直接跳，"
                "改完还会等反馈稳定。"
                "**偏压要变号、或者变化很大时，优先用它而不是 SetBias** —— "
                "直接用 SetBias 跨零是撞针的经典路径，而参数范围和安全门"
                "**都不会拦你**。"
            ),
            parameters=[
                ParameterSpec(
                    name="bias_v",
                    type="float",
                    description=(
                        "目标样品偏压，是一个 VOLT 量。偏压的量级不大，所以"
                        "写普通字符串就行：'-2'（−2 V）、'0.05' 或 '50m'（50 mV）。"
                    ),
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="settle_s",
                    type="float",
                    description=(
                        "改完之后等一会儿，让反馈的暂态衰减掉再进行下一次采集。"
                        "留空用默认值。"
                    ),
                    unit="s",
                    required=False,
                    min_value=0.0,
                    max_value=60.0,
                ),
                ParameterSpec(
                    name="allow_stop_in_deadband",
                    type="bool",
                    description=(
                        "允许**目标**偏压落在低偏压死区里（|V| < 50 mV）。"
                        "默认关：恒流反馈开着时停在那里会把针尖往表面上驱。"
                        "**只有在反馈已关时才开它。**"
                    ),
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=6.0,
            composition_level=2,
            tags=["bias", "safety", "settle", "composite"],
        )

    @staticmethod
    def _read_bias(context, calls) -> "float | None":
        rec = context.safe_call("Bias_Get")
        calls.append(rec)
        if rec.error:
            return None
        parsed = getattr(rec, "return_value", None)
        if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
            return None
        vals = parsed[2]
        if not isinstance(vals, (list, tuple)) or not vals:
            return None
        try:
            return float(vals[0])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _feedback_on(context, calls) -> "bool | None":
        rec = context.safe_call("ZCtrl_OnOffGet")
        calls.append(rec)
        if rec.error:
            return None
        parsed = getattr(rec, "return_value", None)
        if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
            return None
        vals = parsed[2]
        if not isinstance(vals, (list, tuple)) or not vals:
            return None
        try:
            return bool(int(vals[0]))
        except (TypeError, ValueError):
            return None

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        target = float(params["bias_v"])
        allow_deadband = bool(params.get("allow_stop_in_deadband", False))

        start = self._read_bias(context, calls)
        if start is None:
            return SkillResult(
                skill_name="BiasSettleChange", success=False,
                error=("读不到当前偏压(Bias_Get)—— 不知道起点就无法判断这次"
                       "改变会不会穿过零点,拒绝执行。"),
                nanonis_calls=calls)

        feedback_on = self._feedback_on(context, calls)

        # 目标落在死区里:反馈开着时这是个不能停的地方。
        in_deadband = abs(target) < ZERO_DEADBAND_V
        if in_deadband and not allow_deadband and feedback_on is not False:
            return SkillResult(
                skill_name="BiasSettleChange", success=False,
                error=(
                    f"目标偏压 {target:.4g} V 落在低偏压死区(|V| < "
                    f"{ZERO_DEADBAND_V} V)内,而 Z 反馈"
                    f"{'开着' if feedback_on else '状态未知'}。恒流反馈在这里"
                    "维持不住电流,会把针尖一直推向表面。先关反馈"
                    "(ZControllerOnOff),或改用一个更大的偏压。"),
                data={"bias_v_start": start, "bias_v_target": target,
                      "feedback_on": feedback_on,
                      "deadband_v": ZERO_DEADBAND_V},
                nanonis_calls=calls)

        crosses_zero = (start > 0) != (target > 0) and start != 0 and target != 0
        delta = abs(target - start)
        strategy = "direct"
        slew = DEFAULT_SLEW_V_PER_S

        if crosses_zero:
            # 穿零:用最快的合理斜坡跨过死区。停在死区里的每一毫秒,反馈都在
            # 把针尖往下推。
            strategy = "ramp_through_zero"
            slew = CROSS_ZERO_SLEW_V_PER_S
        elif delta > RAMP_THRESHOLD_V:
            strategy = "ramp"
            slew = DEFAULT_SLEW_V_PER_S

        if strategy == "direct":
            res = context.run("SetBias", {"bias_v": target})
        else:
            res = context.run("SetBiasRamp", {
                "bias_v_end": target,
                "bias_v_start": start,
                "slew_rate_v_per_s": slew,
            })

        if not getattr(res, "success", False):
            return SkillResult(
                skill_name="BiasSettleChange", success=False,
                error=f"偏压变更失败({strategy}): {getattr(res, 'error', '')}",
                data={"bias_v_start": start, "bias_v_target": target,
                      "strategy": strategy},
                nanonis_calls=calls)

        settle = params.get("settle_s")
        if settle is None:
            settle = (SMALL_CHANGE_SETTLE_S
                      if (delta <= RAMP_THRESHOLD_V and not crosses_zero)
                      else DEFAULT_SETTLE_S)
        settle = float(settle)
        if settle > 0:
            time.sleep(settle)

        return SkillResult(
            skill_name="BiasSettleChange", success=True,
            data={
                "bias_v_start": start,
                "bias_v": target,
                "delta_v": target - start,
                "crossed_zero": crosses_zero,
                "strategy": strategy,
                "slew_rate_v_per_s": slew if strategy != "direct" else None,
                "settle_s": settle,
                "feedback_on": feedback_on,
            },
            summary=(f"偏压 {start:.4g} V → {target:.4g} V"
                     f"({'穿零斜坡' if crosses_zero else strategy}),"
                     f"稳定 {settle:.3g} s"),
            nanonis_calls=calls)
