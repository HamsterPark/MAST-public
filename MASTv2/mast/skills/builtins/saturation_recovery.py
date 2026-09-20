# -*- coding: utf-8 -*-
"""RecoverTipFromSaturation —— 撞针之后放大器锁在满量程时，把针尖退到脱离饱和。

## 它治的是什么

一次扫大图时针尖撞上东西，电流放大器锁死在满量程附近的一个固定读数。
症状很迷惑人：

```
撤针前          I 已锁在满量程附近   Z 已在压电上限附近
降偏压到 0.2 V   I 还是那个读数        ← 一点不变
WithdrawTip     I 还是那个读数        ← 还是一点不变
```

**「降偏压不变、撤针也不变」不是读数坏了**，有两层原因叠在一起：

1. 放大器已经饱和，读的是量程端点不是电流 —— 所以它当然不随偏压变；
2. **更要命的**：那些命令**根本没到仪器**。当时另一个调用还持有仪器锁
   （报错形如：「仪器正被占用：技能直调 API 正在执行……，已持有 Ns」）。
   原先以为自己在撤针，其实每一条都被挡在门外。

⇒ 所以本技能**第一步是确认仪器没有被别的链路占着**。在别人握着锁的时候
「撤针失败但读数没变」与「撤针成功但确实没变」长得一模一样，
而这两者要做的事完全相反。

## 顺序

压电退针买到的余量只有 ~1 µm，撞进去之后往往不够 —— 曾经 Z 已经在压电上限附近
仍然饱和。真正解开的是**粗动 Z 退针 200 步**（逐级 20/30/50/100/200）。
所以阶梯是：查锁 → 降偏压 → 压电退针 → 粗动 Z 逐级退，每级都读电流，脱离就停。

**逐级而不是一次退到底**：退得比需要的多，回来就要多花几分钟重新进针；
而每一级之后读一次电流，代价只有一次读数。
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

# 放大器满量程参考（A），用于饱和检查；必须与当前前置放大器档位一致。
# 贴近量程边界的读数不能当作可解析的结电流。
_SATURATION_A = 9.5e-9

#: 判「已脱离」的电流上限（A）。留得比隧穿工作点宽，
#: 因为退开途中经过场发射区时读数仍可能有几百 pA。
_RECOVERED_A = 1e-10

#: 粗动 Z 退针的阶梯（步）。逐级加大：退多了要多花几分钟才进得回来。
_LADDER = (20, 30, 50, 100, 200, 400)

#: 横移/退针期间的安全偏压（V）。与 relocate 用同一个理由：
#: 带着成像偏压时几十 nm 距离就场发射，读数没法用来判断。
_SAFE_BIAS_V = 0.2


def is_saturated(current_a):
    """读数是否贴在量程端点。None ⇒ 读不到，**不是**「没饱和」。"""
    if current_a is None:
        return None
    return abs(float(current_a)) >= _SATURATION_A


class RecoverTipFromSaturation(BaseSkill):
    """Back the tip off until the current amplifier leaves saturation."""

    _settle_s = 1.2

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RecoverTipFromSaturation",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "撞针之后电流放大器锁在满量程时，把针尖一级级退开直到读数脱离饱和。"
                "\n\n**第一步是查仪器锁**，不是撤针。撞针后连发"
                "降偏压、撤针，读数却一动不动 —— 不是命令无效，是它们"
                "**根本没到仪器**：另一个调用还持有仪器锁。"
                "「被挡在门外」与「发出去了但没效果」长得一模一样，而两者要做的事相反。"
                "\n\n**压电退针往往不够。** 曾经 Z 已经在压电上限附近仍然饱和，"
                "真正解开的是**粗动 Z 退 200 步**。所以顺序是：查锁 → 降偏压 → 压电退针 → "
                "粗动 Z 逐级退（20/30/50/100/200/400），每级读一次电流，脱离就停。"
                "\n\n**逐级而不是一次退到底**：退过头就要多花几分钟重新进针，"
                "而每级多读一次电流几乎不要钱。"
                "\n\n读不到电流时说「判不了」，**不当作已恢复** —— 那会让调用方"
                "以为可以继续，而针尖可能还压在样品上。"
            ),
            parameters=[
                ParameterSpec(
                    name="max_coarse_steps", type="int",
                    description=("粗动 Z 最多退多少步（累计）。到点仍未脱离就如实返回未恢复。"),
                    required=False, default=800, min_value=0, max_value=20000),
                ParameterSpec(
                    name="safe_bias_v", type="float", unit="V",
                    description=("退针期间的偏压。带着成像偏压时几十 nm 距离就场发射，"
                                 "读数没法用来判断。"),
                    required=False, default=_SAFE_BIAS_V,
                    min_value=0.0, max_value=2.0),
            ],
            estimated_duration_s=180.0,
            composition_level=1,
            tags=["tip", "recovery", "crash", "saturation", "撞针", "饱和"],
        )

    # ── 读数 ──────────────────────────────────────────────────────────
    @staticmethod
    def _current(context):
        res = context.run("GetCurrent", {})
        return (getattr(res, "data", None) or {}).get("current_a")

    def _settled_current(self, context, n=3):
        import statistics

        vals = []
        for _ in range(n):
            v = self._current(context)
            if v is not None:
                vals.append(float(v))
            time.sleep(0.2)
        return statistics.median(vals) if vals else None

    def execute(self, context, params: dict) -> SkillResult:
        budget = int(params.get("max_coarse_steps") or 800)
        safe_bias = float(params.get("safe_bias_v", _SAFE_BIAS_V))

        # ① 查锁。**不查就撤针,正是这类事故的形状。**
        busy = context.run("MotorMove", {"direction": "z-retract", "steps": 0})
        busy_err = str(getattr(busy, "error", "") or "")
        if "占用" in busy_err or "busy" in busy_err.lower():
            return SkillResult(
                skill_name="RecoverTipFromSaturation", success=False,
                error=("仪器正被另一条链路占用，**撤针命令到不了仪器**：" + busy_err[:200]
                       + " —— 先中止对方，否则「撤针了但读数没变」会被读成「撤针无效」，"
                         "而真相是它根本没发出去。"),
                data={"blocked_by_lock": True})

        before = self._settled_current(context)
        sat = is_saturated(before)
        steps_log = []
        if sat is None:
            return SkillResult(
                skill_name="RecoverTipFromSaturation", success=False,
                error="读不到电流 —— **判不了**是否饱和。不当作已恢复：针尖可能还压在样品上。",
                data={"current_a": None})
        if not sat:
            return SkillResult(
                skill_name="RecoverTipFromSaturation", success=True,
                data={"outcome": "not_saturated", "current_a": before,
                      "saturation_a": _SATURATION_A, "steps": [],
                      "message": ("电流 %.4g A 未贴量程端点 —— 没有饱和，**一步都没做**。"
                                  % before)})

        # ② 降偏压：带着成像偏压时读数会被场发射污染
        context.run("SetBias", {"bias_v": safe_bias})
        time.sleep(0.5)

        # ③ 压电退针（买 ~1 µm；撞进去之后常常不够）
        context.run("WithdrawTip", {})
        time.sleep(self._settle_s)
        after_piezo = self._settled_current(context)
        steps_log.append({"stage": "piezo_withdraw", "current_a": after_piezo,
                          "saturated": is_saturated(after_piezo)})
        if is_saturated(after_piezo) is False:
            return SkillResult(
                skill_name="RecoverTipFromSaturation", success=True,
                data={"outcome": "recovered", "by": "piezo_withdraw",
                      "current_before_a": before, "current_after_a": after_piezo,
                      "coarse_steps_used": 0, "steps": steps_log,
                      "message": "压电退针即脱离饱和，没有动粗动。"})

        # ④ 粗动 Z 逐级退
        used = 0
        for n in _LADDER:
            if used + n > budget:
                break
            res = context.run("MotorMove", {"direction": "z-retract", "steps": int(n)})
            ok = bool(getattr(res, "success", False))
            time.sleep(self._settle_s)
            now = self._settled_current(context)
            used += n if ok else 0
            steps_log.append({"stage": "coarse_z", "steps": n, "ok": ok,
                              "cumulative": used, "current_a": now,
                              "saturated": is_saturated(now),
                              "error": str(getattr(res, "error", "") or "")[:160]})
            if not ok:
                continue
            if now is not None and abs(now) <= _RECOVERED_A:
                return SkillResult(
                    skill_name="RecoverTipFromSaturation", success=True,
                    data={"outcome": "recovered", "by": "coarse_z",
                          "current_before_a": before, "current_after_a": now,
                          "coarse_steps_used": used, "steps": steps_log,
                          "message": ("粗动 Z 退 %d 步后电流 %.4g A，脱离饱和。"
                                      "**针尖已远离样品，要继续工作需重新进针。**"
                                      % (used, now))})

        last = steps_log[-1].get("current_a") if steps_log else before
        return SkillResult(
            skill_name="RecoverTipFromSaturation", success=False,
            error=("退了 %d 步仍未脱离饱和（最后读数 %s）。**没有假装恢复** —— "
                   "继续退之前先确认 z-retract 方向配置是对的：方向反了的话，"
                   "每一步都在往样品里扎。"
                   % (used, ("%.4g A" % last) if last is not None else "读不到")),
            data={"outcome": "not_recovered", "coarse_steps_used": used,
                  "steps": steps_log, "current_after_a": last})
