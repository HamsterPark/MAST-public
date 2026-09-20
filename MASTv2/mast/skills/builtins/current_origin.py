# -*- coding: utf-8 -*-
"""ClassifyUnexplainedCurrent：以偏压依赖和零偏压读数分析可疑电流。

首先检查饱和；满量程读数也可能随偏压不变，不能把它误判为串扰。
有效非零偏压点用于拟合 n = dln|I| / dln|V|，零偏压点用于估计偏置与噪声底。
联合信息区分结电流、强非线性响应、偏压无关成分与不可判定结果。
阈值是工作流分类参数，需要结合具体测量条件解释。

observed_current_a 可提供待复现的读数。静态扫描不能复现时返回 transient，
提示应在原触发过程中观测；静态未出现不等于该电流不存在。
"""

from __future__ import annotations

import logging
import math
import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.builtins.saturation_recovery import is_saturated

logger = logging.getLogger(__name__)

#: 判「0 V 下没有电流」的上限（A）。比典型隧穿工作点（20–120 pA）低一个量级以上。
_ZERO_FLOOR_A = 2e-12

# 噪声底由零偏压读数估计，并设置绝对下限。
# 固定的较高阈值可能删掉仍有意义的小电流点，使幂指数拟合不足两点。
# 不能用无法适配当前噪声的常量代替已获得的零偏压信息。
_ABS_FLOOR_A = 2e-14

#: 自标定倍数：高于 0 V 读数这么多倍才算「读到了」。
_FLOOR_MULT = 3.0


def noise_floor_from_zero(i_zero_a):
    """噪声底由 0 V 那一点自标定。读不到 0 V 时退回绝对下限。

    0 V 读数就是「确定没有结电流时这台机器读到什么」的直接测量。
    """
    if i_zero_a is None:
        return _ABS_FLOOR_A
    return max(abs(float(i_zero_a)) * _FLOOR_MULT, _ABS_FLOOR_A)

# 用偏压依赖的幂指数辅助区分近线性响应与强非线性响应。
# 指数阈值是工作流分类参数，需结合零偏压、饱和状态和测量条件解释。
_N_FLAT = 0.5
_N_FIELD_EMISSION = 3.0

#: 「与偏压无关」的直接判据：最大偏压下的电流不到 0 V 读数的这么多倍。
#:
#: ⚠ 这一条**必须用原始读数的比值**，不能靠对数拟合。第一版只有拟合，
#: 而拟合用的噪声底是从 0 V 自标定来的（3×|I(0V)|）—— 于是「与偏压无关」
#: 这个 case（它按定义就与 I(0V) 同量级）**永远被自己的底线判成
#: 「没有可测的电流」**：判别表最该抓的那一格被自标定吃掉了。
#: 原始读数的比值保留与偏压无关的响应，避免拟合阈值遮蔽这一判别。
_FLAT_RATIO = 2.0

#: 静态复现判据：静态读数与被怀疑读数差到几倍以上算「没复现」。
_REPRODUCE_RATIO = 3.0


def effective_exponent(points, floor_a):
    """由 [(V, I)] 拟 n = dln|I|/dln|V|。点不够或值非法时返回 (None, used)。

    只用 |V|>0 且 |I| 高于噪声底的点 —— 噪声底上的读数是「读不到」，
    喂进去会把指数拉平（这与 barrier_height 那次是同一个坑）。
    """
    usable = [(abs(float(v)), abs(float(i)))
              for v, i in points
              if v is not None and i is not None
              and abs(float(v)) > 1e-9 and abs(float(i)) > floor_a]
    if len(usable) < 2:
        return None, usable
    xs = [math.log(v) for v, _ in usable]
    ys = [math.log(i) for _, i in usable]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None, usable
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    if not math.isfinite(slope):
        return None, usable
    return float(slope), usable


def classify(i_zero_a, exponent, i_max_a):
    """判别表。返回 (verdict, 一句话)。任何一项读不到都返回 undetermined。"""
    if i_max_a is None:
        return "undetermined", "最大偏压下读不到电流 —— 判不了。**「读不到」不是「没有」。**"
    # ⚠ 这里用**绝对**下限，而且本函数**故意不收**自标定的噪声底。
    #   第一版收了，于是「与偏压无关」这一格（按定义与 I(0V) 同量级）
    #   被 3×|I(0V)| 的底恒判成「没有可测的电流」—— 判别表最该抓的格子被
    #   自己的底线吃掉了。自标定的底只回答「这一点的偏压依赖有没有意义」
    #   （给 effective_exponent 用），回答不了「有没有电流」。
    #   参数删掉而不是留着不用：留着的话下一个人会以为它在起作用。
    if abs(i_max_a) < _ABS_FLOOR_A:
        return ("no_measurable_current",
                "所有偏压下都在噪声底以下 —— 没有可测的电流。"
                "若针尖本该在隧穿，这说明它已经退开或反馈没 engage。")
    if i_zero_a is None:
        return "undetermined", "0 V 下读不到 —— **0 V 那一点正是判别点**，缺了它判不了。"
    if abs(i_zero_a) >= _ZERO_FLOOR_A:
        # 直接用原始读数的比值，不依赖拟合能不能算出来
        ratio = abs(i_max_a) / max(abs(i_zero_a), 1e-18)
        if ratio <= _FLAT_RATIO:
            return ("not_a_junction_current",
                    "0 V 下仍有 %.3g A，而最大偏压下也只有 %.3g A（%.1f 倍）—— "
                    "**这不是结电流**：它几乎不随偏压变，而隧穿与场发射在 0 V 下"
                    "都必须是零。偏置漂移／某处串扰／前置放大器断了，三者都长这样；"
                    "分开它们要另一个观察：把可疑的动作停下来看它消不消失。"
                    % (i_zero_a, i_max_a, ratio))
        if exponent is None or exponent < _N_FLAT:
            return ("not_a_junction_current",
                    "0 V 下仍有 %.3g A —— **这不是（纯粹的）结电流**，"
                    "而偏压依赖又拟不出来（本底把点都盖住了）。"
                    "先把 0 V 本底的来源找掉再判结。" % i_zero_a)
        return ("mixed",
                "0 V 下有 %.3g A 的本底，但电流确实随偏压变（%.0f 倍，n=%.1f）—— "
                "一个真的结电流叠在一个偏置上。先把本底的来源找掉再判结。"
                % (i_zero_a, ratio, exponent))
    if exponent is None:
        return "undetermined", "0 V 干净，但非零偏压的点不够（或都在噪声底上），拟不出指数。"
    if exponent >= _N_FIELD_EMISSION:
        return ("field_emission",
                "0 V 干净而电流随偏压极度超线性（n=%.1f）—— **场发射**，不是隧穿。"
                "针尖离表面太远而偏压太高。降偏压或进针；"
                "带着成像偏压做横移/粗动时最常撞见它。" % exponent)
    if exponent < _N_FLAT:
        return ("not_a_junction_current",
                "电流几乎不随偏压变（n=%.1f）而 0 V 下又是干净的 —— 自相矛盾，"
                "多半是读数被别的东西钳住了。先查量程与增益。" % exponent)
    return ("junction_current",
            "0 V 干净、电流随偏压近似线性（n=%.1f）—— 这是真的结电流（隧穿/接触）。"
            % exponent)


class ClassifyUnexplainedCurrent(BaseSkill):
    """Classify an unexplained current by its bias dependence."""

    _settle_s = 0.4

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ClassifyUnexplainedCurrent",
            version="1.0.0",
            category=SkillCategory.WRITE,   # 要动偏压与 Z 控制器，虽然都还原
            safety_level=SafetyLevel.AUTO,
            description=(
                "用偏压依赖区分可疑电流的来源：隧穿、场发射、偏置或串扰，以及放大器饱和。\n\n先检查饱和；满量程时不同偏压的读数都可能不变，不能因此判成串扰。零偏压读数用于检查偏压无关成分，解释时应结合偏置与噪声。\n\n测量期间关闭反馈，避免 Z 跟踪 setpoint 使 I(V) 变成反馈环响应；结束后还原偏压与反馈。observed_current_a 可提供引起怀疑的读数。静态测量无法复现时返回 transient，需要在触发该电流的过程中重新观察。"
            ),
            parameters=[
                ParameterSpec(
                    name="test_biases_v", type="str",
                    description=("逗号分隔的测试偏压。**0 会被自动补上** —— "
                                 "缺了它整个判别就立不住。"),
                    required=False, default="2.0,1.0,0.5,0.0"),
                ParameterSpec(
                    name="observed_current_a", type="float", unit="A",
                    description=("引起怀疑的那个电流读数。给了就与同偏压下的静态读数比："
                                 "复现不出来 ⇒ 判 transient（只在动的时候存在）。"),
                    required=False, default=None),
                ParameterSpec(
                    name="repeats", type="int",
                    description="每个偏压读几次取中位数。",
                    required=False, default=3, min_value=1, max_value=15),
            ],
            estimated_duration_s=25.0,
            composition_level=1,
            tags=["current", "diagnosis", "field-emission", "crosstalk",
                  "场发射", "串扰", "电流来源"],
        )

    # ── 读数 ──────────────────────────────────────────────────────────
    @staticmethod
    def _read_current(context):
        res = context.run("GetCurrent", {})
        return (getattr(res, "data", None) or {}).get("current_a")

    def _median_current(self, context, n):
        import statistics

        vals = []
        for _ in range(max(1, n)):
            v = self._read_current(context)
            if v is not None:
                vals.append(float(v))
            time.sleep(0.05)
        return statistics.median(vals) if vals else None

    # ── 主流程 ────────────────────────────────────────────────────────
    def execute(self, context, params: dict) -> SkillResult:
        raw = str(params.get("test_biases_v") or "2.0,1.0,0.5,0.0")
        try:
            biases = [float(t) for t in raw.split(",") if t.strip()]
        except ValueError:
            return SkillResult(
                skill_name="ClassifyUnexplainedCurrent", success=False,
                error="test_biases_v 解析不了：%r —— 要逗号分隔的数。" % raw)
        if not biases:
            return SkillResult(
                skill_name="ClassifyUnexplainedCurrent", success=False,
                error="test_biases_v 是空的。")
        zero_added = not any(abs(b) < 1e-9 for b in biases)
        if zero_added:
            biases.append(0.0)
        # 从大到小：结束时停在最小偏压上比停在最大偏压上安全
        biases.sort(key=lambda b: -abs(b))

        repeats = int(params.get("repeats") or 3)
        observed = params.get("observed_current_a")
        observed = float(observed) if observed is not None else None

        # ① 饱和先查。饱和会伪装成「随偏压不变」，判别表会判反。
        first = self._median_current(context, repeats)
        if is_saturated(first):
            return SkillResult(
                skill_name="ClassifyUnexplainedCurrent", success=True,
                data={"verdict": "amplifier_saturated", "current_a": first,
                      "message": ("电流 %.4g A 贴在量程端点 —— 放大器**饱和**，"
                                  "读的是端点不是电流。偏压依赖判别对饱和读数无效"
                                  "（每个偏压都会读到同一个数，看起来像「与偏压无关」）。"
                                  "先跑 RecoverTipFromSaturation 把针尖退开。"
                                  % first),
                      "next_skill": "RecoverTipFromSaturation"})

        # ② 记住工作点，**结束必须还原**
        gb = context.run("GetBias", {})
        bias0 = (getattr(gb, "data", None) or {}).get("bias_v")

        # ③ 先读**初态**，结束时还原成它 —— 不是还原成 True。
        #
        # ⚠ 第一版在 finally 里无条件 `enable: True`。如果用户本来把反馈关着
        #   （手动操作中／已退针），这样会替他打开 —— 而开反馈会驱动 Z 去够
        #   setpoint。这和「诊断把偏压留在 0 V」是同一类错，方向相反：
        #   **还原的对象是「进来时的样子」，不是「一般情况下该是的样子」。**
        #   读不到初态就拒答：猜错的代价是一根针尖。
        zstate = context.run("GetZControllerState", {})
        fb_was_on = (getattr(zstate, "data", None) or {}).get("controller_on")
        if fb_was_on is None:
            return SkillResult(
                skill_name="ClassifyUnexplainedCurrent", success=False,
                error=("读不到 Z 反馈的当前状态（ZCtrl_OnOffGet）—— **判不了**它进来时"
                       "是开还是关，也就无从还原。不做：万一用户本来关着反馈，"
                       "我结束时替他开回来会把 Z 驱向 setpoint。"),
                data={"controller_on": None})

        # ④ 关反馈：开着的话 I(V) 描述的是反馈环不是结
        zres = context.run("ZControllerOnOff", {"enable": False})
        zdata = getattr(zres, "data", None) or {}
        fb_off = bool(zdata.get("verified")) and zdata.get("z_controller_on") is False
        # 关不掉就不做 —— 反馈开着量出来的 I(V) 会把人引到错的结论上
        if not fb_off:
            return SkillResult(
                skill_name="ClassifyUnexplainedCurrent", success=False,
                error=("关不掉 Z 反馈（读回 %r）—— **不在反馈开着的时候量 I(V)**："
                       "Z 会去追 setpoint，量到的是反馈环的响应不是结的性质，"
                       "而它长得像一条很正常的曲线。"
                       % zdata.get("z_controller_on")),
                data={"z_controller_on": zdata.get("z_controller_on")})

        points = []
        restored = False
        try:
            for b in biases:
                context.run("SetBias", {"bias_v": b})
                time.sleep(self._settle_s)
                i = self._median_current(context, repeats)
                points.append({"bias_v": b, "current_a": i,
                               "saturated": is_saturated(i)})
        finally:
            # ⑤ 还原**进来时的样子** —— 无论成败。
            #    诊断把偏压留在 0 V 上比不做诊断更糟（skill_sets_its_own_working_point）；
            #    而把反馈还原成写死的 True 比留在 0 V 更糟 —— 见上面 ③ 的注释。
            if bias0 is not None:
                context.run("SetBias", {"bias_v": float(bias0)})
            context.run("ZControllerOnOff", {"enable": bool(fb_was_on)})
            restored = True

        pairs = [(p["bias_v"], p["current_a"]) for p in points]
        i_zero = next((p["current_a"] for p in points if abs(p["bias_v"]) < 1e-9), None)
        nonzero = [p for p in points if abs(p["bias_v"]) > 1e-9
                   and p["current_a"] is not None]
        i_max = max((abs(p["current_a"]) for p in nonzero), default=None)
        floor = noise_floor_from_zero(i_zero)
        exponent, used = effective_exponent(pairs, floor)

        verdict, message = classify(i_zero, exponent, i_max)

        data = {"verdict": verdict, "message": message,
                "points": points, "exponent_n": exponent,
                "i_zero_a": i_zero, "i_max_a": i_max, "noise_floor_a": floor,
                "fit_points": len(used), "zero_bias_added": zero_added,
                "bias_restored_to": bias0, "feedback_restored": restored,
                "feedback_was_on": fb_was_on}

        # ⑤ 静态复现：串扰假设说的是「动的时候」，静态量不到它
        if observed is not None:
            same_bias = None
            if bias0 is not None:
                candidates = [p for p in points if p["current_a"] is not None]
                if candidates:
                    same_bias = min(candidates,
                                    key=lambda p: abs(p["bias_v"] - float(bias0)))
            static = abs(same_bias["current_a"]) if same_bias else None
            data["observed_current_a"] = observed
            data["static_at_same_bias_a"] = static
            if static is None or abs(observed) <= 0:
                data["reproduced"] = None
            else:
                hi = max(abs(observed), static)
                lo = max(min(abs(observed), static), 1e-18)
                ratio = hi / lo
                data["reproduce_ratio"] = ratio
                data["reproduced"] = bool(ratio <= _REPRODUCE_RATIO)
                if not data["reproduced"]:
                    data["verdict"] = "transient"
                    data["static_verdict"] = verdict
                    data["message"] = (
                        "静态复现不出来：被怀疑的读数 %.3g A，同偏压静止时只有 %.3g A"
                        "（差 %.0f 倍）—— 这股电流**只在某件事发生时存在**"
                        "（压电运动、偏压斜坡、扫描）。静态 I(V) 判不了它，"
                        "**要在那个动作进行中重测**。这不是「没有」，是「没在这儿」。"
                        % (observed, static, ratio))

        return SkillResult(
            skill_name="ClassifyUnexplainedCurrent", success=True, data=data)
