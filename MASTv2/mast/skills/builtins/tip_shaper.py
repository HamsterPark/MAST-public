"""Tip shaper skill for hardware-controlled tip conditioning.

vendored from v1 mast/skills/builtins/tip_shaper.py 2026-04-23. Zero behavioural changes.
1 skill: TipShape.
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
from mast.skills.builtins._tip_policy import (
    ALLOW_ON_QPLUS,
    apply_tip_policy,
    shaper_bias_default,
    policy_fields_for_result,
    qplus_gate,
    resolved_lift_height_m,
)
from mast.skills.builtins._tip_xy import tip_xy_fields


class TipShape(BaseSkill):
    """Run the hardware tip shaper procedure."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TipShape",
            version="1.0.0",
            category=SkillCategory.WRITE,
            capabilities=frozenset({"tip_shaping"}),
            # AUTO (2026-06-11 safety re-scoping): the Z ramp here is FINE Z
            # (z-controller piezo, bounded by the Nanonis Z-piezo range), NOT the
            # open-loop coarse stepper — so it cannot crash the instrument the way
            # a coarse sample approach can. Tip conditioning runs autonomously.
            safety_level=SafetyLevel.AUTO,
            description="运行硬件的 tip shaper 流程（受控修针）。",
            parameters=[
                ParameterSpec(
                    name="switch_off_delay_s",
                    type="float",
                    description="关闭 controller 之前，对 Z 位置做平均的时长",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="change_bias",
                    type="bool",
                    description=(
                        "在第一段 Z ramp 之前，把偏压跳变到 bias_v。"
                        "**默认关闭**（2026-08-11）。workflow 层本来就"
                        "拒绝这条路：TipShaper 只能**一步**改完偏压，"
                        "而一步跳变本身就是一记冲击"
                        "（_tip_phases.py）。请先自己把偏压设好（用带 slew rate "
                        "的 SetBias），再在这一项关闭的情况下扎针。"
                    ),
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="bias_v",
                    type="float",
                    description=(
                        "第一段 Z ramp 之前施加的偏压（仅当 "
                        "change_bias=True 时生效）。**省略**它则跟随**当前**的"
                        "成像偏压。没有 3 V 兜底：偏压若读不到，这个技能"
                        "直接拒绝执行。"
                    ),
                    unit="V",
                    required=False,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="tip_lift_m",
                    type="float",
                    description=(
                        "第一段 Z ramp 的距离（相对当前 Z 的相对量），单位"
                        "**米**。常用 -2n 到 2n（±2 nm）。硬上限 "
                        "±100n（±100 nm）—— 由 ±1 µm 收紧而来，那是常规扎针的 500×，"
                        "而且没有任何全局兜底（2026-07-03 "
                        "复审）。示例：-3n（= -3 nm）✓；5（= 5 米！）✗。"
                    ),
                    unit="m",
                    required=False,
                    default=0.0,
                    min_value=-1e-7,
                    max_value=1e-7,
                ),
                ParameterSpec(
                    name="lift_time_1_s",
                    type="float",
                    description="第一段 Z ramp 的时长",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="bias_lift_v",
                    type="float",
                    description=(
                        "在第一段 Z ramp**刚结束之后**施加的偏压。⚠️ 与 "
                        "bias_v 不同，这一项是**无条件**施加的 —— change_bias "
                        "**不能**把它解除（厂商原文：'Bias (V) … if Change Bias "
                        "is True' 对比 'Bias Lift (V) … applied just after the "
                        "first Z ramping'）。**省略**它则跟随 bias_v，也就是"
                        "当前的成像偏压。没有 3 V 兜底。"
                    ),
                    unit="V",
                    required=False,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="bias_settling_s",
                    type="float",
                    description="施加偏压值之后的等待时长",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="lift_height_m",
                    type="float",
                    description=(
                        "第二段 Z ramp 的高度，单位**米** —— 也就是**退回来**的那一段。"
                        "**省略**它则默认取 -tip_lift_m（扎进去 x，就拉"
                        "回来 x），这也是每个 composite 早已在用的"
                        "规则。常用 1n 到 5n（1–5 nm）。硬上限 ±100n"
                        "（±100 nm）—— 由 ±1 µm 收紧而来（2026-07-03 复审）。"
                        "示例：2n（= 2 nm）✓；1（= 1 米！）✗。"
                    ),
                    unit="m",
                    required=False,
                    min_value=-1e-7,
                    max_value=1e-7,
                ),
                ParameterSpec(
                    name="lift_time_2_s",
                    type="float",
                    description="第二段 Z ramp 的时长",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="end_wait_s",
                    type="float",
                    description="恢复初始偏压之后的等待时长",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="restore_feedback",
                    type="bool",
                    description="结束时恢复 Z-controller 的初始状态",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name=ALLOW_ON_QPLUS,
                    type="bool",
                    description=(
                        "在已登记的 qPlus 针尖上显式放行这个动作。Z "
                        "ramp 可能**不可逆地**毁掉石英音叉（传感器必须"
                        "物理更换并重新标定），所以除非设了这一项，"
                        "否则 qPlus 针尖一律**拒绝**。若没有登记针尖，"
                        "则本项忽略。"
                    ),
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="timeout_ms",
                    type="int",
                    description="整个流程的超时（-1 = 永远等待）",
                    unit="ms",
                    required=False,
                    default=-1,
                    min_value=-1,
                ),
            ],
            estimated_duration_s=10.0,
            composition_level=1,
            tags=["tip", "shaper", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []
        # 针尖门 + 方案表。两件事:
        #   * qPlus 传感器上做 Z 下压有毁掉音叉的风险(不可逆,要拆机重装并重新
        #     标定 f₀/Q),默认拒绝,要做得显式签字;
        #   * bias / lift 这些数字不该由模型现编,也不该是一个对所有针尖都一样的
        #     常量 —— 没给就按当前针尖的材料/制备/形态查方案表。
        gate = qplus_gate("TipShape", params, calls)
        if gate is not None:
            return gate
        # bias 缺省 = 此刻的成像偏压,不是写死的 3 V(2026-08-10 要求：
        # 「shaper 应该自带 bias 变成扫图 bias,而不是锁死 3V」)。
        # 见 _tip_policy.shaper_bias_default。注意这一步在 apply_tip_policy **之前**:
        # 方案表若给了 shaper_bias_v,那是一个有主的值,照旧胜出。
        _bias_before_policy = params.get("bias_v")
        params, plan = apply_tip_policy(
            params, ("shaper_bias_v", "shaper_lift_v"),
            {"shaper_bias_v": "bias_v", "shaper_lift_v": "bias_lift_v"})
        if plan is not None and not plan.ok:
            return SkillResult(
                skill_name="TipShape", success=False,
                error="；".join(plan.refusals), nanonis_calls=calls)

        # 方案表没给、调用方也没给 ⇒ 跟随**当前成像偏压**;读不到就拒绝,
        # **不回落到 3.0**(那个 3.0 追到 c307d16 的 stub,没有物理来源)。
        bias_v = params.get("bias_v")
        bias_src = "explicit" if _bias_before_policy is not None else "policy"
        if bias_v is None:
            bias_v, why = shaper_bias_default(context)
            bias_src = "read"
            if bias_v is None:
                return SkillResult(
                    skill_name="TipShape", success=False,
                    error=(f"读不到当前偏压({why}),而 bias_v 既没有显式给出、"
                           "方案表也没有 —— 拒绝用写死的 3 V 代替。"),
                    nanonis_calls=calls)

        # ⚠️ ``Bias Lift (V)`` 是**无条件**施加的 —— 厂商同一句话里一个带条件一个
        # 不带:「Bias (V) … **if Change Bias is True**」/「Bias Lift (V) … applied
        # **just after the first Z ramping**」(nanonis_spm NanonisClass.py:3949+)。
        # 所以 ``change_bias=False`` **不等于不加电**,上面那个 3 V 照打。
        # 2026-08-11 之前它的 ParameterSpec 默认就是 3.0,于是「跟随成像偏压」这段
        # 在工具路径上是死代码:pydantic 先把 3.0 灌进来,这里就永远不是 None。
        # 现在没给 = None = 跟随 bias_v(= 成像偏压)。安全门那一侧见
        # ``core.safety.is_electrical_pulse``:它以前一见 change_bias=False 就短路,
        # 从不看这个字段。
        bias_lift_v = params.get("bias_lift_v")
        if bias_lift_v is None:
            bias_lift_v = bias_v

        # WHERE this plunge happens, read now rather than reconstructed later.
        # Tip forming leaves a permanent crater and debris field; the scan map's
        # avoidance model has to know its centre to a few nm, and after
        # restore_feedback the tip may no longer be here. Best-effort: an
        # unreadable position simply isn't reported (see _tip_xy).
        spot = tip_xy_fields(context)
        # TipShaper_PropsSet(Switch_Off_Delay, Change_Bias, Bias_V, Tip_Lift_m,
        #   Lift_Time_1_s, Bias_Lift_V, Bias_Settling_Time_s, Lift_Height_m,
        #   Lift_Time_2_s, End_Wait_Time_s, Restore_Feedback)
        # Change_Bias / Restore_Feedback: 0=no change, 1=True, 2=False
        # change_bias 缺省 **False**(2026-08-11):工作流层自己拒绝走这条路 ——
        # 「TipShaper 的 change-bias 只能**阶跃**改偏压…那一次阶跃本身就是一记
        # 冲量」(``composite/_tip_phases.py``)。裸技能的默认不该与那条判断相反。
        change_bias = 1 if params.get("change_bias", False) else 2
        restore_fb = 1 if params.get("restore_feedback", True) else 2

        rec_props = context.safe_call(
            "TipShaper_PropsSet",
            params.get("switch_off_delay_s", 0.1),
            change_bias,
            bias_v,
            params.get("tip_lift_m", 0.0),
            params.get("lift_time_1_s", 0.1),
            bias_lift_v,
            params.get("bias_settling_s", 0.1),
            resolved_lift_height_m(params),
            params.get("lift_time_2_s", 0.1),
            params.get("end_wait_s", 0.1),
            restore_fb,
        )
        calls.append(rec_props)
        if rec_props.error:
            # ⑫ make a "module not running" error actionable (field trace s306):
            # append a hint to open+start the Tip Shaper module in Nanonis.
            from mast.skills.composite._preflight import module_down_hint
            return SkillResult(
                skill_name="TipShape",
                success=False,
                error=rec_props.error + module_down_hint(rec_props.error, "Tip Shaper"),
                nanonis_calls=calls,
            )

        # TipShaper_Start(Wait_until_finished, Timeout_ms)
        timeout_ms = params.get("timeout_ms", -1)
        rec_start = context.safe_call("TipShaper_Start", 1, timeout_ms)
        calls.append(rec_start)
        if rec_start.error:
            from mast.skills.composite._preflight import module_down_hint
            return SkillResult(
                skill_name="TipShape",
                success=False,
                error=rec_start.error + module_down_hint(rec_start.error, "Tip Shaper"),
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="TipShape",
            success=True,
            # ``bias_lift_v`` 在回包里,因为它是**无条件施加**的那一个 —— 以前回包
            # 只写 bias_v,于是「我关掉了 change_bias」的调用方看着一份没有电压的
            # 回执,而针尖上刚刚过了 3 V。回包要说的是**实际下发的那一组**。
            data={"bias_v": bias_v, "bias_v_source": bias_src,
                  "bias_lift_v": bias_lift_v,
                  "change_bias": change_bias == 1,
                  "lift_height_m": resolved_lift_height_m(params),
                  "completed": True,
                  **spot, **policy_fields_for_result(plan)},
            nanonis_calls=calls,
        )
