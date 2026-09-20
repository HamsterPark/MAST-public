"""电脉冲与 Z/电流读回，共用 io.z_trace 的前后稳定值判据。

非阻塞发起脉冲后在同一连接轮询，分别记录稳定高度变化与过程瞬态。
返回的 up/down 是 Z 数值方向；物理退开方向依赖 instrument_profile.z_extend_sign，
本层不把高度变化直接等同于修针成功。

若固件阻塞发起调用，结果标记 start_blocked，此时曲线主要反映事后状态。
前后稳定值比较与完整过程记录必须区分，不能声称捕获了未采集的过程。
"""
from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io import z_trace
from mast.skills.base import BaseSkill
from mast.skills.builtins._readback_stream import (
    channel_block,
    save_trace,
    stream_with_action,
    trace_ref,
)
from mast.skills.builtins._tip_xy import tip_xy_fields

# 脉冲后需要等待 Z 稳定，默认延迟是待验证的工程估计。
# 尾窗仍在变化时会低估稳定高度差，应按设备响应检查窗口。
DEFAULT_POST_ROLL_S = 0.3


class BiasPulseWithReadback(BaseSkill):
    """打一发电脉冲,并在脉冲前后流式采 Z/电流,判定 Z 跳变的方向与幅度。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="BiasPulseWithReadback",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # capabilities 才是三档操作模式的门控真源(safety_level 不是):
            # 少了这个标签,SAFE 模式看不见这个技能,照跑。
            capabilities=frozenset({"bias_pulse"}),
            # AUTO,与 BiasPulse 同级:脉冲被 Nanonis 自身量程 + 全局 ±10 V
            # SafetyGate 双重限制,伤不到硬件。
            safety_level=SafetyLevel.AUTO,
            description=(
                "发一个偏压脉冲，同时连续采集 Z 与电流，然后报告 Z 在脉冲前"
                "的稳定值与脉冲后的稳定值之间跳了多远、往哪个方向跳"
                "（脉冲进行中的瞬态不计）。当这一发脉冲的目的是修针、"
                "而你需要知道它到底有没有起作用时，用它而不是 BiasPulse。"
            ),
            parameters=[
                ParameterSpec(
                    name="bias_v", type="float", unit="V",
                    description="脉冲电压。贵金属上修针通常用 ±10 V；"
                                "更温和的做法用 ±3–7 V。",
                    required=True, min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="width_s", type="float", unit="s",
                    description="脉冲宽度（由硬件计时）",
                    required=False, default=0.5, min_value=1e-3, max_value=2.0),
                ParameterSpec(
                    name="z_hold", type="int",
                    description="脉冲期间的 Z controller：0=不变，1=保持，"
                                "2=不保持。默认 1 —— feedback 若还开着，一发脉冲"
                                "会把电流抬高几个数量级，Z 会一路追着它"
                                "直接扎进表面。",
                    required=False, default=1, allowed_values=[0, 1, 2]),
                ParameterSpec(
                    name="absolute", type="bool",
                    description="True=绝对偏压，False=相对当前值",
                    required=False, default=True),
                ParameterSpec(
                    name="poll_hz", type="float", unit="Hz",
                    description="目标逐帧轮询速率（每帧各采一次电流+Z；"
                                "受 RTT 限制，每通道约 1 kHz 封顶）",
                    required=False, default=2000.0, min_value=10.0, max_value=8000.0),
                ParameterSpec(
                    name="pre_roll_s", type="float", unit="s",
                    description="脉冲**之前**采集的基线（这就是 z1）",
                    required=False, default=0.1, min_value=0.01, max_value=5.0),
                ParameterSpec(
                    name="post_roll_s", type="float", unit="s",
                    description="脉冲结束之后继续采集；它稳定下来的尾段"
                                "就是 z3",
                    required=False, default=DEFAULT_POST_ROLL_S,
                    min_value=0.02, max_value=5.0),
                ParameterSpec(
                    name="max_capture_s", type="float", unit="s",
                    description="整个采集窗口的硬上限",
                    required=False, default=5.0, min_value=0.2, max_value=30.0),
                ParameterSpec(
                    name="jump_k", type="float",
                    description="跳变灵敏度：|Δ| > median+k·σ_robust 即标记。默认值是算法工作点，应按当前数据的噪声与误报代价验证。"
                                ,
                    required=False, default=z_trace.DEFAULT_JUMP_K,
                    min_value=1.0, max_value=50.0),
                ParameterSpec(
                    name="step_tol_nm", type="float", unit="nm",
                    description="z3−z1 上的死区（nm）；tol = max(此值, "
                                "step_tol_k·σ(baseline))。默认 0.5 nm：值得"
                                "据以动作的脉冲台阶在几十 nm 量级。",
                    required=False, default=0.5, min_value=0.0, max_value=1000.0),
                ParameterSpec(
                    name="step_tol_k", type="float",
                    description="若 step_tol_nm=0，则 tol = z1 基线的 k·σ",
                    required=False, default=4.0, min_value=1.0, max_value=50.0),
            ],
            estimated_duration_s=6.0,
            composition_level=1,
            tags=["bias", "pulse", "readback", "z", "tip", "write"],
        )

    def validate_params(self, params: dict) -> list[str]:
        """标准校验 + 当前针尖的安全包络。

        包络在这里而不是 execute:它必须在任何硬件调用之前拦住,而 validate 的错误
        原样回给调用方(看到的是「为什么被拒」而不是一次静默的 no-op)。
        **超上限拒绝,不夹紧** —— 同一条哲学贯穿粗动电压四重锁:悄悄把 10 V 改成
        8 V 会让用户以为自己做的是他要的实验。

        修针流程默认的 ±10 V 是用户对**金属丝针尖**的做法。铂铱(包络 8 V)、
        磁性/超导针、qPlus(3 V)上会在这里被拒绝,那不是 bug 是保护。
        """
        errors = super().validate_params(params)
        from mast.skills.builtins._tip_policy import apply_tip_policy
        # 方案表管这个量叫 pulse_v,本技能的参数名是 bias_v(全局 ±10 V 安全帽按
        # 参数名子串匹配,叫 pulse_v 也在帽内,但 bias_v 与 BiasPulse 保持一致)。
        _, plan = apply_tip_policy(params, ("pulse_v",), {"pulse_v": "bias_v"})
        if plan is not None and not plan.ok:
            errors.extend(plan.refusals)
        return errors

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        bias_v = float(params["bias_v"])
        width_s = float(params.get("width_s", 0.5))
        z_hold = int(params.get("z_hold", 1))
        absolute = bool(params.get("absolute", True))
        abs_rel = 2 if absolute else 1          # 1=relative, 2=absolute
        poll_hz = float(params.get("poll_hz", 2000.0))
        pre_roll = float(params.get("pre_roll_s", 0.1))
        post_roll = float(params.get("post_roll_s", DEFAULT_POST_ROLL_S))
        max_capture = float(params.get("max_capture_s", 5.0))
        jump_k = float(params.get("jump_k", z_trace.DEFAULT_JUMP_K))

        # 脉冲落在哪里,在开火**之前**读回。脉冲会改造表面(那通常正是目的),扫描
        # 地图要的是它真正的坐标,不是 ≤1 s 前的缓存值。尽力而为。
        spot = tip_xy_fields(context)

        # 固件多留 50% 余量,与 shaper 那一路同口径。
        total_capture = min(pre_roll + width_s * 1.5 + post_roll, max_capture)

        capture = stream_with_action(
            context,
            fire=lambda: context.safe_call(
                "Bias_Pulse", 0, width_s, bias_v, z_hold, abs_rel),
            total_capture_s=total_capture, pre_roll_s=pre_roll,
            poll_hz=poll_hz, calls=calls)

        rec = capture.fire_record
        if rec is not None and rec.error:
            return SkillResult(skill_name="BiasPulseWithReadback", success=False,
                               error=rec.error, nanonis_calls=calls)
        if not capture.fired:
            return SkillResult(
                skill_name="BiasPulseWithReadback", success=False,
                error="Aborted before the pulse was fired — no pulse was applied.",
                nanonis_calls=calls)
        if capture.empty:
            return SkillResult(
                skill_name="BiasPulseWithReadback", success=False,
                error="Pulse fired but no samples were collected (TCP errors); "
                      "the tip state after this pulse is UNKNOWN.",
                nanonis_calls=calls)

        verdict = z_trace.step_verdict(
            capture.z_s, capture.z_t, capture.fire_t_s, post_roll_s=post_roll,
            tol_k=float(params.get("step_tol_k", 4.0)),
            tol_abs_m=float(params.get("step_tol_nm", 0.5)) * 1e-9)

        # wait=0 本该毫秒级返回;若它耗掉了脉冲时长的一半以上,说明固件忽略了它。
        # 绝对下限是 20 ms 而不是 shaper 那边的 200 ms:脉冲的时间尺度比整形短一个
        # 数量级(500 ms 对几秒),照抄 200 ms 会让这个守卫在任何短于 400 ms 的脉冲
        # 上**永远不触发** —— 守卫在,却只在不需要它的地方管用。20 ms 仍远大于
        # 一次 TCP 往返(~1 ms),不会把正常抖动误报成阻塞。
        start_blocked = capture.fire_blocked_s >= max(0.5 * width_s, 0.02)

        data = {
            "completed": True,
            "bias_v": bias_v,
            "width_s": width_s,
            "z_hold": z_hold,
            "absolute": absolute,
            "current": channel_block(capture.current_s, capture.current_t, "a"),
            "z": channel_block(capture.z_s, capture.z_t, "m"),
            "jumps": {
                "current": z_trace.detect_jumps(
                    capture.current_s, capture.current_t, jump_k),
                "z": z_trace.detect_jumps(capture.z_s, capture.z_t, jump_k),
            },
            "step": verdict,
            "timing": {
                "pre_roll_s": pre_roll,
                "post_roll_s": post_roll,
                "capture_s": capture.capture_s,
                "pulse_t_s": capture.fire_t_s,
                "fire_call_blocked_s": capture.fire_blocked_s,
                "start_blocked": start_blocked,
                "aborted": capture.aborted,
                "n_current": len(capture.current_s),
                "n_z": len(capture.z_s),
            },
            **spot,
        }
        if start_blocked:
            data["warning"] = (
                "Bias_Pulse(wait=0) blocked for ~the pulse duration; the firmware "
                "likely ignored it, so the trace reflects the AFTER state rather "
                "than the pulse itself. The before/after verdict still holds.")

        # ── 原始曲线落盘 ──────────────────────────────────────────────────
        # 与扎针那一路同一个口(采集循环共用 ``_readback_stream``,落盘也就共用
        # 一份实现 —— 复制第二份的下场是其中一份的修复到不了另一份)。
        # 判定报的是前后**稳定值**之差,中间那个瞬态峰按设计不参与;而 qPlus 针尖
        # 上「z 大幅弹起、音叉起振」恰恰就在那个峰里。判定不该改,过程要留得下来。
        direction = verdict.get("direction", "insufficient_data")
        saved = save_trace(
            capture, skill="BiasPulseWithReadback",
            meta={
                "bias_v": bias_v,
                "width_s": width_s,
                "z_hold": z_hold,
                "absolute": absolute,
                "poll_hz": poll_hz,
                "pre_roll_s": pre_roll,
                "post_roll_s": post_roll,
                "start_blocked": start_blocked,
                "direction": direction,
                "delta_m": verdict.get("delta_m"),
                "z_min_m": verdict.get("z_min_m"),
                **spot,
            })
        data.update(saved)

        dz_nm = float(verdict.get("delta_m") or 0.0) * 1e9
        cn = {"up": "Z 向上跳变", "down": "Z 向下跳变",
              "none": "Z 无跳变", "insufficient_data": "数据不足,判不了"}[direction]
        summary = f"{bias_v:+.1f} V / {width_s * 1e3:.0f} ms 脉冲 → {cn}"
        if direction in ("up", "down"):
            summary += f" ({dz_nm:+.2f} nm)"
        # 指针写进摘要:摘要是**唯一**穿过工具边界的东西。
        summary += trace_ref(saved)
        return SkillResult(skill_name="BiasPulseWithReadback", success=True,
                           data=data, nanonis_calls=calls, summary=summary)


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(BiasPulseWithReadback, context_provider)
