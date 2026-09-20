"""Tip shaper WITH real-time current+Z readback (2026-06-02).

TipShapeWithReadback runs the hardware tip shaper while concurrently streaming
the tunnelling **current** and **Z position** so the operator/agent can see how
the two channels jump at the instant the apex reshapes — the in-process trace,
not a before/after comparison.

How "concurrent" works on one TCP connection:
  * ``TipShaper_Start(Wait_until_finished=0, …)`` returns IMMEDIATELY — the shaper
    runs on the Nanonis controller (hardware side); the Python client is then
    free to poll ``Current_Get`` / ``ZCtrl_ZPosGet`` on the SAME connection
    during the procedure. (TipShape uses ``Wait_until_finished=1`` which blocks,
    so this readback variant deliberately uses 0.)
  * Sampling is TCP-polled → bounded by RTT (~1 kHz per channel when alternating
    current+Z each frame). Sub-millisecond atomic transients can be under-sampled;
    for those use the hardware oscilloscope (AcquireOsciTrace / Osci1T).
  * Honesty guard: if ``TipShaper_Start`` itself blocks for ≈ the procedure
    duration (some firmware ignores wait=0), ``start_blocked`` is flagged in the
    result — the capture then mostly reflects the AFTER state, not the process.

Safety: AUTO (mirrors TipShape per the 2026-06-11 re-scoping). The Z ramp is
FINE Z (Z-controller piezo, bounded by the Nanonis Z-piezo range), not the
open-loop coarse stepper, so it cannot crash the instrument; tip conditioning
runs autonomously. The plunge/lift excursions are bounded per-spec (±100 nm) and
by the global tip_lift limit. Same TipShaper_PropsSet + TipShaper_Start pair as
TipShape, plus the readback loop. (Earlier docstrings mislabelled this DANGEROUS,
contradicting the AUTO metadata — corrected 2026-07-03.)
"""

from __future__ import annotations

import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io import z_trace
from mast.skills.base import BaseSkill
from mast.skills.builtins._tip_policy import (
    resolved_lift_height_m,
    shaper_bias_default,
)
from mast.skills.builtins._readback_stream import (
    channel_block,
    save_trace,
    scalar as _scalar,
    stats as _stats,
    stream_with_action,
    trace_ref,
)
from mast.skills.builtins._tip_xy import tip_xy_fields


# 判据本体住在 mast.io.z_trace(纯函数,无硬件)。这里只留本技能语义的薄封装:
# 电脉冲要用同一套判据,把它复制一份的下场是两边阈值各自漂移。
_detect_jumps = z_trace.detect_jumps
_median = z_trace._median
_std = z_trace._std


def _stage_boundaries(shaper_start_t, params: dict) -> list[dict]:
    """Approximate stage boundary times (capture clock, seconds) from the
    TipShaper timing params, anchored at ``shaper_start_t``. Order per Nanonis:
    switch_off → Z ramp 1 (plunge) → bias settle → Z ramp 2 (retract) → end
    wait. ESTIMATED — real firmware adds small overheads; calibrate on the sim."""
    if shaper_start_t is None:
        return []
    t = float(shaper_start_t)
    out = [{"stage": "pre_roll", "t_start": 0.0, "t_end": t,
            "note": "feedback-held baseline (z1)"}]
    for name, dur in (
        ("switch_off", params.get("switch_off_delay_s", 0.1)),
        ("z_ramp_1_plunge", params.get("lift_time_1_s", 0.1)),
        ("bias_settle", params.get("bias_settling_s", 0.1)),
        ("z_ramp_2_retract", params.get("lift_time_2_s", 0.1)),
        ("end_wait", params.get("end_wait_s", 0.1)),
    ):
        out.append({"stage": name, "t_start": t, "t_end": t + float(dur)})
        t += float(dur)
    out.append({"stage": "post_roll", "t_start": t, "t_end": None,
                "note": "feedback restored (z3)"})
    return out


#: 恒流(反馈开)下扎针尖时,z_trace 的中性方向读作什么。Z 变大 = 针尖退开 =
#: 表面等效变高,所以向上是「表面上长了东西」。电脉冲那一路读法不同(见
#: ``bias_pulse_readback``)—— 这正是判据保持中性、语义各归各家的原因。
_INDENT_VERDICT = {
    "up": "cluster",
    "down": "tip_changed_or_pit",
    "none": "no_change",
    "insufficient_data": "insufficient_data",
}

_INDENT_ADVICE = {
    "no_change": "没扎上 — 增大向下扎的深度(更负的 tip_lift_m)后重试。",
    "cluster": "扎上了,表面已形成一个 cluster。",
    "tip_changed_or_pit": "针尖状态改变(或扎出一个坑)。",
}


def _three_step_verdict(z_s, z_t, shaper_start_t, *, post_roll_s,
                        tol_k=4.0, tol_abs_m=0.0,
                        current_s=None, current_t=None) -> dict:
    """Constant-current (feedback-ON) indentation verdict from the Z trace.

    Three steps: z1 = pre-plunge baseline (feedback-held), z_min = deepest
    plunge, z3 = settled Z after feedback is restored. Delta = z3 - z1.
    Z larger = tip retracted = surface effectively higher, so:
      * |Delta| <= tol  -> didn't bite ('no_change') — increase plunge depth
      * Delta >  tol     -> a cluster grew on the surface ('cluster')
      * Delta < -tol     -> tip apex changed, or a pit was dug ('tip_changed_or_pit')

    Thin wrapper over :func:`mast.io.z_trace.step_verdict` — the arithmetic and
    the tolerance live there so the bias-pulse path shares one definition.
    """
    # 电流通道用于定位反馈重新接入后的稳定窗口。
    # 缺少电流时只能依赖时间尾窗，而它可能仍覆盖回抬或反馈未接入阶段；
    # feedback_segment_source 必须说明窗口来源，不能把按构造回到基线解释成无变化。
    out = z_trace.step_verdict(z_s, z_t, shaper_start_t, post_roll_s=post_roll_s,
                               tol_k=tol_k, tol_abs_m=tol_abs_m,
                               current_s=current_s, current_t=current_t)
    verdict = _INDENT_VERDICT.get(out.get("direction", ""), "insufficient_data")
    if verdict == "insufficient_data":
        # ⚠️ 这里以前是 ``return {"verdict": "insufficient_data"}`` —— 把整个 dict
        # 丢掉。于是诊断字段(``n_pre`` / ``n_post`` / ``max_gap_s`` …)**恰好在
        # 不需要它们的时候幸存、在需要它们的时候消失**:判不出来正是最想知道
        # 「采了几个点、最长往返多久」的时刻。这就是隔壁 bias_pulse_readback 那句
        # 注释说的「守卫在,却只在不需要它的地方管用」,同一个形状。
        # 保留全部诊断,只把判定换成本层的词汇。
        out = {k: v for k, v in out.items() if k not in ("direction", "z_max_m")}
        out["verdict"] = "insufficient_data"
        # ``feedback_segment_too_short`` 自带一句**具体的**下一步（加长 post_roll_s，
        # 别加深）。默认那句「增大扎入深度」在这种情形下正好是错的方向。
        out.setdefault("advice", _INDENT_ADVICE.get("no_change", ""))
        return out
    out = {k: v for k, v in out.items() if k not in ("direction", "z_max_m")}
    out["verdict"] = verdict
    out["advice"] = _INDENT_ADVICE[verdict]
    return out


class TipShapeWithReadback(BaseSkill):
    """Run the tip shaper while streaming current + Z to capture the apex-change."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TipShapeWithReadback",
            version="1.0.0",
            category=SkillCategory.WRITE,
            capabilities=frozenset({"tip_shaping"}),
            # AUTO (2026-06-11): fine-Z ramp + bias pulse (same as TipShape) —
            # bounded by Nanonis, not the open-loop coarse stepper. Autonomous.
            safety_level=SafetyLevel.AUTO,
            description=(
                "运行硬件 tip shaper，**同时**在流程进行中连续采集电流+Z，"
                "以捕捉针尖顶端发生变化时两路通道各自怎么跳"
                "（用的是 TipShaper_Start wait=0，所以轮询与动作是重叠进行的）。"
            ),
            parameters=[
                # ── shaper params (identical to TipShape) ──
                ParameterSpec(name="switch_off_delay_s", type="float", unit="s",
                              description="关闭 controller 之前对 Z 做平均的时长",
                              required=False, default=0.1, min_value=0.0),
                ParameterSpec(name="change_bias", type="bool",
                              description=(
                                  "在第一段 Z ramp 之前，把偏压跳变到 bias_v。"
                                  "**默认关闭**（2026-08-11）：workflow 层本来就拒绝这条路 —— "
                                  "TipShaper 只能**一步**改完偏压，而一步跳变"
                                  "本身就是一记冲击"
                                  "（_tip_phases.py）。请先自己把偏压设好（用带 slew rate "
                                  "的 SetBias），再在这一项关闭的情况下扎针。"),
                              required=False, default=False),
                ParameterSpec(name="bias_v", type="float", unit="V",
                              description=("第一段 Z ramp 之前的偏压（当 change_bias 开启时）。"
                                           "**省略**它则跟随**当前**的成像偏压 —— 也就是用户的"
                                           "参数组（2026-08-10）。没有 3 V 兜底：偏压若读不到，"
                                           "这个技能直接拒绝执行。"),
                              required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="tip_lift_m", type="float", unit="m",
                              description="第一段 Z ramp（相对量），单位**米**。常用 ±2n，上限 ±100n（±100 nm）。",
                              required=False, default=0.0, min_value=-1e-7, max_value=1e-7),
                ParameterSpec(name="lift_time_1_s", type="float", unit="s",
                              description="第一段 Z ramp 的时长",
                              required=False, default=0.1, min_value=0.0),
                ParameterSpec(name="bias_lift_v", type="float", unit="V",
                              description=("在第一段 Z ramp**刚结束之后**施加的偏压。"
                                           "⚠️ 与 bias_v 不同，这一项是**无条件**施加的 —— "
                                           "change_bias **不能**把它解除（厂商原文：'Bias (V) … "
                                           "if Change Bias is True' 对比 'Bias Lift (V) … applied "
                                           "just after the first Z ramping'）。**省略**它则跟随 "
                                           "bias_v，也就是当前的成像偏压。没有 3 V 兜底。"),
                              required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="bias_settling_s", type="float", unit="s",
                              description="施加偏压值之后的等待时长",
                              required=False, default=0.1, min_value=0.0),
                ParameterSpec(name="lift_height_m", type="float", unit="m",
                              description=("第二段 Z ramp 的高度（也就是**退回来**的那一段），单位米。"
                                           "**省略**它则默认取 -tip_lift_m —— 扎进去 x，"
                                           "就拉回来 x，这是每个 composite 早已在用的"
                                           "规则。常用 1n..5e-9，上限 ±100n（±100 nm）。"),
                              required=False, min_value=-1e-7, max_value=1e-7),
                ParameterSpec(name="lift_time_2_s", type="float", unit="s",
                              description="第二段 Z ramp 的时长",
                              required=False, default=0.1, min_value=0.0),
                ParameterSpec(name="end_wait_s", type="float", unit="s",
                              description="恢复初始偏压之后的等待时长",
                              required=False, default=0.1, min_value=0.0),
                ParameterSpec(name="restore_feedback", type="bool",
                              description="结束时恢复 Z-controller 状态",
                              required=False, default=True),
                ParameterSpec(name="timeout_ms", type="int", unit="ms",
                              description="shaper 的超时（-1 = 永远等待）",
                              required=False, default=-1, min_value=-1),
                # ── readback params ──
                ParameterSpec(name="poll_hz", type="float", unit="Hz",
                              description="目标逐帧轮询速率（每帧各采一次电流+Z；"
                                          "每通道有效速率 ≈ poll_hz，受 RTT 限制、~1 kHz 封顶）",
                              required=False, default=2000.0, min_value=10.0, max_value=8000.0),
                ParameterSpec(name="pre_roll_s", type="float", unit="s",
                              description="shaper 启动**之前**采集的基线",
                              required=False, default=0.05, min_value=0.0, max_value=5.0),
                ParameterSpec(name="post_roll_s", type="float", unit="s",
                              description="在估计的 shaper 结束时刻**之后**继续采集",
                              required=False, default=0.1, min_value=0.0, max_value=5.0),
                ParameterSpec(name="max_capture_s", type="float", unit="s",
                              description="整个采集窗口的硬上限（安全考虑）",
                              required=False, default=5.0, min_value=0.1, max_value=30.0),
                ParameterSpec(name="jump_k", type="float",
                              description="跳变灵敏度：|Δ| > median+k·σ_robust 即标记",
                              required=False, default=5.0, min_value=1.0, max_value=50.0),
                # constant-current three-step indent verdict (z3 vs z1)
                ParameterSpec(name="indent_tol_nm", type="float", unit="nm",
                              description="判定「扎入」用的 z3−z1 死区（nm）；"
                                          "tol = max(此值, indent_tol_k·σ(baseline))。"
                                          "默认 0.02 nm：低于原子团簇的尺度"
                                          "（~0.1–0.3 nm），高于 z 噪声。",
                              required=False, default=0.02, min_value=0.0, max_value=100.0),
                ParameterSpec(name="indent_tol_k", type="float",
                              description="若 indent_tol_nm=0，则 tol = z1 基线的 k·σ",
                              required=False, default=4.0, min_value=1.0, max_value=50.0),
            ],
            estimated_duration_s=12.0,
            composition_level=1,
            tags=["tip", "shaper", "readback", "current", "z", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []

        # bias 缺省沿用当前成像偏压，使用 _tip_policy.shaper_bias_default 共用实现。
        bias_v = params.get("bias_v")
        bias_src = "explicit"
        if bias_v is None:
            bias_v, why = shaper_bias_default(context)
            bias_src = "read"
            if bias_v is None:
                # **不回落到 3.0**:读不到就说读不到,让调用方看见,而不是悄悄
                # 在结上打一个没人要求过的 3 V。
                return SkillResult(
                    skill_name="TipShapeWithReadback", success=False,
                    error=(f"读不到当前偏压({why}),而 bias_v 没有显式给出 —— "
                           "拒绝用写死的 3 V 代替。要么先修好偏压读取,"
                           "要么显式传 bias_v。"),
                    nanonis_calls=calls)
        # bias_v 与 bias_lift_v 的施加条件不同，change_bias=False 不会自动关闭 bias lift。
        # 两个字段都需使用明确参数或当前成像偏压，不能由 schema 默认值提前填入固定高偏压。
        bias_lift_v = params.get("bias_lift_v")
        if bias_lift_v is None:
            bias_lift_v = bias_v

        # ── 1) PropsSet (identical encoding to TipShape) ──
        # change_bias 缺省 **False**(2026-08-11),与工作流层的判断一致:
        # 「TipShaper 的 change-bias 只能**阶跃**改偏压…那一次阶跃本身就是一记冲量」
        # (``composite/_tip_phases.py``)。
        change_bias = 1 if params.get("change_bias", False) else 2
        restore_fb = 1 if params.get("restore_feedback", True) else 2
        rec_props = context.safe_call(
            "TipShaper_PropsSet",
            params.get("switch_off_delay_s", 0.1), change_bias,
            bias_v, params.get("tip_lift_m", 0.0),
            params.get("lift_time_1_s", 0.1), bias_lift_v,
            params.get("bias_settling_s", 0.1), resolved_lift_height_m(params),
            params.get("lift_time_2_s", 0.1), params.get("end_wait_s", 0.1),
            restore_fb,
        )
        calls.append(rec_props)
        if rec_props.error:
            return SkillResult(skill_name="TipShapeWithReadback", success=False,
                               error=rec_props.error, nanonis_calls=calls)

        # ── capture window sizing ──
        poll_hz = float(params.get("poll_hz", 2000.0))
        period = 1.0 / poll_hz
        pre_roll = float(params.get("pre_roll_s", 0.05))
        post_roll = float(params.get("post_roll_s", 0.1))
        max_capture = float(params.get("max_capture_s", 5.0))
        jump_k = float(params.get("jump_k", 5.0))
        # estimated shaper duration from its own timing params (+50% firmware margin)
        shape_s = (params.get("switch_off_delay_s", 0.1)
                   + params.get("lift_time_1_s", 0.1)
                   + params.get("bias_settling_s", 0.1)
                   + params.get("lift_time_2_s", 0.1)
                   + params.get("end_wait_s", 0.1))
        total_capture = min(pre_roll + shape_s * 1.5 + post_roll, max_capture)
        timeout_ms = params.get("timeout_ms", -1)

        # 扎针留下的是**永久痕迹**,所以「在哪扎的」和「扎没扎上」一样是结果的一部分
        # —— ``_tip_xy`` 的模块 docstring 逐字点了「tip forming」的名,而这个技能
        # 一直没接上,只有电脉冲那一路接了(同形状漏一处的老毛病)。在开火**之前**
        # 读:扎针会改造表面,坐标要的是它真正发生的地方,不是 ≤1 s 前的缓存值。
        # 尽力而为,读不到就什么都不报(回退到快照并标 ``pos_src``)。
        spot = tip_xy_fields(context)

        # The poll loop itself lives in _readback_stream: the bias-pulse twin
        # runs the identical loop, and a fix to one of its four fiddly details
        # (absolute-time scheduling, pre-roll, fire timing, abort) has to reach
        # both.
        capture = stream_with_action(
            context,
            fire=lambda: context.safe_call("TipShaper_Start", 0, timeout_ms),
            total_capture_s=total_capture, pre_roll_s=pre_roll,
            poll_hz=poll_hz, calls=calls)
        cur_s, cur_t = capture.current_s, capture.current_t
        z_s, z_t = capture.z_s, capture.z_t
        start_rec = capture.fire_record
        shaper_start_elapsed = capture.fire_t_s
        start_blocked_s = capture.fire_blocked_s

        if start_rec is not None and start_rec.error:
            return SkillResult(skill_name="TipShapeWithReadback", success=False,
                               error=start_rec.error, nanonis_calls=calls)

        if capture.empty:
            return SkillResult(
                skill_name="TipShapeWithReadback", success=False,
                error="No samples collected (TCP errors or aborted before/while shaping).",
                nanonis_calls=calls)

        cap = capture.capture_s
        # wait=0 expected to return in ~ms; if it took ≈ the procedure, the
        # capture mostly reflects the AFTER state (firmware ignored wait=0).
        start_blocked = start_blocked_s >= max(0.5 * shape_s, 0.2)

        data = {
            "completed": True,
            # 实际下发的那个值 + 它是哪来的(显式 / 读来的)。``bias_lift_v`` 也在
            # 回包里,因为它才是**无条件施加**的那一个:以前回包只写 bias_v,于是
            # 「我关掉了 change_bias」的调用方看着一份没有电压的回执,而针尖上刚刚
            # 过了 3 V。回包要说的是实际下发的那一组。
            "bias_v": bias_v,
            "bias_v_source": bias_src,
            "bias_lift_v": bias_lift_v,
            "change_bias": change_bias == 1,
            "lift_height_m": resolved_lift_height_m(params),
            "current": channel_block(cur_s, cur_t, "a"),
            "z": channel_block(z_s, z_t, "m"),
            "jumps": {
                "current": _detect_jumps(cur_s, cur_t, jump_k),
                "z": _detect_jumps(z_s, z_t, jump_k),
            },
            "timing": {
                "pre_roll_s": pre_roll,
                "post_roll_s": post_roll,
                "estimated_shape_s": shape_s,
                "capture_s": cap,
                "shaper_start_t_s": shaper_start_elapsed,
                "start_call_blocked_s": start_blocked_s,
                "start_blocked": start_blocked,
                "n_current": len(cur_s),
                "n_z": len(z_s),
                "fs_current_hz": (len(cur_s) - 1) / cap if cap > 0 and len(cur_s) > 1 else 0.0,
                "fs_z_hz": (len(z_s) - 1) / cap if cap > 0 and len(z_s) > 1 else 0.0,
            },
            **spot,
        }
        # Stage alignment (current/z ↔ TipShaper phases) + constant-current
        # three-step indent verdict (z1 baseline → plunge → z3 restored).
        data["stages"] = _stage_boundaries(shaper_start_elapsed, params)
        data["indent"] = _three_step_verdict(
            z_s, z_t, shaper_start_elapsed, post_roll_s=post_roll,
            tol_k=float(params.get("indent_tol_k", 4.0)),
            tol_abs_m=float(params.get("indent_tol_nm", 0.02)) * 1e-9,
            current_s=capture.current_s, current_t=capture.current_t)

        if start_blocked:
            data["warning"] = ("TipShaper_Start(wait=0) blocked ~the full procedure; "
                               "capture likely reflects the AFTER state, not the "
                               "in-process trace. Consider the hardware oscilloscope.")

        # 保存原始过程曲线，保留最终状态判据没有使用的瞬态信息。
        # Δz = z3 − z1 表示稳定高度的净变化，z_min 只记录，不直接参与该判定。
        # 表面与针尖同时改变时，净变化可能掩盖过程，因此过程记录与最终结论都需保留。
        # 复用落盘后仅返回指针的通路，浮点序列不进入摘要或 checkpointer。
        saved = save_trace(
            capture, skill="TipShapeWithReadback",
            stages=data["stages"],
            meta={
                # 这次扎针**是怎么扎的** —— 下次要把「深度 → 曲线 → 团簇」串起来,
                # 靠的就是这几个数和曲线躺在同一个文件里。
                "tip_lift_m": params.get("tip_lift_m", 0.0),
                "lift_height_m": resolved_lift_height_m(params),
                "bias_v": bias_v,
                "bias_v_source": bias_src,
                "bias_lift_v": bias_lift_v,
                "change_bias": change_bias == 1,
                "switch_off_delay_s": params.get("switch_off_delay_s", 0.1),
                "lift_time_1_s": params.get("lift_time_1_s", 0.1),
                "bias_settling_s": params.get("bias_settling_s", 0.1),
                "lift_time_2_s": params.get("lift_time_2_s", 0.1),
                "end_wait_s": params.get("end_wait_s", 0.1),
                "restore_feedback": restore_fb == 1,
                "poll_hz": poll_hz,
                "pre_roll_s": pre_roll,
                "post_roll_s": post_roll,
                "start_blocked": start_blocked,
                "verdict": data["indent"].get("verdict", ""),
                "delta_m": data["indent"].get("delta_m"),
                "z_min_m": data["indent"].get("z_min_m"),
                **spot,
            })
        data.update(saved)

        # one-line summary for the instrument chat (full traces stay in data)
        _ind = data.get("indent", {})
        _cn = {"no_change": "没扎上", "cluster": "扎上了(cluster)",
               "tip_changed_or_pit": "针尖改变/坑",
               "insufficient_data": "数据不足"}.get(
            _ind.get("verdict", ""), _ind.get("verdict", ""))
        # 摘要是**唯一**穿过工具边界的东西,所以指针必须写在这里 —— 写进 data
        # 而不写进摘要,agent 就还是只看得到那一行判定。
        summary = (f"针尖整形: {_cn} (Δz={_ind.get('delta_m', 0.0) * 1e9:+.2f} nm) — "
                   f"{_ind.get('advice', '')}" + trace_ref(saved))
        return SkillResult(skill_name="TipShapeWithReadback", success=True,
                           data=data, nanonis_calls=calls, summary=summary)


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(TipShapeWithReadback, context_provider)
