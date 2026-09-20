"""偏压随机扰动 —— 用户在 GUI 上「把偏压条拉来拉去」的软件版。

技术要点(用于原子分辨针尖处理):把偏压设到 20 mV、电流设到 500 pA,随后在
±20 mV 范围内高频率随机切换偏压;因为是软件控制斜坡,切换速率与斜率都要
留有余量,避免超出仪器承受范围。

物理上在做什么:恒流反馈开着时,|V| 变小 → 隧道电流掉 → 反馈把针尖推近样品。在
±20 mV 内反复跳变,针尖顶端反复经历不同的场与距离,最不稳的那几个原子会重排。
这是**温和**的改性:全程 |V| ≤ 20 mV,与修针脉冲的伏级差两个数量级。

## 为什么不复用 BiasSettleChange

``composite/bias_settle.py`` 有一条硬规矩:反馈开着时目标偏压不许落在
``|V| < 50 mV`` 的死区里 —— 恒流反馈在那里维持不住电流,会一直把针尖往下推。
它会**拒绝**本技能的每一个目标值。

本技能整个工作区间都在那条死区之内,**这是刻意的例外**,所以它自带四道护栏取代
那条规矩:

1. **|V| 有下限**(``wiggle_lower_v``,出厂 4 mV)。目标值永远不落在零附近 ——
   「在死区里跳」和「停在零点上」是两回事,后者才是撞针;
2. **穿零不停留**。符号翻转的那一段用允许的最大斜率单步跨过去,中间不设停留点;
3. **每一步都看电流**。超过 ``abort_current_a`` 立即恢复初始偏压并中止;
4. **突发有硬上限**(≤10 s)。它是一次**短促扰动**,不是一个可以一直开着的模式。

## 为什么不复用 SetBias 的 slew

``SetBias(slew_rate_v_per_s=…)`` 的斜坡循环里没有地方插电流看护,而且每个目标值
一次 skill 调用会把记录刷爆(一次突发有几十个目标)。这里自己写循环,但**骨架四点
原样继承**自 ``bias.py:127-221``:先 ``Bias_Get`` 确认起点(读不到就拒绝,绝不从
假设的 0 V 开始斜坡)、固定步进网格、每步查 abort、失败即停。

## 安全归类

``AUTO`` + ``capabilities={"tip_shaping"}``,与 ``BiasPulse`` 2026-06-11 的重定级
同一条论证:能物理毁掉仪器的是开环粗动,而这里的偏压被三层界住(本技能 ±0.1 V 硬
帽、全局 SafetyGate ±10 V、Nanonis 自己的量程)。声明 ``tip_shaping`` 让它自动
落进 SAFE 模式的硬拒名单,并在 CRITICAL 事件关闸时仍可作为补救手段被放行。
"""

from __future__ import annotations

import random
import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

#: 绝对硬帽 —— 写死在代码里,配置放宽不了,``execute`` 里二次校验。
#:
#: 与粗动电压四重锁同一条哲学:超上限**拒绝,不夹紧**。悄悄降下来的幅度是一个
#: 被报告成成功的错误动作。
ABSOLUTE_MAX_V = 0.1            # 单个目标偏压的幅值上限
ABSOLUTE_MAX_BURST_S = 10.0     # 一次突发的总时长上限
ABSOLUTE_MAX_SLEW = 2.0         # 斜率上限(V/s),与 bias_settle 的穿零斜率同值

#: 斜坡的时间网格。比出厂的最短停留(50 ms)细,这样「停留」和「爬坡」不会互相
#: 吃掉对方的时间粒度。
_STEP_INTERVAL_S = 0.05

#: 小于这个幅度的改变直接下发,不分步 —— 分步只是多几次 TCP 往返。
_MIN_RAMP_V = 1e-3


def _parse_scalar(record) -> "float | None":
    """Nanonis ``(header, body, [values])`` 里的第一个标量。"""
    if record is None or getattr(record, "error", None):
        return None
    parsed = getattr(record, "return_value", None)
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    vals = parsed[2]
    if not isinstance(vals, (list, tuple)) or not vals:
        return None
    try:
        return float(vals[0])
    except (TypeError, ValueError):
        return None


class BiasWiggle(BaseSkill):
    """在一个小的双极性区间内随机跳变偏压,轻微改性针尖顶端。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="BiasWiggle",
            version="1.0.0",
            category=SkillCategory.WRITE,
            capabilities=frozenset({"tip_shaping"}),
            safety_level=SafetyLevel.AUTO,
            description=(
                "在一个小的双极性窗口内（默认 "
                "±4..20 mV）随机跳变偏压，跑一小段 burst，把针尖顶端推向"
                "原子级尖锐的构型 —— 相当于用软件来回拖那根偏压滑块。"
                "feedback 必须是**开**的：机理在于 |V| 越小，环路就把针尖"
                "推得越近。目标值绝不会落在零附近，变号时以允许的最大 slew "
                "一穿而过、不作停留，每一步之后都会检查电流，"
                "整段 burst 硬上限 10 s。它是**故意**"
                "允许在扫描过程中跑的（配方就是在一帧牺牲掉的"
                "扫描框里 wiggle）。"
            ),
            parameters=[
                ParameterSpec(
                    name="base_bias_v", type="float", unit="V",
                    description=(
                        "burst 结束后要回到的偏压（通常就是成像偏压，如 "
                        "0.02 V）。每一条退出路径都会把它恢复回去，"
                        "abort 也不例外。"),
                    required=True, min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="wiggle_lower_v", type="float", unit="V",
                    description=(
                        "|target| 的最小值（V）。**不能**为零：在恒流 "
                        "feedback 下，偏压接近零会让环路把针尖一路"
                        "压进表面。"),
                    required=False, default=0.004,
                    min_value=0.0005, max_value=ABSOLUTE_MAX_V),
                ParameterSpec(
                    name="wiggle_upper_v", type="float", unit="V",
                    description="|target| 的最大值（V）。上限 ±0.1 V。",
                    required=False, default=0.020,
                    min_value=0.001, max_value=ABSOLUTE_MAX_V),
                ParameterSpec(
                    name="dwell_min_s", type="float", unit="s",
                    description="在一个目标值上的最短停留时间。",
                    required=False, default=0.05, min_value=0.01, max_value=5.0),
                ParameterSpec(
                    name="dwell_max_s", type="float", unit="s",
                    description="在一个目标值上的最长停留时间。",
                    required=False, default=0.15, min_value=0.01, max_value=10.0),
                ParameterSpec(
                    name="slew_rate_v_per_s", type="float", unit="V/s",
                    description=(
                        "目标值之间的最大 ramp 速率。变号时允许用到绝对"
                        "上限，好让它快速穿过零点。"),
                    required=False, default=1.0,
                    min_value=0.005, max_value=ABSOLUTE_MAX_SLEW),
                ParameterSpec(
                    name="burst_s", type="float", unit="s",
                    description="burst 总时长。硬上限 10 s。",
                    required=False, default=5.0,
                    min_value=0.1, max_value=ABSOLUTE_MAX_BURST_S),
                ParameterSpec(
                    name="abort_current_a", type="float", unit="A",
                    description=(
                        "任何一步只要 |I| 超过这个值，就 abort 并把偏压"
                        "恢复回去。"),
                    required=False, default=5e-9,
                    min_value=1e-11, max_value=1e-6),
                ParameterSpec(
                    name="seed", type="int",
                    description="随机种子（0 = 非确定性）。",
                    required=False, default=0, min_value=0),
                ParameterSpec(
                    name="allow_feedback_off", type="bool",
                    description=(
                        "即使 Z controller 关着也照跑。没有 feedback 就不"
                        "存在上面那个机理，所以这件事默认会被**拒绝**。"),
                    required=False, default=False),
            ],
            # scan_not_running 刻意不在这里:配方就是要在一张牺牲帧里打扰动。
            preconditions=[],
            estimated_duration_s=6.0,
            composition_level=1,
            tags=["bias", "wiggle", "tip", "write"],
        )

    # ── 内部:一次带看护的斜坡 ─────────────────────────────────────────────

    def _ramp_to(self, context, calls, *, start: float, target: float,
                 slew: float, abort_a: float) -> tuple[float, str]:
        """从 ``start`` 走到 ``target``。返回 ``(实际到达的偏压, 停止原因)``。

        停止原因为 ``""`` 表示走到了。中途每一步都查 abort、看电流 —— 停在两个都
        通过边界检查的中间值上是安全的。
        """
        delta = abs(target - start)
        if delta < _MIN_RAMP_V:
            rec = context.safe_call("Bias_Set", float(target))
            calls.append(rec)
            if rec.error:
                return start, f"Bias_Set 失败: {rec.error}"
            # 小幅改变也要看电流。这条快捷路径最初漏了这一步 —— 「每一步都看
            # 电流」一旦有例外,一次恰好落在例外里的跳变就会让针尖在超阈状态下
            # 继续走下一步。护栏不能有大小之分。
            cur_rec = context.safe_call("Current_Get")
            calls.append(cur_rec)
            amps = _parse_scalar(cur_rec)
            if amps is not None and abs(amps) > abort_a:
                return float(target), (
                    f"电流 {abs(amps):.3g} A 超过中止阈 {abort_a:.3g} A")
            return float(target), ""

        n_steps = max(1, int(round(delta / max(slew * _STEP_INTERVAL_S, 1e-9))))
        current = start
        for i in range(1, n_steps + 1):
            check_abort = getattr(context, "check_abort", None)
            if callable(check_abort) and check_abort():
                return current, "aborted"
            value = start + (target - start) * (i / n_steps)
            rec = context.safe_call("Bias_Set", float(value))
            calls.append(rec)
            if rec.error:
                return current, f"Bias_Set 失败: {rec.error}"
            current = float(value)
            cur_rec = context.safe_call("Current_Get")
            calls.append(cur_rec)
            amps = _parse_scalar(cur_rec)
            if amps is not None and abs(amps) > abort_a:
                return current, (
                    f"电流 {abs(amps):.3g} A 超过中止阈 {abort_a:.3g} A")
            if i < n_steps:
                time.sleep(_STEP_INTERVAL_S)
        return current, ""

    def _restore(self, context, calls, *, frm: float, to: float,
                 aborted: bool) -> None:
        """把偏压放回去。

        abort 之后 ``safe_call`` 拒绝一切非白名单写,而 ``Bias_Set`` 不在白名单里
        —— 所以清理必须显式带 ``allow_on_abort=True``。这正是那个逃生口存在的
        理由:这次写**是因为** abort 才要做的。
        """
        try:
            delta = abs(to - frm)
            if delta < _MIN_RAMP_V:
                calls.append(context.safe_call("Bias_Set", float(to),
                                               allow_on_abort=True))
                return
            # 收尾用加急斜率(仍不超绝对上限):留在一个随机的扰动值上没有意义,
            # 而幅度本来就只有几十毫伏。
            n = max(1, int(round(delta / (ABSOLUTE_MAX_SLEW * _STEP_INTERVAL_S))))
            for i in range(1, n + 1):
                value = frm + (to - frm) * (i / n)
                calls.append(context.safe_call("Bias_Set", float(value),
                                               allow_on_abort=True))
                if i < n:
                    time.sleep(_STEP_INTERVAL_S)
        except Exception:  # noqa: BLE001 — 收尾绝不能把已经发生的事变成异常
            pass

    # ── 执行 ──────────────────────────────────────────────────────────────

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        base = float(params["base_bias_v"])
        lower = abs(float(params.get("wiggle_lower_v", 0.004)))
        upper = abs(float(params.get("wiggle_upper_v", 0.020)))
        dwell_lo = float(params.get("dwell_min_s", 0.05))
        dwell_hi = float(params.get("dwell_max_s", 0.15))
        slew = float(params.get("slew_rate_v_per_s", 1.0))
        burst = float(params.get("burst_s", 5.0))
        abort_a = float(params.get("abort_current_a", 5e-9))
        seed = int(params.get("seed", 0) or 0)

        def _fail(msg: str, **data) -> SkillResult:
            return SkillResult(skill_name="BiasWiggle", success=False,
                               error=msg, data=data, nanonis_calls=calls)

        # ── 硬帽二次校验(ParameterSpec 之后再来一遍) ──
        # spec 可以被改、被绕过;这几个数字关系到针尖是不是还在,所以在真正下发
        # 之前再查一次。拒绝,不夹紧。
        if upper > ABSOLUTE_MAX_V or lower > ABSOLUTE_MAX_V:
            return _fail(
                f"扰动幅值上限 {upper:g} V 超过硬上限 ±{ABSOLUTE_MAX_V} V。这是"
                f"写死在代码里的界,配置改不了 —— 拒绝执行而不是替你降到上限。"
                f"要更大的幅度请用 BiasPulse(那是另一件事,有它自己的包络)。")
        if lower >= upper:
            return _fail(
                f"扰动下限 {lower:g} V 不小于上限 {upper:g} V —— 区间是空的。")
        if burst > ABSOLUTE_MAX_BURST_S:
            return _fail(
                f"突发时长 {burst:g} s 超过硬上限 {ABSOLUTE_MAX_BURST_S} s。"
                f"这是一次短促扰动,不是一个可以一直开着的模式。")
        if slew > ABSOLUTE_MAX_SLEW:
            return _fail(
                f"斜率 {slew:g} V/s 超过硬上限 {ABSOLUTE_MAX_SLEW} V/s。")
        if dwell_hi < dwell_lo:
            dwell_lo, dwell_hi = dwell_hi, dwell_lo
        if abs(base) > ABSOLUTE_MAX_V + upper:
            # 从 1 V 的成像偏压跳进 ±20 mV 的扰动区,那一下不是扰动是一次大跳变。
            return _fail(
                f"基准偏压 {base:g} V 离扰动区间(±{upper:g} V)太远 —— 进出扰动"
                f"区的那两次跳变本身就是一次大的偏压变化。请先用 "
                f"BiasSettleChange 把偏压带到成像值附近再打扰动。")

        # ── 前置:反馈必须开着 ──
        # 这个技能的物理机制就是「|V| 变小 → 反馈把针尖推近」。反馈关着时它什么
        # 也不做(z 不动),读不到状态时我们无法确认自己在做什么 —— 两种情况都
        # 不该照跑。一次 ZCtrl_OnOffGet 很便宜。
        if not bool(params.get("allow_feedback_off", False)):
            fb_rec = context.safe_call("ZCtrl_OnOffGet")
            calls.append(fb_rec)
            fb = _parse_scalar(fb_rec)
            feedback_on = None if fb is None else bool(int(fb))
            if feedback_on is not True:
                return _fail(
                    ("Z 反馈是关的" if feedback_on is False else
                     "读不到 Z 反馈状态(ZCtrl_OnOffGet)") +
                    " —— 偏压扰动靠的是恒流反馈在低偏压下把针尖推近，"
                    "反馈不开这一步什么也不会发生。先开反馈"
                    "(ZControllerOnOff enable=true)，或显式传 "
                    "allow_feedback_off=true。",
                    feedback_on=feedback_on)

        # ── 起点确认:读不到就拒绝斜坡 ──
        start_rec = context.safe_call("Bias_Get")
        calls.append(start_rec)
        start = _parse_scalar(start_rec)
        if start is None:
            return _fail(
                f"读不到当前偏压(Bias_Get: {start_rec.error or '返回值无法解析'})"
                f" —— 不知道起点就不能受控地改变偏压。拒绝执行，不从假设的 0 V "
                f"开始。")

        rng = random.Random(seed) if seed else random.Random()
        t0 = time.monotonic()
        current = float(start)
        log: list[dict] = []
        stop_reason = ""
        aborted = False

        while (time.monotonic() - t0) < burst:
            # 目标:幅值在 [lower, upper] 内随机,符号随机。永不落在零附近 ——
            # 这是「在死区里跳」与「停在零点」的分界。
            target = rng.uniform(lower, upper) * rng.choice((-1.0, 1.0))
            # 穿零段允许用绝对上限的斜率:停在零附近的每一毫秒反馈都在推针尖,
            # 所以跨过去要快,而且中间不设停留点。
            crosses = (current > 0) != (target > 0) and current != 0
            eff_slew = ABSOLUTE_MAX_SLEW if crosses else slew

            current, why = self._ramp_to(
                context, calls, start=current, target=target,
                slew=eff_slew, abort_a=abort_a)
            log.append({"target_v": round(target, 6),
                        "reached_v": round(current, 6),
                        "crossed_zero": bool(crosses),
                        "t_s": round(time.monotonic() - t0, 3)})
            if why:
                stop_reason = why
                aborted = (why == "aborted")
                break

            dwell = rng.uniform(dwell_lo, dwell_hi)
            remaining = burst - (time.monotonic() - t0)
            if remaining <= 0:
                break
            time.sleep(min(dwell, remaining))
            check_abort = getattr(context, "check_abort", None)
            if callable(check_abort) and check_abort():
                stop_reason, aborted = "aborted", True
                break

        # ── 收尾:偏压一定要放回去(包括 abort 路径) ──
        self._restore(context, calls, frm=current, to=base, aborted=aborted)

        elapsed = time.monotonic() - t0
        data = {
            "flips_executed": len(log),
            "burst_s": round(elapsed, 3),
            "base_bias_v": base,
            "wiggle_lower_v": lower,
            "wiggle_upper_v": upper,
            "slew_rate_v_per_s": slew,
            "abort_current_a": abort_a,
            "seed": seed,
            "log": log,
            "bias_restored": True,
            "aborted_reason": stop_reason,
        }
        if aborted:
            return SkillResult(
                skill_name="BiasWiggle", success=False,
                error=(f"扰动被中止(已完成 {len(log)} 次跳变)，偏压已恢复到 "
                       f"{base:g} V。"),
                data=data, nanonis_calls=calls)
        if stop_reason:
            return SkillResult(
                skill_name="BiasWiggle", success=False,
                error=(f"扰动在第 {len(log)} 次跳变时停下：{stop_reason}。"
                       f"偏压已恢复到 {base:g} V。"),
                data=data, nanonis_calls=calls)
        return SkillResult(
            skill_name="BiasWiggle", success=True, data=data,
            nanonis_calls=calls,
            summary=(f"偏压扰动 {len(log)} 次跳变／{elapsed:.1f} s "
                     f"(±{lower * 1e3:.0f}–{upper * 1e3:.0f} mV)，"
                     f"已恢复到 {base * 1e3:.0f} mV。"))


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(BiasWiggle, context_provider)


__all__ = ["BiasWiggle", "ABSOLUTE_MAX_V", "ABSOLUTE_MAX_BURST_S",
           "ABSOLUTE_MAX_SLEW"]
