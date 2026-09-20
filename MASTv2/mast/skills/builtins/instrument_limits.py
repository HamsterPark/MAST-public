"""The safety bounds themselves: Z limits, piezo limits, withdraw rate, SafeTip.

MAST's entire safety argument is "Nanonis bounds it" — the bias range, the piezo
range, the Z limits, the SafeTip threshold. That argument is why TipShape, a skill
that deliberately drives the tip into the surface, is merely AUTO.

A 2026-07-13 census found MAST could **read** every one of those bounds and **set**
none of them:

    ZCtrl_LimitsGet        ✓      ZCtrl_LimitsSet        ✗
    ZCtrl_WithdrawRateGet  ✓      ZCtrl_WithdrawRateSet  ✗
    Piezo_XYZLimitsGet     ✓      Piezo_XYZLimitsSet     ✗
    SafeTip_PropsGet       ✓      SafeTip_PropsSet       ✗
    ZCtrl_HomePropsGet     ✓      ZCtrl_Home             ✗   (can see home, can't go there)

So the agent could observe that the tip-protection threshold was wrong for this tip,
and not fix it; could see the Z limits were set for a different sample height, and
not correct them; could enable the Z limits (``ZCtrl_LimitsEnabledSet`` was wrapped)
without being able to say what they should be.

WIDENING A GUARDRAIL
====================
An agent that can widen its own limits has, in the strict sense, no limits.
The user outputs are the one exception: a Nanonis user output is a
small-signal line, and the wiring already takes care of that at the
physical layer.

— i.e. the physical layer is the real barrier and the software limit is a
convenience. That reasoning is sound for a user output and it is **not** sound for
the Z piezo: there is no wiring that stops a piezo from crashing a tip.

So these follow the SetUserOutputLimits pattern rather than blocking outright:
CONFIRM-gated, and **every change is recorded to the diagnostics ledger with the
before/after values and an explicit `widened` flag**. A widening is legal, visible,
and attributable. Silently is the only way it must not happen.
"""

from __future__ import annotations

import logging

from mast.core.diagnostics import record as diag_record
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """统一解包并返回回包 body，见 io.nanonis_files.decode_reply。

    不把 (error, raw_bytes, body) 信封直接当成读数交给调用方。
    """
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


def _floats(value, n: int) -> list[float] | None:
    """Pull n floats out of a Nanonis return value. None when we cannot."""
    out: list[float] = []
    stack = [value]
    seen = 0
    while stack and seen < 128:
        seen += 1
        v = stack.pop(0)
        if isinstance(v, (list, tuple)):
            stack = list(v) + stack
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out[:n] if len(out) >= n else None


class SetZLimits(BaseSkill):
    """The Z travel bounds. The last software barrier between a piezo and a tip."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZLimits",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定 Z 控制器的位置上限与下限，单位**米**。\n"
                "\n"
                "这两个界限规定了 Z 压电伸出与回缩都不得越过的边界 —— 是压电与针尖之间最后一道软件屏障。"
                "把它们放宽是合法的（更高的样品、更长的针尖确实需要更多行程），并且会连同改动前后的取值一起**记入**诊断账本。"
                "\n"
                "\n"
                "**限值不处于启用状态时，它们什么也不做**（Nanonis 原文：'When the Z position limits are not enabled, this function has no effect'）"
                "。请用 enable=true，或去核对 GetZControllerState.z_limits_enabled。"
                "\n"
                "\n"
                "取值单位是**米**：500 nm 的限值要写成 500n，不是 500。"
            ),
            parameters=[
                ParameterSpec(name="z_high_limit_m", type="float",
                              description="Z 的上界，单位**米**（500 nm = 500n）",
                              unit="m", required=True,
                              min_value=-1e-4, max_value=1e-4),
                ParameterSpec(name="z_low_limit_m", type="float",
                              description="Z 的下界，单位**米**",
                              unit="m", required=True,
                              min_value=-1e-4, max_value=1e-4),
                ParameterSpec(name="enable", type="bool",
                              description="同时**启用**这两个限值（不启用它们就没有任何作用）",
                              required=False, default=True),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["z", "limits", "safety", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetZLimits"
        hi = float(params["z_high_limit_m"])
        lo = float(params["z_low_limit_m"])
        if lo >= hi:
            return _fail(name,
                         f"z_low_limit_m ({lo:.3e} m) 必须小于 z_high_limit_m ({hi:.3e} m)"
                         "——限值反了", [])
        calls: list = []

        # Read the CURRENT limits first. A limits change we cannot describe is a
        # limits change we cannot audit: "widened from what?" has to have an answer.
        cur = context.safe_call("ZCtrl_LimitsGet")
        calls.append(cur)
        before = None if cur.error else _floats(_rv(cur), 2)
        if before is None:
            # Not fatal — but say so. Do NOT silently proceed as if before == after.
            logger.warning("SetZLimits: 无法读取当前 Z 限值（%s）", cur.error or "未能解析")

        rec = context.safe_call("ZCtrl_LimitsSet", hi, lo)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"ZCtrl_LimitsSet failed: {rec.error}", calls)

        widened = None
        if before is not None:
            b_hi, b_lo = before[0], before[1]
            widened = bool(hi > b_hi or lo < b_lo)

        want_enable = bool(params.get("enable", True))
        if want_enable:
            rec_en = context.safe_call("ZCtrl_LimitsEnabledSet", 1)
            calls.append(rec_en)
            if rec_en.error:
                return _fail(name,
                             f"限值已写入但**未能启用**（ZCtrl_LimitsEnabledSet 失败："
                             f"{rec_en.error}）——未启用的限值不起任何作用", calls)

        # enable=False is a legal request, and it used to produce a cheerful
        # success message for a write that does NOTHING. The docstring above
        # already quotes Nanonis ("has no effect" when the limits are not
        # enabled); saying it in prose and not in the RESULT is how a caller ends
        # up believing a narrowed limit is protecting them. Read the flag back —
        # do not infer it — and let the summary carry the answer (KNOWN_ISSUES
        # §1.3). Deliberately NOT auto-enabling: the caller said enable=False,
        # and quietly doing the opposite of an explicit argument is its own bug.
        enabled_now: bool | None = True if want_enable else None
        if not want_enable:
            rec_en_get = context.safe_call("ZCtrl_LimitsEnabledGet")
            calls.append(rec_en_get)
            if not rec_en_get.error:
                flags = _floats(_rv(rec_en_get), 1)
                enabled_now = bool(flags[0]) if flags else None

        # Read back. The whole point of this file.
        after = context.safe_call("ZCtrl_LimitsGet")
        calls.append(after)
        actual = None if after.error else _floats(_rv(after), 2)

        diag_record(
            "note", subject="SetZLimits",
            reason="Z 位置限值被修改" + ("（放宽）" if widened else ""),
            before=list(before) if before else None,
            requested=[hi, lo], after=list(actual) if actual else None,
            widened=widened, enabled=want_enable, enabled_now=enabled_now,
        )

        if actual is not None and (abs(actual[0] - hi) > 1e-12 or abs(actual[1] - lo) > 1e-12):
            return SkillResult(
                skill_name=name, success=False,
                error=(f"Z 限值未按要求生效：要求 [{lo:.3e}, {hi:.3e}] m，"
                       f"读回 [{actual[1]:.3e}, {actual[0]:.3e}] m"),
                data={"requested": [lo, hi], "actual": [actual[1], actual[0]]},
                nanonis_calls=calls)

        if enabled_now is False:
            inert = ("；⚠ **限值当前未启用（z_limits_enabled = 0），这对数字不起任何"
                     "作用** —— 要让它生效请以 enable=true 重新调用")
        elif enabled_now is None and not want_enable:
            inert = "；⚠ 未能读回启用状态，无法确认这对限值是否真的在生效"
        else:
            inert = ""

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"z_low_limit_m": lo, "z_high_limit_m": hi,
                  "before": list(before) if before else None,
                  "widened": widened, "verified": actual is not None,
                  "enabled": enabled_now},
            summary=(f"Z 限值 = [{lo:.3e}, {hi:.3e}] m"
                     + ("（**已放宽**，已记入诊断台账）" if widened else "")
                     + ("" if actual is not None else "（未能读回确认）")
                     + inert),
        )


class SetWithdrawRate(BaseSkill):
    """How fast Withdraw retracts."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetWithdrawRate",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定 Z 的退针速率，单位米每秒 —— 也就是 Withdraw（或紧急退针、或 SafeTip）"
                "触发时针尖回缩得有多快。\n"
                "\n"
                "这在两个方向上都是一个安全参数。太**慢**，紧急退针来不及把针尖撤离；太**快**，退针本身就可能把针尖抖松、"
                "或者把扫描器激起振铃。Nanonis 的默认值是个合理的起点；要改就得有理由。"
            ),
            parameters=[
                ParameterSpec(name="rate_m_per_s", type="float",
                              description="退针速率，单位 m/s",
                              unit="m/s", required=True,
                              min_value=1e-9, max_value=1e-2),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["z", "withdraw", "safety", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetWithdrawRate"
        rate = float(params["rate_m_per_s"])
        calls: list = []

        cur = context.safe_call("ZCtrl_WithdrawRateGet")
        calls.append(cur)
        before = None if cur.error else _floats(_rv(cur), 1)

        rec = context.safe_call("ZCtrl_WithdrawRateSet", rate)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"ZCtrl_WithdrawRateSet failed: {rec.error}", calls)

        after = context.safe_call("ZCtrl_WithdrawRateGet")
        calls.append(after)
        actual = None if after.error else _floats(_rv(after), 1)

        diag_record("note", subject="SetWithdrawRate", reason="退针速率被修改",
                    before=before[0] if before else None, requested=rate,
                    after=actual[0] if actual else None)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"rate_m_per_s": actual[0] if actual else rate,
                  "before": before[0] if before else None,
                  "verified": actual is not None},
            summary=f"退针速率 = {rate:.3e} m/s",
        )


class HomeZController(BaseSkill):
    """Send Z to its configured home position."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="HomeZController",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把 Z 移到它配置好的 **HOME** 位置。\n"
                "\n"
                "Home 是用户设定的一个安全／中性的停放点 —— 调用它之前先用 GetZControllerState.home 读一下，"
                "因为这个位置不是 MAST 选的，而一个为另一块样品配的 home 就只是个普通的 Z 位置而已。"
                "\n"
                "\n"
                "这会**移动**压电。若你想让针尖确定无疑地离开表面，请用 Withdraw，不要用 Home —— 退针去的是安全的极端位置，"
                "而 home 去的是 home 碰巧在的地方。"
            ),
            parameters=[],
            estimated_duration_s=3.0,
            composition_level=0,
            tags=["z", "home", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        home = context.safe_call("ZCtrl_HomePropsGet")
        calls.append(home)
        rec = context.safe_call("ZCtrl_Home")
        calls.append(rec)
        if rec.error:
            return _fail("HomeZController", f"ZCtrl_Home failed: {rec.error}", calls)
        pos = context.safe_call("ZCtrl_ZPosGet")
        calls.append(pos)
        z = _floats(_rv(pos), 1) if not pos.error else None
        return SkillResult(
            skill_name="HomeZController", success=True, nanonis_calls=calls,
            data={"home_props": None if home.error else _rv(home),
                  "z_m": z[0] if z else None},
            summary=(f"Z 已回到 home（当前 Z = {z[0]:.3e} m）" if z else "Z 已回到 home"),
        )


class SetActiveZController(BaseSkill):
    """Which Z controller is the active one (multi-controller rigs)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetActiveZController",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "选定哪一个 Z 控制器处于**激活**状态。只有在装了不止一个的机器上才有意义 —— 先调 GetZControllerState / ListZControllers。"
                "\n"
                "\n"
                "切换激活的控制器，会改变其余每一个 Z 技能所对话的是哪一个环。弄错了，就意味着你在给一个并没有托着针尖的环设设定值。"
            ),
            parameters=[
                ParameterSpec(name="controller_index", type="int",
                              description="Z 控制器序号（取自控制器列表）",
                              required=True, min_value=0, max_value=15),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["z", "controller", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["controller_index"])
        rec = context.safe_call("ZCtrl_ActiveCtrlSet", idx)
        if rec.error:
            return _fail("SetActiveZController", f"ZCtrl_ActiveCtrlSet failed: {rec.error}", [rec])
        diag_record("note", subject="SetActiveZController",
                    reason="活动 Z 控制器被切换", controller_index=idx)
        return SkillResult(skill_name="SetActiveZController", success=True,
                           nanonis_calls=[rec], data={"controller_index": idx},
                           summary=f"活动 Z 控制器 = {idx}")


class SetPiezoLimits(BaseSkill):
    """The XYZ piezo voltage limits."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPiezoLimits",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定压电的 X/Y/Z **电压**限值（并启用它们）。\n"
                "\n"
                "这些限值以伏特为单位框住扫描器能被驱动到多远，是在量程标定把它们换算成米之前起作用的。它们保护的既是针尖，"
                "也同样是压电本身（过压会让它**永久**退极化）。\n"
                "\n"
                "把它们放宽会连同改动前后的取值一起记入诊断账本。当前值用 GetPiezoConfig 读 —— 并注意这些是**伏特**，"
                "不是米；把两者连起来的是量程标定。"
            ),
            parameters=[
                ParameterSpec(name="x_low_v", type="float", description="X 下限（V）",
                              unit="V", required=True, min_value=-300.0, max_value=300.0),
                ParameterSpec(name="x_high_v", type="float", description="X 上限（V）",
                              unit="V", required=True, min_value=-300.0, max_value=300.0),
                ParameterSpec(name="y_low_v", type="float", description="Y 下限（V）",
                              unit="V", required=True, min_value=-300.0, max_value=300.0),
                ParameterSpec(name="y_high_v", type="float", description="Y 上限（V）",
                              unit="V", required=True, min_value=-300.0, max_value=300.0),
                ParameterSpec(name="z_low_v", type="float", description="Z 下限（V）",
                              unit="V", required=True, min_value=-300.0, max_value=300.0),
                ParameterSpec(name="z_high_v", type="float", description="Z 上限（V）",
                              unit="V", required=True, min_value=-300.0, max_value=300.0),
                ParameterSpec(name="enable", type="bool",
                              description="启用这些限值（不启用它们就没有任何作用）",
                              required=False, default=True),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["piezo", "limits", "safety", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetPiezoLimits"
        axes = {}
        for ax in ("x", "y", "z"):
            lo = float(params[f"{ax}_low_v"])
            hi = float(params[f"{ax}_high_v"])
            if lo >= hi:
                return _fail(name, f"{ax}_low_v ({lo:g} V) 必须小于 {ax}_high_v ({hi:g} V)", [])
            axes[ax] = (lo, hi)
        calls: list = []

        cur = context.safe_call("Piezo_XYZLimitsGet")
        calls.append(cur)
        before = None if cur.error else _floats(_rv(cur), 6)

        # Piezo_XYZLimitsSet(Enable_limits, X_low, X_high, Y_low, Y_high, Z_low, Z_high)
        rec = context.safe_call(
            "Piezo_XYZLimitsSet",
            1 if bool(params.get("enable", True)) else 0,
            axes["x"][0], axes["x"][1],
            axes["y"][0], axes["y"][1],
            axes["z"][0], axes["z"][1],
        )
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Piezo_XYZLimitsSet failed: {rec.error}", calls)

        after = context.safe_call("Piezo_XYZLimitsGet")
        calls.append(after)
        actual = None if after.error else _floats(_rv(after), 6)

        widened = None
        if before is not None:
            # before is [x_low, x_high, y_low, y_high, z_low, z_high] (Nanonis order).
            new = [axes["x"][0], axes["x"][1], axes["y"][0], axes["y"][1],
                   axes["z"][0], axes["z"][1]]
            widened = any(new[i] < before[i] for i in (0, 2, 4)) or \
                      any(new[i] > before[i] for i in (1, 3, 5))

        diag_record("note", subject="SetPiezoLimits",
                    reason="压电电压限值被修改" + ("（放宽）" if widened else ""),
                    before=before, after=actual, widened=widened,
                    enabled=bool(params.get("enable", True)))

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"limits_v": {k: list(v) for k, v in axes.items()},
                  "before": before, "widened": widened,
                  "verified": actual is not None},
            summary=("压电电压限值已设置"
                     + ("（**已放宽**，已记入诊断台账）" if widened else "")),
        )


class SetSafeTipProps(BaseSkill):
    """The tip-protection system's own configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSafeTipProps",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置 SafeTip —— 那套自动的针尖保护系统：它盯着一路信号，一旦越过阈值就退针。"
                "\n"
                "\n"
                "MAST 从前只能**读**这份配置而设不了它，这意味着 agent 明明看得出阈值对当前这根针尖是错的，"
                "却拿它没办法。\n"
                "\n"
                "**threshold 是要紧的那个数。** 定高了，SafeTip 永远不触发 —— 这份保护成了摆设。"
                "定低了，它会在噪声上触发，把好好的扫描中止掉。先用 GetSafeTipProps / GetSafeTipSignal 读一下当前值和被盯着的那路信号。"
                "\n"
                "\n"
                "auto_recovery 会在一次 SafeTip 事件之后恢复 Z 控制器，**前提是它原本就是开着的**；"
                "auto_pause_scan 则把扫描暂停下来，而不是让它在一个针尖刚被拉离的表面上继续磨下去。"
            ),
            parameters=[
                ParameterSpec(name="threshold", type="float",
                              description="触发阈值，用被盯着那路信号自己的单位",
                              required=True),
                ParameterSpec(name="auto_recovery", type="bool",
                              description="SafeTip 事件之后恢复 Z 控制器（前提是它原本开着）",
                              required=False, default=True),
                ParameterSpec(name="auto_pause_scan", type="bool",
                              description="发生 SafeTip 事件时暂停扫描",
                              required=False, default=True),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["safetip", "safety", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetSafeTipProps"
        calls: list = []

        cur = context.safe_call("SafeTip_PropsGet")
        calls.append(cur)
        before = None if cur.error else _rv(cur)

        thr = float(params["threshold"])
        rec = context.safe_call(
            "SafeTip_PropsSet",
            1 if bool(params.get("auto_recovery", True)) else 0,
            1 if bool(params.get("auto_pause_scan", True)) else 0,
            thr,
        )
        calls.append(rec)
        if rec.error:
            return _fail(name, f"SafeTip_PropsSet failed: {rec.error}", calls)

        after = context.safe_call("SafeTip_PropsGet")
        calls.append(after)
        actual = None if after.error else _rv(after)

        diag_record("note", subject="SetSafeTipProps",
                    reason="针尖保护(SafeTip)配置被修改", threshold=thr,
                    auto_recovery=bool(params.get("auto_recovery", True)),
                    auto_pause_scan=bool(params.get("auto_pause_scan", True)))

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"threshold": thr,
                  "auto_recovery": bool(params.get("auto_recovery", True)),
                  "auto_pause_scan": bool(params.get("auto_pause_scan", True)),
                  "before": before, "verified": actual is not None},
            summary=f"SafeTip 阈值 = {thr:g}",
        )
