"""Z controller skills: position, setpoint, on/off.

vendored from v1 mast/skills/builtins/zcontrol.py 2026-04-23

2026-07-13 — the Z on/off state is now READ BACK from the real-time controller
(``ZCtrl_OnOffGet``) instead of echoed back from the request. See
``mast.skills.verify``: Nanonis' own manual says the Z-Controller MODULE and the
real-time controller can disagree, and that you must ask the latter before starting
anything that depends on the loop being off. TryEngageController's
``needs_auto_approach=True`` is exactly such a thing — it authorises the open-loop
coarse stepper, which has no current-feedback stop.
"""

from __future__ import annotations

from mast.core.state import coerce_number, reply_scalar
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.verify import values_match, verify_z_controller
from mast.skills.verify import z_off_or_reason as _z_off_or_reason


def _setpoint_from(record) -> "float | None":
    """The setpoint out of a ``ZCtrl_SetpntGet`` record, or None if unreadable.

    None is deliberately distinct from any number: "could not read" must never be
    confusable with "reads zero" — a zero setpoint is itself one of the corrupted
    values this readback exists to catch.
    """
    if record is None or getattr(record, "error", None):
        return None
    parsed = getattr(record, "return_value", None)
    if (isinstance(parsed, (list, tuple)) and len(parsed) > 2
            and isinstance(parsed[2], (list, tuple)) and parsed[2]):
        try:
            return float(parsed[2][0])
        except (TypeError, ValueError):  # pragma: no cover — defensive
            return None
    try:
        return float(parsed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class GetZPosition(BaseSkill):
    """Read current Z position."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZPosition",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前的 Z piezo 位置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "position", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_ZPosGet")
        if record.error:
            return SkillResult(
                skill_name="GetZPosition",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        z_pos_m = reply_scalar(parsed, field="z_pos_m")
        # 缓存值必须是已解析的数值标量，不能把原始回包中的单元素元组写入 z_m。
        if z_pos_m is None:
            return SkillResult(
                skill_name="GetZPosition",
                success=False,
                error=("读到的 Z 位置不是一个数(回包解不出) —— 拒绝把它当成读数。"
                       "多半是 Nanonis 数值字段的元组包装没解开。"),
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GetZPosition",
            success=True,
            data={"z_pos_m": z_pos_m},
            nanonis_calls=[record],
        )


class SetSetpoint(BaseSkill):
    """Set tunneling current setpoint."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSetpoint",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z controller 的隧道电流 setpoint。",
            parameters=[
                ParameterSpec(
                    name="setpoint_a",
                    type="float",
                    description=(
                        "隧道电流 setpoint，是一个**安培量**（SI）。"
                        "STM 的 setpoint 极小：1 pA … 100 nA；常见值"
                        "是 100 pA。请写成**带 SI 前缀的字符串** —— "
                        "'50p' 表示 50 pA，'1n' 表示 1 nA，'100p' 表示 100 pA。"
                        "⚠️ 这里前缀**不是**可选的：指数写法和光秃秃的 "
                        "'1.5' 都会被拒绝。这是刻意的 —— 一个丢了数量级的裸数字"
                        "仍然是个合法数字，"
                        "于是 1.5（= 1.5 安培，约为真实 setpoint 的 ~1e10×）"
                        "就会一路畅通无阻。你若想说的是 1.5 nA，请写 '1.5n'。"
                    ),
                    unit="A",
                    required=True,
                    min_value=1e-12,   # 1 pA
                    max_value=100e-9,  # 100 nA
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["z", "setpoint", "write", "readback"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        setpoint_a = params["setpoint_a"]
        calls = []

        # Read first, for the same reason SetZCtrlGain does: this is what we put
        # back if the write does not land, and it is a reading rather than a
        # remembered claim. (2026-08-03 the setpoint was among the values that
        # arrived corrupted — SetSetpoint(0) — so it gets the same treatment.)
        prior_rec = context.safe_call("ZCtrl_SetpntGet")
        calls.append(prior_rec)
        prior = _setpoint_from(prior_rec)

        record = context.safe_call("ZCtrl_SetpntSet", setpoint_a)
        calls.append(record)
        if record.error:
            return SkillResult(
                skill_name="SetSetpoint",
                success=False,
                error=record.error,
                nanonis_calls=calls,
            )

        back_rec = context.safe_call("ZCtrl_SetpntGet")
        calls.append(back_rec)
        got = _setpoint_from(back_rec)

        if got is None:
            return SkillResult(
                skill_name="SetSetpoint",
                success=True,
                data={
                    "setpoint_a": setpoint_a,
                    "readback_ok": None,
                    "readback_unavailable": True,
                    "note": (
                        "设定点已写入,但回读失败,无法确认硬件真的接受了这个值。"
                        "进针前请在 Nanonis 面板上人工核对。"
                    ),
                },
                nanonis_calls=calls,
            )

        ok, detail = values_match(setpoint_a, got)
        if not ok:
            restored = False
            if prior is not None:
                r = context.safe_call("ZCtrl_SetpntSet", prior)
                calls.append(r)
                restored = not r.error
            return SkillResult(
                skill_name="SetSetpoint",
                success=False,
                error=(
                    f"写后回读不一致 —— 硬件里的设定点不是刚才请求的值: {detail}。"
                    + ("已还原为写入前的值。" if restored
                       else "无法还原为写入前的值。")
                    + "**不要进针**,先在 Nanonis 面板上人工核对设定点。"
                ),
                data={
                    "requested": setpoint_a,
                    "readback": got,
                    "prior": prior,
                    "restored": restored,
                    "readback_ok": False,
                },
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="SetSetpoint",
            success=True,
            data={"setpoint_a": setpoint_a, "readback": got, "readback_ok": True},
            nanonis_calls=calls,
        )


class GetSetpoint(BaseSkill):
    """Read the tunneling current setpoint (Z-controller).

    Added 2026-06-08 (F9): the read-skill set was asymmetric — GetBias /
    GetCurrent / GetZPosition existed but there was no getter for the feedback
    setpoint, so an agent could SET but not READ it (it could only infer it from
    the live-state block). Mirrors GetBias: READ / AUTO, ZCtrl_SetpntGet.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSetpoint",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前的隧道电流 setpoint（Z controller）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "setpoint", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_SetpntGet")
        if record.error:
            return SkillResult(
                skill_name="GetSetpoint",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list); parsed[2][0] is
        # the setpoint in amperes (mirrors GetBias's defensive extraction).
        parsed = record.return_value
        setpoint_a = (
            parsed[2][0]
            if isinstance(parsed, (list, tuple)) and len(parsed) > 2
            and isinstance(parsed[2], (list, tuple)) and parsed[2]
            else parsed
        )
        return SkillResult(
            skill_name="GetSetpoint",
            success=True,
            data={"setpoint_a": setpoint_a},
            nanonis_calls=[record],
        )


class ZControllerOnOff(BaseSkill):
    """Enable or disable Z controller."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ZControllerOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="启用或禁用 Z controller（feedback 环路）。",
            parameters=[
                ParameterSpec(
                    name="enable",
                    type="bool",
                    description="True 为启用，False 为禁用",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "controller", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        enable = bool(params["enable"])
        record = context.safe_call("ZCtrl_OnOffSet", int(enable))
        if record.error:
            return SkillResult(
                skill_name="ZControllerOnOff",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )

        # READ IT BACK. This used to return data={"z_controller_on": enable} —
        # echoing the request as though it were the state. A write that returns
        # without a TCP error has been *accepted*, not necessarily *applied*: the
        # real-time controller runs on its own clock and Nanonis' manual explicitly
        # warns about the delay. Downstream, `z_controller_on: False` is what makes
        # TryEngageController recommend a COARSE APPROACH — a claim that turns into
        # a tip crash if it was wrong.
        v = verify_z_controller(context, expect=enable)
        calls = [record, v["record"]]

        if v["verified"] and v["matches"] is False:
            # Say HOW LONG we waited and what budget the rig itself declared.
            #
            # the real question here:
            #     [ZControllerOnOff] failed: 要求 ON，实时控制器回报 OFF。
            #     这是不是一个误判？
            # Neither a human nor anyone reading the log could tell —
            # because the two numbers that answer it were computed by
            # verify_z_controller (`waited_s`, `switch_off_delay_s`) and then
            # dropped on the floor. Yet another "measured but never surfaced".
            #
            # With them present the two causes separate at a glance:
            #   waited ≈ the declared delay  → the write really is not landing
            #                                  (wiring / module / RT config)
            #   waited ≪ the declared delay  → we gave up too early, and the
            #                                  settle budget is what to raise.
            waited = v.get("waited_s")
            budget = v.get("switch_off_delay_s")
            timing = ""
            if isinstance(waited, (int, float)):
                timing = f" 已等待 {waited:.2f}s"
                if isinstance(budget, (int, float)) and budget > 0:
                    timing += f"（本机声明的 switch-off 延迟 {budget:.2f}s）"
                    if waited < budget * 0.9:
                        timing += ("——**等待时间短于机器自己声明的延迟**，"
                                   "这更像是判早了而不是写入没生效")
                    else:
                        timing += "——已等满声明的延迟，写入很可能确实没生效"
                else:
                    timing += "（本机未声明 switch-off 延迟）"
            return SkillResult(
                skill_name="ZControllerOnOff", success=False,
                error=(f"Z 反馈开关未生效：要求 {'ON' if enable else 'OFF'}，"
                       f"实时控制器回报 {'ON' if v['on'] else 'OFF'}。{timing}"),
                data={"z_controller_on": v["on"], "requested": enable,
                      "verified": True,
                      "waited_s": waited, "switch_off_delay_s": budget},
                nanonis_calls=calls,
            )

        if not v["verified"]:
            # The write went through; the readback did not. Say so instead of
            # silently upgrading "unknown" to "as requested".
            return SkillResult(
                skill_name="ZControllerOnOff", success=True,
                data={"z_controller_on": None, "requested": enable,
                      "verified": False, "verify_error": v["error"]},
                summary=(f"已下发 Z 反馈 {'ON' if enable else 'OFF'}，"
                         f"但无法读回确认（{v['error']}）——请勿据此认定它已生效。"),
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="ZControllerOnOff",
            success=True,
            data={"z_controller_on": v["on"], "requested": enable, "verified": True},
            nanonis_calls=calls,
        )


def _scalar(parsed):
    """Defensively pull the scalar out of a Nanonis (err, raw, parsed_list) tuple."""
    try:
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            inner = parsed[2]
            if isinstance(inner, (list, tuple)) and inner:
                return float(inner[0])
            return float(inner)
        return float(parsed)
    except (TypeError, ValueError, IndexError):
        return None


class TryEngageController(BaseSkill):
    """Try to establish tunneling by switching the Z-controller ON — WITHOUT a
    coarse motor auto-approach.

    Disambiguates a "进针 / engage the tip" request: if the tip is already within
    the fine-Z range of the surface, turning feedback on pulls into ~setpoint
    tunneling → engaged, no coarse approach needed. If the setpoint cannot be
    reached by feedback alone (tip too far / retracted), the controller is switched
    back OFF (so Z is not left fruitlessly maxed out) and the result flags
    ``needs_auto_approach=True`` — the caller should then run ``AutoApproach``.

    Only moves the fine Z piezo (bounded ~µm range); never drives the coarse motor,
    so it cannot crash the tip the way a blind coarse approach could.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TryEngageController",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在**不做**粗动 auto-approach 的前提下，靠打开 Z-controller 尝试"
                "进入隧道状态。若隧道电流达到 ~setpoint，说明针已进上，feedback "
                "保持开启；否则把 controller 重新关掉，并在结果里置 "
                "needs_auto_approach=True。当收到含糊的「进针/engage」请求时，"
                "**先**用它，再考虑退回 AutoApproach。"
            ),
            parameters=[
                ParameterSpec(
                    name="settle_s", type="float",
                    description="一边轮询电流、一边让 feedback 稳定下来的秒数。",
                    required=False, default=1.5, min_value=0.1, max_value=30.0),
                ParameterSpec(
                    name="poll_hz", type="int",
                    description="稳定期间每秒轮询电流的次数。",
                    required=False, default=10, min_value=1, max_value=100),
                ParameterSpec(
                    name="engage_fraction", type="float",
                    description="|current| 必须达到 setpoint 的这个比例，才算进上针。",
                    required=False, default=0.5, min_value=0.05, max_value=1.0),
            ],
            estimated_duration_s=2.5,
            composition_level=1,
            tags=["z", "controller", "approach", "engage", "tip"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import time

        settle_s = float(params.get("settle_s", 1.5))
        poll_hz = int(params.get("poll_hz", 10))
        frac = float(params.get("engage_fraction", 0.5))
        calls = []

        rec_sp = context.safe_call("ZCtrl_SetpntGet")
        calls.append(rec_sp)
        if rec_sp.error:
            return SkillResult(skill_name="TryEngageController", success=False,
                               error=f"could not read setpoint: {rec_sp.error}",
                               nanonis_calls=calls)
        sp = _scalar(rec_sp.return_value)
        setpoint_a = abs(sp) if sp is not None else 0.0

        rec_on = context.safe_call("ZCtrl_OnOffSet", 1)
        calls.append(rec_on)
        if rec_on.error:
            return SkillResult(skill_name="TryEngageController", success=False,
                               error=f"could not turn on Z-controller: {rec_on.error}",
                               nanonis_calls=calls)

        peak = 0.0
        n_valid = 0  # how many current reads actually parsed
        n = max(1, int(settle_s * poll_hz))
        dt = settle_s / n
        _abort = getattr(context, "check_abort", None)
        for _ in range(n):
            # The settle loop only READS, so an abort here is not a hardware
            # hazard (the post-abort gate lets reads through by design) — but a
            # run that has been stopped should not keep the operator waiting out
            # the full settle window before it says so.
            if callable(_abort) and _abort():
                return SkillResult(
                    skill_name="TryEngageController", success=False,
                    error=("aborted by operator while waiting for the feedback to "
                           "settle — engagement is UNVERIFIED. Do not retry, and "
                           "do not assume the tip is engaged."),
                    data={"engaged": False, "aborted": True},
                    nanonis_calls=calls,
                )
            rec_c = context.safe_call("Current_Get")
            calls.append(rec_c)
            if not rec_c.error:
                cur = _scalar(rec_c.return_value)
                if cur is not None:
                    n_valid += 1
                    peak = max(peak, abs(cur))
            time.sleep(dt)

        threshold = frac * setpoint_a

        # FAIL-SAFE (2026-07-03 review): recommending a coarse AutoApproach is
        # ONLY safe when the current-feedback chain works — that feedback is what
        # STOPS the coarse stepper before a crash. If the current measurement was
        # dead (every Current_Get failed/unparsed) OR the setpoint is non-positive
        # (unreadable / genuinely 0), we CANNOT judge tunnelling, so we must NOT
        # set needs_auto_approach — otherwise the very failure of the measurement
        # chain would trigger a blind coarse approach. Turn the controller back
        # OFF and report a diagnostic (needs_auto_approach=False → ApproachTip
        # stops rather than approaching).
        if n_valid == 0 or setpoint_a <= 0:
            rec_off = context.safe_call("ZCtrl_OnOffSet", 0)
            calls.append(rec_off)
            # Verified against the REAL-TIME controller, not echoed back from the
            # request. v["on"] is None when we could not determine it — and None is
            # NOT False. See mast.skills.verify.
            v = verify_z_controller(context, expect=False)
            calls.append(v["record"])
            reason = ("current measurement chain returned no valid reading"
                      if n_valid == 0 else
                      "setpoint is non-positive / unreadable")
            return SkillResult(
                skill_name="TryEngageController", success=False,
                error=(f"cannot assess tunnelling ({reason}) — refusing to flag a "
                       f"coarse approach on a broken feedback chain; check the "
                       f"preamp / current range / setpoint."),
                data={"engaged": False, "z_controller_on": v["on"],
                      "z_controller_verified": v["verified"],
                      "needs_auto_approach": False, "peak_current_a": peak,
                      "setpoint_a": setpoint_a, "valid_current_reads": n_valid},
                nanonis_calls=calls)

        engaged = peak >= threshold

        if engaged:
            return SkillResult(
                skill_name="TryEngageController", success=True,
                data={"engaged": True, "z_controller_on": True,
                      "needs_auto_approach": False, "peak_current_a": peak,
                      "setpoint_a": setpoint_a,
                      "message": "Tunneling established — Z-controller ON, tip engaged."},
                nanonis_calls=calls)

        # THE decision point. Returning needs_auto_approach=True tells the agent to
        # run a COARSE MOTOR APPROACH — the open-loop stepper that has no
        # current-feedback stop and is the one action in this system that reliably
        # destroys a tip. It is only safe with the Z feedback OPEN.
        #
        # This used to switch the loop off and assert "z_controller_on": False in the
        # same breath, without checking rec_off.error and without ever reading the
        # hardware. If that write had failed — or merely not landed yet; Nanonis'
        # manual warns the module and the real-time controller disagree during the
        # communication delay — the agent would drive the coarse stepper into the
        # surface with the feedback still closed.
        rec_off = context.safe_call("ZCtrl_OnOffSet", 0)
        calls.append(rec_off)
        v = verify_z_controller(context, expect=False)
        calls.append(v["record"])

        if not (v["verified"] and v["on"] is False):
            # FAIL CLOSED. Cannot confirm the loop is open ⇒ do NOT flag the coarse
            # approach. The failure of the check must never be what authorises the
            # dangerous action.
            #
            # Report what we ACTUALLY know: True when the RT controller says the loop
            # is still closed (we know, and it is bad), None when the read failed (we
            # do not know). Collapsing both into None would throw away the one that
            # tells the operator their OFF write is not landing.
            why = (f"实时控制器回报 Z 反馈仍然闭合（Z-Controller 模块可能已显示 Off——"
                   "两者不是一回事，Nanonis 手册要求以实时控制器为准）"
                   if v["verified"] else
                   f"无法确认 Z 反馈是否已断开（ZCtrl_OnOffGet 读取失败：{v['error']}）")
            return SkillResult(
                skill_name="TryEngageController", success=False,
                error=(f"未能建立隧穿（峰值 |I|={peak:.2e} A < {threshold:.2e} A），"
                       f"但也**不能**建议粗进针：{why}。"
                       "反馈环还在的时候跑开环粗进针马达就是撞针。"),
                data={"engaged": False, "z_controller_on": v["on"],
                      "z_controller_verified": v["verified"],
                      "needs_auto_approach": False,   # ← the fail-closed bit
                      "peak_current_a": peak, "setpoint_a": setpoint_a},
                nanonis_calls=calls)

        return SkillResult(
            skill_name="TryEngageController", success=True,
            data={"engaged": False, "z_controller_on": False,
                  "z_controller_verified": True,
                  "needs_auto_approach": True, "peak_current_a": peak,
                  "setpoint_a": setpoint_a,
                  "message": (
                      f"Could not reach setpoint by feedback alone (peak |I|={peak:.2e} A "
                      f"< {threshold:.2e} A). Z-controller confirmed OFF against the "
                      f"real-time controller. Run AutoApproach (coarse motor) to "
                      f"engage the tip.")},
            nanonis_calls=calls)


class SetZPosition(BaseSkill):
    """Set Z position directly (Z controller must be OFF)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZPosition",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="直接设置 Z piezo 位置。Z controller 必须处于 OFF。",
            parameters=[
                ParameterSpec(
                    name="z_pos_m",
                    type="float",
                    description="Z 位置，单位米",
                    unit="m",
                    required=True,
                    # ±10 µm 包络:这是设 **绝对** Z 的唯一入口,此前没有任何
                    # ParameterSpec 边界(SetTipLift/SetHomeProps 都有 ±1 µm)。
                    # 真实压电量程约 1.5 µm,留一个数量级余量给非常规标定,
                    # 挡住的是数量级写错(米当纳米)这类输入。
                    min_value=-1e-5,
                    max_value=1e-5,
                ),
            ],
            preconditions=["z_controller_off"],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "position", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        z_pos_m = params["z_pos_m"]
        record = context.safe_call("ZCtrl_ZPosSet", z_pos_m)
        if record.error:
            return SkillResult(
                skill_name="SetZPosition",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetZPosition",
            success=True,
            data={"z_pos_m": z_pos_m},
            nanonis_calls=[record],
        )


class SetTipLift(BaseSkill):
    """Set tip lift amount (retraction when Z controller turns off)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetTipLift",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z controller 关闭时针尖回撤的量。",
            parameters=[
                ParameterSpec(
                    name="tip_lift_m",
                    type="float",
                    description="针尖抬起量，单位米",
                    unit="m",
                    required=True,
                    # Fine-Z config: ±1 µm hard cap (mirrors TipShape's tip_lift_m
                    # bound in tip_shaper.py). Without this, an LLM passing e.g.
                    # tip_lift_m=5 (= 5 metres!) reaches the hardware unchecked.
                    min_value=-1e-6,
                    max_value=1e-6,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "tip_lift", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        tip_lift_m = params["tip_lift_m"]
        record = context.safe_call("ZCtrl_TipLiftSet", tip_lift_m)
        if record.error:
            return SkillResult(
                skill_name="SetTipLift",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetTipLift",
            success=True,
            data={"tip_lift_m": tip_lift_m},
            nanonis_calls=[record],
        )


class GetTipLift(BaseSkill):
    """Read current tip lift setting."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetTipLift",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前的针尖抬起量。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "tip_lift", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_TipLiftGet")
        if record.error:
            return SkillResult(
                skill_name="GetTipLift",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        tip_lift_m = reply_scalar(parsed, field="tip_lift_m")
        if tip_lift_m is None:
            return SkillResult(
                skill_name="GetTipLift",
                success=False,
                error=("读到的 tip lift 不是一个数(回包解不出) —— 拒绝把它当成读数。"
                       "扎针深度用错数量级的后果落在针尖上,宁可失败。"),
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GetTipLift",
            success=True,
            data={"tip_lift_m": tip_lift_m},
            nanonis_calls=[record],
        )


class SetZLimitsEnabled(BaseSkill):
    """Enable or disable Z position limits."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZLimitsEnabled",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="启用或禁用 Z 位置的安全限位。",
            parameters=[
                ParameterSpec(
                    name="enabled",
                    type="bool",
                    description=(
                        "True 为启用 Z 限位，False 为禁用。默认为 "
                        "True：不带参数、走默认的调用会**启用** Z 软限位"
                        "（除非用户显式传 "
                        "enabled=False，否则它们应当保持开启）。"
                    ),
                    # Default ON (safety fix): a bare/defaulted call enables the
                    # Z soft-limits. Disabling requires an explicit enabled=False
                    # (the disable path is intentionally NOT gated — see audit #4).
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "limits", "safety", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # Default ON: a missing/None ``enabled`` ENABLES the Z limits (see metadata).
        enabled = params.get("enabled")
        if enabled is None:
            enabled = True
        record = context.safe_call("ZCtrl_LimitsEnabledSet", 1 if enabled else 0)
        if record.error:
            return SkillResult(
                skill_name="SetZLimitsEnabled",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetZLimitsEnabled",
            success=True,
            data={"enabled": enabled},
            nanonis_calls=[record],
        )


class GetZLimitsEnabled(BaseSkill):
    """Read whether Z position limits are enabled."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZLimitsEnabled",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 Z 位置安全限位是否启用。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "limits", "safety", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_LimitsEnabledGet")
        if record.error:
            return SkillResult(
                skill_name="GetZLimitsEnabled",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        raw = reply_scalar(parsed, field="z_limits_enabled")
        # ⚠️ 这里**不能**写 ``bool(raw)``:``reply_scalar`` 读不到时返回 ``None``,
        # 而 ``bool(None) is False`` —— 那就把「我没读到」报成了「限位是关的」。
        # 一个关于安全限位的假否定,比读失败危险得多。
        # (这次统一时差点原样保留 ``bool(raw)``,而那正是这一版在修的
        #  那类错的镜像:上一处是「读不到 ⇒ 出故障」,这里会是「读不到 ⇒ 没开」。)
        if raw is None:
            return SkillResult(
                skill_name="GetZLimitsEnabled",
                success=False,
                error=("读不到 Z 限位的启用状态(回包解不出一个数)—— "
                       "**不要**把它当成「限位已关闭」。"),
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GetZLimitsEnabled",
            success=True,
            data={"enabled": bool(raw)},
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# ZCtrl — CtrlListGet, HomePropsGet/Set, SwitchOffDelaySet, WithdrawRateGet
# ---------------------------------------------------------------------------


class GetZCtrlList(BaseSkill):
    """Get the list of available Z controllers and the active controller index."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZCtrlList",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 Z controller 列表，以及当前生效的 controller 索引。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "controller", "list", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_CtrlListGet")
        if record.error:
            return SkillResult(
                skill_name="GetZCtrlList",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is the (error_string, raw_bytes, Variables) triple.
        # ZCtrl.CtrlListGet ResponseTypes = ["i", "i", "*+c", "i"], so Variables
        # (== record.return_value[2]) is
        #   [list_size(int), num_controllers(int), controller_names(list[str]),
        #    active_index(int)].
        # The real data lives in Variables — the previous code iterated the
        # OUTER triple (['', b'...', [...]]) and never reached the names array,
        # so `controllers` was always [] and `active_index` always 0.
        parsed = record.return_value
        variables = (
            parsed[2]
            if isinstance(parsed, (list, tuple)) and len(parsed) > 2
            else parsed
        )
        controllers: list = []
        active_index = 0
        if isinstance(variables, (list, tuple)):
            for item in variables:
                if isinstance(item, (list, tuple)) and all(
                    isinstance(s, str) for s in item
                ):
                    controllers = list(item)
                elif isinstance(item, int) and controllers:
                    active_index = item
        return SkillResult(
            skill_name="GetZCtrlList",
            success=True,
            data={"controllers": controllers, "active_index": active_index},
            nanonis_calls=[record],
        )


class GetHomeProps(BaseSkill):
    """Get Z controller Home properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetHomeProps",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 Z controller 的 Home 位置模式与数值。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "home", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_HomePropsGet")
        if record.error:
            return SkillResult(
                skill_name="GetHomeProps",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is the (error_string, raw_bytes, Variables) triple.
        # ZCtrl.HomePropsGet ResponseTypes = ["H", "f"], so Variables
        # (== record.return_value[2]) is [rel_or_abs(int), home_pos_m(float)].
        # The previous code read parsed[0]/parsed[1] (the error string '' and the
        # raw bytes), so int('') raised ValueError and this skill ALWAYS failed.
        parsed = record.return_value
        variables = (
            parsed[2]
            if isinstance(parsed, (list, tuple)) and len(parsed) > 2
            else parsed
        )
        # 0=absolute, 1=relative
        rel_or_abs = 0
        home_pos_m = 0.0
        if isinstance(variables, (list, tuple)) and len(variables) >= 2:
            rel_or_abs = int(variables[0])
            home_pos_m = float(variables[1])
        return SkillResult(
            skill_name="GetHomeProps",
            success=True,
            data={
                "mode": "relative" if rel_or_abs == 1 else "absolute",
                "home_position_m": home_pos_m,
            },
            nanonis_calls=[record],
        )


class SetHomeProps(BaseSkill):
    """Set Z controller Home properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetHomeProps",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z controller 的 Home 位置模式与数值。",
            parameters=[
                ParameterSpec(
                    name="rel_or_abs",
                    type="int",
                    description="0=不变，1=绝对，2=相对",
                    required=True,
                    min_value=0,
                    max_value=2,
                ),
                ParameterSpec(
                    name="home_position_m",
                    type="float",
                    description="Home 位置，单位米",
                    unit="m",
                    required=True,
                    # Fine-Z config: ±1 µm cap (the Home position is a fine-Z
                    # target, same scale as TipShape / SetTipLift). A sanity
                    # ceiling (rig-tunable) so an absurd home position can't be
                    # set; without it this Z config reached hardware unbounded.
                    min_value=-1e-6,
                    max_value=1e-6,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "home", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "ZCtrl_HomePropsSet",
            params["rel_or_abs"],
            params["home_position_m"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetHomeProps",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetHomeProps",
            success=True,
            data={
                "rel_or_abs": params["rel_or_abs"],
                "home_position_m": params["home_position_m"],
            },
            nanonis_calls=[record],
        )


class SetSwitchOffDelay(BaseSkill):
    """Set the Z controller switch-off delay."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSwitchOffDelay",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z controller 的关断延时，单位秒。",
            parameters=[
                ParameterSpec(
                    name="delay_s",
                    type="float",
                    description="关断延时，单位秒",
                    unit="s",
                    required=True,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "switchoff", "delay", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "ZCtrl_SwitchOffDelaySet", params["delay_s"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetSwitchOffDelay",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetSwitchOffDelay",
            success=True,
            data={"delay_s": params["delay_s"]},
            nanonis_calls=[record],
        )


class GetWithdrawRate(BaseSkill):
    """Get the Z controller withdraw slew rate."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetWithdrawRate",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 Z controller 的退针 slew rate，单位 m/s。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "withdraw", "rate", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZCtrl_WithdrawRateGet")
        if record.error:
            return SkillResult(
                skill_name="GetWithdrawRate",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is the (error_string, raw_bytes, Variables) triple.
        # ZCtrl.WithdrawRateGet ResponseTypes = ["f"], so Variables
        # (== record.return_value[2]) is [rate(float)]. The previous code read
        # parsed[0] (the error string '') so float('') raised ValueError and this
        # skill ALWAYS failed.
        parsed = record.return_value
        variables = (
            parsed[2]
            if isinstance(parsed, (list, tuple)) and len(parsed) > 2
            else parsed
        )
        rate = 0.0
        if isinstance(variables, (list, tuple)) and len(variables) > 0:
            rate = float(variables[0])
        elif isinstance(variables, (int, float)):
            rate = float(variables)
        return SkillResult(
            skill_name="GetWithdrawRate",
            success=True,
            data={"withdraw_rate_m_per_s": rate},
            nanonis_calls=[record],
        )
