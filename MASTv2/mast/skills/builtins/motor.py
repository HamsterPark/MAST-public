"""Motor skills — coarse motor movement (DANGEROUS).

vendored from v1 mast/skills/builtins/motor.py 2026-04-23.
7 skills: MotorMove, MotorGetPos, StopMotor, GetMotorFreqAmp, SetMotorFreqAmp,
          MotorMoveClosedLoop, GetMotorStepCounter.

2026-07-31 — the drive voltage became operator-only. ``SetMotorFreqAmp`` now
consults ``mast.core.coarse_drive`` (per-rig declared ceiling, refuse-don't-clamp)
and ``GetMotorFreqAmp`` was added so a coarse move can verify the drive BEFORE it
steps. See ``docs/v2/design/coarse_motion_intelligence.md`` §4 for the four locks.

Note on which lateral path is the autonomous one: ``MotorMove(x±/y±)`` is the raw
primitive — it checks only the fine-Z piezo (~1 µm of clearance, and it passes on
unknown state) and records nothing about where the stage has been. The guarded
composite ``RelocateCoarseXY`` is the one the agent should call.
"""

from __future__ import annotations

from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)


class MotorMove(BaseSkill):
    """Move the coarse (pan-type stepper) motor one axis at a time.

    **横向粗动是常规操作,不是禁区。** 三档,危险性完全不同 ——

    ====================  ==========  ============================================
    方向                   门禁         为什么
    ====================  ==========  ============================================
    ``x±`` / ``y±``       CONFIRM     横移撞不到样品(样品在 Z 方向)。换区就靠它。
    ``z-retract``         CONFIRM     远离样品。是**变安全**的方向。
    ``z-approach``        人工批准     开环朝样品走、没有反馈停 —— 唯一真正危险的那一档。
    ====================  ==========  ============================================

    ⚠️ 2026-08-24 更正:这段自述原来写的是「DANGEROUS … Requires human approval」,
    对**所有**方向一概而论 —— 而同一个类的 ``metadata()`` 从 2026-06-11 起就是
    ``CONFIRM``,并且只把 ``z-approach`` 交给 ``is_coarse_sample_approach()`` 去挡。
    **自述与 metadata 矛盾了两个多月,而 agent 读的是自述**,于是横向换区被当成
    需要人守着的动作,自主流程宁可绕路也不用它。结论是:粗动本不该被设定得
    这么可怕。

    先决条件 ``z_controller_off`` 仍然要满足(先退针再动粗动马达)——那是防撞,
    与「危不危险」的分档无关。换区请优先用 ``RelocateCoarseXY``:它把退针、
    确认清空、分段移动、对账、重进针都串好了;这条原语是它的底层。
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MotorMove",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # CONFIRM baseline (2026-06-11 safety re-scoping): X/Y coarse moves and
            # Z RETRACT (away from sample) can't wreck the instrument — LLM/operator
            # confirm path. The ONE physically-dangerous case — an open-loop coarse
            # Z step TOWARD the sample ('z-approach', pan-type stepper, no feedback
            # stop) — is gated to HUMAN approval by is_coarse_sample_approach() in
            # SkillExecutor (manual path) and BLOCKED outright on the autonomous
            # agent path by SafetyGateMiddleware. The z_controller_off precondition
            # still applies (withdraw before stepping the coarse motor).
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "驱动粗动马达（pan 型步进）。x±/y± 为横向；"
                "'z-approach' 朝**样品方向**走步（DANGEROUS —— 需要人工"
                "批准）；'z-retract' 朝**远离样品**走步（安全）。"
            ),
            parameters=[
                ParameterSpec(
                    name="direction",
                    type="str",
                    description=(
                        "马达方向：'x+','x-','y+','y-'（横向）、"
                        "'z-approach'（朝样品）、'z-retract'（远离样品）"
                    ),
                    required=True,
                    allowed_values=["x+", "x-", "y+", "y-", "z-approach", "z-retract"],
                ),
                ParameterSpec(
                    name="steps",
                    type="int",
                    description="马达走步数",
                    required=True,
                    min_value=1,
                    max_value=1000,
                ),
            ],
            preconditions=["z_controller_off"],
            estimated_duration_s=10.0,
            composition_level=0,
            # tags 不参与任何门禁(门禁看 safety_level),它只被人和 agent 读。
            # 原来挂着一个一刀切的 "dangerous",与上面的分档矛盾 —— 去掉。
            tags=["motor", "coarse", "relocation"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        direction = params["direction"]
        steps = params["steps"]

        # 0=X+,1=X-,2=Y+,3=Y-,4=Z+,5=Z-. 'z-approach' = toward sample (Z-, the
        # DANGEROUS open-loop step gated by is_coarse_sample_approach); 'z-retract'
        # = away (Z+). The danger gate keys off the semantic direction string, not
        # this code, so it stays correct if a rig needs the two Z codes swapped.
        dir_map = {"x+": 0, "x-": 1, "y+": 2, "y-": 3, "z-approach": 5, "z-retract": 4}
        # Fail LOUDLY on an unknown direction rather than silently moving X+. The
        # allowed_values guard now covers the executor/composite paths, but a
        # direct execute() with a typo'd direction must not send the coarse motor
        # off along the wrong axis (2026-07-03 review).
        if direction not in dir_map:
            return SkillResult(
                skill_name="MotorMove", success=False,
                error=(f"unknown motor direction {direction!r}; expected one of "
                       f"{sorted(dir_map)}"),
            )
        dir_code = dir_map[direction]

        # Lateral coarse moves (x/y) drag the tip sideways. Right after STS the
        # Z-controller is OFF but the tip is still near the surface (withdrawn is
        # only guaranteed by an explicit retract), so a lateral step would scrape
        # it. Refuse a lateral move when we KNOW the tip is not withdrawn
        # (unknown state passes, matching the precondition fail-open). Z moves are
        # exempt: 'z-retract' moves away, and 'z-approach' is human-gated
        # elsewhere and legitimately steps a non-withdrawn tip.
        if direction in ("x+", "x-", "y+", "y-"):
            try:
                snap = context.state.snapshot() if getattr(context, "state", None) else None
            except Exception:
                snap = None
            if snap is not None and getattr(snap, "withdrawn", None) is False:
                return SkillResult(
                    skill_name="MotorMove", success=False,
                    error=("tip is not withdrawn — retract the tip (WithdrawTip) "
                           "before a lateral coarse motor move to avoid scraping "
                           "the surface."),
                )

        record = context.safe_call("Motor_StartMove", dir_code, steps, 0, 1)
        if record.error:
            return SkillResult(
                skill_name="MotorMove",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )

        # ⑫ A successful lateral coarse move IS the "换区" escape from a
        # repeatedly-crashing spot — clear the tip-crash blocks so the fresh
        # region starts with a clean slate (see mast.core.tip_crash_tracker).
        if direction in ("x+", "x-", "y+", "y-"):
            try:
                from mast.core.tip_crash_tracker import get_tip_crash_tracker
                get_tip_crash_tracker().note_recovery()
            except Exception:  # noqa: BLE001 — recovery bookkeeping is best-effort
                pass

        return SkillResult(
            skill_name="MotorMove",
            success=True,
            data={"direction": direction, "steps": steps},
            nanonis_calls=[record],
        )


class MotorGetPos(BaseSkill):
    """Get current coarse motor position."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MotorGetPos",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取粗动马达位置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["motor", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Motor_PosGet", 0, 500)
        if record.error:
            return SkillResult(
                skill_name="MotorGetPos",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                data = {"x_m": float(vals[0]), "y_m": float(vals[1]), "z_m": float(vals[2])}
        return SkillResult(
            skill_name="MotorGetPos",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


class StopMotor(BaseSkill):
    """Emergency stop motor movement."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopMotor",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="紧急停止所有马达运动。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["motor", "stop", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Motor_StopMove")
        if record.error:
            return SkillResult(
                skill_name="StopMotor",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="StopMotor",
            success=True,
            data={"stopped": True},
            nanonis_calls=[record],
        )


class GetMotorFreqAmp(BaseSkill):
    """Read back the coarse stepper's drive frequency and amplitude.

    The counterpart to :class:`SetMotorFreqAmp`, and the input to the pre-move
    check every coarse motion runs. The declared per-rig ceiling constrains what
    MAST *writes*; it says nothing about what the drive is set to right now —
    the Nanonis UI is right there, and a previous session may have left it high.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetMotorFreqAmp",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读回粗动马达当前的驱动频率与幅度(电压),并与本机声明的耐压上限比对。"
                "**只读** —— 设置驱动电压是用户的权限,agent 不能改。"
                "粗动移动前会自动做这个核对;读不到就拒绝粗动(读不到 ≠ 没问题)。"
            ),
            parameters=[
                ParameterSpec(
                    name="axis",
                    type="str",
                    description="马达轴：'all'、'x'、'y' 或 'z'",
                    required=False,
                    default="all",
                    allowed_values=["all", "x", "y", "z"],
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["motor", "coarse", "read", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        axis = str(params.get("axis", "all") or "all").lower()
        axis_int = {"all": 0, "x": 1, "y": 2, "z": 3}.get(axis, 0)
        record = context.safe_call("Motor_FreqAmpGet", axis_int)
        if record.error:
            return SkillResult(
                skill_name="GetMotorFreqAmp", success=False,
                error=record.error, nanonis_calls=[record])
        freq, amp = _parse_freq_amp(record.return_value)
        from mast.core import coarse_drive

        ok, reason = coarse_drive.readback_matches(amp, freq)
        return SkillResult(
            skill_name="GetMotorFreqAmp",
            success=True,
            data={
                "axis": axis,
                "frequency_hz": freq,
                "amplitude_v": amp,
                "declared_max_amplitude_v": coarse_drive.max_amplitude_v(),
                "within_declared_limit": bool(ok),
                "note": reason,
            },
            nanonis_calls=[record],
        )


def _parse_freq_amp(parsed) -> tuple[float | None, float | None]:
    """按 frequency_hz、amplitude_v 顺序解析 Motor_FreqAmpGet。
    
    形状不符时返回 (None, None)，不能猜测幅度以授权粗动。
    字段意义由协议布局确定，不能靠两个数的大小推断频率和电压顺序。"""
    if not (isinstance(parsed, (list, tuple)) and len(parsed) > 2):
        return None, None
    body = parsed[2]
    vals: list[float] = []
    if isinstance(body, (int, float)) and not isinstance(body, bool):
        vals = [float(body)]
    elif isinstance(body, (list, tuple)):
        for el in body:
            if isinstance(el, (int, float)) and not isinstance(el, bool):
                vals.append(float(el))
            elif isinstance(el, (list, tuple)):
                vals.extend(float(x) for x in el
                            if isinstance(x, (int, float)) and not isinstance(x, bool))
    if len(vals) < 2:
        return (vals[0] if vals else None), None
    return vals[0], vals[1]


class SetMotorFreqAmp(BaseSkill):
    """Set motor frequency and amplitude — OPERATOR-ONLY, four independent locks.

    See :mod:`mast.core.coarse_drive` for why. The short version: the voltage the
    controller can output is not the voltage this rig's piezo stack survives,
    nothing reads back which rig you are on, and getting it wrong is
    unrecoverable.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetMotorFreqAmp",
            version="1.1.0",
            category=SkillCategory.WRITE,
            # Stays CONFIRM as the BASELINE; the real gating is elsewhere and is
            # not expressible as a level: safety.is_coarse_drive_change forces
            # human approval on the manual path and blocks the autonomous one,
            # the advanced_capabilities gate keeps the tool out of the agent's
            # list entirely, and coarse_drive.authorize() refuses anything above
            # (or without) the operator's declaration.
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置粗动马达的驱动频率与幅度(电压)。**这是用户的参数,不是 agent 的。**"
                "写入必须落在【高级】页声明的本机耐压上限之内;未声明则一律拒绝,"
                "超上限**直接拒绝而不是降到上限**(悄悄降下来会让调用方以为自己设的是"
                "另一个值)。"),
            parameters=[
                ParameterSpec(
                    name="frequency_hz",
                    type="float",
                    description="马达驱动频率，单位 Hz",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                    # Coarse-stepper drive frequency ceiling — real steppers run
                    # ≤ a few kHz; 20 kHz is a generous cap that still rejects a
                    # hallucinated absurd value (2026-07-03 review).
                    max_value=20000.0,
                ),
                ParameterSpec(
                    name="amplitude_v",
                    type="float",
                    description="马达驱动幅度，单位伏特",
                    unit="V",
                    required=True,
                    min_value=0.0,
                    # OUTER bound only, and a rig-independent one: the real limit
                    # is the operator's declaration in mast.core.coarse_drive,
                    # enforced in execute(). 400 V is the most any coarse-motion
                    # controller in this class outputs, so anything above it is a
                    # units/magnitude error rather than an aggressive setting.
                    # It used to be 200 V, which is neither a real controller
                    # ceiling nor any particular rig's stack limit — a number
                    # that protects nothing while looking like it does.
                    max_value=400.0,
                ),
                ParameterSpec(
                    name="axis",
                    type="str",
                    description="马达轴：'all'、'x'、'y' 或 'z'",
                    required=False,
                    default="all",
                    allowed_values=["all", "x", "y", "z"],
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["motor", "frequency", "amplitude", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        frequency_hz = params["frequency_hz"]
        amplitude_v = params["amplitude_v"]
        axis = params.get("axis", "all")

        # Lock 3 + 4: the operator's per-rig ceiling and the absolute one. This
        # runs INSIDE execute rather than as a ParameterSpec bound because the
        # limit is runtime state (a declaration that can be made, changed, or
        # never made at all) and because "no declaration" must refuse rather than
        # fall back to a default — a bound cannot express that.
        from mast.core import coarse_drive

        ok, reason = coarse_drive.authorize(amplitude_v, frequency_hz, str(axis))
        if not ok:
            return SkillResult(
                skill_name="SetMotorFreqAmp", success=False, error=reason)

        axis_map = {"all": 0, "x": 1, "y": 2, "z": 3}
        axis_int = axis_map.get(axis.lower(), 0)

        record = context.safe_call(
            "Motor_FreqAmpSet", frequency_hz, amplitude_v, axis_int,
        )
        if record.error:
            return SkillResult(
                skill_name="SetMotorFreqAmp",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetMotorFreqAmp",
            success=True,
            data={
                "frequency_hz": frequency_hz,
                "amplitude_v": amplitude_v,
                "axis": axis,
            },
            nanonis_calls=[record],
        )


class MotorMoveClosedLoop(BaseSkill):
    """Move the coarse motor in closed loop to a target position.

    危险性与 :class:`MotorMove` 同源:**取决于目标点在哪个方向**,不是这个技能
    本身危险。朝样品走的那一段仍由 ``is_coarse_sample_approach()`` 挡;横向与
    远离样品的移动是常规操作(metadata 是 ``CONFIRM``)。

    ⚠️ 2026-08-24 与 :class:`MotorMove` 一起更正:原自述写「Requires human
    approval」而 metadata 一直是 CONFIRM。

    并非所有马达控制模块都支持闭环 —— 不支持时它会明说,不会假装移动过。
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MotorMoveClosedLoop",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "闭环粗动到目标 XYZ。DANGEROUS:可能与样品相撞 —— "
                "横向与远离样品的移动是常规操作;朝样品走的那一段由人工批准挡着。"
                "并非所有控制器都支持闭环。"
            ),
            parameters=[
                ParameterSpec(
                    name="absolute",
                    type="bool",
                    description="True 表示绝对位置，False 表示相对移动",
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="target_x_m",
                    type="float",
                    description="目标 X 位置，单位米",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="target_y_m",
                    type="float",
                    description="目标 Y 位置，单位米",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="target_z_m",
                    type="float",
                    description="目标 Z 位置，单位米",
                    unit="m",
                    required=True,
                    # Coarse-motor safety CEILING, NOT the true coarse Z range
                    # (which is rig-specific and unknown here). ±1e-4 m (±100 µm)
                    # rejects absurd / fat-fingered Z targets that could drive the
                    # stepper hard into the sample; the real per-rig range is
                    # enforced by Nanonis + the human-approval gate (any Z-bearing
                    # closed-loop move is treated as a coarse sample approach by
                    # is_coarse_sample_approach). target_x/y_m are deliberately
                    # left unbounded here — they are already clamped by the global
                    # x_m/y_m bounds check (mast.core.safety._GLOBAL_CHECKS).
                    min_value=-1e-4,
                    max_value=1e-4,
                ),
                ParameterSpec(
                    name="wait",
                    type="bool",
                    description="等到移动结束再返回",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="group",
                    type="int",
                    description="马达组（0-5）",
                    required=False,
                    default=0,
                    min_value=0,
                    max_value=5,
                ),
            ],
            preconditions=["z_controller_off"],
            estimated_duration_s=30.0,
            composition_level=0,
            tags=["motor", "coarse", "closed_loop", "dangerous"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        abs_rel = 1 if params.get("absolute", False) else 0
        tx = params["target_x_m"]
        ty = params["target_y_m"]
        tz = params["target_z_m"]
        wait = 1 if params.get("wait", True) else 0
        group = params.get("group", 0)

        record = context.safe_call(
            "Motor_StartClosedLoop", abs_rel, tx, ty, tz, wait, group,
        )
        if record.error:
            return SkillResult(
                skill_name="MotorMoveClosedLoop",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="MotorMoveClosedLoop",
            success=True,
            data={
                "absolute": bool(abs_rel),
                "target_x_m": tx,
                "target_y_m": ty,
                "target_z_m": tz,
            },
            nanonis_calls=[record],
        )


class GetMotorStepCounter(BaseSkill):
    """Read motor step counter values for X, Y, Z.

    Optionally resets counters after reading. Only available on Attocube ANC150.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetMotorStepCounter",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取 X、Y、Z 三轴的步数计数器值。"
                "可选择读完后清零。仅适用于 Attocube ANC150。"
            ),
            parameters=[
                ParameterSpec(
                    name="reset_x",
                    type="bool",
                    description="读完后把 X 步数计数器清零",
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="reset_y",
                    type="bool",
                    description="读完后把 Y 步数计数器清零",
                    required=False,
                    default=False,
                ),
                ParameterSpec(
                    name="reset_z",
                    type="bool",
                    description="读完后把 Z 步数计数器清零",
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["motor", "step_counter", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rx = 1 if params.get("reset_x", False) else 0
        ry = 1 if params.get("reset_y", False) else 0
        rz = 1 if params.get("reset_z", False) else 0

        record = context.safe_call("Motor_StepCounterGet", rx, ry, rz)
        if record.error:
            return SkillResult(
                skill_name="GetMotorStepCounter",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                data = {
                    "step_counter_x": int(vals[0]),
                    "step_counter_y": int(vals[1]),
                    "step_counter_z": int(vals[2]),
                }
        return SkillResult(
            skill_name="GetMotorStepCounter",
            success=True,
            data=data,
            nanonis_calls=[record],
        )
