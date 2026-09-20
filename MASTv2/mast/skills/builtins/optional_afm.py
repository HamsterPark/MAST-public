"""AFM-side optional modules: Kelvin controller (KPFM), CPD compensation,
interferometer, beam deflection, laser.

**All optional, all OFF by default** (设置 → 硬件模块).

These are the modules an AFM has and an STM does not. MAST's core is an STM
pipeline; nothing here is on the tunnelling path. But the operator's rig is a TERS
setup with optical hardware, and the licences are all present — so the skills
exist, switched off, ready for the day the hardware is.

  * **KelvinCtrl** — the KPFM feedback loop: it modulates the bias, demodulates the
    resulting force, and servos the DC bias until the electrostatic force nulls.
    That DC bias IS the contact potential difference. The loop DRIVES THE BIAS,
    which is why switching it on is a CONFIRM-gated act, not a free one.
  * **CPDComp** — sweeps the bias to find the CPD parabola's apex directly, rather
    than servoing to it.
  * **Interf** — the interferometric deflection detector and its own PI loop.
  * **BeamDefl** — the optical-lever deflection detector (horizontal / vertical /
    sum), plus auto-zeroing.
  * **Laser** — on/off and setpoint power.
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


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """统一解包并返回回包 body，见 io.nanonis_files.decode_reply。

    不把 (error, raw_bytes, body) 信封直接当成读数交给调用方。
    """
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


def _multi_read(name: str, reads) -> SkillResult:
    """Read several getters; one failure blanks its own key, not the whole result.

    ``reads`` is ``((key, thunk), …)``; each thunk does ONE safe_call with a LITERAL
    verb. A verb behind a variable is invisible to every safety tool in this repo —
    they all grep ``safe_call("…")``. See readback._read_many.
    """
    calls: list = []
    data: dict = {}
    for key, thunk in reads:
        rec = thunk()
        calls.append(rec)
        data[key] = None if rec.error else _rv(rec)
    return SkillResult(skill_name=name, success=True, nanonis_calls=calls, data=data)


# ─────────────────────────────────────────────────────────────────────────────
# Kelvin controller (KPFM)
# ─────────────────────────────────────────────────────────────────────────────

class ConfigureKelvinController(BaseSkill):
    """Gains, setpoint, modulation and — critically — the bias limits."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureKelvinController",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置 Kelvin(KPFM)控制器：它伺服的解调信号、P 增益与时间常数、设定值、AC 调制，"
                "以及**偏压限值**。\n"
                "\n"
                "偏压限值是这里最要紧的参数。Kelvin 环靠驱动 DC 偏压把静电力归零 —— 环没调好或信号有噪声时它会一直驱动下去，"
                "而限值就是唯一能阻止它在针尖处于隧穿距离时把偏压推到轨上的东西。把限值设成你真正预期 CPD 会落在的范围（通常一两伏）"
                "，不要设成硬件最大值。\n"
                "\n"
                "配置**不会**打开这个环 —— 打开是 SetKelvinControllerOnOff 的事。"
            ),
            parameters=[
                ParameterSpec(name="bias_high_limit_v", type="float",
                              description="Kelvin 环可以把偏压驱动到的上界",
                              unit="V", required=True, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="bias_low_limit_v", type="float",
                              description="Kelvin 环可以把偏压驱动到的下界",
                              unit="V", required=True, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="setpoint", type="float",
                              description="环的设定值（通常为 0 —— 把力归零）",
                              required=False, default=0.0),
                ParameterSpec(name="p_gain", type="float",
                              description="比例增益",
                              required=False, default=1.0, min_value=0.0),
                ParameterSpec(name="time_constant_s", type="float",
                              description="积分时间常数",
                              unit="s", required=False, default=0.01,
                              min_value=1e-6, max_value=100.0),
                ParameterSpec(name="slope", type="int",
                              description="环的斜率：0 = 负，1 = 正",
                              required=False, default=0, min_value=0, max_value=1),
                ParameterSpec(name="control_signal_index", type="int",
                              description="环所伺服的解调信号",
                              required=False, default=0, min_value=0, max_value=127),
                ParameterSpec(name="modulation_frequency_hz", type="float",
                              description="AC 调制频率",
                              unit="Hz", required=False, default=1000.0,
                              min_value=0.0, max_value=1e6),
                ParameterSpec(name="modulation_amplitude", type="float",
                              description="AC 调制幅度（V）",
                              unit="V", required=False, default=0.1,
                              min_value=0.0, max_value=10.0),
                ParameterSpec(name="modulation_phase_deg", type="float",
                              description="AC 调制相位",
                              unit="deg", required=False, default=0.0,
                              min_value=-360.0, max_value=360.0),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["kpfm", "kelvin", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureKelvinController"
        hi = float(params["bias_high_limit_v"])
        lo = float(params["bias_low_limit_v"])
        if lo >= hi:
            return _fail(name,
                         f"bias_low_limit_v ({lo:g} V) 必须小于 bias_high_limit_v ({hi:g} V)"
                         "——限值反了，Kelvin 环会立刻把偏压推到轨上", [])
        calls: list = []

        # Limits FIRST. If any later call fails we want the guard rails already in
        # place, not a tuned loop with the old (possibly wide-open) limits.
        rec = context.safe_call("KelvinCtrl_BiasLimitsSet", hi, lo)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"KelvinCtrl_BiasLimitsSet failed: {rec.error}", calls)

        rec = context.safe_call("KelvinCtrl_CtrlSignalSet",
                                int(params.get("control_signal_index", 0) or 0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"KelvinCtrl_CtrlSignalSet failed: {rec.error}", calls)

        rec = context.safe_call("KelvinCtrl_GainSet",
                                float(params.get("p_gain", 1.0) or 1.0),
                                float(params.get("time_constant_s", 0.01) or 0.01),
                                int(params.get("slope", 0) or 0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"KelvinCtrl_GainSet failed: {rec.error}", calls)

        rec = context.safe_call("KelvinCtrl_SetpntSet",
                                float(params.get("setpoint", 0.0) or 0.0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"KelvinCtrl_SetpntSet failed: {rec.error}", calls)

        rec = context.safe_call(
            "KelvinCtrl_ModParamsSet",
            float(params.get("modulation_frequency_hz", 1000.0) or 1000.0),
            float(params.get("modulation_amplitude", 0.1) or 0.1),
            float(params.get("modulation_phase_deg", 0.0) or 0.0),
        )
        calls.append(rec)
        if rec.error:
            return _fail(name, f"KelvinCtrl_ModParamsSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"bias_high_limit_v": hi, "bias_low_limit_v": lo,
                  "setpoint": float(params.get("setpoint", 0.0) or 0.0),
                  "controller_on": False},
            summary=(f"Kelvin 控制器已配置：偏压限 [{lo:g}, {hi:g}] V"
                     "（环仍关闭——需 SetKelvinControllerOnOff 才生效）"),
        )


class SetKelvinControllerOnOff(BaseSkill):
    """Switch the KPFM loop (and its modulation) on or off."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetKelvinControllerOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "打开或关闭 Kelvin(KPFM)反馈环，连同它的 AC 调制一起。\n"
                "\n"
                "打开时须当心：从那一刻起，这个环就会在针尖处于工作距离的情况下，持续地、自主地**驱动偏压**。"
                "环没调好、或解调信号有噪声，它会把偏压一路推到 ConfigureKelvinController 所设的限值为止 —— 所以先把那些限值设好，"
                "而且要设紧。关闭则永远是安全的，偏压会停在环最后放它的地方（用 GetBias 读回）。"
            ),
            parameters=[
                ParameterSpec(name="on", type="bool",
                              description="True = 闭合 Kelvin 环（它随即开始驱动偏压）",
                              required=True),
                ParameterSpec(name="modulation_on", type="bool",
                              description="AC 调制随之一起开关",
                              required=False, default=True),
                ParameterSpec(name="ac_mode", type="bool",
                              description="AC 模式（相对 DC）",
                              required=False, default=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["kpfm", "kelvin", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetKelvinControllerOnOff"
        on = bool(params["on"])
        calls: list = []

        if bool(params.get("modulation_on", True)):
            rec = context.safe_call("KelvinCtrl_ModOnOffSet",
                                    1 if bool(params.get("ac_mode", True)) else 0,
                                    1 if on else 0)
            calls.append(rec)
            if rec.error:
                return _fail(name, f"KelvinCtrl_ModOnOffSet failed: {rec.error}", calls)

        rec = context.safe_call("KelvinCtrl_CtrlOnOffSet", 1 if on else 0)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"KelvinCtrl_CtrlOnOffSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"controller_on": on},
            summary=("Kelvin 环已闭合——它现在正在自主驱动偏压" if on
                     else "Kelvin 环已断开（偏压停在环最后给的值）"),
        )


class GetKelvinController(BaseSkill):
    """The loop's state, setpoint, gains and limits."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetKelvinController",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读 Kelvin 控制器：环是否闭合、它的设定值、增益、偏压限值、调制参数，以及解调幅度。"
                "读一下幅度就能判断这个环有没有东西可伺服 —— 幅度接近零意味着环在追噪声。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["kpfm", "kelvin", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetKelvinController", (
            ("controller_on", lambda: context.safe_call("KelvinCtrl_CtrlOnOffGet")),
            ("setpoint", lambda: context.safe_call("KelvinCtrl_SetpntGet")),
            ("gains", lambda: context.safe_call("KelvinCtrl_GainGet")),
            ("bias_limits", lambda: context.safe_call("KelvinCtrl_BiasLimitsGet")),
            ("modulation", lambda: context.safe_call("KelvinCtrl_ModParamsGet")),
            ("modulation_on", lambda: context.safe_call("KelvinCtrl_ModOnOffGet")),
            ("amplitude", lambda: context.safe_call("KelvinCtrl_AmpGet")),
            ("control_signal", lambda: context.safe_call("KelvinCtrl_CtrlSignalGet")),
        ))


class RunCpdCompensation(BaseSkill):
    """Sweep the bias to find the CPD parabola directly."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunCpdCompensation",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "打开 CPD 补偿模块并运行它：它会在给定范围内**扫描偏压**以直接找出接触电位差的抛物线，"
                "而不是像 Kelvin 环那样伺服到它。\n"
                "\n"
                "当心：这会在针尖处于工作距离时把偏压扫过 ±range_v。range_v 是**以零为中心的半幅** —— range_v=2 表示从 -2 V 扫到 +2 V。"
                "把它限制在 CPD 实际所在的那一两伏内；近距离下做大范围扫描，你就不是在测量样品，而是在改造它。"
                "\n"
                "\n"
                "用 GetCpdCompensation 读结果。"
            ),
            parameters=[
                ParameterSpec(name="range_v", type="float",
                              description="要扫描的偏压**半幅**（以零为中心的 ±range_v）",
                              unit="V", required=True, min_value=0.01, max_value=10.0),
                ParameterSpec(name="speed_hz", type="float",
                              description="扫描速度",
                              unit="Hz", required=False, default=100.0,
                              min_value=0.01, max_value=100_000.0),
                ParameterSpec(name="averaging", type="int",
                              description="每点平均次数",
                              required=False, default=1, min_value=1, max_value=10_000),
            ],
            estimated_duration_s=10.0,
            composition_level=0,
            tags=["kpfm", "cpd", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "RunCpdCompensation"
        calls: list = []
        rec = context.safe_call("CPDComp_Open")
        calls.append(rec)
        if rec.error:
            return _fail(name, f"CPDComp_Open failed: {rec.error}", calls)
        rec = context.safe_call("CPDComp_ParamsSet",
                                float(params.get("speed_hz", 100.0) or 100.0),
                                float(params["range_v"]),
                                int(params.get("averaging", 1) or 1))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"CPDComp_ParamsSet failed: {rec.error}", calls)
        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"range_v": float(params["range_v"]),
                  "speed_hz": float(params.get("speed_hz", 100.0) or 100.0)},
            summary=(f"CPD 补偿已启动：±{float(params['range_v']):g} V —— "
                     "用 GetCpdCompensation 读结果"),
        )


class GetCpdCompensation(BaseSkill):
    """The measured CPD."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetCpdCompensation",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读 CPD 补偿模块测得的接触电位差，以及它当前的参数。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["kpfm", "cpd", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetCpdCompensation", (
            ("cpd", lambda: context.safe_call("CPDComp_DataGet")),
            ("params", lambda: context.safe_call("CPDComp_ParamsGet")),
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Interferometer
# ─────────────────────────────────────────────────────────────────────────────

class ConfigureInterferometer(BaseSkill):
    """PI gains + the working piezo point; optionally null the deflection."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureInterferometer",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置干涉式偏转探测器：它的 PI 环增益与符号，以及工作点压电电压。\n"
                "\n"
                "null_deflection=true 会额外跑一次归零偏转例程，把干涉仪压电移到能让悬臂落在干涉条纹最陡（最灵敏）"
                "那一段的位置。信任任何偏转读数之前先做这一步 —— 偏离条纹时，探测器既不灵敏又非线性。"
            ),
            parameters=[
                ParameterSpec(name="integral", type="float",
                              description="PI 积分增益",
                              required=False, default=1.0),
                ParameterSpec(name="proportional", type="float",
                              description="PI 比例增益",
                              required=False, default=1.0),
                ParameterSpec(name="sign", type="int",
                              description="环的符号：0 = 负，1 = 正",
                              required=False, default=0, min_value=0, max_value=1),
                ParameterSpec(name="w_piezo", type="float",
                              description="工作点压电电压",
                              unit="V", required=False, default=0.0),
                ParameterSpec(name="null_deflection", type="bool",
                              description="跑归零偏转例程（停在最陡的条纹上）",
                              required=False, default=False),
            ],
            estimated_duration_s=3.0,
            composition_level=0,
            tags=["interferometer", "afm", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureInterferometer"
        calls: list = []

        rec = context.safe_call("Interf_CtrlPropsSet",
                                float(params.get("integral", 1.0) or 1.0),
                                float(params.get("proportional", 1.0) or 1.0),
                                int(params.get("sign", 0) or 0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Interf_CtrlPropsSet failed: {rec.error}", calls)

        rec = context.safe_call("Interf_WPiezoSet", float(params.get("w_piezo", 0.0) or 0.0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Interf_WPiezoSet failed: {rec.error}", calls)

        if bool(params.get("null_deflection", False)):
            rec = context.safe_call("Interf_CtrlNullDefl")
            calls.append(rec)
            if rec.error:
                return _fail(name, f"Interf_CtrlNullDefl failed: {rec.error}", calls)

        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"w_piezo": float(params.get("w_piezo", 0.0) or 0.0),
                                 "nulled": bool(params.get("null_deflection", False))},
                           summary="干涉仪已配置")


class SetInterferometerOnOff(BaseSkill):
    """Close or open the interferometer's PI loop."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetInterferometerOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "打开或关闭干涉仪的 PI 控制环。reset=true 会先复位这个环的积分器 —— 如果环已经积分饱和、"
                "卡在轨上，就该这么做。"
            ),
            parameters=[
                ParameterSpec(name="on", type="bool",
                              description="True = 闭合这个环",
                              required=True),
                ParameterSpec(name="reset", type="bool",
                              description="开关之前先复位这个环",
                              required=False, default=False),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["interferometer", "afm", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetInterferometerOnOff"
        calls: list = []
        if bool(params.get("reset", False)):
            rec = context.safe_call("Interf_CtrlReset")
            calls.append(rec)
            if rec.error:
                return _fail(name, f"Interf_CtrlReset failed: {rec.error}", calls)
        on = bool(params["on"])
        rec = context.safe_call("Interf_CtrlOnOffSet", 1 if on else 0)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Interf_CtrlOnOffSet failed: {rec.error}", calls)
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"controller_on": on},
                           summary=f"干涉仪控制环已{'闭合' if on else '断开'}")


class GetInterferometer(BaseSkill):
    """Deflection value, loop state, piezo working point."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetInterferometer",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读干涉仪：当前的偏转值、它的环是否闭合、它的增益，以及压电工作点。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["interferometer", "afm", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetInterferometer", (
            ("value", lambda: context.safe_call("Interf_ValGet")),
            ("controller_on", lambda: context.safe_call("Interf_CtrlOnOffGet")),
            ("gains", lambda: context.safe_call("Interf_CtrlPropsGet")),
            ("w_piezo", lambda: context.safe_call("Interf_WPiezoGet")),
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Beam deflection (optical lever)
# ─────────────────────────────────────────────────────────────────────────────

class ConfigureBeamDeflection(BaseSkill):
    """Name / units / calibration / offset for one deflection axis."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureBeamDeflection",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定**一条**光杠杆偏转轴的标定：vertical（法向力）、horizontal（横向力／摩擦力）"
                "，或强度和 sum。\n"
                "\n"
                "标定把光电二极管的伏特换算成物理单位（偏转的 N/m、力的 nN）。AFM 下游报出来的每一个力的数值都被它缩放过 —— 标定错了不会失败，"
                "它只会安静地让此后每一个力都差一个固定倍数。"
            ),
            parameters=[
                ParameterSpec(name="axis", type="str",
                              description="vertical（法向）、horizontal（横向），或 sum（强度）",
                              required=True,
                              allowed_values=["vertical", "horizontal", "sum"]),
                ParameterSpec(name="name", type="str",
                              description="在 Nanonis 里显示的信号名",
                              required=True),
                ParameterSpec(name="units", type="str",
                              description="标定之后的物理单位（例如 'nN'、'nm'）",
                              required=True),
                ParameterSpec(name="calibration", type="float",
                              description="每伏对应多少物理单位",
                              required=True),
                ParameterSpec(name="offset", type="float",
                              description="以物理单位表示的偏置",
                              required=False, default=0.0),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["beam_deflection", "afm", "calibration", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureBeamDeflection"
        axis = str(params["axis"])
        # Three literal branches. This used to resolve the verb through a dict and
        # call safe_call(verb, …) — with a comment claiming the greps could still see
        # it. They could not; the comment was simply wrong, and an AST check caught
        # the lie. Every safety tool here finds Nanonis calls by grepping
        # safe_call("…"): a verb reached through a variable is outside the safety net
        # while looking exactly like it is inside.
        sig = (str(params["name"]), str(params["units"]),
               float(params["calibration"]), float(params.get("offset", 0.0) or 0.0))
        if axis == "vertical":
            verb = "BeamDefl_VerConfigSet"
            rec = context.safe_call("BeamDefl_VerConfigSet", *sig)
        elif axis == "horizontal":
            verb = "BeamDefl_HorConfigSet"
            rec = context.safe_call("BeamDefl_HorConfigSet", *sig)
        elif axis == "sum":
            verb = "BeamDefl_IntConfigSet"
            rec = context.safe_call("BeamDefl_IntConfigSet", *sig)
        else:
            return _fail(name, f"未知 axis：{axis!r}（应为 vertical/horizontal/sum）", [])
        if rec.error:
            return _fail(name, f"{verb} failed: {rec.error}", [rec])
        return SkillResult(skill_name=name, success=True, nanonis_calls=[rec],
                           data={"axis": axis,
                                 "calibration": float(params["calibration"])},
                           summary=f"光杠杆 {axis} 轴已标定")


class GetBeamDeflection(BaseSkill):
    """All three axes' configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetBeamDeflection",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读光杠杆偏转探测器三条轴（vertical、horizontal、sum）的配置：名称、"
                "单位、标定与偏置。信任任何力的数值之前，先核一下标定。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["beam_deflection", "afm", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetBeamDeflection", (
            ("vertical", lambda: context.safe_call("BeamDefl_VerConfigGet")),
            ("horizontal", lambda: context.safe_call("BeamDefl_HorConfigGet")),
            ("sum", lambda: context.safe_call("BeamDefl_IntConfigGet")),
        ))


class AutoZeroBeamDeflection(BaseSkill):
    """Null the deflection signal's offset."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AutoZeroBeamDeflection",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "自动归零光杠杆偏转信号：测出它当前的值，并把它作为偏置扣掉，于是「零偏转」就等于悬臂当前的静止位置。"
                "\n"
                "\n"
                "做这件事时针尖必须**已退针**、悬臂处于自由状态。在接触状态下自动归零，等于把带载的偏转定义成了零 —— 此后你测到的每一个力，"
                "都会被你原本就施加着的那份载荷偏移掉。"
            ),
            parameters=[
                ParameterSpec(name="deflection_signal", type="int",
                              description="要归零的偏转信号序号",
                              required=False, default=0, min_value=0, max_value=127),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["beam_deflection", "afm", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("BeamDefl_AutoOffset",
                                int(params.get("deflection_signal", 0) or 0))
        if rec.error:
            return _fail("AutoZeroBeamDeflection", f"BeamDefl_AutoOffset failed: {rec.error}", [rec])
        return SkillResult(skill_name="AutoZeroBeamDeflection", success=True,
                           nanonis_calls=[rec], summary="光杠杆偏转已自动归零")


# ─────────────────────────────────────────────────────────────────────────────
# Laser
# ─────────────────────────────────────────────────────────────────────────────

class SetLaserOnOff(BaseSkill):
    """Laser on/off."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetLaserOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "打开或关闭激光。\n"
                "\n"
                "打开时是 DANGEROUS。激光既是对眼睛的危害，也是结区的一份热负载；MAST 看不到快门是否关着、"
                "显微镜旁有没有人、光束打向哪里。只有在用户要求时才打开它。\n"
                "\n"
                "关闭则永远允许 —— 包括在 abort 之后：对激光而言，「关」毫无歧义地就是安全态。"
            ),
            parameters=[
                ParameterSpec(name="on", type="bool",
                              description="True = 激光开",
                              required=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["laser", "optical", "dangerous", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        on = bool(params["on"])
        rec = context.safe_call("Laser_OnOffSet", 1 if on else 0)
        if rec.error:
            return _fail("SetLaserOnOff", f"Laser_OnOffSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetLaserOnOff", success=True, nanonis_calls=[rec],
                           data={"laser_on": on},
                           summary=f"激光已{'开启' if on else '关闭'}")


class SetLaserPower(BaseSkill):
    """Laser setpoint."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetLaserPower",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定激光的功率设定值。**不会**打开激光 —— 那是 SetLaserOnOff 的事。"
                "趁激光关着的时候设功率，是预置一个值的安全做法：设好、用 GetLaser 读回、再打开。"
            ),
            parameters=[
                ParameterSpec(name="setpoint", type="float",
                              description="激光功率设定值（用激光模块自己的单位）",
                              required=True, min_value=0.0),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["laser", "optical", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("Laser_PropsSet", float(params["setpoint"]))
        if rec.error:
            return _fail("SetLaserPower", f"Laser_PropsSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetLaserPower", success=True, nanonis_calls=[rec],
                           data={"setpoint": float(params["setpoint"])},
                           summary=f"激光功率设定值 = {float(params['setpoint']):g}（激光开关未变）")


class GetLaser(BaseSkill):
    """Is it on, and at what power?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetLaser",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读激光：它是否处于开启状态、实测功率，以及设定值。在假定激光是关着的之前，先读这一条。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["laser", "optical", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetLaser", (
            ("laser_on", lambda: context.safe_call("Laser_OnOffGet")),
            ("power", lambda: context.safe_call("Laser_PowerGet")),
            ("setpoint", lambda: context.safe_call("Laser_PropsGet")),
        ))
