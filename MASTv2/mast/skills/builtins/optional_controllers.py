"""Generic PI controllers, the MCVA5 preamp, PLL analysis, OC Sync, tip recorder.

**All optional, all OFF by default** (设置 → 硬件模块).

TWO GENERATIONS OF THE SAME MODULE
==================================
Nanonis ships the generic PI controller twice, and they are NOT the same API:

  * ``PICtrl_*``    — the **V5e** generic PI controller. Indexed: every call takes a
                      ``Controller_Index``, because there are several. The operator's
                      rig is a V5e, so this is the one that matters.
  * ``GenPICtrl_*`` — the **V5** generic PI controller. Singleton, no index, and it
                      has an analogue-output section the V5e one does not.

MAST wraps both, because a licence file does not tell you which generation the
crate actually has. ``ConfigurePiController`` drives the V5e (indexed) one;
``SetGenericPiOutput`` / ``GetGenericPiController`` drive the V5 one. If a call
errors with "module not available", you have the other generation — switch skills,
not settings.

WHAT A GENERIC PI LOOP IS
=========================
Point it at any input signal, give it a setpoint, and it drives any output until
the input matches. That is a very sharp tool: the output can be a piezo, the bias,
a laser, a heater. The module does not know and cannot warn you. Which is why
closing one of these loops is DANGEROUS — see SetPiControllerOnOff.
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


def _multi_read(name: str, reads, data: dict | None = None) -> SkillResult:
    """``reads`` = ((key, thunk), …); each thunk does ONE safe_call with a LITERAL
    verb — every safety tool in this repo greps safe_call("…"). See
    readback._read_many."""
    calls: list = []
    out: dict = dict(data or {})
    for key, thunk in reads:
        rec = thunk()
        calls.append(rec)
        out[key] = None if rec.error else _rv(rec)
    return SkillResult(skill_name=name, success=True, nanonis_calls=calls, data=out)


# ─────────────────────────────────────────────────────────────────────────────
# Generic PI controller — V5e (indexed)
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurePiController(BaseSkill):
    """Input, output, setpoint, gains and output limits for one PI loop."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigurePiController",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置一个通用 PI 控制器（V5e）：它**监视**哪个信号（输入）、**驱动**哪个信号（控制信号）"
                "、设定值、增益，以及输出限值。\n"
                "\n"
                "输出限值是这里最要紧的参数。这个环会毫无边界地驱动它的控制信号，直到输入达到设定值为止 —— 如果设定值根本到不了（传感器坏了、"
                "符号弄反了），它就会一路驱动到限值并停在那儿。把限值设成输出可以安全占据的范围，不要设成硬件最大值。"
                "\n"
                "\n"
                "slope 会把环的作用方向反过来。slope 弄错的环不会振荡 —— 它会朝**背离**设定值的方向一路跑到限值。"
                "\n"
                "\n"
                "配置并不会闭合这个环；闭合是 SetPiControllerOnOff 的事。"
            ),
            parameters=[
                ParameterSpec(name="controller_index", type="int",
                              description="第几个 PI 控制器（0 起算）",
                              required=True, min_value=0, max_value=15),
                ParameterSpec(name="input_index", type="int",
                              description="环所**监视**的信号",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="control_signal_index", type="int",
                              description="环所**驱动**的信号",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="setpoint", type="float",
                              description="输入信号的目标值",
                              required=True),
                ParameterSpec(name="output_lower_limit", type="float",
                              description="环可以把输出驱动到的最低值",
                              required=True),
                ParameterSpec(name="output_upper_limit", type="float",
                              description="环可以把输出驱动到的最高值",
                              required=True),
                ParameterSpec(name="p_gain", type="float",
                              description="比例增益",
                              required=False, default=1.0, min_value=0.0),
                ParameterSpec(name="i_gain", type="float",
                              description="积分增益",
                              required=False, default=0.0, min_value=0.0),
                ParameterSpec(name="slope", type="int",
                              description="0 = 负，1 = 正。slope **弄错**会让输出一路跑到限值。",
                              required=False, default=0, min_value=0, max_value=1),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["pi_controller", "feedback", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigurePiController"
        idx = int(params["controller_index"])
        lo = float(params["output_lower_limit"])
        hi = float(params["output_upper_limit"])
        if lo >= hi:
            return _fail(name,
                         f"output_lower_limit ({lo:g}) 必须小于 output_upper_limit ({hi:g})"
                         "——限值反了，环会立刻把输出推到轨上", [])
        calls: list = []

        # Limits FIRST — same reasoning as the Kelvin loop: if a later call fails,
        # the guard rails are already in place.
        rec = context.safe_call("PICtrl_CtrlChPropsSet", idx, lo, hi)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PICtrl_CtrlChPropsSet failed: {rec.error}", calls)

        rec = context.safe_call("PICtrl_InputChSet", idx, int(params["input_index"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PICtrl_InputChSet failed: {rec.error}", calls)

        rec = context.safe_call("PICtrl_CtrlChSet", idx, int(params["control_signal_index"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PICtrl_CtrlChSet failed: {rec.error}", calls)

        rec = context.safe_call("PICtrl_PropsSet", idx,
                                float(params["setpoint"]),
                                float(params.get("p_gain", 1.0) or 1.0),
                                float(params.get("i_gain", 0.0) or 0.0),
                                int(params.get("slope", 0) or 0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PICtrl_PropsSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"controller_index": idx, "setpoint": float(params["setpoint"]),
                  "output_limits": [lo, hi], "controller_on": False},
            summary=(f"PI 控制器 {idx} 已配置：输出限 [{lo:g}, {hi:g}]"
                     "（环仍开路——需 SetPiControllerOnOff 才闭合）"),
        )


class SetPiControllerOnOff(BaseSkill):
    """Close or open one PI loop."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPiControllerOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "闭合或断开一个通用 PI 控制环（V5e）。\n"
                "\n"
                "闭合时是 DANGEROUS。从那一刻起，这个环就会自主地**驱动它的输出** —— 而这个输出可能是一块压电、"
                "是偏压、是加热器、是激光。模块本身并不知道是哪一个。先调 GetPiController 读一下这个环实际接的是什么、"
                "它的输出限值是多少。\n"
                "\n"
                "断开这个环是安全的，输出会停在环最后放它的地方 —— 它并不会被送回任何静止值。"
            ),
            parameters=[
                ParameterSpec(name="controller_index", type="int",
                              description="第几个 PI 控制器（0 起算）",
                              required=True, min_value=0, max_value=15),
                ParameterSpec(name="on", type="bool",
                              description="True = 闭合此环（它随即开始驱动）",
                              required=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pi_controller", "feedback", "dangerous", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["controller_index"])
        on = bool(params["on"])
        rec = context.safe_call("PICtrl_OnOffSet", idx, 1 if on else 0)
        if rec.error:
            return _fail("SetPiControllerOnOff", f"PICtrl_OnOffSet failed: {rec.error}", [rec])
        return SkillResult(
            skill_name="SetPiControllerOnOff", success=True, nanonis_calls=[rec],
            data={"controller_index": idx, "controller_on": on},
            summary=(f"PI 控制器 {idx} 已闭合——它现在正在自主驱动输出" if on
                     else f"PI 控制器 {idx} 已开路（输出停在环最后给的值）"),
        )


class GetPiController(BaseSkill):
    """What one PI loop watches, drives, and is set to."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPiController",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读一个通用 PI 控制器（V5e）：它是否闭合、监视哪个信号、**驱动哪个信号**、它的设定值、"
                "增益与输出限值。\n"
                "\n"
                "闭合任何一个不是你自己配的环之前，先读这个。控制信号才是要紧的那一项 —— 那是环将要去移动的东西。"
            ),
            parameters=[
                ParameterSpec(name="controller_index", type="int",
                              description="第几个 PI 控制器（0 起算）",
                              required=True, min_value=0, max_value=15),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pi_controller", "feedback", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["controller_index"])
        return _multi_read("GetPiController", (
            ("controller_on", lambda: context.safe_call("PICtrl_OnOffGet", idx)),
            ("input_signal", lambda: context.safe_call("PICtrl_InputChGet", idx)),
            ("control_signal", lambda: context.safe_call("PICtrl_CtrlChGet", idx)),
            ("props", lambda: context.safe_call("PICtrl_PropsGet", idx)),
            ("output_limits", lambda: context.safe_call("PICtrl_CtrlChPropsGet", idx)),
        ), data={"controller_index": idx})


# ─────────────────────────────────────────────────────────────────────────────
# Generic PI controller — V5 (singleton, with an analogue output)
# ─────────────────────────────────────────────────────────────────────────────

class SetGenericPiOutput(BaseSkill):
    """The V5 generic PI controller's analogue output value."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetGenericPiOutput",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "直接设定 V5 通用 PI 控制器的模拟输出，用它**自己的物理单位**（不是伏特 —— 模块会套用它自己的标定）"
                "。\n"
                "\n"
                "当心：这是直接写到一个 MAST 看不见其接线的物理输出上。先读 GetGenericPiController —— 它会报出这个输出的名称、"
                "单位与限值，那是你弄清自己将要驱动什么的唯一途径。\n"
                "\n"
                "直接写输出只有在环**断开**时才讲得通。环闭合时，控制器会在下一拍就把你设的值覆盖掉。"
                "\n"
                "\n"
                "这是 V5 模块。V5e 机箱上请改用 ConfigurePiController；这里报错就说明你手上是另一代。"
            ),
            parameters=[
                ParameterSpec(name="value", type="float",
                              description="输出值，用该输出**自己的**物理单位（见 GetGenericPiController）",
                              required=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pi_controller", "output", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("GenPICtrl_AOValSet", float(params["value"]))
        if rec.error:
            return _fail("SetGenericPiOutput", f"GenPICtrl_AOValSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetGenericPiOutput", success=True, nanonis_calls=[rec],
                           data={"value": float(params["value"])},
                           summary=f"通用 PI(V5) 模拟输出 = {float(params['value']):g}（物理单位）")


class GetGenericPiController(BaseSkill):
    """The V5 controller's state, output value, and output calibration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetGenericPiController",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读 V5 通用 PI 控制器：环的状态、设定值与增益、模拟输出的当前值，以及 —— 写任何东西之前你都需要的那一项 —— 输出的**名称、"
                "单位与限值**。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pi_controller", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetGenericPiController", (
            ("controller_on", lambda: context.safe_call("GenPICtrl_OnOffGet")),
            ("props", lambda: context.safe_call("GenPICtrl_PropsGet")),
            ("output_value", lambda: context.safe_call("GenPICtrl_AOValGet")),
            ("output_props", lambda: context.safe_call("GenPICtrl_AOPropsGet")),
            ("modulation_channel", lambda: context.safe_call("GenPICtrl_ModChGet")),
            ("demod_channel", lambda: context.safe_call("GenPICtrl_DemodChGet")),
        ))


# ─────────────────────────────────────────────────────────────────────────────
# MCVA5 preamplifier
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurePreamp(BaseSkill):
    """MCVA5 gain / coupling / input mode for one channel."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigurePreamp",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定 MCVA5 前置放大器某个通道的增益、耦合方式（AC/DC）与输入模式。\n"
                "\n"
                "改前放增益，会改变同一个物理电流**读出来是多少**。Z 控制器闭合时，反馈环会把这看成电流突变，"
                "并据此**移动针尖**。改增益之前先退针，或者先断开 Z 环。\n"
                "\n"
                "前放编号与通道编号沿用 MCVA5 自己的编号方式。"
            ),
            parameters=[
                ParameterSpec(name="preamp", type="int",
                              description="前放编号（MCVA5 编号方式）",
                              required=True, min_value=0, max_value=7),
                ParameterSpec(name="channel", type="int",
                              description="该前放上的通道号",
                              required=True, min_value=0, max_value=7),
                ParameterSpec(name="gain", type="int",
                              description="增益档位序号 —— 不传则保持不变",
                              required=False, default=None, min_value=0, max_value=31),
                ParameterSpec(name="coupling", type="int",
                              description="耦合方式序号（AC/DC）—— 不传则保持不变",
                              required=False, default=None, min_value=0, max_value=7),
                ParameterSpec(name="input_mode", type="int",
                              description="输入模式序号 —— 不传则保持不变",
                              required=False, default=None, min_value=0, max_value=7),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["preamp", "mcva5", "gain", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigurePreamp"
        pre, ch = int(params["preamp"]), int(params["channel"])
        calls: list = []
        touched: list[str] = []

        if params.get("gain") is not None:
            rec = context.safe_call("MCVA5_GainSet", pre, ch, int(params["gain"]))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"MCVA5_GainSet failed: {rec.error}", calls)
            touched.append("gain")
        if params.get("coupling") is not None:
            rec = context.safe_call("MCVA5_CouplingSet", pre, ch, int(params["coupling"]))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"MCVA5_CouplingSet failed: {rec.error}", calls)
            touched.append("coupling")
        if params.get("input_mode") is not None:
            rec = context.safe_call("MCVA5_InputModeSet", pre, ch, int(params["input_mode"]))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"MCVA5_InputModeSet failed: {rec.error}", calls)
            touched.append("input_mode")

        if not touched:
            return _fail(name, "gain / coupling / input_mode 至少要给一个", calls)
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"preamp": pre, "channel": ch, "changed": touched},
                           summary=f"MCVA5 前放 {pre}/{ch} 已设置：{', '.join(touched)}")


class GetPreamp(BaseSkill):
    """MCVA5 channel state."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPreamp",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读 MCVA5 前置放大器某个通道：它的增益、耦合方式与输入模式。解读任何电流之前先读增益 —— 没有它，"
                "ADC 报出来的数字什么也不是。"
            ),
            parameters=[
                ParameterSpec(name="preamp", type="int",
                              description="前放编号",
                              required=True, min_value=0, max_value=7),
                ParameterSpec(name="channel", type="int",
                              description="通道号",
                              required=True, min_value=0, max_value=7),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["preamp", "mcva5", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        pre, ch = int(params["preamp"]), int(params["channel"])
        return _multi_read("GetPreamp", (
            ("gain", lambda: context.safe_call("MCVA5_GainGet", pre, ch)),
            ("coupling", lambda: context.safe_call("MCVA5_CouplingGet", pre, ch)),
            ("input_mode", lambda: context.safe_call("MCVA5_InputModeGet", pre, ch)),
        ), data={"preamp": pre, "channel": ch})


# ─────────────────────────────────────────────────────────────────────────────
# PLL analysis: Zoom FFT / phase sweep / signal analyser
# ─────────────────────────────────────────────────────────────────────────────

class RunPllZoomFft(BaseSkill):
    """Zoomed FFT of a PLL channel."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunPllZoomFft",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "打开 PLL Zoom-FFT 并在某个通道上启动它，带窗函数与平均。只读 —— FFT 不驱动任何东西。"
                "\n"
                "\n"
                "用它来看噪声本底和共振附近的杂散峰：zoom FFT 能分辨出普通频谱糊掉的那些结构。restart_averaging=true 会把旧的平均丢掉，"
                "改动过针尖或驱动之后你正需要这么做。"
            ),
            parameters=[
                ParameterSpec(name="channel_index", type="int",
                              description="要分析的 PLL 通道",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="fft_window", type="int",
                              description="FFT 窗类型序号（0 = 矩形窗；通常该用 Hann）",
                              required=False, default=1, min_value=0, max_value=8),
                ParameterSpec(name="averaging_mode", type="int",
                              description="平均模式序号",
                              required=False, default=0, min_value=0, max_value=4),
                ParameterSpec(name="weighting_mode", type="int",
                              description="加权模式序号",
                              required=False, default=0, min_value=0, max_value=4),
                ParameterSpec(name="count", type="int",
                              description="平均次数",
                              required=False, default=10, min_value=1, max_value=10_000),
                ParameterSpec(name="restart_averaging", type="bool",
                              description="丢弃已有的平均",
                              required=False, default=True),
            ],
            estimated_duration_s=3.0,
            composition_level=0,
            tags=["pll", "fft", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "RunPllZoomFft"
        calls: list = []
        rec = context.safe_call("PLLZoomFFT_Open")
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PLLZoomFFT_Open failed: {rec.error}", calls)
        rec = context.safe_call("PLLZoomFFT_ChSet", int(params["channel_index"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PLLZoomFFT_ChSet failed: {rec.error}", calls)
        rec = context.safe_call("PLLZoomFFT_PropsSet",
                                int(params.get("fft_window", 1) or 1),
                                int(params.get("averaging_mode", 0) or 0),
                                int(params.get("weighting_mode", 0) or 0),
                                int(params.get("count", 10) or 10))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PLLZoomFFT_PropsSet failed: {rec.error}", calls)
        if bool(params.get("restart_averaging", True)):
            rec = context.safe_call("PLLZoomFFT_AvgRestart")
            calls.append(rec)
            if rec.error:
                return _fail(name, f"PLLZoomFFT_AvgRestart failed: {rec.error}", calls)
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"channel_index": int(params["channel_index"])},
                           summary="PLL Zoom FFT 已启动 —— 用 GetPllZoomFftData 读频谱")


class GetPllZoomFftData(BaseSkill):
    """The zoom-FFT spectrum."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPllZoomFftData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 PLL Zoom-FFT 的频谱及其当前设置。",
            parameters=[],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["pll", "fft", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetPllZoomFftData", (
            ("spectrum", lambda: context.safe_call("PLLZoomFFT_DataGet")),
            ("props", lambda: context.safe_call("PLLZoomFFT_PropsGet")),
            ("channel", lambda: context.safe_call("PLLZoomFFT_ChGet")),
        ))


class RunPllPhaseSweep(BaseSkill):
    """Sweep the PLL phase to find the right operating point."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunPllPhaseSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在一个调制器上扫 PLL 的相位，并（可选地）返回得到的曲线。\n"
                "\n"
                "这是你找出 PLL 实际锁在哪个相位上的办法。扫描期间它**驱动激励** —— 一根悬臂或音叉会在整段扫描里被摇动 —— 但它不移动针尖。"
                "StopPllPhaseSweep 能中止它，按 中止 也能。"
            ),
            parameters=[
                ParameterSpec(name="modulator_index", type="int",
                              description="第几个 PLL 调制器（0 起算）",
                              required=False, default=0, min_value=0, max_value=7),
                ParameterSpec(name="get_data", type="bool",
                              description="是否返回扫描曲线",
                              required=False, default=True),
            ],
            estimated_duration_s=20.0,
            composition_level=0,
            tags=["pll", "phase", "sweep", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params.get("modulator_index", 0) or 0)
        rec = context.safe_call("PLLPhasSwp_Start", idx,
                                1 if bool(params.get("get_data", True)) else 0)
        if rec.error:
            return _fail("RunPllPhaseSweep", f"PLLPhasSwp_Start failed: {rec.error}", [rec])
        return SkillResult(skill_name="RunPllPhaseSweep", success=True, nanonis_calls=[rec],
                           data={"modulator_index": idx, "curve": _rv(rec)},
                           summary=f"PLL 相位扫描完成（调制器 {idx}）")


class StopPllPhaseSweep(BaseSkill):
    """Kill a running phase sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopPllPhaseSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "停止一次正在跑的 PLL 相位扫描。永远允许，包括在 abort 之后。"
            ),
            parameters=[
                ParameterSpec(name="modulator_index", type="int",
                              description="第几个 PLL 调制器（0 起算）",
                              required=False, default=0, min_value=0, max_value=7),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pll", "phase", "stop", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params.get("modulator_index", 0) or 0)
        rec = context.safe_call("PLLPhasSwp_Stop", idx)
        if rec.error:
            return _fail("StopPllPhaseSweep", f"PLLPhasSwp_Stop failed: {rec.error}", [rec])
        return SkillResult(skill_name="StopPllPhaseSweep", success=True, nanonis_calls=[rec],
                           data={"modulator_index": idx},
                           summary=f"PLL 相位扫描已停止（调制器 {idx}）")


class ConfigurePllSignalAnalyzer(BaseSkill):
    """The PLL signal analyser's channel, timebase, FFT and trigger."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigurePllSignalAnalyzer",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "配置 PLL 信号分析仪：用哪个通道、时基，以及 FFT 窗／平均。它就是架在 PLL 自身信号上的一台示波器 + FFT —— 只读，"
                "不驱动任何东西。\n"
                "\n"
                "用 GetPllSignalAnalyzerData 把时域波形和频谱一起取回来。"
            ),
            parameters=[
                ParameterSpec(name="channel_index", type="int",
                              description="要分析的 PLL 通道",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="timebase", type="float",
                              description="时基 —— 不传则保持不变",
                              required=False, default=None),
                ParameterSpec(name="update_rate", type="int",
                              description="刷新率 —— 不传则保持不变",
                              required=False, default=None, min_value=1),
                ParameterSpec(name="fft_window", type="int",
                              description="FFT 窗类型序号",
                              required=False, default=1, min_value=0, max_value=8),
                ParameterSpec(name="averaging_mode", type="int",
                              description="平均模式序号",
                              required=False, default=0, min_value=0, max_value=4),
                ParameterSpec(name="weighting_mode", type="int",
                              description="加权模式序号",
                              required=False, default=0, min_value=0, max_value=4),
                ParameterSpec(name="count", type="int",
                              description="平均次数",
                              required=False, default=10, min_value=1, max_value=10_000),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["pll", "analyzer", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigurePllSignalAnalyzer"
        calls: list = []
        rec = context.safe_call("PLLSignalAnlzr_Open")
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PLLSignalAnlzr_Open failed: {rec.error}", calls)

        rec = context.safe_call("PLLSignalAnlzr_ChSet", int(params["channel_index"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PLLSignalAnlzr_ChSet failed: {rec.error}", calls)

        tb, ur = params.get("timebase"), params.get("update_rate")
        if tb is not None or ur is not None:
            cur = context.safe_call("PLLSignalAnlzr_TimebaseGet")
            calls.append(cur)
            if cur.error:
                return _fail(name, f"PLLSignalAnlzr_TimebaseGet failed: {cur.error}", calls)
            got = _rv(cur)
            cur_tb, cur_ur = (got[0], got[1]) if isinstance(got, (list, tuple)) and len(got) >= 2 \
                else (0.0, 1)
            rec = context.safe_call("PLLSignalAnlzr_TimebaseSet",
                                    float(tb) if tb is not None else float(cur_tb),
                                    int(ur) if ur is not None else int(cur_ur))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"PLLSignalAnlzr_TimebaseSet failed: {rec.error}", calls)

        rec = context.safe_call("PLLSignalAnlzr_FFTPropsSet",
                                int(params.get("fft_window", 1) or 1),
                                int(params.get("averaging_mode", 0) or 0),
                                int(params.get("weighting_mode", 0) or 0),
                                int(params.get("count", 10) or 10))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"PLLSignalAnlzr_FFTPropsSet failed: {rec.error}", calls)

        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"channel_index": int(params["channel_index"])},
                           summary="PLL 信号分析仪已配置")


class GetPllSignalAnalyzerData(BaseSkill):
    """The analyser's time trace and spectrum."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPllSignalAnalyzerData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读 PLL 信号分析仪：示波器时域波形**和** FFT 频谱，外加触发状态。\n"
                "\n"
                "rearm=true 会在读之前重新武装触发 —— 触发不是自由运行时就该这么做，否则你拿回来的是上一次的捕获。"
            ),
            parameters=[
                ParameterSpec(name="rearm", type="bool",
                              description="先重新武装触发",
                              required=False, default=False),
            ],
            estimated_duration_s=3.0,
            composition_level=0,
            tags=["pll", "analyzer", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        if bool(params.get("rearm", False)):
            rec = context.safe_call("PLLSignalAnlzr_TrigRearm")
            calls.append(rec)
        result = _multi_read("GetPllSignalAnalyzerData", (
            ("trace", lambda: context.safe_call("PLLSignalAnlzr_OsciDataGet")),
            ("spectrum", lambda: context.safe_call("PLLSignalAnlzr_FFTDataGet")),
            ("trigger", lambda: context.safe_call("PLLSignalAnlzr_TrigGet")),
        ))
        result.nanonis_calls = calls + list(result.nanonis_calls or [])
        return result


# ─────────────────────────────────────────────────────────────────────────────
# OC Sync + tip recorder
# ─────────────────────────────────────────────────────────────────────────────

class ConfigureOcSync(BaseSkill):
    """OC Sync on/off angles."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureOcSync",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定 OC Sync 模块的开／关相位角，单位是**度**。它们把振荡控制的输出闸门化，"
                "使其只在振荡的某个相位窗内触发 —— 这是在振荡探针上做相位分辨（pump-probe 式）"
                "测量的基础。\n"
                "\n"
                "link_channels 会把通道 2 的角度绑到通道 1 上。"
            ),
            parameters=[
                ParameterSpec(name="ch1_on_deg", type="float",
                              description="通道 1 的 ON 相位角",
                              unit="deg", required=True, min_value=-360.0, max_value=360.0),
                ParameterSpec(name="ch1_off_deg", type="float",
                              description="通道 1 的 OFF 相位角",
                              unit="deg", required=True, min_value=-360.0, max_value=360.0),
                ParameterSpec(name="ch2_on_deg", type="float",
                              description="通道 2 的 ON 相位角",
                              unit="deg", required=True, min_value=-360.0, max_value=360.0),
                ParameterSpec(name="ch2_off_deg", type="float",
                              description="通道 2 的 OFF 相位角",
                              unit="deg", required=True, min_value=-360.0, max_value=360.0),
                ParameterSpec(name="link_channels", type="bool",
                              description="把通道 2 的角度绑到通道 1 上",
                              required=False, default=False),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["oc_sync", "phase", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureOcSync"
        calls: list = []
        rec = context.safe_call("OCSync_AnglesSet",
                                float(params["ch1_on_deg"]), float(params["ch1_off_deg"]),
                                float(params["ch2_on_deg"]), float(params["ch2_off_deg"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"OCSync_AnglesSet failed: {rec.error}", calls)
        link = 1 if bool(params.get("link_channels", False)) else 0
        rec = context.safe_call("OCSync_LinkAnglesSet", link, link)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"OCSync_LinkAnglesSet failed: {rec.error}", calls)
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"ch1_on_deg": float(params["ch1_on_deg"]),
                                 "ch1_off_deg": float(params["ch1_off_deg"])},
                           summary="OC Sync 相位角已设置")


class GetOcSync(BaseSkill):
    """OC Sync angles and links."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetOcSync",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 OC Sync 模块的相位角与通道绑定关系。",
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["oc_sync", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetOcSync", (
            ("angles", lambda: context.safe_call("OCSync_AnglesGet")),
            ("links", lambda: context.safe_call("OCSync_LinkAnglesGet")),
        ))


class ConfigureTipRecorder(BaseSkill):
    """Buffer size for the tip-move recorder."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureTipRecorder",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "设定针尖移动记录器的缓冲区大小，并可选地清空它。\n"
                "\n"
                "这个记录器把每一次针尖移动都记进一个环形缓冲区 —— 它是你重建针尖**去过哪里**的依据，"
                "而那正是一次无人值守运行出岔子之后你想要的东西。清空会把这段历史丢掉，所以要清就在一次运行**开始时**清，"
                "而不是在诊断它的时候清。"
            ),
            parameters=[
                ParameterSpec(name="buffer_size", type="int",
                              description="保留多少次针尖移动",
                              required=True, min_value=1, max_value=1_000_000),
                ParameterSpec(name="clear", type="bool",
                              description="清空缓冲区（会把移动历史丢掉）",
                              required=False, default=False),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["tip_recorder", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureTipRecorder"
        calls: list = []
        rec = context.safe_call("TipRec_BufferSizeSet", int(params["buffer_size"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"TipRec_BufferSizeSet failed: {rec.error}", calls)
        if bool(params.get("clear", False)):
            rec = context.safe_call("TipRec_BufferClear")
            calls.append(rec)
            if rec.error:
                return _fail(name, f"TipRec_BufferClear failed: {rec.error}", calls)
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"buffer_size": int(params["buffer_size"]),
                                 "cleared": bool(params.get("clear", False))},
                           summary=f"针尖记录器缓冲区 = {params['buffer_size']} 条")


class GetTipRecorderData(BaseSkill):
    """The recorded tip trajectory."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetTipRecorderData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从针尖记录器的缓冲区里读出记录下来的针尖移动历史。用它重建针尖去了哪里 —— 一次无人值守运行停在意料之外的地方时，"
                "这是第一个该看的东西。"
            ),
            parameters=[],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["tip_recorder", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return _multi_read("GetTipRecorderData", (
            ("moves", lambda: context.safe_call("TipRec_DataGet")),
            ("buffer_size", lambda: context.safe_call("TipRec_BufferSizeGet")),
        ))
