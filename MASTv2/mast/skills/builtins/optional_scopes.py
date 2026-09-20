"""Oscilloscopes — high-resolution (OsciHR), dual-channel (Osci2T), Signal Chart.

**Optional hardware.** Every skill here belongs to a module that ships OFF (设置 →
硬件模块). Off means these are not in the agent's tool list at all — see
``mast.skills.hardware_modules`` for why that is stronger than refusing the call.

Nanonis has THREE scopes and they are not interchangeable:

  * ``Osci1T`` — the plain 1-channel scope. Always present; MAST already wraps it
    (``experimental_features`` / signal capture). Not here.
  * ``Osci2T`` — 2-channel, for looking at two signals against each other on one
    timebase. The pump-probe workhorse.
  * ``OsciHR`` — the high-resolution scope: real triggering (level / digital,
    slope, hysteresis, pre-trigger, arm modes), oversampling, and a PSD section
    with windowing and averaging. This is the one you reach for when you want to
    SEE the noise, not just record it.

Shaped as tasks. ``ConfigureHighResScope`` takes the trigger, the sample count and
the oversampling together — the API wants ten separate Set calls to express one
decision ("trigger on channel 3, rising, at 50 pA, keep 1024 samples"), and making
the model chain ten calls is ten chances to get one wrong and no way to tell.
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

# OsciHR_TrigModeSet(Trigger_mode): 0=immediate, 1=level, 2=digital
_TRIG_MODES = {"immediate": 0, "level": 1, "digital": 2}
# slope: 0 = falling, 1 = rising
_SLOPES = {"falling": 0, "rising": 1}
# OsciHR_OsciDataGet(Osci_index, Data_to_get, Timeout_s):
#   Data_to_get 0 = current, 1 = next trigger, 2 = wait for next trigger
_DATA_NEXT_TRIGGER = 2


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """用 io.nanonis_files.decode_reply 提取回包内容，排除错误与原始字节信封。
    
    原始 bytes 不应进入需要 JSON 序列化的结果；所有调用方共用同一解码入口。"""
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


class ConfigureHighResScope(BaseSkill):
    """Trigger + sampling for the high-resolution oscilloscope, in one call."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureHighResScope",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "配置高分辨率示波器（OsciHR）：用哪个信号、采多少点、过采样，以及触发。\n"
                "\n"
                "配置示波器不会碰仪器 —— 它只决定**什么被数字化**。没有东西会动，也没有东西被施加出去。"
                "\n"
                "\n"
                "trigger_mode 为 'immediate' 时，一 Run 就立刻捕获；'level' 会等 trigger_channel 按给定方向穿过 trigger_level；"
                "'digital' 则等一条数字线。信号序号来自信号目录（ListSignalNames）"
                "。"
            ),
            parameters=[
                ParameterSpec(name="signal_index", type="int",
                              description="要数字化的信号（来自 ListSignalNames）",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="samples", type="int",
                              description="每次采集捕获的点数",
                              required=False, default=1024,
                              min_value=1, max_value=1_000_000),
                ParameterSpec(name="oversampling_index", type="int",
                              description="过采样序号（0 = 不过采样；越高平均越多、越慢）",
                              required=False, default=0, min_value=0, max_value=10),
                ParameterSpec(name="trigger_mode", type="str",
                              description="immediate | level | digital",
                              required=False, default="immediate",
                              allowed_values=list(_TRIG_MODES)),
                ParameterSpec(name="trigger_channel", type="int",
                              description="触发源通道（level 模式用）",
                              required=False, default=0, min_value=0, max_value=127),
                ParameterSpec(name="trigger_level", type="float",
                              description="触发阈值，用触发通道自己的物理单位",
                              required=False, default=0.0),
                ParameterSpec(name="trigger_slope", type="str",
                              description="rising | falling",
                              required=False, default="rising",
                              allowed_values=list(_SLOPES)),
                ParameterSpec(name="trigger_hysteresis", type="float",
                              description="触发迟滞（防止在噪声上反复触发）",
                              required=False, default=0.0, min_value=0.0),
                ParameterSpec(name="osci_index", type="int",
                              description="第几个 OsciHR 实例（0 起算）",
                              required=False, default=0, min_value=0, max_value=7),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["oscilloscope", "osci_hr", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureHighResScope"
        idx = int(params.get("osci_index", 0) or 0)
        mode = str(params.get("trigger_mode", "immediate") or "immediate")
        slope = str(params.get("trigger_slope", "rising") or "rising")
        calls: list = []

        rec = context.safe_call("OsciHR_ChSet", idx, int(params["signal_index"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"OsciHR_ChSet failed: {rec.error}", calls)

        rec = context.safe_call("OsciHR_SamplesSet", int(params.get("samples", 1024) or 1024))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"OsciHR_SamplesSet failed: {rec.error}", calls)

        rec = context.safe_call("OsciHR_OversamplSet",
                                int(params.get("oversampling_index", 0) or 0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"OsciHR_OversamplSet failed: {rec.error}", calls)

        rec = context.safe_call("OsciHR_TrigModeSet", _TRIG_MODES[mode])
        calls.append(rec)
        if rec.error:
            return _fail(name, f"OsciHR_TrigModeSet failed: {rec.error}", calls)

        if mode == "level":
            # Literal verbs — the safety tools grep safe_call("…"); a verb behind a
            # variable is invisible to every one of them.
            for verb, thunk in (
                ("OsciHR_TrigLevChSet", lambda: context.safe_call(
                    "OsciHR_TrigLevChSet", int(params.get("trigger_channel", 0) or 0))),
                ("OsciHR_TrigLevValSet", lambda: context.safe_call(
                    "OsciHR_TrigLevValSet", float(params.get("trigger_level", 0.0) or 0.0))),
                ("OsciHR_TrigLevSlopeSet", lambda: context.safe_call(
                    "OsciHR_TrigLevSlopeSet", _SLOPES[slope])),
                ("OsciHR_TrigLevHystSet", lambda: context.safe_call(
                    "OsciHR_TrigLevHystSet",
                    float(params.get("trigger_hysteresis", 0.0) or 0.0))),
            ):
                rec = thunk()
                calls.append(rec)
                if rec.error:
                    return _fail(name, f"{verb} failed: {rec.error}", calls)
        elif mode == "digital":
            rec = context.safe_call("OsciHR_TrigDigChSet",
                                    int(params.get("trigger_channel", 0) or 0))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"OsciHR_TrigDigChSet failed: {rec.error}", calls)
            rec = context.safe_call("OsciHR_TrigDigSlopeSet", _SLOPES[slope])
            calls.append(rec)
            if rec.error:
                return _fail(name, f"OsciHR_TrigDigSlopeSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"osci_index": idx, "signal_index": int(params["signal_index"]),
                  "samples": int(params.get("samples", 1024) or 1024),
                  "trigger_mode": mode},
            summary=(f"OsciHR#{idx} 已配置：信号 {params['signal_index']}，"
                     f"{params.get('samples', 1024)} 点，触发={mode}"),
        )


class RunHighResScope(BaseSkill):
    """Arm and run one acquisition."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunHighResScope",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "启动高分辨率示波器（并重新武装它的触发）。之后调 GetHighResScopeData 把波形取回来。"
                "\n"
                "\n"
                "对仪器而言这是只读的：跑示波器是把一个信号数字化，它不驱动任何东西。"
            ),
            parameters=[
                ParameterSpec(name="rearm", type="bool",
                              description="启动前先重新武装触发",
                              required=False, default=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["oscilloscope", "osci_hr", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        if bool(params.get("rearm", True)):
            rec = context.safe_call("OsciHR_TrigRearm")
            calls.append(rec)
            if rec.error:
                return _fail("RunHighResScope", f"OsciHR_TrigRearm failed: {rec.error}", calls)
        rec = context.safe_call("OsciHR_Run")
        calls.append(rec)
        if rec.error:
            return _fail("RunHighResScope", f"OsciHR_Run failed: {rec.error}", calls)
        return SkillResult(skill_name="RunHighResScope", success=True,
                           nanonis_calls=calls, summary="OsciHR 已启动")


class GetHighResScopeData(BaseSkill):
    """Collect the captured trace (and optionally the PSD)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetHighResScopeData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读高分辨率示波器捕获到的波形，并可选地读它的功率谱密度。\n"
                "\n"
                "wait_for_trigger=true 会在示波器上**阻塞**，直到下一次触发发生、"
                "或 timeout_s 到期 —— 触发是 level/digital 时，请在 RunHighResScope 之后这样用。"
                "timeout_s 给短了你会拿到一个错误，而不是半条波形。"
            ),
            parameters=[
                ParameterSpec(name="wait_for_trigger", type="bool",
                              description="等下一次触发，而不是取当前缓冲区",
                              required=False, default=True),
                ParameterSpec(name="timeout_s", type="float",
                              description="等触发最多等多久",
                              unit="s", required=False, default=10.0,
                              min_value=0.1, max_value=600.0),
                ParameterSpec(name="include_psd", type="bool",
                              description="同时读 PSD 部分",
                              required=False, default=False),
                ParameterSpec(name="osci_index", type="int",
                              description="第几个 OsciHR 实例（0 起算）",
                              required=False, default=0, min_value=0, max_value=7),
            ],
            estimated_duration_s=10.0,
            composition_level=0,
            tags=["oscilloscope", "osci_hr", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "GetHighResScopeData"
        idx = int(params.get("osci_index", 0) or 0)
        timeout = float(params.get("timeout_s", 10.0) or 10.0)
        mode = _DATA_NEXT_TRIGGER if bool(params.get("wait_for_trigger", True)) else 0
        calls: list = []

        rec = context.safe_call("OsciHR_OsciDataGet", idx, mode, timeout)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"OsciHR_OsciDataGet failed: {rec.error}", calls)
        data: dict = {"osci_index": idx, "trace": _rv(rec)}

        if bool(params.get("include_psd", False)):
            rec = context.safe_call("OsciHR_PSDDataGet", mode, timeout)
            calls.append(rec)
            if rec.error:
                # The trace IS in hand. Losing the PSD must not throw it away.
                data["psd_error"] = str(rec.error)
            else:
                data["psd"] = _rv(rec)

        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data=data, summary=f"OsciHR#{idx} 数据已取回")


class GetHighResScopeStatus(BaseSkill):
    """What the scope is currently set to."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetHighResScopeStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "回读高分辨率示波器当前的配置：通道、采样点数、过采样与触发模式。在信任一次由别人（或 Nanonis 界面）"
                "设好的捕获之前先读它。"
            ),
            parameters=[
                ParameterSpec(name="osci_index", type="int",
                              description="第几个 OsciHR 实例（0 起算）",
                              required=False, default=0, min_value=0, max_value=7),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["oscilloscope", "osci_hr", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params.get("osci_index", 0) or 0)
        calls: list = []
        data: dict = {"osci_index": idx}
        # Each read is independent — one unsupported getter must not blank the rest.
        # Literal verbs: every safety tool in this repo (abort-policy check,
        # security audit, API-coverage census) finds Nanonis calls by grepping
        # safe_call("…"). A verb behind a variable is invisible to all of them.
        for key, thunk in (
            ("signal_index", lambda: context.safe_call("OsciHR_ChGet", idx)),
            ("samples", lambda: context.safe_call("OsciHR_SamplesGet")),
            ("oversampling_index", lambda: context.safe_call("OsciHR_OversamplGet")),
            ("trigger_mode", lambda: context.safe_call("OsciHR_TrigModeGet")),
        ):
            rec = thunk()
            calls.append(rec)
            data[key] = None if rec.error else _rv(rec)
        return SkillResult(skill_name="GetHighResScopeStatus", success=True,
                           nanonis_calls=calls, data=data)


class ConfigureDualScope(BaseSkill):
    """Two channels, one timebase (Osci2T)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureDualScope",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "配置 2 通道示波器：两路信号、时基，以及触发。需要把两个信号放在**同一条**时基上互相对照时用它（一对 pump-probe、"
                "电流对偏压、锁相的 X/Y）。\n"
                "\n"
                "timebase_index 是从 Nanonis 固定的时基表里挑一项，它**不是**以秒为单位的时间。"
                "配置示波器不驱动任何东西。"
            ),
            parameters=[
                ParameterSpec(name="channel_a", type="int",
                              description="通道 A 的信号序号",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="channel_b", type="int",
                              description="通道 B 的信号序号",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="timebase_index", type="int",
                              description="时基序号（0 起算，取自 Nanonis 的固定表 —— 不是秒）",
                              required=False, default=0, min_value=0, max_value=15),
                ParameterSpec(name="trigger_mode", type="int",
                              description="0 = immediate，1 = level，2 = auto",
                              required=False, default=0, min_value=0, max_value=2),
                ParameterSpec(name="trigger_channel", type="int",
                              description="触发源：0 = 通道 A，1 = 通道 B",
                              required=False, default=0, min_value=0, max_value=1),
                ParameterSpec(name="trigger_slope", type="int",
                              description="0 = 下降沿，1 = 上升沿",
                              required=False, default=1, min_value=0, max_value=1),
                ParameterSpec(name="trigger_level", type="float",
                              description="触发阈值，用触发通道自己的单位",
                              required=False, default=0.0),
                ParameterSpec(name="trigger_hysteresis", type="float",
                              description="触发迟滞",
                              required=False, default=0.0, min_value=0.0),
                ParameterSpec(name="trigger_position", type="float",
                              description="触发点在记录中的位置（0..1）",
                              required=False, default=0.0, min_value=0.0, max_value=1.0),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["oscilloscope", "osci_2t", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureDualScope"
        calls: list = []

        rec = context.safe_call("Osci2T_ChSet", int(params["channel_a"]),
                                int(params["channel_b"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Osci2T_ChSet failed: {rec.error}", calls)

        rec = context.safe_call("Osci2T_TimebaseSet",
                                int(params.get("timebase_index", 0) or 0))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Osci2T_TimebaseSet failed: {rec.error}", calls)

        rec = context.safe_call(
            "Osci2T_TrigSet",
            int(params.get("trigger_mode", 0) or 0),
            int(params.get("trigger_channel", 0) or 0),
            int(params.get("trigger_slope", 1) or 1),
            float(params.get("trigger_level", 0.0) or 0.0),
            float(params.get("trigger_hysteresis", 0.0) or 0.0),
            float(params.get("trigger_position", 0.0) or 0.0),
        )
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Osci2T_TrigSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"channel_a": int(params["channel_a"]),
                  "channel_b": int(params["channel_b"])},
            summary=f"Osci2T 已配置：A={params['channel_a']} B={params['channel_b']}",
        )


class GetDualScopeData(BaseSkill):
    """Run the 2-channel scope and read both traces."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDualScopeData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "跑 2 通道示波器并把两路波形都读回来。先调 ConfigureDualScope 选定信号与时基。"
                "\n"
                "\n"
                "返回的通道 A 与通道 B 共用一条时间轴。"
            ),
            parameters=[
                ParameterSpec(name="run_first", type="bool",
                              description="读之前先启动一次新的采集",
                              required=False, default=True),
                ParameterSpec(name="data_to_get", type="int",
                              description="0 = 当前缓冲区，1 = 下一次触发，2 = 等待下一次触发",
                              required=False, default=1, min_value=0, max_value=2),
            ],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["oscilloscope", "osci_2t", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "GetDualScopeData"
        calls: list = []
        if bool(params.get("run_first", True)):
            rec = context.safe_call("Osci2T_Run")
            calls.append(rec)
            if rec.error:
                return _fail(name, f"Osci2T_Run failed: {rec.error}", calls)
        rec = context.safe_call("Osci2T_DataGet", int(params.get("data_to_get", 1) or 1))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"Osci2T_DataGet failed: {rec.error}", calls)
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"trace": _rv(rec)}, summary="Osci2T 双通道数据已取回")


class ConfigureSignalChart(BaseSkill):
    """Which two signals the Signal Chart shows."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureSignalChart",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "打开 Signal Chart 并设定它显示哪两路信号。这是 Nanonis 机器上的一个**显示**模块 —— 它改变的是用户在那边看到的东西，"
                "与测量本身无关。"
            ),
            parameters=[
                ParameterSpec(name="channel_a", type="int",
                              description="图表 A 的信号序号",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="channel_b", type="int",
                              description="图表 B 的信号序号",
                              required=True, min_value=0, max_value=127),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["signal_chart", "display", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureSignalChart"
        calls: list = []
        rec = context.safe_call("SignalChart_Open")
        calls.append(rec)
        if rec.error:
            return _fail(name, f"SignalChart_Open failed: {rec.error}", calls)
        rec = context.safe_call("SignalChart_ChsSet", int(params["channel_a"]),
                                int(params["channel_b"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"SignalChart_ChsSet failed: {rec.error}", calls)
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"channel_a": int(params["channel_a"]),
                                 "channel_b": int(params["channel_b"])},
                           summary="Signal Chart 已设置")
