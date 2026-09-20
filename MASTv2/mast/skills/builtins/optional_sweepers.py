"""High-Speed Sweeper (HSSwp) and the Arbitrary/RF Generator (APRFGen).

**Optional hardware, both OFF by default** (设置 → 硬件模块).

These two are the sharpest things in the optional set, and for opposite reasons:

  * **HSSwp** sweeps an *arbitrary signal* — you tell it which one by index. Point
    it at the bias and it is a fast bias sweep; point it at a piezo and it moves
    the tip. The module has no idea which you meant. So ``RunHighSpeedSweep`` is
    the sharpest thing here: a machine for driving whatever you name, quickly.
    It can also switch the Z-controller off for the duration (``ZCtrlOffSet``),
    which is exactly what you want for spectroscopy and exactly what ruins a tip
    if the sweep then drives Z.
  * **APRFGen** puts RF *power* out. ``StartRfGenerator`` is DANGEROUS not because
    it moves anything but because dBm into a tunnel junction is energy, and the
    agent has no feel for how much.

Both therefore carry an explicit stop skill, and both stop verbs (``HSSwp_Stop``,
``APRFGen_SwpStop``, plus ``APRFGen_RFOutOnOffSet`` in its OFF form) are on the
post-abort allow-list in ``core.execution_context`` — pressing 中止 must kill a
running sweep and cut the RF, not leave them running because the abort flag is up.
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

# APRFGen_FreqSwpStart(Direction) / PowerSwpStart / ListSwpStart: 0 = down, 1 = up
_DIRECTIONS = {"up": 1, "down": 0}
# APRFGen_RFOutOnOffSet(RF_Output): 0 = off, 1 = on
_RF_OFF, _RF_ON = 0, 1


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """用 io.nanonis_files.decode_reply 提取回包内容，排除错误与原始字节信封。
    
    原始 bytes 不应进入需要 JSON 序列化的结果；所有调用方共用同一解码入口。"""
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


def _channels(raw) -> list[int] | None:
    """'0, 2, 14' → [0, 2, 14]. None when unparseable."""
    try:
        out = [int(x) for x in str(raw).replace(",", " ").split()]
    except ValueError:
        return None
    return out or None


# ─────────────────────────────────────────────────────────────────────────────
# High-Speed Sweeper
# ─────────────────────────────────────────────────────────────────────────────

class ConfigureHighSpeedSweep(BaseSkill):
    """Sweep signal, limits, points, timing, channels — one decision, one call."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureHighSpeedSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置高速扫频器（High-Speed Sweeper）：扫哪个信号、扫什么范围、多少个点、"
                "时序，以及记录哪些通道。\n"
                "\n"
                "**用它之前先读这一段。** 被扫的信号是按**序号**从可扫信号表里挑的 —— 调 GetHighSpeedSweepStatus 才看得到那张表。"
                "HSSwp 你点名什么它就扫什么：偏压、一块压电、一路输出。它既不知道也不在乎是哪一个。"
                "序号点错，就是在高速驱动错误的东西。\n"
                "\n"
                "relative_limits=true 会让 start/stop 变成相对该信号当前值的**偏移量**（例如在当前偏压附近 ±0.5 V）"
                "；false 则是绝对值。把这个弄反，是扫到一个你从没打算去的值的经典途径。\n"
                "\n"
                "z_controller_off=true 会在整段扫描期间抬掉 Z 反馈 —— 做谱学时这是对的（针尖绝不能去追电流）"
                "，而如果被扫的信号会牵动 Z，它就是一份撞针风险。配置本身还什么都不改变；真正执行的是 RunHighSpeedSweep。"
            ),
            parameters=[
                ParameterSpec(name="sweep_signal_index", type="int",
                              description="**可扫信号表**里的序号（不是通用信号目录）",
                              required=True, min_value=0, max_value=127),
                ParameterSpec(name="start", type="float",
                              description="扫描起点（绝对值；relative_limits 时则为偏移量）",
                              required=True),
                ParameterSpec(name="stop", type="float",
                              description="扫描终点（绝对值；relative_limits 时则为偏移量）",
                              required=True),
                ParameterSpec(name="relative_limits", type="bool",
                              description="start/stop 是相对该信号**当前**值的偏移量",
                              required=False, default=False),
                ParameterSpec(name="points", type="int",
                              description="每次扫描的点数",
                              required=False, default=256, min_value=2, max_value=100_000),
                ParameterSpec(name="acquire_channels", type="str",
                              description="扫描期间要记录的信号序号，逗号分隔（例如 '0,24'）",
                              required=True),
                ParameterSpec(name="settling_time_s", type="float",
                              description="每点的稳定时间",
                              unit="s", required=False, default=1e-4,
                              min_value=0.0, max_value=10.0),
                ParameterSpec(name="integration_time_s", type="float",
                              description="每点的积分时间",
                              unit="s", required=False, default=1e-4,
                              min_value=0.0, max_value=10.0),
                ParameterSpec(name="initial_settling_time_s", type="float",
                              description="第一个点之前的稳定时间",
                              unit="s", required=False, default=1e-3,
                              min_value=0.0, max_value=60.0),
                ParameterSpec(name="max_slew_time_s", type="float",
                              description="允许用来爬到起始值的最长时间",
                              unit="s", required=False, default=1.0,
                              min_value=0.0, max_value=60.0),
                ParameterSpec(name="backward_sweep", type="bool",
                              description="同时反向扫一遍（stop → start）",
                              required=False, default=True),
                ParameterSpec(name="num_sweeps", type="int",
                              description="平均多少次扫描（0 = 一直连续扫到被停止为止）",
                              required=False, default=1, min_value=0, max_value=10_000),
                ParameterSpec(name="z_controller_off", type="bool",
                              description="扫描期间抬掉 Z 反馈（做谱学时是对的；如果被扫信号会牵动 Z，则是一份撞针风险）",
                              required=False, default=False),
                ParameterSpec(name="z_offset_m", type="float",
                              description="控制器关闭期间施加的 Z 偏移",
                              unit="m", required=False, default=0.0,
                              min_value=-1e-6, max_value=1e-6),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["sweep", "hs_sweeper", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureHighSpeedSweep"
        chs = _channels(params["acquire_channels"])
        if not chs:
            return _fail(name,
                         f"acquire_channels 解析失败：{params['acquire_channels']!r}"
                         "（应为逗号分隔的信号索引）", [])
        calls: list = []

        rec = context.safe_call("HSSwp_SwpChSignalSet",
                                int(params["sweep_signal_index"]), 0)  # 0 = not a timed sweep
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_SwpChSignalSet failed: {rec.error}", calls)

        rec = context.safe_call("HSSwp_SwpChLimitsSet",
                                1 if bool(params.get("relative_limits", False)) else 0,
                                float(params["start"]), float(params["stop"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_SwpChLimitsSet failed: {rec.error}", calls)

        rec = context.safe_call("HSSwp_SwpChNumPtsSet", int(params.get("points", 256) or 256))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_SwpChNumPtsSet failed: {rec.error}", calls)

        rec = context.safe_call(
            "HSSwp_SwpChTimingSet",
            float(params.get("initial_settling_time_s", 1e-3) or 1e-3),
            float(params.get("settling_time_s", 1e-4) or 1e-4),
            float(params.get("integration_time_s", 1e-4) or 1e-4),
            float(params.get("max_slew_time_s", 1.0) or 1.0),
        )
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_SwpChTimingSet failed: {rec.error}", calls)

        rec = context.safe_call("HSSwp_SwpChBwdSwSet",
                                1 if bool(params.get("backward_sweep", True)) else 0)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_SwpChBwdSwSet failed: {rec.error}", calls)

        n = int(params.get("num_sweeps", 1) or 1)
        rec = context.safe_call("HSSwp_NumSweepsSet", max(n, 1), 1 if n == 0 else 0)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_NumSweepsSet failed: {rec.error}", calls)

        rec = context.safe_call("HSSwp_AcqChsSet", chs)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_AcqChsSet failed: {rec.error}", calls)

        # HSSwp_ZCtrlOffSet(Z_Controller_Off, Z_Controller_Index, Z_Averaging_Time,
        #                   Z_Offset, Z_Control_Time)
        z_off = bool(params.get("z_controller_off", False))
        rec = context.safe_call("HSSwp_ZCtrlOffSet", 1 if z_off else 0, 0, 0.05,
                                float(params.get("z_offset_m", 0.0) or 0.0), 0.05)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_ZCtrlOffSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"sweep_signal_index": int(params["sweep_signal_index"]),
                  "start": float(params["start"]), "stop": float(params["stop"]),
                  "relative_limits": bool(params.get("relative_limits", False)),
                  "points": int(params.get("points", 256) or 256),
                  "acquire_channels": chs,
                  "z_controller_off": z_off},
            summary=(f"HSSwp 已配置：信号 {params['sweep_signal_index']}，"
                     f"{params['start']}→{params['stop']}"
                     f"（{'相对' if params.get('relative_limits') else '绝对'}），"
                     f"{params.get('points', 256)} 点"
                     + ("，Z 反馈将关闭" if z_off else "")),
        )


class RunHighSpeedSweep(BaseSkill):
    """Execute the configured sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunHighSpeedSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "执行由 ConfigureHighSpeedSweep 配好的那次扫描。它会把你点名的那路信号 —— 偏压、"
                "一块压电、一路输出 —— 从起点**驱动**到终点，而且很快。\n"
                "\n"
                "当心：扫频器对信号本身是不可知的。它会毫不犹豫地把一块压电开到硬限位上、或者把偏压拉过一个会毁掉针尖的范围，"
                "因为它就是被这么吩咐的。永远先 ConfigureHighSpeedSweep，并核对你点的到底是什么。"
                "\n"
                "\n"
                "wait=true 会阻塞到扫描结束（或 timeout_s 到期）并返回数据；wait=false 立刻返回，"
                "之后由你用 GetHighSpeedSweepStatus 轮询。StopHighSpeedSweep 能中止它，"
                "按 中止 也能。"
            ),
            parameters=[
                ParameterSpec(name="wait", type="bool",
                              description="阻塞直到扫描结束",
                              required=False, default=True),
                ParameterSpec(name="timeout_s", type="float",
                              description="wait=true 时最多等多久",
                              unit="s", required=False, default=60.0,
                              min_value=1.0, max_value=3600.0),
            ],
            estimated_duration_s=60.0,
            composition_level=0,
            tags=["sweep", "hs_sweeper", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "RunHighSpeedSweep"
        wait = bool(params.get("wait", True))
        timeout = float(params.get("timeout_s", 60.0) or 60.0)
        calls: list = []
        rec = context.safe_call("HSSwp_Start", 1 if wait else 0, timeout)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"HSSwp_Start failed: {rec.error}", calls)
        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"waited": wait, "result": _rv(rec)},
            summary="高速扫描完成" if wait else "高速扫描已启动（后台运行）",
        )


class StopHighSpeedSweep(BaseSkill):
    """Kill a running sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopHighSpeedSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "立即停止正在跑的高速扫描。永远允许 —— 包括在 abort 之后，那正是「停止」这件事存在的意义。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["sweep", "hs_sweeper", "stop", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("HSSwp_Stop")
        if rec.error:
            return _fail("StopHighSpeedSweep", f"HSSwp_Stop failed: {rec.error}", [rec])
        return SkillResult(skill_name="StopHighSpeedSweep", success=True,
                           nanonis_calls=[rec], summary="高速扫描已停止")


class GetHighSpeedSweepStatus(BaseSkill):
    """Is it running, and what is it set to?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetHighSpeedSweepStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读高速扫频器的状态：是否有扫描在跑、当前的扫描信号与限值，以及**可扫信号表**。\n"
                "\n"
                "请在 ConfigureHighSpeedSweep **之前**调它 —— 扫描信号是按序号从那张表里挑的，"
                "而那张表不是通用信号目录。序号靠猜，就是在扫错东西。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["sweep", "hs_sweeper", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        data: dict = {}
        # Literal verbs: every safety tool in this repo (abort-policy check,
        # security audit, API-coverage census) finds Nanonis calls by grepping
        # safe_call("…"). A verb behind a variable is invisible to all of them.
        for key, thunk in (
            ("status", lambda: context.safe_call("HSSwp_StatusGet")),
            ("sweepable_signals", lambda: context.safe_call("HSSwp_SwpChSigListGet")),
            ("sweep_signal", lambda: context.safe_call("HSSwp_SwpChSignalGet")),
            ("limits", lambda: context.safe_call("HSSwp_SwpChLimitsGet")),
            ("points", lambda: context.safe_call("HSSwp_SwpChNumPtsGet")),
            ("acquire_channels", lambda: context.safe_call("HSSwp_AcqChsGet")),
        ):
            rec = thunk()
            calls.append(rec)
            data[key] = None if rec.error else _rv(rec)
        return SkillResult(skill_name="GetHighSpeedSweepStatus", success=True,
                           nanonis_calls=calls, data=data)


# ─────────────────────────────────────────────────────────────────────────────
# Arbitrary / RF Generator
# ─────────────────────────────────────────────────────────────────────────────

class ConfigureRfGenerator(BaseSkill):
    """Frequency and power, without turning the output on."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureRfGenerator",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定 RF 源的频率与功率，但**不**打开输出。用它来预置一组设置、核对一遍，然后才调 StartRfGenerator。"
                "\n"
                "\n"
                "power_dbm 的单位是 dBm，这是一个**对数**标度：+10 dBm 是 0 dBm 的十倍功率，"
                "不是多百分之十。频率单位是 Hz —— 2.5 GHz 要写成 2.5e9，写 2.5 就是 2.5 Hz，"
                "而且不会有任何提示。"
            ),
            parameters=[
                ParameterSpec(name="frequency_hz", type="float",
                              description="RF 频率，单位**赫兹**（2.5 GHz = 2.5e9）",
                              unit="Hz", required=True,
                              min_value=0.0, max_value=4e10),
                ParameterSpec(name="power_dbm", type="float",
                              description="RF 功率，单位 dBm（对数标度）",
                              unit="dBm", required=True,
                              min_value=-100.0, max_value=30.0),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["rf", "aprf_gen", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureRfGenerator"
        calls: list = []
        # Force_RF_On = 0 on both: set the value, do NOT switch the output on. That
        # is the entire reason this skill is separate from StartRfGenerator.
        rec = context.safe_call("APRFGen_FreqSet", 0, float(params["frequency_hz"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"APRFGen_FreqSet failed: {rec.error}", calls)
        rec = context.safe_call("APRFGen_PowerSet", 0, float(params["power_dbm"]))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"APRFGen_PowerSet failed: {rec.error}", calls)
        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"frequency_hz": float(params["frequency_hz"]),
                  "power_dbm": float(params["power_dbm"]), "output_on": False},
            summary=(f"RF 已配置：{float(params['frequency_hz']):.6g} Hz，"
                     f"{float(params['power_dbm']):.4g} dBm（输出仍关闭）"),
        )


class StartRfGenerator(BaseSkill):
    """Switch the RF output ON."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StartRfGenerator",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "以当前配置好的频率与功率打开 RF 输出。\n"
                "\n"
                "DANGEROUS：打进隧道结里的 dBm 就是**能量**。RF 功率会耦合进针尖和样品，"
                "量够大就会改变、乃至毁掉这两者。先调 ConfigureRfGenerator 并确认那些数字 —— GetRfGeneratorStatus 显示的才是实际设进去的值。"
                "\n"
                "\n"
                "StopRfGenerator 能切断它，按 中止 也能。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["rf", "aprf_gen", "dangerous", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("APRFGen_RFOutOnOffSet", _RF_ON)
        if rec.error:
            return _fail("StartRfGenerator", f"APRFGen_RFOutOnOffSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="StartRfGenerator", success=True,
                           nanonis_calls=[rec], data={"output_on": True},
                           summary="RF 输出已开启")


class StopRfGenerator(BaseSkill):
    """Cut the RF output and any running sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopRfGenerator",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "停掉任何正在跑的 RF 扫描，**并且**关闭 RF 输出。永远允许，包括在 abort 之后。"
                "两件事都做，且按这个顺序 —— 只停扫描而不切断输出，会让 RF 停在扫描走到的那个值上继续输出。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["rf", "aprf_gen", "stop", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "StopRfGenerator"
        calls: list = []
        rec = context.safe_call("APRFGen_SwpStop")
        calls.append(rec)
        sweep_err = rec.error
        # Cut the output EVEN IF the sweep-stop failed. A failed sweep-stop is the
        # case where you most need the output off.
        rec = context.safe_call("APRFGen_RFOutOnOffSet", _RF_OFF)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"APRFGen_RFOutOnOffSet(off) failed: {rec.error}", calls)
        msg = "RF 扫描已停止，输出已关闭"
        if sweep_err:
            msg = f"RF 输出已关闭（扫描停止报错：{sweep_err}）"
        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"output_on": False}, summary=msg)


class RunRfFrequencySweep(BaseSkill):
    """Sweep the RF frequency between two limits."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunRfFrequencySweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "以当前功率把 RF 频率从一端扫到另一端，每个点驻留一段时间。\n"
                "\n"
                "DANGEROUS，理由与 StartRfGenerator 相同 —— 整段扫描期间它都在往外发 RF。"
                "请先用 ConfigureRfGenerator 把功率设好。\n"
                "\n"
                "限值单位是**赫兹**。dwell_s 是每点的驻留时间；points 是点数。总时长 ≈ points × dwell_s × repetitions。"
                "auto_off=true 会在扫描结束时切断 RF —— 除非你特别希望输出停在末频率上继续，"
                "否则就让它开着。"
            ),
            parameters=[
                ParameterSpec(name="lower_hz", type="float",
                              description="扫描下限，单位**赫兹**",
                              unit="Hz", required=True, min_value=0.0, max_value=4e10),
                ParameterSpec(name="upper_hz", type="float",
                              description="扫描上限，单位**赫兹**",
                              unit="Hz", required=True, min_value=0.0, max_value=4e10),
                ParameterSpec(name="points", type="int",
                              description="整段扫描的点数",
                              required=False, default=101, min_value=2, max_value=100_000),
                ParameterSpec(name="dwell_s", type="float",
                              description="每个点的驻留时间",
                              unit="s", required=False, default=0.01,
                              min_value=1e-6, max_value=60.0),
                ParameterSpec(name="repetitions", type="int",
                              description="这段扫描重复多少遍",
                              required=False, default=1, min_value=1, max_value=10_000),
                ParameterSpec(name="direction", type="str",
                              description="up（lower→upper）或 down",
                              required=False, default="up",
                              allowed_values=list(_DIRECTIONS)),
                ParameterSpec(name="auto_off", type="bool",
                              description="扫描结束时关闭 RF 输出",
                              required=False, default=True),
            ],
            estimated_duration_s=30.0,
            composition_level=0,
            tags=["rf", "aprf_gen", "sweep", "dangerous", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "RunRfFrequencySweep"
        lo = float(params["lower_hz"])
        hi = float(params["upper_hz"])
        if lo >= hi:
            return _fail(name, f"lower_hz ({lo:g}) 必须小于 upper_hz ({hi:g})", [])
        calls: list = []

        rec = context.safe_call("APRFGen_FreqSwpLimitsSet", lo, hi)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"APRFGen_FreqSwpLimitsSet failed: {rec.error}", calls)

        # APRFGen_FreqSwpPropsSet(Mode, Dwell_s, Repetitions, Infinite, Points,
        #                         Off_s, AutoOff). Infinite=0 always — an agent must
        #                         never start an unbounded RF sweep.
        auto_off = 1 if bool(params.get("auto_off", True)) else 0
        rec = context.safe_call(
            "APRFGen_FreqSwpPropsSet", 0,
            float(params.get("dwell_s", 0.01) or 0.01),
            float(params.get("repetitions", 1) or 1),
            0,
            int(params.get("points", 101) or 101),
            0.0, auto_off,
        )
        calls.append(rec)
        if rec.error:
            return _fail(name, f"APRFGen_FreqSwpPropsSet failed: {rec.error}", calls)

        direction = _DIRECTIONS[str(params.get("direction", "up") or "up")]
        rec = context.safe_call("APRFGen_FreqSwpStart", direction)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"APRFGen_FreqSwpStart failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"lower_hz": lo, "upper_hz": hi,
                  "points": int(params.get("points", 101) or 101),
                  "dwell_s": float(params.get("dwell_s", 0.01) or 0.01),
                  "auto_off": bool(params.get("auto_off", True))},
            summary=(f"RF 频率扫描已启动：{lo:.6g}→{hi:.6g} Hz，"
                     f"{params.get('points', 101)} 点"
                     + ("，结束后自动关输出" if auto_off else "，结束后输出保持开启")),
        )


class GetRfGeneratorStatus(BaseSkill):
    """Frequency, power, and whether the output is live."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetRfGeneratorStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读 RF 源的频率、功率，以及 —— 真正要紧的那一项 —— **输出是否开着**。在假定 RF 是关的之前，"
                "先核这一条。"
            ),
            parameters=[],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["rf", "aprf_gen", "read", "optional-hardware"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        data: dict = {}
        # Literal verbs: every safety tool in this repo (abort-policy check,
        # security audit, API-coverage census) finds Nanonis calls by grepping
        # safe_call("…"). A verb behind a variable is invisible to all of them.
        for key, thunk in (
            ("output_on", lambda: context.safe_call("APRFGen_RFOutOnOffGet")),
            ("frequency_hz", lambda: context.safe_call("APRFGen_FreqGet")),
            ("power_dbm", lambda: context.safe_call("APRFGen_PowerGet")),
            ("freq_sweep_limits", lambda: context.safe_call("APRFGen_FreqSwpLimitsGet")),
        ):
            rec = thunk()
            calls.append(rec)
            data[key] = None if rec.error else _rv(rec)
        return SkillResult(skill_name="GetRfGeneratorStatus", success=True,
                           nanonis_calls=calls, data=data)
