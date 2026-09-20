"""The last eight write-side holes: settings MAST could read and not change.

The 2026-07-13 census found the instrument surface was asymmetric in both directions.
37 settings could be written and not read back (fixed: readback.py, and every write
skill that mattered now verifies). These are the other eight — MAST could READ them
and had no way to SET them, so the agent could see a misconfiguration and do nothing
about it.

Individually small. Collectively they are the difference between an agent that can
observe the instrument and one that can operate it.
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


class SetWaveformSignal(BaseSkill):
    """Which signal the 2-channel function generator drives."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetWaveformSignal",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "选定一个函数发生器通道**驱动哪一路信号**。\n"
                "\n"
                "这个参数决定了波形实际上在做什么。同样一个 1 V 正弦波，接在空闲输出上是无害的测试信号，"
                "接在隧道结上就是 1 V 的偏压调制 —— 发生器分不出这两者的区别。先用 GetMiscInstrumentConfig.fungen2_signal 读一下当前的指派。"
                "\n"
                "\n"
                "波形本身用 ConfigureWaveform 配置；用 StartWaveform 启动。"
            ),
            parameters=[
                ParameterSpec(name="channel", type="int",
                              description="函数发生器通道（1 起算）",
                              required=True, min_value=1, max_value=2),
                ParameterSpec(name="signal_index", type="int",
                              description="该通道驱动的信号（来自 ListSignalNames）",
                              required=True, min_value=0, max_value=127),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["fungen", "waveform", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        ch = int(params["channel"])
        sig = int(params["signal_index"])
        rec = context.safe_call("FunGen2Ch_SignalSet", ch, sig)
        calls = [rec]
        if rec.error:
            return _fail("SetWaveformSignal", f"FunGen2Ch_SignalSet failed: {rec.error}", calls)
        back = context.safe_call("FunGen2Ch_SignalGet", ch)
        calls.append(back)
        return SkillResult(
            skill_name="SetWaveformSignal", success=True, nanonis_calls=calls,
            data={"channel": ch, "signal_index": sig,
                  "readback": None if back.error else _rv(back)},
            summary=f"函数发生器通道 {ch} 现在驱动信号 {sig}",
        )


class SetLockInDemodPhaseRegister(BaseSkill):
    """Which phase register a lock-in demodulator uses."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetLockInDemodPhaseRegister",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "设定某个锁相解调器参考**哪一个相位寄存器**。\n"
                "\n"
                "解调器的相位必须参考到产生该信号的那个调制上。指到错的寄存器不会失败 —— 它会把 X 转到 Y 里去，"
                "于是一条 dI/dV 就变成了一个看起来像 dI/dV、实际不是的东西。用 GetLockInConfig 读回来核对。"
            ),
            parameters=[
                ParameterSpec(name="demodulator", type="int",
                              description="解调器编号（1 起算）",
                              required=True, min_value=1, max_value=8),
                ParameterSpec(name="phase_register", type="int",
                              description="相位寄存器序号",
                              required=True, min_value=0, max_value=8),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["lockin", "demod", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        d = int(params["demodulator"])
        reg = int(params["phase_register"])
        rec = context.safe_call("LockIn_DemodPhasRegSet", d, reg)
        if rec.error:
            return _fail("SetLockInDemodPhaseRegister",
                         f"LockIn_DemodPhasRegSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetLockInDemodPhaseRegister", success=True,
                           nanonis_calls=[rec],
                           data={"demodulator": d, "phase_register": reg},
                           summary=f"解调器 {d} 的相位寄存器 = {reg}")


class SetLockInFrequencySweepSignal(BaseSkill):
    """Which signal the lock-in frequency sweep sweeps."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetLockInFrequencySweepSignal",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "选定锁相**频率扫描**扫的是哪一路信号。\n"
                "\n"
                "这个扫描会把你点名的东西在整个频率范围内驱动一遍 —— 找共振就是这么找的。点错信号，就是在驱动错的东西。"
                "当前值用 GetMiscInstrumentConfig.lockin_freqswp_signal 读。"
            ),
            parameters=[
                ParameterSpec(name="signal_index", type="int",
                              description="要扫的信号（来自 ListSignalNames）",
                              required=True, min_value=0, max_value=127),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["lockin", "sweep", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        sig = int(params["signal_index"])
        rec = context.safe_call("LockInFreqSwp_SignalSet", sig)
        if rec.error:
            return _fail("SetLockInFrequencySweepSignal",
                         f"LockInFreqSwp_SignalSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetLockInFrequencySweepSignal", success=True,
                           nanonis_calls=[rec], data={"signal_index": sig},
                           summary=f"锁相频率扫描的信号 = {sig}")


class SetPllExcitationAdd(BaseSkill):
    """Add the PLL's excitation to the output, or not."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPllExcitationAdd",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把 PLL 的激励信号叠加到它的输出上（或停止叠加）。\n"
                "\n"
                "Add 开着时，调制器的激励会到达悬臂／音叉 —— 探针正在被**驱动**。关掉时，环仍然在跟踪，"
                "但什么都不驱动。当前状态用 GetPllConfig.add_on_off 读。"
            ),
            parameters=[
                ParameterSpec(name="modulator", type="int",
                              description="调制器序号",
                              required=True, min_value=1, max_value=8),
                ParameterSpec(name="add", type="bool",
                              description="True = 激励会到达输出",
                              required=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pll", "excitation", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        m = int(params["modulator"])
        add = bool(params["add"])
        rec = context.safe_call("PLL_AddOnOffSet", m, 1 if add else 0)
        if rec.error:
            return _fail("SetPllExcitationAdd", f"PLL_AddOnOffSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetPllExcitationAdd", success=True,
                           nanonis_calls=[rec], data={"modulator": m, "add": add},
                           summary=f"PLL 调制器 {m} 的激励{'已接入输出' if add else '已断开'}")


class SetPllDemodHarmonic(BaseSkill):
    """Which harmonic a PLL demodulator locks to."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPllDemodHarmonic",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "设定某个 PLL 解调器锁到**第几次谐波**（1 = 基频）。\n"
                "\n"
                "更高次谐波携带的是关于针尖-样品相互作用的另一类信息，而一个锁在根本不存在的谐波上的解调器，"
                "会非常自信地报出噪声。用 GetPllConfig.demod_harmonic 读回来核对。"
            ),
            parameters=[
                ParameterSpec(name="demodulator", type="int",
                              description="解调器序号",
                              required=True, min_value=1, max_value=8),
                ParameterSpec(name="harmonic", type="int",
                              description="谐波次数（1 = 基频）",
                              required=True, min_value=1, max_value=16),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pll", "demod", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        d = int(params["demodulator"])
        h = int(params["harmonic"])
        rec = context.safe_call("PLL_DemodHarmonicSet", d, h)
        if rec.error:
            return _fail("SetPllDemodHarmonic", f"PLL_DemodHarmonicSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetPllDemodHarmonic", success=True,
                           nanonis_calls=[rec], data={"demodulator": d, "harmonic": h},
                           summary=f"PLL 解调器 {d} 锁定 {h} 次谐波")


class ConfigureScopeTrigger(BaseSkill):
    """The 1-channel scope's trigger."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureScopeTrigger",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "设定 1 通道示波器的触发：模式、沿、电平与迟滞。只读 —— 示波器做的是数字化，它不驱动任何东西。"
                "\n"
                "\n"
                "电平用的是被触发通道自己的物理单位。迟滞的作用，是让一路有噪声的信号不会在阈值附近每抖一下就重新触发一次。"
            ),
            parameters=[
                ParameterSpec(name="trigger_mode", type="int",
                              description="0 = immediate，1 = level，2 = auto",
                              required=False, default=1, min_value=0, max_value=2),
                ParameterSpec(name="trigger_slope", type="int",
                              description="0 = 下降沿，1 = 上升沿",
                              required=False, default=1, min_value=0, max_value=1),
                ParameterSpec(name="trigger_level", type="float",
                              description="阈值，用该通道自己的单位",
                              required=False, default=0.0),
                ParameterSpec(name="trigger_hysteresis", type="float",
                              description="迟滞（防止在噪声上反复触发）",
                              required=False, default=0.0, min_value=0.0),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["oscilloscope", "trigger", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # Osci1T_TrigSet(TriggerMode, TriggerSlope, TriggerLevel, TriggerHysteresis)
        rec = context.safe_call(
            "Osci1T_TrigSet",
            int(params.get("trigger_mode", 1) or 1),
            int(params.get("trigger_slope", 1) or 1),
            float(params.get("trigger_level", 0.0) or 0.0),
            float(params.get("trigger_hysteresis", 0.0) or 0.0),
        )
        if rec.error:
            return _fail("ConfigureScopeTrigger", f"Osci1T_TrigSet failed: {rec.error}", [rec])
        return SkillResult(
            skill_name="ConfigureScopeTrigger", success=True, nanonis_calls=[rec],
            data={"trigger_mode": int(params.get("trigger_mode", 1) or 1),
                  "trigger_level": float(params.get("trigger_level", 0.0) or 0.0)},
            summary="示波器触发已配置",
        )


class SetPatternExperiment(BaseSkill):
    """Which experiment a grid/line pattern runs at each point."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPatternExperiment",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "选定一个图案（网格／线／点云）在每个点上跑**哪一个实验**，外加文件基名与测量前延时。"
                "\n"
                "\n"
                "这个参数是把一组坐标变成一次测量的那一项。一个指向错误实验的网格会跑上几个小时，并在每一个点上产出错的东西 —— 而它此前是只读的（GetScanPatternConfig 看得见它；"
                "没有任何东西设得了它）。\n"
                "\n"
                "`pre_measure_delay_s` 是每个点测量之前的稳定时间；给短了，每条曲线都会拖着「走到这里」"
                "那段移动的尾巴。"
            ),
            parameters=[
                ParameterSpec(name="experiment", type="int",
                              description="实验序号（每个点上跑哪一种测量）",
                              required=True, min_value=0, max_value=31),
                ParameterSpec(name="basename", type="str",
                              description="存盘数据的文件基名",
                              required=False, default="mast_pattern"),
                ParameterSpec(name="pre_measure_delay_s", type="float",
                              description="每个点测量之前的稳定时间",
                              unit="s", required=False, default=0.1,
                              min_value=0.0, max_value=60.0),
                ParameterSpec(name="save_scan_channels", type="bool",
                              description="每个点上同时保存扫描通道",
                              required=False, default=False),
                ParameterSpec(name="external_vi_path", type="str",
                              description="外部 VI 路径（不用就留空）",
                              required=False, default=""),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["pattern", "grid", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # Pattern_PropsSet(Selected_experiment, Basename, External_VI_path,
        #                  Pre_measure_delay_s, Save_scan_channels)
        rec = context.safe_call(
            "Pattern_PropsSet",
            int(params["experiment"]),
            str(params.get("basename", "mast_pattern") or "mast_pattern"),
            str(params.get("external_vi_path", "") or ""),
            float(params.get("pre_measure_delay_s", 0.1) or 0.1),
            1 if bool(params.get("save_scan_channels", False)) else 0,
        )
        calls = [rec]
        if rec.error:
            return _fail("SetPatternExperiment", f"Pattern_PropsSet failed: {rec.error}", calls)
        back = context.safe_call("Pattern_PropsGet")
        calls.append(back)
        return SkillResult(
            skill_name="SetPatternExperiment", success=True, nanonis_calls=calls,
            data={"experiment": int(params["experiment"]),
                  "readback": None if back.error else _rv(back)},
            summary=f"图案实验 = {params['experiment']}",
        )


class SetPointShootProps(BaseSkill):
    """Point-and-shoot: what happens at each clicked point."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPointShootProps",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置 Follow-Me 的 point-and-shoot：测完之后扫描是否自动续跑、"
                "文件基名，以及测量前延时。\n"
                "\n"
                "`auto_resume` 是值得想一想的那一个。开着的话，每做完一次 point-and-shoot 测量扫描就会接着跑 —— 做普查时你要的正是这个，"
                "而如果那次测量可能改变了针尖，你要的就不是这个。"
            ),
            parameters=[
                ParameterSpec(name="auto_resume", type="bool",
                              description="每次点测之后续跑扫描",
                              required=True),
                ParameterSpec(name="basename", type="str",
                              description="文件基名",
                              required=False, default="mast_ps"),
                ParameterSpec(name="use_own_basename", type="bool",
                              description="用上面这个基名，而不是本次会话的基名",
                              required=False, default=True),
                ParameterSpec(name="pre_measure_delay_s", type="float",
                              description="每次测量之前的稳定时间",
                              unit="s", required=False, default=0.1,
                              min_value=0.0, max_value=60.0),
                ParameterSpec(name="external_vi_path", type="str",
                              description="外部 VI 路径（不用就留空）",
                              required=False, default=""),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["folme", "point-and-shoot", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # FolMe_PSPropsSet(Auto_resume, Use_own_basename, Basename, External_VI_path,
        #                  Pre_measure_delay_s)
        rec = context.safe_call(
            "FolMe_PSPropsSet",
            1 if bool(params["auto_resume"]) else 0,
            1 if bool(params.get("use_own_basename", True)) else 0,
            str(params.get("basename", "mast_ps") or "mast_ps"),
            str(params.get("external_vi_path", "") or ""),
            float(params.get("pre_measure_delay_s", 0.1) or 0.1),
        )
        calls = [rec]
        if rec.error:
            return _fail("SetPointShootProps", f"FolMe_PSPropsSet failed: {rec.error}", calls)
        back = context.safe_call("FolMe_PSPropsGet")
        calls.append(back)
        return SkillResult(
            skill_name="SetPointShootProps", success=True, nanonis_calls=calls,
            data={"auto_resume": bool(params["auto_resume"]),
                  "readback": None if back.error else _rv(back)},
            summary=f"point-and-shoot 已配置（扫描{'自动续跑' if params['auto_resume'] else '不自动续跑'}）",
        )
