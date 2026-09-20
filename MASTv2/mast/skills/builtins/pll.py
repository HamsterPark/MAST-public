"""PLL (Phase-Locked Loop) skills for qPlus/oscillation control.

vendored from v1 mast/skills/builtins/pll.py 2026-04-23. Zero behavioural changes.
36 skills: ConfigurePLL, GetPLLStatus, PLLOnOff, ConfigurePLLExcitation,
           AcquirePLLFreqSweep, PLLSignalAnalyzer, GetPLLAddOnOff,
           SetPLLAmpCtrlBandwidth, GetPLLAmpCtrlOnOff, SetPLLAmpCtrlSetpnt,
           GetPLLDemodFilter, SetPLLDemodFilter, GetPLLDemodHarmonic,
           GetPLLDemodInput, SetPLLDemodInput, SetPLLDemodPhasRef,
           GetPLLExcRange, SetPLLFreqExcOverwrite, GetPLLFreqRange,
           SetPLLFreqRange, PLLFreqShiftAutoCenter, GetPLLInpCalibr,
           SetPLLInpCalibr, GetPLLInpProps, SetPLLInpProps, SetPLLInpRange,
           PLLPerfectPLLUpdtZTC, SetPLLPhasCtrlBandwidth, GetPLLPhasCtrlOnOff,
           GetPLLSignalAnlzrCh, GetPLLSignalAnlzrFFTProps,
           GetPLLSignalAnlzrTimebase, PLLSignalAnlzrTrigAuto,
           SetPLLSignalAnlzrTrig, GetPLLFreqSwpParams, StopPLLFreqSwp.
"""

from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class ConfigurePLL(BaseSkill):
    """Configure PLL frequency and controller gains."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigurePLL",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 PLL 的中心频率、频移与控制器增益。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="center_freq_hz",
                    type="float",
                    description="中心频率，单位 Hz",
                    unit="Hz",
                    required=False,
                ),
                ParameterSpec(
                    name="freq_shift_hz",
                    type="float",
                    description="频移，单位 Hz",
                    unit="Hz",
                    required=False,
                ),
                ParameterSpec(
                    name="amp_p_gain",
                    type="float",
                    description="幅度控制器的 P 增益（V/m）",
                    required=False,
                ),
                ParameterSpec(
                    name="amp_time_constant_s",
                    type="float",
                    description="幅度控制器的时间常数",
                    unit="s",
                    required=False,
                ),
                ParameterSpec(
                    name="phas_p_gain",
                    type="float",
                    description="相位控制器的 P 增益（Hz/deg）",
                    required=False,
                ),
                ParameterSpec(
                    name="phas_time_constant_s",
                    type="float",
                    description="相位控制器的时间常数",
                    unit="s",
                    required=False,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["pll", "configure", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        calls = []

        # NOTE: use ``params.get(k) is not None`` rather than ``k in params``.
        # In the agent path the pydantic args_schema materialises every declared
        # optional parameter as a key with value None, so ``"center_freq_hz" in
        # params`` is ALWAYS True and would push None to the hardware
        # (PLL_CenterFreqSet(mod, None), etc.). Gating on a non-None value
        # restores "only write what the caller actually supplied".
        if params.get("center_freq_hz") is not None:
            rec = context.safe_call("PLL_CenterFreqSet", mod, params["center_freq_hz"])
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigurePLL", success=False,
                                   error=rec.error, nanonis_calls=calls)

        if params.get("freq_shift_hz") is not None:
            rec = context.safe_call("PLL_FreqShiftSet", mod, params["freq_shift_hz"])
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigurePLL", success=False,
                                   error=rec.error, nanonis_calls=calls)

        if params.get("amp_p_gain") is not None:
            tc = params.get("amp_time_constant_s")
            rec = context.safe_call("PLL_AmpCtrlGainSet", mod,
                                    params["amp_p_gain"],
                                    tc if tc is not None else 1e-3)
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigurePLL", success=False,
                                   error=rec.error, nanonis_calls=calls)

        if params.get("phas_p_gain") is not None:
            tc = params.get("phas_time_constant_s")
            rec = context.safe_call("PLL_PhasCtrlGainSet", mod,
                                    params["phas_p_gain"],
                                    tc if tc is not None else 1e-3)
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigurePLL", success=False,
                                   error=rec.error, nanonis_calls=calls)

        return SkillResult(
            skill_name="ConfigurePLL",
            success=True,
            data={"modulator_index": mod},
            nanonis_calls=calls,
        )


class GetPLLStatus(BaseSkill):
    """Read PLL status and parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 PLL 当前状态：频率、增益、excitation。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["pll", "status", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        calls = []
        data: dict = {"modulator_index": mod}

        rec = context.safe_call("PLL_CenterFreqGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            data["center_freq_hz"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        rec = context.safe_call("PLL_FreqShiftGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            data["freq_shift_hz"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        rec = context.safe_call("PLL_AmpCtrlGainGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                data["amp_p_gain"] = float(v[0])
                data["amp_time_constant_s"] = float(v[1])
                data["amp_i_gain"] = float(v[2])

        rec = context.safe_call("PLL_PhasCtrlGainGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                data["phas_p_gain"] = float(v[0])
                data["phas_time_constant_s"] = float(v[1])

        rec = context.safe_call("PLL_ExcitationGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            data["excitation_v"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        return SkillResult(
            skill_name="GetPLLStatus",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class PLLOnOff(BaseSkill):
    """Turn PLL output and controllers on or off."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PLLOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="开或关 PLL 输出、相位控制器与幅度控制器。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="output_on",
                    type="bool",
                    description="启用 PLL 输出",
                    required=True,
                ),
                ParameterSpec(
                    name="phase_ctrl_on",
                    type="bool",
                    description="启用相位控制器",
                    required=False,
                    default=True,
                ),
                ParameterSpec(
                    name="amp_ctrl_on",
                    type="bool",
                    description="启用幅度控制器",
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=1,
            tags=["pll", "onoff", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        calls = []

        rec = context.safe_call("PLL_OutOnOffSet", mod, int(params["output_on"]))
        calls.append(rec)
        if rec.error:
            return SkillResult(skill_name="PLLOnOff", success=False,
                               error=rec.error, nanonis_calls=calls)

        rec = context.safe_call("PLL_PhasCtrlOnOffSet", mod,
                                int(params.get("phase_ctrl_on", True)))
        calls.append(rec)

        rec = context.safe_call("PLL_AmpCtrlOnOffSet", mod,
                                int(params.get("amp_ctrl_on", True)))
        calls.append(rec)

        return SkillResult(
            skill_name="PLLOnOff",
            success=True,
            data={
                "output_on": params["output_on"],
                "phase_ctrl_on": params.get("phase_ctrl_on", True),
                "amp_ctrl_on": params.get("amp_ctrl_on", True),
            },
            nanonis_calls=calls,
        )


class ConfigurePLLExcitation(BaseSkill):
    """Configure PLL excitation parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigurePLLExcitation",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 PLL 的 excitation 幅度与输出量程。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="excitation_v",
                    type="float",
                    description="excitation 幅度，单位伏特",
                    unit="V",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="output_range",
                    type="float",
                    description="excitation 输出量程",
                    required=False,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=1,
            tags=["pll", "excitation", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        calls = []

        rec = context.safe_call("PLL_ExcitationSet", mod, params["excitation_v"])
        calls.append(rec)
        if rec.error:
            return SkillResult(skill_name="ConfigurePLLExcitation", success=False,
                               error=rec.error, nanonis_calls=calls)

        # ``params.get(k) is not None`` — not ``k in params``: the agent-path
        # pydantic schema materialises the omitted optional ``output_range`` as
        # None, so the membership test would always fire and call
        # PLL_ExcRangeSet(mod, None).
        if params.get("output_range") is not None:
            rec = context.safe_call("PLL_ExcRangeSet", mod, params["output_range"])
            calls.append(rec)

        return SkillResult(
            skill_name="ConfigurePLLExcitation",
            success=True,
            data={"excitation_v": params["excitation_v"], "modulator_index": mod},
            nanonis_calls=calls,
        )


class AcquirePLLFreqSweep(BaseSkill):
    """Acquire a PLL frequency sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquirePLLFreqSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="跑一次 PLL 扫频，找出共振峰。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="num_points",
                    type="int",
                    description="频率点数",
                    required=True,
                    min_value=2,
                    max_value=10000,
                ),
                ParameterSpec(
                    name="period_s",
                    type="float",
                    description="每个频率点上的测量时间",
                    unit="s",
                    required=True,
                    min_value=0.001,
                ),
                ParameterSpec(
                    name="settling_time_s",
                    type="float",
                    description="设定起始频率之后的等待时间",
                    unit="s",
                    required=False,
                    default=0.1,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="sweep_up",
                    type="bool",
                    description="从下限扫到上限",
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=60.0,
            composition_level=1,
            tags=["pll", "sweep", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        calls = []

        rec = context.safe_call("PLLFreqSwp_Open", mod)
        calls.append(rec)
        if rec.error:
            return SkillResult(skill_name="AcquirePLLFreqSweep", success=False,
                               error=rec.error, nanonis_calls=calls)

        rec = context.safe_call("PLLFreqSwp_ParamsSet", mod,
                                params["num_points"],
                                params["period_s"],
                                params.get("settling_time_s", 0.1))
        calls.append(rec)
        if rec.error:
            return SkillResult(skill_name="AcquirePLLFreqSweep", success=False,
                               error=rec.error, nanonis_calls=calls)

        direction = 1 if params.get("sweep_up", True) else 0
        rec = context.safe_call("PLLFreqSwp_Start", mod, 1, direction)
        calls.append(rec)
        if rec.error:
            return SkillResult(skill_name="AcquirePLLFreqSweep", success=False,
                               error=rec.error, nanonis_calls=calls)

        data: dict = {"completed": True}
        parsed = rec.return_value
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 8:
                data["resonance_freq_hz"] = float(vals[6])
                data["q_factor"] = float(vals[7])
                # 落盘(2026-07-31):在此之前这两个数只活在这一次的 SkillResult 里,
                # 想知道当前音叉的 f₀/Q 就得重扫一次 —— 而它们是判断「这支传感器
                # 还好不好」的基本量。写回 instrument_profile 的**实测**槽位
                # (与针尖行上的标称值分开存),换针尖时自动清除。
                # Best-effort:记账失败绝不能弄坏一次成功的扫描。
                try:
                    from mast.core.instrument_profile import set_qplus_resonance
                    set_qplus_resonance(data["resonance_freq_hz"], data["q_factor"])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("qPlus 共振写回失败(不影响本次扫描): %s", exc)

        return SkillResult(
            skill_name="AcquirePLLFreqSweep",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class PLLSignalAnalyzer(BaseSkill):
    """Acquire PLL signal analyzer data (oscilloscope and FFT)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PLLSignalAnalyzer",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="打开 PLL 信号分析仪，并采集示波器/FFT 数据。",
            parameters=[
                ParameterSpec(
                    name="channel_index",
                    type="int",
                    description="要分析的信号通道索引",
                    required=True,
                    min_value=0,
                ),
                ParameterSpec(
                    name="get_fft",
                    type="bool",
                    description="同时采集 FFT 数据",
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=1,
            tags=["pll", "analyzer", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []

        rec = context.safe_call("PLLSignalAnlzr_Open")
        calls.append(rec)
        if rec.error:
            return SkillResult(skill_name="PLLSignalAnalyzer", success=False,
                               error=rec.error, nanonis_calls=calls)

        rec = context.safe_call("PLLSignalAnlzr_ChSet", params["channel_index"])
        calls.append(rec)
        if rec.error:
            return SkillResult(skill_name="PLLSignalAnalyzer", success=False,
                               error=rec.error, nanonis_calls=calls)

        rec_osci = context.safe_call("PLLSignalAnlzr_OsciDataGet")
        calls.append(rec_osci)
        data: dict = {"channel_index": params["channel_index"]}
        if not rec_osci.error:
            data["osci_data"] = str(rec_osci.return_value)

        if params.get("get_fft", False):
            rec_fft = context.safe_call("PLLSignalAnlzr_FFTDataGet")
            calls.append(rec_fft)
            if not rec_fft.error:
                data["fft_data"] = str(rec_fft.return_value)

        return SkillResult(
            skill_name="PLLSignalAnalyzer",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class GetPLLAddOnOff(BaseSkill):
    """Read the Add external signal to output status."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLAddOnOff",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回 Add external signal to output 是开还是关。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "add", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_AddOnOffGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLAddOnOff", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["add_on"] = bool(int(v[0]) if isinstance(v, (list, tuple)) else int(v))
        return SkillResult(skill_name="GetPLLAddOnOff", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLAmpCtrlBandwidth(BaseSkill):
    """Set the amplitude controller bandwidth."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLAmpCtrlBandwidth",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置幅度控制器的带宽。使用当前的 Q factor 与幅度对 excitation "
                "之比（取自之前的一次扫频）。"
            ),
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="bandwidth_hz",
                    type="float",
                    description="带宽，单位 Hz",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "amplitude", "bandwidth", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_AmpCtrlBandwidthSet", mod, params["bandwidth_hz"])
        if record.error:
            return SkillResult(skill_name="SetPLLAmpCtrlBandwidth", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLAmpCtrlBandwidth", success=True,
            data={"modulator_index": mod, "bandwidth_hz": params["bandwidth_hz"]},
            nanonis_calls=[record],
        )


class GetPLLAmpCtrlOnOff(BaseSkill):
    """Read the amplitude controller on/off status."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLAmpCtrlOnOff",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回幅度控制器是开还是关。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "amplitude", "onoff", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_AmpCtrlOnOffGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLAmpCtrlOnOff", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["amp_ctrl_on"] = bool(int(v[0]) if isinstance(v, (list, tuple)) else int(v))
        return SkillResult(skill_name="GetPLLAmpCtrlOnOff", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLAmpCtrlSetpnt(BaseSkill):
    """Set the amplitude controller setpoint."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLAmpCtrlSetpnt",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置幅度控制器的 setpoint，单位米。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="setpoint_m",
                    type="float",
                    description="幅度 setpoint，单位米",
                    unit="m",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "amplitude", "setpoint", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_AmpCtrlSetpntSet", mod, params["setpoint_m"])
        if record.error:
            return SkillResult(skill_name="SetPLLAmpCtrlSetpnt", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLAmpCtrlSetpnt", success=True,
            data={"modulator_index": mod, "setpoint_m": params["setpoint_m"]},
            nanonis_calls=[record],
        )


class GetPLLDemodFilter(BaseSkill):
    """Read the demodulator low-pass filter order."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLDemodFilter",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回 PLL lock-in 之后那个低通滤波器的阶数。",
            parameters=[
                ParameterSpec(
                    name="demodulator_index",
                    type="int",
                    description="解调器索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "demod", "filter", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator_index", 1)
        record = context.safe_call("PLL_DemodFilterGet", demod)
        if record.error:
            return SkillResult(skill_name="GetPLLDemodFilter", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"demodulator_index": demod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["filter_order"] = int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        return SkillResult(skill_name="GetPLLDemodFilter", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLDemodFilter(BaseSkill):
    """Set the demodulator low-pass filter order."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLDemodFilter",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 PLL lock-in 之后那个低通滤波器的阶数。",
            parameters=[
                ParameterSpec(
                    name="demodulator_index",
                    type="int",
                    description="解调器索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="filter_order",
                    type="int",
                    description="滤波器阶数（unsigned int16）",
                    required=True,
                    min_value=0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "demod", "filter", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator_index", 1)
        record = context.safe_call("PLL_DemodFilterSet", demod, params["filter_order"])
        if record.error:
            return SkillResult(skill_name="SetPLLDemodFilter", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLDemodFilter", success=True,
            data={"demodulator_index": demod, "filter_order": params["filter_order"]},
            nanonis_calls=[record],
        )


class GetPLLDemodHarmonic(BaseSkill):
    """Read which harmonic the PLL lock-in demodulator is set to."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLDemodHarmonic",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回 PLL lock-in 解调器中选中的谐波。",
            parameters=[
                ParameterSpec(
                    name="demodulator_index",
                    type="int",
                    description="解调器索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "demod", "harmonic", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator_index", 1)
        record = context.safe_call("PLL_DemodHarmonicGet", demod)
        if record.error:
            return SkillResult(skill_name="GetPLLDemodHarmonic", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"demodulator_index": demod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["harmonic"] = int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        return SkillResult(skill_name="GetPLLDemodHarmonic", success=True,
                           data=data, nanonis_calls=[record])


class GetPLLDemodInput(BaseSkill):
    """Read the demodulator input and frequency generator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLDemodInput",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回选中解调器的输入与频率发生器。",
            parameters=[
                ParameterSpec(
                    name="demodulator_index",
                    type="int",
                    description="解调器索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "demod", "input", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator_index", 1)
        record = context.safe_call("PLL_DemodInputGet", demod)
        if record.error:
            return SkillResult(skill_name="GetPLLDemodInput", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"demodulator_index": demod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                data["input"] = int(v[0])
                data["frequency_generator"] = int(v[1])
        return SkillResult(skill_name="GetPLLDemodInput", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLDemodInput(BaseSkill):
    """Set the demodulator input and frequency generator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLDemodInput",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置选中解调器的输入与频率发生器。",
            parameters=[
                ParameterSpec(
                    name="demodulator_index",
                    type="int",
                    description="解调器索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="input",
                    type="int",
                    description="输入索引（0 = 不改）",
                    required=True,
                ),
                ParameterSpec(
                    name="frequency_generator",
                    type="int",
                    description="频率发生器索引（0 = 不改）",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "demod", "input", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator_index", 1)
        record = context.safe_call("PLL_DemodInputSet", demod,
                                   params["input"], params["frequency_generator"])
        if record.error:
            return SkillResult(skill_name="SetPLLDemodInput", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLDemodInput", success=True,
            data={
                "demodulator_index": demod,
                "input": params["input"],
                "frequency_generator": params["frequency_generator"],
            },
            nanonis_calls=[record],
        )


class SetPLLDemodPhasRef(BaseSkill):
    """Set the demodulator phase reference."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLDemodPhasRef",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置选中解调器的参考相位。",
            parameters=[
                ParameterSpec(
                    name="demodulator_index",
                    type="int",
                    description="解调器索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="phase_reference_deg",
                    type="float",
                    description="参考相位，单位度",
                    unit="deg",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "demod", "phase", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator_index", 1)
        record = context.safe_call("PLL_DemodPhasRefSet", demod,
                                   params["phase_reference_deg"])
        if record.error:
            return SkillResult(skill_name="SetPLLDemodPhasRef", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLDemodPhasRef", success=True,
            data={
                "demodulator_index": demod,
                "phase_reference_deg": params["phase_reference_deg"],
            },
            nanonis_calls=[record],
        )


class GetPLLExcRange(BaseSkill):
    """Read the PLL excitation output range."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLExcRange",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "返回 excitation 输出量程的索引（0=10V, 1=1V, 2=0.1V, 3=0.01V, 4=0.001V）。"
            ),
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "excitation", "range", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_ExcRangeGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLExcRange", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        range_map = {0: "10V", 1: "1V", 2: "0.1V", 3: "0.01V", 4: "0.001V"}
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            idx = int(v[0]) if isinstance(v, (list, tuple)) else int(v)
            data["output_range_index"] = idx
            data["output_range"] = range_map.get(idx, f"unknown({idx})")
        return SkillResult(skill_name="GetPLLExcRange", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLFreqExcOverwrite(BaseSkill):
    """Set the frequency shift and/or excitation overwrite signals."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLFreqExcOverwrite",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置用于覆写 Frequency Shift 和/或 Excitation 的信号。仅在对应控制器未激活时有效。"
                "不改则填 -2。"
            ),
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="excitation_overwrite_index",
                    type="int",
                    description="excitation 覆写信号的索引（-2 = 不改）",
                    required=True,
                ),
                ParameterSpec(
                    name="frequency_overwrite_index",
                    type="int",
                    description="频率覆写信号的索引（-2 = 不改）",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "overwrite", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        exc_idx = params["excitation_overwrite_index"]
        freq_idx = params["frequency_overwrite_index"]
        record = context.safe_call("PLL_FreqExcOverwriteSet", mod, exc_idx, freq_idx)
        if record.error:
            return SkillResult(skill_name="SetPLLFreqExcOverwrite", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLFreqExcOverwrite", success=True,
            data={
                "modulator_index": mod,
                "excitation_overwrite_index": exc_idx,
                "frequency_overwrite_index": freq_idx,
            },
            nanonis_calls=[record],
        )


class GetPLLFreqRange(BaseSkill):
    """Read the PLL frequency range."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLFreqRange",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回振荡控制模块的频率量程。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "frequency", "range", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_FreqRangeGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLFreqRange", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["frequency_range_hz"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)
        return SkillResult(skill_name="GetPLLFreqRange", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLFreqRange(BaseSkill):
    """Set the PLL frequency range."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLFreqRange",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置振荡控制模块的频率量程。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="frequency_range_hz",
                    type="float",
                    description="频率量程，单位 Hz",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "frequency", "range", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_FreqRangeSet", mod, params["frequency_range_hz"])
        if record.error:
            return SkillResult(skill_name="SetPLLFreqRange", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLFreqRange", success=True,
            data={"modulator_index": mod, "frequency_range_hz": params["frequency_range_hz"]},
            nanonis_calls=[record],
        )


class PLLFreqShiftAutoCenter(BaseSkill):
    """Auto-center the PLL frequency shift."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PLLFreqShiftAutoCenter",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "频移自动归中：把当前频移加到中心频率上，并把频移复位为零。"
            ),
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "frequency", "autocenter", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_FreqShiftAutoCenter", mod)
        if record.error:
            return SkillResult(skill_name="PLLFreqShiftAutoCenter", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="PLLFreqShiftAutoCenter", success=True,
            data={"modulator_index": mod},
            nanonis_calls=[record],
        )


class GetPLLInpCalibr(BaseSkill):
    """Read the PLL input calibration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLInpCalibr",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回振荡控制模块的输入标定（m/V）。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "input", "calibration", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_InpCalibrGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLInpCalibr", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["calibration_m_per_v"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)
        return SkillResult(skill_name="GetPLLInpCalibr", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLInpCalibr(BaseSkill):
    """Set the PLL input calibration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLInpCalibr",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置振荡控制模块的输入标定（m/V）。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="calibration_m_per_v",
                    type="float",
                    description="输入标定，单位 m/V",
                    unit="m/V",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "input", "calibration", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_InpCalibrSet", mod, params["calibration_m_per_v"])
        if record.error:
            return SkillResult(skill_name="SetPLLInpCalibr", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLInpCalibr", success=True,
            data={"modulator_index": mod, "calibration_m_per_v": params["calibration_m_per_v"]},
            nanonis_calls=[record],
        )


class GetPLLInpProps(BaseSkill):
    """Read the PLL input properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLInpProps",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回 PLL 的输入属性（差分输入、1/10 分压器）。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "input", "properties", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_InpPropsGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLInpProps", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                data["differential_input"] = bool(int(v[0]))
                data["divider_1_10"] = bool(int(v[1]))
        return SkillResult(skill_name="GetPLLInpProps", success=True,
                           data=data, nanonis_calls=[record])


class SetPLLInpProps(BaseSkill):
    """Set the PLL input properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLInpProps",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 PLL 的输入属性（差分输入、1/10 分压器）。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="differential_input",
                    type="bool",
                    description="启用差分输入",
                    required=True,
                ),
                ParameterSpec(
                    name="divider_1_10",
                    type="bool",
                    description="启用 1/10 分压器",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "input", "properties", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_InpPropsSet", mod,
                                   int(params["differential_input"]),
                                   int(params["divider_1_10"]))
        if record.error:
            return SkillResult(skill_name="SetPLLInpProps", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLInpProps", success=True,
            data={
                "modulator_index": mod,
                "differential_input": params["differential_input"],
                "divider_1_10": params["divider_1_10"],
            },
            nanonis_calls=[record],
        )


class SetPLLInpRange(BaseSkill):
    """Set the PLL input range."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLInpRange",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置振荡控制模块的输入量程（m）。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="input_range_m",
                    type="float",
                    description="输入量程，单位米",
                    unit="m",
                    required=True,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "input", "range", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_InpRangeSet", mod, params["input_range_m"])
        if record.error:
            return SkillResult(skill_name="SetPLLInpRange", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLInpRange", success=True,
            data={"modulator_index": mod, "input_range_m": params["input_range_m"]},
            nanonis_calls=[record],
        )


class PLLPerfectPLLUpdtZTC(BaseSkill):
    """Update the Z-Controller time constant via PerfectPLL."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PLLPerfectPLLUpdtZTC",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="用 PerfectPLL 算法更新 Z 控制器的时间常数。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["pll", "perfectpll", "ztc", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_PerfectPLLUpdtZTC", mod)
        if record.error:
            return SkillResult(skill_name="PLLPerfectPLLUpdtZTC", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="PLLPerfectPLLUpdtZTC", success=True,
            data={"modulator_index": mod},
            nanonis_calls=[record],
        )


class SetPLLPhasCtrlBandwidth(BaseSkill):
    """Set the phase controller bandwidth."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLPhasCtrlBandwidth",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置相位控制器的带宽。使用当前的 Q factor（取自之前的一次扫频）。"
            ),
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="bandwidth_hz",
                    type="float",
                    description="带宽，单位 Hz",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "phase", "bandwidth", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_PhasCtrlBandwidthSet", mod, params["bandwidth_hz"])
        if record.error:
            return SkillResult(skill_name="SetPLLPhasCtrlBandwidth", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLPhasCtrlBandwidth", success=True,
            data={"modulator_index": mod, "bandwidth_hz": params["bandwidth_hz"]},
            nanonis_calls=[record],
        )


class GetPLLPhasCtrlOnOff(BaseSkill):
    """Read the phase controller on/off status."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLPhasCtrlOnOff",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回相位控制器是开还是关。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "phase", "onoff", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLL_PhasCtrlOnOffGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLPhasCtrlOnOff", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["phase_ctrl_on"] = bool(int(v[0]) if isinstance(v, (list, tuple)) else int(v))
        return SkillResult(skill_name="GetPLLPhasCtrlOnOff", success=True,
                           data=data, nanonis_calls=[record])


class GetPLLSignalAnlzrCh(BaseSkill):
    """Read the PLL Signal Analyzer channel."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLSignalAnlzrCh",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回 PLL Signal Analyzer 当前的通道索引。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "analyzer", "channel", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("PLLSignalAnlzr_ChGet")
        if record.error:
            return SkillResult(skill_name="GetPLLSignalAnlzrCh", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            data["channel_index"] = int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        return SkillResult(skill_name="GetPLLSignalAnlzrCh", success=True,
                           data=data, nanonis_calls=[record])


class GetPLLSignalAnlzrFFTProps(BaseSkill):
    """Read the PLL Signal Analyzer FFT/spectrum configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLSignalAnlzrFFTProps",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "返回 FFT 配置：窗函数、平均模式、加权模式与次数。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "analyzer", "fft", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("PLLSignalAnlzr_FFTPropsGet")
        if record.error:
            return SkillResult(skill_name="GetPLLSignalAnlzrFFTProps", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        window_map = {
            0: "None", 1: "Hanning", 2: "Hamming", 3: "Blackman-Harris",
            4: "Exact Blackman", 5: "Blackman", 6: "Flat Top",
            7: "4 Term B-Harris", 8: "7 Term B-Harris", 9: "Low Sidelobe",
        }
        avg_map = {0: "None", 1: "Vector", 2: "RMS", 3: "Peak Hold"}
        weight_map = {0: "Linear", 1: "Exponential"}
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                win_idx = int(v[0])
                avg_idx = int(v[1])
                wgt_idx = int(v[2])
                data["fft_window_index"] = win_idx
                data["fft_window"] = window_map.get(win_idx, f"unknown({win_idx})")
                data["averaging_mode_index"] = avg_idx
                data["averaging_mode"] = avg_map.get(avg_idx, f"unknown({avg_idx})")
                data["weighting_mode_index"] = wgt_idx
                data["weighting_mode"] = weight_map.get(wgt_idx, f"unknown({wgt_idx})")
                data["count"] = int(v[3])
        return SkillResult(skill_name="GetPLLSignalAnlzrFFTProps", success=True,
                           data=data, nanonis_calls=[record])


class GetPLLSignalAnlzrTimebase(BaseSkill):
    """Read the PLL Signal Analyzer time base and update rate."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLSignalAnlzrTimebase",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回 PLL Signal Analyzer 的时基索引与刷新率。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "analyzer", "timebase", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("PLLSignalAnlzr_TimebaseGet")
        if record.error:
            return SkillResult(skill_name="GetPLLSignalAnlzrTimebase", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                data["timebase_index"] = int(v[0])
                data["update_rate"] = int(v[1])
        return SkillResult(skill_name="GetPLLSignalAnlzrTimebase", success=True,
                           data=data, nanonis_calls=[record])


class PLLSignalAnlzrTrigAuto(BaseSkill):
    """Set the PLL Signal Analyzer trigger to auto (pre-defined values)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PLLSignalAnlzrTrigAuto",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="把 PLL Signal Analyzer 的触发参数设为预定义值。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "analyzer", "trigger", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("PLLSignalAnlzr_TrigAuto")
        if record.error:
            return SkillResult(skill_name="PLLSignalAnlzrTrigAuto", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(skill_name="PLLSignalAnlzrTrigAuto", success=True,
                           data={}, nanonis_calls=[record])


class SetPLLSignalAnlzrTrig(BaseSkill):
    """Set the PLL Signal Analyzer trigger configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPLLSignalAnlzrTrig",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 PLL Signal Analyzer 中的触发配置。",
            parameters=[
                ParameterSpec(
                    name="trigger_mode",
                    type="int",
                    description="触发模式（0=不改，1=Immediate，2=Level）",
                    required=True,
                    min_value=0,
                    max_value=2,
                ),
                ParameterSpec(
                    name="trigger_source",
                    type="int",
                    description="触发源的信号索引",
                    required=True,
                ),
                ParameterSpec(
                    name="trigger_slope",
                    type="int",
                    description="触发沿（0=不改，1=Rising，2=Falling）",
                    required=True,
                    min_value=0,
                    max_value=2,
                ),
                ParameterSpec(
                    name="trigger_level",
                    type="float",
                    description="触发电平",
                    required=True,
                ),
                ParameterSpec(
                    name="trigger_position_s",
                    type="float",
                    description="触发位置，单位秒",
                    unit="s",
                    required=True,
                ),
                ParameterSpec(
                    name="arming_mode",
                    type="int",
                    description="布防模式（0=不改，1=Manual，2=Automatic）",
                    required=True,
                    min_value=0,
                    max_value=2,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "analyzer", "trigger", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "PLLSignalAnlzr_TrigSet",
            params["trigger_mode"],
            params["trigger_source"],
            params["trigger_slope"],
            params["trigger_level"],
            params["trigger_position_s"],
            params["arming_mode"],
        )
        if record.error:
            return SkillResult(skill_name="SetPLLSignalAnlzrTrig", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="SetPLLSignalAnlzrTrig", success=True,
            data={
                "trigger_mode": params["trigger_mode"],
                "trigger_source": params["trigger_source"],
                "trigger_slope": params["trigger_slope"],
                "trigger_level": params["trigger_level"],
                "trigger_position_s": params["trigger_position_s"],
                "arming_mode": params["arming_mode"],
            },
            nanonis_calls=[record],
        )


class GetPLLFreqSwpParams(BaseSkill):
    """Read the PLL frequency sweep parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPLLFreqSwpParams",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="返回扫频参数：点数、周期、建立时间。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "sweep", "params", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLLFreqSwp_ParamsGet", mod)
        if record.error:
            return SkillResult(skill_name="GetPLLFreqSwpParams", success=False,
                               error=record.error, nanonis_calls=[record])
        parsed = record.return_value
        data: dict = {"modulator_index": mod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            v = parsed[2]
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                data["num_points"] = int(v[0])
                data["period_s"] = float(v[1])
                data["settling_time_s"] = float(v[2])
        return SkillResult(skill_name="GetPLLFreqSwpParams", success=True,
                           data=data, nanonis_calls=[record])


class StopPLLFreqSwp(BaseSkill):
    """Stop the PLL frequency sweep."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopPLLFreqSwp",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="停止 PLL Frequency Sweep 模块中正在进行的扫描。",
            parameters=[
                ParameterSpec(
                    name="modulator_index",
                    type="int",
                    description="调制器/PLL 索引（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["pll", "sweep", "stop", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator_index", 1)
        record = context.safe_call("PLLFreqSwp_Stop", mod)
        if record.error:
            return SkillResult(skill_name="StopPLLFreqSwp", success=False,
                               error=record.error, nanonis_calls=[record])
        return SkillResult(
            skill_name="StopPLLFreqSwp", success=True,
            data={"modulator_index": mod},
            nanonis_calls=[record],
        )
