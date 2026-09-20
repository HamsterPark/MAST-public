"""Lock-in amplifier configuration skills.

vendored from v1 mast/skills/builtins/lockin.py 2026-04-23. Zero behavioural changes.
13 skills: ConfigureLockIn, ConfigureLockInDemod, GetLockInConfig,
           GetDemodHPFilter, GetDemodHarmonic, GetDemodLPFilter,
           GetDemodPhase, GetDemodPhasReg, SetDemodRTSignals,
           GetDemodSignal, SetDemodSyncFilter,
           SetModHarmonic, SetModPhasReg, SetModSignal.
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply, scalar_int_from_reply
from mast.skills.base import BaseSkill


def _extract(parsed, index: int = 0, default=None):
    """Extract a value from the Nanonis TCP return tuple ``(header, body, [vals])``."""
    if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
        vals = parsed[2]
        if isinstance(vals, (list, tuple)):
            if len(vals) > index:
                return vals[index]
        else:
            if index == 0:
                return vals
    return default


class ConfigureLockIn(BaseSkill):
    """Configure lock-in amplifier modulation."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureLockIn",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 lock-in 放大器调制的开/关与各项参数。",
            parameters=[
                ParameterSpec(
                    name="mod_on",
                    type="bool",
                    description="启用或禁用调制",
                    required=True,
                ),
                # 可选参数不声明默认值，避免模型把 schema 中的默认建议变成显式写入。
                # 适配器没有把缺省参数自动填进 execute；但一旦模型显式传值，
                # is not None 守卫无法区分它来自用户意图还是 schema 建议。
                # 因此未请求的设置保持省略，由实际参数存在性控制写操作。
                ParameterSpec(
                    name="amplitude_v",
                    type="float",
                    description=("调制幅度，单位伏特。省略 = 幅度保持原样。显式写 0 依然有效，"
                                 "而且依然表示 0 V —— 正因如此，本参数绝不能对外声称默认值是 0。"),
                    unit="V",
                    required=False,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ParameterSpec(
                    name="frequency_hz",
                    type="float",
                    description=("调制频率，单位 Hz。省略 = 频率保持原样。"),
                    unit="Hz",
                    required=False,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="phase_deg",
                    type="float",
                    # No default: omit phase unless the caller explicitly requests a write.
                    # Modulator phase may be unsupported by a device; demodulator reference phase
                    # is a separate setting exposed by ConfigureLockInDemod.
                    description=("调制相位，单位度。只有明确要修改时才提供；省略表示不写调制侧相位。设备未必支持此字段；若要修改解调参考相位，请使用 ConfigureLockInDemod。"
                                 ),
                    unit="deg",
                    required=False,
                    min_value=-360.0,
                    max_value=360.0,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["lockin", "modulation", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod_on = params["mod_on"]
        modulator = 1
        calls = []

        # Set frequency / amplitude / phase BEFORE enabling the modulation, so it
        # turns on with the CORRECT values — the old code enabled the modulation
        # FIRST, briefly exciting the tunnelling junction with the STALE previous
        # amplitude/frequency. Set each whenever provided,
        # allowing explicit 0 (e.g. amplitude=0 to make it safe before enabling);
        # the old `>0` / `!=0` guards made 0 impossible to command. Frequency=0
        # is still skipped (an invalid modulation frequency).
        frequency_hz = params.get("frequency_hz")
        amplitude_v = params.get("amplitude_v")
        phase_deg = params.get("phase_deg")
        frequency_written = False
        amplitude_written = False

        if frequency_hz is not None and frequency_hz > 0:
            rec_freq = context.safe_call("LockIn_ModPhasFreqSet", modulator, frequency_hz)
            calls.append(rec_freq)
            if rec_freq.error:
                return SkillResult(
                    skill_name="ConfigureLockIn", success=False,
                    error=rec_freq.error, nanonis_calls=calls,
                )
            frequency_written = True

        if amplitude_v is not None and amplitude_v >= 0:
            rec_amp = context.safe_call("LockIn_ModAmpSet", modulator, amplitude_v)
            calls.append(rec_amp)
            if rec_amp.error:
                return SkillResult(
                    skill_name="ConfigureLockIn", success=False,
                    error=rec_amp.error, nanonis_calls=calls,
                )
            amplitude_written = True

        if phase_deg is not None:
            rec_phase = context.safe_call("LockIn_ModPhasSet", modulator, phase_deg)
            calls.append(rec_phase)
            if rec_phase.error:
                return SkillResult(
                    skill_name="ConfigureLockIn", success=False,
                    error=rec_phase.error, nanonis_calls=calls,
                )

        # THEN enable/disable the modulation with the values already in place.
        rec_onoff = context.safe_call("LockIn_ModOnOffSet", modulator, int(mod_on))
        calls.append(rec_onoff)
        if rec_onoff.error:
            return SkillResult(
                skill_name="ConfigureLockIn",
                success=False,
                error=rec_onoff.error,
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="ConfigureLockIn",
            success=True,
            data={
                "mod_on": mod_on,
                "amplitude_v": amplitude_v if amplitude_written else None,
                "amplitude_written": amplitude_written,
                "frequency_hz": frequency_hz if frequency_written else None,
                "frequency_written": frequency_written,
                # Report the phase ONLY when we wrote one. It used to report
                # 0.0 unconditionally — a result that states a phase the skill
                # never set, on a register it never touched. Same family as
                # every other field label that says something the code did not
                # do; here it would read as "the phase is now 0°" while the
                # rig's calibrated phase sat untouched at some other value.
                "phase_deg": phase_deg,
                "phase_written": phase_deg is not None,
            },
            nanonis_calls=calls,
        )


class ConfigureLockInDemod(BaseSkill):
    """Configure lock-in demodulator settings."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureLockInDemod",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 lock-in 解调器：信号、谐波、滤波器、相位。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="signal_index",
                    type="int",
                    description="解调信号索引（0-127）",
                    required=False,
                    min_value=0,
                    max_value=127,
                ),
                ParameterSpec(
                    name="harmonic",
                    type="int",
                    description="谐波次数（1=基频）",
                    required=False,
                    min_value=1,
                ),
                ParameterSpec(
                    name="lp_order",
                    type="int",
                    description="低通滤波器阶数（-1=不改，0=关，1-8）",
                    required=False,
                    min_value=-1,
                    max_value=8,
                ),
                ParameterSpec(
                    name="lp_cutoff_hz",
                    type="float",
                    description="低通滤波器截止频率（0=不改）",
                    unit="Hz",
                    required=False,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="hp_order",
                    type="int",
                    description="高通滤波器阶数（-1=不改，0=关，1-8）",
                    required=False,
                    min_value=-1,
                    max_value=8,
                ),
                ParameterSpec(
                    name="hp_cutoff_hz",
                    type="float",
                    description="高通滤波器截止频率（0=不改）",
                    unit="Hz",
                    required=False,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="phase_deg",
                    type="float",
                    description="解调器的参考相位",
                    unit="deg",
                    required=False,
                    min_value=-360.0,
                    max_value=360.0,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["lockin", "demodulator", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        calls = []

        # NOTE: use ``params.get(key) is not None`` rather than ``key in params``.
        # On the agent path the pydantic args_schema materialises EVERY optional
        # field — unset ones arrive as ``None`` — so ``"key" in params`` is always
        # True and we would otherwise call the hardware setter with ``None``
        # (struct.pack failure / corrupt write on real Nanonis). Only act on the
        # sub-settings the caller actually supplied.
        if params.get("signal_index") is not None:
            rec = context.safe_call("LockIn_DemodSignalSet", demod, params["signal_index"])
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigureLockInDemod", success=False,
                                   error=rec.error, nanonis_calls=calls)

        if params.get("harmonic") is not None:
            rec = context.safe_call("LockIn_DemodHarmonicSet", demod, params["harmonic"])
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigureLockInDemod", success=False,
                                   error=rec.error, nanonis_calls=calls)

        if params.get("lp_order") is not None or params.get("lp_cutoff_hz") is not None:
            order = params.get("lp_order")
            cutoff = params.get("lp_cutoff_hz")
            order = -1 if order is None else order
            cutoff = 0.0 if cutoff is None else cutoff
            rec = context.safe_call("LockIn_DemodLPFilterSet", demod, order, cutoff)
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigureLockInDemod", success=False,
                                   error=rec.error, nanonis_calls=calls)

        if params.get("hp_order") is not None or params.get("hp_cutoff_hz") is not None:
            order = params.get("hp_order")
            cutoff = params.get("hp_cutoff_hz")
            order = -1 if order is None else order
            cutoff = 0.0 if cutoff is None else cutoff
            rec = context.safe_call("LockIn_DemodHPFilterSet", demod, order, cutoff)
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigureLockInDemod", success=False,
                                   error=rec.error, nanonis_calls=calls)

        if params.get("phase_deg") is not None:
            rec = context.safe_call("LockIn_DemodPhasSet", demod, params["phase_deg"])
            calls.append(rec)
            if rec.error:
                return SkillResult(skill_name="ConfigureLockInDemod", success=False,
                                   error=rec.error, nanonis_calls=calls)

        return SkillResult(
            skill_name="ConfigureLockInDemod",
            success=True,
            data={"demodulator": demod},
            nanonis_calls=calls,
        )


class GetLockInConfig(BaseSkill):
    """Read back complete lock-in configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetLockInConfig",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "把 lock-in 完整读回来：开/关、幅度、频率、相位，以及 —— 以前只写不可读的那部分 "
                "—— 它调制的到底是哪一路信号、谐波次数、调制器与解调器的相位寄存器、"
                "解调器的实时信号及其 sync 滤波器。\n"
                "\n"
                "相信任何一条 dI/dV 之前，先查 `modulated_signal`：调制错了信号的 lock-in "
                "会给出一条完美干净、却彻头彻尾错误的曲线，而且它没有任何一处看起来是错的。"
            ),
            parameters=[
                ParameterSpec(
                    name="modulator",
                    type="int",
                    description="调制器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）；默认取调制器的编号",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=1.5,
            composition_level=1,
            tags=["lockin", "config", "read", "readback", "verify"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator", 1)
        calls = []
        data: dict = {"modulator": mod}

        rec = context.safe_call("LockIn_ModOnOffGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            data["mod_on"] = bool(v[0]) if isinstance(v, (list, tuple)) else bool(v)

        rec = context.safe_call("LockIn_ModAmpGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            data["amplitude"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        rec = context.safe_call("LockIn_ModPhasFreqGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            data["frequency_hz"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        rec = context.safe_call("LockIn_ModPhasGet", mod)
        calls.append(rec)
        if not rec.error and isinstance(rec.return_value, (list, tuple)) and len(rec.return_value) > 2:
            v = rec.return_value[2]
            data["phase_deg"] = float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        # 2026-07-13 — the readback half that was never wired. MAST could SET each of
        # these and had no way to read it back, so every "the lock-in is configured"
        # was the skill repeating its own request to itself. `modulated_signal` is the
        # one that matters: a lock-in modulating the wrong signal produces a perfectly
        # clean dI/dV-shaped curve that is not dI/dV, and nothing about it looks wrong.
        demod = int(params.get("demodulator", mod) or mod)
        data["demodulator"] = demod
        # Literal verbs: every safety tool in this repo (abort-policy check,
        # security audit, API-coverage census) finds Nanonis calls by grepping
        # safe_call("…"). A verb behind a variable is invisible to all of them.
        for key, thunk in (
            ("modulated_signal", lambda: context.safe_call("LockIn_ModSignalGet", mod)),
            ("harmonic", lambda: context.safe_call("LockIn_ModHarmonicGet", mod)),
            ("mod_phase_register", lambda: context.safe_call("LockIn_ModPhasRegGet", mod)),
            ("demod_phase_register", lambda: context.safe_call("LockIn_DemodPhasRegGet", demod)),
            ("demod_rt_signals", lambda: context.safe_call("LockIn_DemodRTSignalsGet", demod)),
            ("demod_sync_filter", lambda: context.safe_call("LockIn_DemodSyncFilterGet", demod)),
        ):
            rec = thunk()
            calls.append(rec)
            # 解析数值而不是把三段协议回包原样返回；包含单元素元组时需正确解包。
            # 读取失败或无法解码返回 None，不能用零替代。
            data[key] = None if rec.error else scalar_int_from_reply(rec.return_value)

        return SkillResult(
            skill_name="GetLockInConfig",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class GetDemodHPFilter(BaseSkill):
    """Read the high-pass filter settings of a lock-in demodulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDemodHPFilter",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个 lock-in 解调器的高通滤波器阶数与截止频率。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "hp", "filter", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        record = context.safe_call("LockIn_DemodHPFilterGet", demod)
        if record.error:
            return SkillResult(
                skill_name="GetDemodHPFilter", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"demodulator": demod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {
                    "demodulator": demod,
                    "hp_filter_order": int(vals[0]),
                    "hp_cutoff_hz": float(vals[1]),
                }
        return SkillResult(
            skill_name="GetDemodHPFilter", success=True,
            data=data, nanonis_calls=[record],
        )


class GetDemodHarmonic(BaseSkill):
    """Read the harmonic of a lock-in demodulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDemodHarmonic",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个 lock-in 解调器的谐波次数。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "harmonic", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        record = context.safe_call("LockIn_DemodHarmonicGet", demod)
        if record.error:
            return SkillResult(
                skill_name="GetDemodHarmonic", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"demodulator": demod, "raw": decode_reply(parsed)}
        harmonic = _extract(parsed, 0)
        if harmonic is not None:
            data = {"demodulator": demod, "harmonic": int(harmonic)}
        return SkillResult(
            skill_name="GetDemodHarmonic", success=True,
            data=data, nanonis_calls=[record],
        )


class GetDemodLPFilter(BaseSkill):
    """Read the low-pass filter settings of a lock-in demodulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDemodLPFilter",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个 lock-in 解调器的低通滤波器阶数与截止频率。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "lp", "filter", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        record = context.safe_call("LockIn_DemodLPFilterGet", demod)
        if record.error:
            return SkillResult(
                skill_name="GetDemodLPFilter", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"demodulator": demod, "raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {
                    "demodulator": demod,
                    "lp_filter_order": int(vals[0]),
                    "lp_cutoff_hz": float(vals[1]),
                }
        return SkillResult(
            skill_name="GetDemodLPFilter", success=True,
            data=data, nanonis_calls=[record],
        )


class GetDemodPhase(BaseSkill):
    """Read the reference phase of a lock-in demodulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDemodPhase",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个 lock-in 解调器的参考相位。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "phase", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        record = context.safe_call("LockIn_DemodPhasGet", demod)
        if record.error:
            return SkillResult(
                skill_name="GetDemodPhase", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"demodulator": demod, "raw": decode_reply(parsed)}
        phase = _extract(parsed, 0)
        if phase is not None:
            data = {"demodulator": demod, "phase_deg": float(phase)}
        return SkillResult(
            skill_name="GetDemodPhase", success=True,
            data=data, nanonis_calls=[record],
        )


class GetDemodPhasReg(BaseSkill):
    """Read the phase register index of a lock-in demodulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDemodPhasReg",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个 lock-in 解调器的相位寄存器索引（1-8）。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "phase", "register", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        record = context.safe_call("LockIn_DemodPhasRegGet", demod)
        if record.error:
            return SkillResult(
                skill_name="GetDemodPhasReg", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"demodulator": demod, "raw": decode_reply(parsed)}
        idx = _extract(parsed, 0)
        if idx is not None:
            data = {"demodulator": demod, "phase_register_index": int(idx)}
        return SkillResult(
            skill_name="GetDemodPhasReg", success=True,
            data=data, nanonis_calls=[record],
        )


class SetDemodRTSignals(BaseSkill):
    """Set which RT signals are available from a lock-in demodulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetDemodRTSignals",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置某个 lock-in 解调器的 RT 信号（X/Y 或 R/phi）。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="rt_signals",
                    type="int",
                    description="0 = X/Y，1 = R/phi",
                    required=True,
                    min_value=0,
                    max_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "rt", "signals", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        rt = params["rt_signals"]
        record = context.safe_call("LockIn_DemodRTSignalsSet", demod, rt)
        if record.error:
            return SkillResult(
                skill_name="SetDemodRTSignals", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetDemodRTSignals", success=True,
            data={"demodulator": demod, "rt_signals": rt},
            nanonis_calls=[record],
        )


class GetDemodSignal(BaseSkill):
    """Read the demodulated signal index of a lock-in demodulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDemodSignal",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个 lock-in 解调器的解调信号索引（0-127）。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "signal", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        record = context.safe_call("LockIn_DemodSignalGet", demod)
        if record.error:
            return SkillResult(
                skill_name="GetDemodSignal", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"demodulator": demod, "raw": decode_reply(parsed)}
        idx = _extract(parsed, 0)
        if idx is not None:
            data = {"demodulator": demod, "signal_index": int(idx)}
        return SkillResult(
            skill_name="GetDemodSignal", success=True,
            data=data, nanonis_calls=[record],
        )


class SetDemodSyncFilter(BaseSkill):
    """Switch the synchronous filter of a lock-in demodulator on or off."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetDemodSyncFilter",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="开/关某个 lock-in 解调器的 sync 滤波器。",
            parameters=[
                ParameterSpec(
                    name="demodulator",
                    type="int",
                    description="解调器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="sync_filter_on",
                    type="bool",
                    description="True 启用 sync 滤波器，False 禁用",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "demodulator", "sync", "filter", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        demod = params.get("demodulator", 1)
        on = int(params["sync_filter_on"])
        record = context.safe_call("LockIn_DemodSyncFilterSet", demod, on)
        if record.error:
            return SkillResult(
                skill_name="SetDemodSyncFilter", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetDemodSyncFilter", success=True,
            data={"demodulator": demod, "sync_filter_on": bool(on)},
            nanonis_calls=[record],
        )


class SetModHarmonic(BaseSkill):
    """Set the harmonic of a lock-in modulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetModHarmonic",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置某个 lock-in 调制器的谐波次数。",
            parameters=[
                ParameterSpec(
                    name="modulator",
                    type="int",
                    description="调制器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="harmonic",
                    type="int",
                    description="谐波次数（1 = 基频）",
                    required=True,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "modulator", "harmonic", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator", 1)
        harmonic = params["harmonic"]
        record = context.safe_call("LockIn_ModHarmonicSet", mod, harmonic)
        if record.error:
            return SkillResult(
                skill_name="SetModHarmonic", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetModHarmonic", success=True,
            data={"modulator": mod, "harmonic": harmonic},
            nanonis_calls=[record],
        )


class SetModPhasReg(BaseSkill):
    """Set the phase register index of a lock-in modulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetModPhasReg",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="把某个 lock-in 调制器指派到一个相位寄存器（1-8）。",
            parameters=[
                ParameterSpec(
                    name="modulator",
                    type="int",
                    description="调制器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="phase_register_index",
                    type="int",
                    description="相位寄存器索引（1-8）",
                    required=True,
                    min_value=1,
                    max_value=8,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "modulator", "phase", "register", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator", 1)
        idx = params["phase_register_index"]
        record = context.safe_call("LockIn_ModPhasRegSet", mod, idx)
        if record.error:
            return SkillResult(
                skill_name="SetModPhasReg", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetModPhasReg", success=True,
            data={"modulator": mod, "phase_register_index": idx},
            nanonis_calls=[record],
        )


class SetModSignal(BaseSkill):
    """Set the modulated signal of a lock-in modulator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetModSignal",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="为某个 lock-in 调制器选择被调制的信号（按索引 0-127）。",
            parameters=[
                ParameterSpec(
                    name="modulator",
                    type="int",
                    description="调制器编号（从 1 开始）",
                    required=False,
                    default=1,
                    min_value=1,
                ),
                ParameterSpec(
                    name="signal_index",
                    type="int",
                    description="信号索引（0-127），取自 Signals 列表",
                    required=True,
                    min_value=0,
                    max_value=127,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["lockin", "modulator", "signal", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mod = params.get("modulator", 1)
        sig = params["signal_index"]
        record = context.safe_call("LockIn_ModSignalSet", mod, sig)
        if record.error:
            return SkillResult(
                skill_name="SetModSignal", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetModSignal", success=True,
            data={"modulator": mod, "signal_index": sig},
            nanonis_calls=[record],
        )
