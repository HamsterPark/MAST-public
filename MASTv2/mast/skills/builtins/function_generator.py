"""Function generators — the other half of "Nanonis output control".

Nanonis has two waveform generators and MAST had a skill for neither:

  * ``FunGen1Ch_*`` (7 API methods) — the single-channel generator.
  * ``FunGen2Ch_*`` (13 methods) — the two-channel generator, with a selectable
    waveform SHAPE per channel and an on/off per channel.

They belong with ``UserOut``: this is the same gap, with a time axis on it. A user
output holds a value; a function generator *sweeps* one — a modulation, a
reference, a ramp, a chopper drive. For a pump–probe / TERS setup that is not a
nicety.

Same safety posture as ``user_output.py``, for the same reason: **MAST does not
know what is wired to the output.** But a Nanonis output is a small-signal
line and the real protection is in the wiring. So these
are ``CONFIRM``, not blocked; they say plainly that MAST cannot see the far end;
and they are not on the post-abort allow-list, so an abort stops them from
starting. An abort does NOT stop a generator already running — call ``StopWaveform``
for that, which IS what an operator pressing 中止 would want next.
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

# FunGen2Ch_WaveformSet(Channel_index, Shape)
_SHAPES = {"sine": 0, "square": 1, "triangle": 2, "sawtooth": 3, "ramp": 4}


def _values(record) -> list:
    rv = getattr(record, "return_value", None)
    if isinstance(rv, (list, tuple)) and len(rv) > 2 and isinstance(rv[2], (list, tuple)):
        return list(rv[2])
    return []


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


class ConfigureWaveform(BaseSkill):
    """Set a function generator's amplitude / frequency / shape."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureWaveform",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置一台 Nanonis 函数发生器：幅度、频率，以及（仅 2 通道发生器）波形形状。"
                "它驱动的是 MAST 看不见的 EXTERNAL 硬件 —— 一路调制、一路 lock-in 参考、一个斩波器、"
                "一条栅极斜坡。配置不等于启动；之后还要调用 StartWaveform。"
            ),
            parameters=[
                ParameterSpec(
                    name="generator", type="str",
                    description="'1ch'（单通道）或 '2ch'（双通道）",
                    required=True, allowed_values=["1ch", "2ch"],
                ),
                ParameterSpec(
                    name="amplitude", type="float",
                    description="波形幅度，用该输出的物理单位",
                    required=True,
                ),
                ParameterSpec(
                    name="frequency_hz", type="float",
                    description=(
                        "1 通道发生器的频率（Hz）。2 通道发生器那边 Nanonis 收的是 PERIOD（s）—— "
                        "你传频率，它会被换算。"
                    ),
                    unit="Hz", required=True, min_value=1e-6, max_value=1e6,
                ),
                ParameterSpec(
                    name="channel", type="int",
                    description="通道索引（仅 2 通道发生器）",
                    required=False, default=1, min_value=1, max_value=2,
                ),
                ParameterSpec(
                    name="shape", type="str",
                    description="sine/square/triangle/sawtooth/ramp（仅 2 通道）",
                    required=False, default="sine",
                    allowed_values=list(_SHAPES),
                ),
                ParameterSpec(
                    name="polarity", type="int",
                    description="0 = 双极性，1 = 单极性（厂商约定）",
                    required=False, default=0, min_value=0, max_value=1,
                ),
                ParameterSpec(
                    name="direction", type="int",
                    description="0 = 先向上，1 = 先向下",
                    required=False, default=0, min_value=0, max_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["output", "function_generator", "waveform", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        gen = str(params["generator"]).strip().lower()
        amp = float(params["amplitude"])
        freq = float(params["frequency_hz"])
        pol = int(params.get("polarity", 0) or 0)
        direction = int(params.get("direction", 0) or 0)
        calls = []

        if gen == "1ch":
            rec = context.safe_call("FunGen1Ch_PropsSet", amp, freq, pol, direction)
            calls.append(rec)
            if rec.error:
                return _fail("ConfigureWaveform", rec.error, calls)
            return SkillResult(
                skill_name="ConfigureWaveform", success=True,
                data={"generator": "1ch", "amplitude": amp, "frequency_hz": freq},
                nanonis_calls=calls,
            )

        if gen != "2ch":
            return _fail("ConfigureWaveform",
                         f"generator 必须是 '1ch' 或 '2ch'，收到 {gen!r}", [])

        ch = int(params.get("channel", 1) or 1)
        shape = str(params.get("shape", "sine")).strip().lower()
        if shape not in _SHAPES:
            return _fail("ConfigureWaveform",
                         f"shape 必须是 {list(_SHAPES)} 之一，收到 {shape!r}", [])

        rec = context.safe_call("FunGen2Ch_WaveformSet", ch, _SHAPES[shape])
        calls.append(rec)
        if rec.error:
            return _fail("ConfigureWaveform", f"WaveformSet failed: {rec.error}", calls)

        # NB: the 2-channel generator takes a PERIOD (s), not a frequency. Passing a
        # frequency straight through here would set a period of e.g. 1000 s for a
        # 1 kHz request — off by six orders of magnitude, silently, on hardware
        # nobody can see. Convert.
        period_s = 1.0 / freq
        add_zero = 0
        rec = context.safe_call("FunGen2Ch_PropsSet", ch, amp, period_s,
                                pol, direction, add_zero)
        calls.append(rec)
        if rec.error:
            return _fail("ConfigureWaveform", f"PropsSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name="ConfigureWaveform", success=True,
            data={"generator": "2ch", "channel": ch, "shape": shape,
                  "amplitude": amp, "frequency_hz": freq, "period_s": period_s},
            nanonis_calls=calls,
        )


class StartWaveform(BaseSkill):
    """Start a function generator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StartWaveform",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "启动一台函数发生器，运行指定的周期数（0 = 一直跑到被停止）。先用 ConfigureWaveform "
                "配置它。这会在一条输出线上放出一个真实信号 —— MAST 看不见另一端接的是什么。"
            ),
            parameters=[
                ParameterSpec(
                    name="generator", type="str",
                    description="'1ch' 或 '2ch'",
                    required=True, allowed_values=["1ch", "2ch"],
                ),
                ParameterSpec(
                    name="periods", type="int",
                    description="要运行的周期数；0 = 连续",
                    required=False, default=0, min_value=0, max_value=1000000,
                ),
                ParameterSpec(
                    name="wait_until_finished", type="bool",
                    description="阻塞直到这次 burst 结束（periods=0 时本项被忽略）",
                    required=False, default=False,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["output", "function_generator", "waveform", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        gen = str(params["generator"]).strip().lower()
        periods = int(params.get("periods", 0) or 0)
        wait = 1 if params.get("wait_until_finished", False) else 0
        if gen not in ("1ch", "2ch"):
            return _fail("StartWaveform",
                         f"generator 必须是 '1ch' 或 '2ch'，收到 {gen!r}", [])
        # Literal verbs, deliberately. Every safety tool MAST has reads the
        # Nanonis verb back out of the source by grepping `safe_call("…")` —
        # the post-abort allow-list guard, the coverage census, any future
        # audit. A verb hidden behind a variable is INVISIBLE to all of them:
        # my own guard caught this file adding FunGen*_Stop to the abort
        # allow-list and then reporting it as a phantom, because the call site
        # was `safe_call(verb)`. Two branches cost three lines and keep the
        # safety tooling honest.
        if gen == "1ch":
            rec = context.safe_call("FunGen1Ch_Start", periods, wait)
        else:
            rec = context.safe_call("FunGen2Ch_Start", periods, wait)
        if rec.error:
            return _fail("StartWaveform", rec.error, [rec])
        return SkillResult(
            skill_name="StartWaveform", success=True,
            data={"generator": gen, "periods": periods,
                  "mode": "continuous" if periods == 0 else "burst"},
            nanonis_calls=[rec],
        )


class StopWaveform(BaseSkill):
    """Stop a function generator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopWaveform",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "停止一台正在运行的函数发生器。STOP 永远是允许的 —— 对一路输出来说，"
                "这是你永远希望自己做得到的那件事。"
            ),
            parameters=[
                ParameterSpec(
                    name="generator", type="str",
                    description="'1ch' 或 '2ch'",
                    required=True, allowed_values=["1ch", "2ch"],
                ),
            ],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["output", "function_generator", "stop"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        gen = str(params["generator"]).strip().lower()
        if gen not in ("1ch", "2ch"):
            return _fail("StopWaveform",
                         f"generator 必须是 '1ch' 或 '2ch'，收到 {gen!r}", [])
        if gen == "1ch":
            rec = context.safe_call("FunGen1Ch_Stop")
        else:
            rec = context.safe_call("FunGen2Ch_Stop")
        if rec.error:
            return _fail("StopWaveform", rec.error, [rec])
        return SkillResult(skill_name="StopWaveform", success=True,
                           data={"generator": gen, "stopped": True},
                           nanonis_calls=[rec])


class GetWaveformStatus(BaseSkill):
    """Is a generator running, and with what settings?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetWaveformStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读一台函数发生器的状态与当前设置（幅度 / 频率 / 形状 / idle 值）。"
                "启动一台不是你自己配置的发生器之前，先查一下。"
            ),
            parameters=[
                ParameterSpec(
                    name="generator", type="str",
                    description="'1ch' 或 '2ch'",
                    required=True, allowed_values=["1ch", "2ch"],
                ),
                ParameterSpec(
                    name="channel", type="int",
                    description="通道索引（仅 2 通道发生器）",
                    required=False, default=1, min_value=1, max_value=2,
                ),
            ],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["output", "function_generator", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        gen = str(params["generator"]).strip().lower()
        if gen not in ("1ch", "2ch"):
            return _fail("GetWaveformStatus",
                         f"generator 必须是 '1ch' 或 '2ch'，收到 {gen!r}", [])
        calls = []
        if gen == "1ch":
            st = context.safe_call("FunGen1Ch_StatusGet")
            pr = context.safe_call("FunGen1Ch_PropsGet")
            idle = context.safe_call("FunGen1Ch_IdleGet")
            calls = [st, pr, idle]
        else:
            ch = int(params.get("channel", 1) or 1)
            st = context.safe_call("FunGen2Ch_StatusGet")
            pr = context.safe_call("FunGen2Ch_PropsGet", ch)
            idle = context.safe_call("FunGen2Ch_WaveformGet", ch)
            on = context.safe_call("FunGen2Ch_OnOffGet", ch)
            sig = context.safe_call("FunGen2Ch_SignalGet", ch)
            calls = [st, pr, idle, on, sig]
        if st.error:
            return _fail("GetWaveformStatus", st.error, calls)
        return SkillResult(
            skill_name="GetWaveformStatus", success=True,
            data={"generator": gen,
                  "status": _values(st),
                  "props": _values(pr) if not pr.error else [],
                  "extra": _values(idle) if not idle.error else [],
                  # 2-channel only: which output line the channel drives, and
                  # whether that channel is enabled.
                  "channel_on": (_values(calls[3]) if len(calls) > 3
                                 and not calls[3].error else None),
                  "output_signal": (_values(calls[4]) if len(calls) > 4
                                    and not calls[4].error else None)},
            nanonis_calls=calls,
        )


class SetWaveformIdleValue(BaseSkill):
    """Set the value a generator rests at when it is not running."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetWaveformIdleValue",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置函数发生器停止时保持的 IDLE 值。这是一条输出线的静息状态，所以它比听上去更要紧："
                "外部硬件在两次 burst 之间、以及一次 StopWaveform 之后看到的就是它。"
                "它**不一定**就是「关」—— 那取决于你的接线，而接线 MAST 看不见。"
            ),
            parameters=[
                ParameterSpec(
                    name="generator", type="str",
                    description="'1ch' 或 '2ch'",
                    required=True, allowed_values=["1ch", "2ch"],
                ),
                ParameterSpec(
                    name="idle_value", type="float",
                    description="静息值，用该输出的物理单位",
                    required=True,
                ),
                ParameterSpec(
                    name="device", type="int",
                    description="设备索引（仅 2 通道发生器）",
                    required=False, default=1, min_value=1, max_value=2,
                ),
            ],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["output", "function_generator", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        gen = str(params["generator"]).strip().lower()
        val = float(params["idle_value"])
        if gen == "1ch":
            rec = context.safe_call("FunGen1Ch_IdleSet", val)
        elif gen == "2ch":
            rec = context.safe_call("FunGen2Ch_IdleSet",
                                    int(params.get("device", 1) or 1), val)
        else:
            return _fail("SetWaveformIdleValue",
                         f"generator 必须是 '1ch' 或 '2ch'，收到 {gen!r}", [])
        if rec.error:
            return _fail("SetWaveformIdleValue", rec.error, [rec])
        return SkillResult(
            skill_name="SetWaveformIdleValue", success=True,
            data={"generator": gen, "idle_value": val}, nanonis_calls=[rec])


class SetWaveformChannelOnOff(BaseSkill):
    """Enable / disable one channel of the 2-channel generator."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetWaveformChannelOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "启用或禁用 2 通道函数发生器中的一个通道，而不停掉另一个。"
            ),
            parameters=[
                ParameterSpec(
                    name="channel", type="int",
                    description="通道索引（1 或 2）",
                    required=True, min_value=1, max_value=2,
                ),
                ParameterSpec(
                    name="on", type="bool",
                    description="True = 启用，False = 禁用",
                    required=True,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "function_generator", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        ch = int(params["channel"])
        on = 1 if params["on"] else 0
        rec = context.safe_call("FunGen2Ch_OnOffSet", ch, on)
        if rec.error:
            return _fail("SetWaveformChannelOnOff", rec.error, [rec])
        return SkillResult(
            skill_name="SetWaveformChannelOnOff", success=True,
            data={"channel": ch, "on": bool(on)}, nanonis_calls=[rec])


__all__ = ["ConfigureWaveform", "StartWaveform", "StopWaveform",
           "GetWaveformStatus", "SetWaveformIdleValue", "SetWaveformChannelOnOff"]
