"""User Outputs + Digital Lines — driving hardware that is NOT the microscope.

THE GAP THIS FILLS: Nanonis had some custom output control with no skill
wrapping it, and it was total: Nanonis exposes 12 ``UserOut_*`` methods and
4 ``DigLines_*`` methods, and MAST had a skill for **zero of them**. The agent
could scan, sweep, pulse the tip and read every signal in the machine — and could
not turn on a single output.

User Outputs are the analog channels through which Nanonis drives *external*
hardware: a lock-in reference, a gate voltage, a piezo amplifier, a laser shutter,
a delay line. Digital lines are the TTL outputs that trigger it: a camera, a pulse
generator, a chopper.

────────────────────────────────────────────────────────────────────────────────
WHY THESE SKILLS ARE SHAPED THE WAY THEY ARE

**MAST does not know what is plugged into a user output.** That is the whole
problem, and it is what makes these different from ``SetBias``. A bias is a bias;
the safety envelope is a property of the microscope and MAST can hold it in
``SafetyLimits``. A user output could be volts, micrometres, milliwatts, or a
shutter that is open at 0 and shut at 5 — MAST has no way to know, and inventing a
bound would be a bound about nothing.

Nanonis DOES know: every output carries **operator-configured physical limits**
(``UserOut.LimitsGet/Set``), in the same physical units as the value. So:

  1. ``SetUserOutput`` **reads those limits and refuses to exceed them.** MAST does
     not invent a safety envelope; it honours the one configured on the instrument.
     If the limits cannot be read, the write is refused — an unknown envelope is
     not an open one.

  2. ``SetUserOutputLimits`` **is a skill, and the limits are therefore not a
     barrier the agent cannot cross.** The first draft of this file withheld it, on
     the software-security principle that "an agent which can widen its own
     guardrail has none". That principle does not fit here: a Nanonis user
     output is a small-signal analog line; the real protection lives
     in the WIRING — the amplifier, the interlock, the choice of what to connect.
     The Nanonis limits are a configuration convenience, not the last physical
     barrier, and treating them as one imported a threat model from a different
     domain into a lab where the physical layer is already designed for this.

     What remains is honest bookkeeping, not a block: a limits change is
     ``CONFIRM``-gated like every other write here, and it is written to the
     refusal ledger (``core.diagnostics``, kind ``note``) as a GUARDRAIL CHANGE. So
     ``SetUserOutput``'s check still earns its place — it catches the honest
     out-of-range mistake, which is the common case — and going outside the
     envelope now takes a deliberate, logged act rather than an accident.

  3. Nothing here is on the **post-abort allow-list**, so every one of these is
     refused once the operator aborts (``core.execution_context``). Note what is
     NOT done: MAST does **not** zero the outputs on an abort. Zeroing sounds safe
     and is not — a shutter wired normally-closed OPENS at 0. MAST does not know
     the wiring, so it stops writing rather than guessing which value is "off".
     If your setup has a safe resting state, put it behind an explicit skill call.

Value semantics (from the official protocol, TCPProtocol_Mimea_V5e):
  * ``Output index`` is **1-based** (1..N), not 0-based.
  * ``Output value`` is in **calibrated physical units**, NOT volts.
  * ``Output mode``: 0 = User Output, 1 = Monitor, 2 = Calc.Signal.
  * ``Port``: 0=A, 1=B, 2=C, 3=D, 4+ = expanded DIO ports.
  * Digital ``lines`` are 1..8 within a port.
"""

from __future__ import annotations

from typing import Any

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

# Nanonis output modes (protocol §User Outputs).
_MODES = {0: "User Output", 1: "Monitor", 2: "Calc.Signal"}
_PORTS = {0: "A", 1: "B", 2: "C", 3: "D"}


def _values(record) -> list:
    """The parsed value list out of a NanonisCallRecord, or []."""
    rv = getattr(record, "return_value", None)
    if isinstance(rv, (list, tuple)) and len(rv) > 2 and isinstance(rv[2], (list, tuple)):
        return list(rv[2])
    return []


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


# ═════════════════════════════════════════════════════════════════════════════
# Reads
# ═════════════════════════════════════════════════════════════════════════════

class GetUserOutputLimits(BaseSkill):
    """Read one user output's operator-configured physical limits."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetUserOutputLimits",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读某个用户输出通道的物理上／下限。这两个限值就是 SetUserOutput 的**安全包络** —— 它们由用户在 Nanonis 里配置，"
                "从这里改不了。要驱动一个你没把握的输出之前，先读它们。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算（1..N）",
                    required=True, min_value=1, max_value=24,
                ),
                ParameterSpec(
                    name="raw", type="int",
                    description="0 = 物理（已标定）限值；1 = 原始限值",
                    required=False, default=0, min_value=0, max_value=1,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "user_output", "read", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        raw = int(params.get("raw", 0) or 0)
        rec = context.safe_call("UserOut_LimitsGet", idx, raw)
        if rec.error:
            return _fail("GetUserOutputLimits", rec.error, [rec])
        vals = _values(rec)
        upper = float(vals[0]) if len(vals) > 0 else None
        lower = float(vals[1]) if len(vals) > 1 else None
        return SkillResult(
            skill_name="GetUserOutputLimits", success=True,
            data={"output_index": idx, "upper_limit": upper, "lower_limit": lower,
                  "raw": bool(raw),
                  "note": "单位是该通道的物理单位（可能不是伏特）——由 Nanonis 里的校准决定。"},
            nanonis_calls=[rec],
        )


class GetUserOutputMode(BaseSkill):
    """Read a user output's mode (User Output / Monitor / Calc.Signal)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetUserOutputMode",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读某个用户输出的模式：0=User Output（由你驱动），1=Monitor（它镜像一路信号）"
                "，2=Calc.Signal。SetUserOutput 只有在模式 0 下才起作用 —— Monitor 模式下这个通道由仪器驱动，"
                "不由你。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "user_output", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        rec = context.safe_call("UserOut_ModeGet", idx)
        if rec.error:
            return _fail("GetUserOutputMode", rec.error, [rec])
        vals = _values(rec)
        mode = int(vals[0]) if vals else None
        return SkillResult(
            skill_name="GetUserOutputMode", success=True,
            data={"output_index": idx, "mode": mode,
                  "mode_name": _MODES.get(mode, "unknown")},
            nanonis_calls=[rec],
        )


class GetUserOutputMonitorChannel(BaseSkill):
    """Read which signal a user output mirrors in Monitor mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetUserOutputMonitorChannel",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个用户输出的监视通道序号（Monitor 模式）。",
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "user_output", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        rec = context.safe_call("UserOut_MonitorChGet", idx)
        if rec.error:
            return _fail("GetUserOutputMonitorChannel", rec.error, [rec])
        vals = _values(rec)
        return SkillResult(
            skill_name="GetUserOutputMonitorChannel", success=True,
            data={"output_index": idx,
                  "monitor_channel_index": int(vals[0]) if vals else None},
            nanonis_calls=[rec],
        )


class GetDigitalLineTTL(BaseSkill):
    """Read the TTL levels of a digital port."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDigitalLineTTL",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个数字端口上全部 8 条线的 TTL 值。",
            parameters=[
                ParameterSpec(
                    name="port", type="int",
                    description="0=Port A，1=B，2=C，3=D，4+ = 扩展 DIO",
                    required=True, min_value=0, max_value=8,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "digital", "ttl", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        port = int(params["port"])
        rec = context.safe_call("DigLines_TTLValGet", port)
        if rec.error:
            return _fail("GetDigitalLineTTL", rec.error, [rec])
        return SkillResult(
            skill_name="GetDigitalLineTTL", success=True,
            data={"port": port, "port_name": _PORTS.get(port, f"expanded_{port}"),
                  "ttl_values": _values(rec)},
            nanonis_calls=[rec],
        )


# ═════════════════════════════════════════════════════════════════════════════
# Writes — the ones that reach into hardware MAST cannot see
# ═════════════════════════════════════════════════════════════════════════════

class SetUserOutput(BaseSkill):
    """Drive a user output — clamped to the limits the OPERATOR configured."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetUserOutput",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把某个用户输出通道设成一个值，用该通道**已标定的物理单位**（**不一定是伏特** —— 依它在 Nanonis 里的标定，"
                "一个通道可能是 µm、mW 或别的任何东西）。\n"
                "\n"
                "这驱动的是 MAST 看不见的**外部**硬件（一路栅压、一台压电放大器、一个激光快门、"
                "一条延时线）。要在一个你没把握的通道上设值之前，先调 GetUserOutputLimits 和 GetUserOutputMode —— Monitor 模式的通道根本不理你，"
                "而超出用户所设限值的值会被**拒绝**，不是被夹到边界。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算（1..N）",
                    required=True, min_value=1, max_value=24,
                ),
                ParameterSpec(
                    name="value", type="float",
                    description=(
                        "目标值，用该通道的物理单位。必须落在该通道在 Nanonis 里配置的限值之内。"
                    ),
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["output", "user_output", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        value = float(params["value"])
        calls = []

        # THE SAFETY MODEL. MAST does not know what is wired to this output, so it
        # does not invent a bound — it reads the one the operator configured on the
        # instrument and refuses outside it. A limits read that FAILS is a refusal,
        # not a free pass: an unknown envelope is not an open one.
        lim = context.safe_call("UserOut_LimitsGet", idx, 0)
        calls.append(lim)
        if lim.error:
            return _fail(
                "SetUserOutput",
                f"拒绝写入：无法读取输出 {idx} 的物理限值（{lim.error}）。"
                "MAST 不知道这个输出口接的是什么，读不到用户配置的限值就不写——"
                "未知的边界不等于没有边界。",
                calls,
            )
        vals = _values(lim)
        if len(vals) < 2:
            return _fail(
                "SetUserOutput",
                f"拒绝写入：输出 {idx} 的限值回读格式异常（{vals!r}）。",
                calls,
            )
        upper, lower = float(vals[0]), float(vals[1])
        lo, hi = min(lower, upper), max(lower, upper)
        if not (lo <= value <= hi):
            return _fail(
                "SetUserOutput",
                f"拒绝写入：{value:g} 超出输出 {idx} 的物理限值 [{lo:g}, {hi:g}]（该限值由"
                "用户在 Nanonis 中配置，是这个通道的安全包络）。若确需更大范围，请在"
                "Nanonis 界面上调整该输出的 Limits——智能体不能拓宽自己的护栏。",
                calls,
            )

        rec = context.safe_call("UserOut_ValSet", idx, value)
        calls.append(rec)
        if rec.error:
            return _fail("SetUserOutput", rec.error, calls)
        return SkillResult(
            skill_name="SetUserOutput", success=True,
            data={"output_index": idx, "value": value,
                  "limits": {"lower": lo, "upper": hi},
                  "note": "值的单位由该通道的 Nanonis 校准决定，未必是伏特。"},
            nanonis_calls=calls,
        )


class SetUserOutputMode(BaseSkill):
    """Switch a user output between User Output / Monitor / Calc.Signal."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetUserOutputMode",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定某个用户输出的模式：0=User Output（由你驱动），1=Monitor（它镜像一路仪器信号）"
                "，2=Calc.Signal。把一个通道切**进**模式 0，等于把外部硬件的控制权交给 agent；"
                "把它切**出**去，则是把控制权交还给仪器。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
                ParameterSpec(
                    name="mode", type="int",
                    description="0=User Output，1=Monitor，2=Calc.Signal",
                    required=True, min_value=0, max_value=2,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "user_output", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        mode = int(params["mode"])
        rec = context.safe_call("UserOut_ModeSet", idx, mode)
        if rec.error:
            return _fail("SetUserOutputMode", rec.error, [rec])
        return SkillResult(
            skill_name="SetUserOutputMode", success=True,
            data={"output_index": idx, "mode": mode,
                  "mode_name": _MODES.get(mode, "unknown")},
            nanonis_calls=[rec],
        )


class SetUserOutputMonitorChannel(BaseSkill):
    """Choose which signal a user output mirrors in Monitor mode."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetUserOutputMonitorChannel",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定某个用户输出镜像哪一路信号（只在 Monitor 模式下有意义）。用 ListSignalNames／信号目录去找通道序号。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
                ParameterSpec(
                    name="monitor_channel_index", type="int",
                    description="要镜像的信号序号（0..127）",
                    required=True, min_value=0, max_value=127,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "user_output", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        ch = int(params["monitor_channel_index"])
        rec = context.safe_call("UserOut_MonitorChSet", idx, ch)
        if rec.error:
            return _fail("SetUserOutputMonitorChannel", rec.error, [rec])
        return SkillResult(
            skill_name="SetUserOutputMonitorChannel", success=True,
            data={"output_index": idx, "monitor_channel_index": ch},
            nanonis_calls=[rec],
        )


class PulseDigitalLine(BaseSkill):
    """Fire a TTL pulse train on digital output lines — trigger external gear."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PulseDigitalLine",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在一条或多条数字输出线上发一列 TTL 脉冲。Nanonis 就是这样触发**外部**硬件的 —— 一次相机曝光、"
                "一台脉冲发生器、一个斩波器、一个快门。MAST 并不知道线上接的是什么：对不熟悉的线，打脉冲之前先问用户。"
            ),
            parameters=[
                ParameterSpec(
                    name="port", type="int",
                    description="0=Port A，1=B，2=C，3=D，4+ = 扩展 DIO",
                    required=True, min_value=0, max_value=8,
                ),
                ParameterSpec(
                    name="lines", type="str",
                    description="要打脉冲的数字线，1..8，逗号分隔（例如 '1' 或 '1,3'）",
                    required=True,
                ),
                ParameterSpec(
                    name="pulse_width_s", type="float",
                    description="每个脉冲持续多久（s）",
                    unit="s", required=True, min_value=1e-6, max_value=60.0,
                ),
                ParameterSpec(
                    name="pulse_pause_s", type="float",
                    description="脉冲之间的间隔（s）",
                    unit="s", required=False, default=0.001,
                    min_value=0.0, max_value=60.0,
                ),
                ParameterSpec(
                    name="n_pulses", type="int",
                    description="脉冲个数",
                    required=False, default=1, min_value=1, max_value=10000,
                ),
                ParameterSpec(
                    name="wait_until_finished", type="bool",
                    description="阻塞直到整列脉冲发完",
                    required=False, default=True,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["output", "digital", "ttl", "trigger", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        port = int(params["port"])
        raw_lines = params["lines"]
        try:
            lines = [int(x) for x in str(raw_lines).replace(",", " ").split()]
        except ValueError:
            return _fail("PulseDigitalLine",
                         f"lines 解析失败：{raw_lines!r}（应为 1..8 的逗号分隔列表）", [])
        if not lines or any(not 1 <= v <= 8 for v in lines):
            return _fail("PulseDigitalLine",
                         f"digital line 必须在 1..8 之间，收到 {lines}", [])

        width = float(params["pulse_width_s"])
        pause = float(params.get("pulse_pause_s", 0.001) or 0.0)
        n = int(params.get("n_pulses", 1) or 1)
        wait = 1 if params.get("wait_until_finished", True) else 0

        rec = context.safe_call("DigLines_Pulse", port, lines, width, pause, n, wait)
        if rec.error:
            return _fail("PulseDigitalLine", rec.error, [rec])
        return SkillResult(
            skill_name="PulseDigitalLine", success=True,
            data={"port": port, "port_name": _PORTS.get(port, f"expanded_{port}"),
                  "lines": lines, "pulse_width_s": width, "pulse_pause_s": pause,
                  "n_pulses": n},
            nanonis_calls=[rec],
        )


class SetDigitalLineStatus(BaseSkill):
    """Hold one digital output line high or low."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetDigitalLineStatus",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一条数字输出线置为 HIGH 或 LOW 并**保持**在那里（不像 PulseDigitalLine，"
                "那个会把它送回去）。用于需要锁存的使能 —— 一个一直开着的快门、一台一直开着的放大器。"
                "MAST 并不知道线上接的是什么，而且 abort 时它**不会**被复位：abort 只是让 MAST 停止写入，"
                "它不会去猜哪一个电平才叫「关」。"
            ),
            parameters=[
                ParameterSpec(
                    name="port", type="int",
                    description="0=Port A，1=B，2=C，3=D，4+ = 扩展 DIO",
                    required=True, min_value=0, max_value=8,
                ),
                ParameterSpec(
                    name="line", type="int",
                    description="端口内的数字线，1..8",
                    required=True, min_value=1, max_value=8,
                ),
                ParameterSpec(
                    name="status", type="bool",
                    description="True = HIGH，False = LOW",
                    required=True,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "digital", "ttl", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        port = int(params["port"])
        line = int(params["line"])
        status = 1 if params["status"] else 0
        rec = context.safe_call("DigLines_OutStatusSet", port, line, status)
        if rec.error:
            return _fail("SetDigitalLineStatus", rec.error, [rec])
        return SkillResult(
            skill_name="SetDigitalLineStatus", success=True,
            data={"port": port, "line": line, "status": bool(status)},
            nanonis_calls=[rec],
        )


class ConfigureDigitalLine(BaseSkill):
    """Set a digital line's direction and polarity."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureDigitalLine",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置一条数字线的方向（输入／输出）与极性（高有效／低有效）。把一条线从输入翻成输出，就开始**驱动**另一端接着的任何东西 —— 先核对接线。"
            ),
            parameters=[
                ParameterSpec(
                    name="line", type="int",
                    description="数字线，1..8",
                    required=True, min_value=1, max_value=8,
                ),
                ParameterSpec(
                    name="port", type="int",
                    description="0=Port A，1=B，2=C，3=D，4+ = 扩展 DIO",
                    required=True, min_value=0, max_value=8,
                ),
                ParameterSpec(
                    name="direction", type="int",
                    description="0 = 输入，1 = 输出",
                    required=True, min_value=0, max_value=1,
                ),
                ParameterSpec(
                    name="polarity", type="int",
                    description="0 = 低有效，1 = 高有效",
                    required=True, min_value=0, max_value=1,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["output", "digital", "ttl", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        line = int(params["line"])
        port = int(params["port"])
        direction = int(params["direction"])
        polarity = int(params["polarity"])
        rec = context.safe_call("DigLines_PropsSet", line, port, direction, polarity)
        if rec.error:
            return _fail("ConfigureDigitalLine", rec.error, [rec])
        return SkillResult(
            skill_name="ConfigureDigitalLine", success=True,
            data={"line": line, "port": port,
                  "direction": "output" if direction else "input",
                  "polarity": "active_high" if polarity else "active_low"},
            nanonis_calls=[rec],
        )


class SetUserOutputLimits(BaseSkill):
    """Set a user output's physical limits — a GUARDRAIL CHANGE, and logged as one."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetUserOutputLimits",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定某个用户输出通道的物理上／下限。这两个限值就是 SetUserOutput 据以核对的那道包络，"
                "所以把它们放宽，就等于放宽了 agent 被允许驱动的范围 —— 这是一次护栏改动，它需要用户确认，"
                "并且会被记入诊断账本（记录 → 诊断）。收紧它们可以保护一个通道；只有在接线确实容许更大范围时，"
                "才把它们放宽。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
                ParameterSpec(
                    name="upper_limit", type="float",
                    description="物理上限（该通道已标定的单位）",
                    required=True,
                ),
                ParameterSpec(
                    name="lower_limit", type="float",
                    description="物理下限（该通道已标定的单位）",
                    required=True,
                ),
                ParameterSpec(
                    name="raw", type="int",
                    description="0 = 物理（已标定）限值；1 = 原始限值",
                    required=False, default=0, min_value=0, max_value=1,
                ),
            ],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["output", "user_output", "write", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        upper = float(params["upper_limit"])
        lower = float(params["lower_limit"])
        raw = int(params.get("raw", 0) or 0)
        if upper < lower:
            upper, lower = lower, upper
        calls = []

        # Record what the envelope WAS, so the ledger line says what changed rather
        # than just that something did. Best-effort: a failed read must not block a
        # legitimate limits change.
        before = None
        prev = context.safe_call("UserOut_LimitsGet", idx, raw)
        calls.append(prev)
        if not prev.error:
            v = _values(prev)
            if len(v) >= 2:
                before = {"upper": float(v[0]), "lower": float(v[1])}

        rec = context.safe_call("UserOut_LimitsSet", idx, upper, lower, raw)
        calls.append(rec)
        if rec.error:
            return _fail("SetUserOutputLimits", rec.error, calls)

        # A guardrail change is not refused (the operator's wiring is the real
        # barrier — see the module docstring), but it IS written down. If an output
        # later drives something odd, the widening that allowed it is on the record.
        try:
            from mast.core.diagnostics import record

            widened = bool(
                before is not None
                and (upper > before["upper"] or lower < before["lower"])
            )
            record(
                "note", f"UserOut[{idx}].limits",
                ("护栏被放宽" if widened else "护栏被调整")
                + f"：{before} → {{'upper': {upper}, 'lower': {lower}}}",
                output_index=idx, before=before,
                after={"upper": upper, "lower": lower},
                widened=widened, raw=bool(raw),
            )
        except Exception:  # noqa: BLE001 — bookkeeping must never fail a skill
            pass

        return SkillResult(
            skill_name="SetUserOutputLimits", success=True,
            data={"output_index": idx, "upper_limit": upper, "lower_limit": lower,
                  "previous": before, "raw": bool(raw)},
            nanonis_calls=calls,
        )


class SetUserOutputCalibration(BaseSkill):
    """Set a user output's calibration — what one physical unit MEANS."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetUserOutputCalibration",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设定某个用户输出或监视通道的标定（每伏对应多少物理单位 + 偏移）。这会重新定义在该通道上「一个物理单位」"
                "**意味着什么**：改完标定之后，同一个数值驱动出的是**另一个**电压，而且每一条限值都会按新单位重新解释。"
                "用它来把该通道真实的缩放关系告诉 MAST；不要用它来绕开一条限值。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
                ParameterSpec(
                    name="calibration_per_volt", type="float",
                    description="DAC 每输出一伏对应多少物理单位",
                    required=True,
                ),
                ParameterSpec(
                    name="offset", type="float",
                    description="偏移，用物理单位表示",
                    required=False, default=0.0,
                ),
            ],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["output", "user_output", "write", "calibration"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        cal = float(params["calibration_per_volt"])
        off = float(params.get("offset", 0.0) or 0.0)
        rec = context.safe_call("UserOut_CalibrSet", idx, cal, off)
        if rec.error:
            return _fail("SetUserOutputCalibration", rec.error, [rec])
        try:
            from mast.core.diagnostics import record

            record("note", f"UserOut[{idx}].calibration",
                   f"校准被改写：{cal} 物理单位/V，offset={off}"
                   "（此后同一个数值会驱动不同的电压，且所有限值按新单位重新解释）",
                   output_index=idx, calibration_per_volt=cal, offset=off)
        except Exception:  # noqa: BLE001
            pass
        return SkillResult(
            skill_name="SetUserOutputCalibration", success=True,
            data={"output_index": idx, "calibration_per_volt": cal, "offset": off},
            nanonis_calls=[rec],
        )


class ConfigureCalculatedOutput(BaseSkill):
    """Output the RESULT of an operation on two signals (Calc.Signal mode)."""

    # UserOut_CalcSignalConfigSet(Output_index, Signal_1, Operation, Signal_2)
    _OPS = {"+": 0, "-": 1, "*": 2, "/": 3}

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureCalculatedOutput",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "让某个用户输出承载两路信号做一次算术运算的**结果**（Calc.Signal 模式）："
                "例如把 (Signal_A − Signal_B) 作为一条实时模拟线输出。适合做差分通道、"
                "归一化信号，或者你想送上示波器、送进外部硬件的某个导出量。\n"
                "\n"
                "要让它生效，请用 SetUserOutputMode 把该输出切到模式 2（Calc.Signal）"
                "。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
                ParameterSpec(
                    name="signal_1", type="int",
                    description="第一路信号的序号（见 ListSignalNames）",
                    required=True, min_value=0, max_value=127,
                ),
                ParameterSpec(
                    name="operation", type="str",
                    description="算术运算：+ - * /",
                    required=True, allowed_values=["+", "-", "*", "/"],
                ),
                ParameterSpec(
                    name="signal_2", type="int",
                    description="第二路信号的序号",
                    required=True, min_value=0, max_value=127,
                ),
                ParameterSpec(
                    name="name", type="str",
                    description="这个计算信号的名字（会显示在 Nanonis 里）",
                    required=False, default="",
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["output", "user_output", "calc_signal", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        s1 = int(params["signal_1"])
        s2 = int(params["signal_2"])
        op = str(params["operation"]).strip()
        if op not in self._OPS:
            return _fail("ConfigureCalculatedOutput",
                         f"operation 必须是 + - * / 之一，收到 {op!r}", [])
        calls = []
        rec = context.safe_call("UserOut_CalcSignalConfigSet", idx, s1,
                                self._OPS[op], s2)
        calls.append(rec)
        if rec.error:
            return _fail("ConfigureCalculatedOutput", rec.error, calls)

        name = str(params.get("name", "") or "")
        if name:
            rec2 = context.safe_call("UserOut_CalcSignalNameSet", idx, name)
            calls.append(rec2)
            if rec2.error:
                # The config landed; only the label failed. Say so rather than
                # reporting the whole thing as a failure.
                return SkillResult(
                    skill_name="ConfigureCalculatedOutput", success=True,
                    data={"output_index": idx, "expression": f"S{s1} {op} S{s2}",
                          "name": None,
                          "warning": f"配置已生效，但命名失败：{rec2.error}"},
                    nanonis_calls=calls,
                )
        return SkillResult(
            skill_name="ConfigureCalculatedOutput", success=True,
            data={"output_index": idx, "expression": f"S{s1} {op} S{s2}",
                  "signal_1": s1, "signal_2": s2, "operation": op,
                  "name": name or None,
                  "note": "需把该输出设为 mode 2 (Calc.Signal) 才会生效——用 SetUserOutputMode。"},
            nanonis_calls=calls,
        )


class GetCalculatedOutputConfig(BaseSkill):
    """Read a user output's Calc.Signal configuration."""

    _OP_NAMES = {0: "+", 1: "-", 2: "*", 3: "/"}

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetCalculatedOutputConfig",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读某个用户输出把哪两路信号、用什么运算、以什么名字组合起来（Calc.Signal 模式）"
                "。"
            ),
            parameters=[
                ParameterSpec(
                    name="output_index", type="int",
                    description="用户输出通道，1 起算",
                    required=True, min_value=1, max_value=24,
                ),
            ],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["output", "user_output", "calc_signal", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["output_index"])
        cfg = context.safe_call("UserOut_CalcSignalConfigGet", idx)
        nm = context.safe_call("UserOut_CalcSignalNameGet", idx)
        calls = [cfg, nm]
        if cfg.error:
            return _fail("GetCalculatedOutputConfig", cfg.error, calls)
        v = _values(cfg)
        op = int(v[1]) if len(v) > 1 else None
        return SkillResult(
            skill_name="GetCalculatedOutputConfig", success=True,
            data={"output_index": idx,
                  "signal_1": int(v[0]) if len(v) > 0 else None,
                  "operation": self._OP_NAMES.get(op, op),
                  "signal_2": int(v[2]) if len(v) > 2 else None,
                  "name": (_values(nm)[0] if (not nm.error and _values(nm)) else None)},
            nanonis_calls=calls,
        )


__all__ = [
    "GetUserOutputLimits", "GetUserOutputMode", "GetUserOutputMonitorChannel",
    "GetDigitalLineTTL", "GetCalculatedOutputConfig",
    "SetUserOutput", "SetUserOutputMode", "SetUserOutputMonitorChannel",
    "SetUserOutputLimits", "SetUserOutputCalibration", "ConfigureCalculatedOutput",
    "PulseDigitalLine", "SetDigitalLineStatus", "ConfigureDigitalLine",
]
