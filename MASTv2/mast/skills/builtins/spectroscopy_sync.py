"""External synchronisation for spectroscopy — the pump-probe half MAST never had.

Nanonis lets bias- and Z-spectroscopy fire, and be fired by, the outside world:

  * **TTL sync** — pulse a TTL line at a chosen time relative to each spectroscopy
    point, for a chosen duration. This is how a spectrum triggers a pump laser, a
    shutter, a spectrometer's exposure.
  * **Digital sync** — gate on a digital line.
  * **Pulse-sequence sync** — run a pulse sequence (the pulse generator's programmed
    train) for N periods at each point.

MAST wrapped **every one of the getters** for these — ``GetSTSTTLSync``,
``GetSTSDigSync``, ``GetSTSPulseSeqSync``, ``GetZSpectrTTLSync`` and the rest — and
**none of the setters**. So the agent could read the pump-probe synchronisation
configuration and could not change it. On a TERS rig, where the whole point is
correlating a tunnelling spectrum with an optical event, that is the missing half.

ALSO HERE
=========
  * ``BiasSpectr_StatusGet`` / ``ZSpectr_StatusGet`` — **is the spectroscopy still
    running?** Neither was wrapped, so a spectroscopy could only be run blocking.
    Now it can be polled, which is what an agent that also wants to watch the
    current needs.
  * The alternate-Z-setpoint machinery (``AltZCtrlSet`` / ``ZOffRevertSet``): move
    to a different, usually closer, Z setpoint for the duration of the sweep and
    revert afterwards. The getters existed; the setters did not.
  * ``ZSpectr_RetractSecondSet`` — a SECOND retract condition on Z spectroscopy. A
    safety feature, and it was read-only.

UNITS AND OFF-BY-ONES THAT WILL BITE
====================================
  * TTL times are in **SECONDS**. 1 ms is 0.001, and passing 1 means the line stays
    on for a second — long enough to dump a laser pulse train into the junction.
  * ``TTL_line`` and ``Pulse_Sequence_Nr`` follow Nanonis' own numbering. Read the
    current configuration with GetSTSTTLSync before assuming a base.
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

_POLARITY = {"low_active": 0, "high_active": 1}
# BiasSpectr_StatusGet / ZSpectr_StatusGet: 0 = not running, 1 = running
_RUNNING = 1


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """用 io.nanonis_files.decode_reply 提取回包内容，排除错误与原始字节信封。
    
    原始 bytes 不应进入需要 JSON 序列化的结果；所有调用方共用同一解码入口。"""
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


def _first_int(value):
    stack = [value]
    seen = 0
    while stack and seen < 64:
        seen += 1
        v = stack.pop(0)
        if isinstance(v, (list, tuple)):
            stack = list(v) + stack
            continue
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
    return None


class SetSpectroscopyTtlSync(BaseSkill):
    """Pulse a TTL line in step with each spectroscopy point."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSpectroscopyTtlSync",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "让谱学在每一个点上给一条 TTL 线发脉冲 —— 泵浦激光、"
                "快门或光谱仪曝光就挂在这个钩子上。\n"
                "\n"
                "时间一律以 SECONDS 计。`time_to_on_s` 是从该点开始到线变为有效之间的延迟；"
                "`on_duration_s` 是它保持有效的时长。1 ms 就是 0.001 —— 传 1 "
                "会让这条线在扫描的每一个点上都保持整整一秒。\n"
                "\n"
                "设 line=0 即可 DISABLE 这个同步。先用 GetSTSTTLSync（bias）或 GetZSpectrTTLSync（Z）"
                "读回当前配置 —— 线号用的是 Nanonis 的编号，不是 MAST 的，"
                "脉冲发错线就会触发接在那条线上的任何东西。"
            ),
            parameters=[
                ParameterSpec(name="which", type="str",
                              description="bias（bias 谱学）或 z（Z 谱学）",
                              required=True, allowed_values=["bias", "z"]),
                ParameterSpec(name="line", type="int",
                              description="TTL 线号（Nanonis 编号）。0 = 关掉该同步。",
                              required=True, min_value=0, max_value=8),
                ParameterSpec(name="polarity", type="str",
                              description="high_active（线拉高）或 low_active",
                              required=False, default="high_active",
                              allowed_values=list(_POLARITY)),
                ParameterSpec(name="time_to_on_s", type="float",
                              description="从该点开始到线变为有效之间的延迟，单位 SECONDS",
                              unit="s", required=False, default=0.0,
                              min_value=0.0, max_value=10.0),
                ParameterSpec(name="on_duration_s", type="float",
                              description="线保持有效的时长，单位 SECONDS（1 ms = 0.001）",
                              unit="s", required=False, default=1e-3,
                              min_value=0.0, max_value=10.0),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["spectroscopy", "sync", "ttl", "pump-probe", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetSpectroscopyTtlSync"
        which = str(params["which"])
        line = int(params["line"])
        pol = _POLARITY[str(params.get("polarity", "high_active") or "high_active")]
        t_on = float(params.get("time_to_on_s", 0.0) or 0.0)
        dur = float(params.get("on_duration_s", 1e-3) or 1e-3)

        # Literal verbs — every MAST safety tool (the safety audit, the abort-policy
        # check, the API coverage census) greps for safe_call("literal"). A verb
        # behind a variable is invisible to all of them.
        if which == "bias":
            rec = context.safe_call("BiasSpectr_TTLSyncSet", line, pol, t_on, dur)
            back = context.safe_call("BiasSpectr_TTLSyncGet")
        else:
            rec = context.safe_call("ZSpectr_TTLSyncSet", line, pol, t_on, dur)
            back = context.safe_call("ZSpectr_TTLSyncGet")
        calls = [rec, back]
        if rec.error:
            return _fail(name, f"TTL sync set failed ({which}): {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"which": which, "line": line, "polarity": params.get("polarity"),
                  "time_to_on_s": t_on, "on_duration_s": dur,
                  "readback": None if back.error else _rv(back)},
            summary=(f"{which} 谱学 TTL 同步已{'关闭' if line == 0 else '设置'}"
                     + ("" if line == 0 else
                        f"：线 {line}，延迟 {t_on:g} s，持续 {dur:g} s")),
        )


class SetSpectroscopyPulseSync(BaseSkill):
    """Digital-line gating and pulse-sequence sync."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSpectroscopyPulseSync",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置谱学的数字线门控和/或它的脉冲序列同步。\n"
                "\n"
                "`digital_sync` 用一条数字线给扫描加门控。`pulse_sequence_nr` "
                "在每一个谱学点上运行脉冲发生器里某个已编程的序列，运行 `pulse_periods` 个周期 —— "
                "这就是隧道一侧做 pump-probe 延时扫描的机制。\n"
                "\n"
                "把其中任一项设为 0 即可关掉它。省略某个参数则保持它原样不动。"
            ),
            parameters=[
                ParameterSpec(name="which", type="str",
                              description="bias 或 z",
                              required=True, allowed_values=["bias", "z"]),
                ParameterSpec(name="digital_sync", type="int",
                              description="数字同步线（0 = 关）—— 省略则保持不变",
                              required=False, default=None, min_value=0, max_value=8),
                ParameterSpec(name="pulse_sequence_nr", type="int",
                              description="脉冲序列编号（0 = 关）—— 省略则保持不变",
                              required=False, default=None, min_value=0, max_value=8),
                ParameterSpec(name="pulse_periods", type="int",
                              description="每个点上跑该序列多少个周期",
                              required=False, default=1, min_value=1, max_value=1_000_000),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["spectroscopy", "sync", "pulse", "pump-probe", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetSpectroscopyPulseSync"
        which = str(params["which"])
        calls: list = []
        touched: list[str] = []

        dig = params.get("digital_sync")
        if dig is not None:
            if which == "bias":
                rec = context.safe_call("BiasSpectr_DigSyncSet", int(dig))
            else:
                rec = context.safe_call("ZSpectr_DigSyncSet", int(dig))
            calls.append(rec)
            if rec.error:
                return _fail(name, f"digital sync set failed: {rec.error}", calls)
            touched.append(f"digital_sync={int(dig)}")

        seq = params.get("pulse_sequence_nr")
        if seq is not None:
            periods = int(params.get("pulse_periods", 1) or 1)
            if which == "bias":
                rec = context.safe_call("BiasSpectr_PulseSeqSyncSet", int(seq), periods)
            else:
                rec = context.safe_call("ZSpectr_PulseSeqSyncSet", int(seq), periods)
            calls.append(rec)
            if rec.error:
                return _fail(name, f"pulse-sequence sync set failed: {rec.error}", calls)
            touched.append(f"pulse_seq={int(seq)}×{periods}")

        if not touched:
            return _fail(name, "digital_sync / pulse_sequence_nr 至少要给一个", calls)

        return SkillResult(skill_name=name, success=True, nanonis_calls=calls,
                           data={"which": which, "changed": touched},
                           summary=f"{which} 谱学同步已设置：{', '.join(touched)}")


class SetSpectroscopyZControl(BaseSkill):
    """The alternate Z setpoint used during the sweep, and what happens after."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSpectroscopyZControl",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "配置谱学如何对待 Z 控制器：扫描前它要移到的 ALTERNATE SETPOINT，以及事后是否恢复 Z "
                "偏移。\n"
                "\n"
                "alternate setpoint 是让你在与扫描时不同的（通常更近、电流更大的）"
                "针尖-样品间距上取谱的办法：环路先稳定到 `setpoint`，等待 `settling_time_s`，"
                "然后环路才断开、扫描才开始。\n"
                "\n"
                "**更近的 setpoint 意味着更小的间隙。** 决定反馈松手之前针尖靠得多近的，就是这个参数。"
                "把它抬到远高于扫描用的 setpoint 会把针尖往里推；再配上一段大范围的 bias 扫描，"
                "那就是在改造表面，而不是在测量它。\n"
                "\n"
                "`revert_z_offset` 会在事后恢复扫描前的 Z。除非你就是要针尖停在扫描结束时的位置，"
                "否则让它保持 ON。"
            ),
            parameters=[
                ParameterSpec(name="use_alternate_setpoint", type="bool",
                              description="扫描前先移到一个 alternate Z setpoint",
                              required=True),
                ParameterSpec(name="setpoint", type="float",
                              description="alternate 的 Z 控制器 setpoint（电流，单位 A）",
                              unit="A", required=False, default=0.0),
                ParameterSpec(name="settling_time_s", type="float",
                              description="断开之前，让环路在该值上稳定多久",
                              unit="s", required=False, default=0.1,
                              min_value=0.0, max_value=60.0),
                ParameterSpec(name="revert_z_offset", type="bool",
                              description="事后恢复扫描前的 Z 偏移",
                              required=False, default=True),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "zcontroller", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetSpectroscopyZControl"
        use = bool(params["use_alternate_setpoint"])
        calls: list = []

        rec = context.safe_call("BiasSpectr_AltZCtrlSet",
                                1 if use else 0,
                                float(params.get("setpoint", 0.0) or 0.0),
                                float(params.get("settling_time_s", 0.1) or 0.1))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"BiasSpectr_AltZCtrlSet failed: {rec.error}", calls)

        rec = context.safe_call("BiasSpectr_ZOffRevertSet",
                                1 if bool(params.get("revert_z_offset", True)) else 0)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"BiasSpectr_ZOffRevertSet failed: {rec.error}", calls)

        back = context.safe_call("BiasSpectr_AltZCtrlGet")
        calls.append(back)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"use_alternate_setpoint": use,
                  "setpoint": float(params.get("setpoint", 0.0) or 0.0),
                  "revert_z_offset": bool(params.get("revert_z_offset", True)),
                  "readback": None if back.error else _rv(back)},
            summary=(f"谱学 Z 控制已设置：备用设定点"
                     f"{'启用 = %.3e A' % float(params.get('setpoint', 0.0) or 0.0) if use else '关闭'}"),
        )


class SetZSpectroscopySecondRetract(BaseSkill):
    """A second, signal-based retract condition for Z spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZSpectroscopySecondRetract",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "给 Z 谱学加第二条（SECOND）retract 条件：当选定信号越过阈值时，中止扫描并退针。\n"
                "\n"
                "这是一个安全特性，而在此之前它一直是只读的。Z 扫描会在开环状态下把针尖推向表面 —— "
                "retract 条件就是在电流说「到了」的时候把它停住的那个东西，而不是等扫描走到量程尽头。"
                "\n"
                "\n"
                "comparison：0 = 信号高于阈值时退针，1 = 低于阈值时退针。"
                "弄反了就意味着这个条件永远不会触发 —— 用 GetZSpectrRetract2nd 读回来核对。"
            ),
            parameters=[
                ParameterSpec(name="enable", type="bool",
                              description="启用第二条 retract 条件",
                              required=True),
                ParameterSpec(name="signal_index", type="int",
                              description="要盯的信号（取自 ListSignalNames）",
                              required=False, default=0, min_value=0, max_value=127),
                ParameterSpec(name="threshold", type="float",
                              description="阈值，用该信号自己的单位",
                              required=False, default=0.0),
                ParameterSpec(name="comparison", type="int",
                              description="0 = 高于阈值时退针，1 = 低于阈值时退针",
                              required=False, default=0, min_value=0, max_value=1),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["spectroscopy", "zspectr", "retract", "safety", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetZSpectroscopySecondRetract"
        # ZSpectr_RetractSecondSet(Second_condition, Threshold, Signal_index, Comparison)
        rec = context.safe_call(
            "ZSpectr_RetractSecondSet",
            1 if bool(params["enable"]) else 0,
            float(params.get("threshold", 0.0) or 0.0),
            int(params.get("signal_index", 0) or 0),
            int(params.get("comparison", 0) or 0),
        )
        calls = [rec]
        if rec.error:
            return _fail(name, f"ZSpectr_RetractSecondSet failed: {rec.error}", calls)
        back = context.safe_call("ZSpectr_RetractSecondGet")
        calls.append(back)
        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"enabled": bool(params["enable"]),
                  "signal_index": int(params.get("signal_index", 0) or 0),
                  "threshold": float(params.get("threshold", 0.0) or 0.0),
                  "readback": None if back.error else _rv(back)},
            summary=("Z 谱学第二退回条件已" +
                     ("启用" if params["enable"] else "关闭")),
        )


class SetMlsLockinPerSegment(BaseSkill):
    """Lock-in per MLS segment."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetMlsLockinPerSegment",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "按多线段（MLS）bias 谱学的每一个 SEGMENT 分别开关 lock-in，"
                "而不是对整条扫描一次性开关。\n"
                "\n"
                "MLS 让你在一条曲线里以不同速度扫不同的 bias 区间。这个开关让 lock-in（dI/dV）"
                "只在你想要的那些段里运行 —— 慢的、密的那些 —— "
                "而不必拖着它的时间常数走完快的那些段。"
            ),
            parameters=[
                ParameterSpec(name="enable", type="bool",
                              description="lock-in 按段分别配置（相对于对整条扫描统一配置）",
                              required=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["spectroscopy", "mls", "lockin", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("BiasSpectr_MLSLockinPerSegSet",
                                1 if bool(params["enable"]) else 0)
        if rec.error:
            return _fail("SetMlsLockinPerSegment",
                         f"BiasSpectr_MLSLockinPerSegSet failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetMlsLockinPerSegment", success=True,
                           nanonis_calls=[rec], data={"enabled": bool(params["enable"])},
                           summary=f"MLS 分段锁相已{'启用' if params['enable'] else '关闭'}")


class GetSpectroscopyStatus(BaseSkill):
    """Is the spectroscopy still running?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSpectroscopyStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "查询 bias 谱学或 Z 谱学当前是不是正在 RUNNING。\n"
                "\n"
                "在此之前 MAST 只能以阻塞方式跑谱学 —— 启动，然后干等。有了它，你可以先启动再轮询："
                "盯着电流、查中止标志、上报进度。一大片谱的网格于是从「只能等它跑完」"
                "变成了「可以盯着它跑」。"
            ),
            parameters=[
                ParameterSpec(name="which", type="str",
                              description="bias | z | both",
                              required=False, default="both",
                              allowed_values=["bias", "z", "both"]),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "status", "read", "poll"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        which = str(params.get("which", "both") or "both")
        calls: list = []
        data: dict = {"which": which}
        if which in ("bias", "both"):
            rec = context.safe_call("BiasSpectr_StatusGet")
            calls.append(rec)
            v = None if rec.error else _first_int(_rv(rec))
            data["bias_running"] = None if v is None else (v == _RUNNING)
        if which in ("z", "both"):
            rec = context.safe_call("ZSpectr_StatusGet")
            calls.append(rec)
            v = None if rec.error else _first_int(_rv(rec))
            data["z_running"] = None if v is None else (v == _RUNNING)
        running = [k for k in ("bias_running", "z_running") if data.get(k) is True]
        return SkillResult(
            skill_name="GetSpectroscopyStatus", success=True, nanonis_calls=calls,
            data=data,
            summary=("谱学运行中：" + ", ".join(running)) if running else "谱学未在运行",
        )
