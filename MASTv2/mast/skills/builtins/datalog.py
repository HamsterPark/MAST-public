"""Data logging — record signals over time, to file or over TCP.

Nanonis has two loggers and MAST had a skill for neither:

  * **Data Logger** (``DataLog_*``, 8 API methods) — records selected channels to a
    file on the Nanonis machine, for a set duration or until stopped. This is what
    you use to watch a drift, a temperature settle, a slow degradation of the tip.
  * **TCP Logger** (``TCPLog_*``, 5 methods) — streams selected channels over TCP.
    Same idea, but the data comes to you.

Without these the agent could read a signal ONCE (``Signals_ValGet``) and had no
way to say "watch this for ten minutes". A long autonomous run that wants to know
whether the thermal drift has settled had to poll — burning tool calls, and
sampling at whatever cadence the LLM loop happened to run at.

Shaped as tasks: ``StartDataLog`` takes the channels and the duration together,
rather than making the model chain ``DataLog_Open`` → ``ChsSet`` → ``PropsSet`` →
``Start`` and get one of them wrong. The API's four calls are an implementation
detail, not four decisions.
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

# DataLog_PropsSet(Acquisition_mode, hours, minutes, seconds, Averaging, Basename,
#                  Comment, Modules_names_size, Modules_names)
# Acquisition mode: 0 = no change, 1 = continuous, 2 = timed
_MODE_TIMED = 2
_MODE_CONTINUOUS = 1


def _values(record) -> list:
    rv = getattr(record, "return_value", None)
    if isinstance(rv, (list, tuple)) and len(rv) > 2 and isinstance(rv[2], (list, tuple)):
        return list(rv[2])
    return []


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _channels(raw) -> list[int] | None:
    """'0, 2, 14' → [0, 2, 14]. None when unparseable."""
    try:
        out = [int(x) for x in str(raw).replace(",", " ").split()]
    except ValueError:
        return None
    return out or None


class StartDataLog(BaseSkill):
    """Record channels to a file on the Nanonis machine."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StartDataLog",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "把一路或多路信号通道录进一个文件，录固定时长、或一直录到被停止。想**盯着**某个量随时间怎么走时用它，"
                "而不是去轮询：热漂移的稳定过程、针尖的缓慢退化、一次长退火期间的电流。\n"
                "\n"
                "通道序号取自信号目录（0=Current，24=Z，…）—— 拿不准就调 ListSignalNames。"
                "录制不会碰仪器；它只读。"
            ),
            parameters=[
                ParameterSpec(
                    name="channels", type="str",
                    description="要录的信号序号，逗号分隔（例如 '0,24'）",
                    required=True,
                ),
                ParameterSpec(
                    name="duration_s", type="float",
                    description=(
                        "录制时长，单位秒。不传（或传 0）则一直录到 StopDataLog 为止。"
                    ),
                    unit="s", required=False, default=0.0,
                    min_value=0.0, max_value=86400.0,
                ),
                ParameterSpec(
                    name="basename", type="str",
                    description="Nanonis 机器上的文件基名",
                    required=False, default="mast_log",
                ),
                ParameterSpec(
                    name="averaging", type="int",
                    description="每个记录点平均多少个采样",
                    required=False, default=1, min_value=1, max_value=100000,
                ),
                ParameterSpec(
                    name="comment", type="str",
                    description="存进日志文件头的注释",
                    required=False, default="",
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["datalog", "record", "monitor"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        chs = _channels(params["channels"])
        if not chs:
            return _fail("StartDataLog",
                         f"channels 解析失败：{params['channels']!r}（应为逗号分隔的索引）", [])
        total = float(params.get("duration_s", 0.0) or 0.0)
        avg = int(params.get("averaging", 1) or 1)
        base = str(params.get("basename", "mast_log") or "mast_log")
        comment = str(params.get("comment", "") or "")
        calls = []

        rec = context.safe_call("DataLog_Open")
        calls.append(rec)
        if rec.error:
            return _fail("StartDataLog", f"DataLog_Open failed: {rec.error}", calls)

        rec = context.safe_call("DataLog_ChsSet", chs)
        calls.append(rec)
        if rec.error:
            return _fail("StartDataLog", f"DataLog_ChsSet failed: {rec.error}", calls)

        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        seconds = float(total % 60)
        mode = _MODE_TIMED if total > 0 else _MODE_CONTINUOUS
        rec = context.safe_call("DataLog_PropsSet", mode, hours, minutes, seconds,
                                avg, base, comment, 0, [])
        calls.append(rec)
        if rec.error:
            return _fail("StartDataLog", f"DataLog_PropsSet failed: {rec.error}", calls)

        rec = context.safe_call("DataLog_Start")
        calls.append(rec)
        if rec.error:
            return _fail("StartDataLog", f"DataLog_Start failed: {rec.error}", calls)

        return SkillResult(
            skill_name="StartDataLog", success=True,
            data={"channels": chs, "duration_s": total, "basename": base,
                  "mode": "timed" if total > 0 else "continuous",
                  "averaging": avg},
            nanonis_calls=calls,
        )


class StopDataLog(BaseSkill):
    """Stop the data logger."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopDataLog",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="停掉 Nanonis 数据记录器并关闭文件。",
            parameters=[],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["datalog", "record", "stop"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("DataLog_Stop")
        if rec.error:
            return _fail("StopDataLog", rec.error, [rec])
        return SkillResult(skill_name="StopDataLog", success=True,
                           data={"stopped": True}, nanonis_calls=[rec])


class GetDataLogStatus(BaseSkill):
    """Is the data logger running, and on what?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDataLogStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读数据记录器的状态（在跑／已停）、它配置的通道，以及它的各项属性。开一次新记录之前先看这个 —— 在一个正在跑的记录之上再开一个，"
                "会把前一个丢掉。"
            ),
            parameters=[],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["datalog", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        st = context.safe_call("DataLog_StatusGet")
        chs = context.safe_call("DataLog_ChsGet")
        props = context.safe_call("DataLog_PropsGet")
        calls = [st, chs, props]
        if st.error:
            return _fail("GetDataLogStatus", st.error, calls)
        return SkillResult(
            skill_name="GetDataLogStatus", success=True,
            data={"status": _values(st),
                  "channels": _values(chs) if not chs.error else [],
                  "props": _values(props) if not props.error else []},
            nanonis_calls=calls,
        )


class StartTcpLog(BaseSkill):
    """Stream channels over TCP instead of to a file."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StartTcpLog",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "启动 TCP 记录器：把选定的通道通过 TCP 流式送出，而不是录到 Nanonis 机器上的文件里。"
                "数据应当送到**这边**来的时候用它。只读 —— 不碰任何硬件。"
            ),
            parameters=[
                ParameterSpec(
                    name="channels", type="str",
                    description="要流式送出的信号序号，逗号分隔",
                    required=True,
                ),
                ParameterSpec(
                    name="oversampling", type="int",
                    description="每个送出点平均多少个采样",
                    required=False, default=10, min_value=1, max_value=100000,
                ),
            ],
            estimated_duration_s=0.6,
            composition_level=0,
            tags=["tcplog", "record", "stream"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        chs = _channels(params["channels"])
        if not chs:
            return _fail("StartTcpLog",
                         f"channels 解析失败：{params['channels']!r}", [])
        over = int(params.get("oversampling", 10) or 10)
        calls = []

        rec = context.safe_call("TCPLog_ChsSet", len(chs), chs)
        calls.append(rec)
        if rec.error:
            return _fail("StartTcpLog", f"TCPLog_ChsSet failed: {rec.error}", calls)

        rec = context.safe_call("TCPLog_OversamplSet", over)
        calls.append(rec)
        if rec.error:
            return _fail("StartTcpLog", f"TCPLog_OversamplSet failed: {rec.error}", calls)

        rec = context.safe_call("TCPLog_Start")
        calls.append(rec)
        if rec.error:
            return _fail("StartTcpLog", f"TCPLog_Start failed: {rec.error}", calls)

        return SkillResult(
            skill_name="StartTcpLog", success=True,
            data={"channels": chs, "oversampling": over},
            nanonis_calls=calls,
        )


class StopTcpLog(BaseSkill):
    """Stop the TCP logger."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopTcpLog",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="停掉 Nanonis TCP 记录器的数据流。",
            parameters=[],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["tcplog", "record", "stop"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("TCPLog_Stop")
        if rec.error:
            return _fail("StopTcpLog", rec.error, [rec])
        return SkillResult(skill_name="StopTcpLog", success=True,
                           data={"stopped": True}, nanonis_calls=[rec])


class GetTcpLogStatus(BaseSkill):
    """Is the TCP logger streaming?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetTcpLogStatus",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 TCP 记录器的状态（在流式送出／已停／出错）。",
            parameters=[],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["tcplog", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("TCPLog_StatusGet")
        if rec.error:
            return _fail("GetTcpLogStatus", rec.error, [rec])
        return SkillResult(skill_name="GetTcpLogStatus", success=True,
                           data={"status": _values(rec)}, nanonis_calls=[rec])


__all__ = ["StartDataLog", "StopDataLog", "GetDataLogStatus",
           "StartTcpLog", "StopTcpLog", "GetTcpLogStatus"]
