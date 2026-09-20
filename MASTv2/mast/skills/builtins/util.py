"""Utility skills: UI lock, settings save/load.

vendored from v1 mast/skills/builtins/util.py 2026-04-23"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill


class LockNanonisUI(BaseSkill):
    """Lock the Nanonis UI, restricting operator interaction until it is unlocked.

    Lock and unlock are separate operations because they have different effects
    on the ability to intervene. This skill declares its own safety level; the
    common execution path applies the configured checks. Unlocking remains an
    independent recovery operation.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="LockNanonisUI",
            version="2.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "锁定 Nanonis 软件 —— 这会在它上面盖一个**模态窗口**，并**阻止用户与仪器交互**，"
                "直到被解锁为止。\n"
                "\n"
                "DANGEROUS，而且不是因为它对硬件做了什么。这里其余每一道安全机制 —— 安全闸门、"
                "模式闸门、abort 按钮、审批提示 —— 归根结底都依赖于「有个人能走到显微镜跟前接管」"
                "。这个技能把那件事拿掉了。它需要人工批准正是因为这个原因，而且基本上没有哪个自主任务需要它："
                "用户若想在无人值守运行期间锁住界面，他自己锁就是了 —— 这恰恰就是重点所在。\n"
                "\n"
                "UnlockNanonisUI 是它的反面，并且永远允许。"
            ),
            parameters=[],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "lock", "dangerous", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_Lock")
        if record.error:
            return SkillResult(
                skill_name="LockNanonisUI",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="LockNanonisUI",
            success=True,
            data={"locked": True},
            summary="Nanonis 界面已锁定——用户在解锁前无法在仪器端操作",
            nanonis_calls=[record],
        )


class UnlockNanonisUI(BaseSkill):
    """Unlock the Nanonis UI — give the operator their instrument back."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="UnlockNanonisUI",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "解锁 Nanonis 软件，把仪器的控制权还给用户。\n"
                "\n"
                "永远允许，无需批准。恢复一个人介入的能力，从来不是危险的那个方向 —— 而「上一次运行结束时界面还锁着」"
                "是真实会发生的事。"
            ),
            parameters=[],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "lock", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_UnLock")
        if record.error:
            return SkillResult(
                skill_name="UnlockNanonisUI",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="UnlockNanonisUI",
            success=True,
            data={"locked": False},
            summary="Nanonis 界面已解锁",
            nanonis_calls=[record],
        )


class SaveSettings(BaseSkill):
    """Save or load Nanonis settings."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SaveSettings",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="把 Nanonis 设置存到 .ini 文件，或从 .ini 文件载入。",
            parameters=[
                ParameterSpec(
                    name="action",
                    type="str",
                    description="'save' 或 'load'",
                    required=True,
                    allowed_values=["save", "load"],
                ),
                ParameterSpec(
                    name="file_path",
                    type="str",
                    description="设置 .ini 文件的路径（use_session=True 时忽略）",
                    required=False,
                    default="",
                ),
                ParameterSpec(
                    name="use_session",
                    type="bool",
                    description="用当前会话文件，而不是 file_path",
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["util", "settings", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        action = params["action"]
        file_path = params.get("file_path", "")
        use_session = int(params.get("use_session", False))
        if action == "save":
            record = context.safe_call("Util_SettingsSave", file_path, use_session)
        else:
            record = context.safe_call("Util_SettingsLoad", file_path, use_session)
        if record.error:
            return SkillResult(
                skill_name="SaveSettings",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SaveSettings",
            success=True,
            data={
                "action": action,
                "file_path": file_path,
                "use_session": bool(use_session),
            },
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Util — AcqPeriodGet, LayoutLoad, LayoutSave, RTFreqGet, RTFreqSet,
#        RTOversamplGet, RTOversamplSet, SessionPathGet, SessionPathSet,
#        UnLock
# ---------------------------------------------------------------------------


class GetAcqPeriod(BaseSkill):
    """Get the TCP Receiver acquisition period."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetAcqPeriod",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 TCP Receiver 里的采集周期（s）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "acquisition", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_AcqPeriodGet")
        if record.error:
            return SkillResult(
                skill_name="GetAcqPeriod", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # Util.AcqPeriodGet ResponseTypes=["f"] -> parsed[2][0] is the
        # acquisition period (s). Reading parsed[0] gave the empty error
        # string and float("") crashed on real hardware.
        parsed = record.return_value
        period_s = 0.0
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                period_s = float(vals[0])
            elif isinstance(vals, (int, float)):
                period_s = float(vals)
        elif isinstance(parsed, (int, float)):
            period_s = float(parsed)
        return SkillResult(
            skill_name="GetAcqPeriod", success=True,
            data={"acquisition_period_s": period_s},
            nanonis_calls=[record],
        )


class LoadLayout(BaseSkill):
    """Load a Nanonis layout from an .ini file."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="LoadLayout",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="从 .ini 文件载入一份 Nanonis 布局。",
            parameters=[
                ParameterSpec(
                    name="file_path",
                    type="str",
                    description="布局 .ini 文件的路径（use_session=True 时忽略）",
                    required=False,
                    default="",
                ),
                ParameterSpec(
                    name="use_session",
                    type="bool",
                    description="从当前会话文件载入布局",
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["util", "layout", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        file_path = params.get("file_path", "")
        use_session = int(params.get("use_session", False))
        record = context.safe_call("Util_LayoutLoad", file_path, use_session)
        if record.error:
            return SkillResult(
                skill_name="LoadLayout", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="LoadLayout", success=True,
            data={"file_path": file_path, "use_session": bool(use_session)},
            nanonis_calls=[record],
        )


class SaveLayout(BaseSkill):
    """Save current Nanonis layout to an .ini file."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SaveLayout",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="把当前 Nanonis 布局存到 .ini 文件。",
            parameters=[
                ParameterSpec(
                    name="file_path",
                    type="str",
                    description="布局 .ini 文件的路径（use_session=True 时忽略）",
                    required=False,
                    default="",
                ),
                ParameterSpec(
                    name="use_session",
                    type="bool",
                    description="把布局存到当前会话文件",
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["util", "layout", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        file_path = params.get("file_path", "")
        use_session = int(params.get("use_session", False))
        record = context.safe_call("Util_LayoutSave", file_path, use_session)
        if record.error:
            return SkillResult(
                skill_name="SaveLayout", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SaveLayout", success=True,
            data={"file_path": file_path, "use_session": bool(use_session)},
            nanonis_calls=[record],
        )


class GetRTFreq(BaseSkill):
    """Get the Real Time controller frequency."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetRTFreq",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读实时控制器频率，单位 Hz。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "rt", "frequency", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_RTFreqGet")
        if record.error:
            return SkillResult(
                skill_name="GetRTFreq", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # Util.RTFreqGet ResponseTypes=["f"] -> parsed[2][0] is the RT
        # frequency (Hz). Reading parsed[0] gave the empty error string and
        # float("") crashed on real hardware.
        parsed = record.return_value
        freq_hz = 0.0
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                freq_hz = float(vals[0])
            elif isinstance(vals, (int, float)):
                freq_hz = float(vals)
        elif isinstance(parsed, (int, float)):
            freq_hz = float(parsed)
        return SkillResult(
            skill_name="GetRTFreq", success=True,
            data={"rt_frequency_hz": freq_hz},
            nanonis_calls=[record],
        )


class SetRTFreq(BaseSkill):
    """Set the Real Time controller frequency."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetRTFreq",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设定实时控制器频率，单位 Hz。",
            parameters=[
                ParameterSpec(
                    name="frequency_hz",
                    type="float",
                    description="RT 频率，单位 Hz",
                    unit="Hz",
                    required=True,
                    min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "rt", "frequency", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_RTFreqSet", params["frequency_hz"])
        if record.error:
            return SkillResult(
                skill_name="SetRTFreq", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetRTFreq", success=True,
            data={"rt_frequency_hz": params["frequency_hz"]},
            nanonis_calls=[record],
        )


class GetRTOversample(BaseSkill):
    """Get the Real-time oversampling value."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetRTOversample",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读 TCP Receiver 里的实时过采样值。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "rt", "oversampling", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_RTOversamplGet")
        if record.error:
            return SkillResult(
                skill_name="GetRTOversample", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # Util.RTOversamplGet ResponseTypes=["i"] -> parsed[2][0] is the RT
        # oversampling value. Reading parsed[0] gave the empty error string
        # and int("") crashed on real hardware.
        parsed = record.return_value
        oversampling = 0
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                oversampling = int(vals[0])
            elif isinstance(vals, (int, float)):
                oversampling = int(vals)
        elif isinstance(parsed, int):
            oversampling = parsed
        return SkillResult(
            skill_name="GetRTOversample", success=True,
            data={"rt_oversampling": oversampling},
            nanonis_calls=[record],
        )


class SetRTOversample(BaseSkill):
    """Set the Real-time oversampling value."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetRTOversample",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设定 TCP Receiver 里的实时过采样值。",
            parameters=[
                ParameterSpec(
                    name="oversampling",
                    type="int",
                    description="RT 过采样值",
                    required=True,
                    min_value=1,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "rt", "oversampling", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "Util_RTOversamplSet", params["oversampling"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetRTOversample", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetRTOversample", success=True,
            data={"rt_oversampling": params["oversampling"]},
            nanonis_calls=[record],
        )


def _parse_session_path(record) -> str:
    """The session folder out of a ``Util_SessionPathGet`` record.

    ``return_value`` is the Nanonis triplet ``(error_string, raw_bytes,
    parsed_list)``, and ``Util.SessionPathGet`` declares
    ``ResponseTypes=["i", "*-c"]`` — so ``parsed_list`` is
    ``[path_size_int, path_string]`` and the folder lives at ``parsed[2][1]``.
    An older scan over the whole triplet never looked inside ``parsed[2]`` (it
    saw the empty error string and the raw bytes) and returned an empty path
    even when a real one was there.

    ONE parser, shared by the getter and by ``SetSessionPath``'s readback: a
    verification whose reader is a second, separately-written parser can agree
    with itself while both halves are wrong.
    """
    parsed = getattr(record, "return_value", None)
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
        vals = parsed[2]
        if isinstance(vals, (list, tuple)):
            # Prefer the last string entry (the path follows its size int).
            for item in reversed(vals):
                if isinstance(item, str):
                    return item
        elif isinstance(vals, str):
            return vals
    return ""


def _same_session_path(a: str, b: str) -> bool:
    """Do these two name the same folder?

    Deliberately tolerant about the things Nanonis is free to normalise and
    that do NOT mean the write missed: separator direction, a trailing
    separator, and (on Windows, where this rig lives) case. Deliberately strict
    about everything else — the point is to catch a path that came back
    DIFFERENT, and a comparison generous enough to never fail is the echo it
    replaced wearing a different hat.
    """
    def norm(p: str) -> str:
        return str(p or "").strip().replace("\\", "/").rstrip("/").casefold()
    return norm(a) == norm(b)


class GetSessionPath(BaseSkill):
    """Get the current Nanonis session path."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSessionPath",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读当前 Nanonis 会话文件夹路径。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "session", "path", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_SessionPathGet")
        if record.error:
            return SkillResult(
                skill_name="GetSessionPath", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="GetSessionPath", success=True,
            data={"session_path": _parse_session_path(record)},
            nanonis_calls=[record],
        )


class SetSessionPath(BaseSkill):
    """Set the Nanonis session folder path."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSessionPath",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设定 Nanonis 会话文件夹路径。",
            parameters=[
                ParameterSpec(
                    name="session_path",
                    type="str",
                    description="会话文件夹路径",
                    required=True,
                ),
                ParameterSpec(
                    name="save_settings_to_previous",
                    type="bool",
                    description=(
                        "切换之前，把仪器**当前**的设置**写进旧**会话的设置文件里。默认 True，所以改会话文件夹并不是一次纯粹的导航操作 —— 它同时会把状态存进你正要离开的那个文件夹。"
                        "传 False 则只换位置、不碰旧会话。"
                    ),
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["util", "session", "path", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        session_path = params["session_path"]
        save_prev = 1 if params.get("save_settings_to_previous", True) else 0
        record = context.safe_call(
            "Util_SessionPathSet", session_path, save_prev,
        )
        if record.error:
            return SkillResult(
                skill_name="SetSessionPath", success=False,
                error=record.error, nanonis_calls=[record],
            )

        # Read back the session path instead of echoing the requested value. TCP
        # acceptance alone does not establish that a setting was applied, and this
        # path controls where subsequent scans are written.
        back = context.safe_call("Util_SessionPathGet")
        calls = [record, back]
        if back.error:
            # The write went through; the readback did not. Say so rather than
            # silently promoting "unknown" to "as requested".
            return SkillResult(
                skill_name="SetSessionPath", success=True,
                data={"session_path": None, "requested": session_path,
                      "verified": False, "verify_error": back.error,
                      "save_settings_to_previous": bool(save_prev)},
                summary=(f"已下发会话路径 {session_path}，但无法读回确认"
                         f"（{back.error}）——请勿据此认定它已生效。"),
                nanonis_calls=calls,
            )

        actual = _parse_session_path(back)
        if not _same_session_path(actual, session_path):
            return SkillResult(
                skill_name="SetSessionPath", success=False,
                error=(f"会话路径未生效：要求 {session_path!r}，"
                       f"读回 {actual!r}。后续扫描仍会存到读回的那个目录。"),
                data={"session_path": actual, "requested": session_path,
                      "verified": True,
                      "save_settings_to_previous": bool(save_prev)},
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="SetSessionPath", success=True,
            # The path Nanonis reports, not the one we asked for — they can
            # differ harmlessly (separator, trailing slash) and the instrument's
            # spelling is the one that will show up in every saved file.
            data={"session_path": actual, "requested": session_path,
                  "verified": True,
                  "save_settings_to_previous": bool(save_prev)},
            nanonis_calls=calls,
        )


class UnlockNanonisUI(BaseSkill):
    """Unlock the Nanonis user interface."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="UnlockNanonisUI",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="解锁 Nanonis 界面（关掉 Lock 模态窗口）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["util", "unlock", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Util_UnLock")
        if record.error:
            return SkillResult(
                skill_name="UnlockNanonisUI", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="UnlockNanonisUI", success=True,
            data={"locked": False},
            nanonis_calls=[record],
        )
