"""v2 unit tests for mast.skills.builtins.util.

Skills covered: LockNanonisUI (CONFIRM), SaveSettings (CONFIRM),
  GetAcqPeriod (AUTO), LoadLayout (CONFIRM), SaveLayout (CONFIRM),
  GetRTFreq (AUTO), SetRTFreq (CONFIRM), GetRTOversample (AUTO),
  SetRTOversample (CONFIRM), GetSessionPath (AUTO), SetSessionPath (CONFIRM),
  UnlockNanonisUI (AUTO) — 12 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_util.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.util import (
    GetAcqPeriod,
    GetRTFreq,
    GetRTOversample,
    GetSessionPath,
    LoadLayout,
    LockNanonisUI,
    SaveLayout,
    SaveSettings,
    SetRTFreq,
    SetRTOversample,
    SetSessionPath,
    UnlockNanonisUI,
)

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_lock_nanonis_ui_shape():
    """2026-07-13: split into LockNanonisUI (DANGEROUS) + UnlockNanonisUI (AUTO).

    It used to be one skill with a `lock: bool`, at CONFIRM — and CONFIRM derives no
    HITL gate on the agent path, so an autonomous run could put a modal lock over the
    operator's Nanonis without anyone being asked. Locking the UI does not endanger
    the hardware; it removes the human's ability to intervene, which is the backstop
    every other safety mechanism in MAST falls back on. The two directions are now
    asymmetric — the same shape as Laser_OnOffSet (abort-safe only in its OFF form).
    """
    tool = wrap_skill(LockNanonisUI, make_provider())
    assert tool.name == "LockNanonisUI"
    assert tool.metadata["danger_level"] == "DANGEROUS", (
        "锁住 Nanonis 界面 = 锁住人的介入能力。CONFIRM 在 agent 路径上不触发 HITL——"
        "自治运行时可以在没人同意的情况下把用户锁在仪器外面。"
    )
    assert "lock" not in tool.args_schema.model_fields, (
        "lock 参数还在——方向必须拆开，不能靠一个布尔量决定要不要人批准"
    )


def test_save_settings_shape():
    tool = wrap_skill(SaveSettings, make_provider())
    assert tool.name == "SaveSettings"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "action" in fields
    assert fields["action"].is_required()
    assert "file_path" in fields
    assert not fields["file_path"].is_required()
    assert "use_session" in fields
    assert not fields["use_session"].is_required()


def test_get_acq_period_shape():
    tool = wrap_skill(GetAcqPeriod, make_provider())
    assert tool.name == "GetAcqPeriod"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_set_rt_freq_shape():
    tool = wrap_skill(SetRTFreq, make_provider())
    assert tool.name == "SetRTFreq"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "frequency_hz" in fields
    assert fields["frequency_hz"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["frequency_hz"].annotation is str
    # unit annotation should be present in description
    assert "Hz" in (fields["frequency_hz"].description or "")


def test_set_session_path_shape():
    tool = wrap_skill(SetSessionPath, make_provider())
    assert tool.name == "SetSessionPath"
    fields = tool.args_schema.model_fields
    assert "session_path" in fields
    assert fields["session_path"].is_required()
    assert "save_settings_to_previous" in fields
    assert not fields["save_settings_to_previous"].is_required()


def test_set_rt_oversample_int_param():
    tool = wrap_skill(SetRTOversample, make_provider())
    fields = tool.args_schema.model_fields
    assert "oversampling" in fields
    assert fields["oversampling"].annotation is int


def test_unlock_nanonis_ui_is_auto():
    tool = wrap_skill(UnlockNanonisUI, make_provider())
    assert tool.name == "UnlockNanonisUI"
    assert tool.metadata["danger_level"] == "AUTO"


def test_skill_source_points_to_util_module():
    tool = wrap_skill(GetAcqPeriod, make_provider())
    assert tool.metadata["skill_source"].endswith(".util")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_lock_nanonis_ui_executes_lock():
    canned = {"Util_Lock": {"return_value": ("", b"", [])}}
    tool = wrap_skill(LockNanonisUI, make_provider(canned))
    result = _invoke(tool, lock=True)
    update = result.update
    assert update["executed_skills"] == ["LockNanonisUI"]


def test_unlock_nanonis_ui_calls_unlock_and_needs_no_approval():
    """Unlocking gives the operator their instrument back. That direction is never
    the dangerous one, and gating it would be perverse — a run that left the UI
    locked is a real thing that happens."""
    canned = {"Util_UnLock": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(UnlockNanonisUI, capturing_provider)
    assert tool.metadata["danger_level"] == "AUTO"
    _invoke(tool)
    last_ctx = instances[-1]
    unlock_calls = [c for c in last_ctx.calls if c[0] == "Util_UnLock"]
    assert len(unlock_calls) == 1


def test_lock_nanonis_ui_calls_lock():
    canned = {"Util_Lock": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(LockNanonisUI, capturing_provider)
    _invoke(tool)
    last_ctx = instances[-1]
    assert [c for c in last_ctx.calls if c[0] == "Util_Lock"]
    # And it must never reach for the unlock verb — the directions are separate skills.
    assert not [c for c in last_ctx.calls if c[0] == "Util_UnLock"]


def test_set_rt_freq_calls_correct_method():
    canned = {"Util_RTFreqSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetRTFreq, capturing_provider)
    _invoke(tool, frequency_hz=50000.0)
    last_ctx = instances[-1]
    freq_calls = [c for c in last_ctx.calls if c[0] == "Util_RTFreqSet"]
    assert len(freq_calls) == 1
    assert freq_calls[0][1] == (50000.0,)


def test_save_settings_missing_action():
    tool = wrap_skill(SaveSettings, make_provider())
    result = _invoke(tool)  # missing required action param
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "action" in msg_content
    )


# ── Triplet-fixture regression tests (real Nanonis return shape) ───────────────
#
# These getters return the (error_string, raw_bytes, parsed_list) triplet:
#   Util.AcqPeriodGet     ResponseTypes=["f"]        -> period   at parsed[2][0]
#   Util.RTFreqGet        ResponseTypes=["f"]        -> freq     at parsed[2][0]
#   Util.RTOversamplGet   ResponseTypes=["i"]        -> oversamp at parsed[2][0]
#   Util.SessionPathGet   ResponseTypes=["i","*-c"]  -> path     at parsed[2][1]
# The old code read parsed[0] (the empty error string): float("")/int("")
# crashed for the numeric getters, and the path scan never inspected parsed[2]
# so it returned an empty session path on real hardware.

def test_get_acq_period_real_triplet_value():
    ctx = FakeCtx(canned={"Util_AcqPeriodGet": {"return_value": ("", b"", [0.002])}})
    res = GetAcqPeriod().execute(ctx, {})
    assert res.success
    assert res.data["acquisition_period_s"] == pytest.approx(0.002)


def test_get_rt_freq_real_triplet_value():
    ctx = FakeCtx(canned={"Util_RTFreqGet": {"return_value": ("", b"", [40000.0])}})
    res = GetRTFreq().execute(ctx, {})
    assert res.success
    assert res.data["rt_frequency_hz"] == pytest.approx(40000.0)


def test_get_rt_oversample_real_triplet_value():
    ctx = FakeCtx(canned={"Util_RTOversamplGet": {"return_value": ("", b"", [10])}})
    res = GetRTOversample().execute(ctx, {})
    assert res.success
    assert res.data["rt_oversampling"] == 10


def test_get_session_path_real_triplet_value():
    # parsed_list = [path_size_int, path_string] -> path at parsed[2][1].
    rv = ("", b"", [27, r"C:\Nanonis\Session\demo"])
    ctx = FakeCtx(canned={"Util_SessionPathGet": {"return_value": rv}})
    res = GetSessionPath().execute(ctx, {})
    assert res.success
    assert res.data["session_path"] == r"C:\Nanonis\Session\demo"


def test_get_session_path_empty_triplet_yields_empty_string():
    ctx = FakeCtx(canned={"Util_SessionPathGet": {"return_value": ("", b"", [0, ""])}})
    res = GetSessionPath().execute(ctx, {})
    assert res.success
    assert res.data["session_path"] == ""


# ── SetSessionPath: verify, don't echo ───────────────────────────────────────
#
# It used to return data={"session_path": <the value we asked for>} — the
# request re-labelled as the state. Same anti-pattern ZControllerOnOff was fixed
# for, and the same rebuttal: a write that returns without a TCP error has been
# ACCEPTED, not necessarily APPLIED.
#
# This one was checkable the whole time. Util_SessionPathGet exists and is
# already used in production (scan_extra.py resolves the real save directory
# with it) — the verification was available and simply was not done.
#
# It matters because the session path is WHERE EVERY SUBSEQUENT SCAN IS
# WRITTEN. A silently-refused change sends hours of data into the previous
# sample's folder while everything downstream believes otherwise.

_SET_OK = {"Util_SessionPathSet": {"return_value": ("", b"", [])}}


def _sess(path):
    return {"return_value": ("", b"", [len(path), path])}


def test_set_session_path_reads_the_value_back():
    ctx = FakeCtx(canned={**_SET_OK,
                          "Util_SessionPathGet": _sess(r"C:\N\demo")})
    res = SetSessionPath().execute(ctx, {"session_path": r"C:\N\demo"})
    assert res.success
    assert res.data["verified"] is True
    assert res.data["session_path"] == r"C:\N\demo"
    assert [c[0] for c in ctx.calls] == ["Util_SessionPathSet",
                                         "Util_SessionPathGet"]


def test_a_path_that_did_not_take_is_a_failure_not_a_success():
    """THE test. Under the old code this returned success with the requested
    path echoed back, and nothing anywhere could tell it had not landed."""
    ctx = FakeCtx(canned={**_SET_OK,
                          "Util_SessionPathGet": _sess(r"C:\N\OLD_SAMPLE")})
    res = SetSessionPath().execute(ctx, {"session_path": r"C:\N\new_sample"})
    assert res.success is False
    assert res.data["session_path"] == r"C:\N\OLD_SAMPLE"   # what IS
    assert res.data["requested"] == r"C:\N\new_sample"      # what was ASKED
    assert "OLD_SAMPLE" in res.error and "new_sample" in res.error


def test_reported_path_is_the_instruments_spelling_not_ours():
    """Nanonis' spelling is what shows up in every saved file."""
    ctx = FakeCtx(canned={**_SET_OK,
                          "Util_SessionPathGet": _sess(r"C:\N\Demo\\")})
    res = SetSessionPath().execute(ctx, {"session_path": "c:/n/demo"})
    assert res.success
    assert res.data["session_path"] == r"C:\N\Demo\\"


@pytest.mark.parametrize("asked,back", [
    (r"C:\N\demo",   r"C:\N\demo" + "\\"),   # trailing separator
    (r"C:\N\demo",   "C:/N/demo"),            # separator direction
    (r"C:\N\Demo",   r"c:\n\demo"),           # case (Windows)
])
def test_harmless_normalisations_are_not_false_alarms(asked, back):
    """A comparison that fires on a trailing backslash would be worse than the
    echo it replaced — the operator learns to ignore it."""
    ctx = FakeCtx(canned={**_SET_OK, "Util_SessionPathGet": _sess(back)})
    assert SetSessionPath().execute(ctx, {"session_path": asked}).success


@pytest.mark.parametrize("asked,back", [
    (r"C:\N\demo",   r"C:\N\demo2"),
    (r"C:\N\demo",   r"C:\N\dem"),
    (r"C:\N\demo",   ""),
    (r"C:\N\a\b",    r"C:\N\b\a"),
])
def test_the_comparison_is_still_strict_about_real_differences(asked, back):
    """Sensitivity: the tolerance above must not have made it blind."""
    assert SetSessionPath().execute(
        FakeCtx(canned={**_SET_OK, "Util_SessionPathGet": _sess(back)}),
        {"session_path": asked}).success is False


def test_unreadable_readback_says_unknown_rather_than_claiming_success():
    """The write went through; the readback did not. Do not promote "unknown"
    to "as requested" — that is the echo returning by the back door."""
    ctx = FakeCtx(canned={**_SET_OK,
                          "Util_SessionPathGet": {"error": "module down"}})
    res = SetSessionPath().execute(ctx, {"session_path": r"C:\N\demo"})
    assert res.success is True            # the write itself did not fail
    assert res.data["verified"] is False
    assert res.data["session_path"] is None
    assert res.data["requested"] == r"C:\N\demo"
    assert "无法读回确认" in res.summary


def test_a_failed_write_never_reaches_the_readback():
    ctx = FakeCtx(canned={"Util_SessionPathSet": {"error": "refused"}})
    res = SetSessionPath().execute(ctx, {"session_path": r"C:\N\demo"})
    assert res.success is False
    assert [c[0] for c in ctx.calls] == ["Util_SessionPathSet"]


def test_save_settings_flag_is_passed_through_and_described():
    """Default True means changing the folder ALSO writes current settings into
    the folder being left. That side effect was not stated in the parameter
    description; an operator reading only the name would not expect a write."""
    ctx = FakeCtx(canned={**_SET_OK,
                          "Util_SessionPathGet": _sess(r"C:\N\demo")})
    SetSessionPath().execute(ctx, {"session_path": r"C:\N\demo"})
    assert ctx.calls[0] == ("Util_SessionPathSet", (r"C:\N\demo", 1))

    ctx2 = FakeCtx(canned={**_SET_OK,
                           "Util_SessionPathGet": _sess(r"C:\N\demo")})
    SetSessionPath().execute(ctx2, {"session_path": r"C:\N\demo",
                                    "save_settings_to_previous": False})
    assert ctx2.calls[0] == ("Util_SessionPathSet", (r"C:\N\demo", 0))

    spec = next(p for p in SetSessionPath().metadata().parameters
                if p.name == "save_settings_to_previous")
    assert "当前" in (spec.description or "")


def test_both_skills_share_one_path_parser():
    """A verification whose reader is a second, separately-written parser can
    agree with itself while both halves are wrong."""
    import inspect

    import mast.skills.builtins.util as util

    for cls in (util.GetSessionPath, util.SetSessionPath):
        assert "_parse_session_path" in source_of(cls.execute)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
