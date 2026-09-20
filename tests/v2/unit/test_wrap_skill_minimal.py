"""Phase 4 mini — verify wrap_skill end-to-end with vendored v1 skills.

This test exercises:
  - mast.skills.base.BaseSkill (vendored)
  - mast.core.types.SkillMetadata / ParameterSpec (vendored)
  - mast.skills.builtins.bias.{GetBias, SetBias, GetBiasCalibration} (vendored)
  - mast.agents._shared.skill_adapter.wrap_skill (v2 native)

If this passes, the same wrap_skill pattern can batch-port the remaining
124 builtin skills + 5 composite skills + 21 paper skills in Phase 4 proper.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_wrap_skill_minimal.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
# pytest's rootdir-based sys.path injection puts D:\...\MAST first, where
# the v1 mast/ shadows MASTv2/mast/. Force MASTv2/ ahead, and purge any
# already-cached v1 mast.* modules. Using importlib-mode at the file level
# would also work but requires changing pytest config.
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    # remove any prior occurrence + push to front
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
from mast.core.types import SafetyLevel, SkillCategory, NanonisCallRecord
from mast.skills.builtins.bias import (
    GetBias,
    GetBiasCalibration,
    GetCurrent,
    SetBias,
    SetBiasCalibration,
    SetBiasRange,
)


# ─────────────────────────────────────────────────────────────────────
# Test helpers — fake ExecutionContext that intercepts Nanonis calls
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    """Minimal ExecutionContext substitute. Only implements safe_call."""
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method,
                args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    """Build a context_provider callable that returns a fresh FakeCtx each call."""
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


# ─────────────────────────────────────────────────────────────────────
# Static introspection — wrap_skill must expose correct tool shape
# ─────────────────────────────────────────────────────────────────────

def test_wrap_get_bias_tool_shape():
    """GetBias is AUTO read-only with no params → empty Pydantic schema."""
    tool = wrap_skill(GetBias, make_provider())
    assert tool.name == "GetBias"
    assert "bias" in tool.description.lower()
    assert tool.metadata["danger_level"] == "AUTO"
    assert tool.metadata["skill_source"].endswith(".bias")
    # GetBias has no parameters
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_wrap_set_bias_tool_shape():
    """SetBias is CONFIRM write with bias_v required + slew_rate_v_per_s optional."""
    tool = wrap_skill(SetBias, make_provider())
    assert tool.name == "SetBias"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "bias_v" in fields
    assert "slew_rate_v_per_s" in fields
    assert fields["bias_v"].is_required()
    assert not fields["slew_rate_v_per_s"].is_required()
    # description should mention V/s unit annotation
    assert "V" in (fields["bias_v"].description or "")


def test_wrap_confirm_skill_marked():
    """SetBiasCalibration was DANGEROUS, now CONFIRM (v0.3.22 — confirm pane
    never wired, so DANGEROUS path stalled silently)."""
    tool = wrap_skill(SetBiasCalibration, make_provider())
    assert tool.metadata["danger_level"] == "CONFIRM"


def test_wrap_skill_int_param():
    """SetBiasRange uses int range_index — Pydantic schema must reflect that."""
    tool = wrap_skill(SetBiasRange, make_provider())
    fields = tool.args_schema.model_fields
    assert "range_index" in fields
    # int annotation
    assert fields["range_index"].annotation is int


# ─────────────────────────────────────────────────────────────────────
# End-to-end execution — invoke the tool and inspect Command.update
# ─────────────────────────────────────────────────────────────────────

def _invoke(tool, **kwargs) -> Any:
    """Call the tool's underlying _run with manually-supplied injected args.

    The InjectedToolCallId / InjectedState annotations are populated by LangGraph
    at runtime; we substitute test values to bypass.
    """
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


def test_get_bias_succeeds_with_canned_response():
    """GetBias should produce Command(update={executed_skills, messages, ...})."""
    canned = {
        "Bias_Get": {"return_value": ("", b"", [1.234])},
    }
    tool = wrap_skill(GetBias, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetBias"]
    msgs = update["messages"]
    assert len(msgs) == 1
    # The skill summary should mention bias_v ~ 1.234
    assert "1.234" in msgs[0].content or "bias_v" in msgs[0].content


def test_set_bias_with_param():
    """SetBias passes bias_v through ctx.safe_call('Bias_Set', target)."""
    canned = {"Bias_Set": {"return_value": ("", b"", [])}}
    tool = wrap_skill(SetBias, make_provider(canned))
    result = _invoke(tool, bias_v=0.5)
    update = result.update
    assert update["executed_skills"] == ["SetBias"]


def test_get_bias_propagates_nanonis_error():
    """If safe_call returns an error, the tool surfaces it (rolled_back path or error msg)."""
    canned = {"Bias_Get": {"error": "TCP connection refused"}}
    tool = wrap_skill(GetBias, make_provider(canned))
    result = _invoke(tool)
    msg_content = result.update["messages"][0].content
    # GetBias.execute returns SkillResult(success=False, error=...) — the wrap_skill
    # treats this as a successful tool return (not an exception). The summary or
    # stringified data dict should include the error string.
    assert ("TCP connection refused" in msg_content
            or "False" in msg_content
            or "error" in msg_content.lower())


def test_validate_params_blocks_missing_required():
    """SetBias requires bias_v — invoking with empty params should fail validation."""
    tool = wrap_skill(SetBias, make_provider())
    # Calling without bias_v should be flagged by validate_params
    result = _invoke(tool)  # no bias_v provided
    msg_content = result.update["messages"][0].content
    assert ("precondition_failed" in msg_content
            or "Missing required" in msg_content
            or "bias_v" in msg_content)


# ─────────────────────────────────────────────────────────────────────
# Precondition fail-CLOSED — a crash in check_preconditions must NOT let
# a hardware action through. Preconditions like
# z_controller_off (tip withdrawn before MotorMove) are safety-relevant.
# ─────────────────────────────────────────────────────────────────────

class _StateWithSnapshot:
    """Minimal ctx.state stub whose snapshot() returns a non-None state."""

    def snapshot(self):
        return object()  # any non-None state triggers the precondition branch


@dataclass
class _CtxWithState(FakeCtx):
    state: Any = None


def _provider_with_state():
    return lambda: _CtxWithState(state=_StateWithSnapshot())


def _make_skill(*, precond_raises: bool):
    """Build a tiny BaseSkill subclass; check_preconditions raises or returns []."""
    from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata, SkillResult
    from mast.skills.base import BaseSkill

    executed = {"ran": False}

    class _Probe(BaseSkill):
        def metadata(self):
            return SkillMetadata(
                name="ProbeSkill", version="1.0.0", category=SkillCategory.WRITE,
                safety_level=SafetyLevel.CONFIRM, description="probe",
                parameters=[],
            )

        def validate_params(self, params):
            return []

        def check_preconditions(self, state):
            if precond_raises:
                raise RuntimeError("precondition checker exploded")
            return []

        def execute(self, ctx, params):
            executed["ran"] = True
            return SkillResult(skill_name="ProbeSkill", success=True, summary="executed")

    return _Probe, executed


def test_precondition_exception_fails_closed():
    """check_preconditions raising must REFUSE execution (fail-closed), not run the skill."""
    skill_cls, executed = _make_skill(precond_raises=True)
    tool = wrap_skill(skill_cls, _provider_with_state())
    result = _invoke(tool)
    msg = result.update["messages"][0]
    assert "precondition_failed" in msg.content
    assert msg.status == "error"
    # The exception type/message should be surfaced, not swallowed.
    assert "RuntimeError" in msg.content
    # Crucially: the hardware action must NOT have executed.
    assert executed["ran"] is False
    # Failure path also logs to error_log.
    assert any("ProbeSkill" in e for e in result.update.get("error_log", []))


def test_precondition_ok_still_executes():
    """Normal path unchanged: no exception + preconditions met → skill executes."""
    skill_cls, executed = _make_skill(precond_raises=False)
    tool = wrap_skill(skill_cls, _provider_with_state())
    result = _invoke(tool)
    assert executed["ran"] is True
    assert "precondition_failed" not in result.update["messages"][0].content
    assert result.update["executed_skills"] == ["ProbeSkill"]


# ─────────────────────────────────────────────────────────────────────
# Sanity — calling sequence is recorded by the fake ctx
# ─────────────────────────────────────────────────────────────────────

def test_set_bias_calls_bias_set_method():
    canned = {"Bias_Set": {"return_value": ("", b"", [])}}
    provider = make_provider(canned)
    # Capture which ctx instance is used
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = provider()
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetBias, capturing_provider)
    _invoke(tool, bias_v=0.7)
    # The ctx records every safe_call
    last_ctx = instances[-1]
    assert any(call[0] == "Bias_Set" for call in last_ctx.calls)
    # The argument we passed should appear in the call
    bias_set_calls = [c for c in last_ctx.calls if c[0] == "Bias_Set"]
    assert bias_set_calls[0][1] == (0.7,)


# ─────────────────────────────────────────────────────────────────────
# Training-log recorder (RFC P1) — opt-in, fail-safe skill-call capture
# ─────────────────────────────────────────────────────────────────────

def _make_raising_skill():
    from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata
    from mast.skills.base import BaseSkill

    class _Boom(BaseSkill):
        def metadata(self):
            return SkillMetadata(name="BoomSkill", version="1.0.0",
                                 category=SkillCategory.WRITE,
                                 safety_level=SafetyLevel.CONFIRM,
                                 description="boom", parameters=[])

        def validate_params(self, params):
            return []

        def execute(self, ctx, params):
            raise RuntimeError("kaboom")

    return _Boom


def test_recorder_captures_successful_call():
    canned = {"Bias_Set": {"return_value": ("", b"", [])}}
    rec: list[dict] = []
    tool = wrap_skill(SetBias, make_provider(canned), recorder=rec.append)
    _invoke(tool, bias_v=0.5)
    assert len(rec) == 1
    p = rec[0]
    assert p["skill"] == "SetBias"
    assert p["params"] == {"bias_v": 0.5}
    assert p["tool_call_id"] == "test-call-1"
    assert p["danger_level"] == "CONFIRM"
    assert p["success"] is True
    assert p["rolled_back"] is False
    assert isinstance(p["duration_ms"], int) and p["duration_ms"] >= 0


def test_recorder_captures_failure_rollback():
    rec: list[dict] = []
    tool = wrap_skill(_make_raising_skill(), make_provider(), recorder=rec.append)
    _invoke(tool)
    assert len(rec) == 1
    p = rec[0]
    assert p["skill"] == "BoomSkill"
    assert p["success"] is False
    assert p["rolled_back"] is True
    assert "RuntimeError" in p["error"] and "kaboom" in p["error"]


def test_recorder_none_default_is_noop():
    # No recorder → behaves exactly as before (no crash, normal result).
    canned = {"Bias_Set": {"return_value": ("", b"", [])}}
    tool = wrap_skill(SetBias, make_provider(canned))
    result = _invoke(tool, bias_v=0.3)
    assert result.update["executed_skills"] == ["SetBias"]


def test_recorder_failure_never_breaks_skill():
    canned = {"Bias_Set": {"return_value": ("", b"", [])}}

    def boom_recorder(payload):
        raise ValueError("recorder itself broke")

    tool = wrap_skill(SetBias, make_provider(canned), recorder=boom_recorder)
    result = _invoke(tool, bias_v=0.5)  # must NOT raise
    assert result.update["executed_skills"] == ["SetBias"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
