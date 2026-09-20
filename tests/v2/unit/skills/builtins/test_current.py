"""v2 unit tests for mast.skills.builtins.current.

Skills covered: SetCurrentGain (CONFIRM), GetCurrentBEEM (AUTO),
                SetCurrentCalibration (CONFIRM) — 3 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_current.py -x -v
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
from mast.skills.builtins.current import (
    GetCurrentBEEM,
    SetCurrentCalibration,
    SetCurrentGain,
)


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

def test_set_current_gain_shape():
    tool = wrap_skill(SetCurrentGain, make_provider())
    assert tool.name == "SetCurrentGain"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "gain_index" in fields
    assert fields["gain_index"].is_required()
    assert "filter_index" in fields
    assert not fields["filter_index"].is_required()
    assert fields["gain_index"].annotation is int


def test_get_current_beem_shape():
    tool = wrap_skill(GetCurrentBEEM, make_provider())
    assert tool.name == "GetCurrentBEEM"
    assert tool.metadata["danger_level"] == "AUTO"
    # No parameters
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_set_current_calibration_shape():
    tool = wrap_skill(SetCurrentCalibration, make_provider())
    assert tool.name == "SetCurrentCalibration"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "calibration" in fields
    assert "offset" in fields
    assert fields["calibration"].is_required()
    assert fields["offset"].is_required()
    assert "gain_index" in fields
    assert not fields["gain_index"].is_required()
    assert fields["calibration"].annotation is float


def test_skill_source_points_to_current_module():
    tool = wrap_skill(SetCurrentGain, make_provider())
    assert tool.metadata["skill_source"].endswith(".current")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_set_current_gain_executes():
    canned = {"Current_GainSet": {"return_value": ("", b"", [])}}
    tool = wrap_skill(SetCurrentGain, make_provider(canned))
    result = _invoke(tool, gain_index=3)
    update = result.update
    assert update["executed_skills"] == ["SetCurrentGain"]


def test_set_current_gain_calls_correct_method():
    canned = {"Current_GainSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetCurrentGain, capturing_provider)
    _invoke(tool, gain_index=2, filter_index=1)
    last_ctx = instances[-1]
    gain_calls = [c for c in last_ctx.calls if c[0] == "Current_GainSet"]
    assert len(gain_calls) == 1
    assert gain_calls[0][1] == (2, 1)


def test_get_current_beem_succeeds():
    canned = {"Current_BEEMGet": {"return_value": [5e-10]}}
    tool = wrap_skill(GetCurrentBEEM, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetCurrentBEEM"]
    msg_content = update["messages"][0].content
    assert "beem_current_a" in msg_content or "5e-10" in msg_content or "5" in msg_content


def test_set_current_calibration_missing_required():
    tool = wrap_skill(SetCurrentCalibration, make_provider())
    result = _invoke(tool)  # missing calibration + offset
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "calibration" in msg_content
    )


# ── Triplet-fixture regression tests (real Nanonis return shape) ───────────────
#
# Current.BEEMGet ResponseTypes=["f"] -> return_value is the
# (error_string, raw_bytes, parsed_list) triplet with the BEEM current (A) at
# parsed[2][0]. The old code did float(parsed[0]) == float("") which raised
# ValueError on real hardware, so the read ALWAYS crashed.

def test_get_current_beem_real_triplet_value():
    ctx = FakeCtx(canned={"Current_BEEMGet": {"return_value": ("", b"", [5e-10])}})
    res = GetCurrentBEEM().execute(ctx, {})
    assert res.success
    assert res.data["beem_current_a"] == pytest.approx(5e-10)


def test_get_current_beem_real_triplet_does_not_crash_on_empty_header():
    # The crucial regression: a real triplet whose [0] is the empty error
    # string must NOT be float()'d. Value comes from parsed[2][0].
    ctx = FakeCtx(canned={"Current_BEEMGet": {"return_value": ("", b"\x00", [1.25e-9])}})
    res = GetCurrentBEEM().execute(ctx, {})
    assert res.success
    assert res.data["beem_current_a"] == pytest.approx(1.25e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
