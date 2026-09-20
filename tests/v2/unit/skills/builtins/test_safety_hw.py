"""v2 unit tests for mast.skills.builtins.safety_hw.

Skills covered: EnableSafeTip (AUTO), GetSafeTipStatus (AUTO),
               GetSafeTipProps (AUTO), GetSafeTipSignal (AUTO) — 4 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_safety_hw.py -x -v
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
from mast.skills.builtins.safety_hw import (
    EnableSafeTip,
    GetSafeTipProps,
    GetSafeTipSignal,
    GetSafeTipStatus,
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

def test_enable_safe_tip_shape():
    tool = wrap_skill(EnableSafeTip, make_provider())
    assert tool.name == "EnableSafeTip"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "enable" in fields
    # Default ON (safety fix): `enable` is now optional and defaults to True so a
    # bare/defaulted call ENABLES SafeTip; only an explicit enable=False disables.
    assert not fields["enable"].is_required()
    assert fields["enable"].default is True
    assert fields["enable"].annotation is bool


def test_get_safe_tip_status_shape():
    tool = wrap_skill(GetSafeTipStatus, make_provider())
    assert tool.name == "GetSafeTipStatus"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_get_safe_tip_props_shape():
    tool = wrap_skill(GetSafeTipProps, make_provider())
    assert tool.name == "GetSafeTipProps"
    assert tool.metadata["danger_level"] == "AUTO"


def test_get_safe_tip_signal_shape():
    tool = wrap_skill(GetSafeTipSignal, make_provider())
    assert tool.name == "GetSafeTipSignal"
    assert tool.metadata["danger_level"] == "AUTO"


def test_skill_source_points_to_safety_hw_module():
    tool = wrap_skill(EnableSafeTip, make_provider())
    assert tool.metadata["skill_source"].endswith(".safety_hw")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_enable_safe_tip_executes():
    canned = {"SafeTip_OnOffSet": {"return_value": ("", b"", [])}}
    tool = wrap_skill(EnableSafeTip, make_provider(canned))
    result = _invoke(tool, enable=True)
    update = result.update
    assert update["executed_skills"] == ["EnableSafeTip"]


def test_enable_safe_tip_calls_correct_method():
    canned = {"SafeTip_OnOffSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(EnableSafeTip, capturing_provider)
    _invoke(tool, enable=True)
    last_ctx = instances[-1]
    enable_calls = [c for c in last_ctx.calls if c[0] == "SafeTip_OnOffSet"]
    assert len(enable_calls) == 1
    # enable=True → int(True) = 1
    assert enable_calls[0][1] == (1,)


def test_get_safe_tip_status_executes():
    canned = {"SafeTip_OnOffGet": {"return_value": ("", b"", [1])}}
    tool = wrap_skill(GetSafeTipStatus, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetSafeTipStatus"]


def test_enable_safe_tip_bare_call_defaults_on():
    # Default ON (safety fix): a bare/defaulted call (no `enable` arg) ENABLES
    # SafeTip — it must NOT fail as "missing required" and must call OnOffSet(1).
    canned = {"SafeTip_OnOffSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(EnableSafeTip, capturing_provider)
    result = _invoke(tool)  # no enable arg
    assert result.update["executed_skills"] == ["EnableSafeTip"]
    assert result.update["messages"][0].status == "success"
    enable_calls = [c for c in instances[-1].calls if c[0] == "SafeTip_OnOffSet"]
    assert len(enable_calls) == 1
    assert enable_calls[0][1] == (1,)  # defaulted to enable=True → int(True)=1


def test_enable_safe_tip_explicit_disable_still_works():
    # The disable path is intentionally NOT gated: enable=False must still pass
    # through and call OnOffSet(0).
    canned = {"SafeTip_OnOffSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(EnableSafeTip, capturing_provider)
    result = _invoke(tool, enable=False)
    assert result.update["messages"][0].status == "success"
    enable_calls = [c for c in instances[-1].calls if c[0] == "SafeTip_OnOffSet"]
    assert enable_calls[0][1] == (0,)  # int(False) = 0 → disable


# ── Triplet-read regression tests (real nanonis_spm return shape) ──────────────
#
# nanonis_spm quickSend/parseGeneralResponse returns
#     [error_string, raw_bytes, Variables]
# where Variables is the parsed data list ordered by the method's ResponseTypes.
#   SafeTip.PropsGet  ResponseTypes ["H", "H", "f"] -> Variables = [auto_recovery,
#                                                                    auto_pause_scan,
#                                                                    threshold]
#   SafeTip.SignalGet ResponseTypes ["f"]           -> Variables = [signal_value]
# The real data lives at parsed[2][i]; parsed[0] is the (empty) error string and
# parsed[1] is the raw response bytes. Reading parsed[0]/parsed[1] is the bug.


def _ctx(canned):
    return FakeCtx(canned=canned)


def test_get_safe_tip_props_reads_variables_not_error_string():
    # error string parsed[0]="" (empty), raw bytes parsed[1]=b"...",
    # real values live in parsed[2].
    canned = {
        "SafeTip_PropsGet": {"return_value": ("", b"\x00\x01\x00\x00\x3f", [1, 0, 0.25])}
    }
    skill = GetSafeTipProps()
    result = skill.execute(_ctx(canned), {})
    assert result.success is True
    # auto_recovery=1 -> True, auto_pause_scan=0 -> False, threshold=0.25
    assert result.data["auto_recovery"] is True
    assert result.data["auto_pause_scan"] is False
    assert result.data["threshold"] == pytest.approx(0.25)


def test_get_safe_tip_props_distinguishes_recovery_and_pause():
    # Swapped flags: recovery off, pause on — proves index ordering inside parsed[2].
    canned = {
        "SafeTip_PropsGet": {"return_value": ("", b"", [0, 1, 1.5])}
    }
    skill = GetSafeTipProps()
    result = skill.execute(_ctx(canned), {})
    assert result.data["auto_recovery"] is False
    assert result.data["auto_pause_scan"] is True
    assert result.data["threshold"] == pytest.approx(1.5)


def test_get_safe_tip_props_not_fooled_by_nonempty_error_position():
    # Defensive: even if parsed[0] were a truthy string and parsed[1] truthy bytes,
    # the skill must read parsed[2], never coerce parsed[0]/parsed[1].
    canned = {
        "SafeTip_PropsGet": {"return_value": ("", b"\xff\xff", [1, 1, 2.0])}
    }
    skill = GetSafeTipProps()
    result = skill.execute(_ctx(canned), {})
    assert result.data["auto_recovery"] is True
    assert result.data["auto_pause_scan"] is True
    assert result.data["threshold"] == pytest.approx(2.0)


def test_get_safe_tip_props_propagates_error():
    canned = {"SafeTip_PropsGet": {"return_value": None, "error": "boom"}}
    skill = GetSafeTipProps()
    result = skill.execute(_ctx(canned), {})
    assert result.success is False
    assert result.error == "boom"


def test_get_safe_tip_signal_reads_float_from_variables():
    # Single float32 at parsed[2][0]. The old bug did float(parsed[0]) on the
    # empty error string -> ValueError at runtime against real hardware.
    canned = {"SafeTip_SignalGet": {"return_value": ("", b"\x3d\xcc\xcc\xcd", [0.1])}}
    skill = GetSafeTipSignal()
    result = skill.execute(_ctx(canned), {})
    assert result.success is True
    assert result.data["signal_value"] == pytest.approx(0.1)


def test_get_safe_tip_signal_negative_value():
    canned = {"SafeTip_SignalGet": {"return_value": ("", b"", [-3.5])}}
    skill = GetSafeTipSignal()
    result = skill.execute(_ctx(canned), {})
    assert result.success is True
    assert result.data["signal_value"] == pytest.approx(-3.5)


def test_get_safe_tip_signal_empty_error_string_does_not_crash():
    # Regression for float(parsed[0]) == float("") ValueError: an empty error
    # string at parsed[0] must never be parsed as the signal value.
    canned = {"SafeTip_SignalGet": {"return_value": ("", b"", [42.0])}}
    skill = GetSafeTipSignal()
    result = skill.execute(_ctx(canned), {})  # must not raise ValueError
    assert result.success is True
    assert result.data["signal_value"] == pytest.approx(42.0)


def test_get_safe_tip_signal_propagates_error():
    canned = {"SafeTip_SignalGet": {"return_value": None, "error": "no signal"}}
    skill = GetSafeTipSignal()
    result = skill.execute(_ctx(canned), {})
    assert result.success is False
    assert result.error == "no signal"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
