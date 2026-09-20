"""v2 unit tests for mast.skills.builtins.signals.

Skills covered: GetSignalsAddRT (AUTO), GetSignalRange (AUTO),
                GetSignalValues (AUTO) — 3 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_signals.py -x -v
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
from mast.skills.builtins.signals import (
    GetSignalRange,
    GetSignalValues,
    GetSignalsAddRT,
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

def test_get_signals_add_rt_shape():
    tool = wrap_skill(GetSignalsAddRT, make_provider())
    assert tool.name == "GetSignalsAddRT"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_get_signal_range_shape():
    tool = wrap_skill(GetSignalRange, make_provider())
    assert tool.name == "GetSignalRange"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "signal_index" in fields
    assert fields["signal_index"].is_required()
    assert fields["signal_index"].annotation is int


def test_get_signal_values_shape():
    tool = wrap_skill(GetSignalValues, make_provider())
    assert tool.name == "GetSignalValues"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "signal_indexes" in fields
    assert fields["signal_indexes"].is_required()
    assert "wait_for_newest" in fields
    assert not fields["wait_for_newest"].is_required()


def test_skill_source_points_to_signals_module():
    tool = wrap_skill(GetSignalRange, make_provider())
    assert tool.metadata["skill_source"].endswith(".signals")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_get_signal_range_succeeds():
    canned = {
        "Signals_RangeGet": {
            "return_value": ("", b"", [[10.0, -10.0]])
        }
    }
    tool = wrap_skill(GetSignalRange, make_provider(canned))
    result = _invoke(tool, signal_index=5)
    update = result.update
    assert update["executed_skills"] == ["GetSignalRange"]


def test_get_signal_range_calls_correct_method():
    canned = {"Signals_RangeGet": {"return_value": ("", b"", [[5.0, -5.0]])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(GetSignalRange, capturing_provider)
    _invoke(tool, signal_index=14)
    last_ctx = instances[-1]
    range_calls = [c for c in last_ctx.calls if c[0] == "Signals_RangeGet"]
    assert len(range_calls) == 1
    assert range_calls[0][1] == (14,)


def test_get_signal_values_executes():
    canned = {"Signals_ValsGet": {"return_value": ("", b"", [[1.0, 2.0, 3.0]])}}
    tool = wrap_skill(GetSignalValues, make_provider(canned))
    result = _invoke(tool, signal_indexes="0,1,2")
    update = result.update
    assert update["executed_skills"] == ["GetSignalValues"]


def test_get_signals_add_rt_error_propagated():
    canned = {"Signals_AddRTGet": {"error": "channel not available"}}
    tool = wrap_skill(GetSignalsAddRT, make_provider(canned))
    result = _invoke(tool)
    msg_content = result.update["messages"][0].content
    assert (
        "channel not available" in msg_content
        or "False" in msg_content
        or "error" in msg_content.lower()
    )


# ── Realistic-triplet parse regressions ─────────────────────────────────────
# return_value is always [error_string, raw_bytes, Variables] where Variables
# is positional per the method's ResponseTypes. Earlier code read the WRONG
# Variables indices for Signals_AddRTGet and Signals_ValsGet.

def test_get_signal_range_parses_real_triplet():
    """Signals_RangeGet ResponseTypes ["f","f"] → Variables=[max, min]."""
    canned = {"Signals_RangeGet": {"return_value": ("", b"\x00", [10.5, -10.5])}}
    skill = GetSignalRange()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {"signal_index": 7})
    assert res.success
    assert res.data["signal_index"] == 7
    assert res.data["max_limit"] == 10.5
    assert res.data["min_limit"] == -10.5


def test_get_signals_add_rt_parses_real_triplet():
    """Signals_AddRTGet ResponseTypes ["i","i","*+c","i","*-c","i","*-c"]:
    Variables = [names_size, num_signals, names_list, sz1, rt1, sz2, rt2].

    Regression: the available names live at index 2 and the two assigned
    signal names at indices 4 and 6 — NOT 0/1/2 as the old code assumed.
    """
    canned = {
        "Signals_AddRTGet": {
            "return_value": (
                "", b"\x00",
                [40, 3, ["Sig A", "Sig B", "Sig C"], 5, "Sig A", 5, "Sig B"],
            )
        }
    }
    skill = GetSignalsAddRT()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["available_rt_signals"] == ["Sig A", "Sig B", "Sig C"]
    assert res.data["num_rt_signals"] == 3
    assert res.data["internal_23_signal"] == "Sig A"
    assert res.data["internal_24_signal"] == "Sig B"
    # The integer size fields must NOT leak in as "signal names".
    assert "40" not in res.data["internal_23_signal"]
    assert "3" not in res.data["internal_23_signal"]


def test_get_signal_values_parses_real_triplet():
    """Signals_ValsGet ResponseTypes ["i","*f"]:
    Variables = [values_size, values_array]. The readings are at index 1.

    decodeArray yields single-element tuples like (1.0,), which must be
    unwrapped to floats.
    """
    canned = {
        "Signals_ValsGet": {
            "return_value": (
                "", b"\x00",
                [3, [(1.5e-9,), (-2.0e-9,), (3.25e-9,)]],
            )
        }
    }
    skill = GetSignalValues()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {"signal_indexes": "0,1,2"})
    assert res.success
    assert res.data["values"] == [1.5e-9, -2.0e-9, 3.25e-9]
    # The leading size (3) must NOT be mistaken for a reading.
    assert 3.0 not in res.data["values"]


def test_get_signal_values_plain_float_array():
    """Some parser builds return a plain float list (no tuple wrapping)."""
    canned = {
        "Signals_ValsGet": {
            "return_value": ("", b"\x00", [2, [1.0, 2.0]]),
        }
    }
    skill = GetSignalValues()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {"signal_indexes": "5,6"})
    assert res.success
    assert res.data["values"] == [1.0, 2.0]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
