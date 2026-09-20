"""``allowed_values`` must reach the model as an enum, not as a bare string.

Same shape of gap as the numeric bounds before them: ``ParameterSpec`` had
carried ``allowed_values`` on a dozen skills, ``validate_params`` enforced it,
and the model saw ``{"type": "string"}`` — the legal set existed only in prose it
could skip. A constraint the model can only learn by reading is a constraint it
can ignore.

Literal is what survives ``convert_to_openai_tool``; ``json_schema_extra`` is
dropped there (measured, and the reason the numeric bounds needed ge/le).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/agents/test_allowed_values_in_schema.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
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
from langchain_core.utils.function_calling import convert_to_openai_tool

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.atom_track import AtomTrackQuickCompStart
from mast.skills.builtins.motor import MotorMove


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is not None:
            return NanonisCallRecord(method=method, args=args,
                                     return_value=entry.get("return_value"),
                                     error=entry.get("error", ""))
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def _provider(canned=None):
    canned = canned or {}
    instances: list[FakeCtx] = []

    def provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    return provider, instances


def _invoke(tool, **kwargs):
    return tool.func(tool_call_id="t1", state={}, **kwargs)


def test_string_enum_reaches_the_provider_payload():
    tool = wrap_skill(MotorMove, _provider()[0])
    props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert set(props["direction"]["enum"]) == {
        "x+", "x-", "y+", "y-", "z-approach", "z-retract",
    }


def test_int_enum_reaches_the_provider_payload():
    tool = wrap_skill(AtomTrackQuickCompStart, _provider()[0])
    props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert props["compensation_type"]["enum"] == [0, 1]


def test_an_illegal_choice_is_rejected_and_the_legal_set_is_shown():
    """Rejection has to name the alternatives, or the retry is a guess."""
    provider, instances = _provider({"Motor_StartMove": {"return_value": None}})
    tool = wrap_skill(MotorMove, provider)
    result = _invoke(tool, direction="up", steps=3)
    assert not any(c[0] == "Motor_StartMove" for c in instances[-1].calls)
    msg = result.update["messages"][0].content
    assert "z-approach" in msg or "direction" in msg


def test_a_legal_choice_still_executes():
    provider, instances = _provider({"Motor_StartMove": {"return_value": None}})
    tool = wrap_skill(MotorMove, provider)
    result = _invoke(tool, direction="x+", steps=3)
    assert result.update["executed_skills"] == ["MotorMove"]


def test_parameters_without_a_declared_set_are_untouched():
    """Only enums change shape; everything else keeps its numeric bounds."""
    from mast.skills.builtins.zctrl_gain import SetZCtrlGain

    tool = wrap_skill(SetZCtrlGain, _provider()[0])
    props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert "enum" not in props["p_gain"]
    # dimensioned → string + SI range in the description, not `maximum`
    assert "1u" in props["p_gain"]["description"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
