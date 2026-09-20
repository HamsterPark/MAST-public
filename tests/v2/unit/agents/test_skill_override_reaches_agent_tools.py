"""Admin per-skill overrides must reach the AGENT tool path, not just the menu.

The gap this pins shut
======================
``SkillRegistry._get_metadata`` applies ``skill_overrides.json``, so the skill
menu, the GUI and ``ExecutionContext.run``'s approval decision have always
honoured admin overrides. ``wrap_skill`` did not: ``instrument_control/tools.py``
takes only the CLASS out of ``registry.list_skills()``, and the adapter then asked
the fresh instance for its metadata again — the declared one.

Nothing on the model-facing side saw the override: not the JSON schema, not the
``skill_metadata`` handed to SafetyGate, not ``validate_params``. An admin got a
stored file and a 200 response for a bound that was enforced nowhere the model
could reach. That is how a ``p_gain`` ceiling written by an admin override
would have behaved after the restart it was waiting for: still absent.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/agents/test_skill_override_reaches_agent_tools.py -x -v
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

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from mast.admin.override_store import ConfigOverrideRegistry
from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.zctrl_gain import SetZCtrlGain

P_OK, T_OK, I_OK = 3e-12, 1.6667e-05, 1.8e-07


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is not None:
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"), error=entry.get("error", ""),
            )
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


@pytest.fixture
def overrides(tmp_path):
    """A registry pointed at a temp dir, torn down so no test leaks its file."""
    ConfigOverrideRegistry.reset()

    def write(payload: dict):
        (tmp_path / "skill_overrides.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        ConfigOverrideRegistry.reset()
        ConfigOverrideRegistry.get(tmp_path)

    ConfigOverrideRegistry.get(tmp_path)
    yield write
    ConfigOverrideRegistry.reset()


def test_tightened_bound_reaches_the_model_facing_schema(overrides):
    overrides({"SetZCtrlGain": {"parameters": {"p_gain": {"max_value": 1e-7}}}})
    tool = wrap_skill(SetZCtrlGain, _provider()[0])
    props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    # Dimensioned params are strings now, so the advertised bound is the SI range
    # in the description — 1e-7 renders as "100n".
    assert "100n" in props["p_gain"]["description"]


def test_tightened_bound_reaches_the_metadata_safetygate_reads(overrides):
    """SafetyGate checks the RAW args against ``tool.metadata['skill_metadata']``."""
    overrides({"SetZCtrlGain": {"parameters": {"p_gain": {"max_value": 1e-7}}}})
    tool = wrap_skill(SetZCtrlGain, _provider()[0])
    meta = tool.metadata["skill_metadata"]
    spec = {p.name: p for p in meta.parameters}["p_gain"]
    assert spec.max_value == 1e-7


def test_tightened_bound_is_actually_enforced(overrides):
    overrides({"SetZCtrlGain": {"parameters": {"p_gain": {"max_value": 1e-7}}}})
    canned = {"ZCtrl_GainSet": {"return_value": ("", b"", [])}}
    provider, instances = _provider(canned)
    tool = wrap_skill(SetZCtrlGain, provider)
    _invoke(tool, p_gain="500n", time_constant_s=T_OK, i_gain=I_OK)
    assert not any(c[0] == "ZCtrl_GainSet" for c in instances[-1].calls)


def test_widened_bound_does_not_leave_the_tool_contradicting_itself(overrides):
    """The half-fix would advertise the wider bound and still refuse the value.

    ``validate_params`` reads ``self.metadata()`` on the skill INSTANCE. Applying
    the override only to the schema lets pydantic accept a value that
    validate_params then rejects — a tool whose declared range is a lie. Only a
    WIDENING override exposes this; tightening hides it, because pydantic refuses
    first and the two layers never get to disagree.
    """
    overrides({"SetZCtrlGain": {"parameters": {"p_gain": {"max_value": 2e-6}}}})
    canned = {
        "ZCtrl_GainSet": {"return_value": ("", b"", [])},
        "ZCtrl_GainGet": {"return_value": ("", b"", [1.5e-6, T_OK, I_OK])},
    }
    provider, instances = _provider(canned)
    tool = wrap_skill(SetZCtrlGain, provider)
    _invoke(tool, p_gain="1.5u", time_constant_s=T_OK, i_gain=I_OK)
    sets = [c for c in instances[-1].calls if c[0] == "ZCtrl_GainSet"]
    assert sets, "widened bound was advertised but the write was still refused"


def test_safety_level_override_reaches_the_agent_hitl_gate(overrides):
    """Raising a skill to DANGEROUS must gate the agent path, not only the menu."""
    overrides({"SetZCtrlGain": {"safety_level": "dangerous"}})
    tool = wrap_skill(SetZCtrlGain, _provider()[0])
    assert tool.metadata["danger_level"] == "DANGEROUS"


def test_no_override_file_leaves_the_declared_metadata_alone(overrides):
    tool = wrap_skill(SetZCtrlGain, _provider()[0])
    props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert "1u" in props["p_gain"]["description"]  # the declared 1e-6 ceiling


def test_a_corrupt_override_file_does_not_stop_the_agent_building(tmp_path):
    """Fail-open: a broken file must not make an agent unbuildable."""
    ConfigOverrideRegistry.reset()
    (tmp_path / "skill_overrides.json").write_text("{not json", encoding="utf-8")
    ConfigOverrideRegistry.get(tmp_path)
    try:
        tool = wrap_skill(SetZCtrlGain, _provider()[0])
        props = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
        assert "1u" in props["p_gain"]["description"]
    finally:
        ConfigOverrideRegistry.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
