"""v2 unit tests for mast.skills.builtins.bias_pulse.

Skills covered: BiasPulse (AUTO) — 1 skill total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_bias_pulse.py -x -v
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
from mast.skills.builtins.bias_pulse import BiasPulse


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

def test_bias_pulse_shape():
    tool = wrap_skill(BiasPulse, make_provider())
    assert tool.name == "BiasPulse"
    # AUTO: a bias pulse only changes the tip apex (no instrument damage); the
    # Nanonis bias range bounds it, so autonomous agents may run it ungated.
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "width_s" in fields
    assert "bias_v" in fields
    assert fields["width_s"].is_required()
    assert fields["bias_v"].is_required()
    assert "z_hold" in fields
    assert not fields["z_hold"].is_required()
    assert "absolute" in fields
    assert not fields["absolute"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["width_s"].annotation is str
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["bias_v"].annotation is str


def test_skill_source_points_to_bias_pulse_module():
    tool = wrap_skill(BiasPulse, make_provider())
    assert tool.metadata["skill_source"].endswith(".bias_pulse")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_bias_pulse_executes():
    canned = {"Bias_Pulse": {"return_value": ("", b"", [])}}
    tool = wrap_skill(BiasPulse, make_provider(canned))
    result = _invoke(tool, width_s=1e-3, bias_v=2.0)
    update = result.update
    assert update["executed_skills"] == ["BiasPulse"]


def test_bias_pulse_calls_correct_method_absolute():
    canned = {"Bias_Pulse": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(BiasPulse, capturing_provider)
    # absolute=True → abs_rel=2
    _invoke(tool, width_s=1e-3, bias_v=2.0, z_hold=1, absolute=True)
    last_ctx = instances[-1]
    pulse_calls = [c for c in last_ctx.calls if c[0] == "Bias_Pulse"]
    assert len(pulse_calls) == 1
    assert pulse_calls[0][1] == (1, 1e-3, 2.0, 1, 2)  # wait=1, width, bias, z_hold, abs_rel=2


def test_bias_pulse_missing_required():
    tool = wrap_skill(BiasPulse, make_provider())
    result = _invoke(tool)  # missing width_s + bias_v
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "width_s" in msg_content
    )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
