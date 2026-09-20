"""v2 unit tests for mast.skills.builtins.capture_signal_buffer.

Skill covered: CaptureSignalBuffer (AUTO) — high-rate TCP polling of one
signal for N seconds, returns a (timestamps, values) trace.

Nanonis methods exercised: Current_Get / ZCtrl_ZPosGet / Bias_Get (fast
paths) and Signals_ValGet (generic signal-index path).

Tests use tiny duration_s windows (0.05 s) so the internal poll loop
terminates fast — no slow tests.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_capture_signal_buffer.py -x -v
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
from mast.skills.builtins.capture_signal_buffer import CaptureSignalBuffer


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


# A signal-value payload: _decoded() unwraps the (err, b"", [value]) tuple.
def _val(x: float):
    return ("", b"", [x])


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_capture_signal_buffer_shape():
    tool = wrap_skill(CaptureSignalBuffer, make_provider())
    assert tool.name == "CaptureSignalBuffer"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert all(not v.is_required() for v in fields.values())
    for name in ("channel", "duration_s", "poll_hz", "include_samples"):
        assert name in fields
    assert fields["channel"].annotation is str
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["duration_s"].annotation is str
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["poll_hz"].annotation is str
    assert fields["include_samples"].annotation is bool


def test_capture_signal_buffer_skill_source():
    tool = wrap_skill(CaptureSignalBuffer, make_provider())
    assert tool.metadata["skill_source"].endswith(".capture_signal_buffer")


# ── Execution tests (fast-path channels) ──────────────────────────────────────

def test_capture_signal_buffer_executes_current():
    canned = {"Current_Get": {"return_value": _val(1.2e-9)}}
    tool = wrap_skill(CaptureSignalBuffer, make_provider(canned))
    result = _invoke(tool, channel="current", duration_s=0.05, poll_hz=500.0)
    update = result.update
    assert update["executed_skills"] == ["CaptureSignalBuffer"]
    assert "error_log" not in update


def test_capture_signal_buffer_current_uses_current_get():
    canned = {"Current_Get": {"return_value": _val(1.2e-9)}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(CaptureSignalBuffer, capturing_provider)
    _invoke(tool, channel="current", duration_s=0.05, poll_hz=500.0)
    methods = {c[0] for c in instances[-1].calls}
    assert methods == {"Current_Get"}
    # at least one sample collected in a 50 ms window
    assert len(instances[-1].calls) >= 1


def test_capture_signal_buffer_z_channel():
    canned = {"ZCtrl_ZPosGet": {"return_value": _val(3.3e-9)}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(CaptureSignalBuffer, capturing_provider)
    result = _invoke(tool, channel="z", duration_s=0.05, poll_hz=400.0)
    assert result.update["executed_skills"] == ["CaptureSignalBuffer"]
    assert {c[0] for c in instances[-1].calls} == {"ZCtrl_ZPosGet"}


def test_capture_signal_buffer_bias_channel():
    canned = {"Bias_Get": {"return_value": _val(0.5)}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(CaptureSignalBuffer, capturing_provider)
    result = _invoke(tool, channel="bias", duration_s=0.05, poll_hz=400.0)
    assert result.update["executed_skills"] == ["CaptureSignalBuffer"]
    assert {c[0] for c in instances[-1].calls} == {"Bias_Get"}


def test_capture_signal_buffer_signal_index_path():
    """Numeric channel string → Signals_ValGet with (idx, 0) args."""
    canned = {"Signals_ValGet": {"return_value": _val(7.0)}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(CaptureSignalBuffer, capturing_provider)
    result = _invoke(tool, channel="24", duration_s=0.05, poll_hz=50.0)
    assert result.update["executed_skills"] == ["CaptureSignalBuffer"]
    valget = [c for c in instances[-1].calls if c[0] == "Signals_ValGet"]
    assert len(valget) >= 1
    # wait_for_newest=0 → args (idx, 0)
    assert valget[0][1] == (24, 0)


# ── Boundary / error paths ────────────────────────────────────────────────────

def test_capture_signal_buffer_unknown_channel():
    """Non-numeric, non-fast-path channel → graceful failure before polling."""
    tool = wrap_skill(CaptureSignalBuffer, make_provider())
    result = _invoke(tool, channel="frobnicate", duration_s=0.05)
    update = result.update
    assert update["executed_skills"] == ["CaptureSignalBuffer"]
    assert "error_log" in update
    assert "Unknown channel" in str(result)


def test_capture_signal_buffer_all_tcp_errors():
    """Every poll errors → 'No samples collected' failure result."""
    canned = {"Current_Get": {"return_value": None, "error": "TCP timeout"}}
    tool = wrap_skill(CaptureSignalBuffer, make_provider(canned))
    result = _invoke(tool, channel="current", duration_s=0.05, poll_hz=500.0)
    update = result.update
    assert update["executed_skills"] == ["CaptureSignalBuffer"]
    assert "error_log" in update
    assert "No samples collected" in str(result)


def test_capture_signal_buffer_abort_stops_loop():
    """A context.check_abort() returning True ends the loop immediately."""

    @dataclass
    class AbortingCtx(FakeCtx):
        def check_abort(self) -> bool:
            return True

    canned = {"Current_Get": {"return_value": _val(1.0e-9)}}
    tool = wrap_skill(CaptureSignalBuffer, lambda: AbortingCtx(canned=canned))
    result = _invoke(tool, channel="current", duration_s=10.0, poll_hz=500.0)
    update = result.update
    # immediate abort → zero samples → failure, but test returns fast
    assert update["executed_skills"] == ["CaptureSignalBuffer"]
    assert "error_log" in update


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
