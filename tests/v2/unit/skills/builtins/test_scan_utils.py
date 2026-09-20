"""v2 unit tests for mast.skills.builtins.scan_utils.

Skills covered: GetScanFrame (AUTO), WaitScanComplete (AUTO) — 2 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_scan_utils.py -x -v
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
from mast.skills.builtins.scan_utils import GetScanFrame, WaitScanComplete


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

def test_get_scan_frame_shape():
    tool = wrap_skill(GetScanFrame, make_provider())
    assert tool.name == "GetScanFrame"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_wait_scan_complete_shape():
    tool = wrap_skill(WaitScanComplete, make_provider())
    assert tool.name == "WaitScanComplete"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "timeout_ms" in fields
    assert not fields["timeout_ms"].is_required()
    assert fields["timeout_ms"].annotation is int


def test_skill_source_points_to_scan_utils_module():
    tool = wrap_skill(GetScanFrame, make_provider())
    assert tool.metadata["skill_source"].endswith(".scan_utils")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_get_scan_frame_executes():
    canned = {
        "Scan_FrameGet": {
            "return_value": ("", b"", [0.0, 0.0, 100e-9, 100e-9, 0.0])
        }
    }
    tool = wrap_skill(GetScanFrame, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetScanFrame"]


def test_wait_scan_complete_executes():
    """v0.3.14: polling implementation. Status 0 = finished → success."""
    canned = {
        # Scan_StatusGet returns (err, raw, [status]); 0 = not scanning
        "Scan_StatusGet": {"return_value": ("", b"", [0])},
    }
    tool = wrap_skill(WaitScanComplete, make_provider(canned))
    result = _invoke(tool, timeout_ms=5000)
    update = result.update
    assert update["executed_skills"] == ["WaitScanComplete"]


def test_wait_scan_complete_uses_polling_not_blocking_call():
    """v0.3.14: must call Scan_StatusGet, NOT Scan_WaitEndOfScan."""
    canned = {"Scan_StatusGet": {"return_value": ("", b"", [0])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(WaitScanComplete, capturing_provider)
    _invoke(tool, timeout_ms=2000)
    last_ctx = instances[-1]
    methods = [c[0] for c in last_ctx.calls]
    assert "Scan_StatusGet" in methods, f"expected polling, got {methods}"
    assert "Scan_WaitEndOfScan" not in methods, (
        "Scan_WaitEndOfScan blocks the TCP socket for the full timeout — "
        "polling Scan_StatusGet is the v0.3.14 fix"
    )


# ── Realistic-triplet parse test (return_value = [err, raw, Variables]) ──────

def test_get_scan_frame_parses_real_triplet():
    """Scan_FrameGet ResponseTypes ["f","f","f","f","f"] →
    [center_x, center_y, width, height, angle] — confirm index mapping."""
    canned = {
        "Scan_FrameGet": {
            "return_value": ("", b"\x00", [1e-7, -2e-7, 3e-7, 4e-7, 30.0])
        }
    }
    skill = GetScanFrame()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["center_x_m"] == 1e-7
    assert res.data["center_y_m"] == -2e-7
    assert res.data["width_m"] == 3e-7
    assert res.data["height_m"] == 4e-7
    assert res.data["angle_deg"] == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
