"""v2 unit tests for mast.skills.builtins.acquire_osci_trace.

Skill covered: AcquireOsciTrace (AUTO) — pulls one buffered trace from the
Nanonis Osci1T (1-channel oscilloscope) module.

Nanonis methods exercised: Osci1T_Run, Osci1T_ChSet, Osci1T_DataGet.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_acquire_osci_trace.py -x -v
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
from mast.skills.builtins.acquire_osci_trace import AcquireOsciTrace


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


# A valid Osci1T_DataGet payload: [t0, dt, n, y_array].
# The skill's _decoded() unwraps the (err, b"", [...]) convention.
def _osci_ok(t0=0.0, dt=5e-5, n=8):
    y = [float(i) * 1e-9 for i in range(n)]
    return ("", b"", [t0, dt, n, y])


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_acquire_osci_trace_shape():
    tool = wrap_skill(AcquireOsciTrace, make_provider())
    assert tool.name == "AcquireOsciTrace"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    # both params are optional
    assert all(not v.is_required() for v in fields.values())
    assert "data_to_get" in fields
    assert "signal_index" in fields
    assert fields["data_to_get"].annotation is int
    assert fields["signal_index"].annotation is int


def test_acquire_osci_trace_skill_source():
    tool = wrap_skill(AcquireOsciTrace, make_provider())
    assert tool.metadata["skill_source"].endswith(".acquire_osci_trace")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_acquire_osci_trace_executes():
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_DataGet": {"return_value": _osci_ok(n=8)},
    }
    tool = wrap_skill(AcquireOsciTrace, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["AcquireOsciTrace"]
    # success path → no error_log entry
    assert "error_log" not in update


def test_acquire_osci_trace_issues_run_and_dataget():
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_DataGet": {"return_value": _osci_ok(n=8)},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(AcquireOsciTrace, capturing_provider)
    _invoke(tool)  # defaults: data_to_get=0, signal_index=-1
    methods = [c[0] for c in instances[-1].calls]
    assert "Osci1T_Run" in methods
    assert "Osci1T_DataGet" in methods
    # signal_index=-1 default → no Osci1T_ChSet
    assert "Osci1T_ChSet" not in methods
    # data_to_get default forwarded as positional arg
    dataget = [c for c in instances[-1].calls if c[0] == "Osci1T_DataGet"][0]
    assert dataget[1] == (0,)


def test_acquire_osci_trace_assigns_signal_channel():
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_ChSet": {"return_value": ("", b"", [0])},
        "Osci1T_DataGet": {"return_value": _osci_ok(n=4)},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(AcquireOsciTrace, capturing_provider)
    _invoke(tool, signal_index=3, data_to_get=2)
    calls = instances[-1].calls
    chset = [c for c in calls if c[0] == "Osci1T_ChSet"]
    assert len(chset) == 1
    assert chset[0][1] == (3,)
    dataget = [c for c in calls if c[0] == "Osci1T_DataGet"][0]
    assert dataget[1] == (2,)


# ── Boundary / error paths ────────────────────────────────────────────────────

def test_acquire_osci_trace_module_not_loaded():
    """Osci1T_Run returns a NeedModule error → graceful fallback hint."""
    canned = {
        "Osci1T_Run": {"return_value": None, "error": "NeedModule: Osci1T"},
    }
    tool = wrap_skill(AcquireOsciTrace, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["AcquireOsciTrace"]
    # failure path → error logged, summary mentions the fallback skill
    assert "error_log" in update
    assert "CaptureSignalBuffer" in str(result)


def test_acquire_osci_trace_dataget_error():
    """Osci1T_DataGet errors after a successful Run → failure result."""
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_DataGet": {"return_value": None, "error": "TCP timeout"},
    }
    tool = wrap_skill(AcquireOsciTrace, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["AcquireOsciTrace"]
    assert "error_log" in update
    assert "Osci1T_DataGet failed" in str(result)


def test_acquire_osci_trace_bad_shape():
    """DataGet returns too few fields → 'Unexpected response shape' failure."""
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_DataGet": {"return_value": ("", b"", [0.0, 5e-5])},  # only 2 fields
    }
    tool = wrap_skill(AcquireOsciTrace, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["AcquireOsciTrace"]
    assert "error_log" in update


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
