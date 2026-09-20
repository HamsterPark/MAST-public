"""v2 unit tests for the experimental current-capture support skills.

Skills covered (all AUTO):
  - ListSignalChannels (signals.py)      — enumerate signal channels + flag current
  - GetOsciTimebases (acquire_osci_trace) — list Osci1T sample-rate timebases
  - SetOsciTimebase (acquire_osci_trace)  — select a timebase by index

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_exp_capture_skills.py -x -v
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
from mast.skills.builtins.acquire_osci_trace import (
    GetOsciTimebases,
    SetOsciTimebase,
)
from mast.skills.builtins.signals import ListSignalChannels


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


# ── ListSignalChannels ─────────────────────────────────────────────────────────

def _names_ok(names: list[str]):
    # Signals.NamesGet Variables = [size, n, names_list]
    return ("", b"", [0, len(names), list(names)])


def test_list_channels_shape():
    tool = wrap_skill(ListSignalChannels, make_provider())
    assert tool.name == "ListSignalChannels"
    assert tool.metadata["danger_level"] == "AUTO"


def test_list_channels_flags_current():
    names = ["Bias (V)", "Current (A)", "Z (m)", "Current 2 (A)", "LI Demod 1 X (A)"]
    canned = {"Signals_NamesGet": {"return_value": _names_ok(names)}}
    skill = ListSignalChannels()
    res = skill.execute(FakeCtx(canned=canned), {})
    assert res.success
    assert res.data["n_channels"] == 5
    # "Current (A)" (idx 1) and "Current 2 (A)" (idx 3) start with 'current'.
    assert res.data["current_indices"] == [1, 3]
    assert res.data["channels"][1] == {"index": 1, "name": "Current (A)"}


def test_list_channels_error_propagated():
    canned = {"Signals_NamesGet": {"error": "TCP down"}}
    skill = ListSignalChannels()
    res = skill.execute(FakeCtx(canned=canned), {})
    assert not res.success
    assert "TCP down" in res.error


def test_list_channels_unparseable():
    canned = {"Signals_NamesGet": {"return_value": ("", b"", [0])}}
    skill = ListSignalChannels()
    res = skill.execute(FakeCtx(canned=canned), {})
    assert not res.success


# ── GetOsciTimebases ───────────────────────────────────────────────────────────

def _timebases_ok(current_index: int, dts: list[float]):
    # Osci1T.TimebaseGet Variables = [current_index, n, [dt...]]
    return ("", b"", [current_index, len(dts), list(dts)])


def test_get_timebases_shape():
    tool = wrap_skill(GetOsciTimebases, make_provider())
    assert tool.name == "GetOsciTimebases"
    assert tool.metadata["danger_level"] == "AUTO"


def test_get_timebases_computes_fs():
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_TimebaseGet": {"return_value": _timebases_ok(1, [5e-5, 1e-4])},
    }
    skill = GetOsciTimebases()
    res = skill.execute(FakeCtx(canned=canned), {})
    assert res.success
    assert res.data["n_timebases"] == 2
    assert res.data["current_index"] == 1
    tb0 = res.data["timebases"][0]
    assert tb0["dt_s"] == 5e-5
    assert abs(tb0["fs_hz"] - 20000.0) < 1e-6   # 1/5e-5 = 20 kHz
    assert abs(res.data["timebases"][1]["fs_hz"] - 10000.0) < 1e-6


def test_get_timebases_unwraps_tuple_array():
    """decodeArray can yield single-element tuples like (5e-5,)."""
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_TimebaseGet": {"return_value": ("", b"", [0, 2, [(5e-5,), (1e-4,)]])},
    }
    skill = GetOsciTimebases()
    res = skill.execute(FakeCtx(canned=canned), {})
    assert res.success
    assert abs(res.data["timebases"][0]["fs_hz"] - 20000.0) < 1e-6


def test_get_timebases_module_not_loaded():
    canned = {"Osci1T_Run": {"error": "NeedModule: Osci1T"}}
    skill = GetOsciTimebases()
    res = skill.execute(FakeCtx(canned=canned), {})
    assert not res.success
    assert "CaptureSignalBuffer" in res.error  # fallback hint


# ── SetOsciTimebase ────────────────────────────────────────────────────────────

def test_set_timebase_shape():
    tool = wrap_skill(SetOsciTimebase, make_provider())
    assert tool.name == "SetOsciTimebase"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert fields["timebase_index"].is_required()


def test_set_timebase_calls_set_with_index():
    canned = {
        "Osci1T_Run": {"return_value": ("", b"", [0])},
        "Osci1T_TimebaseSet": {"return_value": ("", b"", [0])},
    }
    ctx = FakeCtx(canned=canned)
    res = SetOsciTimebase().execute(ctx, {"timebase_index": 2})
    assert res.success
    assert res.data["timebase_index"] == 2
    set_calls = [c for c in ctx.calls if c[0] == "Osci1T_TimebaseSet"]
    assert set_calls and set_calls[0][1] == (2,)


def test_set_timebase_module_not_loaded():
    canned = {"Osci1T_Run": {"error": "NeedModule: Osci1T"}}
    res = SetOsciTimebase().execute(FakeCtx(canned=canned), {"timebase_index": 0})
    assert not res.success


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
