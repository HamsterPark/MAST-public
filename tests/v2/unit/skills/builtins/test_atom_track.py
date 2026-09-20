"""v2 unit tests for mast.skills.builtins.atom_track.

Skills covered: ConfigureAtomTrack (CONFIRM), AtomTrackDriftComp (CONFIRM),
               AtomTrackQuickCompStart (CONFIRM), AtomTrackStatusGet (AUTO)
               — 4 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_atom_track.py -x -v
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
from typing import Any, get_args

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.atom_track import (
    AtomTrackDriftComp,
    AtomTrackQuickCompStart,
    AtomTrackStatusGet,
    ConfigureAtomTrack,
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

def test_configure_atom_track_shape():
    tool = wrap_skill(ConfigureAtomTrack, make_provider())
    assert tool.name == "ConfigureAtomTrack"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "integral_gain" in fields
    assert "frequency_hz" in fields
    assert "amplitude_m" in fields
    assert fields["integral_gain"].is_required()
    assert fields["frequency_hz"].is_required()
    assert fields["amplitude_m"].is_required()
    assert "phase_deg" in fields
    assert not fields["phase_deg"].is_required()
    assert fields["integral_gain"].annotation is float


def test_atom_track_drift_comp_shape():
    tool = wrap_skill(AtomTrackDriftComp, make_provider())
    assert tool.name == "AtomTrackDriftComp"
    assert tool.metadata["danger_level"] == "CONFIRM"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_atom_track_quick_comp_start_shape():
    tool = wrap_skill(AtomTrackQuickCompStart, make_provider())
    assert tool.name == "AtomTrackQuickCompStart"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "compensation_type" in fields
    assert fields["compensation_type"].is_required()
    # allowed_values reaches the model as a Literal enum, not a bare int — an
    # explicit set is what the model needs, and it is strictly stronger than the
    # int range it sits inside.
    assert get_args(fields["compensation_type"].annotation) == (0, 1)


def test_atom_track_status_get_shape():
    tool = wrap_skill(AtomTrackStatusGet, make_provider())
    assert tool.name == "AtomTrackStatusGet"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "control" in fields
    assert fields["control"].is_required()
    assert get_args(fields["control"].annotation) == (0, 1, 2)


def test_skill_source_points_to_atom_track_module():
    tool = wrap_skill(ConfigureAtomTrack, make_provider())
    assert tool.metadata["skill_source"].endswith(".atom_track")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_configure_atom_track_executes():
    canned = {
        "AtomTrack_PropsSet": {"return_value": ("", b"", [])},
        "AtomTrack_CtrlSet": {"return_value": ("", b"", [])},
    }
    tool = wrap_skill(ConfigureAtomTrack, make_provider(canned))
    result = _invoke(tool, integral_gain=1.0, frequency_hz=100.0, amplitude_m=1e-10)
    update = result.update
    assert update["executed_skills"] == ["ConfigureAtomTrack"]


def test_configure_atom_track_calls_ctrl_set_for_modulation_and_controller():
    canned = {
        "AtomTrack_PropsSet": {"return_value": ("", b"", [])},
        "AtomTrack_CtrlSet": {"return_value": ("", b"", [])},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(ConfigureAtomTrack, capturing_provider)
    _invoke(
        tool,
        integral_gain=1.0, frequency_hz=100.0, amplitude_m=1e-10,
        enable_modulation=True, enable_controller=True,
    )
    last_ctx = instances[-1]
    ctrl_calls = [c for c in last_ctx.calls if c[0] == "AtomTrack_CtrlSet"]
    # Should have 2: modulation (0) and controller (1)
    assert len(ctrl_calls) == 2
    ctrl_types = {c[1][0] for c in ctrl_calls}
    assert 0 in ctrl_types  # modulation
    assert 1 in ctrl_types  # controller


def test_atom_track_drift_comp_executes():
    canned = {"AtomTrack_DriftComp": {"return_value": ("", b"", [])}}
    tool = wrap_skill(AtomTrackDriftComp, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["AtomTrackDriftComp"]


def test_atom_track_quick_comp_start_executes():
    canned = {"AtomTrack_QuickCompStart": {"return_value": ("", b"", [])}}
    tool = wrap_skill(AtomTrackQuickCompStart, make_provider(canned))
    result = _invoke(tool, compensation_type=1)
    update = result.update
    assert update["executed_skills"] == ["AtomTrackQuickCompStart"]


def test_atom_track_status_get_executes():
    canned = {"AtomTrack_StatusGet": {"return_value": [1]}}
    tool = wrap_skill(AtomTrackStatusGet, make_provider(canned))
    result = _invoke(tool, control=0)
    update = result.update
    assert update["executed_skills"] == ["AtomTrackStatusGet"]


def test_configure_atom_track_missing_required():
    tool = wrap_skill(ConfigureAtomTrack, make_provider())
    result = _invoke(tool)  # missing integral_gain, frequency_hz, amplitude_m
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "integral_gain" in msg_content
    )


# ── Triplet-fixture regression tests (real Nanonis return shape) ───────────────
#
# AtomTrack.StatusGet ResponseTypes=["H"] -> return_value is the
# (error_string, raw_bytes, parsed_list) triplet with Status at parsed[2][0].
# The old code read parsed[0] (the empty error string), so bool("") == False
# made it ALWAYS report "Off" for every control on real hardware.

def test_atom_track_status_get_on_real_triplet():
    ctx = FakeCtx(canned={"AtomTrack_StatusGet": {"return_value": ("", b"\x00\x01", [1])}})
    res = AtomTrackStatusGet().execute(ctx, {"control": 1})
    assert res.success
    assert res.data["status"] is True
    assert res.data["control"] == "Controller"


def test_atom_track_status_get_off_real_triplet():
    ctx = FakeCtx(canned={"AtomTrack_StatusGet": {"return_value": ("", b"\x00\x00", [0])}})
    res = AtomTrackStatusGet().execute(ctx, {"control": 0})
    assert res.success
    assert res.data["status"] is False
    assert res.data["control"] == "Modulation"


# ── ConfigureAtomTrack: enable failures must NOT be silently swallowed ─────────

def test_configure_atom_track_propagates_modulation_enable_failure():
    canned = {
        "AtomTrack_PropsSet": {"return_value": ("", b"", [])},
        # CtrlSet for modulation (0,1) reports a hardware error.
    }
    # Add an erroring CtrlSet entry; FakeCtx returns it for every CtrlSet call.
    canned["AtomTrack_CtrlSet"] = {"return_value": None, "error": "module not open"}
    ctx = FakeCtx(canned=canned)
    res = ConfigureAtomTrack().execute(
        ctx,
        {"integral_gain": 1.0, "frequency_hz": 100.0, "amplitude_m": 1e-10,
         "enable_modulation": True, "enable_controller": True},
    )
    assert res.success is False
    assert "modulation" in (res.error or "")
    # Must have stopped after the first failing CtrlSet (no controller call).
    ctrl_calls = [c for c in ctx.calls if c[0] == "AtomTrack_CtrlSet"]
    assert len(ctrl_calls) == 1


def test_configure_atom_track_propagates_controller_enable_failure():
    canned = {
        "AtomTrack_PropsSet": {"return_value": ("", b"", [])},
    }

    # Modulation succeeds, controller fails. FakeCtx can't branch by args, so
    # build a ctx whose CtrlSet errors only for AT_control == 1 (controller).
    @dataclass
    class _BranchCtx(FakeCtx):
        def safe_call(self, method, *args, role="main"):
            if method == "AtomTrack_CtrlSet" and args and args[0] == 1:
                self.calls.append((method, args))
                return NanonisCallRecord(method=method, args=args,
                                         error="controller refused")
            return super().safe_call(method, *args, role=role)

    ctx = _BranchCtx(canned={
        "AtomTrack_PropsSet": {"return_value": ("", b"", [])},
        "AtomTrack_CtrlSet": {"return_value": ("", b"", [])},
    })
    res = ConfigureAtomTrack().execute(
        ctx,
        {"integral_gain": 1.0, "frequency_hz": 100.0, "amplitude_m": 1e-10,
         "enable_modulation": True, "enable_controller": True},
    )
    assert res.success is False
    assert "controller" in (res.error or "")


def test_configure_atom_track_success_reports_enabled_flags():
    canned = {
        "AtomTrack_PropsSet": {"return_value": ("", b"", [])},
        "AtomTrack_CtrlSet": {"return_value": ("", b"", [])},
    }
    ctx = FakeCtx(canned=canned)
    res = ConfigureAtomTrack().execute(
        ctx,
        {"integral_gain": 1.0, "frequency_hz": 100.0, "amplitude_m": 1e-10,
         "enable_modulation": True, "enable_controller": False},
    )
    assert res.success is True
    assert res.data["modulation_enabled"] is True
    assert res.data["controller_enabled"] is False
    ctrl_calls = [c for c in ctx.calls if c[0] == "AtomTrack_CtrlSet"]
    assert len(ctrl_calls) == 1  # only modulation enabled


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
