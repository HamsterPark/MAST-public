"""v2 unit tests for mast.skills.builtins.pattern.

Skills covered (7):
  RunGridExperiment (CONFIRM), OpenPatternExperiment (CONFIRM),
  PausePatternExperiment (CONFIRM), SetPatternLine (CONFIRM),
  SetPatternCloud (CONFIRM), GetPatternCloud (AUTO), GetPatternProps (AUTO).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_pattern.py -x -v
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
from mast.skills.builtins.pattern import (
    GetPatternCloud,
    GetPatternProps,
    OpenPatternExperiment,
    PausePatternExperiment,
    RunGridExperiment,
    SetPatternCloud,
    SetPatternLine,
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

@pytest.mark.parametrize("skill_cls,expected_name,expected_danger", [
    (RunGridExperiment, "RunGridExperiment", "CONFIRM"),
    (OpenPatternExperiment, "OpenPatternExperiment", "CONFIRM"),
    (PausePatternExperiment, "PausePatternExperiment", "CONFIRM"),
    (SetPatternLine, "SetPatternLine", "CONFIRM"),
    (SetPatternCloud, "SetPatternCloud", "CONFIRM"),
    (GetPatternCloud, "GetPatternCloud", "AUTO"),
    (GetPatternProps, "GetPatternProps", "AUTO"),
])
def test_pattern_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".pattern")


def test_run_grid_experiment_required_fields():
    tool = wrap_skill(RunGridExperiment, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["nx"].is_required()
    assert fields["ny"].is_required()
    assert fields["nx"].annotation is int
    assert fields["ny"].annotation is int
    assert not fields["center_x_m"].is_required()
    assert not fields["wait_timeout_s"].is_required()


def test_set_pattern_line_required_fields():
    tool = wrap_skill(SetPatternLine, make_provider())
    fields = tool.args_schema.model_fields
    for req in ("num_points", "p1_x_m", "p1_y_m", "p2_x_m", "p2_y_m"):
        assert req in fields
        assert fields[req].is_required()


def test_set_pattern_cloud_required_fields():
    tool = wrap_skill(SetPatternCloud, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["x_coords"].is_required()
    assert fields["y_coords"].is_required()


# ── Execution tests ───────────────────────────────────────────────────────────

def test_open_pattern_experiment_executes():
    canned = {"Pattern_ExpOpen": {"return_value": None}}
    tool = wrap_skill(OpenPatternExperiment, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["OpenPatternExperiment"]


def test_pause_pattern_experiment_executes():
    canned = {"Pattern_ExpPause": {"return_value": None}}
    tool = wrap_skill(PausePatternExperiment, make_provider(canned))
    result = _invoke(tool, pause=True)
    assert result.update["executed_skills"] == ["PausePatternExperiment"]


def test_pause_maps_to_1():
    """pause=True → Pattern_ExpPause called with 1."""
    canned = {"Pattern_ExpPause": {"return_value": None}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(PausePatternExperiment, capturing_provider)
    _invoke(tool, pause=True)
    ctx = instances[-1]
    calls = [c for c in ctx.calls if c[0] == "Pattern_ExpPause"]
    assert calls[0][1][0] == 1


def test_set_pattern_line_executes():
    canned = {"Pattern_LineSet": {"return_value": None}}
    tool = wrap_skill(SetPatternLine, make_provider(canned))
    result = _invoke(tool, num_points=10, p1_x_m=0.0, p1_y_m=0.0, p2_x_m=50e-9, p2_y_m=50e-9)
    assert result.update["executed_skills"] == ["SetPatternLine"]


def test_set_pattern_cloud_executes():
    canned = {"Pattern_CloudSet": {"return_value": None}}
    tool = wrap_skill(SetPatternCloud, make_provider(canned))
    result = _invoke(tool, x_coords="[0.0, 1e-9]", y_coords="[0.0, 1e-9]")
    assert result.update["executed_skills"] == ["SetPatternCloud"]


def test_set_pattern_cloud_mismatched_coords():
    """Mismatched x/y lengths returns error without calling Nanonis."""
    tool = wrap_skill(SetPatternCloud, make_provider())
    result = _invoke(tool, x_coords="[0.0, 1e-9]", y_coords="[0.0]")
    msg = result.update["messages"][0]
    assert msg.status == "error"
    assert "equal length" in msg.content


def test_set_pattern_cloud_invalid_json():
    tool = wrap_skill(SetPatternCloud, make_provider())
    result = _invoke(tool, x_coords="not-json", y_coords="[0.0]")
    msg = result.update["messages"][0]
    assert msg.status == "error"


def test_get_pattern_cloud_executes():
    canned = {
        "Pattern_CloudGet": {
            "return_value": ("", b"", [2, [0.0, 1e-9], [0.0, 2e-9]])
        }
    }
    tool = wrap_skill(GetPatternCloud, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetPatternCloud"]


def test_get_pattern_props_executes():
    canned = {"Pattern_PropsGet": {"return_value": ("", b"", [10, 2, [], 5, "STS", 0, "", 0.1, 1])}}
    tool = wrap_skill(GetPatternProps, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetPatternProps"]


def test_run_grid_experiment_gridset_fails():
    """If Pattern_GridSet errors, RunGridExperiment returns error immediately."""
    canned = {"Pattern_GridSet": {"error": "GridSet failed"}}
    tool = wrap_skill(RunGridExperiment, make_provider(canned))
    result = _invoke(tool, nx=3, ny=3)
    msg = result.update["messages"][0]
    assert msg.status == "error"


# ── Pattern_GridSet argument-passing regression ─────────────────────────────
# Pattern_GridSet signature:
#   (Set_active_pattern, nx, ny, Grid_Scan_frame, Cx, Cy, W, H, Angle)
# Grid_Scan_frame=1 makes Nanonis size the grid to the scan frame and ignore
# the explicit Cx/Cy/W/H/Angle. The fix forces it to 0 so the user's grid
# geometry is actually honoured.

def _run_phase_setup(nx, ny, **kw):
    """Drive RunGridExperiment._phase_setup_grid directly with a FakeCtx and a
    minimal executor stub, returning the recorded Pattern_GridSet call args."""
    skill = RunGridExperiment()

    class _Exec:
        def set_partial(self, *a, **k):
            pass

    skill._executor = _Exec()
    skill._call_log = []
    ctx = FakeCtx(canned={"Pattern_GridSet": {"return_value": ("", b"", [])}})
    params = {"nx": nx, "ny": ny, **kw}
    res = skill._phase_setup_grid(params, ctx)
    grid_calls = [c for c in ctx.calls if c[0] == "Pattern_GridSet"]
    return res, grid_calls


def test_gridset_grid_scan_frame_is_zero():
    """Grid_Scan_frame (4th arg) must be 0 so explicit geometry is used."""
    res, grid_calls = _run_phase_setup(3, 4)
    assert res.success
    assert len(grid_calls) == 1
    args = grid_calls[0][1]
    # (set_active=1, nx, ny, grid_scan_frame, cx, cy, w, h, angle)
    assert args[0] == 1
    assert args[1] == 3
    assert args[2] == 4
    assert args[3] == 0, "Grid_Scan_frame must be 0, else user geometry is ignored"


def test_gridset_user_geometry_passed_through():
    """User-supplied center/width/height/angle reach Pattern_GridSet verbatim."""
    res, grid_calls = _run_phase_setup(
        5, 5,
        center_x_m=1e-7, center_y_m=-2e-7,
        width_m=3e-7, height_m=4e-7, angle_deg=15.0,
    )
    assert res.success
    args = grid_calls[0][1]
    assert args[3] == 0
    assert args[4] == 1e-7    # center_x_m
    assert args[5] == -2e-7   # center_y_m
    assert args[6] == 3e-7    # width_m
    assert args[7] == 4e-7    # height_m
    assert args[8] == 15.0    # angle_deg


# ── Realistic-triplet parse tests (return_value = [err, raw, Variables]) ─────

def test_get_pattern_cloud_parses_real_triplet():
    """Pattern_CloudGet ResponseTypes ["i","**f","**f"] →
    Variables = [num_points, x_array, y_array]."""
    canned = {
        "Pattern_CloudGet": {
            "return_value": ("", b"\x00\x00", [3, [0.0, 1e-9, 2e-9], [0.0, -1e-9, -2e-9]])
        }
    }
    skill = GetPatternCloud()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["num_points"] == 3
    assert res.data["x_coords"] == [0.0, 1e-9, 2e-9]
    assert res.data["y_coords"] == [0.0, -1e-9, -2e-9]


def test_get_pattern_props_parses_real_triplet():
    """Pattern_PropsGet ResponseTypes
    ["i","i","*+c","i","*-c","i","*-c","f","I"] → 9 positional Variables."""
    canned = {
        "Pattern_PropsGet": {
            "return_value": (
                "", b"\x00",
                [24, 2, ["STS", "IV"], 3, "STS", 0, "", 0.25, 1],
            )
        }
    }
    skill = GetPatternProps()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["num_experiments"] == 2
    assert res.data["experiments"] == ["STS", "IV"]
    assert res.data["selected_experiment"] == "STS"
    assert res.data["pre_measure_delay_s"] == 0.25
    assert res.data["save_scan_channels"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
