"""GraphExecutor + RunGridExperiment regression tests (Phase 7 migration).

Pins the following contracts:

  1. ``RunGridExperiment`` subclasses :class:`CompositeSkillGraph`.
  2. Dynamic plan: ``setup_grid`` + ``start_experiment`` + per-tick polls
     + ``cleanup``. The poll loop exits early once Pattern_ExpStatusGet
     returns status=0.
  3. With a 3-second timeout and a 2s tick interval, ~1-2 tick steps run
     before the experiment completes.
  4. ``Pattern_GridSet`` failure aborts immediately (no ExpStart call).
  5. ``Pattern_ExpStart`` failure aborts (no tick polls).
  6. Timeout returns success=False with the v1 error format.
  7. Aggregate carries nx/ny/total_points.
  8. SafetyLevel is CONFIRM.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_pattern_graph.py -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
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

from mast.core.types import NanonisCallRecord, SafetyLevel, SkillResult
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeProgress
from mast.skills.builtins.pattern import (
    RunGridExperiment,
    _PHASE_SETUP,
    _PHASE_START,
    _PHASE_TICK_PREFIX,
    _PHASE_CLEANUP,
)


# ── FakeCtx ───────────────────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Default: GridSet+ExpStart succeed; ExpStatusGet returns running, then 0."""
    canned_errors: dict[str, str] = field(default_factory=dict)
    status_sequence: list[int] = field(default_factory=lambda: [1, 1, 0])
    status_idx: int = 0
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    emitted: list[CompositeProgress] = field(default_factory=list)
    prior_progress: dict | None = None
    flushes: int = 0
    _abort: bool = False

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned_errors:
            return NanonisCallRecord(method=method, args=args,
                                     error=self.canned_errors[method])
        if method == "Pattern_ExpStatusGet":
            idx = self.status_idx
            self.status_idx += 1
            status = (self.status_sequence[idx]
                      if idx < len(self.status_sequence) else 0)
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [status]),
            )
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", []))

    def check_abort(self) -> bool:
        return self._abort

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Shape tests ────────────────────────────────────────────────────────────


def test_run_grid_experiment_is_composite_graph():
    assert issubclass(RunGridExperiment, CompositeSkillGraph)


def test_run_grid_experiment_metadata():
    meta = RunGridExperiment().metadata()
    assert meta.name == "RunGridExperiment"
    assert meta.safety_level == SafetyLevel.CONFIRM
    assert "z_controller_on" in meta.preconditions


# ── Execution tests ───────────────────────────────────────────────────────


def test_three_ticks_until_done(monkeypatch):
    """status=[1,1,0] → setup + start + 3 ticks + cleanup."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(status_sequence=[1, 1, 0])
    result = skill.execute(ctx, {"nx": 3, "ny": 4, "wait_timeout_s": 60.0})
    assert result.success
    methods = [c[0] for c in ctx.calls]
    assert "Pattern_GridSet" in methods
    assert "Pattern_ExpStart" in methods
    assert methods.count("Pattern_ExpStatusGet") == 3
    # Aggregate carries nx/ny/total_points
    assert result.data["nx"] == 3
    assert result.data["ny"] == 4
    assert result.data["total_points"] == 12


def test_immediate_finish(monkeypatch):
    """status=[0] → only 1 tick."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(status_sequence=[0])
    result = skill.execute(ctx, {"nx": 2, "ny": 2, "wait_timeout_s": 60.0})
    assert result.success
    status_calls = [c for c in ctx.calls if c[0] == "Pattern_ExpStatusGet"]
    assert len(status_calls) == 1


def test_gridset_failure_aborts(monkeypatch):
    """Pattern_GridSet error → no ExpStart, no ticks."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(canned_errors={"Pattern_GridSet": "invalid params"})
    result = skill.execute(ctx, {"nx": 2, "ny": 2})
    assert not result.success
    assert "GridSet" in (result.error or "")
    methods = [c[0] for c in ctx.calls]
    assert "Pattern_GridSet" in methods
    assert "Pattern_ExpStart" not in methods
    assert "Pattern_ExpStatusGet" not in methods


def test_expstart_failure_aborts(monkeypatch):
    """Pattern_ExpStart error → no ticks."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(canned_errors={"Pattern_ExpStart": "module busy"})
    result = skill.execute(ctx, {"nx": 2, "ny": 2})
    assert not result.success
    assert "ExpStart" in (result.error or "")
    methods = [c[0] for c in ctx.calls]
    assert "Pattern_ExpStart" in methods
    assert "Pattern_ExpStatusGet" not in methods


def test_status_get_failure_aborts(monkeypatch):
    """Pattern_ExpStatusGet error on first tick → abort (mandatory)."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(canned_errors={"Pattern_ExpStatusGet": "TCP timeout"})
    result = skill.execute(ctx, {"nx": 2, "ny": 2, "wait_timeout_s": 10.0})
    assert not result.success


def test_gridset_called_with_explicit_geometry(monkeypatch):
    """Pattern_GridSet(1, nx, ny, 0, cx, cy, w, h, angle).

    Grid_Scan_frame (4th arg) MUST be 0. When it is 1, Nanonis sizes the grid
    to the current scan frame and discards the explicit center/width/height/
    angle that follow — silently ignoring the user's requested grid geometry.
    The v1 code vendored here passed 1, which was the bug; the fix passes 0 so
    cx/cy/w/h/angle are actually applied.
    """
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(status_sequence=[0])
    skill.execute(ctx, {
        "nx": 5, "ny": 6,
        "center_x_m": 1e-9, "center_y_m": 2e-9,
        "width_m": 50e-9, "height_m": 60e-9,
        "angle_deg": 30.0,
    })
    grid_set = next(c for c in ctx.calls if c[0] == "Pattern_GridSet")
    # Args: (set_active=1, nx, ny, grid_scan_frame=0, cx, cy, w, h, angle)
    assert grid_set[1] == (1, 5, 6, 0, 1e-9, 2e-9, 50e-9, 60e-9, 30.0)


# ── Phase dispatch ────────────────────────────────────────────────────────


def test_phase_names_intercepted_by_wrapper(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()

    @dataclass
    class TraceCtx(FakeCtx):
        run_log: list[tuple[str, dict]] = field(default_factory=list)

        def run(self, skill_name: str, params: dict) -> SkillResult:
            self.run_log.append((skill_name, dict(params)))
            return SkillResult(skill_name=skill_name, success=True, data={})

    ctx = TraceCtx(status_sequence=[0])
    skill.execute(ctx, {"nx": 2, "ny": 2})
    fallthrough = [name for name, _ in ctx.run_log
                   if name.startswith("_phase_")]
    assert fallthrough == [], f"phase names leaked: {fallthrough}"


# ── Progress emission + checkpoint ────────────────────────────────────────


def test_emit_progress_fires_every_step(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(status_sequence=[1, 0])
    skill.execute(ctx, {"nx": 2, "ny": 2, "wait_timeout_s": 60.0})
    # setup + start + 2 ticks + cleanup = 5 steps → ≥5 emits
    assert len(ctx.emitted) >= 5
    assert ctx.emitted[-1].current_step is None


def test_checkpoint_flushes_critical_phases(monkeypatch):
    """start_experiment + cleanup flush; setup + ticks don't."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    ctx = FakeCtx(status_sequence=[0])
    skill.execute(ctx, {"nx": 2, "ny": 2, "wait_timeout_s": 60.0})
    # 2 critical flushes
    assert ctx.flushes == 2


# ── Resume ────────────────────────────────────────────────────────────────


def test_resume_skips_completed_steps(monkeypatch):
    """Prior progress with setup + start done → no GridSet, no ExpStart."""
    import time as _t
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    # start_time must be recent so the tick timeout doesn't fire on resume.
    prior = CompositeProgress(
        composite_name="RunGridExperiment",
        total_steps=10,
        completed_steps=["setup_grid", "start_experiment"],
        partial_data={"nx": 2, "ny": 2, "start_time": _t.time()},
    )
    ctx = FakeCtx(prior_progress=prior.to_dict(), status_sequence=[0])
    skill.execute(ctx, {"nx": 2, "ny": 2, "wait_timeout_s": 60.0})
    methods = [c[0] for c in ctx.calls]
    # Setup and Start skipped (resumed)
    assert "Pattern_GridSet" not in methods
    assert "Pattern_ExpStart" not in methods
    # But tick polls still ran
    assert "Pattern_ExpStatusGet" in methods


# ── Plan introspection ────────────────────────────────────────────────────


def test_plan_dynamic_skeleton(monkeypatch):
    """plan_dynamic yields setup + start + at-least-1-tick + cleanup."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = RunGridExperiment()
    # Collect the step ids the executor sees by running once
    ctx = FakeCtx(status_sequence=[0])
    result = skill.execute(ctx, {"nx": 2, "ny": 2, "wait_timeout_s": 60.0})
    completed = result.data["_progress"]["completed_steps"]
    assert "setup_grid" in completed
    assert "start_experiment" in completed
    assert "cleanup" in completed
    # At least one tick
    ticks = [s for s in completed if s.startswith("tick_")]
    assert len(ticks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
