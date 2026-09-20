"""GraphExecutor + FullScan regression tests (Phase 7 framework).

Pins the following contracts:

  1. ``plan(params)`` returns 4 :class:`CompositeStep` items:
     ConfigureScan -> SetScanSpeed -> StartScan -> WaitScanComplete.
  2. Running an empty-context executor walks every step exactly once and
     emits a :class:`CompositeProgress` snapshot whose ``completed_steps``
     length matches the plan.
  3. **Resume**: re-running with a prior ``composite_progress`` covering
     the first 3 steps causes the executor to skip them.
  4. ``WaitScanComplete.data["timed_out"] is True`` is converted into a
     hard SkillResult failure (matches v1 behaviour).
  5. Post-scan crash detection: data with near-zero variance triggers a
     ``CRASH_DETECTED`` failure (matches v1 ``_check_scan_data``).
  6. ``line_time_s`` is capped at 0.1 (v1 ``min(line_time_s, 0.1)``).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_full_scan_graph.py -x -v
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

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
)
from mast.skills.composite.full_scan import FullScan


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context — supports run(), safe_call(), + progress hooks."""
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    safe_call_log: list[tuple[str, tuple]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    # When True, WaitScanComplete reports a timeout
    wait_times_out: bool = False
    # When set, WaitScanComplete reports a scan that stopped part-way (v6.1.2)
    wait_lines_done: int | None = None
    wait_lines_total: int = 512
    # Canned Scan_FrameDataGrab data — varying = good, flat = crash
    scan_data: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0])

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "WaitScanComplete":
            done = (self.wait_lines_total if self.wait_lines_done is None
                    else self.wait_lines_done)
            stopped_early = done < self.wait_lines_total
            outcome = ("timed_out" if self.wait_times_out
                       else "stopped_early" if stopped_early else "completed")
            return SkillResult(
                skill_name=skill_name,
                success=True,
                data={"timed_out": self.wait_times_out, "polls": 1,
                      "stopped_early": stopped_early, "outcome": outcome,
                      "lines_done": done, "lines_total": self.wait_lines_total,
                      "lines_verified": True},
            )
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.safe_call_log.append((method, args))
        if method == "Scan_FrameDataGrab":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", self.scan_data),
            )
        return NanonisCallRecord(method=method, args=args, return_value=None)

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(
            CompositeProgress.from_dict(progress.to_dict())
        )

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Tests ────────────────────────────────────────────────────────────────


def test_plan_step_count_is_4():
    """4 main-flow steps: configure / set_speed / start_scan / wait_scan."""
    skill = FullScan()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
    })
    assert len(plan) == 4
    step_ids = [s.step_id for s in plan]
    assert step_ids == ["configure", "set_speed", "start_scan", "wait_scan"]
    skill_names = [s.skill_name for s in plan]
    assert skill_names == [
        "ConfigureScan", "SetScanSpeed", "StartScan", "WaitScanComplete",
    ]
    # All step_ids unique
    assert len({s.step_id for s in plan}) == 4


def test_plan_honours_requested_line_time():
    """Review 2026-07-03: the requested line time is HONOURED (the ParameterSpec
    bounds it to 0.01–60 s); the old min(...,0.1) silently clamped every slow
    high-quality scan back to 0.1 s/line."""
    skill = FullScan()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
        "line_time_s": 1.0,  # request 1 s — must be honoured, not clamped
    })
    set_speed = next(s for s in plan if s.step_id == "set_speed")
    assert set_speed.params["fwd_line_time"] == 1.0
    assert set_speed.params["bwd_line_time"] == 1.0


def test_plan_wait_timeout_param_in_ms():
    """WaitScanComplete consumes timeout in ms — verify ms conversion."""
    skill = FullScan()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
        "wait_timeout_s": 120.0,
    })
    wait = next(s for s in plan if s.step_id == "wait_scan")
    assert wait.params["timeout_ms"] == 120_000


def test_full_execution_walks_every_step():
    """All 4 sub-skills invoked once with non-flat scan data → success."""
    skill = FullScan()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 1e-9, "center_y_m": 2e-9,
        "width_m": 50e-9, "height_m": 50e-9,
    })
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names == [
        "ConfigureScan", "SetScanSpeed", "StartScan", "WaitScanComplete",
    ]
    assert result.success
    # Aggregate echoes input geometry
    assert result.data["center_x_m"] == 1e-9
    assert result.data["center_y_m"] == 2e-9
    assert result.data["width_m"] == 50e-9
    assert result.data["height_m"] == 50e-9


def test_progress_snapshot_carries_into_result():
    """`_progress` snapshot is lifted into result.data for the adapter."""
    skill = FullScan()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
    })
    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "FullScan"
    # All 4 graph steps completed
    assert len(snap["completed_steps"]) == 4
    assert snap["aborted"] is False


def test_wait_timeout_becomes_hard_failure():
    """v1 semantics: WaitScanComplete.timed_out=True → fail."""
    skill = FullScan()
    ctx = FakeCtx(wait_times_out=True)
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
        "wait_timeout_s": 30.0,
    })
    assert not result.success
    assert "timed out" in result.error.lower()


# ── a scan that stopped part-way (v6.1.2) ────────────────────────────────


def _scan(ctx, **over):
    params = {"center_x_m": 0.0, "center_y_m": 0.0,
              "width_m": 50e-9, "height_m": 50e-9}
    params.update(over)
    return FullScan().execute(ctx, params)


def test_stopped_early_becomes_a_hard_failure():
    """Any reason a scan stops part-way — the operator's Stop button, a
    Nanonis-side halt, a safety halt — used to arrive here as
    success=True/timed_out=False, i.e. indistinguishable from a finished frame,
    and FullScan went on to crash-check, save and report a clean scan."""
    result = _scan(FakeCtx(wait_lines_done=123, wait_lines_total=512))
    assert not result.success
    assert "123/512" in result.error
    assert result.data.get("stopped_early") is True


def test_a_truncated_scan_never_reaches_the_crash_check():
    """The crash check reads Scan_FrameDataGrab. Over a frame that is mostly
    unacquired rows it is not measuring a tip — it must not run at all, and it
    certainly must not be the thing that decides the scan was fine."""
    ctx = FakeCtx(wait_lines_done=1, wait_lines_total=512)
    _scan(ctx)
    assert not [c for c in ctx.safe_call_log if c[0] == "Scan_FrameDataGrab"]


def test_a_timeout_is_reported_as_a_timeout_not_as_an_early_stop():
    """Both truncate the frame; they call for different actions (raise the
    timeout vs find out who stopped the scan), so they must not merge."""
    result = _scan(FakeCtx(wait_times_out=True, wait_lines_done=10),
                   wait_timeout_s=30.0)
    assert not result.success
    assert "timed out" in result.error.lower()


def test_a_completed_scan_carries_the_evidence_it_completed():
    """'512/512, verified' is what makes a clean result a CHECKED claim rather
    than an assumed one."""
    result = _scan(FakeCtx())
    assert result.success
    assert result.data["wait_outcome"] == "completed"
    assert result.data["scan_lines_done"] == result.data["scan_lines_total"]
    assert result.data["scan_lines_verified"] is True


def test_crash_detection_flat_data():
    """v1 semantics: near-zero-variance scan data → CRASH_DETECTED fail."""
    skill = FullScan()
    # All-zero scan data → data_range < 1e-25 → crash
    ctx = FakeCtx(scan_data=[0.0, 0.0, 0.0, 0.0])
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
    })
    assert not result.success
    assert "CRASH_DETECTED" in result.error
    assert result.data.get("crash_indicator") is True


def test_resume_skips_completed_steps():
    """Prior progress covers first 3 steps → only wait_scan runs fresh."""
    skill = FullScan()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
    })
    completed = [s.step_id for s in plan[:3]]
    prior = CompositeProgress(
        composite_name="FullScan",
        total_steps=4,
        completed_steps=list(completed),
        partial_data={},
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
    })
    # Only wait_scan should have triggered a fresh sub-skill invocation
    sub_names = [name for name, _ in ctx.run_log]
    assert sub_names == ["WaitScanComplete"]
    snap = result.data["_progress"]
    assert len(snap["completed_steps"]) == 4
    assert result.success


def test_progress_emitted_every_step():
    """emit_progress fires after each successful step + once on finish."""
    skill = FullScan()
    ctx = FakeCtx()
    skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
    })
    # 4 steps → ≥ 5 emits (per-step + final)
    assert len(ctx.emitted) >= 5
    # Final emit has current_step=None
    assert ctx.emitted[-1].current_step is None


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
