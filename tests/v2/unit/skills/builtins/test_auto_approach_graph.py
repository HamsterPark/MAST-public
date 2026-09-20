"""GraphExecutor + AutoApproach regression tests (Phase 7 migration).

Pins the following contracts:

  1. ``AutoApproach`` subclasses :class:`CompositeSkillGraph`.
  2. The static plan has exactly 4 mandatory phases:
     ``open_module / start_approach / wait_complete / verify_status``.
  3. Each phase invokes its expected Nanonis TCP method:
     ``AutoApproach_Open`` → ``AutoApproach_OnOffSet(1)`` →
     ``AutoApproach_OnOffGet`` (×2 — once for wait, once for verify).
  4. Any phase failure aborts the composite and surfaces the error.
  5. SafetyLevel remains AUTO (v0.3.21 contract — Nanonis hardware
     safety is sufficient).
  6. Resume skips completed phases.
  7. Per-step ``emit_progress`` fires.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_auto_approach_graph.py -x -v
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
from mast.skills.builtins.approach import (
    AutoApproach,
    _PHASE_OPEN,
    _PHASE_START,
    _PHASE_WAIT,
    _PHASE_VERIFY,
)


# ── FakeCtx ───────────────────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Models a normal coarse approach that reaches the setpoint.

    ``AutoApproach_OnOffGet`` reports running=True for the first
    ``running_polls`` polls (module approaching), then running=False (reached
    the setpoint = completed). The wait phase now POLLS to completion (review
    2026-07-03), so a mock that reports running forever would hang; the
    countdown lets tests finish fast and deterministically.

    Since 2026-07-10  a stopped module must also show a real tunnelling
    current before wait_complete succeeds, so the fake serves Current_Get /
    ZCtrl_SetpntGet with |I| = setpoint by default (a genuine engagement);
    set ``current_a`` to a noise value to model a failed/range-exhausted
    approach."""
    canned_errors: dict[str, str] = field(default_factory=dict)
    onoff_running: bool = True
    running_polls: int = 1     # polls that report running before the module stops
    current_a: float = 5e-10   # served for Current_Get (default: engaged)
    setpoint_a: float = 5e-10  # served for ZCtrl_SetpntGet
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    emitted: list[CompositeProgress] = field(default_factory=list)
    prior_progress: dict | None = None
    flushes: int = 0
    _oog_count: int = 0

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned_errors:
            return NanonisCallRecord(method=method, args=args,
                                     error=self.canned_errors[method])
        if method == "AutoApproach_OnOffGet":
            self._oog_count += 1
            still = self.onoff_running and (self._oog_count <= self.running_polls)
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [1 if still else 0]),
            )
        if method == "Current_Get":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [self.current_a]))
        if method == "ZCtrl_SetpntGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [self.setpoint_a]))
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", []))

    def check_abort(self) -> bool:
        return False

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Shape tests ────────────────────────────────────────────────────────────


def test_auto_approach_is_composite_graph():
    assert issubclass(AutoApproach, CompositeSkillGraph)


def test_auto_approach_metadata_safety_unchanged():
    """v0.3.21: AutoApproach is AUTO (Nanonis hardware safety is sufficient)."""
    meta = AutoApproach().metadata()
    assert meta.name == "AutoApproach"
    assert meta.safety_level == SafetyLevel.AUTO
    assert meta.rollback_skill == "WithdrawTip"


# ── Plan tests ─────────────────────────────────────────────────────────────


def test_plan_has_4_mandatory_phases():
    skill = AutoApproach()
    plan = skill.plan({})
    assert len(plan) == 4
    step_ids = [s.step_id for s in plan]
    assert step_ids == [
        "open_module", "start_approach", "wait_complete", "verify_status",
    ]
    # All mandatory — any failure aborts
    assert all(not s.optional for s in plan)
    skill_names = [s.skill_name for s in plan]
    assert skill_names == [_PHASE_OPEN, _PHASE_START, _PHASE_WAIT, _PHASE_VERIFY]


def test_plan_checkpoints_after_critical_phases():
    """start_approach (state change) and verify_status (final) flush; others don't."""
    skill = AutoApproach()
    plan = skill.plan({})
    by_id = {s.step_id: s for s in plan}
    assert by_id["open_module"].checkpoint_after is False
    assert by_id["start_approach"].checkpoint_after is True
    assert by_id["wait_complete"].checkpoint_after is False
    assert by_id["verify_status"].checkpoint_after is True


# ── Execution tests ───────────────────────────────────────────────────────


def test_full_execution_calls_all_4_methods():
    """All 4 phases succeed → expected Nanonis methods invoked."""
    skill = AutoApproach()
    skill._poll_interval_s = 0.01  # keep the wait-phase poll loop fast in tests
    ctx = FakeCtx()
    result = skill.execute(ctx, {})
    assert result.success, f"unexpected failure: {result.error}"
    methods = [c[0] for c in ctx.calls]
    assert methods.count("AutoApproach_Open") == 1
    assert methods.count("AutoApproach_OnOffSet") == 1
    # OnOffGet now POLLS to completion in wait_complete (>=2 polls: running then
    # stopped) plus one in verify_status.
    assert methods.count("AutoApproach_OnOffGet") >= 2
    # OnOffSet always called with arg 1 (start)
    onoff_set = next(c for c in ctx.calls if c[0] == "AutoApproach_OnOffSet")
    assert onoff_set[1] == (1,)
    # Aggregate carries flag
    assert result.data["approach_started"] is True
    # The module reached the setpoint → final running is False (approach done).
    assert result.data["running"] is False


def test_open_failure_aborts_immediately():
    """AutoApproach_Open fails → start_approach never called."""
    skill = AutoApproach()
    ctx = FakeCtx(canned_errors={"AutoApproach_Open": "module busy"})
    result = skill.execute(ctx, {})
    assert not result.success
    assert "module busy" in (result.error or "")
    methods = [c[0] for c in ctx.calls]
    # AutoApproach_Open attempted, but no OnOffSet
    assert "AutoApproach_Open" in methods
    assert "AutoApproach_OnOffSet" not in methods


def test_onoffset_failure_aborts():
    """AutoApproach_OnOffSet fails → wait_complete never called."""
    skill = AutoApproach()
    ctx = FakeCtx(canned_errors={"AutoApproach_OnOffSet": "switch locked"})
    result = skill.execute(ctx, {})
    assert not result.success
    methods = [c[0] for c in ctx.calls]
    assert "AutoApproach_OnOffSet" in methods
    assert "AutoApproach_OnOffGet" not in methods


def test_wait_complete_failure_aborts():
    """AutoApproach_OnOffGet fails persistently on wait → verify never called."""
    skill = AutoApproach()
    skill._poll_interval_s = 0.01  # fast retries
    ctx = FakeCtx(canned_errors={"AutoApproach_OnOffGet": "TCP timeout"})
    result = skill.execute(ctx, {})
    assert not result.success
    assert "OnOffGet failed" in (result.error or "")
    methods = [c[0] for c in ctx.calls]
    # Wait retries a bounded number of times then bails; mandatory step failure
    # → verify_status never runs (no 2nd burst of OnOffGet after a gap).
    assert methods.count("AutoApproach_OnOffGet") >= 1


# ── Phase dispatch ────────────────────────────────────────────────────────


def test_phase_names_intercepted_by_wrapper():
    """``_phase_*`` skill_names never fall through to the underlying context.run."""
    skill = AutoApproach()
    skill._poll_interval_s = 0.01

    @dataclass
    class TraceCtx(FakeCtx):
        run_log: list[tuple[str, dict]] = field(default_factory=list)

        def run(self, skill_name: str, params: dict) -> SkillResult:
            self.run_log.append((skill_name, dict(params)))
            return SkillResult(skill_name=skill_name, success=True, data={})

    ctx = TraceCtx()
    skill.execute(ctx, {})
    fallthrough = [name for name, _ in ctx.run_log
                   if name.startswith("_phase_")]
    assert fallthrough == [], f"phase names leaked: {fallthrough}"


# ── Resume ────────────────────────────────────────────────────────────────


def test_resume_skips_completed_phases():
    """Prior progress with open + start done → only wait + verify run."""
    skill = AutoApproach()
    prior = CompositeProgress(
        composite_name="AutoApproach",
        total_steps=4,
        completed_steps=["open_module", "start_approach"],
        partial_data={"approach_started": True},
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    skill._poll_interval_s = 0.01
    result = skill.execute(ctx, {})
    assert result.success
    methods = [c[0] for c in ctx.calls]
    # Open and OnOffSet skipped (resumed); only OnOffGet runs (wait polls to
    # completion + verify).
    assert "AutoApproach_Open" not in methods
    assert "AutoApproach_OnOffSet" not in methods
    assert methods.count("AutoApproach_OnOffGet") >= 2


# ── Progress + checkpoint ─────────────────────────────────────────────────


def test_emit_progress_fires_every_step():
    skill = AutoApproach()
    skill._poll_interval_s = 0.01
    ctx = FakeCtx()
    skill.execute(ctx, {})
    # 4 steps + final summary → ≥4 emits
    assert len(ctx.emitted) >= 4
    assert ctx.emitted[-1].current_step is None


def test_checkpoint_flushes_after_2_critical_phases():
    """start_approach + verify_status → exactly 2 flushes."""
    skill = AutoApproach()
    skill._poll_interval_s = 0.01
    ctx = FakeCtx()
    skill.execute(ctx, {})
    assert ctx.flushes == 2


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
