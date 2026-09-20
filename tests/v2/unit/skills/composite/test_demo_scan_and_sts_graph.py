"""GraphExecutor + DemoScanAndSTS regression tests (Phase 7 framework).

Pins the following contracts:

  1. ``plan(params)`` returns ``6 + 2 * sts_count`` :class:`CompositeStep`
     items: ConfigureScan -> SetScanSpeed -> StartScan -> WaitScanComplete
     -> SaveScan -> ConfigureSTS -> (MoveToXY + AcquireSTS) x sts_count.
  2. Running an empty-context executor walks every step exactly once and
     emits a :class:`CompositeProgress` snapshot whose ``completed_steps``
     length matches the plan (when all sub-skills succeed).
  3. **Resume**: re-running with a prior ``composite_progress`` covering
     the first 6 setup steps causes the executor to skip them — only the
     (move + sts) pairs run fresh.
  4. emit_progress fires after every successful step + once on finish.
  5. ``sts_count`` defaults to 5 — the plan has 16 total steps.
  6. STS positions follow the v1 layout: center + corners of inner square
     (offset = scan_size / 3).
  7. ``WaitScanComplete`` is the only "wait" — the v1 ``wait_scan_complete``
     helper is no longer reachable from the migrated skill.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/composite/test_demo_scan_and_sts_graph.py -x -v
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

from mast.core.types import SkillResult
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
)
from mast.skills.composite.demo_scan_and_sts import DemoScanAndSTS


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context — supports run(), emit_progress, checkpoint_flush."""
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    # Per-skill failure switch: when set, that skill returns success=False
    failing_skills: set[str] = field(default_factory=set)
    # WaitScanComplete carries data={"timed_out": ..., "polls": ...}
    wait_times_out: bool = False
    # v6.1.3 — a frame that stopped part-way (neither flag was consumed here
    # before; see KNOWN_ISSUES §2.24).
    wait_lines_done: int | None = None
    wait_lines_total: int = 512

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name in self.failing_skills:
            return SkillResult(
                skill_name=skill_name, success=False, error="canned failure",
            )
        if skill_name == "WaitScanComplete":
            done = (self.wait_lines_total if self.wait_lines_done is None
                    else self.wait_lines_done)
            stopped_early = done < self.wait_lines_total
            return SkillResult(
                skill_name=skill_name, success=True,
                data={"timed_out": self.wait_times_out, "polls": 1,
                      "stopped_early": stopped_early,
                      "outcome": ("timed_out" if self.wait_times_out
                                  else "stopped_early" if stopped_early
                                  else "completed"),
                      "lines_done": done, "lines_total": self.wait_lines_total,
                      "lines_verified": True},
            )
        return SkillResult(skill_name=skill_name, success=True, data={})

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(
            CompositeProgress.from_dict(progress.to_dict())
        )

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Tests ────────────────────────────────────────────────────────────────


def test_plan_step_count_default_sts_count():
    """Default sts_count=5 → 6 setup + 2*5 = 16 steps."""
    skill = DemoScanAndSTS()
    plan = skill.plan({})
    assert len(plan) == 16
    # First six are the fixed setup pipeline
    expected_setup_ids = [
        "configure_scan", "set_scan_speed", "start_scan",
        "wait_scan", "save_scan", "configure_sts",
    ]
    assert [s.step_id for s in plan[:6]] == expected_setup_ids
    # Then 5 (move + sts) pairs
    for idx in range(1, 6):
        assert plan[5 + 2 * idx - 1].step_id == f"move_{idx}"
        assert plan[5 + 2 * idx].step_id == f"sts_{idx}"
    # All ids unique
    assert len({s.step_id for s in plan}) == 16


# ── an unfinished frame is reported, not fatal (v6.1.3, §2.24) ──────────
#
# Before v6.1.3 this composite consumed NEITHER flag: a timed-out or
# interrupted frame flowed into SaveScan and the whole STS run as if it were a
# finished image. The fix reports rather than aborts, and that is a judgement —
# so these tests pin BOTH halves of it. Dropping either one turns the fix into
# a different behaviour that still looks tested.


def _truncated(**over):
    return FakeCtx(wait_lines_done=100, wait_lines_total=512, **over)


def test_an_unfinished_frame_is_reported_in_the_result():
    """The half that must not regress into silence."""
    result = DemoScanAndSTS().execute(_truncated(), {})
    assert result.data["scan_completed"] is False
    assert result.data["scan_outcome"] == "stopped_early"
    assert (result.data["scan_lines_done"],
            result.data["scan_lines_total"]) == (100, 512)


def test_an_unfinished_frame_does_not_kill_the_sts_run():
    """The other half. The STS points come from _build_sts_positions(center,
    size) — geometry, never the image — so a truncated frame invalidates not
    one spectrum. Aborting would throw away good data to punish an unrelated
    step, and this skill exists to run a guaranteed demo sequence."""
    result = DemoScanAndSTS().execute(_truncated(), {})
    assert result.success
    assert result.data["sts_succeeded"] == 5
    assert result.data["sts_failed"] == 0


def test_a_timed_out_frame_is_reported_the_same_way():
    """The other half of "both halves" — timeout was equally unconsumed."""
    result = DemoScanAndSTS().execute(FakeCtx(wait_times_out=True), {})
    assert result.data["scan_completed"] is False
    assert result.data["scan_outcome"] == "timed_out"
    assert result.success                      # still reported, still not fatal


def test_scan_completed_is_not_the_same_field_as_scan_saved():
    """A truncated frame can be saved perfectly well, and then the .sxm on disk
    looks like a complete image to everything downstream. Collapsing the two
    would put the lie back."""
    result = DemoScanAndSTS().execute(_truncated(), {})
    assert result.data["scan_saved"] is True
    assert result.data["scan_completed"] is False


def test_a_complete_frame_reports_completed():
    """Control — otherwise scan_completed could just be hardwired False."""
    result = DemoScanAndSTS().execute(FakeCtx(), {})
    assert result.data["scan_completed"] is True
    assert result.data["scan_outcome"] == "completed"


def test_plan_step_count_with_sts_count_3():
    """sts_count=3 → 6 + 2*3 = 12 steps."""
    skill = DemoScanAndSTS()
    plan = skill.plan({"sts_count": 3})
    assert len(plan) == 12


def test_plan_step_count_with_sts_count_1():
    """sts_count=1 → 6 + 2 = 8 steps (single center STS)."""
    skill = DemoScanAndSTS()
    plan = skill.plan({"sts_count": 1})
    assert len(plan) == 8
    # The lone STS is at the scan center
    move = next(s for s in plan if s.step_id == "move_1")
    assert move.params["x_m"] == 0.0
    assert move.params["y_m"] == 0.0


def test_plan_skill_names_are_real_sub_skills():
    """Every step.skill_name must be a real registered skill (no synthetic phases)."""
    skill = DemoScanAndSTS()
    plan = skill.plan({"sts_count": 2})
    real_skills = {
        "ConfigureScan", "SetScanSpeed", "StartScan", "WaitScanComplete",
        "SaveScan", "ConfigureSTS", "MoveToXY", "AcquireSTS",
    }
    plan_skills = {s.skill_name for s in plan}
    assert plan_skills.issubset(real_skills)


def test_wait_scan_complete_timeout_ms_conversion():
    """scan_timeout_s=180 → WaitScanComplete params.timeout_ms == 180_000."""
    skill = DemoScanAndSTS()
    plan = skill.plan({"scan_timeout_s": 180.0})
    wait = next(s for s in plan if s.step_id == "wait_scan")
    assert wait.params["timeout_ms"] == 180_000


def test_sts_positions_center_plus_corners():
    """v1 layout: 5 STS = center + 4 corners of inner square (offset = size/3)."""
    skill = DemoScanAndSTS()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "scan_size_m": 30e-9, "sts_count": 5,
    })
    offset = 30e-9 / 3.0
    expected = [
        (0.0, 0.0),
        (-offset, -offset),
        (offset, -offset),
        (offset, offset),
        (-offset, offset),
    ]
    actual = []
    for idx in range(1, 6):
        m = next(s for s in plan if s.step_id == f"move_{idx}")
        actual.append((m.params["x_m"], m.params["y_m"]))
    for (ax, ay), (ex, ey) in zip(actual, expected):
        assert ax == pytest.approx(ex)
        assert ay == pytest.approx(ey)


def test_full_execution_walks_every_step():
    """Fresh context (no resume) → every step invoked exactly once."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"sts_count": 5})
    # 16 invocations: 6 setup + 5 move + 5 sts
    assert len(ctx.run_log) == 16
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names.count("ConfigureScan") == 1
    assert skill_names.count("SetScanSpeed") == 1
    assert skill_names.count("StartScan") == 1
    assert skill_names.count("WaitScanComplete") == 1
    assert skill_names.count("SaveScan") == 1
    assert skill_names.count("ConfigureSTS") == 1
    assert skill_names.count("MoveToXY") == 5
    assert skill_names.count("AcquireSTS") == 5
    assert result.success
    assert result.data["sts_total"] == 5
    assert result.data["sts_succeeded"] == 5
    assert result.data["sts_failed"] == 0


def test_progress_snapshot_carries_into_result():
    """`_progress` snapshot is lifted into result.data for the adapter."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"sts_count": 2})
    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "DemoScanAndSTS"
    # 10 steps for sts_count=2 (6 setup + 2 move + 2 sts)
    assert len(snap["completed_steps"]) == 10
    assert snap["aborted"] is False


def test_resume_skips_setup_steps():
    """Prior progress covers the 6 setup steps → only sts pairs run fresh."""
    skill = DemoScanAndSTS()
    plan = skill.plan({"sts_count": 3})
    first_six = [s.step_id for s in plan[:6]]  # all setup
    prior = CompositeProgress(
        composite_name="DemoScanAndSTS",
        total_steps=12,
        completed_steps=list(first_six),
        partial_data={
            "scan_size_m": 50e-9,
            "scan_center": (0.0, 0.0),
            "sts_total": 3,
            "sts_succeeded": 0,
            "sts_failed": 0,
            "scan_saved": True,
        },
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {"sts_count": 3})
    # Only move/sts pairs ran fresh: 2*3 = 6 fresh sub-skill invocations
    assert len(ctx.run_log) == 6
    snap = result.data["_progress"]
    # All 12 steps completed (6 resumed + 6 new)
    assert len(snap["completed_steps"]) == 12
    assert result.success
    # All 3 STS succeeded (succeeded counter starts at 0, +3 from this run)
    assert result.data["sts_succeeded"] == 3


def test_progress_emitted_every_step():
    """emit_progress fires after each step + once on finish.

    sts_count=1 → 8 steps → >= 9 emits (per-step + final summary).
    """
    skill = DemoScanAndSTS()
    ctx = FakeCtx()
    skill.execute(ctx, {"sts_count": 1})
    assert len(ctx.emitted) >= 9
    # Final emit clears current_step
    assert ctx.emitted[-1].current_step is None


def test_save_scan_optional_failure_continues():
    """SaveScan failure is non-fatal (v1 behaviour) — rest of plan still runs."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx(failing_skills={"SaveScan"})
    result = skill.execute(ctx, {"sts_count": 2})
    assert result.success, f"unexpected failure: {result.error}"
    assert result.data["scan_saved"] is False
    # All other steps should have still run
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names.count("ConfigureSTS") == 1
    assert skill_names.count("AcquireSTS") == 2


def test_mandatory_failure_aborts():
    """ConfigureScan failure is mandatory → aborts immediately."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx(failing_skills={"ConfigureScan"})
    result = skill.execute(ctx, {"sts_count": 5})
    assert not result.success
    # Only ConfigureScan should have run before abort
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names == ["ConfigureScan"]


def test_move_failure_skips_matching_sts():
    """v1 behaviour: when MoveToXY fails, the matching AcquireSTS is skipped.

    The dynamic plan looks at executor.progress.failed_steps and omits the
    sts_N step when move_N has been recorded as failed. This avoids
    acquiring spectra at the wrong (unmoved) position.
    """
    skill = DemoScanAndSTS()
    ctx = FakeCtx(failing_skills={"MoveToXY"})
    result = skill.execute(ctx, {"sts_count": 3})
    skill_names = [name for name, _ in ctx.run_log]
    # All 3 MoveToXY attempts ran (and failed), but no AcquireSTS should run.
    assert skill_names.count("MoveToXY") == 3
    assert skill_names.count("AcquireSTS") == 0
    # sts_failed reflects 3 unmet points
    assert result.data["sts_succeeded"] == 0
    assert result.data["sts_failed"] == 3
    # Composite still succeeds since the failures are all optional
    assert result.success


def test_sts_failure_does_not_abort():
    """An AcquireSTS failure is optional — composite continues to next point."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx(failing_skills={"AcquireSTS"})
    result = skill.execute(ctx, {"sts_count": 3})
    # Composite still finishes since STS steps are optional
    assert result.success
    assert result.data["sts_succeeded"] == 0
    assert result.data["sts_failed"] == 3


def test_checkpoint_flushes_at_marked_steps():
    """Marked steps: wait_scan, save_scan, and every sts_N step (per spectrum)."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx()
    skill.execute(ctx, {"sts_count": 2})
    # 2 fixed checkpoints (wait_scan + save_scan) + 2 sts checkpoints
    assert ctx.flushes == 4


def test_make_tool_wraps_to_v2_adapter():
    """make_tool() should produce a LangChain StructuredTool via wrap_skill."""
    from mast.skills.composite.demo_scan_and_sts import make_tool

    def _provider():
        return None

    tool = make_tool(_provider)
    assert tool.name == "DemoScanAndSTS"
    fields = tool.args_schema.model_fields
    # All params optional (defaults provided) so an LLM can call with {}
    assert "center_x_m" in fields
    assert "scan_size_m" in fields
    assert "sts_count" in fields
    required = [k for k, v in fields.items() if v.is_required()]
    assert required == [], f"all params should be optional; required: {required}"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
