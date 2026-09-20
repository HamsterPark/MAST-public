"""GraphExecutor + GridSTS regression tests (Phase 7 framework).

Pins three contracts:

  1. ``plan(params)`` is a list of :class:`CompositeStep` whose length
     equals ``1 + 2 * nx * ny`` (one ConfigureSTS + per-point Move+STS).
  2. Running an empty-context executor walks every step and emits a
     :class:`CompositeProgress` snapshot whose ``completed_steps`` length
     equals the step count (when all sub-skills succeed).
  3. **Resume**: re-running with a ``prior_state.composite_progress``
     entry whose ``completed_steps`` already covers half the plan causes
     the executor to skip those steps (no ``context.run`` invocations
     for skipped IDs).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_grid_sts_graph.py -x -v
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
    GraphExecutor,
)
from mast.skills.composite.grid_sts import GridSTS


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context — supports run() and the optional progress hooks."""
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        # Every sub-skill succeeds with empty data
        return SkillResult(skill_name=skill_name, success=True, data={})

    def emit_progress(self, progress: CompositeProgress) -> None:
        # Append a *copy* so later mutation in the executor doesn't
        # rewrite history.
        self.emitted.append(
            CompositeProgress.from_dict(progress.to_dict())
        )

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Tests ────────────────────────────────────────────────────────────────


def test_plan_step_count_3x3():
    """3×3 grid → 1 ConfigureSTS + 9 × (Move + STS) = 19 steps."""
    skill = GridSTS()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 3, "ny": 3, "spacing_m": 1e-9,
    })
    assert len(plan) == 1 + 2 * 3 * 3
    assert plan[0].step_id == "configure"
    assert plan[0].skill_name == "ConfigureSTS"
    # Move/STS pattern alternates
    assert plan[1].step_id == "move_0_0"
    assert plan[2].step_id == "sts_0_0"
    # All steps have unique ids
    assert len({s.step_id for s in plan}) == len(plan)


def test_plan_step_count_1x4():
    """1×4 grid → 1 + 4 × 2 = 9 steps."""
    skill = GridSTS()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 1, "ny": 4, "spacing_m": 1e-9,
    })
    assert len(plan) == 1 + 2 * 1 * 4


def test_full_execution_walks_every_step():
    """Fresh context (no resume) → every step invoked exactly once."""
    skill = GridSTS()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 2, "ny": 2, "spacing_m": 1e-9,
    })
    # 1 ConfigureSTS + 2*2 = 4 points × (Move + STS) = 9 invocations
    assert len(ctx.run_log) == 9
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names.count("ConfigureSTS") == 1
    assert skill_names.count("MoveToXY") == 4
    assert skill_names.count("AcquireSTS") == 4
    assert result.success
    assert result.data["succeeded"] == 4
    assert result.data["failed"] == 0


def test_progress_snapshot_carries_into_result():
    """`_progress` snapshot is lifted into result.data for the adapter."""
    skill = GridSTS()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 2, "ny": 2, "spacing_m": 1e-9,
    })
    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "GridSTS"
    assert len(snap["completed_steps"]) == 9   # all 9 steps logged
    assert snap["aborted"] is False


def test_resume_skips_completed_steps():
    """Prior progress with first 5 steps marked done → only 4 fresh runs."""
    skill = GridSTS()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 2, "ny": 2, "spacing_m": 1e-9,
    })
    first_five = [s.step_id for s in plan[:5]]   # configure, move_0_0, sts_0_0, move_1_0, sts_1_0
    prior = CompositeProgress(
        composite_name="GridSTS",
        total_steps=9,
        completed_steps=list(first_five),
        partial_data={"nx": 2, "ny": 2, "spacing_m": 1e-9,
                      "succeeded": 2, "failed": 0},
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 2, "ny": 2, "spacing_m": 1e-9,
    })
    # Exactly 4 NEW sub-skill invocations (move_0_1, sts_0_1, move_1_1, sts_1_1)
    assert len(ctx.run_log) == 4
    snap = result.data["_progress"]
    # All 9 steps now in completed_steps (5 resumed + 4 new)
    assert len(snap["completed_steps"]) == 9
    # succeeded increments only on actually-run sts steps
    assert result.data["succeeded"] == 2 + 2  # prior 2 + 2 new
    assert result.success


def test_progress_emitted_every_step():
    """emit_progress fires after each successful step + once on finish."""
    skill = GridSTS()
    ctx = FakeCtx()
    skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 1, "ny": 1, "spacing_m": 1e-9,
    })
    # 1 ConfigureSTS + 1 Move + 1 STS = 3 steps → ≥ 4 emits (per-step + final)
    assert len(ctx.emitted) >= 4
    # Final emit has current_step=None
    assert ctx.emitted[-1].current_step is None


def test_checkpoint_flushes_only_on_marked_steps():
    """Steps with checkpoint_after=True trigger flush; others don't.

    For a 1×1 grid: configure (True) + move (False) + sts (True) = 2 flushes.
    """
    skill = GridSTS()
    ctx = FakeCtx()
    skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "nx": 1, "ny": 1, "spacing_m": 1e-9,
    })
    assert ctx.flushes == 2


class TestPointCount:
    """A grid point is ONE spectrum — but it costs TWO plan steps (move + sts),
    plus one configure step up front. ``aggregate`` reported ``total_steps - 1``,
    i.e. **2·nx·ny** — double the real point count. A 3×3 grid narrated itself as
    18 points, and a total failure said "All 18 spectra failed" for 9 real
    spectra.

    (— "之前第五个点扫描不到也有可能是下标0或1开始的问题" —
    suspected the point count was wrong. It was. Just not in the 0-vs-1 indexing
    the operator guessed, which is why nobody found it by looking there.)
    """

    def _aggregate(self, nx: int, ny: int, **partial) -> dict:
        skill = GridSTS()
        n_steps = len(skill.plan({
            "center_x_m": 0.0, "center_y_m": 0.0,
            "nx": nx, "ny": ny, "spacing_m": 1e-9,
        }))
        progress = CompositeProgress(
            composite_name="GridSTS",
            total_steps=n_steps,
            partial_data={"nx": nx, "ny": ny, **partial},
        )
        return skill.aggregate({}, progress)

    @pytest.mark.parametrize("nx,ny", [(3, 3), (1, 4), (4, 5), (1, 1)])
    def test_total_points_is_the_number_of_spectra(self, nx, ny):
        data = self._aggregate(nx, ny)
        assert data["total_points"] == nx * ny, (
            f"a {nx}×{ny} grid reports {data['total_points']} points instead of "
            f"{nx * ny} — the plan's move+sts step PAIRS are being counted as "
            f"separate spectra")

    def test_the_failure_message_quotes_the_real_count(self):
        """"All 18 spectra failed" for a 3×3 grid is a lie the operator has no
        way to check."""
        data = self._aggregate(3, 3, succeeded=0, failed=9)
        assert f"All {data['total_points']} spectra failed" == "All 9 spectra failed"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
