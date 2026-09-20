"""GraphExecutor + SurveySurface_TileScan regression tests (Phase 7 framework).

Pins the following contracts:

  1. ``plan(params)`` is a list of :class:`CompositeStep` whose length
     equals ``n*n * (4 + assess?)`` (per tile: configure, speed, start,
     wait, optional assess).
  2. Running an empty-context executor walks every step exactly once and
     emits a :class:`CompositeProgress` snapshot whose ``completed_steps``
     length equals the step count.
  3. **Resume**: re-running with prior progress covering one full tile's
     steps causes the executor to skip them.
  4. AssessImageQuality returns ``fft_quality``; the recommended tile
     comes from the highest quality.
  5. Pre-flight grid validation still fires (tile_size_m >= total_size_m
     fails fast, > 8x8 grid rejected).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_survey_surface_graph.py -x -v
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
from mast.skills.composite.survey_surface import SurveySurface_TileScan


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context — supports run() + the optional progress hooks."""
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    # Quality value cycle: each AssessImageQuality call returns the next
    # value in this list (used to assert "best tile" selection).
    quality_cycle: list[float] = field(default_factory=list)
    _quality_idx: int = 0
    # If True, WaitScanComplete reports timed_out=True
    wait_timed_out: bool = False
    # If set, WaitScanComplete reports a tile stopped part-way (v6.1.3)
    wait_lines_done: int | None = None
    wait_lines_total: int = 256

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "AssessImageQuality":
            if self.quality_cycle:
                q = self.quality_cycle[
                    self._quality_idx % len(self.quality_cycle)
                ]
                self._quality_idx += 1
            else:
                q = 0.5
            return SkillResult(
                skill_name=skill_name,
                success=True,
                data={"fft_quality": q, "label": "ok", "snr_db": 10.0},
            )
        if skill_name == "WaitScanComplete":
            done = (self.wait_lines_total if self.wait_lines_done is None
                    else self.wait_lines_done)
            stopped_early = done < self.wait_lines_total
            return SkillResult(
                skill_name=skill_name,
                success=True,
                data={"timed_out": self.wait_timed_out, "polls": 1,
                      "stopped_early": stopped_early,
                      "outcome": ("timed_out" if self.wait_timed_out
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


def test_plan_step_count_2x2_with_assess():
    """2×2 grid with assess_quality → 4 tiles × 5 steps = 20 steps."""
    skill = SurveySurface_TileScan()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": True,
    })
    assert len(plan) == 2 * 2 * 5
    # Verify step IDs are unique
    assert len({s.step_id for s in plan}) == len(plan)
    # First tile's first step should be the ConfigureScan
    assert plan[0].step_id == "tile_0_0:configure"
    assert plan[0].skill_name == "ConfigureScan"


def test_plan_step_count_no_assess():
    """2×2 grid without assess_quality → 4 tiles × 4 steps = 16 steps."""
    skill = SurveySurface_TileScan()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": False,
    })
    assert len(plan) == 2 * 2 * 4
    # No assess step anywhere
    assert all("assess" not in s.step_id for s in plan)


def test_plan_step_count_4x4_default():
    """Defaults: 200nm / 50nm → 4x4 = 16 tiles → 16 * 5 = 80 steps."""
    skill = SurveySurface_TileScan()
    plan = skill.plan({})  # all defaults
    assert len(plan) == 4 * 4 * 5


def test_plan_phase_ordering_in_each_tile():
    """Each tile's 5 steps come in (configure, speed, start, wait, assess) order."""
    skill = SurveySurface_TileScan()
    plan = skill.plan({
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": True,
    })
    phases = []
    for s in plan:
        _, _, phase = s.step_id.partition(":")
        phases.append(phase)
    # First tile's phases
    assert phases[0:5] == ["configure", "speed", "start", "wait", "assess"]


def test_full_execution_walks_every_step():
    """Fresh context (no resume) → every step invoked exactly once."""
    skill = SurveySurface_TileScan()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": True,
    })
    # 2x2 = 4 tiles × 5 steps each = 20 invocations
    assert len(ctx.run_log) == 20
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names.count("ConfigureScan") == 4
    assert skill_names.count("SetScanSpeed") == 4
    assert skill_names.count("StartScan") == 4
    assert skill_names.count("WaitScanComplete") == 4
    assert skill_names.count("AssessImageQuality") == 4
    assert result.success
    assert result.data["grid_n"] == 2
    assert result.data["tile_count"] == 4
    assert result.data["success_count"] == 4
    assert result.data["fail_count"] == 0


def test_recommended_tile_selects_highest_quality():
    """Best tile = highest fft_quality value seen."""
    skill = SurveySurface_TileScan()
    # 4 tiles → 4 assess calls; #2 (index=1) has the max quality
    ctx = FakeCtx(quality_cycle=[0.1, 0.9, 0.3, 0.5])
    result = skill.execute(ctx, {
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": True,
    })
    assert result.success
    best = result.data["recommended_tile"]
    assert best is not None
    assert best["quality"] == pytest.approx(0.9)


def test_progress_snapshot_carries_into_result():
    """`_progress` snapshot is lifted into result.data."""
    skill = SurveySurface_TileScan()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": False,
    })
    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "SurveySurface_TileScan"
    # 2x2 = 4 tiles × 4 phases (no assess) = 16 completed steps
    assert len(snap["completed_steps"]) == 16
    assert snap["aborted"] is False


def test_resume_skips_completed_steps():
    """Prior progress covering one full tile (5 steps) → remaining run fresh."""
    skill = SurveySurface_TileScan()
    plan = skill.plan({
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": True,
    })
    # Mark first tile's 5 steps as done.
    completed = [s.step_id for s in plan[:5]]
    prior = CompositeProgress(
        composite_name="SurveySurface_TileScan",
        total_steps=20,
        completed_steps=list(completed),
        partial_data={
            "grid_n": 2,
            "tile_size_m": 50e-9,
            "total_size_m": 100e-9,
            "assess_quality": True,
            # Pre-populate tile_0_0 record so aggregate can render it
            "tile_records": {
                "0_0": {
                    "index": 1, "row": 0, "col": 0, "success": True,
                    "quality": 0.7,
                },
            },
        },
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": True,
    })
    # 15 NEW invocations (20 plan steps - 5 prior)
    assert len(ctx.run_log) == 15
    snap = result.data["_progress"]
    # All 20 now completed (5 resumed + 15 fresh)
    assert len(snap["completed_steps"]) == 20
    # All 4 tiles successful in aggregate
    assert result.data["success_count"] == 4


def test_progress_emitted_every_step():
    """emit_progress fires after each successful step + once on finish."""
    skill = SurveySurface_TileScan()
    ctx = FakeCtx()
    skill.execute(ctx, {
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": False,
    })
    # 2x2 × 4 steps = 16 completed steps → ≥ 17 emits (per-step + final)
    assert len(ctx.emitted) >= 17
    # Final emit has current_step=None
    assert ctx.emitted[-1].current_step is None


def test_grid_validation_tile_too_large():
    """tile_size_m >= total_size_m → fail before any sub-skill runs."""
    skill = SurveySurface_TileScan()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "total_size_m": 50e-9, "tile_size_m": 50e-9,
    })
    assert not result.success
    assert "tile_size_m" in result.error or "FullScan" in result.error
    # No sub-skills should have been invoked
    assert ctx.run_log == []


def test_grid_validation_too_many_tiles():
    """grid > 8x8 → fail with 'too many' message."""
    skill = SurveySurface_TileScan()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "total_size_m": 1e-6, "tile_size_m": 50e-9,  # 20x20 = 400 tiles
    })
    assert not result.success
    assert "too many" in result.error.lower() or "8x8" in result.error
    assert ctx.run_log == []


def test_wait_timed_out_marks_tile_failed():
    """If WaitScanComplete reports timed_out=True, the tile is recorded
    as failed but the survey continues (partial success)."""
    skill = SurveySurface_TileScan()
    ctx = FakeCtx(wait_timed_out=True)
    result = skill.execute(ctx, {
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": False,
    })
    # All 4 tiles "succeeded" at the skill level (executor saw no
    # failures), but on_step_result marks them failed when timed_out.
    assert result.success  # partial-success semantics
    assert result.data["fail_count"] == 4
    assert result.data["success_count"] == 0


def test_a_tile_stopped_part_way_is_marked_failed_but_the_survey_continues():
    """v6.1.3 (KNOWN_ISSUES §2.24). Same policy as the timeout above, and that
    is the point: this is NOT a new decision, it follows the one already made
    at this call site. A survey exists to map many tiles — one truncated tile
    says nothing about the next, and aborting the grid would throw away the
    tiles that did scan."""
    result = SurveySurface_TileScan().execute(
        FakeCtx(wait_lines_done=40, wait_lines_total=256), {
            "total_size_m": 100e-9, "tile_size_m": 50e-9,
            "assess_quality": False,
        })
    assert result.success                      # partial-success semantics
    assert result.data["fail_count"] == 4
    assert result.data["success_count"] == 0


def test_a_truncated_tile_says_stopped_not_timeout():
    """Raising the timeout fixes one and does nothing for the other, so the
    per-tile error has to tell them apart."""
    result = SurveySurface_TileScan().execute(
        FakeCtx(wait_lines_done=40, wait_lines_total=256), {
            "total_size_m": 100e-9, "tile_size_m": 50e-9,
            "assess_quality": False,
        })
    errors = [t.get("error", "") for t in result.data["tiles"]]
    assert all("stopped early" in e for e in errors), errors
    assert not any("timeout" in e for e in errors), errors
    assert all("40/256" in e for e in errors), errors


def test_complete_tiles_still_pass():
    """Control for both tests above."""
    result = SurveySurface_TileScan().execute(FakeCtx(), {
        "total_size_m": 100e-9, "tile_size_m": 50e-9, "assess_quality": False,
    })
    assert result.data["success_count"] == 4
    assert result.data["fail_count"] == 0


def test_tile_centers_match_grid_geometry():
    """Tile centers form the expected raster grid."""
    skill = SurveySurface_TileScan()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "total_size_m": 100e-9, "tile_size_m": 50e-9,
        "assess_quality": False,
    })
    tiles = result.data["tiles"]
    assert len(tiles) == 4
    # 2x2 grid: x0/y0 = -25e-9, then +50e-9 step
    centers = [(t["row"], t["col"], t["center_x_m"], t["center_y_m"]) for t in tiles]
    # Row 0 = bottom, row 1 = top
    assert centers[0] == pytest.approx((0, 0, -25e-9, -25e-9))
    assert centers[1] == pytest.approx((0, 1, 25e-9, -25e-9))
    assert centers[2] == pytest.approx((1, 0, -25e-9, 25e-9))
    assert centers[3] == pytest.approx((1, 1, 25e-9, 25e-9))


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
