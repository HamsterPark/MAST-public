"""GraphExecutor + TipPulse regression tests (Phase 7 framework).

Pins the following contracts:

  1. ``plan(params)`` is a list of :class:`CompositeStep` whose length
     equals ``1 + count`` (one ``snapshot_bias`` + N ``pulse_<i>``).
  2. Running an empty-context executor walks every step exactly once and
     emits a :class:`CompositeProgress` snapshot whose ``completed_steps``
     length equals the step count.
  3. **Resume**: re-running with a prior ``composite_progress`` entry
     covering half the plan causes the executor to skip those steps.
  4. ``original_bias_v`` from the GetBias snapshot survives into the
     final SkillResult.data via :meth:`on_step_result`.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_tip_pulse_graph.py -x -v
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
from mast.skills.composite.tip_pulse import TipPulse


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context — supports run() + the optional progress hooks."""
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    # GetBias sub-skill returns this bias value
    canned_bias_v: float = -0.5

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "GetBias":
            return SkillResult(
                skill_name=skill_name,
                success=True,
                data={"bias_v": self.canned_bias_v},
            )
        # All other sub-skills (BiasPulse) succeed
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


def test_plan_step_count_default_count_1():
    """count=1 → 1 snapshot + 1 pulse = 2 steps."""
    skill = TipPulse()
    plan = skill.plan({"pulse_v": 3.0, "duration_s": 0.1, "count": 1})
    assert len(plan) == 2
    assert plan[0].step_id == "snapshot_bias"
    assert plan[0].skill_name == "GetBias"
    assert plan[1].step_id == "pulse_1"
    assert plan[1].skill_name == "BiasPulse"


def test_plan_step_count_count_5():
    """count=5 → 1 snapshot + 5 pulses = 6 steps."""
    skill = TipPulse()
    plan = skill.plan({"pulse_v": 2.0, "duration_s": 0.05, "count": 5})
    assert len(plan) == 6
    # All step_ids unique
    assert len({s.step_id for s in plan}) == 6
    # Last pulse is the one that checkpoints
    assert plan[-1].step_id == "pulse_5"
    assert plan[-1].checkpoint_after is True
    # Intermediate pulses don't checkpoint (tight inner loop)
    assert plan[1].checkpoint_after is False
    assert plan[2].checkpoint_after is False


def test_plan_pulse_step_carries_correct_params():
    """Each BiasPulse step must carry width_s=duration_s, bias_v=pulse_v."""
    skill = TipPulse()
    plan = skill.plan({"pulse_v": 4.5, "duration_s": 0.2, "count": 2})
    pulses = [s for s in plan if s.step_id.startswith("pulse_")]
    assert len(pulses) == 2
    for step in pulses:
        assert step.skill_name == "BiasPulse"
        assert step.params["width_s"] == 0.2
        assert step.params["bias_v"] == 4.5
        assert step.params["absolute"] is True
        # z_hold=1 (hold Z during the pulse), NOT 0 ("no change").
        #
        # This assertion used to require 0, pinning a real defect in place: with
        # the Z controller left running, a several-volt pulse drives the tunnel
        # current up by orders of magnitude and the feedback loop chases it —
        # straight into the surface. BiasPulse's own default is 1 for that
        # reason; TipPulse was overriding it to 0 (2026-08-01 fix).
        assert step.params["z_hold"] == 1


def test_full_execution_walks_every_step():
    """count=3 → 1 GetBias + 3 BiasPulse = 4 run() invocations."""
    skill = TipPulse()
    ctx = FakeCtx(canned_bias_v=-1.2)
    result = skill.execute(ctx, {"pulse_v": 3.0, "duration_s": 0.1, "count": 3})
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names == ["GetBias", "BiasPulse", "BiasPulse", "BiasPulse"]
    assert result.success
    # original_bias_v threaded from GetBias via on_step_result
    assert result.data["original_bias_v"] == -1.2
    assert result.data["pulse_v"] == 3.0
    assert result.data["duration_s"] == 0.1
    assert result.data["count"] == 3


def test_progress_snapshot_carries_into_result():
    """`_progress` snapshot is lifted into result.data for the adapter."""
    skill = TipPulse()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"pulse_v": 2.5, "count": 2})
    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "TipPulse"
    # 1 snapshot + 2 pulses = 3 completed_steps
    assert len(snap["completed_steps"]) == 3
    assert snap["aborted"] is False


def test_resume_skips_completed_steps():
    """Prior progress covering snapshot+pulse_1 → only pulse_2 runs fresh."""
    skill = TipPulse()
    plan = skill.plan({"pulse_v": 3.0, "duration_s": 0.1, "count": 2})
    completed = [plan[0].step_id, plan[1].step_id]  # snapshot_bias, pulse_1
    prior = CompositeProgress(
        composite_name="TipPulse",
        total_steps=3,
        completed_steps=list(completed),
        partial_data={
            "pulse_v": 3.0,
            "duration_s": 0.1,
            "count": 2,
            "original_bias_v": -0.7,
        },
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {"pulse_v": 3.0, "duration_s": 0.1, "count": 2})
    # Only pulse_2 should have triggered a fresh sub-skill invocation
    assert len(ctx.run_log) == 1
    assert ctx.run_log[0][0] == "BiasPulse"
    # original_bias_v preserved across resume (came from prior partial_data)
    assert result.data["original_bias_v"] == -0.7
    assert result.success


def test_progress_emitted_every_step():
    """emit_progress fires after each successful step + once on finish."""
    skill = TipPulse()
    ctx = FakeCtx()
    skill.execute(ctx, {"pulse_v": 3.0, "count": 1})
    # 1 GetBias + 1 BiasPulse = 2 steps → ≥ 3 emits (per-step + final)
    assert len(ctx.emitted) >= 3
    # Final emit has current_step=None
    assert ctx.emitted[-1].current_step is None


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
