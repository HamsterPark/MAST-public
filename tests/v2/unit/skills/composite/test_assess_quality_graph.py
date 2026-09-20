"""GraphExecutor + AssessImageQuality regression tests (Phase 7 framework).

Pins the following contracts:

  1. ``plan(params)`` returns 6 :class:`CompositeStep` items:
     load -> fft -> roughness -> noise -> ssim (optional) -> training_sample
     (optional). All step_ids are unique. The first four are mandatory; the
     last two are optional.
  2. Running an empty-context executor walks every step exactly once and
     emits a :class:`CompositeProgress` snapshot whose ``completed_steps``
     length covers all mandatory phases (with optional phases either
     completed or recorded in ``failed_steps``).
  3. **Resume**: re-running with a prior ``composite_progress`` covering
     the first 3 phases causes the executor to skip them — only noise +
     optionals run.
  4. emit_progress fires after every successful step + once on finish.
  5. The phase wrapper (_PhaseCtx) routes ``_phase_*`` skill_names to
     internal compute methods so the executor never has to know about
     the synthetic phase identifiers.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/composite/test_assess_quality_graph.py -x -v
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

import numpy as np
import pytest

from mast.core.types import SkillResult
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
)
from mast.skills.composite.assess_quality import (
    AssessImageQuality,
    _PHASE_FFT,
    _PHASE_LOAD,
    _PHASE_NOISE,
    _PHASE_ROUGHNESS,
    _PHASE_SSIM,
    _PHASE_TRAINING,
)


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context — passes through to the synthetic phase dispatch."""
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0

    def run(self, skill_name: str, params: dict) -> SkillResult:
        # This is only ever called via the _PhaseCtx wrapper, but the wrapper
        # only intercepts _phase_* names. Any *registered* skill (none in this
        # test) would fall through here. Record any such accidental fallthrough.
        self.run_log.append((skill_name, dict(params)))
        return SkillResult(skill_name=skill_name, success=True, data={})

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(
            CompositeProgress.from_dict(progress.to_dict())
        )

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Helpers ───────────────────────────────────────────────────────────────


def _patch_io(monkeypatch, image_shape=(32, 32)):
    """Stub out the file/scan helpers so we can run without real .sxm.

    Returns a deterministic float64 array so quality metrics stay stable.
    """
    img = np.linspace(0.0, 1.0, num=image_shape[0] * image_shape[1])
    img = img.reshape(image_shape).astype(np.float64)

    def _fake_load(path):  # noqa: ANN001
        return img

    def _fake_find(context):  # noqa: ANN001
        return "/tmp/fake_scan.sxm"

    monkeypatch.setattr(
        AssessImageQuality, "_load_sxm_image", staticmethod(_fake_load),
    )
    monkeypatch.setattr(
        AssessImageQuality, "_find_latest_scan", staticmethod(_fake_find),
    )
    # F2 (2026-06-08): _phase_load now existence-checks the scan path via the
    # stubbable _scan_file_exists seam before loading. The fake path doesn't
    # exist on disk, so stub existence True alongside the stubbed loader.
    monkeypatch.setattr(
        AssessImageQuality, "_scan_file_exists", staticmethod(lambda path: True),
    )
    # SSIM helper returns None → optional ssim phase fails gracefully
    monkeypatch.setattr(
        AssessImageQuality,
        "_compute_fwd_bwd_ssim",
        staticmethod(lambda path: None),
    )
    # Training-sample helper is a no-op (no planner attached)
    monkeypatch.setattr(
        AssessImageQuality,
        "_record_training_sample",
        staticmethod(lambda ctx, image, fft_score: None),
    )
    return img


# ── Tests ────────────────────────────────────────────────────────────────


def test_plan_step_count_is_6():
    """6 phases: load / fft / roughness / noise / ssim / training_sample."""
    skill = AssessImageQuality()
    plan = skill.plan({"scan_path": "/tmp/x.sxm"})
    assert len(plan) == 6
    step_ids = [s.step_id for s in plan]
    assert step_ids == [
        "load", "fft", "roughness", "noise", "ssim", "training_sample",
    ]
    skill_names = [s.skill_name for s in plan]
    assert skill_names == [
        _PHASE_LOAD, _PHASE_FFT, _PHASE_ROUGHNESS,
        _PHASE_NOISE, _PHASE_SSIM, _PHASE_TRAINING,
    ]
    # All ids unique
    assert len({s.step_id for s in plan}) == 6


def test_plan_mandatory_vs_optional():
    """First 4 phases are mandatory; ssim + training_sample are optional."""
    skill = AssessImageQuality()
    plan = skill.plan({"scan_path": ""})
    by_id = {s.step_id: s for s in plan}
    assert by_id["load"].optional is False
    assert by_id["fft"].optional is False
    assert by_id["roughness"].optional is False
    assert by_id["noise"].optional is False
    assert by_id["ssim"].optional is True
    assert by_id["training_sample"].optional is True


def test_plan_load_step_carries_scan_path():
    """load step forwards user-provided scan_path."""
    skill = AssessImageQuality()
    plan = skill.plan({"scan_path": "/data/foo.sxm"})
    load = next(s for s in plan if s.step_id == "load")
    assert load.params["scan_path"] == "/data/foo.sxm"


def test_full_execution_with_stubbed_io(monkeypatch):
    """All mandatory phases succeed → final result carries FFT/RMS/noise."""
    _patch_io(monkeypatch)
    skill = AssessImageQuality()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"scan_path": ""})
    assert result.success, f"expected success, got error: {result.error}"
    assert "fft_quality" in result.data
    assert "rms_roughness_m" in result.data
    assert "noise_sigma" in result.data
    assert result.data["scan_path"] == "/tmp/fake_scan.sxm"
    # ssim is optional and our stub returns None → key NOT present
    assert "fwd_bwd_ssim" not in result.data
    # Synthetic phase names are intercepted by _PhaseCtx — none should fall
    # through to the underlying context.run
    assert ctx.run_log == []


def test_progress_snapshot_carries_into_result(monkeypatch):
    """`_progress` snapshot is lifted into result.data for the adapter."""
    _patch_io(monkeypatch)
    skill = AssessImageQuality()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"scan_path": ""})
    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "AssessImageQuality"
    # 4 mandatory phases done; ssim + training_sample are recorded as either
    # completed (training_sample stub succeeds) or failed (ssim returns None
    # → its phase returns SkillResult(success=False) → recorded in
    # failed_steps because step.optional=True).
    completed = set(snap["completed_steps"])
    assert {"load", "fft", "roughness", "noise"}.issubset(completed)
    # ssim phase fails optional (stub returns None) → in failed_steps
    assert "ssim" in snap["failed_steps"]
    # training_sample stub returns success → completed
    assert "training_sample" in completed
    assert snap["aborted"] is False


def test_progress_emitted_every_step(monkeypatch):
    """emit_progress fires after each successful/failed step + once on finish."""
    _patch_io(monkeypatch)
    skill = AssessImageQuality()
    ctx = FakeCtx()
    skill.execute(ctx, {"scan_path": ""})
    # 6 steps → >= 6 emits (per-step success/fail + final summary)
    assert len(ctx.emitted) >= 6
    # Final emit clears current_step
    assert ctx.emitted[-1].current_step is None


def test_resume_skips_completed_phases(monkeypatch):
    """Prior progress covers first 3 phases → only noise + optionals run."""
    _patch_io(monkeypatch)
    skill = AssessImageQuality()
    plan = skill.plan({"scan_path": ""})
    completed = [s.step_id for s in plan[:3]]  # load, fft, roughness
    # Seed prior partial_data so aggregate() can still surface FFT/RMS from
    # the resumed run (the actual compute won't re-run for these phases).
    prior = CompositeProgress(
        composite_name="AssessImageQuality",
        total_steps=6,
        completed_steps=list(completed),
        partial_data={
            "scan_path": "/tmp/fake_scan.sxm",
            "fft_quality": 0.42,
            "rms_roughness_m": 1e-10,
        },
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {"scan_path": ""})
    snap = result.data["_progress"]
    # All 6 phases accounted for (4 completed + ssim failed-optional +
    # training completed)
    accounted = set(snap["completed_steps"]) | set(snap["failed_steps"])
    assert accounted == {
        "load", "fft", "roughness", "noise",
        "ssim", "training_sample",
    }
    assert result.success
    # The resumed FFT score survives in the final data
    assert result.data["fft_quality"] == 0.42


def test_checkpoint_flushes_after_noise(monkeypatch):
    """Only the noise step (and optionally training_sample) trigger flush.

    The plan marks ``noise`` with checkpoint_after=True (core metrics done).
    All other steps have checkpoint_after=False so they should NOT flush.
    """
    _patch_io(monkeypatch)
    skill = AssessImageQuality()
    ctx = FakeCtx()
    skill.execute(ctx, {"scan_path": ""})
    # Exactly 1 flush from `noise`. (ssim/training_sample have checkpoint_after=False)
    assert ctx.flushes == 1


def test_missing_scan_aborts(monkeypatch):
    """If load can't find a scan and none is provided → mandatory load fails."""
    # _find_latest_scan returns None
    monkeypatch.setattr(
        AssessImageQuality, "_find_latest_scan",
        staticmethod(lambda ctx: None),
    )
    skill = AssessImageQuality()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"scan_path": ""})
    assert not result.success
    # Either explicit "No scan file" or executor's abort reason
    err = (result.error or "").lower()
    assert "no scan" in err or "aborted" in err


def test_make_tool_wraps_to_v2_adapter():
    """make_tool() should produce a LangChain StructuredTool via wrap_skill."""
    from mast.skills.composite.assess_quality import make_tool

    def _provider():
        return None

    tool = make_tool(_provider)
    assert tool.name == "AssessImageQuality"
    fields = tool.args_schema.model_fields
    assert "scan_path" in fields
    # scan_path is optional with default
    assert not fields["scan_path"].is_required()


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
