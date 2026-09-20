"""GraphExecutor + TrackDrift_ReferenceScan tests (Phase 7 dynamic-plan migration).

Pins:

  1. **First call** (no reference image path) yields SetBias + FullScan,
     captures the just-grabbed image to partial_data, and reports
     ``compensated=False`` with the "Reference captured" message.
  2. **Tracking call** (with reference) yields SetBias + FullScan, runs
     cross-correlation in-process, and yields ``apply_compensation`` only
     when the measured drift is non-trivial.
  3. **No-drift case**: when the live image matches the reference (zero
     shift), the optional compensation step is NOT yielded — only the
     two mandatory scan steps execute.
  4. **Resume**: re-running with SetBias + FullScan already in
     completed_steps replays the analysis from grabbed image data and
     still yields apply_compensation if drift is significant.
  5. Aggregate carries ``drift_x_m`` / ``drift_y_m`` / ``compensated``
     for downstream agents to decide on follow-up actions.

Run:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_drift_track_graph.py -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
import tempfile
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

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.drift_track import TrackDrift_ReferenceScan
from mast.skills.composite.graph_executor import CompositeProgress


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context for drift tracking.

    ``current_image_arr`` is the array returned by Scan_FrameDataGrab.
    """
    current_image_arr: np.ndarray | None = None
    prior_progress: dict | None = None
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    safe_calls: list[tuple[str, tuple]] = field(default_factory=list)
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    run_results: dict[str, SkillResult] = field(default_factory=dict)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name in self.run_results:
            return self.run_results[skill_name]
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.safe_calls.append((method, args))
        if method == "Scan_FrameDataGrab":
            if self.current_image_arr is None:
                return NanonisCallRecord(
                    method=method, args=args, return_value=None,
                )
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", self.current_image_arr.ravel().tolist()),
            )
        return NanonisCallRecord(method=method, args=args)

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(
            CompositeProgress.from_dict(progress.to_dict())
        )

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Helpers ─────────────────────────────────────────────────────────────


def _params(**overrides: Any) -> dict:
    base = dict(
        ref_x_m=10e-9,
        ref_y_m=20e-9,
        ref_width_m=20e-9,
        bias_v=-0.5,
        ref_image_path="",
    )
    base.update(overrides)
    return base


# ── Tests ────────────────────────────────────────────────────────────────


def test_metadata_version_bumped_to_v2():
    md = TrackDrift_ReferenceScan().metadata()
    assert md.version.startswith("2."), (
        f"expected v2.x version, got {md.version}"
    )


def test_first_call_yields_setbias_and_fullscan_only():
    """Empty ref_image_path → no compensation step yielded."""
    img = np.random.RandomState(42).rand(16, 16)
    ctx = FakeCtx(current_image_arr=img)
    skill = TrackDrift_ReferenceScan()
    result = skill.execute(ctx, _params())

    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names == ["SetBias", "FullScan"]
    assert result.success
    # Reference-capture path is success with compensated=False.
    assert result.data["compensated"] is False
    assert "Reference captured" in result.data["message"]
    assert result.data["drift_x_m"] == 0.0
    assert result.data["drift_y_m"] == 0.0


def test_zero_drift_skips_compensation():
    """Identical ref + current images → drift ≈ 0 → no apply_compensation
    step is yielded.

    Use an odd-shape image so correlate2d(mode="same") returns a peak
    at the exact center (8,8) for the autocorrelation case; with an
    even-shape image scipy's center is between samples, producing a
    1-pixel artifact even for identical inputs.
    """
    img = np.zeros((17, 17))
    img[8, 8] = 1.0   # single bright pixel exactly at center
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.npy"
        np.save(ref_path, img)

        ctx = FakeCtx(current_image_arr=img.copy())
        skill = TrackDrift_ReferenceScan()
        result = skill.execute(
            ctx, _params(ref_image_path=str(ref_path)),
        )

    skill_names = [name for name, _ in ctx.run_log]
    # SetBias + FullScan only — compensation skipped (zero drift).
    assert skill_names == ["SetBias", "FullScan"]
    assert result.success
    assert result.data["compensated"] is False
    # Drift effectively zero.
    assert abs(result.data["drift_x_m"]) < 1e-15
    assert abs(result.data["drift_y_m"]) < 1e-15


def test_nonzero_drift_yields_compensation():
    """Shifted live image → drift detected → apply_compensation step yielded.

    Constructs a clearly-shifted pair so the xcorr peak lands away from
    center.
    """
    ref = np.zeros((17, 17))
    ref[8, 8] = 1.0   # peak at row=8 col=8
    cur = np.zeros((17, 17))
    cur[8, 10] = 1.0   # peak shifted right by 2 columns

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.npy"
        np.save(ref_path, ref)

        ctx = FakeCtx(current_image_arr=cur)
        skill = TrackDrift_ReferenceScan()
        result = skill.execute(
            ctx, _params(ref_image_path=str(ref_path), ref_width_m=20e-9),
        )

    skill_names = [name for name, _ in ctx.run_log]
    # Three sub-skills: SetBias + FullScan + ConfigureScan (compensation).
    assert skill_names == ["SetBias", "FullScan", "ConfigureScan"]
    assert result.success
    assert result.data["compensated"] is True
    # drift_x_m should be non-zero (the shift is along columns → x-axis).
    assert abs(result.data["drift_x_m"]) > 0


def test_missing_reference_file_aborts():
    """Non-existent ref_image_path → composite aborts with failure."""
    ctx = FakeCtx(current_image_arr=np.zeros((8, 8)))
    skill = TrackDrift_ReferenceScan()
    result = skill.execute(
        ctx, _params(ref_image_path="/nonexistent/ref.npy"),
    )

    assert not result.success
    assert "Failed to load reference" in (result.error or "")
    # SetBias + FullScan ran, ConfigureScan did NOT.
    skill_names = [name for name, _ in ctx.run_log]
    assert "ConfigureScan" not in skill_names


def test_compensation_step_failure_doesnt_abort():
    """A failed ConfigureScan (optional step) leaves compensated=False but
    the composite still succeeds with drift reported."""
    ref = np.zeros((17, 17)); ref[8, 8] = 1.0
    cur = np.zeros((17, 17)); cur[8, 10] = 1.0

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.npy"
        np.save(ref_path, ref)

        ctx = FakeCtx(
            current_image_arr=cur,
            run_results={
                "ConfigureScan": SkillResult(
                    skill_name="ConfigureScan", success=False,
                    error="hardware busy",
                ),
            },
        )
        skill = TrackDrift_ReferenceScan()
        result = skill.execute(
            ctx, _params(ref_image_path=str(ref_path)),
        )

    # ConfigureScan was attempted but the optional flag prevented abort.
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names == ["SetBias", "FullScan", "ConfigureScan"]
    # Drift was measured; compensated reports False because the apply failed.
    assert result.success
    assert result.data["compensated"] is False


def test_resume_skips_completed_setbias_and_scan():
    """Prior progress with set_bias + ref_scan done → only apply_compensation
    runs fresh (if drift is non-zero)."""
    ref = np.zeros((17, 17)); ref[8, 8] = 1.0
    cur = np.zeros((17, 17)); cur[8, 10] = 1.0

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.npy"
        np.save(ref_path, ref)

        prior = CompositeProgress(
            composite_name="TrackDrift_ReferenceScan",
            total_steps=3,
            completed_steps=["set_bias", "ref_scan"],
            partial_data={},
        )
        ctx = FakeCtx(
            current_image_arr=cur,
            prior_progress=prior.to_dict(),
        )
        skill = TrackDrift_ReferenceScan()
        result = skill.execute(
            ctx, _params(ref_image_path=str(ref_path)),
        )

    skill_names = [name for name, _ in ctx.run_log]
    assert "SetBias" not in skill_names
    assert "FullScan" not in skill_names
    assert skill_names == ["ConfigureScan"]
    assert result.success
    assert result.data["compensated"] is True


def test_progress_snapshot_lifted_into_result():
    img = np.random.RandomState(0).rand(16, 16)
    ctx = FakeCtx(current_image_arr=img)
    skill = TrackDrift_ReferenceScan()
    result = skill.execute(ctx, _params())

    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "TrackDrift_ReferenceScan"
    # Two mandatory steps completed (reference-capture branch).
    assert "set_bias" in snap["completed_steps"]
    assert "ref_scan" in snap["completed_steps"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
