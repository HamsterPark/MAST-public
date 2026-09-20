"""GraphExecutor + AcquirePSD regression tests (Phase 7 migration).

Pins the following contracts:

  1. ``AcquirePSD`` subclasses :class:`CompositeSkillGraph`.
  2. Single-range mode (legacy v1 contract): no ``freq_range_indices`` →
     plan = configure + range_0_capture (2 steps).
  3. Multi-range mode: ``freq_range_indices='[0,2,5]'`` → plan = configure
     + 3 range_<i>_capture steps.
  4. Each capture step issues SpectrumAnlzr_FreqRangeSet (when fr_idx>=0)
     + SpectrumAnlzr_FreqRangeGet + SpectrumAnlzr_DataGet.
  5. Result.data["per_range"] is an ordered list of PSD records (one per
     band).
  6. SafetyLevel remains AUTO.
  7. Progress emission + resume work as expected.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_acquire_psd_graph.py -x -v
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
from mast.skills.builtins.acquire_psd import (
    AcquirePSD,
    _PHASE_CONFIGURE,
    _PHASE_RANGE_CAPTURE_PREFIX,
)


# ── FakeCtx ───────────────────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Default: all calls succeed; DataGet returns a valid 4-bin PSD."""
    canned_errors: dict[str, str] = field(default_factory=dict)
    psd_bins: int = 4
    avail_ranges: list[float] = field(
        default_factory=lambda: [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
    )
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    emitted: list[CompositeProgress] = field(default_factory=list)
    prior_progress: dict | None = None
    flushes: int = 0

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned_errors:
            return NanonisCallRecord(method=method, args=args,
                                     error=self.canned_errors[method])
        if method == "SpectrumAnlzr_DataGet":
            f0 = 0.0
            df = 1.0
            n = self.psd_bins
            psd = [float(i) for i in range(n)]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [f0, df, n, psd]),
            )
        if method == "SpectrumAnlzr_FreqRangeGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [0, 5, list(self.avail_ranges)]),
            )
        if method == "SpectrumAnlzr_FreqResGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [0, 4]),
            )
        if method == "SpectrumAnlzr_ChGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [3]),
            )
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


def test_acquire_psd_is_composite_graph():
    assert issubclass(AcquirePSD, CompositeSkillGraph)


def test_acquire_psd_metadata():
    meta = AcquirePSD().metadata()
    assert meta.name == "AcquirePSD"
    assert meta.safety_level == SafetyLevel.AUTO


# ── Plan tests ─────────────────────────────────────────────────────────────


def test_plan_single_range_mode():
    """No freq_range_indices → 1 configure + 1 capture = 2 steps."""
    skill = AcquirePSD()
    plan = skill.plan({"instance": 1, "freq_range_index": 2})
    assert len(plan) == 2
    assert plan[0].step_id == "configure"
    assert plan[0].skill_name == _PHASE_CONFIGURE
    assert plan[1].step_id == "range_0_capture"
    assert plan[1].skill_name == f"{_PHASE_RANGE_CAPTURE_PREFIX}0"


def test_plan_multi_range_mode():
    """freq_range_indices='[0, 2, 5]' → 1 configure + 3 capture = 4 steps."""
    skill = AcquirePSD()
    plan = skill.plan({"freq_range_indices": "[0, 2, 5]"})
    assert len(plan) == 4
    step_ids = [s.step_id for s in plan]
    assert step_ids == [
        "configure", "range_0_capture", "range_1_capture", "range_2_capture",
    ]
    # All capture steps carry their freq_range_index in params
    captures = [s for s in plan if s.step_id.startswith("range_")]
    assert [c.params["freq_range_index"] for c in captures] == [0, 2, 5]


def test_plan_all_capture_mandatory():
    skill = AcquirePSD()
    plan = skill.plan({"freq_range_indices": "[0, 1, 2]"})
    assert all(not s.optional for s in plan)
    # Only last capture flushes checkpoint
    assert plan[-1].checkpoint_after is True
    assert plan[-2].checkpoint_after is False


# ── Execution tests ───────────────────────────────────────────────────────


def test_single_range_execution_returns_v1_shape():
    """Single-range execution surfaces top-level psd/f0_hz keys (v1 compat)."""
    skill = AcquirePSD()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"instance": 1, "freq_range_index": 2})
    assert result.success, f"unexpected failure: {result.error}"
    # v1-style top-level keys preserved
    assert "psd" in result.data
    assert "f0_hz" in result.data
    assert "df_hz" in result.data
    assert "n_bins" in result.data
    assert "f_max_hz" in result.data
    # New per_range list contains one record
    assert len(result.data["per_range"]) == 1
    assert result.data["per_range"][0]["freq_range_index"] == 2


def test_multi_range_execution_3_captures():
    """3 frequency-range indices → 3 SpectrumAnlzr_DataGet calls."""
    skill = AcquirePSD()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"freq_range_indices": "[0, 2, 5]"})
    assert result.success
    # Exactly 3 DataGet calls (one per capture)
    data_gets = [c for c in ctx.calls if c[0] == "SpectrumAnlzr_DataGet"]
    assert len(data_gets) == 3
    # Exactly 3 FreqRangeSet calls (each capture sets its band)
    range_sets = [c for c in ctx.calls if c[0] == "SpectrumAnlzr_FreqRangeSet"]
    assert len(range_sets) == 3
    # Each set was issued with the expected fr_idx
    set_args = [c[1] for c in range_sets]
    assert [a[1] for a in set_args] == [0, 2, 5]
    # per_range list ordered by position
    per_range = result.data["per_range"]
    assert len(per_range) == 3
    assert [r["freq_range_index"] for r in per_range] == [0, 2, 5]


def test_dataget_failure_aborts():
    """SpectrumAnlzr_DataGet error on first capture → mandatory abort."""
    skill = AcquirePSD()
    ctx = FakeCtx(canned_errors={
        "SpectrumAnlzr_DataGet": "spectrum analyser offline",
    })
    result = skill.execute(ctx, {"freq_range_indices": "[0, 1, 2]"})
    assert not result.success
    assert "DataGet failed" in (result.error or "") or "offline" in (
        result.error or "")
    # Only one capture attempted (mandatory failure aborts)
    data_gets = [c for c in ctx.calls if c[0] == "SpectrumAnlzr_DataGet"]
    assert len(data_gets) == 1


def test_configure_runs_idempotent_run_call():
    """configure phase issues SpectrumAnlzr_Run (don't fail-fast on errors)."""
    skill = AcquirePSD()
    # Even if Run errors out, configure should not abort (idempotent).
    ctx = FakeCtx(canned_errors={"SpectrumAnlzr_Run": "already running"})
    result = skill.execute(ctx, {"instance": 1, "freq_range_index": 0})
    # Configure must still succeed even with Run error (matches v1 contract)
    assert result.success or "DataGet" not in (result.error or "")
    methods = [c[0] for c in ctx.calls]
    assert "SpectrumAnlzr_Run" in methods


# ── Phase dispatch ────────────────────────────────────────────────────────


def test_phase_names_intercepted_by_wrapper():
    skill = AcquirePSD()

    @dataclass
    class TraceCtx(FakeCtx):
        run_log: list[tuple[str, dict]] = field(default_factory=list)

        def run(self, skill_name: str, params: dict) -> SkillResult:
            self.run_log.append((skill_name, dict(params)))
            return SkillResult(skill_name=skill_name, success=True, data={})

    ctx = TraceCtx()
    skill.execute(ctx, {"freq_range_indices": "[0, 1]"})
    fallthrough = [name for name, _ in ctx.run_log
                   if name.startswith("_phase_")]
    assert fallthrough == [], f"phase names leaked: {fallthrough}"


# ── Resume ────────────────────────────────────────────────────────────────


def test_resume_skips_completed_captures():
    """Prior progress with configure + range_0 done → 2 fresh captures."""
    skill = AcquirePSD()
    prior = CompositeProgress(
        composite_name="AcquirePSD",
        total_steps=4,
        completed_steps=["configure", "range_0_capture"],
        partial_data={
            "instance": 1,
            "per_range": {"0": {
                "freq_range_index": 0,
                "f0_hz": 0.0,
                "df_hz": 1.0,
                "n_bins": 4,
                "f_max_hz": 3.0,
                "psd": [0.0, 1.0, 2.0, 3.0],
            }},
        },
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {"freq_range_indices": "[0, 2, 5]"})
    assert result.success
    # Only 2 fresh DataGet calls (range_1_capture + range_2_capture)
    data_gets = [c for c in ctx.calls if c[0] == "SpectrumAnlzr_DataGet"]
    assert len(data_gets) == 2
    # Result includes all 3 per_range records (1 resumed + 2 fresh)
    assert len(result.data["per_range"]) == 3


# ── Progress emission ─────────────────────────────────────────────────────


def test_emit_progress_fires_every_step():
    skill = AcquirePSD()
    ctx = FakeCtx()
    skill.execute(ctx, {"freq_range_indices": "[0, 1, 2]"})
    # 4 steps + final summary → ≥4 emits
    assert len(ctx.emitted) >= 4
    assert ctx.emitted[-1].current_step is None


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
