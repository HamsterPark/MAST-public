"""GraphExecutor + ConditionTip regression tests (Phase 7 dynamic-plan migration).

Pins the dynamic-plan contract:

  1. ``plan_dynamic(params, executor)`` yields *five* steps per attempt
     (pulse / configure / set_speed / start_scan / assess) with step_ids
     ``a{N}:<phase>`` so multiple attempts can coexist in the same plan.
  2. ``wait_scan_complete`` runs synchronously between StartScan and
     AssessImageQuality — it is NOT a sub-skill step.
  3. **Early exit on quality**: if attempt 3's AssessImageQuality returns
     ``fft_quality >= target_quality``, attempts 4..max are NOT yielded.
  4. **Resume**: if prior progress already covers attempts 1 + 2 worth of
     step_ids (10 steps), only attempt 3+ steps execute fresh.
  5. **Failure path**: when every attempt under-shoots target, the
     composite fails but the SkillResult carries the full
     ``quality_history`` for diagnostic purposes.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_condition_tip_graph.py -x -v
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
from typing import Any, Callable

import pytest

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.condition_tip import ConditionTip
from mast.skills.composite.graph_executor import CompositeProgress


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context driving ConditionTip.

    ``quality_by_attempt`` maps attempt number -> fft_quality returned by
    AssessImageQuality. Any attempt not in the dict returns 0.0.

    ``scan_status_seq`` is a list of integers returned by Scan_StatusGet
    polls (0 = scan done). The fake pops the first element each poll and
    keeps the last value once the list empties.
    """
    quality_by_attempt: dict[int, float] = field(default_factory=dict)
    scan_status_seq: list[int] = field(default_factory=lambda: [0])
    prior_progress: dict | None = None
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    safe_calls: list[tuple[str, tuple]] = field(default_factory=list)
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    _attempt_counter: int = 0

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "AssessImageQuality":
            # Track which attempt this is from the previous TipPulse count.
            quality = self.quality_by_attempt.get(self._attempt_counter, 0.0)
            return SkillResult(
                skill_name=skill_name, success=True,
                data={"fft_quality": quality},
            )
        if skill_name == "TipPulse":
            self._attempt_counter += 1
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.safe_calls.append((method, args))
        if method == "Scan_StatusGet":
            if self.scan_status_seq:
                status = (
                    self.scan_status_seq.pop(0)
                    if len(self.scan_status_seq) > 1
                    else self.scan_status_seq[0]
                )
            else:
                status = 0
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [status]),
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


# ── Helpers ─────────────────────────────────────────────────────────────


def _params(**overrides: Any) -> dict:
    """Sensible defaults for ConditionTip in unit tests (no real hardware)."""
    base = dict(
        pulse_v=3.0,
        max_attempts=5,
        target_quality=0.3,
        center_x_m=0.0,
        center_y_m=0.0,
        scan_width_m=10e-9,
    )
    base.update(overrides)
    return base


def _attempt_step_ids(attempt: int) -> list[str]:
    return [
        f"a{attempt}:pulse",
        f"a{attempt}:configure",
        f"a{attempt}:set_speed",
        f"a{attempt}:start_scan",
        f"a{attempt}:assess",
    ]


# ── Tests ────────────────────────────────────────────────────────────────


def test_plan_dynamic_yields_five_steps_per_attempt():
    """The dynamic plan yields pulse/configure/set_speed/start_scan/assess
    for each attempt — five steps per attempt — and quality < target keeps
    iterating to max_attempts.

    To enumerate the plan in isolation we walk plan_dynamic directly with a
    no-op fake executor (we only care about the yielded step_ids and order).
    """
    skill = ConditionTip()
    skill._all_calls = []   # plan_dynamic uses self._all_calls indirectly

    class _StubExec:
        progress = type("P", (), {
            "aborted": False, "completed_steps": [],
            "partial_data": {"center_x_m": 0.0, "center_y_m": 0.0}})()
        sub_results: dict = {}

        def set_total_steps(self, n): self.total = n
        def set_partial(self, k, v): self.progress.partial_data[k] = v

    # We must intercept the synchronous wait_scan_complete helper too —
    # patch via skill._context that returns a stub for Scan_StatusGet.
    class _NullCtx:
        def safe_call(self, *args, **kwargs):
            return NanonisCallRecord(
                method=args[0] if args else "",
                args=tuple(args[1:]),
                return_value=("", b"", [0]),
            )

        def check_abort(self):
            return False

    skill._context = _NullCtx()
    yielded: list[str] = []
    gen = skill.plan_dynamic(_params(max_attempts=2), _StubExec())
    # Iterate through the generator manually, simulating "result was 0.0
    # quality" so it keeps going.
    while True:
        try:
            step = next(gen)
        except StopIteration:
            break
        yielded.append(step.step_id)

    # Initial best-effort assessment (a0: configure/set_speed/start_scan/assess,
    # NO pulse) precedes the pulse loop (审查: don't pulse a tip
    # that's already good), then two attempts × five steps.
    a0_scan = ["a0:configure", "a0:set_speed", "a0:start_scan", "a0:assess"]
    assert yielded == a0_scan + _attempt_step_ids(1) + _attempt_step_ids(2)


def test_metadata_version_bumped_to_v2():
    """Phase 7 graph migration bumps the version major to 2.x."""
    md = ConditionTip().metadata()
    assert md.version.startswith("2."), (
        f"expected v2.x version, got {md.version}"
    )


def test_early_exit_on_attempt_3():
    """Quality = 0.5 (> target 0.3) on attempt 3 → only 3 attempts run.

    Expected sub-skill invocations:
      attempts 1+2: each runs (TipPulse, ConfigureScan, SetScanSpeed,
                                StartScan, AssessImageQuality) = 5 invocations
      attempt 3: same 5 (with assess returning quality=0.5)
      attempts 4 + 5: NOT yielded → not invoked

    Total: 5 × 3 = 15 sub-skill invocations.
    """
    ctx = FakeCtx(
        quality_by_attempt={1: 0.1, 2: 0.2, 3: 0.5},
        scan_status_seq=[0],   # scan always done immediately
    )
    skill = ConditionTip()
    result = skill.execute(ctx, _params(max_attempts=5))

    skill_names = [name for name, _ in ctx.run_log]
    # a0 initial assessment adds one scan+assess (no pulse) before the 3 attempts.
    assert skill_names.count("TipPulse") == 3
    assert skill_names.count("ConfigureScan") == 4
    assert skill_names.count("SetScanSpeed") == 4
    assert skill_names.count("StartScan") == 4
    assert skill_names.count("AssessImageQuality") == 4

    assert result.success
    assert result.data["attempts"] == 3
    assert result.data["final_quality"] == pytest.approx(0.5)
    assert result.data["quality_history"] == [pytest.approx(0.1),
                                              pytest.approx(0.2),
                                              pytest.approx(0.5)]
    assert result.data["target_quality"] == pytest.approx(0.3)


def test_failure_all_attempts_below_target():
    """No attempt reaches target → composite fails, history is complete."""
    ctx = FakeCtx(
        quality_by_attempt={1: 0.05, 2: 0.10, 3: 0.15},
        scan_status_seq=[0],
    )
    skill = ConditionTip()
    result = skill.execute(ctx, _params(max_attempts=3, target_quality=0.5))

    skill_names = [name for name, _ in ctx.run_log]
    # a0 initial assess (1) + 3 attempts = 4 AssessImageQuality calls.
    assert skill_names.count("AssessImageQuality") == 4

    assert not result.success
    assert "Target quality" in result.error
    assert "0.5" in result.error
    assert result.data["attempts"] == 3
    assert result.data["quality_history"] == [pytest.approx(0.05),
                                              pytest.approx(0.10),
                                              pytest.approx(0.15)]
    # Even on failure, _progress survives so the wrap_skill adapter can
    # lift it into composite_progress.
    assert isinstance(result.data["_progress"], dict)


def test_resume_skips_completed_attempts():
    """Prior progress has attempts 1 + 2 fully done (10 step_ids) →
    only attempt 3's 5 steps execute on resume.
    """
    # Build prior progress covering attempts 1 + 2.
    prior_step_ids = _attempt_step_ids(1) + _attempt_step_ids(2)
    prior = CompositeProgress(
        composite_name="ConditionTip",
        total_steps=25,   # 5 attempts × 5 steps
        completed_steps=list(prior_step_ids),
        partial_data={
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "target_quality": 0.3,
            "max_attempts": 5,
            "attempts": 2,
            "final_quality": 0.1,
            "quality_history": [0.05, 0.1],
            "target_reached": False,
        },
    )
    # FakeCtx._attempt_counter starts at 0 and increments on each fresh
    # TipPulse the fake sees. Resume skips the prior 2 attempts, so the
    # FIRST fresh TipPulse the fake observes IS the user-facing attempt 3 —
    # the fake registers it as its own counter=1.
    ctx = FakeCtx(
        quality_by_attempt={1: 0.5},
        scan_status_seq=[0],
        prior_progress=prior.to_dict(),
    )

    skill = ConditionTip()
    result = skill.execute(ctx, _params(max_attempts=5))

    # Exactly attempt 3's five steps execute fresh (resume skips attempts 1+2).
    assert len(ctx.run_log) == 5
    skill_names = [name for name, _ in ctx.run_log]
    assert skill_names == ["TipPulse", "ConfigureScan", "SetScanSpeed",
                           "StartScan", "AssessImageQuality"]

    assert result.success
    # History carries the prior runs + the new one
    assert result.data["quality_history"] == [pytest.approx(0.05),
                                              pytest.approx(0.1),
                                              pytest.approx(0.5)]
    assert result.data["attempts"] == 3
    assert result.data["final_quality"] == pytest.approx(0.5)


def test_progress_snapshot_in_skill_result():
    """The `_progress` dict is lifted into result.data for the adapter."""
    ctx = FakeCtx(
        quality_by_attempt={1: 0.5},
        scan_status_seq=[0],
    )
    skill = ConditionTip()
    result = skill.execute(ctx, _params(max_attempts=5))

    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "ConditionTip"
    # a0 initial assess (4 steps) + attempt 1 (5 steps) = 9
    assert len(snap["completed_steps"]) == 9
    assert snap["aborted"] is False
    # partial_data carries the full quality history + attempts counter
    assert snap["partial_data"]["attempts"] == 1
    assert snap["partial_data"]["target_reached"] is True


def test_scan_timeout_aborts_composite():
    """If wait_scan_complete returns False, the composite aborts cleanly
    and quality_history reflects only the completed attempts."""
    ctx = FakeCtx(
        quality_by_attempt={1: 0.1},
        # Status always 1 (busy) → wait_scan_complete returns False after
        # timeout. We use a single small status_seq=[1] which keeps repeating
        # the last value (status busy forever).
        scan_status_seq=[1],
    )
    skill = ConditionTip()
    # Use a small max_attempts but the FIRST wait will time out (the helper
    # uses timeout_s=60 / 0.5 = 120 polls). We monkey-patch the helper to
    # short-circuit to False quickly for the test:
    import mast.skills.composite.condition_tip as ct_mod

    def _wait_false(context, timeout_s=60.0, poll_interval_s=0.5,
                    call_accumulator=None):
        # Mimic v1: stop scan, then return False.
        rec = context.safe_call("Scan_StatusGet")
        if call_accumulator is not None:
            call_accumulator.append(rec)
        return False

    orig = ct_mod.wait_scan_complete
    ct_mod.wait_scan_complete = _wait_false
    try:
        result = skill.execute(ctx, _params(max_attempts=3))
    finally:
        ct_mod.wait_scan_complete = orig

    assert not result.success
    assert "timed out" in result.error or "aborted" in result.error
    # Only attempt 1 got far enough to do pulse/configure/set_speed/start_scan
    # (assess was never reached because the wait failed)
    skill_names = [name for name, _ in ctx.run_log]
    assert "TipPulse" in skill_names
    assert "AssessImageQuality" not in skill_names
    # quality_history is empty since no assess happened
    assert result.data["quality_history"] == []


def test_resolve_center_from_state_when_params_omit():
    """If center_x_m / center_y_m are None in params, run_composite reads
    them from context.state.snapshot()."""

    @dataclass
    class _State:
        x_pos_m: float = 1.5e-9
        y_pos_m: float = 2.5e-9

    @dataclass
    class _StateProvider:
        x_pos_m: float = 1.5e-9
        y_pos_m: float = 2.5e-9

        def snapshot(self) -> _State:
            return _State(self.x_pos_m, self.y_pos_m)

    class CtxWithState(FakeCtx):
        state = _StateProvider()

    ctx = CtxWithState(
        quality_by_attempt={1: 0.5},
        scan_status_seq=[0],
    )
    skill = ConditionTip()
    params = _params(max_attempts=2)
    params.pop("center_x_m")
    params.pop("center_y_m")
    result = skill.execute(ctx, params)

    # First ConfigureScan should have used the snapshot's center.
    configure_calls = [
        p for name, p in ctx.run_log if name == "ConfigureScan"
    ]
    assert configure_calls
    assert configure_calls[0]["center_x_m"] == pytest.approx(1.5e-9)
    assert configure_calls[0]["center_y_m"] == pytest.approx(2.5e-9)
    assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
