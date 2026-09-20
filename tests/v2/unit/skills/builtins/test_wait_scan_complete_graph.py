"""GraphExecutor + WaitScanComplete regression tests (Phase 7 migration).

Pins the following contracts:

  1. ``WaitScanComplete`` subclasses :class:`CompositeSkillGraph`.
  2. Polling is exposed as ``_phase_poll_<i>`` synthetic steps via the
     ``_WaitScanCompletePhaseCtx`` wrapper (no real sub-skill calls).
  3. With ``Scan_StatusGet`` returning status=0 after N polls, the executor
     runs exactly N poll steps + 1 finalize step.
  4. Aborts via ``check_abort()`` cause the composite to issue
     ``Scan_Action(1, 0)`` and return success=False (v1 contract).
  5. Per-step ``emit_progress`` fires after each successful step plus a
     final summary; resume skips already-completed poll steps.
  6. **status 0 is not "finished"** — the line count decides (v6.1.2).

WHY (6) IS NEW. Until v6.1.2 the ONLY thing this file's context double could
say about a scan was ``Scan_StatusGet`` → 0 or 1. There was no frame, no buffer,
no notion of a line count, so "the scan stopped" and "the frame is complete"
were the same event BY CONSTRUCTION and no test here could have told them
apart. That is the defect's own shape reproduced in the test double: a stub more
regular than the instrument cannot fail the way the instrument does. ``FakeCtx``
now carries a real scan buffer, NaN-filled past the acquired front, and replies
with the 1-tuple channel-id shape the rig actually sends (``[(0,), (30,)]``).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_wait_scan_complete_graph.py -x -v
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

from mast.core.types import NanonisCallRecord, SafetyLevel, SkillResult
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeProgress
from mast.skills.builtins.scan_utils import (
    WaitScanComplete,
    _PHASE_POLL_PREFIX,
    _PHASE_FINALIZE,
)


# ── FakeCtx with scriptable Scan_StatusGet responses ─────────────────────


@dataclass
class FakeCtx:
    """Returns scan-status=running for the first N polls, then finished (0).

    Also models the scan buffer: ``lines`` configured, ``lines_done`` of them
    acquired, the rest NaN — which is how Nanonis leaves rows it has not
    reached. ``lines_done=None`` means "fully acquired".
    """
    statuses: list[int] = field(default_factory=lambda: [1, 1, 0])
    abort_after: int | None = None     # trigger check_abort() after N polls
    poll_count: int = 0
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    emitted: list[CompositeProgress] = field(default_factory=list)
    prior_progress: dict | None = None
    flushes: int = 0
    _aborted: bool = False
    # ── scan buffer ──
    lines: int = 64
    pixels: int = 64
    lines_done: int | None = None      # None → complete
    buffer_readable: bool = True       # False → Scan_BufferGet errors
    frame_readable: bool = True        # False → Scan_FrameDataGrab errors

    def _frame(self) -> np.ndarray:
        arr = np.arange(self.lines * self.pixels,
                        dtype=np.float64).reshape(self.lines, self.pixels)
        done = self.lines if self.lines_done is None else self.lines_done
        arr[done:, :] = np.nan
        return arr

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method == "Scan_StatusGet":
            idx = self.poll_count
            self.poll_count += 1
            status = self.statuses[idx] if idx < len(self.statuses) else 0
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [status]),
            )
        if method == "Scan_BufferGet":
            if not self.buffer_readable:
                return NanonisCallRecord(method=method, args=args,
                                         error="buffer unreadable")
            # Channel ids as 1-TUPLES — the shape the rig actually sends
            # (2026-08-04). Bare ints here are what let a real crash hide.
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [2, [(0,), (30,)], self.pixels, self.lines]),
            )
        if method == "Scan_FrameDataGrab":
            if not self.frame_readable:
                return NanonisCallRecord(method=method, args=args,
                                         error="frame unreadable")
            # Heterogeneous body, exactly as nanonis_spm decodes it:
            # [name_len, name, rows, cols, data_2D, direction]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [5, "Z (m)", self.lines, self.pixels,
                                        self._frame(), 1]),
            )
        # Scan_Action(1, 0) used to stop a running scan on abort
        if method == "Scan_Action":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", []))
        return NanonisCallRecord(method=method, args=args,
                                 error=f"unmocked: {method}")

    def check_abort(self) -> bool:
        if self.abort_after is not None and self.poll_count >= self.abort_after:
            self._aborted = True
            return True
        return False

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Shape tests ────────────────────────────────────────────────────────────


def test_wait_scan_complete_is_composite_graph():
    """WaitScanComplete migrated from BaseSkill → CompositeSkillGraph."""
    assert issubclass(WaitScanComplete, CompositeSkillGraph)


def test_wait_scan_complete_metadata_preserved():
    """Safety level + name unchanged from v1 contract."""
    meta = WaitScanComplete().metadata()
    assert meta.name == "WaitScanComplete"
    assert meta.safety_level == SafetyLevel.AUTO


# ── Execution: full poll-loop ─────────────────────────────────────────────


def test_three_polls_then_done(monkeypatch):
    """Scan finishes on the 3rd poll → 3 poll steps + 1 finalize."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()
    ctx = FakeCtx(statuses=[1, 1, 0])
    result = skill.execute(ctx, {"timeout_ms": 60000})
    assert result.success
    assert result.data["timed_out"] is False
    assert result.data["polls"] == 3
    # Scan_StatusGet called 3 times
    status_calls = [c for c in ctx.calls if c[0] == "Scan_StatusGet"]
    assert len(status_calls) == 3
    # 3 poll steps + 1 finalize completed
    snap = result.data["_progress"]
    poll_completed = [s for s in snap["completed_steps"]
                      if s.startswith("poll_")]
    assert len(poll_completed) == 3
    assert "finalize" in snap["completed_steps"]


def test_single_poll_terminal(monkeypatch):
    """Scan already finished → 1 poll + 1 finalize."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()
    ctx = FakeCtx(statuses=[0])
    result = skill.execute(ctx, {"timeout_ms": 60000})
    assert result.success
    assert result.data["polls"] == 1


def test_abort_invokes_scan_action_stop(monkeypatch):
    """check_abort() True → Scan_Action(1, 0) + success=False."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()
    ctx = FakeCtx(statuses=[1, 1, 1, 0], abort_after=0)
    result = skill.execute(ctx, {"timeout_ms": 60000})
    assert not result.success
    assert "aborted" in (result.error or "").lower()
    # Scan_Action stop was issued
    methods = [c[0] for c in ctx.calls]
    assert "Scan_Action" in methods
    stop_call = next(c for c in ctx.calls if c[0] == "Scan_Action")
    assert stop_call[1] == (1, 0)


def test_uses_polling_not_blocking_call(monkeypatch):
    """v0.3.14 invariant: must NOT call Scan_WaitEndOfScan."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()
    ctx = FakeCtx(statuses=[0])
    skill.execute(ctx, {"timeout_ms": 5000})
    methods = [c[0] for c in ctx.calls]
    assert "Scan_StatusGet" in methods
    assert "Scan_WaitEndOfScan" not in methods


# ── "not scanning" vs "finished" (v6.1.2) ────────────────────────────────


def test_a_complete_frame_is_completed(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    result = WaitScanComplete().execute(
        FakeCtx(statuses=[1, 0], lines=64, lines_done=64), {"timeout_ms": 60000})
    assert result.success
    assert result.data["outcome"] == "completed"
    assert result.data["stopped_early"] is False
    assert result.data["lines_verified"] is True
    assert (result.data["lines_done"], result.data["lines_total"]) == (64, 64)


def test_a_scan_stopped_part_way_is_not_completed(monkeypatch):
    """The 2026-08-04 event: a frame stopped at 24 %, no file produced. Before
    this, the status alone made it indistinguishable from a finished scan."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    result = WaitScanComplete().execute(
        FakeCtx(statuses=[1, 0], lines=512, lines_done=123), {"timeout_ms": 60000})
    assert result.data["outcome"] == "stopped_early"
    assert result.data["stopped_early"] is True
    assert result.data["timed_out"] is False        # NOT a timeout — it stopped
    assert (result.data["lines_done"], result.data["lines_total"]) == (123, 512)


def test_one_missing_line_still_counts_as_stopped_early(monkeypatch):
    """No fudge factor. 511/512 is a frame that did not finish, and inventing a
    tolerance here would just be a threshold nobody calibrated."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    result = WaitScanComplete().execute(
        FakeCtx(statuses=[0], lines=512, lines_done=511), {"timeout_ms": 60000})
    assert result.data["stopped_early"] is True


@pytest.mark.parametrize("broken", ["buffer", "frame"])
def test_an_unreadable_buffer_falls_back_to_the_old_behaviour(monkeypatch, broken):
    """Fail OPEN: unmeasurable is not truncated. Calling an unreadable scan
    'stopped early' would break every install whose reply we cannot parse, to
    guard against a case we cannot see. The result says it was not verified."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(statuses=[0], lines=512, lines_done=100,
                  buffer_readable=(broken != "buffer"),
                  frame_readable=(broken != "frame"))
    result = WaitScanComplete().execute(ctx, {"timeout_ms": 60000})
    assert result.success
    if broken == "frame":
        assert result.data["outcome"] == "completed"
        assert result.data["stopped_early"] is False
        assert result.data["lines_verified"] is False
    else:
        # The buffer is what supplies the configured total; the frame still
        # carries its own row count, so this case IS still measurable.
        assert result.data["outcome"] == "stopped_early"
        assert result.data["lines_verified"] is True


def test_a_frame_that_is_not_the_configured_size_is_not_judged(monkeypatch):
    """The grabbed frame and the configured buffer must describe the SAME
    object. When they disagree — a reply we mis-shaped, a buffer resized under
    us, an instrument that does not allocate the whole frame — dividing one by
    the other manufactures a truncation out of a parsing difference. This is the
    guard that keeps a flat-list simulator reply from failing every scan."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    ctx = FakeCtx(statuses=[0], lines=512, lines_done=512)
    # Buffer says 512 lines; the grab hands back a 4×4 frame.
    real = ctx.safe_call

    def shrunk(method, *args, role="main"):
        if method == "Scan_FrameDataGrab":
            arr = np.ones((4, 4), dtype=np.float64)
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [5, "Z (m)", 4, 4, arr, 1]))
        return real(method, *args, role=role)

    ctx.safe_call = shrunk                       # type: ignore[method-assign]
    result = WaitScanComplete().execute(ctx, {"timeout_ms": 60000})
    assert result.data["outcome"] == "completed"
    assert result.data["stopped_early"] is False
    assert result.data["lines_verified"] is False


def test_the_frame_row_count_is_the_fallback_denominator(monkeypatch):
    """With no configured line count, the grabbed frame's own rows do the job."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(statuses=[0], lines=64, lines_done=64, buffer_readable=False)
    result = WaitScanComplete().execute(ctx, {"timeout_ms": 60000})
    assert result.data["outcome"] == "completed"
    assert result.data["lines_total"] == 64


def test_the_line_check_costs_one_grab_not_one_per_poll(monkeypatch):
    """A 512² frame is ≈1 MB. At 0.5 s polling, doing this per poll would push
    2 MB/s down the same socket the scan is using."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(statuses=[1, 1, 1, 1, 0], lines=64, lines_done=64)
    WaitScanComplete().execute(ctx, {"timeout_ms": 60000})
    grabs = [c for c in ctx.calls if c[0] == "Scan_FrameDataGrab"]
    assert len(grabs) == 1
    assert len([c for c in ctx.calls if c[0] == "Scan_BufferGet"]) == 1
    assert len([c for c in ctx.calls if c[0] == "Scan_StatusGet"]) == 5


def test_the_grab_uses_an_acquired_channel_id(monkeypatch):
    """Scan_FrameDataGrab's channel arg must be one of the ACQUIRED ids from
    Scan_BufferGet — and those arrive as 1-tuples on the instrument. Passing a tuple
    through would be the v6.1.1 SetScanBuffer crash all over again."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(statuses=[0], lines=64, lines_done=64)
    WaitScanComplete().execute(ctx, {"timeout_ms": 60000})
    grab = next(c for c in ctx.calls if c[0] == "Scan_FrameDataGrab")
    assert grab[1] == (0, 1)                 # channel 0, forward — plain ints


def test_a_timeout_is_still_a_timeout_not_a_stop(monkeypatch):
    """Three distinct outcomes, not two dressed up. A timeout already stops the
    scan itself, so it must not be re-labelled as an early stop."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    result = WaitScanComplete().execute(
        FakeCtx(statuses=[1, 1, 1, 1], lines=512, lines_done=10),
        {"timeout_ms": 0})
    assert result.success
    assert result.data["timed_out"] is True
    assert result.data["outcome"] == "timed_out"
    assert result.data["stopped_early"] is False


def test_an_abort_reports_the_abort_outcome(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    result = WaitScanComplete().execute(
        FakeCtx(statuses=[1, 1, 0], abort_after=0), {"timeout_ms": 60000})
    assert not result.success
    assert result.data["outcome"] == "aborted"


def test_a_stopped_scan_is_not_grabbed_twice_on_resume(monkeypatch):
    """Resume must not re-measure a frame the instrument has since overwritten:
    the terminal partials are what a resumed run reads."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    prior = CompositeProgress(
        composite_name="WaitScanComplete",
        total_steps=10,
        completed_steps=["poll_0"],
        partial_data={"polls": 1, "scan_done": True, "timed_out": False,
                      "aborted": False, "stopped_early": True,
                      "outcome": "stopped_early", "lines_done": 123,
                      "lines_total": 512, "lines_verified": True},
    )
    ctx = FakeCtx(statuses=[0], lines=512, lines_done=512,
                  prior_progress=prior.to_dict())
    result = WaitScanComplete().execute(ctx, {"timeout_ms": 60000})
    assert result.data["outcome"] == "stopped_early"
    assert result.data["lines_done"] == 123
    assert not [c for c in ctx.calls if c[0] == "Scan_FrameDataGrab"]


# ── Progress emission ─────────────────────────────────────────────────────


def test_emit_progress_fires_per_step(monkeypatch):
    """Each poll + finalize emits a CompositeProgress snapshot."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()
    ctx = FakeCtx(statuses=[1, 0])
    skill.execute(ctx, {"timeout_ms": 60000})
    # 2 polls + 1 finalize → at least 3 emits + 1 final summary
    assert len(ctx.emitted) >= 3
    # Final emit clears current_step
    assert ctx.emitted[-1].current_step is None


def test_phase_dispatch_intercepts_phase_skill_names(monkeypatch):
    """The synthetic ``_phase_*`` skill names are intercepted (no real sub-call)."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()

    @dataclass
    class TraceCtx(FakeCtx):
        run_log: list[tuple[str, dict]] = field(default_factory=list)

        def run(self, skill_name: str, params: dict) -> SkillResult:
            self.run_log.append((skill_name, dict(params)))
            return SkillResult(skill_name=skill_name, success=True, data={})

    ctx = TraceCtx(statuses=[0])
    skill.execute(ctx, {"timeout_ms": 60000})
    # _phase_* never falls through to ctx.run — only Scan_StatusGet sees it
    fallthrough = [name for name, _ in ctx.run_log
                   if name.startswith("_phase_")]
    assert fallthrough == [], (
        f"phase names should be intercepted, leaked: {fallthrough}"
    )


# ── Resume ────────────────────────────────────────────────────────────────


def test_resume_skips_completed_polls(monkeypatch):
    """Prior progress with 2 completed polls → fresh run skips them."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()
    # Seed prior progress: poll_0 and poll_1 already done.
    prior = CompositeProgress(
        composite_name="WaitScanComplete",
        total_steps=10,
        completed_steps=["poll_0", "poll_1"],
        partial_data={"polls": 2, "scan_done": False,
                      "timed_out": False, "aborted": False},
    )
    ctx = FakeCtx(statuses=[0], prior_progress=prior.to_dict())
    result = skill.execute(ctx, {"timeout_ms": 60000})
    assert result.success
    # Only 1 new Scan_StatusGet call (skipped poll_0 and poll_1)
    status_calls = [c for c in ctx.calls if c[0] == "Scan_StatusGet"]
    assert len(status_calls) == 1
    # The new poll has a unique id (poll_2, not collision with poll_0/1)
    snap = result.data["_progress"]
    new_polls = [s for s in snap["completed_steps"] if s.startswith("poll_")]
    # Combined (resumed + new) ≥ 3 unique ids
    assert len(set(new_polls)) >= 3


# ── Checkpoint flush ──────────────────────────────────────────────────────


def test_finalize_flushes_checkpoint(monkeypatch):
    """``finalize`` step has checkpoint_after=True → exactly 1 flush."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    skill = WaitScanComplete()
    ctx = FakeCtx(statuses=[0])
    skill.execute(ctx, {"timeout_ms": 60000})
    # Polls have checkpoint_after=False → only finalize flushes.
    assert ctx.flushes == 1


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
