"""Sidecar resume hygiene — .

A 9-day-old, fully-completed ``AutoApproach.json`` sidecar survived in
``experiments/composite_progress/`` and was resumed by every later
AutoApproach invocation: all 4 steps were "already completed", so the
composite returned instant success without touching the instrument — 进针
was declared at noise-level current. Resume must only bridge the
crash/interrupt window of the SAME logical run:

  * TERMINAL sidecars (all steps completed) are finished runs whose cleanup
    was missed → discard + delete, never resume.
  * STALE sidecars (last update older than the resume window) → discard +
    delete, never resume.
  * FRESH, PARTIAL sidecars → still resume (the P2-G crash-recovery
    behaviour this mechanism exists for).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_sidecar_staleness.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import json
import time

import pytest

from mast.skills.composite import graph_executor as ge_mod
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)


class _Res:
    def __init__(self, success=True, data=None, error=""):
        self.success = success
        self.data = data or {}
        self.error = error
        self.nanonis_calls = []


class FakeCtx:
    def __init__(self):
        self.calls = []

    def run(self, skill_name, params):
        self.calls.append((skill_name, dict(params)))
        return _Res()


@pytest.fixture(autouse=True)
def _tmp_sidecar_dir(tmp_path, monkeypatch):
    d = tmp_path / "composite_progress"
    d.mkdir()
    import re as _re

    def _path(name, run_id=""):
        safe = _re.sub(r"[^\w一-鿿-]", "_", str(name))[:80] or "composite"
        if run_id:
            return d / f"{safe}__{_re.sub(r'[^\w-]', '_', str(run_id))[:40]}.json"
        return d / f"{safe}.json"

    monkeypatch.setattr(ge_mod, "_sidecar_path", _path)
    monkeypatch.setattr(ge_mod, "_sidecar_dir", lambda: d)
    return d


def _write_sidecar(dir_: Path, name: str, *, completed, total, age_s: float,
                   run_id: str = ""):
    stem = f"{name}__{run_id}" if run_id else name
    p = dir_ / f"{stem}.json"
    prog = CompositeProgress(
        composite_name=name, total_steps=total,
        completed_steps=list(completed),
        started_at=time.time() - age_s - 5.0,
        last_update_at=time.time() - age_s,
    )
    p.write_text(json.dumps(prog.to_dict()), encoding="utf-8")
    return p


def _plan():
    return [
        CompositeStep(step_id="a", skill_name="SkillA", params={}),
        CompositeStep(step_id="b", skill_name="SkillB", params={}),
    ]


def test_terminal_sidecar_discarded_and_deleted(_tmp_sidecar_dir):
    """The field failure: a fully-completed leftover must NOT short-circuit a
    new run into a no-op success — every step must actually execute."""
    p = _write_sidecar(_tmp_sidecar_dir, "Demo", completed=["a", "b"],
                       total=2, age_s=10.0)  # fresh but TERMINAL
    ctx = FakeCtx()
    ex = GraphExecutor(composite_name="Demo", context=ctx)
    assert ex.run_plan(iter(_plan())) is True
    # both steps genuinely ran (nothing was resume-skipped)
    assert [c[0] for c in ctx.calls] == ["SkillA", "SkillB"]
    # nothing terminal was resumed
    assert ex.progress.completed_steps == ["a", "b"]


def test_stale_partial_sidecar_discarded(_tmp_sidecar_dir):
    """A partial sidecar older than the resume window (e.g. 9 days) is a
    leftover from a previous session, not an interrupted run — start fresh."""
    _write_sidecar(_tmp_sidecar_dir, "Demo", completed=["a"], total=2,
                   age_s=9 * 24 * 3600.0)
    ctx = FakeCtx()
    ex = GraphExecutor(composite_name="Demo", context=ctx)
    ex.run_plan(iter(_plan()))
    assert [c[0] for c in ctx.calls] == ["SkillA", "SkillB"]


def test_fresh_partial_sidecar_still_resumes(_tmp_sidecar_dir):
    """The legitimate P2-G case is untouched: a fresh, partial sidecar (the
    run crashed seconds ago) resumes and skips the completed step."""
    _write_sidecar(_tmp_sidecar_dir, "Demo", completed=["a"], total=2,
                   age_s=3.0)
    ctx = FakeCtx()
    ex = GraphExecutor(composite_name="Demo", context=ctx)
    assert ex.run_plan(iter(_plan())) is True
    # step "a" was resume-skipped; only "b" ran
    assert [c[0] for c in ctx.calls] == ["SkillB"]


def test_terminal_sidecar_file_removed(_tmp_sidecar_dir):
    """Discarding also deletes the file so it cannot poison the NEXT run."""
    p = _write_sidecar(_tmp_sidecar_dir, "Demo", completed=["a", "b"],
                       total=2, age_s=10.0)
    GraphExecutor(composite_name="Demo", context=FakeCtx())
    assert not p.exists()


class _CurrentCtx(FakeCtx):
    """FakeCtx whose safe_call serves an engaged tunnelling current, so
    AutoApproach's wait/verify phases succeed."""

    def __init__(self):
        super().__init__()
        self.safe_calls = []
        self._oog = 0

    def safe_call(self, method, *args, role="main"):
        from mast.core.types import NanonisCallRecord
        self.safe_calls.append(method)
        if method == "AutoApproach_OnOffGet":
            self._oog += 1
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [1 if self._oog <= 1 else 0]))
        if method == "Current_Get":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [5e-10]))
        if method == "ZCtrl_SetpntGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [5e-10]))
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", []))

    def check_abort(self):
        return False


class _RunCtx(FakeCtx):
    """A context that carries a run_id, like the orchestrator's real one."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id


def test_abort_from_inside_a_dynamic_plan_is_control_flow_not_failure():
    """An operator abort raised from INSIDE a dynamic plan generator (ConditionTip's
    wait-for-scan helper does exactly this) must abort the composite — not escape
    run_plan, where skill_adapter's generic `except Exception` would catch it and
    fire the skill's ROLLBACK, undoing work the abort never touched."""
    from mast.skills.composite._base import AbortRequested

    ctx = FakeCtx()
    ex = GraphExecutor(composite_name="Demo", context=ctx)

    def _dynamic_plan():
        yield CompositeStep(step_id="a", skill_name="SkillA", params={})
        raise AbortRequested("wait_scan_complete")   # operator stopped us mid-plan

    ok = ex.run_plan(_dynamic_plan())        # must NOT raise
    assert ok is False
    assert ex.progress.aborted is True
    assert "abort" in (ex.progress.aborted_reason or "").lower()
    assert [c[0] for c in ctx.calls] == ["SkillA"]   # step a ran, then we stopped


def test_abort_raised_by_a_sub_skill_is_not_counted_as_a_failure():
    """Same rule one level down: a sub-skill raising AbortRequested aborts the
    composite rather than being routed through the failure path."""
    from mast.skills.composite._base import AbortRequested

    class _AbortingCtx(FakeCtx):
        def run(self, skill_name, params):
            self.calls.append((skill_name, dict(params)))
            raise AbortRequested(skill_name)

    ctx = _AbortingCtx()
    ex = GraphExecutor(composite_name="Demo", context=ctx)
    ok = ex.run_plan(iter(_plan()))
    assert ok is False
    assert ex.progress.aborted is True
    assert ex.progress.failed_steps == [], "an abort is not a failed step"


def test_sidecar_is_scoped_to_the_run(_tmp_sidecar_dir):
    """The structural fix: the sidecar KEY carries the run id, so two runs of the
    same composite cannot share a progress file at all — the cross-run reuse that
    faked 进针 becomes impossible rather than merely guarded against."""
    ctx_a = _RunCtx("run-A")
    GraphExecutor(composite_name="Demo", context=ctx_a).run_plan(
        iter([CompositeStep(step_id="a", skill_name="SkillA", params={})]))
    ctx_b = _RunCtx("run-B")
    ex_b = GraphExecutor(composite_name="Demo", context=ctx_b)
    assert ex_b._sidecar_path != GraphExecutor(
        composite_name="Demo", context=_RunCtx("run-A"))._sidecar_path
    ex_b.run_plan(iter([CompositeStep(step_id="a", skill_name="SkillA", params={})]))
    assert [c[0] for c in ctx_b.calls] == ["SkillA"], \
        "a different run must NOT resume-skip the previous run's step"


def test_same_run_still_resumes(_tmp_sidecar_dir):
    """The legitimate P2-G case is untouched: a run INTERRUPTED mid-plan (a
    partial sidecar) resumes within the SAME run and skips what it already did.

    Note the interplay with the terminal rule: a run that reached its end clears
    its sidecar, so only a genuinely interrupted one has anything to resume."""
    _write_sidecar(_tmp_sidecar_dir, "Demo", completed=["a"], total=2,
                   age_s=3.0, run_id="run-X")
    ctx = _RunCtx("run-X")           # the SAME run, resumed after the interrupt
    ex = GraphExecutor(composite_name="Demo", context=ctx)
    ex.run_plan(iter(_plan()))
    assert [c[0] for c in ctx.calls] == ["SkillB"], \
        "step a completed in THIS run — it must be resume-skipped"


def test_other_runs_partial_progress_is_not_inherited(_tmp_sidecar_dir):
    """A partial sidecar from a DIFFERENT run is invisible: run-B re-executes
    everything rather than inheriting run-A's half-finished plan."""
    _write_sidecar(_tmp_sidecar_dir, "Demo", completed=["a"], total=2,
                   age_s=3.0, run_id="run-A")
    ctx = _RunCtx("run-B")
    ex = GraphExecutor(composite_name="Demo", context=ctx)
    ex.run_plan(iter(_plan()))
    assert [c[0] for c in ctx.calls] == ["SkillA", "SkillB"]


def test_sweep_deletes_ancient_sidecars(_tmp_sidecar_dir):
    import os
    import time as _t
    old = _tmp_sidecar_dir / "Demo__run-old.json"
    old.write_text("{}", encoding="utf-8")
    ancient = _t.time() - 48 * 3600
    os.utime(old, (ancient, ancient))
    fresh = _tmp_sidecar_dir / "Demo__run-new.json"
    fresh.write_text("{}", encoding="utf-8")

    ge_mod._sweep_stale_sidecars()
    assert not old.exists()
    assert fresh.exists()


def test_auto_approach_clears_sidecar_after_completion(_tmp_sidecar_dir, monkeypatch):
    """AutoApproach overrides run_composite and used to BYPASS the only
    clear_sidecar() call site (_base._graph_execute) — its sidecar therefore
    lived forever and every later approach resume-skipped all 4 phases into an
    instant fake success (the 2026-07-10 #42/#75 field failure). A completed
    run must leave NO sidecar behind, and a back-to-back second run must
    re-execute the hardware phases."""
    from mast.skills.builtins.approach import AutoApproach
    monkeypatch.setattr(AutoApproach, "_poll_interval_s", 0.01, raising=False)
    monkeypatch.setattr(AutoApproach, "_grace_s", 0.05, raising=False)

    ctx1 = _CurrentCtx()
    res1 = AutoApproach().execute(ctx1, {})
    assert res1.success
    sidecar = _tmp_sidecar_dir / "AutoApproach.json"
    assert not sidecar.exists(), "completed run must clear its sidecar"

    # Second run seconds later (well inside the resume window) must actually
    # talk to the instrument again, not resume-skip into a no-op.
    ctx2 = _CurrentCtx()
    res2 = AutoApproach().execute(ctx2, {})
    assert res2.success
    assert "AutoApproach_OnOffSet" in ctx2.safe_calls, (
        "second approach must re-execute start_approach, not resume-skip")
