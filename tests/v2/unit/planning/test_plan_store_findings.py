"""v2 unit tests for mast.planning.plan_store fixes.

Covers review findings:
  #80  update_progress() now PERSISTS phase status into the definition JSON
       (previously only rewrote Markdown -> status lost on reload).
  #133 update_status(notes=...) persists notes (abort_plan note was dropped).
  #134 connections are closed via contextlib.closing (no handle leak).
  #135 redundant `import sqlite3` removed from __init__ (smoke: ctor still works).

No LLM / network. Pure SQLite on a tmp_path DB.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/planning/ -q -p no:randomly
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
_REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path[:] = [
    p for p in sys.path
    if not p or Path(p).resolve() != _REPO_ROOT_PATH
]
while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.planning.plan_store import (
    ExperimentPlan, PlanPhase, PlanStatus, PlanStore,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_store(tmp_path) -> PlanStore:
    return PlanStore(db_path=tmp_path / "plans.db", plans_dir=tmp_path / "md")


def _make_plan() -> ExperimentPlan:
    return ExperimentPlan(
        plan_id="",
        name="测试计划",
        goal="验证持久化",
        phases=[
            PlanPhase(id="p0", name="表面制备",
                      steps=[{"skill": "FullScan", "params": {}}]),
            PlanPhase(id="p1", name="成像",
                      steps=[{"skill": "GridSTS", "params": {}},
                             {"skill": "AssessQuality", "params": {}}]),
        ],
        status=PlanStatus.DRAFT,
    )


# ── #135: constructor smoke (no leftover sqlite3 import dependency) ───

def test_ctor_creates_db_and_dir(tmp_path):
    store = _make_store(tmp_path)
    assert (tmp_path / "plans.db").exists()
    assert (tmp_path / "md").is_dir()
    # And a basic save/load round-trips
    pid = store.save(_make_plan())
    assert store.load(pid) is not None


# ── #80: update_progress persists phase status into definition JSON ──

def test_update_progress_persists_phase_status(tmp_path):
    store = _make_store(tmp_path)
    pid = store.save(_make_plan())

    # Mark phase 0 as done, advance into phase 1 step 1.
    store.update_progress(pid, phase_idx=0, step_idx=0, phase_status="done")

    # Reload from a FRESH store instance so nothing comes from memory.
    store2 = _make_store(tmp_path)
    reloaded = store2.load(pid)
    assert reloaded is not None
    # The phase status must have round-tripped through the definition column.
    assert reloaded.phases[0].status == "done", (
        "phase status was not persisted into definition JSON "
    )
    assert reloaded.current_phase_idx == 0
    assert reloaded.current_step_idx == 0


def test_update_progress_running_phase_persists(tmp_path):
    store = _make_store(tmp_path)
    pid = store.save(_make_plan())
    store.update_progress(pid, phase_idx=1, step_idx=1, phase_status="running")

    reloaded = _make_store(tmp_path).load(pid)
    assert reloaded.phases[1].status == "running"
    assert reloaded.current_phase_idx == 1
    assert reloaded.current_step_idx == 1


def test_update_progress_missing_plan_is_noop(tmp_path):
    store = _make_store(tmp_path)
    # Should not raise when plan_id doesn't exist.
    store.update_progress("nope", 0, 0, "done")
    assert store.load("nope") is None


# ── #133: update_status persists notes ──────────────────────────────

def test_update_status_persists_notes(tmp_path):
    store = _make_store(tmp_path)
    pid = store.save(_make_plan())
    store.update_status(pid, PlanStatus.ABORTED, notes="Aborted: tip crashed")

    reloaded = _make_store(tmp_path).load(pid)
    assert reloaded.status == PlanStatus.ABORTED
    assert reloaded.notes == "Aborted: tip crashed", (
        "notes were not persisted by update_status "
    )


def test_update_status_without_notes_preserves_existing(tmp_path):
    store = _make_store(tmp_path)
    plan = _make_plan()
    plan.notes = "original note"
    pid = store.save(plan)
    # Update status WITHOUT notes -> existing note must be preserved.
    store.update_status(pid, PlanStatus.RUNNING)

    reloaded = _make_store(tmp_path).load(pid)
    assert reloaded.status == PlanStatus.RUNNING
    assert reloaded.notes == "original note"


# ── #134: connections do not leak (smoke via many ops) ──────────────

def test_many_ops_do_not_leak_handles(tmp_path):
    # If connections leaked, a tight loop of save/load/update would
    # eventually exhaust handles / lock the WAL. 200 cycles is plenty to
    # surface a regression without being slow.
    store = _make_store(tmp_path)
    pid = store.save(_make_plan())
    for i in range(200):
        store.update_progress(pid, phase_idx=i % 2, step_idx=0,
                              phase_status="running")
        store.update_status(pid, PlanStatus.RUNNING)
        assert store.load(pid) is not None
        store.list_plans()
    # Final state is still consistent.
    assert store.load(pid).status == PlanStatus.RUNNING


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-p", "no:randomly"])
