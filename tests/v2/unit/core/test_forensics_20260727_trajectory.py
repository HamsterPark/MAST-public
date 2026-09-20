"""2026-07-27 field — the agent training log recorded nothing.

`trajectories.experiment_id REFERENCES experiments(id)` points at the V2
experiments table and the store runs with `PRAGMA foreign_keys = ON`. The
runtime handed it the V1 ExperimentLog's UUID, so every INSERT died on a
FOREIGN KEY constraint — silently, at DEBUG level, behind a ULID that had
already been minted locally and returned to a caller who had no way to know.
Over a full day of runs the table gained 0 rows; the single row from two days
earlier survived only because it was created 18 seconds before that day's
experiment existed, when the v1 id happened to be None.

Two more silent-write failures in the same family, found while fixing it:
  * every FINISHED run closed with `exit_status="completed"`, which is not in
    the schema's CHECK list ('success','aborted','failed','timeout') — so no
    completed run ever got an exit status either;
  * once a trajectory INSERT is rejected, every one of its steps then fails its
    own foreign key, one swallowed exception per step.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_forensics_20260727_trajectory.py -q
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
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import logging  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.logging.v2.trace_sink import QueuedTraceSink  # noqa: E402

_V1_STYLE_UUID = "441ebe7c-2a73-4f08-8cdf-5034728022df"


@pytest.fixture()
def live_v2(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.logging.v2.live import open_live_v2
    repos, eid = open_live_v2()
    assert repos is not None and eid
    return repos, eid


def _bare_runtime(**attrs):
    """A CoreRuntime with its real methods but none of its construction — the
    trajectory path only reads the attributes set here. `__new__` (not
    SimpleNamespace) so the methods under test really are the class's own."""
    obj = CoreRuntime.__new__(CoreRuntime)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


@pytest.fixture()
def rt(live_v2):
    """Runtime wired to a real v2 store behind a real sink. Its V1 experiment
    log holds a v1-style UUID — the value that used to reach the v2 foreign
    key."""
    repos, eid = live_v2
    sink = QueuedTraceSink(repos)
    yield _bare_runtime(
        _v2_repos=repos, _v2_eid=eid, _trace_sink=sink,
        _active_traj=None, _active_thread_id=None,
        _experiment_log=SimpleNamespace(current_experiment_id=_V1_STYLE_UUID,
                                        current_sample_id="v1-sample-id"))
    sink.close()


# ── the mechanism, stated as a test so it can't be re-broken quietly ──

def test_the_v2_store_really_rejects_a_v1_experiment_id(live_v2):
    repos, eid = live_v2
    with pytest.raises(Exception, match="FOREIGN KEY"):
        repos.trajectories.begin(thread_id="t", experiment_id=_V1_STYLE_UUID)
    assert repos.trajectories.begin(thread_id="t", experiment_id=eid)


def test_the_v2_store_really_rejects_exit_status_completed(live_v2):
    repos, eid = live_v2
    tid = repos.trajectories.begin(thread_id="t", experiment_id=eid)
    with pytest.raises(Exception, match="CHECK constraint failed"):
        repos.trajectories.end(tid, exit_status="completed")


# ── the fix: a run with an OPEN experiment must persist its trajectory ──

def test_begin_trajectory_persists_while_an_experiment_is_open(rt, live_v2):
    """The scenario that failed 100% of the time: an experiment IS open, so the
    v1 id is not None and used to be written into the v2 foreign key."""
    repos, eid = live_v2
    tid = CoreRuntime.begin_trajectory(
        rt, "agents-ce41b51220f6",
        operator_intent={"instruction": "取五个点，各做一个2V~-2V的STS"})
    assert tid
    rt._trace_sink.flush()

    row = repos.trajectories.get(tid)
    assert row is not None, "the trajectory must actually be IN the database"
    assert row["thread_id"] == "agents-ce41b51220f6"
    assert row["experiment_id"] == eid          # the v2 id, not the v1 UUID
    assert rt._trace_sink.stats()["failed"] == 0


def test_steps_land_because_their_trajectory_exists(rt, live_v2):
    """`trajectory_steps.trajectory_id REFERENCES trajectories(id)` — when the
    parent INSERT is rejected the steps go down with it."""
    repos, _ = live_v2
    tid = CoreRuntime.begin_trajectory(rt, "agents-ec9ec60d2ead")
    rt._trace_sink.record_step(trajectory_id=tid, step_type="tool_call",
                               agent_id="IC", output={"skill": "AcquireSTS"})
    rt._trace_sink.flush()
    steps = repos.steps.by_trajectory(tid)
    assert len(steps) == 1 and steps[0]["agent_id"] == "IC"


def test_end_trajectory_maps_completed_onto_the_stored_vocabulary(rt, live_v2):
    repos, _ = live_v2
    tid = CoreRuntime.begin_trajectory(rt, "agents-5d260bcf353c")
    CoreRuntime.end_trajectory(rt, "completed")     # what the caller says
    rt._trace_sink.flush()
    assert repos.trajectories.get(tid)["exit_status"] == "success"
    assert rt._trace_sink.stats()["failed"] == 0


def test_end_trajectory_keeps_aborted_as_aborted(rt, live_v2):
    repos, _ = live_v2
    tid = CoreRuntime.begin_trajectory(rt, "agents-f898e226b844")
    CoreRuntime.end_trajectory(rt, "aborted")
    rt._trace_sink.flush()
    assert repos.trajectories.get(tid)["exit_status"] == "aborted"


def test_unknown_exit_status_stores_null_and_says_so(rt, live_v2, caplog):
    """An unrecognised label must not be quietly rounded to a plausible one."""
    repos, _ = live_v2
    tid = CoreRuntime.begin_trajectory(rt, "agents-e0c922eb1f1d")
    with caplog.at_level(logging.WARNING, logger="mast.core.runtime"):
        CoreRuntime.end_trajectory(rt, "half-way-ish")
    rt._trace_sink.flush()
    assert repos.trajectories.get(tid)["exit_status"] is None
    assert any("half-way-ish" in r.getMessage() for r in caplog.records)


def test_no_v2_scope_means_no_experiment_link_not_a_bogus_one(live_v2):
    """No live v2 session → the trajectory is written unlinked, which the
    schema allows, rather than with an id from the other store."""
    repos, _ = live_v2
    sink = QueuedTraceSink(repos)
    try:
        obj = _bare_runtime(
            _v2_repos=repos, _v2_eid=None, _trace_sink=sink, _active_traj=None,
            _experiment_log=SimpleNamespace(current_experiment_id=_V1_STYLE_UUID,
                                            current_sample_id=None))
        tid = CoreRuntime.begin_trajectory(obj, "agents-a68270103fbd")
        sink.flush()
        row = repos.trajectories.get(tid)
        assert row is not None and row["experiment_id"] is None
        assert sink.stats()["failed"] == 0
    finally:
        sink.close()


# ── the swallow itself: a systematic failure must be visible ──

class _BrokenRepos:
    class trajectories:
        @staticmethod
        def begin(**kw):
            raise RuntimeError("FOREIGN KEY constraint failed")

        @staticmethod
        def end(*a, **kw):
            pass

    class steps:
        @staticmethod
        def record(**kw):
            raise RuntimeError("FOREIGN KEY constraint failed")


def test_a_rejected_write_is_logged_at_warning_not_debug(caplog):
    """It failed at a 100% rate all day and the service log had NOT ONE LINE
    about it, because the worker swallowed at DEBUG."""
    sink = QueuedTraceSink(_BrokenRepos())
    try:
        with caplog.at_level(logging.DEBUG, logger="mast.logging.v2.trace_sink"):
            sink.begin_trajectory(thread_id="t")
            sink.flush()
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warns, "a rejected trajectory write must be logged at WARNING+"
        assert "LOSING DATA" in warns[0].getMessage()
        assert sink.stats()["failed"] == 1
    finally:
        sink.close()


def test_steps_of_a_dead_trajectory_are_dropped_as_counted_orphans(caplog):
    """Not one swallowed foreign-key error per step: the sink remembers that
    the parent INSERT was rejected and drops the orphans, counted."""
    sink = QueuedTraceSink(_BrokenRepos())
    try:
        with caplog.at_level(logging.WARNING, logger="mast.logging.v2.trace_sink"):
            tid = sink.begin_trajectory(thread_id="t")
            for _ in range(9):
                sink.record_step(trajectory_id=tid, step_type="tool_call")
            sink.flush()
        st = sink.stats()
        assert st["failed"] == 1, "only the parent write failed"
        assert st["orphan_dropped"] == 9
        assert st["dead_trajectories"] == 1
        assert any("its INSERT was rejected" in r.getMessage()
                   for r in caplog.records)
    finally:
        sink.close()


def test_healthy_sink_reports_clean_stats(live_v2):
    repos, eid = live_v2
    sink = QueuedTraceSink(repos)
    try:
        tid = sink.begin_trajectory(thread_id="t", experiment_id=eid)
        sink.record_step(trajectory_id=tid, step_type="agent_turn")
        sink.flush()
        assert sink.stats() == {"dropped": 0, "failed": 0, "orphan_dropped": 0,
                                "dead_trajectories": 0, "last_error": None}
    finally:
        sink.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
