"""start_experiment / start_sample idempotency + the resume primitive.

(2026-07-19/20): one NiI2/Au(111) study spawned SIX experiments and
eight samples across four session restarts, because every ``start_experiment``
minted a fresh row — the agent, having no memory of the run already in progress,
re-created it under a slightly different name each time.

The fix is opt-in so no existing caller changes behaviour:
  * ``start_experiment(name, goal, reuse_open=True)`` attaches to an already-open
    same-named experiment instead of duplicating it (the agent path opts in);
  * ``ExperimentLog.resume_experiment(id)`` attaches to an existing run WITHOUT
    creating a row (the reusable form of the runtime's startup auto-restore);
  * the default (``reuse_open=False``) is create-always, exactly as before.

Run:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/logging/test_experiment_idempotency.py -q
"""
from __future__ import annotations

import json

from mast.logging.storage import ExperimentStorage
from mast.logging.experiment_log import ExperimentLog


def _log(tmp_path) -> ExperimentLog:
    return ExperimentLog(ExperimentStorage(str(tmp_path / "exp.db")))


# ── default behaviour is UNCHANGED (create-always) ────────────────────────────

class TestDefaultUnchanged:
    def test_same_name_without_optin_still_creates_two(self, tmp_path):
        """Two starts without opt-in = two experiments, pointer on the new one.

        REVISED 2026-07-28: the old assertion was ``status == "superseded"``.
        Experiments no longer have a terminal state — 「没必要做归档。有的实验
        可能过了十年重启。」 Switching away leaves the row completely untouched
        so it can be switched back to at any time.
        """
        log = _log(tmp_path)
        e1 = log.start_experiment("NiI2")
        e2 = log.start_experiment("NiI2")           # no reuse_open → create-always
        assert e1 != e2
        assert log.current_experiment_id == e2
        # The one we left is untouched and still reachable.
        assert log._storage.get_experiment(e1)["end_time"] is None
        assert log.switch_experiment(e1).ok
        assert log.current_experiment_id == e1

    def test_sample_default_still_ends_and_recreates(self, tmp_path):
        """A second same-named sample without opt-in still creates a new row —
        but the previous sample is NOT ended (samples get swapped back in)."""
        log = _log(tmp_path)
        log.start_experiment("exp")
        s1 = log.start_sample("filmA")
        s2 = log.start_sample("filmA")              # no reuse_active → recreate
        assert s1 != s2
        assert log._storage.get_sample(s1)["end_time"] is None
        assert log.current_sample_id == s2
        assert log.switch_sample(s1).ok             # 切回上一块样品
        assert log.current_sample_id == s1


# ── idempotent reuse (opt-in) ─────────────────────────────────────────────────

class TestExperimentReuse:
    def test_same_name_reuses_within_session(self, tmp_path):
        log = _log(tmp_path)
        e1 = log.start_experiment("NiI2 quality", reuse_open=True)
        e2 = log.start_experiment("NiI2 quality", reuse_open=True)
        assert e1 == e2, "a re-issued same-name start must not duplicate"
        assert log._last_start_reused is True
        # only ONE experiment row exists
        assert len(log._storage.list_experiments()) == 1

    def test_case_and_whitespace_insensitive(self, tmp_path):
        log = _log(tmp_path)
        e1 = log.start_experiment("NiI2", reuse_open=True)
        e2 = log.start_experiment("  nii2  ", reuse_open=True)
        assert e1 == e2

    def test_different_name_creates_new(self, tmp_path):
        log = _log(tmp_path)
        e1 = log.start_experiment("NiI2", reuse_open=True)
        e2 = log.start_experiment("Au(111) clean", reuse_open=True)
        assert e1 != e2
        assert log._last_start_reused is False

    def test_cross_session_attaches_to_running_row(self, tmp_path):
        """A fresh ExperimentLog on the SAME storage (a restart) must attach to the
        still-running experiment, not create a 2nd — this is the 6-duplicates bug."""
        store = ExperimentStorage(str(tmp_path / "exp.db"))
        first = ExperimentLog(store)
        eid = first.start_experiment("NiI2 study", reuse_open=True)

        restarted = ExperimentLog(store)             # new session, current is None
        eid2 = restarted.start_experiment("NiI2 study", reuse_open=True)
        assert eid2 == eid
        assert restarted._last_start_reused is True
        assert len(store.list_experiments()) == 1

    def test_any_same_named_experiment_is_reused(self, tmp_path):
        """A same-named start reuses the row REGARDLESS of the row's age/status.

        REPLACES ``test_ended_experiment_is_not_reused`` (2026-07-28). That test
        asserted a completed run must not absorb a new same-named start — the
        premise is gone: there is no "completed". 「有的实验可能过了十年重启」,
        so a same-named start must attach to the existing experiment instead of
        minting the duplicates the field actually saw (5 identically-named rows
        two minutes apart).
        """
        log = _log(tmp_path)
        e1 = log.start_experiment("NiI2", reuse_open=True)
        log.end_experiment()                          # 现在只是清指针，不写终态
        assert log.current_experiment_id is None
        e2 = log.start_experiment("NiI2", reuse_open=True)
        assert e1 == e2, "同名 start 必须回到原来那条实验，而不是造一条新的"
        assert log._last_start_reused is True
        assert len(log._storage.list_experiments()) == 1

    def test_reuse_restores_active_sample(self, tmp_path):
        store = ExperimentStorage(str(tmp_path / "exp.db"))
        first = ExperimentLog(store)
        first.start_experiment("NiI2", reuse_open=True)
        sid = first.start_sample("film1")

        restarted = ExperimentLog(store)
        restarted.start_experiment("NiI2", reuse_open=True)
        assert restarted.current_sample_id == sid, "reuse must restore the active sample"


class TestSampleReuse:
    def test_same_name_active_sample_reused(self, tmp_path):
        log = _log(tmp_path)
        log.start_experiment("exp")
        s1 = log.start_sample("NiI2 film", reuse_active=True)
        s2 = log.start_sample("NiI2 film", reuse_active=True)
        assert s1 == s2
        assert log._last_sample_reused is True

    def test_different_sample_name_creates_new(self, tmp_path):
        log = _log(tmp_path)
        log.start_experiment("exp")
        s1 = log.start_sample("filmA", reuse_active=True)
        s2 = log.start_sample("filmB", reuse_active=True)
        assert s1 != s2
        assert log._last_sample_reused is False


# ── the resume primitive ──────────────────────────────────────────────────────

class TestResumePrimitive:
    def test_resume_attaches_without_creating(self, tmp_path):
        store = ExperimentStorage(str(tmp_path / "exp.db"))
        first = ExperimentLog(store)
        eid = first.start_experiment("run")
        sid = first.start_sample("s1")

        other = ExperimentLog(store)
        assert other.resume_experiment(eid) is True
        assert other.current_experiment_id == eid
        assert other.current_sample_id == sid
        assert len(store.list_experiments()) == 1, "resume must not create a row"

    def test_resume_unknown_id_returns_false(self, tmp_path):
        log = _log(tmp_path)
        assert log.resume_experiment("does-not-exist") is False
        assert log.current_experiment_id is None


# ── storage helper ────────────────────────────────────────────────────────────

class TestStorageHelper:
    def test_find_running_by_name_skips_ended(self, tmp_path):
        """Legacy helper keeps its status-filtered behaviour (still used by
        nothing on the hot path, kept so old callers don't change meaning)."""
        s = ExperimentStorage(str(tmp_path / "exp.db"))
        e1 = s.create_experiment("NiI2")
        s.end_experiment(e1, "completed")
        assert s.find_running_experiment_by_name("NiI2") is None
        e2 = s.create_experiment("NiI2")
        found = s.find_running_experiment_by_name("nii2")   # case-insensitive
        assert found is not None and found["id"] == e2

    def test_find_experiment_by_name_ignores_status(self, tmp_path):
        """The replacement helper finds a same-named row whatever its status.

        This is what makes 「做了一个月这个又回去做那个」 work: an experiment
        that some older code left marked 'completed' is still the experiment the
        operator means when they type its name.
        """
        s = ExperimentStorage(str(tmp_path / "exp.db"))
        e1 = s.create_experiment("NiI2")
        s.end_experiment(e1, "completed")           # 历史遗留的终态值
        found = s.find_experiment_by_name("nii2")   # case-insensitive + CJK-safe
        assert found is not None and found["id"] == e1

    def test_find_experiment_by_name_prefers_recently_active(self, tmp_path):
        s = ExperimentStorage(str(tmp_path / "exp.db"))
        e1 = s.create_experiment("dup")
        e2 = s.create_experiment("dup")
        s.set_active_scope(e1, None)                # e1 是最近动过的那条
        found = s.find_experiment_by_name("dup")
        assert found is not None and found["id"] == e1

    def test_find_running_by_name_empty_is_none(self, tmp_path):
        s = ExperimentStorage(str(tmp_path / "exp.db"))
        assert s.find_running_experiment_by_name("") is None


# ── agent-facing meta-tool surfaces reuse ─────────────────────────────────────

class TestMetaToolSurfacing:
    def _tools(self, log):
        from mast.agents._shared.meta_tools import make_meta_tools
        return {t.name: t for t in make_meta_tools(lambda: {"experiment_log": log})}

    def test_start_experiment_tool_reuses_and_reports(self, tmp_path):
        log = _log(tmp_path)
        tools = self._tools(log)
        r1 = json.loads(tools["start_experiment"].invoke({"name": "NiI2"}))
        r2 = json.loads(tools["start_experiment"].invoke({"name": "NiI2"}))
        assert r1["experiment_id"] == r2["experiment_id"], "tool created a duplicate"
        assert r2["reused"] is True
        assert "复用" in r2.get("message", "")
        assert len(log._storage.list_experiments()) == 1

    def test_start_sample_tool_reuses_and_reports(self, tmp_path):
        log = _log(tmp_path)
        tools = self._tools(log)
        tools["start_experiment"].invoke({"name": "exp"})
        r1 = json.loads(tools["start_sample"].invoke({"name": "film1"}))
        r2 = json.loads(tools["start_sample"].invoke({"name": "film1"}))
        assert r1["sample_id"] == r2["sample_id"]
        assert r2["reused"] is True
