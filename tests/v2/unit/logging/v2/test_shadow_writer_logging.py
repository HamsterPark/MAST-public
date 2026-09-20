"""Logging-group tests for the ShadowLogger v2 runtime writer (finding [29]).

[29] notes the entire v2 logging schema is read but never written by any live
code path — ShadowLogger is the documented strangler-fig write adapter but has
no caller, so the v2 store stays empty and the Records UI silently shows demo
mock.

The *wiring* of ShadowLogger into the executor / orchestrator (or making
records_api surface an honest empty state) lives in files owned by other
parallel agents (core/executor.py, gui/records_api.py) and is therefore left to
the integration step — see the report. What these tests DO cover, entirely
inside logging/v2/, is that ShadowLogger is a genuinely functional v2 writer:

  * its v2 side really persists experiments / actions / observations into a live
    ExperimentStoreV2 (so once wired, the Records tab gets real data); and
  * the end_experiment v1-side signature bug (it called the v1
    ExperimentLog.end_experiment(id, status) with TWO positional args while the
    real v1 method takes only `status`) is fixed — that TypeError was a concrete
    blocker to ever adopting ShadowLogger as the live writer.

Run from MASTv2:
    ../.venv-v2-py313/Scripts/python.exe -m pytest \
        ../tests/v2/unit/logging/v2/test_shadow_writer_logging.py -v
"""
from __future__ import annotations

# ── Force MASTv2/ to the head of sys.path BEFORE importing mast.* ─────
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from dataclasses import dataclass

import pytest

from mast.logging.v2.repos import build_repos
from mast.logging.v2.shadow import ShadowConfig, ShadowLogger
from mast.logging.v2.storage import ExperimentStoreV2


# ── Minimal stand-ins for a v1 ActionRecord / result / state ─────────

@dataclass
class _FakeResult:
    success: bool = True
    elapsed_s: float = 0.05
    error: str | None = None


@dataclass
class _FakeStateAfter:
    bias_v: float | None = None
    current_a: float | None = None
    z_pos_m: float | None = None
    setpoint_a: float | None = None


@dataclass
class _FakeActionRecord:
    id: str = "v1-action-1"
    approval_source: str = "llm"
    skill_name: str = "SetBias"
    parameters: dict = None
    result: _FakeResult = None
    state_after: _FakeStateAfter = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {"bias_v": -2.0}


class _RecordingV1Log:
    """Stands in for the v1 ExperimentLog. Records exactly how it was called so
    we can prove ShadowLogger uses the REAL v1 signatures (no TypeError)."""

    def __init__(self):
        self.started: list[tuple] = []
        self.ended: list[tuple] = []
        self.actions: list = []

    def start_experiment(self, name, goal=""):
        self.started.append((name, goal))
        return "v1-exp-1"

    def end_experiment(self, status="completed"):
        # Real v1 takes exactly ONE positional arg (status). If ShadowLogger
        # passed an id positionally (the old bug) this would be called with 2
        # args and raise TypeError before we ever recorded it.
        self.ended.append((status,))

    def log_skill_execution(self, record):
        self.actions.append(record)


# ── Tests ─────────────────────────────────────────────────────────────

def _make_v2_repos(tmp_path):
    store = ExperimentStoreV2(tmp_path / "shadow_v2.db")
    return store, build_repos(store)


def test_shadow_v2_side_persists_real_rows(tmp_path):
    """The v2 side of ShadowLogger genuinely writes experiments/actions/
    observations into a live ExperimentStoreV2 — the data the Records UI would
    render once a caller is wired in."""
    store, repos = _make_v2_repos(tmp_path)
    cfg = ShadowConfig(write_v1=False, write_v2=True)
    sl = ShadowLogger(v1_log=None, v2_repos=repos, cfg=cfg)

    ids = sl.start_experiment(name="Au(111) shadow test", sample_label="Au sample")
    exp_v2 = ids["v2"]
    assert exp_v2

    rec = _FakeActionRecord(
        result=_FakeResult(success=True, elapsed_s=0.1),
        state_after=_FakeStateAfter(bias_v=-1.5, current_a=1e-10, z_pos_m=2e-7),
    )
    out = sl.log_action(v1_record=rec, experiment_id_v2=exp_v2)
    assert "v2" in out

    sl.end_experiment(v2_experiment_id=exp_v2, exit_status="success",
                      conclusion="done")

    counts = store.table_counts()
    assert counts["experiments"] == 1, counts
    assert counts["actions"] == 1, counts
    # 3 non-None scalar state fields -> 3 observations
    assert counts["observations"] == 3, counts

    # The experiment is closed with the right exit status.
    exp = repos.experiments.get(exp_v2)
    assert exp["exit_status"] == "success"
    assert exp["conclusion"] == "done"
    # The action landed succeeded.
    actions = repos.actions.for_experiment(exp_v2)
    assert len(actions) == 1
    assert actions[0]["status"] == "succeeded"


def test_shadow_end_experiment_uses_correct_v1_signature(tmp_path):
    """Regression: ShadowLogger.end_experiment must call the v1 log with ONE
    positional arg (status), not (id, status). The old code raised TypeError on
    every real v1-enabled call — proving it was never an adoptable live writer.
    """
    store, repos = _make_v2_repos(tmp_path)
    v1 = _RecordingV1Log()
    cfg = ShadowConfig(write_v1=True, write_v2=True)
    sl = ShadowLogger(v1_log=v1, v2_repos=repos, cfg=cfg)

    ids = sl.start_experiment(name="dual-write test")
    assert ids["v1"] == "v1-exp-1"
    assert "v2" in ids

    # This used to raise TypeError: end_experiment() takes 1..2 positional args
    # but 3 were given. It must now succeed.
    sl.end_experiment(v1_experiment_id="v1-exp-1", v2_experiment_id=ids["v2"],
                      exit_status="success")

    # v1 was called with exactly one positional arg, mapped to v1 vocabulary.
    assert v1.ended == [("completed",)], v1.ended

    # aborted maps onto v1 'aborted'.
    ids2 = sl.start_experiment(name="dual-write test 2")
    sl.end_experiment(v1_experiment_id="v1-exp-1", v2_experiment_id=ids2["v2"],
                      exit_status="aborted")
    assert v1.ended[-1] == ("aborted",)


def test_shadow_dual_write_keeps_v1_v2_mapping(tmp_path):
    """When both sides are enabled, the action mirrors into v2 and the v2
    experiment is closeable via the remembered v1->v2 mapping (no explicit v2
    id needed)."""
    store, repos = _make_v2_repos(tmp_path)
    v1 = _RecordingV1Log()
    cfg = ShadowConfig(write_v1=True, write_v2=True)
    sl = ShadowLogger(v1_log=v1, v2_repos=repos, cfg=cfg)

    ids = sl.start_experiment(name="mapping test")
    rec = _FakeActionRecord(result=_FakeResult(success=False, error="tip crash",
                                               elapsed_s=1.0),
                            state_after=_FakeStateAfter(bias_v=0.5))
    sl.log_action(v1_record=rec, experiment_id_v2=ids["v2"])

    # Close via the v1 id only — ShadowLogger resolves it to the v2 id.
    sl.end_experiment(v1_experiment_id=ids["v1"], exit_status="success")

    exp = repos.experiments.get(ids["v2"])
    assert exp["ended_at"] is not None
    actions = repos.actions.for_experiment(ids["v2"])
    assert len(actions) == 1
    assert actions[0]["status"] == "failed"
    assert actions[0]["error"] == "tip crash"
    # 1 non-None scalar -> 1 observation
    assert store.table_counts()["observations"] == 1
