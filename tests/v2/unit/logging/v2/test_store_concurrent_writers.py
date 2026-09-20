"""The v2 store must not lose a row to a write-lock collision.

WAL permits exactly one writer at a time. That did not matter much while the
only v2 writers were the trace-sink worker and one post_hook; since 2026-07-27
every skill call writes its action row from the calling thread while the sink
worker writes the matching step from its own, so collisions are ordinary. A
dropped row here is invisible — the write is fire-and-forget by design — which
is the exact failure mode the 07-27 forensics is about.

NO production change backs these tests. The behaviour is already correct, but
only by accident of a default: ``sqlite3.connect``'s ``timeout`` argument IS the
busy timeout and defaults to 5 s, and ``ExperimentStoreV2.connect`` never passes
it. An explicit ``PRAGMA busy_timeout`` would be pure decoration (verified: with
and without it these tests behave identically). Passing ``timeout=0`` there
turns both tests red — which is precisely why the invariant is worth pinning
before somebody "tunes" that call.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/logging/v2/test_store_concurrent_writers.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import sqlite3  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from mast.logging.v2.repos import build_repos  # noqa: E402
from mast.logging.v2.storage import ExperimentStoreV2  # noqa: E402

_HOLD_S = 0.4


@pytest.fixture()
def repos_and_experiment(tmp_path):
    store = ExperimentStoreV2(tmp_path / "v2.db")
    repos = build_repos(store)
    cid = repos.campaigns.create(title="c", hypothesis="", hypothesis_kind="exploratory",
                                 goal={}, created_by="test")
    sid = repos.samples.create(label="s", material="")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid, title="e",
                                  exp_type="ad_hoc")
    return repos, eid, tmp_path / "v2.db"


def test_a_second_writer_waits_instead_of_losing_the_row(repos_and_experiment):
    repos, eid, db_path = repos_and_experiment

    # The lock holder lives entirely on its own thread — a sqlite3 connection
    # may only be touched by the thread that created it.
    lock_taken = threading.Event()

    def _hold():
        h = sqlite3.connect(str(db_path))
        try:
            h.execute("PRAGMA journal_mode = WAL")
            h.execute("BEGIN IMMEDIATE")         # take the single write lock
            lock_taken.set()
            time.sleep(_HOLD_S)
            h.rollback()
        finally:
            h.close()

    t = threading.Thread(target=_hold, daemon=True)
    t.start()
    assert lock_taken.wait(5.0), "could not acquire the write lock to hold"
    try:
        t0 = time.monotonic()
        aid = repos.actions.begin(experiment_id=eid, agent_id="instrument_control",
                                  action_type="StartScan", params={"bias_v": -1.0})
        waited = time.monotonic() - t0
    finally:
        t.join(timeout=10.0)

    assert aid, "the action row must survive a write-lock collision"
    assert waited >= _HOLD_S * 0.7, "it must have WAITED for the lock, not raced past it"
    assert repos.actions.get(aid)["action_type"] == "StartScan"


def test_concurrent_writers_all_land(repos_and_experiment):
    """8 threads × 10 action rows — none may vanish."""
    repos, eid, _ = repos_and_experiment
    errors: list[BaseException] = []

    def _writer(n: int) -> None:
        try:
            for i in range(10):
                repos.actions.begin(experiment_id=eid, agent_id="instrument_control",
                                    action_type=f"S{n}", params={"i": i})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert not errors, f"writes were rejected: {errors[:3]}"
    assert len(repos.actions.for_experiment(eid, limit=1000)) == 80


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
