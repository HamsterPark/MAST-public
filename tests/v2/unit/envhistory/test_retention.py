"""Retention: the only code in this subsystem that deletes anything.

Three properties get hard coverage, because getting any of them wrong destroys
data that cannot be recovered:

  * alarm/error readings are NEVER pruned — they are the evidence behind an
    ``_on_env_alarm`` shutdown and behind every post-mortem;
  * the cutoff comparison actually works. ``environment_log.timestamp`` is a
    LOCAL naive ISO string, so a UTC or tz-aware cutoff would compare
    lexicographically against a different format — failing either open (deleting
    too much) or shut (deleting nothing while reporting success);
  * deletion is batched, so it never holds the write lock long enough to stall
    the experiment log sharing the same database file.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
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

from datetime import datetime, timedelta

import pytest

from mast.logging.storage import ExperimentStorage


@pytest.fixture()
def storage(tmp_path):
    return ExperimentStorage(tmp_path / "exp.db")


def _seed(storage, *, days_ago: float, status: str = "ok",
          sensor: str = "temperature", n: int = 1) -> None:
    """Insert rows with an explicit timestamp (log_environment stamps `now`)."""
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    with storage._connect() as conn:                    # noqa: SLF001 — test seam
        conn.executemany(
            "INSERT INTO environment_log"
            " (timestamp, sensor_name, value, unit, status) VALUES (?,?,?,?,?)",
            [(ts, sensor, 77.0, "K", status)] * n,
        )


def _count(storage, **where) -> int:
    q = "SELECT COUNT(*) AS n FROM environment_log"
    params: list = []
    if where.get("status"):
        q += " WHERE status = ?"
        params.append(where["status"])
    with storage._connect() as conn:                    # noqa: SLF001
        return int(conn.execute(q, params).fetchone()["n"])


def _cutoff(days: float) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat()


def test_old_ok_rows_are_pruned(storage):
    _seed(storage, days_ago=30, n=50)
    _seed(storage, days_ago=1, n=10)
    deleted = storage.prune_environment_log(_cutoff(14))
    assert deleted == 50
    assert _count(storage) == 10


def test_alarm_and_error_rows_are_exempt_forever(storage):
    _seed(storage, days_ago=365, status="alarm", n=3)
    _seed(storage, days_ago=365, status="error", n=2)
    _seed(storage, days_ago=365, status="ok", n=40)
    deleted = storage.prune_environment_log(_cutoff(14))
    assert deleted == 40
    assert _count(storage, status="alarm") == 3
    assert _count(storage, status="error") == 2


def test_warning_rows_are_prunable_by_default(storage):
    """`warning` is advisory and high-volume; the buckets keep its statistics."""
    _seed(storage, days_ago=30, status="warning", n=5)
    assert storage.prune_environment_log(_cutoff(14)) == 5


def test_keep_statuses_is_configurable(storage):
    _seed(storage, days_ago=30, status="warning", n=5)
    assert storage.prune_environment_log(
        _cutoff(14), keep_statuses=("alarm", "error", "warning")) == 0


def test_cutoff_format_matches_what_log_environment_writes(storage):
    """Regression guard for the format mismatch this comparison invites.

    ``log_environment`` stamps ``datetime.now().isoformat()`` — local, naive.
    Comparing against a UTC or tz-aware string is a lexicographic comparison
    between different formats, and it fails SILENTLY.
    """
    storage.log_environment("temperature", 77.0, "K", "ok")
    # A cutoff far in the future must sweep the row just written.
    future = (datetime.now() + timedelta(days=1)).isoformat()
    assert storage.prune_environment_log(future) == 1
    # …and a cutoff in the past must not.
    storage.log_environment("temperature", 77.0, "K", "ok")
    assert storage.prune_environment_log(_cutoff(1)) == 0
    assert _count(storage) == 1


def test_deletion_is_batched(storage):
    _seed(storage, days_ago=30, n=25)
    deleted = storage.prune_environment_log(_cutoff(14), batch_rows=10,
                                            pause_s=0.0)
    assert deleted == 25
    assert _count(storage) == 0


def test_max_batches_bounds_a_single_sweep(storage):
    """A sweep must not run unbounded — it shares the DB with the experiment
    log, and the next tick will finish the job anyway."""
    _seed(storage, days_ago=30, n=25)
    deleted = storage.prune_environment_log(_cutoff(14), batch_rows=10,
                                            max_batches=1, pause_s=0.0)
    assert deleted == 10
    assert _count(storage) == 15


def test_nothing_to_do_is_zero_not_an_error(storage):
    assert storage.prune_environment_log(_cutoff(14)) == 0


def test_scoped_history_still_readable_after_a_sweep(storage):
    """The API's raw→buckets fallback keys off "no rows", so a sweep must
    genuinely leave none rather than leave unreadable ones."""
    _seed(storage, days_ago=30, n=5)
    _seed(storage, days_ago=0.5, n=5)
    storage.prune_environment_log(_cutoff(14))
    rows = storage.get_environment_history("temperature")
    assert len(rows) == 5
    assert all(r["status"] == "ok" for r in rows)


def test_prune_leaves_other_tables_alone(storage):
    exp_id = storage.create_experiment("t", "desc") if hasattr(
        storage, "create_experiment") else None
    _seed(storage, days_ago=30, n=5)
    storage.prune_environment_log(_cutoff(14))
    if exp_id:
        assert storage.get_experiment(exp_id) is not None
