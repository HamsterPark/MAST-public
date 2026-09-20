"""Tests for gui/records_api._experiments N+1 fix ().

The old code opened one SQLite connection *per experiment* (up to 500) to
count scan files. The fix issues a single GROUP BY pass over one connection.
These tests pin both the connection-count regression and correct mapping of
scan counts onto experiments. No real DB / network — a fake store records how
many times ``connect()`` is invoked and serves canned rows.
"""
from __future__ import annotations

# ── path bootstrap: MASTv2/ must win over any v1 mast on sys.path ──
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found above " + str(Path(__file__).resolve()))


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from contextlib import contextmanager  # noqa: E402

import pytest  # noqa: E402

import mast.webui.records_api as records_api  # noqa: E402

assert "MASTv2" in records_api.__file__.replace("\\", "/"), (
    f"records_api resolved to v1 path: {records_api.__file__}"
)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Returns the GROUP BY scan-count rows for the one expected query."""

    def __init__(self, scan_rows):
        self._scan_rows = scan_rows

    def execute(self, sql, params=()):
        # _experiments issues exactly one GROUP BY query (no params).
        assert "GROUP BY a.experiment_id" in sql, sql
        return _FakeCursor(self._scan_rows)


class _FakeStore:
    def __init__(self, scan_rows):
        self._scan_rows = scan_rows
        self.connect_calls = 0

    @contextmanager
    def connect(self):
        self.connect_calls += 1
        yield _FakeConn(self._scan_rows)


class _FakeRepos:
    def __init__(self, store):
        self.store = store


def _exp(eid, **over):
    base = {
        "id": eid,
        "campaign_id": "camp-1",
        "sample_id": "samp-1",
        "title": f"exp {eid}",
        "exp_type": "stm",
        "action_count": 3,
        "observation_count": 2,
    }
    base.update(over)
    return base


class TestExperimentsN1:
    def test_single_connection_regardless_of_count(self):
        """#68: 200 experiments must NOT open 200 connections."""
        experiments = [_exp(f"e{i}") for i in range(200)]
        scan_rows = [{"eid": "e0", "c": 5}, {"eid": "e7", "c": 2}]
        store = _FakeStore(scan_rows)
        repos = _FakeRepos(store)

        out = records_api._experiments(repos, experiments)

        assert store.connect_calls == 1, (
            f"expected 1 connection for the GROUP BY, got {store.connect_calls}"
        )
        assert len(out) == 200

    def test_scan_counts_mapped_correctly(self):
        experiments = [_exp("e0"), _exp("e1"), _exp("e2")]
        scan_rows = [{"eid": "e0", "c": 5}, {"eid": "e2", "c": 9}]
        store = _FakeStore(scan_rows)
        out = records_api._experiments(_FakeRepos(store), experiments)

        by_id = {e["id"]: e for e in out}
        assert by_id["e0"]["scan_count"] == 5
        assert by_id["e1"]["scan_count"] == 0  # absent → default 0
        assert by_id["e2"]["scan_count"] == 9

    def test_empty_experiments_one_query(self):
        store = _FakeStore([])
        out = records_api._experiments(_FakeRepos(store), [])
        # Still one connection (the GROUP BY runs once), zero rows out.
        assert store.connect_calls == 1
        assert out == []

    def test_other_fields_preserved(self):
        experiments = [_exp("e0", action_count=7, observation_count=4)]
        store = _FakeStore([{"eid": "e0", "c": 1}])
        out = records_api._experiments(_FakeRepos(store), experiments)
        assert out[0]["action_count"] == 7
        assert out[0]["obs_count"] == 4
        assert out[0]["scan_count"] == 1
