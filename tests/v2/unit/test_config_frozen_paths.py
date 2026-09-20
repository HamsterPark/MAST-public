"""Frozen-path safety for MASTConfig — the test class that was MISSING and let the
installed-app "实验记录不可用" bug ship (2026-06-29).

The installed launcher sets MAST2_PROJECT_ROOT (the user-data root) and chdir's to
it, but not every process/thread that touches storage inherits that cwd — the
``--service-mode`` instance runs with cwd = C:\\Windows\\System32. db_path /
experiments_dir were bare RELATIVE paths, so they resolved against that wrong cwd
→ ExperimentStorage couldn't create the DB → the agent got "实验记录不可用" and the
records/vision panels were empty. The fix makes them resolve against
``_project_root()`` (which honours MAST2_PROJECT_ROOT), never cwd.

Dev unit tests passed all along because in dev cwd == repo root, so the relative
path happened to resolve correctly — exactly why this needed a CWD-independent test.
"""
from __future__ import annotations

from pathlib import Path

from mast.config import MASTConfig
from mast.logging.experiment_log import ExperimentLog
from mast.logging.storage import ExperimentStorage


def test_db_path_resolves_to_env_root_not_cwd(tmp_path, monkeypatch):
    root = tmp_path / "dataroot"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(root))
    monkeypatch.chdir(elsewhere)  # process cwd != data root (the frozen failure mode)

    cfg = MASTConfig()
    db = Path(str(cfg.db_path)).resolve()
    exp = Path(str(cfg.experiments_dir)).resolve()

    assert str(root.resolve()) in str(db), f"db_path {db} not under env root {root}"
    assert str(elsewhere.resolve()) not in str(db), "db_path followed cwd — frozen bug"
    assert str(root.resolve()) in str(exp), f"experiments_dir {exp} not under env root"


def test_storage_builds_under_env_root_with_foreign_cwd(tmp_path, monkeypatch):
    # End-to-end: the exact failure — build ExperimentStorage from cfg.db_path while
    # cwd is a foreign dir. It must create the DB under the env root (writable), not
    # fail, so start_experiment works (→ NOT "实验记录不可用").
    root = tmp_path / "dataroot"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(root))
    monkeypatch.chdir(elsewhere)

    cfg = MASTConfig()
    st = ExperimentStorage(cfg.db_path)
    el = ExperimentLog(st)
    eid = el.start_experiment("frozen-path-regression", "verify writable under env root")

    db = Path(str(cfg.db_path)).resolve()
    assert eid  # start_experiment succeeded
    assert db.exists() and str(root.resolve()) in str(db)
