"""Tests for the one-click full-history export (``mast.logging.export_all``).

Each test points ``MAST2_PROJECT_ROOT`` at a temp dir populated with fake
experiments/ databases + logs and artifacts/ (a small data product, a heavy
blob, a heavy cache dir), then asserts the zip bundles the right things and
that the snapshotted DB is a valid, complete copy.
"""
from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest


def _make_db(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany("INSERT INTO t (v) VALUES (?)",
                         [(f"row{i}",) for i in range(rows)])
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    exp = tmp_path / "experiments"
    exp.mkdir()
    _make_db(exp / "mast_experiments_v2.db", rows=5)
    _make_db(exp / "mast_experiments.db", rows=2)
    (exp / "chat_history.jsonl").write_text('{"role":"user"}\n', encoding="utf-8")
    (exp / ".mast_gui.lock").write_text("1", encoding="utf-8")
    (exp / "logs").mkdir()
    (exp / "logs" / "run.log").write_text("hello", encoding="utf-8")
    art = tmp_path / "artifacts"
    (art / "current_traces").mkdir(parents=True)
    (art / "current_traces" / "t1.csv").write_text(
        "t_s,current_a\n0,1\n", encoding="utf-8")
    (art / "model.pt").write_bytes(b"\x00" * 1024)        # heavy suffix
    (art / "vision_backbone").mkdir()
    (art / "vision_backbone" / "blob.bin").write_bytes(b"\x00" * 2048)  # heavy dir
    return tmp_path


def test_export_light_mode(fake_root):
    from mast.logging.export_all import export_all_history

    dest = fake_root / "exports" / "x.zip"
    man = export_all_history(dest)
    assert dest.exists()
    names = set(zipfile.ZipFile(dest).namelist())

    # DBs snapshotted under databases/
    assert "databases/mast_experiments_v2.db" in names
    assert "databases/mast_experiments.db" in names
    # experiments non-db files included, db sidecars + lock excluded
    assert "experiments/chat_history.jsonl" in names
    assert "experiments/logs/run.log" in names
    assert "experiments/.mast_gui.lock" not in names
    # small data product kept; heavy blob + heavy dir pruned in light mode
    assert "artifacts/current_traces/t1.csv" in names
    assert "artifacts/model.pt" not in names
    assert not any(n.startswith("artifacts/vision_backbone") for n in names)
    # manifest present + accurate
    assert "MANIFEST.json" in names
    parsed = json.loads(zipfile.ZipFile(dest).read("MANIFEST.json"))
    assert parsed["include_heavy"] is False
    assert len(parsed["databases"]) >= 2
    assert any(s["reason"] == "heavy-suffix" for s in parsed["skipped"])
    assert man["file_count"] == len(parsed["files"])


def test_exported_db_is_valid_snapshot(fake_root):
    from mast.logging.export_all import export_all_history

    dest = fake_root / "exports" / "x.zip"
    export_all_history(dest)
    out = fake_root / "extracted"
    with zipfile.ZipFile(dest) as zf:
        zf.extract("databases/mast_experiments_v2.db", out)
    conn = sqlite3.connect(str(out / "databases" / "mast_experiments_v2.db"))
    try:
        n = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    finally:
        conn.close()
    assert n == 5  # all rows survived the online backup


def test_export_heavy_mode_includes_blobs(fake_root):
    from mast.logging.export_all import export_all_history

    dest = fake_root / "exports" / "heavy.zip"
    man = export_all_history(dest, include_heavy=True)
    names = set(zipfile.ZipFile(dest).namelist())
    assert "artifacts/model.pt" in names
    assert any(n.startswith("artifacts/vision_backbone") for n in names)
    assert man["include_heavy"] is True


def test_progress_callback_invoked(fake_root):
    from mast.logging.export_all import export_all_history

    seen: list[str] = []
    export_all_history(fake_root / "exports" / "p.zip",
                       progress=seen.append)
    assert seen
    assert any(("数据库" in m) or ("打包" in m) for m in seen)


def test_missing_dirs_do_not_crash(tmp_path, monkeypatch):
    # A pristine project root with no experiments/ or artifacts/ must still
    # produce a valid (near-empty) zip rather than raising.
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    from mast.logging.export_all import export_all_history

    dest = tmp_path / "exports" / "empty.zip"
    man = export_all_history(dest)
    assert dest.exists()
    assert "MANIFEST.json" in set(zipfile.ZipFile(dest).namelist())
    assert man["databases"] == []
