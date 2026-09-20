"""Corpus export, and above all the weak-label alignment guard.

The vision journal timestamps are ``time.monotonic_ns()``. Treating them as
wall-clock offline yields labels that look reasonable and sit on the wrong
segments — a failure that is invisible until a model trained on them behaves
strangely. These tests pin the refusal: no anchor, no weak labels.
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

import json
import sqlite3
import time

import numpy as np
import pytest

from mast.monitoring import features as F
from mast.monitoring.export import collect_weak_labels, export_corpus
from mast.monitoring.store import CurrentMonitorStore, segment_npy_path

FS = 20000.0


@pytest.fixture()
def store(tmp_path):
    s = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path / "data")
    yield s
    s.close()


def _seed(store, *, t_start, pinned=True, label=None, with_file=True) -> int:
    n = int(FS)
    y = 100e-12 + np.random.default_rng(int(t_start) % 997).normal(0, 2e-12, n)
    npy_path, nbytes = None, 0
    if with_file:
        p = segment_npy_path(store.data_dir, t_start, FS)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, y.astype(np.float32))
        npy_path, nbytes = str(p), p.stat().st_size
    sid = store.add_segment(
        {"t_start": t_start, "t_end": t_start + 1, "fs_hz": FS, "n_samples": n,
         "npy_path": npy_path, "npy_bytes": nbytes, "pinned": pinned,
         "pin_reason": "test", "channel_name": "Current (A)"},
        F.envelope(y, FS, 100).tobytes(), 0.01)
    feats = F.compute_segment_features([y], FS)
    feats["t_start"] = t_start
    store.add_features(sid, feats, {"ctx_skill": ""}, "ok")
    if label:
        store.add_label(t_start=t_start, t_end=t_start + 1, label=label,
                        segment_id=sid)
    return sid


def _make_wal(path: Path, *, rows, anchor_payload=None) -> None:
    """A minimal stand-in for the vision buffer WAL."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE tip_status_journal (seqno INTEGER PRIMARY KEY, "
                 "t_mono_ns INTEGER NOT NULL, quality TEXT NOT NULL, "
                 "confidence REAL NOT NULL, scan_id TEXT, frame_idx INTEGER)")
    conn.execute("CREATE TABLE event_journal (event_id TEXT PRIMARY KEY, "
                 "seqno INTEGER NOT NULL, t_mono_ns INTEGER NOT NULL, "
                 "kind TEXT NOT NULL, severity TEXT NOT NULL, payload_json TEXT)")
    for i, (t_mono_ns, quality) in enumerate(rows):
        conn.execute("INSERT INTO tip_status_journal VALUES (?,?,?,?,?,?)",
                     (i + 1, t_mono_ns, quality, 0.9, "scan-1", i))
    if anchor_payload is not None:
        conn.execute("INSERT INTO event_journal VALUES (?,?,?,?,?,?)",
                     ("e1", 99, anchor_payload[0], "tip_quality_drop", "critical",
                      json.dumps(anchor_payload[1])))
    conn.commit()
    conn.close()


# ── export ───────────────────────────────────────────────────────────────────

def test_export_writes_waveforms_sidecars_and_dataset(store, tmp_path):
    now = time.time()
    _seed(store, t_start=now - 2, label="good")
    _seed(store, t_start=now - 1, label="bad")
    out = tmp_path / "corpus"

    manifest = export_corpus(out_dir=out, store=store, fmt="jsonl")

    assert manifest["segments"] == 2
    assert manifest["waveforms_copied"] == 2
    assert manifest["human_labels"] == 2
    assert (out / "manifest.json").is_file()
    assert (out / "dataset.jsonl").is_file()
    npys = list((out / "segments").glob("*.npy"))
    sidecars = list((out / "segments").glob("*.json"))
    assert len(npys) == 2 and len(sidecars) == 2

    rec = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert rec["fs_hz"] == FS
    assert rec["label_human"] in ("good", "bad")
    assert rec["rms_detrended_a"] is not None      # features travel with the data
    assert rec["npy"].startswith("segments/")


def test_export_defaults_to_pinned_only(store, tmp_path):
    now = time.time()
    _seed(store, t_start=now - 2, pinned=True)
    _seed(store, t_start=now - 1, pinned=False)

    pinned = export_corpus(out_dir=tmp_path / "a", store=store, fmt="jsonl")
    assert pinned["segments"] == 1

    every = export_corpus(out_dir=tmp_path / "b", store=store, fmt="jsonl",
                          pinned_only=False)
    assert every["segments"] == 2


def test_export_records_segments_whose_waveform_was_swept(store, tmp_path):
    """A swept segment still carries features and a label — it belongs in the
    manifest, just without an .npy."""
    now = time.time()
    _seed(store, t_start=now, with_file=False, label="bad")
    m = export_corpus(out_dir=tmp_path / "c", store=store, fmt="jsonl")
    assert m["segments"] == 1
    assert m["waveforms_copied"] == 0
    assert m["waveforms_missing"] == 1
    rec = json.loads((tmp_path / "c" / "dataset.jsonl").read_text().strip())
    assert rec["npy"] == "" and rec["label_human"] == "bad"


def test_export_time_range(store, tmp_path):
    now = time.time()
    for i in range(5):
        _seed(store, t_start=now + i)
    m = export_corpus(out_dir=tmp_path / "d", store=store, fmt="jsonl",
                      since=now + 1, until=now + 3)
    assert m["segments"] == 2


def test_export_on_an_empty_store_is_an_empty_manifest(store, tmp_path):
    m = export_corpus(out_dir=tmp_path / "e", store=store, fmt="jsonl")
    assert m["segments"] == 0
    assert (tmp_path / "e" / "manifest.json").is_file()


def test_parquet_is_written_when_pandas_is_available(store, tmp_path):
    _seed(store, t_start=time.time())
    m = export_corpus(out_dir=tmp_path / "f", store=store, fmt="parquet")
    assert "dataset.jsonl" in m["files"]           # always
    if "dataset.parquet" in m["files"]:
        import pandas as pd
        df = pd.read_parquet(tmp_path / "f" / "dataset.parquet")
        assert len(df) == 1 and "rms_detrended_a" in df.columns


# ── weak labels ──────────────────────────────────────────────────────────────

def test_no_weak_labels_without_a_wall_clock_anchor(store, tmp_path):
    """The whole point: monotonic stamps cannot be converted offline, so the
    exporter refuses rather than inventing an offset."""
    wal = tmp_path / "vision.sqlite"
    _make_wal(wal, rows=[(1_000_000_000, "bad"), (2_000_000_000, "good")])
    assert collect_weak_labels(wal) == []

    _seed(store, t_start=time.time())
    m = export_corpus(out_dir=tmp_path / "g", store=store, fmt="jsonl",
                      weak_labels=True, wal_path=wal)
    assert m["weak_labels_attached"] == 0
    rec = json.loads((tmp_path / "g" / "dataset.jsonl").read_text().strip())
    assert rec["label_weak"] is None


def test_weak_labels_align_through_a_frame_path_anchor(store, tmp_path):
    now = time.time()
    frames = tmp_path / "frames"
    frames.mkdir()
    # A frame written at `now`, named by the repo's epoch-millisecond convention.
    frame = frames / f"frame_ch0_dir0_{int(now * 1000):013d}.png"
    frame.write_bytes(b"png")

    mono_at_now = 5_000_000_000
    wal = tmp_path / "vision.sqlite"
    _make_wal(wal,
              rows=[(mono_at_now, "bad"), (mono_at_now + 10_000_000_000, "good")],
              anchor_payload=(mono_at_now, {"frame_path": str(frame)}))

    labels = collect_weak_labels(wal)
    assert len(labels) == 2
    assert labels[0]["ts"] == pytest.approx(now, abs=1.0)
    assert labels[0]["align_method"] == "frame_anchor"
    # The second row sits 10 s past the anchor; its placement is just as precise
    # (both clocks advance at the same rate), so the error bound must not grow.
    assert labels[1]["ts"] == pytest.approx(now + 10.0, abs=1.0)
    assert labels[1]["align_error_bound_s"] == labels[0]["align_error_bound_s"]
    assert labels[1]["anchor_distance_s"] == pytest.approx(10.0)

    _seed(store, t_start=now)
    m = export_corpus(out_dir=tmp_path / "h", store=store, fmt="jsonl",
                      weak_labels=True, wal_path=wal)
    assert m["weak_labels_attached"] == 1
    rec = json.loads((tmp_path / "h" / "dataset.jsonl").read_text().strip())
    assert rec["label_weak"] == "bad"
    assert rec["label_weak_align_method"] == "frame_anchor"
    assert rec["label_weak_align_error_s"] == pytest.approx(0.0, abs=1.0)


def test_a_weak_label_never_overwrites_a_human_one(store, tmp_path):
    now = time.time()
    frames = tmp_path / "frames"
    frames.mkdir()
    frame = frames / f"f_{int(now * 1000):013d}.png"
    frame.write_bytes(b"png")
    wal = tmp_path / "vision.sqlite"
    _make_wal(wal, rows=[(1_000, "bad")],
              anchor_payload=(1_000, {"frame_path": str(frame)}))

    _seed(store, t_start=now, label="good")        # human says good
    export_corpus(out_dir=tmp_path / "i", store=store, fmt="jsonl",
                  weak_labels=True, wal_path=wal)
    rec = json.loads((tmp_path / "i" / "dataset.jsonl").read_text().strip())
    assert rec["label_human"] == "good"            # separate columns, no clobber
    assert rec["label_weak"] == "bad"


def test_far_away_journal_rows_are_not_attached(store, tmp_path):
    now = time.time()
    frames = tmp_path / "frames"
    frames.mkdir()
    frame = frames / f"f_{int(now * 1000):013d}.png"
    frame.write_bytes(b"png")
    wal = tmp_path / "vision.sqlite"
    # a verdict 200 s away from the anchor — inside the anchor window, but far
    # from any segment
    _make_wal(wal, rows=[(1_000 + 200_000_000_000, "bad")],
              anchor_payload=(1_000, {"frame_path": str(frame)}))

    _seed(store, t_start=now)
    m = export_corpus(out_dir=tmp_path / "j", store=store, fmt="jsonl",
                      weak_labels=True, wal_path=wal)
    assert m["weak_labels_attached"] == 0


def test_missing_wal_file_is_not_an_error(tmp_path):
    assert collect_weak_labels(tmp_path / "nope.sqlite") == []


def test_a_thirteen_digit_run_that_is_not_a_timestamp_is_rejected(tmp_path):
    from mast.monitoring.export import _wall_clock_from_path
    assert _wall_clock_from_path("/no/such/file_9999999999999.png") is None


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_runs_end_to_end(store, tmp_path, monkeypatch, capsys):
    from mast.monitoring import export as E
    from mast.monitoring import store as ST

    _seed(store, t_start=time.time(), label="good")
    monkeypatch.setattr(ST, "set_store_for_test", ST.set_store_for_test)
    ST.set_store_for_test(store)
    try:
        rc = E.main(["--out", str(tmp_path / "cli"), "--format", "jsonl"])
    finally:
        ST.set_store_for_test(None)

    assert rc == 0
    out = capsys.readouterr().out
    assert "导出完成" in out
    assert (tmp_path / "cli" / "dataset.jsonl").is_file()
    assert "人工标签 1" in out
