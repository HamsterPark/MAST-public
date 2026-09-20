"""RunStatsStore — the per-agent-type duration learner behind smarter auto-
backgrounding (item ②).

Pinned here:
  * record / count / p50 basics;
  * is_slow is PREFER-MISSING: None below min_samples, then p50 vs threshold;
  * the ring is BOUNDED (recent-biased, file can't grow unbounded);
  * JSON persistence round-trips; a corrupt/absent file degrades to empty;
  * junk inputs (blank type / negative / NaN) are ignored, never raise.
"""
from __future__ import annotations

import json

from mast.core.run_stats import RunStatsStore


def test_record_count_and_p50():
    s = RunStatsStore()
    for d in (10.0, 20.0, 30.0):
        s.record("literature", d)
    assert s.count("literature") == 3
    assert s.p50("literature") == 20.0
    # an unseen type is empty, not an error
    assert s.count("data_processing") == 0
    assert s.p50("data_processing") is None


def test_is_slow_is_prefer_missing_below_min_samples():
    s = RunStatsStore()
    # 3 samples, min_samples=5 → not enough data → None (fall back to whitelist)
    for d in (100.0, 100.0, 100.0):
        s.record("literature", d)
    assert s.is_slow("literature", threshold_s=45.0, min_samples=5) is None
    # a type with ZERO samples is also None
    assert s.is_slow("paper_writing", threshold_s=45.0, min_samples=5) is None


def test_is_slow_true_when_median_exceeds_threshold():
    s = RunStatsStore()
    for d in (60.0, 70.0, 80.0, 90.0, 100.0):     # p50 = 80 > 45
        s.record("literature", d)
    assert s.is_slow("literature", threshold_s=45.0, min_samples=5) is True


def test_is_slow_false_when_median_below_threshold():
    s = RunStatsStore()
    for d in (5.0, 6.0, 7.0, 8.0, 9.0):           # p50 = 7 < 45
        s.record("literature", d)
    assert s.is_slow("literature", threshold_s=45.0, min_samples=5) is False


def test_ring_is_bounded_and_recent_biased():
    s = RunStatsStore(max_per_type=4)
    for d in (1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0):
        s.record("literature", d)
    assert s.count("literature") == 4            # only the last 4 kept
    assert s.p50("literature") == 100.0          # old fast runs dropped out


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "stats.json"
    s1 = RunStatsStore(path=path)
    for d in (12.0, 34.0, 56.0):
        s1.record("literature", d)
    assert path.exists()
    # a fresh store over the SAME file sees the history (survives a "restart")
    s2 = RunStatsStore(path=path)
    assert s2.count("literature") == 3
    assert s2.p50("literature") == 34.0


def test_corrupt_file_degrades_to_empty(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text("{ this is not json", encoding="utf-8")
    s = RunStatsStore(path=path)          # must not raise
    assert s.count("literature") == 0
    # and it can still record + persist over the bad file
    s.record("literature", 42.0)
    assert s.count("literature") == 1


def test_junk_inputs_are_ignored():
    s = RunStatsStore()
    s.record("", 10.0)                    # blank type
    s.record("literature", -5.0)         # negative
    s.record("literature", float("nan")) # NaN
    s.record("literature", "oops")       # non-numeric  # type: ignore[arg-type]
    assert s.count("literature") == 0
    assert s.count("") == 0


def test_snapshot_shape(tmp_path):
    s = RunStatsStore()
    for d in (10.0, 20.0):
        s.record("literature", d)
    snap = s.snapshot()
    assert snap["literature"]["count"] == 2
    assert snap["literature"]["p50"] == 10.0
    # snapshot is JSON-safe
    json.dumps(snap)
