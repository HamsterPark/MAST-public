"""The store under the thread mix it actually sees in production.

One SQLite connection behind one lock is a deliberate choice (same as the usage
ledger), but it only holds if every path really takes the lock and none of them
holds it across slow I/O. In production the writers and readers are genuinely
concurrent: the acquisition daemon writes a segment every second, API requests
read on the server's thread pool, the retention sweep deletes files, and the
evidence renderer writes back from its own thread.

This test does not assert timings — it asserts that nothing raises and nothing
deadlocks, which is what a lock-discipline bug looks like from the outside.
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

import threading
import time
import traceback

import numpy as np
import pytest

from mast.monitoring import features as F
from mast.monitoring.store import CurrentMonitorStore, segment_npy_path

FS = 20000.0
DURATION_S = 4.0


@pytest.fixture()
def store(tmp_path):
    s = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    yield s
    s.close()


def test_mixed_readers_and_writers_never_raise_or_deadlock(store, tmp_path):
    errors: list[tuple[str, str]] = []
    counts = {"write": 0, "read": 0, "sweep": 0, "pin": 0, "evidence": 0}
    stop = threading.Event()
    # Short 0.05 s segments keep the test quick while still exercising the
    # full write path (npy + envelope + features) on every iteration.
    sample = 100e-12 + np.random.default_rng(0).normal(0, 2e-12, int(FS * 0.05))
    env = F.envelope(sample, FS, 100).tobytes()
    feats = F.compute_segment_features([sample], FS)

    def write():
        t = time.time()
        p = segment_npy_path(store.data_dir, t, FS)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, sample.astype(np.float32))
        sid = store.add_segment(
            {"t_start": t, "t_end": t + 0.05, "fs_hz": FS, "n_samples": sample.size,
             "npy_path": str(p), "npy_bytes": p.stat().st_size}, env, 0.01)
        store.add_features(sid, {**feats, "t_start": t}, {}, "ok")

    def read():
        store.storage_stats()
        store.latest_feature()
        store.features_query(limit=50)
        store.segments_query(limit=20)
        store.live_tail(window_s=30)
        store.alerts_query()

    def sweep():
        # An aggressive budget so the sweep is always deleting something while
        # the writer is still creating files.
        store.retention_sweep(keep_hours=0.001, keep_gb=0.0005)

    def pin():
        for s in store.segments_query(limit=5)["segments"][:2]:
            store.set_pin(int(s["id"]), True, "race")

    def evidence():
        aid = store.add_alert(ts=time.time(), level="warn", rule="rms_high",
                              summary_zh="x")
        if aid:
            store.set_alert_evidence(aid, str(tmp_path / "fake.png"))

    def loop(name, fn):
        def run():
            while not stop.is_set():
                try:
                    fn()
                    counts[name] += 1
                except Exception:  # noqa: BLE001 — recorded, then the thread exits
                    errors.append((name, traceback.format_exc()))
                    return
        return run

    jobs = ([("write", write)] + [("read", read)] * 4 +
            [("sweep", sweep), ("pin", pin), ("evidence", evidence)])
    threads = [threading.Thread(target=loop(n, f), daemon=True) for n, f in jobs]
    for t in threads:
        t.start()
    time.sleep(DURATION_S)
    stop.set()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "a worker did not finish — likely a deadlock"

    assert not errors, "\n\n".join(f"[{n}]\n{e}" for n, e in errors[:2])
    assert counts["write"] > 0 and counts["read"] > 0
    assert counts["sweep"] > 0 and counts["evidence"] > 0


def test_sweeping_while_reading_never_yields_a_half_read_segment(store):
    """The sweep deletes .npy files that a reader may be opening. The reader
    must either get the samples or fall back to the envelope — never a partial
    array and never an exception."""
    y = 100e-12 + np.random.default_rng(1).normal(0, 2e-12, int(FS * 0.05))
    ids = []
    for i in range(20):
        t = time.time() - 10 + i * 0.05
        p = segment_npy_path(store.data_dir, t, FS)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, y.astype(np.float32))
        ids.append(store.add_segment(
            {"t_start": t, "t_end": t + 0.05, "fs_hz": FS, "n_samples": y.size,
             "npy_path": str(p), "npy_bytes": p.stat().st_size},
            F.envelope(y, FS, 100).tobytes(), 0.01))

    bad: list[str] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            for sid in ids:
                out = store.read_segment_decimated(sid, max_points=200)
                if out is None:
                    continue
                if not out["i_a"]:
                    bad.append(f"segment {sid} returned an empty trace")
                if out["source"] not in ("raw", "envelope"):
                    bad.append(f"segment {sid} bad source {out['source']}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for _ in range(5):
        store.retention_sweep(keep_hours=0.0, keep_gb=100.0)
        time.sleep(0.05)
    stop.set()
    t.join(timeout=10.0)

    assert not t.is_alive()
    assert not bad, bad[:3]
