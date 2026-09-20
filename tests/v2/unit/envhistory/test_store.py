"""Store round-trips, and the one claim the whole query layer rests on:
**re-aggregating buckets in SQL gives the same answer as aggregating the raw
readings.**

If that is only approximately true, then zooming out on a trend quietly changes
the numbers — and the operator has no way to tell which zoom level is lying.
So it is checked against numpy over the raw values, not against a second
implementation of the same formula.

The other thing checked hard is that a downsampled view can never hide a fault:
the merged ``worst_status`` has to come out of a severity ranking, not out of
SQLite's lexicographic MAX (where 'warning' > 'error' > 'alarm').
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

import numpy as np
import pytest

from mast.envhistory.buckets import BucketAccumulator
from mast.envhistory.store import EnvHistoryStore, get_store, set_store_for_test

T0 = 1_800_000_000.0
DT = 60.0


@pytest.fixture()
def store(tmp_path):
    s = EnvHistoryStore(tmp_path / "eh.sqlite")
    set_store_for_test(s)
    yield s
    set_store_for_test(None)


def _fill(store, sensor="temperature", n_readings=180, unit="K",
          value=lambda i: 77.0 + 0.01 * i, status=lambda i: "ok"):
    """Feed n readings at 2 s spacing, write every completed bucket, return the
    raw values that entered the statistics."""
    acc = BucketAccumulator(sensor)
    raw, rows = [], []
    for i in range(n_readings):
        st = status(i)
        v = value(i)
        b = acc.add(T0 + i * 2.0, v, unit, st, dt_s=DT)
        if st in ("ok", "warning"):
            raw.append(v)
        if b is not None:
            rows.append(b)
    tail = acc.flush()
    if tail is not None:
        rows.append(tail)
    store.upsert_buckets(rows)
    return raw


def test_schema_is_idempotent(tmp_path):
    p = tmp_path / "eh.sqlite"
    EnvHistoryStore(p).close()
    s = EnvHistoryStore(p)          # re-open must not raise
    assert s.storage_stats()["bucket_rows"] == 0
    s.close()


def test_upsert_is_idempotent(store):
    _fill(store, n_readings=60)
    before = store.storage_stats()["bucket_rows"]
    _fill(store, n_readings=60)     # replay the same window
    assert store.storage_stats()["bucket_rows"] == before


def test_series_returns_native_buckets_when_they_fit(store):
    _fill(store, n_readings=180)
    d = store.series("temperature")
    assert d["thinned"] is False
    assert d["bucket_s_effective"] == DT
    assert d["unit"] == "K"
    assert len(d["points"]) == 6


def test_rebucketing_is_exact_not_approximate(store):
    """The load-bearing claim: a merged view equals aggregating the raw data."""
    raw = _fill(store, n_readings=180)
    d = store.series("temperature", max_points=1)
    assert d["thinned"] is True
    assert len(d["points"]) == 1
    p = d["points"][0]
    assert p["n"] == len(raw)
    assert abs(p["mean"] - float(np.mean(raw))) < 1e-9
    assert p["min"] == min(raw)
    assert p["max"] == max(raw)
    # Pooled variance, computed in SQL from per-bucket (mean, std, n).
    assert abs(p["std"] - float(np.std(raw))) < 1e-6


def test_rebucketing_preserves_extremes_of_a_spike(store):
    """A single excursion inside one minute must survive a month-wide view."""
    def value(i):
        return 1e-9 if i == 77 else 2e-11
    _fill(store, sensor="tunnel_current", n_readings=180, unit="A", value=value)
    d = store.series("tunnel_current", max_points=1)
    assert d["points"][0]["max"] == 1e-9


def test_downsampling_cannot_hide_an_alarm(store):
    """SQLite's MAX() on TEXT would rank 'warning' above 'error' and 'alarm'."""
    def status(i):
        return "alarm" if i == 5 else "ok"
    _fill(store, n_readings=180, status=status)
    coarse = store.series("temperature", max_points=1)
    assert coarse["points"][0]["worst_status"] == "alarm"


def test_series_respects_the_time_window(store):
    _fill(store, n_readings=180)
    d = store.series("temperature", since=T0 + 120)
    assert d["points"]
    assert all(p["ts"] >= T0 + 120 for p in d["points"])


def test_series_of_an_unknown_sensor_is_empty_not_an_error(store):
    d = store.series("nonexistent")
    assert d["points"] == [] and d["total"] == 0


def test_list_sensors_reports_range_and_unit(store):
    _fill(store, n_readings=180)
    _fill(store, sensor="vacuum", n_readings=60, unit="Pa",
          value=lambda i: 1e-8)
    rows = {r["sensor"]: r for r in store.list_sensors()}
    assert set(rows) == {"temperature", "vacuum"}
    assert rows["temperature"]["unit"] == "K"
    assert rows["temperature"]["first_ts"] == T0


def test_spectrum_blob_round_trips_without_numpy(store):
    freqs = [1.0 * 2 ** (i / 8.0) for i in range(64)]
    psd = [1e-24 / f for f in freqs]
    rid = store.add_spectrum(ts=T0, channel="current", span_s=1800.0,
                             n_segments=8, fs_hz=20000.0,
                             freqs=freqs, psd=psd,
                             ctx={"ctx_bias_v": 0.5, "ctx_zctrl_on": True},
                             experiment_id="E1")
    assert rid
    got = store.spectrum(rid)
    assert got["n_points"] == len(freqs)
    assert got["channel"] == "current"
    assert got["experiment_id"] == "E1"
    assert got["ctx_zctrl_on"] == 1
    # float32 storage — compare with the precision that implies, not exactly.
    assert np.allclose(got["freqs_hz"], freqs, rtol=1e-6)
    assert np.allclose(got["psd"], psd, rtol=1e-6)


def test_spectrum_metadata_list_carries_no_arrays(store):
    store.add_spectrum(ts=T0, channel="z", span_s=30.0, n_segments=30,
                       fs_hz=20000.0, freqs=[1.0, 2.0], psd=[1e-24, 5e-25],
                       unit="m^2/Hz")
    rows = store.spectra_query(channel="z")
    assert len(rows) == 1
    assert rows[0]["unit"] == "m^2/Hz"
    assert "freqs" not in rows[0] and "psd" not in rows[0]


def test_spectra_query_filters_by_channel_and_time(store):
    for i, ch in enumerate(("current", "z", "current")):
        store.add_spectrum(ts=T0 + i * 100, channel=ch, span_s=1.0, n_segments=1,
                           fs_hz=1000.0, freqs=[1.0], psd=[1.0])
    assert len(store.spectra_query(channel="current")) == 2
    assert len(store.spectra_query(since=T0 + 50)) == 2


def test_spectrum_rejects_mismatched_arrays(store):
    assert store.add_spectrum(ts=T0, channel="current", span_s=1.0, n_segments=1,
                              fs_hz=1000.0, freqs=[1.0, 2.0], psd=[1.0]) is None
    assert store.add_spectrum(ts=T0, channel="current", span_s=1.0, n_segments=1,
                              fs_hz=1000.0, freqs=[], psd=[]) is None


def test_missing_spectrum_is_none(store):
    assert store.spectrum(9999) is None


def test_spectrum_context_survives_the_metadata_list(store):
    """挑一条谱是按「哪一条是在 -1.2 V 隧穿下测的」挑的，不是按行号。

    这几列一直在库里，只是 ``spectra_query`` 的 SELECT 里没有 —— 于是下拉框里
    只有一串时间戳，而两条不同工作点的谱放在一起对比是没有意义的。
    """
    store.add_spectrum(ts=T0, channel="current", span_s=1.0, n_segments=1,
                       fs_hz=1000.0, freqs=[1.0], psd=[1.0],
                       ctx={"ctx_bias_v": -1.2, "ctx_setpoint_a": 100e-12,
                            "ctx_zctrl_on": True},
                       ctx_stable=False)
    row = store.spectra_query(channel="current")[0]
    assert row["ctx_bias_v"] == -1.2
    assert row["ctx_zctrl_on"] == 1
    assert row["ctx_stable"] == 0


def test_ctx_stable_is_three_state_not_two(store):
    """``None`` = 没记过（老行），``False`` = 记了而且变过。

    压成布尔就等于替一条永久记录断言「工作点没变」，而事后没有第二次机会补测。
    """
    a = store.add_spectrum(ts=T0, channel="current", span_s=1.0, n_segments=1,
                           fs_hz=1000.0, freqs=[1.0], psd=[1.0])
    b = store.add_spectrum(ts=T0 + 1, channel="current", span_s=1.0, n_segments=1,
                           fs_hz=1000.0, freqs=[1.0], psd=[1.0], ctx_stable=True)
    assert store.spectrum(a)["ctx_stable"] is None
    assert store.spectrum(b)["ctx_stable"] == 1


def test_an_old_db_without_the_new_column_gets_it_added(tmp_path):
    """旧版本数据库升级必须新增缺失列；CREATE TABLE IF NOT EXISTS 不会修改已有结构。
    测试核验迁移后写入仍然成功，避免异常被吞掉后静默停止归档。"""
    import sqlite3

    from mast.envhistory.store import EnvHistoryStore

    db = tmp_path / "old.sqlite"
    # v1 的 env_spectra：没有 ctx_stable。
    con = sqlite3.connect(str(db))
    con.executescript(
        "CREATE TABLE env_spectra ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
        " channel TEXT NOT NULL, span_s REAL NOT NULL, n_segments INTEGER NOT NULL,"
        " fs_hz REAL NOT NULL, f_lo_hz REAL NOT NULL, f_hi_hz REAL NOT NULL,"
        " n_points INTEGER NOT NULL, freqs BLOB NOT NULL, psd BLOB NOT NULL,"
        " unit TEXT NOT NULL DEFAULT 'A^2/Hz',"
        " quietness TEXT NOT NULL DEFAULT 'quiet',"
        " ctx_bias_v REAL, ctx_setpoint_a REAL, ctx_zctrl_on INTEGER,"
        " experiment_id TEXT, sample_id TEXT, created_at REAL NOT NULL);"
    )
    con.commit()
    con.close()
    assert "ctx_stable" not in _columns(db, "env_spectra"), "前提没成立，这条测试什么也没证明"

    st = EnvHistoryStore(db)
    assert "ctx_stable" in _columns(db, "env_spectra")
    # 而且写得进去、读得出来 —— 补了列却写不进等于没补。
    rid = st.add_spectrum(ts=T0, channel="current", span_s=1.0, n_segments=1,
                          fs_hz=1000.0, freqs=[1.0], psd=[1.0], ctx_stable=False)
    assert rid and st.spectrum(rid)["ctx_stable"] == 0


def _columns(db_path, table: str) -> set:
    import sqlite3
    con = sqlite3.connect(str(db_path))
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def test_writes_never_raise_after_the_connection_is_closed(store):
    """A monitoring write failing must never take acquisition down."""
    acc = BucketAccumulator("temperature")
    acc.add(T0, 77.0, "K", "ok", dt_s=DT)
    b = acc.flush()
    store.close()
    assert store.upsert_buckets([b]) == 0            # swallowed, reported as 0
    assert store.add_spectrum(ts=T0, channel="current", span_s=1.0,
                              n_segments=1, fs_hz=1.0,
                              freqs=[1.0], psd=[1.0]) is None
    assert store.series("temperature")["points"] == []   # reads degrade too
    assert store.list_sensors() == []


def test_storage_stats_counts_both_tables(store):
    _fill(store, n_readings=180)
    store.add_spectrum(ts=T0, channel="current", span_s=1.0, n_segments=1,
                       fs_hz=1.0, freqs=[1.0], psd=[1.0])
    st = store.storage_stats()
    assert st["bucket_rows"] == 6
    assert st["spectra_rows"] == 1
    assert st["db_bytes"] > 0
    assert st["oldest_bucket_ts"] == T0


def test_singleton_is_isolated_by_the_conftest_fixture():
    """The autouse fixture must have redirected the process singleton away from
    the operator's real archive."""
    p = str(get_store().path).replace("\\", "/")
    assert "/experiments/env_history/" not in p
