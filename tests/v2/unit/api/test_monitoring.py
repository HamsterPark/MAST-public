"""Contract tests for the 电流监控 endpoints (/api/monitoring/*).

Two things get pinned here beyond the happy paths:

* **never 500** — this app must boot standalone with no core wired, and the
  monitoring package may not be importable at all. Every endpoint answers with
  ``degraded=True`` instead;
* **the store is readable without the daemon** — a stopped monitor must still
  show history, because that is what an operator opens the page to look at
  after something went wrong.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.monitoring import router
from mast.monitoring import features as F
from mast.monitoring import service as SVC
from mast.monitoring import store as ST
from mast.monitoring import thresholds as TH

FS = 20000.0


@pytest.fixture(autouse=True)
def _isolate(tmp_path):
    ST.set_store_for_test(ST.CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path))
    SVC.set_service_for_test(None)
    TH.set_monitor_thresholds(None)
    yield
    ST.set_store_for_test(None)
    SVC.set_service_for_test(None)
    TH.set_monitor_thresholds(None)


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _seed(*, t_start=None, pinned=False, level="ok", with_file=True,
          skill="", scanning=True) -> int:
    store = ST.get_store()
    t = time.time() if t_start is None else t_start
    n = int(FS)
    y = 100e-12 + np.random.default_rng(int(t) % 1000).normal(0, 2e-12, n)
    npy_path, nbytes = None, 0
    if with_file:
        p = ST.segment_npy_path(store.data_dir, t, FS)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, y.astype(np.float32))
        npy_path, nbytes = str(p), p.stat().st_size
    sid = store.add_segment(
        {"t_start": t, "t_end": t + 1, "fs_hz": FS, "n_samples": n,
         "npy_path": npy_path, "npy_bytes": nbytes, "pinned": pinned,
         "channel_name": "Current (A)"},
        F.envelope(y, FS, 100).tobytes(), 0.01)
    feats = F.compute_segment_features([y], FS)
    feats["t_start"] = t
    store.add_features(sid, feats, {"ctx_scanning": scanning, "ctx_bias_v": -1.2,
                                    "ctx_setpoint_a": 100e-12, "ctx_skill": skill},
                       level)
    return sid


# ── degradation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    "/api/monitoring/status",
    "/api/monitoring/config",
    "/api/monitoring/live-trace",
    "/api/monitoring/features",
    "/api/monitoring/alerts",
    "/api/monitoring/segments",
    "/api/monitoring/segments/999/data",
    "/api/monitoring/alerts/999/evidence",
])
def test_every_get_answers_200_on_an_empty_install(path):
    r = _client().get(path)
    assert r.status_code == 200


def test_status_degrades_instead_of_500_when_the_package_is_missing(monkeypatch):
    monkeypatch.setattr(SVC, "get_service",
                        lambda: (_ for _ in ()).throw(ImportError("no monitoring")))
    b = _client().get("/api/monitoring/status").json()
    assert b["degraded"] is True
    assert b["detail"]


def test_store_failure_degrades_the_series_endpoints(monkeypatch):
    monkeypatch.setattr(ST, "get_store",
                        lambda: (_ for _ in ()).throw(RuntimeError("db gone")))
    for path in ("/api/monitoring/live-trace", "/api/monitoring/features",
                 "/api/monitoring/alerts", "/api/monitoring/segments"):
        b = _client().get(path).json()
        assert b["degraded"] is True, path


def test_a_store_that_is_there_but_unreadable_also_degrades():
    """数据库对象存在但查询失败时，端点必须报告 degraded，不能把故障转换成正常的空告警列表。"""
    ST.get_store().close()
    for path in ("/api/monitoring/alerts", "/api/monitoring/features",
                 "/api/monitoring/segments", "/api/monitoring/live-trace",
                 "/api/monitoring/status"):
        r = _client().get(path)
        assert r.status_code == 200, f"{path} 不许 500"
        b = r.json()
        assert b["degraded"] is True, f"{path} 把一次读失败报成了正常结果"

    # 而且**不许**同时报出一个像模像样的计数。
    b = _client().get("/api/monitoring/alerts").json()
    assert b["total"] == 0 and b["alerts"] == []
    assert b["detail"], "degraded 了却不说为什么 —— 停得住,解释不了"


# ── status ───────────────────────────────────────────────────────────────────

def test_status_without_a_daemon_still_reports_history():
    """A stopped monitor must not blank the page — the store is what the
    operator came to look at."""
    _seed()
    b = _client().get("/api/monitoring/status").json()
    assert b["running"] is False
    assert b["segments_total"] == 1
    assert b["latest"] is not None
    assert b["latest"]["rms_detrended_a"] is not None
    assert b["latest"]["verdict"] == "ok"
    assert "rtn_score" in b["latest"]["metrics"]


def test_status_relays_the_daemon_state():
    class FakeSvc:
        def status(self):
            return {"running": True, "state": "running", "detail": "",
                    "strategy": "osci1t", "fs_hz": FS, "channel_name": "Current (A)",
                    "n_buffer": 1024, "connected": True, "segments_done": 7,
                    "gaps_total_s": 0.5, "last_segment_ts": 123.0,
                    "enabled_in_settings": True, "alerts_enabled": True,
                    "segment_seconds": 1.0, "retention_hours": 24.0,
                    "retention_gb": 4.0, "retry_in_s": 0.0}

    SVC.set_service_for_test(FakeSvc())
    b = _client().get("/api/monitoring/status").json()
    assert b["running"] is True and b["state"] == "running"
    assert b["strategy"] == "osci1t" and b["fs_hz"] == FS
    assert b["segments_done"] == 7 and b["connected"] is True


def test_status_reflects_the_settings_switch():
    TH.set_monitor_thresholds({"cm_enabled": 0.0})
    b = _client().get("/api/monitoring/status").json()
    assert b["enabled_in_settings"] is False


# ── control ──────────────────────────────────────────────────────────────────

def test_start_and_stop_relay_to_the_daemon():
    calls = []

    class FakeSvc:
        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

        def status(self):
            return {"running": True}

    SVC.set_service_for_test(FakeSvc())
    c = _client()
    assert c.post("/api/monitoring/start").json()["ok"] is True
    assert c.post("/api/monitoring/stop").json()["ok"] is True
    assert calls == ["start", "stop"]


def test_start_without_a_daemon_is_degraded_not_an_error():
    b = _client().post("/api/monitoring/start").json()
    assert b["ok"] is False and b["degraded"] is True
    assert b["note"]


def test_stop_without_a_daemon_is_ok():
    b = _client().post("/api/monitoring/stop").json()
    assert b["ok"] is True and b["running"] is False


# ── config ───────────────────────────────────────────────────────────────────

def test_config_ships_the_knob_catalogue():
    b = _client().get("/api/monitoring/config").json()
    keys = {k["key"] for k in b["knobs"]}
    assert set(TH.EDITABLE_KEYS) == keys
    for knob in b["knobs"]:
        assert knob["min"] <= knob["default"] <= knob["max"]
        assert knob["label_zh"]
    assert b["enabled"] is True


def test_config_reflects_live_threshold_changes():
    TH.set_monitor_thresholds({"cm_keep_gb": 9.0})
    b = _client().get("/api/monitoring/config").json()
    assert b["retention_gb"] == 9.0
    assert next(k for k in b["knobs"] if k["key"] == "cm_keep_gb")["value"] == 9.0


# ── series ───────────────────────────────────────────────────────────────────

def test_live_trace_returns_a_min_max_band():
    now = time.time()
    for i in range(3):
        _seed(t_start=now - 3 + i)
    b = _client().get("/api/monitoring/live-trace?window_s=60").json()
    assert b["n_segments"] == 3
    assert len(b["t_s"]) == len(b["i_min_a"]) == len(b["i_max_a"]) > 0
    assert all(lo <= hi for lo, hi in zip(b["i_min_a"], b["i_max_a"]))


def test_live_trace_window_and_points_are_clamped():
    from mast.api.routes.monitoring import _MAX_TRACE_WINDOW_S

    b = _client().get("/api/monitoring/live-trace?window_s=999999&max_points=99999").json()
    assert b["window_s"] == _MAX_TRACE_WINDOW_S
    # 6 h, raised from 600 s (real-time current view needed to cover more time). Asserted
    # as a literal too: importing the constant alone would let the ceiling be
    # deleted to 1 s and this test would still pass.
    assert _MAX_TRACE_WINDOW_S == 21600.0


def test_live_trace_decimates_a_long_window_without_losing_the_extremes():
    """The two-pass reduction must stay min/max, not subsampling.

    A long window reduces inside each segment BEFORE concatenating, so this is
    the case where a spike could quietly be averaged away — and the whole point
    of an envelope band is that it cannot be.
    """
    now = time.time()
    for i in range(40):
        _seed(t_start=now - 40 + i)
    # 100 is the route's own floor for max_points, so this is the tightest
    # budget an actual caller can request.
    b = _client().get("/api/monitoring/live-trace?window_s=600&max_points=100").json()
    full = _client().get("/api/monitoring/live-trace?window_s=600&max_points=5000").json()

    assert 0 < len(b["t_s"]) <= 100
    assert len(full["t_s"]) > len(b["t_s"]), "not decimated — the test proves nothing"
    assert len(b["t_s"]) == len(b["i_min_a"]) == len(b["i_max_a"])
    assert all(lo <= hi for lo, hi in zip(b["i_min_a"], b["i_max_a"]))
    # The reduced band must still bracket the un-reduced one: decimation may
    # widen a bucket, never narrow it past a real excursion.
    assert min(b["i_min_a"]) <= min(full["i_min_a"]) + 1e-18
    assert max(b["i_max_a"]) >= max(full["i_max_a"]) - 1e-18


def test_features_range_and_metrics_map():
    now = time.time()
    for i in range(5):
        _seed(t_start=now + i)
    b = _client().get(f"/api/monitoring/features?since={now + 1}&until={now + 4}").json()
    assert b["total"] == 3
    row = b["rows"][0]
    assert row["scanning"] is True
    assert row["bias_v"] == pytest.approx(-1.2)
    assert "rms_detrended_a" in row["metrics"]
    assert "ctx_bias_v" not in row["metrics"]      # context is not a metric


def test_features_thinning_keeps_the_worst_verdict():
    """Chart downsampling must never hide an alert that fired."""
    now = time.time()
    for i in range(20):
        _seed(t_start=now + i, level="critical" if i == 9 else "ok")
    b = _client().get(f"/api/monitoring/features?since={now}&max_points=5").json()
    assert b["thinned"] is True
    assert any(r["verdict"] == "critical" for r in b["rows"])


# ── alerts ───────────────────────────────────────────────────────────────────

def test_alerts_list_and_level_filter():
    store = ST.get_store()
    store.add_alert(ts=time.time(), level="warn", rule="rms_high", summary_zh="噪声偏高")
    store.add_alert(ts=time.time(), level="critical", rule="saturation",
                    summary_zh="电流饱和", emitted_buffer=True)
    c = _client()
    assert c.get("/api/monitoring/alerts").json()["total"] == 2
    crit = c.get("/api/monitoring/alerts?level=critical").json()
    assert crit["total"] == 1
    assert crit["alerts"][0]["rule"] == "saturation"
    assert crit["alerts"][0]["emitted_buffer"] is True
    assert crit["alerts"][0]["summary_zh"] == "电流饱和"


def test_alert_evidence_is_base64(tmp_path):
    png = tmp_path / "ev.png"
    png.write_bytes(b"\x89PNG-body")
    aid = ST.get_store().add_alert(ts=time.time(), level="critical", rule="freeze",
                                   summary_zh="冻结", evidence_png=str(png))
    b = _client().get(f"/api/monitoring/alerts/{aid}/evidence").json()
    assert b["ok"] is True
    import base64
    assert base64.b64decode(b["png_b64"]) == b"\x89PNG-body"


def test_alert_evidence_missing_is_not_ok_but_still_200():
    aid = ST.get_store().add_alert(ts=time.time(), level="warn", rule="x",
                                   summary_zh="y")
    r = _client().get(f"/api/monitoring/alerts/{aid}/evidence")
    assert r.status_code == 200 and r.json()["ok"] is False


# ── segments ─────────────────────────────────────────────────────────────────

def test_segment_list_filters_by_pinned_and_label():
    a = _seed(pinned=True)
    b_id = _seed()
    c = _client()
    only_pinned = c.get("/api/monitoring/segments?pinned=true").json()
    assert [s["seg_id"] for s in only_pinned["segments"]] == [a]
    assert only_pinned["pinned_count"] == 1

    c.post(f"/api/monitoring/segments/{b_id}/label", json={"label": "bad"})
    bad = c.get("/api/monitoring/segments?label=bad").json()
    assert [s["seg_id"] for s in bad["segments"]] == [b_id]
    unl = c.get("/api/monitoring/segments?label=unlabeled").json()
    assert [s["seg_id"] for s in unl["segments"]] == [a]


def test_segment_data_is_decimated_and_keeps_the_extremes():
    sid = _seed()
    b = _client().get(f"/api/monitoring/segments/{sid}/data?max_points=500").json()
    assert b["ok"] is True and b["source"] == "raw" and b["decimated"] is True
    assert len(b["i_a"]) <= 500
    assert b["n_samples_raw"] == int(FS)
    assert b["meta"]["channel_name"] == "Current (A)"


def test_segment_data_includes_a_psd_on_request():
    sid = _seed()
    b = _client().get(f"/api/monitoring/segments/{sid}/data?include_psd=true").json()
    assert b["psd"] is not None
    assert b["psd"]["output"] == "power"
    assert len(b["psd"]["freqs_hz"]) == len(b["psd"]["spectrum"]) > 0
    # computed from the full-rate samples, not the decimated view
    assert b["psd"]["nyquist_hz"] == pytest.approx(FS / 2)


def test_segment_data_falls_back_to_the_envelope_after_a_sweep():
    sid = _seed(t_start=time.time() - 100 * 3600)
    ST.get_store().retention_sweep(keep_hours=1.0, keep_gb=100.0)
    b = _client().get(f"/api/monitoring/segments/{sid}/data").json()
    assert b["ok"] is True and b["source"] == "envelope"
    assert len(b["i_a"]) > 0
    assert b["meta"]["has_file"] is False        # the UI says so


def test_missing_segment_is_not_ok_but_still_200():
    r = _client().get("/api/monitoring/segments/4242/data")
    assert r.status_code == 200 and r.json()["ok"] is False


# ── pin / label ──────────────────────────────────────────────────────────────

def test_pin_and_unpin_round_trip():
    sid = _seed()
    c = _client()
    assert c.post(f"/api/monitoring/segments/{sid}/pin",
                  json={"pinned": True, "reason": "interesting"}).json()["ok"] is True
    assert ST.get_store().segment_meta(sid)["pinned"] == 1
    c.post(f"/api/monitoring/segments/{sid}/pin", json={"pinned": False})
    assert ST.get_store().segment_meta(sid)["pinned"] == 0


def test_labelling_pins_the_segment():
    """A judged segment is a corpus item; letting the sweep delete the waveform
    behind a label would leave a verdict with nothing to train on."""
    sid = _seed()
    b = _client().post(f"/api/monitoring/segments/{sid}/label",
                       json={"label": "good", "note": "干净"}).json()
    assert b["ok"] is True and b["label"] == "good" and b["pinned"] is True
    assert ST.get_store().segment_meta(sid)["pinned"] == 1

    rows = ST.get_store().labels_query()
    assert rows[0]["label"] == "good" and rows[0]["note"] == "干净"


def test_relabelling_replaces_rather_than_appends():
    sid = _seed()
    c = _client()
    c.post(f"/api/monitoring/segments/{sid}/label", json={"label": "good"})
    c.post(f"/api/monitoring/segments/{sid}/label", json={"label": "bad"})
    labels = [r["label"] for r in ST.get_store().labels_query()]
    assert labels == ["bad"]


def test_clearing_a_label_leaves_the_pin_alone():
    sid = _seed()
    c = _client()
    c.post(f"/api/monitoring/segments/{sid}/label", json={"label": "good"})
    b = c.post(f"/api/monitoring/segments/{sid}/label", json={"label": None}).json()
    assert b["ok"] is True and b["label"] is None
    assert ST.get_store().labels_query() == []
    # still pinned: the operator can unpin explicitly, but clearing a verdict
    # should not quietly make the data eligible for deletion
    assert ST.get_store().segment_meta(sid)["pinned"] == 1


def test_labelling_a_missing_segment_is_not_ok_but_still_200():
    r = _client().post("/api/monitoring/segments/999/label", json={"label": "good"})
    assert r.status_code == 200 and r.json()["ok"] is False


# ── route registration ───────────────────────────────────────────────────────

def test_literal_paths_are_not_shadowed_by_the_id_parameter():
    """`/segments` must not be captured by `/segments/{seg_id}/…`."""
    paths = {r.path for r in _client().app.routes}
    assert "/api/monitoring/segments" in paths
    assert "/api/monitoring/segments/{seg_id}/data" in paths
    r = _client().get("/api/monitoring/segments")
    assert r.status_code == 200 and "segments" in r.json()


# ── 辅助通道 (Z / qPlus 振幅) ────────────────────────────────────────────────

def _seed_aux(n=5, *, base_ts=None, scanning=False, verdict="ok"):
    store = ST.get_store()
    t0 = time.time() if base_ts is None else base_ts
    for i in range(n):
        store.add_aux_sample(
            t0 + i,
            {"z_m": 1e-9 + i * 1e-11, "z_drift_m_per_s": 1e-11,
             "amp_m": 100e-12, "df_hz": -5.0, "z_step_m": 0.0},
            {"ctx_scanning": scanning, "ctx_zctrl_on": True, "ctx_skill": ""},
            segment_id=i + 1, verdict=verdict,
            rules=["z_step"] if verdict == "warn" else [])
    return t0


def test_aux_series_returns_aligned_columns():
    t0 = _seed_aux(5)
    b = _client().get("/api/monitoring/aux/series",
                      params={"since": t0 - 10}).json()
    from mast.api.routes.monitoring import _AUX_SERIES_COLUMNS
    assert b["total"] == 5 and len(b["t_s"]) == 5
    assert set(b["series"]) == set(_AUX_SERIES_COLUMNS)
    assert all(len(v) == 5 for v in b["series"].values())
    assert b["verdicts"] == ["ok"] * 5


def test_aux_series_context_rides_along_for_remote_calibration():
    """标定要按「在不在扫描」分组 —— 远程调用方也得拿得到这一层。"""
    t0 = _seed_aux(3, scanning=False)
    b = _client().get("/api/monitoring/aux/series", params={"since": t0 - 10}).json()
    assert b["scanning"] == [False, False, False]
    assert b["z_ctrl_on"] == [True, True, True]
    assert b["skills"] == ["", "", ""]


def test_aux_series_columns_are_whitelisted():
    """未经校验的列名会让查询点名 ``rules``，拿回字符串塞进承诺是浮点的字段。"""
    t0 = _seed_aux(3)
    b = _client().get("/api/monitoring/aux/series",
                      params={"since": t0 - 10,
                              "columns": "z_drift_m_per_s,rules,extra_json"}).json()
    assert set(b["series"]) == {"z_drift_m_per_s"}


def test_aux_series_unknown_columns_fall_back_to_the_defaults():
    t0 = _seed_aux(3)
    b = _client().get("/api/monitoring/aux/series",
                      params={"since": t0 - 10, "columns": "nonsense"}).json()
    from mast.api.routes.monitoring import _AUX_SERIES_COLUMNS
    assert set(b["series"]) == set(_AUX_SERIES_COLUMNS)


def test_aux_series_missing_readings_stay_null_not_zero():
    """塌陷的 qPlus 振幅**真的就是** 0 —— 「没读到」必须区分得开。"""
    ST.get_store().add_aux_sample(time.time(), {"z_m": 1e-9}, {})   # 无 amp/df
    b = _client().get("/api/monitoring/aux/series").json()
    assert b["series"]["amp_m"] == [None]
    assert b["series"]["z_m"][0] is not None


def test_aux_series_thinning_keeps_the_warn_row():
    # 200 行 + max_points 的下界是 50（低于这个数的图没有意义），所以要真的抽稀
    # 就得有比 50 多的行。
    t0 = _seed_aux(200)
    ST.get_store().add_aux_sample(t0 + 201, {"z_m": 1e-9}, {}, verdict="warn")
    b = _client().get("/api/monitoring/aux/series",
                      params={"since": t0 - 10, "max_points": 5000}).json()
    b2 = _client().get("/api/monitoring/aux/series",
                       params={"since": t0 - 10, "max_points": 50}).json()
    assert b["thinned"] is False and b2["thinned"] is True
    assert len(b2["t_s"]) < len(b["t_s"])
    assert "warn" in b2["verdicts"], "抽稀不该把一条告警藏起来"


def test_aux_series_never_500s_without_the_monitoring_package(monkeypatch):
    monkeypatch.setattr("mast.api.routes.monitoring._store",
                        lambda: (_ for _ in ()).throw(ImportError("no numpy")))
    r = _client().get("/api/monitoring/aux/series")
    assert r.status_code == 200 and r.json()["degraded"] is True


def test_status_aux_is_none_without_a_daemon():
    """没有守护进程 ≠ 这台机器没有 Z 通道。两件事不能混为一谈。"""
    b = _client().get("/api/monitoring/status").json()
    assert b["aux"] is None


def test_status_carries_the_aux_snapshot_when_the_daemon_runs():
    class FakeSvc:
        def status(self):
            return {"running": True, "state": "running",
                    "aux": {"enabled": True, "alerts_enabled": False,
                            "interval_s": 1.0, "window_s": 300.0,
                            "sampled": 12, "skipped_busy": 1,
                            "amp_tau_s": 0.125, "amp_oversampled": True,
                            "z_limits_m": [-1e-6, 1e-6], "detail": "ok",
                            "channels": [
                                {"kind": "z", "label_zh": "Z 位置", "unit": "m",
                                 "available": True, "signal_index": 30,
                                 "signal_name": "Z (m)", "judged": True,
                                 "value": 1e-9, "ts": 1.0, "verdict": "unjudged",
                                 "note": "先跑标定", "metrics": {"z_m": 1e-9}},
                                {"kind": "amplitude", "available": False,
                                 "judged": True, "verdict": "unavailable",
                                 "note": "这台机器没有 qPlus"},
                            ]}}

    SVC.set_service_for_test(FakeSvc())
    b = _client().get("/api/monitoring/status").json()
    aux = b["aux"]
    assert aux["sampled"] == 12 and aux["amp_oversampled"] is True
    kinds = {c["kind"]: c for c in aux["channels"]}
    assert kinds["z"]["signal_index"] == 30
    assert kinds["amplitude"]["available"] is False
    assert kinds["amplitude"]["verdict"] == "unavailable"


def test_every_declared_aux_field_survives_the_mapping():
    """守护进程已经提供且响应模型已经声明的字段必须完整映射。
    逐字段检查可以发现新增模型字段遗漏于映射层的问题，避免默认 None 掩盖缺失。"""
    from mast.api.schemas_monitoring import AuxSnapshot

    # 每个标量字段一个**与默认值不同**的哨兵值:丢掉任何一个都会显形。
    scalars = {
        "enabled": True,
        "alerts_enabled": True,
        "interval_s": 1.0,
        "observed_interval_s": 1.25,   # 与 interval_s 刻意不等:两者不许互相顶替
        "window_s": 300.0,
        "sampled": 7,
        "skipped_busy": 2,
        "last_ts": 12345.0,
        "baseline_amp_m": 2.0e-10,
        "amp_tau_s": 0.125,
        "amp_oversampled": True,

        # 软限值启用状态与实际余量读数必须一起传递，避免单位或读回来源不一致。

        "z_limits_enabled": False,
        "z_travel_source": "Piezo_RangeGet/2",
        "detail": "ok",
    }
    declared = (set(AuxSnapshot.model_fields)
                - {"channels", "z_limits_m", "z_travel_m"})
    missing = declared - set(scalars)
    assert not missing, f"AuxSnapshot 新增了字段但没进这条闸门:{sorted(missing)}"

    raw = dict(scalars, z_limits_m=[-1e-6, 1e-6],
               z_travel_m=[-160.0e-9, 160.0e-9], channels=[])

    class FakeSvc:
        def status(self):
            return {"running": True, "state": "running", "aux": raw}

    SVC.set_service_for_test(FakeSvc())
    aux = _client().get("/api/monitoring/status").json()["aux"]
    for k, v in scalars.items():
        assert aux[k] == v, f"字段 {k} 在映射层被丢掉或改写了:{aux[k]!r} != {v!r}"
    assert aux["z_limits_m"] == [-1e-6, 1e-6]
    assert aux["z_travel_m"] == [-160.0e-9, 160.0e-9]


#: ``MonitoringStatus`` 里**不由守护进程字典供货**的字段,连同豁免理由。
#:
#: 闸门要**同时**编码「哪些必须活着过桥」和「哪些本来就不走这座桥」——只有前者
#: 的话,下一个人会把 latest/aux 这类塞进哨兵字典,发现对不上,然后把断言放宽,
#: 于是闸门变成一条永远绿的正则。
_STATUS_NOT_FROM_DAEMON: dict[str, str] = {
    "segments_total": "来自 store.storage_stats(),不经守护进程",
    "segments_on_disk": "来自 store.storage_stats()",
    "store_bytes": "来自 store.storage_stats()",
    "pinned_count": "来自 store.storage_stats()",
    "latest": "来自 store.latest_feature(),是最近一段的特征行",
    "aux": "嵌套模型,逐字段由 test_every_declared_aux_field_survives_the_mapping 守",
    "degraded": "只在 _guarded 的降级分支里置位,正常路径恒 False",
    "detail_error": "同上,只在降级分支里有值",
    "baseline": ("来自 service.baseline_status() 的单独调用,不经守护进程的 status "
                 "字典 —— 它读的是 store 里的活跃基线行,而那不是采集状态"),
}


def test_every_declared_status_field_survives_the_mapping():
    """守护进程、响应 schema 与路由映射必须完整传递 hr_available 等状态字段。
    从端到端 HTTP 响应验证，避免按映射层遗漏字段构造的替身掩盖同一遗漏。"""
    from mast.api.schemas_monitoring import MonitoringStatus

    # 每个字段一个**与模型默认值不同**的哨兵:映射层丢掉任何一个,响应都会回落到
    # 默认值,于是显形。
    sentinels: dict = {
        "running": True,
        "enabled_in_settings": True,
        "alerts_enabled": True,
        "state": "probing",
        "detail": "示例明细",
        "retry_in_s": 12.5,
        "strategy": "osci2t",
        "hr_available": True,
        "fs_hz": 2000.0,
        "channel_name": "Current (A)",
        "n_buffer": 12800,
        "rt_freq_hz": 20000.0,
        "timebases_s": [6.4, 0.128],
        "timebase_index": 5,
        "timebase_check": "ok",
        "pump_stats": {"fresh": 11, "duplicate": 3},
        "segment_seconds": 2.5,
        "connected": True,
        "segments_done": 7,
        "gaps_total_s": 0.5,
        "last_segment_ts": 123.0,
        "retention_hours": 48.0,
        "retention_gb": 7.0,
    }

    declared = set(MonitoringStatus.model_fields)
    unaccounted = declared - set(sentinels) - set(_STATUS_NOT_FROM_DAEMON)
    assert not unaccounted, (
        "MonitoringStatus 新增了字段但既没进哨兵表、也没写进豁免名单:"
        f"{sorted(unaccounted)}")
    # 反方向:哨兵表点名的字段模型必须真的声明。⑲ 的一半正是「schema 少一个字段」,
    # 而 declared 会跟着一起缩,上面那条断言看不见它。
    undeclared = set(sentinels) - declared
    assert not undeclared, (
        f"哨兵表点名了 MonitoringStatus 并未声明的字段:{sorted(undeclared)} —— "
        "response_model 会把它们静默过滤掉,正是缺陷⑲的形状")

    # 哨兵必须与默认值不同,否则「丢了」和「传了」长得一样,这条闸门自己就成了
    # 那种能编译但匹配不到任何东西的正则。
    defaults = MonitoringStatus()
    for key, value in sentinels.items():
        assert getattr(defaults, key) != value, (
            f"哨兵 {key}={value!r} 与模型默认值相同,这一格守不住任何东西")

    class FakeSvc:
        def status(self):
            return dict(sentinels)

    SVC.set_service_for_test(FakeSvc())
    body = _client().get("/api/monitoring/status").json()
    assert body["degraded"] is False, body.get("detail_error")
    for key, value in sentinels.items():
        assert body[key] == value, (
            f"字段 {key} 在映射层被丢掉或改写了:{body[key]!r} != {value!r}")


def test_config_knobs_carry_the_group_for_ui_sectioning():
    knobs = {k["key"]: k for k in _client().get("/api/monitoring/config").json()["knobs"]}
    assert knobs["cm_z_drift_warn_m_per_s"]["group"] == "aux"
    assert knobs["cm_rms_warn_a"]["group"] == "current"
    assert knobs["cm_aux_alerts_enabled"]["is_bool"] is True
    assert knobs["cm_aux_alerts_enabled"]["default"] == 0.0


def test_aux_series_literal_path_is_not_shadowed():
    paths = {r.path for r in _client().app.routes}
    assert "/api/monitoring/aux/series" in paths

# 告警确认必须通过 HTTP 路由可达，并正确写回 ack 状态。

def test_the_ack_route_exists_and_is_not_shadowed():
    """可达性:路由真的注册了,而且是字面路径不是被 {alert_id} 之类吃掉。"""
    paths = {r.path for r in _client().app.routes}
    assert "/api/monitoring/alerts/{alert_id}/ack" in paths


def test_acking_an_alert_actually_lands_in_the_table():
    store = ST.get_store()
    aid = store.add_alert(ts=time.time(), level="critical", rule="saturation",
                          summary_zh="贴轨")
    r = _client().post(f"/api/monitoring/alerts/{aid}/ack")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["acked"] is True

    rows, _ = store.alerts_query(limit=10)
    row = next(x for x in rows if x["id"] == aid)
    assert row["acked"] == 1, "路由答了 ok,但表里没落账"


def test_ack_does_not_touch_the_agent_delivery_subject():
    """人点掉 ≠ agent 看过。合成一个字段会让两个都问不出来。"""
    store = ST.get_store()
    aid = store.add_alert(ts=time.time(), level="critical", rule="saturation",
                          summary_zh="贴轨")
    _client().post(f"/api/monitoring/alerts/{aid}/ack")

    rows, _ = store.alerts_query(limit=10)
    row = next(x for x in rows if x["id"] == aid)
    assert row["acked"] == 1
    assert row["delivered_agent"] == 0, "点掉顺手替 agent 确认了"
    # 而且人点掉的这条**仍然**要送给 agent。
    left = store.undelivered_alerts(0.0, limit=50)
    assert any(x["id"] == aid for x in left), "人点掉之后 agent 就再也看不到了"


def test_the_alert_row_exposes_whether_the_agent_saw_it():
    """delivered_agent 必须可从 API 读取，用于核验告警是否送达代理。"""
    store = ST.get_store()
    aid = store.add_alert(ts=time.time(), level="critical", rule="saturation",
                          summary_zh="贴轨")
    store.mark_alerts_delivered([aid])
    body = _client().get("/api/monitoring/alerts").json()
    row = next(x for x in body["alerts"] if x["id"] == aid)
    assert row["delivered_agent"] is True
    assert row["acked"] is False
