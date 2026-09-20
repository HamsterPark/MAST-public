"""環境歷史 endpoints.

Two house rules get explicit coverage because they are what let this page exist
at all: every handler degrades instead of returning 500, and no handler touches
hardware. The third is the honesty requirement on
``/experiments/{id}/environment`` — when the per-reading detail has aged out and
the answer comes from statistics buckets instead, the response has to say so.
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
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.env_history import router
from mast.envhistory import store as ST
from mast.envhistory import thresholds as TH
from mast.envhistory.buckets import BucketAccumulator
from mast.envhistory.recorder import set_recorder

T0 = 1_800_000_000.0
DT = 60.0


@pytest.fixture()
def store(tmp_path):
    s = ST.EnvHistoryStore(tmp_path / "eh.sqlite")
    ST.set_store_for_test(s)
    TH.set_env_history_thresholds(None)
    set_recorder(None)
    yield s
    ST.set_store_for_test(None)
    TH.set_env_history_thresholds(None)
    set_recorder(None)


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _seed(store, sensor="temperature", n=180, unit="K"):
    acc = BucketAccumulator(sensor)
    rows = []
    for i in range(n):
        b = acc.add(T0 + i * 2.0, 77.0 + 0.01 * i, unit, "ok", dt_s=DT)
        if b:
            rows.append(b)
    tail = acc.flush()
    if tail:
        rows.append(tail)
    store.upsert_buckets(rows)


# ── config / status ─────────────────────────────────────────────────────────

def test_config_ships_the_knob_catalogue(store):
    r = _client().get("/api/env-history/config")
    assert r.status_code == 200
    j = r.json()
    assert j["degraded"] is False
    keys = {k["key"] for k in j["knobs"]}
    assert "eh_enabled" in keys and "eh_raw_keep_days" in keys
    assert j["enabled"] is True          # recording is the default
    assert j["z_enabled"] is False       # the one TCP-issuing knob is not


def test_status_without_a_recorder_still_reports_the_store(store):
    _seed(store)
    j = _client().get("/api/env-history/status").json()
    assert j["recording"] is False       # nothing is recording…
    assert j["store"]["bucket_rows"] == 6   # …but the archive is readable
    assert j["degraded"] is False


def test_status_reflects_a_live_recorder(store):
    from mast.envhistory.recorder import EnvHistoryRecorder
    rec = EnvHistoryRecorder(store_getter=lambda: store)
    set_recorder(rec)
    j = _client().get("/api/env-history/status").json()
    assert j["recording"] is True
    assert set(j["spectra"]) == {"current", "z"}


# ── series ──────────────────────────────────────────────────────────────────

def test_sensors_lists_what_is_archived(store):
    _seed(store)
    _seed(store, sensor="vacuum", n=60, unit="Pa")
    names = {s["sensor"] for s in _client().get("/api/env-history/sensors").json()["sensors"]}
    assert names == {"temperature", "vacuum"}


def test_series_returns_points(store):
    _seed(store)
    j = _client().get("/api/env-history/series?sensor=temperature").json()
    assert j["unit"] == "K"
    assert len(j["points"]) == 6
    assert j["thinned"] is False
    assert j["bucket_s_effective"] == DT


def test_series_thins_and_says_so(store):
    _seed(store, n=3600)     # two hours of buckets
    j = _client().get("/api/env-history/series?sensor=temperature&max_points=10").json()
    assert j["thinned"] is True
    assert len(j["points"]) <= 10
    assert j["bucket_s_effective"] > DT


def test_max_points_is_clamped_server_side(store):
    _seed(store)
    j = _client().get("/api/env-history/series?sensor=temperature&max_points=99999999").json()
    assert j["degraded"] is False        # a huge request is bounded, not fatal


def test_series_of_an_unknown_sensor_is_empty_not_404(store):
    j = _client().get("/api/env-history/series?sensor=nope").json()
    assert j["points"] == [] and j["degraded"] is False


def test_series_respects_since(store):
    _seed(store)
    j = _client().get(f"/api/env-history/series?sensor=temperature&since={T0 + 120}").json()
    assert all(p["ts"] >= T0 + 120 for p in j["points"])


# ── spectra ─────────────────────────────────────────────────────────────────

def test_spectra_list_carries_metadata_only(store):
    store.add_spectrum(ts=T0, channel="current", span_s=1800.0, n_segments=8,
                       fs_hz=20000.0, freqs=[1.0, 2.0, 4.0],
                       psd=[1e-24, 5e-25, 2e-25])
    j = _client().get("/api/env-history/spectra?channel=current").json()
    assert len(j["spectra"]) == 1
    assert j["spectra"][0]["n_points"] == 3
    assert "psd" not in j["spectra"][0]


def test_one_spectrum_returns_the_arrays(store):
    rid = store.add_spectrum(ts=T0, channel="z", span_s=30.0, n_segments=30,
                             fs_hz=20000.0, freqs=[1.0, 2.0],
                             psd=[1e-24, 5e-25], unit="m^2/Hz")
    j = _client().get(f"/api/env-history/spectra/{rid}").json()
    assert j["found"] is True
    assert len(j["freqs_hz"]) == 2 and len(j["psd"]) == 2
    assert j["unit"] == "m^2/Hz"


def test_a_missing_spectrum_is_found_false_not_404(store):
    j = _client().get("/api/env-history/spectra/424242").json()
    assert j["found"] is False and j["degraded"] is False


def test_the_literal_spectra_path_is_not_shadowed_by_the_id_path(store):
    """Registration order matters: `/spectra` must not be parsed as an id."""
    r = _client().get("/api/env-history/spectra")
    assert r.status_code == 200
    assert "spectra" in r.json()


# ── per-experiment environment ──────────────────────────────────────────────

class _Storage:
    """Just enough ExperimentStorage for the endpoint."""

    def __init__(self, rows=None, exp=None):
        self._rows = rows or []
        self._exp = exp or {}

    def list_environment_sensors(self, experiment_id=None):
        return [{"sensor_name": n} for n in
                sorted({r["sensor_name"] for r in self._rows})] or [
            {"sensor_name": "temperature"}]

    def get_environment_history(self, sensor_name, **kw):
        return [r for r in self._rows if r["sensor_name"] == sensor_name]

    def get_experiment(self, experiment_id):
        return self._exp


def _iso(dt):
    return dt.isoformat()


def test_experiment_environment_prefers_raw_rows(store):
    now = datetime.now()
    rows = [{"timestamp": _iso(now - timedelta(minutes=i)),
             "sensor_name": "temperature", "value": 77.0 + i * 0.01,
             "unit": "K", "status": "ok", "sample_id": "S1"} for i in range(5)]
    ctx = AppContext()
    ctx.wire(experiment_storage=_Storage(rows))
    j = _client(ctx).get("/api/experiments/E1/environment").json()
    assert j["source"] == "raw"
    assert len(j["rows"]) == 5
    assert j["rows"][0]["unit"] == "K"
    assert j["rows"][0]["ts"] > 0        # ISO parsed into epoch for plotting


def test_experiment_environment_falls_back_to_buckets_and_says_so(store):
    """After a sweep the per-reading detail is gone; the buckets remain. The
    response must not present coarser data as if it were the raw record."""
    _seed(store)
    ctx = AppContext()
    ctx.wire(experiment_storage=_Storage(rows=[], exp={
        "started_at": _iso(datetime.fromtimestamp(T0)),
        "last_active_at": _iso(datetime.fromtimestamp(T0 + 400)),
    }))
    j = _client(ctx).get("/api/experiments/E1/environment").json()
    assert j["source"] == "buckets"
    assert j["points"]
    assert j["bucket_s_effective"] > 0


def test_experiment_environment_without_storage_degrades(store):
    j = _client().get("/api/experiments/E1/environment").json()
    assert j["degraded"] is True
    assert j["source"] == "none"


def test_experiment_environment_survives_a_storage_that_raises(store):
    class _Boom:
        def list_environment_sensors(self, **kw):
            raise RuntimeError("db gone")

    ctx = AppContext()
    ctx.wire(experiment_storage=_Boom())
    r = _client(ctx).get("/api/experiments/E1/environment")
    assert r.status_code == 200          # never a 500
    assert r.json()["degraded"] is True


# ── degradation ─────────────────────────────────────────────────────────────

def test_every_endpoint_degrades_when_the_store_is_gone(store):
    store.close()
    c = _client()
    for url in ("/api/env-history/status", "/api/env-history/sensors",
                "/api/env-history/series?sensor=temperature",
                "/api/env-history/spectra", "/api/env-history/spectra/1"):
        r = c.get(url)
        assert r.status_code == 200, url


def test_no_endpoint_touches_hardware(store, monkeypatch):
    """Zero TCP is the reason this page can poll at all."""
    import mast.core.connection as conn

    def explode(*a, **kw):
        raise AssertionError("an env-history endpoint issued a Nanonis call")

    monkeypatch.setattr(conn.ConnectionPool, "safe_call", explode, raising=False)
    _seed(store)
    c = _client()
    for url in ("/api/env-history/config", "/api/env-history/status",
                "/api/env-history/sensors",
                "/api/env-history/series?sensor=temperature",
                "/api/env-history/spectra"):
        assert c.get(url).status_code == 200
