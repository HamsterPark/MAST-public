"""Phase-3 contract tests for the Vision / scan-map / experimental one-shot slice.

Mirrors the read-only contract tests: every endpoint returns its declared status
and shape, and the degraded paths are empty-but-not-broken (never 500). The API
under test is NOT mounted in app.py yet (integration wires that) — we build a
throwaway FastAPI with just this router and a bare AppContext (standalone: no
BufferService / no live app wired → everything degrades), exactly per the brief.

The FFT endpoint is pure compute (no live core), so it is exercised with a real
sinusoid and asserted to produce a non-degraded spectrum whose peak lands at the
injected frequency — the end-to-end one-shot proof.
"""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.vision import router


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture()
def client() -> TestClient:
    return _client()


# ── GET /api/vision/recent ──────────────────────────────────────────

def test_vision_recent_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/vision/recent")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["frames"] == []
    assert body["count"] == 0


# ── GET /api/vision/pulse ───────────────────────────────────────────

def test_vision_pulse_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/vision/pulse")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    # idle defaults — empty but not broken
    assert body["alert_level"] == "idle"
    assert body["trend"] == "idle"
    assert body["dino_score"] is None
    assert body["recent_count"] == 0


class _StubBufferForPulse:
    """Minimal BufferService surface the pulse rollup touches."""

    def get_event_history(self, since_seqno=-1, limit=200):
        return []

    def get_stats(self):
        return {}

    def get_latest_tip_status(self):
        from mast.buffer.schemas import TipQuality, TipStatus
        return TipStatus(seqno=1, quality=TipQuality.GOOD, confidence=0.9,
                         scan_id="s1", frame_idx=0), 1


def _wired_client() -> TestClient:
    app = FastAPI()
    ctx = AppContext()
    ctx.buffer = _StubBufferForPulse()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_vision_pulse_flags_safe_mode_override() -> None:
    """In SAFE every tip verdict is rewritten to "good" at its producer, so the
    tip_quality this endpoint reports is manufactured. The flag is how a human
    is told that — without it the operator reads a mode as a measurement."""
    from mast.core.operating_mode import bind_mode_source

    c = _wired_client()
    try:
        bind_mode_source(lambda: "auto")
        assert c.get("/api/vision/pulse").json()["safe_mode"] is False

        bind_mode_source(lambda: "safe")
        body = c.get("/api/vision/pulse").json()
        assert body["safe_mode"] is True
        assert body["tip_quality"] == "good"

        bind_mode_source(None)          # unbound → no claim of an override
        assert c.get("/api/vision/pulse").json()["safe_mode"] is False
    finally:
        bind_mode_source(None)
    assert body["critical_count"] == 0


# ── GET /api/scan-map ───────────────────────────────────────────────

def test_scan_map_degrades_unwired(client: TestClient) -> None:
    """核心没接线时的降级形状。**前提由这条测试自己制造,不靠套件顺序。**

    2026-08-12:并行跑(pytest-xdist)时这条红了,串行绿。原因不是并行——
    ``markers`` 里的「计划路线」来自 ``PlanOverlay`` **进程级单例**,别的目录里
    某个测试往里写过东西,这条就看到非空。串行能过,只是因为那个测试恰好排在后面。

    **一个断言「什么都没有」的测试,必须自己把它清空。** 否则它测的是套件顺序,
    而套件顺序是没人拥有的东西 —— 换个跑法、加一个文件、开并行,它就变了。
    """
    from mast.io.plan_overlay import get_plan_overlay
    get_plan_overlay().set_plan([])          # 显式制造「没有计划路线」这个前提

    r = client.get("/api/scan-map")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["markers"] == []
    assert body["marker_count"] == 0
    assert body["frame"] is None
    assert body["image_b64"] is None
    # tip_xyz is always present (empty), never null
    assert body["tip_xyz"] == {"x_m": None, "y_m": None, "z_m": None}


# ── POST /api/experimental/fft ──────────────────────────────────────

def test_fft_too_few_samples_not_degraded(client: TestClient) -> None:
    # < 4 samples → faithful empty result (compute_fft's n<4 guard), not an error.
    r = client.post("/api/experimental/fft", json={"samples": [1.0, 2.0], "fs_hz": 10.0})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False
    assert body["spectrum"] == []
    assert body["n_samples"] == 2


def test_fft_empty_samples(client: TestClient) -> None:
    r = client.post("/api/experimental/fft", json={"samples": []})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False
    assert body["freqs_hz"] == [] and body["spectrum"] == []


def test_fft_real_sinusoid_peak(client: TestClient) -> None:
    # Pure one-shot compute (no live core). Inject a 50 Hz tone at fs=1000 Hz.
    fs = 1000.0
    f0 = 50.0
    n = 1024
    samples = [math.sin(2.0 * math.pi * f0 * (i / fs)) for i in range(n)]
    r = client.post(
        "/api/experimental/fft",
        json={"samples": samples, "fs_hz": fs, "window": "hann",
              "output": "magnitude", "unit": "V", "channel_name": "test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["degraded"] is False
    assert body["n_samples"] == n
    assert body["fs_hz"] == fs
    assert body["nyquist_hz"] == fs / 2.0
    assert len(body["freqs_hz"]) == len(body["spectrum"]) == n // 2 + 1
    assert body["unit"] == "V" and body["channel_name"] == "test"
    # Peak bin (skip DC) must land at the injected 50 Hz.
    spec = body["spectrum"]
    freqs = body["freqs_hz"]
    peak_idx = max(range(1, len(spec)), key=lambda i: spec[i])
    assert abs(freqs[peak_idx] - f0) <= body["df_hz"] * 1.5


def test_fft_derives_fs_from_duration(client: TestClient) -> None:
    n = 64
    samples = [float(i % 8) for i in range(n)]
    r = client.post("/api/experimental/fft",
                    json={"samples": samples, "duration_s": 0.63})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["fs_hz"] > 0.0  # derived from (n-1)/duration


# ── POST /api/experimental/mosaic ───────────────────────────────────

def test_mosaic_empty_dir(client: TestClient, tmp_path) -> None:
    # No .sxm files → core returns a 'no valid scans' result, ok=False, not 500.
    r = client.post("/api/experimental/mosaic", json={"directory": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["placed"] == 0
    assert body["image_b64"] is None
    # degraded only if the core itself was unavailable; with numpy/matplotlib
    # present it runs and returns an honest 'no scans' error.
    assert body["degraded"] is False
    assert body["error"]


def test_mosaic_missing_dir(client: TestClient) -> None:
    # A non-existent directory still must not 500 — empty/degraded result.
    r = client.post("/api/experimental/mosaic",
                    json={"directory": "Z:/definitely/not/here", "channel": "Z"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["placed"] == 0


# ── POST /api/experimental/monitor/start | stop ─────────────────────

def test_monitor_start_degrades_unwired(client: TestClient) -> None:
    r = client.post("/api/experimental/monitor/start",
                    json={"channel": "current", "interval_s": 5.0})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert body["status"]["running"] is False


def test_monitor_start_validates_interval(client: TestClient) -> None:
    # interval below the schema floor (0.5) is a 422 (Pydantic validation).
    r = client.post("/api/experimental/monitor/start",
                    json={"channel": "current", "interval_s": 0.1})
    assert r.status_code == 422


def test_monitor_stop_degrades_unwired(client: TestClient) -> None:
    r = client.post("/api/experimental/monitor/stop")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


# ── _resolve_scan_path (2026-07-01: stale / uncaptured-session-dir fallback) ──
# 近期帧 thumbnails rendered '未解码缩略图' when the event's file_path no longer
# resolved (scan moved, or saved under a Nanonis session dir the event never
# captured). The handler now recovers by basename across the live scan dirs.

def test_resolve_scan_path_returns_existing(tmp_path) -> None:
    from mast.api.routes.vision import _resolve_scan_path

    f = tmp_path / "scan_001.sxm"
    f.write_bytes(b"x")
    assert _resolve_scan_path(None, str(f)) == str(f)


def test_resolve_scan_path_none_for_empty() -> None:
    from mast.api.routes.vision import _resolve_scan_path

    assert _resolve_scan_path(None, None) is None
    assert _resolve_scan_path(None, "") is None


def test_resolve_scan_path_basename_match_via_session_dir(tmp_path) -> None:
    """A dead path is recovered by basename in the live Nanonis session dir — the
    win once the session dir is actually captured (Util_SessionPathGet)."""
    from mast.api.routes.vision import _resolve_scan_path

    real = tmp_path / "scan_042.sxm"
    real.write_bytes(b"x")
    stale = r"C:\gone\old\scan_042.sxm"  # same basename, dead path

    class _App:
        def _resolve_session_dir(self):
            return str(tmp_path)

    assert _resolve_scan_path(_App(), stale) == str(real)


# ── scan_images surface-mosaic underlay (2026-07-01 scan-map redesign) ──────
# get_scan_map places saved .sxm scans on the map by their REAL stage footprint
# (centre/size from the header) + a b64 thumbnail — the map shows the actual
# surface, not just outlines.

def test_recent_scan_images_places_sxm_by_footprint(tmp_path, monkeypatch) -> None:
    import shutil
    import time as _t
    from pathlib import Path as _P
    from types import SimpleNamespace
    from mast.api.routes.vision import _recent_scan_images, _SCAN_IMG_CACHE

    src = _P(__file__).resolve().parents[4] / "stm-datasets" / "repos" / "ML-STM" / "example data" / "STM_WTip_WSe2-SL445_023.sxm"
    if not src.is_file():
        import pytest
        pytest.skip("no .sxm fixture")
    dst = tmp_path / "scan_023.sxm"
    shutil.copy2(src, dst)
    import os
    os.utime(dst, (_t.time(), _t.time()))  # fresh mtime so it's a "recent" scan
    _SCAN_IMG_CACHE["key"] = None  # bust the module cache
    # The search also walks <project_root>/working-sessions, i.e. the developer's
    # OWN scans. This used to read `imgs[0]` and pass only because the copy above
    # was the newest file of all — when the underlay was reordered oldest-first
    # (#94, so the newest frame paints on top) index 0 became a real scan from
    # 20260327 and the footprint assertions failed against someone's data.
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "_root"))

    app = SimpleNamespace(_resolve_session_dir=lambda: str(tmp_path), config=None)
    imgs = _recent_scan_images(app)
    assert imgs, "no scan images assembled from a real .sxm"
    # Pick the file this test wrote — position in the list is the DRAW ORDER, a
    # separate contract (see tests/v2/unit/io/test_scan_map_underlay.py).
    im = next(i for i in imgs if i.name == "scan_023.sxm")
    assert im.image_b64  # a real base64 thumbnail
    assert abs(im.center_x_m - 3.877405e-08) < 1e-12  # scan_offset from header
    assert abs(im.width_m - 4.999998e-09) < 1e-15      # scan_range from header
    assert im.name.endswith(".sxm")


def test_recent_scan_images_never_raises() -> None:
    from mast.api.routes.vision import _recent_scan_images
    # No app → no session dir, but it still falls back to <root>/working-sessions
    # (so scans show even before a session dir resolves). Contract: return a list,
    # never raise.
    out = _recent_scan_images(None)
    assert isinstance(out, list)


def test_scan_map_endpoint_end_to_end(tmp_path) -> None:
    """Capstone: a fully-wired app (storage markers + live state + session .sxm)
    → GET /api/scan-map returns EVERY op type's marker + the live frame/tip + the
    surface-mosaic scan image. This is the recording→display path the operator's
    goal checks ('验证所有记录能正确记录和显示')."""
    import os
    import shutil
    import time as _t
    from pathlib import Path as _P
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mast.api.context import AppContext
    from mast.api.routes.vision import router, _SCAN_IMG_CACHE
    from mast.core.types import HardwareState
    from mast.logging.storage import ExperimentStorage

    EXP, SAMP = "e2e-exp", "e2e-samp"
    storage = ExperimentStorage(str(tmp_path / "e.db"))
    for kind, x, y, w, h in [("scan", 1e-8, 2e-8, 5e-8, 5e-8), ("sts", 2e-8, 3e-8, None, None),
                             ("tip_shape", 3e-8, 1e-8, None, None), ("pulse", 4e-8, 2e-8, None, None),
                             ("move", 6e-8, -3e-8, None, None)]:
        storage.log_marker(kind=kind, x_m=x, y_m=y, w_m=w, h_m=h, label=kind,
                           skill_name="sim", experiment_id=EXP, sample_id=SAMP)

    scans_dir = tmp_path / "sess"
    scans_dir.mkdir()
    src = _P(__file__).resolve().parents[4] / "stm-datasets" / "repos" / "ML-STM" / "example data" / "STM_WTip_WSe2-SL445_023.sxm"
    have_sxm = src.is_file()
    if have_sxm:
        d = scans_dir / "s.sxm"
        shutil.copy2(src, d)
        os.utime(d, (_t.time(), _t.time()))
    _SCAN_IMG_CACHE["key"] = None

    st = HardwareState()
    st.scan_center_x_m, st.scan_center_y_m = 1e-8, 1e-8
    st.scan_width_m, st.scan_height_m = 5e-8, 5e-8
    st.x_pos_m, st.y_pos_m, st.z_pos_m = 1e-8, 1e-8, 1e-9

    app_stub = SimpleNamespace(
        _state=SimpleNamespace(snapshot=lambda: st),
        _experiment_log=SimpleNamespace(current_experiment_id=EXP, current_sample_id=SAMP),
        _storage=storage,
        _resolve_session_dir=lambda: str(scans_dir),
        config=None,
    )
    ctx = AppContext()
    ctx.app = app_stub
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")

    body = TestClient(app).get("/api/scan-map").json()
    assert body["degraded"] is False
    kinds = {m["kind"] for m in body["markers"]}
    assert {"scan", "sts", "tip_shape", "pulse", "move"} <= kinds, f"missing op kinds: {kinds}"
    assert body["frame"] and body["frame"]["width_m"] == 5e-8   # live current frame
    assert body["tip_xyz"]["x_m"] == 1e-8                        # live tip
    if have_sxm:
        assert body["scan_images"], "surface-mosaic scan image missing"
        assert body["scan_images"][0]["image_b64"]
        assert body["scan_images"][0]["width_m"]  # placed by real footprint


# ── POST /api/scan-map/import — operator imports their OWN scans/spectra ─────

def _import_client(tmp_path, *, exp_id="imp-e"):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from mast.api.context import AppContext
    from mast.api.routes.vision import router
    from mast.logging.storage import ExperimentStorage

    storage = ExperimentStorage(str(tmp_path / "e.db"))
    app_stub = SimpleNamespace(
        _storage=storage,
        _experiment_log=SimpleNamespace(current_experiment_id=exp_id, current_sample_id="imp-s"),
        _resolve_session_dir=lambda: None, _state=None, config=None)
    ctx = AppContext()
    ctx.app = app_stub
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app), storage


def test_scan_map_import_places_operator_files(tmp_path) -> None:
    import shutil
    from pathlib import Path as _P
    from mast.api.routes.vision import _SCAN_IMG_CACHE

    imp = tmp_path / "mydata"
    imp.mkdir()
    sxm = _P(__file__).resolve().parents[4] / "stm-datasets" / "repos" / "ML-STM" / "example data" / "STM_WTip_WSe2-SL445_023.sxm"
    dat = _P(__file__).resolve().parents[2] / "fixtures" / "nanonis" / "bias_spectroscopy_200pt.dat"
    have_sxm = sxm.is_file()
    if have_sxm:
        shutil.copy2(sxm, imp / "myscan.sxm")
    if dat.is_file():
        shutil.copy2(dat, imp / "myspec.dat")
    _SCAN_IMG_CACHE["key"] = None

    client, storage = _import_client(tmp_path)
    body = client.post("/api/scan-map/import", json={"path": str(imp)}).json()
    assert body["ok"], body
    if have_sxm:
        assert body["scans"] >= 1

    rows = storage.get_markers("imp-e", "imp-s")
    assert rows and all(r["source"] == "import" for r in rows)
    if have_sxm:
        assert any(r["kind"] == "scan" and (r["meta"] or {}).get("file") for r in rows)
        # the imported .sxm (outside any searched dir) still shows as a thumbnail,
        # via get_scan_map passing marker meta.file as extra_paths.
        smap = client.get("/api/scan-map").json()
        assert any(m["kind"] == "scan" for m in smap["markers"])
        assert smap["scan_images"], "imported scan not shown as a surface thumbnail"


def test_scan_map_import_requires_active_experiment(tmp_path) -> None:
    client, _ = _import_client(tmp_path, exp_id=None)
    body = client.post("/api/scan-map/import", json={"path": str(tmp_path)}).json()
    assert body["ok"] is False and "实验" in body["message"]


def test_scan_map_import_degrades_standalone(client: TestClient) -> None:
    # the shared client fixture wires NO live app → import degrades, never 500s.
    body = client.post("/api/scan-map/import", json={"path": "whatever"}).json()
    assert body["ok"] is False and body["degraded"] is True


# ── OpenAPI contract — all paths exposed for the frontend typegen ───

def test_openapi_exposes_all_paths(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    paths = set(spec.get("paths", {}))
    assert {
        "/api/vision/recent",
        "/api/vision/pulse",
        "/api/scan-map",
        "/api/experimental/fft",
        "/api/experimental/mosaic",
        "/api/experimental/monitor/start",
        "/api/experimental/monitor/stop",
    } <= paths
    schemas = spec.get("components", {}).get("schemas", {})
    assert {"VisionPulseResponse", "ScanMapResponse", "FFTResponse",
            "MosaicResponse", "MonitorActionResponse"} <= set(schemas)
