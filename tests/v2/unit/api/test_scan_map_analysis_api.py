"""The map-analysis endpoint: the operator sees what the agent acts on.

``GET /api/scan-map/analysis`` calls the same ``analyze_map`` the agent's
``get_map_analysis`` tool calls. That shared path is the point — the button exists
so the programmatic decisions (coverage, keep-out zones, next position, whether to
relocate) can be inspected, and a second implementation would defeat it.

``POST /api/scan-map/coarse-move`` covers a blind spot: a coarse move made by hand
in Nanonis is unobservable, and every marker silently keeps a coordinate that no
longer points at the same surface.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return p / "MASTv2"
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != str(_ROOT):
    while str(_ROOT) in sys.path:
        sys.path.remove(str(_ROOT))
    sys.path.insert(0, str(_ROOT))
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.app import create_app  # noqa: E402
from mast.io.map_analysis import AnalysisConfig  # noqa: E402
from mast.logging.storage import ExperimentStorage  # noqa: E402


class _Log:
    current_experiment_id = "e1"
    current_sample_id = "s1"


class _FakeApp:
    """The three attributes the two routes read off the live core."""

    def __init__(self, storage, cfg):
        self._storage = storage
        self._experiment_log = _Log()
        self._state = None
        self._cfg = cfg

    def build_map_analysis_config(self):
        return self._cfg


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """A TestClient whose app.state.ctx exposes a fake core over a temp DB.

    The DB path is passed to ExperimentStorage directly, so there is no chance of
    redirecting one env var while storage reads another — the root cause every
    time this repo's tests have written into the operator's real records."""
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "exp.db"))
    storage = ExperimentStorage(str(tmp_path / "exp.db"))
    cfg = AnalysisConfig(piezo_half_range_m=1.5e-6, frame_size_m=100e-9,
                         strategy="center_first", tip_shape_r_m=30e-9,
                         crash_r_m=150e-9)
    app = create_app()
    fake = _FakeApp(storage, cfg)

    from mast.api.routes import vision as vision_route
    monkeypatch.setattr(vision_route, "_get_app", lambda ctx: fake)
    return TestClient(app), storage


def _mark(storage, kind, x_nm=0.0, y_nm=0.0, **kw):
    return storage.log_marker(
        kind=kind, x_m=x_nm * 1e-9, y_m=y_nm * 1e-9,
        experiment_id="e1", sample_id="s1", **kw)


@pytest.fixture(autouse=True)
def _empty_plan_overlay():
    """The route lives in a PROCESS-GLOBAL singleton, so a test that publishes
    one leaks it into every later test in this interpreter. Cleared on both
    sides: nothing inherited, nothing left behind."""
    from mast.io.plan_overlay import get_plan_overlay
    get_plan_overlay().clear()
    yield
    get_plan_overlay().clear()


# ── analysis endpoint ──────────────────────────────────────────────────────

def test_analysis_on_an_empty_record_recommends_the_route_start(wired):
    client, _ = wired
    body = client.get("/api/scan-map/analysis").json()
    assert body["degraded"] is False
    assert body["coverage_pct"] == 0.0
    assert body["usable_pct"] == 100.0
    assert body["strategy"] == "center_first"
    assert body["next_position"]["x_m"] == 0.0     # the low-creep centre
    assert body["coarse_advice"]["suggest"] is False


def test_analysis_reports_keepout_zones_with_geometry(wired):
    """The frontend draws these discs, so it needs centre + radius, not a count."""
    client, storage = wired
    _mark(storage, "tip_shape", 300, 0, label="修针尖")
    _mark(storage, "crash", -300, 0, label="撞针")
    body = client.get("/api/scan-map/analysis").json()
    zones = {z["kind"]: z for z in body["avoid_zones"]}
    assert zones["tip_shape"]["radius_m"] == pytest.approx(30e-9)
    assert zones["crash"]["radius_m"] == pytest.approx(150e-9)
    assert zones["crash"]["x_m"] == pytest.approx(-300e-9)
    assert body["damage_counts"] == {"tip_shape": 1, "crash": 1}


def test_analysis_moves_the_recommendation_off_damaged_surface(wired):
    client, storage = wired
    _mark(storage, "crash", 0, 0)          # ruin the centre
    body = client.get("/api/scan-map/analysis").json()
    assert (body["next_position"]["x_m"], body["next_position"]["y_m"]) != (0.0, 0.0)
    assert "避让区" in body["next_position"]["reason"]


def test_analysis_ignores_a_dead_coordinate_system(wired):
    """After a lateral coarse move the old numbers address different surface."""
    client, storage = wired
    _mark(storage, "crash", 0, 0)
    _mark(storage, "coarse_move")
    body = client.get("/api/scan-map/analysis").json()
    assert body["current_epoch"] == 1
    assert body["markers_total"] == 2
    assert body["markers_current_epoch"] == 0
    assert body["avoid_zones"] == []
    # The centre is fresh surface again.
    assert body["next_position"]["x_m"] == 0.0


def test_analysis_matches_the_agent_tool_on_the_same_record(wired):
    """One computation, two audiences — if these can differ, the panel is lying
    about what the agent will do."""
    import json
    client, storage = wired
    _mark(storage, "tip_shape", 0, 0)
    _mark(storage, "scan", 300, 0, w_m=1e-7, h_m=1e-7)

    from mast.agents._shared.meta_tools import make_meta_tools
    cfg = AnalysisConfig(piezo_half_range_m=1.5e-6, frame_size_m=100e-9,
                         strategy="center_first", tip_shape_r_m=30e-9,
                         crash_r_m=150e-9)
    tools = {t.name: t for t in make_meta_tools(
        lambda: {"storage": storage, "experiment_log": _Log(),
                 "map_analysis_cfg": lambda: cfg})}
    tool_out = json.loads(tools["get_map_analysis"].invoke({}))
    api_out = client.get("/api/scan-map/analysis").json()

    assert tool_out["coverage_pct"] == pytest.approx(api_out["coverage_pct"], abs=0.01)
    assert tool_out["usable_pct"] == pytest.approx(api_out["usable_pct"], abs=0.01)
    assert tool_out["coord_epoch"] == api_out["current_epoch"]
    assert tool_out["next_position"]["x_m"] == api_out["next_position"]["x_m"]
    assert tool_out["next_position"]["reason"] == api_out["next_position"]["reason"]


def test_analysis_degrades_without_a_live_core():
    """Never a 500: the page must render and say why it is empty."""
    client = TestClient(create_app())
    body = client.get("/api/scan-map/analysis").json()
    assert body["degraded"] is True
    assert body["detail"]
    assert body["upcoming"] == []      # no core, no route — not a stale one


# ── the route ahead (drawn on the map, not just the next step) ─────────────

def test_analysis_returns_the_route_ahead(wired):
    """The map draws where the survey is going, so one recommendation is not
    enough — and the route has to start at that recommendation."""
    client, storage = wired
    _mark(storage, "crash", 0, 0)              # push the start off centre
    body = client.get("/api/scan-map/analysis").json()
    up = body["upcoming"]
    assert len(up) >= 2
    assert (up[0]["x_m"], up[0]["y_m"]) == (
        body["next_position"]["x_m"], body["next_position"]["y_m"])


def test_route_ahead_covers_distinct_surface(wired):
    """Ghost frames that overlap would read as one blurred blob and would not
    be executable as a plan."""
    client, _ = wired
    up = client.get("/api/scan-map/analysis").json()["upcoming"]
    frame = 100e-9
    pts = [(p["x_m"], p["y_m"]) for p in up]
    assert len(pts) >= 4
    for i, (x, y) in enumerate(pts):
        for (px, py) in pts[i + 1:]:
            assert abs(x - px) >= frame or abs(y - py) >= frame


# ── fields the map poll carries for the view controls ─────────────────────

def test_scan_map_carries_the_piezo_half_range(wired):
    """The whole-range view must work with the analysis layers switched off, so
    the bound rides on the 3 s poll rather than the ~1 s analysis call."""
    client, _ = wired
    assert client.get("/api/scan-map").json()["piezo_half_range_m"] == pytest.approx(1.5e-6)


def test_scan_map_carries_the_plan_in_execution_order(wired):
    """A client numbers the steps and highlights the next one from the array
    order alone — so that order has to survive the trip, and the title (the one
    thing it cannot derive) has to come along."""
    from mast.io.exp_map import MapMarker
    from mast.io.plan_overlay import get_plan_overlay

    client, _ = wired
    get_plan_overlay().set_plan(
        [MapMarker(kind="scan", x_m=i * 200e-9, y_m=0.0, w_m=1e-7, h_m=1e-7,
                   label=f"第 {i + 1} 张") for i in range(3)],
        title="普查 3 张")
    body = client.get("/api/scan-map").json()
    planned = [m for m in body["markers"] if m["status"] == "planned"]
    assert [m["label"] for m in planned] == ["第 1 张", "第 2 张", "第 3 张"]
    assert body["plan_title"] == "普查 3 张"
    # And a planned step is not an operation that happened.
    assert body["marker_count"] == 0


# ── coarse-move backfill ───────────────────────────────────────────────────

def test_backfill_starts_a_new_generation(wired):
    client, storage = wired
    _mark(storage, "tip_shape", 0, 0)
    res = client.post("/api/scan-map/coarse-move",
                      json={"direction": "x+", "steps": 100, "note": "手动换区"})
    body = res.json()
    assert body["ok"] is True
    assert body["new_coord_epoch"] == 1
    # And the analysis immediately stops counting the old damage.
    assert client.get("/api/scan-map/analysis").json()["avoid_zones"] == []


def test_backfill_does_not_rewrite_history(wired):
    """It dates the boundary from the report, because there is no way to know
    when during the record the move actually happened."""
    client, storage = wired
    _mark(storage, "tip_shape", 0, 0)
    client.post("/api/scan-map/coarse-move", json={"direction": "x+"})
    rows = storage.get_markers("e1", "s1")
    assert [r["kind"] for r in rows] == ["tip_shape", "coarse_move"]
    assert [r["coord_epoch"] for r in rows] == [0, 0]
    assert rows[-1]["meta"]["manual_backfill"] is True


def test_backfill_accepts_an_empty_body(wired):
    """The operator may not remember the direction or the step count; the
    boundary itself is the load-bearing fact."""
    client, _ = wired
    assert client.post("/api/scan-map/coarse-move", json={}).json()["ok"] is True


def test_backfill_degrades_without_a_live_core():
    client = TestClient(create_app())
    body = client.post("/api/scan-map/coarse-move", json={}).json()
    assert body["ok"] is False
    assert body["degraded"] is True


# ── generation reaches the map itself ──────────────────────────────────────

def test_scan_map_exposes_the_generation_for_fading(wired):
    """The canvas fades stale markers on the ordinary 3 s poll, without needing
    the analysis endpoint — so both fields have to ride on /scan-map."""
    client, storage = wired
    _mark(storage, "scan", 0, 0, w_m=1e-7, h_m=1e-7)
    _mark(storage, "coarse_move")
    _mark(storage, "scan", 300, 0, w_m=1e-7, h_m=1e-7)
    body = client.get("/api/scan-map").json()
    assert body["current_epoch"] == 1
    epochs = [m["coord_epoch"] for m in body["markers"]
              if m["status"] != "planned"]
    assert epochs == [0, 0, 1]
