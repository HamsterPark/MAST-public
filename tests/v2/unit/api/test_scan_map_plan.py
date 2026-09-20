"""The planned route must actually reach the scan map.

`show_plan_on_map` answered
"计划路线已显示在扫描地图上,执行时会自动推进。" and the map stayed empty.

It wrote into the process-local ``PlanOverlay`` singleton — which had NO READER.
``GET /api/scan-map`` only ever read ``storage.get_markers()`` plus the hardware
snapshot; nothing in the codebase called ``PlanOverlay.snapshot()`` outside tests.
The frontend was ready the whole time (``ScanMapCanvas`` filters
``status === "planned"`` and draws a dashed route, and the legend has a 「计划」
entry that could never light up) — it was simply never sent any planned markers.

This is the same family as the failures 2026-07-10 catalogued: *the reported
success was assumed, not verified*. ``PlanOverlay.set_plan`` cannot fail, so
"nothing threw" was the entire basis for the claim.

``test_exp_map.py`` covered the ``PlanOverlay`` class itself (set/clear/advance
all passed) — which is exactly why this stayed green while the wiring was dead.
These tests assert the WIRING: does a published plan come back out of the HTTP
endpoint.
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
from mast.io.exp_map import MapMarker  # noqa: E402
from mast.io.plan_overlay import get_plan_overlay  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clean_overlay():
    """The overlay is a process-level singleton — leaking a plan between tests
    would make later assertions lie."""
    get_plan_overlay().clear()
    yield
    get_plan_overlay().clear()


def _plan(*steps):
    get_plan_overlay().set_plan(
        [MapMarker(kind=k, x_m=x, y_m=y, w_m=1e-7, h_m=1e-7, label=lbl,
                   status="planned", source="plan")
         for k, x, y, lbl in steps],
        title="route",
    )


def _planned(body) -> list[dict]:
    return [m for m in (body.get("markers") or []) if m.get("status") == "planned"]


def test_published_plan_appears_in_the_response(client):
    """THE regression: this returned zero planned markers before the fix."""
    _plan(("scan", 1e-7, 2e-7, "step1"),
          ("scan", 3e-7, 4e-7, "step2"),
          ("sts", 5e-7, 6e-7, "step3"))
    pl = _planned(client.get("/api/scan-map").json())
    assert len(pl) == 3
    assert [m["label"] for m in pl] == ["step1", "step2", "step3"], (
        "order is the route order — the frontend draws the polyline in array order"
    )
    assert all(m["source"] == "plan" for m in pl)


def test_empty_overlay_yields_no_planned_markers(client):
    assert _planned(client.get("/api/scan-map").json()) == []


def test_planned_steps_do_not_inflate_marker_count(client):
    """`marker_count` means "operations recorded", not "shapes drawn"."""
    _plan(("scan", 1e-7, 2e-7, "a"), ("scan", 2e-7, 3e-7, "b"))
    body = client.get("/api/scan-map").json()
    assert len(_planned(body)) == 2
    assert body["marker_count"] == 0


def test_advance_removes_the_reached_step(client):
    """The tool promises "执行时会自动推进" — that must be observable."""
    _plan(("scan", 1e-7, 2e-7, "a"), ("scan", 3e-7, 4e-7, "b"))
    get_plan_overlay().advance(1e-7, 2e-7)
    assert len(_planned(client.get("/api/scan-map").json())) == 1


def test_clear_removes_the_route(client):
    _plan(("scan", 1e-7, 2e-7, "a"))
    get_plan_overlay().clear()
    assert _planned(client.get("/api/scan-map").json()) == []


def test_plan_survives_a_degraded_map(client):
    """No live core wired (the standalone case) still shows the plan — it is
    process-local state and does not need one."""
    _plan(("scan", 1e-7, 2e-7, "a"))
    body = client.get("/api/scan-map").json()
    assert body["degraded"] is True
    assert len(_planned(body)) == 1


def test_no_wasted_png_render(client):
    """Nothing in the frontend reads ScanMapResponse.image_b64 (the map is drawn
    client-side by react-konva). Rendering it cost ~110 ms of matplotlib plus
    ~30 KB of base64 on every 3-second poll, per client, and was discarded."""
    assert client.get("/api/scan-map").json()["image_b64"] is None


# ── the tool must report what it can verify ─────────────────────────────────

def _call(tool, **kw):
    import json
    return json.loads(tool.invoke(kw))


def _plan_tools():
    """The two map-plan tools, built off a minimal provider.

    The plan tools touch only the PlanOverlay singleton, so an empty context is
    enough — no live core, no storage.
    """
    from mast.agents._shared.meta_tools import make_meta_tools
    tools = {t.name: t for t in make_meta_tools(lambda: {})}
    return tools["show_plan_on_map"], tools["clear_plan_on_map"]


def test_show_plan_reports_the_stored_count_not_the_input_count():
    show, _ = _plan_tools()
    r = _call(show, steps=[{"kind": "scan", "x_m": 1e-7, "y_m": 2e-7},
                           {"kind": "scan", "x_m": 3e-7, "y_m": 4e-7}])
    assert r["success"] is True
    assert r["steps"] == len(get_plan_overlay().snapshot()) == 2


def test_show_plan_claims_only_what_it_verified():
    """The old message asserted the operator's screen state ("已显示在扫描地图上")
    — something the tool cannot observe, and which was false for months."""
    show, _ = _plan_tools()
    r = _call(show, steps=[{"kind": "scan", "x_m": 1e-7, "y_m": 2e-7}])
    assert "已发布" in r["message"]


def test_show_plan_fails_when_the_store_does_not_take_it(monkeypatch):
    """The product gate itself: a write that does not land must not report success."""
    from mast.io import plan_overlay as po
    show, _ = _plan_tools()
    monkeypatch.setattr(po.PlanOverlay, "set_plan", lambda self, *a, **k: None)
    r = _call(show, steps=[{"kind": "scan", "x_m": 1e-7, "y_m": 2e-7}])
    assert r["success"] is False
    assert "回读为空" in r["error"]


def test_clear_says_how_many_it_removed():
    show, clear = _plan_tools()
    _call(show, steps=[{"kind": "scan", "x_m": 1e-7, "y_m": 2e-7},
                       {"kind": "scan", "x_m": 3e-7, "y_m": 4e-7}])
    assert _call(clear)["cleared"] == 2
    again = _call(clear)
    assert again["cleared"] == 0
    assert "本来就没有" in again["message"], "clearing nothing must not read as success"
