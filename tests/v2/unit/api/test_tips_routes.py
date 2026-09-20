"""针尖登记 API 契约。

要点:
  * ``/tips/current`` 必须**声明在** ``/tips/{tip_id}`` 之前,否则 "current" 会
    被当成一个 tip_id —— 这是本仓踩过的形状(群聊转录端点被 {agent_id} 遮蔽)。
  * 没接上核心时**降级不 500**:返回 degraded=True,前端据此显示「读不到」
    而不是「没有针尖」。这两件事在 UI 上必须能分开。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.tips import router
from mast.core import instrument_profile as iprof
from mast.core import tip_state


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _wired(tmp_path) -> tuple[TestClient, object]:
    from mast.logging.storage import ExperimentStorage

    storage = ExperimentStorage(str(tmp_path / "tips.db"))
    ctx = AppContext()
    ctx.wire(experiment_storage=storage)
    return _client(ctx), storage


# ── 降级 ────────────────────────────────────────────────────────────────────

def test_current_degrades_without_a_store() -> None:
    r = _client().get("/api/tips/current")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["tip"] is None


def test_list_degrades_without_a_store() -> None:
    r = _client().get("/api/tips")
    assert r.status_code == 200
    assert r.json()["degraded"] is True and r.json()["tips"] == []


def test_writes_degrade_without_a_store() -> None:
    c = _client()
    assert c.post("/api/tips", json={"material": "W"}).json()["ok"] is False
    assert c.post("/api/tips/current/remove").json()["ok"] is False
    assert c.patch("/api/tips/x", json={"note": "y"}).json()["ok"] is False


def test_degraded_is_distinct_from_no_tip_registered(tmp_path) -> None:
    """「读不到」和「没有针尖」在 UI 上是两回事,不能都渲染成空。"""
    c, _ = _wired(tmp_path)
    body = c.get("/api/tips/current").json()
    assert body["degraded"] is False
    assert body["registered"] is False
    assert body["hint"]


# ── 路由顺序 ────────────────────────────────────────────────────────────────

def test_current_is_not_swallowed_by_the_id_route(tmp_path) -> None:
    """FastAPI 按声明序匹配。/tips/{tip_id} 若在前,"current" 就成了一个 id。"""
    c, _ = _wired(tmp_path)
    r = c.get("/api/tips/current")
    assert r.status_code == 200
    assert "registered" in r.json()          # CurrentTipResponse 的形状


def test_vocabulary_endpoint_is_reachable(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    body = c.get("/api/tips/vocabulary").json()
    assert any(m["value"] == "W" for m in body["materials"])
    assert any(f["value"] == "etched" for f in body["fabrications"])
    assert any(f["value"] == "qplus" for f in body["forms"])


# ── 实路径 ──────────────────────────────────────────────────────────────────

def test_register_then_current_round_trips(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    r = c.post("/api/tips", json={"material": "钨", "fabrication": "电化学腐蚀",
                                  "wire_diameter_mm": 0.25})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["tip"]["material"] == "W"
    assert body["tip"]["fabrication"] == "etched"

    cur = c.get("/api/tips/current").json()
    assert cur["registered"] is True
    assert cur["tip"]["wire_diameter_mm"] == 0.25
    tip_state.set_current_tip(None)


def test_register_reports_cleared_calibration(tmp_path) -> None:
    """前端要能告诉用户「这些标定被清了,需要重做」。"""
    c, _ = _wired(tmp_path)
    c.post("/api/tips", json={"material": "W"})
    iprof.set_profile({"didv_at_contact_v": 2e-3, "qplus_amplitude_baseline": 9.0})

    body = c.post("/api/tips", json={"material": "PtIr"}).json()
    assert "didv_at_contact_v" in body["cleared_calibration"]
    assert "qplus_amplitude_baseline" in body["cleared_calibration"]
    iprof.set_profile({})
    tip_state.set_current_tip(None)


def test_list_returns_the_change_history(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    c.post("/api/tips", json={"material": "W", "name": "first"})
    c.post("/api/tips", json={"material": "PtIr", "name": "second"})
    tips = c.get("/api/tips").json()["tips"]
    assert [t["name"] for t in tips] == ["second", "first"]
    assert tips[1]["removed_at"], "上一根针应已退役"
    tip_state.set_current_tip(None)


def test_retired_row_carries_the_archived_calibration(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    c.post("/api/tips", json={"material": "W"})
    iprof.set_profile({"didv_at_contact_v": 3.3e-3})
    c.post("/api/tips", json={"material": "PtIr"})

    retired = [t for t in c.get("/api/tips").json()["tips"] if t["removed_at"]][0]
    assert retired["retire_snapshot"]["didv_at_contact_v"] == 3.3e-3
    iprof.set_profile({})
    tip_state.set_current_tip(None)


def test_patch_updates_and_does_not_clear_calibration(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    tip_id = c.post("/api/tips", json={"material": "W"}).json()["tip"]["id"]
    iprof.set_profile({"didv_at_contact_v": 2e-3})

    body = c.patch(f"/api/tips/{tip_id}", json={"note": "补记一句"}).json()
    assert body["ok"] is True and body["tip"]["note"] == "补记一句"
    assert iprof.get_profile().get("didv_at_contact_v") == 2e-3
    iprof.set_profile({})
    tip_state.set_current_tip(None)


def test_patch_on_a_missing_tip_is_an_honest_miss(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    body = c.patch("/api/tips/no-such-id", json={"note": "x"}).json()
    assert body["ok"] is False and body["error"]


def test_remove_current(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    c.post("/api/tips", json={"material": "W"})
    body = c.post("/api/tips/current/remove").json()
    assert body["ok"] is True and body["changed"] is True
    assert c.get("/api/tips/current").json()["registered"] is False
    tip_state.set_current_tip(None)


def test_remove_with_nothing_registered_is_not_an_error(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    body = c.post("/api/tips/current/remove").json()
    assert body["ok"] is True and body["changed"] is False


def test_unknown_vocabulary_comes_back_as_warnings(tmp_path) -> None:
    c, _ = _wired(tmp_path)
    body = c.post("/api/tips", json={"material": "镝钪合金"}).json()
    assert body["ok"] is True
    assert body["warnings"]
    tip_state.set_current_tip(None)


def test_router_is_mounted_in_the_real_app() -> None:
    """路由写了但没进 app.py 的 modules 列表 = 端点根本不存在(本仓踩过)。"""
    from mast.api.app import create_app

    paths = {r.path for r in create_app().routes}
    assert "/api/tips/current" in paths
    assert "/api/tips" in paths
