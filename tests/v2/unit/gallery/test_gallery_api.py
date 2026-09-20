# -*- coding: utf-8 -*-
"""``/api/gallery/*`` 契约（设计文档 §4.2、T3、T18、D9、D10）。

一次性 app 的写法照 ``tests/v2/unit/api/test_records_export.py``；另有一条对真实
``create_app()`` 做同样的 GET（与 ``test_boot_smoke`` 装配的是同一个 app），核对不落盘。
"""
from __future__ import annotations

import threading

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.gallery import router
from mast.gallery import service

GETS = [
    "/api/gallery/config", "/api/gallery/status", "/api/gallery/index", "/api/gallery/marks",
    "/api/gallery/marks/export/json", "/api/gallery/marks/export/md",
    "/api/gallery/marks/export/csv", "/api/gallery/marks/export/series_csv",
]


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_every_get_is_read_only_on_a_fresh_state_dir(gallery_dir):
    c = _client()
    for url in GETS:
        r = c.get(url)
        assert r.status_code == 200, (url, r.text)
        if not url.startswith("/api/gallery/marks/export"):
            # /marks 原样返回存盘的文档（不经 Pydantic 补默认键），没有 degraded 键即「没降级」。
            assert r.json().get("degraded") in (False, None), (url, r.json())
    assert c.get("/api/gallery/thumb/nope").status_code == 404
    assert not gallery_dir.exists(), "GET 在状态目录里留下了东西"


def test_the_real_app_gets_are_read_only_too(gallery_dir):
    """``test_boot_smoke`` 装配真实 app、逐个请求所有 GET；图库那几条在这里对着同一个
    app 走一遍，并核对全程没有创建状态目录。"""
    from mast.api.app import create_app

    c = TestClient(create_app())
    for url in GETS:
        assert c.get(url).status_code == 200, url
    assert c.get("/api/gallery/thumb/nope").status_code == 404
    assert not gallery_dir.exists()


def test_config_post_validates_before_writing(gallery_dir, tmp_path):
    c = _client()
    r = c.post("/api/gallery/config", json={"roots": [{"path": str(tmp_path / "missing")}]})
    body = r.json()
    assert r.status_code == 200 and body["ok"] is False and "不是已存在的目录" in body["detail"]
    assert not gallery_dir.exists()
    data = tmp_path / "SPM"
    data.mkdir()
    body = c.post("/api/gallery/config", json={"roots": [{"path": str(data)}], "workers": 3}).json()
    assert body["ok"] and body["workers"] == 3
    assert body["roots"] == [{"name": "SPM", "path": str(data), "enabled": True, "exists": True}]
    assert c.get("/api/gallery/config").json()["roots"][0]["name"] == "SPM"


def test_build_then_index_then_thumbnail(gallery_dir, tmp_path, synth, t0_ns):
    root = tmp_path / "SPM"
    synth.sxm(root / "d" / "a.sxm", np.random.default_rng(0).normal(0, 1e-11, (64, 64)), mtime_ns=t0_ns)
    V = np.linspace(-1.0, 1.0, 51)
    synth.dat(root / "d" / "s.dat", ["Bias calc (V)", "Current (A)"],
              np.column_stack([V, V * 1e-10]), mtime_ns=t0_ns + 10**9)
    c = _client()
    assert c.post("/api/gallery/config", json={"roots": [{"name": "SPM", "path": str(root)}]}).json()["ok"]
    started = c.post("/api/gallery/build", json={"force": False}).json()
    assert started["degraded"] is False
    assert service.wait(120)
    st = c.get("/api/gallery/status").json()
    assert st["phase"] == "done" and st["running"] is False and st["n_files"] == 2, st

    r = c.get("/api/gallery/index", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"
    doc = r.json()
    assert doc["built"] is True and len(doc["items"]) == 2
    plain = c.get("/api/gallery/index", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.json() == doc

    th = next(it["th"] for it in doc["items"] if it["k"] == "f")
    img = c.get(th)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert "immutable" in img.headers["cache-control"]
    for bad in ("/api/gallery/thumb/%2E%2E/config.json",
                "/api/gallery/thumb/..%5Cconfig.json",
                "/api/gallery/thumb/SPM/%2E%2E/%2E%2E/config.json"):
        assert c.get(bad).status_code == 404, bad


def test_a_second_build_request_reports_the_running_build(gallery_dir, monkeypatch):
    from mast.gallery import build

    gate = threading.Event()

    def slow(**kw):
        gate.wait(30)
        return {}

    monkeypatch.setattr(build, "run_build", slow)
    c = _client()
    first = c.post("/api/gallery/build", json={}).json()
    second = c.post("/api/gallery/build", json={}).json()
    assert first["running"] and second["running"] and first["started"] == second["started"]
    assert c.post("/api/gallery/build/cancel").json()["message"].startswith("正在取消")
    gate.set()
    assert service.wait(30)
    assert c.get("/api/gallery/status").json()["running"] is False


def test_a_status_read_while_a_build_is_being_started_never_says_running_false(gallery_dir, monkeypatch):
    """状态请求恰好落在「发布新构建」与「线程真正启动」之间时，不许读出 phase=inventory 而 running=False
    —— 轮询方会把它当成构建已经结束（出图任务槽在端到端里撞上过同一个窗口）。"""
    from mast.gallery import build

    gate = threading.Event()
    seen = []

    def slow(**kw):
        gate.wait(30)
        return {}

    class PeekingThread(threading.Thread):
        def start(self):
            seen.append(service.get_status())
            super().start()
            seen.append(service.get_status())

    monkeypatch.setattr(build, "run_build", slow)
    monkeypatch.setattr(service.threading, "Thread", PeekingThread)
    first = service.start_build()
    gate.set()
    assert service.wait(30)
    assert len(seen) == 2, "the peeking thread class was not used"
    for st in (*seen, first):
        assert not (st["phase"] == "inventory" and st["running"] is False), st
    assert (first["running"], first["phase"]) == (True, "inventory")


def test_marks_patch_read_export_and_import(gallery_dir):
    c = _client()
    r = c.post("/api/gallery/marks/patch",
               json={"items": {"R/a.sxm": {"r": 2, "tags": ["超结构"], "ts": 10}}, "tags": ["超结构", "漂移"]})
    body = r.json()
    assert body["ok"] is True and body["rev"] == 1 and body["updated"]
    doc = c.get("/api/gallery/marks").json()
    assert doc["items"]["R/a.sxm"] == {"r": 2, "tags": ["超结构"], "ts": 10}, "只存客户端发来的键"
    assert doc["tags"] == ["超结构", "漂移"]
    md = c.get("/api/gallery/marks/export/md")
    assert md.status_code == 200 and 'filename="marks.md"' in md.headers["content-disposition"]
    assert c.post("/api/gallery/marks/patch",
                  json={"items": {"R/a.sxm": {"del": True, "ts": 11}}}).json()["rev"] == 2
    assert "R/a.sxm" not in c.get("/api/gallery/marks").json()["items"]
    imp = c.post("/api/gallery/marks/import",
                 json={"doc": {"items": {"b.sxm": {"r": 1, "t": "2001-09-13 00:00:00"}}},
                       "key_prefix": "R"}).json()
    assert imp["ok"] and imp["items"] == 1 and imp["unmatched"] == 1
    assert c.get("/api/gallery/marks/export/xlsx").status_code == 422


def test_a_corrupt_marks_file_degrades_instead_of_500(gallery_dir):
    gallery_dir.mkdir(parents=True)
    (gallery_dir / "marks.json").write_text("{broken", encoding="utf-8")
    c = _client()
    r = c.get("/api/gallery/marks")
    assert r.status_code == 200 and r.json()["degraded"] is True
    r = c.post("/api/gallery/marks/patch", json={"items": {"R/a.sxm": {"r": 1, "ts": 1}}})
    assert r.status_code == 200 and r.json()["ok"] is False and r.json()["degraded"] is True
    assert (gallery_dir / "marks.json").read_text(encoding="utf-8") == "{broken"
    assert c.get("/api/gallery/marks/export/md").status_code == 503
