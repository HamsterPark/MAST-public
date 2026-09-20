# -*- coding: utf-8 -*-
"""出图产物的列表 / 文件 / 预览（D16 / D21，T32）、单槽任务（T33）与路由契约。"""
from __future__ import annotations

import io
import os
import re
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from mast.api.schemas_gallery_figures import GalleryFigureJobStatus, GalleryFiguresList, GalleryStsLinePlan
from mast.gallery import paths
from mast.gallery.figures import common as C
from mast.gallery.figures import service, store

FILES = ("样例_叠加_配准表.csv", "样例_叠加_主图_2x.png", "样例_叠加.png")


def _fake_figure(files=FILES, size=(1200, 600)) -> str:
    lay = paths.layout()
    for name in files:
        if name.endswith(".png"):
            buf = io.BytesIO()
            Image.new("RGB", size, (200, 100, 50)).save(buf, "PNG")
            data = buf.getvalue()
        else:
            data = b"a,b\r\n1,2\r\n"
        paths.atomic_write_bytes(C.category_dir(lay, "series") / name, data)
    return C.write_figure_json(lay, "series", "样例_叠加", kind="series_stack", title="样例", files=list(files),
                               ids=["x"], series=["S"], options={"anchor": "darkest"}, summary={"帧数": 3},
                               detail={"big": list(range(10))})


def _client() -> TestClient:
    from mast.api.context import AppContext
    from mast.api.routes.gallery_figures import router

    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── store ───────────────────────────────────────────────────────────────


def test_a_fresh_state_dir_lists_empty_and_creates_nothing(gallery_dir):
    got = store.list_figures()
    assert [c["key"] for c in got["categories"]] == ["frames", "grids", "sts_lines", "sts_stitch", "series"]
    assert all(c["figures"] == [] for c in got["categories"]) and got["degraded"] is False
    GalleryFiguresList(**got)
    assert store.figure_file("series/x.png") is None and store.preview_file("series/x.png") is None
    assert not gallery_dir.exists()


def test_listing_puts_the_main_image_first_and_previews_images_only(gallery_dir):
    key = _fake_figure()
    (C.category_dir(paths.layout(), "series") / ("样例_叠加.png" + paths.TMP_MARK + "abc")).write_bytes(b"partial")
    got = store.list_figures()
    [fig] = next(c for c in got["categories"] if c["key"] == "series")["figures"]
    assert fig["key"] == key == "series/样例_叠加" and fig["kind"] == "series_stack" and fig["summary"] == {"帧数": 3}
    assert [f["name"] for f in fig["files"]] == ["样例_叠加.png", "样例_叠加_配准表.csv", "样例_叠加_主图_2x.png",
                                                 "样例_叠加.figure.json"]
    assert [f["preview_url"] is not None for f in fig["files"]] == [True, False, True, False]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", fig["created"])
    assert "detail" not in fig
    GalleryFiguresList(**got)


def test_files_and_previews_stay_inside_figures(gallery_dir):
    _fake_figure()
    assert store.figure_file("series/样例_叠加.png").name == "样例_叠加.png"
    for bad in ("../marks.json", "series/../../config.json", "series", "series/样例_叠加.png.tmp-abc"):
        assert store.figure_file(bad) is None, bad
    prev = store.preview_file("series/样例_叠加.png")
    with Image.open(prev) as im:
        assert im.format == "JPEG" and im.size == (480, 240)
    assert store.figure_file(".previews/series/样例_叠加.png.jpg") is None
    assert store.preview_file("series/样例_叠加_配准表.csv") is None


def test_a_preview_of_a_missing_source_writes_nothing(gallery_dir):
    _fake_figure()
    assert store.preview_file("series/没有这张.png") is None
    assert store.preview_file("nope") is None
    assert not (gallery_dir / "figures" / C.PREVIEW_DIR).exists()


def test_a_preview_is_cached_until_the_source_is_newer(gallery_dir):
    _fake_figure()
    prev = store.preview_file("series/样例_叠加.png")
    first = prev.stat().st_mtime_ns
    assert store.preview_file("series/样例_叠加.png") == prev and prev.stat().st_mtime_ns == first
    src = gallery_dir / "figures" / "series" / "样例_叠加.png"
    old = src.stat().st_mtime_ns - 10 * 10**9
    os.utime(prev, ns=(old, old))
    store.preview_file("series/样例_叠加.png")
    assert prev.stat().st_mtime_ns > old + 5 * 10**9


# ── service ─────────────────────────────────────────────────────────────


def test_status_keys_are_the_schema_fields():
    assert set(service.STATUS_KEYS) == set(GalleryFigureJobStatus.model_fields)
    assert set(service.get_status()) == set(GalleryFigureJobStatus.model_fields)


def test_a_job_without_an_index_ends_in_error_not_in_an_exception(gallery_dir):
    res = service.run_job("grid_sheets")
    assert res["phase"] == "error" and "构建" in res["message"] and res["running"] is False
    assert service.run_job("nope")["phase"] == "error"
    assert not gallery_dir.exists()


def test_one_slot_a_second_start_reports_the_running_job(gallery_dir, monkeypatch):
    gate = threading.Event()
    seen = []

    def fake(kind, ids, series, options, *, state, progress, cancel):
        seen.append((kind, state))
        gate.wait(30)
        progress.update(running=False, phase="cancelled" if cancel.is_set() else "done", finished=C.now_str())
        return progress.snapshot()

    monkeypatch.setattr(service, "run_job", fake)
    first = service.start("grid_sheets")
    assert (first["running"], first["phase"], first["kind"]) == (True, "running", "grid_sheets")
    second = service.start("sts_lines", series=["S"])
    assert (second["running"], second["kind"]) == (True, "grid_sheets")
    assert "正在取消" in service.cancel()["message"]
    gate.set()
    assert service.wait(10)
    st = service.get_status()
    assert (st["running"], st["phase"]) == (False, "cancelled")
    assert [k for k, _s in seen] == ["grid_sheets"] and os.path.samefile(seen[0][1].parent, gallery_dir.parent)


def test_a_status_read_while_a_job_is_being_started_never_says_running_false(gallery_dir, monkeypatch):
    """状态请求恰好落在「发布新任务」与「线程真正启动」之间时，不许读出 phase=running 而 running=False
    —— 轮询方会把它当成任务已经结束（端到端撞上过）。用一个在 ``start()`` 前后各查一次状态的线程类钉住那个窗口。"""
    gate = threading.Event()
    seen = []

    def fake(kind, ids, series, options, *, state, progress, cancel):
        gate.wait(30)
        progress.update(running=False, phase="done", finished=C.now_str())
        return progress.snapshot()

    class PeekingThread(threading.Thread):
        def start(self):
            seen.append(service.get_status())
            super().start()
            seen.append(service.get_status())

    monkeypatch.setattr(service, "run_job", fake)
    monkeypatch.setattr(service.threading, "Thread", PeekingThread)
    first = service.start("grid_sheets")
    gate.set()
    assert service.wait(10)
    assert len(seen) == 2, "the peeking thread class was not used"
    for st in (*seen, first):
        assert not (st["phase"] == "running" and st["running"] is False), st
    assert (first["running"], first["phase"]) == (True, "running")


# ── 路由 ────────────────────────────────────────────────────────────────


def test_routes_honour_the_contract_on_a_cold_state_dir(gallery_dir):
    c = _client()
    r = c.get("/api/gallery/figures")
    assert r.status_code == 200 and GalleryFiguresList(**r.json()).degraded is False
    r = c.get("/api/gallery/figures/status")
    assert r.status_code == 200 and r.json()["phase"] == "idle"
    assert c.get("/api/gallery/figures/file/nope").status_code == 404
    assert c.get("/api/gallery/figures/preview/nope").status_code == 404
    plan = GalleryStsLinePlan(**c.post("/api/gallery/figures/sts_lines/plan", json={"series": ["S1"]}).json())
    assert plan.ok is False and plan.degraded is False and plan.detail
    r = c.post("/api/gallery/figures/run", json={"kind": "grid_sheets"})
    assert r.status_code == 200 and r.json()["kind"] == "grid_sheets"
    assert service.wait(30)
    st = c.get("/api/gallery/figures/status").json()
    assert st["phase"] == "error" and "构建" in st["message"]
    assert not gallery_dir.exists()


def test_the_plan_route_reads_the_index_only(lab, monkeypatch):
    ds = lab.line_dataset()
    lab.build()
    lab.mark(series=ds["series"])

    def boom(*_a, **_k):
        raise AssertionError("预演不许读原始文件")

    monkeypatch.setattr("mast.io.nanonis_files.read_dat", boom)
    body = _client().post("/api/gallery/figures/sts_lines/plan", json={"series": list(ds["series"])}).json()
    plan = GalleryStsLinePlan(**body)
    assert plan.ok and plan.n_stations == 6 and plan.reference == "SA" and plan.line_name == "测试线 · L9 过缺陷"
    assert [b.label for b in plan.blocks] == ["区组0", "区组1", "区组2 （中断）", "区组3"]
    assert plan.blocks[2].unmatched == [ds["outlier"]] and len(plan.direction) == 2


def test_the_real_app_serves_the_figure_routes_without_touching_disk(gallery_dir):
    from mast.api.app import create_app

    c = TestClient(create_app())
    assert c.get("/api/gallery/figures").status_code == 200
    assert c.get("/api/gallery/figures/status").status_code == 200
    assert c.get("/api/gallery/figures/preview/nope").status_code == 404
    assert not gallery_dir.exists()
