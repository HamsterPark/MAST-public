"""近期标注帧的「点开放大」—— 放大的必须是**同一张**图（已知问题 #93）。

已知问题:近期标注帧不能点开放大。网格从前渲染的是一个没有任何 handler 的 `<img>`。

这一批加了 ``GET /api/vision/recent-frame/{seqno}``。它最容易出的错不是 404，是
**打开另一帧**：``scan_complete`` 借哪一个 .sxm 是**按位置**分配的（``scan_ptr``
沿着最近的扫描文件往下走），所以「第 N 条事件的图是哪一个文件」是整批的性质，
不是第 N 条自己的性质。任何第二次独立推导都会在批次边界上给出不同答案，而症状是
一个看起来完全正常的放大图 —— #76/#78 的伪造历史，多了一次点击。

所以这里钉的是**行为**：同一批事件，缩略图那条路与放大那条路必须选中同一个文件。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/api/test_recent_frame_zoom.py -q
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.vision import router


class _Ev:
    """A BufferService VisionEvent, reduced to what the recent strip reads."""

    def __init__(self, seqno, kind, payload=None, cause_ref=None):
        self.seqno = seqno
        self.kind = kind
        self.severity = "info"
        self.cause_ref = cause_ref
        self.payload = payload or {}
        self.t_mono_ns = 0


class _Buf:
    def __init__(self, events):
        self._events = events

    def get_event_history(self, since_seqno=-1, limit=200):
        return list(self._events)


def _client(events, app_obj=None) -> TestClient:
    api = FastAPI()
    ctx = AppContext()
    ctx.buffer = _Buf(events)
    if app_obj is not None:
        ctx.app = app_obj
    api.state.ctx = ctx
    api.include_router(router, prefix="/api")
    return TestClient(api)


_EXAMPLES = (Path(__file__).resolve().parents[4] / "stm-datasets" / "repos"
             / "ML-STM" / "example data")


def _copy_example(n: str, dst_dir: Path) -> Path:
    src = _EXAMPLES / f"STM_WTip_WSe2-SL445_{n}.sxm"
    if not src.is_file():
        pytest.skip("no .sxm fixture")
    dst = dst_dir / f"scan_{n}.sxm"
    shutil.copy2(src, dst)
    return dst


@pytest.fixture()
def sxm(tmp_path) -> Path:
    return _copy_example("023", tmp_path)


def test_enlarge_404s_when_nothing_is_wired() -> None:
    """没有缓冲区就没有事件 —— 404，不是 500,也不是一张随便什么图。"""
    api = FastAPI()
    api.state.ctx = AppContext()
    api.include_router(router, prefix="/api")
    r = TestClient(api).get("/api/vision/recent-frame/7")
    assert r.status_code == 404


def test_enlarge_404s_for_a_tile_that_has_no_picture() -> None:
    """扫描中判读没有留存画面 —— 它的格子上就不该有放大按钮，端点也照实说没有。

    「这一帧没有图」和「后端坏了」必须是两句话；这里是前者，而它长成 404 而不是
    一张借来的图。
    """
    c = _client([_Ev(3, "feature_of_interest")])
    assert c.get("/api/vision/recent-frame/3").status_code == 404


def test_enlarge_serves_the_persisted_frame_png(tmp_path) -> None:
    png = tmp_path / "milestone.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    c = _client([_Ev(11, "feature_of_interest", {"frame_path": str(png)})])

    r = c.get("/api/vision/recent-frame/11")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == png.read_bytes(), "放大给的不是这一帧存下来的那张图"
    # 借图是按位置分配的，所以同一个 seqno 的答案**会**随新扫描到来而合法地改变。
    assert "no-store" in r.headers.get("cache-control", "")


def test_enlarge_and_thumbnail_pick_the_same_file(tmp_path, monkeypatch) -> None:
    """同一批事件里,放大与缩略图必须落在**同一个文件**上。

    两条 scan_complete 借的是不同的 .sxm(池子里第 0 个和第 1 个)。放大端点如果自己
    重新数一遍而口径稍有不同,拿回来的就是隔壁那一帧 —— 而它看上去完全正常。

    断言不是「两张图不一样」:第一版正是那么写的,而两个文件是**同一份的拷贝**,
    于是两侧都对、断言必红,红的原因还跟被测代码无关。这里改成拿每一个 seqno 的
    返回**逐字节比对它应得的那个文件的渲染**。
    """
    from mast.api.routes import vision as vr
    from mast.webui.scan_preview import render_scan_thumbnail

    a = _copy_example("023", tmp_path)
    b = _copy_example("022", tmp_path)
    # 借图池的顺序由这里定死,免得断言依赖 mtime 的先后。
    pool = [str(a), str(b)]
    monkeypatch.setattr(vr, "_recent_scan_paths", lambda _app, _n: list(pool))

    def _expected(path: str) -> bytes:
        import base64
        uri = render_scan_thumbnail(path, size=vr._ENLARGED_PX)
        return base64.b64decode(uri.split(",", 1)[1])

    want = {2: _expected(pool[0]), 1: _expected(pool[1])}
    assert want[1] != want[2], (
        "两个夹具文件渲染出来一模一样 —— 这条测试分不出「拿对了」和「拿错了」")

    events = [_Ev(1, "scan_complete"), _Ev(2, "scan_complete")]  # 旧→新
    c = _client(events, app_obj=SimpleNamespace(_resolve_session_dir=lambda: None,
                                                config=None))

    listed = c.get("/api/vision/recent").json()["frames"]
    by_seq = {f["seqno"]: f for f in listed}
    # newest-first: seqno 2 拿池子里第 0 个, seqno 1 拿第 1 个
    assert by_seq[2]["file_path"] == pool[0]
    assert by_seq[1]["file_path"] == pool[1]

    for seq in (1, 2):
        r = c.get(f"/api/vision/recent-frame/{seq}")
        assert r.status_code == 200, seq
        assert r.content == want[seq], f"#{seq} 放大打开的是另一帧"


def test_enlarged_render_is_bigger_than_the_strip_thumbnail(sxm) -> None:
    """放大必须真的取一张更大的图。

    把 160 px 的缩略图在浏览器里拉大,是「同样的 160 px,格子更大」—— 一个看起来
    做完了的放大按钮。这条比较的是**字节层面**的两张 PNG 的像素尺寸。
    """
    import struct

    def _png_size(blob: bytes) -> tuple[int, int]:
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", blob[16:24])   # IHDR width/height
        return w, h

    c = _client([_Ev(5, "feature_of_interest", {"file_path": str(sxm)})],
                app_obj=SimpleNamespace(_resolve_session_dir=lambda: None, config=None))

    import base64
    thumb_b64 = c.get("/api/vision/recent").json()["frames"][0]["image_b64"]
    assert thumb_b64, "缩略图这一侧就没出图,下面的比较没有意义"
    tw, th = _png_size(base64.b64decode(thumb_b64))

    big = c.get("/api/vision/recent-frame/5")
    assert big.status_code == 200
    bw, bh = _png_size(big.content)

    assert bw > tw * 2 and bh > th * 2, (
        f"放大后 {bw}x{bh} vs 缩略图 {tw}x{th} —— 这不叫放大")
