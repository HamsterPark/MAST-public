"""桌面截图诊断应区分不可访问的会话、抓取失败与纯色图像，并将原因保留给调用方。"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

_MASTV2 = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)

import PIL
import pytest
from PIL import Image

from mast.core import desktop_capture as dc


def _img(color, size=(40, 30)):
    return Image.new("RGB", size, color)


def test_a_uniform_grab_is_flagged_blank_but_still_returns_the_image(monkeypatch):
    """blank ≠ 失败：图确实抓回来了，而「分辨率是多少」是判断有没有桌面的线索，
    丢掉它等于丢掉诊断本身。"""
    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "win32"}))
    monkeypatch.setattr(PIL, "ImageGrab",
                        type("M", (), {"grab": staticmethod(lambda **kw: _img("white"))}),
                        raising=False)
    cap = dc.capture(max_width=1600)
    assert cap.ok is True
    assert cap.blank is True
    assert cap.png, "blank 时也必须把图带回来"
    assert cap.raw_width == 40 and cap.raw_height == 30
    assert any("没有交互桌面" in n for n in cap.notes)


def test_a_real_looking_grab_is_not_flagged_blank(monkeypatch):
    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "win32"}))
    img = _img("black")
    img.putpixel((0, 0), (255, 0, 0))        # 一个像素不同就不是纯色
    monkeypatch.setattr(PIL, "ImageGrab", type("M", (), {"grab": staticmethod(lambda **kw: img)}), raising=False)
    cap = dc.capture()
    assert cap.ok is True and cap.blank is False


def test_no_desktop_is_its_own_reason(monkeypatch):
    """SSH / 服务方式跑时的典型下场。必须单独归类并说清怎么办。"""
    def _boom(**kw):
        raise OSError("screen grab failed")

    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "win32"}))
    monkeypatch.setattr(PIL, "ImageGrab", type("M", (), {"grab": staticmethod(_boom)}), raising=False)
    cap = dc.capture()
    assert cap.ok is False
    assert cap.reason == "no_desktop", "笼统的『截图失败』会让人去查错的方向"
    assert cap.png == b""
    assert any("交互会话" in n for n in cap.notes)


def test_non_windows_degrades_honestly(monkeypatch):
    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "linux"}))
    cap = dc.capture()
    assert cap.ok is False and cap.reason == "unsupported_platform"


def test_capture_never_raises(monkeypatch):
    """诊断工具自己坏掉不该变成 500。"""
    def _boom(**kw):
        raise RuntimeError("something else entirely")

    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "win32"}))
    monkeypatch.setattr(PIL, "ImageGrab", type("M", (), {"grab": staticmethod(_boom)}), raising=False)
    cap = dc.capture()
    assert cap.ok is False and cap.reason == "error"


def test_downscale_keeps_aspect_and_reports_the_raw_size(monkeypatch):
    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "win32"}))
    img = _img("black", size=(3200, 1800))
    img.putpixel((0, 0), (1, 2, 3))
    monkeypatch.setattr(PIL, "ImageGrab", type("M", (), {"grab": staticmethod(lambda **kw: img)}), raising=False)
    cap = dc.capture(max_width=1600)
    assert (cap.width, cap.height) == (1600, 900)
    assert (cap.raw_width, cap.raw_height) == (3200, 1800), (
        "缩放后必须还能说出原始分辨率 —— 1024×768 这种假值正是没有桌面的信号"
    )
    assert Image.open(io.BytesIO(cap.png)).size == (1600, 900)


def test_the_endpoint_never_500s_and_marks_degraded(monkeypatch):
    from mast.api.routes import diagnostics as route

    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "linux"}))
    resp = route.get_screenshot(max_width=1600)
    assert resp.ok is False and resp.degraded is True
    assert resp.reason == "unsupported_platform"
    assert resp.png_b64 == ""


def test_the_endpoint_returns_base64_png(monkeypatch):
    from mast.api.routes import diagnostics as route

    monkeypatch.setattr(dc, "sys", type("S", (), {"platform": "win32"}))
    img = _img("black", size=(20, 10))
    img.putpixel((0, 0), (9, 9, 9))
    monkeypatch.setattr(PIL, "ImageGrab", type("M", (), {"grab": staticmethod(lambda **kw: img)}), raising=False)
    resp = route.get_screenshot(max_width=1600)
    assert resp.ok is True and resp.degraded is False
    assert Image.open(io.BytesIO(base64.b64decode(resp.png_b64))).size == (20, 10)
