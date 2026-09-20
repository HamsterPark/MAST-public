"""GET /api/vision/recent falls back to a recent real .sxm when a scan event
carries no file_path.

审查: scan_monitor._emit_milestone_event puts only scan_id (never
file_path) in the event payload, so get_vision_recent never had a path to render
→ the 近期帧 thumbnail grid showed only 未解码/无图像. The route now borrows the
recent real scans (newest-first) for scan-related events.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/api/test_vision_recent_fallback.py -x -v
"""
from __future__ import annotations

import os
import shutil
import time as _t
from pathlib import Path as _P
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.vision import router
from mast.webui.scan_preview import _THUMB_CACHE

_SXM = (
    _P(__file__).resolve().parents[4]
    / "stm-datasets" / "repos" / "ML-STM" / "example data" / "STM_WTip_WSe2-SL445_023.sxm"
)


class _Buf:
    """Minimal BufferService stand-in: get_event_history returns preset events."""

    def __init__(self, events):
        self._events = events

    def get_event_history(self, since_seqno=-1, limit=200):
        return self._events


def _ev(kind: str, seqno: int):
    # scan_monitor's real payload shape: scan_id + summary_zh, NO file_path/path.
    return SimpleNamespace(
        kind=kind, severity="info", seqno=seqno, cause_ref=f"scan#{seqno}",
        t_mono_ns=0, payload={"scan_id": "s1", "summary_zh": f"里程碑 {seqno}"},
    )


def _client(events, session_dir):
    app_stub = SimpleNamespace(
        _resolve_session_dir=lambda: str(session_dir) if session_dir else None,
        _storage=None, _state=None, config=None,
    )
    ctx = AppContext()
    ctx.buffer = _Buf(events)   # _get_buffer prefers ctx.buffer
    ctx.app = app_stub
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.mark.skipif(not _SXM.is_file(), reason="sample .sxm not present")
def test_scan_events_without_file_path_get_a_thumbnail(tmp_path):
    """scan_complete still borrows the matching real scan (that file IS its
    product). In-scan pulses without a persisted frame get NO image any more —
    borrowing showed the PREVIOUS scan as 'the current pulse', so an honest placeholder beats a misleading image."""
    _THUMB_CACHE.clear()
    scans = tmp_path / "sess"
    scans.mkdir()
    dst = scans / "NiI2_0002.sxm"
    shutil.copy2(_SXM, dst)
    os.utime(dst, (_t.time(), _t.time()))

    events = [_ev("feature_of_interest", 1), _ev("scan_complete", 2)]
    body = _client(events, scans).get("/api/vision/recent").json()

    assert body["degraded"] is False
    assert body["count"] == 2
    by_kind = {f["kind"]: f for f in body["frames"]}
    assert by_kind["scan_complete"]["image_b64"], "scan_complete fallback missing"
    assert not by_kind["feature_of_interest"]["image_b64"], (
        "in-scan pulse must NOT borrow a stale scan image")


def test_pulse_prefers_persisted_true_frame(tmp_path):
    """A milestone event that carries frame_path (the PNG of the EXACT partial
    frame the model analysed, persisted by scan_monitor since 2026-07-10) must
    serve THAT image — for in-scan pulses and completions alike."""
    png = tmp_path / "s1_f3.png"
    # tiny valid PNG (1×1 black pixel)
    png.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000c4944415408d763f8cfc00000030101"
        "00c9fe92ef0000000049454e44ae426082"))
    ev = SimpleNamespace(
        kind="feature_of_interest", severity="info", seqno=7, cause_ref="scan#7",
        t_mono_ns=0,
        payload={"scan_id": "s1", "summary_zh": "里程碑 3",
                 "frame_path": str(png)},
    )
    body = _client([ev], None).get("/api/vision/recent").json()
    assert body["degraded"] is False
    f = body["frames"][0]
    import base64
    assert f["image_b64"] == base64.b64encode(png.read_bytes()).decode("ascii")
    assert f["file_path"] == str(png)


def test_recent_frame_response_well_formed(tmp_path):
    # No session dir → the fallback may still find a scan in working-sessions /
    # experiments_dir (env-dependent), so don't assert on the image; assert the
    # response is well-formed and never crashes, and the narration flows through.
    events = [_ev("scan_complete", 1)]
    body = _client(events, None).get("/api/vision/recent").json()
    assert body["degraded"] is False
    assert body["count"] == 1
    f = body["frames"][0]
    assert f["kind"] == "scan_complete"
    assert "image_b64" in f            # present (fallback thumbnail OR None)
    assert f["summary"] == "里程碑 1"   # summary_zh flowed through (2026-07-03 fix)


# ── 电流监控事件的证据图（「近期标注帧 无图像」） ──────────────
#
# 那张图是事件发出后约 1 s 才在守护线程上画好的，路径只写进 alerts 行；
# 缓冲事件早就带着 frame_path="" 发出去了，而没有任何一步回头补它。
# 于是图在磁盘上，显示这个事件的面板却看不见。这里在**读**侧接上。

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415408d763f8cfc00000030101"
    "00c9fe92ef0000000049454e44ae426082")


def _monitor_ev(seqno: int, cause_ref: str):
    """电流监控发出的事件：frame_path 恒为空字符串（service 传 None）。"""
    return SimpleNamespace(
        kind="tip_quality_drop", severity="critical", seqno=seqno,
        cause_ref=cause_ref, t_mono_ns=0,
        payload={"summary_zh": "电流持续饱和", "frame_path": "",
                 "source": "current_monitor"},
    )


@pytest.fixture
def _monitor_store(tmp_path):
    from mast.monitoring import store as ST
    ST.set_store_for_test(ST.CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path))
    yield ST.get_store()
    ST.set_store_for_test(None)


def test_current_monitor_event_picks_up_its_evidence_png(tmp_path, _monitor_store):
    png = tmp_path / "ev.png"
    png.write_bytes(_PNG)
    sid = _monitor_store.add_segment(
        {"t_start": 1.0, "t_end": 2.0, "fs_hz": 1000.0, "n_samples": 1000,
         "npy_path": None, "npy_bytes": 0, "channel_name": "Current (A)"},
        b"", 0.01)
    _monitor_store.add_alert(ts=1.5, level="critical", rule="saturation",
                             summary_zh="电流持续饱和", segment_id=sid,
                             evidence_png=str(png))

    body = _client([_monitor_ev(1, f"current_monitor#{sid}")], None).get(
        "/api/vision/recent").json()
    import base64
    assert body["frames"][0]["image_b64"] == base64.b64encode(_PNG).decode("ascii")


def test_an_event_whose_segment_has_no_evidence_gets_no_image(tmp_path, _monitor_store):
    """WARN 刻意不渲染证据图。没有就是没有 —— 不许借别的。"""
    sid = _monitor_store.add_segment(
        {"t_start": 1.0, "t_end": 2.0, "fs_hz": 1000.0, "n_samples": 1000,
         "npy_path": None, "npy_bytes": 0, "channel_name": "Current (A)"},
        b"", 0.01)
    _monitor_store.add_alert(ts=1.5, level="warn", rule="rms_high",
                             summary_zh="噪声偏高", segment_id=sid,
                             evidence_png=None)
    body = _client([_monitor_ev(1, f"current_monitor#{sid}")], None).get(
        "/api/vision/recent").json()
    assert not body["frames"][0]["image_b64"]


def test_a_scan_event_never_picks_up_a_current_monitor_png(tmp_path, _monitor_store):
    """段号与扫描号是两个互不相关的编号空间。

    按数字撞上就取图，等于在一个扫描事件底下画一条电流波形 —— #76/#78
    那条「不许伪造历史」的规则在这里同样适用，所以只认 ``current_monitor#``
    这个前缀。
    """
    png = tmp_path / "ev.png"
    png.write_bytes(_PNG)
    sid = _monitor_store.add_segment(
        {"t_start": 1.0, "t_end": 2.0, "fs_hz": 1000.0, "n_samples": 1000,
         "npy_path": None, "npy_bytes": 0, "channel_name": "Current (A)"},
        b"", 0.01)
    _monitor_store.add_alert(ts=1.5, level="critical", rule="saturation",
                             summary_zh="电流持续饱和", segment_id=sid,
                             evidence_png=str(png))
    # 同一个数字，但前缀是 scan#。
    ev = SimpleNamespace(
        kind="feature_of_interest", severity="info", seqno=1,
        cause_ref=f"scan#{sid}", t_mono_ns=0,
        payload={"summary_zh": "里程碑", "frame_path": ""})
    body = _client([ev], None).get("/api/vision/recent").json()
    assert not body["frames"][0]["image_b64"]


def test_cause_ref_parsing_rejects_everything_that_is_not_a_segment():
    from mast.api.routes.vision import _segment_of
    assert _segment_of("current_monitor#42") == 42
    # 没有段号的告警：cause_ref 就是裸的 "current_monitor"。
    assert _segment_of("current_monitor") is None
    assert _segment_of("scan#42") is None
    assert _segment_of("current_monitor#abc") is None
    assert _segment_of(None) is None
    assert _segment_of("") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
