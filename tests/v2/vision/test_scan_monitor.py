"""Scan-progress vision monitor — unit tests (no Nanonis, no real model).

Drives ScanVisionMonitor._run() synchronously with a controllable clock +
stub pool/buffer/vision, so milestone firing, publishing, self-termination,
abort, and fail-safe behaviour are all deterministic.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.vision.module import (  # noqa: E402
    SegmentationResult, TipCoarseResult, TipFineResult,
)
from mast.vision.scan_monitor import (  # noqa: E402
    ScanVisionMonitor, _parsed_values, start_scan_vision_monitor,
)
from mast.vision.seg_utils import encode_rle  # noqa: E402


# ── stubs ─────────────────────────────────────────────────────────────

class _Rec:
    def __init__(self, vals, error=""):
        self.error = error
        self.return_value = ("", b"", vals)


class StubPool:
    """Reports 'scanning' for the first `running_polls` status reads, then idle.
    Scan_FrameDataGrab returns a flat pixels*lines frame."""
    def __init__(self, pixels=8, lines=8, running_polls=8):
        self.pixels, self.lines = pixels, lines
        self.running_polls = running_polls
        self.status_calls = 0
        self.frame_grabs = 0

    def safe_call(self, method, *args, role="main"):
        if method == "Scan_StatusGet":
            self.status_calls += 1
            running = 1 if self.status_calls <= self.running_polls else 0
            return _Rec([running])
        if method == "Scan_FrameDataGrab":
            self.frame_grabs += 1
            n = self.pixels * self.lines
            return _Rec(list(np.linspace(0, 30, n).astype(float)))
        return _Rec([])


class StubBuffer:
    def __init__(self):
        self.seq = 0
        self.tips, self.regions, self.progress, self.events = [], [], [], []

    def next_seq(self):
        self.seq += 1
        return self.seq

    def put_tip_status(self, ts): self.tips.append(ts)
    def put_region(self, rm): self.regions.append(rm)
    def put_progress(self, p): self.progress.append(p)
    def emit_event(self, ev): self.events.append(ev)


class StubVision:
    def __init__(self, label="good", raise_on=None):
        self.label = label
        self.raise_on = raise_on or set()
        self.scan_sizes = []
        self.calls = {"coarse": 0, "fine": 0, "segment": 0}

    def set_scan_size_nm(self, nm): self.scan_sizes.append(nm)

    def assess_tip_coarse(self, image):
        self.calls["coarse"] += 1
        if "coarse" in self.raise_on:
            raise RuntimeError("boom")
        return TipCoarseResult(label=self.label, confidence=0.8,
                               embedding_sha="abc123", scan_size_nm=5.0,
                               tip_radius_nm=0.3, sharpness_log10=-1.2)

    def assess_tip_fine(self, image):
        self.calls["fine"] += 1
        return TipFineResult(label="M0", top2=[("M0", 0.7), ("M1", 0.2)],
                             is_usable=True, morph="M0", switching=False,
                             drift=False, perturbation=False, multi_tip=False,
                             n_tips=1.0)

    def segment(self, image, classes=None, *, level=None):
        self.calls["segment"] += 1
        seg = np.zeros((8, 8), dtype=np.uint8)
        return SegmentationResult(mask_rle=encode_rle(seg), shape=(8, 8),
                                  class_counts={"TERRACE": 64}, level=1,
                                  classes=["TERRACE", "STEP", "DEFECT", "CONTAMINATION"])


def _clock():
    """Returns a time_fn advancing 1.0 per call (t0=0, then 1,2,3,...)."""
    state = {"n": 0}
    def fn():
        v = float(state["n"]); state["n"] += 1; return v
    return fn


def _monitor(pool, buf, vision, pixels=8, total_lines=8, **kw):
    return ScanVisionMonitor(
        pool, scan_id="testscan", buffer=buf,
        vision_getter=lambda: vision, poll_interval_s=0.0,
        scan_size_nm=5.0, total_lines=total_lines, pixels=pixels, total_time_s=8.0,
        time_fn=_clock(), **kw,
    )


# ── tests ─────────────────────────────────────────────────────────────

def test_parsed_values_extracts_index2():
    assert _parsed_values(_Rec([1, 2, 3])) == [1, 2, 3]
    assert _parsed_values(_Rec([], error="bad")) is None
    assert _parsed_values(None) is None


def test_all_eight_milestones_fire_and_publish():
    # 128-line frame so every milestone (≥16 acquired rows) has enough data for
    # the classical segmenter (segmentation now runs network-free on the acquired
    # region — it no longer calls vision.segment; the deployed C head hallucinated
    # contamination, see the diagnostic).
    pool, buf, vision = StubPool(pixels=128, lines=128, running_polls=8), StubBuffer(), StubVision()
    mon = _monitor(pool, buf, vision, pixels=128, total_lines=128)
    mon._run()  # synchronous (poll=0, stub clock)
    # 8 milestones fired (7 partial + final 100%)
    assert mon._fired == {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}
    # coarse at every milestone; fine only at the final. Segment is now the
    # classical (network-free) path, so vision.segment is NOT called.
    assert vision.calls["coarse"] == 8
    assert vision.calls["segment"] == 0
    assert vision.calls["fine"] == 1
    # buffer got tip statuses + classical regions + a milestone event each. The
    # real-time classical detectors may add their own TIP_QUALITY_DROP alerts
    # (deduped) on top, so count the MILESTONE events specifically.
    from mast.buffer.schemas import VisionEventType
    milestone_events = [e for e in buf.events if e.kind != VisionEventType.TIP_QUALITY_DROP]
    assert len(buf.tips) == 8
    assert len(buf.regions) == 8
    assert len(milestone_events) == 8
    # scan size was set before each assessment
    assert all(s == 5.0 for s in vision.scan_sizes)


def test_final_event_is_scan_complete_with_summary():
    from mast.buffer.schemas import VisionEventType
    pool, buf, vision = StubPool(running_polls=8), StubBuffer(), StubVision()
    _monitor(pool, buf, vision)._run()
    final = buf.events[-1]
    assert final.kind == VisionEventType.SCAN_COMPLETE
    assert final.payload["milestone"] == 1.0
    assert "summary_zh" in final.payload and final.payload["summary_zh"]
    assert "tip_fine" in final.payload  # fine included at 100%
    # partial events are FEATURE_OF_INTEREST
    assert buf.events[0].kind == VisionEventType.FEATURE_OF_INTEREST


def test_bad_tip_midscan_emits_warn():
    from mast.buffer.schemas import Severity
    pool, buf, vision = StubPool(running_polls=8), StubBuffer(), StubVision(label="bad")
    _monitor(pool, buf, vision)._run()
    assert any(ev.severity == Severity.WARN for ev in buf.events)
    # tip status quality reflects 'bad'
    from mast.buffer.schemas import TipQuality
    assert buf.tips[0].quality == TipQuality.BAD


def test_progress_published_each_poll():
    pool, buf, vision = StubPool(running_polls=8), StubBuffer(), StubVision()
    _monitor(pool, buf, vision)._run()
    assert len(buf.progress) >= 7
    assert buf.progress[0].lines_total == 8
    assert buf.progress[-1].line_idx <= 8


def test_operator_abort_skips_final_milestone():
    pool, buf, vision = StubPool(running_polls=100), StubBuffer(), StubVision()
    abort = threading.Event()
    mon = _monitor(pool, buf, vision, abort_event=abort)
    abort.set()  # operator aborted before run
    mon._run()
    # aborted before loop → no milestones, NO final 100% (operator abort)
    assert 1.0 not in mon._fired
    assert len(buf.events) == 0


def test_milestone_failure_does_not_crash_monitor():
    pool, buf = StubPool(running_polls=8), StubBuffer()
    vision = StubVision(raise_on={"coarse"})  # coarse raises every time
    mon = _monitor(pool, buf, vision)
    mon._run()  # must not raise
    # all milestones attempted, each emitted a vision_error event instead
    from mast.buffer.schemas import VisionEventType
    assert any(ev.kind == VisionEventType.VISION_ERROR for ev in buf.events)


def test_reshape_uses_pixels_then_falls_back():
    arr = list(range(64))
    out = ScanVisionMonitor._reshape(arr, 8)
    assert out.shape == (8, 8)
    out2 = ScanVisionMonitor._reshape(arr, 0)  # unknown pixels → square
    assert out2.shape == (8, 8)
    assert ScanVisionMonitor._reshape(None, 8) is None


def test_grab_frame_zeros_unacquired_rows():
    pool, buf, vision = StubPool(pixels=8, lines=8), StubBuffer(), StubVision()
    mon = _monitor(pool, buf, vision)
    frame = mon._grab_frame(0.5)  # half acquired
    assert frame.shape == (2, 8, 8)
    # rows 4..8 zeroed (50% of 8)
    assert np.all(frame[:, 4:] == 0.0)
    assert not np.all(frame[:, :4] == 0.0)


# ── Fix 5: (2,H,W) frame must be adapted for non-pair-aware backends ───

class _PairAwareVision:
    """VIGIL-like: _backend exposes _to_fwd_bwd (accepts the (2,H,W) pair)."""
    class _BE:
        @staticmethod
        def _to_fwd_bwd(image):  # presence is the capability marker
            return image, image
    _backend = _BE()


class _SingleChannelVision:
    """Mock/Legacy-like: backend takes a single (H,W) channel only."""
    class _BE:
        pass
    _backend = _BE()


def test_adapt_frame_collapses_pair_for_single_channel_backend():
    pool, buf, vision = StubPool(), StubBuffer(), StubVision()
    mon = _monitor(pool, buf, vision)
    pair = np.stack([np.ones((8, 8)), np.zeros((8, 8))]).astype(np.float32)  # (2,8,8)
    out = mon._adapt_frame_for_backend(_SingleChannelVision(), pair)
    assert out.shape == (8, 8)                 # collapsed to forward channel
    assert np.all(out == 1.0)                  # forward (channel 0) kept


def test_adapt_frame_keeps_pair_for_pair_aware_backend():
    pool, buf, vision = StubPool(), StubBuffer(), StubVision()
    mon = _monitor(pool, buf, vision)
    pair = np.zeros((2, 8, 8), dtype=np.float32)
    out = mon._adapt_frame_for_backend(_PairAwareVision(), pair)
    assert out.shape == (2, 8, 8)              # untouched for VIGIL


def test_adapt_frame_passthrough_single_channel_input():
    pool, buf, vision = StubPool(), StubBuffer(), StubVision()
    mon = _monitor(pool, buf, vision)
    single = np.zeros((8, 8), dtype=np.float32)
    out = mon._adapt_frame_for_backend(_SingleChannelVision(), single)
    assert out.shape == (8, 8)                 # already (H,W) → unchanged


def test_start_helper_noop_without_buffer(monkeypatch):
    from mast.buffer import active
    active.set_active_buffer(None)
    mon = start_scan_vision_monitor(StubPool(), scan_id="x")
    assert mon is None


def test_start_helper_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MAST_SCAN_VISION_MONITOR", "0")
    mon = start_scan_vision_monitor(StubPool(), scan_id="x")
    assert mon is None


def test_start_helper_blank_scan_id_gets_fallback(monkeypatch):
    from mast.buffer import active
    from mast.vision import scan_monitor as sm
    active.set_active_buffer(StubBuffer())
    try:
        mon = start_scan_vision_monitor(
            StubPool(running_polls=0), scan_id="",  # no active experiment
            vision_getter=lambda: StubVision(), poll_interval_s=0.0,
            total_lines=8, pixels=8, scan_size_nm=5.0, total_time_s=1.0,
        )
        assert mon is not None and mon._scan_id.startswith("scan-")
    finally:
        sm.stop_active_monitor(join_timeout=2.0)
        active.set_active_buffer(None)


def test_supersede_stops_prior_monitor():
    """HIGH-1: a second StartScan must stop the prior monitor (no leak)."""
    from mast.buffer import active
    from mast.vision import scan_monitor as sm
    active.set_active_buffer(StubBuffer())
    try:
        pool = StubPool(running_polls=10_000_000)  # stays "running"
        kw = dict(vision_getter=lambda: StubVision(), poll_interval_s=0.02,
                  total_lines=8, pixels=8, scan_size_nm=5.0, total_time_s=1000.0)
        mon1 = start_scan_vision_monitor(pool, scan_id="a", **kw)
        assert mon1 is not None and sm._CURRENT_MONITOR is mon1
        mon2 = start_scan_vision_monitor(pool, scan_id="b", **kw)
        assert mon2 is not None and mon2 is not mon1
        assert sm._CURRENT_MONITOR is mon2
        assert mon1._stop.is_set()                 # prior superseded
        assert not mon1._thread.is_alive()         # and joined (no orphan)
        sm.stop_active_monitor(join_timeout=2.0)
        assert sm._CURRENT_MONITOR is None
        assert mon2._stop.is_set()
    finally:
        sm.stop_active_monitor(join_timeout=2.0)
        active.set_active_buffer(None)


# ── Fix #143: an early-stopped scan must NOT be recorded as complete ───

class StubPoolAborted:
    """Scanning for `running_polls` reads, then idle — but the scan buffer never
    fills past `acquired_frac`: the rows beyond it stay all-NaN (a scan STOPPED
    early leaves its unacquired rows NaN). This is the fingerprint the monitor
    must read to refuse a bogus 100 % completion."""
    def __init__(self, pixels=8, lines=8, running_polls=6, acquired_frac=0.5):
        self.pixels, self.lines = pixels, lines
        self.running_polls = running_polls
        self.acquired_frac = acquired_frac
        self.status_calls = 0

    def safe_call(self, method, *args, role="main"):
        if method == "Scan_StatusGet":
            self.status_calls += 1
            running = 1 if self.status_calls <= self.running_polls else 0
            return _Rec([running])
        if method == "Scan_FrameDataGrab":
            arr = np.linspace(0, 30, self.pixels * self.lines).astype(float)
            arr = arr.reshape(self.lines, self.pixels)
            n_acq = int(round(self.acquired_frac * self.lines))
            arr[n_acq:] = np.nan               # never-acquired rows stay NaN
            return _Rec(list(arr.ravel()))
        return _Rec([])


def test_early_stop_is_not_recorded_as_complete():
    """The scan goes idle at ~50 % (NaN front) → NO SCAN_COMPLETE; an honest
    WARN 'ended early' event is emitted instead."""
    from mast.buffer.schemas import Severity, VisionEventType
    pool = StubPoolAborted(running_polls=6, acquired_frac=0.5)
    buf, vision = StubBuffer(), StubVision()
    _monitor(pool, buf, vision)._run()

    assert 1.0 not in _fired_of(buf)               # no fake 100 %
    assert all(ev.kind != VisionEventType.SCAN_COMPLETE for ev in buf.events)
    incompletes = [ev for ev in buf.events if ev.payload.get("incomplete")]
    assert len(incompletes) == 1
    ev = incompletes[0]
    assert ev.kind == VisionEventType.FEATURE_OF_INTEREST
    assert ev.severity == Severity.WARN
    assert "提前结束" in ev.payload["summary_zh"]


def _fired_of(buf):
    """Milestones represented in the emitted events (SCAN_COMPLETE ⇒ 1.0)."""
    from mast.buffer.schemas import VisionEventType
    return {1.0 if ev.kind == VisionEventType.SCAN_COMPLETE
            else ev.payload.get("milestone")
            for ev in buf.events}


def test_full_scan_still_completes_when_buffer_fills():
    """Guard the fix doesn't over-fire: a scan whose buffer has NO NaN rows at
    idle (fully acquired) still yields the authoritative SCAN_COMPLETE."""
    from mast.buffer.schemas import VisionEventType
    pool, buf, vision = StubPool(running_polls=8), StubBuffer(), StubVision()
    _monitor(pool, buf, vision)._run()
    assert any(ev.kind == VisionEventType.SCAN_COMPLETE for ev in buf.events)
    assert not any(ev.payload.get("incomplete") for ev in buf.events)


# ── Fix #141: partial-frame PNGs must not render all-black ─────────────

def test_contrast_range_stretches_over_acquired_pixels():
    """The zero-filled unacquired region must NOT drive the colour scale — with a
    Z-height DC offset that collapses the acquired strip to black (feedback
    #141). The scale keys off the non-zero (acquired) pixels."""
    a = np.zeros((8, 8), dtype=np.float32)
    a[:2] = 20.0 + np.linspace(0.0, 0.1, 16).reshape(2, 8).astype(np.float32)
    vmin, vmax = ScanVisionMonitor._contrast_range(a)
    assert vmin > 10.0            # anchored to the ~20 strip, not the 0.0 fill
    assert vmax > vmin


def test_contrast_range_all_zero_is_safe():
    vmin, vmax = ScanVisionMonitor._contrast_range(np.zeros((4, 4), np.float32))
    assert vmax > vmin           # degenerate but never a zero-width range


def test_persist_frame_png_skips_blank_frame(tmp_path):
    """An all-zero (nothing-acquired) frame is not a real image → no PNG written;
    a frame with real data does persist."""
    pool, buf, vision = StubPool(), StubBuffer(), StubVision()
    mon = _monitor(pool, buf, vision)
    assert mon._persist_frame_png(np.zeros((8, 8), dtype=np.float32), 1) is None
    a = np.zeros((8, 8), dtype=np.float32)
    a[:4] = 5.0
    out = mon._persist_frame_png(a, 2)
    assert out is not None and Path(out).is_file()


# ── Fix #144: is_monitor_running signals a live system scan ────────────

def test_is_monitor_running_reflects_thread_liveness():
    from mast.buffer import active
    from mast.vision import scan_monitor as sm
    sm.stop_active_monitor(join_timeout=2.0)
    assert sm.is_monitor_running() is False
    active.set_active_buffer(StubBuffer())
    try:
        pool = StubPool(running_polls=10_000_000)   # stays "running"
        mon = start_scan_vision_monitor(
            pool, scan_id="live", vision_getter=lambda: StubVision(),
            poll_interval_s=0.02, total_lines=8, pixels=8, scan_size_nm=5.0,
            total_time_s=1000.0)
        assert mon is not None
        assert sm.is_monitor_running() is True
        sm.stop_active_monitor(join_timeout=2.0)
        assert sm.is_monitor_running() is False     # thread exited
    finally:
        sm.stop_active_monitor(join_timeout=2.0)
        active.set_active_buffer(None)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
