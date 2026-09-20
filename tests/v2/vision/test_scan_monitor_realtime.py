"""Real-time network-free detectors in the live scan monitor.

The monitor now runs mast.vision.tip_change + scan_artifacts on the ACQUIRED
region of each partial-milestone frame, so a mid-scan tip change / feedback
oscillation / bad scan-lines are caught DURING the scan (early abort) instead of
only after a whole frame is burned. See docs/v2/benchmarks/vision_v25_diagnostic/.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import asyncio  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from mast.vision.scan_monitor import ScanVisionMonitor  # noqa: E402


def _lattice(H=256, W=256, period=7.3, seed=0):
    """Rotated incommensurate lattice + noise floor — physically plausible
    (a perfect axis-aligned sin·sin grid has whole-row zero crossings no real
    scan produces; see test_tip_change.py)."""
    yy, xx = np.mgrid[0:H, 0:W]
    th = 0.3
    x2 = xx * np.cos(th) + yy * np.sin(th)
    y2 = -xx * np.sin(th) + yy * np.cos(th)
    return (np.sin(x2 * 2 * np.pi / period + 0.7)
            * np.sin(y2 * 2 * np.pi / (period * 1.13) + 1.1)
            + 0.05 * np.random.RandomState(seed).randn(H, W)).astype(np.float32)


def _partial(fwd, acquired_rows):
    """(2,H,W) with rows past `acquired_rows` zeroed in BOTH channels (as
    _grab_frame yields for a down-scan)."""
    bwd = fwd + 0.05 * np.random.RandomState(9).randn(*fwd.shape).astype(np.float32)
    fwd = fwd.copy(); bwd = bwd.copy()
    fwd[acquired_rows:] = 0.0
    bwd[acquired_rows:] = 0.0
    return np.stack([fwd, bwd]).astype(np.float32)


def _partial_with_change(change_row=96, acquired=160):
    """Mid-scan tip change in the z-offset mode — the physical main mode of
    real events (apex length changes → row-DC jump; VIGIL truth validation)."""
    fwd = _lattice()
    fwd[change_row:acquired] += 0.6
    fwd[change_row:acquired] = ndi.gaussian_filter(fwd[change_row:acquired], 1.5) \
        + 0.15 * np.random.RandomState(1).randn(acquired - change_row, fwd.shape[1])
    return _partial(fwd, acquired)


def _mon(scan_id="t"):
    return ScanVisionMonitor(None, scan_id=scan_id)   # no pool/buffer → alerts no-op


def test_acquired_crop_keeps_only_scanned_rows():
    crop = ScanVisionMonitor._acquired_crop(_partial(_lattice(), 160))
    assert crop is not None
    assert 150 <= crop.shape[1] <= 170          # ~160 acquired rows
    assert crop.shape[0] == 2


def test_acquired_crop_too_few_rows_is_none():
    assert ScanVisionMonitor._acquired_crop(np.zeros((2, 256, 256), np.float32)) is None
    assert ScanVisionMonitor._acquired_crop(_partial(_lattice(), 30)) is None


def test_realtime_check_detects_mid_scan_tip_change():
    mon = _mon("t1")
    rt = mon._realtime_check(_partial_with_change(96, 160), 0.6)
    assert rt["tip_change"]["changed"] is True
    assert abs(int(rt["tip_change"]["row"]) - 96) <= 12
    assert "tip_change" in mon._rt_alerted          # alert fired (deduped)


def test_realtime_check_clean_partial_no_change():
    mon = _mon("t2")
    rt = mon._realtime_check(_partial(_lattice(seed=3), 160), 0.6)
    assert rt.get("tip_change", {}).get("changed") is False
    assert "tip_change" not in mon._rt_alerted


def test_realtime_alert_dedup_once_per_scan():
    mon = _mon("t3")
    frame = _partial_with_change(96, 160)
    mon._realtime_check(frame, 0.6)
    assert mon._rt_alerted == {"tip_change"} or "tip_change" in mon._rt_alerted
    n_after_first = len(mon._rt_alerted)
    mon._realtime_check(frame, 0.7)                 # same condition, later milestone
    assert len(mon._rt_alerted) == n_after_first    # not re-added → alert only once


def test_realtime_check_oscillation():
    mon = _mon("t4")
    H = 256
    xx = np.mgrid[0:H, 0:H][1]
    ring = (1.5 * np.sin(xx * 2 * np.pi / 6) + 0.05 * np.random.RandomState(0).randn(H, H)).astype(np.float32)
    rt = mon._realtime_check(_partial(ring, 200), 0.8)
    assert rt["artifacts"]["oscillation"] is True
    assert "oscillation" in mon._rt_alerted


def test_realtime_check_double_tip_data_only_no_alert():
    """double_tip was DEMOTED (2026-07-27): physics-truth validation measured
    AUC 0.442 on labels / 0.625 on injected ghosts (misses ~4 of 5 at FPR 5 %),
    so it must never raise a repair-the-tip alert — the score stays visible as
    advisory data, explicitly marked unreliable."""
    mon = _mon("t6")
    mon._scan_size_nm = 8.0
    H = 256
    yy, xx = np.mgrid[0:H, 0:H]
    base = np.zeros((H, H), np.float32)
    for y, x in zip(np.random.RandomState(0).randint(20, 140, 14),
                    np.random.RandomState(1).randint(20, H - 20, 14)):
        base += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 9.0))
    dbl = 0.5 * base + 0.5 * ndi.shift(base, (16, 12), mode="nearest")
    rt = mon._realtime_check(_partial(dbl.astype(np.float32), 160), 0.6)
    assert rt["double_tip"]["is_double"] is True          # data still reported
    assert rt["double_tip"]["reliability"] == "unreliable"
    assert "double_tip" not in mon._rt_alerted            # but NO alert
    assert "metrics" in rt                       # tip_metrics attached


def test_realtime_check_never_raises_on_junk():
    mon = _mon("t5")
    assert mon._realtime_check(np.zeros((2, 256, 256), np.float32), 0.1) == {}   # all-zero → no crop
    assert isinstance(mon._realtime_check(_lattice()[None].repeat(2, 0), 0.5), dict)


# ── segment upgrade: classical (no hallucination), not the deployed C head ────
def test_monitor_segment_is_classical_no_hallucination():
    """The monitor now segments with the network-free classical segmenter on the
    ACQUIRED region, padded to full shape — a clean lattice is ~all terrace, NOT
    the contamination the deployed Head C hallucinates."""
    from mast.vision.seg_utils import decode_rle
    mon = _mon("seg")
    mon._scan_size_nm = 8.0
    seg, summary = mon._segment_classical_frame(_partial(_lattice(), 160))
    assert seg is not None
    assert seg.level == 0
    assert seg.classes == ["TERRACE", "STEP", "DEFECT", "CONTAMINATION"]
    assert seg.shape == (256, 256)                       # padded back to full frame
    assert summary is not None and "DEFECT" in summary   # decision summary rides along
    mask = decode_rle(seg.mask_rle, seg.shape)
    assert (mask[:160] == 0).mean() > 0.85               # acquired lattice → terrace
    assert (mask[:160] == 3).mean() < 0.05               # ~no hallucinated contamination
    assert (mask[160:] == 0).all()                       # unscanned rows → terrace


def test_segment_classical_frame_too_few_rows_none():
    mon = _mon("seg2")
    assert mon._segment_classical_frame(_partial(_lattice(), 10)) == (None, None)


# ── monitor emits the correct mid-scan-change event to the buffer ─────────────
class _StubBuffer:
    def __init__(self):
        self.events = []
        self._seq = 0

    def next_seq(self):
        self._seq += 1
        return self._seq

    def emit_event(self, ev):
        self.events.append(ev)


def test_monitor_emits_critical_tip_change_event():
    from mast.buffer.schemas import Severity, VisionEventType
    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="emit", buffer=buf)
    mon._scan_size_nm = 6.0
    mon._realtime_check(_partial_with_change(96, 160), 0.6)
    drops = [e for e in buf.events if e.kind == VisionEventType.TIP_QUALITY_DROP
             and e.payload.get("signal") == "tip_change"]
    assert len(drops) == 1
    ev = drops[0]
    assert ev.severity == Severity.CRITICAL
    assert ev.payload["recommend"] == ["StopScan", "ConditionTip"]
    assert "扫描中途针尖状态突变" in ev.payload["summary_zh"]


# ── end-to-end: monitor → buffer → IC agent buffer_hitl interrupt ─────────────
class _Capture:
    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return None


@pytest_asyncio.fixture
async def _live_buffer(tmp_path):
    from mast.buffer.service import BufferService
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        yield buf
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_e2e_mid_scan_change_reaches_the_agent_without_stopping_it(_live_buffer):
    """Full flow: monitor detects a mid-scan tip change → emits CRITICAL
    TIP_QUALITY_DROP to the buffer → the IC agent's buffer_hitl middleware
    RECORDS it and does NOT interrupt.

    ⚠️ **This assertion was reversed on 2026-08-05, deliberately.** It used to
    read "buffer_hitl did not interrupt on the mid-scan tip change" as a
    failure message — the interrupt was the point. One night on the instrument then
    measured this class of verdict at **0 true positives**, twice suspending a
    composite mid-relocation on a junction whose σ was 6.6 fA, and the operator
    decided that guesses about tip and image quality leave the interrupt path.

    What is asserted here is the half that must NOT change: the monitor still
    sees it, the buffer still takes it, and the agent still gets the event_id
    in its state. Surfacing without stopping is the whole design —
    ``docs/v2/design/au111_tip_forge_uninterrupted.md`` §五.

    To put the interrupt back you would have to answer: **what changed to make
    the vision tip verdict worth stopping an experiment for?**

    ⑰(2026-08-08)扩到了全部事件:``interrupt_fn`` 注入口本身已经删掉,因为模块里
    根本没有 interrupt 调用点了(``test_buffer_hitl.py`` 用 AST 钉着这一条)。
    所以这里不再需要捕获器 —— 「没弹框」现在由**结构**保证,而这条测试要钉的是
    另一半:事件照样一路到达 agent 的 state。"""
    from mast.agents._shared.buffer_hitl import make_buffer_hitl_middleware
    from mast.buffer.schemas import VisionEventType

    mw = make_buffer_hitl_middleware(buffer=_live_buffer)
    # A second subscriber, so the "still published" half is asserted against the
    # real bus rather than inferred from the middleware's own bookkeeping.
    witness = _live_buffer.subscribe(VisionEventType.TIP_QUALITY_DROP)
    try:
        mon = ScanVisionMonitor(None, scan_id="e2e", buffer=_live_buffer)
        mon._scan_size_nm = 6.0
        mon._realtime_check(_partial_with_change(96, 160), 0.6)   # emits the CRITICAL event
        await asyncio.sleep(0)                                     # drain the fanout
        update = mw.before_model(state={}, runtime=None)
        assert update is not None and update.get("event_refs"), (
            "…but the event must still reach the agent's state")
        assert mw.gate_state()["closed"] is False, "the tool gate must stay open"
        assert mw.gate_state()["recorded_not_escalated"] >= 1
        # And it is genuinely the same event KIND on the bus — the demotion is by
        # payload at the consumer, not the monitor having stopped publishing.
        seen = witness.get_nowait()
        assert seen.kind is VisionEventType.TIP_QUALITY_DROP
        assert seen.payload.get("signal") == "tip_change"
    finally:
        _live_buffer.unsubscribe(VisionEventType.TIP_QUALITY_DROP, witness)
        mw.close()
