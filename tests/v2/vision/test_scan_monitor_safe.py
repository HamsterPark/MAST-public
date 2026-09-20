"""SAFE mode at the live scan monitor.

The monitor is where a mid-scan tip change becomes the CRITICAL that aborts the
running composite and raises a HITL interrupt. In SAFE that is exactly the
behaviour the operator asked NOT to have, so the alert is demoted to INFO — it
still lands in the record and the GUI, it just stops driving the machine.

What must NOT change: oscillation / bad-line alerts (scan problems, not tip
repair), and the ability to raise the real CRITICAL the moment the operator
switches back mid-scan.
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

from mast.buffer.schemas import Severity, VisionEventType  # noqa: E402
from mast.core.operating_mode import bind_mode_source  # noqa: E402
from mast.vision.scan_monitor import ScanVisionMonitor  # noqa: E402


@pytest.fixture(autouse=True)
def _unbind_after():
    yield
    bind_mode_source(None)
    # Two tests below swap in the mock backend; do not leave it for the next file.
    from mast.vision.module import VisionModule
    VisionModule._instance = None


def _safe():
    bind_mode_source(lambda: "safe")


def _lattice(H=256, W=256, period=7.3, seed=0):
    yy, xx = np.mgrid[0:H, 0:W]
    th = 0.3
    x2 = xx * np.cos(th) + yy * np.sin(th)
    y2 = -xx * np.sin(th) + yy * np.cos(th)
    return (np.sin(x2 * 2 * np.pi / period + 0.7)
            * np.sin(y2 * 2 * np.pi / (period * 1.13) + 1.1)
            + 0.05 * np.random.RandomState(seed).randn(H, W)).astype(np.float32)


def _partial(fwd, acquired_rows):
    bwd = fwd + 0.05 * np.random.RandomState(9).randn(*fwd.shape).astype(np.float32)
    fwd = fwd.copy(); bwd = bwd.copy()
    fwd[acquired_rows:] = 0.0
    bwd[acquired_rows:] = 0.0
    return np.stack([fwd, bwd]).astype(np.float32)


def _partial_with_change(change_row=96, acquired=160):
    fwd = _lattice()
    fwd[change_row:acquired] += 0.6
    fwd[change_row:acquired] = ndi.gaussian_filter(fwd[change_row:acquired], 1.5) \
        + 0.15 * np.random.RandomState(1).randn(acquired - change_row, fwd.shape[1])
    return _partial(fwd, acquired)


class _StubBuffer:
    def __init__(self):
        self.events = []
        self.tip_status = []
        self._seq = 0

    def next_seq(self):
        self._seq += 1
        return self._seq

    def emit_event(self, ev):
        self.events.append(ev)

    def put_tip_status(self, ts):
        self.tip_status.append(ts)


def _drops(buf, signal="tip_change"):
    return [e for e in buf.events
            if e.kind == VisionEventType.TIP_QUALITY_DROP
            and e.payload.get("signal") == signal]


# ── tip_change: demoted, not deleted ────────────────────────────────────────

def test_tip_change_demoted_to_info_in_safe():
    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="safe1", buffer=buf)
    mon._scan_size_nm = 6.0
    _safe()
    rt = mon._realtime_check(_partial_with_change(96, 160), 0.6)

    # The detector still ran and still reports the truth as DATA.
    assert rt["tip_change"]["changed"] is True

    evs = _drops(buf)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.severity == Severity.INFO          # not CRITICAL → no halt, no HITL
    assert ev.payload["safe_mode_suppressed"] is True
    assert "recommend" not in ev.payload         # no StopScan/ConditionTip advice
    assert "安全模式" in ev.payload["summary_zh"]


def test_tip_change_critical_intact_outside_safe():
    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="auto1", buffer=buf)
    mon._scan_size_nm = 6.0
    bind_mode_source(lambda: "auto")
    mon._realtime_check(_partial_with_change(96, 160), 0.6)

    ev = _drops(buf)[0]
    assert ev.severity == Severity.CRITICAL
    assert ev.payload["recommend"] == ["StopScan", "ConditionTip"]
    assert "safe_mode_suppressed" not in ev.payload


def test_semi_keeps_the_critical():
    """SEMI allows shallow shaping and routes pulses to HITL — it needs the real
    alert to make that choice."""
    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="semi1", buffer=buf)
    mon._scan_size_nm = 6.0
    bind_mode_source(lambda: "semi")
    mon._realtime_check(_partial_with_change(96, 160), 0.6)
    assert _drops(buf)[0].severity == Severity.CRITICAL


# ── the dedup trap: switching mode mid-scan ─────────────────────────────────

def test_switching_out_of_safe_midscan_can_still_raise_critical():
    """`_rt_alerted` fires each alert at most once per scan. If the SAFE-demoted
    alert used the SAME dedup key, an operator who switched back to auto during
    the scan would never get the real CRITICAL for that scan — the alert would
    look 'already sent'. Separate keys are what make the switch honest."""
    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="switch", buffer=buf)
    mon._scan_size_nm = 6.0
    frame = _partial_with_change(96, 160)

    _safe()
    mon._realtime_check(frame, 0.5)
    assert _drops(buf)[0].severity == Severity.INFO

    bind_mode_source(lambda: "auto")             # operator flips mid-scan
    mon._realtime_check(frame, 0.7)

    sevs = [e.severity for e in _drops(buf)]
    assert Severity.CRITICAL in sevs, "post-switch CRITICAL was eaten by dedup"


def test_safe_alert_still_deduped_within_safe():
    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="dedup", buffer=buf)
    mon._scan_size_nm = 6.0
    frame = _partial_with_change(96, 160)
    _safe()
    mon._realtime_check(frame, 0.5)
    mon._realtime_check(frame, 0.7)
    assert len(_drops(buf)) == 1                 # one per scan, as before


def test_learned_quality_bad_demoted_in_safe(monkeypatch):
    """The other tip alert in _realtime_check. Untested in the first pass — and
    an untested half of a two-branch override is how one of them drifts."""
    class _Q:
        score = 0.11
        tier = "bad"

    class _StubVision:
        def assess_quality(self, img, scan_size_nm=None):
            return _Q()

    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="lq", buffer=buf)
    mon._vision = _StubVision()
    _safe()
    mon._realtime_check(_partial(_lattice(seed=4), 200), 0.8)

    evs = _drops(buf, signal="learned_quality")
    assert len(evs) == 1
    assert evs[0].severity == Severity.INFO
    assert evs[0].payload["safe_mode_suppressed"] is True
    assert evs[0].payload["tier"] == "bad"       # the score stays visible as data


def test_learned_quality_warn_intact_outside_safe(monkeypatch):
    class _Q:
        score = 0.11
        tier = "bad"

    class _StubVision:
        def assess_quality(self, img, scan_size_nm=None):
            return _Q()

    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="lq2", buffer=buf)
    mon._vision = _StubVision()
    bind_mode_source(lambda: "auto")
    mon._realtime_check(_partial(_lattice(seed=4), 200), 0.8)
    assert _drops(buf, signal="learned_quality")[0].severity == Severity.WARN


def test_milestone_severity_matches_the_demoted_alert(monkeypatch):
    """Self-consistency: the dedicated tip_change alert is demoted to INFO in
    SAFE, so the milestone event for the same frame must not stay at WARN.
    One finding, one severity."""
    from mast.vision.module import VisionModule

    monkeypatch.setenv("MAST_VISION_BACKEND", "mock")
    VisionModule._instance = None
    vm = VisionModule.get()

    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="sev", buffer=buf)
    _safe()
    coarse = vm.assess_tip_coarse(np.random.RandomState(0).randn(64, 64).astype(np.float32))
    rt = {"tip_change": {"changed": True, "row": 96}}
    mon._emit_milestone_event(buf, 0.5, False, coarse, None, rt=rt)

    ev = [e for e in buf.events if e.kind is VisionEventType.FEATURE_OF_INTEREST][0]
    assert ev.severity == Severity.INFO
    # the detection itself is still recorded truthfully
    assert ev.payload["classical"]["tip_change"]["changed"] is True


def test_milestone_severity_is_warn_outside_safe(monkeypatch):
    from mast.vision.module import VisionModule

    monkeypatch.setenv("MAST_VISION_BACKEND", "mock")
    VisionModule._instance = None
    vm = VisionModule.get()

    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="sev2", buffer=buf)
    bind_mode_source(lambda: "auto")
    coarse = vm.assess_tip_coarse(np.random.RandomState(0).randn(64, 64).astype(np.float32))
    mon._emit_milestone_event(buf, 0.5, False, coarse, None,
                              rt={"tip_change": {"changed": True, "row": 96}})
    ev = [e for e in buf.events if e.kind is VisionEventType.FEATURE_OF_INTEREST][0]
    assert ev.severity == Severity.WARN


# ── scan-problem alerts are NOT tip repair and must survive SAFE ────────────

def test_oscillation_warn_survives_safe():
    """Feedback ringing is fixed by retuning the loop, not by repairing the tip.
    Suppressing it would blind the agent to a fault it could actually fix."""
    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="osc", buffer=buf)
    H = 256
    xx = np.mgrid[0:H, 0:H][1]
    ring = (1.5 * np.sin(xx * 2 * np.pi / 6)
            + 0.05 * np.random.RandomState(0).randn(H, H)).astype(np.float32)
    _safe()
    rt = mon._realtime_check(_partial(ring, 200), 0.8)
    assert rt["artifacts"]["oscillation"] is True
    evs = _drops(buf, signal="oscillation")
    assert len(evs) == 1
    assert evs[0].severity == Severity.WARN
    assert "safe_mode_suppressed" not in evs[0].payload


# ── TipStatus publication + milestone payload ──────────────────────────────

def test_publish_coarse_writes_good_in_safe(monkeypatch):
    """_publish_coarse is the ONLY producer of a TipStatus in the whole system.
    With the facade override upstream it must write GOOD, which is what makes
    every downstream consumer (buffer edge, halt hook, HITL, full_scan's vision
    note, the agent's read_latest_tip_status) consistent."""
    from mast.buffer.schemas import TipQuality
    from mast.vision.module import VisionModule

    monkeypatch.setenv("MAST_VISION_BACKEND", "mock")
    VisionModule._instance = None
    vm = VisionModule.get()

    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="pub", buffer=buf)
    _safe()
    coarse = vm.assess_tip_coarse(np.random.RandomState(0).randn(64, 64).astype(np.float32))
    mon._publish_coarse(buf, coarse, ordinal=1)

    assert len(buf.tip_status) == 1
    assert buf.tip_status[0].quality is TipQuality.GOOD


def test_milestone_payload_carries_raw_verdict_and_flag(monkeypatch):
    """The human-facing truth lives here: the milestone event payload keeps the
    model's real verdict in tip_coarse.safe_mode_raw plus a top-level safe_mode
    flag, so a record from SAFE is never mistaken for a measured 'good tip'."""
    from mast.vision.module import VisionModule

    monkeypatch.setenv("MAST_VISION_BACKEND", "mock")
    VisionModule._instance = None
    vm = VisionModule.get()

    buf = _StubBuffer()
    mon = ScanVisionMonitor(None, scan_id="ms", buffer=buf)
    _safe()
    coarse = vm.assess_tip_coarse(np.random.RandomState(0).randn(64, 64).astype(np.float32))
    mon._emit_milestone_event(buf, 0.5, False, coarse, None)

    ev = [e for e in buf.events if e.kind is VisionEventType.FEATURE_OF_INTEREST][0]
    assert ev.payload["safe_mode"] is True
    assert ev.payload["tip_coarse"]["label"] == "good"
    assert ev.payload["tip_coarse"]["safe_mode_raw"] == {"label": "bad", "confidence": 0.5}


# ── end-to-end: SAFE must not raise the IC agent's HITL interrupt ───────────

@pytest_asyncio.fixture
async def _live_buffer(tmp_path):
    from mast.buffer.service import BufferService
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        yield buf
    finally:
        await buf.stop()


class _Capture:
    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return None


@pytest.mark.asyncio
async def test_e2e_safe_mid_scan_change_does_not_interrupt(_live_buffer):
    """The mirror of test_e2e_mid_scan_change_triggers_ic_interrupt: same frame,
    same wiring, SAFE on → the IC agent is NOT interrupted, so the experiment
    SAFE told it to focus on keeps running.

    ⑰(2026-08-08):「不打断」现在对**所有**模式都成立,而且是结构性的(模块里
    没有 interrupt 调用点,``test_buffer_hitl.py`` 用 AST 钉着)。所以这条测试
    改钉 SAFE **特有**的那一半 —— 生产者侧的降级:SAFE 把针尖判定改写成 good,
    于是同一帧发出来的 ``tip_quality_drop`` 是 **INFO**,连「值得单独通知一行」
    的门槛都够不到,而事件本身照样到达 agent 的 state。

    这里刻意**不去查全局诊断台账**。第一版就是那么写的,它绿了 —— 因为
    ``diagnostics`` 是进程级环形缓冲,同一次 pytest 里另一个文件写进去的
    ``notice_only`` 行让 ``assert rows`` 通过了,而这条测试自己一行都没写。
    断言要对着**这个中间件自己的账**(``recorded_not_escalated``)。"""
    from mast.agents._shared.buffer_hitl import make_buffer_hitl_middleware

    _safe()
    mw = make_buffer_hitl_middleware(buffer=_live_buffer, get_mode=lambda: "safe")
    try:
        mon = ScanVisionMonitor(None, scan_id="e2e-safe", buffer=_live_buffer)
        mon._scan_size_nm = 6.0
        mon._realtime_check(_partial_with_change(96, 160), 0.6)
        await asyncio.sleep(0)
        update = mw.before_model(state={}, runtime=None)
        assert update is not None and update.get("event_refs"), (
            "SAFE 模式下事件也必须照样到达 agent 的 state")
        assert mw.gate_state()["closed"] is False
        assert mw.gate_state()["recorded_not_escalated"] == 0, (
            "SAFE 模式下这一帧不该产生 CRITICAL —— 判定在生产者侧就被改写成 good 了")
    finally:
        mw.close()


def test_the_safe_mode_suggestion_still_avoids_tip_repair():
    """#43 的落点:SAFE 模式下建议动作不能是「去修针」。

    SafetyGate 在 SAFE 下硬拦修针,所以建议 ``tip_prep`` 是在推荐一件系统自己不肯
    做的事 —— 用户看到过并提了这条。它原来只活在审批框的 payload 里;⑰ 把框删
    掉之后,同一个函数改为给**通知**用,这条断言跟着搬过来,否则 #43 的修复会随框
    一起消失而没人发现。"""
    from mast.agents._shared.buffer_hitl import _suggest_action
    from mast.buffer.schemas import VisionEventType

    assert _suggest_action(VisionEventType.TIP_QUALITY_DROP,
                           safe_mode=True) == "manual_tip_check"
    assert _suggest_action(VisionEventType.TIP_QUALITY_DROP,
                           safe_mode=False) == "tip_prep"
