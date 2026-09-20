"""SAFE-mode safety net at the buffer's tip-quality edge + the halt hook.

Two independent guards, both deliberately *last-resort*:

1. ``BufferService._maybe_emit_quality_drop`` — in SAFE every tip verdict is
   rewritten at its producer, so a BAD TipStatus arriving here means a producer
   bypassed the override. Do not escalate (a CRITICAL halts the running
   composite and interrupts the agent — precisely what SAFE promises not to do),
   but log loudly: this branch firing IS the designed hole detector.

2. ``make_tip_halt_hook`` — the discrimination that keeps SAFE from switching off
   physical protection. Vision and the current monitor publish CRITICALs under
   the SAME event kind; only the vision tip verdict is suppressed. Current-
   monitor saturation / freeze / giant-spike mean the tip is railed into the
   surface or the measurement chain is dead, and those must halt in every mode.
"""
from __future__ import annotations

import logging
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

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from mast.buffer.schemas import (  # noqa: E402
    Severity,
    TipQuality,
    TipStatus,
    VisionEvent,
    VisionEventType,
)
from mast.buffer.service import BufferService  # noqa: E402
from mast.core.operating_mode import bind_mode_source  # noqa: E402


@pytest.fixture(autouse=True)
def _unbind_after():
    yield
    bind_mode_source(None)


def _safe():
    bind_mode_source(lambda: "safe")


@pytest_asyncio.fixture
async def buf(tmp_path):
    b = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


def _ts(seqno: int, quality: TipQuality) -> TipStatus:
    return TipStatus(seqno=seqno, quality=quality, confidence=0.9,
                     scan_id="s1", frame_idx=seqno)


def _drops(events):
    return [e for e in events if e.kind is VisionEventType.TIP_QUALITY_DROP]


# ── 1. the buffer safety net ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bad_tip_status_in_safe_is_warn_not_critical(buf, caplog):
    _safe()
    buf.put_tip_status(_ts(1, TipQuality.GOOD))
    with caplog.at_level(logging.WARNING, logger="mast.buffer.service"):
        buf.put_tip_status(_ts(2, TipQuality.BAD))

    ev = _drops(buf.get_event_history(since_seqno=-1, limit=50))
    assert len(ev) == 1
    assert ev[0].severity is Severity.WARN            # → no halt, no HITL
    assert ev[0].payload["safe_mode_suppressed"] is True
    assert "覆写链有漏" in caplog.text, "the hole detector must say so out loud"


@pytest.mark.asyncio
async def test_bad_tip_status_outside_safe_stays_critical(buf):
    bind_mode_source(lambda: "auto")
    buf.put_tip_status(_ts(1, TipQuality.GOOD))
    buf.put_tip_status(_ts(2, TipQuality.BAD))

    ev = _drops(buf.get_event_history(since_seqno=-1, limit=50))
    assert len(ev) == 1
    assert ev[0].severity is Severity.CRITICAL
    assert "safe_mode_suppressed" not in ev[0].payload


@pytest.mark.asyncio
async def test_unbound_holder_keeps_historical_critical(buf):
    """The regression guard: no runtime bound → byte-for-byte old behaviour."""
    bind_mode_source(None)
    buf.put_tip_status(_ts(1, TipQuality.GOOD))
    buf.put_tip_status(_ts(2, TipQuality.BAD))
    assert _drops(buf.get_event_history(since_seqno=-1, limit=50))[0].severity is Severity.CRITICAL


# ── 【必测】mode switch auto → safe → auto leaves a clean rising edge ────────

@pytest.mark.asyncio
async def test_mode_switch_round_trip_keeps_edge_clean(buf):
    """The reason SAFE writes GOOD to the store instead of leaving BAD there.

    The quality-drop event is RISING-EDGE triggered (good→bad). If SAFE let real
    BADs land in the store, `_last_emitted_quality` would sit at BAD for the whole
    SAFE stretch, and the first genuine bad tip after switching back to auto would
    be a bad→bad transition — no edge, no event, no alarm. Overriding at the
    producer keeps the latch at GOOD, so the switch back is honest immediately.
    """
    bind_mode_source(lambda: "auto")
    buf.put_tip_status(_ts(1, TipQuality.GOOD))

    # SAFE stretch: producers write GOOD (that is what the facade override does).
    _safe()
    for i in range(2, 6):
        buf.put_tip_status(_ts(i, TipQuality.GOOD))
    assert _drops(buf.get_event_history(since_seqno=-1, limit=50)) == []

    # Operator switches back; the very next real BAD must fire exactly one CRITICAL.
    bind_mode_source(lambda: "auto")
    buf.put_tip_status(_ts(6, TipQuality.BAD))

    ev = _drops(buf.get_event_history(since_seqno=-1, limit=50))
    assert len(ev) == 1, "expected exactly one rising-edge CRITICAL after the switch"
    assert ev[0].severity is Severity.CRITICAL
    assert ev[0].seqno == 6


@pytest.mark.asyncio
async def test_suppressed_drop_does_not_poison_the_latch(buf, caplog):
    """The safety net's own round trip — the case the first implementation got
    wrong.

    The latch records what the alarm chain last ACTED on. If a suppressed drop
    moved it to BAD, the first genuine bad tip after switching back to auto would
    be a bad→bad transition: no rising edge, no event, no alarm. That is exactly
    the failure the "SAFE writes good to the store" decision exists to prevent,
    reintroduced through the one path that can still receive a BAD in SAFE.
    """
    bind_mode_source(lambda: "auto")
    buf.put_tip_status(_ts(1, TipQuality.GOOD))

    # A producer bypasses the override and writes a real BAD while SAFE is on.
    _safe()
    with caplog.at_level(logging.WARNING, logger="mast.buffer.service"):
        buf.put_tip_status(_ts(2, TipQuality.BAD))
    assert _drops(buf.get_event_history(since_seqno=-1, limit=50))[0].severity is Severity.WARN

    # Operator switches back; the next real BAD must still be a rising edge.
    bind_mode_source(lambda: "auto")
    buf.put_tip_status(_ts(3, TipQuality.BAD))

    evs = _drops(buf.get_event_history(since_seqno=-1, limit=50))
    assert [e.severity for e in evs] == [Severity.WARN, Severity.CRITICAL]
    assert evs[-1].seqno == 3


@pytest.mark.asyncio
async def test_stale_bad_is_not_read_back_in_safe(buf):
    """The read side. Writing "good" at the producer covers what is published
    WHILE safe is on; it does nothing about what was already in the slot when the
    operator switched — and that is the likeliest case of all, because the
    operator switches to SAFE precisely after the system called the tip bad."""
    bind_mode_source(lambda: "auto")
    buf.put_tip_status(_ts(1, TipQuality.BAD))
    assert buf.get_latest_tip_status()[0].quality is TipQuality.BAD

    _safe()
    ts, _seq = buf.get_latest_tip_status()
    assert ts.quality is TipQuality.GOOD
    assert ts.safe_mode is True                  # marked as a mode, not a measurement
    assert all(t.quality is TipQuality.GOOD for t in buf.get_tip_history(0))

    bind_mode_source(lambda: "auto")
    assert buf.get_latest_tip_status()[0].quality is TipQuality.BAD   # truth restored


# ── 2. 【必测】the halt hook: SAFE must not switch off physical protection ───

class _App:
    def __init__(self):
        self.halts = []
        self.sources = []

    def raise_tip_halt(self, reason, event_id="", seqno=None, source="", **_kw):
        self.halts.append(reason)
        self.sources.append(source)


def _ev(payload, seqno=1):
    return VisionEvent(seqno=seqno, kind=VisionEventType.TIP_QUALITY_DROP,
                       severity=Severity.CRITICAL, payload=payload,
                       cause_ref="test")


def _hook(app):
    from mast.core.runtime import make_tip_halt_hook
    return make_tip_halt_hook(app)


def test_current_monitor_saturation_still_halts_in_safe():
    """SAFE means "do not repair the tip", NEVER "switch off the protections".
    Saturation = the tip is railed into the surface / the preamp is pinned. A
    naive "skip tip_quality_drop halts in SAFE" would have swallowed this."""
    _safe()
    app = _App()
    _hook(app)(_ev({"signal": "current_saturation", "source": "current_monitor",
                    "summary_zh": "隧道电流贴轨"}))
    assert app.halts == ["隧道电流贴轨"]
    # …and the latch is told which kind it is, so a mode switch between arming
    # and consuming still knows not to drop it.
    assert app.sources == ["current_monitor"]


@pytest.mark.parametrize("signal", ["current_freeze", "current_giant_spike"])
def test_other_current_monitor_criticals_still_halt_in_safe(signal):
    _safe()
    app = _App()
    _hook(app)(_ev({"signal": signal, "source": "current_monitor",
                    "summary_zh": "电流监控严重告警"}))
    assert app.halts, f"{signal} must halt even in SAFE"


def test_current_signal_prefix_alone_is_enough():
    """Defence in depth: either marker (source OR the current_* signal prefix)
    keeps the halt, so a payload that loses one still protects the instrument."""
    _safe()
    app = _App()
    _hook(app)(_ev({"signal": "current_saturation", "summary_zh": "贴轨"}))
    assert app.halts == ["贴轨"]


def test_vision_tip_change_halt_is_skipped_in_safe():
    _safe()
    app = _App()
    _hook(app)(_ev({"signal": "tip_change", "summary_zh": "扫描中途针尖状态突变"}))
    assert app.halts == []


def test_unknown_vision_signal_is_skipped_in_safe():
    """Fail toward SAFE's contract for vision: a future vision signal with no
    marker is suppressed by default; the current monitor's explicit markers are
    what preserve the halt."""
    _safe()
    app = _App()
    _hook(app)(_ev({"signal": "some_future_vision_rule", "summary_zh": "x"}))
    assert app.halts == []


# ── ⑰-C1(2026-08-09):视觉来源在**任何模式**下都不再 halt ──────────────────
#
# 下面两条原名 ``test_vision_tip_change_halts_outside_safe`` /
# ``test_unbound_holder_halts_as_before``,断言的是「SAFE 之外视觉照样 halt」——
# 那是本节的对照组:它证明 SAFE 的抑制是**模式**造成的,不是判据坏了。
#
# 定案把视觉 halt 整条割掉(「也割掉」),所以对照组的答案反了:视觉在 AUTO、
# 在没绑定模式源时,同样不 halt。本节要守的分界线**没变**,只是换了一句话说:
# 现在守的是「**物理来源**在任何模式下都照旧 halt」(上面那四条,一个字没改),
# 而不再是「视觉在 SAFE 之外照旧 halt」。
#
# 两条都保留而不是删掉:一条被删掉的测试和一条从没写过的测试长得一模一样,而
# 「SAFE 曾经是唯一让视觉安静下来的东西」是这个文件存在的理由之一。


def test_vision_tip_change_no_longer_halts_outside_safe():
    """语义反转的对照组。⑰-C1 之前这里是 ``halts == ["扫描中途针尖状态突变"]``。"""
    bind_mode_source(lambda: "auto")
    app = _App()
    _hook(app)(_ev({"signal": "tip_change", "summary_zh": "扫描中途针尖状态突变"}))
    assert app.halts == [], (
        "视觉针尖判定又开始中止流程了 —— 若这是刻意的，请先回答 ⑰-C1 的翻盘观测:"
        "哪一次中止避免了真实损害，而拒绝型防护都没接住？")
    assert app.sources == []


def test_unbound_mode_holder_does_not_halt_on_vision_either():
    """没绑定模式源(离线/测试)时也一样 —— 豁免不依赖能不能读到模式。

    这一条以前叫 ``..._halts_as_before``,守的是「模式读不到时退回旧行为」。
    ⑰-C1 之后旧行为本身没了,而它守的性质换成:**豁免不能依赖一个可能缺席的读取器**。
    """
    bind_mode_source(None)
    app = _App()
    _hook(app)(_ev({"signal": "tip_change", "summary_zh": "扫描中途针尖状态突变"}))
    assert app.halts == []


def test_the_physical_halt_is_what_this_section_now_guards():
    """把本节现在真正的分界线写成一条断言:**物理来源在任何模式下都照旧 halt**。

    上面四条已经分别覆盖 SAFE 下的三个物理信号;这一条补 AUTO 那一侧,免得
    「物理 halt 只在 SAFE 下被测过」——那会让「C1 顺手把物理也割了」逃过去。
    """
    bind_mode_source(lambda: "auto")
    app = _App()
    _hook(app)(_ev({"signal": "current_saturation", "source": "current_monitor",
                    "summary_zh": "隧道电流贴轨"}))
    assert app.halts == ["隧道电流贴轨"]
    assert app.sources == ["current_monitor"]


def test_warn_severity_never_halts_in_any_mode():
    """Pre-existing contract, re-pinned: only CRITICAL halts. The SAFE net
    demotes to WARN, and this is why that demotion is sufficient."""
    _safe()
    app = _App()
    ev = VisionEvent(seqno=1, kind=VisionEventType.TIP_QUALITY_DROP,
                     severity=Severity.WARN, payload={"signal": "current_saturation",
                                                      "source": "current_monitor"},
                     cause_ref="t")
    _hook(app)(ev)
    assert app.halts == []
