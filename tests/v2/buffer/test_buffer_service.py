"""BufferService tests — single-thread + cross-thread + edge events + slow sub + WAL."""
from __future__ import annotations

import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import asyncio
import threading
import time

import pytest

from mast.buffer.schemas import (
    ScanProgress,
    Severity,
    TipQuality,
    TipStatus,
    VisionEvent,
    VisionEventType,
    make_e_stop,
    make_setpoint_change,
    make_vision_error,
)
from mast.buffer.service import BufferService, _LatestSlot


# ─────────────────────────────────────────────────────────────────────
# _LatestSlot — atomic value+seqno pair under threading.Lock
# ─────────────────────────────────────────────────────────────────────

class TestLatestSlot:
    def test_initial_get_returns_none(self):
        slot: _LatestSlot[int] = _LatestSlot(threading.Lock())
        assert slot.get() == (None, -1)

    def test_put_then_get(self):
        slot: _LatestSlot[int] = _LatestSlot(threading.Lock())
        slot.put(42, 1)
        assert slot.get() == (42, 1)
        slot.put(43, 2)
        assert slot.get() == (43, 2)

    def test_concurrent_writes_no_torn_pair(self):
        """200-iter race: writer thread + reader thread, value/seqno must stay paired."""
        lock = threading.Lock()
        slot: _LatestSlot[int] = _LatestSlot(lock)
        stop = threading.Event()
        torn = []

        def writer():
            for i in range(2000):
                slot.put(i * 10, i)  # value = seqno * 10
                if stop.is_set():
                    break

        def reader():
            for _ in range(2000):
                v, s = slot.get()
                if v is not None and s >= 0 and v != s * 10:
                    torn.append((v, s))
                if stop.is_set():
                    break

        t_w = threading.Thread(target=writer)
        t_r = threading.Thread(target=reader)
        t_w.start()
        t_r.start()
        t_w.join(timeout=5)
        t_r.join(timeout=5)
        stop.set()
        assert torn == []


# ─────────────────────────────────────────────────────────────────────
# BufferService basic flow
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_put_get_tip_status(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite")
    await buf.start()
    try:
        seq = buf.next_seq()
        ts = TipStatus(seqno=seq, quality=TipQuality.GOOD, confidence=0.9,
                       scan_id="s1", frame_idx=0)
        buf.put_tip_status(ts)
        latest, latest_seq = buf.get_latest_tip_status()
        assert latest is not None
        assert latest.confidence == 0.9
        assert latest_seq == seq
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_seqno_monotonic(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        s1 = buf.next_seq()
        s2 = buf.next_seq()
        s3 = buf.next_seq()
        assert s1 < s2 < s3
        assert s2 == s1 + 1 and s3 == s2 + 1
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_tip_history_returns_since(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        for i in range(5):
            seq = buf.next_seq()
            buf.put_tip_status(TipStatus(seqno=seq, quality=TipQuality.GOOD,
                                          confidence=0.5, scan_id="s", frame_idx=i))
        history = buf.get_tip_history(since_seq=2)
        # Should include seqnos 3, 4, 5
        assert [t.seqno for t in history] == [3, 4, 5]
    finally:
        await buf.stop()


# ─────────────────────────────────────────────────────────────────────
# Edge-triggered event emission
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quality_drop_edge_triggered(tmp_path):
    """First good→bad emits an event; subsequent bad→bad does NOT."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        q = buf.subscribe(VisionEventType.TIP_QUALITY_DROP)

        # Frame 1: GOOD (no event)
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.GOOD, confidence=0.9,
                                      scan_id="s", frame_idx=0))
        # Frame 2: BAD (rising edge — emit)
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.BAD, confidence=0.9,
                                      scan_id="s", frame_idx=1))
        # Frame 3: BAD again (no edge — no emit)
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.BAD, confidence=0.9,
                                      scan_id="s", frame_idx=2))
        # Frame 4: GOOD (back; no emit because we're tracking only DROPs)
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.GOOD, confidence=0.9,
                                      scan_id="s", frame_idx=3))
        # Frame 5: BAD (rising edge again — emit)
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.BAD, confidence=0.9,
                                      scan_id="s", frame_idx=4))

        # Allow the asyncio loop to fan out (call_soon_threadsafe is queued)
        # Since producer ran in same thread/loop here, fanout already executed.
        # Drain queue:
        events = []
        for _ in range(10):
            try:
                ev = q.get_nowait()
                events.append(ev)
            except asyncio.QueueEmpty:
                break
        assert len(events) == 2  # only 2 rising edges
        assert all(ev.kind == VisionEventType.TIP_QUALITY_DROP for ev in events)
        assert events[0].severity == Severity.CRITICAL  # bad
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_slow_subscriber_drops_oldest(tmp_path):
    """Subscriber with size=2 receives only the 2 most recent events."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        q = buf.subscribe(VisionEventType.TIP_QUALITY_DROP, max_size=2)
        # Force 5 rising edges by alternating GOOD/BAD
        for i in range(5):
            buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                          quality=TipQuality.GOOD, confidence=0.9,
                                          scan_id="s", frame_idx=2 * i))
            buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                          quality=TipQuality.BAD, confidence=0.9,
                                          scan_id="s", frame_idx=2 * i + 1))
        events = []
        while True:
            try:
                events.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        # Queue capped at 2; with drop-oldest, the latest 2 events should be present
        assert len(events) == 2
        # The latest events should reference the highest seqnos
        assert events[-1].seqno >= 9
    finally:
        await buf.stop()


# ─────────────────────────────────────────────────────────────────────
# Cross-thread emission (vision producer thread → agent coroutine)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cross_thread_emission(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        q = buf.subscribe(VisionEventType.TIP_QUALITY_DROP)

        def producer_thread():
            # Simulate vision thread: GOOD → BAD edge
            buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                          quality=TipQuality.GOOD, confidence=0.9,
                                          scan_id="s", frame_idx=0))
            buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                          quality=TipQuality.BAD, confidence=0.9,
                                          scan_id="s", frame_idx=1))

        t = threading.Thread(target=producer_thread, daemon=True)
        t.start()
        t.join(timeout=2)

        # Wait briefly for the loop to fan out (call_soon_threadsafe)
        ev = await asyncio.wait_for(q.get(), timeout=2.0)
        assert ev.kind == VisionEventType.TIP_QUALITY_DROP
        assert ev.severity == Severity.CRITICAL
    finally:
        await buf.stop()


# ─────────────────────────────────────────────────────────────────────
# WAL persistence — restart restores last seqno
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wal_restores_seqno_on_restart(tmp_path):
    wal_path = tmp_path / "buf.sqlite"
    buf = BufferService(wal_path=wal_path)
    await buf.start()
    try:
        for _ in range(7):
            seq = buf.next_seq()
            buf.put_tip_status(TipStatus(seqno=seq, quality=TipQuality.GOOD,
                                          confidence=0.5, scan_id="s", frame_idx=0))
        # Wait briefly for WAL writes to flush
        await asyncio.sleep(0.2)
    finally:
        await buf.stop()

    # Restart: a new BufferService should pick up where we left off
    buf2 = BufferService(wal_path=wal_path)
    await buf2.start()
    try:
        next_after_restore = buf2.next_seq()
        assert next_after_restore == 8  # 7 was last; 8 is next
    finally:
        await buf2.stop()


# ─────────────────────────────────────────────────────────────────────
# Compass §3.1 / §5.2 / §6 extensions:
#   - factory helpers for VISION_ERROR / SETPOINT_CHANGE / E_STOP
#   - wait_for_event (rising-edge), catchup (WAL replay)
#   - get_event_history (in-memory ring), subscribe_all (fanout)
#   - get_stats
#
# All new tests use fresh fixtures so they don't perturb the 10 existing tests.
# ─────────────────────────────────────────────────────────────────────

def _make_event(buf, kind: VisionEventType, payload: dict | None = None,
                severity: Severity = Severity.INFO) -> VisionEvent:
    return VisionEvent(
        seqno=buf.next_seq(),
        kind=kind,
        severity=severity,
        payload=payload or {},
    )


# ── Factory helpers --------------------------------------------------------

def test_make_vision_error_payload_shape():
    ev = make_vision_error(
        "cuda_oom",
        "torch.cuda.OutOfMemoryError: tried to allocate 4 GB",
        seqno=42,
    )
    assert ev.kind == VisionEventType.VISION_ERROR
    assert ev.severity == Severity.CRITICAL
    assert ev.payload["error_kind"] == "cuda_oom"
    assert ev.payload["detail"].startswith("torch.cuda.OutOfMemoryError")
    assert ev.seqno == 42


def test_make_vision_error_rejects_bad_error_kind():
    with pytest.raises(ValueError, match="not in"):
        make_vision_error("disk_full", "x", seqno=1)


def test_make_vision_error_truncates_long_detail():
    long = "x" * 5000
    ev = make_vision_error("model_error", long, seqno=1)
    assert len(ev.payload["detail"]) == 1024


def test_make_setpoint_change_partial_payload():
    ev = make_setpoint_change(bias_v=1.5, current_a=None, source="planner",
                              seqno=7)
    assert ev.kind == VisionEventType.SETPOINT_CHANGE
    assert ev.severity == Severity.INFO  # default for setpoint changes
    assert ev.payload == {"bias_v": 1.5, "current_a": None, "source": "planner"}


def test_make_setpoint_change_rejects_bad_source():
    with pytest.raises(ValueError, match="source="):
        make_setpoint_change(bias_v=1.0, current_a=1e-9, source="cron",
                             seqno=1)


def test_make_e_stop_payload_shape():
    ev = make_e_stop("watchdog", "no heartbeat for 5s", seqno=3)
    assert ev.kind == VisionEventType.E_STOP
    assert ev.severity == Severity.CRITICAL
    assert ev.payload == {"reason": "watchdog",
                          "detail": "no heartbeat for 5s"}


def test_make_e_stop_rejects_bad_reason():
    with pytest.raises(ValueError, match="reason="):
        make_e_stop("alien_attack", "x", seqno=1)


# ── wait_for_event ---------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_for_event_rising_edge(tmp_path):
    """A future emit with seqno > since_seqno unblocks the waiter."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        async def emitter():
            await asyncio.sleep(0.05)  # let waiter park
            buf.emit_event(make_vision_error(
                "cuda_oom", "alloc failed", seqno=buf.next_seq(),
            ))
        emitter_task = asyncio.create_task(emitter())
        ev = await buf.wait_for_event(
            VisionEventType.VISION_ERROR, since_seqno=-1, timeout=2.0,
        )
        await emitter_task
        assert ev is not None
        assert ev.kind == VisionEventType.VISION_ERROR
        assert ev.payload["error_kind"] == "cuda_oom"
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_wait_for_event_returns_immediately_if_already_satisfied(tmp_path):
    """If an event with seqno > since already exists, return without blocking."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        buf.emit_event(make_setpoint_change(
            bias_v=2.0, current_a=None, source="user",
            seqno=buf.next_seq(),
        ))
        # since_seqno=-1, current seqno=1 → match immediately
        ev = await asyncio.wait_for(
            buf.wait_for_event(VisionEventType.SETPOINT_CHANGE, since_seqno=-1),
            timeout=0.5,
        )
        assert ev is not None
        assert ev.payload["bias_v"] == 2.0
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_wait_for_event_timeout(tmp_path):
    """No emission within timeout → returns None."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        ev = await buf.wait_for_event(
            VisionEventType.E_STOP, since_seqno=-1, timeout=0.15,
        )
        assert ev is None
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_wait_for_event_ignores_lower_seqno(tmp_path):
    """A 'since_seqno' floor that's above existing events keeps the waiter parked."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        # Emit one event first; its seqno will be 1.
        buf.emit_event(make_e_stop("user", "manual abort",
                                   seqno=buf.next_seq()))
        # Wait with floor=100 → timeout because no event has higher seqno
        ev = await buf.wait_for_event(
            VisionEventType.E_STOP, since_seqno=100, timeout=0.15,
        )
        assert ev is None
    finally:
        await buf.stop()


# ── catchup (WAL replay) ---------------------------------------------------

@pytest.mark.asyncio
async def test_catchup_from_wal(tmp_path):
    """Mix put_tip_status (triggers WAL tip + maybe edge event) + emit_event;
    catchup(since=0) returns all events with seqno > 0 in chronological order."""
    wal_path = tmp_path / "buf.sqlite"
    buf = BufferService(wal_path=wal_path)
    await buf.start()
    try:
        # 3 explicit events
        buf.emit_event(make_vision_error("model_error", "load failed",
                                         seqno=buf.next_seq()))
        buf.emit_event(make_setpoint_change(bias_v=1.0, current_a=1e-9,
                                            source="planner",
                                            seqno=buf.next_seq()))
        buf.emit_event(make_e_stop("vacuum", "chamber pressure spike",
                                   seqno=buf.next_seq()))
        # Trigger an edge-emitted TIP_QUALITY_DROP from GOOD → BAD
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.GOOD, confidence=0.9,
                                      scan_id="s", frame_idx=0))
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.BAD, confidence=0.9,
                                      scan_id="s", frame_idx=1))

        # Let WAL writes flush.
        await asyncio.sleep(0.2)

        events = await buf.catchup(since_seqno=0)
        kinds = [e.kind for e in events]
        # Must include all 3 emitted + the edge-emitted drop (4 total)
        assert VisionEventType.VISION_ERROR in kinds
        assert VisionEventType.SETPOINT_CHANGE in kinds
        assert VisionEventType.E_STOP in kinds
        assert VisionEventType.TIP_QUALITY_DROP in kinds
        # Strict chronological order by seqno
        assert [e.seqno for e in events] == sorted(e.seqno for e in events)
        assert len(events) == 4
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_catchup_filter_by_kinds(tmp_path):
    """catchup(kinds=[X]) restricts to that subset."""
    wal_path = tmp_path / "buf.sqlite"
    buf = BufferService(wal_path=wal_path)
    await buf.start()
    try:
        buf.emit_event(make_vision_error("preprocess_fail", "bad shape",
                                         seqno=buf.next_seq()))
        buf.emit_event(make_setpoint_change(bias_v=0.5, current_a=None,
                                            source="monitor",
                                            seqno=buf.next_seq()))
        buf.emit_event(make_e_stop("force", "manual",
                                   seqno=buf.next_seq()))
        await asyncio.sleep(0.2)

        only_estop = await buf.catchup(
            since_seqno=0, kinds=[VisionEventType.E_STOP],
        )
        assert len(only_estop) == 1
        assert only_estop[0].kind == VisionEventType.E_STOP
        assert only_estop[0].payload["reason"] == "force"
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_catchup_since_seqno_floor(tmp_path):
    """catchup(since=N) returns only events with seqno > N."""
    wal_path = tmp_path / "buf.sqlite"
    buf = BufferService(wal_path=wal_path)
    await buf.start()
    cutoff: int = -1
    try:
        for i in range(5):
            buf.emit_event(make_setpoint_change(
                bias_v=float(i), current_a=None, source="planner",
                seqno=buf.next_seq(),
            ))
            if i == 2:
                cutoff = buf.get_stats()["events_published"]  # current seqno
        await asyncio.sleep(0.2)
        out = await buf.catchup(since_seqno=cutoff)
        # Only events 4 and 5 should be returned
        seqnos = [e.seqno for e in out]
        assert all(s > cutoff for s in seqnos)
        assert len(out) == 5 - cutoff
    finally:
        await buf.stop()


# ── get_event_history (in-memory ring) -------------------------------------

@pytest.mark.asyncio
async def test_get_event_history_ring_buffer(tmp_path):
    """history_size=100 → only the last 100 of 150 events kept."""
    buf = BufferService(
        wal_path=tmp_path / "buf.sqlite",
        wal_enabled=False,
        history_size=100,
    )
    await buf.start()
    try:
        for i in range(150):
            buf.emit_event(make_setpoint_change(
                bias_v=float(i), current_a=None, source="planner",
                seqno=buf.next_seq(),
            ))
        hist = buf.get_event_history(since_seqno=-1, limit=10000)
        assert len(hist) == 100
        # Last 100 emitted have seqnos 51..150 (1-indexed via next_seq)
        assert hist[0].seqno == 51
        assert hist[-1].seqno == 150
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_get_event_history_respects_limit_and_since(tmp_path):
    """limit truncates newest-first; since_seqno filters lower."""
    buf = BufferService(
        wal_path=tmp_path / "buf.sqlite",
        wal_enabled=False,
        history_size=100,
    )
    await buf.start()
    try:
        for i in range(20):
            buf.emit_event(make_e_stop("user", f"abort {i}",
                                       seqno=buf.next_seq()))
        # limit=5 → newest 5
        last5 = buf.get_event_history(since_seqno=-1, limit=5)
        assert len(last5) == 5
        assert [e.seqno for e in last5] == [16, 17, 18, 19, 20]
        # since_seqno=15 → 16..20
        after15 = buf.get_event_history(since_seqno=15, limit=100)
        assert [e.seqno for e in after15] == [16, 17, 18, 19, 20]
        # limit=0 → empty
        assert buf.get_event_history(limit=0) == []
    finally:
        await buf.stop()


# ── subscribe_all ----------------------------------------------------------

@pytest.mark.asyncio
async def test_subscribe_all_receives_all_kinds(tmp_path):
    """A subscribe_all() queue receives events regardless of kind."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        q_all = buf.subscribe_all()
        buf.emit_event(make_vision_error("context_loss", "driver reset",
                                         seqno=buf.next_seq()))
        buf.emit_event(make_setpoint_change(bias_v=3.0, current_a=None,
                                            source="user",
                                            seqno=buf.next_seq()))
        buf.emit_event(make_e_stop("temperature", "too hot",
                                   seqno=buf.next_seq()))

        kinds = []
        for _ in range(3):
            ev = await asyncio.wait_for(q_all.get(), timeout=1.0)
            kinds.append(ev.kind)
        assert VisionEventType.VISION_ERROR in kinds
        assert VisionEventType.SETPOINT_CHANGE in kinds
        assert VisionEventType.E_STOP in kinds
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_subscribe_all_plus_targeted_subscribe(tmp_path):
    """subscribe_all receives every event AND targeted subscribe still works."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        q_all = buf.subscribe_all()
        q_estop = buf.subscribe(VisionEventType.E_STOP)
        buf.emit_event(make_vision_error("cuda_oom", "x", seqno=buf.next_seq()))
        buf.emit_event(make_e_stop("force", "y", seqno=buf.next_seq()))

        all_evs = []
        for _ in range(2):
            all_evs.append(await asyncio.wait_for(q_all.get(), timeout=1.0))
        estop_ev = await asyncio.wait_for(q_estop.get(), timeout=1.0)
        assert len(all_evs) == 2
        assert estop_ev.kind == VisionEventType.E_STOP
        # The targeted queue should NOT have received the vision_error
        assert q_estop.empty()
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_unsubscribe_all_stops_delivery(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        q = buf.subscribe_all()
        buf.unsubscribe_all(q)
        buf.emit_event(make_setpoint_change(
            bias_v=1.0, current_a=None, source="planner",
            seqno=buf.next_seq(),
        ))
        # Give the loop a tick so any pending fanout would have fired.
        await asyncio.sleep(0.05)
        assert q.empty()
    finally:
        await buf.stop()


# ── get_stats --------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_stats_counts_publishes(tmp_path):
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        before = buf.get_stats()
        assert before["events_published"] == 0
        for _ in range(3):
            buf.emit_event(make_vision_error(
                "model_error", "x", seqno=buf.next_seq(),
            ))
        after = buf.get_stats()
        assert after["events_published"] == 3
        # subscribers_active starts at 0; subscribe bumps it.
        assert after["subscribers_active"] == 0
        _q = buf.subscribe(VisionEventType.E_STOP)
        assert buf.get_stats()["subscribers_active"] == 1
        buf.unsubscribe(VisionEventType.E_STOP, _q)
        assert buf.get_stats()["subscribers_active"] == 0
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_get_stats_counts_drops(tmp_path):
    """A queue size of 1 forces drop-oldest; counter should reflect it."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        # tiny queue → drops will happen on the 2nd event
        _q = buf.subscribe(VisionEventType.E_STOP, max_size=1)
        for i in range(5):
            buf.emit_event(make_e_stop("user", f"abort {i}",
                                       seqno=buf.next_seq()))
        # Yield the loop to let fanout complete in case it was scheduled.
        await asyncio.sleep(0.05)
        stats = buf.get_stats()
        # 5 published, queue size 1 → at least 4 oldest drops
        assert stats["events_published"] == 5
        assert stats["events_dropped_oldest"] >= 4
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_get_stats_counts_wal_writes(tmp_path):
    """WAL-enabled mode bumps wal_event_writes / wal_tip_writes counters."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite")  # wal_enabled default True
    await buf.start()
    try:
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(),
                                      quality=TipQuality.GOOD,
                                      confidence=0.9, scan_id="s",
                                      frame_idx=0))
        buf.emit_event(make_setpoint_change(
            bias_v=1.0, current_a=None, source="planner",
            seqno=buf.next_seq(),
        ))
        await asyncio.sleep(0.2)  # flush
        s = buf.get_stats()
        assert s["wal_tip_writes"] >= 1
        assert s["wal_event_writes"] >= 1
        assert s["wal_writes_failed"] == 0
    finally:
        await buf.stop()


# ─────────────────────────────────────────────────────────────────────
# stop()/start() must clear edge/history/latest state so a
# restarted instance never replays a previous session's edges or events.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_restart_clears_event_history_and_latest_id(tmp_path):
    """After stop()/start(), in-memory event history + rising-edge ids reset."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        for _ in range(3):
            buf.emit_event(make_e_stop("user", "abort", seqno=buf.next_seq()))
        assert len(buf.get_event_history(since_seqno=-1, limit=100)) == 3
    finally:
        await buf.stop()

    # History should be gone immediately after stop().
    assert buf.get_event_history(since_seqno=-1, limit=100) == []

    await buf.start()
    try:
        # Fresh session: no replay of the prior session's events.
        assert buf.get_event_history(since_seqno=-1, limit=100) == []
        # Rising-edge fast-path must not see the prior session's high id:
        # wait_for_event(since=-1) with no new emit must time out, NOT return a
        # stale E_STOP from before the restart.
        ev = await buf.wait_for_event(
            VisionEventType.E_STOP, since_seqno=-1, timeout=0.1,
        )
        assert ev is None
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_restart_resets_quality_edge_state(tmp_path):
    """A BAD quality at stop() must not suppress the next GOOD→BAD edge on restart."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        q = buf.subscribe(VisionEventType.TIP_QUALITY_DROP)
        # GOOD → BAD: one rising edge this session.
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(), quality=TipQuality.GOOD,
                                     confidence=0.9, scan_id="s", frame_idx=0))
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(), quality=TipQuality.BAD,
                                     confidence=0.9, scan_id="s", frame_idx=1))
        drained = []
        while True:
            try:
                drained.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        assert len(drained) == 1  # _last_emitted_quality now BAD
    finally:
        await buf.stop()

    # Restart: _last_emitted_quality must be reset to None, so an *immediate*
    # GOOD→BAD again produces a fresh rising edge (not swallowed as BAD→BAD).
    await buf.start()
    try:
        q2 = buf.subscribe(VisionEventType.TIP_QUALITY_DROP)
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(), quality=TipQuality.GOOD,
                                     confidence=0.9, scan_id="s", frame_idx=0))
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(), quality=TipQuality.BAD,
                                     confidence=0.9, scan_id="s", frame_idx=1))
        drained2 = []
        while True:
            try:
                drained2.append(q2.get_nowait())
            except asyncio.QueueEmpty:
                break
        assert len(drained2) == 1  # edge re-armed after restart
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_restart_clears_latest_slots(tmp_path):
    """get_latest_* must return (None, -1) after a stop()/start() cycle."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        buf.put_tip_status(TipStatus(seqno=buf.next_seq(), quality=TipQuality.GOOD,
                                     confidence=0.7, scan_id="s", frame_idx=0))
        buf.put_progress(ScanProgress(seqno=buf.next_seq(), scan_id="s",
                                      line_idx=5, lines_total=10, eta_s=1.0))
        latest_tip, _ = buf.get_latest_tip_status()
        assert latest_tip is not None
    finally:
        await buf.stop()

    await buf.start()
    try:
        assert buf.get_latest_tip_status() == (None, -1)
        assert buf.get_latest_progress() == (None, -1)
        assert buf.get_latest_region() == (None, -1)
    finally:
        await buf.stop()


@pytest.mark.asyncio
async def test_restart_preserves_seqno_with_wal(tmp_path):
    """_reset_volatile_state must NOT clobber the WAL-restored monotonic seqno."""
    wal_path = tmp_path / "buf.sqlite"
    buf = BufferService(wal_path=wal_path)  # WAL enabled
    await buf.start()
    try:
        for _ in range(4):
            seq = buf.next_seq()
            buf.put_tip_status(TipStatus(seqno=seq, quality=TipQuality.GOOD,
                                         confidence=0.5, scan_id="s", frame_idx=0))
        await asyncio.sleep(0.2)  # flush WAL
    finally:
        await buf.stop()

    # New instance from same WAL: seqno must resume at 5 even though
    # start() runs _reset_volatile_state() (which must leave _seqno alone).
    buf2 = BufferService(wal_path=wal_path)
    await buf2.start()
    try:
        assert buf2.next_seq() == 5
    finally:
        await buf2.stop()


# ─────────────────────────────────────────────────────────────────────
# _latest_event_seqno guarded by _event_history_lock.
# Cross-thread stress: producer thread emits while loop thread polls
# wait_for_event; no torn (id, history) pair should surface.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_latest_event_seqno_concurrent_emit_and_wait(tmp_path):
    """Vision-thread emits while loop-thread waits; every wait must resolve to a
    real event with seqno > floor (never a phantom from a torn fast-path read)."""
    buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
    await buf.start()
    try:
        n = 300

        def producer():
            for _ in range(n):
                buf.emit_event(make_e_stop("user", "x", seqno=buf.next_seq()))

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        floor = -1
        # Drain rising edges. Each call returns the newest matching event, so
        # `floor` can jump forward by many seqnos at once — loop on floor, not a
        # per-iteration counter. The invariant under test is that the fast-path
        # id table (read under the shared lock) never promises a match the
        # history scan can't deliver: every non-None return must satisfy
        # seqno > floor.
        while floor < n:
            ev = await buf.wait_for_event(
                VisionEventType.E_STOP, since_seqno=floor, timeout=2.0,
            )
            if ev is None:
                break
            assert ev.kind == VisionEventType.E_STOP
            assert ev.seqno > floor  # torn read would violate this
            floor = ev.seqno
        t.join(timeout=2)
        # We must have advanced the floor to the highest emitted seqno.
        assert buf.get_stats()["events_published"] == n
        assert floor == n  # consumed all the way to the last emit
    finally:
        await buf.stop()


def test_latest_slot_clear():
    """_LatestSlot.clear() returns the slot to the empty (None, -1) state."""
    slot: _LatestSlot[int] = _LatestSlot(threading.Lock())
    slot.put(99, 7)
    assert slot.get() == (99, 7)
    slot.clear()
    assert slot.get() == (None, -1)


# ─────────────────────────────────────────────────────────────────────
# 修复项 (2026-06-11): synchronous critical hooks — the physical E_STOP path.
# Must fire inline on the publisher thread, even with NO asyncio loop
# (history-only degraded mode), so a threading.Event abort works always.
# ─────────────────────────────────────────────────────────────────────

class TestCriticalHooks:
    def _buf(self, tmp_path):
        return BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)

    def test_hook_fires_synchronously_without_loop(self, tmp_path):
        """No start(), no loop: emit_event must still invoke the hook inline."""
        buf = self._buf(tmp_path)
        abort = threading.Event()

        def hook(ev):
            if ev.kind is VisionEventType.E_STOP:
                abort.set()

        buf.register_critical_hook(hook)
        buf.emit_event(make_e_stop("watchdog", "test", seqno=1))
        assert abort.is_set()  # synchronous — no sleep/poll needed

    def test_hook_ignores_other_kinds(self, tmp_path):
        buf = self._buf(tmp_path)
        abort = threading.Event()
        buf.register_critical_hook(
            lambda ev: abort.set() if ev.kind is VisionEventType.E_STOP else None)
        buf.emit_event(make_vision_error("model_error", "x", seqno=1))
        assert not abort.is_set()

    def test_failing_hook_does_not_break_emit_or_other_hooks(self, tmp_path):
        buf = self._buf(tmp_path)
        seen = []

        def bad(ev):
            raise RuntimeError("boom")

        buf.register_critical_hook(bad)
        buf.register_critical_hook(lambda ev: seen.append(ev.kind))
        buf.emit_event(make_e_stop("user", "test", seqno=2))
        assert seen == [VisionEventType.E_STOP]
        assert buf.get_stats()["events_published"] == 1

    def test_register_idempotent_and_unregister(self, tmp_path):
        buf = self._buf(tmp_path)
        hits = []

        def hook(ev):
            hits.append(1)

        buf.register_critical_hook(hook)
        buf.register_critical_hook(hook)  # duplicate — must not double-fire
        buf.emit_event(make_e_stop("user", "a", seqno=3))
        assert len(hits) == 1
        buf.unregister_critical_hook(hook)
        buf.emit_event(make_e_stop("user", "b", seqno=4))
        assert len(hits) == 1


class TestAbortEventSharing:
    """修复项 contract: ExecutionContext.check_abort sees a SHARED Event."""

    def test_shared_event_reaches_check_abort(self):
        from mast.core.execution_context import ExecutionContext
        from mast.core.registry import SkillRegistry
        shared = threading.Event()
        ctx = ExecutionContext(pool=None, state=None,
                               registry=SkillRegistry(), abort_event=shared)
        assert ctx.check_abort() is False
        shared.set()  # what the E_STOP hook / GUI abort does
        assert ctx.check_abort() is True


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
