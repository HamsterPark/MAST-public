"""Realtime channels — WebSocket (primary) + SSE (fallback) + polling (last resort).

Per RFC §2.8 / §3.7. Decoupled from agent execution: the buffer/event streams
push from the vision thread independent of any agent run, so the frontend can
assume they are always live. Degrades safely when no BufferService is wired
(standalone dev): snapshots come back empty (degraded=True) and the WS sends
periodic pings until the client disconnects.

Routes here use FULL paths (/ws/*, /sse/*, /api/buffer/*) and are included on the
app WITHOUT a prefix (see app.py).
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from mast.api.schemas_realtime import BufferSnapshot, RegionRef, WsFrame

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

_PING_TIMEOUT_S = 15.0


def _build_snapshot(ctx) -> BufferSnapshot:
    """Read a point-in-time snapshot from the live BufferService, or a degraded
    empty snapshot when none is wired. Never raises."""
    buffer = getattr(ctx, "buffer_service", None)
    if buffer is None:
        return BufferSnapshot(degraded=True)
    try:
        tip, tip_seq = buffer.get_latest_tip_status()
        prog, prog_seq = buffer.get_latest_progress()
        region, _region_seq = buffer.get_latest_region()
        region_ref = (
            RegionRef(seqno=region.seqno, scan_id=region.scan_id, shape=region.shape)
            if region is not None
            else None
        )
        try:
            recent = buffer.get_event_history(limit=20)
        except Exception:
            recent = []
        return BufferSnapshot(
            tip=tip,
            tip_seqno=tip_seq,
            progress=prog,
            progress_seqno=prog_seq,
            region=region_ref,
            recent_events=list(recent or []),
            degraded=False,
        )
    except Exception as exc:  # any buffer hiccup → degrade, never break the stream
        logger.warning("buffer snapshot failed: %s", exc)
        return BufferSnapshot(degraded=True)


# ── polling fallback (REST) ────────────────────────────────────────────
@router.get("/api/buffer/snapshot", response_model=BufferSnapshot)
def buffer_snapshot(request: Request) -> BufferSnapshot:
    return _build_snapshot(request.app.state.ctx)


# ── WebSocket: vision/scan buffer stream ───────────────────────────────
@router.websocket("/ws/buffer")
async def ws_buffer(ws: WebSocket) -> None:
    await ws.accept()
    ctx = ws.app.state.ctx
    buffer = getattr(ctx, "buffer_service", None)

    # First frame is always a snapshot so a fresh client renders immediately.
    await ws.send_json(WsFrame(kind="snapshot", snapshot=_build_snapshot(ctx)).model_dump(mode="json"))

    queue = None
    try:
        if buffer is not None:
            try:
                queue = buffer.subscribe_all(max_size=100)
            except Exception as exc:
                logger.warning("subscribe_all failed: %s", exc)
                queue = None

        while True:
            if queue is not None:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=_PING_TIMEOUT_S)
                    await ws.send_json(
                        WsFrame(kind="event", event=ev).model_dump(mode="json")
                    )
                except asyncio.TimeoutError:
                    await ws.send_json(WsFrame(kind="ping").model_dump(mode="json"))
            else:
                # Degraded: no producer. Wait on the client (so a disconnect is
                # noticed immediately) and heartbeat on idle.
                try:
                    await asyncio.wait_for(ws.receive_text(), timeout=_PING_TIMEOUT_S)
                except asyncio.TimeoutError:
                    await ws.send_json(WsFrame(kind="ping").model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ws_buffer loop ended: %s", exc)
    finally:
        if queue is not None and buffer is not None:
            try:
                buffer.unsubscribe_all(queue)
            except Exception:
                pass


# ── WebSocket: hardware event bus ──────────────────────────────────────
#
# How many undelivered events a single slow client may hold before we start
# dropping the OLDEST. Bounded on purpose: an unbounded queue turns one stalled
# browser tab into a server-side memory leak. When it overflows the client is
# TOLD (``dropped``) rather than silently handed a hole — it can then re-sync
# with ``?since=`` instead of believing it saw everything.
_WS_QUEUE_MAX = 500


def _event_frame(seq: int, event) -> dict:
    """One wire frame for a bus event. Flat on purpose: the client switches on
    ``type`` and reads ``data``; ``seq`` is the reconnect cursor."""
    payload = event.to_json_dict()
    return {"kind": "event", "seq": int(seq),
            "type": payload.get("type"), "data": payload.get("data") or {},
            "ts": payload.get("ts")}


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    """Live push of :class:`mast.core.events.EventBus` events.

    Was a stub that only pinged — a "contract-correct socket" nobody could get
    data out of, so every consumer polled instead. With Tailscale as the
    transport there is no reason for that: a WebSocket rides the same WireGuard
    tunnel and costs one connection instead of N pollers.

    Contract (the frontend client is written against exactly this):

    * ``GET /ws/events?since=<seq>`` — on connect, everything in the bus history
      with ``seq > since`` is replayed FIRST, in order, before any live event.
      ``since=0`` (or absent) replays whatever the ring buffer still holds.
      This is what makes a reconnect lossless rather than a fresh start.
    * frames are ``{"kind":"event","seq":N,"type":…,"data":{…},"ts":…}``
      or ``{"kind":"ping"}``. Unknown ``type`` values must be ignored by
      clients — new event kinds are added over time.
    * ``{"kind":"dropped","from":a,"to":b}`` says the client was too slow and
      lost that range; it should re-sync from ``since=b``.

    Degradation: the HTTP polling path (``EventBus.recent_events_since``) stays
    exactly as it was. A blocked Upgrade, a dead socket, or a client that never
    connects all fall back to it — this endpoint adds a faster road, it does not
    remove the old one.
    """
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=_WS_QUEUE_MAX)
    dropped: list[int] = []          # seqs lost to backpressure
    handle = None
    bus = None

    try:
        from mast.core.events import EventBus

        bus = EventBus.get()
    except Exception as exc:  # noqa: BLE001 — degrade to a ping-only socket
        logger.warning("ws_events: EventBus unavailable (%s); ping-only", exc)

    try:
        since = int(ws.query_params.get("since", "0") or 0)
    except (TypeError, ValueError):
        since = 0

    def _on_event(seq: int, event) -> None:
        # Runs on the PUBLISHING thread (state.refresh's 1 Hz poller, a skill
        # thread, …) — never touch the socket here. Hand the frame to the loop.
        def _put() -> None:
            try:
                queue.put_nowait((seq, event))
            except asyncio.QueueFull:
                try:
                    old_seq, _ = queue.get_nowait()
                    dropped.append(int(old_seq))
                    queue.put_nowait((seq, event))
                except Exception:  # noqa: BLE001
                    dropped.append(int(seq))
        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass                      # loop already closing

    try:
        if bus is not None:
            # Replay BEFORE subscribing would lose anything published in
            # between; subscribing first can duplicate. Duplicates are the
            # survivable direction — the client dedupes on `seq` — so we
            # subscribe first and replay after.
            handle = bus.subscribe_with_id(_on_event)
            missed, _latest = bus.recent_events_since(since)
            for item in missed:
                await ws.send_json({
                    "kind": "event", "seq": int(item.get("id", 0)),
                    "type": item.get("type"), "data": item.get("data") or {},
                    "ts": item.get("ts")})

        while True:
            try:
                seq, event = await asyncio.wait_for(queue.get(),
                                                    timeout=_PING_TIMEOUT_S)
            except asyncio.TimeoutError:
                await ws.send_json(WsFrame(kind="ping").model_dump(mode="json"))
                continue
            if dropped:
                lost, dropped[:] = list(dropped), []
                await ws.send_json({"kind": "dropped",
                                    "from": min(lost), "to": max(lost)})
            await ws.send_json(_event_frame(seq, event))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("ws_events loop ended: %s", exc)
    finally:
        if bus is not None and handle is not None:
            try:
                bus.unsubscribe(handle)
            except Exception:  # noqa: BLE001
                pass


# ── SSE fallback for the buffer stream ─────────────────────────────────
@router.get("/sse/buffer")
async def sse_buffer(request: Request) -> StreamingResponse:
    ctx = request.app.state.ctx
    buffer = getattr(ctx, "buffer_service", None)

    async def gen():
        snap = WsFrame(kind="snapshot", snapshot=_build_snapshot(ctx))
        yield f"data: {json.dumps(snap.model_dump(mode='json'))}\n\n"
        if buffer is None:
            # Degraded: no producer to stream. End the response cleanly so the
            # client falls back to polling /api/buffer/snapshot (or reconnects);
            # an infinite empty stream would only pin a connection for nothing.
            return
        try:
            queue = buffer.subscribe_all(max_size=100)
        except Exception:
            return
        try:
            while not await request.is_disconnected():
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=_PING_TIMEOUT_S)
                    frame = WsFrame(kind="event", event=ev)
                    yield f"data: {json.dumps(frame.model_dump(mode='json'))}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            try:
                buffer.unsubscribe_all(queue)
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


__all__ = ["router"]
