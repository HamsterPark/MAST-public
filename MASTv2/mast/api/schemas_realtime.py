"""Realtime channel schemas — the typed envelopes for WS/SSE/polling.

Per RFC §3.7: the realtime channels are DECOUPLED from agent execution (the
buffer/event bus pushes from the vision thread / event loop, independent of any
agent run) and the frontend assumes they are ALWAYS available. Tensors/ndarrays
NEVER cross these frames — only scalars, ids, and (elsewhere) b64 thumbnails.

The buffer payload models (TipStatus/ScanProgress/VisionEvent) are reused
verbatim from mast.buffer.schemas so there is one definition shared by producer,
core, and API.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from mast.buffer.schemas import ScanProgress, TipStatus, VisionEvent


class RegionRef(BaseModel):
    """A RegionMap reference WITHOUT the RLE mask bytes (those are fetched
    separately via the vision endpoint). Keeps realtime frames small + JSON-safe."""

    seqno: int
    scan_id: str
    shape: tuple[int, int]


class BufferSnapshot(BaseModel):
    """Point-in-time snapshot of the buffer streams — sent as the first WS frame
    and served by the polling-fallback REST endpoint."""

    tip: Optional[TipStatus] = None
    tip_seqno: int = -1
    progress: Optional[ScanProgress] = None
    progress_seqno: int = -1
    region: Optional[RegionRef] = None
    recent_events: list[VisionEvent] = Field(default_factory=list)
    degraded: bool = False


class WsFrame(BaseModel):
    """Discriminated WS frame envelope. ``kind`` distinguishes payloads so the
    typed frontend can switch on it."""

    kind: Literal["snapshot", "event", "ping", "hardware_state"]
    snapshot: Optional[BufferSnapshot] = None
    event: Optional[VisionEvent] = None
    # hardware_state carries a flat scalar dict (bias/current/z/setpoint/...);
    # kept as an open dict because the live readings set varies by instrument.
    hardware: Optional[dict] = None


__all__ = ["RegionRef", "BufferSnapshot", "WsFrame"]
