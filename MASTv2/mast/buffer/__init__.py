"""mast.buffer — s-scale handshake layer between VisionModule and agents.

Phase 2 fills:
  schemas.py  TipStatus / RegionMap / ScanProgress / VisionEvent (Pydantic frozen)
  service.py  BufferService: _LatestSlot + edge-triggered events + aiosqlite WAL

In-process asyncio, not Redis (loopback latency eats vision budget) and not
LangGraph Store (no change-feed). See docs/v2/architecture/buffer.md and
reference/compass_artifact_wf-cc956796 §3.
"""
