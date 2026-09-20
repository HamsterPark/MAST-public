"""Buffer-layer LLM summarizer (Phase 5 follow-up — design 2026-05-14).

Bridges ms-scale Vision module → s-scale Buffer → min-scale IC agent.

VisionModule produces structured Pydantic results (TipCoarseResult,
SegmentationResult, etc.) which are precise but tedious for an LLM to
quote verbatim in chat replies. The buffer_summarizer node consumes raw
vision outputs, asks a lightweight LLM (Kimi K2.6, but K2.5 / DeepSeek
Flash are valid drop-ins) to produce a 1-2 sentence Chinese summary,
and writes the summarised text into the BufferService so the IC agent's
``read_latest_tip_status`` / ``get_scan_progress`` tools return both
the raw schema AND a natural-language summary the LLM can quote.

Architecture:

    ┌────────────────────┐
    │  VisionModule      │  ms scale
    │  (DINOv3/legacy)   │
    └────────┬───────────┘
             │ TipCoarseResult / SegmentationResult / ...
             ▼
    ┌────────────────────┐
    │ buffer_summarizer  │  s scale — 1 LLM call per vision event
    │  (this package)    │
    └────────┬───────────┘
             │ TipStatus.summary_text: "针尖良好，置信度 0.94"
             ▼
    ┌────────────────────┐
    │  BufferService     │
    └────────┬───────────┘
             │ read_latest_tip_status() tool
             ▼
    ┌────────────────────┐
    │  IC agent (LLM)    │  min scale
    └────────────────────┘

Stub for now; implementation lands when DINOv3 is trained and
VisionModule emits real events. See `node.py` for the LangChain
runnable + `prompt.py` for the system prompt.
"""

from mast.agents.buffer_summarizer.node import (
    BufferSummarizerNode,
    summarize_tip_status,
    summarize_segmentation,
)

__all__ = [
    "BufferSummarizerNode",
    "summarize_tip_status",
    "summarize_segmentation",
]
