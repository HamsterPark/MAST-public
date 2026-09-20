"""Streaming (realtime) voice providers — WebSocket ASR + TTS.

Separate from the batch ``mast.voice.dashscope`` client (kept for graceful
degradation). These classes hold ONE DashScope realtime WebSocket each, so the
two-connection design is a natural "separated streaming ASR + TTS":

  * ``DashScopeRealtimeASR``  — speech (16 kHz PCM16 in) → partial/final text,
        with server-side VAD sentence endpointing.
  * ``DashScopeRealtimeTTS``  — text chunks in → 24 kHz PCM16 audio frames out,
        for "synthesize while the LLM is still generating".

Both talk the OpenAI-Realtime-style dialect DashScope exposes at
``wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=...`` (Bearer auth in the
handshake header, base64 audio in JSON frames). No ``dashscope`` SDK dependency —
handwritten over ``websockets`` (already a transitive dep via uvicorn[standard]).
"""

from __future__ import annotations

from mast.voice.realtime.dashscope_rt import (
    DashScopeRealtimeASR,
    DashScopeRealtimeTTS,
    RealtimeVoiceError,
)

__all__ = [
    "DashScopeRealtimeASR",
    "DashScopeRealtimeTTS",
    "RealtimeVoiceError",
]
