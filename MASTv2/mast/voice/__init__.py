"""Voice interaction layer: ASR (speech → text) + TTS (text → speech).

Default provider is Aliyun DashScope (Qwen3-TTS-flash + Qwen3-ASR-flash).
The provider abstraction allows swapping in Whisper / ElevenLabs / OpenAI
without touching GUI code.
"""

from __future__ import annotations

from mast.voice.dashscope import (
    DashScopeVoiceClient,
    VoiceProviderError,
    TTS_VOICES,
)

__all__ = [
    "DashScopeVoiceClient",
    "VoiceProviderError",
    "TTS_VOICES",
]
