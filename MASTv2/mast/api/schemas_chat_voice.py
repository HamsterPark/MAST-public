"""Pydantic request/response models for the chat_voice slice — voice ASR/TTS and
curated chat quick-prompts.

These shapes mirror the kept backends faithfully:

* ASR / TTS follow ``mast.voice.DashScopeVoiceClient`` (qwen3-asr-flash returns
  ``{"text", "emotion", "language", "request_id"}``; qwen3-tts-flash returns
  24 kHz mono WAV bytes which we relay base64-encoded so the SPA can feed an
  ``<audio>`` element without a binary fetch). The allowed voice list is
  ``mast.voice.TTS_VOICES``.
* quick-prompts mirror ``ConfigOverrideRegistry.get_quick_prompts()`` (the
  ``quick_prompts.json`` override ``{"prompts": [...]}``) merged over a small
  curated built-in fallback so the chat composer always has starter prompts.

Per the house rules these are the SINGLE SOURCE OF TYPES for this slice. Every
endpoint has a ``response_model``; results degrade safely (``degraded=True``)
when the live voice subsystem / override store is absent — never a 500.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── voice: ASR (speech → text) ─────────────────────────────────────────


class TranscribeRequest(BaseModel):
    """ASR request. Audio is supplied either as a base64 blob (+ container
    ``mime``) or as an ``https://…`` URL — mirrors
    ``DashScopeVoiceClient.transcribe`` accepting bytes / path / URL."""

    audio_b64: Optional[str] = Field(
        default=None, description="base64-encoded audio bytes"
    )
    audio_url: Optional[str] = Field(
        default=None, description="https URL to the audio (alternative to audio_b64)"
    )
    mime: str = Field(default="wav", description="audio container: wav/mp3/ogg/flac")


class TranscribeResult(BaseModel):
    """ASR result — the qwen3-asr-flash payload shape."""

    ok: bool = False
    text: str = ""
    emotion: str = ""
    language: str = ""
    request_id: str = ""
    error: Optional[str] = None
    degraded: bool = False


# ── voice: TTS (text → speech) ─────────────────────────────────────────


class SynthesizeRequest(BaseModel):
    """TTS request. ``voice`` defaults to the client's configured default
    (Cherry) when omitted — mirrors ``DashScopeVoiceClient.synthesize``."""

    text: str
    voice: Optional[str] = None


class SynthesizeResult(BaseModel):
    """TTS result. The WAV bytes are relayed base64-encoded (audio_b64) so the
    SPA can build a ``data:`` URL directly; no binary endpoint needed."""

    ok: bool = False
    audio_b64: str = ""
    mime: str = "audio/wav"
    voice: str = ""
    bytes: int = 0
    error: Optional[str] = None
    degraded: bool = False


class VoicesResponse(BaseModel):
    """Available TTS voices (mast.voice.TTS_VOICES) + the configured default +
    whether the voice subsystem is actually reachable (has a key)."""

    voices: list[str] = Field(default_factory=list)
    default_voice: str = ""
    enabled: bool = False
    degraded: bool = False


# ── chat: quick-prompts ────────────────────────────────────────────────


class QuickPrompt(BaseModel):
    """One quick-prompt chip for the chat composer. ``source`` marks whether it
    came from the curated built-in set or the operator override file."""

    label: str = ""
    prompt: str = ""
    source: str = "curated"  # "curated" | "override"
    meta: dict[str, Any] = Field(default_factory=dict)


class QuickPromptsResponse(BaseModel):
    """Curated quick-prompts merged with operator overrides
    (ConfigOverrideRegistry.get_quick_prompts)."""

    prompts: list[QuickPrompt] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


__all__ = [
    "TranscribeRequest",
    "TranscribeResult",
    "SynthesizeRequest",
    "SynthesizeResult",
    "VoicesResponse",
    "QuickPrompt",
    "QuickPromptsResponse",
]
