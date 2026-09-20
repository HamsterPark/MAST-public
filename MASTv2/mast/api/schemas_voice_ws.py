"""Typed frames for the ``/ws/voice`` full-duplex voice channel.

The socket carries TWO frame kinds:

* **binary frames** — raw little-endian PCM16 mono audio.
    · up   (client → server): microphone @ 16 kHz
    · down (server → client): TTS @ 24 kHz
* **JSON control frames** — everything else, shapes defined below.

These models are the SINGLE SOURCE OF TYPES for the channel (mirrored by the TS
``voiceSessionStore`` on the frontend). Down-frames are Pydantic (built by the
route → ``model_dump``); up-frames are tolerant-parsed (``parse_client_frame``)
so a slightly-off client never 500s the socket.

House rules honored: the channel DEGRADES (a ``degraded`` frame) rather than
erroring when the voice subsystem / key is absent; nothing here blocks.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ── session states (down: {"type":"state", "state": <one of these>}) ──────
VoiceState = Literal[
    "idle",          # channel open, not listening
    "listening",     # capturing mic, waiting for speech / endpoint
    "transcribing",  # got audio, ASR finalizing
    "thinking",      # agent turn running (pre-first-token)
    "speaking",      # streaming TTS playback
    "degraded",      # realtime unavailable — see the degraded frame
]

# ── interaction modes ─────────────────────────────────────────────────────
VoiceMode = Literal["ptt", "wake", "duplex"]


# ═══════════════════════════ down-frames (server → client) ═════════════════
class HelloAck(BaseModel):
    type: Literal["hello_ack"] = "hello_ack"
    mode: VoiceMode = "ptt"
    voice: str = "Cherry"
    streaming: bool = True          # False → running the batch fallback
    narrate: bool = True
    degraded: bool = False
    message: str = ""


class StateFrame(BaseModel):
    type: Literal["state"] = "state"
    state: VoiceState = "idle"


class AsrPartial(BaseModel):
    type: Literal["asr_partial"] = "asr_partial"
    text: str = ""
    emotion: str = ""


class AsrFinal(BaseModel):
    type: Literal["asr_final"] = "asr_final"
    text: str = ""


class ReplyDelta(BaseModel):
    type: Literal["reply_delta"] = "reply_delta"
    text: str = ""


class Narration(BaseModel):
    """Spoken side-channel about agent execution (e.g. 「扫描完成」)."""
    type: Literal["narration"] = "narration"
    text: str = ""


class TtsBegin(BaseModel):
    """Precedes a run of binary audio frames — tells the client the PCM rate."""
    type: Literal["tts_begin"] = "tts_begin"
    sample_rate: int = 24000


class TtsDone(BaseModel):
    type: Literal["tts_done"] = "tts_done"


class WakeFrame(BaseModel):
    """Wake-word mode armed/disarmed (down). ``armed`` → the next utterance is
    taken as a command; disarmed → back to standby waiting for the wake word."""
    type: Literal["wake"] = "wake"
    armed: bool = False


class InterruptFrame(BaseModel):
    """A HITL / DANGEROUS approval is pending — the client opens the existing
    approval modal. Voice NEVER auto-approves; it only announces + requests."""
    type: Literal["interrupt"] = "interrupt"
    interrupt_id: str = ""
    skill: str = ""
    rationale: str = ""
    spoken: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ErrorFrame(BaseModel):
    type: Literal["error"] = "error"
    message: str = ""


class DegradedFrame(BaseModel):
    type: Literal["degraded"] = "degraded"
    message: str = ""
    reason: str = ""   # e.g. "no_key" | "realtime_denied" | "no_engine"


# ═══════════════════════════ up-frames (client → server) ═══════════════════
class HelloFrame(BaseModel):
    type: Literal["hello"] = "hello"
    mode: VoiceMode = "ptt"
    conversation_id: Optional[str] = None
    voice: Optional[str] = None
    narrate: bool = True
    language: Optional[str] = None


class CommitFrame(BaseModel):
    type: Literal["commit"] = "commit"


class BargeInFrame(BaseModel):
    type: Literal["barge_in"] = "barge_in"


class TextFrame(BaseModel):
    type: Literal["text"] = "text"
    content: str = ""


class SetModeFrame(BaseModel):
    type: Literal["set_mode"] = "set_mode"
    mode: VoiceMode = "ptt"


class StopFrame(BaseModel):
    type: Literal["stop"] = "stop"


class ByeFrame(BaseModel):
    type: Literal["bye"] = "bye"


_CLIENT_FRAMES = {
    "hello": HelloFrame,
    "commit": CommitFrame,
    "barge_in": BargeInFrame,
    "text": TextFrame,
    "set_mode": SetModeFrame,
    "stop": StopFrame,
    "bye": ByeFrame,
}


def parse_client_frame(obj: Any) -> Optional[BaseModel]:
    """Tolerant-parse a decoded JSON control frame into its model, or None if it
    is unrecognized / malformed (the caller simply ignores None)."""
    if not isinstance(obj, dict):
        return None
    model = _CLIENT_FRAMES.get(str(obj.get("type") or ""))
    if model is None:
        return None
    try:
        return model.model_validate(obj)
    except Exception:
        return None


__all__ = [
    "VoiceState",
    "VoiceMode",
    "HelloAck",
    "StateFrame",
    "AsrPartial",
    "AsrFinal",
    "ReplyDelta",
    "Narration",
    "TtsBegin",
    "TtsDone",
    "WakeFrame",
    "InterruptFrame",
    "ErrorFrame",
    "DegradedFrame",
    "HelloFrame",
    "CommitFrame",
    "BargeInFrame",
    "TextFrame",
    "SetModeFrame",
    "StopFrame",
    "ByeFrame",
    "parse_client_frame",
]
