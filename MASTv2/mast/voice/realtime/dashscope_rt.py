"""DashScope realtime voice — streaming ASR + streaming TTS over WebSocket.

Endpoint (both): ``wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=<model>``
Auth: ``Authorization: Bearer <DASHSCOPE_API_KEY>`` in the WS handshake header.

Protocol is OpenAI-Realtime-*style* (JSON control frames, base64 audio):

ASR (``qwen3-asr-flash-realtime``)
  → ``session.update`` {input_audio_format, sample_rate, input_audio_transcription,
     turn_detection: server_vad}
  → ``input_audio_buffer.append`` {audio: base64 pcm16/16k}   (~100 ms slices)
  ← ``input_audio_buffer.speech_started`` / ``.speech_stopped``   (VAD endpoints)
  ← ``conversation.item.input_audio_transcription.text``   {text}   (partial)
  ← ``conversation.item.input_audio_transcription.completed`` {transcript} (final)

TTS (``qwen3-tts-flash-realtime``)
  → ``session.update`` {voice, response_format: PCM_24000HZ_MONO_16BIT, mode:server_commit}
  → ``input_text_buffer.append`` {text}                        (LLM tokens, by sentence)
  → ``session.finish``
  ← ``response.audio.delta`` {delta: base64 pcm16/24k}          (incremental audio)
  ← ``response.done`` / ``session.finished``

NOTE (待验证): several field names/enums come from the 2026 DashScope docs and are
tolerant-parsed here (multiple candidate keys accepted). The ``open()`` reader
logs every RAW server frame at DEBUG so the P0 smoke can reveal the real shapes
and this module can be tightened. Nothing here is imported at process start beyond
stdlib + ``websockets``; ``glossary`` is lazy-imported so a missing knowledge pack
never breaks voice.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)

_REALTIME_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

#: Voices verified against qwen3-tts-flash(-realtime) — shared with the batch client.
TTS_VOICES: tuple[str, ...] = ("Cherry", "Ethan", "Chelsie", "Serena", "Dylan")


class RealtimeVoiceError(RuntimeError):
    """Raised on handshake failure or a fatal server error frame."""


async def _connect(url: str, api_key: str, *, timeout: float = 20.0):
    """Open a websockets client to *url* with Bearer auth. Import websockets lazily
    so the module imports even where the client lib is somehow absent.

    ``timeout`` is the handshake budget — generous by default because a corporate
    / campus VPN can make the TLS+WS handshake to Aliyun take double-digit seconds
    (measured ~17 s on the MAST dev box). Callers on a fast link can lower it."""
    try:
        import websockets
    except Exception as exc:  # pragma: no cover - websockets is a uvicorn[standard] dep
        raise RealtimeVoiceError(f"websockets unavailable: {exc}") from exc
    try:
        # websockets>=14: top-level connect is the asyncio client; headers via
        # additional_headers. max_size=None so large base64 audio frames are fine.
        return await asyncio.wait_for(
            websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                max_size=None,
                open_timeout=timeout,
            ),
            timeout=timeout + 5.0,
        )
    except RealtimeVoiceError:
        raise
    except Exception as exc:
        raise RealtimeVoiceError(f"realtime connect failed ({url}): {exc}") from exc


def _first(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return default


# ═══════════════════════════════════════════════════════════════════════════
#  ASR — speech → text
# ═══════════════════════════════════════════════════════════════════════════
class DashScopeRealtimeASR:
    """One realtime ASR connection. Push 16 kHz PCM16 with :meth:`send_audio`;
    consume normalized events from :meth:`events` (single reader coroutine)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "qwen3-asr-flash-realtime",
        language: Optional[str] = None,   # None → auto-detect (zh/en mixed ok)
        use_glossary: bool = True,
    ) -> None:
        if not api_key:
            raise RealtimeVoiceError("empty DashScope API key.")
        self._api_key = api_key
        self._model = model
        self._language = language
        self._use_glossary = use_glossary
        self._ws: Any = None
        self._closed = False

    async def open(
        self,
        *,
        vad: bool = True,
        vad_threshold: float = 0.2,
        vad_silence_ms: int = 800,
        connect_timeout: float = 20.0,
    ) -> None:
        url = f"{_REALTIME_ENDPOINT}?model={self._model}"
        self._ws = await _connect(url, self._api_key, timeout=connect_timeout)
        transcription: dict[str, Any] = {}
        if self._language:
            transcription["language"] = self._language
        session: dict[str, Any] = {
            "modalities": ["text"],
            "input_audio_format": "pcm",       # 16-bit little-endian
            "sample_rate": 16000,
            "input_audio_transcription": transcription,
            "turn_detection": (
                {
                    "type": "server_vad",
                    "threshold": vad_threshold,
                    "silence_duration_ms": vad_silence_ms,
                }
                if vad
                else None
            ),
        }
        await self._ws.send(json.dumps({"type": "session.update", "session": session}))
        logger.info("ASR realtime session opened (model=%s, vad=%s)", self._model, vad)

    async def send_audio(self, pcm16_16k: bytes) -> None:
        """Append one ~100 ms slice of raw 16 kHz PCM16 mono audio."""
        if self._closed or self._ws is None or not pcm16_16k:
            return
        b64 = base64.b64encode(pcm16_16k).decode("ascii")
        try:
            await self._ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": b64})
            )
        except Exception as exc:
            logger.debug("ASR send_audio failed: %s", exc)

    async def commit(self) -> None:
        """Manual endpoint (PTT release / no-VAD mode): commit the buffered audio."""
        if self._closed or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as exc:
            logger.debug("ASR commit failed: %s", exc)

    async def events(self) -> AsyncIterator[dict]:
        """Yield normalized events until the socket closes:

          {"type": "speech_started"}
          {"type": "speech_stopped"}
          {"type": "partial", "text": str, "emotion": str}
          {"type": "final",   "text": str, "emotion": str}
          {"type": "error",   "message": str}
        """
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                msg = _parse(raw)
                if msg is None:
                    continue
                logger.debug("ASR ← %s", _preview(msg))
                out = self._normalize(msg)
                if out is not None:
                    yield out
        except Exception as exc:
            if not self._closed:
                logger.debug("ASR events loop ended: %s", exc)
        finally:
            self._closed = True

    def _normalize(self, msg: dict) -> Optional[dict]:
        t = str(msg.get("type") or "")
        if t.endswith("speech_started"):
            return {"type": "speech_started"}
        if t.endswith("speech_stopped"):
            return {"type": "speech_stopped"}
        emotion = _first(msg, "emotion")
        if t.endswith("input_audio_transcription.completed") or t.endswith(".done"):
            text = _first(msg, "transcript", "text", "stash")
            return {"type": "final", "text": self._post(text), "emotion": emotion}
        if "input_audio_transcription" in t or t.endswith(".delta") or t.endswith(".text"):
            # qwen3-asr-flash-realtime streams the growing partial in ``stash``
            # (the ``text`` field stays empty until the ``completed`` event).
            text = _first(msg, "stash", "text", "transcript", "delta")
            if text:
                return {"type": "partial", "text": text, "emotion": emotion}
            return None
        if t.endswith("error") or msg.get("error"):
            return {"type": "error", "message": _err(msg)}
        return None

    def _post(self, text: str) -> str:
        text = (text or "").strip()
        if text and self._use_glossary:
            try:
                from mast.knowledge.glossary import normalize_asr

                return normalize_asr(text)
            except Exception:
                return text
        return text

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


# ═══════════════════════════════════════════════════════════════════════════
#  TTS — text → speech
# ═══════════════════════════════════════════════════════════════════════════
class DashScopeRealtimeTTS:
    """One realtime TTS connection (typically one per assistant turn). Feed text
    with :meth:`append_text` as the LLM produces it; drain 24 kHz PCM16 frames
    from :meth:`audio_frames`. :meth:`cancel` supports barge-in (stop now)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "qwen3-tts-flash-realtime",
        voice: str = "Cherry",
        use_glossary: bool = True,
    ) -> None:
        if not api_key:
            raise RealtimeVoiceError("empty DashScope API key.")
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._use_glossary = use_glossary
        self._ws: Any = None
        self._closed = False

    async def open(
        self,
        *,
        voice: Optional[str] = None,
        response_format: str = "pcm",   # enum: mp3 | wav | pcm | opus
        sample_rate: int = 24000,
        mode: str = "server_commit",
        connect_timeout: float = 20.0,
    ) -> None:
        if voice:
            self._voice = voice
        url = f"{_REALTIME_ENDPOINT}?model={self._model}"
        self._ws = await _connect(url, self._api_key, timeout=connect_timeout)
        session = {
            "voice": self._voice,
            "response_format": response_format,
            "sample_rate": sample_rate,
            "mode": mode,
        }
        await self._ws.send(json.dumps({"type": "session.update", "session": session}))
        logger.info("TTS realtime session opened (model=%s, voice=%s)", self._model, self._voice)

    async def append_text(self, text: str) -> None:
        """Append a text chunk (ideally a full sentence) to synthesize. Glossary
        read-aloud substitution (STM abbreviations / formulae) is applied here."""
        if self._closed or self._ws is None:
            return
        send_text = text or ""
        if send_text.strip() and self._use_glossary:
            try:
                from mast.knowledge.glossary import read_as_for_tts

                send_text = read_as_for_tts(send_text)
            except Exception:
                pass
        try:
            await self._ws.send(
                json.dumps({"type": "input_text_buffer.append", "text": send_text})
            )
        except Exception as exc:
            logger.debug("TTS append_text failed: %s", exc)

    async def finish(self) -> None:
        """Signal end-of-input: no more text; drain remaining audio then close."""
        if self._closed or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "input_text_buffer.commit"}))
            await self._ws.send(json.dumps({"type": "session.finish"}))
        except Exception as exc:
            logger.debug("TTS finish failed: %s", exc)

    async def audio_frames(self) -> AsyncIterator[bytes]:
        """Yield raw 24 kHz PCM16 audio bytes as they arrive; ends on
        response.done / session.finished / socket close."""
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                msg = _parse(raw)
                if msg is None:
                    continue
                t = str(msg.get("type") or "")
                logger.debug("TTS ← %s", _preview(msg))
                if t.endswith("audio.delta") or (t.endswith(".delta") and (msg.get("delta") or msg.get("audio"))):
                    b64 = _first(msg, "delta", "audio")
                    if b64:
                        try:
                            yield base64.b64decode(b64)
                        except Exception:
                            continue
                elif t.endswith("session.finished") or t.endswith("response.done"):
                    break
                elif t.endswith("error") or msg.get("error"):
                    logger.warning("TTS server error: %s", _err(msg))
                    break
        except Exception as exc:
            if not self._closed:
                logger.debug("TTS audio loop ended: %s", exc)

    async def cancel(self) -> None:
        """Barge-in: stop synthesis immediately and drop the connection."""
        if self._ws is not None and not self._closed:
            try:
                await self._ws.send(json.dumps({"type": "response.cancel"}))
            except Exception:
                pass
        await self.close()

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


# ── shared frame helpers ───────────────────────────────────────────────────
def _parse(raw: Any) -> Optional[dict]:
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return None
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _err(msg: dict) -> str:
    e = msg.get("error")
    if isinstance(e, dict):
        return _first(e, "message", "code", default=str(e))
    return _first(msg, "message", default=str(e or msg.get("type") or "error"))


def _preview(msg: dict) -> str:
    """Compact one-line log of a server frame with big base64 blobs elided."""
    t = msg.get("type", "?")
    keys = []
    for k, v in msg.items():
        if k == "type":
            continue
        if isinstance(v, str) and len(v) > 40:
            keys.append(f"{k}=<{len(v)}b>")
        else:
            keys.append(f"{k}={v!r}"[:60])
    return f"{t} {' '.join(keys)}".strip()


__all__ = [
    "DashScopeRealtimeASR",
    "DashScopeRealtimeTTS",
    "RealtimeVoiceError",
    "TTS_VOICES",
]
