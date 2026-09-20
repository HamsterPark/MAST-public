"""Aliyun DashScope voice client — TTS + ASR via the multimodal-generation endpoint.

Two operations are exposed:

  * `synthesize(text, voice="Cherry") -> bytes`
        Calls qwen3-tts-flash. Returns 24 kHz mono WAV bytes.
        DashScope returns an OSS URL (20-min TTL); this client immediately
        downloads it so callers get raw bytes they can stream into a Gradio
        `gr.Audio` widget without worrying about expiry.

  * `transcribe(audio, mime="wav") -> dict`
        Calls qwen3-asr-flash. Accepts raw bytes, a path, or an https URL.
        Returns ``{"text": str, "emotion": str, "language": str}``.

Both endpoints use one URL:
    POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
with Bearer auth. No DashScope SDK is required — just httpx.

The TTS path uses an in-memory + on-disk LRU cache keyed by
``sha1((voice, text, model))`` so the same prompt does not pay the API
round-trip twice.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_DASHSCOPE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)

#: Voices verified against qwen3-tts-flash on 2026-05-07.
TTS_VOICES: tuple[str, ...] = ("Cherry", "Ethan", "Chelsie", "Serena", "Dylan")


class VoiceProviderError(RuntimeError):
    """Raised when DashScope returns a non-200 response or unexpected payload."""


class DashScopeVoiceClient:
    """Thin httpx wrapper around DashScope qwen3-tts-flash + qwen3-asr-flash."""

    def __init__(
        self,
        api_key: str,
        *,
        tts_model: str = "qwen3-tts-flash",
        asr_model: str = "qwen3-asr-flash",
        default_voice: str = "Cherry",
        cache_dir: Path | None = None,
        timeout_s: float = 60.0,
        use_glossary: bool = True,
    ) -> None:
        if not api_key:
            raise VoiceProviderError("Empty DashScope API key.")
        self._api_key = api_key
        self._tts_model = tts_model
        self._asr_model = asr_model
        self._default_voice = default_voice
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._use_glossary = use_glossary
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, bytes] = {}
        self._mem_cache_lock = threading.Lock()
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=15.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        """Close the underlying httpx client."""
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self) -> "DashScopeVoiceClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── TTS ──────────────────────────────────────────────────────────

    @property
    def default_voice(self) -> str:
        return self._default_voice

    def set_voice(self, voice: str) -> None:
        if voice not in TTS_VOICES:
            logger.warning("Unknown voice %r; allowed: %s", voice, TTS_VOICES)
        self._default_voice = voice

    def _cache_key(self, text: str, voice: str) -> str:
        h = hashlib.sha1()
        h.update(self._tts_model.encode("utf-8"))
        h.update(b"|")
        h.update(voice.encode("utf-8"))
        h.update(b"|")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> bytes | None:
        with self._mem_cache_lock:
            cached = self._mem_cache.get(key)
        if cached is not None:
            return cached
        if self._cache_dir is not None:
            path = self._cache_dir / f"{key}.wav"
            if path.exists():
                data = path.read_bytes()
                with self._mem_cache_lock:
                    self._mem_cache[key] = data
                return data
        return None

    def _cache_put(self, key: str, data: bytes) -> None:
        with self._mem_cache_lock:
            self._mem_cache[key] = data
        if self._cache_dir is not None:
            try:
                (self._cache_dir / f"{key}.wav").write_bytes(data)
            except OSError as exc:
                logger.debug("TTS cache write failed: %s", exc)

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Convert *text* → 24 kHz mono WAV bytes.

        When `use_glossary=True`, applies `glossary.read_as_for_tts()` before
        sending — STM abbreviations and chemical formulae get spelled out so
        the TTS engine reads them naturally to a Chinese listener.

        Raises:
            VoiceProviderError on API failure or unparseable response.
        """
        if not text or not text.strip():
            raise VoiceProviderError("synthesize: empty text.")
        v = voice or self._default_voice
        # Glossary preprocessing: substitute terms with TTS-friendly readings
        send_text = text
        if self._use_glossary:
            try:
                from mast.knowledge.glossary import read_as_for_tts
                send_text = read_as_for_tts(text)
            except ImportError:
                pass
        key = self._cache_key(send_text, v)
        cached = self._cache_get(key)
        if cached is not None:
            logger.debug("TTS cache hit (%d bytes) for voice=%s", len(cached), v)
            return cached

        body = {
            "model": self._tts_model,
            "input": {"text": send_text},
            "parameters": {"voice": v},
        }
        try:
            resp = self._http.post(_DASHSCOPE_ENDPOINT, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else ""
            raise VoiceProviderError(
                f"DashScope TTS HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except Exception as exc:
            raise VoiceProviderError(f"DashScope TTS request failed: {exc}") from exc

        payload = resp.json()
        audio = (payload.get("output") or {}).get("audio") or {}
        url = audio.get("url") or ""
        inline = audio.get("data") or ""

        if inline:
            try:
                wav_bytes = base64.b64decode(inline)
            except Exception as exc:
                raise VoiceProviderError(f"DashScope TTS bad base64: {exc}") from exc
        elif url:
            try:
                # Use a fresh httpx request — OSS URL doesn't accept our auth header.
                r2 = httpx.get(url, timeout=30.0)
                r2.raise_for_status()
                wav_bytes = r2.content
            except Exception as exc:
                raise VoiceProviderError(f"DashScope TTS OSS download failed: {exc}") from exc
        else:
            raise VoiceProviderError(
                f"DashScope TTS returned no audio (request_id={payload.get('request_id')})"
            )

        if not wav_bytes:
            raise VoiceProviderError("DashScope TTS returned empty audio bytes.")
        self._cache_put(key, wav_bytes)
        logger.info(
            "TTS synthesised %d chars → %d bytes (voice=%s, model=%s)",
            len(text), len(wav_bytes), v, self._tts_model,
        )
        try:  # book TTS cost (chars) into the 用量·花销 ledger — fail-safe
            from mast.billing.capture import record_tts
            record_tts(model=self._tts_model, chars=len(text))
        except Exception:  # noqa: BLE001
            pass
        return wav_bytes

    # ── ASR ──────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        mime: str = "wav",
    ) -> dict[str, Any]:
        """Convert speech audio → ``{"text", "emotion", "language"}``.

        Args:
            audio: raw bytes, a local file path, or an https://… URL.
            mime:  audio container (`wav`, `mp3`, `ogg`, `flac`); ignored if a
                   URL is passed.

        Raises:
            VoiceProviderError on API failure or unparseable response.
        """
        audio_field = self._build_audio_field(audio, mime)
        body = {
            "model": self._asr_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"audio": audio_field}],
                    }
                ]
            },
            "parameters": {},
        }
        try:
            resp = self._http.post(_DASHSCOPE_ENDPOINT, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else ""
            raise VoiceProviderError(
                f"DashScope ASR HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except Exception as exc:
            raise VoiceProviderError(f"DashScope ASR request failed: {exc}") from exc

        payload = resp.json()
        try:  # book ASR cost (audio seconds) — fail-safe; skip if no duration given
            _u = payload.get("usage") or {}
            _sec = (_u.get("duration") or _u.get("seconds")
                    or _u.get("audio_seconds") or _u.get("audio_duration"))
            if _sec:
                from mast.billing.capture import record_asr
                record_asr(model=self._asr_model, seconds=float(_sec))
        except Exception:  # noqa: BLE001
            pass
        choices = (payload.get("output") or {}).get("choices") or []
        if not choices:
            raise VoiceProviderError(
                f"DashScope ASR returned no choices (request_id={payload.get('request_id')})"
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        annotations = message.get("annotations") or []

        text = ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text += str(block["text"])
        elif isinstance(content, str):
            text = content

        emotion = ""
        language = ""
        for ann in annotations:
            if isinstance(ann, dict):
                if "emotion" in ann and not emotion:
                    emotion = str(ann["emotion"])
                if "language" in ann and not language:
                    language = str(ann["language"])

        cleaned_text = text.strip()
        if self._use_glossary and cleaned_text:
            try:
                from mast.knowledge.glossary import normalize_asr
                cleaned_text = normalize_asr(cleaned_text)
            except ImportError:
                pass
        result = {
            "text": cleaned_text,
            "emotion": emotion,
            "language": language,
            "request_id": payload.get("request_id", ""),
        }
        logger.info(
            "ASR transcribed → %d chars (lang=%s, emotion=%s)",
            len(result["text"]), language, emotion,
        )
        return result

    @staticmethod
    def _build_audio_field(audio: bytes | str | Path, mime: str) -> str:
        """Convert *audio* to either an https URL or a `data:audio/...;base64,...` string."""
        if isinstance(audio, str) and audio.startswith(("http://", "https://")):
            return audio
        if isinstance(audio, str):
            audio = Path(audio)
        if isinstance(audio, Path):
            audio_bytes = audio.read_bytes()
            if mime == "wav" and audio.suffix.lower() in (".mp3", ".ogg", ".flac", ".m4a"):
                mime = audio.suffix.lower().lstrip(".")
        else:
            audio_bytes = audio
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return f"data:audio/{mime};base64,{b64}"


__all__ = ["DashScopeVoiceClient", "VoiceProviderError", "TTS_VOICES"]
