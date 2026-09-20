"""chat_voice slice — voice ASR/TTS + curated chat quick-prompts.

Parity rebuild Wave A. The old Gradio chat panel exposed voice input/output
(speech → instruction, assistant reply → speech) and a set of curated/override
quick-prompt chips; the TS SPA lost the controls but the LOGIC still lives in
the kept core (``mast.voice.DashScopeVoiceClient`` for ASR/TTS,
``ConfigOverrideRegistry.get_quick_prompts`` for the override prompts). This
module re-exposes that logic as typed endpoints — it only RELAYS into the core;
no voice/business logic is reimplemented here.

Endpoints:

  * ``POST /api/voice/transcribe``   — ASR (qwen3-asr-flash)
  * ``POST /api/voice/synthesize``   — TTS (qwen3-tts-flash), WAV relayed base64
  * ``GET  /api/voice/voices``       — available TTS voices + configured default
  * ``GET  /api/chat/quick-prompts`` — curated + override quick-prompts

NOTE (no duplication): ``POST /api/feedback`` already exists (routes/records.py,
the conversation-feedback write) and ``/api/realtime/snapshot`` already serves
the data strip (routes/realtime.py) — both are intentionally NOT re-added here.

GRACEFUL DEGRADATION (house rule 2): the API boots STANDALONE. The voice client
needs a DashScope key; with none (or any backend raise) the handlers return a
valid degraded result (``degraded=True``), never a 500. Heavy backends
(``mast.voice``, ``mast.config``, ``mast.admin.override_store``) are
LAZY-imported inside each handler in try/except.
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Request

from mast.api.schemas_chat_voice import (
    QuickPrompt,
    QuickPromptsResponse,
    SynthesizeRequest,
    SynthesizeResult,
    TranscribeRequest,
    TranscribeResult,
    VoicesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat_voice"])


# ── curated built-in quick-prompts ─────────────────────────────────────
#
# A small STM-flavoured starter set so the chat composer always has chips even
# with no operator override file. Operator overrides (quick_prompts.json) are
# merged ON TOP of these (see GET /api/chat/quick-prompts). Edit freely — these
# are presentation hints, not behaviour; the agent core remains the source of
# truth for what each instruction actually does.
_CURATED_QUICK_PROMPTS: tuple[tuple[str, str], ...] = (
    ("当前状态", "报告当前仪器状态：偏压、电流、Z 控制器与扫描状态。"),
    ("扫一张图", "用当前参数扫描一张图并把结果显示出来。"),
    ("STS 谱", "在当前位置采集一条 STS 偏压谱。"),
    ("调针", "针尖质量看起来不好，请执行一次温和的调针流程。"),
    ("撤针", "安全撤针：关闭 Z 控制器并把针退到安全高度。"),
)


def _curated_prompts() -> list[QuickPrompt]:
    return [
        QuickPrompt(label=label, prompt=prompt, source="curated")
        for label, prompt in _CURATED_QUICK_PROMPTS
    ]


def _make_voice_client():
    """Lazy-build a DashScopeVoiceClient from config, or return None.

    Mirrors core.runtime's construction (enabled + api_key gate). Returns None
    when voice is disabled, no key is configured, or the backend is unavailable
    — callers degrade gracefully on None. Never raises.
    """
    try:
        from mast.config import MASTConfig
        from mast.voice import DashScopeVoiceClient

        cfg = MASTConfig()
        voice = cfg.voice
        if not getattr(voice, "enabled", False) or not getattr(voice, "api_key", ""):
            return None
        try:
            cache_dir = cfg.experiments_dir / voice.cache_dir_name
        except Exception:
            cache_dir = None
        return DashScopeVoiceClient(
            api_key=voice.api_key,
            tts_model=voice.tts_model,
            asr_model=voice.asr_model,
            default_voice=voice.default_voice,
            cache_dir=cache_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive (no key/no deps)
        logger.warning("voice client construct failed: %s", exc)
        return None


# ── POST /api/voice/transcribe (ASR) ───────────────────────────────────


@router.post("/voice/transcribe", response_model=TranscribeResult)
def voice_transcribe(body: TranscribeRequest, request: Request) -> TranscribeResult:
    """Speech → text via DashScopeVoiceClient.transcribe (qwen3-asr-flash).

    Accepts either a base64 audio blob (+ mime) or an https audio URL. Degrades
    to ``ok=false, degraded=true`` when no voice key is configured or the ASR
    call fails — never a 500."""
    if not body.audio_b64 and not body.audio_url:
        return TranscribeResult(ok=False, error="no audio supplied", degraded=False)

    client = _make_voice_client()
    if client is None:
        return TranscribeResult(
            ok=False, error="voice subsystem unavailable (no DashScope key)",
            degraded=True,
        )

    try:
        if body.audio_url:
            audio: bytes | str = body.audio_url
        else:
            try:
                audio = base64.b64decode(body.audio_b64 or "", validate=False)
            except Exception:
                return TranscribeResult(
                    ok=False, error="invalid base64 audio", degraded=False
                )
        result = client.transcribe(audio, mime=body.mime or "wav")
        return TranscribeResult(
            ok=True,
            text=str(result.get("text", "") or ""),
            emotion=str(result.get("emotion", "") or ""),
            language=str(result.get("language", "") or ""),
            request_id=str(result.get("request_id", "") or ""),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("voice transcribe failed: %s", exc)
        return TranscribeResult(ok=False, error=str(exc), degraded=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


# ── POST /api/voice/synthesize (TTS) ───────────────────────────────────


@router.post("/voice/synthesize", response_model=SynthesizeResult)
def voice_synthesize(body: SynthesizeRequest, request: Request) -> SynthesizeResult:
    """Text → 24 kHz mono WAV via DashScopeVoiceClient.synthesize
    (qwen3-tts-flash). The WAV bytes are relayed base64-encoded so the SPA can
    build a ``data:audio/wav;base64,…`` URL directly. Degrades to
    ``ok=false, degraded=true`` with no key / on failure — never a 500."""
    text = (body.text or "").strip()
    if not text:
        return SynthesizeResult(ok=False, error="empty text", degraded=False)

    client = _make_voice_client()
    if client is None:
        return SynthesizeResult(
            ok=False, error="voice subsystem unavailable (no DashScope key)",
            degraded=True,
        )

    try:
        wav = client.synthesize(text, voice=body.voice)
        return SynthesizeResult(
            ok=True,
            audio_b64=base64.b64encode(wav).decode("ascii"),
            mime="audio/wav",
            voice=body.voice or getattr(client, "default_voice", ""),
            bytes=len(wav),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("voice synthesize failed: %s", exc)
        return SynthesizeResult(ok=False, error=str(exc), degraded=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


# ── GET /api/voice/voices ──────────────────────────────────────────────


@router.get("/voice/voices", response_model=VoicesResponse)
def voice_voices(request: Request) -> VoicesResponse:
    """Available TTS voices (mast.voice.TTS_VOICES) + the configured default +
    whether the voice subsystem is actually reachable (key present). Always 200;
    the voice list is static so it is served even without a key (``enabled``
    reports reachability)."""
    try:
        from mast.voice import TTS_VOICES

        voices = list(TTS_VOICES)
    except Exception as exc:
        logger.warning("voice list import failed: %s", exc)
        return VoicesResponse(degraded=True)

    default_voice = voices[0] if voices else ""
    enabled = False
    try:
        from mast.config import MASTConfig

        vcfg = MASTConfig().voice
        default_voice = getattr(vcfg, "default_voice", default_voice) or default_voice
        enabled = bool(getattr(vcfg, "enabled", False) and getattr(vcfg, "api_key", ""))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("voice config read failed: %s", exc)

    return VoicesResponse(
        voices=voices,
        default_voice=default_voice,
        enabled=enabled,
        degraded=False,
    )


# ── GET /api/chat/quick-prompts ────────────────────────────────────────


@router.get("/chat/quick-prompts", response_model=QuickPromptsResponse)
def chat_quick_prompts(request: Request) -> QuickPromptsResponse:
    """Curated built-in quick-prompts merged with operator overrides
    (ConfigOverrideRegistry.get_quick_prompts → quick_prompts.json). Always
    returns the curated set; the override store is best-effort (``degraded=true``
    if it cannot be read)."""
    prompts = _curated_prompts()
    degraded = False
    try:
        from mast.admin.override_store import ConfigOverrideRegistry

        registry = ConfigOverrideRegistry.get()
        for entry in registry.get_quick_prompts() or []:
            if not isinstance(entry, dict):
                continue
            prompt_text = str(entry.get("prompt", "") or "")
            if not prompt_text:
                continue
            prompts.append(
                QuickPrompt(
                    label=str(entry.get("label", "") or prompt_text[:24]),
                    prompt=prompt_text,
                    source="override",
                    meta={
                        k: v
                        for k, v in entry.items()
                        if k not in ("label", "prompt")
                    },
                )
            )
    except Exception as exc:
        logger.warning("quick-prompts override read failed: %s", exc)
        degraded = True

    return QuickPromptsResponse(
        prompts=prompts, count=len(prompts), degraded=degraded
    )


__all__ = ["router"]
