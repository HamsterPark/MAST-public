"""WebSocket ``/ws/voice`` — full-duplex voice channel (ASR ⇄ agent ⇄ TTS).

Mirrors the ``mast/api/ws.py`` house pattern: full path (no ``/api`` prefix),
accept-first, graceful degradation. All voice logic lives in
``mast.voice.session.VoiceSessionOrchestrator``; this route is only the transport
seam — it demuxes binary (PCM) vs JSON (control) frames and relays the
orchestrator's outbound frames back.

DEGRADES, never 500s: with no conversation engine / no DashScope key the
orchestrator sends a single ``degraded`` frame and idles until the client leaves.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mast.api.schemas_voice_ws import (
    BargeInFrame,
    ByeFrame,
    CommitFrame,
    HelloFrame,
    SetModeFrame,
    StopFrame,
    TextFrame,
    parse_client_frame,
)
from mast.voice.session import VoiceSessionOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])


def _voice_cfg(ctx: Any) -> Optional[Any]:
    """Resolve ``config.voice`` from the wired context, else lazily from
    ``MASTConfig()`` (matches routes/chat_voice._make_voice_client)."""
    cfg = getattr(ctx, "config", None)
    if cfg is not None and getattr(cfg, "voice", None) is not None:
        return cfg.voice
    try:
        from mast.config import MASTConfig

        return MASTConfig().voice
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("voice cfg resolve failed: %s", exc)
        return None


def _default_conversation_id(ctx: Any) -> str:
    """The active private conversation (the 仪器 chat), best-effort."""
    store = getattr(ctx, "conversation_store", None)
    if store is None:
        return ""
    try:
        rows = store.list(kind="private", limit=1)
        if rows:
            return str(rows[0].get("conversation_id") or "")
    except Exception as exc:  # noqa: BLE001
        logger.debug("default conversation resolve failed: %s", exc)
    return ""


def _hitl_factory(ctx: Any):
    """Build ``(conversation_id, abort) -> resolver`` for the voice session.

    Same seam as the typed chat stream — one implementation, so a spoken
    approval and a typed one publish the same card into the same store and are
    answered by the same endpoint. Returns ``None`` with no live store
    (standalone dev), which keeps the honest "nobody can approve this here"
    behaviour rather than pretending a gate exists.
    """
    app = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    if app is None or not getattr(app, "_orch_interrupts", None):
        return None
    from mast.api.routes.chat_stream import _make_hitl_resolver

    def factory(conversation_id: str, abort):
        agent_id = "instrument_control"
        store = getattr(ctx, "conversation_store", None)
        if store is not None and conversation_id:
            try:
                row = store.get(conversation_id)
                agent_id = (row or {}).get("agent_id") or agent_id
            except Exception as exc:  # noqa: BLE001
                logger.debug("voice hitl owner lookup failed: %s", exc)
        return _make_hitl_resolver(ctx, agent_id, conversation_id, abort)

    return factory


@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket) -> None:
    await ws.accept()
    ctx = ws.app.state.ctx
    engine = getattr(ctx, "conversation_engine", None)
    voice_cfg = _voice_cfg(ctx)

    async def send(frame: Any) -> None:
        if isinstance(frame, (bytes, bytearray)):
            await ws.send_bytes(bytes(frame))
        else:
            await ws.send_json(frame)

    orch = VoiceSessionOrchestrator(
        voice_cfg=voice_cfg,
        conversation_engine=engine,
        conversation_id=_default_conversation_id(ctx),
        send=send,
        hitl_resolver_factory=_hitl_factory(ctx),
    )

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is not None:
                await orch.on_audio(data)
                continue

            text = msg.get("text")
            if text is None:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            frame = parse_client_frame(obj)
            if frame is None:
                continue

            if isinstance(frame, HelloFrame):
                await orch.start(frame)
            elif isinstance(frame, CommitFrame):
                await orch.on_commit()
            elif isinstance(frame, TextFrame):
                await orch.on_text(frame.content)
            elif isinstance(frame, SetModeFrame):
                await orch.on_set_mode(frame.mode)
            elif isinstance(frame, StopFrame):
                await orch.on_stop()
            elif isinstance(frame, BargeInFrame):
                await orch.on_barge_in()
            elif isinstance(frame, ByeFrame):
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive; never crash the worker
        logger.warning("ws_voice loop ended: %s", exc)
    finally:
        await orch.close()


__all__ = ["router"]
