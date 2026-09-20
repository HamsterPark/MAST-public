"""chat_voice slice contract tests — voice ASR/TTS + chat quick-prompts.

Per the house test rule, the router under test is mounted on a throwaway
FastAPI app with a fresh AppContext (integration wires it into
``mast.api.app`` separately). We assert:

  * every endpoint returns its defined status + the schema-shaped body;
  * standalone (no DashScope key) voice paths degrade — empty, never broken,
    never 500;
  * a stubbed voice client drives transcribe/synthesize through their live
    relay paths (no network, no key needed);
  * quick-prompts always serve the curated built-ins and merge override rows
    from a REAL ConfigOverrideRegistry over a temp overrides dir.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes import chat_voice
from mast.api.routes.chat_voice import router


# ── throwaway app ──────────────────────────────────────────────────────


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


class _StubVoiceClient:
    """A no-network stand-in for DashScopeVoiceClient."""

    default_voice = "Cherry"

    def __init__(self, *, raise_on=None):
        self._raise_on = raise_on or set()
        self.closed = False
        self.last_transcribe = None
        self.last_synthesize = None

    def transcribe(self, audio, *, mime="wav"):
        if "transcribe" in self._raise_on:
            raise RuntimeError("asr boom")
        self.last_transcribe = (audio, mime)
        return {
            "text": "扫一张图",
            "emotion": "neutral",
            "language": "zh",
            "request_id": "req-123",
        }

    def synthesize(self, text, voice=None):
        if "synthesize" in self._raise_on:
            raise RuntimeError("tts boom")
        self.last_synthesize = (text, voice)
        return b"RIFF....WAVEfake-wav-bytes"

    def close(self):
        self.closed = True


# ── voice/transcribe (ASR) ─────────────────────────────────────────────


def test_transcribe_degrades_without_key(monkeypatch) -> None:
    monkeypatch.setattr(chat_voice, "_make_voice_client", lambda: None)
    r = _client().post(
        "/api/voice/transcribe",
        json={"audio_b64": base64.b64encode(b"x").decode()},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert body["text"] == ""


def test_transcribe_no_audio_is_not_degraded() -> None:
    # Missing audio is a client error shape, not a backend-degrade.
    r = _client().post("/api/voice/transcribe", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False
    assert "no audio" in (body["error"] or "")


def test_transcribe_live_relay(monkeypatch) -> None:
    stub = _StubVoiceClient()
    monkeypatch.setattr(chat_voice, "_make_voice_client", lambda: stub)
    r = _client().post(
        "/api/voice/transcribe",
        json={"audio_b64": base64.b64encode(b"hello").decode(), "mime": "wav"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["text"] == "扫一张图"
    assert body["language"] == "zh"
    assert body["request_id"] == "req-123"
    assert body["degraded"] is False
    # bytes were decoded and relayed to the client
    assert stub.last_transcribe[0] == b"hello"
    assert stub.closed is True


def test_transcribe_url_relay(monkeypatch) -> None:
    stub = _StubVoiceClient()
    monkeypatch.setattr(chat_voice, "_make_voice_client", lambda: stub)
    r = _client().post(
        "/api/voice/transcribe",
        json={"audio_url": "https://example.com/a.wav"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert stub.last_transcribe[0] == "https://example.com/a.wav"


def test_transcribe_backend_raise_degrades(monkeypatch) -> None:
    stub = _StubVoiceClient(raise_on={"transcribe"})
    monkeypatch.setattr(chat_voice, "_make_voice_client", lambda: stub)
    r = _client().post(
        "/api/voice/transcribe",
        json={"audio_b64": base64.b64encode(b"x").decode()},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert stub.closed is True


# ── voice/synthesize (TTS) ─────────────────────────────────────────────


def test_synthesize_degrades_without_key(monkeypatch) -> None:
    monkeypatch.setattr(chat_voice, "_make_voice_client", lambda: None)
    r = _client().post("/api/voice/synthesize", json={"text": "你好"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert body["audio_b64"] == ""


def test_synthesize_empty_text_not_degraded() -> None:
    r = _client().post("/api/voice/synthesize", json={"text": "   "})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False
    assert "empty" in (body["error"] or "")


def test_synthesize_live_relay(monkeypatch) -> None:
    stub = _StubVoiceClient()
    monkeypatch.setattr(chat_voice, "_make_voice_client", lambda: stub)
    r = _client().post(
        "/api/voice/synthesize", json={"text": "你好", "voice": "Ethan"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["mime"] == "audio/wav"
    assert body["voice"] == "Ethan"
    assert body["bytes"] > 0
    # base64 round-trips back to the stub's bytes
    assert base64.b64decode(body["audio_b64"]) == b"RIFF....WAVEfake-wav-bytes"
    assert stub.last_synthesize == ("你好", "Ethan")
    assert stub.closed is True


def test_synthesize_backend_raise_degrades(monkeypatch) -> None:
    stub = _StubVoiceClient(raise_on={"synthesize"})
    monkeypatch.setattr(chat_voice, "_make_voice_client", lambda: stub)
    r = _client().post("/api/voice/synthesize", json={"text": "你好"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


# ── voice/voices ───────────────────────────────────────────────────────


def test_voices_lists_static_set() -> None:
    # The voice list is static (mast.voice.TTS_VOICES); served even without key.
    r = _client().get("/api/voice/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert isinstance(body["voices"], list) and len(body["voices"]) >= 1
    assert "Cherry" in body["voices"]
    assert body["default_voice"]
    assert isinstance(body["enabled"], bool)


# ── chat/quick-prompts ─────────────────────────────────────────────────


def test_quick_prompts_curated_always_present() -> None:
    r = _client().get("/api/chat/quick-prompts")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(body["prompts"]) >= 1
    # curated built-ins are always present (overrides may add more on top)
    assert any(p["source"] == "curated" for p in body["prompts"])
    assert all(p["source"] in ("curated", "override") for p in body["prompts"])
    assert any(p["prompt"] for p in body["prompts"])


def test_quick_prompts_merges_overrides(monkeypatch, tmp_path) -> None:
    from mast.admin.override_store import ConfigOverrideRegistry

    ovr_dir = tmp_path / "overrides"
    ovr_dir.mkdir()
    (ovr_dir / "quick_prompts.json").write_text(
        json.dumps(
            {
                "prompts": [
                    {"label": "我的提示", "prompt": "做点别的", "tag": "x"},
                    {"prompt": "没有标签"},
                    {"label": "空", "prompt": ""},  # dropped (no prompt text)
                ]
            }
        ),
        encoding="utf-8",
    )
    ConfigOverrideRegistry.reset()
    monkeypatch.setattr(
        ConfigOverrideRegistry, "get", classmethod(lambda cls: cls(ovr_dir))
    )
    try:
        r = _client().get("/api/chat/quick-prompts")
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] is False
        override_rows = [p for p in body["prompts"] if p["source"] == "override"]
        # The two non-empty override prompts merged; the empty one dropped.
        assert len(override_rows) == 2
        labelled = next(p for p in override_rows if p["label"] == "我的提示")
        assert labelled["prompt"] == "做点别的"
        assert labelled["meta"].get("tag") == "x"
        # curated still present
        assert any(p["source"] == "curated" for p in body["prompts"])
    finally:
        ConfigOverrideRegistry.reset()


def test_quick_prompts_degrades_on_store_failure(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name == "mast.admin.override_store":
            raise RuntimeError("store boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = _client().get("/api/chat/quick-prompts")
    assert r.status_code == 200
    body = r.json()
    # curated survives; degraded flag set because override read failed
    assert body["degraded"] is True
    assert any(p["source"] == "curated" for p in body["prompts"])
