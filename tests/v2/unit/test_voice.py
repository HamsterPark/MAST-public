"""Unit tests for MASTv2 mast.voice.dashscope (vendored from v1 tests).

Same coverage; the path bootstrap below pins ``mast.voice`` to the v2 copy.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
# pytest's rootdir-based sys.path injection puts D:\...\MAST first, where
# the v1 mast/ shadows MASTv2/mast/. Force MASTv2/ ahead, and purge any
# already-cached v1 mast.* modules. Same canonical block as
# test_wrap_skill_minimal.py.
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import base64  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from mast.voice import DashScopeVoiceClient, TTS_VOICES, VoiceProviderError  # noqa: E402
import mast.voice.dashscope as _ds_module  # noqa: E402

assert "MASTv2" in _ds_module.__file__.replace("\\", "/"), (
    f"voice module resolved to v1 path: {_ds_module.__file__}"
)


# ── helpers ────────────────────────────────────────────────────────────

def _mock_tts(wav: bytes = b"RIFFXXXXWAVEfake_payload"):
    """Return a MockTransport that responds to TTS calls with inline base64."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["model"].startswith("qwen3-tts")
        assert "input" in body and "text" in body["input"]
        return httpx.Response(
            200,
            json={
                "output": {
                    "audio": {
                        "data": base64.b64encode(wav).decode(),
                        "expires_at": 0,
                        "id": "mock-id",
                        "url": "",
                    },
                    "finish_reason": "stop",
                },
                "usage": {"characters": len(body["input"]["text"])},
                "request_id": "mock-tts-req",
            },
        )
    return httpx.MockTransport(handler)


def _mock_asr(text: str = "你好世界", emotion: str = "neutral", lang: str = "zh"):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["model"].startswith("qwen3-asr")
        msgs = body["input"]["messages"]
        assert msgs[0]["role"] == "user"
        assert "audio" in msgs[0]["content"][0]
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {
                            "annotations": [
                                {"emotion": emotion, "language": lang, "type": "audio_info"}
                            ],
                            "content": [{"text": text}],
                            "role": "assistant",
                        },
                    }]
                },
                "usage": {"input_tokens": 16, "output_tokens": 9, "total_tokens": 25},
                "request_id": "mock-asr-req",
            },
        )
    return httpx.MockTransport(handler)


def _client_with_transport(transport: httpx.MockTransport, **kw) -> DashScopeVoiceClient:
    c = DashScopeVoiceClient(api_key="sk-test", **kw)
    c._http.close()
    c._http = httpx.Client(
        transport=transport,
        timeout=10,
        headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
    )
    return c


# ── construction ───────────────────────────────────────────────────────

def test_empty_api_key_raises():
    with pytest.raises(VoiceProviderError):
        DashScopeVoiceClient(api_key="")


def test_voices_constant():
    assert "Cherry" in TTS_VOICES
    assert "Ethan" in TTS_VOICES
    assert "Chelsie" in TTS_VOICES
    assert "Serena" in TTS_VOICES
    assert "Dylan" in TTS_VOICES
    assert len(TTS_VOICES) == 5


def test_set_voice_unknown_warns_but_accepts(caplog):
    c = DashScopeVoiceClient(api_key="sk-test")
    c.set_voice("RoboticGremlin")
    assert c.default_voice == "RoboticGremlin"


# ── TTS ────────────────────────────────────────────────────────────────

def test_synthesize_returns_wav_bytes():
    c = _client_with_transport(_mock_tts(b"RIFFheaderWAVEdata"))
    wav = c.synthesize("你好")
    assert wav == b"RIFFheaderWAVEdata"


def test_synthesize_empty_text_raises():
    c = _client_with_transport(_mock_tts())
    with pytest.raises(VoiceProviderError):
        c.synthesize("   ")


def test_synthesize_uses_default_voice_when_omitted(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "output": {
                    "audio": {"data": base64.b64encode(b"wav").decode(), "url": ""},
                    "finish_reason": "stop",
                },
                "usage": {"characters": 1},
                "request_id": "x",
            },
        )

    c = _client_with_transport(httpx.MockTransport(handler), default_voice="Ethan")
    c.synthesize("hi")
    assert captured["body"]["parameters"]["voice"] == "Ethan"


def test_synthesize_overridden_voice_wins():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"output": {"audio": {"data": base64.b64encode(b"wav").decode()}, "finish_reason": "stop"}, "usage": {}, "request_id": "x"},
        )

    c = _client_with_transport(httpx.MockTransport(handler), default_voice="Cherry")
    c.synthesize("hi", voice="Dylan")
    assert captured["body"]["parameters"]["voice"] == "Dylan"


def test_synthesize_caches_in_memory():
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={"output": {"audio": {"data": base64.b64encode(b"wav-bytes").decode()}, "finish_reason": "stop"}, "usage": {}, "request_id": "x"},
        )

    c = _client_with_transport(httpx.MockTransport(handler))
    a = c.synthesize("hello")
    b = c.synthesize("hello")
    assert a == b == b"wav-bytes"
    assert call_count["n"] == 1, "second call should hit cache, not network"


def test_synthesize_on_disk_cache(tmp_path: Path):
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={"output": {"audio": {"data": base64.b64encode(b"persisted").decode()}, "finish_reason": "stop"}, "usage": {}, "request_id": "x"},
        )

    c1 = _client_with_transport(httpx.MockTransport(handler), cache_dir=tmp_path)
    c1.synthesize("once")
    c1.close()

    # Second client, fresh memory cache, but disk cache should hit
    c2 = _client_with_transport(httpx.MockTransport(handler), cache_dir=tmp_path)
    out = c2.synthesize("once")
    assert out == b"persisted"
    assert call_count["n"] == 1


def test_synthesize_oss_url_path(monkeypatch):
    """When DashScope returns only a URL (no inline data), client downloads it."""
    def api_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"output": {"audio": {"url": "https://oss.example/clip.wav", "data": ""}, "finish_reason": "stop"}, "usage": {}, "request_id": "x"},
        )

    fetched = {"called": False}

    def fake_get(url, timeout):
        fetched["called"] = True
        assert url == "https://oss.example/clip.wav"
        return httpx.Response(
            200,
            content=b"oss-wav-content",
            request=httpx.Request("GET", url),
        )

    c = _client_with_transport(httpx.MockTransport(api_handler))
    monkeypatch.setattr("mast.voice.dashscope.httpx.get", fake_get)
    out = c.synthesize("via oss")
    assert out == b"oss-wav-content"
    assert fetched["called"]


def test_synthesize_no_audio_raises():
    def handler(request):
        return httpx.Response(200, json={"output": {"audio": {}, "finish_reason": "stop"}, "request_id": "x"})

    c = _client_with_transport(httpx.MockTransport(handler))
    with pytest.raises(VoiceProviderError, match="returned no audio"):
        c.synthesize("hi")


def test_synthesize_http_error_raises():
    def handler(request):
        return httpx.Response(429, text="rate limited")

    c = _client_with_transport(httpx.MockTransport(handler))
    with pytest.raises(VoiceProviderError, match="HTTP 429"):
        c.synthesize("hi")


# ── ASR ────────────────────────────────────────────────────────────────

def test_transcribe_bytes():
    c = _client_with_transport(_mock_asr(text="你好世界", emotion="happy", lang="zh"))
    out = c.transcribe(b"\x00\x01\x02fake-wav-bytes", mime="wav")
    assert out["text"] == "你好世界"
    assert out["emotion"] == "happy"
    assert out["language"] == "zh"


def test_transcribe_path(tmp_path: Path):
    p = tmp_path / "test.wav"
    p.write_bytes(b"RIFFXXXX")
    c = _client_with_transport(_mock_asr(text="from file"))
    out = c.transcribe(p)
    assert out["text"] == "from file"


def test_transcribe_https_url_passthrough():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": [{"text": "from url"}], "annotations": []}}]}, "usage": {}, "request_id": "x"},
        )

    c = _client_with_transport(httpx.MockTransport(handler))
    c.transcribe("https://example.com/clip.wav")
    audio_field = captured["body"]["input"]["messages"][0]["content"][0]["audio"]
    assert audio_field == "https://example.com/clip.wav"  # passes through, no base64


def test_transcribe_empty_choices_raises():
    def handler(request):
        return httpx.Response(200, json={"output": {"choices": []}, "request_id": "x"})

    c = _client_with_transport(httpx.MockTransport(handler))
    with pytest.raises(VoiceProviderError, match="no choices"):
        c.transcribe(b"x", mime="wav")


def test_transcribe_http_error_raises():
    def handler(request):
        return httpx.Response(401, text="bad key")

    c = _client_with_transport(httpx.MockTransport(handler))
    with pytest.raises(VoiceProviderError, match="HTTP 401"):
        c.transcribe(b"x", mime="wav")


def test_transcribe_string_content_handled():
    """Some responses return content as a plain string instead of [{text:...}]."""
    def handler(request):
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "plain", "annotations": []}}]}, "request_id": "x"},
        )

    c = _client_with_transport(httpx.MockTransport(handler))
    out = c.transcribe(b"x", mime="wav")
    assert out["text"] == "plain"


# ── Optional live test (network) ───────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("MAST_VOICE_LIVE_TESTS"),
    reason="set MAST_VOICE_LIVE_TESTS=1 to hit real DashScope",
)
def test_live_roundtrip_tts_then_asr():
    from mast.config import _load_provider_key
    key = _load_provider_key("dashscope")
    if not key:
        pytest.skip("no dashscope key configured")
    c = DashScopeVoiceClient(api_key=key)
    wav = c.synthesize("测试一下", voice="Cherry")
    assert wav.startswith(b"RIFF") and b"WAVE" in wav[:16]
    asr = c.transcribe(wav, mime="wav")
    assert "测试" in asr["text"]
    c.close()
