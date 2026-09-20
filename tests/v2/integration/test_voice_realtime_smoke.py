"""Live DashScope realtime voice smoke — TTS → (resample 24k→16k) → ASR round trip.

SKIPPED BY DEFAULT. Needs network + a DashScope key WITH realtime entitlement
(``qwen3-tts-flash-realtime`` / ``qwen3-asr-flash-realtime`` activated in the
Model Studio console). Run explicitly:

    MAST_VOICE_RT_SMOKE=1 PYTHONPATH=MASTv2 \
      .venv-v2-py313/Scripts/python.exe -m pytest \
      tests/v2/integration/test_voice_realtime_smoke.py -s

It self-skips (not fails) when the key is missing OR the account is not entitled
(the server closes the realtime WS with "account in good standing"), so it stays
green in CI while remaining a one-command validation once realtime is enabled.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MAST_VOICE_RT_SMOKE") != "1",
    reason="live realtime smoke — set MAST_VOICE_RT_SMOKE=1 to run",
)

TEXT = "扫描已完成，在样品表面发现清晰的原子台阶。"


def _resample_24k_to_16k(pcm24: bytes) -> bytes:
    import numpy as np
    from scipy.signal import resample_poly

    x = np.frombuffer(pcm24, dtype="<i2").astype("float32")
    y = resample_poly(x, up=2, down=3)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


async def _drain(agen, timeout: float):
    ait = agen.__aiter__()
    out = []
    while True:
        try:
            out.append(await asyncio.wait_for(ait.__anext__(), timeout))
        except (StopAsyncIteration, asyncio.TimeoutError):
            return out


async def _run() -> tuple[int, str]:
    from mast.config import _load_provider_key
    from mast.voice.realtime import DashScopeRealtimeASR, DashScopeRealtimeTTS

    key = _load_provider_key("dashscope")
    if not key:
        pytest.skip("no DashScope key configured")

    # TTS
    tts = DashScopeRealtimeTTS(key, voice="Cherry")
    await tts.open(connect_timeout=45.0)
    await tts.append_text(TEXT)
    await tts.finish()
    frames = await _drain(tts.audio_frames(), 60.0)
    await tts.close()
    pcm24 = b"".join(frames)
    if not pcm24:
        pytest.skip("realtime TTS produced no audio (account not entitled for realtime?)")

    # ASR on the synthesized audio
    pcm16 = _resample_24k_to_16k(pcm24)
    # vad=False (manual commit) matches how VoiceSessionOrchestrator drives ASR
    # (the client's PTT / VAD is the endpoint authority → asr.commit()).
    asr = DashScopeRealtimeASR(key, language="zh")
    await asr.open(vad=False, connect_timeout=45.0)

    async def feed():
        for i in range(0, len(pcm16), 3200):
            await asr.send_audio(pcm16[i:i + 3200])
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)
        await asr.commit()

    task = asyncio.create_task(feed())
    transcript = ""
    ait = asr.events().__aiter__()
    while True:
        try:
            ev = await asyncio.wait_for(ait.__anext__(), 60.0)
        except (StopAsyncIteration, asyncio.TimeoutError):
            break
        if ev["type"] == "final":
            transcript = ev["text"]
            break
        if ev["type"] == "error":
            break
    task.cancel()
    await asr.close()
    return len(pcm24), transcript


def test_realtime_round_trip():
    n_bytes, transcript = asyncio.run(_run())
    assert n_bytes > 0
    # loose check — ASR of synthesized speech should recover the keyword
    assert "扫描" in transcript or "台阶" in transcript, f"got: {transcript!r}"
