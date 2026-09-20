"""VoiceSessionOrchestrator — drive one full-duplex voice session over ``/ws/voice``.

Owns the state machine for ONE connected client and bridges three worlds:

  microphone PCM  ──▶  ASR (persistent realtime streaming, manual-commit; the
                            │      client's PTT/VAD is the endpoint authority.
                            │      Falls back to batch on-commit if unavailable.)
                            │  partial (stash) → live caption; final → run turn
                            ▼
                  ConversationEngine.stream_events()  (the SAME IC agent graph)
                            │  token / tool / final events
                            ▼
                  sentence text  ──▶  TTS (realtime streaming, batch fallback)  ──▶  audio down

Design rules honored:
* **Never blocks the event loop** — the sync ``stream_events`` generator and the
  sync batch httpx calls run in threads; results cross back via the loop.
* **Speaks while the agent generates** — assistant tokens are coalesced into
  sentences and each finished sentence is synthesized immediately (realtime: one
  persistent TTS WS fed per sentence; batch: one synth per sentence), so the
  first words are heard before the reply is complete. Tool executions are
  narrated aloud ("开始扫描" / "扫描完成").
* **Graceful degradation** — realtime TTS is probed once in the background; until
  proven it uses the batch client, which works wherever the batch voice does. A
  missing key → a single ``degraded`` frame.
* **Voice never bypasses safety** — a HITL / DANGEROUS interrupt is ANNOUNCED and
  surfaced to the approval modal; the turn stops. No auto-approval here.

``send`` is an async callable the route supplies: ``await send(frame_dict)`` for a
JSON control frame, ``await send(pcm_bytes)`` for a binary audio frame.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from typing import Any, Awaitable, Callable, Optional

from mast.api.schemas_voice_ws import (
    AsrFinal,
    AsrPartial,
    DegradedFrame,
    HelloAck,
    HelloFrame,
    InterruptFrame,
    Narration,
    ReplyDelta,
    StateFrame,
    TtsBegin,
    TtsDone,
    WakeFrame,
)

logger = logging.getLogger(__name__)

_MAX_UTTERANCE_BYTES = 16000 * 2 * 60  # 60 s of 16 kHz PCM16 — safety cap
_TTS_SAMPLE_RATE = 24000
_SENTENCE_ENDS = "。！？!?；;\n"
_SOFT_BREAKS = _SENTENCE_ENDS + "，、,"

#: tool → short spoken phrase for execution narration (unmapped tools stay silent
#: so raw tool JSON is never read aloud).
_NARRATE_START: dict[str, str] = {
    "scan_image": "开始扫描", "scan": "开始扫描", "start_scan": "开始扫描",
    "bias_spectroscopy": "开始采集能谱", "sts": "开始采集能谱",
    "auto_approach": "开始自动逼近", "approach": "开始自动逼近",
    "withdraw_tip": "撤针", "withdraw": "撤针",
    "tip_shaping": "开始修针", "shape_tip": "开始修针",
    "pulse": "施加电脉冲", "tip_pulse": "施加电脉冲",
    "move_tip": "移动针尖", "set_bias": "设置偏压",
    "set_setpoint": "设置电流设定点",
    # 注：``ask_user`` **刻意不在这张表里**。v1 里它由 ``interrupt`` 事件驱动
    # （``_announce_interrupt``），加进来会让 v1 说两遍 —— 而 v1 现在是生产。
    # v2 那边由 ``chat/engine_v2._event_frame`` 把它翻译成同样的 ``interrupt`` 帧，
    # 走的是下面那个已有的分支。
}
_NARRATE_END: dict[str, str] = {
    "scan_image": "扫描完成", "scan": "扫描完成", "start_scan": "扫描完成",
    "bias_spectroscopy": "能谱采集完成", "sts": "能谱采集完成",
    "auto_approach": "逼近完成", "approach": "逼近完成",
    "tip_shaping": "修针完成", "shape_tip": "修针完成",
    "pulse": "脉冲完成", "tip_pulse": "脉冲完成",
    "move_tip": "针尖已移动",
}


def _narration_for(tool: str, phase: str) -> str:
    table = _NARRATE_START if phase == "start" else _NARRATE_END
    return table.get((tool or "").strip().lower(), "")


#: wake words (lowercased) that arm the assistant in "wake" mode. Kept broad so
#: ASR spelling variants of "MAST" all trigger. Overridable via config in P5.
_WAKE_WORDS: tuple[str, ...] = ("mast", "马斯特", "马斯特", "嘿马斯特", "hey mast", "小马")


def _strip_wake(text: str) -> tuple[bool, str]:
    """(wake_found, command_after_wake). Matches the first wake word anywhere and
    returns whatever follows it (a one-shot 「MAST 扫一张图」)."""
    low = (text or "").lower()
    for w in _WAKE_WORDS:
        i = low.find(w)
        if i >= 0:
            rest = text[i + len(w):].lstrip(" ，,。.:：!！?？、")
            return True, rest.strip()
    return False, ""


def _rightmost(buf: str, chars: str) -> int:
    best = -1
    for c in chars:
        i = buf.rfind(c)
        if i > best:
            best = i
    return best


class _SentenceAggregator:
    """Coalesce a token stream into speakable sentences. Emits as soon as a
    sentence boundary is available past a minimum length (so TTS isn't called on
    single characters), with a hard soft-break flush for run-ons."""

    def __init__(self, min_len: int = 8, hard_len: int = 60) -> None:
        self._buf = ""
        self._min = min_len
        self._hard = hard_len

    def push(self, text: str) -> list[str]:
        self._buf += text or ""
        out: list[str] = []
        while True:
            stripped = self._buf.strip()
            if len(stripped) >= self._min:
                idx = _rightmost(self._buf, _SENTENCE_ENDS)
                if idx >= 0:
                    out.append(self._buf[: idx + 1].strip())
                    self._buf = self._buf[idx + 1:]
                    continue
            if len(self._buf) >= self._hard:
                idx = _rightmost(self._buf, _SOFT_BREAKS)
                if idx < 0:
                    idx = len(self._buf) - 1
                out.append(self._buf[: idx + 1].strip())
                self._buf = self._buf[idx + 1:]
                continue
            break
        return [s for s in out if s]

    def flush(self) -> list[str]:
        s = self._buf.strip()
        self._buf = ""
        return [s] if s else []


class VoiceSessionOrchestrator:
    def __init__(
        self,
        *,
        voice_cfg: Any,
        conversation_engine: Any,
        conversation_id: str,
        send: Callable[[Any], Awaitable[None]],
        hitl_resolver_factory: Optional[Callable[[str, threading.Event], Any]] = None,
    ) -> None:
        self._cfg = voice_cfg
        self._engine = conversation_engine
        self._conv_id = conversation_id
        self._send = send
        # (conversation_id, abort_event) -> resolver | None. Voice used to hit a
        # DANGEROUS gate, say "请在审批面板确认后继续" and stop — while the
        # interrupt was never published, so that panel stayed empty and the
        # sentence was simply untrue. With a resolver the pause is published,
        # the turn waits, and an approval carries it through.
        self._hitl_factory = hitl_resolver_factory

        # session params (overridable via hello / set_mode)
        self._mode: str = getattr(voice_cfg, "default_mode", "ptt") if voice_cfg else "ptt"
        self._voice: str = getattr(voice_cfg, "default_voice", "Cherry") if voice_cfg else "Cherry"
        self._narrate: bool = bool(getattr(voice_cfg, "narrate_execution", True)) if voice_cfg else True
        self._language: Optional[str] = None

        # provider config (forward-compatible before the P5 config extension)
        self._api_key: str = (getattr(voice_cfg, "api_key", "") or "") if voice_cfg else ""
        self._streaming: bool = bool(getattr(voice_cfg, "streaming", True)) if voice_cfg else True
        self._tts_model_rt: str = getattr(voice_cfg, "tts_model_rt", "qwen3-tts-flash-realtime") if voice_cfg else "qwen3-tts-flash-realtime"
        self._asr_model_rt: str = getattr(voice_cfg, "asr_model_rt", "qwen3-asr-flash-realtime") if voice_cfg else "qwen3-asr-flash-realtime"
        self._batch_client: Any = None
        self._realtime_tts_ok: Optional[bool] = None  # None=untried, True/False sticky

        # runtime state
        self._audio_buf = bytearray()
        self._armed = False  # wake mode: a wake word has been heard, awaiting command
        self._turn_task: Optional[asyncio.Task] = None
        self._probe_task: Optional[asyncio.Task] = None
        # realtime streaming ASR (opened in the background; batch on-commit until/if ready)
        self._asr: Any = None
        self._asr_reader: Optional[asyncio.Task] = None
        self._asr_open_task: Optional[asyncio.Task] = None
        self._use_realtime_asr = False
        self._transcribing = False   # a batch commit is being transcribed
        self._turn_running = False   # a turn (run/route) is in progress
        self._abort = threading.Event()
        self._speak_cancel = asyncio.Event()
        self._closed = False

    # ── lifecycle ──────────────────────────────────────────────────────
    async def start(self, hello: HelloFrame) -> None:
        self._mode = hello.mode or self._mode
        if hello.voice:
            self._voice = hello.voice
        self._narrate = bool(hello.narrate)
        self._language = hello.language
        if hello.conversation_id:
            self._conv_id = hello.conversation_id

        degraded = not self._api_key or self._engine is None
        await self._emit(
            HelloAck(
                mode=self._mode, voice=self._voice, streaming=self._streaming,
                narrate=self._narrate, degraded=degraded,
                message=("语音子系统未就绪（缺少 DashScope 密钥或对话引擎）" if degraded else ""),
            )
        )
        if degraded:
            await self._emit(DegradedFrame(
                message="语音不可用", reason="no_key" if not self._api_key else "no_engine"))
            await self._state("degraded")
            return
        # probe realtime TTS once, off the critical path (so an entitled account
        # auto-upgrades to streaming without a silent first turn on a denied one).
        if self._streaming and self._realtime_tts_ok is None:
            self._probe_task = asyncio.create_task(self._probe_realtime())
        # open the realtime streaming ASR in the background (a slow-VPN handshake
        # must NOT delay the listening state); until it's ready, commits use batch.
        if self._streaming and self._api_key:
            self._asr_open_task = asyncio.create_task(self._try_open_realtime_asr())
        await self._state(self._standby())

    async def close(self) -> None:
        self._closed = True
        self._abort.set()
        self._speak_cancel.set()
        for t in (self._turn_task, self._probe_task, self._asr_reader, self._asr_open_task):
            if t is not None and not t.done():
                t.cancel()
        if self._asr is not None:
            try:
                await self._asr.close()
            except Exception:
                pass
        if self._batch_client is not None:
            try:
                self._batch_client.close()
            except Exception:
                pass

    # ── inbound frames (from the route) ────────────────────────────────
    async def on_audio(self, pcm16: bytes) -> None:
        if self._closed or not pcm16:
            return
        # always buffer (batch fallback safety net) + stream to realtime ASR if up
        if len(self._audio_buf) < _MAX_UTTERANCE_BYTES:
            self._audio_buf.extend(pcm16)
        if self._use_realtime_asr and self._asr is not None:
            await self._asr.send_audio(pcm16)

    def _busy(self) -> bool:
        return self._turn_running

    async def on_commit(self) -> None:
        """Endpoint (PTT release / client-VAD speech end).

        Realtime ASR: flush the streaming recognizer — the final arrives via the
        ASR reader. Batch: transcribe the buffered utterance. The turn then runs as
        a background task so the route stays responsive to barge-in / stop."""
        if self._closed:
            return
        if self._use_realtime_asr and self._asr is not None:
            await self._asr.commit()
            return
        if self._busy() or self._transcribing:
            return
        self._transcribing = True
        self._turn_task = asyncio.create_task(self._batch_commit())

    async def _batch_commit(self) -> None:
        try:
            pcm = bytes(self._audio_buf)
            self._audio_buf.clear()
            if len(pcm) < 3200:  # < ~100 ms — ignore stray taps
                await self._state(self._standby())
                return
            await self._state("transcribing")
            text = await asyncio.to_thread(self._batch_transcribe, pcm)
            await self._on_final_text(text)
        finally:
            self._transcribing = False

    async def _on_final_text(self, text: str) -> None:
        """A finalized transcript (realtime ASR or batch) → run/route a turn.

        Awaits the turn (so the caller — batch-commit task or ASR reader — blocks
        through it). That is fine: barge-in / stop arrive on the ROUTE's receive
        loop (a separate channel from the ASR reader) and halt the turn via the
        abort / speak-cancel flags, not by unblocking this await."""
        text = (text or "").strip()
        await self._emit(AsrFinal(text=text))
        if not text:
            await self._state(self._standby())
            return
        if self._turn_running:
            return
        await self._route_turn(text)

    async def _route_turn(self, text: str) -> None:
        if self._mode == "wake":
            await self._handle_wake(text)
        else:
            await self._run_turn(text)

    async def _handle_wake(self, text: str) -> None:
        """Wake-word gate: in standby, only a transcript containing the wake word
        proceeds. 「MAST 扫一张图」 runs in one shot; a bare 「MAST」 arms the next
        utterance. After any command the assistant returns to standby."""
        if self._armed:
            self._armed = False
            await self._emit(WakeFrame(armed=False))
            await self._run_turn(text)
            return
        found, rest = _strip_wake(text)
        if not found:
            await self._state("listening")  # ignore chatter; stay in standby
            return
        if rest:
            await self._run_turn(rest)  # one-shot; stays disarmed
        else:
            self._armed = True
            await self._emit(WakeFrame(armed=True))
            await self._emit(Narration(text="已唤醒，请讲"))
            await self._state("listening")

    async def on_text(self, content: str) -> None:
        content = (content or "").strip()
        if self._closed or self._busy() or not content:
            return
        self._turn_task = asyncio.create_task(self._run_turn(content))

    async def on_barge_in(self) -> None:
        self._speak_cancel.set()
        self._abort.set()
        self._audio_buf.clear()  # the interrupted utterance's tail is stale
        await self._state(self._standby())

    async def on_stop(self) -> None:
        self._speak_cancel.set()
        self._abort.set()
        t = self._turn_task
        if t is not None and not t.done():
            t.cancel()
        await self._state(self._standby())

    async def on_set_mode(self, mode: str) -> None:
        self._mode = mode
        self._armed = False
        await self._state(self._standby())

    # ── turn execution (streaming: speak sentences as they form) ────────
    async def _run_turn(self, user_text: str) -> None:
        self._abort.clear()
        self._speak_cancel.clear()
        self._turn_running = True
        agg = _SentenceAggregator()
        speaker = _TurnSpeaker(self)
        spoke_any = False
        interrupted = False
        final_text = ""
        try:
            await self._state("thinking")
            await speaker.begin()
            async for ev in self._iter_turn_events(user_text):
                if self._speak_cancel.is_set():
                    break
                etype = ev.get("type")
                if etype == "token":
                    txt = ev.get("text") or ""
                    if txt:
                        await self._emit(ReplyDelta(text=txt))
                        for sentence in agg.push(txt):
                            if not speaker.speaking:
                                await self._state("speaking")
                            await speaker.say(sentence)
                            spoke_any = True
                elif etype == "tool_start" and self._narrate:
                    phrase = _narration_for(ev.get("name", ""), "start")
                    if phrase:
                        await self._emit(Narration(text=phrase))
                        await speaker.say(phrase)
                elif etype == "tool_end" and self._narrate:
                    phrase = _narration_for(ev.get("name", ""), "end")
                    if phrase:
                        await self._emit(Narration(text=phrase))
                        await speaker.say(phrase)
                elif etype == "interrupt":
                    # ``waiting`` means the turn is about to BLOCK on the
                    # operator and will carry on once they answer — so it is
                    # announced, but the turn is not over.
                    waiting = bool(ev.get("waiting"))
                    if not waiting:
                        interrupted = True
                    await self._announce_interrupt(
                        ev.get("payload") or {}, speaker, waiting=waiting)
                elif etype == "notice":
                    text = str(ev.get("text") or "")
                    if text:
                        await self._emit(Narration(text=text))
                        await speaker.say(text)
                elif etype == "error":
                    await self._emit(DegradedFrame(
                        message=ev.get("message", "对话出错"), reason="turn_error"))
                elif etype == "final":
                    final_text = (ev.get("text") or "").strip()
            # flush any tail sentence; if tokens never streamed, speak the final.
            if not self._speak_cancel.is_set() and not interrupted:
                tail = agg.flush()
                for sentence in tail:
                    if not speaker.speaking:
                        await self._state("speaking")
                    await speaker.say(sentence)
                    spoke_any = True
                if not spoke_any and final_text:
                    await self._state("speaking")
                    await speaker.say(final_text)
        except asyncio.CancelledError:
            await speaker.cancel()
        except Exception as exc:  # noqa: BLE001 — never break the socket
            logger.warning("voice turn failed: %s", exc)
            await self._emit(DegradedFrame(message=str(exc), reason="turn_error"))
        finally:
            await speaker.end()
            self._turn_running = False
            self._turn_task = None
            if not self._closed:
                await self._state(self._standby())

    async def _iter_turn_events(self, user_text: str):
        """Bridge the SYNC ``stream_events`` generator into async land via a queue."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        # Bind the resolver to THIS turn's conversation and abort event. The
        # conversation can change mid-session (a hello frame may switch it), so
        # it is read now rather than at construction; the abort event is what
        # lets barge-in / stop cut short a wait for an operator.
        resolver = None
        if self._hitl_factory is not None:
            try:
                resolver = self._hitl_factory(self._conv_id, self._abort)
            except Exception as exc:  # noqa: BLE001 — never break a turn
                logger.warning("voice hitl resolver factory failed: %s", exc)

        def worker() -> None:
            try:
                for ev in self._engine.stream_events(
                    self._conv_id, user_text, abort=self._abort,
                    hitl_resolver=resolver,
                ):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(
                    q.put_nowait, {"type": "error", "message": str(exc)})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, sentinel)

        threading.Thread(target=worker, daemon=True, name="voice-turn").start()
        while True:
            ev = await q.get()
            if ev is sentinel:
                return
            yield ev

    async def _announce_interrupt(self, payload: dict, speaker: "_TurnSpeaker",
                                  *, waiting: bool = False) -> None:
        """Say what is being asked, and say TRUTHFULLY what happens next.

        Both halves used to be wrong. The payload was the raw HITLRequest, whose
        keys are ``action_requests`` / ``review_configs`` — so ``skill`` was
        missing and this announced 「需要人工确认：操作」, naming nothing. And it
        promised the panel would carry the run on, while private-chat interrupts
        were never published there at all.
        """
        skill = str(payload.get("skill") or payload.get("kind") or "操作")
        rationale = str(payload.get("rationale") or "")
        if waiting:
            spoken = f"需要人工确认：{skill}。请在审批面板处理，我等你的结果。"
        else:
            spoken = f"需要人工确认：{skill}。这次没有得到处理，本轮先停下了。"
        await self._emit(InterruptFrame(
            # LangGraph's own id: the panel's event_id is minted when the
            # resolver publishes, which is downstream of this.
            interrupt_id=str(payload.get("event_id") or payload.get("lg_id") or ""),
            skill=skill, rationale=rationale, spoken=spoken, payload=payload))
        await speaker.say(spoken)

    # ── batch client (works wherever batch voice works) ────────────────
    def _batch(self):
        if self._batch_client is None and self._api_key:
            try:
                from mast.voice import DashScopeVoiceClient

                self._batch_client = DashScopeVoiceClient(
                    api_key=self._api_key,
                    tts_model=getattr(self._cfg, "tts_model", "qwen3-tts-flash"),
                    asr_model=getattr(self._cfg, "asr_model", "qwen3-asr-flash"),
                    default_voice=self._voice,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("batch voice client build failed: %s", exc)
                self._batch_client = None
        return self._batch_client

    def _batch_transcribe(self, pcm16_16k: bytes) -> str:
        client = self._batch()
        if client is None:
            return ""
        try:
            wav = _pcm_to_wav(pcm16_16k, rate=16000)
            result = client.transcribe(wav, mime="wav")
            return str(result.get("text", "") or "")
        except Exception as exc:  # noqa: BLE001
            logger.info("batch transcribe failed: %s", exc)
            return ""

    def _batch_synthesize(self, text: str) -> bytes:
        client = self._batch()
        if client is None:
            return b""
        try:
            return client.synthesize(text, voice=self._voice)
        except Exception as exc:  # noqa: BLE001
            logger.info("batch synthesize failed: %s", exc)
            return b""

    async def _speak_batch_sentence(self, text: str) -> None:
        """Batch synth ONE sentence → 24 kHz WAV → strip header → PCM frames down."""
        if self._speak_cancel.is_set() or not text.strip():
            return
        wav = await asyncio.to_thread(self._batch_synthesize, text)
        if not wav:
            return
        pcm = _wav_to_pcm(wav)
        step = _TTS_SAMPLE_RATE * 2 // 20  # ~50 ms frames
        for i in range(0, len(pcm), step):
            if self._speak_cancel.is_set():
                break
            await self._emit(pcm[i:i + step])

    # ── realtime TTS one-time probe (background) ───────────────────────
    async def _probe_realtime(self) -> None:
        try:
            from mast.voice.realtime import DashScopeRealtimeTTS

            tts = DashScopeRealtimeTTS(
                self._api_key, model=self._tts_model_rt, voice=self._voice)
            await tts.open(voice=self._voice, connect_timeout=10.0)
            await tts.append_text("。")
            await tts.finish()
            got = False
            async for frame in tts.audio_frames():
                if frame:
                    got = True
                    break
            await tts.close()
            self._realtime_tts_ok = bool(got)
            logger.info("realtime TTS probe: %s",
                        "available" if got else "unavailable → batch fallback")
        except Exception as exc:  # noqa: BLE001
            logger.info("realtime TTS probe failed (%s) → batch fallback", exc)
            self._realtime_tts_ok = False

    # ── realtime streaming ASR (client is the endpoint authority) ──────
    async def _try_open_realtime_asr(self) -> None:
        """Open a persistent realtime ASR (manual-commit mode: the client's PTT /
        VAD drives the endpoint via ``on_commit`` → ``asr.commit()``). On any
        failure (e.g. not entitled) stays on batch-on-commit."""
        try:
            from mast.voice.realtime import DashScopeRealtimeASR

            asr = DashScopeRealtimeASR(
                self._api_key, model=self._asr_model_rt, language=self._language)
            await asr.open(vad=False, connect_timeout=20.0)
            if self._closed:
                await asr.close()
                return
            self._asr = asr
            self._use_realtime_asr = True
            self._asr_reader = asyncio.create_task(self._asr_event_loop())
            logger.info("realtime ASR streaming active")
        except Exception as exc:  # noqa: BLE001
            logger.info("realtime ASR unavailable (%s) → batch ASR on commit", exc)
            self._use_realtime_asr = False
            self._asr = None

    async def _asr_event_loop(self) -> None:
        """Consume realtime ASR events: partial → live caption; final → run a turn."""
        try:
            async for ev in self._asr.events():
                if self._closed:
                    break
                t = ev.get("type")
                if t == "partial":
                    await self._emit(AsrPartial(
                        text=ev.get("text", ""), emotion=ev.get("emotion", "")))
                elif t == "final":
                    self._audio_buf.clear()
                    await self._on_final_text(ev.get("text", ""))
                elif t == "error":
                    logger.info("realtime ASR error: %s", ev.get("message"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("realtime ASR reader ended: %s", exc)
        finally:
            # connection gone → subsequent commits fall back to batch transcription
            self._use_realtime_asr = False

    # ── outbound helpers ────────────────────────────────────────────────
    async def _emit(self, frame: Any) -> None:
        if self._closed:
            return
        try:
            if isinstance(frame, (bytes, bytearray)):
                await self._send(bytes(frame))
            else:
                await self._send(frame.model_dump())
        except Exception as exc:  # noqa: BLE001 — client gone / socket closed
            logger.debug("voice emit failed: %s", exc)
            self._closed = True

    async def _state(self, state: str) -> None:
        await self._emit(StateFrame(state=state))  # type: ignore[arg-type]

    def _standby(self) -> str:
        """Where the machine rests between turns: continuous modes keep listening;
        PTT goes idle."""
        return "listening" if self._mode in ("duplex", "wake") else "idle"


# ── per-turn speaker: realtime streaming with a batch fallback ─────────────
class _TurnSpeaker:
    """Speaks a turn's sentences. Uses realtime TTS (one persistent WS fed per
    sentence, drained in the background) ONLY when the one-time probe confirmed
    it; otherwise synthesizes each sentence via the batch client. Both stream
    audio down as it is produced."""

    def __init__(self, orch: VoiceSessionOrchestrator) -> None:
        self._o = orch
        self._rt: Any = None
        self._drain: Optional[asyncio.Task] = None
        self._use_rt = False
        self._began = False
        self._any_audio = False

    @property
    def speaking(self) -> bool:
        return self._began

    async def begin(self) -> None:
        self._began = True
        await self._o._emit(TtsBegin(sample_rate=_TTS_SAMPLE_RATE))
        if self._o._realtime_tts_ok is True and self._o._api_key and self._o._streaming:
            try:
                from mast.voice.realtime import DashScopeRealtimeTTS

                self._rt = DashScopeRealtimeTTS(
                    self._o._api_key, model=self._o._tts_model_rt, voice=self._o._voice)
                await self._rt.open(voice=self._o._voice)
                self._use_rt = True
                self._drain = asyncio.create_task(self._drain_rt())
            except Exception as exc:  # noqa: BLE001
                logger.info("realtime TTS open failed (%s) → batch", exc)
                self._use_rt = False
                self._rt = None

    async def _drain_rt(self) -> None:
        try:
            async for frame in self._rt.audio_frames():
                if self._o._speak_cancel.is_set():
                    await self._rt.cancel()
                    break
                if frame:
                    self._any_audio = True
                    await self._o._emit(frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("realtime TTS drain ended: %s", exc)

    async def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self._o._speak_cancel.is_set():
            return
        if self._use_rt and self._rt is not None:
            await self._rt.append_text(text)
        else:
            await self._o._speak_batch_sentence(text)

    async def cancel(self) -> None:
        self._o._speak_cancel.set()
        if self._rt is not None:
            try:
                await self._rt.cancel()
            except Exception:
                pass
        if self._drain is not None and not self._drain.done():
            self._drain.cancel()

    async def end(self) -> None:
        if self._use_rt and self._rt is not None:
            try:
                await self._rt.finish()
                if self._drain is not None:
                    await self._drain
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._rt.close()
            except Exception:
                pass
            if not self._any_audio:
                # realtime produced nothing → don't use it again this session.
                self._o._realtime_tts_ok = False
        if self._began:
            await self._o._emit(TtsDone())


# ── WAV <-> PCM helpers (in-memory) ────────────────────────────────────────
def _pcm_to_wav(pcm: bytes, *, rate: int, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_to_pcm(wav_bytes: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.readframes(w.getnframes())
    except Exception:
        return wav_bytes  # not a WAV container (already raw?) — return as-is


__all__ = ["VoiceSessionOrchestrator"]
