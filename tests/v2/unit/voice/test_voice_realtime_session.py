"""Unit tests for the voice rewrite — the parts that need neither a microphone,
network, nor the live instrument graph:

  * WAV <-> PCM helpers round-trip
  * ConversationEngine.stream_events() unpacking + event emission (fake graph)
  * VoiceSessionOrchestrator turn flow + graceful degradation (fake engine)

Realtime DashScope audio itself is validated separately by the P0 smoke
(``scratchpad/voice_rt_smoke.py``) once the account has realtime entitlement.
"""

from __future__ import annotations

import asyncio
import types

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from mast.chat.engine import ConversationEngine
from mast.voice.session import (
    VoiceSessionOrchestrator,
    _SentenceAggregator,
    _pcm_to_wav,
    _strip_wake,
    _wav_to_pcm,
)
from mast.api.schemas_voice_ws import HelloFrame


# ── sentence aggregator ────────────────────────────────────────────────────
def test_sentence_aggregator_splits_on_punctuation():
    agg = _SentenceAggregator(min_len=8)
    # short fragment held back until it forms a long-enough sentence
    assert agg.push("好的，") == []
    out = agg.push("现在开始扫描。再采一条谱")
    assert out == ["好的，现在开始扫描。"]
    # tail flushed at end
    assert agg.flush() == ["再采一条谱"]


def test_sentence_aggregator_hard_break_on_runon():
    agg = _SentenceAggregator(min_len=8, hard_len=20)
    long = "这是一段没有句号的很长的文字需要在软断点强制切分继续说"
    out = agg.push(long)
    assert out  # forced a soft-break flush rather than buffering unbounded
    assert "".join(out) + "".join(agg.flush()) == long


# ── WAV helpers ────────────────────────────────────────────────────────────
def test_wav_pcm_roundtrip():
    pcm = bytes(range(256)) * 8
    wav = _pcm_to_wav(pcm, rate=16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert _wav_to_pcm(wav) == pcm


def test_wav_to_pcm_passthrough_on_raw():
    raw = b"\x01\x02\x03\x04"  # not a WAV container
    assert _wav_to_pcm(raw) == raw


# ── stream_events on a fake graph ──────────────────────────────────────────
class _FakeGraph:
    def __init__(self, items, final_msgs):
        self._items = items
        self._final = final_msgs

    def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
        return iter(self._items)

    def get_state(self, cfg):
        return types.SimpleNamespace(values={"messages": self._final})


class _FakeStore:
    def get(self, cid):
        return {"agent_id": "instrument_control", "thread_id": "t1",
                "title": "新对话", "last_message_preview": ""}

    def touch(self, *a, **k):
        pass


def test_stream_events_emits_token_tool_final():
    ns = ("agent:x",)
    ai_call = AIMessage(
        content="", tool_calls=[{"id": "c1", "name": "scan_image", "args": {}}])
    tool_res = ToolMessage(content="ok", name="scan_image", tool_call_id="c1")
    items = [
        (ns, "messages", (AIMessageChunk(content="正在"), {})),
        (ns, "messages", (AIMessageChunk(content="扫描"), {})),
        (ns, "updates", {"model": {"messages": [ai_call]}}),
        (ns, "updates", {"tools": {"messages": [tool_res]}}),
    ]
    final = [AIMessage(content="扫描完成，发现清晰的原子台阶。")]
    eng = ConversationEngine(
        graph_factory=lambda aid: _FakeGraph(items, final),
        checkpointer=None, store=_FakeStore())

    evs = list(eng.stream_events("conv1", "扫一张图"))
    types_seen = [e["type"] for e in evs]
    assert "token" in types_seen
    assert "".join(e["text"] for e in evs if e["type"] == "token") == "正在扫描"
    starts = [e for e in evs if e["type"] == "tool_start"]
    ends = [e for e in evs if e["type"] == "tool_end"]
    assert starts and starts[0]["name"] == "scan_image"
    assert ends and ends[0]["name"] == "scan_image"
    final_ev = evs[-1]
    assert final_ev["type"] == "final"
    assert "扫描完成" in final_ev["text"]
    assert final_ev["aborted"] is False


def test_stream_events_missing_conversation():
    eng = ConversationEngine(
        graph_factory=lambda aid: _FakeGraph([], []),
        checkpointer=None,
        store=types.SimpleNamespace(get=lambda cid: None))
    evs = list(eng.stream_events("nope", "hi"))
    assert evs == [{"type": "error", "message": "会话不存在"}]


# ── orchestrator turn flow (fake engine, batch fallback) ───────────────────
class _FakeEngine:
    def __init__(self, events):
        self._events = events

    def stream_events(self, conv_id, user_text, *, abort=None, hitl_resolver=None):
        yield from self._events


def _run(coro):
    return asyncio.run(coro)


def test_orchestrator_turn_flow_batch():
    frames: list = []

    async def send(f):
        frames.append(f)

    cfg = types.SimpleNamespace(
        default_mode="ptt", default_voice="Cherry", narrate_execution=True,
        api_key="sk-test", streaming=False,  # force batch path
        tts_model="qwen3-tts-flash", asr_model="qwen3-asr-flash")

    engine = _FakeEngine([
        {"type": "tool_start", "name": "scan_image"},
        {"type": "tool_end", "name": "scan_image"},
        # tokens stream the full reply incrementally (as a real provider does)
        {"type": "token", "text": "扫描完成，"},
        {"type": "token", "text": "发现清晰的台阶。"},
        {"type": "final", "text": "扫描完成，发现清晰的台阶。", "aborted": False},
    ])
    orch = VoiceSessionOrchestrator(
        voice_cfg=cfg, conversation_engine=engine,
        conversation_id="conv1", send=send)
    # stub batch synth so no network is touched
    orch._batch_synthesize = lambda text: _pcm_to_wav(b"\x01\x02" * 240, rate=24000)

    async def drive():
        await orch.start(HelloFrame(mode="ptt", voice="Cherry", narrate=True))
        await orch.on_text("扫一张图")
        if orch._turn_task is not None:
            await orch._turn_task
        await orch.close()

    _run(drive())

    kinds = [f["type"] for f in frames if isinstance(f, dict)]
    assert kinds[0] == "hello_ack"
    assert "state" in kinds
    # reply_delta tokens accumulate to the full reply
    reply = "".join(
        f["text"] for f in frames if isinstance(f, dict) and f["type"] == "reply_delta")
    assert reply == "扫描完成，发现清晰的台阶。"
    narrations = [f["text"] for f in frames if isinstance(f, dict) and f["type"] == "narration"]
    assert "开始扫描" in narrations and "扫描完成" in narrations
    assert any(f.get("type") == "tts_begin" for f in frames if isinstance(f, dict))
    assert any(isinstance(f, (bytes, bytearray)) for f in frames)  # audio frames
    assert any(f.get("type") == "tts_done" for f in frames if isinstance(f, dict))
    # ends back at idle (ptt)
    states = [f["state"] for f in frames if isinstance(f, dict) and f["type"] == "state"]
    assert states[-1] == "idle"


# ── HITL over voice ───────────────────────────────────────────────────────
#
# Voice used to announce 「需要人工确认：操作」 — naming nothing, because the
# payload was the raw HITLRequest with no ``skill`` — and then promise the run
# would continue from the approval panel, which never received the interrupt at
# all. Both halves are pinned here.
class _RecordingEngine:
    """Captures the resolver it is handed, then replays scripted events."""

    def __init__(self, events):
        self._events = events
        self.seen: list = []

    def stream_events(self, conv_id, user_text, *, abort=None, hitl_resolver=None):
        self.seen.append({"conv_id": conv_id, "abort": abort,
                          "resolver": hitl_resolver})
        yield from self._events


def _voice_cfg_batch():
    return types.SimpleNamespace(
        default_mode="ptt", default_voice="Cherry", narrate_execution=True,
        api_key="sk-test", streaming=False,
        tts_model="qwen3-tts-flash", asr_model="qwen3-asr-flash")


def _drive_one_turn(orch, text="打个脉冲"):
    async def drive():
        await orch.start(HelloFrame(mode="ptt", voice="Cherry", narrate=True))
        await orch.on_text(text)
        if orch._turn_task is not None:
            await orch._turn_task
        await orch.close()

    _run(drive())


def test_voice_hands_the_resolver_to_the_engine():
    frames: list = []

    async def send(f):
        frames.append(f)

    engine = _RecordingEngine([{"type": "final", "text": "好的。", "aborted": False}])
    made: list = []

    def factory(conv_id, abort):
        made.append((conv_id, abort))
        return lambda interrupted: None

    orch = VoiceSessionOrchestrator(
        voice_cfg=_voice_cfg_batch(), conversation_engine=engine,
        conversation_id="c-voice", send=send, hitl_resolver_factory=factory)
    orch._batch_synthesize = lambda t: _pcm_to_wav(b"\x00\x00" * 120, rate=24000)
    _drive_one_turn(orch)

    # Bound to THIS turn's conversation and abort event — the conversation can
    # change mid-session, and barge-in must be able to cut a wait short.
    assert made and made[0][0] == "c-voice"
    assert made[0][1] is orch._abort
    assert engine.seen[0]["resolver"] is not None
    assert engine.seen[0]["abort"] is orch._abort


def test_a_waiting_interrupt_names_the_skill_and_keeps_the_turn_alive():
    frames: list = []

    async def send(f):
        frames.append(f)

    engine = _RecordingEngine([
        {"type": "interrupt", "waiting": True,
         "payload": {"kind": "dangerous", "skill": "BiasPulse",
                     "rationale": "危险偏压", "lg_id": "lg-7"}},
        {"type": "token", "text": "已执行。"},
        {"type": "final", "text": "已执行。", "aborted": False},
    ])
    orch = VoiceSessionOrchestrator(
        voice_cfg=_voice_cfg_batch(), conversation_engine=engine,
        conversation_id="c1", send=send,
        hitl_resolver_factory=lambda c, a: (lambda i: {"decisions": []}))
    orch._batch_synthesize = lambda t: _pcm_to_wav(b"\x00\x00" * 120, rate=24000)
    _drive_one_turn(orch)

    intr = [f for f in frames if isinstance(f, dict) and f["type"] == "interrupt"]
    assert intr, "the operator must be told what is being waited for"
    assert intr[0]["skill"] == "BiasPulse"
    assert "BiasPulse" in intr[0]["spoken"]
    assert "等你的结果" in intr[0]["spoken"], "a waiting turn must not claim it stopped"
    assert intr[0]["interrupt_id"] == "lg-7"
    # The turn carried on: the post-approval reply was still spoken.
    reply = "".join(f["text"] for f in frames
                    if isinstance(f, dict) and f["type"] == "reply_delta")
    assert reply == "已执行。"


def test_an_unanswered_interrupt_says_the_turn_stopped():
    frames: list = []

    async def send(f):
        frames.append(f)

    engine = _RecordingEngine([
        {"type": "interrupt",
         "payload": {"kind": "dangerous", "skill": "SetBias", "rationale": ""}},
    ])
    orch = VoiceSessionOrchestrator(
        voice_cfg=_voice_cfg_batch(), conversation_engine=engine,
        conversation_id="c1", send=send)
    orch._batch_synthesize = lambda t: _pcm_to_wav(b"\x00\x00" * 120, rate=24000)
    _drive_one_turn(orch)

    intr = [f for f in frames if isinstance(f, dict) and f["type"] == "interrupt"]
    assert intr and intr[0]["skill"] == "SetBias"
    assert "本轮先停下" in intr[0]["spoken"]


def test_a_stale_approval_notice_is_spoken():
    frames: list = []

    async def send(f):
        frames.append(f)

    engine = _RecordingEngine([
        {"type": "notice", "text": "上一轮遗留的人工审批已过期，已按拒绝自动处理，现在继续。"},
        {"type": "final", "text": "好的。", "aborted": False},
    ])
    orch = VoiceSessionOrchestrator(
        voice_cfg=_voice_cfg_batch(), conversation_engine=engine,
        conversation_id="c1", send=send)
    orch._batch_synthesize = lambda t: _pcm_to_wav(b"\x00\x00" * 120, rate=24000)
    _drive_one_turn(orch)

    narrations = [f["text"] for f in frames
                  if isinstance(f, dict) and f["type"] == "narration"]
    assert any("已过期" in n for n in narrations)


def test_a_broken_resolver_factory_never_breaks_the_turn():
    frames: list = []

    async def send(f):
        frames.append(f)

    engine = _RecordingEngine([{"type": "final", "text": "好的。", "aborted": False}])

    def boom(conv_id, abort):
        raise RuntimeError("no store")

    orch = VoiceSessionOrchestrator(
        voice_cfg=_voice_cfg_batch(), conversation_engine=engine,
        conversation_id="c1", send=send, hitl_resolver_factory=boom)
    orch._batch_synthesize = lambda t: _pcm_to_wav(b"\x00\x00" * 120, rate=24000)
    _drive_one_turn(orch)

    assert engine.seen[0]["resolver"] is None
    assert any(f.get("type") == "tts_done" for f in frames if isinstance(f, dict))


def test_strip_wake():
    assert _strip_wake("MAST 扫一张图")[0] is True
    assert _strip_wake("MAST 扫一张图")[1] == "扫一张图"
    assert _strip_wake("马斯特，当前状态") == (True, "当前状态")
    assert _strip_wake("mast") == (True, "")
    assert _strip_wake("随便扫一张图") == (False, "")


def test_wake_mode_gating():
    frames: list = []

    async def send(f):
        frames.append(f)

    cfg = types.SimpleNamespace(
        default_mode="wake", default_voice="Cherry", narrate_execution=False,
        api_key="sk-test", streaming=False, tts_model="x", asr_model="y")
    engine = _FakeEngine([{"type": "final", "text": "好的。", "aborted": False}])
    orch = VoiceSessionOrchestrator(
        voice_cfg=cfg, conversation_engine=engine, conversation_id="c1", send=send)
    orch._batch_synthesize = lambda t: _pcm_to_wav(b"\x00\x00" * 120, rate=24000)
    scripted = iter(["随便说点什么", "MAST 扫一张图", "MAST", "扫一张图"])
    orch._batch_transcribe = lambda pcm: next(scripted)

    async def drive():
        await orch.start(HelloFrame(mode="wake"))
        for _ in range(4):
            await orch.on_audio(b"\x00" * 4000)
            await orch.on_commit()
            if orch._turn_task is not None:
                await orch._turn_task
        await orch.close()

    _run(drive())

    turns = [f for f in frames if isinstance(f, dict) and f["type"] == "tts_done"]
    armed = [f for f in frames if isinstance(f, dict) and f["type"] == "wake" and f["armed"]]
    # chatter ignored; "MAST 扫一张图" one-shot; bare "MAST" arms; next utterance runs
    assert len(turns) == 2
    assert len(armed) == 1


class _FakeAsr:
    def __init__(self, events):
        self._events = events
        self.sent: list = []
        self.committed = 0

    async def send_audio(self, pcm):
        self.sent.append(pcm)

    async def commit(self):
        self.committed += 1

    async def events(self):
        for e in self._events:
            yield e

    async def close(self):
        pass


def test_realtime_asr_streaming_path():
    frames: list = []

    async def send(f):
        frames.append(f)

    cfg = types.SimpleNamespace(
        default_mode="duplex", default_voice="Cherry", narrate_execution=False,
        api_key="sk-test", streaming=True, tts_model="x", asr_model="y")
    engine = _FakeEngine([{"type": "final", "text": "好的。", "aborted": False}])
    orch = VoiceSessionOrchestrator(
        voice_cfg=cfg, conversation_engine=engine, conversation_id="c1", send=send)
    orch._batch_synthesize = lambda t: _pcm_to_wav(b"\x00\x00" * 120, rate=24000)
    fake = _FakeAsr([
        {"type": "partial", "text": "扫一", "emotion": "neutral"},
        {"type": "partial", "text": "扫一张图", "emotion": "neutral"},
        {"type": "final", "text": "扫一张图", "emotion": ""},
    ])
    orch._asr = fake
    orch._use_realtime_asr = True

    async def drive():
        await orch.on_audio(b"\x01\x02" * 100)  # forwards to realtime ASR
        await orch._asr_event_loop()            # partials → captions, final → turn
        await orch.close()

    _run(drive())

    assert fake.sent, "mic audio should be forwarded to realtime ASR"
    partials = [f for f in frames if isinstance(f, dict) and f["type"] == "asr_partial"]
    assert partials and partials[-1]["text"] == "扫一张图"
    finals = [f for f in frames if isinstance(f, dict) and f["type"] == "asr_final"]
    assert finals and finals[0]["text"] == "扫一张图"
    assert any(f.get("type") == "tts_done" for f in frames if isinstance(f, dict))
    assert orch._use_realtime_asr is False  # reader end → falls back to batch


def test_orchestrator_degrades_without_key():
    frames: list = []

    async def send(f):
        frames.append(f)

    cfg = types.SimpleNamespace(
        default_mode="ptt", default_voice="Cherry", narrate_execution=True,
        api_key="", streaming=True, tts_model="x", asr_model="y")
    orch = VoiceSessionOrchestrator(
        voice_cfg=cfg, conversation_engine=None,
        conversation_id="", send=send)

    _run(orch.start(HelloFrame()))

    ack = frames[0]
    assert ack["type"] == "hello_ack" and ack["degraded"] is True
    assert any(f.get("type") == "degraded" for f in frames if isinstance(f, dict))
    states = [f["state"] for f in frames if isinstance(f, dict) and f["type"] == "state"]
    assert states and states[-1] == "degraded"
