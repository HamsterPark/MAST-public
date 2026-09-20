"""群聊内的压缩功能并未显性体现。

SYMPTOM AT THE TIME
-------------------
Context compaction WAS running in 群聊 (``MASTApp._chat_agent_middleware`` attaches
``make_compaction_middleware`` to every orchestrator agent), but nothing in the
system said so:

  * ``SummarizationMiddleware.before_model`` returns
    ``[RemoveMessage(REMOVE_ALL), summary, *preserved]`` and carries no record of
    how much history it just replaced;
  * the 群聊 SSE bridge saw that update, classified the summary message as a
    ``HumanMessage`` — "operator echo — already in `start`" — and DROPPED it;
  * so no ``compaction`` row was ever written to ``conversation_messages`` and
    ``grep -rn 'compact|summariz' frontend/src/components/agents/`` matched zero
    lines. The only trace anywhere was the memory sink, which overwrites ONE
    path (``summaries/chat-running.md``) and is not tied to the conversation.

The operator scrolling back through a long group chat was therefore reading a
history the system had silently rewritten, in a view that looked untouched.

WHAT THESE TESTS PIN
--------------------
  1. the middleware STAMPS the compaction onto the summary message, with counts
     derived from the real lists (not estimated), and survives a real
     ``stream_mode="updates", subgraphs=True`` graph stream — the exact shape the
     群聊 bridge drives;
  2. the bridge turns that stamp into a live ``compaction`` SSE frame AND a
     durable ``compaction`` transcript row (no record ⇒ nothing to display);
  3. the bridge no longer re-persists the preserved tail the compaction update
     re-emits — before this change a compaction on a RESUMED conversation
     duplicated up to ``keep`` rows of history;
  4. HONESTY: a compaction whose counts did not come through renders as a
     compaction with NO counts, never with a plausible-looking number.
"""

from __future__ import annotations

import json
import sys
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.orchestrator import (
    _COMPACTION_META_KEY,
    _compaction_event,
    _compaction_line,
    router as orch_router,
)
from mast.agents._shared.compaction_mw import (
    COMPACTION_META_KEY,
    make_compaction_middleware,
)


# ── the literal the bridge matches on must equal the one the mw writes ───────
def test_bridge_compaction_key_matches_middleware() -> None:
    """The bridge deliberately keeps this as a LITERAL (it stays free of heavy
    agent imports, same rule as ``_AUTO_BG_MARKER``). A silent drift between the
    two spellings would restore the exact #11 symptom — compaction running,
    nothing displayed — with every test still green."""
    assert _COMPACTION_META_KEY == COMPACTION_META_KEY


# ══════════════════════════════════════════════════════════════════════════════
# 1. the middleware records the compaction (through a REAL graph stream)
# ══════════════════════════════════════════════════════════════════════════════
def _fake_model(reply: str):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    return GenericFakeChatModel(messages=iter([AIMessage(reply)] * 50))


def _seed_history(turns: int):
    from langchain_core.messages import AIMessage, HumanMessage
    out = []
    for i in range(turns):
        out.append(HumanMessage(f"turn {i} " + "x" * 50))
        out.append(AIMessage(f"reply {i} " + "y" * 50))
    return out


def _compacting_agent(*, keep: int, trigger):
    from langchain.agents import create_agent
    mw = make_compaction_middleware(
        model_id="kimi-k2.6",
        summarizer_model=_fake_model("摘要：已完成 Au(111) 粗定位"),
        keep_messages=keep,
    )
    # Force the trigger so the compaction fires on a short scripted history
    # instead of on a real 180k-token conversation.
    mw.trigger = trigger
    mw._trigger_conditions = [trigger]
    return create_agent(model=_fake_model("ok"), tools=[], middleware=[mw])


def _stream_updates(agent, history):
    """Drive the agent EXACTLY as the 群聊 bridge does."""
    for _ns, chunk in agent.stream({"messages": history},
                                   stream_mode="updates", subgraphs=True):
        for node_name, node_state in (chunk or {}).items():
            yield node_name, ((node_state or {}).get("messages") or [])


def test_compaction_is_recorded_on_the_summary_message() -> None:
    """Before #11 this state update carried no evidence a compaction happened."""
    agent = _compacting_agent(keep=4, trigger=("messages", 6))
    events = [(_compaction_event(msgs), msgs)
              for _n, msgs in _stream_updates(agent, _seed_history(10))]
    hits = [(e, msgs) for e, msgs in events if e is not None]
    assert len(hits) == 1, "exactly one compaction should be reported"
    event, msgs = hits[0]

    # Counts are DERIVED from the two real lists, so they must be exact.
    assert event["before"] == 20          # the seeded history
    assert event["kept"] == 4             # keep_messages
    assert event["removed"] == 16         # 20 - 4
    assert event["removed"] + event["kept"] == event["before"]
    # …and they must describe the update actually emitted: RemoveMessage +
    # summary + the preserved tail.
    assert len(msgs) == event["kept"] + 2

    # The summary body is carried (upstream's English preamble stripped) so the
    # panel can offer 摘要正文 without the operator digging in the DB.
    assert event["summary"] == "摘要：已完成 Au(111) 粗定位"
    assert "Here is a summary" not in event["summary"]


def test_compaction_reports_token_threshold_only_when_it_has_one() -> None:
    """``tokens_before_estimate`` is an ESTIMATE and is labelled as one in the
    UI; ``trigger_tokens`` is only reported for a token-based trigger. Under a
    message-count trigger there IS no token threshold, so inventing one would be
    the exact dishonesty #11 is about."""
    by_msgs = _compacting_agent(keep=4, trigger=("messages", 6))
    event = next(e for _n, msgs in _stream_updates(by_msgs, _seed_history(10))
                 if (e := _compaction_event(msgs)) is not None)
    assert "trigger_tokens" not in event
    assert event["tokens_before_estimate"] > 0

    by_tokens = _compacting_agent(keep=4, trigger=("tokens", 200))
    event = next(e for _n, msgs in _stream_updates(by_tokens, _seed_history(10))
                 if (e := _compaction_event(msgs)) is not None)
    assert event["trigger_tokens"] == 200
    assert event["tokens_before_estimate"] >= 200   # it is why we compacted


def test_no_compaction_no_marker() -> None:
    """A run under the threshold must not produce a divider out of nowhere."""
    agent = _compacting_agent(keep=4, trigger=("messages", 500))
    assert all(_compaction_event(msgs) is None
               for _n, msgs in _stream_updates(agent, _seed_history(3)))


# ══════════════════════════════════════════════════════════════════════════════
# 2. the bridge displays it (live frame) and records it (durable transcript)
# ══════════════════════════════════════════════════════════════════════════════
def _msg(cls_name, content="", additional_kwargs=None, mid=None):
    """Minimal stand-in for a LangChain message as the bridge sees it: the bridge
    dispatches on ``__class__.__name__``, so the fake needs a real class name."""
    cls = type(cls_name, (), {})
    m = cls()
    m.content = content
    m.tool_calls = []
    m.additional_kwargs = additional_kwargs or {}
    m.id = mid
    return m


_SUMMARY_TEXT = "摘要：已在 Au(111) 上完成粗定位与 3 次 STS"


def _compaction_update(event: dict | None = None, *, preserved=("保留原文甲", "保留原文乙")):
    """A node update shaped exactly like the one SummarizationMiddleware emits."""
    if event is None:
        event = {"removed": 16, "kept": len(preserved), "before": 16 + len(preserved),
                 "summary": _SUMMARY_TEXT, "trigger_tokens": 168000,
                 "tokens_before_estimate": 171234}
    ak = {"lc_source": "summarization"}
    if event:
        ak[_COMPACTION_META_KEY] = event
    msgs = [_msg("RemoveMessage", "", mid="__remove_all__"),
            _msg("HumanMessage", f"Here is a summary…\n\n{_SUMMARY_TEXT}", ak)]
    for i, text in enumerate(preserved):
        msgs.append(_msg("AIMessage", text, mid=f"preserved-{i}"))
    return {"_MemorySinkSummarization.before_model": {"messages": msgs}}


class _CompactingOrchestrator:
    """Streams a supervisor hop, a COMPACTION, then a normal agent message."""

    def __init__(self, event: dict | None = None):
        self._event = event
        self._explicit = event is not None

    def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
        yield ((), {"supervisor": {"messages": [_msg("AIMessage", "路由到 instrument_control",
                                                     mid="sup-1")]}})
        upd = (_compaction_update(self._event) if self._explicit
               else _compaction_update())
        yield (("instrument_control:abc",), upd)
        yield (("instrument_control:abc",),
               {"agent": {"messages": [_msg("AIMessage", "扫描完成", mid="a-1")]}})


class _FakeLiveApp:
    def __init__(self, orchestrator, conv_store=None):
        self._orchestrator = orchestrator
        self._orch_abort = threading.Event()
        self._orch_running = False
        self._orch_interrupts = {"lock": threading.Lock(), "pending": {},
                                 "resolved": {}, "events": {}}
        self._conv_store = conv_store
        self._agents_api_state = {"lock": threading.Lock(), "holds": {},
                                  "interjects": [], "task": None}

    def _build_orchestrator(self, **kwargs):
        return self._orchestrator is not None


def _client(orchestrator, conv_store=None) -> TestClient:
    ctx = AppContext()
    ctx.live_app = _FakeLiveApp(orchestrator, conv_store)  # type: ignore[attr-defined]
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(orch_router, prefix="/api")
    return TestClient(app)


def _frames(resp) -> list[dict]:
    return [json.loads(line[len("data:"):].strip())
            for line in resp.iter_lines() if line.startswith("data:")]


def _run(c, task="扫一张图") -> list[dict]:
    with c.stream("POST", "/api/agents/run-task", json={"task": task}) as r:
        return _frames(r)


@pytest.fixture()
def store(tmp_path):
    from mast.chat.store import ConversationStore
    return ConversationStore(tmp_path / "exp.sqlite")


def test_compaction_surfaces_as_a_live_frame(store) -> None:
    """#11 core: the operator must be TOLD, while it is happening."""
    frames = _run(_client(_CompactingOrchestrator(), store))
    comp = [f for f in frames if f.get("kind") == "compaction"]
    assert len(comp) == 1, f"expected one compaction frame, got kinds " \
                           f"{[f.get('kind') for f in frames]}"
    f = comp[0]
    # attributed to the SUBGRAPH owner — compaction is per-agent context
    assert f["agent"] == "instrument_control"
    assert "压缩" in f["text"]
    assert "16" in f["text"] and "2" in f["text"]
    assert f["compaction"]["removed"] == 16
    assert f["compaction"]["kept"] == 2
    assert f["compaction"]["tokens_before_estimate"] == 171234
    assert f["compaction"]["summary"] == _SUMMARY_TEXT


def test_compaction_is_persisted_for_replay(store) -> None:
    """The divider has to survive a reload: on reconnect the client replays the
    transcript, and a conversation that was compacted must still SAY so — that
    is the case where the rewritten history is least obvious."""
    c = _client(_CompactingOrchestrator(), store)
    frames = _run(c)
    cid = next(f for f in frames if f["kind"] == "start")["conversation_id"]

    rows = store.messages_for(cid)
    comp = [r for r in rows if r["kind"] == "compaction"]
    assert len(comp) == 1, f"kinds persisted: {[r['kind'] for r in rows]}"
    assert comp[0]["agent_id"] == "instrument_control"
    assert "压缩" in comp[0]["text"]
    meta = json.loads(comp[0]["meta"])
    assert meta["removed"] == 16 and meta["kept"] == 2
    assert meta["summary"] == _SUMMARY_TEXT

    # and it comes back through the read endpoint the TS client actually calls
    # (LITERAL path — /agents/<x>/transcript would be shadowed by {agent_id})
    body = c.get("/api/agents/group-transcript",
                 params={"conversation_id": cid}).json()
    assert body["degraded"] is False
    replayed = [e for e in body["entries"] if e["kind"] == "compaction"]
    assert len(replayed) == 1
    assert json.loads(replayed[0]["meta"])["removed"] == 16


def test_compaction_does_not_duplicate_the_preserved_tail(store) -> None:
    """The compaction update re-emits the preserved messages verbatim. Rendering
    them as new messages would mean a compaction — whose whole job is to SHORTEN
    the history — appended up to ``keep`` duplicate rows to the durable
    transcript every time it fired on a resumed conversation."""
    c = _client(_CompactingOrchestrator(), store)
    frames = _run(c)
    cid = next(f for f in frames if f["kind"] == "start")["conversation_id"]

    texts = [r["text"] for r in store.messages_for(cid)]
    assert "保留原文甲" not in texts
    assert "保留原文乙" not in texts
    # the genuine post-compaction message still gets through
    assert "扫描完成" in texts
    # …and no message frame echoed them either
    msg_texts = [f.get("text") for f in frames if f.get("kind") == "message"]
    assert "保留原文甲" not in msg_texts


# ══════════════════════════════════════════════════════════════════════════════
# 3. honesty — no counts means no counts, not a guess
# ══════════════════════════════════════════════════════════════════════════════
def test_compaction_without_counts_states_only_what_it_knows(store) -> None:
    c = _client(_CompactingOrchestrator({"summary": ""}), store)
    frames = _run(c)
    f = next(f for f in frames if f.get("kind") == "compaction")
    assert "压缩" in f["text"]
    # no digits at all: an operator must not be able to read a count off a row
    # that never carried one.
    assert not any(ch.isdigit() for ch in f["text"]), f["text"]
    assert "removed" not in f["compaction"]
    assert "summary" not in f["compaction"]   # empty summary is not carried


def test_compaction_line_degrades_by_step() -> None:
    """Each fact drops out on its own; the divider itself never does."""
    assert _compaction_line({"removed": 16, "kept": 4}) == \
        "上下文压缩：较早的 16 条消息已被摘要替代，最近 4 条保留原文"
    assert _compaction_line({"removed": 16}) == \
        "上下文压缩：较早的 16 条消息已被摘要替代"
    assert _compaction_line({}) == "上下文压缩：此处较早的对话已被摘要替代"
    # a garbled count must not render as a number
    assert _compaction_line({"removed": "many", "kept": 4}) == \
        "上下文压缩：此处较早的对话已被摘要替代"


def test_ordinary_messages_are_not_mistaken_for_a_compaction() -> None:
    """An operator message that quotes the preamble is not a compaction —
    matching is on the stamped key, never on message text."""
    assert _compaction_event([_msg("HumanMessage", "Here is a summary of the "
                                                   "conversation to date: 我总结一下")]) is None
    assert _compaction_event([_msg("AIMessage", "已完成")]) is None
    assert _compaction_event([]) is None
    # a non-dict under the key is ignored rather than crashing the stream
    assert _compaction_event([_msg("HumanMessage", "x",
                                   {_COMPACTION_META_KEY: "nope"})]) is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. background runs merge into the SAME transcript — same rule applies there
# ══════════════════════════════════════════════════════════════════════════════
def test_background_run_marks_its_compaction_in_the_group_transcript() -> None:
    """A detached background run writes into the group conversation the operator
    is reading. Its bridge had the same "HumanMessage ⇒ operator echo ⇒ drop"
    skip, so a compaction inside one rewrote that transcript just as silently."""
    import threading as _th

    from mast.core.background_runs import BG_TAG, BackgroundRunManager
    from mast.core.runtime import (
        _background_compaction_event,
        _background_compaction_line,
    )

    # the detector/formatter agree with the foreground bridge's copies
    event = {"removed": 16, "kept": 2}
    assert _background_compaction_line(event) == _compaction_line(event)
    assert _background_compaction_line({}) == _compaction_line({})
    summary_msg = _msg("HumanMessage", "x", {_COMPACTION_META_KEY: event})
    assert _background_compaction_event([summary_msg]) == event
    assert _background_compaction_event([_msg("AIMessage", "普通消息")]) is None

    rows: list[tuple] = []

    def _run_fn(instruction, thread_id, agents, emit, abort):
        emit("literature", "agent", "找到 3 篇相关文献")
        mark = getattr(emit, "compaction", None)
        assert mark is not None, "emit must expose the compaction channel"
        mark("literature", _background_compaction_line(event))
        return "综述完成"

    mgr = BackgroundRunManager(
        run_fn=_run_fn,
        transcript_sink=lambda cid, kind, agent_id, role, text: rows.append(
            (cid, kind, agent_id, text)))
    rec = mgr.spawn(instruction="查文献", conversation_id="conv-1",
                    agents=("literature",))
    deadline = _th.Event()
    for _ in range(200):                      # ≤2 s, no fixed sleep
        if mgr.get(rec["run_id"])["status"] in ("done", "failed", "aborted"):
            break
        deadline.wait(0.01)
    assert mgr.get(rec["run_id"])["status"] == "done"

    comp = [r for r in rows if r[1] == "compaction"]
    assert len(comp) == 1, f"kinds emitted: {[r[1] for r in rows]}"
    assert comp[0][0] == "conv-1"             # merged into the group conversation
    assert comp[0][2] == "literature"
    assert comp[0][3] == f"{BG_TAG}上下文压缩：较早的 16 条消息已被摘要替代，最近 2 条保留原文"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
