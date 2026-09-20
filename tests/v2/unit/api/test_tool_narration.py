"""群聊 must read as sentences, not as clipped argument soup.

SYMPTOM (群聊面板)
------------------
群聊显示**太多太杂**，应转写自然语言 + 参数折叠。Three shapes were named:

    show_plan_on_map(steps=[{'kind': 'scan', 'label': '+1V 50nm', 'w_m': 5.0, , )
    {"seqno": 1036, "progress": {"seqno": 1036, "scan_id": "…", "line_idx": 2, …}}
    [HANDOFF → data_processing] 5-point STS grid acquired (40 nm spacing, …)

The first was built by::

    preview = ", ".join(f"{k}={v}" for k, v in list(targs.items())[:4])[:120]
    text    = f"{tname}({preview})"

Two silent losses in two lines: the join is cut at 120 characters and the ``)``
pasted on AFTERWARDS (hence the ``, )``), and every argument past the fourth
disappears without a trace. Nothing in the rendered string distinguishes "this
call had two arguments" from "this call had nine and you are seeing four of
them, one of them half-way through".

The second is the raw JSON the tool handed the model — correct for an LLM,
unreadable for a person watching a run.

WHAT THESE TESTS PIN
--------------------
* a summary NEVER invents behaviour — an unmapped tool gets its own name and an
  argument count, which is true of any tool that will ever exist;
* the curated phrase map cannot rot into fiction: every key must be a REAL tool
  name taken from the agents' live tool lists (the same failure mode that let
  ``TOOL_ACCESS`` reference two tools that existed nowhere — see
  tests/v2/unit/agents/test_artifact_class_status.py);
* the arguments SURVIVE. Summarising the transcript must not mean deleting the
  detail from the only durable record of the run, so the sidecar round-trips
  through the conversation store and back out of the API;
* nothing is cut without saying so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.api.tool_narration import (  # noqa: E402
    NARRATORS,
    narrate_tool_call,
    narrate_tool_result,
    summarize_args,
)


# ════════════════════════════════════════════════════════════════════════
# The summary may never claim more than it knows
# ════════════════════════════════════════════════════════════════════════

class TestSummariesAreHonest:
    def test_an_unknown_tool_gets_its_name_and_a_count_only(self):
        """This system has ~490 tools and 66 curated phrases, so the fallback is
        the COMMON path. It must be a sentence that is true of any tool at all."""
        out = narrate_tool_call("SomeSkillNobodyHasRead", {"a": 1, "b": 2, "c": 3})
        assert out == "SomeSkillNobodyHasRead · 3 个参数"

    def test_an_unknown_tool_with_no_args_is_just_its_name(self):
        assert narrate_tool_call("QuitNanonis", {}) == "QuitNanonis"

    def test_the_tool_name_always_survives_for_an_unknown_tool(self):
        """The operator has to be able to tell WHICH tool ran. A summary that
        drops the name in favour of a vague phrase would be worse than the raw
        call it replaced."""
        for name in ("ConfigureLockIn", "AcquireSTS", "EmergencyRetract"):
            assert name in narrate_tool_call(name, {"x": 1})

    def test_a_curated_phrase_counts_real_steps_rather_than_guessing(self):
        """`show_plan_on_map` — the call from the report. The "（4 步）" is
        len(args["steps"]), not a guess."""
        args = {"steps": [{"kind": "scan"}] * 4}
        assert narrate_tool_call("show_plan_on_map", args) == "在扫描地图上预览计划路线（4 步）"

    def test_a_curated_phrase_omits_what_it_was_not_given(self):
        """LLMs leave optional keys out constantly. A summary reading
        "写入长期记忆 · None" is exactly the small lie this module refuses."""
        assert narrate_tool_call("memory_write", {}) == "写入长期记忆"
        assert "None" not in narrate_tool_call("start_experiment", {})
        assert "None" not in narrate_tool_call("create_plan", {"steps": []})

    def test_a_raising_narrator_falls_back_instead_of_breaking_the_stream(self,
                                                                         monkeypatch):
        """A malformed argument shape must cost a nice phrase, never a frame."""
        def _boom(_a):
            raise ValueError("bad shape")

        monkeypatch.setitem(NARRATORS, "memory_write", _boom)
        assert narrate_tool_call("memory_write", {"path": "x"}) == "memory_write · 1 个参数"

    def test_non_dict_args_do_not_crash(self):
        for weird in (None, [], "steps", 42):
            assert narrate_tool_call("show_plan_on_map", weird)


# ════════════════════════════════════════════════════════════════════════
# The phrase map cannot rot into fiction
# ════════════════════════════════════════════════════════════════════════

def test_every_curated_name_is_a_real_tool():
    """A curated phrase for a tool that does not exist is dead weight that
    silently never fires — and, worse, reads as documentation of a capability.

    This is the same guard ``TOOL_ACCESS`` needed and did not have: two of its
    keys named tools that existed nowhere in the tree, and the check that was
    supposed to catch that compared the map against a hard-coded copy of itself.
    Here the truth side is the agents' REAL assembled tool lists.
    """
    from mast.agents._shared.artifacts import agent_tool_names

    real = set().union(*agent_tool_names().values())
    assert len(real) > 100, "tool discovery degraded — this guard would pass vacuously"
    invented = sorted(k for k in NARRATORS if k not in real)
    assert invented == [], (
        f"tool_narration names {len(invented)} tool(s) that do not exist: {invented}")


# ════════════════════════════════════════════════════════════════════════
# Nothing is cut in half, and nothing is cut in silence
# ════════════════════════════════════════════════════════════════════════

class TestNoSilentTruncation:
    def test_a_summary_never_ends_mid_argument(self):
        """The reported artefact was a string ending "…'w_m': 5.0, , )" — a
        clipped join with a paren stuck on the end. A summary is generated text
        now, so no argument value can be cut in half by construction."""
        args = {f"k{i}": f"value-{i}" * 30 for i in range(12)}
        out = narrate_tool_call("ConfigureScan", args)
        assert out == "ConfigureScan · 12 个参数"
        assert ", )" not in out and "…" not in out

    def test_the_arguments_come_back_whole(self):
        """The panel is where the detail the summary dropped has to remain. If
        the expander showed a trimmed copy, this change would just be a nicer
        way of losing the record."""
        args = {"steps": [{"kind": "scan", "label": "+1V 50nm", "w_m": 5.0}] * 4}
        text, clipped = summarize_args(args)
        assert clipped is False
        assert json.loads(text) == args          # byte-for-byte the same object

    def test_an_oversized_payload_is_flagged_not_quietly_trimmed(self):
        big = {"blob": "x" * 40_000}
        text, clipped = summarize_args(big)
        assert clipped is True, "a clipped payload that reads as complete is the bug"
        assert len(text) < len(json.dumps(big))

    def test_empty_args_produce_no_panel(self):
        assert summarize_args({}) == ("", False)
        assert summarize_args(None) == ("", False)

    def test_a_digested_return_keeps_as_much_as_the_store_ever_kept(self):
        """A summarised row must not hold LESS than the old raw row did. The
        store clips a message's text to 8000 on the way in, so that — not the
        route's nominal 200k — is what "the full return" has always meant."""
        from mast.api.routes.orchestrator import _META_DETAIL_MAX

        import mast.chat.store as store_mod

        assert _META_DETAIL_MAX >= 8_000
        assert "[:8000]" in Path(store_mod.__file__).read_text(encoding="utf-8"), (
            "the store's own text cap moved — re-check _META_DETAIL_MAX against it")


# ════════════════════════════════════════════════════════════════════════
# A tool RETURN digest is a transcription, not an interpretation
# ════════════════════════════════════════════════════════════════════════

class TestResultDigest:
    def test_every_token_of_the_digest_appears_in_the_payload(self):
        payload = {"seqno": 1036, "line_idx": 2, "advancing": False,
                   "age_s": 31.2, "note": "扫描很可能已经停止"}
        digest = narrate_tool_result(json.dumps(payload, ensure_ascii=False))
        for key in payload:
            assert key in digest
        assert "1036" in digest and "31.2" in digest
        assert "扫描很可能已经停止" in digest, "the one field that MATTERS was dropped"

    def test_booleans_keep_their_json_spelling(self):
        """The digest claims to be verbatim. Python's "False" is not what the
        payload says."""
        digest = narrate_tool_result('{"advancing": false}')
        assert "advancing=false" in digest and "False" not in digest

    def test_nested_only_payloads_are_not_summarised_down_to_one_field(self):
        """The exact shape from the report: everything hides one level down.
        Reporting just "seqno=1036" would silently drop the entire payload."""
        raw = '{"seqno": 1036, "progress": {"seqno": 1036, "line_idx": 2}}'
        digest = narrate_tool_result(raw)
        assert "seqno=1036" in digest
        assert "另有 1 项" in digest, f"the nested payload vanished: {digest!r}"

    def test_prose_and_handoffs_are_left_alone(self):
        """A handoff note already reads fine; digesting it would only damage it.
        Returning "" tells the caller to show the text as it is."""
        assert narrate_tool_result(
            "[HANDOFF → data_processing] 5-point STS grid acquired") == ""
        assert narrate_tool_result("已保存草稿 Au111_report_v001.md") == ""
        assert narrate_tool_result("") == ""

    def test_malformed_json_is_left_alone_rather_than_guessed_at(self):
        assert narrate_tool_result('{"seqno": 1036, "progress"') == ""


# ════════════════════════════════════════════════════════════════════════
# The sidecar survives the round trip — summary ⇏ data loss
# ════════════════════════════════════════════════════════════════════════

class TestSidecarRoundTrip:
    def _store(self, tmp_path):
        from mast.chat.store import ConversationStore

        return ConversationStore(tmp_path / "conversations.db")

    def test_meta_round_trips_through_the_transcript(self, tmp_path):
        store = self._store(tmp_path)
        conv = store.create("_supervisor", kind="group")
        args_json, _ = summarize_args({"steps": [{"kind": "scan"}] * 4})
        meta = json.dumps({"tool": "show_plan_on_map", "args": args_json,
                           "args_clipped": False}, ensure_ascii=False)

        store.append_message(conv["conversation_id"], agent_id="instrument_control",
                             role="tool", text="在扫描地图上预览计划路线（4 步）",
                             meta=meta)

        (row,) = store.messages_for(conv["conversation_id"])
        assert row["text"] == "在扫描地图上预览计划路线（4 步）"
        back = json.loads(row["meta"])
        assert back["tool"] == "show_plan_on_map"
        assert json.loads(back["args"])["steps"] == [{"kind": "scan"}] * 4

    def test_a_row_written_without_meta_still_reads(self, tmp_path):
        """Every pre-2026-07-28 row has no sidecar. Those must keep rendering —
        with their old raw text and simply nothing to expand."""
        store = self._store(tmp_path)
        conv = store.create("_supervisor", kind="group")
        store.append_message(conv["conversation_id"], role="tool",
                             text="show_plan_on_map(steps=[{'kind': 'scan', , )")
        (row,) = store.messages_for(conv["conversation_id"])
        assert row["meta"] == ""
        assert row["text"].startswith("show_plan_on_map")

    def test_the_agent_activity_feed_carries_the_sidecar_too(self, tmp_path):
        """The per-agent 群聊 feed reads its own query — it had to learn the
        column separately, so it gets its own test."""
        store = self._store(tmp_path)
        conv = store.create("_supervisor", kind="group")
        store.append_message(conv["conversation_id"], agent_id="data_processing",
                             role="tool", text="绘制扫描图",
                             meta='{"tool": "plot_scan", "args": "{}"}')
        (row,) = store.agent_activity("data_processing")
        assert json.loads(row["meta"])["tool"] == "plot_scan"

    def test_an_old_database_gains_the_column(self, tmp_path):
        """ALTER-TABLE migration: CREATE TABLE IF NOT EXISTS is a no-op on a
        deployed DB, so a new column only ever arrives this way."""
        import sqlite3

        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY, agent_id TEXT, thread_id TEXT,
                title TEXT, kind TEXT, created_at TEXT, updated_at TEXT,
                last_message_preview TEXT, experiment_id TEXT, archived INTEGER);
            CREATE TABLE conversation_messages (
                conversation_id TEXT NOT NULL, seq INTEGER NOT NULL,
                agent_id TEXT DEFAULT '', role TEXT DEFAULT '',
                kind TEXT DEFAULT 'message', text TEXT DEFAULT '',
                t REAL DEFAULT 0, PRIMARY KEY (conversation_id, seq));
        """)
        conn.commit()
        conn.close()

        from mast.chat.store import ConversationStore

        store = ConversationStore(db)   # opening it runs the migration
        conv = store.create("_supervisor", kind="group")
        store.append_message(conv["conversation_id"], role="tool", text="x",
                             meta='{"tool": "plot_scan"}')
        (row,) = store.messages_for(conv["conversation_id"])
        assert json.loads(row["meta"])["tool"] == "plot_scan"


# ════════════════════════════════════════════════════════════════════════
# …and out through the API the panel actually reads
# ════════════════════════════════════════════════════════════════════════

def test_the_live_stream_emits_a_summary_plus_structured_args(tmp_path):
    """END TO END through the REAL stream, with the shape from the report.

    The old frame for this exact call was::

        show_plan_on_map(steps=[{'kind': 'scan', 'label': '+1V 50nm', 'w_m': 5.0, , )

    Reusing test_orchestrator.py's fakes on purpose: they drive the actual
    route, so this asserts what the panel would receive, not what a helper
    returns in isolation."""
    from tests.v2.unit.api.test_orchestrator import (  # noqa: PLC0415
        _AIMessage, _client, _FakeLiveApp, _frames, _live_ctx,
    )
    from mast.chat.store import ConversationStore

    # The route buckets on ``m.__class__.__name__``, so the fake has to carry
    # the real class name — that string IS the contract being exercised.
    class ToolMessage:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []
            self.id = None

    steps = [{"kind": "scan", "label": "+1V 50nm", "w_m": 5.0},
             {"kind": "sts", "label": "grid"}, {"kind": "scan"}, {"kind": "scan"}]

    class _Orch:
        def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
            yield (("instrument_control:abc",), {"agent": {"messages": [
                _AIMessage("", tool_calls=[
                    {"name": "show_plan_on_map", "args": {"steps": steps}}])]}})
            yield (("instrument_control:abc",), {"agent": {"messages": [
                ToolMessage('{"seqno": 1036, "progress": '
                             '{"seqno": 1036, "line_idx": 2}}')]}})

    store = ConversationStore(tmp_path / "conversations.db")
    app = _FakeLiveApp(orchestrator=_Orch(), conv_store=store)
    resp = _client(_live_ctx(app)).post("/api/agents/run-task",
                                        json={"task": "预览计划"})
    frames = _frames(resp)
    tool_frames = [f for f in frames
                   if f.get("kind") == "message" and f.get("role") == "tool"]
    assert tool_frames, f"no tool frame in the stream: {frames}"

    call = tool_frames[0]
    assert call["text"] == "在扫描地图上预览计划路线（4 步）"
    assert call["tool"] == "show_plan_on_map"
    assert ", )" not in call["text"], "the clipped-join artefact is back"
    # …and the arguments the summary dropped are RIGHT THERE, complete.
    assert json.loads(call["args"])["steps"] == steps
    assert call["args_clipped"] is False

    ret = tool_frames[1]
    assert ret["text"].startswith("seqno=1036")
    assert "另有 1 项" in ret["text"], "the nested progress payload vanished"
    assert '"line_idx": 2' in ret["detail"], "the raw return must stay reachable"


def test_group_transcript_endpoint_returns_the_sidecar(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mast.api.context import AppContext
    from mast.api.routes.orchestrator import router
    from mast.chat.store import ConversationStore

    store = ConversationStore(tmp_path / "conversations.db")
    conv = store.create("_supervisor", kind="group")
    store.append_message(conv["conversation_id"], agent_id="instrument_control",
                         role="tool", text="读取扫描进度",
                         meta='{"tool": "get_scan_progress", "args": "{}"}')

    ctx = AppContext()
    monkeypatch.setattr(ctx, "conversation_store", store, raising=False)
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")

    body = TestClient(app).get(
        "/api/agents/group-transcript",
        params={"conversation_id": conv["conversation_id"]}).json()
    if body.get("degraded"):
        pytest.skip("no conversation store wired into this AppContext shape")
    (entry,) = body["entries"]
    assert entry["text"] == "读取扫描进度"
    assert json.loads(entry["meta"])["tool"] == "get_scan_progress"
