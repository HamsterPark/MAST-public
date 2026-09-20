"""Unit tests for ConversationStore — focused on the durable group transcript.

The 群聊 (multi-agent orchestrator) run-task path used to stream ephemerally and
index nothing, so a conversation vanished on a tab switch and an agent's team-run
messages were invisible in its per-agent view. These tests pin the persistence
layer that restores both: a durable transcript table, re-readable per conversation
(reconnect) and per agent (the bridge).
"""

from __future__ import annotations

from mast.chat.store import ConversationStore


def _store(tmp_path):
    return ConversationStore(tmp_path / "exp.sqlite")


# ── group conversation rows ──────────────────────────────────────────────────
def test_create_group_conversation(tmp_path) -> None:
    s = _store(tmp_path)
    row = s.create("_supervisor", kind="group", title="测 Kondo", thread_id="agents-abc")
    assert row["kind"] == "group"
    assert row["thread_id"] == "agents-abc"
    # listable as a group, NOT mixed into a per-agent private list
    groups = s.list(kind="group")
    assert any(g["conversation_id"] == row["conversation_id"] for g in groups)
    assert s.list(agent_id="instrument_control") == []


# ── transcript append / read (reconnect) ─────────────────────────────────────
def test_transcript_append_and_read_in_order(tmp_path) -> None:
    s = _store(tmp_path)
    cid = s.create("_supervisor", kind="group", thread_id="agents-1")["conversation_id"]
    s.append_message(cid, kind="operator", role="user", text="在 Au(111) 上测 Kondo")
    s.append_message(cid, kind="message", agent_id="literature", role="agent", text="查到参数")
    s.append_message(cid, kind="message", agent_id="instrument_control", role="agent", text="扫描完成")
    s.append_message(cid, kind="done", role="assistant", text="任务完成")

    entries = s.messages_for(cid)
    assert [e["seq"] for e in entries] == [1, 2, 3, 4]  # stable per-conversation order
    assert [e["kind"] for e in entries] == ["operator", "message", "message", "done"]
    assert entries[0]["text"] == "在 Au(111) 上测 Kondo"
    assert entries[2]["agent_id"] == "instrument_control"


def test_transcript_isolated_per_conversation(tmp_path) -> None:
    s = _store(tmp_path)
    a = s.create("_supervisor", kind="group", thread_id="agents-a")["conversation_id"]
    b = s.create("_supervisor", kind="group", thread_id="agents-b")["conversation_id"]
    s.append_message(a, kind="message", agent_id="literature", text="A1")
    s.append_message(b, kind="message", agent_id="literature", text="B1")
    s.append_message(a, kind="message", agent_id="literature", text="A2")
    assert [e["text"] for e in s.messages_for(a)] == ["A1", "A2"]
    assert [e["text"] for e in s.messages_for(b)] == ["B1"]
    # seq restarts per conversation
    assert [e["seq"] for e in s.messages_for(a)] == [1, 2]
    assert [e["seq"] for e in s.messages_for(b)] == [1]


# ── per-agent bridge (group-activity) ────────────────────────────────────────
def test_agent_activity_filters_by_agent_and_kind(tmp_path) -> None:
    s = _store(tmp_path)
    cid = s.create("_supervisor", kind="group", title="任务X", thread_id="agents-x")["conversation_id"]
    s.append_message(cid, kind="operator", role="user", text="用户的话")  # not an agent message
    s.append_message(cid, kind="message", agent_id="instrument_control", role="agent", text="IC 在群聊里的发言")
    s.append_message(cid, kind="message", agent_id="literature", role="agent", text="文献 agent 的发言")
    s.append_message(cid, kind="done", role="assistant", text="完成")  # terminal, not an agent message

    ic = s.agent_activity("instrument_control")
    assert len(ic) == 1
    assert ic[0]["text"] == "IC 在群聊里的发言"
    assert ic[0]["conversation_title"] == "任务X"  # enriched with the group title
    # operator/done rows never leak into a per-agent feed
    assert all(e["text"] != "用户的话" for e in ic)
    assert s.agent_activity("data_processing") == []


def test_agent_activity_spans_multiple_group_runs(tmp_path) -> None:
    s = _store(tmp_path)
    c1 = s.create("_supervisor", kind="group", thread_id="agents-1")["conversation_id"]
    c2 = s.create("_supervisor", kind="group", thread_id="agents-2")["conversation_id"]
    s.append_message(c1, kind="message", agent_id="instrument_control", text="run1", t=100.0)
    s.append_message(c2, kind="message", agent_id="instrument_control", text="run2", t=200.0)
    feed = s.agent_activity("instrument_control")
    assert {e["text"] for e in feed} == {"run1", "run2"}
    # newest first
    assert feed[0]["text"] == "run2"


# ── per-conversation row cap (no unbounded growth of the shared DB) ──────────
def test_transcript_capped_per_conversation(tmp_path, monkeypatch) -> None:
    import mast.chat.store as store_mod
    monkeypatch.setattr(store_mod, "_MAX_TRANSCRIPT_ROWS", 5)
    s = _store(tmp_path)
    cid = s.create("_supervisor", kind="group", thread_id="agents-cap")["conversation_id"]
    for i in range(12):
        s.append_message(cid, kind="message", agent_id="literature", text=f"m{i}")
    rows = s.messages_for(cid)
    # only the most recent 5 survive (oldest trimmed on write), still in order
    assert [r["text"] for r in rows] == ["m7", "m8", "m9", "m10", "m11"]
    assert [r["seq"] for r in rows] == [8, 9, 10, 11, 12]  # seq stays monotonic


# ── separate DB file + one-time legacy migration ─────────────────────────────
def test_from_storage_uses_separate_db_and_migrates(tmp_path) -> None:
    """from_storage points at a SEPARATE mast_conversations.db (physical decoupling
    from the experiment DB) and migrates conversations that used to live in the
    shared experiment DB."""
    exp_db = tmp_path / "mast_experiments.db"
    legacy = ConversationStore(exp_db)  # OLD layout: chats in the experiment DB
    cid = legacy.create("instrument_control", kind="private", title="legacy chat")["conversation_id"]
    legacy.append_message(cid, kind="message", agent_id="instrument_control", text="old msg")
    gid = legacy.create("_supervisor", kind="group", thread_id="agents-old")["conversation_id"]
    legacy.append_message(gid, kind="message", agent_id="literature", text="old group msg")

    class _FakeStorage:
        _db_path = exp_db

    store = ConversationStore.from_storage(_FakeStorage())
    assert store._db_path.name == "mast_conversations.db"  # separate file
    assert store._db_path != exp_db
    # both conversations + their transcripts migrated
    ids = {c["conversation_id"] for c in store.list()} | {
        g["conversation_id"] for g in store.list(kind="group")}
    assert {cid, gid} <= ids
    assert [m["text"] for m in store.messages_for(cid)] == ["old msg"]
    assert [m["text"] for m in store.messages_for(gid)] == ["old group msg"]
    # legacy chat tables dropped from the experiment DB (data truly moved out)
    import sqlite3
    with sqlite3.connect(str(exp_db)) as raw:
        tbls = {r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "conversations" not in tbls and "conversation_messages" not in tbls


def test_from_storage_migration_idempotent(tmp_path) -> None:
    exp_db = tmp_path / "mast_experiments.db"
    legacy = ConversationStore(exp_db)
    legacy.create("instrument_control", kind="private", title="c1")

    class _S:
        _db_path = exp_db

    ConversationStore.from_storage(_S())
    store2 = ConversationStore.from_storage(_S())  # second build must not duplicate
    assert len(store2.list()) == 1


def test_migration_self_heals_after_failed_first_attempt(tmp_path) -> None:
    """REGRESSION: a failed first migration must NOT permanently strand legacy
    chats. Simulate the post-failure state — legacy DB still holds the chats, and
    the new DB already has a bootstrapped '新对话' row (as the chat engine seeds
    right after build). The old COUNT(*)>0 guard skipped migration forever here;
    the set-difference copy must still recover the stranded chats."""
    exp_db = tmp_path / "mast_experiments.db"
    legacy = ConversationStore(exp_db)
    cid = legacy.create("instrument_control", kind="private", title="stranded chat")["conversation_id"]
    legacy.append_message(cid, kind="message", agent_id="instrument_control", text="stranded msg")

    # pre-seed the NEW db with a bootstrap row (mimics _build_chat_engine seeding
    # a default conversation after a migration that errored out)
    conv_db = tmp_path / "mast_conversations.db"
    ConversationStore(conv_db).create("instrument_control", kind="private", title="新对话")

    class _S:
        _db_path = exp_db

    healed = ConversationStore.from_storage(_S())
    titles = {c["title"] for c in healed.list()}
    assert "stranded chat" in titles  # recovered despite the pre-existing bootstrap row
    assert "新对话" in titles          # bootstrap row preserved, not duplicated/lost
    assert len(healed.list()) == 2
    assert [m["text"] for m in healed.messages_for(cid)] == ["stranded msg"]


def test_migration_picks_up_late_legacy_rows(tmp_path) -> None:
    """A row written to the legacy DB AFTER the first split-boot (older build run
    in parallel / a downgrade) is still migrated — the copy is by set-difference,
    not a one-shot empty guard."""
    exp_db = tmp_path / "mast_experiments.db"
    ConversationStore(exp_db).create("instrument_control", kind="private", title="first")

    class _S:
        _db_path = exp_db

    s1 = ConversationStore.from_storage(_S())
    assert len(s1.list()) == 1
    # an old build re-creates the legacy table and writes a late row
    ConversationStore(exp_db).create("instrument_control", kind="private", title="late")
    s2 = ConversationStore.from_storage(_S())
    assert {c["title"] for c in s2.list()} == {"first", "late"}


# ── experiment_id is a provenance tag, NOT a scope (chat spans experiments) ───
def test_experiment_id_is_provenance_not_filter(tmp_path) -> None:
    s = _store(tmp_path)
    s.create("instrument_control", kind="private", title="exp A chat", experiment_id="expA")
    s.create("instrument_control", kind="private", title="exp B chat", experiment_id="expB")
    s.create("instrument_control", kind="private", title="no-exp chat")  # experiment_id None
    # list(agent_id=...) returns ALL of them regardless of experiment_id — a chat
    # is never hidden by the "current" experiment, so it spans experiments.
    titles = {c["title"] for c in s.list(agent_id="instrument_control")}
    assert titles == {"exp A chat", "exp B chat", "no-exp chat"}
    # the tag is preserved for provenance
    rows = {c["title"]: c["experiment_id"] for c in s.list(agent_id="instrument_control")}
    assert rows["exp A chat"] == "expA" and rows["no-exp chat"] is None


# ── delete cascades the transcript ───────────────────────────────────────────
def test_delete_cascades_transcript(tmp_path) -> None:
    s = _store(tmp_path)
    cid = s.create("_supervisor", kind="group", thread_id="agents-del")["conversation_id"]
    s.append_message(cid, kind="message", agent_id="literature", text="will be purged")
    assert s.messages_for(cid)
    assert s.delete(cid) is True
    assert s.messages_for(cid) == []  # no orphan transcript rows
    assert s.agent_activity("literature") == []
