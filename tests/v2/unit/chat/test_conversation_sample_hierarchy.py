"""— 实验/样品/群聊 的层次不对。

Operator, verbatim:

    实验/样品/群聊应有层级：先建实验与样品会话——实验会话应持久，样品会话可频繁新建；
      一个实验对应若干样品，一个样品对应若干群聊，这个层次此前不存在。

The real hierarchy is 实验 → 若干样品 → 每个样品若干群聊. ``samples.experiment_id``
already carried the first edge, but ``conversations`` had NO ``sample_id`` at all
— every chat hung straight off the experiment, so the middle level did not exist
in the data and could not exist in the UI.

These pin the storage half: the column, the backward-compatible migration (the
operator's live DB predates it), and the fact that untagged rows stay findable
instead of vanishing.
"""

from __future__ import annotations

import sqlite3

import pytest

from mast.chat.store import ConversationStore


_OLD_SCHEMA = """
    CREATE TABLE conversations (
        conversation_id      TEXT PRIMARY KEY,
        agent_id             TEXT NOT NULL,
        thread_id            TEXT NOT NULL UNIQUE,
        title                TEXT NOT NULL DEFAULT '新对话',
        kind                 TEXT NOT NULL DEFAULT 'private',
        created_at           TEXT NOT NULL,
        updated_at           TEXT NOT NULL,
        last_message_preview TEXT NOT NULL DEFAULT '',
        experiment_id        TEXT,
        archived             INTEGER NOT NULL DEFAULT 0
    )
"""


@pytest.fixture()
def store(tmp_path) -> ConversationStore:
    return ConversationStore(tmp_path / "conv.db")


def test_chat_is_filed_under_a_sample(store: ConversationStore) -> None:
    row = store.create("_supervisor", kind="group", title="扫描 Si(111)",
                       experiment_id="E1", sample_id="S1")
    assert row["experiment_id"] == "E1"
    assert row["sample_id"] == "S1"
    assert store.get(row["conversation_id"])["sample_id"] == "S1"


def test_one_experiment_many_samples_many_chats(store: ConversationStore) -> None:
    """The shape the operator described, end to end."""
    for sample, titles in (("S1", ["a", "b"]), ("S2", ["c", "d", "e"])):
        for t in titles:
            store.create("_supervisor", kind="group", title=t,
                         experiment_id="E1", sample_id=sample)

    assert len(store.list(kind="group", experiment_id="E1")) == 5
    assert [r["title"] for r in store.list(kind="group", sample_id="S1")] == ["b", "a"]
    assert len(store.list(kind="group", sample_id="S2")) == 3
    # a filter is opt-in: the unfiltered list still spans everything
    assert len(store.list(kind="group")) == 5


def test_untagged_chats_are_listed_not_hidden(store: ConversationStore) -> None:
    """No active sample must not mean "no chat". They collect in the NULL bucket
    the UI renders as 未归属样品."""
    store.create("_supervisor", kind="group", title="无样品", experiment_id="E1")
    store.create("_supervisor", kind="group", title="有样品",
                 experiment_id="E1", sample_id="S1")

    assert len(store.list(kind="group")) == 2
    orphans = store.list(kind="group", sample_id="none")
    assert [r["title"] for r in orphans] == ["无样品"]
    assert orphans[0]["sample_id"] is None


def test_old_database_migrates_and_keeps_its_rows(tmp_path) -> None:
    """The operator's live mast_conversations.db has no sample_id column.

    CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so without the
    ALTER TABLE every read would raise 'no such column' — the whole chat history
    would disappear on upgrade. Old rows must survive with sample_id NULL."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO conversations VALUES "
        "('old1','_supervisor','t1','旧群聊','group','2026-01-01','2026-01-01','','E0',0)")
    conn.commit()
    conn.close()

    store = ConversationStore(db)
    rows = store.list(kind="group")
    assert [r["conversation_id"] for r in rows] == ["old1"]
    assert rows[0]["sample_id"] is None
    assert rows[0]["experiment_id"] == "E0"
    # and it lands in the explicit unassigned bucket rather than nowhere
    assert [r["conversation_id"] for r in store.list(sample_id="none")] == ["old1"]

    # migration is idempotent — reopening must not duplicate the column / throw
    again = ConversationStore(db)
    assert len(again.list(kind="group")) == 1
    new = again.create("_supervisor", kind="group", experiment_id="E0", sample_id="S9")
    assert again.get(new["conversation_id"])["sample_id"] == "S9"


def test_legacy_db_split_migration_carries_sample_id(tmp_path) -> None:
    """from_storage() copies conversations out of the experiment DB by column
    intersection; sample_id has to be in that list or the tag is dropped on the
    one boot that moves the data."""
    exp_db = tmp_path / "mast_experiments.db"
    seed = ConversationStore(exp_db)  # same schema as the split target
    seed.create("_supervisor", kind="group", title="迁移我",
                experiment_id="E1", sample_id="S1")

    class _Storage:
        _db_path = str(exp_db)

    moved = ConversationStore.from_storage(_Storage())
    rows = moved.list(kind="group")
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "S1"
    assert rows[0]["experiment_id"] == "E1"
