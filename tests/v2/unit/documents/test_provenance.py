"""「这份文档是哪次对话、哪个 run 产出的」——  provenance 的自动填充。

设计文档：docs/v2/design/document_and_library_management.md §3.4

保存工具是裸 ``@tool`` 函数，没有 ctx 注入。对话引擎其实早就把 conversation_id /
run_id 写进了自己的 thread-local，但一直没有公共访问器、零消费者。
``core/turn_context`` 就是那个中立落点。
"""

from __future__ import annotations

import threading

import pytest

from mast.core.turn_context import clear_turn, current_turn, set_turn, turn_scope
from mast.documents import reset_caches, store
from mast.logging.storage import ExperimentStorage


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = tmp_path / "MAST-Data" / "experiments"
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "db" / "exp.db"))
    root.mkdir(parents=True, exist_ok=True)
    reset_caches()
    clear_turn()
    yield root
    clear_turn()
    reset_caches()


@pytest.fixture()
def eid(env):
    from mast.agents._shared.data_paths import experiment_db_path
    st = ExperimentStorage(experiment_db_path())
    e = st.create_experiment("E", "")
    st.set_active_scope(e, None, updated_by="test")
    return e


def test_empty_by_default(env):
    assert current_turn() == {"conversation_id": None, "run_id": None, "agent_id": None}


def test_set_and_clear(env):
    set_turn(conversation_id="conv-1", run_id="run-1", agent_id="paper_writing")
    t = current_turn()
    assert (t["conversation_id"], t["run_id"], t["agent_id"]) == (
        "conv-1", "run-1", "paper_writing")
    clear_turn()
    assert current_turn()["conversation_id"] is None


def test_partial_set_does_not_wipe_the_others(env):
    set_turn(conversation_id="conv-1", run_id="run-1")
    set_turn(run_id="run-2")
    t = current_turn()
    assert t["conversation_id"] == "conv-1" and t["run_id"] == "run-2"


def test_turn_scope_restores_instead_of_clearing(env):
    """嵌套回合（群聊 super-step 里派生子调用）离开时要**恢复**外层，不是清空。"""
    set_turn(conversation_id="outer", run_id="run-outer")
    with turn_scope(conversation_id="inner", run_id="run-inner"):
        assert current_turn()["conversation_id"] == "inner"
    assert current_turn()["conversation_id"] == "outer"
    assert current_turn()["run_id"] == "run-outer"


def test_is_thread_local(env):
    """群聊、私聊、后台 run 三条线程同时在跑 —— 一个全局变量会串档。"""
    set_turn(conversation_id="main-thread")
    seen: dict[str, object] = {}

    def worker() -> None:
        seen["before"] = current_turn()["conversation_id"]
        set_turn(conversation_id="worker-thread")
        seen["after"] = current_turn()["conversation_id"]

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["before"] is None, "别的线程不该看到本线程的回合"
    assert seen["after"] == "worker-thread"
    assert current_turn()["conversation_id"] == "main-thread"


def test_document_records_the_turn_it_came_from(env, eid):
    set_turn(conversation_id="conv-abc", run_id="chat-xyz-0001")
    res = store().save(text="正文", kind="experiment_report", title="带来历",
                       created_by="agent:paper_writing")
    entry = store().get(res.doc_id)
    assert entry.meta.conversation_id == "conv-abc"
    assert entry.meta.run_id == "chat-xyz-0001"
    assert entry.versions[-1].conversation_id == "conv-abc"
    assert entry.versions[-1].run_id == "chat-xyz-0001"


def test_each_version_keeps_its_own_turn(env, eid):
    """v001 和 v002 可能来自不同的对话 —— 版本行各记各的。"""
    set_turn(conversation_id="conv-1", run_id="run-1")
    a = store().save(text="第一版", kind="experiment_report", title="多轮")
    clear_turn()
    set_turn(conversation_id="conv-2", run_id="run-2")
    store().save(text="第二版", doc_id=a.doc_id)

    entry = store().get(a.doc_id)
    assert entry.versions[0].conversation_id == "conv-1"
    assert entry.versions[1].conversation_id == "conv-2"
    # 文档头部保留**首次**的来历（它是这份文档的出身，不随后续修订漂移）
    assert entry.meta.conversation_id == "conv-1"


def test_missing_turn_context_never_blocks_the_save(env, eid):
    """拿不到 provenance 就存 null —— 绝不因为记账缺一格而丢掉整篇内容。"""
    clear_turn()
    res = store().save(text="没有来历也要存下来", kind="experiment_report", title="无来历")
    assert res.ok
    entry = store().get(res.doc_id)
    assert entry.meta.conversation_id is None and entry.meta.run_id is None


def test_explicit_argument_wins_over_thread_local(env, eid):
    set_turn(conversation_id="from-tl", run_id="tl-run")
    res = store().save(text="正文", kind="experiment_report", title="显式覆盖",
                       conversation_id="explicit", run_id="explicit-run")
    entry = store().get(res.doc_id)
    assert entry.meta.conversation_id == "explicit"
    assert entry.meta.run_id == "explicit-run"


def test_engine_exposes_the_conversation_id_accessor():
    """``active_conversation_id`` 此前不存在（值写了但没人能读）。"""
    from mast.chat.engine import ConversationEngine
    assert hasattr(ConversationEngine, "active_conversation_id")
    assert hasattr(ConversationEngine, "_set_turn_context")
    assert hasattr(ConversationEngine, "_clear_turn_context")
