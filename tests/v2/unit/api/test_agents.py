"""Phase-3 Agents domain API contract tests (non-streaming).

Guards the typed seam for the Agents tab: tool catalog, model+thinking table,
and the conversation CRUD + messages + chat-abort surface. Asserts:
  - every endpoint returns its defined status + schema-shaped body;
  - the degraded path (no live core wired) is empty-but-not-broken, never 500;
  - the wired path (a fake ConversationStore / engine on ctx) returns real data.

The router is NOT yet mounted in mast.api.app (integration wires that), so
these tests build a throwaway FastAPI app exactly as the task prescribes.
"""

from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
# pytest's rootdir-based sys.path injection puts D:\...\MAST first, where the v1
# mast/ shadows MASTv2/mast/. Force MASTv2/ ahead, and purge any already-cached
# v1 mast.* modules (canonical block, see tests/v2/unit/test_wrap_skill_minimal.py).
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _mod_name in list(sys.modules):
    if _mod_name == "mast" or _mod_name.startswith("mast."):
        _mod_path = getattr(sys.modules[_mod_name], "__file__", "") or ""
        if "MASTv2" not in _mod_path.replace("\\", "/"):
            del sys.modules[_mod_name]

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.agents import router


# ── throwaway app (router not yet mounted in app.py) ───────────────────


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── fakes for the wired path ───────────────────────────────────────────


class _FakeStore:
    """Minimal ConversationStore stand-in (only the surface the routes touch)."""

    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}
        self._n = 0

    def list(self, *, agent_id=None, kind=None, experiment_id=None, sample_id=None,
             include_archived=False, limit=200):
        return [
            r for r in self._rows.values()
            if (agent_id is None or r["agent_id"] == agent_id) and not r["archived"]
        ]

    # sample_id: a conversation now hangs off the SAMPLE it was started on
    # (实验 → 样品 → 会话, ).
    def create(self, agent_id, *, kind="private", title=None, experiment_id=None,
               sample_id=None, thread_id=None):
        self._n += 1
        cid = f"conv{self._n}"
        row = {
            "conversation_id": cid, "agent_id": agent_id,
            "thread_id": thread_id or f"thread-{cid}",
            "title": title or "新对话", "kind": kind,
            "created_at": "2026-06-20T00:00:00", "updated_at": "2026-06-20T00:00:00",
            "last_message_preview": "", "experiment_id": experiment_id,
            "sample_id": sample_id,
            "archived": False,
        }
        self._rows[cid] = row
        return row

    def rename(self, conversation_id, title):
        row = self._rows.get(conversation_id)
        if not row:
            return False
        row["title"] = title
        return True

    def delete(self, conversation_id, *, checkpointer=None):
        return self._rows.pop(conversation_id, None) is not None


class _FakeEngine:
    _checkpointer = None

    def get_messages(self, conversation_id):
        return [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]


# ═══════════════════════════════════════════════════════════════════════
#  Degraded path (standalone, no live core)
# ═══════════════════════════════════════════════════════════════════════


def test_tools_degrades_or_real() -> None:
    """tools either degrades cleanly OR returns a shaped catalog (registry may be
    importable in this venv) — but NEVER 500s."""
    r = _client().get("/api/agents/tools")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["agents"], list)
    assert isinstance(body["degraded"], bool)
    for a in body["agents"]:
        assert "agent_id" in a and isinstance(a["tools"], list)
        assert a["count"] == len(a["tools"])
        for t in a["tools"]:
            assert "name" in t and "safety_level" in t


def test_models_degrades_or_real() -> None:
    r = _client().get("/api/agents/models")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["agents"], list)
    assert isinstance(body["degraded"], bool)
    for a in body["agents"]:
        assert "agent_id" in a and "model" in a


def test_conversations_list_degrades_unwired() -> None:
    r = _client().get("/api/agents/instrument_control/conversations")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["conversations"] == [] and body["count"] == 0


def test_conversation_create_degrades_unwired() -> None:
    r = _client().post(
        "/api/agents/instrument_control/conversations", json={"title": "x"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["conversation"] is None


def test_conversation_create_accepts_empty_body() -> None:
    # body is optional — POST with no JSON must still degrade cleanly, not 422.
    r = _client().post("/api/agents/literature/conversations")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_conversation_rename_degrades_unwired() -> None:
    r = _client().patch(
        "/api/agents/instrument_control/conversations/c1", json={"title": "new"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_conversation_delete_degrades_unwired() -> None:
    r = _client().delete("/api/agents/instrument_control/conversations/c1")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_messages_degrades_unwired() -> None:
    r = _client().get(
        "/api/agents/instrument_control/messages", params={"conversation_id": "c1"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["conversation_id"] == "c1"
    assert body["messages"] == [] and body["count"] == 0


def test_messages_requires_conversation_id() -> None:
    # conversation_id is a required query param → 422 when missing.
    r = _client().get("/api/agents/instrument_control/messages")
    assert r.status_code == 422


def test_chat_abort_degrades_unwired() -> None:
    r = _client().post("/api/agents/instrument_control/chat/abort")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


# ═══════════════════════════════════════════════════════════════════════
#  Wired path (fake store / engine on ctx)
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def wired_ctx() -> AppContext:
    ctx = AppContext()
    # The leader wires these real singletons at integration time; the routes
    # read them off ctx via getattr, so attaching fakes exercises the live path.
    ctx.conversation_store = _FakeStore()  # type: ignore[attr-defined]
    ctx.conversation_engine = _FakeEngine()  # type: ignore[attr-defined]
    ctx.chat_abort = lambda agent_id: True  # type: ignore[attr-defined]
    return ctx


def test_conversation_crud_roundtrip_wired(wired_ctx: AppContext) -> None:
    c = _client(wired_ctx)

    # create
    r = c.post(
        "/api/agents/instrument_control/conversations", json={"title": "first"}
    )
    assert r.status_code == 200
    created = r.json()
    assert created["ok"] is True and created["degraded"] is False
    conv = created["conversation"]
    assert conv["agent_id"] == "instrument_control"
    assert conv["title"] == "first"
    cid = conv["conversation_id"]

    # list shows it
    r = c.get("/api/agents/instrument_control/conversations")
    body = r.json()
    assert body["degraded"] is False and body["count"] == 1
    assert body["conversations"][0]["conversation_id"] == cid

    # rename
    r = c.patch(
        f"/api/agents/instrument_control/conversations/{cid}",
        json={"title": "renamed"},
    )
    assert r.json() == {"ok": True, "degraded": False}

    # rename unknown id → ok False, but not degraded (store is wired)
    r = c.patch(
        "/api/agents/instrument_control/conversations/nope", json={"title": "z"}
    )
    assert r.json() == {"ok": False, "degraded": False}

    # delete
    r = c.delete(f"/api/agents/instrument_control/conversations/{cid}")
    assert r.json() == {"ok": True, "degraded": False}

    # list now empty (still not degraded)
    r = c.get("/api/agents/instrument_control/conversations")
    body = r.json()
    assert body["degraded"] is False and body["count"] == 0


def test_messages_wired(wired_ctx: AppContext) -> None:
    """线上格式的严格契约。

    ⚠️ 2026-08-12 多了一个 ``t``（消息发生的 epoch 秒，请求：「agent 的发言也
    带上时间就更好了」）。这条测试当场红了 —— **那正是它存在的理由**：
    回包多一个字段是一次契约变更，必须有人看见并决定，不能悄悄发生。

    ``t: None`` 而不是「没有这个键」：``None`` 在这里是**有含义的** ——
    「不知道它是什么时候说的」（重启前就在 checkpoint 里的历史）。前端
    ``fmtClock`` 对无效值返回空串，所以它渲染成「不显示时间」，**绝不显示 1970**。
    这个替身给的正是没有时间戳的那一份。
    """
    r = _client(wired_ctx).get(
        "/api/agents/instrument_control/messages",
        params={"conversation_id": "c1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False and body["count"] == 2
    assert body["messages"][0] == {"role": "user", "content": "hello", "t": None}
    assert body["messages"][1] == {"role": "assistant", "content": "world", "t": None}


def test_chat_abort_wired(wired_ctx: AppContext) -> None:
    """老式 hook（只收 agent_id、返回 bool）仍然接得住。

    契约在 2026-08-11 收紧过：回包多了 ``signalled`` / ``conversation_ids`` /
    ``reason`` / ``caveat``，因为原来那个 ``ok`` 的含义是「有个 Event 对象」而不是
    「这一轮收到了停止信号」。
    """
    r = _client(wired_ctx).post("/api/agents/instrument_control/chat/abort")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["signalled"] == 1
    assert body["caveat"], "报了 ok 就必须同时说清楚 ok 的边界"


def test_chat_abort_refuses_to_name_a_conversation_a_legacy_hook_cannot_target(
        wired_ctx: AppContext) -> None:
    """点名停某一条会话，而通道接不住 ⇒ **拒绝**，不假装停对了。

    私聊的多个会话都挂在同一个 agent 下。悄悄退回「停这个 agent 的全部」会连带停掉
    并发的另一条 —— 那是用一个安静的错误换一个安静的成功。
    """
    r = _client(wired_ctx).post("/api/agents/instrument_control/chat/abort",
                                json={"conversation_id": "c1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["reason"]


def test_create_then_isolated_per_agent(wired_ctx: AppContext) -> None:
    c = _client(wired_ctx)
    c.post("/api/agents/literature/conversations", json={"title": "lit"})
    c.post("/api/agents/data_processing/conversations", json={"title": "dp"})
    lit = c.get("/api/agents/literature/conversations").json()
    dp = c.get("/api/agents/data_processing/conversations").json()
    assert lit["count"] == 1 and dp["count"] == 1
    assert lit["conversations"][0]["agent_id"] == "literature"
    assert dp["conversations"][0]["agent_id"] == "data_processing"


# ═══════════════════════════════════════════════════════════════════════
#  Group activity bridge — an agent's 群聊 messages surface in its own view
#  (the bridge the rewrite dropped: team-run replies were invisible per-agent)
# ═══════════════════════════════════════════════════════════════════════


def test_group_activity_degrades_unwired() -> None:
    r = _client().get("/api/agents/instrument_control/group-activity")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["entries"] == [] and body["count"] == 0
    assert body["agent_id"] == "instrument_control"


def test_group_activity_wired_surfaces_agent_messages(tmp_path) -> None:
    """A real ConversationStore with a 群聊 transcript: the per-agent endpoint
    returns ONLY that agent's group messages — the restored bridge."""
    from mast.chat.store import ConversationStore

    store = ConversationStore(tmp_path / "exp.sqlite")
    cid = store.create("_supervisor", kind="group", title="测 Kondo",
                       thread_id="agents-1")["conversation_id"]
    store.append_message(cid, kind="operator", role="user", text="用户任务")
    store.append_message(cid, kind="message", agent_id="instrument_control",
                         role="agent", text="IC 在群聊里完成扫描")
    store.append_message(cid, kind="message", agent_id="literature",
                         role="agent", text="文献给了参数")

    ctx = AppContext()
    ctx.conversation_store = store  # type: ignore[attr-defined]
    c = _client(ctx)

    r = c.get("/api/agents/instrument_control/group-activity")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["count"] == 1
    assert body["entries"][0]["text"] == "IC 在群聊里完成扫描"
    assert body["entries"][0]["conversation_title"] == "测 Kondo"
    # other agents / operator turns never leak in
    assert all("用户任务" != e["text"] for e in body["entries"])

    lit = c.get("/api/agents/literature/group-activity").json()
    assert lit["count"] == 1 and lit["entries"][0]["text"] == "文献给了参数"
