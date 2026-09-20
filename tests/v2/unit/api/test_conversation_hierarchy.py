"""实验/样品/群聊 的层次不对（API 层）。

层次应为:一个实验若干样品，一个样品若干群聊。

The storage half is pinned in tests/v2/unit/chat/test_conversation_sample_
hierarchy.py. This file pins the seam the UI actually reads:

  * a new chat (private or 群聊) is filed under the sample that is ACTIVE right
    now, without the client having to know about samples at all;
  * the list endpoints hand back sample_id **and** the display names, so the
    client can draw 实验 → 样品 → 群聊 without an N+1 fetch and without showing
    the operator raw UUIDs (which is what #28 was about);
  * a chat with no sample — every row created before this change — still lists.
"""

from __future__ import annotations

import json
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.agents import router as agents_router
from mast.api.routes.orchestrator import router as orch_router
from mast.chat.store import ConversationStore


# ── a live-log stand-in: what the operator currently has open ────────────────
class _FakeLog:
    def __init__(self, experiment_id=None, sample_id=None):
        self.current_experiment_id = experiment_id
        self.current_sample_id = sample_id


class _FakeStorage:
    """Minimal ExperimentStorage surface used for name resolution."""

    def __init__(self, experiments: dict, samples: dict):
        self._experiments = experiments      # {exp_id: name}
        self._samples = samples              # {exp_id: [(sample_id, name), ...]}

    def get_experiment(self, eid):
        name = self._experiments.get(eid)
        return {"id": eid, "name": name} if name else None

    def get_samples(self, eid):
        return [{"id": sid, "name": n} for sid, n in self._samples.get(eid, [])]


class _AIMessage:
    def __init__(self, content=""):
        self.content = content
        self.tool_calls = []
        self.id = None


class _FakeOrchestrator:
    """Enough of the graph surface for one clean super-step."""

    def stream(self, *args, **kwargs):
        yield (("instrument_control:abc",),
               {"agent": {"messages": [_AIMessage("扫描完成")]}})


class _FakeLiveApp:
    def __init__(self, conv_store):
        self._orchestrator = _FakeOrchestrator()
        self._build_ok = False
        self._orch_abort = threading.Event()
        self._orch_running = False
        self._orch_interrupts = {"lock": threading.Lock(), "pending": {},
                                 "resolved": {}, "events": {}}
        self._conv_store = conv_store
        self._agents_api_state = {"lock": threading.Lock(), "holds": {},
                                  "interjects": [], "task": None}

    def _build_orchestrator(self, **kwargs):
        return False


def _client(ctx: AppContext) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(agents_router, prefix="/api")
    app.include_router(orch_router, prefix="/api")
    return TestClient(app)


def _ctx(store, *, live_app=None, storage=None) -> AppContext:
    ctx = AppContext()
    ctx.conversation_store = store            # type: ignore[attr-defined]
    if live_app is not None:
        ctx.live_app = live_app               # type: ignore[attr-defined]
    if storage is not None:
        ctx._experiment_storage = storage     # type: ignore[attr-defined]
    return ctx


def _patch_active(monkeypatch, log) -> None:
    monkeypatch.setattr("mast.logging.experiment_log.get_active_log",
                        lambda: log, raising=False)


# ── private chats ────────────────────────────────────────────────────────────


def test_new_private_chat_inherits_the_active_sample(tmp_path, monkeypatch) -> None:
    """The client sends no sample_id; the server files the chat where the
    operator is standing. Without this the middle level of the hierarchy stays
    empty no matter what the UI draws."""
    _patch_active(monkeypatch, _FakeLog("E1", "S1"))
    store = ConversationStore(tmp_path / "conv.db")
    c = _client(_ctx(store))

    r = c.post("/api/agents/instrument_control/conversations", json={"title": "调针尖"})
    assert r.status_code == 200
    conv = r.json()["conversation"]
    assert conv["experiment_id"] == "E1"
    assert conv["sample_id"] == "S1"

    listed = c.get("/api/agents/instrument_control/conversations").json()["conversations"]
    assert [x["sample_id"] for x in listed] == ["S1"]


def test_explicit_sample_id_wins_over_the_active_one(tmp_path, monkeypatch) -> None:
    _patch_active(monkeypatch, _FakeLog("E1", "S1"))
    store = ConversationStore(tmp_path / "conv.db")
    c = _client(_ctx(store))
    conv = c.post("/api/agents/literature/conversations",
                  json={"sample_id": "S-other"}).json()["conversation"]
    assert conv["sample_id"] == "S-other"


def test_no_active_sample_still_creates_a_listable_chat(tmp_path, monkeypatch) -> None:
    """Nothing active is normal (fresh boot, between samples). The chat must be
    created and listed with sample_id null — never rejected, never hidden."""
    _patch_active(monkeypatch, _FakeLog(None, None))
    store = ConversationStore(tmp_path / "conv.db")
    c = _client(_ctx(store))
    conv = c.post("/api/agents/literature/conversations", json={}).json()["conversation"]
    assert conv["sample_id"] is None
    assert conv["experiment_id"] is None
    listed = c.get("/api/agents/literature/conversations").json()["conversations"]
    assert len(listed) == 1


# ── 群聊 (the level the operator was complaining about) ──────────────────────


def test_group_chat_is_filed_under_the_active_sample(tmp_path, monkeypatch) -> None:
    _patch_active(monkeypatch, _FakeLog("E1", "S1"))
    store = ConversationStore(tmp_path / "conv.db")
    app = _FakeLiveApp(store)
    c = _client(_ctx(store, live_app=app))

    with c.stream("POST", "/api/agents/run-task", json={"task": "扫描一下"}) as r:
        frames = [json.loads(ln[len("data:"):].strip())
                  for ln in r.iter_lines() if ln.startswith("data:")]
    assert frames[-1]["kind"] == "done"

    rows = store.list(kind="group")
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "E1"
    assert rows[0]["sample_id"] == "S1"


def test_group_conversations_expose_the_three_levels_with_names(
    tmp_path, monkeypatch,
) -> None:
    """GET /agents/group-conversations must return ids AND names for both parent
    levels — the UI draws 实验 → 样品 → 群聊 straight off this response."""
    _patch_active(monkeypatch, _FakeLog(None, None))
    store = ConversationStore(tmp_path / "conv.db")
    store.create("_supervisor", kind="group", title="群聊 A",
                 experiment_id="E1", sample_id="S1")
    store.create("_supervisor", kind="group", title="群聊 B",
                 experiment_id="E1", sample_id="S1")
    store.create("_supervisor", kind="group", title="群聊 C",
                 experiment_id="E1", sample_id="S2")
    store.create("_supervisor", kind="group", title="老群聊")  # pre-#17 row

    storage = _FakeStorage(
        experiments={"E1": "Si(111) 表面重构"},
        samples={"E1": [("S1", "样品一"), ("S2", "样品二")]},
    )
    c = _client(_ctx(store, storage=storage))
    body = c.get("/api/agents/group-conversations").json()
    assert body["degraded"] is False
    by_title = {x["title"]: x for x in body["conversations"]}

    assert by_title["群聊 A"]["sample_id"] == "S1"
    assert by_title["群聊 A"]["sample_name"] == "样品一"
    assert by_title["群聊 A"]["experiment_name"] == "Si(111) 表面重构"
    assert by_title["群聊 C"]["sample_name"] == "样品二"

    # one experiment → two samples → three chats, plus the untagged one
    assert {x["sample_id"] for x in body["conversations"]} == {"S1", "S2", None}
    old = by_title["老群聊"]
    assert old["sample_id"] is None and old["sample_name"] is None
    assert len(body["conversations"]) == 4


def test_group_conversations_survive_a_missing_storage(tmp_path, monkeypatch) -> None:
    """No ExperimentStorage wired (standalone dev) → no names, but the list and
    the ids still come back. A name lookup must never cost the operator the
    history itself."""
    _patch_active(monkeypatch, _FakeLog(None, None))
    store = ConversationStore(tmp_path / "conv.db")
    store.create("_supervisor", kind="group", title="群聊 A",
                 experiment_id="E1", sample_id="S1")
    c = _client(_ctx(store))
    body = c.get("/api/agents/group-conversations").json()
    row = body["conversations"][0]
    assert row["sample_id"] == "S1"
    assert row["experiment_name"] is None
    assert row["sample_name"] is None


def test_name_lookup_failure_degrades_to_ids(tmp_path, monkeypatch) -> None:
    """A raising storage (locked DB, deleted experiment) must not 500 the list."""
    _patch_active(monkeypatch, _FakeLog(None, None))
    store = ConversationStore(tmp_path / "conv.db")
    store.create("_supervisor", kind="group", title="群聊 A",
                 experiment_id="E1", sample_id="S1")

    class _Boom:
        def get_experiment(self, eid):
            raise RuntimeError("database is locked")

        def get_samples(self, eid):
            raise RuntimeError("database is locked")

    c = _client(_ctx(store, storage=_Boom()))
    body = c.get("/api/agents/group-conversations").json()
    assert body["degraded"] is False
    assert body["conversations"][0]["sample_id"] == "S1"
    assert body["conversations"][0]["sample_name"] is None
