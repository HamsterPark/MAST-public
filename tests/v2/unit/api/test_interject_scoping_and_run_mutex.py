"""Interjections must not leak across tasks, and two runs must not share the
process's single slots (审计 严重级).

The interjection queue is a process-level singleton whose entries carried no run
identity, and ``run-task`` never cleaned it. An interjection queued into a window
with no consumer — the run's final LLM call, an abort, an exception; ``active``
is still True but no super-step is coming — survived to be drained by the FIRST
supervisor hop of the NEXT task. Carrying "@instrument_control", that stale text
became a DETERMINISTIC hard route into a brand-new experiment: of everything in
that queue it is the one path that touches hardware.

The concurrency guard next to it was a TOCTOU: it read the task slot at the top
of the generator and wrote it seconds later, with graph building and DB writes in
between, and only ever compared conversation_id. Two streams could both pass and
then fight over ``st["task"]`` and ``_orch_run_id`` — both single slots.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_interject_scoping_and_run_mutex.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import json  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import mast.api.routes.orchestrator as O  # noqa: E402
from mast.api.routes.agents_control import router as control_router  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────

class _EndingGraph:
    def stream(self, *_a, **_kw):
        yield ((), {"supervisor": {
            "messages": [AIMessage(content="[SUPERVISOR → __end__] 完成")],
            "active_agent": "__end__"}})


def _live(graph=None):
    live = types.SimpleNamespace()
    live._orchestrator = graph or _EndingGraph()
    live._orch_abort = threading.Event()
    live._orch_abort_emergency = False
    live._orch_running = False
    live._orch_run_id = ""
    live._orch_interrupts = {"lock": threading.Lock(), "pending": {},
                             "resolved": {}, "events": {}}
    live._conv_store = None
    live._agents_api_state = {"lock": threading.Lock(), "holds": {},
                              "interjects": [], "task": None}
    live._build_orchestrator = lambda **_k: True
    return live


def _client(live):
    app = FastAPI()
    app.include_router(O.router, prefix="/api")
    app.include_router(control_router, prefix="/api")
    app.state.ctx = types.SimpleNamespace(live_app=live)
    return TestClient(app)


def _frames(body: str) -> list[dict]:
    return [json.loads(ln[len("data:"):].strip())
            for ln in body.splitlines()
            if ln.startswith("data:") and ln[len("data:"):].strip()]


# ─────────────────────────────────────────────────────────────────────
# Interjection scoping
# ─────────────────────────────────────────────────────────────────────

def test_an_interjection_is_stamped_with_the_live_task():
    live = _live()
    c = _client(live)
    live._agents_api_state["task"] = {"active": True, "id": "task-A",
                                      "conversation_id": ""}

    r = c.post("/api/agents/instrument_control/interject",
               json={"text": "改成 5nm"}).json()
    assert r["ok"] is True
    q = live._agents_api_state["interjects"]
    assert len(q) == 1
    assert q[0]["task_id"] == "task-A", "an entry with no run identity is a leak"


def test_a_stale_interjection_is_dropped_not_delivered_to_the_next_task():
    """THE leak: text queued against task-A reaching task-B's first hop, where
    '@instrument_control' becomes a hard route on a new experiment."""
    from mast.core.runtime import CoreRuntime

    app = CoreRuntime.__new__(CoreRuntime)
    app._agents_api_state = {
        "lock": threading.Lock(),
        "task": {"active": True, "id": "task-B", "conversation_id": ""},
        "interjects": [
            {"id": "sa1", "agent_id": "instrument_control",
             "text": "把偏压设成 5V", "t": 0.0, "task_id": "task-A"},
            {"id": "sa2", "agent_id": "data_processing",
             "text": "顺便分析一下", "t": 0.0, "task_id": "task-B"},
        ],
    }
    out = app._orch_control_provider()

    assert out["interjections"] == ["(指向 data_processing) 顺便分析一下"]
    assert out["directed_targets"] == ["data_processing"], (
        "a previous task's @instrument_control became this task's hard route")
    assert app._agents_api_state["interjects"] == []


def test_an_unstamped_legacy_entry_is_still_delivered():
    """Entries written before the stamp exist in live processes mid-upgrade —
    dropping them would silently eat the operator's words."""
    from mast.core.runtime import CoreRuntime

    app = CoreRuntime.__new__(CoreRuntime)
    app._agents_api_state = {
        "lock": threading.Lock(),
        "task": {"active": True, "id": "task-B"},
        "interjects": [{"id": "sa0", "agent_id": "_supervisor",
                        "text": "慢一点", "t": 0.0}],
    }
    assert app._orch_control_provider()["interjections"] == ["慢一点"]


def test_a_new_run_starts_with_an_empty_interjection_queue():
    live = _live()
    c = _client(live)
    live._agents_api_state["interjects"] = [
        {"id": "sa9", "agent_id": "instrument_control",
         "text": "上一个任务遗留的指令", "t": 0.0, "task_id": "task-old"}]

    c.post("/api/agents/run-task", json={"task": "新任务"})
    assert live._agents_api_state["interjects"] == []


def test_an_unknown_agent_id_is_refused_not_silently_queued():
    """/hold has validated ids since day one; interject did not. A typo used to
    be accepted, queued, then dropped by the directed-route step — while the
    text still went through carrying "(指向 <typo>)", so the operator read their
    own instruction back and believed it had been routed."""
    live = _live()
    c = _client(live)
    live._agents_api_state["task"] = {"active": True, "id": "task-A",
                                      "conversation_id": ""}

    r = c.post("/api/agents/instrument_controll/interject",
               json={"text": "改成 5nm"}).json()
    assert r["ok"] is False and r["degraded"] is True
    assert "未知的智能体" in r["detail"]
    assert live._agents_api_state["interjects"] == []


def test_a_valid_agent_id_still_works():
    live = _live()
    c = _client(live)
    live._agents_api_state["task"] = {"active": True, "id": "task-A",
                                      "conversation_id": ""}
    for aid in ("instrument_control", "_supervisor", "__all__"):
        live._agents_api_state["interjects"] = []
        assert c.post(f"/api/agents/{aid}/interject",
                      json={"text": "x"}).json()["ok"] is True


# ─────────────────────────────────────────────────────────────────────
# Run mutex
# ─────────────────────────────────────────────────────────────────────

def test_a_second_run_is_refused_while_one_is_active():
    live = _live()
    c = _client(live)
    # An already-streaming run holds the slot.
    live._agents_api_state["task"] = {"active": True, "id": "task-A",
                                      "description": "扫一整张大图",
                                      "conversation_id": "conv-A"}

    frames = _frames(c.post("/api/agents/run-task",
                            json={"task": "另一个任务",
                                  "conversation_id": "conv-B"}).text)
    err = [f for f in frames if f["kind"] == "error"]
    assert err, [f["kind"] for f in frames]
    assert "另一个群聊任务" in err[0]["message"]
    assert frames[-1]["kind"] == "done"
    # …and the incumbent's slot is untouched.
    assert live._agents_api_state["task"]["id"] == "task-A"


def test_the_claim_and_the_check_are_one_critical_section():
    """The TOCTOU itself: the guard must not be a read that a second stream can
    slip past before the write lands."""
    src = (Path(_MASTV2_ROOT) / "mast" / "api" / "routes" /
           "orchestrator.py").read_text(encoding="utf-8", errors="replace")
    i = src.find('"active": True, "id": task_id')
    assert i > 0
    before = src[max(0, i - 1400):i]
    assert "st_lock" in before and '_held.get("active")' in before, (
        "the busy check is not inside the same lock as the slot write")


def test_the_slot_is_released_so_the_next_run_can_claim_it():
    live = _live()
    c = _client(live)
    for _ in range(3):
        live._orchestrator = _EndingGraph()
        frames = _frames(c.post("/api/agents/run-task", json={"task": "t"}).text)
        assert not [f for f in frames if f["kind"] == "error"], frames
    assert live._agents_api_state["task"]["active"] is False


def test_an_active_run_on_the_SAME_conversation_still_gets_its_own_message():
    live = _live()
    c = _client(live)
    live._agents_api_state["task"] = {"active": True, "id": "task-A",
                                      "description": "扫图",
                                      "conversation_id": "conv-A"}
    frames = _frames(c.post("/api/agents/run-task",
                            json={"task": "再来一次",
                                  "conversation_id": "conv-A"}).text)
    err = [f for f in frames if f["kind"] == "error"]
    # Which of the two refusal texts appears depends on whether the durable
    # conversation id resolved (this harness has no ConversationStore, so it
    # does not) — what must hold is that the second stream is REFUSED and told
    # what to do about it.
    assert err, [f["kind"] for f in frames]
    assert "中止" in err[0]["message"]
    assert frames[-1]["kind"] == "done"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
