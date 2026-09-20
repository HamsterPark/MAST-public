"""run-task publishes an ``ask_user`` interrupt (and waits the way it declared).

Two things are pinned here, and the first is the one that kills the feature if
it regresses: an interrupt ``kind`` the publisher does not recognise falls
through to an EMPTY result, and ``_drive`` then logs "unparseable __interrupt__"
and terminates the run. That is exactly how buffer_hitl was silently dropped in
2026-07-06. So a question the agent asked would end the run instead of reaching
the operator.

The second is the timeout policy. Unlike an approval — where nobody answering
must mean "do not run the hardware action" — an ``ask_user`` question executes
nothing, so its silence policy is the ASKING AGENT's declaration:
``continue`` hands back "unanswered, use your stated fallback", ``halt`` stops
the run with the checkpoint intact. Both must still emit the ``approval_timeout``
frame first (2026-07-28: a timeout that emitted nothing made the card simply
vanish), and the heartbeat must not promise "按拒绝处理", which is not what
happens to a question.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/api/test_ask_user_publish.py -x -v
"""
from __future__ import annotations

import threading

import mast.api.routes.orchestrator as orch
from mast.api.routes.orchestrator import _await_resolution, _publish_interrupt


class _App:
    pass


def _fresh_app() -> _App:
    app = _App()
    app._orch_interrupts = {"pending": {}, "events": {}, "resolved": {},
                            "lock": threading.Lock()}
    return app


_ASK = {
    "kind": "ask_user",
    "question": "接下来先扫哪个区域？无人回答我会默认扫 B 区。",
    "header": "区域选择",
    "options": [{"label": "A 区", "description": "缺陷密集"},
                {"label": "B 区", "description": "平坦台面"}],
    "multi_select": False,
    "allow_custom": True,
    "timeout_action": "continue",
    "agent_id": "",
}


class TestPublish:
    def test_ask_user_is_published_not_unparseable(self):
        app = _fresh_app()
        published = _publish_interrupt(app, "literature", (dict(_ASK),), "t1")

        assert len(published) == 1, \
            "ask_user must NOT fall through to empty (that terminates the run)"
        p = published[0]
        assert p["kind"] == "ask_user"
        assert p["agent_id"] == "literature"
        assert p["allowed_decisions"] == ["answer"]
        # Registered in the LIVE store, which is what agents_control.resolve drains.
        assert p["event_id"] in app._orch_interrupts["pending"]
        assert p["event_id"] in app._orch_interrupts["events"]

    def test_structured_question_rides_in_ask(self):
        app = _fresh_app()
        p = _publish_interrupt(app, "literature", (dict(_ASK),), "t1")[0]
        ask = p["ask"]
        assert ask["question"].startswith("接下来先扫哪个区域？")
        assert [o["label"] for o in ask["options"]] == ["A 区", "B 区"]
        assert ask["options"][0]["description"] == "缺陷密集"
        assert ask["multi_select"] is False
        assert ask["allow_custom"] is True
        assert ask["timeout_action"] == "continue"

    def test_legacy_fields_keep_an_unaware_client_readable(self):
        # A client that predates this kind renders skill / rationale / params.
        # It gets no buttons (P2 fixes that), but it must still SHOW the question
        # rather than an empty approval box.
        app = _fresh_app()
        p = _publish_interrupt(app, "literature", (dict(_ASK),), "t1")[0]
        assert p["skill"] == "向用户提问"
        assert "先扫哪个区域" in p["rationale"]
        assert p["params"]["options"] == ["A 区", "B 区"]
        assert "先扫哪个区域" in p["params"]["question"]

    def test_self_reported_agent_id_wins_over_the_namespace(self):
        # The supervisor's ask node is a PARENT-graph node: its namespace is
        # empty, so `owner` falls back to instrument_control. Without the
        # self-report the supervisor's question would be attributed to an agent
        # that never asked it.
        app = _fresh_app()
        payload = dict(_ASK, agent_id="_supervisor")
        p = _publish_interrupt(app, "", (payload,), "t1")[0]
        assert p["agent_id"] == "_supervisor"

    def test_unknown_interrupt_still_empty(self):
        # The new branch must key on kind exactly and not swallow arbitrary dicts.
        app = _fresh_app()
        assert _publish_interrupt(app, "literature", ({"kind": "mystery"},), "t2") == []


def _drain(gen):
    beats, result = [], None
    for what, payload in gen:
        if what == "beat":
            beats.append(payload)
        else:
            result = payload
    return beats, result


class TestAwaitResolution:
    def test_answer_resumes_with_the_single_value_shape(self, monkeypatch):
        app = _fresh_app()
        p = _publish_interrupt(app, "literature", (dict(_ASK),), "t1")[0]
        eid = p["event_id"]
        answer = {"selected": ["B 区"], "custom_text": "", "note": ""}
        app._orch_interrupts["resolved"][eid] = answer
        app._orch_interrupts["events"][eid].set()

        _beats, result = _drain(_await_resolution(app, eid, "ask_user", None))
        # NOT wrapped in {"decisions": [...]} — the tool resumes on this dict.
        assert result == answer
        # …and the entry is cleaned out of the live store.
        assert eid not in app._orch_interrupts["pending"]

    def test_continue_timeout_is_fail_open_and_says_so_first(self, monkeypatch):
        monkeypatch.setattr(orch, "_MAX_APPROVAL_WAIT_S", 0.05)
        monkeypatch.setattr(orch, "_APPROVAL_BEAT_S", 0.01)
        app = _fresh_app()
        p = _publish_interrupt(app, "literature", (dict(_ASK),), "t1")[0]

        beats, result = _drain(
            _await_resolution(app, p["event_id"], "ask_user", None))

        # Invariant (2026-07-28): the operator is TOLD the wait ended.
        assert any(b.get("subkind") == "approval_timeout" for b in beats)
        assert result["timeout"] is True
        assert result["selected"] == [] and result["custom_text"] == ""
        assert "未收到用户回答" in result["note"]

    def test_heartbeat_does_not_promise_rejection_for_a_question(self, monkeypatch):
        monkeypatch.setattr(orch, "_MAX_APPROVAL_WAIT_S", 0.05)
        monkeypatch.setattr(orch, "_APPROVAL_BEAT_S", 0.01)
        app = _fresh_app()
        p = _publish_interrupt(app, "literature", (dict(_ASK),), "t1")[0]

        beats, _ = _drain(_await_resolution(app, p["event_id"], "ask_user", None))
        waiting = [b for b in beats if b.get("subkind") == "awaiting_approval"]
        assert waiting, "the wait must still heartbeat (idle-TCP reapers, #34)"
        for b in waiting:
            assert "按拒绝处理" not in b["text"], \
                "nothing is being rejected — a question has no verdict"
            assert "回答" in b["text"]

    def test_halt_timeout_stops_the_run(self, monkeypatch):
        monkeypatch.setattr(orch, "_MAX_APPROVAL_WAIT_S", 0.05)
        monkeypatch.setattr(orch, "_APPROVAL_BEAT_S", 0.01)
        app = _fresh_app()
        payload = dict(_ASK, timeout_action="halt")
        p = _publish_interrupt(app, "literature", (payload,), "t1")[0]

        beats, result = _drain(
            _await_resolution(app, p["event_id"], "ask_user", None))

        assert any(b.get("subkind") == "approval_timeout" for b in beats)
        # None → _drive's stop path (run ends, checkpoint intact).
        assert result is None
        assert any("停下" in b["text"] and "进度已保存" in b["text"] for b in beats
                   if b.get("subkind") == "approval_timeout")

    def test_dangerous_timeout_still_fails_closed(self, monkeypatch):
        # Regression guard: the ask_user branch must not have relaxed approvals.
        monkeypatch.setattr(orch, "_MAX_APPROVAL_WAIT_S", 0.05)
        monkeypatch.setattr(orch, "_APPROVAL_BEAT_S", 0.01)
        app = _fresh_app()
        chunk = ({"action_requests": [{"name": "SetBias", "args": {"bias_v": 2.0},
                                       "description": "set bias"}],
                  "review_configs": [{"action_name": "SetBias",
                                      "allowed_decisions": ["approve", "reject"]}]},)
        p = _publish_interrupt(app, "instrument_control", chunk, "t1")[0]

        beats, result = _drain(
            _await_resolution(app, p["event_id"], "dangerous", None))
        assert result == {"decisions": [{"type": "reject",
                                         "message": result["decisions"][0]["message"]}]}
        assert result["decisions"][0]["type"] == "reject"
        assert any("按拒绝处理" in b["text"] for b in beats)

    def test_abort_clears_the_pending_entry(self):
        # An aborted run's worker is gone, so its pending interrupt can never be
        # resolved; leaving it in the store makes the approval panel offer a card
        # whose resolve wakes nobody.
        app = _fresh_app()
        p = _publish_interrupt(app, "literature", (dict(_ASK),), "t1")[0]
        ev = threading.Event()
        ev.set()

        _beats, result = _drain(
            _await_resolution(app, p["event_id"], "ask_user", ev))
        assert result is None
        assert p["event_id"] not in app._orch_interrupts["pending"]
        assert p["event_id"] not in app._orch_interrupts["events"]


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
