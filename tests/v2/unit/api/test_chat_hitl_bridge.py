"""Private chat can actually be answered — the resolver closes the loop.

Before this, ``ConversationEngine.stream_turn`` had a resume loop and a
``hitl_resolver`` parameter that NO CALLER EVER PASSED. A private-chat turn that
hit a DANGEROUS skill printed "需要人工审批,但当前入口未接审批处理器" and stopped
with the graph parked in its checkpoint; meanwhile the approval panel polled
``_orch_interrupts`` — which the private path never wrote to — and reported
"当前无待处理中断". The gate and its UI were both present and not connected.

These tests drive the real closure end to end: a background thread plays the
engine (calls the resolver and blocks), the test plays the operator (resolves
via the store the way the REST endpoint does), and the assertion is on the
resume value the engine would have fed back into ``Command(resume=…)`` — the
DANGEROUS envelope and the single-value ask_user/workflow_human shape are
different, and getting that wrong resumes the graph with something the tool
cannot read.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/api/test_chat_hitl_bridge.py -x -v
"""
from __future__ import annotations

import threading

import pytest

from mast.api import hitl_bridge as bridge
from mast.api.routes.chat_stream import _make_hitl_resolver


class _App:
    def __init__(self) -> None:
        self._orch_interrupts = {"pending": {}, "events": {}, "resolved": {},
                                 "lock": threading.Lock()}


class _Ctx:
    def __init__(self, app=None) -> None:
        self.live_app = app


class _Intr:
    """A LangGraph Interrupt: the value plus the id a resume must address."""

    def __init__(self, value, id_="lg-1") -> None:
        self.value = value
        self.id = id_


def _run_resolver(resolver, interrupted, out: dict) -> threading.Thread:
    """Play the engine: call the resolver on another thread (it blocks)."""
    def _target():
        out["value"] = resolver(interrupted)
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


def _wait_for_pending(store, timeout: float = 3.0) -> str:
    deadline = threading.Event()
    for _ in range(int(timeout * 100)):
        with store["lock"]:
            ids = list(store["pending"])
        if ids:
            return ids[0]
        deadline.wait(0.01)
    raise AssertionError("resolver never published a pending interrupt")


def _operator_answers(store, event_id: str, value) -> None:
    """What agents_control.resolve does once it has built the decision."""
    with store["lock"]:
        store["resolved"][event_id] = value
    store["events"][event_id].set()


class TestResolverWiring:
    def test_no_live_store_means_no_resolver(self):
        # Standalone dev: keep the honest "no approval channel here" notice
        # rather than pretending a gate exists.
        assert _make_hitl_resolver(_Ctx(None), "literature", "c1", None) is None
        assert _make_hitl_resolver(_Ctx(object()), "literature", "c1", None) is None

    def test_ask_user_resumes_with_the_single_value(self):
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "literature", "c1", None)
        assert resolver is not None

        out: dict = {}
        t = _run_resolver(resolver, ({
            "kind": "ask_user", "question": "先扫哪个区域？",
            "options": [{"label": "A 区"}, {"label": "B 区"}],
            "multi_select": False, "allow_custom": True,
            "timeout_action": "continue",
        },), out)

        eid = _wait_for_pending(store)
        # The panel can see it — this is the half that was missing entirely.
        assert store["pending"][eid]["kind"] == "ask_user"
        assert store["pending"][eid]["agent_id"] == "literature"

        answer = {"selected": ["B 区"], "custom_text": "", "note": ""}
        _operator_answers(store, eid, answer)
        t.join(timeout=3)
        assert not t.is_alive()
        # Verbatim — NOT wrapped in the {"decisions": [...]} envelope.
        assert out["value"] == answer

    def test_dangerous_approval_resumes_with_the_decisions_envelope(self):
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c2", None)

        out: dict = {}
        t = _run_resolver(resolver, ({
            "action_requests": [{"name": "SetBias", "args": {"bias_v": 2.0},
                                 "description": "set bias"}],
            "review_configs": [{"action_name": "SetBias",
                                "allowed_decisions": ["approve", "reject"]}],
        },), out)

        eid = _wait_for_pending(store)
        assert store["pending"][eid]["skill"] == "SetBias"
        _operator_answers(store, eid, {"type": "approve"})
        t.join(timeout=3)
        assert out["value"] == {"decisions": [{"type": "approve"}]}

    def test_several_action_requests_merge_into_one_envelope(self):
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c3", None)

        out: dict = {}
        t = _run_resolver(resolver, ({
            "action_requests": [
                {"name": "SetBias", "args": {}, "description": "a"},
                {"name": "MoveProbeXY", "args": {}, "description": "b"},
            ],
            "review_configs": [],
        },), out)

        # One pending per request (the old code took the first and dropped the
        # rest, so the resume carried fewer decisions than the gate expected).
        for _ in range(300):
            with store["lock"]:
                if len(store["pending"]) == 2:
                    break
            threading.Event().wait(0.01)
        with store["lock"]:
            ids = list(store["pending"])
        assert len(ids) == 2
        for i in ids:
            _operator_answers(store, i, {"type": "approve"})
        t.join(timeout=3)
        assert out["value"] == {"decisions": [{"type": "approve"}, {"type": "approve"}]}

    def test_abort_gives_up_instead_of_resuming(self):
        app = _App()
        store = app._orch_interrupts
        ev = threading.Event()
        resolver = _make_hitl_resolver(_Ctx(app), "literature", "c4", ev)

        out: dict = {}
        t = _run_resolver(resolver, ({
            "kind": "ask_user", "question": "?", "options": [],
            "multi_select": False, "allow_custom": True,
        },), out)
        _wait_for_pending(store)
        ev.set()
        t.join(timeout=3)
        assert not t.is_alive()
        # None → the engine breaks out and leaves the graph in its checkpoint.
        assert out["value"] is None

    def test_unparseable_interrupt_does_not_hang_the_turn(self):
        app = _App()
        resolver = _make_hitl_resolver(_Ctx(app), "literature", "c5", None)
        out: dict = {}
        t = _run_resolver(resolver, ({"kind": "mystery"},), out)
        t.join(timeout=3)
        assert not t.is_alive(), "an unknown interrupt must not block forever"
        assert out["value"] is None


class TestResumeIsAddressedByInterruptId:
    """A resume must name the interrupt it answers.

    ``Command(resume=<bare value>)`` is BROADCAST to every pending interrupt.
    Serially that is harmless, but the moment two branches are paused at once
    one skill's approval also answers the other's pending SetBias — verified on
    the run-task path 2026-07-11, which is why it assembles ``{id: value}``.
    Private chat resumed with a bare value until this test.
    """

    def test_dangerous_resume_is_keyed_by_the_interrupt_id(self):
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c6", None)

        out: dict = {}
        t = _run_resolver(resolver, (_Intr({
            "action_requests": [{"name": "SetBias", "args": {"bias_v": 2.0},
                                 "description": "set bias"}],
            "review_configs": [],
        }, "lg-abc"),), out)

        eid = _wait_for_pending(store)
        assert store["pending"][eid]["lg_id"] == "lg-abc"
        _operator_answers(store, eid, {"type": "approve"})
        t.join(timeout=3)
        assert out["value"] == {"lg-abc": {"decisions": [{"type": "approve"}]}}

    def test_ask_user_keeps_its_single_value_under_the_id(self):
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "literature", "c7", None)

        out: dict = {}
        t = _run_resolver(resolver, (_Intr({
            "kind": "ask_user", "question": "先扫哪个区域？", "options": [],
            "multi_select": False, "allow_custom": True,
        }, "lg-q"),), out)

        eid = _wait_for_pending(store)
        answer = {"selected": ["A"], "custom_text": "", "note": ""}
        _operator_answers(store, eid, answer)
        t.join(timeout=3)
        # Addressed, but NOT wrapped: the ask tool reads the answer directly.
        assert out["value"] == {"lg-q": answer}

    def test_requests_sharing_one_interrupt_merge_under_that_id(self):
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c8", None)

        out: dict = {}
        t = _run_resolver(resolver, (_Intr({
            "action_requests": [
                {"name": "SetBias", "args": {}, "description": "a"},
                {"name": "MoveProbeXY", "args": {}, "description": "b"},
            ],
            "review_configs": [],
        }, "lg-two"),), out)

        for _ in range(300):
            with store["lock"]:
                if len(store["pending"]) == 2:
                    break
            threading.Event().wait(0.01)
        with store["lock"]:
            ids = list(store["pending"])
        assert len(ids) == 2
        for i in ids:
            _operator_answers(store, i, {"type": "approve"})
        t.join(timeout=3)
        # ONE resume value for the interrupt, carrying one decision per request
        # — the middleware raises if the counts disagree.
        assert out["value"] == {
            "lg-two": {"decisions": [{"type": "approve"}, {"type": "approve"}]}}

    def test_an_interrupt_without_an_id_still_resumes(self):
        # Older LangGraph builds expose no id. The private graph is serial, so
        # the broadcast is equivalent — a mapping keyed on None would match
        # nothing and hang the turn.
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c9", None)

        out: dict = {}
        t = _run_resolver(resolver, (_Intr({
            "action_requests": [{"name": "SetBias", "args": {}, "description": ""}],
            "review_configs": [],
        }, None),), out)
        eid = _wait_for_pending(store)
        _operator_answers(store, eid, {"type": "approve"})
        t.join(timeout=3)
        assert out["value"] == {"decisions": [{"type": "approve"}]}


class TestNobodyAnswers:
    """No operator must not mean a wedged conversation."""

    def test_timeout_fails_closed_as_reject(self, monkeypatch):
        # The wait limits are read from the module at call time precisely so
        # this can shorten them; a def-time default would make this test pass
        # in 0.0 s without ever exercising the timeout.
        monkeypatch.setattr(bridge, "MAX_APPROVAL_WAIT_S", 0.2)
        monkeypatch.setattr(bridge, "APPROVAL_BEAT_S", 0.05)
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c10", None)

        out: dict = {}
        t = _run_resolver(resolver, (_Intr({
            "action_requests": [{"name": "SetBias", "args": {"bias_v": 9.0},
                                 "description": "危险偏压"}],
            "review_configs": [],
        }, "lg-t"),), out)
        t.join(timeout=10)
        assert not t.is_alive(), "an unanswered approval must not wait forever"

        resume = out["value"]["lg-t"]
        assert [d["type"] for d in resume["decisions"]] == ["reject"]
        assert "超时" in resume["decisions"][0]["message"]
        # …and the card is gone, so the panel does not offer a resolve that
        # would wake nobody.
        assert not store["pending"] and not store["events"] and not store["resolved"]

    def test_store_is_emptied_on_the_answered_path_too(self):
        app = _App()
        store = app._orch_interrupts
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c11", None)
        out: dict = {}
        t = _run_resolver(resolver, (_Intr({
            "action_requests": [{"name": "SetBias", "args": {}, "description": ""}],
            "review_configs": [],
        }, "lg-ok"),), out)
        eid = _wait_for_pending(store)
        _operator_answers(store, eid, {"type": "approve"})
        t.join(timeout=3)
        assert not store["pending"] and not store["events"] and not store["resolved"]

    def test_store_is_emptied_on_abort(self):
        app = _App()
        store = app._orch_interrupts
        ev = threading.Event()
        resolver = _make_hitl_resolver(_Ctx(app), "instrument_control", "c12", ev)
        out: dict = {}
        t = _run_resolver(resolver, (_Intr({
            "action_requests": [{"name": "SetBias", "args": {}, "description": ""}],
            "review_configs": [],
        }, "lg-ab"),), out)
        _wait_for_pending(store)
        ev.set()
        t.join(timeout=3)
        assert out["value"] is None
        assert not store["pending"] and not store["events"] and not store["resolved"]


class TestSharedImplementationStillServesRunTask:
    def test_orchestrator_wrappers_delegate(self):
        # The run-task path keeps its own names + its own patchable constants;
        # only the body moved.
        from mast.api.routes import orchestrator as orch

        app = _App()
        published = orch._publish_interrupt(app, "literature", ({
            "kind": "ask_user", "question": "q", "options": [],
            "multi_select": False, "allow_custom": True,
        },), "t1")
        assert len(published) == 1
        assert published[0]["event_id"] in app._orch_interrupts["pending"]

    def test_namespace_still_maps_to_the_owning_agent(self):
        from mast.api.routes import orchestrator as orch

        app = _App()
        # An empty namespace is the parent graph → the historical fallback.
        p = orch._publish_interrupt(app, "", ({
            "action_requests": [{"name": "X", "args": {}, "description": ""}],
            "review_configs": [],
        },), "t2")[0]
        assert p["agent_id"] == "instrument_control"

    def test_bridge_defaults_match_the_run_task_constants(self):
        from mast.api.routes import orchestrator as orch

        assert bridge.MAX_APPROVAL_WAIT_S == orch._MAX_APPROVAL_WAIT_S
        assert bridge.APPROVAL_BEAT_S == orch._APPROVAL_BEAT_S


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
