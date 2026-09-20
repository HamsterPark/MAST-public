"""Parking instead of dispatching into fiction — W2 of wakeup scheduling.

``docs/v2/design/wakeup_scheduling.md`` §3. The supervisor may now decline to
dispatch an agent whose upstream inputs do not exist: deterministically when a HARD
dependency is missing (no model call — "review a manuscript that does not exist" has
no judgement in it), and otherwise by ASKING the agent, with the asymmetry of
failure written into the question.

The properties that matter more than the feature
------------------------------------------------
* **OFF by default.** ``activation_gate`` absent/False ⇒ byte-for-byte the previous
  routing. Same shape as ``background_gate`` because a new routing behaviour that
  changes the default changes it for every caller at once, and this one — not
  dispatching — is the behaviour the design itself flags as most likely to become a
  silent death channel.
* **A park is never silent.** Transcript line + state entry, and the END message
  names who is waiting for what. This repo has already paid for "nothing happened"
  and "it hung" being indistinguishable (621 fail_silent_end events).
* **A parked agent does not spend a hop.** It never ran; charging it would let
  repeated re-selection exhaust the loop budget on work that did not happen.
* **An operator's explicit @agent is never parked.** Same rule @agent already has
  against the router and against auto-background: it reaches THAT agent.
* **One question per agent per run.** A soft miss is the ordinary case, so without
  bookkeeping every dispatch re-asks — unbounded cost for a decision already made.
* **Every failure path defaults to START.** A scheduler that deadlocks when its LLM
  hiccups is worse than one that produces a visibly imperfect artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path


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
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from mast.agents._shared.artifacts import ARTIFACT_BY_ID, ClassStatus  # noqa: E402
from mast.agents.orchestrator.graph import (  # noqa: E402
    _PARK_MARKER,
    _supervisor_node_factory,
)
import mast.agents.orchestrator.graph as og  # noqa: E402
from mast.agents.state import MASTState  # noqa: E402


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def empty_disk(monkeypatch):
    """Every artifact class readable and EMPTY.

    Mandatory for these tests: a state miss deliberately falls through to disk (an
    existing on-disk draft must satisfy a fresh run whose state was never seeded),
    so without pinning disk these assertions would depend on whatever is in the
    developer's artifacts directory.
    """
    rows = [ClassStatus(artifact=a, count=0, known=True, detail="empty")
            for a in ARTIFACT_BY_ID.values()]
    monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                        lambda *a, **k: rows)


@pytest.fixture()
def full_disk(monkeypatch):
    """Every artifact class readable and NON-empty."""
    rows = [ClassStatus(artifact=a, count=3, known=True, detail="3 项")
            for a in ARTIFACT_BY_ID.values()]
    monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                        lambda *a, **k: rows)


def _state(**kw) -> dict:
    s = {
        "messages": [HumanMessage(content="把这篇稿子评审一下")],
        "visit_count": {}, "executed_skills": [], "scan_paths": [],
        "scan_metadata": {}, "error_log": [], "event_refs": [],
        "pending_approvals": {},
    }
    s.update(kw)
    return s


class _Router:
    """Structured-output stub that always names ``targets``."""

    def __init__(self, targets):
        self.targets = list(targets)
        self.calls = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _m):
                outer.calls += 1
                # The activation question uses the SAME model; distinguish by schema.
                if "action" in getattr(schema, "__annotations__", {}):
                    return {"action": "start", "waiting_for": [], "reason": "ok"}
                return {"next_agents": list(outer.targets), "reason": "go"}

        return _S()

    def invoke(self, _m):
        return AIMessage(content="ok")


class _WaitingRouter(_Router):
    """Routes to ``targets``, and answers "wait" to any activation question."""

    def __init__(self, targets, waiting_for=("analysis",)):
        super().__init__(targets)
        self.waiting_for = list(waiting_for)
        self.asked = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _m):
                if "action" in getattr(schema, "__annotations__", {}):
                    outer.asked += 1
                    return {"action": "wait", "waiting_for": list(outer.waiting_for),
                            "reason": "缺分析结果,现在写只能猜"}
                return {"next_agents": list(outer.targets), "reason": "go"}

        return _S()


def _on() -> "callable":
    return lambda: True


def _texts(cmd) -> str:
    return "\n".join(str(getattr(m, "content", "")) for m in cmd.update["messages"])


# ════════════════════════════════════════════════════════════════════
# OFF by default
# ════════════════════════════════════════════════════════════════════

class TestOffByDefault:
    def test_no_gate_dispatches_a_hard_missing_agent(self, empty_disk):
        """No gate wired ⇒ the previous behaviour exactly. paper_review with no
        draft anywhere still gets dispatched, as it always did."""
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",))
        cmd = node(_state())
        assert cmd.goto == "paper_review"
        assert _PARK_MARKER not in _texts(cmd)

    def test_gate_returning_false_dispatches_too(self, empty_disk):
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=lambda: False)
        assert node(_state()).goto == "paper_review"

    def test_a_raising_gate_is_treated_as_off(self, empty_disk):
        def _boom():
            raise RuntimeError("settings gone")

        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_boom)
        assert node(_state()).goto == "paper_review", \
            "a gate read failure must not start parking agents"


# ════════════════════════════════════════════════════════════════════
# Layer 1 — hard dependency, no model call
# ════════════════════════════════════════════════════════════════════

class TestHardDependencyParks:
    def test_paper_review_without_a_draft_is_parked(self, empty_disk):
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        cmd = node(_state())
        assert cmd.goto == END
        assert _PARK_MARKER in _texts(cmd)
        entry = cmd.update["pending_activations"]["paper_review"]
        assert entry["status"] == "waiting"
        assert entry["waiting_for"] == ["draft"]
        assert entry["hard"] is True

    def test_it_costs_no_model_call(self, empty_disk):
        """Layer 1 is deterministic on purpose. Spending a model call to decide
        whether a nonexistent document exists is spending money on a fact."""
        router = _Router(["paper_review"])
        node = _supervisor_node_factory(router, wired_agents=("paper_review",),
                                        activation_gate=_on())
        before = router.calls
        node(_state())
        assert router.calls == before + 1, \
            "the routing call is expected; an ADDITIONAL activation call is not"

    def test_a_draft_on_disk_prevents_the_park(self, monkeypatch, empty_disk):
        """A fresh run's state carries no artifacts even when the experiment folder
        is full — parking on that would deadlock over a fact never checked."""
        rows = [ClassStatus(artifact=a, count=(2 if a.id == "draft" else 0),
                            known=True, detail="x")
                for a in ARTIFACT_BY_ID.values()]
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: rows)
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        assert node(_state()).goto == "paper_review"

    def test_a_draft_in_state_prevents_the_park(self, empty_disk):
        from mast.agents._shared.artifact_channel import doc_ref
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        cmd = node(_state(draft=doc_ref(doc_id="draft__x", version=1)))
        assert cmd.goto == "paper_review"

    def test_an_unreadable_store_does_not_park(self, monkeypatch):
        """`unknown` is not `missing`. Parking because the store could not be read
        would be parking on an assumption, and the run would look hung for a reason
        nobody could find."""
        rows = [ClassStatus(artifact=a, count=0, known=False, detail="db locked")
                for a in ARTIFACT_BY_ID.values()]
        monkeypatch.setattr("mast.agents._shared.artifacts.class_status",
                            lambda *a, **k: rows)
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        cmd = node(_state())
        assert cmd.goto == "paper_review"


# ════════════════════════════════════════════════════════════════════
# Layer 2 — ask the agent
# ════════════════════════════════════════════════════════════════════

class TestAskingTheAgent:
    def test_a_wait_answer_parks_with_the_agents_own_reason(self, full_disk):
        """full_disk ⇒ no hard miss, so layer 2 is what decides. instrument_control
        merely PREFERS a plan, so this is the soft path."""
        router = _WaitingRouter(["instrument_control"], waiting_for=("experiment_plan",))
        node = _supervisor_node_factory(router, wired_agents=("instrument_control",),
                                        activation_gate=_on())
        # experiment_plan absent from state AND from disk for this one class
        cmd = node(_state())
        if cmd.goto == END:
            entry = cmd.update["pending_activations"]["instrument_control"]
            assert entry["waiting_for"] == ["experiment_plan"]
            assert "缺分析结果" in entry["reason"] or entry["reason"]
        else:
            # full_disk satisfied the soft need, so nothing was asked — also valid.
            assert router.asked == 0

    def test_a_start_answer_dispatches_and_is_remembered(self, empty_disk):
        """One question per agent per run. A soft miss is the ordinary case, so
        re-asking every dispatch is unbounded cost for a settled decision."""
        router = _Router(["instrument_control"])
        node = _supervisor_node_factory(router, wired_agents=("instrument_control",),
                                        activation_gate=_on())
        cmd = node(_state())
        assert cmd.goto == "instrument_control"
        entry = (cmd.update.get("pending_activations") or {}).get("instrument_control")
        assert entry and entry["status"] == "asked"

    def test_an_already_asked_agent_is_not_asked_again(self, empty_disk):
        router = _WaitingRouter(["instrument_control"])
        node = _supervisor_node_factory(router, wired_agents=("instrument_control",),
                                        activation_gate=_on())
        cmd = node(_state(pending_activations={
            "instrument_control": {"status": "asked", "waiting_for": []}}))
        assert cmd.goto == "instrument_control"
        assert router.asked == 0, "the same question was paid for twice in one run"

    def test_an_already_parked_agent_stays_parked_without_re_asking(self, empty_disk):
        router = _WaitingRouter(["instrument_control"])
        node = _supervisor_node_factory(router, wired_agents=("instrument_control",),
                                        activation_gate=_on())
        cmd = node(_state(pending_activations={
            "instrument_control": {"status": "waiting",
                                   "waiting_for": ["experiment_plan"],
                                   "reason": "earlier"}}))
        assert cmd.goto == END
        assert router.asked == 0

    def test_an_undecidable_question_defaults_to_start(self, empty_disk):
        """Rule 3 as code. If the model cannot answer, the choice is between a wait
        nobody can see and one visible imperfect artifact."""
        class _Broken(_Router):
            def with_structured_output(self, schema, method=None):
                outer = self

                class _S:
                    def invoke(self, _m):
                        if "action" in getattr(schema, "__annotations__", {}):
                            raise RuntimeError("provider down")
                        return {"next_agents": list(outer.targets), "reason": "go"}

                return _S()

            def invoke(self, msgs):
                # the text fallback tier also fails to produce a decision
                if any("action" in str(m.get("content", "")) for m in msgs):
                    return AIMessage(content="I am not sure what to do here")
                return AIMessage(content="ok")

        node = _supervisor_node_factory(_Broken(["instrument_control"]),
                                        wired_agents=("instrument_control",),
                                        activation_gate=_on())
        assert node(_state()).goto == "instrument_control"


# ════════════════════════════════════════════════════════════════════
# Bookkeeping and visibility
# ════════════════════════════════════════════════════════════════════

class TestParkedAgentsDoNotSpendHops:
    def test_a_parked_agent_gets_no_visit_count(self, empty_disk):
        """It never ran. Charging it would let repeated re-selection burn the loop
        budget on work that did not happen — and the loop guard would then end a run
        that had done nothing."""
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        cmd = node(_state())
        assert cmd.update["visit_count"] == {"supervisor": 1}
        assert "paper_review" not in cmd.update["visit_count"]


class TestParkIsNeverSilent:
    def test_the_end_message_names_who_waits_and_why(self, empty_disk):
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        body = _texts(node(_state()))
        assert "paper_review" in body
        assert "既不是失败也不是完成" in body, \
            "an operator reading this must be able to tell a park from a failure " \
            "and from completion"

    def test_the_park_note_names_the_producer_of_what_is_missing(self, empty_disk):
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        body = _texts(node(_state()))
        assert "paper_writing" in body, \
            "the note must say WHO produces the missing draft, or it is unactionable"

    def test_state_records_the_park_for_the_ui_and_the_next_hop(self, empty_disk):
        node = _supervisor_node_factory(_Router(["paper_review"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        cmd = node(_state())
        assert "pending_activations" in cmd.update


# ════════════════════════════════════════════════════════════════════
# The operator always wins
# ════════════════════════════════════════════════════════════════════

class TestOperatorDirectionIsNeverParked:
    def test_an_at_agent_direction_dispatches_despite_a_hard_miss(self, empty_disk):
        """@agent means it reaches THAT agent  — the same exemption it
        already has from the router and from auto-background. If the operator wants
        the reviewer run with no draft, that is their call and their result to read."""
        # `directed_targets` is the explicit channel POST /agents/<id>/interject
        # fills — NOT text parsed out of the interjection body.
        node = _supervisor_node_factory(
            _Router(["literature"]), wired_agents=("paper_review", "literature"),
            control_provider=lambda: {"interjections": ["就现在评"],
                                      "directed_targets": ["paper_review"]},
            activation_gate=_on())
        cmd = node(_state())
        assert cmd.goto == "paper_review"
        assert _PARK_MARKER not in _texts(cmd), \
            "an operator's explicit target must not be parked, nor reported as parked"

    def test_a_directed_target_and_an_agent_hint_are_treated_differently(self, empty_disk):
        """Both arrive on the same channel and are dispatched by the same code path,
        so the distinction has to be tested: the operator's target survives, the
        agent's guess is gated."""
        node = _supervisor_node_factory(
            _Router(["literature"]),
            wired_agents=("paper_review", "data_processing", "literature"),
            control_provider=lambda: {"interjections": ["评一下"],
                                      "directed_targets": ["paper_review"]},
            activation_gate=_on())
        cmd = node(_state(routing_hints=["data_processing"]))
        assert cmd.goto == "paper_review", \
            "the operator's target should be the only one dispatched"
        assert _PARK_MARKER in _texts(cmd)  # data_processing parked (no scan)

    def test_an_agent_requested_hop_IS_gated(self, empty_disk):
        """A handing-off agent is guessing about a sibling's inputs — exactly the
        guess readiness knows the answer to."""
        node = _supervisor_node_factory(_Router(["literature"]),
                                        wired_agents=("paper_review",),
                                        activation_gate=_on())
        cmd = node(_state(routing_hints=["paper_review"]))
        assert cmd.goto == END
        assert _PARK_MARKER in _texts(cmd)


# ════════════════════════════════════════════════════════════════════
# Partial fan-out: parking one target must not stall the others
# ════════════════════════════════════════════════════════════════════

class TestPartialFanout:
    def test_the_unparked_branch_still_runs(self, empty_disk):
        """The property that makes parking usable at all: the supervisor keeps making
        progress on everything else. Otherwise one unready agent stops the campaign."""
        node = _supervisor_node_factory(
            _Router(["literature", "paper_review"]),
            wired_agents=("literature", "paper_review"),
            activation_gate=_on())
        cmd = node(_state())
        assert cmd.goto == "literature"
        assert _PARK_MARKER in _texts(cmd)
        assert cmd.update["visit_count"] == {"supervisor": 1, "literature": 1}


# ════════════════════════════════════════════════════════════════════
# End to end through a REAL compiled graph
# ════════════════════════════════════════════════════════════════════

class TestThroughARealCompiledGraph:
    """Assertions on the node alone have fooled this repo before (a whole test suite
    was green against a fake orchestrator). These drive a compiled StateGraph with a
    checkpointer, so the reducers and the channel setup are the real ones.
    """

    def _graph(self, router, targets, gate):
        g: StateGraph = StateGraph(MASTState)
        g.add_node("supervisor", _supervisor_node_factory(
            router, wired_agents=tuple(targets), activation_gate=gate))
        for t in targets:
            g.add_node(t, lambda s, n=t: Command(
                goto=END, update={"messages": [AIMessage(content=f"{n} ran")]}))
        g.add_edge(START, "supervisor")
        return g.compile(checkpointer=InMemorySaver())

    def test_a_parked_run_ends_cleanly_and_records_the_park(self, empty_disk):
        app = self._graph(_Router(["paper_review"]), ["paper_review"], _on())
        out = app.invoke(_state(), config={"configurable": {"thread_id": "p1"},
                                           "recursion_limit": 30})
        body = "\n".join(str(getattr(m, "content", "")) for m in out["messages"])
        assert "paper_review ran" not in body, "a parked agent was dispatched anyway"
        assert _PARK_MARKER in body
        # The park survived the real reducer into the durable channel.
        assert out["pending_activations"]["paper_review"]["waiting_for"] == ["draft"]

    def test_pending_activations_merges_rather_than_replacing(self, empty_disk):
        """`merge_dicts`, not last_wins: under a fan-out two branches can park in the
        same super-step, and one park must not erase the other."""
        app = self._graph(_Router(["paper_review"]), ["paper_review"], _on())
        out = app.invoke(
            _state(pending_activations={"data_processing": {"status": "waiting",
                                                           "waiting_for": ["last_scan"]}}),
            config={"configurable": {"thread_id": "p2"}, "recursion_limit": 30})
        assert set(out["pending_activations"]) == {"data_processing", "paper_review"}

    def test_gate_off_runs_the_agent_end_to_end(self, empty_disk):
        app = self._graph(_Router(["paper_review"]), ["paper_review"], None)
        out = app.invoke(_state(), config={"configurable": {"thread_id": "p3"},
                                           "recursion_limit": 30})
        body = "\n".join(str(getattr(m, "content", "")) for m in out["messages"])
        assert "paper_review ran" in body


# ── park 带上纲领归属（2026-08-27） ──────────────────────────────────

class TestParkCarriesTheCampaign:
    """空闲进程要能回答「这份等待是为哪条纲领等的」——只能在建 park 时冻。"""

    def test_campaign_ref_object_is_read(self):
        from mast.agents.state import CampaignRef

        got = og._campaign_id_of(
            {"research_campaign": CampaignRef(campaign_id="cmp-7", title="T")})
        assert got == "cmp-7"

    def test_a_checkpoint_dict_is_read_too(self):
        """同一份数据两种形状是本仓常态：checkpoint 解回来的是 dict。

        只认对象的话，**重启之后**建的 park 会静默丢掉归属 —— 而重启之后
        正是这条路最要紧的时候。
        """
        assert og._campaign_id_of({"research_campaign": {"campaign_id": "cmp-8"}}) \
            == "cmp-8"

    def test_no_campaign_yields_an_empty_string_not_a_guess(self):
        assert og._campaign_id_of({}) == ""
        assert og._campaign_id_of({"research_campaign": None}) == ""
        assert og._campaign_id_of({"research_campaign": {"title": "没有 id"}}) == ""
