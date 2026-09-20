"""W5 — a woken run can SEE the mainline's work, and its own work gets back.

``docs/v2/design/wakeup_scheduling.md`` §5. Three pieces:

* ``seed_artifacts`` — a background run is state-ISOLATED (own ``thread_id``, own
  ``InMemorySaver``), which is what makes it safe to run concurrently and also what
  made it blind. The snapshot crosses BY VALUE because LangGraph checkpoints are
  per-thread and two runs writing one thread corrupt each other. It works at all only
  because the channel carries POINTERS — the bodies are on disk, which is shared and
  authoritative, so ``load_document`` in the woken run reads the real latest text.
* the **return inbox** — a woken run cannot write the mainline's checkpoint either,
  so its products are posted and drained at the mainline's next dispatch. Without it
  a woken agent's work would exist on disk and be invisible to everyone, which is the
  same as not having done it.
* ``request_activation`` — "一个 agent 可以提要求说我觉得应该激活我的朋友", as a
  suggestion routed through the supervisor, never a direct jump.

The two properties easiest to get wrong, both tested here:

* the snapshot must be COPIED, not referenced — a reference silently turns "snapshot"
  into "whatever it is now", which is the one thing §1.4 forbids;
* the merge back must respect VERSIONS. The channels use ``last_wins``, which is
  right for concurrent writes in one super-step and wrong for a delivery from
  outside: the mainline may have moved on, and blind last-wins would overwrite a
  newer pointer with the older one the woken run started from.
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

from mast.agents._shared.artifact_channel import doc_ref  # noqa: E402
from mast.agents._shared.environment_tools import request_activation  # noqa: E402
from mast.agents.orchestrator.graph import _artifact_is_newer  # noqa: E402
from mast.core import park_board as pb  # noqa: E402


def _call(**args):
    """Invoke the tool the way the framework does — a full ToolCall envelope.

    ``InjectedToolCallId`` REQUIRES it; a bare kwargs dict raises. Recorded here
    because it is a real trap: a test that called it the convenient way would fail
    for a reason that has nothing to do with the tool's behaviour.
    """
    return request_activation.invoke({
        "args": args, "name": "request_activation",
        "type": "tool_call", "id": "tc-1", "tool_call_id": "tc-1",
    })


# ════════════════════════════════════════════════════════════════════
# request_activation — proposes, never commands
# ════════════════════════════════════════════════════════════════════

class TestRequestActivation:
    def test_it_writes_a_routing_hint_not_a_jump(self):
        """The 2026-05-30 removal of agent→agent dispatch stands: a direct edge makes
        an A→B→A ping-pong bounded by nothing but recursion_limit. Going through the
        channel the supervisor already consumes keeps every loop guard, the budget
        gate and the activation check in the path."""
        cmd = _call(agent="paper_review", why="草稿写完了")
        assert cmd.update["routing_hints"] == ["paper_review"]
        assert not getattr(cmd, "goto", None), \
            "the tool jumped directly to an agent instead of proposing"

    def test_instrument_control_can_never_be_requested(self):
        """The sole hardware agent stays foreground and interactive, so no automatic
        mechanism may ever end in a hardware command."""
        cmd = _call(agent="instrument_control", why="扫一下")
        assert "routing_hints" not in cmd.update
        assert "instrument_control" in cmd.update["messages"][0].content

    def test_an_unknown_agent_is_refused_with_the_real_options(self):
        cmd = _call(agent="wizard", why="x")
        assert "routing_hints" not in cmd.update
        body = cmd.update["messages"][0].content
        assert "literature" in body and "paper_review" in body

    def test_a_missing_reason_is_refused(self):
        """"Activate X" with no why gives the supervisor nothing to judge."""
        cmd = _call(agent="paper_review", why="  ")
        assert "routing_hints" not in cmd.update

    def test_waiting_for_must_be_in_the_closed_set(self):
        """Free text cannot be matched against a future arrival — accepting it would
        create a request that can never be satisfied."""
        cmd = _call(agent="paper_writing", why="x", waiting_for="更多好数据")
        assert "routing_hints" not in cmd.update
        assert "永远不会被叫醒" in cmd.update["messages"][0].content

    def test_a_valid_waiting_for_is_accepted(self):
        cmd = _call(agent="paper_writing", why="x", waiting_for="analysis")
        assert cmd.update["routing_hints"] == ["paper_writing"]

    def test_an_instruction_rides_along(self):
        """The park entry has an ``instruction`` field; without this parameter that
        field could never be filled from this path."""
        cmd = _call(agent="paper_review", why="该评审了", instruction="重点看误差分析")
        assert "重点看误差分析" in cmd.update["messages"][0].content

    def test_the_tool_result_is_paired_with_its_call_id(self):
        """An unpaired tool result is the 400 this repo has already debugged
        (tool_call_id mismatch → orphan ToolMessage)."""
        cmd = _call(agent="paper_review", why="x")
        assert cmd.update["messages"][0].tool_call_id == "tc-1"

    def test_it_reaches_every_agent_not_just_instrument_control(self):
        """``spawn_background_task`` already existed but is a META-tool, so only
        instrument_control had it — five of six agents had no way at all to say
        "someone else should do this next"."""
        import inspect

        from mast.core import runtime as rt
        src = inspect.getsource(rt)
        assert src.count("ENVIRONMENT_TOOLS") >= 2, \
            "request_activation reaches only one of {group, private chat}"


# ════════════════════════════════════════════════════════════════════
# Version-aware merge — last_wins is wrong for an outside delivery
# ════════════════════════════════════════════════════════════════════

class TestVersionAwareMerge:
    def test_an_empty_slot_accepts_anything(self):
        assert _artifact_is_newer(doc_ref(doc_id="d", version=1), None)
        assert _artifact_is_newer(doc_ref(doc_id="d", version=1), {})

    def test_a_newer_version_wins(self):
        assert _artifact_is_newer(doc_ref(doc_id="d", version=5),
                                  doc_ref(doc_id="d", version=3))

    def test_an_older_version_does_not_overwrite(self):
        """THE case this exists for: a woken run produced from a snapshot taken when
        it was spawned, while the mainline moved on. Blind last-wins would show the
        agent a stale summary AND a stale version to continue from."""
        assert not _artifact_is_newer(doc_ref(doc_id="d", version=2),
                                      doc_ref(doc_id="d", version=7))

    def test_an_equal_version_does_not_overwrite(self):
        assert not _artifact_is_newer(doc_ref(doc_id="d", version=4),
                                      doc_ref(doc_id="d", version=4))

    def test_an_unversioned_incoming_does_not_overwrite_an_occupied_slot(self):
        """Refusing an ambiguous overwrite is the safe direction: the woken run's
        product is still on disk and still discoverable, whereas a clobbered pointer
        is silently gone."""
        assert not _artifact_is_newer({"summary": "x"},
                                      doc_ref(doc_id="d", version=1))

    def test_dicts_and_objects_compare_the_same_way(self):
        assert _artifact_is_newer({"doc_id": "d", "version": 9},
                                  doc_ref(doc_id="d", version=2))


# ════════════════════════════════════════════════════════════════════
# The return inbox
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_PARK_BOARD_DIR", str(tmp_path))
    pb.set_board_for_test(None)
    yield
    pb.set_board_for_test(None)


@pytest.fixture()
def board(tmp_path):
    b = pb.ParkBoard(tmp_path / "pb.json")
    pb.set_board_for_test(b)
    return b


class TestReturnInbox:
    def test_products_posted_by_a_woken_run_can_be_drained(self, board):
        board.post_return(run_id="r1", park_id="", agent="paper_review",
                          artifacts={"review": doc_ref(doc_id="rv1", version=1)})
        drained = board.drain_returns()
        assert len(drained) == 1
        assert "review" in drained[0]["artifacts"]

    def test_draining_twice_yields_nothing_the_second_time(self, board):
        """Marking inside the same lock as the read is what makes a concurrent second
        drain (a fan-out where two branches dispatch) return nothing rather than the
        same products twice."""
        board.post_return(run_id="r1", park_id="", agent="a",
                          artifacts={"review": doc_ref(doc_id="rv1", version=1)})
        assert len(board.drain_returns()) == 1
        assert board.drain_returns() == []

    def test_a_re_post_of_the_same_run_is_deduped(self, board):
        board.post_return(run_id="r1", park_id="", agent="a",
                          artifacts={"review": doc_ref(doc_id="rv1", version=1)})
        board.post_return(run_id="r1", park_id="", agent="a",
                          artifacts={"review": doc_ref(doc_id="rv1", version=1)})
        assert len(board.drain_returns()) == 1

    def test_an_empty_product_set_posts_nothing(self, board):
        assert board.post_return(run_id="r1", park_id="", agent="a",
                                 artifacts={}) == {}
        assert board.peek_returns() == []

    def test_the_inbox_survives_a_restart(self, tmp_path):
        """A woken run can finish while nothing else is running; the delivery has to
        wait on disk until the mainline next dispatches.

        Regression, 2026-07-30: the channel carries ``DocRef`` — a Pydantic model, not
        a dict — and ``json.dumps`` refuses it. Because ``_save`` is best-effort the
        write failed, logged at debug level, and ``post_return`` still reported
        success: the inbox worked perfectly until the process ended, which for a
        mailbox whose whole job is outliving the process is the worst failure
        available. Hence ``_jsonable`` on the way in.
        """
        p = tmp_path / "b.json"
        b1 = pb.ParkBoard(p)
        b1.post_return(run_id="r1", park_id="", agent="a",
                       artifacts={"review": doc_ref(doc_id="rv1", version=1)})
        b2 = pb.ParkBoard(p)
        drained = b2.drain_returns()
        assert len(drained) == 1
        # and the pointer is still usable after the round trip
        assert drained[0]["artifacts"]["review"]["doc_id"] == "rv1"
        assert drained[0]["artifacts"]["review"]["version"] == 1

    def test_a_pydantic_payload_really_reaches_the_file(self, tmp_path):
        """Direct check on the file, not on the in-memory copy — the bug above was
        invisible from memory."""
        import json as _json

        p = tmp_path / "b.json"
        b = pb.ParkBoard(p)
        b.post_return(run_id="r1", park_id="", agent="a",
                      artifacts={"review": doc_ref(doc_id="rv1", version=3)})
        raw = _json.loads(p.read_text(encoding="utf-8"))
        assert raw["inbox"][0]["artifacts"]["review"]["version"] == 3

    def test_a_pydantic_snapshot_on_a_park_also_persists(self, tmp_path):
        """Same hazard, the other payload: a park's ``artifact_snapshot`` is what a
        woken run is seeded from, so losing it silently would wake an agent blind."""
        import json as _json

        p = tmp_path / "b.json"
        b = pb.ParkBoard(p)
        b.park("paper_review", waiting_for=["draft"], experiment_id="e1",
               artifact_snapshot={"draft": doc_ref(doc_id="d9", version=2)})
        raw = _json.loads(p.read_text(encoding="utf-8"))
        assert raw["parks"][0]["artifact_snapshot"]["draft"]["doc_id"] == "d9"

    def test_peek_does_not_consume(self, board):
        board.post_return(run_id="r1", park_id="", agent="a",
                          artifacts={"review": doc_ref(doc_id="rv1", version=1)})
        assert len(board.peek_returns()) == 1
        assert len(board.drain_returns()) == 1


class TestParkClosureOnReturn:
    def test_a_park_is_only_done_once_its_products_return(self, board):
        """``woken`` is deliberately not terminal: a detached run on an InMemorySaver
        can die and take its work with it, and a board that called ``woken`` finished
        would have no record that the chain broke."""
        rec = board.park("paper_review", waiting_for=["draft"], experiment_id="e1")
        board.mark_woken(rec["park_id"], "r1")
        assert board.get(rec["park_id"])["status"] == "woken"
        board.mark_done(rec["park_id"], note="产物已回主线")
        assert board.get(rec["park_id"])["status"] == "done"

    def test_a_woken_park_is_not_offered_to_a_new_run_as_waiting(self, board):
        """It is in flight, not waiting — re-parking it would double-dispatch."""
        rec = board.park("paper_review", waiting_for=["draft"], experiment_id="e1")
        board.mark_woken(rec["park_id"], "r1")
        assert board.as_state_cache("e1") == {}


class TestSeedArtifactsPlumbing:
    def test_spawn_copies_the_snapshot_rather_than_referencing_it(self):
        """A reference silently turns "snapshot" into "whatever it is now" — the one
        thing §1.4 says this must never be, because the mainline's channel keeps
        changing after spawn returns."""
        from mast.core.background_runs import BackgroundRunManager

        mgr = BackgroundRunManager(run_fn=lambda *a, **k: "")
        live = {"draft": doc_ref(doc_id="d1", version=1)}
        rec = mgr.spawn(instruction="go", agents=("paper_review",),
                        seed_artifacts=live)
        live["draft"] = doc_ref(doc_id="d1", version=99)   # mainline moves on
        run = mgr.get(rec["run_id"])
        assert run is not None
        stored = mgr._runs[rec["run_id"]].seed_artifacts
        assert stored["draft"].version == 1, \
            "the snapshot followed the live state — it is a reference, not a copy"
        mgr.abort(rec["run_id"])

    def test_a_seeded_run_records_when_the_snapshot_was_taken(self):
        """A snapshot goes stale between spawn and execution, and the woken agent is
        told so rather than being allowed to assume it is live."""
        from mast.core.background_runs import BackgroundRunManager

        mgr = BackgroundRunManager(run_fn=lambda *a, **k: "")
        rec = mgr.spawn(instruction="go", agents=("paper_review",),
                        seed_artifacts={"draft": doc_ref(doc_id="d1", version=1)})
        assert mgr._runs[rec["run_id"]].snapshot_at > 0
        assert rec["snapshot_at"] > 0
        mgr.abort(rec["run_id"])

    def test_an_unseeded_run_is_unchanged(self):
        """Zero behaviour change for every existing caller."""
        from mast.core.background_runs import BackgroundRunManager

        mgr = BackgroundRunManager(run_fn=lambda *a, **k: "")
        rec = mgr.spawn(instruction="go", agents=("literature",))
        assert mgr._runs[rec["run_id"]].seed_artifacts is None
        assert rec["snapshot_at"] == 0.0
        mgr.abort(rec["run_id"])

    def test_the_park_id_links_the_run_back_to_its_board_entry(self):
        from mast.core.background_runs import BackgroundRunManager

        mgr = BackgroundRunManager(run_fn=lambda *a, **k: "")
        rec = mgr.spawn(instruction="go", agents=("paper_review",), park_id="pk-1")
        assert rec["park_id"] == "pk-1"
        mgr.abort(rec["run_id"])

    def test_the_seed_notice_says_it_is_a_snapshot_and_how_to_get_the_truth(self):
        """Not dressing a snapshot up as live data is the whole of the honesty here."""
        from mast.core.runtime import CoreRuntime

        text = CoreRuntime._seed_notice({"draft": doc_ref(doc_id="d", version=1)},
                                        __import__("time").time() - 600)
        assert "快照" in text
        assert "load_document" in text and "最新版" in text
        assert "survey_environment" in text
