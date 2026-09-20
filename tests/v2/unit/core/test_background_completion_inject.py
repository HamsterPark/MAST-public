"""Cross-run dependency injection (item ③).

When a background run finishes cleanly the runtime fires ``_on_background_complete``
(the manager's completion_sink). It injects an ADVISORY note into the FOREGROUND
supervisor's interjection queue — surfacing the result + any EXPLICITLY declared
follow-up — so the operator/supervisor can decide the next step.

Hard contract pinned here:
  * the note is delivered ONLY while a foreground task is actively streaming;
  * it is a PLAIN (agent_id="_supervisor") note the router weighs — never a
    directed dispatch, never a tool/skill/hardware call (nothing auto-executes);
  * the queue is bounded so a burst of completions can't grow it unbounded.
"""
from __future__ import annotations

import threading

from mast.core.background_runs import BackgroundRun
from mast.core.runtime import CoreRuntime


def _rt(active: bool) -> CoreRuntime:
    # build a bare runtime shell — we only exercise the injection method, so we
    # skip the heavy setup() and hand it just the Agents-API state it reads.
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._agents_api_state = {
        "lock": threading.RLock(),
        "interjects": [],
        "task": {"active": active, "conversation_id": "c1"},
    }
    return rt


def _done_run(**kw) -> BackgroundRun:
    base = dict(run_id="r1", conversation_id="c1", instruction="综述石墨烯",
                agents=("literature",), title="综述石墨烯", status="done",
                final_text="找到 12 篇文献")
    base.update(kw)
    return BackgroundRun(**base)


def test_completion_injects_a_supervisor_hint_when_task_active():
    rt = _rt(active=True)
    rt._on_background_complete(_done_run(
        on_done={"next_agent": "experiment_design", "note": "据综述拟方案"}))
    q = rt._agents_api_state["interjects"]
    assert len(q) == 1
    note = q[0]
    # non-directed → the router decides; nothing is auto-dispatched
    assert note["agent_id"] == "_supervisor"
    assert note["kind"] == "background_completion"
    assert "综述石墨烯" in note["text"]
    assert "experiment_design" in note["text"]      # the DECLARED follow-up
    assert "不要自动" in note["text"]                 # advisory-only wording


def test_completion_is_silent_when_no_active_foreground_task():
    rt = _rt(active=False)
    rt._on_background_complete(_done_run())
    # nobody to hint (result is already in the durable transcript) → no interjection
    assert rt._agents_api_state["interjects"] == []


def test_completion_without_declared_followup_still_surfaces_result():
    rt = _rt(active=True)
    rt._on_background_complete(_done_run(on_done=None))
    q = rt._agents_api_state["interjects"]
    assert len(q) == 1
    assert "已完成" in q[0]["text"]
    assert "experiment_design" not in q[0]["text"]   # no invented follow-up


def test_completion_queue_is_bounded():
    rt = _rt(active=True)
    cap = CoreRuntime._BG_COMPLETION_NOTE_CAP
    for i in range(cap + 5):
        rt._on_background_complete(_done_run(run_id=f"r{i}"))
    assert len(rt._agents_api_state["interjects"]) == cap


def test_completion_never_raises_on_missing_state():
    # a runtime with no agents-api state must degrade silently, never raise
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._agents_api_state = None
    rt._on_background_complete(_done_run())   # no exception
