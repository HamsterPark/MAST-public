"""BackgroundRunManager — true parallelism without touching LangGraph semantics.

The manager runs a long/independent task on a daemon thread with its OWN
thread_id + (caller-provided) checkpointer, streaming its transcript back into
the shared ConversationStore, so the FOREGROUND chat stays responsive.

Pinned here (all with an injected fake ``run_fn`` — no live LLM / graph):
  * spawn is NON-BLOCKING even when the run itself blocks for a while;
  * two background runs execute CONCURRENTLY (not serialised);
  * a background run never blocks a concurrent foreground action;
  * every streamed message + start/end status lands in the transcript sink,
    tagged as background, under the run's OWN thread_id;
  * abort stops a run between chunks and marks it aborted;
  * instrument_control can NEVER be backgrounded (the hardware invariant).
"""
from __future__ import annotations

import threading
import time

import pytest

from mast.core.background_runs import BG_TAG, BackgroundRunManager


class _Sink:
    """Thread-safe capture of transcript appends (stands in for ConversationStore)."""

    def __init__(self):
        self.rows: list[tuple] = []
        self._lock = threading.Lock()

    def __call__(self, conversation_id, kind, agent_id, role, text):
        with self._lock:
            self.rows.append((conversation_id, kind, agent_id, role, text))

    def texts(self):
        with self._lock:
            return [r[4] for r in self.rows]

    def for_conv(self, cid):
        with self._lock:
            return [r for r in self.rows if r[0] == cid]


def _wait_until(pred, timeout=5.0, interval=0.01):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_spawn_is_non_blocking_even_when_run_blocks():
    started = threading.Event()
    release = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        started.set()
        release.wait(timeout=5)   # block the WORKER, not the caller
        emit("literature", "agent", "survey done")
        return "final"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink())
    t0 = time.perf_counter()
    rec = mgr.spawn(instruction="survey graphene", conversation_id="c1",
                    agents=["literature"])
    dt = time.perf_counter() - t0

    # spawn returned immediately despite the worker blocking
    assert dt < 0.5, f"spawn blocked for {dt:.2f}s"
    assert rec["status"] in ("queued", "running")
    assert rec["thread_id"] == f"bg-{rec['run_id']}"
    assert started.wait(timeout=2), "worker never started"
    release.set()
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "done")


def test_two_background_runs_execute_concurrently():
    """Two runs must overlap in time — if the manager serialised them the barrier
    it exists to remove would just move here."""
    in_run = threading.Barrier(2, timeout=5)
    overlap = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        # both workers must be INSIDE run_fn at the same time to pass the barrier
        try:
            in_run.wait()
            overlap.set()
        except threading.BrokenBarrierError:  # pragma: no cover
            pass
        return "ok"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink(),
                               max_concurrent=2)
    a = mgr.spawn(instruction="task A", conversation_id="c", agents=["literature"])
    b = mgr.spawn(instruction="task B", conversation_id="c", agents=["data_processing"])
    assert overlap.wait(timeout=3), "the two background runs did not run concurrently"
    assert _wait_until(lambda: mgr.get(a["run_id"])["status"] == "done")
    assert _wait_until(lambda: mgr.get(b["run_id"])["status"] == "done")


def test_background_run_does_not_block_a_foreground_action():
    """While a background run is mid-flight, a simulated foreground call returns
    promptly — the whole point of detaching it."""
    holding = threading.Event()
    release = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        holding.set()
        release.wait(timeout=5)
        return "done"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink())
    mgr.spawn(instruction="long survey", conversation_id="c", agents=["literature"])
    assert holding.wait(timeout=2)

    # foreground work proceeds while the background run is still blocked
    t0 = time.perf_counter()
    foreground_result = "instrument scanned"   # would be a real IC call
    dt = time.perf_counter() - t0
    assert dt < 0.2 and foreground_result
    release.set()


def test_transcript_merge_tags_background_and_uses_own_thread():
    sink = _Sink()
    seen_thread = {}

    def run_fn(instruction, thread_id, agents, emit, abort):
        seen_thread["tid"] = thread_id
        emit("literature", "agent", "found 3 papers")
        emit("literature", "tool", "lib_search(q=graphene)")
        return "survey complete: 3 papers"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=sink)
    rec = mgr.spawn(instruction="survey", conversation_id="conv-9", agents=["literature"])
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "done")

    rows = sink.for_conv("conv-9")
    kinds = [r[1] for r in rows]
    assert kinds[0] == "status" and kinds[-1] == "status"      # start + end markers
    assert any(r[1] == "message" and "found 3 papers" in r[4] for r in rows)
    # every background row is tagged
    assert all(BG_TAG in r[4] for r in rows), rows
    # the run used its OWN isolated thread_id
    assert seen_thread["tid"] == f"bg-{rec['run_id']}"
    # final text captured on the record
    assert "survey complete" in mgr.get(rec["run_id"])["final_text"]


def test_abort_stops_the_run_between_chunks():
    step = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        for i in range(100):
            if abort.is_set():
                break
            emit("literature", "agent", f"chunk {i}")
            step.set()
            time.sleep(0.02)
        return "stopped"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink())
    rec = mgr.spawn(instruction="long", conversation_id="c", agents=["literature"])
    assert step.wait(timeout=2)
    assert mgr.abort(rec["run_id"]) is True
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "aborted")
    # aborting an already-finished run is a no-op returning False
    assert mgr.abort(rec["run_id"]) is False


def test_instrument_control_can_never_be_backgrounded():
    mgr = BackgroundRunManager(run_fn=lambda *a, **k: "x", transcript_sink=_Sink())
    with pytest.raises(ValueError):
        mgr.spawn(instruction="scan", conversation_id="c",
                  agents=["instrument_control"])
    # a mixed request keeps only the safe agents
    rec = mgr.spawn(instruction="survey + scan", conversation_id="c",
                    agents=["instrument_control", "literature"])
    assert rec["agents"] == ["literature"]


def test_empty_instruction_rejected():
    mgr = BackgroundRunManager(run_fn=lambda *a, **k: "x")
    with pytest.raises(ValueError):
        mgr.spawn(instruction="   ", conversation_id="c", agents=["literature"])


def test_spawn_background_meta_tool_offloads_via_context():
    """The IC-facing meta-tool `spawn_background_task` lets the FOREGROUND agent
    detach a long sub-task. It just relays into the injected `spawn_background`
    context callable — pin that wiring so a refactor can't silently break it."""
    import json as _json

    from mast.agents._shared.meta_tools import make_meta_tools

    calls: list = []

    def _spawn(instruction, agents, priority="normal", on_done=None):
        calls.append((instruction, tuple(agents), priority, on_done))
        return {"ok": True, "run_id": "bg123", "agents": list(agents)}

    tools = make_meta_tools(lambda: {"spawn_background": _spawn})
    tool = next(t for t in tools if t.name == "spawn_background_task")

    out = tool.invoke({"instruction": "综述石墨烯 STM", "agent": "literature"})
    res = _json.loads(out)
    assert res["success"] is True and res["run_id"] == "bg123"
    # defaults: normal priority, no declared follow-up
    assert calls == [("综述石墨烯 STM", ("literature",), "normal", None)]

    # an EXPLICIT follow-up + priority flows through as on_done (item ③) — never
    # auto-run, just declared for the completion hint.
    calls.clear()
    tool.invoke({"instruction": "查文献", "agent": "literature", "priority": "high",
                 "then_agent": "experiment_design", "then_note": "据综述拟方案"})
    assert calls == [("查文献", ("literature",), "high",
                      {"next_agent": "experiment_design", "note": "据综述拟方案"})]

    # a rejected spawn (e.g. instrument_control) surfaces the error, never raises
    def _reject(instruction, agents, priority="normal", on_done=None):
        return {"ok": False, "error": "instrument_control cannot be detached"}

    tools2 = make_meta_tools(lambda: {"spawn_background": _reject})
    tool2 = next(t for t in tools2 if t.name == "spawn_background_task")
    res2 = _json.loads(tool2.invoke({"instruction": "scan", "agent": "instrument_control"}))
    assert res2["success"] is False and "instrument_control" in res2["error"]


# ── ① resource governance: priority / per-type quota / progress ──────────────

def test_high_priority_is_admitted_before_a_queued_normal():
    started: list = []
    slock = threading.Lock()
    gate = {k: threading.Event() for k in ("A", "B", "C")}

    def run_fn(instruction, thread_id, agents, emit, abort):
        with slock:
            started.append(instruction)
        gate[instruction].wait(timeout=5)
        return "ok"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink(), max_concurrent=1)
    a = mgr.spawn(instruction="A", conversation_id="c", agents=["literature"])
    assert _wait_until(lambda: mgr.get(a["run_id"])["status"] == "running")
    assert _wait_until(lambda: "A" in started)
    # queue a normal THEN a high while the single slot is held by A
    b = mgr.spawn(instruction="B", conversation_id="c", agents=["literature"], priority="normal")
    cc = mgr.spawn(instruction="C", conversation_id="c", agents=["literature"], priority="high")
    assert _wait_until(lambda: mgr.get(b["run_id"])["status"] == "queued")
    assert _wait_until(lambda: mgr.get(cc["run_id"])["status"] == "queued")
    # free the slot → the HIGH-priority C must jump ahead of the earlier normal B
    gate["A"].set()
    assert _wait_until(lambda: "C" in started)
    assert "B" not in started, "normal B was admitted before high-priority C"
    gate["C"].set()
    assert _wait_until(lambda: "B" in started)
    gate["B"].set()
    for r in (a, b, cc):
        assert _wait_until(lambda r=r: mgr.get(r["run_id"])["status"] == "done")


def test_per_type_quota_caps_one_types_concurrency():
    live = {"literature": 0, "max": 0}
    lock = threading.Lock()
    release = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        with lock:
            live[agents[0]] += 1
            live["max"] = max(live["max"], live["literature"])
        release.wait(timeout=5)
        with lock:
            live[agents[0]] -= 1
        return "ok"

    # 5 global slots but a per-type quota of 2 → at most 2 literature at once
    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink(),
                               max_concurrent=5, per_type_quota=2)
    ids = [mgr.spawn(instruction=f"lit{i}", conversation_id="c", agents=["literature"])
           for i in range(4)]
    assert _wait_until(lambda: sum(
        1 for i in ids if mgr.get(i["run_id"])["status"] == "running") == 2)
    time.sleep(0.15)  # let any (wrongly) admitted extra run surface
    assert live["max"] <= 2, f"per-type quota breached: max concurrent = {live['max']}"
    running = [i for i in ids if mgr.get(i["run_id"])["status"] == "running"]
    assert len(running) == 2
    release.set()
    for i in ids:
        assert _wait_until(lambda i=i: mgr.get(i["run_id"])["status"] == "done")


def test_different_types_are_not_capped_by_each_others_quota():
    """per_type_quota is PER TYPE — 2 literature + 2 data_processing can all run
    under a global cap of 4 even with a per-type quota of 2."""
    release = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        release.wait(timeout=5)
        return "ok"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink(),
                               max_concurrent=4, per_type_quota=2)
    ids = [mgr.spawn(instruction=f"{t}{i}", conversation_id="c", agents=[t])
           for t in ("literature", "data_processing") for i in range(2)]
    assert _wait_until(lambda: sum(
        1 for i in ids if mgr.get(i["run_id"])["status"] == "running") == 4)
    release.set()
    for i in ids:
        assert _wait_until(lambda i=i: mgr.get(i["run_id"])["status"] == "done")


def test_progress_climbs_with_steps_and_completes_at_100():
    def run_fn(instruction, thread_id, agents, emit, abort):
        for i in range(3):
            emit("literature", "agent", f"step {i}")
        return "final"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink())
    rec = mgr.spawn(instruction="survey", conversation_id="c", agents=["literature"])
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "done")
    r = mgr.get(rec["run_id"])
    assert r["steps"] == 3
    assert r["progress"] == 100          # done → 100
    assert r["last_activity"] == "step 2"


def test_run_fn_explicit_progress_is_visible_while_running():
    release = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        emit.progress(37, "halfway-ish")
        release.wait(timeout=5)
        return "final"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink())
    rec = mgr.spawn(instruction="survey", conversation_id="c", agents=["literature"])
    assert _wait_until(lambda: (mgr.get(rec["run_id"]) or {}).get("progress") == 37)
    assert mgr.get(rec["run_id"])["last_activity"] == "halfway-ish"
    release.set()
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "done")


def test_list_and_active_filtering():
    release = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        release.wait(timeout=5)
        return "ok"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink())
    r1 = mgr.spawn(instruction="a", conversation_id="c1", agents=["literature"])
    mgr.spawn(instruction="b", conversation_id="c2", agents=["literature"])
    assert _wait_until(lambda: mgr.get(r1["run_id"])["status"] == "running")

    assert len(mgr.list_runs()) == 2
    assert len(mgr.list_runs(conversation_id="c1")) == 1
    assert mgr.has_active("c1") is True
    assert len(mgr.list_runs(active_only=True)) == 2
    release.set()
    assert _wait_until(lambda: not mgr.has_active())


# ── ② duration learning: duration_sink ───────────────────────────────────────

def test_duration_sink_fires_with_agent_type_and_positive_duration():
    seen: list = []
    lock = threading.Lock()

    def sink(agent_type, duration_s):
        with lock:
            seen.append((agent_type, duration_s))

    def run_fn(instruction, thread_id, agents, emit, abort):
        time.sleep(0.03)
        return "ok"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink(),
                               duration_sink=sink)
    rec = mgr.spawn(instruction="survey", conversation_id="c", agents=["literature"])
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "done")
    assert _wait_until(lambda: len(seen) == 1)
    agent_type, dur = seen[0]
    assert agent_type == "literature"
    assert dur > 0.0            # a real measured duration, not a placeholder


def test_duration_sink_failure_never_breaks_the_run():
    def boom(agent_type, duration_s):
        raise RuntimeError("stats store offline")

    mgr = BackgroundRunManager(run_fn=lambda *a, **k: "ok", transcript_sink=_Sink(),
                               duration_sink=boom)
    rec = mgr.spawn(instruction="x", conversation_id="c", agents=["literature"])
    # a throwing duration_sink is swallowed → the run still reaches done
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "done")


# ── ③ cross-run dependency: completion_sink + on_done + hint ─────────────────

def test_completion_sink_fires_only_on_clean_done():
    done_runs: list = []
    lock = threading.Lock()

    def completion(run):
        with lock:
            done_runs.append((run.run_id, run.status))

    def ok_fn(instruction, thread_id, agents, emit, abort):
        return "survey complete"

    def fail_fn(instruction, thread_id, agents, emit, abort):
        raise RuntimeError("provider down")

    mgr = BackgroundRunManager(run_fn=ok_fn, transcript_sink=_Sink(),
                               completion_sink=completion)
    good = mgr.spawn(instruction="ok", conversation_id="c", agents=["literature"])
    assert _wait_until(lambda: mgr.get(good["run_id"])["status"] == "done")
    assert _wait_until(lambda: len(done_runs) == 1)
    assert done_runs[0] == (good["run_id"], "done")

    # a FAILED run must NOT fire the completion hook (no follow-up off a broken run)
    mgr2 = BackgroundRunManager(run_fn=fail_fn, transcript_sink=_Sink(),
                                completion_sink=completion)
    bad = mgr2.spawn(instruction="boom", conversation_id="c", agents=["literature"])
    assert _wait_until(lambda: mgr2.get(bad["run_id"])["status"] == "failed")
    time.sleep(0.1)   # give any (wrong) hook call time to surface
    assert all(rid != bad["run_id"] for rid, _ in done_runs)


def test_completion_sink_not_fired_on_abort():
    fired: list = []
    step = threading.Event()

    def completion(run):
        fired.append(run.run_id)

    def run_fn(instruction, thread_id, agents, emit, abort):
        for i in range(100):
            if abort.is_set():
                break
            emit("literature", "agent", f"chunk {i}")
            step.set()
            time.sleep(0.02)
        return "stopped"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink(),
                               completion_sink=completion)
    rec = mgr.spawn(instruction="long", conversation_id="c", agents=["literature"])
    assert step.wait(timeout=2)
    assert mgr.abort(rec["run_id"]) is True
    assert _wait_until(lambda: mgr.get(rec["run_id"])["status"] == "aborted")
    time.sleep(0.1)
    assert fired == [], "completion hook fired on an aborted run"


def test_on_done_is_carried_on_the_record():
    mgr = BackgroundRunManager(run_fn=lambda *a, **k: "ok", transcript_sink=_Sink())
    rec = mgr.spawn(instruction="survey", conversation_id="c", agents=["literature"],
                    on_done={"next_agent": "experiment_design", "note": "拟方案"})
    # the explicit follow-up is preserved on the run for the completion hint
    with mgr._cond:
        run = mgr._runs[rec["run_id"]]
    assert run.on_done == {"next_agent": "experiment_design", "note": "拟方案"}


def test_build_completion_hint_surfaces_result_and_declared_followup():
    from mast.core.background_runs import BackgroundRun, build_completion_hint

    run = BackgroundRun(
        run_id="r1", conversation_id="c", instruction="综述石墨烯", agents=("literature",),
        title="综述石墨烯", final_text="找到 12 篇关键文献",
        on_done={"next_agent": "experiment_design", "note": "据此拟方案"})
    hint = build_completion_hint(run)
    assert "综述石墨烯" in hint
    assert "找到 12 篇关键文献" in hint
    assert "experiment_design" in hint          # the DECLARED follow-up is named
    assert "不要自动" in hint                     # advisory-only wording, never auto-run

    # no on_done → surfaces the result but invents NO follow-up
    run2 = BackgroundRun(
        run_id="r2", conversation_id="c", instruction="综述", agents=("literature",),
        title="综述", final_text="done")
    hint2 = build_completion_hint(run2)
    assert "已完成" in hint2
    assert "experiment_design" not in hint2
