"""The refusal ledger — and proof that the refusals actually reach it.

A diagnostics module that nobody writes to is worse than none: it looks like
coverage. So the point of this file is NOT to test the ring buffer (that part is
20 lines and obvious). It is to prove that the four places which REFUSE things
really do leave a line behind — because the whole reason this exists is that they
didn't:

    进针失败  — refused by WHICH layer? on what state? Nobody knew,
                              because an unmet precondition produced a ToolMessage
                              to the model and then vanished.
    STS 在第五点停下 — executed-and-failed, resume-skipped, and aborted all
                              look identical from outside. Nothing wrote down which.
    智能体在预条件或安全门上反复失败而空转
                            — the operator diagnosed this from the symptom, because
                              the system could not.

The ledger must also never be able to hurt the thing it observes: a diagnostics
failure is a lost line, never a failed skill.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.core import diagnostics as diag  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    diag.clear()
    diag.set_run_id("")
    yield
    diag.clear()


# ════════════════════════════════════════════════════════════════════════
# THE POINT: the refusal sites really write
# ════════════════════════════════════════════════════════════════════════

class TestRefusalsReachTheLedger:
    def test_an_unmet_precondition_says_which_one_and_on_what_state(self):
        """#28. 「进针功能调用失败」 with no cause. The answer needs BOTH halves:
        WHICH condition, and the live state it was judged against. Neither was
        recorded anywhere — the unmet list went to the model as a ToolMessage and
        evaporated."""
        from mast.core.types import HardwareState
        from mast.skills.builtins.approach import AutoApproach

        # AutoApproach declares `bias_nonzero` — and 「进针功能调用失败」 with the
        # bias at 0 is EXACTLY the shape the operator hit. Now it leaves a record
        # that names the condition and the state it was judged against.
        skill = AutoApproach()
        state = HardwareState(bias_v=0.0, z_controller_on=False)
        unmet = skill.check_preconditions(state)
        assert unmet, "test is vacuous — bias_v=0 must violate bias_nonzero"

        rows = diag.recent(kinds=("precondition_block",))
        assert rows, "an unmet precondition left NO trace — #28 all over again"
        r = rows[0]
        assert r["subject"] == "AutoApproach"
        assert r["reason"], "the refusal has no reason"
        assert r["unmet"] == unmet
        assert r["declared"] == ["bias_nonzero"]
        assert "bias_v" in r["state"], (
            "the refusal does not record the STATE it was judged against — half "
            "the answer is missing, and it is the half that says WHY")
        assert r["state"]["bias_v"] == 0.0

    def test_an_abort_refusal_names_the_verb_and_its_args(self):
        """A post-abort instrument write is refused. Which one, with what args?"""
        from mast.core.execution_context import ExecutionContext
        from mast.core.types import NanonisCallRecord

        class _Pool:
            def safe_call(self, m, *a, role="main"):
                return NanonisCallRecord(method=m, args=a)

        ev = threading.Event()
        ev.set()
        ctx = ExecutionContext(pool=_Pool(), state=None, registry=None,
                               abort_event=ev, run_id="task-7")
        ctx.safe_call("Bias_Set", 5.0)

        rows = diag.recent(kinds=("abort_block",))
        assert rows, "the abort gate refused a write and left no trace"
        assert rows[0]["subject"] == "Bias_Set"
        assert rows[0]["args"] == [5.0]
        assert rows[0]["run_id"] == "task-7", "the refusal is not tied to its run"

    def test_a_resume_skipped_step_is_recorded_as_a_DECISION(self):
        """#90. A skipped step and a step that ran-and-succeeded are
        indistinguishable from outside. If a stale sidecar makes the executor skip
        the leading points, the composite silently starts in the middle and the
        operator sees it "stop early" — with nothing to look at."""
        from mast.core.types import SkillResult
        from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor

        class _Ctx:
            def __init__(self):
                self.ran = []

            def run(self, name, params, version=None):
                self.ran.append(name)
                return SkillResult(skill_name=name, success=True)

        ctx = _Ctx()
        ex = GraphExecutor(composite_name="GridSTS", context=ctx)
        ex.progress.completed_steps.extend(["sts_0_0", "sts_0_1"])   # a stale sidecar
        plan = [CompositeStep(step_id=f"sts_0_{i}", skill_name="STS", params={})
                for i in range(4)]
        ex.run_plan(plan)

        assert ctx.ran == ["STS", "STS"], "the skip did not happen — test is vacuous"
        skips = diag.recent(kinds=("step_skip",))
        assert len(skips) == 2, (
            f"the executor skipped 2 steps and recorded {len(skips)} — an operator "
            f"seeing it 'stop early' still has nothing to read")
        assert {r["subject"] for r in skips} == {"GridSTS.sts_0_0", "GridSTS.sts_0_1"}

    def test_an_abort_records_WHICH_step_it_stopped_on(self):
        """"停在第五点" needs a step id, not just "the composite stopped"."""
        from mast.core.types import SkillResult
        from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor

        class _Ctx:
            def run(self, name, params, version=None):
                return SkillResult(skill_name=name, success=False, error="tip crashed")

        ex = GraphExecutor(composite_name="GridSTS", context=_Ctx())
        ex.run_plan([CompositeStep(step_id="sts_0_4", skill_name="STS", params={})])

        aborts = diag.recent(kinds=("step_abort",))
        assert aborts, "the composite stopped and did not say where"
        assert aborts[0]["subject"] == "GridSTS.sts_0_4"
        assert "tip crashed" in aborts[0]["reason"]

        fails = diag.recent(kinds=("step_fail",))
        assert fails and fails[0]["subject"] == "GridSTS.sts_0_4"


# ════════════════════════════════════════════════════════════════════════
# It must never be able to hurt what it observes
# ════════════════════════════════════════════════════════════════════════

class TestItCannotBreakAnything:
    def test_an_unserialisable_value_does_not_raise(self):
        """A tensor / socket / Nanonis client must never reach the ledger — and
        must never crash the skill that tried to log one either."""
        diag.record("note", "s", "r", arr=np.zeros((4, 4)), sock=object())
        r = diag.recent()[0]
        assert isinstance(r["arr"], (str, list))
        assert isinstance(r["sock"], str)

    def test_a_dead_disk_disables_the_log_and_nothing_else(self, monkeypatch):
        monkeypatch.setattr(diag, "_dir",
                            lambda: (_ for _ in ()).throw(OSError("read-only")))
        diag.record("note", "s", "still recorded in memory")
        assert diag.recent()[0]["reason"] == "still recorded in memory"

    def test_a_huge_error_blob_is_capped(self):
        diag.record("note", "s", "x" * 50_000)
        assert len(diag.recent()[0]["reason"]) < 1000, (
            "one pathological traceback would become the whole log")

    def test_the_buffer_is_bounded(self):
        for i in range(diag._MAX_ENTRIES + 200):
            diag.record("note", f"s{i}", "r")
        assert len(diag.recent(10_000)) == diag._MAX_ENTRIES

    def test_it_is_thread_safe(self):
        def _spam():
            for i in range(200):
                diag.record("note", f"t{i}", "r")

        ts = [threading.Thread(target=_spam) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        seqs = [r["seq"] for r in diag.recent(10_000)]
        assert len(seqs) == len(set(seqs)), "concurrent writes collided on seq"


# ════════════════════════════════════════════════════════════════════════
# Reading it
# ════════════════════════════════════════════════════════════════════════

class TestReading:
    def test_summary_shows_the_shape_of_a_spin(self):
        """#31 without anyone reading a log: ONE subject dominating a count IS the
        spin, and the summary says so at a glance."""
        for _ in range(7):
            diag.record("precondition_block", "AutoApproach", "bias_nonzero 未满足")
        diag.record("safety_block", "SetBias", "超出全局上限")

        s = diag.summary()
        assert s["total"] == 8
        assert s["by_kind"]["precondition_block"] == 7
        assert s["top_refusals"][0]["what"] == "precondition_block:AutoApproach"
        assert s["top_refusals"][0]["count"] == 7

    def test_filters_isolate_one_run(self):
        diag.set_run_id("run-A")
        diag.record("safety_block", "SetBias", "a")
        diag.set_run_id("run-B")
        diag.record("safety_block", "SetBias", "b")
        assert len(diag.recent(run_id="run-A")) == 1
        assert diag.recent(run_id="run-A")[0]["reason"] == "a"

    def test_it_survives_the_run_that_produced_it(self, tmp_path):
        """The ledger appends to a JSONL, so a post-mortem is possible after the
        process that refused things is gone. An in-memory-only trace would be
        useless for exactly the case it exists for."""
        diag.record("safety_block", "TipShape", "safe 模式禁止修针")
        log = tmp_path / "artifacts" / "diagnostics" / "refusals.jsonl"
        assert log.is_file(), "nothing on disk — the record dies with the process"
        assert "TipShape" in log.read_text(encoding="utf-8")
