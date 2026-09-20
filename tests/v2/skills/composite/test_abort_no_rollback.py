"""An abort is CONTROL FLOW, not a failure — it must never trigger a rollback.

The operator presses 中止. Somewhere deep in a composite, ``step()`` sees the
flag and raises ``AbortRequested``. That exception then travels up through
whatever happens to be on the stack — and every generic ``except Exception`` on
the way treats it as a crash:

  * ``GraphExecutor`` used to route it into ``_handle_failure``, which appends a
    ``failed_step`` and aborts the plan as a FAILURE;
  * ``skill_adapter._run``'s ``except Exception`` then called
    ``skill.rollback(ctx, params)`` and reported
    ``[X] rolled_back: AbortRequested: …``.

Both are wrong, and the second is dangerous: rollback UNDOES completed work —
it re-drives the very instrument the operator just told us to stop, minutes
after they stopped it. (Today no skill overrides ``rollback``, so the live blast
radius is the lie in the record; the guard is what keeps it that way when the
first one does.) It also poisons the training log: a run the operator cancelled
was recorded as a skill that crashed and got rolled back.

The rule, matching how ``GraphInterrupt`` (a HITL pause) is already handled:
an abort stops the work, reports honestly, and rolls back NOTHING.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.agents._shared.skill_adapter import wrap_skill  # noqa: E402
from mast.core.execution_context import ExecutionContext  # noqa: E402
from mast.core.types import (  # noqa: E402
    NanonisCallRecord,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill  # noqa: E402
from mast.skills.composite._base import AbortRequested  # noqa: E402
from mast.skills.composite.graph_executor import (  # noqa: E402
    CompositeStep,
    GraphExecutor,
)

ROLLBACKS: list[str] = []


@pytest.fixture(autouse=True)
def _clear():
    ROLLBACKS.clear()
    yield
    ROLLBACKS.clear()


class _Pool:
    def __init__(self):
        self.calls: list[tuple] = []

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", [0]))


def _meta(name: str) -> SkillMetadata:
    return SkillMetadata(
        name=name, version="1.0.0", category=SkillCategory.COMPOSITE,
        safety_level=SafetyLevel.AUTO, description="test double",
        estimated_duration_s=0.1, composition_level=1,
    )


class _AbortingSkill(BaseSkill):
    """Raises AbortRequested from execute() — what a composite does when its
    ``step()`` primitive sees the flag while the plan is running."""

    def metadata(self):
        return _meta("AbortingSkill")

    def execute(self, context, params: dict) -> SkillResult:
        raise AbortRequested("AbortingSkill")

    def rollback(self, context, params: dict) -> None:
        ROLLBACKS.append("AbortingSkill")


class _CrashingSkill(BaseSkill):
    """A genuine bug — this one SHOULD roll back."""

    def metadata(self):
        return _meta("CrashingSkill")

    def execute(self, context, params: dict) -> SkillResult:
        raise RuntimeError("the piezo driver exploded")

    def rollback(self, context, params: dict) -> None:
        ROLLBACKS.append("CrashingSkill")


def _tool(skill_cls, records: list):
    pool = _Pool()

    def _provider():
        return ExecutionContext(pool=pool, state=None, registry=None,
                                abort_event=threading.Event())

    return wrap_skill(skill_cls, _provider, recorder=records.append), pool


# ════════════════════════════════════════════════════════════════════════
# skill_adapter — the outer guard
# ════════════════════════════════════════════════════════════════════════

class TestAbortDoesNotRollBack:
    def test_abort_never_calls_rollback(self):
        records: list = []
        tool, _pool = _tool(_AbortingSkill, records)
        out = tool.func(tool_call_id="t1", state={})

        assert ROLLBACKS == [], (
            "an abort fired the skill's ROLLBACK — undoing work the operator's "
            "stop never touched, and re-driving the instrument they just stopped")
        summary = out.update["messages"][0].content
        assert "aborted" in summary
        assert "rolled_back" not in summary, (
            f"the operator is told their abort was a crash + rollback: {summary}")
        assert "STOP now" in summary          # the model must not retry

    def test_the_record_says_aborted_not_rolled_back(self):
        """The trajectory log feeds #8 agent training. Recording an operator's
        abort as a rolled-back crash teaches exactly the wrong lesson."""
        records: list = []
        tool, _ = _tool(_AbortingSkill, records)
        tool.func(tool_call_id="t1", state={})

        assert records, "the skill emitted no record at all"
        rec = records[-1]
        assert rec.get("success") is False
        assert rec.get("rolled_back") is not True, (
            "the record claims a rollback happened; nothing was rolled back")
        assert "abort" in (rec.get("error") or "").lower()

    def test_the_abort_lands_in_the_error_trail(self):
        """Silently dropping it would leave the operator with no trace of WHY the
        run stopped."""
        records: list = []
        tool, _ = _tool(_AbortingSkill, records)
        out = tool.func(tool_call_id="t1", state={})
        assert out.update.get("error_log"), "the abort left no trace in error_log"

    def test_a_real_crash_still_rolls_back(self):
        """The guard must be surgical: a genuine exception is still a failure and
        must still roll back, or the fix would have disabled rollback entirely."""
        records: list = []
        tool, _ = _tool(_CrashingSkill, records)
        out = tool.func(tool_call_id="t1", state={})

        assert ROLLBACKS == ["CrashingSkill"], "a real crash must still roll back"
        assert "rolled_back" in out.update["messages"][0].content
        assert records[-1].get("rolled_back") is True


# ════════════════════════════════════════════════════════════════════════
# GraphExecutor — an abort is not a failed step
# ════════════════════════════════════════════════════════════════════════

class _Ctx:
    """A context whose abort flag can be tripped after N sub-skill calls."""

    def __init__(self, abort_after: int | None = None, raise_abort_at: int | None = None):
        self.ran: list[str] = []
        self._abort = threading.Event()
        self._abort_after = abort_after
        self._raise_at = raise_abort_at

    def check_abort(self) -> bool:
        return self._abort.is_set()

    def run(self, skill_name, params, version=None):
        self.ran.append(skill_name)
        n = len(self.ran)
        if self._raise_at is not None and n == self._raise_at:
            raise AbortRequested(skill_name)
        if self._abort_after is not None and n >= self._abort_after:
            self._abort.set()
        return SkillResult(skill_name=skill_name, success=True)


def _plan(n: int = 3, *, optional: bool = False) -> list[CompositeStep]:
    return [CompositeStep(step_id=f"s{i}", skill_name=f"Skill{i}", params={},
                          optional=optional)
            for i in range(n)]


def _exec(ctx, tmp_path) -> GraphExecutor:
    return GraphExecutor(composite_name="SpyComposite", context=ctx)


class TestGraphExecutorTreatsAbortAsControlFlow:
    def test_external_abort_flag_stops_the_plan_without_failing_a_step(self, tmp_path):
        ctx = _Ctx(abort_after=1)          # flag trips after the first sub-skill
        ex = _exec(ctx, tmp_path)
        ok = ex.run_plan(_plan(3))

        assert ok is False
        assert ex.progress.aborted is True
        assert ex.progress.failed_steps == [], (
            "the abort was recorded as a FAILED step — it is not a failure, and "
            "one level up a failed step is what triggers the rollback")
        assert ctx.ran == ["Skill0"], "the plan kept driving hardware after abort"

    def test_abort_raised_from_inside_a_sub_skill_is_not_a_failure(self, tmp_path):
        ctx = _Ctx(raise_abort_at=2)       # the 2nd sub-skill raises AbortRequested
        ex = _exec(ctx, tmp_path)
        ok = ex.run_plan(_plan(3))

        assert ok is False
        assert ex.progress.aborted is True
        assert ex.progress.failed_steps == []
        assert "abort" in (ex.progress.aborted_reason or "").lower()
        assert ctx.ran == ["Skill0", "Skill1"], "step 3 ran after the abort"

    def test_abort_raised_from_a_dynamic_plan_generator(self, tmp_path):
        """A dynamic plan computes its next step from the last result — so the
        abort can surface from the GENERATOR, outside any step's try block. It
        used to escape run_plan entirely and land in skill_adapter's generic
        handler, which rolled the composite back."""
        ctx = _Ctx()

        def _gen():
            yield CompositeStep(step_id="s0", skill_name="Skill0", params={})
            raise AbortRequested("dynamic plan")

        ex = _exec(ctx, tmp_path)
        ok = ex.run_plan(_gen())

        assert ok is False
        assert ex.progress.aborted is True
        assert ex.progress.failed_steps == []
        assert ctx.ran == ["Skill0"]

    def test_an_abort_in_an_OPTIONAL_step_still_stops_the_plan(self, tmp_path):
        """The sharpest form of "an abort is not a failure".

        ``_handle_failure`` deliberately CONTINUES past a failed *optional* step.
        So if an abort were routed through the failure path — which is exactly
        what happened before ``_is_abort_requested`` — the plan would march on to
        the next step and keep issuing instrument commands AFTER the operator
        stopped it. The abort must short-circuit before the optional-step logic
        ever sees it."""
        ctx = _Ctx(raise_abort_at=2)
        ex = _exec(ctx, tmp_path)
        ok = ex.run_plan(_plan(3, optional=True))

        assert ok is False
        assert ex.progress.aborted is True
        assert ctx.ran == ["Skill0", "Skill1"], (
            "the plan CONTINUED past the operator's abort because the step was "
            f"optional — it kept driving the instrument: ran={ctx.ran}")
        assert ex.progress.failed_steps == [], "the abort was logged as a failure"

    def test_a_real_failure_of_an_optional_step_is_recorded_and_continues(self, tmp_path):
        """The contrast that proves the guard is surgical rather than a blanket
        "never fail": a genuine failure of an optional step is still recorded and
        the plan still carries on."""
        class _FailingCtx(_Ctx):
            def run(self, skill_name, params, version=None):
                self.ran.append(skill_name)
                if skill_name == "Skill1":
                    return SkillResult(skill_name=skill_name, success=False,
                                       error="setpoint out of range")
                return SkillResult(skill_name=skill_name, success=True)

        ctx = _FailingCtx()
        ex = _exec(ctx, tmp_path)
        ok = ex.run_plan(_plan(3, optional=True))

        assert ok is True                                   # optional → carry on
        assert ex.progress.aborted is False
        assert ex.progress.failed_steps == ["s1"]
        assert ctx.ran == ["Skill0", "Skill1", "Skill2"]

    def test_a_fatal_step_failure_aborts_with_the_error_as_the_reason(self, tmp_path):
        """A non-optional failure stops the plan too — but ``aborted_reason`` must
        carry the ERROR, not "aborted by operator". That field is the only thing
        distinguishing "the operator stopped us" from "the instrument refused",
        and the operator-facing UI reads it."""
        class _FailingCtx(_Ctx):
            def run(self, skill_name, params, version=None):
                self.ran.append(skill_name)
                if skill_name == "Skill1":
                    return SkillResult(skill_name=skill_name, success=False,
                                       error="setpoint out of range")
                return SkillResult(skill_name=skill_name, success=True)

        ctx = _FailingCtx()
        ex = _exec(ctx, tmp_path)
        ok = ex.run_plan(_plan(3))

        assert ok is False
        assert ex.progress.aborted is True
        assert "setpoint out of range" in (ex.progress.aborted_reason or "")
        assert "operator" not in (ex.progress.aborted_reason or "").lower(), (
            "an instrument failure is being reported to the operator as their "
            "own abort")
