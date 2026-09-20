"""Abort must reach the INSTRUMENT, not just the graph loop (systemic gap).

Before this, abort was honoured only BETWEEN LangGraph super-steps: nothing in
agents/ or the skill layer ever checked it. So pressing 中止 during a scan or an
approach did nothing until the whole step returned — the agent's ReAct loop kept
issuing tool calls and each one drove the hardware. The operator's report was
blunt: "中止后群聊进入后端运行？控制不了了？"

The gate now sits at the three doors every hardware action must pass:
  1. skill_adapter.wrap_skill  — the ONE door an LLM tool call enters a skill by
  2. ExecutionContext.run      — the door a composite's sub-steps enter by
  3. ExecutionContext.safe_call— the door EVERY Nanonis command enters by

and it is deliberately asymmetric, because a stop is itself a hardware sequence:

    read anything · stop anything · start nothing

Getting that policy wrong in the permissive direction is a hardware-safety bug
(a post-abort Scan_Action(0,…) would START a scan), so it is pinned hard here.
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

import pytest  # noqa: E402

from mast.core.execution_context import (  # noqa: E402
    ExecutionContext,
    _is_abort_safe,
)
from mast.core.types import NanonisCallRecord  # noqa: E402


class _Pool:
    """Records every call that actually REACHED the instrument."""

    def __init__(self):
        self.calls: list[tuple] = []

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        return NanonisCallRecord(method=method, args=args,
                                 return_value=("", b"", [0]))


# ════════════════════════════════════════════════════════════════════════
# The post-abort hardware policy — the part that can hurt the instrument
# ════════════════════════════════════════════════════════════════════════

class TestAbortSafePolicy:
    @pytest.mark.parametrize("method,args", [
        ("Scan_StatusGet", ()),
        ("Current_Get", ()),
        ("ZCtrl_SetpntGet", ()),
        ("AutoApproach_OnOffGet", ()),
    ])
    def test_reads_always_allowed(self, method, args):
        """A stop sequence has to be able to READ status."""
        assert _is_abort_safe(method, args) is True

    @pytest.mark.parametrize("method,args", [
        ("ZCtrl_Withdraw", (1, -1)),          # retract — always the safe way
        ("Motor_StopMove", ()),
        ("Scan_Action", (1, 0)),              # action=1 ⇒ STOP the scan
        ("Scan_Action", (2, 0)),              # action=2 ⇒ PAUSE
        ("AutoApproach_OnOffSet", (0,)),      # 0 ⇒ OFF
        ("BiasSpectr_Stop", ()),
    ])
    def test_stops_always_allowed(self, method, args):
        """Blocking the stop verbs would strand the tip mid-approach."""
        assert _is_abort_safe(method, args) is True

    @pytest.mark.parametrize("method,args", [
        ("Bias_Set", (5.0,)),
        ("ZCtrl_SetpntSet", (1e-9,)),
        ("FolMe_XYPosSet", (1e-8, 1e-8, 1)),
        ("Motor_StartMove", (0, 10, 1)),
        ("BiasSpectr_Start", (1, "x")),
    ])
    def test_writes_refused(self, method, args):
        assert _is_abort_safe(method, args) is False

    def test_scan_action_start_is_refused(self):
        """THE trap: Scan_Action(action, direction) — action 0 is START. A rule
        that read the *direction* argument instead would let a post-abort call
        start a brand-new scan."""
        assert _is_abort_safe("Scan_Action", (0, 0)) is False
        assert _is_abort_safe("Scan_Action", (0, 1)) is False

    def test_auto_approach_restart_is_refused(self):
        """Same verb, opposite meaning: OnOffSet(1) would RESTART the approach
        we are aborting."""
        assert _is_abort_safe("AutoApproach_OnOffSet", (1,)) is False

    def test_unknown_verb_refused(self):
        """Fail closed: a verb nobody vetted is not assumed safe."""
        assert _is_abort_safe("Some_NewWrite", (1,)) is False

    def test_malformed_args_refused(self):
        """Can't prove it's a stop → refuse."""
        assert _is_abort_safe("Scan_Action", ()) is False
        assert _is_abort_safe("AutoApproach_OnOffSet", ("nonsense",)) is False


# ════════════════════════════════════════════════════════════════════════
# ExecutionContext.safe_call — the deepest door
# ════════════════════════════════════════════════════════════════════════

class TestAbortIsAUnion:
    """P0 (2026-07-11): there is MORE THAN ONE legitimate stop in this system,
    and they are different threading.Events — the orchestrator/E-STOP event, and
    a private chat's / voice session's own stop.

    The private-chat context used to be built as
    ``abort_event=session_event or _orch_abort``, so the session event WON and a
    composite running under the main chat polled only that one. E_STOP sets
    ``_orch_abort`` — which meant **pressing the emergency stop could not stop a
    composite running in the private chat**: check_abort() returned False and
    every gate below it let the hardware keep going.

    A context must therefore abort when ANY of its stop sources fires.
    """

    def test_estop_reaches_a_private_chat_context(self):
        session_stop = threading.Event()      # the chat's own Stop button
        orch_stop = threading.Event()         # E-STOP / run-task abort
        pool = _Pool()
        ctx = ExecutionContext(pool=pool, state=None, registry=None,
                               abort_event=[session_stop, orch_stop])
        assert ctx.check_abort() is False

        orch_stop.set()                       # ← the emergency stop fires
        assert ctx.check_abort() is True, "E-STOP must reach the chat's context"
        # and the hardware gate must actually bite
        rec = ctx.safe_call("Bias_Set", 5.0)
        assert rec.error and "aborted" in rec.error
        assert pool.calls == []

    def test_chat_stop_still_works(self):
        session_stop = threading.Event()
        orch_stop = threading.Event()
        ctx = ExecutionContext(pool=_Pool(), state=None, registry=None,
                               abort_event=[session_stop, orch_stop])
        session_stop.set()
        assert ctx.check_abort() is True

    def test_single_event_still_accepted(self):
        """Back-compat: the orchestrator path passes one Event."""
        ev = threading.Event()
        ctx = ExecutionContext(pool=_Pool(), state=None, registry=None,
                               abort_event=ev)
        assert ctx.check_abort() is False
        ev.set()
        assert ctx.check_abort() is True

    def test_no_event_is_never_aborted(self):
        ctx = ExecutionContext(pool=_Pool(), state=None, registry=None)
        assert ctx.check_abort() is False


class TestSafeCallGate:
    def _ctx(self, aborted: bool):
        ev = threading.Event()
        if aborted:
            ev.set()
        pool = _Pool()
        return ExecutionContext(pool=pool, state=None, registry=None,
                                abort_event=ev), pool

    def test_write_blocked_and_never_reaches_hardware(self):
        ctx, pool = self._ctx(aborted=True)
        rec = ctx.safe_call("Bias_Set", 5.0)
        assert rec.error and "aborted by operator" in rec.error
        assert pool.calls == [], "the instrument must NOT have been touched"

    def test_stop_still_reaches_hardware(self):
        ctx, pool = self._ctx(aborted=True)
        rec = ctx.safe_call("Scan_Action", 1, 0)
        assert not rec.error
        assert pool.calls == [("Scan_Action", (1, 0))]

    def test_read_still_reaches_hardware(self):
        ctx, pool = self._ctx(aborted=True)
        ctx.safe_call("Current_Get")
        assert pool.calls == [("Current_Get", ())]

    def test_explicit_escape_hatch(self):
        """Cleanup code that runs BECAUSE of the abort can opt out."""
        ctx, pool = self._ctx(aborted=True)
        rec = ctx.safe_call("Bias_Set", 0.0, allow_on_abort=True)
        assert not rec.error
        assert pool.calls == [("Bias_Set", (0.0,))]

    def test_no_abort_no_gate(self):
        ctx, pool = self._ctx(aborted=False)
        ctx.safe_call("Bias_Set", 5.0)
        assert pool.calls == [("Bias_Set", (5.0,))]


# ════════════════════════════════════════════════════════════════════════
# ExecutionContext.run — a mid-flight composite must not fire its next step
# ════════════════════════════════════════════════════════════════════════

class _Registry:
    def __init__(self, ran: list):
        self._ran = ran

    def get(self, name, version=None):
        ran = self._ran

        class _Skill:
            def metadata(self):  # pragma: no cover — not reached when gated
                raise AssertionError("must not be reached under abort")

            def execute(self, ctx, params):  # pragma: no cover
                ran.append(name)
                raise AssertionError("must not be reached under abort")

        return _Skill

    def _get_metadata(self, cls):  # pragma: no cover
        raise AssertionError("must not be reached under abort")


class TestSubSkillGate:
    def test_sub_skill_refused_under_abort(self):
        ev = threading.Event()
        ev.set()
        ran: list = []
        ctx = ExecutionContext(pool=_Pool(), state=None, registry=_Registry(ran),
                               abort_event=ev)
        res = ctx.run("SetBias", {"bias_v": 1.0})
        assert res.success is False
        assert "aborted by operator" in (res.error or "")
        assert "do not retry" in (res.error or "").lower()
        assert ran == [], "the sub-skill must never have executed"


# ════════════════════════════════════════════════════════════════════════
# skill_adapter — the door the LLM's tool calls come through
# ════════════════════════════════════════════════════════════════════════

class TestAgentToolGate:
    def test_tool_call_refused_under_abort(self):
        from mast.agents._shared.skill_adapter import wrap_skill
        from mast.skills.builtins.bias import GetBias

        ev = threading.Event()
        ev.set()
        pool = _Pool()

        def _provider():
            return ExecutionContext(pool=pool, state=None, registry=None,
                                    abort_event=ev)

        tool = wrap_skill(GetBias, _provider)
        out = tool.func(tool_call_id="t1", state={})
        content = out.update["messages"][0].content
        assert "aborted" in content
        assert "STOP now" in content          # the model is told not to retry
        assert pool.calls == [], "no hardware call may be issued after abort"

    def test_tool_call_runs_normally_when_not_aborted(self):
        from mast.agents._shared.skill_adapter import wrap_skill
        from mast.skills.builtins.bias import GetBias

        pool = _Pool()

        def _provider():
            return ExecutionContext(pool=pool, state=None, registry=None,
                                    abort_event=threading.Event())

        tool = wrap_skill(GetBias, _provider)
        tool.func(tool_call_id="t1", state={})
        assert pool.calls, "the skill must run normally with no abort latched"
