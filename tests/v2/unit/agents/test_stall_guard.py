"""StallGuard — end a spin, and leave a record of what it was.

    「运行出错：任务步数达到上限——可能某个智能体在预条件或安全门上反复失败而空转。
     已停止本次运行。」

The operator wrote that diagnosis themselves, from the symptom, because the system
could not. And this middleware was ALREADY RUNNING when it happened. Three reasons
it did nothing, each pinned below:

  1. it was wired into **instrument_control only** — the other five agents had no
     stall detection at all;
  2. it only **nudged**, once, and then refused to stack a second — so a model that
     ignored the nudge spun on until the recursion cap killed the whole run with
     nothing to show for it. (Raising the cap 150→500 makes a runaway take LONGER.
     It is not detection.)
  3. it only matched **byte-identical** error text — ``timed out after 3.02s`` and
     ``timed out after 3.14s`` are the same spin and looked like two different
     failures, so the counter never reached the threshold. Any error carrying a
     number, a timestamp, or a path could not trip it. Which is most of them.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/agents/test_stall_guard.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from mast.agents._shared.stall_guard_mw import _MARKER, StallGuardMiddleware  # noqa: E402
from mast.core import diagnostics as diag  # noqa: E402

_AGENTS = ("literature", "experiment_design", "instrument_control",
           "data_processing", "paper_writing", "paper_review")


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    diag.clear()
    yield
    diag.clear()


class _Req:
    """Minimal stand-in for the LangChain model-call request (has .messages, no
    .override → the guard assigns .messages directly, like the real per-call path)."""

    def __init__(self, messages):
        self.messages = list(messages)


def _fail(name: str, err: str) -> ToolMessage:
    return ToolMessage(content=f"[{name}] failed: {err}", tool_call_id="tc",
                       name=name, status="error")


def _guard(agent: str = "instrument_control") -> StallGuardMiddleware:
    return StallGuardMiddleware(agent_name=agent)


def _run(g: StallGuardMiddleware, msgs) -> tuple[int, str]:
    """(directives injected, stop-text). A non-empty stop-text means the guard
    ended the turn instead of letting the model spin on."""
    req = _Req(msgs)
    out = g.wrap_model_call(req, lambda r: AIMessage(content="<model ran>"))
    text = str(getattr(out, "content", ""))
    stopped = "<model ran>" not in text
    n = sum(1 for m in req.messages if _MARKER in str(getattr(m, "content", "")))
    return n, (text if stopped else "")


# ════════════════════════════════════════════════════════════════════════
# The original contract — still holds
# ════════════════════════════════════════════════════════════════════════

def test_trips_on_repeated_identical_failure():
    msgs = [HumanMessage(content="进针")] + [
        _fail("AutoApproach",
              "precondition_failed: Cannot verify precondition 'bias_nonzero'")
        for _ in range(3)
    ]
    n, stop = _run(_guard(), msgs)
    assert n == 1 and not stop


def test_no_trip_below_threshold():
    msgs = [HumanMessage(content="x")] + [
        _fail("GetBias", "TimeoutError: timed out") for _ in range(2)]
    assert _run(_guard(), msgs) == (0, "")


def test_no_trip_on_varied_failures():
    """Three DIFFERENT tools each failing once is not a spin."""
    msgs = [
        HumanMessage(content="x"),
        _fail("GetBias", "timed out"),
        _fail("GetCurrent", "timed out"),
        _fail("GetZPosition", "timed out"),
    ]
    assert _run(_guard(), msgs) == (0, "")


def test_does_not_stack_a_second_directive_immediately():
    """Already nudged and awaiting the agent's next move → give it a turn."""
    msgs = [_fail("AutoApproach", "bias_nonzero") for _ in range(3)] + [
        HumanMessage(content=_MARKER + " 停止重试并升级")
    ]
    n, stop = _run(_guard(), msgs)
    assert n == 1 and not stop


# ════════════════════════════════════════════════════════════════════════
# (3) A spin is "the same failure again", not "the same BYTES again"
# ════════════════════════════════════════════════════════════════════════

class TestSignatureNormalisation:
    def test_a_timeout_that_reports_its_duration_still_trips(self):
        """The exact reason the guard was decorative: an error carrying a number
        never matched itself, so the counter never reached the threshold."""
        msgs = [HumanMessage(content="读电流")] + [
            _fail("GetCurrent", f"TimeoutError: timed out after {d}s")
            for d in ("3.02", "3.14", "2.98")
        ]
        n, _ = _run(_guard(), msgs)
        assert n == 1, ("three timeouts differing only in their duration read as "
                        "three DIFFERENT failures — this is the whole bug")

    def test_a_path_in_the_error_still_trips(self):
        msgs = [HumanMessage(content="x")] + [
            _fail("load_scan", f"file not found: D:\\Data\\scan_{i:03d}.sxm")
            for i in range(3)
        ]
        assert _run(_guard(), msgs)[0] == 1

    def test_a_safety_refusal_counts_as_a_spin_too(self):
        """A model retrying a BLOCKED action spins just as surely as one retrying a
        failing one. The old signature only looked for `failed:`."""
        msgs = [HumanMessage(content="x")] + [
            ToolMessage(content="[MotorMove] blocked: 开环粗逼近须人工执行",
                        tool_call_id="t", name="MotorMove")
            for _ in range(3)
        ]
        assert _run(_guard(), msgs)[0] == 1

    def test_genuinely_different_errors_still_do_not_trip(self):
        """Normalisation must not collapse every failure into the same failure."""
        msgs = [
            HumanMessage(content="x"),
            _fail("Scan", "tip crash detected"),
            _fail("Scan", "piezo out of range"),
            _fail("Scan", "scan buffer empty"),
        ]
        assert _run(_guard(), msgs) == (0, "")


# ════════════════════════════════════════════════════════════════════════
# (2) It must ESCALATE — a nudge the model ignores is not a guard
# ════════════════════════════════════════════════════════════════════════

class TestEscalation:
    def _spinning(self, nudges: int):
        msgs = [HumanMessage(content="进针")]
        for _ in range(nudges):
            msgs += [_fail("AutoApproach", "precondition_failed: bias_nonzero")] * 3
            msgs += [HumanMessage(content=_MARKER + " 停止重试并升级")]
        msgs += [_fail("AutoApproach", "precondition_failed: bias_nonzero")] * 3
        return msgs

    def test_second_trip_escalates_the_wording(self):
        n, stop = _run(_guard(), self._spinning(nudges=1))
        assert n == 2 and not stop, "a second, harder directive should follow"

    def test_after_two_ignored_nudges_the_turn_is_STOPPED(self):
        """The failure the operator actually hit: the model ignored the directive
        and burned the whole recursion budget, ending with 「任务步数达到上限」 and
        nothing to show. Stop the turn instead, with a readable conclusion."""
        _n, stop = _run(_guard(), self._spinning(nudges=2))
        assert stop, "the agent ignored two directives and was allowed to spin on"
        assert "空转保护" in stop
        assert "AutoApproach" in stop, "the conclusion does not name the blocker"
        assert "诊断" in stop, "the operator is not told where to look"

    def test_a_stopped_turn_never_calls_the_model(self):
        """If the model still gets called, the budget still burns — the stop is
        cosmetic."""
        called: list[int] = []

        def _handler(_r):
            called.append(1)
            return AIMessage(content="<model ran>")

        _guard().wrap_model_call(_Req(self._spinning(nudges=2)), _handler)
        assert not called, "the model was called anyway — the recursion budget " \
                           "still burns and we are back where we started"


# ════════════════════════════════════════════════════════════════════════
# (1) Every agent gets one — a spin is not an instrument problem
# ════════════════════════════════════════════════════════════════════════

def test_every_agent_has_a_stall_guard():
    """Wired into instrument_control ONLY, so a literature agent — or a
    data_processing agent stuck re-guessing a path it will never find  —
    could spin to the recursion cap with no detection at all."""
    missing = [
        a for a in _AGENTS
        if "StallGuardMiddleware("
        not in (Path(_MASTV2_ROOT) / "mast" / "agents" / a / "graph.py").read_text(
            encoding="utf-8")
    ]
    assert not missing, f"these agents have no stall detection: {missing}"


# ════════════════════════════════════════════════════════════════════════
# Every trip is written down — #31 was undiagnosable, not merely unhandled
# ════════════════════════════════════════════════════════════════════════

class TestItLeavesARecord:
    def test_a_nudge_is_recorded_with_what_it_was_spinning_on(self):
        msgs = [HumanMessage(content="x")] + [
            _fail("AutoApproach", "precondition_failed: bias_nonzero")] * 4
        _run(_guard("instrument_control"), msgs)

        rows = diag.recent(kinds=("stall",))
        assert rows, "the guard tripped and left no trace"
        r = rows[0]
        assert r["subject"] == "instrument_control:AutoApproach"
        assert r["count"] >= 3
        assert r["stopped"] is False
        assert "bias_nonzero" in r["signature"]

    def test_a_forced_stop_is_recorded_as_such(self):
        msgs = [HumanMessage(content="x")]
        for _ in range(2):
            msgs += [_fail("GetBias", "TimeoutError: timed out after 3.0s")] * 3
            msgs += [HumanMessage(content=_MARKER + " 停止重试")]
        msgs += [_fail("GetBias", "TimeoutError: timed out after 3.1s")] * 3
        _run(_guard("data_processing"), msgs)

        stops = [r for r in diag.recent(kinds=("stall",)) if r.get("stopped")]
        assert stops, "the guard force-stopped a turn and did not record it"
        assert stops[0]["subject"] == "data_processing:GetBias"


# ════════════════════════════════════════════════════════════════════════
# (NEW) Acknowledged / complied — stop firing once the agent has given up
#
# the agent said 「本轮不再重试 SetSetpoint」 and DID stop, but
# the guard kept re-firing on the stale safety-gate failures still in the window,
# forcing four re-explanations across 25 minutes. A guard that punishes compliance
# is worse than no guard.
# ════════════════════════════════════════════════════════════════════════

def _safety_block(name: str, err: str) -> ToolMessage:
    """A safety-gate value rejection, as it reaches the model: status='error',
    content is the gate's own line (no 'failed:'/'blocked' token)."""
    return ToolMessage(content=err, tool_call_id="tc", name=name, status="error")


_SETPOINT_ERR = ("[safety_gate] global_bounds_violation: Parameter 'setpoint_a' "
                 "= 1.5 above global safety maximum 1e-07")


def _run_capture(g: StallGuardMiddleware, msgs) -> tuple[list[str], str]:
    """(injected directive texts, stop-text)."""
    req = _Req(msgs)
    out = g.wrap_model_call(req, lambda r: AIMessage(content="<model ran>"))
    injected = [str(m.content) for m in req.messages
                if _MARKER in str(getattr(m, "content", ""))]
    text = str(getattr(out, "content", ""))
    return injected, (text if "<model ran>" not in text else "")


class TestAcknowledgedSpinGoesQuiet:
    def test_no_re_nudge_after_the_agent_stops_repeating(self):
        """Warned once, the agent complies (does other work, never re-issues the
        bad call). The guard must NOT nudge again on the stale failures."""
        msgs = [HumanMessage(content="设定点")]
        msgs += [_safety_block("SetSetpoint", _SETPOINT_ERR)] * 3
        msgs += [HumanMessage(content=_MARKER + " 停止重试并升级")]
        # complied: a different, SUCCESSFUL call — no new SetSetpoint failure
        msgs += [ToolMessage(content="{'current_a': 5.1e-11}",
                             tool_call_id="t2", name="GetCurrent")]
        n, stop = _run(_guard(), msgs)
        assert n == 1 and not stop, "the guard re-fired at an agent that complied"

    def test_forced_stop_does_not_repeat_once_the_agent_stops(self):
        """After a forced stop, if the agent stops repeating the failure, the next
        turn must not be stopped AGAIN on the same stale signature (the 25-minute
        bug: five stalls after the agent had already given up)."""
        spinning = [HumanMessage(content="设定点")]
        for _ in range(2):
            spinning += [_safety_block("SetSetpoint", _SETPOINT_ERR)] * 3
            spinning += [HumanMessage(content=_MARKER + " 停止重试")]
        spinning += [_safety_block("SetSetpoint", _SETPOINT_ERR)] * 3

        g = _guard()
        _n1, stop1 = _run(g, spinning)
        assert stop1, "the forced stop should fire the first time"

        # next turn: the stop message is now in history AND the agent complied
        after = spinning + [AIMessage(content=stop1),
                            ToolMessage(content="ok", tool_call_id="t9",
                                        name="GetZPosition")]
        _n2, stop2 = _run(g, after)
        assert not stop2, "the guard force-stopped the same signature again after " \
                          "the agent had already complied"

    def test_a_second_spin_on_a_DIFFERENT_tool_is_still_caught(self):
        """Compliance suppression must not blind the guard: if the agent stops the
        first failure but starts spinning a DIFFERENT one, that must still trip."""
        msgs = [HumanMessage(content="x")]
        msgs += [_safety_block("SetSetpoint", _SETPOINT_ERR)] * 3
        msgs += [HumanMessage(content=_MARKER + " 停止重试")]
        # abandoned SetSetpoint, now spinning GetBias instead
        msgs += [_fail("GetBias", "TimeoutError: timed out after 3.0s")] * 3
        n, _stop = _run(_guard(), msgs)
        assert n == 2, "a fresh spin on a different tool went undetected"


# ════════════════════════════════════════════════════════════════════════
# (NEW) A parameter typo is not a hardware fault — do not send the operator
# to the machine for it.
# ════════════════════════════════════════════════════════════════════════

class TestInputErrorVsHardware:
    def test_input_error_directive_does_not_summon_the_operator(self):
        msgs = [HumanMessage(content="设定点")] + [
            _safety_block("SetSetpoint", _SETPOINT_ERR)] * 3
        injected, _ = _run_capture(_guard("instrument_control"), msgs)
        assert injected, "no directive was injected"
        d = injected[-1]
        assert "参数" in d, "the directive does not name it as a parameter error"
        # It may name request_user_action, but ONLY to forbid it — the typo must
        # not be escalated to the operator.
        assert "不要" in d and "request_user_action" in d, (
            "a value typo must be told NOT to call request_user_action")
        assert "无需" in d and "机台" in d, (
            "the directive should say the operator is not needed at the machine")

    def test_hardware_directive_STILL_offers_the_operator_route(self):
        """The operator escalation must survive for genuine hardware/precondition
        spins — this is the path that was correct all along."""
        msgs = [HumanMessage(content="进针")] + [
            _fail("AutoApproach", "precondition_failed: bias_nonzero")] * 3
        injected, _ = _run_capture(_guard(), msgs)
        assert injected and "request_user_action" in injected[-1]

    def test_input_error_stop_message_says_no_operator_needed(self):
        spinning = [HumanMessage(content="设定点")]
        for _ in range(2):
            spinning += [_safety_block("SetSetpoint", _SETPOINT_ERR)] * 3
            spinning += [HumanMessage(content=_MARKER + " 停止重试")]
        spinning += [_safety_block("SetSetpoint", _SETPOINT_ERR)] * 3
        _n, stop = _run(_guard(), spinning)
        assert stop and "空转保护" in stop
        assert "无需用户" in stop, "the stop still sends the operator to a typo"
        assert "SetSetpoint" in stop

    def test_stall_category_is_recorded(self):
        """The ledger must carry which kind of stall this was, so a post-mortem can
        separate 'agent typo' from 'hardware needs a human'."""
        # input error
        diag.clear()
        _run(_guard("instrument_control"),
             [HumanMessage(content="x")] + [_safety_block("SetSetpoint", _SETPOINT_ERR)] * 3)
        rows = diag.recent(kinds=("stall",))
        assert rows and rows[0].get("category") == "input_error"
        # hardware
        diag.clear()
        _run(_guard("instrument_control"),
             [HumanMessage(content="x")] + [_fail("AutoApproach",
                                                  "precondition_failed: bias_nonzero")] * 3)
        rows = diag.recent(kinds=("stall",))
        assert rows and rows[0].get("category") == "hardware"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
