"""StallGuard must retain nudge counts across model turns and escalate repeated rejected inputs to a stop."""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
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
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402

from mast.agents._shared.stall_guard_mw import (  # noqa: E402
    _MAX_NUDGES,
    _THRESHOLD,
    StallGuardMiddleware,
)

#: The real rejection text from the run.
_ERR = ("SetSetpoint blocked: [safety_gate] global_bounds_violation: "
        "物理荒谬值: 'setpoint_a' = 1.5 A 在物理上不可能"
        "——请改用科学计数法，切勿重试相同数值。")

_OTHER = ("MoveToXY blocked: [safety_gate] global_bounds_violation: "
          "'center_y_m' = 1.7031e-06 超出全局安全上限 1.5e-06")


class _Req:
    """Rebuilt every turn WITHOUT prior nudges — the shape that broke it."""

    def __init__(self, msgs):
        self.messages = msgs

    def override(self, messages=None, **kw):
        self.messages = messages
        return self


def _spin(err=_ERR, n=None, tool="SetSetpoint"):
    n = _THRESHOLD + 1 if n is None else n
    return [ToolMessage(content=err, tool_call_id=f"t{i}", name=tool,
                        status="error") for i in range(n)]


# ════════════════════════════════════════════════════════════════════════════

def test_escalation_reaches_a_forced_stop():
    """The regression. Each turn arrives with no nudge in the transcript —
    exactly what LangGraph hands us — and the guard must still count."""
    g = StallGuardMiddleware("instrument_control")
    stops = 0
    for turn in range(1, _MAX_NUDGES + 3):
        _req, stop = g._decide(_Req(_spin()))
        if stop:
            stops = turn
            break
    assert stops, (
        f"{_MAX_NUDGES + 2} turns of an identical rejection and the guard never "
        "stopped the turn — this is the 11-retry / zero-action run"
    )
    assert stops == _MAX_NUDGES + 1, f"stopped on turn {stops}, expected "
    f"{_MAX_NUDGES + 1}"


def test_it_nudges_before_it_stops():
    """A stop on the FIRST spin would be its own failure mode — a retry after a
    correction is normal behaviour, not a spin."""
    g = StallGuardMiddleware("instrument_control")
    req, stop = g._decide(_Req(_spin()))
    assert not stop, "stopped on the first spin without ever nudging"
    assert any("stall-guard" in str(getattr(m, "content", "") or "")
               for m in req.messages), "no nudge was injected"


def test_the_stop_message_explains_itself():
    g = StallGuardMiddleware("instrument_control")
    stop = ""
    for _ in range(_MAX_NUDGES + 2):
        _r, stop = g._decide(_Req(_spin()))
        if stop:
            break
    assert "空转保护" in stop
    assert "SetSetpoint" in stop, "the stop does not name the blocking tool"


def test_a_different_signature_gets_its_own_budget():
    """Two distinct spins must not share a counter — otherwise one abandoned
    typo would instantly stop the turn on an unrelated later failure."""
    g = StallGuardMiddleware("instrument_control")
    for _ in range(_MAX_NUDGES):
        g._decide(_Req(_spin()))
    _req, stop = g._decide(_Req(_spin(err=_OTHER, tool="MoveToXY")))
    assert not stop, "an unrelated failure inherited another signature's count"


def test_the_ledger_resets_after_a_stop():
    """After the turn is stopped, a genuinely new spin later deserves the full
    nudge sequence again rather than an instant stop."""
    g = StallGuardMiddleware("instrument_control")
    for _ in range(_MAX_NUDGES + 2):
        _r, stop = g._decide(_Req(_spin()))
        if stop:
            break
    _req, stop2 = g._decide(_Req(_spin()))
    assert not stop2, "a later spin was stopped instantly instead of nudged"


def test_below_threshold_is_left_alone():
    """One or two failures are a normal retry, not a spin."""
    g = StallGuardMiddleware("instrument_control")
    req, stop = g._decide(_Req(_spin(n=_THRESHOLD - 1)))
    assert not stop
    assert not any("stall-guard" in str(getattr(m, "content", "") or "")
                   for m in req.messages)


def test_ledger_does_not_grow_without_bound():
    g = StallGuardMiddleware("instrument_control")
    for i in range(200):
        g._decide(_Req(_spin(err=f"Tool{i} blocked: distinct error {i}",
                             tool=f"Tool{i}")))
    assert len(g._nudged) <= 64


def test_guard_never_raises_on_junk():
    """A guard that throws takes the agent down with it."""
    g = StallGuardMiddleware("x")
    for msgs in ([], [None], ["not a message"], [ToolMessage(
            content="", tool_call_id="t", name="T")]):
        req, stop = g._decide(_Req(list(msgs)))
        assert stop == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
