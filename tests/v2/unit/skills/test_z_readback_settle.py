"""Z-controller readback must survive Nanonis's switch-off delay.

both directions of the SAME defect:

  #7  `[ApproachTip] failed: … 实时控制器回报 Z 反馈仍然闭合` — we asked for OFF
      and read back ON.
  #8  `[ZControllerOnOff] failed: 要求 ON，实时控制器回报 OFF` — the mirror image.

Root cause: the readback happened with ZERO wait after the write. Nanonis
Z-Controllers have a per-controller SWITCH-OFF DELAY (Mimea manual,
Z-Controller Configuration §6): on a switch-off request the controller
"continues to run for the specified time" while it averages Z. Recommended
values are 10–100 ms; our write→read gap is ~0.2 ms on loopback and 1–3 ms over
LAN. Reading immediately therefore returns the PRE-WRITE state essentially every
time — a deterministic failure, not a flaky race.

The delay is a per-CONTROLLER property, which is why this surfaced after the rig
moved its control signal to log Current: switching the active controller
switches its switch-off delay with it.

v1 diagnosed this exact race in 2026-03 and fixed it with a 100 ms retry, but
that fix lives in core/executor.py and only guards the manual/GUI path — agents
go through wrap_skill → skill.execute and bypass it.

These tests pin BOTH the fix and the safety properties it must not weaken.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return p / "MASTv2"
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != str(_ROOT):
    while str(_ROOT) in sys.path:
        sys.path.remove(str(_ROOT))
    sys.path.insert(0, str(_ROOT))
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.skills.verify import verify_z_controller  # noqa: E402


class _Rec:
    def __init__(self, return_value=None, error=None):
        self.return_value = return_value
        self.error = error


class ScriptedCtx:
    """A Nanonis whose OnOffGet returns the PRE-WRITE state for the first N reads.

    That is exactly the shape of the switch-off-delay window: the RT controller
    is still running and honestly reports so, until the delay elapses.
    """

    def __init__(self, sequence, *, delay=0.05, status=4, abort=False):
        self.sequence = list(sequence)
        self._i = 0
        self.delay = delay
        self.status = status
        self.reads = 0
        self.verbs: list[str] = []
        self._abort = abort

    def check_abort(self):
        return self._abort

    def safe_call(self, verb, *args):
        self.verbs.append(verb)
        if verb == "ZCtrl_SwitchOffDelayGet":
            if self.delay is None:
                return _Rec(None, "not supported")
            return _Rec(("", b"", [self.delay]))
        if verb == "ZCtrl_StatusGet":
            return _Rec(("", b"", [self.status]))
        if verb == "ZCtrl_OnOffGet":
            self.reads += 1
            v = self.sequence[min(self._i, len(self.sequence) - 1)]
            self._i += 1
            if v is None:
                return _Rec(None, "TCP timeout")
            return _Rec(("", b"", [v]))
        return _Rec(("", b"", [0]))


# ── the two field failures ───────────────────────────────────────────────────

def test_off_is_confirmed_after_the_switch_off_delay():
    """#7: asked OFF, RT honestly reports ON during the delay, then OFF."""
    ctx = ScriptedCtx([1, 1, 0])
    v = verify_z_controller(ctx, expect=False)
    assert v["on"] is False
    assert v["matches"] is True
    assert ctx.reads >= 2, "must have polled, not concluded from the first read"
    assert v["waited_s"] > 0


def test_on_is_confirmed_after_the_delay():
    """#8: the mirror image — asked ON, first read still says OFF."""
    ctx = ScriptedCtx([0, 1])
    v = verify_z_controller(ctx, expect=True)
    assert v["on"] is True
    assert v["matches"] is True
    assert ctx.reads >= 2


# ── the safety properties settling must NOT weaken ───────────────────────────

def test_a_loop_that_never_opens_still_fails_closed():
    """THE one that must never regress: waiting must not become assuming."""
    ctx = ScriptedCtx([1] * 40)
    v = verify_z_controller(ctx, expect=False)
    assert v["on"] is True
    assert v["matches"] is False, "never-settling loop must NOT be reported off"


def test_unreadable_stays_None_never_False():
    """`on=None` is not `on=False` — a dead measurement chain must never be the
    trigger for an action that is only safe with the loop open."""
    ctx = ScriptedCtx([None] * 40)
    v = verify_z_controller(ctx, expect=False)
    assert v["on"] is None
    assert v["verified"] is False
    assert v["matches"] is None


def test_abort_leaves_the_loop_but_not_the_conclusion():
    """Being cancelled must never be reported as 'confirmed off'."""
    ctx = ScriptedCtx([1] * 40, abort=True)
    v = verify_z_controller(ctx, expect=False)
    assert v["matches"] is False
    assert v["on"] is True


# ── budget comes from the rig, not a constant ────────────────────────────────

def test_budget_is_read_from_the_active_controller():
    ctx = ScriptedCtx([1, 1, 0], delay=0.05)
    v = verify_z_controller(ctx, expect=False)
    assert v["switch_off_delay_s"] == pytest.approx(0.05)
    assert "ZCtrl_SwitchOffDelayGet" in ctx.verbs


def test_unreadable_delay_falls_back_and_still_settles():
    """A rig that won't report its delay must still get a settle budget."""
    ctx = ScriptedCtx([1, 1, 0], delay=None)
    v = verify_z_controller(ctx, expect=False)
    assert v["switch_off_delay_s"] is None
    assert v["matches"] is True, "fallback budget must still allow settling"


def test_a_misconfigured_delay_cannot_hang_the_skill():
    """A 1e6-second switch-off delay must not wedge the run."""
    import time
    ctx = ScriptedCtx([1] * 400, delay=1e6)
    t0 = time.monotonic()
    v = verify_z_controller(ctx, expect=False)
    assert (time.monotonic() - t0) < 12.0, "ceiling must cap the wait"
    assert v["matches"] is False


# ── explicit opt-out keeps the old single-read shape ─────────────────────────

def test_settle_false_reads_exactly_once():
    """Pins the OLD behaviour so the bug itself stays described in the suite."""
    ctx = ScriptedCtx([1, 0])
    v = verify_z_controller(ctx, expect=False, settle=False)
    assert ctx.reads == 1
    assert v["on"] is True and v["matches"] is False


def test_settle_is_on_by_default():
    """A call site that forgets the argument must get the SAFE behaviour."""
    ctx = ScriptedCtx([1, 0])
    verify_z_controller(ctx, expect=False)
    assert ctx.reads >= 2, "default must poll, or we are back at the 2026-07 race"


def test_agreeing_first_read_costs_no_extra_round_trips():
    """Settling must not tax the common case."""
    ctx = ScriptedCtx([0])
    v = verify_z_controller(ctx, expect=False)
    assert ctx.reads == 1
    assert "ZCtrl_SwitchOffDelayGet" not in ctx.verbs
    assert v["waited_s"] < 0.05


def test_expect_none_never_polls():
    """With nothing to compare against there is no 'disagreement' to settle."""
    ctx = ScriptedCtx([1])
    v = verify_z_controller(ctx, expect=None)
    assert ctx.reads == 1
    assert v["matches"] is None
