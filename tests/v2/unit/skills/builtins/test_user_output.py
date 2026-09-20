"""User Outputs + Digital Lines — the outputs MAST could not drive (2026-07-13).

The operator's report: "nanonis 有一些自定义输出控制…没有对应的 skill". It was
total — 12 ``UserOut_*`` + 4 ``DigLines_*`` methods in the Nanonis API, and MAST
had a skill for **none** of them. The agent could scan, sweep, pulse the tip and
read every signal in the machine, and could not turn on a single output.

The tests that matter here are the SAFETY ones, because a user output is not a
bias. A bias is a property of the microscope and MAST can hold its envelope in
``SafetyLimits``. A user output drives hardware MAST **cannot see** — a gate
voltage, a piezo amplifier, a laser shutter, a delay line — in units MAST does not
know (the value is in *calibrated physical units*, not volts).

So the safety model is not "MAST invents a bound". It is:

  * ``SetUserOutput`` READS the operator-configured physical limits off the
    instrument and refuses to exceed them;
  * a limits read that FAILS is a REFUSAL, not a free pass — an unknown envelope
    is not an open one;
  * ``UserOut_LimitsSet`` is not a skill at all, because those limits ARE the
    guardrail: an agent that can widen its own guardrail does not have one.

Each of those three is a line that is easy to write the wrong way round, so each
is pinned.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
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

from mast.core.types import NanonisCallRecord, SafetyLevel, SkillCategory  # noqa: E402
from mast.skills.builtins.user_output import (  # noqa: E402
    ConfigureDigitalLine,
    GetDigitalLineTTL,
    GetUserOutputLimits,
    GetUserOutputMode,
    PulseDigitalLine,
    SetDigitalLineStatus,
    SetUserOutput,
    SetUserOutputMode,
)


class Ctx:
    """Records every Nanonis call; canned replies keyed by verb."""

    def __init__(self, canned: dict | None = None):
        self.calls: list[tuple] = []
        self.canned = canned or {}

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is None:
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", []))
        return NanonisCallRecord(method=method, args=args,
                                 return_value=entry.get("return_value", ("", b"", [])),
                                 error=entry.get("error", ""))

    def methods(self) -> list[str]:
        return [m for m, _ in self.calls]


def _limits(upper: float, lower: float) -> dict:
    return {"UserOut_LimitsGet": {"return_value": ("", b"", [upper, lower])}}


# ════════════════════════════════════════════════════════════════════════
# THE safety model — MAST does not invent a bound; it obeys the operator's
# ════════════════════════════════════════════════════════════════════════

class TestSetUserOutputObeysTheOperatorsLimits:
    def test_it_reads_the_limits_before_it_writes(self):
        """A bias has an envelope MAST can hold. A user output does not — MAST has
        no idea whether this channel is volts, micrometres, or a shutter. Nanonis
        knows: the operator configured physical limits. Read them."""
        ctx = Ctx(_limits(5.0, -5.0))
        res = SetUserOutput().execute(ctx, {"output_index": 1, "value": 2.0})
        assert res.success
        assert ctx.methods() == ["UserOut_LimitsGet", "UserOut_ValSet"], (
            "the write went out WITHOUT reading the channel's limits first")
        assert ctx.calls[1] == ("UserOut_ValSet", (1, 2.0))

    def test_a_value_above_the_limit_is_refused_not_clipped(self):
        """Clipping would silently drive a DIFFERENT value than the agent asked
        for — on hardware nobody can see. Refuse and say so."""
        ctx = Ctx(_limits(5.0, -5.0))
        res = SetUserOutput().execute(ctx, {"output_index": 1, "value": 9.0})
        assert res.success is False
        assert "UserOut_ValSet" not in ctx.methods(), "the out-of-range value was written"
        # the refusal must name the envelope it refused against — "out of range"
        # with no range is a dead end for both the agent and the operator
        assert "超出" in res.error
        assert "-5" in res.error and "5" in res.error

    def test_a_value_below_the_limit_is_refused(self):
        ctx = Ctx(_limits(5.0, -5.0))
        res = SetUserOutput().execute(ctx, {"output_index": 1, "value": -9.0})
        assert res.success is False
        assert "UserOut_ValSet" not in ctx.methods()

    def test_the_limits_may_be_returned_in_either_order(self):
        """Nanonis returns (upper, lower). Do not assume the sign — an output can
        be configured 0..10 or -10..0, and a naive `lower <= v <= upper` inverts
        on a channel whose 'upper' is the more negative number."""
        ctx = Ctx(_limits(-1.0, -9.0))          # upper=-1, lower=-9
        assert SetUserOutput().execute(ctx, {"output_index": 2, "value": -5.0}).success
        ctx2 = Ctx(_limits(-1.0, -9.0))
        assert not SetUserOutput().execute(ctx2, {"output_index": 2, "value": 0.0}).success

    def test_a_limits_READ_FAILURE_is_a_refusal_not_a_free_pass(self):
        """The line that is easiest to get exactly backwards. If the limits cannot
        be read, MAST does not know the envelope — and an unknown envelope is not
        an open one. Falling through to the write here would mean: the ONE case
        where we know nothing is the one case where we write anything."""
        ctx = Ctx({"UserOut_LimitsGet": {"error": "communication timeout"}})
        res = SetUserOutput().execute(ctx, {"output_index": 1, "value": 2.0})
        assert res.success is False
        assert "UserOut_ValSet" not in ctx.methods(), (
            "the limits read FAILED and MAST wrote to the output anyway")
        assert "拒绝写入" in res.error

    def test_a_malformed_limits_reply_is_also_a_refusal(self):
        ctx = Ctx({"UserOut_LimitsGet": {"return_value": ("", b"", [3.0])}})  # only 1
        res = SetUserOutput().execute(ctx, {"output_index": 1, "value": 1.0})
        assert res.success is False
        assert "UserOut_ValSet" not in ctx.methods()

    def test_the_boundary_itself_is_allowed(self):
        ctx = Ctx(_limits(5.0, -5.0))
        assert SetUserOutput().execute(ctx, {"output_index": 1, "value": 5.0}).success


class TestWideningTheGuardrailIsAllowedButRecorded:
    """The first draft of this file WITHHELD ``SetUserOutputLimits``, on the
    software-security principle that "an agent which can widen its own guardrail
    has none".

    The operator overruled that, and they were right: *"nanonis 自己的硬件输出很
    小的，我们在硬件接线的时候就会注意的."* A Nanonis user output is a small-signal
    analog line; the real protection lives in the WIRING — the amplifier, the
    interlock, the choice of what to connect. Treating a configuration convenience
    as the last physical barrier imported a threat model from a different domain
    into a lab whose physical layer is already designed for this.

    So the guardrail is crossable. What replaces the block is a RECORD: the change
    is CONFIRM-gated like every other write here, and it lands in the refusal
    ledger. ``SetUserOutput``'s check still earns its place — it catches the honest
    out-of-range mistake, which is the common case — and going outside the envelope
    now takes a deliberate, logged act rather than an accident.
    """

    def test_the_limits_can_be_changed(self):
        from mast.skills.builtins.user_output import SetUserOutputLimits

        ctx = Ctx(_limits(5.0, -5.0))
        res = SetUserOutputLimits().execute(
            ctx, {"output_index": 1, "upper_limit": 10.0, "lower_limit": -10.0})
        assert res.success
        assert ("UserOut_LimitsSet", (1, 10.0, -10.0, 0)) in ctx.calls
        assert res.data["previous"] == {"upper": 5.0, "lower": -5.0}

    def test_a_widening_is_written_to_the_refusal_ledger(self, tmp_path, monkeypatch):
        """Not a block — a record. If an output later drives something odd, the
        widening that allowed it is on disk, with the before/after."""
        monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
        from mast.core import diagnostics as diag
        from mast.skills.builtins.user_output import SetUserOutputLimits

        diag.clear()
        SetUserOutputLimits().execute(
            Ctx(_limits(5.0, -5.0)),
            {"output_index": 3, "upper_limit": 20.0, "lower_limit": -20.0})

        rows = [r for r in diag.recent() if "limits" in r["subject"]]
        assert rows, "a guardrail change left no trace"
        r = rows[0]
        assert r["widened"] is True
        assert r["before"] == {"upper": 5.0, "lower": -5.0}
        assert r["after"] == {"upper": 20.0, "lower": -20.0}
        diag.clear()

    def test_a_narrowing_is_recorded_but_not_flagged_as_widening(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
        from mast.core import diagnostics as diag
        from mast.skills.builtins.user_output import SetUserOutputLimits

        diag.clear()
        SetUserOutputLimits().execute(
            Ctx(_limits(10.0, -10.0)),
            {"output_index": 1, "upper_limit": 2.0, "lower_limit": -2.0})
        rows = [r for r in diag.recent() if "limits" in r["subject"]]
        assert rows and rows[0]["widened"] is False, (
            "tightening a limit is not a guardrail breach and must not read as one")
        diag.clear()

    def test_changing_the_limits_still_needs_confirmation(self):
        from mast.skills.builtins.user_output import SetUserOutputLimits

        m = SetUserOutputLimits().metadata()
        assert m.safety_level != SafetyLevel.AUTO, (
            "widening what the agent may drive must not be an unconfirmed action")

    def test_SetUserOutput_still_checks_the_limits(self):
        """The check is not decorative just because the limits are settable: it
        still catches the honest out-of-range mistake, which is the common case.
        Crossing the envelope now takes a DELIBERATE, logged call — not a typo."""
        ctx = Ctx(_limits(5.0, -5.0))
        assert not SetUserOutput().execute(
            ctx, {"output_index": 1, "value": 9.0}).success
        assert "UserOut_ValSet" not in ctx.methods()


# ════════════════════════════════════════════════════════════════════════
# Protocol semantics — these are easy to get wrong from memory
# ════════════════════════════════════════════════════════════════════════

class TestProtocolSemantics:
    def test_output_index_is_one_based(self):
        """The protocol says 1..N. A 0-based skill would drive the WRONG channel —
        silently, on hardware MAST cannot see."""
        spec = next(p for p in SetUserOutput().metadata().parameters
                    if p.name == "output_index")
        assert spec.min_value == 1, "output_index must be 1-based (protocol §User Outputs)"

    def test_mode_enum_matches_the_protocol(self):
        ctx = Ctx()
        res = SetUserOutputMode().execute(ctx, {"output_index": 1, "mode": 1})
        assert res.data["mode_name"] == "Monitor"       # 0=User Output 1=Monitor 2=Calc
        assert ctx.calls[0] == ("UserOut_ModeSet", (1, 1))

    def test_pulse_passes_lines_as_a_list_of_1_to_8(self):
        ctx = Ctx()
        res = PulseDigitalLine().execute(ctx, {
            "port": 0, "lines": "1,3", "pulse_width_s": 0.01,
            "pulse_pause_s": 0.005, "n_pulses": 4, "wait_until_finished": True,
        })
        assert res.success
        m, args = ctx.calls[0]
        assert m == "DigLines_Pulse"
        assert args == (0, [1, 3], 0.01, 0.005, 4, 1)

    @pytest.mark.parametrize("bad", ["0", "9", "1,0", "abc"])
    def test_pulse_refuses_lines_outside_1_to_8(self, bad):
        ctx = Ctx()
        res = PulseDigitalLine().execute(ctx, {
            "port": 0, "lines": bad, "pulse_width_s": 0.01})
        assert res.success is False
        assert not ctx.calls, "a bad line spec still reached the instrument"

    def test_digital_status_maps_bool_to_int(self):
        ctx = Ctx()
        SetDigitalLineStatus().execute(ctx, {"port": 1, "line": 3, "status": True})
        assert ctx.calls[0] == ("DigLines_OutStatusSet", (1, 3, 1))

    def test_configure_line_argument_order(self):
        """DigLines_PropsSet(Digital_line, Port, Direction, Polarity) — line FIRST,
        port second. Swapping them configures a different line on a different port."""
        ctx = Ctx()
        ConfigureDigitalLine().execute(
            ctx, {"line": 2, "port": 1, "direction": 1, "polarity": 1})
        assert ctx.calls[0] == ("DigLines_PropsSet", (2, 1, 1, 1))


# ════════════════════════════════════════════════════════════════════════
# Reads + metadata hygiene
# ════════════════════════════════════════════════════════════════════════

class TestReads:
    def test_limits_read(self):
        ctx = Ctx(_limits(10.0, 0.0))
        res = GetUserOutputLimits().execute(ctx, {"output_index": 3})
        assert res.success
        assert res.data["upper_limit"] == 10.0 and res.data["lower_limit"] == 0.0
        assert ctx.calls[0] == ("UserOut_LimitsGet", (3, 0))

    def test_mode_read_names_the_mode(self):
        ctx = Ctx({"UserOut_ModeGet": {"return_value": ("", b"", [2])}})
        res = GetUserOutputMode().execute(ctx, {"output_index": 1})
        assert res.data["mode_name"] == "Calc.Signal"

    def test_ttl_read(self):
        ctx = Ctx({"DigLines_TTLValGet": {"return_value": ("", b"", [[0, 1, 0, 1]])}})
        res = GetDigitalLineTTL().execute(ctx, {"port": 0})
        assert res.success and res.data["port_name"] == "A"

    def test_a_read_error_surfaces(self):
        ctx = Ctx({"UserOut_ModeGet": {"error": "no such output"}})
        res = GetUserOutputMode().execute(ctx, {"output_index": 99})
        assert res.success is False and "no such output" in res.error


class TestMetadata:
    WRITES = (SetUserOutput, SetUserOutputMode, PulseDigitalLine,
              SetDigitalLineStatus, ConfigureDigitalLine)
    READS = (GetUserOutputLimits, GetUserOutputMode, GetDigitalLineTTL)

    @pytest.mark.parametrize("cls", WRITES)
    def test_every_write_needs_confirmation(self, cls):
        """These drive hardware MAST cannot see. None of them is AUTO."""
        m = cls().metadata()
        assert m.category == SkillCategory.WRITE
        assert m.safety_level != SafetyLevel.AUTO, (
            f"{m.name} would run unconfirmed — it drives external hardware")

    @pytest.mark.parametrize("cls", READS)
    def test_reads_are_auto(self, cls):
        m = cls().metadata()
        assert m.category == SkillCategory.READ
        assert m.safety_level == SafetyLevel.AUTO

    def test_the_description_says_the_unit_is_unknown(self):
        """The single most dangerous assumption an agent could make here is that
        the value is in volts. It is in the channel's calibrated physical units —
        which might be µm, or mW, or a shutter state."""
        d = SetUserOutput().metadata().description
        assert "不一定是伏特" in d


def test_the_gap_is_actually_closed():
    """The complaint was that Nanonis's output controls had no skill at all. Pin
    the coverage so it cannot silently regress."""
    from mast.agents.instrument_control.tools import discover_instrument_skills

    names = {m.name for m in discover_instrument_skills().list_skills()}
    assert {"SetUserOutput", "GetUserOutputLimits", "GetUserOutputMode",
            "SetUserOutputMode", "SetUserOutputMonitorChannel",
            "GetUserOutputMonitorChannel", "PulseDigitalLine",
            "SetDigitalLineStatus", "GetDigitalLineTTL",
            "ConfigureDigitalLine"} <= names
