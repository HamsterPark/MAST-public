"""The Nanonis modules MAST could not reach at all (2026-07-13).

The operator asked about output control; the census that answered them found the
gap was much wider. Of 684 Nanonis API methods MAST called 266, and **31 modules
had zero coverage**. This file covers the ones closed here — the ones any base
Nanonis controller has, and the ones an autonomous run actually needs:

    Marks       — mark WHERE you did something on the scan image
    DataLog     — record channels to file over time
    TCPLog      — stream channels over TCP
    FunGen1/2Ch — waveform generators: an output that MOVES (the other half of
                  the operator's question — a user output holds a value, a
                  generator sweeps one)
    BiasSwp     — the bias sweeper (distinct from bias spectroscopy)
    Signals     — what a signal's number MEANS (calibration)

The tests below are unit tests over a fake instrument. What they pin is the part
that would fail SILENTLY on real hardware: the argument order and the unit
conversions. A skill that passes 1000 Hz where Nanonis expects a period in seconds
does not error — it just runs the generator six orders of magnitude too slow, on
an output nobody can see.
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

from mast.core.types import NanonisCallRecord  # noqa: E402
from mast.skills.builtins.bias_sweep import (  # noqa: E402
    GetSignalCalibration,
    RunBiasSweep,
    SetAcquisitionPeriod,
)
from mast.skills.builtins.datalog import (  # noqa: E402
    StartDataLog,
    StartTcpLog,
    StopDataLog,
)
from mast.skills.builtins.function_generator import (  # noqa: E402
    ConfigureWaveform,
    StartWaveform,
    StopWaveform,
)
from mast.skills.builtins.marks import (  # noqa: E402
    DrawScanMarker,
    EraseScanMarkers,
    ListScanMarkers,
)


class Ctx:
    def __init__(self, canned: dict | None = None):
        self.calls: list[tuple] = []
        self.canned = canned or {}

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        e = self.canned.get(method, {})
        return NanonisCallRecord(method=method, args=args,
                                 return_value=e.get("return_value", ("", b"", [])),
                                 error=e.get("error", ""))

    def methods(self):
        return [m for m, _ in self.calls]


# ════════════════════════════════════════════════════════════════════════
# The silent-failure class: units and argument order
# ════════════════════════════════════════════════════════════════════════

class TestUnitsAndArgumentOrder:
    def test_the_2ch_generator_takes_a_PERIOD_not_a_frequency(self):
        """Nanonis's 2-channel generator takes ``Time`` (a period, in seconds); the
        1-channel one takes ``Frequency`` (Hz). Passing 1000 Hz straight into the
        2-channel PropsSet sets a period of 1000 SECONDS — a factor of 10^6, with no
        error, on hardware MAST cannot see."""
        ctx = Ctx()
        ConfigureWaveform().execute(ctx, {
            "generator": "2ch", "amplitude": 0.5, "frequency_hz": 1000.0,
            "channel": 1, "shape": "sine"})
        props = next(a for m, a in ctx.calls if m == "FunGen2Ch_PropsSet")
        assert props[2] == pytest.approx(0.001), (
            f"the 2-channel generator was given {props[2]} as its period — a "
            f"1 kHz request became a {props[2]} s period")

    def test_the_1ch_generator_takes_the_frequency_directly(self):
        ctx = Ctx()
        ConfigureWaveform().execute(ctx, {
            "generator": "1ch", "amplitude": 0.5, "frequency_hz": 1000.0})
        props = next(a for m, a in ctx.calls if m == "FunGen1Ch_PropsSet")
        assert props == (0.5, 1000.0, 0, 0)

    def test_datalog_duration_is_split_into_h_m_s(self):
        """DataLog_PropsSet takes hours, minutes, seconds SEPARATELY. Handing it a
        raw seconds count in the hours slot records for 3661 HOURS."""
        ctx = Ctx()
        StartDataLog().execute(ctx, {"channels": "0,24", "duration_s": 3661.5})
        props = next(a for m, a in ctx.calls if m == "DataLog_PropsSet")
        _mode, h, m, s = props[0], props[1], props[2], props[3]
        assert (h, m) == (1, 1) and s == pytest.approx(1.5)

    def test_a_zero_duration_means_continuous_not_zero_seconds(self):
        ctx = Ctx()
        StartDataLog().execute(ctx, {"channels": "0", "duration_s": 0})
        mode = next(a for m, a in ctx.calls if m == "DataLog_PropsSet")[0]
        assert mode == 1, "duration=0 must mean CONTINUOUS, not 'record for 0 s'"

    def test_tcplog_passes_the_channel_COUNT_before_the_channels(self):
        """TCPLog_ChsSet(Num_channels, Channel_indexes) — the count comes first."""
        ctx = Ctx()
        StartTcpLog().execute(ctx, {"channels": "0,2,14", "oversampling": 5})
        assert ("TCPLog_ChsSet", (3, [0, 2, 14])) in ctx.calls

    def test_bias_sweep_limits_are_ordered_low_then_high(self):
        """BiasSwp_LimitsSet(Lower, Upper). Handing it (upper, lower) sweeps
        backwards, or not at all."""
        ctx = Ctx()
        RunBiasSweep().execute(ctx, {"lower_limit_v": 1.0, "upper_limit_v": -1.0})
        lim = next(a for m, a in ctx.calls if m == "BiasSwp_LimitsSet")
        assert lim == (-1.0, 1.0), "the limits were not normalised to (low, high)"


# ════════════════════════════════════════════════════════════════════════
# Marks — the agent can finally say WHERE
# ════════════════════════════════════════════════════════════════════════

class TestMarks:
    def test_a_point_marker_carries_its_label(self):
        ctx = Ctx()
        res = DrawScanMarker().execute(ctx, {
            "kind": "point", "x_m": 1e-9, "y_m": -2e-9,
            "text": "STS #3", "color": "green"})
        assert res.success
        m, a = ctx.calls[0]
        assert m == "Marks_PointDraw"
        assert a[:3] == (1e-9, -2e-9, "STS #3")
        assert a[3] == 0x00FF00

    def test_a_line_marker_needs_both_endpoints(self):
        ctx = Ctx()
        res = DrawScanMarker().execute(ctx, {
            "kind": "line", "x_m": 0.0, "y_m": 0.0})
        assert res.success is False
        assert not ctx.calls, "a half-specified line still reached the instrument"

    def test_erase_vs_hide_are_different_verbs(self):
        ctx = Ctx()
        EraseScanMarkers().execute(ctx, {"kind": "point", "index": 2})
        assert ctx.calls[0] == ("Marks_PointsErase", (2,))
        ctx2 = Ctx()
        EraseScanMarkers().execute(ctx2, {"kind": "line", "index": 1,
                                          "hide_only": True})
        assert ctx2.calls[0] == ("Marks_LinesVisibleSet", (1, 0))

    def test_listing_survives_one_of_the_two_reads_failing(self):
        ctx = Ctx({"Marks_LinesGet": {"error": "no lines"}})
        res = ListScanMarkers().execute(ctx, {})
        assert res.success, "a missing line list killed the point list too"
        assert res.data["lines"] == []


# ════════════════════════════════════════════════════════════════════════
# Sequencing: these skills are TASKS, so they must sequence the API for you
# ════════════════════════════════════════════════════════════════════════

class TestTaskLevelSequencing:
    def test_start_datalog_does_open_chs_props_start_in_that_order(self):
        """The API needs four calls in order. Making the model chain them is how it
        gets one wrong; the skill's job is to be the sequence."""
        ctx = Ctx()
        assert StartDataLog().execute(ctx, {"channels": "0"}).success
        assert ctx.methods() == ["DataLog_Open", "DataLog_ChsSet",
                                 "DataLog_PropsSet", "DataLog_Start"]

    def test_a_failure_midway_stops_the_sequence(self):
        ctx = Ctx({"DataLog_ChsSet": {"error": "bad channel"}})
        res = StartDataLog().execute(ctx, {"channels": "999"})
        assert res.success is False
        assert "DataLog_Start" not in ctx.methods(), (
            "the logger was STARTED after its channel config failed")

    def test_bias_sweep_sequences_open_limits_props_start(self):
        ctx = Ctx()
        assert RunBiasSweep().execute(
            ctx, {"lower_limit_v": -1.0, "upper_limit_v": 1.0}).success
        assert ctx.methods() == ["BiasSwp_Open", "BiasSwp_LimitsSet",
                                 "BiasSwp_PropsSet", "BiasSwp_Start"]

    def test_bad_channels_never_reach_the_instrument(self):
        ctx = Ctx()
        res = StartDataLog().execute(ctx, {"channels": "not,a,number"})
        assert res.success is False and not ctx.calls


# ════════════════════════════════════════════════════════════════════════
# Stops must always be reachable — including after an abort
# ════════════════════════════════════════════════════════════════════════

class TestStopsSurviveAnAbort:
    @pytest.mark.parametrize("verb", [
        "FunGen1Ch_Stop", "FunGen2Ch_Stop", "DataLog_Stop", "TCPLog_Stop",
    ])
    def test_every_new_stop_verb_is_abort_safe(self, verb):
        """A generator left running keeps driving an output line — the exact thing
        an operator pressing 中止 wants to end. If the abort gate refuses the STOP,
        the abort does the opposite of its job."""
        from mast.core.execution_context import _is_abort_safe

        assert _is_abort_safe(verb, ()) is True

    def test_stop_waveform_calls_a_literal_verb(self):
        """Every safety tool MAST has (the abort allow-list guard, the coverage
        census) reads the Nanonis verb back out of the source by grepping
        `safe_call("…")`. A verb hidden behind a variable is invisible to all of
        them — the guard caught exactly that here."""
        import mast.skills.builtins.function_generator as fg

        src = Path(fg.__file__).read_text(encoding="utf-8")
        assert 'safe_call("FunGen1Ch_Stop")' in src
        assert 'safe_call("FunGen2Ch_Stop")' in src
        # CODE lines only — the comment above the fix quotes the bad pattern to
        # explain it, and a naive substring check would flag the explanation.
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        assert "safe_call(verb" not in code, (
            "a Nanonis verb is hidden behind a variable — the safety tooling "
            "cannot see it")

    def test_stopping_a_generator_needs_no_confirmation(self):
        from mast.core.types import SafetyLevel

        assert StopWaveform().metadata().safety_level == SafetyLevel.AUTO, (
            "stopping an output must never be gated — it is the one thing you "
            "always want to be able to do")


# ════════════════════════════════════════════════════════════════════════
# Coverage — pin the gap closed
# ════════════════════════════════════════════════════════════════════════

def test_the_zero_coverage_modules_are_covered_now():
    import re

    from mast.agents.instrument_control.tools import discover_instrument_skills

    called: set[str] = set()
    pat = re.compile(r'safe_call\(\s*["\']([A-Za-z0-9_]+)["\']')
    for p in (Path(_MASTV2_ROOT) / "mast").rglob("*.py"):
        called |= set(pat.findall(p.read_text(encoding="utf-8", errors="replace")))

    for module in ("UserOut", "DigLines", "Marks", "DataLog", "TCPLog",
                   "FunGen1Ch", "FunGen2Ch", "BiasSwp"):
        assert any(v.startswith(module + "_") for v in called), (
            f"{module}_* is back to zero coverage")

    names = {m.name for m in discover_instrument_skills().list_skills()}
    assert {"DrawScanMarker", "StartDataLog", "StartTcpLog", "ConfigureWaveform",
            "StartWaveform", "StopWaveform", "RunBiasSweep",
            "GetSignalCalibration", "SetAcquisitionPeriod",
            "ConfigureCalculatedOutput"} <= names


def test_signal_calibration_says_what_a_number_means():
    ctx = Ctx({"Signals_CalibrGet": {"return_value": ("", b"", [1e-9, 0.5])}})
    res = GetSignalCalibration().execute(ctx, {"signal_index": 0})
    assert res.success
    assert res.data["calibration"] == 1e-9 and res.data["offset"] == 0.5


def test_acquisition_period_is_confirm_gated():
    from mast.core.types import SafetyLevel

    m = SetAcquisitionPeriod().metadata()
    assert m.safety_level != SafetyLevel.AUTO, (
        "the acquisition period affects EVERY measurement the controller makes")


def test_stop_datalog_is_unconditional():
    ctx = Ctx()
    assert StopDataLog().execute(ctx, {}).success
    assert ctx.calls == [("DataLog_Stop", ())]
