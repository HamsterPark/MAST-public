"""v2 unit tests for mast.skills.builtins.pll.

Skills covered (36):
  ConfigurePLL (CONFIRM), GetPLLStatus (AUTO), PLLOnOff (CONFIRM),
  ConfigurePLLExcitation (CONFIRM), AcquirePLLFreqSweep (CONFIRM),
  PLLSignalAnalyzer (AUTO), GetPLLAddOnOff (AUTO),
  SetPLLAmpCtrlBandwidth (CONFIRM), GetPLLAmpCtrlOnOff (AUTO),
  SetPLLAmpCtrlSetpnt (CONFIRM), GetPLLDemodFilter (AUTO),
  SetPLLDemodFilter (CONFIRM), GetPLLDemodHarmonic (AUTO),
  GetPLLDemodInput (AUTO), SetPLLDemodInput (CONFIRM),
  SetPLLDemodPhasRef (CONFIRM), GetPLLExcRange (AUTO),
  SetPLLFreqExcOverwrite (CONFIRM), GetPLLFreqRange (AUTO),
  SetPLLFreqRange (CONFIRM), PLLFreqShiftAutoCenter (CONFIRM),
  GetPLLInpCalibr (AUTO), SetPLLInpCalibr (CONFIRM),
  GetPLLInpProps (AUTO), SetPLLInpProps (CONFIRM), SetPLLInpRange (CONFIRM),
  PLLPerfectPLLUpdtZTC (CONFIRM), SetPLLPhasCtrlBandwidth (CONFIRM),
  GetPLLPhasCtrlOnOff (AUTO), GetPLLSignalAnlzrCh (AUTO),
  GetPLLSignalAnlzrFFTProps (AUTO), GetPLLSignalAnlzrTimebase (AUTO),
  PLLSignalAnlzrTrigAuto (CONFIRM), SetPLLSignalAnlzrTrig (CONFIRM),
  GetPLLFreqSwpParams (AUTO), StopPLLFreqSwp (CONFIRM).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_pll.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.pll import (
    AcquirePLLFreqSweep,
    ConfigurePLL,
    ConfigurePLLExcitation,
    GetPLLAddOnOff,
    GetPLLAmpCtrlOnOff,
    GetPLLDemodFilter,
    GetPLLDemodHarmonic,
    GetPLLDemodInput,
    GetPLLExcRange,
    GetPLLFreqRange,
    GetPLLFreqSwpParams,
    GetPLLInpCalibr,
    GetPLLInpProps,
    GetPLLPhasCtrlOnOff,
    GetPLLSignalAnlzrCh,
    GetPLLSignalAnlzrFFTProps,
    GetPLLSignalAnlzrTimebase,
    GetPLLStatus,
    PLLFreqShiftAutoCenter,
    PLLOnOff,
    PLLPerfectPLLUpdtZTC,
    PLLSignalAnalyzer,
    PLLSignalAnlzrTrigAuto,
    SetPLLAmpCtrlBandwidth,
    SetPLLAmpCtrlSetpnt,
    SetPLLDemodFilter,
    SetPLLDemodInput,
    SetPLLDemodPhasRef,
    SetPLLFreqExcOverwrite,
    SetPLLFreqRange,
    SetPLLInpCalibr,
    SetPLLInpProps,
    SetPLLInpRange,
    SetPLLPhasCtrlBandwidth,
    SetPLLSignalAnlzrTrig,
    StopPLLFreqSwp,
)


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("skill_cls,expected_name,expected_danger", [
    (ConfigurePLL, "ConfigurePLL", "CONFIRM"),
    (GetPLLStatus, "GetPLLStatus", "AUTO"),
    (PLLOnOff, "PLLOnOff", "CONFIRM"),
    (ConfigurePLLExcitation, "ConfigurePLLExcitation", "CONFIRM"),
    (AcquirePLLFreqSweep, "AcquirePLLFreqSweep", "CONFIRM"),
    (PLLSignalAnalyzer, "PLLSignalAnalyzer", "AUTO"),
    (GetPLLAddOnOff, "GetPLLAddOnOff", "AUTO"),
    (SetPLLAmpCtrlBandwidth, "SetPLLAmpCtrlBandwidth", "CONFIRM"),
    (GetPLLAmpCtrlOnOff, "GetPLLAmpCtrlOnOff", "AUTO"),
    (SetPLLAmpCtrlSetpnt, "SetPLLAmpCtrlSetpnt", "CONFIRM"),
    (GetPLLDemodFilter, "GetPLLDemodFilter", "AUTO"),
    (SetPLLDemodFilter, "SetPLLDemodFilter", "CONFIRM"),
    (GetPLLDemodHarmonic, "GetPLLDemodHarmonic", "AUTO"),
    (GetPLLDemodInput, "GetPLLDemodInput", "AUTO"),
    (SetPLLDemodInput, "SetPLLDemodInput", "CONFIRM"),
    (SetPLLDemodPhasRef, "SetPLLDemodPhasRef", "CONFIRM"),
    (GetPLLExcRange, "GetPLLExcRange", "AUTO"),
    (SetPLLFreqExcOverwrite, "SetPLLFreqExcOverwrite", "CONFIRM"),
    (GetPLLFreqRange, "GetPLLFreqRange", "AUTO"),
    (SetPLLFreqRange, "SetPLLFreqRange", "CONFIRM"),
    (PLLFreqShiftAutoCenter, "PLLFreqShiftAutoCenter", "CONFIRM"),
    (GetPLLInpCalibr, "GetPLLInpCalibr", "AUTO"),
    (SetPLLInpCalibr, "SetPLLInpCalibr", "CONFIRM"),
    (GetPLLInpProps, "GetPLLInpProps", "AUTO"),
    (SetPLLInpProps, "SetPLLInpProps", "CONFIRM"),
    (SetPLLInpRange, "SetPLLInpRange", "CONFIRM"),
    (PLLPerfectPLLUpdtZTC, "PLLPerfectPLLUpdtZTC", "CONFIRM"),
    (SetPLLPhasCtrlBandwidth, "SetPLLPhasCtrlBandwidth", "CONFIRM"),
    (GetPLLPhasCtrlOnOff, "GetPLLPhasCtrlOnOff", "AUTO"),
    (GetPLLSignalAnlzrCh, "GetPLLSignalAnlzrCh", "AUTO"),
    (GetPLLSignalAnlzrFFTProps, "GetPLLSignalAnlzrFFTProps", "AUTO"),
    (GetPLLSignalAnlzrTimebase, "GetPLLSignalAnlzrTimebase", "AUTO"),
    (PLLSignalAnlzrTrigAuto, "PLLSignalAnlzrTrigAuto", "CONFIRM"),
    (SetPLLSignalAnlzrTrig, "SetPLLSignalAnlzrTrig", "CONFIRM"),
    (GetPLLFreqSwpParams, "GetPLLFreqSwpParams", "AUTO"),
    (StopPLLFreqSwp, "StopPLLFreqSwp", "CONFIRM"),
])
def test_pll_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".pll")


# ── Required-field schema tests ───────────────────────────────────────────────

def test_pll_on_off_required_output_on():
    tool = wrap_skill(PLLOnOff, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["output_on"].is_required()
    assert fields["output_on"].annotation is bool
    assert not fields["modulator_index"].is_required()


def test_configure_pll_excitation_required_excitation_v():
    tool = wrap_skill(ConfigurePLLExcitation, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["excitation_v"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["excitation_v"].annotation is str
    assert not fields["modulator_index"].is_required()


def test_acquire_pll_freq_sweep_required_fields():
    tool = wrap_skill(AcquirePLLFreqSweep, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["num_points"].is_required()
    assert fields["period_s"].is_required()
    assert not fields["sweep_up"].is_required()


def test_set_pll_amp_ctrl_bandwidth_required():
    tool = wrap_skill(SetPLLAmpCtrlBandwidth, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["bandwidth_hz"].is_required()


def test_set_pll_demod_input_required():
    tool = wrap_skill(SetPLLDemodInput, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["input"].is_required()
    assert fields["frequency_generator"].is_required()


def test_set_pll_signal_anlzr_trig_required_fields():
    tool = wrap_skill(SetPLLSignalAnlzrTrig, make_provider())
    fields = tool.args_schema.model_fields
    for f in ["trigger_mode", "trigger_source", "trigger_slope",
              "trigger_level", "trigger_position_s", "arming_mode"]:
        assert fields[f].is_required(), f"{f} should be required"


def test_set_pll_inp_props_required_fields():
    tool = wrap_skill(SetPLLInpProps, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["differential_input"].is_required()
    assert fields["differential_input"].annotation is bool
    assert fields["divider_1_10"].is_required()


# ── Execution tests ───────────────────────────────────────────────────────────

def test_configure_pll_center_freq():
    canned = {"PLL_CenterFreqSet": {"return_value": None}}
    tool = wrap_skill(ConfigurePLL, make_provider(canned))
    result = _invoke(tool, center_freq_hz=25000.0)
    assert result.update["executed_skills"] == ["ConfigurePLL"]


def test_configure_pll_both_gains():
    canned = {
        "PLL_AmpCtrlGainSet": {"return_value": None},
        "PLL_PhasCtrlGainSet": {"return_value": None},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(ConfigurePLL, capturing_provider)
    _invoke(tool, amp_p_gain=1e-3, phas_p_gain=5.0)
    ctx = instances[-1]
    assert any(c[0] == "PLL_AmpCtrlGainSet" for c in ctx.calls)
    assert any(c[0] == "PLL_PhasCtrlGainSet" for c in ctx.calls)


def test_get_pll_status_executes():
    canned = {
        "PLL_CenterFreqGet": {"return_value": ("", b"", [25000.0])},
        "PLL_FreqShiftGet": {"return_value": ("", b"", [0.0])},
        "PLL_AmpCtrlGainGet": {"return_value": ("", b"", [1e-3, 1e-3, 0.0])},
        "PLL_PhasCtrlGainGet": {"return_value": ("", b"", [5.0, 1e-3])},
        "PLL_ExcitationGet": {"return_value": ("", b"", [0.1])},
    }
    tool = wrap_skill(GetPLLStatus, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetPLLStatus"]


def test_pll_on_off_executes():
    canned = {
        "PLL_OutOnOffSet": {"return_value": None},
        "PLL_PhasCtrlOnOffSet": {"return_value": None},
        "PLL_AmpCtrlOnOffSet": {"return_value": None},
    }
    tool = wrap_skill(PLLOnOff, make_provider(canned))
    result = _invoke(tool, output_on=True)
    assert result.update["executed_skills"] == ["PLLOnOff"]


def test_pll_on_off_error_propagates():
    canned = {"PLL_OutOnOffSet": {"error": "PLL fault"}}
    tool = wrap_skill(PLLOnOff, make_provider(canned))
    result = _invoke(tool, output_on=False)
    assert result.update["messages"][0].status == "error"


def test_configure_pll_excitation_executes():
    canned = {"PLL_ExcitationSet": {"return_value": None}}
    tool = wrap_skill(ConfigurePLLExcitation, make_provider(canned))
    result = _invoke(tool, excitation_v=0.05)
    assert result.update["executed_skills"] == ["ConfigurePLLExcitation"]


def test_acquire_pll_freq_sweep_executes():
    canned = {
        "PLLFreqSwp_Open": {"return_value": None},
        "PLLFreqSwp_ParamsSet": {"return_value": None},
        "PLLFreqSwp_Start": {"return_value": None},
    }
    tool = wrap_skill(AcquirePLLFreqSweep, make_provider(canned))
    result = _invoke(tool, num_points=100, period_s=0.01)
    assert result.update["executed_skills"] == ["AcquirePLLFreqSweep"]


def test_pll_signal_analyzer_executes():
    canned = {
        "PLLSignalAnlzr_Open": {"return_value": None},
        "PLLSignalAnlzr_ChSet": {"return_value": None},
        "PLLSignalAnlzr_OsciDataGet": {"return_value": ("", b"", [0.0])},
    }
    tool = wrap_skill(PLLSignalAnalyzer, make_provider(canned))
    result = _invoke(tool, channel_index=0)
    assert result.update["executed_skills"] == ["PLLSignalAnalyzer"]


def test_set_pll_amp_ctrl_bandwidth_executes():
    canned = {"PLL_AmpCtrlBandwidthSet": {"return_value": None}}
    tool = wrap_skill(SetPLLAmpCtrlBandwidth, make_provider(canned))
    result = _invoke(tool, bandwidth_hz=10.0)
    assert result.update["executed_skills"] == ["SetPLLAmpCtrlBandwidth"]


def test_get_pll_demod_filter_executes():
    canned = {"PLL_DemodFilterGet": {"return_value": ("", b"", [4])}}
    tool = wrap_skill(GetPLLDemodFilter, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetPLLDemodFilter"]


def test_set_pll_demod_filter_executes():
    canned = {"PLL_DemodFilterSet": {"return_value": None}}
    tool = wrap_skill(SetPLLDemodFilter, make_provider(canned))
    result = _invoke(tool, filter_order=4)
    assert result.update["executed_skills"] == ["SetPLLDemodFilter"]


def test_pll_freq_shift_auto_center_executes():
    canned = {"PLL_FreqShiftAutoCenter": {"return_value": None}}
    tool = wrap_skill(PLLFreqShiftAutoCenter, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["PLLFreqShiftAutoCenter"]


def test_set_pll_inp_props_executes():
    canned = {"PLL_InpPropsSet": {"return_value": None}}
    tool = wrap_skill(SetPLLInpProps, make_provider(canned))
    result = _invoke(tool, differential_input=True, divider_1_10=False)
    assert result.update["executed_skills"] == ["SetPLLInpProps"]


def test_pll_perfect_pll_updt_ztc_executes():
    canned = {"PLL_PerfectPLLUpdtZTC": {"return_value": None}}
    tool = wrap_skill(PLLPerfectPLLUpdtZTC, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["PLLPerfectPLLUpdtZTC"]


def test_pll_signal_anlzr_trig_auto_executes():
    canned = {"PLLSignalAnlzr_TrigAuto": {"return_value": None}}
    tool = wrap_skill(PLLSignalAnlzrTrigAuto, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["PLLSignalAnlzrTrigAuto"]


def test_stop_pll_freq_swp_executes():
    canned = {"PLLFreqSwp_Stop": {"return_value": None}}
    tool = wrap_skill(StopPLLFreqSwp, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["StopPLLFreqSwp"]


def test_configure_pll_error_propagates():
    canned = {"PLL_CenterFreqSet": {"error": "PLL unreachable"}}
    tool = wrap_skill(ConfigurePLL, make_provider(canned))
    result = _invoke(tool, center_freq_hz=25000.0)
    assert result.update["messages"][0].status == "error"


# ── Regression: agent-path None-materialisation (optional-param gating) ───────
#
# In the agent path the pydantic args_schema materialises every declared
# optional parameter as a dict key with value None. The old ``"k" in params``
# gating fired for ALL of them and pushed None to the hardware
# (PLL_FreqShiftSet(mod, None), PLL_AmpCtrlGainSet(mod, None, None), ...).
# The fix gates on ``params.get(k) is not None``. These tests drive the FULL
# agent path (tool.invoke with a tool_call dict) so omitted params are really
# materialised to None by pydantic — a bare dict to .func() would NOT reproduce
# the bug.

def _capturing_provider():
    instances: list[FakeCtx] = []

    def provider():
        ctx = FakeCtx(canned={
            "PLL_CenterFreqSet": {"return_value": ("", b"", [])},
            "PLL_FreqShiftSet": {"return_value": ("", b"", [])},
            "PLL_AmpCtrlGainSet": {"return_value": ("", b"", [])},
            "PLL_PhasCtrlGainSet": {"return_value": ("", b"", [])},
            "PLL_ExcitationSet": {"return_value": ("", b"", [])},
            "PLL_ExcRangeSet": {"return_value": ("", b"", [])},
        })
        instances.append(ctx)
        return ctx

    return provider, instances


def _agent_invoke(tool, args: dict):
    """Drive the real langgraph tool path so pydantic materialises omitted
    optional params to None (reproduces the agent runtime, unlike .func())."""
    return tool.invoke({"name": tool.name, "args": args, "id": "tc-1", "type": "tool_call"})


def test_configure_pll_agent_path_center_only_no_none_writes():
    provider, instances = _capturing_provider()
    tool = wrap_skill(ConfigurePLL, provider)
    _agent_invoke(tool, {"center_freq_hz": "25k"})
    methods = [m for m, _ in instances[-1].calls]
    # Only the supplied parameter must hit the hardware.
    assert methods == ["PLL_CenterFreqSet"]
    assert instances[-1].calls[0] == ("PLL_CenterFreqSet", (1, 25000.0))
    # No None ever sent.
    assert all(None not in args for _, args in instances[-1].calls)


def test_configure_pll_agent_path_amp_gain_defaults_time_constant():
    provider, instances = _capturing_provider()
    tool = wrap_skill(ConfigurePLL, provider)
    _agent_invoke(tool, {"amp_p_gain": 2e-3})
    calls = instances[-1].calls
    assert [m for m, _ in calls] == ["PLL_AmpCtrlGainSet"]
    # Time constant must fall back to 1e-3, NOT the materialised None.
    assert calls[0] == ("PLL_AmpCtrlGainSet", (1, 2e-3, 1e-3))


def test_configure_pll_agent_path_phas_gain_defaults_time_constant():
    provider, instances = _capturing_provider()
    tool = wrap_skill(ConfigurePLL, provider)
    _agent_invoke(tool, {"phas_p_gain": 5.0})
    calls = instances[-1].calls
    assert [m for m, _ in calls] == ["PLL_PhasCtrlGainSet"]
    assert calls[0] == ("PLL_PhasCtrlGainSet", (1, 5.0, 1e-3))


def test_configure_pll_agent_path_explicit_time_constant_honoured():
    provider, instances = _capturing_provider()
    tool = wrap_skill(ConfigurePLL, provider)
    _agent_invoke(tool, {"amp_p_gain": 2e-3, "amp_time_constant_s": "5m"})
    assert instances[-1].calls[0] == ("PLL_AmpCtrlGainSet", (1, 2e-3, 5e-3))


def test_configure_pll_excitation_agent_path_no_output_range_when_omitted():
    provider, instances = _capturing_provider()
    tool = wrap_skill(ConfigurePLLExcitation, provider)
    _agent_invoke(tool, {"excitation_v": "0.05"})
    methods = [m for m, _ in instances[-1].calls]
    # PLL_ExcRangeSet must NOT be called with a materialised-None output_range.
    assert methods == ["PLL_ExcitationSet"]
    assert instances[-1].calls[0] == ("PLL_ExcitationSet", (1, 0.05))


def test_configure_pll_excitation_agent_path_output_range_sent_when_given():
    provider, instances = _capturing_provider()
    tool = wrap_skill(ConfigurePLLExcitation, provider)
    _agent_invoke(tool, {"excitation_v": "0.05", "output_range": 2.0})
    methods = [m for m, _ in instances[-1].calls]
    assert methods == ["PLL_ExcitationSet", "PLL_ExcRangeSet"]
    assert instances[-1].calls[1] == ("PLL_ExcRangeSet", (1, 2.0))


# ── Read-path data extraction (real Nanonis triple fixtures) ──────────────────
#
# Every read parses ``record.return_value[2][i]`` (the Variables list), where
# index i maps 1:1 to the method's quickSend ResponseTypes order. These tests
# feed a real ("", b"", [..]) triple and assert the *parsed values*, guarding
# against [0]/[1] misreads (empty error string / raw bytes) and against index
# drift. They call ``execute()`` directly so the returned SkillResult.data is
# inspectable.

def _exec(skill_cls, canned: dict, params: dict | None = None):
    ctx = FakeCtx(canned=canned)
    return skill_cls().execute(ctx, params or {})


def test_get_pll_status_parses_all_values():
    # ResponseTypes: CenterFreq ['d'], FreqShift ['f'], AmpGain ['f','f','f'],
    # PhasGain ['f','f'], Excitation ['f'].
    canned = {
        "PLL_CenterFreqGet": {"return_value": ("", b"", [25000.0])},
        "PLL_FreqShiftGet": {"return_value": ("", b"", [-3.5])},
        "PLL_AmpCtrlGainGet": {"return_value": ("", b"", [1.1e-3, 2.2e-3, 3.3e-3])},
        "PLL_PhasCtrlGainGet": {"return_value": ("", b"", [5.0, 6.0])},
        "PLL_ExcitationGet": {"return_value": ("", b"", [0.123])},
    }
    r = _exec(GetPLLStatus, canned)
    assert r.success
    assert r.data["center_freq_hz"] == 25000.0
    assert r.data["freq_shift_hz"] == -3.5
    assert r.data["amp_p_gain"] == pytest.approx(1.1e-3)
    assert r.data["amp_time_constant_s"] == pytest.approx(2.2e-3)
    assert r.data["amp_i_gain"] == pytest.approx(3.3e-3)
    assert r.data["phas_p_gain"] == 5.0
    assert r.data["phas_time_constant_s"] == 6.0
    assert r.data["excitation_v"] == pytest.approx(0.123)


def test_get_pll_add_on_off_parses_bool():
    r = _exec(GetPLLAddOnOff, {"PLL_AddOnOffGet": {"return_value": ("", b"", [1])}})
    assert r.data["add_on"] is True
    r = _exec(GetPLLAddOnOff, {"PLL_AddOnOffGet": {"return_value": ("", b"", [0])}})
    assert r.data["add_on"] is False


def test_get_pll_amp_ctrl_on_off_parses_bool():
    r = _exec(GetPLLAmpCtrlOnOff, {"PLL_AmpCtrlOnOffGet": {"return_value": ("", b"", [1])}})
    assert r.data["amp_ctrl_on"] is True


def test_get_pll_phas_ctrl_on_off_parses_bool():
    r = _exec(GetPLLPhasCtrlOnOff, {"PLL_PhasCtrlOnOffGet": {"return_value": ("", b"", [0])}})
    assert r.data["phase_ctrl_on"] is False


def test_get_pll_demod_filter_parses_order():
    r = _exec(GetPLLDemodFilter, {"PLL_DemodFilterGet": {"return_value": ("", b"", [4])}})
    assert r.data["filter_order"] == 4


def test_get_pll_demod_harmonic_parses():
    r = _exec(GetPLLDemodHarmonic, {"PLL_DemodHarmonicGet": {"return_value": ("", b"", [3])}})
    assert r.data["harmonic"] == 3


def test_get_pll_demod_input_parses_pair():
    # ResponseTypes ['H','H'] -> input, frequency_generator.
    r = _exec(GetPLLDemodInput, {"PLL_DemodInputGet": {"return_value": ("", b"", [7, 2])}})
    assert r.data["input"] == 7
    assert r.data["frequency_generator"] == 2


def test_get_pll_exc_range_parses_index_and_label():
    r = _exec(GetPLLExcRange, {"PLL_ExcRangeGet": {"return_value": ("", b"", [2])}})
    assert r.data["output_range_index"] == 2
    assert r.data["output_range"] == "0.1V"


def test_get_pll_freq_range_parses():
    r = _exec(GetPLLFreqRange, {"PLL_FreqRangeGet": {"return_value": ("", b"", [100.0])}})
    assert r.data["frequency_range_hz"] == 100.0


def test_get_pll_inp_calibr_parses():
    r = _exec(GetPLLInpCalibr, {"PLL_InpCalibrGet": {"return_value": ("", b"", [1.5e-9])}})
    assert r.data["calibration_m_per_v"] == pytest.approx(1.5e-9)


def test_get_pll_inp_props_parses_pair():
    r = _exec(GetPLLInpProps, {"PLL_InpPropsGet": {"return_value": ("", b"", [1, 0])}})
    assert r.data["differential_input"] is True
    assert r.data["divider_1_10"] is False


def test_get_pll_signal_anlzr_ch_parses():
    r = _exec(GetPLLSignalAnlzrCh, {"PLLSignalAnlzr_ChGet": {"return_value": ("", b"", [3])}})
    assert r.data["channel_index"] == 3


def test_get_pll_signal_anlzr_fft_props_parses():
    # ResponseTypes order: window, averaging, weighting, count.
    canned = {"PLLSignalAnlzr_FFTPropsGet": {"return_value": ("", b"", [1, 2, 1, 16])}}
    r = _exec(GetPLLSignalAnlzrFFTProps, canned)
    assert r.data["fft_window_index"] == 1
    assert r.data["fft_window"] == "Hanning"
    assert r.data["averaging_mode_index"] == 2
    assert r.data["averaging_mode"] == "RMS"
    assert r.data["weighting_mode_index"] == 1
    assert r.data["weighting_mode"] == "Exponential"
    assert r.data["count"] == 16


def test_get_pll_signal_anlzr_timebase_parses_pair():
    canned = {"PLLSignalAnlzr_TimebaseGet": {"return_value": ("", b"", [4, 1000])}}
    r = _exec(GetPLLSignalAnlzrTimebase, canned)
    assert r.data["timebase_index"] == 4
    assert r.data["update_rate"] == 1000


def test_get_pll_freq_swp_params_parses_triple():
    # ResponseTypes ['i','f','f'] -> num_points, period_s, settling_time_s.
    canned = {"PLLFreqSwp_ParamsGet": {"return_value": ("", b"", [256, 0.02, 0.1])}}
    r = _exec(GetPLLFreqSwpParams, canned)
    assert r.data["num_points"] == 256
    assert r.data["period_s"] == pytest.approx(0.02)
    assert r.data["settling_time_s"] == pytest.approx(0.1)


def test_acquire_pll_freq_sweep_parses_resonance_and_q():
    # PLLFreqSwp_Start ResponseTypes order places Resonance freq at Variables[6]
    # and Q factor at Variables[7]:
    #   [0]=names size [1]=num ch [2]=names [3]=rows [4]=cols [5]=data
    #   [6]=resonance freq (f64) [7]=Q (f64) [8]=phase [9]=amp/exc [10]=fit len
    vals = [0, 0, [], 0, 0, [], 23456.7, 18000.0, 90.0, 1.2, 0]
    canned = {
        "PLLFreqSwp_Open": {"return_value": ("", b"", [])},
        "PLLFreqSwp_ParamsSet": {"return_value": ("", b"", [])},
        "PLLFreqSwp_Start": {"return_value": ("", b"", vals)},
    }
    r = _exec(AcquirePLLFreqSweep, canned, {"num_points": 100, "period_s": 0.01})
    assert r.success
    assert r.data["resonance_freq_hz"] == pytest.approx(23456.7)
    assert r.data["q_factor"] == pytest.approx(18000.0)


def test_reads_do_not_crash_on_short_or_missing_variables():
    # Defensive: a too-short / empty Variables list must not raise, just omit.
    r = _exec(GetPLLDemodInput, {"PLL_DemodInputGet": {"return_value": ("", b"", [])}})
    assert r.success
    assert "input" not in r.data
    r = _exec(GetPLLStatus, {})  # all unmocked -> error records, still succeeds
    assert r.success


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
