"""v2 unit tests for mast.skills.builtins.spectroscopy.

Skills covered (38):
  AcquireSTS (CONFIRM), ConfigureSTS (CONFIRM), ConfigureZSpectr (CONFIRM),
  AcquireZSpectr (CONFIRM), ConfigureSTSTiming (CONFIRM),
  StopSTS (AUTO), StopZSpectr (AUTO), ConfigureSTSChannels (CONFIRM),
  ConfigureZSpectrTiming (CONFIRM), GetSTSChannels (AUTO),
  SetSTSChannels (CONFIRM), GetSTSLimits (AUTO), SetSTSAdvancedProps (CONFIRM),
  GetSTSTiming (AUTO), GetSTSAltZCtrl (AUTO), GetZSpectrChannels (AUTO),
  SetZSpectrChannels (CONFIRM), GetZSpectrRange (AUTO), SetZSpectrRange (CONFIRM),
  GetZSpectrRetract (AUTO), SetZSpectrRetract (CONFIRM),
  GetSTSDigSync (AUTO), GetSTSTTLSync (AUTO), GetSTSPulseSeqSync (AUTO),
  GetSTSZOffRevert (AUTO), GetSTSMLSLockinPerSeg (AUTO),
  SetSTSMLSMode (CONFIRM), SetSTSMLSVals (CONFIRM),
  SetSTSSafeCond1 (CONFIRM), GetSTSSafeCond1 (AUTO),
  SetSTSSafeCond2 (CONFIRM), SetZSpectrAdvProps (CONFIRM),
  GetZSpectrDigSync (AUTO), GetZSpectrPulseSeqSync (AUTO),
  GetZSpectrRetract2nd (AUTO), SetZSpectrRetractDelay (CONFIRM),
  GetZSpectrTTLSync (AUTO), GetZSpectrTiming (AUTO).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_spectroscopy.py -x -v
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
from typing import Any, get_args

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.spectroscopy import (
    AcquireSTS,
    AcquireZSpectr,
    ConfigureSTS,
    ConfigureSTSChannels,
    ConfigureSTSTiming,
    ConfigureZSpectr,
    ConfigureZSpectrTiming,
    GetSTSAltZCtrl,
    GetSTSChannels,
    GetSTSDigSync,
    GetSTSLimits,
    GetSTSMLSLockinPerSeg,
    GetSTSPulseSeqSync,
    GetSTSSafeCond1,
    GetSTSTiming,
    GetSTSTTLSync,
    GetSTSZOffRevert,
    GetZSpectrChannels,
    GetZSpectrDigSync,
    GetZSpectrPulseSeqSync,
    GetZSpectrRange,
    GetZSpectrRetract,
    GetZSpectrRetract2nd,
    GetZSpectrTiming,
    GetZSpectrTTLSync,
    SetSTSAdvancedProps,
    SetSTSChannels,
    SetSTSMLSMode,
    SetSTSMLSVals,
    SetSTSSafeCond1,
    SetSTSSafeCond2,
    SetZSpectrAdvProps,
    SetZSpectrChannels,
    SetZSpectrRange,
    SetZSpectrRetract,
    SetZSpectrRetractDelay,
    StopSTS,
    StopZSpectr,
)


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    recv_timeouts: list = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main",
                  recv_timeout_s: float | None = None) -> NanonisCallRecord:
        # ``recv_timeout_s`` 是 2026-09-09 加到真 ``ExecutionContext.safe_call``
        # 上的（长扫掠要抬 socket recv 超时，否则回包落在没人读的 socket 上、
        # 整条连接报废）。替身**必须跟上真接口**，否则一加参数这里就 TypeError；
        # 而且要**记下来** —— 只是吞掉的话，「这次到底传没传预算」就永远没人看得见。
        self.calls.append((method, args))
        self.recv_timeouts.append((method, recv_timeout_s))
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


def _triple(*variables) -> tuple:
    """Build a realistic nanonis_spm return_value triple.

    quickSend -> parseGeneralResponse returns ``[error_string, raw_bytes, Variables]``
    where Variables is the parsed-data list in ResponseTypes order. The CORRECT
    data index is therefore ``return_value[2][i]`` — never [0] (empty error
    string) or [1] (raw bytes). These fixtures reproduce that exact shape so the
    tests fail if a skill regresses to reading [0]/[1]."""
    return ("", b"\x00\x00\x00\x00", list(variables))


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("skill_cls,expected_name,expected_danger", [
    (AcquireSTS, "AcquireSTS", "CONFIRM"),
    (ConfigureSTS, "ConfigureSTS", "CONFIRM"),
    (ConfigureZSpectr, "ConfigureZSpectr", "CONFIRM"),
    (AcquireZSpectr, "AcquireZSpectr", "CONFIRM"),
    (ConfigureSTSTiming, "ConfigureSTSTiming", "CONFIRM"),
    (StopSTS, "StopSTS", "AUTO"),
    (StopZSpectr, "StopZSpectr", "AUTO"),
    (ConfigureSTSChannels, "ConfigureSTSChannels", "CONFIRM"),
    (ConfigureZSpectrTiming, "ConfigureZSpectrTiming", "CONFIRM"),
    (GetSTSChannels, "GetSTSChannels", "AUTO"),
    (SetSTSChannels, "SetSTSChannels", "CONFIRM"),
    (GetSTSLimits, "GetSTSLimits", "AUTO"),
    (SetSTSAdvancedProps, "SetSTSAdvancedProps", "CONFIRM"),
    (GetSTSTiming, "GetSTSTiming", "AUTO"),
    (GetSTSAltZCtrl, "GetSTSAltZCtrl", "AUTO"),
    (GetZSpectrChannels, "GetZSpectrChannels", "AUTO"),
    (SetZSpectrChannels, "SetZSpectrChannels", "CONFIRM"),
    (GetZSpectrRange, "GetZSpectrRange", "AUTO"),
    (SetZSpectrRange, "SetZSpectrRange", "CONFIRM"),
    (GetZSpectrRetract, "GetZSpectrRetract", "AUTO"),
    (SetZSpectrRetract, "SetZSpectrRetract", "CONFIRM"),
    (GetSTSDigSync, "GetSTSDigSync", "AUTO"),
    (GetSTSTTLSync, "GetSTSTTLSync", "AUTO"),
    (GetSTSPulseSeqSync, "GetSTSPulseSeqSync", "AUTO"),
    (GetSTSZOffRevert, "GetSTSZOffRevert", "AUTO"),
    (GetSTSMLSLockinPerSeg, "GetSTSMLSLockinPerSeg", "AUTO"),
    (SetSTSMLSMode, "SetSTSMLSMode", "CONFIRM"),
    (SetSTSMLSVals, "SetSTSMLSVals", "CONFIRM"),
    (SetSTSSafeCond1, "SetSTSSafeCond1", "CONFIRM"),
    (GetSTSSafeCond1, "GetSTSSafeCond1", "AUTO"),
    (SetSTSSafeCond2, "SetSTSSafeCond2", "CONFIRM"),
    (SetZSpectrAdvProps, "SetZSpectrAdvProps", "CONFIRM"),
    (GetZSpectrDigSync, "GetZSpectrDigSync", "AUTO"),
    (GetZSpectrPulseSeqSync, "GetZSpectrPulseSeqSync", "AUTO"),
    (GetZSpectrRetract2nd, "GetZSpectrRetract2nd", "AUTO"),
    (SetZSpectrRetractDelay, "SetZSpectrRetractDelay", "CONFIRM"),
    (GetZSpectrTTLSync, "GetZSpectrTTLSync", "AUTO"),
    (GetZSpectrTiming, "GetZSpectrTiming", "AUTO"),
])
def test_spectroscopy_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".spectroscopy")


# ── Required-field schema tests ───────────────────────────────────────────────

def test_configure_sts_required_fields():
    tool = wrap_skill(ConfigureSTS, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["start_v"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["start_v"].annotation is str
    assert fields["end_v"].is_required()
    assert fields["num_points"].is_required()
    assert fields["num_points"].annotation is int
    assert not fields["z_offset_m"].is_required()


def test_configure_z_spectr_required_fields():
    tool = wrap_skill(ConfigureZSpectr, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["z_offset_m"].is_required()
    assert fields["z_sweep_distance_m"].is_required()
    assert fields["num_points"].is_required()
    assert not fields["backward_sweep"].is_required()


def test_configure_sts_timing_required_fields():
    tool = wrap_skill(ConfigureSTSTiming, make_provider())
    fields = tool.args_schema.model_fields
    for req in ["z_avg_time_s", "init_settling_s", "max_slew_rate_v_s",
                "settling_s", "integration_s"]:
        assert fields[req].is_required(), f"{req} should be required"
    assert not fields["end_settling_s"].is_required()


def test_set_sts_advanced_props_required_fields():
    tool = wrap_skill(SetSTSAdvancedProps, make_provider())
    fields = tool.args_schema.model_fields
    for req in ["reset_bias", "z_controller_hold", "record_final_z", "lockin_run"]:
        assert fields[req].is_required()


def test_set_z_spectr_retract_required_fields():
    tool = wrap_skill(SetZSpectrRetract, make_provider())
    fields = tool.args_schema.model_fields
    for req in ["enabled", "threshold", "signal_index", "comparison"]:
        assert fields[req].is_required()


def test_set_sts_mls_mode_required_mode():
    tool = wrap_skill(SetSTSMLSMode, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["mode"].is_required()
    assert get_args(fields["mode"].annotation) == ("Linear", "MLS")


def test_set_z_spectr_retract_delay_required():
    tool = wrap_skill(SetZSpectrRetractDelay, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["retract_delay_s"].is_required()


# ── Execution tests ───────────────────────────────────────────────────────────

def test_acquire_sts_executes():
    canned = {
        "BiasSpectr_Open": {"return_value": None},
        "BiasSpectr_PropsSet": {"return_value": None},
        "BiasSpectr_Start": {"return_value": None},
    }
    tool = wrap_skill(AcquireSTS, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["AcquireSTS"]


def test_acquire_sts_error_propagates():
    canned = {
        "BiasSpectr_Open": {"return_value": None},
        "BiasSpectr_PropsSet": {"return_value": None},
        "BiasSpectr_Start": {"error": "spectroscopy hardware unavailable"},
    }
    tool = wrap_skill(AcquireSTS, make_provider(canned))
    result = _invoke(tool)
    assert result.update["messages"][0].status == "error"


def test_configure_sts_executes():
    canned = {
        "BiasSpectr_Open": {"return_value": None},
        "BiasSpectr_LimitsSet": {"return_value": None},
        "BiasSpectr_PropsSet": {"return_value": None},
    }
    tool = wrap_skill(ConfigureSTS, make_provider(canned))
    result = _invoke(tool, start_v=-1.0, end_v=1.0, num_points=512)
    assert result.update["executed_skills"] == ["ConfigureSTS"]


def test_configure_sts_open_error_propagates():
    canned = {"BiasSpectr_Open": {"error": "module not responding"}}
    tool = wrap_skill(ConfigureSTS, make_provider(canned))
    result = _invoke(tool, start_v=-1.0, end_v=1.0, num_points=512)
    assert result.update["messages"][0].status == "error"


def test_configure_sts_calls_limits_set():
    canned = {
        "BiasSpectr_Open": {"return_value": None},
        "BiasSpectr_LimitsSet": {"return_value": None},
        "BiasSpectr_PropsSet": {"return_value": None},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(ConfigureSTS, capturing_provider)
    _invoke(tool, start_v=-0.5, end_v=0.5, num_points=256)
    ctx = instances[-1]
    limits_calls = [c for c in ctx.calls if c[0] == "BiasSpectr_LimitsSet"]
    assert len(limits_calls) == 1
    assert limits_calls[0][1] == (-0.5, 0.5)


def test_configure_z_spectr_executes():
    canned = {
        "ZSpectr_Open": {"return_value": None},
        "ZSpectr_RangeSet": {"return_value": None},
        "ZSpectr_PropsSet": {"return_value": None},
    }
    tool = wrap_skill(ConfigureZSpectr, make_provider(canned))
    result = _invoke(tool, z_offset_m=0.0, z_sweep_distance_m=1e-9, num_points=100)
    assert result.update["executed_skills"] == ["ConfigureZSpectr"]


def test_acquire_z_spectr_executes():
    canned = {
        "ZSpectr_Open": {"return_value": None},
        "ZSpectr_PropsSet": {"return_value": None},
        "ZSpectr_Start": {"return_value": None},
    }
    tool = wrap_skill(AcquireZSpectr, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["AcquireZSpectr"]


def test_stop_sts_executes():
    canned = {"BiasSpectr_Stop": {"return_value": None}}
    tool = wrap_skill(StopSTS, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["StopSTS"]


def test_stop_z_spectr_executes():
    canned = {"ZSpectr_Stop": {"return_value": None}}
    tool = wrap_skill(StopZSpectr, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["StopZSpectr"]


def test_configure_sts_channels_executes():
    canned = {"BiasSpectr_ChsSet": {"return_value": None}}
    tool = wrap_skill(ConfigureSTSChannels, make_provider(canned))
    result = _invoke(tool, channel_indexes="0,1,2")
    assert result.update["executed_skills"] == ["ConfigureSTSChannels"]


def test_set_sts_channels_passes_single_list_arg():
    """Regression (2026-06-02): BiasSpectr_ChsSet(Channel_indexes: list) takes
    ONE arg — the library's "+*i" format sends the length. SetSTSChannels used
    to pass len() too → TypeError on hardware. Lock the single-list call.

    Updated 2026-06-10: channel_indexes is now declared str (comma-separated)
    because the agent schema builder has no list type; the agent path therefore
    sends a string which execute() coerces to a real int list."""
    ctx = FakeCtx(canned={"BiasSpectr_ChsSet": {"return_value": None}})
    tool = wrap_skill(SetSTSChannels, lambda: ctx)
    result = _invoke(tool, channel_indexes="0, 1, 2")
    assert result.update["executed_skills"] == ["SetSTSChannels"]
    assert ("BiasSpectr_ChsSet", ([0, 1, 2],)) in ctx.calls


def test_set_z_spectr_channels_passes_single_list_arg():
    """Same regression for ZSpectr_ChsSet (str-coerced agent path)."""
    ctx = FakeCtx(canned={"ZSpectr_ChsSet": {"return_value": None}})
    tool = wrap_skill(SetZSpectrChannels, lambda: ctx)
    result = _invoke(tool, channel_indexes="0, 1")
    assert result.update["executed_skills"] == ["SetZSpectrChannels"]
    assert ("ZSpectr_ChsSet", ([0, 1],)) in ctx.calls


def test_get_sts_limits_reads_variables():
    """Regression (2026-06-10): GetSTSLimits used to read parsed[0]/[1] (the empty
    error string + raw bytes) -> float("") ValueError on hardware. It must read
    the Variables list (parsed[2]) for BiasSpectr.LimitsGet (ResponseTypes
    ["f","f"])."""
    ctx = FakeCtx(canned={"BiasSpectr_LimitsGet": {"return_value": _triple(-1.5, 2.5)}})
    res = GetSTSLimits().execute(ctx, {})
    assert res.success is True
    assert res.data["start_v"] == -1.5
    assert res.data["end_v"] == 2.5


def test_set_sts_advanced_props_executes():
    canned = {"BiasSpectr_AdvPropsSet": {"return_value": None}}
    tool = wrap_skill(SetSTSAdvancedProps, make_provider(canned))
    result = _invoke(tool, reset_bias=1, z_controller_hold=1,
                     record_final_z=1, lockin_run=1)
    assert result.update["executed_skills"] == ["SetSTSAdvancedProps"]


def test_get_sts_timing_reads_variables():
    """GetSTSTiming must read the 8 floats from Variables (parsed[2]), not from
    parsed[i] (which is the error string / raw bytes / out of range)."""
    vals = [0.002, 1e-10, 0.003, 0.15, 0.004, 0.02, 0.005, 0.006]
    ctx = FakeCtx(canned={"BiasSpectr_TimingGet": {"return_value": _triple(*vals)}})
    res = GetSTSTiming().execute(ctx, {})
    assert res.success is True
    assert res.data["z_averaging_time_s"] == 0.002
    assert res.data["z_offset_m"] == 1e-10
    assert res.data["z_control_time_s"] == 0.006


def test_set_z_spectr_range_executes():
    canned = {"ZSpectr_RangeSet": {"return_value": None}}
    tool = wrap_skill(SetZSpectrRange, make_provider(canned))
    result = _invoke(tool, z_offset_m=0.0, z_sweep_distance_m=2e-9)
    assert result.update["executed_skills"] == ["SetZSpectrRange"]


def test_get_sts_dig_sync_reads_variables():
    """BiasSpectr.DigSyncGet ResponseTypes ["H"] -> Variables[0]. Verify the
    label maps the REAL value (2 = Pulse Sequence), not the empty error string."""
    ctx = FakeCtx(canned={"BiasSpectr_DigSyncGet": {"return_value": _triple(2)}})
    res = GetSTSDigSync().execute(ctx, {})
    assert res.success is True
    assert res.data["dig_sync"] == 2
    assert res.data["dig_sync_label"] == "Pulse Sequence"


def test_set_sts_mls_mode_executes():
    canned = {"BiasSpectr_MLSModeSet": {"return_value": None}}
    tool = wrap_skill(SetSTSMLSMode, make_provider(canned))
    result = _invoke(tool, mode="Linear")
    assert result.update["executed_skills"] == ["SetSTSMLSMode"]


def test_set_sts_safe_cond1_fails_fast():
    """Regression (2026-06-10): Bias Spectroscopy has NO safe-condition /
    auto-retract method in nanonis_spm (BiasSpectr_* ends at MLSValsGet — there
    is no BiasSpectr_SafeCond1Set). The skill used to call that non-existent
    method, getting only a cryptic "Method not found" on hardware. It must now
    fail fast WITHOUT touching the wire and point the caller at the Z-Spectroscopy
    auto-retract skills."""
    ctx = FakeCtx()
    res = SetSTSSafeCond1().execute(ctx, {"condition": 2, "threshold": 1e-9,
                                          "signal_index": 0, "comparison": 0})
    assert res.success is False
    assert "Z Spectroscopy" in res.error
    # No non-existent method was ever sent to the wire.
    assert not any(m == "BiasSpectr_SafeCond1Set" for m, _ in ctx.calls)


def test_get_sts_safe_cond1_fails_fast():
    """Same: no BiasSpectr_SafeCond1Get exists — fail fast, no wire call."""
    ctx = FakeCtx()
    res = GetSTSSafeCond1().execute(ctx, {})
    assert res.success is False
    assert "Z Spectroscopy" in res.error
    assert not any(m == "BiasSpectr_SafeCond1Get" for m, _ in ctx.calls)


def test_set_sts_safe_cond2_fails_fast():
    """Same: no BiasSpectr_SafeCond2Set exists — fail fast, no wire call."""
    ctx = FakeCtx()
    res = SetSTSSafeCond2().execute(ctx, {"condition": 1, "threshold": 1e-9,
                                          "signal_index": 0, "comparison": 0})
    assert res.success is False
    assert "Z Spectroscopy" in res.error
    assert not any(m == "BiasSpectr_SafeCond2Set" for m, _ in ctx.calls)


def test_set_z_spectr_adv_props_executes():
    canned = {"ZSpectr_AdvPropsSet": {"return_value": None}}
    tool = wrap_skill(SetZSpectrAdvProps, make_provider(canned))
    result = _invoke(tool, time_between_sweeps_s=0.1, record_final_z=1,
                     lockin_run=1, reset_z=1)
    assert result.update["executed_skills"] == ["SetZSpectrAdvProps"]


def test_get_z_spectr_timing_reads_variables():
    """ZSpectr.TimingGet has 7 floats -> read from Variables (parsed[2])."""
    vals = [0.002, 0.003, 0.12, 0.004, 0.02, 0.005, 0.006]
    ctx = FakeCtx(canned={"ZSpectr_TimingGet": {"return_value": _triple(*vals)}})
    res = GetZSpectrTiming().execute(ctx, {})
    assert res.success is True
    assert res.data["z_averaging_time_s"] == 0.002
    assert res.data["z_control_time_s"] == 0.006


def test_set_z_spectr_retract_delay_executes():
    canned = {"ZSpectr_RetractDelaySet": {"return_value": None}}
    tool = wrap_skill(SetZSpectrRetractDelay, make_provider(canned))
    result = _invoke(tool, retract_delay_s=0.05)
    assert result.update["executed_skills"] == ["SetZSpectrRetractDelay"]


def test_configure_sts_timing_executes():
    canned = {"BiasSpectr_TimingSet": {"return_value": None}}
    tool = wrap_skill(ConfigureSTSTiming, make_provider(canned))
    result = _invoke(tool, z_avg_time_s=0.001, init_settling_s=0.001,
                     max_slew_rate_v_s=0.1, settling_s=0.001, integration_s=0.01)
    assert result.update["executed_skills"] == ["ConfigureSTSTiming"]


# ── Triple-shape data-correctness tests (2026-06-10 spectroscopy fix) ──────────

def test_acquire_sts_parses_spectrum_from_variables():
    """BiasSpectr.Start Variables (["i","i","*+c","i","i","2f","i","*f"]):
    [2]=channel names, [3]=rows, [4]=cols, [5]=2D data. Acquire must read the
    spectrum out of Variables[5], not the header ints."""
    # Real Nanonis layout: rows = CHANNELS, cols = sweep POINTS (row i = channel
    # i's trace). 2 channels × 3 points. (This fixture previously encoded the
    # TRANSPOSED layout that the buggy parser assumed — 审查.)
    rows, cols = 2, 3
    data_2d = [0.0, 0.5, 1.0,       # Bias (V) row
               1e-9, 2e-9, 3e-9]    # Current (A) row
    variables = [16, 2, ["Bias (V)", "Current (A)"], rows, cols, data_2d, 0, []]
    ctx = FakeCtx(canned={
        "BiasSpectr_Open": {"return_value": _triple()},
        "BiasSpectr_PropsSet": {"return_value": _triple()},
        "BiasSpectr_Start": {"return_value": ("", b"", variables)},
    })
    res = AcquireSTS().execute(ctx, {})
    assert res.success is True
    assert res.data["num_points"] == 3
    assert res.data["voltage"] == [0.0, 0.5, 1.0]
    assert res.data["current"] == [1e-9, 2e-9, 3e-9]
    assert res.data["channel_names"] == ["Bias (V)", "Current (A)"]


def test_acquire_z_spectr_parses_spectrum_from_variables():
    """Regression (2026-06-10): AcquireZSpectr used to read Variables[0]/[1]
    (the channel-name byte size + channel count) as the z/current arrays — i.e.
    it returned the channel COUNT, never the spectrum. It must read the 2D data
    from Variables[5] (same layout as BiasSpectr.Start)."""
    # Real Nanonis layout: rows = CHANNELS, cols = sweep POINTS. 2 channels × 3.
    rows, cols = 2, 3
    data_2d = [1e-9, 2e-9, 3e-9,        # Z (m) row
               5e-12, 4e-12, 3e-12]     # Current (A) row
    variables = [16, 2, ["Z (m)", "Current (A)"], rows, cols, data_2d, 0, []]
    ctx = FakeCtx(canned={
        "ZSpectr_Open": {"return_value": _triple()},
        "ZSpectr_PropsSet": {"return_value": _triple()},
        "ZSpectr_Start": {"return_value": ("", b"", variables)},
    })
    res = AcquireZSpectr().execute(ctx, {})
    assert res.success is True
    assert res.data["num_points"] == 3
    assert res.data["z"] == [1e-9, 2e-9, 3e-9]
    assert res.data["current"] == [5e-12, 4e-12, 3e-12]
    assert res.data["channel_names"] == ["Z (m)", "Current (A)"]


@pytest.mark.parametrize("idxs", [
    pytest.param([0, 1, 4], id="bare-ints"),
    pytest.param([(0,), (1,), (4,)], id="1-tuples"),
])
def test_get_sts_channels_reads_variables(idxs):
    """Channel indices and names come from Variables positions 1 and 4. Bare integers and one-tuples both decode, while string names keep their list structure."""
    variables = [3, idxs, 30, 3, ["Current", "Bias", "LIX"]]
    ctx = FakeCtx(canned={"BiasSpectr_ChsGet": {"return_value": ("", b"", variables)}})
    res = GetSTSChannels().execute(ctx, {})
    assert res.success is True
    assert res.data["channel_indexes"] == [0, 1, 4]
    assert res.data["channel_names"] == ["Current", "Bias", "LIX"]


def test_get_sts_channels_synthetic_single_channel_payload():
    """A synthetic single-channel reply must unwrap its integer element while preserving the independent string-list decoding path."""
    variables = [1, [(7,)], 12, 1, ["Test (A)"]]
    ctx = FakeCtx(canned={"BiasSpectr_ChsGet": {"return_value": ("", b"", variables)}})
    res = GetSTSChannels().execute(ctx, {})
    assert res.success is True
    assert res.data["channel_indexes"] == [7]
    assert res.data["channel_names"] == ["Test (A)"]


@pytest.mark.parametrize("idxs", [
    pytest.param([2, 7], id="bare-ints"),
    pytest.param([(2,), (7,)], id="1-tuples"),
])
def test_get_z_spectr_channels_reads_variables(idxs):
    """Byte-for-byte twin of the BiasSpectr case above — same two shapes."""
    variables = [2, idxs, 20, 2, ["Z", "Current"]]
    ctx = FakeCtx(canned={"ZSpectr_ChsGet": {"return_value": ("", b"", variables)}})
    res = GetZSpectrChannels().execute(ctx, {})
    assert res.success is True
    assert res.data["channel_indexes"] == [2, 7]
    assert res.data["channel_names"] == ["Z", "Current"]


def test_get_sts_alt_z_ctrl_reads_variables():
    """BiasSpectr.AltZCtrlGet ["H","f","f"] -> [enabled, setpoint, settling]."""
    ctx = FakeCtx(canned={"BiasSpectr_AltZCtrlGet": {"return_value": _triple(1, 1e-11, 0.05)}})
    res = GetSTSAltZCtrl().execute(ctx, {})
    assert res.success is True
    assert res.data["enabled"] is True
    assert res.data["setpoint"] == 1e-11
    assert res.data["settling_time_s"] == 0.05


def test_get_z_spectr_retract_reads_variables():
    """ZSpectr.RetractGet ["H","f","i","H"] -> [enable, threshold, signal, comp]."""
    ctx = FakeCtx(canned={"ZSpectr_RetractGet": {"return_value": _triple(1, 2e-9, 5, 1)}})
    res = GetZSpectrRetract().execute(ctx, {})
    assert res.success is True
    assert res.data["enabled"] is True
    assert res.data["threshold"] == 2e-9
    assert res.data["signal_index"] == 5
    assert res.data["comparison"] == "<"


def test_get_z_spectr_retract2nd_uses_correct_method_and_reads_variables():
    """Regression (2026-06-10): the skill called the non-existent
    "ZSpectr_Retract2ndGet" (real name is ZSpectr_RetractSecondGet) AND read
    parsed[0..3] instead of Variables. Fix both."""
    ctx = FakeCtx(canned={"ZSpectr_RetractSecondGet": {"return_value": _triple(2, 3e-9, 1, 0)}})
    res = GetZSpectrRetract2nd().execute(ctx, {})
    assert res.success is True
    # Correct method name was used.
    assert ("ZSpectr_RetractSecondGet", ()) in ctx.calls
    assert ("ZSpectr_Retract2ndGet", ()) not in [(m, a) for m, a in ctx.calls]
    assert res.data["condition"] == 2
    assert res.data["condition_label"] == "AND"
    assert res.data["threshold"] == 3e-9
    assert res.data["signal_index"] == 1
    assert res.data["comparison"] == ">"


def test_get_sts_ttl_sync_reads_variables():
    ctx = FakeCtx(canned={"BiasSpectr_TTLSyncGet": {"return_value": _triple(2, 1, 0.01, 0.02)}})
    res = GetSTSTTLSync().execute(ctx, {})
    assert res.success is True
    assert res.data["ttl_line"] == 2
    assert res.data["ttl_polarity"] == 1
    assert res.data["time_to_on_s"] == 0.01
    assert res.data["on_duration_s"] == 0.02


def test_get_sts_pulse_seq_sync_reads_variables():
    ctx = FakeCtx(canned={"BiasSpectr_PulseSeqSyncGet": {"return_value": _triple(3, 7)}})
    res = GetSTSPulseSeqSync().execute(ctx, {})
    assert res.success is True
    assert res.data["pulse_seq_nr"] == 3
    assert res.data["nr_periods"] == 7


def test_get_sts_zoff_revert_reads_variables():
    ctx = FakeCtx(canned={"BiasSpectr_ZOffRevertGet": {"return_value": _triple(1)}})
    res = GetSTSZOffRevert().execute(ctx, {})
    assert res.success is True
    assert res.data["z_off_revert"] is True


def test_get_sts_mls_lockin_per_seg_reads_variables():
    ctx = FakeCtx(canned={"BiasSpectr_MLSLockinPerSegGet": {"return_value": _triple(1)}})
    res = GetSTSMLSLockinPerSeg().execute(ctx, {})
    assert res.success is True
    assert res.data["lockin_per_segment"] is True


def test_get_z_spectr_range_reads_variables():
    ctx = FakeCtx(canned={"ZSpectr_RangeGet": {"return_value": _triple(1e-9, 5e-9)}})
    res = GetZSpectrRange().execute(ctx, {})
    assert res.success is True
    assert res.data["z_offset_m"] == 1e-9
    assert res.data["z_sweep_distance_m"] == 5e-9


# ── Wrong-arg / wrong-method fixes ─────────────────────────────────────────────

def test_set_sts_mls_mode_sends_single_string_arg():
    """Regression (2026-06-10): BiasSpectr_MLSModeSet(Sweep_mode: str) takes ONE
    arg (the "+*c" wire format sends the string length). The skill used to pass
    len(mode), mode -> TypeError on hardware. Lock the single-arg call."""
    ctx = FakeCtx(canned={"BiasSpectr_MLSModeSet": {"return_value": _triple()}})
    res = SetSTSMLSMode().execute(ctx, {"mode": "MLS"})
    assert res.success is True
    assert ("BiasSpectr_MLSModeSet", ("MLS",)) in ctx.calls


def test_set_sts_mls_vals_coerces_string_lists():
    """The per-segment arrays are declared as str (comma-separated) for the agent
    schema; execute() must coerce them to real list[float]/list[int] and pass the
    segment count + 7 arrays to BiasSpectr_MLSValsSet."""
    ctx = FakeCtx(canned={"BiasSpectr_MLSValsSet": {"return_value": _triple()}})
    res = SetSTSMLSVals().execute(ctx, {
        "bias_start_v": "-1.0, 0.0",
        "bias_end_v": "0.0, 1.0",
        "initial_settling_s": "0.01, 0.01",
        "settling_s": "0.001, 0.001",
        "integration_s": "0.01, 0.01",
        "steps": "100, 200",
        "lockin_run": "0, 1",
    })
    assert res.success is True
    assert res.data["num_segments"] == 2
    method, args = next(c for c in ctx.calls if c[0] == "BiasSpectr_MLSValsSet")
    assert args[0] == 2                       # No_Of_Segments
    assert args[1] == [-1.0, 0.0]             # bias_start (floats)
    assert args[6] == [100, 200]              # steps (ints)
    assert args[7] == [0, 1]                  # lockin_run (ints)


def test_set_sts_mls_vals_accepts_real_lists():
    """Programmatic callers may still pass real lists (coercion is idempotent)."""
    ctx = FakeCtx(canned={"BiasSpectr_MLSValsSet": {"return_value": _triple()}})
    res = SetSTSMLSVals().execute(ctx, {
        "bias_start_v": [-1.0], "bias_end_v": [1.0],
        "initial_settling_s": [0.01], "settling_s": [0.001],
        "integration_s": [0.01], "steps": [100], "lockin_run": [1],
    })
    assert res.success is True
    assert res.data["num_segments"] == 1


def test_set_sts_channels_coerces_comma_string():
    """SetSTSChannels now declares channel_indexes as str — a comma string from
    the agent must become a real int list passed as the single ChsSet arg."""
    ctx = FakeCtx(canned={"BiasSpectr_ChsSet": {"return_value": _triple()}})
    res = SetSTSChannels().execute(ctx, {"channel_indexes": "0, 1, 4"})
    assert res.success is True
    assert ("BiasSpectr_ChsSet", ([0, 1, 4],)) in ctx.calls


# ── 「块解不开」不许长得像「测了 0 个点」（普查 A3，2026-08-15）─────────────
#
# `_reshape_spectrum` 对任何不认识的块形状回 `({}, 0)`，而 `success=True` 与
# `acquisition_complete: True` 原来是**无条件**的。于是解析失败的唯一症状是
# `data` 里**少了 `num_points` 这个键** —— 一个缺席的键，调用方没法拿它和
# 「这次真的只有 0 个点」分开。
#
# ## 为什么两个技能的结论不一样（这一段是判据，不是风格）
#
# 「这次采集跑完了」和「这条谱解得出来」**不是同一件事**，而它们在这两个技能
# 里的耦合程度不同：
#
# * **AcquireSTS** —— ConfigureSTS 会打开 autosave，每次采集都有一份 .dat 落盘
#   （`_attach_saved_dat` 的整个存在理由）。所以「内联块解不开」时数据**可能
#   仍然在盘上**。这时报 `success=False` 会把 agent 推去**重扫同一个点** ——
#   多一次针尖停留、多一次针尖变化的机会，而好数据本来就在。
#   ⇒ 解不开但拿到了 path：`success=True`，但**把解不开说出来**。
#   ⇒ 解不开**且**没有 path：内联没有、盘上也找不到 ⇒ 这次采集没有在任何地方
#      留下东西，说成成功就是「故障答成了数据」⇒ `success=False`。
# * **AcquireZSpectr** —— 它根本不调 `_attach_saved_dat`，`ZSpectr_Start(1, "")`
#   也不给 basename。**内联块是唯一的一份。** 解不开 = 什么都没有 ⇒ 一律
#   `success=False`。
#
# 这个不对称是从代码里读出来的，不是从任何一条先例照搬的。

def _unparsable_variables():
    """rows × cols 与实际数据长度对不上 —— 「少一列」的真实形状。"""
    return [16, 2, ["Bias (V)", "Current (A)"], 2, 3, [0.0, 0.5, 1.0], 0, []]


def _sts_ctx(variables):
    return FakeCtx(canned={
        "BiasSpectr_Open": {"return_value": _triple()},
        "BiasSpectr_Start": {"return_value": ("", b"", variables)},
    })


@pytest.fixture
def _no_registry_writes(monkeypatch):
    """别让测试往真的 scan_registry 里写（本仓在这上面付过五次学费）。"""
    from mast.core import scan_registry
    monkeypatch.setattr(scan_registry, "record_scan_path", lambda *a, **k: None)


def _saved_dat(monkeypatch, path):
    """让 `_attach_saved_dat` 找到 / 找不到那份 .dat。"""
    from mast.skills.builtins import scan_extra
    monkeypatch.setattr(scan_extra, "_candidate_save_dirs", lambda ctx: [])
    monkeypatch.setattr(scan_extra, "find_latest_saved",
                        lambda *a, **k: path)


def test_sts_unparsable_block_says_so_instead_of_going_quiet(
        monkeypatch, tmp_path, _no_registry_writes):
    """盘上有 .dat ⇒ 仍然算成功，但「解不开」必须是一个**说出口的事实**。"""
    dat = tmp_path / "STS001.dat"
    dat.write_bytes(b"fake")
    _saved_dat(monkeypatch, dat)

    res = AcquireSTS().execute(_sts_ctx(_unparsable_variables()), {})
    assert res.success is True, "数据在盘上，别把 agent 推去重扫"
    assert res.data["spectrum_parsed"] is False
    assert res.data["spectrum_unparsed_reason"]
    assert res.data["path"] == str(dat)
    assert "num_points" not in res.data, "解不开就不许发明一个点数"


def test_sts_unparsable_block_with_nothing_on_disk_is_a_failure(
        monkeypatch, _no_registry_writes):
    """内联没有、盘上也没有 ⇒ 这次采集没在任何地方留下东西。"""
    _saved_dat(monkeypatch, None)

    res = AcquireSTS().execute(_sts_ctx(_unparsable_variables()), {})
    assert res.success is False
    assert "解" in (res.error or "") or "parse" in (res.error or "").lower()


def test_zspectr_unparsable_block_is_always_a_failure():
    """ZSpectr 不落盘，内联块是唯一的一份。"""
    ctx = FakeCtx(canned={
        "ZSpectr_Open": {"return_value": _triple()},
        "ZSpectr_PropsSet": {"return_value": _triple()},
        "ZSpectr_Start": {"return_value": ("", b"", _unparsable_variables())},
    })
    res = AcquireZSpectr().execute(ctx, {})
    assert res.success is False
    assert res.data.get("spectrum_parsed") is False


@pytest.mark.parametrize("truncated", [
    pytest.param([1, 2, 3], id="variables-太短"),
    pytest.param([16, 2, ["Bias (V)"], 0, 0, [], 0, []], id="rows-cols-是零"),
])
def test_other_unparsable_shapes_take_the_same_road(
        monkeypatch, truncated, _no_registry_writes):
    """五个 bail-out 分支不能只有一个被接上。"""
    _saved_dat(monkeypatch, None)
    res = AcquireSTS().execute(_sts_ctx(truncated), {})
    assert res.success is False


def test_a_good_block_still_reports_parsed_true(monkeypatch,
                                                _no_registry_writes):
    """反向对照：**该放行的时候要放行**，而且新键在成功路径上也必须在。

    没有这一条，上面几条可以被一个「永远报失败」的实现满足；而一个只在失败
    路径上出现的 `spectrum_parsed`，会让按它分支的调用方在成功时读到 KeyError。
    """
    _saved_dat(monkeypatch, None)
    variables = [16, 2, ["Bias (V)", "Current (A)"], 2, 3,
                 [0.0, 0.5, 1.0, 1e-9, 2e-9, 3e-9], 0, []]
    res = AcquireSTS().execute(_sts_ctx(variables), {})
    assert res.success is True
    assert res.data["spectrum_parsed"] is True
    assert res.data["num_points"] == 3
    assert "spectrum_unparsed_reason" not in res.data


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
