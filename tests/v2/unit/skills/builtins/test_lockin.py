"""v2 unit tests for mast.skills.builtins.lockin.

Skills covered (14):
  ConfigureLockIn (CONFIRM), ConfigureLockInDemod (CONFIRM),
  GetLockInConfig (AUTO), GetDemodHPFilter (AUTO), GetDemodHarmonic (AUTO),
  GetDemodLPFilter (AUTO), GetDemodPhase (AUTO), GetDemodPhasReg (AUTO),
  SetDemodRTSignals (CONFIRM), GetDemodSignal (AUTO),
  SetDemodSyncFilter (CONFIRM), SetModHarmonic (CONFIRM),
  SetModPhasReg (CONFIRM), SetModSignal (CONFIRM).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_lockin.py -x -v
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
from mast.skills.builtins.lockin import (
    ConfigureLockIn,
    ConfigureLockInDemod,
    GetDemodHPFilter,
    GetDemodHarmonic,
    GetDemodLPFilter,
    GetDemodPhase,
    GetDemodPhasReg,
    GetDemodSignal,
    GetLockInConfig,
    SetDemodRTSignals,
    SetDemodSyncFilter,
    SetModHarmonic,
    SetModPhasReg,
    SetModSignal,
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
    (ConfigureLockIn, "ConfigureLockIn", "CONFIRM"),
    (ConfigureLockInDemod, "ConfigureLockInDemod", "CONFIRM"),
    (GetLockInConfig, "GetLockInConfig", "AUTO"),
    (GetDemodHPFilter, "GetDemodHPFilter", "AUTO"),
    (GetDemodHarmonic, "GetDemodHarmonic", "AUTO"),
    (GetDemodLPFilter, "GetDemodLPFilter", "AUTO"),
    (GetDemodPhase, "GetDemodPhase", "AUTO"),
    (GetDemodPhasReg, "GetDemodPhasReg", "AUTO"),
    (SetDemodRTSignals, "SetDemodRTSignals", "CONFIRM"),
    (GetDemodSignal, "GetDemodSignal", "AUTO"),
    (SetDemodSyncFilter, "SetDemodSyncFilter", "CONFIRM"),
    (SetModHarmonic, "SetModHarmonic", "CONFIRM"),
    (SetModPhasReg, "SetModPhasReg", "CONFIRM"),
    (SetModSignal, "SetModSignal", "CONFIRM"),
])
def test_lockin_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".lockin")


def test_configure_lockin_required_field():
    tool = wrap_skill(ConfigureLockIn, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["mod_on"].is_required()
    assert fields["mod_on"].annotation is bool
    assert not fields["amplitude_v"].is_required()
    assert not fields["frequency_hz"].is_required()


def test_set_demod_rt_signals_required_field():
    tool = wrap_skill(SetDemodRTSignals, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["rt_signals"].is_required()
    assert fields["rt_signals"].annotation is int
    assert not fields["demodulator"].is_required()


def test_set_mod_harmonic_required_field():
    tool = wrap_skill(SetModHarmonic, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["harmonic"].is_required()
    assert not fields["modulator"].is_required()


def test_set_mod_phas_reg_required_field():
    tool = wrap_skill(SetModPhasReg, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["phase_register_index"].is_required()


def test_set_mod_signal_required_field():
    tool = wrap_skill(SetModSignal, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["signal_index"].is_required()


def test_set_demod_sync_filter_required_field():
    tool = wrap_skill(SetDemodSyncFilter, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["sync_filter_on"].is_required()
    assert fields["sync_filter_on"].annotation is bool


# ── Execution tests ───────────────────────────────────────────────────────────

def test_configure_lockin_mod_off_executes():
    canned = {"LockIn_ModOnOffSet": {"return_value": None}}
    tool = wrap_skill(ConfigureLockIn, make_provider(canned))
    result = _invoke(tool, mod_on=False)
    assert result.update["executed_skills"] == ["ConfigureLockIn"]


def test_configure_lockin_mod_on_with_freq():
    """mod_on=True with frequency_hz > 0 also calls LockIn_ModPhasFreqSet."""
    canned = {
        "LockIn_ModOnOffSet": {"return_value": None},
        "LockIn_ModPhasFreqSet": {"return_value": None},
        "LockIn_ModAmpSet": {"return_value": None},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(ConfigureLockIn, capturing_provider)
    _invoke(tool, mod_on=True, frequency_hz=887.0, amplitude_v=0.05)
    ctx = instances[-1]
    freq_calls = [c for c in ctx.calls if c[0] == "LockIn_ModPhasFreqSet"]
    assert len(freq_calls) == 1
    assert freq_calls[0][1][1] == 887.0


def test_configure_lockin_demod_with_harmonic():
    canned = {"LockIn_DemodHarmonicSet": {"return_value": None}}
    tool = wrap_skill(ConfigureLockInDemod, make_provider(canned))
    result = _invoke(tool, harmonic=2)
    assert result.update["executed_skills"] == ["ConfigureLockInDemod"]


def test_configure_lockin_demod_with_lp_filter():
    canned = {"LockIn_DemodLPFilterSet": {"return_value": None}}
    tool = wrap_skill(ConfigureLockInDemod, make_provider(canned))
    result = _invoke(tool, lp_order=4, lp_cutoff_hz=100.0)
    assert result.update["executed_skills"] == ["ConfigureLockInDemod"]


def test_get_lockin_config_executes():
    canned = {
        "LockIn_ModOnOffGet": {"return_value": ("", b"", [1])},
        "LockIn_ModAmpGet": {"return_value": ("", b"", [0.05])},
        "LockIn_ModPhasFreqGet": {"return_value": ("", b"", [887.0])},
        "LockIn_ModPhasGet": {"return_value": ("", b"", [0.0])},
    }
    tool = wrap_skill(GetLockInConfig, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetLockInConfig"]


def test_get_demod_hp_filter_executes():
    canned = {"LockIn_DemodHPFilterGet": {"return_value": ("", b"", [2, 50.0])}}
    tool = wrap_skill(GetDemodHPFilter, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetDemodHPFilter"]


def test_get_demod_harmonic_executes():
    canned = {"LockIn_DemodHarmonicGet": {"return_value": ("", b"", [1])}}
    tool = wrap_skill(GetDemodHarmonic, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetDemodHarmonic"]


def test_get_demod_lp_filter_executes():
    canned = {"LockIn_DemodLPFilterGet": {"return_value": ("", b"", [3, 100.0])}}
    tool = wrap_skill(GetDemodLPFilter, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetDemodLPFilter"]


def test_get_demod_phase_executes():
    canned = {"LockIn_DemodPhasGet": {"return_value": ("", b"", [90.0])}}
    tool = wrap_skill(GetDemodPhase, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetDemodPhase"]


def test_get_demod_phas_reg_executes():
    canned = {"LockIn_DemodPhasRegGet": {"return_value": ("", b"", [1])}}
    tool = wrap_skill(GetDemodPhasReg, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetDemodPhasReg"]


def test_set_demod_rt_signals_executes():
    canned = {"LockIn_DemodRTSignalsSet": {"return_value": None}}
    tool = wrap_skill(SetDemodRTSignals, make_provider(canned))
    result = _invoke(tool, rt_signals=0)
    assert result.update["executed_skills"] == ["SetDemodRTSignals"]


def test_get_demod_signal_executes():
    canned = {"LockIn_DemodSignalGet": {"return_value": ("", b"", [14])}}
    tool = wrap_skill(GetDemodSignal, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetDemodSignal"]


def test_set_demod_sync_filter_executes():
    canned = {"LockIn_DemodSyncFilterSet": {"return_value": None}}
    tool = wrap_skill(SetDemodSyncFilter, make_provider(canned))
    result = _invoke(tool, sync_filter_on=True)
    assert result.update["executed_skills"] == ["SetDemodSyncFilter"]


def test_set_mod_harmonic_executes():
    canned = {"LockIn_ModHarmonicSet": {"return_value": None}}
    tool = wrap_skill(SetModHarmonic, make_provider(canned))
    result = _invoke(tool, harmonic=2)
    assert result.update["executed_skills"] == ["SetModHarmonic"]


def test_set_mod_phas_reg_executes():
    canned = {"LockIn_ModPhasRegSet": {"return_value": None}}
    tool = wrap_skill(SetModPhasReg, make_provider(canned))
    result = _invoke(tool, phase_register_index=1)
    assert result.update["executed_skills"] == ["SetModPhasReg"]


def test_set_mod_signal_executes():
    canned = {"LockIn_ModSignalSet": {"return_value": None}}
    tool = wrap_skill(SetModSignal, make_provider(canned))
    result = _invoke(tool, signal_index=14)
    assert result.update["executed_skills"] == ["SetModSignal"]


def test_configure_lockin_error_propagates():
    canned = {"LockIn_ModOnOffSet": {"error": "modulator fault"}}
    tool = wrap_skill(ConfigureLockIn, make_provider(canned))
    result = _invoke(tool, mod_on=True)
    assert result.update["messages"][0].status == "error"


def test_configure_lockin_missing_required():
    tool = wrap_skill(ConfigureLockIn, make_provider())
    result = _invoke(tool)  # missing mod_on
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "mod_on" in msg_content
    )


# ── Agent-path materialization (pydantic None) regression ─────────────────────
#
# On the real agent path, langgraph validates the tool call against the
# generated args_schema and materializes EVERY optional field — fields without
# an explicit ParameterSpec.default arrive as ``None``. ``"key" in params`` was
# therefore always True for ConfigureLockInDemod, so the skill would call the
# hardware setters with ``None`` (struct.pack failure / corrupt write on real
# Nanonis). These tests mirror that path by materializing the schema first.

def _agent_params(tool, **supplied) -> dict:
    """Mirror the langgraph agent path: validate supplied kwargs against the
    tool's pydantic args_schema, then ``model_dump`` so unset OPTIONAL fields
    are materialized (those without a ParameterSpec.default become ``None``).

    Then apply the adapter's own SI coercion, because ``_run`` does that before
    the skill sees anything and this helper exists to be a faithful stand-in for
    ``_run``. Dimensioned params are strings in the schema since 2026-08-04; a
    helper that stopped after model_dump would hand the skill ``"0.05"`` where
    the real path hands it ``0.05``, and would then be testing a path nobody
    runs.
    """
    from mast.agents._shared.skill_adapter import _coerce_si_params

    inst = tool.args_schema(**supplied)
    params = inst.model_dump()
    params.pop("tool_call_id", None)
    params, errors = _coerce_si_params(tool.metadata["skill_metadata"], params)
    assert not errors, errors
    return params


def test_configure_lockin_demod_schema_materializes_none():
    """Sanity: the agent path really does inject None for unset optional fields
    that have no explicit default (this is the condition that triggered the bug)."""
    tool = wrap_skill(ConfigureLockInDemod, make_provider())
    params = _agent_params(tool, harmonic=2)
    assert params["harmonic"] == 2
    assert params["signal_index"] is None
    assert params["lp_order"] is None
    assert params["lp_cutoff_hz"] is None
    assert params["hp_order"] is None
    assert params["hp_cutoff_hz"] is None
    assert params["phase_deg"] is None


def test_configure_lockin_demod_agent_path_only_supplied_setter_called():
    """Agent path: supplying only harmonic must NOT call the other 4 setters
    with None. Pre-fix this called every LockIn_Demod*Set with a None arg."""
    canned = {"LockIn_DemodHarmonicSet": {"return_value": ("", b"", [])}}
    ctx = FakeCtx(canned=canned)
    tool = wrap_skill(ConfigureLockInDemod, make_provider())
    params = _agent_params(tool, harmonic=2)

    result = ConfigureLockInDemod().execute(ctx, params)
    assert result.success

    methods = [m for (m, _a) in ctx.calls]
    assert methods == ["LockIn_DemodHarmonicSet"]
    # and the harmonic value (not None) was forwarded
    assert ctx.calls[0][1] == (1, 2)
    # None must never have reached any setter
    for _m, args in ctx.calls:
        assert None not in args


def test_configure_lockin_demod_agent_path_no_settings_calls_nothing():
    """Agent path with only demodulator supplied: every sub-setting is None →
    no hardware setter should fire at all."""
    ctx = FakeCtx(canned={})
    tool = wrap_skill(ConfigureLockInDemod, make_provider())
    params = _agent_params(tool, demodulator=2)

    result = ConfigureLockInDemod().execute(ctx, params)
    assert result.success
    assert ctx.calls == []  # nothing called → no None args sent to hardware


def test_configure_lockin_demod_agent_path_lp_filter_partial():
    """Agent path: supplying only lp_cutoff_hz fires the LP setter with the
    -1 / supplied-cutoff convention and never sends None."""
    canned = {"LockIn_DemodLPFilterSet": {"return_value": ("", b"", [])}}
    ctx = FakeCtx(canned=canned)
    tool = wrap_skill(ConfigureLockInDemod, make_provider())
    params = _agent_params(tool, lp_cutoff_hz="100")

    result = ConfigureLockInDemod().execute(ctx, params)
    assert result.success
    assert [m for (m, _a) in ctx.calls] == ["LockIn_DemodLPFilterSet"]
    # order defaults to -1 (no change), cutoff is the supplied value
    assert ctx.calls[0][1] == (1, -1, 100.0)
    assert None not in ctx.calls[0][1]


def test_configure_lockin_demod_zero_phase_is_honored():
    """phase_deg=0.0 is a real, valid value (not 'unset') and must be applied."""
    canned = {"LockIn_DemodPhasSet": {"return_value": ("", b"", [])}}
    ctx = FakeCtx(canned=canned)
    tool = wrap_skill(ConfigureLockInDemod, make_provider())
    params = _agent_params(tool, phase_deg="0")

    result = ConfigureLockInDemod().execute(ctx, params)
    assert result.success
    assert [m for (m, _a) in ctx.calls] == ["LockIn_DemodPhasSet"]
    assert ctx.calls[0][1] == (1, 0.0)


def test_configure_lockin_demod_signal_index_zero_is_honored():
    """signal_index=0 is a real value (channel 0) and must be applied."""
    canned = {"LockIn_DemodSignalSet": {"return_value": ("", b"", [])}}
    ctx = FakeCtx(canned=canned)
    tool = wrap_skill(ConfigureLockInDemod, make_provider())
    params = _agent_params(tool, signal_index=0)

    result = ConfigureLockInDemod().execute(ctx, params)
    assert result.success
    assert [m for (m, _a) in ctx.calls] == ["LockIn_DemodSignalSet"]
    assert ctx.calls[0][1] == (1, 0)


# ── Triple-tuple read correctness (index vs Nanonis ResponseTypes) ────────────
#
# nanonis_spm returns [error_str, raw_bytes, Variables]; the real data lives in
# [2][i] ordered by the method's ResponseTypes. These tests feed realistic
# triple-tuple fixtures and assert the skill extracts the right value from the
# right index (NOT [0] = error string, NOT [1] = raw bytes).

def test_get_demod_signal_reads_index_2_0():
    # LockIn.DemodSignalGet ResponseTypes ["i"] -> Variables[0]
    canned = {"LockIn_DemodSignalGet": {"return_value": ("", b"\x00", [14])}}
    res = GetDemodSignal().execute(FakeCtx(canned=canned), {"demodulator": 1})
    assert res.success
    assert res.data["signal_index"] == 14


def test_get_demod_harmonic_reads_index_2_0():
    # LockIn.DemodHarmonicGet ResponseTypes ["i"] -> Variables[0]
    canned = {"LockIn_DemodHarmonicGet": {"return_value": ("", b"\x00", [3])}}
    res = GetDemodHarmonic().execute(FakeCtx(canned=canned), {"demodulator": 1})
    assert res.success
    assert res.data["harmonic"] == 3


def test_get_demod_phase_reads_index_2_0():
    # LockIn.DemodPhasGet ResponseTypes ["f"] -> Variables[0]
    canned = {"LockIn_DemodPhasGet": {"return_value": ("", b"\x00", [42.5])}}
    res = GetDemodPhase().execute(FakeCtx(canned=canned), {"demodulator": 1})
    assert res.success
    assert res.data["phase_deg"] == 42.5


def test_get_demod_phas_reg_reads_index_2_0():
    # LockIn.DemodPhasRegGet ResponseTypes ["i"] -> Variables[0]
    canned = {"LockIn_DemodPhasRegGet": {"return_value": ("", b"\x00", [5])}}
    res = GetDemodPhasReg().execute(FakeCtx(canned=canned), {"demodulator": 1})
    assert res.success
    assert res.data["phase_register_index"] == 5


def test_get_demod_hp_filter_reads_order_and_cutoff():
    # LockIn.DemodHPFilterGet ResponseTypes ["i","f"] -> Variables[0], Variables[1]
    canned = {"LockIn_DemodHPFilterGet": {"return_value": ("", b"\x00", [2, 50.0])}}
    res = GetDemodHPFilter().execute(FakeCtx(canned=canned), {"demodulator": 1})
    assert res.success
    assert res.data["hp_filter_order"] == 2
    assert res.data["hp_cutoff_hz"] == 50.0


def test_get_demod_lp_filter_reads_order_and_cutoff():
    # LockIn.DemodLPFilterGet ResponseTypes ["i","f"] -> Variables[0], Variables[1]
    canned = {"LockIn_DemodLPFilterGet": {"return_value": ("", b"\x00", [4, 100.0])}}
    res = GetDemodLPFilter().execute(FakeCtx(canned=canned), {"demodulator": 1})
    assert res.success
    assert res.data["lp_filter_order"] == 4
    assert res.data["lp_cutoff_hz"] == 100.0


def test_get_lockin_config_reads_all_mod_values():
    # ModOnOffGet ["I"], ModAmpGet ["f"], ModPhasFreqGet ["d"], ModPhasGet ["f"]
    canned = {
        "LockIn_ModOnOffGet": {"return_value": ("", b"\x00", [1])},
        "LockIn_ModAmpGet": {"return_value": ("", b"\x00", [0.05])},
        "LockIn_ModPhasFreqGet": {"return_value": ("", b"\x00", [887.0])},
        "LockIn_ModPhasGet": {"return_value": ("", b"\x00", [12.0])},
    }
    res = GetLockInConfig().execute(FakeCtx(canned=canned), {"modulator": 1})
    assert res.success
    assert res.data["mod_on"] is True
    assert res.data["amplitude"] == 0.05
    assert res.data["frequency_hz"] == 887.0
    assert res.data["phase_deg"] == 12.0


def test_get_demod_reads_do_not_use_error_or_raw_slots():
    """Guard against the [0]/[1] mis-read: a NON-empty error string in slot [0]
    is surfaced as an error (not parsed as data), and a populated raw-bytes
    slot [1] is never confused for the value."""
    # Non-empty error string -> safe_call sets record.error -> skill fails,
    # and we never try float('')/float(b'...').
    canned = {"LockIn_DemodPhasGet": {
        "return_value": ("Module not available", b"\xde\xad", [90.0]),
        "error": "Module not available",
    }}
    res = GetDemodPhase().execute(FakeCtx(canned=canned), {"demodulator": 1})
    assert not res.success
    assert "Module not available" in res.error


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
