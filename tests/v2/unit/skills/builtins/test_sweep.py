"""v2 unit tests for mast.skills.builtins.sweep.

Skills covered (11):
  ConfigureBiasSweep (CONFIRM), AcquireBiasSweep (CONFIRM),
  ConfigureLockInSweep (CONFIRM), AcquireLockInSweep (CONFIRM),
  GetLockInSweepLimits (AUTO), GetLockInSweepProps (AUTO),
  GetLockInSweepSignal (AUTO), GenSwpAcqChsGet (AUTO),
  GenSwpPropsGet (AUTO), GenSwpStop (CONFIRM), GenSwpSwpSignalGet (AUTO).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_sweep.py -x -v
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

import numpy as np
import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.sweep import (
    AcquireBiasSweep,
    AcquireLockInSweep,
    ConfigureBiasSweep,
    ConfigureLockInSweep,
    GenSwpAcqChsGet,
    GenSwpPropsGet,
    GenSwpStop,
    GenSwpSwpSignalGet,
    GetLockInSweepLimits,
    GetLockInSweepProps,
    GetLockInSweepSignal,
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
    (ConfigureBiasSweep, "ConfigureBiasSweep", "CONFIRM"),
    (AcquireBiasSweep, "AcquireBiasSweep", "CONFIRM"),
    (ConfigureLockInSweep, "ConfigureLockInSweep", "CONFIRM"),
    (AcquireLockInSweep, "AcquireLockInSweep", "CONFIRM"),
    (GetLockInSweepLimits, "GetLockInSweepLimits", "AUTO"),
    (GetLockInSweepProps, "GetLockInSweepProps", "AUTO"),
    (GetLockInSweepSignal, "GetLockInSweepSignal", "AUTO"),
    (GenSwpAcqChsGet, "GenSwpAcqChsGet", "AUTO"),
    (GenSwpPropsGet, "GenSwpPropsGet", "AUTO"),
    (GenSwpStop, "GenSwpStop", "CONFIRM"),
    (GenSwpSwpSignalGet, "GenSwpSwpSignalGet", "AUTO"),
])
def test_sweep_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".sweep")


def test_configure_bias_sweep_required_fields():
    tool = wrap_skill(ConfigureBiasSweep, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["lower_v"].is_required()
    assert fields["upper_v"].is_required()
    assert fields["num_steps"].is_required()
    assert not fields["period_ms"].is_required()
    assert fields["num_steps"].annotation is int


def test_configure_lockin_sweep_required_fields():
    tool = wrap_skill(ConfigureLockInSweep, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["lower_hz"].is_required()
    assert fields["upper_hz"].is_required()
    assert fields["num_steps"].is_required()
    assert not fields["integration_periods"].is_required()


# ── Execution tests ───────────────────────────────────────────────────────────

def test_configure_bias_sweep_executes():
    canned = {
        "GenSwp_Open": {"return_value": None},
        "GenSwp_SwpSignalSet": {"return_value": None},
        "Signals_MeasNamesGet": {"return_value": None},
        "GenSwp_LimitsSet": {"return_value": None},
        "GenSwp_PropsSet": {"return_value": None},
    }
    tool = wrap_skill(ConfigureBiasSweep, make_provider(canned))
    result = _invoke(tool, lower_v=-1.0, upper_v=1.0, num_steps=100)
    assert result.update["executed_skills"] == ["ConfigureBiasSweep"]


def test_configure_bias_sweep_open_fails():
    canned = {"GenSwp_Open": {"error": "sweeper busy"}}
    tool = wrap_skill(ConfigureBiasSweep, make_provider(canned))
    result = _invoke(tool, lower_v=-1.0, upper_v=1.0, num_steps=100)
    assert result.update["messages"][0].status == "error"


def test_acquire_bias_sweep_executes():
    canned = {
        "GenSwp_Open": {"return_value": None},
        "GenSwp_SwpSignalSet": {"return_value": None},
        "Signals_MeasNamesGet": {"return_value": None},
        "GenSwp_PropsSet": {"return_value": None},
        "GenSwp_Start": {"return_value": None},
    }
    tool = wrap_skill(AcquireBiasSweep, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["AcquireBiasSweep"]


def test_configure_lockin_sweep_executes():
    canned = {
        "LockInFreqSwp_Open": {"return_value": None},
        "LockInFreqSwp_LimitsSet": {"return_value": None},
        "LockInFreqSwp_PropsSet": {"return_value": None},
    }
    tool = wrap_skill(ConfigureLockInSweep, make_provider(canned))
    result = _invoke(tool, lower_hz=100.0, upper_hz=10000.0, num_steps=50)
    assert result.update["executed_skills"] == ["ConfigureLockInSweep"]


def test_acquire_lockin_sweep_executes():
    canned = {
        "LockInFreqSwp_Open": {"return_value": None},
        "LockInFreqSwp_PropsSet": {"return_value": None},
        "LockInFreqSwp_Start": {"return_value": None},
    }
    tool = wrap_skill(AcquireLockInSweep, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["AcquireLockInSweep"]


def test_get_lockin_sweep_limits_executes():
    canned = {"LockInFreqSwp_LimitsGet": {"return_value": ("", b"", [100.0, 10000.0])}}
    tool = wrap_skill(GetLockInSweepLimits, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetLockInSweepLimits"]


def test_get_lockin_sweep_props_executes():
    canned = {"LockInFreqSwp_PropsGet": {"return_value": ("", b"", [50, 3, 0.01, 3, 0.01, 1, 2])}}
    tool = wrap_skill(GetLockInSweepProps, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetLockInSweepProps"]


def test_get_lockin_sweep_signal_executes():
    canned = {"LockInFreqSwp_SignalGet": {"return_value": ("", b"", [5])}}
    tool = wrap_skill(GetLockInSweepSignal, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetLockInSweepSignal"]


def test_gen_swp_acq_chs_get_executes():
    # Realistic three-tuple: (error_string, raw_bytes, Variables) where the
    # Variables list for GenSwp.AcqChsGet (["i","*i","i","i","*+c"]) is
    # [num_channels, channel_indexes, names_size, names_count, names].
    canned = {"GenSwp_AcqChsGet": {
        "return_value": ("", b"", [1, [3], 11, 1, ["Current (A)"]]),
    }}
    tool = wrap_skill(GenSwpAcqChsGet, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GenSwpAcqChsGet"]


def test_gen_swp_props_get_executes():
    # Realistic three-tuple: Variables for GenSwp.PropsGet
    # (["f","f","i","H","I","I","f"]).
    canned = {"GenSwp_PropsGet": {
        "return_value": ("", b"", [100.0, 1e6, 200, 4, 1, 2, 10.0]),
    }}
    tool = wrap_skill(GenSwpPropsGet, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GenSwpPropsGet"]


def test_gen_swp_stop_executes():
    canned = {"GenSwp_Stop": {"return_value": None}}
    tool = wrap_skill(GenSwpStop, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GenSwpStop"]


def test_gen_swp_swp_signal_get_executes():
    # Realistic three-tuple: Variables for GenSwp.SwpSignalGet (["i","*-c"]) is
    # [name_size, channel_name].
    canned = {"GenSwp_SwpSignalGet": {
        "return_value": ("", b"", [8, "Bias (V)"]),
    }}
    tool = wrap_skill(GenSwpSwpSignalGet, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GenSwpSwpSignalGet"]


def test_configure_bias_sweep_missing_required():
    tool = wrap_skill(ConfigureBiasSweep, make_provider())
    result = _invoke(tool)  # missing lower_v, upper_v, num_steps
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "lower_v" in msg_content
        or "num_steps" in msg_content
    )


# ── Three-tuple decode regression tests ────────────────────────────────────────
# nanonis_spm methods return [error_string, raw_bytes, Variables]. The real data
# lives in Variables (return_value[2]); reading return_value[0] (empty error
# string) or [1] (raw bytes) is the bug class being fixed here. These tests feed
# realistic three-tuple fixtures and assert the DECODED values — they fail
# against the pre-fix code that indexed return_value[0]/[1] directly.


def _exec(skill_cls, method: str, return_value, params: dict | None = None):
    """Run a skill's execute() directly against a FakeCtx and return SkillResult."""
    ctx = FakeCtx(canned={method: {"return_value": return_value}})
    return skill_cls().execute(ctx, params or {})


@pytest.mark.parametrize("idxs", [
    pytest.param([3, 7], id="bare-ints"),
    pytest.param([(3,), (7,)], id="1-tuples"),
])
def test_gen_swp_acq_chs_get_decodes_variables(idxs):
    """Both decoded shapes must work.

    ``[(3,), (7,)]`` is what nanonis_spm v1.0.9 hands back for a ``*i`` array —
    ``struct.unpack``'s tuple appended without unwrapping. This stub used to be
    bare ints only, which is why the skill's ``[int(x) for x in idxs]`` looked
    fine in CI and raised ``TypeError`` on EVERY real call (KNOWN_ISSUES §2.21).
    The root cause is fixed in ``mast.core.nanonis_patch``, so **bare ints are
    now the real-machine shape** — the bare case is the production one.

    The tuple case is defence in depth for the MagicMock test path, NOT a guard
    against the patch silently lapsing: it cannot lapse silently (module-level
    assignment at import; a renamed upstream method raises ``AttributeError``
    and the process fails to start). Keep it because
    ``channel_ids_from_buffer`` is shared and already accepts both — do not add
    new tolerance elsewhere on the strength of this test.
    """
    # Variables: [num_channels, channel_indexes, names_size, names_count, names]
    rv = ("", b"\x00rawbytes", [2, idxs, 20, 2, ["Current (A)", "Bias (V)"]])
    result = _exec(GenSwpAcqChsGet, "GenSwp_AcqChsGet", rv)
    assert result.success
    assert result.data["num_channels"] == 2
    assert result.data["channel_indexes"] == [3, 7]
    assert result.data["channel_names"] == ["Current (A)", "Bias (V)"]


def test_gen_swp_props_get_decodes_variables():
    # Variables: [init_settle, max_slew, num_steps, period, autosave, save_dlg, settle]
    rv = ("", b"raw", [50.0, 1.0e6, 256, 4, 1, 2, 12.5])
    result = _exec(GenSwpPropsGet, "GenSwp_PropsGet", rv)
    assert result.success
    assert result.data["initial_settling_time_ms"] == 50.0
    assert result.data["max_slew_rate"] == 1.0e6
    assert result.data["num_steps"] == 256
    assert result.data["period_ms"] == 4.0
    assert result.data["autosave"] == 1
    assert result.data["save_dialog"] == 2
    assert result.data["settling_time_ms"] == 12.5


def test_gen_swp_swp_signal_get_decodes_variables():
    # Variables: [name_size, channel_name]. The name is the SECOND slot —
    # the pre-fix code returned str(parsed[0]) (an int) AND tried
    # int("Bias (V)") for "sweep_direction" → ValueError on real hardware.
    rv = ("", b"raw", [8, "Bias (V)"])
    result = _exec(GenSwpSwpSignalGet, "GenSwp_SwpSignalGet", rv)
    assert result.success
    assert result.data["channel_name"] == "Bias (V)"
    # No fictitious sweep_direction field anymore.
    assert "sweep_direction" not in result.data


def _start_response(rows: int, cols: int, names: list[str]):
    """Build a realistic *_Swp.Start three-tuple.

    Variables = [names_size, num_channels, names, rows, cols, data_2d] where
    data_2d is an np.ndarray of shape (rows, cols).
    """
    data = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    variables = [len("".join(names)), len(names), names, rows, cols, data]
    return ("", b"\x00rawbytes", variables)


def test_acquire_bias_sweep_decodes_2d_data():
    # 3 rows × 5 cols: row0 = bias trace, row1 = current, row2 = dI/dV.
    rv = _start_response(3, 5, ["Bias (V)", "Current (A)", "LIX (A)"])
    canned = {
        "GenSwp_Open": {"return_value": ("", b"", [])},
        "GenSwp_SwpSignalSet": {"return_value": ("", b"", [])},
        "Signals_MeasNamesGet": {"return_value": ("", b"", [])},
        "GenSwp_PropsSet": {"return_value": ("", b"", [])},
        "GenSwp_Start": {"return_value": rv},
    }
    ctx = FakeCtx(canned=canned)
    result = AcquireBiasSweep().execute(ctx, {})
    assert result.success
    # The bias trace must be the FIRST data row [0,1,2,3,4], not the header int.
    assert result.data["bias"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result.data["current"] == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert result.data["dIdV"] == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert result.data["num_points"] == 5
    assert result.data["channel_names"] == ["Bias (V)", "Current (A)", "LIX (A)"]


def test_acquire_lockin_sweep_decodes_2d_data():
    # 3 rows × 4 cols: row0 = frequency, row1 = amplitude, row2 = phase.
    rv = _start_response(3, 4, ["Frequency (Hz)", "Amplitude", "Phase"])
    canned = {
        "LockInFreqSwp_Open": {"return_value": ("", b"", [])},
        "LockInFreqSwp_PropsSet": {"return_value": ("", b"", [])},
        "LockInFreqSwp_Start": {"return_value": rv},
    }
    ctx = FakeCtx(canned=canned)
    result = AcquireLockInSweep().execute(ctx, {})
    assert result.success
    assert result.data["frequency"] == [0.0, 1.0, 2.0, 3.0]
    assert result.data["amplitude"] == [4.0, 5.0, 6.0, 7.0]
    assert result.data["phase"] == [8.0, 9.0, 10.0, 11.0]
    assert result.data["num_points"] == 4
    assert result.data["channel_names"] == ["Frequency (Hz)", "Amplitude", "Phase"]


def test_get_lockin_sweep_limits_decodes_values():
    rv = ("", b"raw", [123.0, 45600.0])
    result = _exec(GetLockInSweepLimits, "LockInFreqSwp_LimitsGet", rv)
    assert result.success
    assert result.data["lower_hz"] == 123.0
    assert result.data["upper_hz"] == 45600.0


def test_get_lockin_sweep_props_decodes_values():
    # Variables for LockInFreqSwp.PropsGet
    # (["H","H","f","H","f","I","I","i","*-c"]).
    rv = ("", b"raw", [64, 5, 0.02, 3, 0.01, 1, 2, 4, "tfunc"])
    result = _exec(GetLockInSweepProps, "LockInFreqSwp_PropsGet", rv)
    assert result.success
    assert result.data["num_steps"] == 64
    assert result.data["integration_periods"] == 5
    assert result.data["settling_periods"] == 3
    assert result.data["basename"] == "tfunc"


def test_get_lockin_sweep_signal_decodes_value():
    rv = ("", b"raw", [7])
    result = _exec(GetLockInSweepSignal, "LockInFreqSwp_SignalGet", rv)
    assert result.success
    assert result.data["sweep_signal_index"] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
