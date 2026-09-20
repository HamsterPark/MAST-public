"""Regression pins for the 2026-07-03 review — spectroscopy / sweep cluster.

Covers:
  * BiasSpectr/ZSpectr.Start data is parsed as rows=CHANNELS, cols=points (was
    transposed → garbage spectra on real hardware).
  * AcquireSTS no longer zeroes the operator's Z offset (no PropsSet).
  * ConfigureSTS pins Z-controller hold + reset bias via AdvPropsSet.
  * AcquireBiasSweep turns feedback OFF during the sweep (Z-Ctrl=1) + restores
    the signal (Reset=1), and no longer wipes the settling times.
  * SetSTSMLSVals validates equal-length segment arrays + the bias bound (str
    params bypass the automatic checks).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/builtins/test_review_2026_07_03_spectroscopy.py -q
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.core.types import NanonisCallRecord
from mast.skills.builtins.spectroscopy import AcquireSTS, ConfigureSTS, SetSTSMLSVals
from mast.skills.builtins.sweep import AcquireBiasSweep


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list = field(default_factory=list)
    recv_timeouts: list = field(default_factory=list)

    def safe_call(self, method, *args, role="main",
                  recv_timeout_s=None):
        # 跟上真 safe_call 的签名（2026-09-09 加的长扫掠 recv 预算），
        # 并记下来 —— 只吞掉的话「传没传预算」永远没人看得见。
        self.calls.append((method, args))
        self.recv_timeouts.append((method, recv_timeout_s))
        entry = self.canned.get(method, {})
        return NanonisCallRecord(method=method, args=args,
                                 return_value=entry.get("return_value"),
                                 error=entry.get("error", ""))


# ── row=channel parse ────────────────────────────────────────────────────────
def test_acquire_sts_rows_are_channels():
    # 2 channels × 3 points, channel-major (row 0 = Bias, row 1 = Current).
    variables = [16, 2, ["Bias (V)", "Current (A)"], 2, 3,
                 [0.0, 0.5, 1.0, 1e-9, 2e-9, 3e-9], 0, []]
    ctx = FakeCtx(canned={"BiasSpectr_Start": {"return_value": ("", b"", variables)}})
    res = AcquireSTS().execute(ctx, {})
    assert res.success
    assert res.data["num_points"] == 3
    assert res.data["voltage"] == [0.0, 0.5, 1.0]
    assert res.data["current"] == [1e-9, 2e-9, 3e-9]


def test_acquire_sts_does_not_zero_z_offset():
    ctx = FakeCtx(canned={"BiasSpectr_Start": {"return_value": ("", b"", [])}})
    AcquireSTS().execute(ctx, {})
    # No PropsSet at all → the operator's configured Z offset is untouched.
    assert not any(m == "BiasSpectr_PropsSet" for m, _ in ctx.calls)


def test_configure_sts_sets_advprops_hold_and_reset():
    ctx = FakeCtx()
    ConfigureSTS().execute(ctx, {"start_v": -1.0, "end_v": 1.0, "num_points": 100})
    adv = [args for m, args in ctx.calls if m == "BiasSpectr_AdvPropsSet"]
    assert adv, "ConfigureSTS must set advanced props"
    # (reset_bias=1 On, z_ctrl_hold=1 On, record_final_z=0, lockin_run=0)
    assert adv[0][0] == 1 and adv[0][1] == 1


# ── sweep feedback + reset ───────────────────────────────────────────────────
def test_acquire_bias_sweep_holds_feedback_and_resets():
    ctx = FakeCtx()
    AcquireBiasSweep().execute(ctx, {})
    starts = [args for m, args in ctx.calls if m == "GenSwp_Start"]
    assert starts, "sweep must start"
    # GenSwp_Start(Get_data, Direction, Basename, Reset_signal, Z-Ctrl)
    get_data, direction, basename, reset_signal, zctrl = starts[0]
    assert reset_signal == 1, "signal must be restored after sweep"
    assert zctrl == 1, "feedback must be turned OFF during the sweep"


def test_acquire_bias_sweep_does_not_wipe_settling():
    ctx = FakeCtx()
    AcquireBiasSweep().execute(ctx, {})
    assert not any(m == "GenSwp_PropsSet" for m, _ in ctx.calls)


# ── MLS validation ───────────────────────────────────────────────────────────
def test_mls_rejects_unequal_length_arrays():
    ctx = FakeCtx()
    res = SetSTSMLSVals().execute(ctx, {
        "bias_start_v": "-1.0, 0.5",
        "bias_end_v": "1.0",            # only 1 element ≠ 2
        "initial_settling_s": "0.1, 0.1",
        "settling_s": "0.1, 0.1",
        "integration_s": "0.1, 0.1",
        "steps": "100, 100",
        "lockin_run": "0, 0",
    })
    assert not res.success
    assert "same" in res.error or "elements" in res.error
    assert not any(m == "BiasSpectr_MLSValsSet" for m, _ in ctx.calls)


def test_mls_rejects_out_of_bound_bias():
    ctx = FakeCtx()
    res = SetSTSMLSVals().execute(ctx, {
        "bias_start_v": "-50.0, 0.5",   # -50 V is outside ±10 V
        "bias_end_v": "1.0, 2.0",
        "initial_settling_s": "0.1, 0.1",
        "settling_s": "0.1, 0.1",
        "integration_s": "0.1, 0.1",
        "steps": "100, 100",
        "lockin_run": "0, 0",
    })
    assert not res.success
    assert "bias" in res.error.lower()
    assert not any(m == "BiasSpectr_MLSValsSet" for m, _ in ctx.calls)


def test_mls_accepts_valid_config():
    ctx = FakeCtx(canned={"BiasSpectr_MLSValsSet": {"return_value": ("", b"", [])}})
    res = SetSTSMLSVals().execute(ctx, {
        "bias_start_v": "-1.0, 0.5",
        "bias_end_v": "1.0, 2.0",
        "initial_settling_s": "0.1, 0.1",
        "settling_s": "0.1, 0.1",
        "integration_s": "0.1, 0.1",
        "steps": "100, 100",
        "lockin_run": "0, 0",
    })
    assert res.success
    assert res.data["num_segments"] == 2
