"""Regression pins for the 2026-07-03 whole-system review — core safety cluster.

Covers:
  * SafetyWatchdog re-arm: an UNCONFIRMED emergency retract must not latch the
    net disarmed forever (it retries after a cooldown); a CONFIRMED retract does
    latch until reset().
  * is_coarse_sample_approach: the absolute-Z human-approval gate must fire for
    truthy string / int forms of ``absolute`` (weak-model JSON), not only ``True``.
  * Global bias envelope now covers spectroscopy/sweep endpoints (start_v/end_v/…).
  * z_offset uses a symmetric RELATIVE bound (negative STS retract allowed).
  * tip-shaper plunge/lift excursions have a global cap (hallucinated ½µm plunge
    rejected).
  * InstrumentState flags a snapshot ``stale`` when the monitor link is down and
    keeps the real timestamp instead of a falsely-fresh clock.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_review_2026_07_03_safety.py -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.config import SafetyLimits
from mast.core.safety import SafetyGuard, is_coarse_sample_approach, _is_truthy_absolute
from mast.core.state import InstrumentState
from mast.core.types import HardwareState, ParameterSpec, SafetyLevel, SkillMetadata
from mast.core.watchdog import SafetyWatchdog


# ── Watchdog re-arm ─────────────────────────────────────────────────────────
class _Rec:
    def __init__(self, current):
        self.error = ""
        self.return_value = (0, 0, [current])


class _Pool:
    def __init__(self, current):
        self._c = current

    def safe_call(self, method, *args, role="main"):
        return _Rec(self._c)


def test_watchdog_rearms_when_retract_unconfirmed():
    """A failed/unconfirmed retract (callback returns False) must NOT latch —
    the net stays armed and retries after the cooldown."""
    fired = []

    def cb():
        fired.append(time.monotonic())
        return False  # retract not confirmed

    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=cb, current_threshold_a=100e-9,
                        interval_s=0.005, window_size=3, retrigger_cooldown_s=0.05)
    wd.daemon = True
    wd.start()
    time.sleep(0.45)
    wd.stop()
    wd.join(timeout=1)
    assert len(fired) >= 2, "unconfirmed retract should re-fire after cooldown"
    assert not wd.is_anomaly_triggered, "must not latch on unconfirmed retract"


def test_watchdog_latches_when_retract_confirmed():
    """A confirmed retract (callback returns True) latches until reset()."""
    fired = []

    def cb():
        fired.append(1)
        return True

    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=cb, current_threshold_a=100e-9,
                        interval_s=0.005, window_size=3, retrigger_cooldown_s=0.05)
    wd.daemon = True
    wd.start()
    time.sleep(0.3)
    wd.stop()
    wd.join(timeout=1)
    assert len(fired) == 1, "confirmed retract should latch (fire exactly once)"
    assert wd.is_anomaly_triggered
    wd.reset()
    assert not wd.is_anomaly_triggered


def test_watchdog_none_return_latches_for_backcompat():
    """A callback returning None (legacy callers) is treated as confirmed."""
    fired = []
    wd = SafetyWatchdog(_Pool(200e-9), on_anomaly=lambda: fired.append(1) or None,
                        current_threshold_a=100e-9, interval_s=0.005, window_size=3)
    wd.daemon = True
    wd.start()
    time.sleep(0.25)
    wd.stop()
    wd.join(timeout=1)
    assert len(fired) == 1
    assert wd.is_anomaly_triggered


# ── Absolute-Z gate truthiness ──────────────────────────────────────────────
def test_absolute_truthy_forms_are_gated():
    for val in ("true", "True", "1", "yes", "on", 1, True):
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"absolute": val, "target_z_m": 0.0}) is True, (
            f"absolute={val!r} with z target must gate")


def test_absolute_falsey_forms_relative_zero_not_gated():
    for val in ("false", "0", "no", "", False, 0, None):
        # relative move with 0 Z = no coarse Z motion → not gated
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"absolute": val, "target_z_m": 0.0}) is False, (
            f"absolute={val!r}, relative 0 Z must NOT gate")


def test_is_truthy_absolute_helper():
    assert _is_truthy_absolute("true") is True
    assert _is_truthy_absolute("garbage") is True   # fail-closed
    assert _is_truthy_absolute("false") is False
    assert _is_truthy_absolute(0) is False
    assert _is_truthy_absolute(None) is False


# ── Global limit coverage ───────────────────────────────────────────────────
def _meta(specs):
    return SkillMetadata(name="X", description="", safety_level=SafetyLevel.AUTO,
                         parameters=specs)


def _viol(guard, specs, params):
    return [v for v in guard.check_parameter_bounds(_meta(specs), params)
            if "global safety" in v]


def test_global_bias_limit_covers_spectroscopy_endpoints():
    g = SafetyGuard(SafetyLimits())
    for pname in ("start_v", "end_v", "lower_v", "upper_v", "bias_start_v",
                  "sts_end_v", "pulse_v", "bias_lift_v"):
        specs = [ParameterSpec(name=pname, type="float", unit="V")]
        assert _viol(g, specs, {pname: -50.0}), f"{pname}=-50 should be rejected"
        assert not _viol(g, specs, {pname: -2.0}), f"{pname}=-2 should pass"


def test_z_offset_uses_relative_bounds():
    g = SafetyGuard(SafetyLimits())
    specs = [ParameterSpec(name="z_offset_m", type="float", unit="m")]
    # negative retract within the fine-Z envelope is legitimate (was wrongly
    # rejected against the absolute z_min_m=0.0 floor).
    assert not _viol(g, specs, {"z_offset_m": -5e-8})
    # an absurd 5 µm offset is still rejected.
    assert _viol(g, specs, {"z_offset_m": -5e-6})


def test_tip_lift_global_cap():
    g = SafetyGuard(SafetyLimits())
    for pname in ("tip_lift_m", "lift_height_m", "deep_depth_m"):
        specs = [ParameterSpec(name=pname, type="float", unit="m")]
        assert _viol(g, specs, {pname: 1e-6}), f"{pname}=1µm should be rejected"
        assert not _viol(g, specs, {pname: 2e-9}), f"{pname}=2nm should pass"


# ── State staleness ─────────────────────────────────────────────────────────
class _StalefulPool:
    def __init__(self):
        self.mode = "ok"

    def safe_call(self, method, *a, role="main"):
        class R:
            error = ""
            return_value = None
        r = R()
        if self.mode == "down":
            r.error = "link down"
            return r
        table = {
            "Bias_Get": [0.5],
            "ZCtrl_StatusGet": [2],
            "Current_Get": [1e-9],
            "ZCtrl_SetpntGet": [1e-9],
            "ZCtrl_ZPosGet": [1e-7],
            "Scan_StatusGet": [0],
        }
        r.return_value = (0, 0, table.get(method, [0.0, 0.0, 0.0, 0.0, 0.0]))
        return r


def test_state_flags_stale_when_link_down():
    p = _StalefulPool()
    st = InstrumentState(p)
    s1 = st.refresh()
    assert s1.stale is False
    assert s1.bias_v == 0.5
    ts1 = s1.timestamp

    p.mode = "down"
    s2 = st.refresh()
    assert s2.stale is True, "all-failed refresh must be flagged stale"
    assert s2.bias_v == 0.5, "value carried forward"
    assert s2.timestamp == ts1, "timestamp must not advertise a fresh clock"
