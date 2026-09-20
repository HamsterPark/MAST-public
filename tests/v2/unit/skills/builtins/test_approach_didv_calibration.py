"""ApproachTip dI/dV-at-contact calibration (② 进针 dI/dV 写标定).

Pins: a verified 进针 records the lock-in dI/dV at contact into the
cross-run instrument_profile calibration (bound to bias/setpoint), via BOTH
success exits (already-engaged + auto-approach). Recording is best-effort and
never blocks 进针: no lock-in configured → skipped, approach still succeeds.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_approach_didv_calibration.py -x -v
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

from mast.core import instrument_profile as ip  # noqa: E402
from mast.core.types import NanonisCallRecord, SkillResult  # noqa: E402
from mast.skills.builtins.approach import ApproachTip  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_profile():
    ip.set_persist_sink(None)
    ip.set_profile({})
    yield
    ip.set_persist_sink(None)
    ip.set_profile({})


class _Ctx:
    """Context with BOTH run() (canned SkillResults) and safe_call() (Nanonis
    reads) — ApproachTip uses run() for sub-skills and safe_call() for the
    dI/dV / bias reads during calibration."""

    def __init__(self, canned=None, *, didv=1.5e-3, bias=0.5, setpoint=1e-10):
        self.canned = canned or {}
        self.didv, self.bias, self.setpoint = didv, bias, setpoint
        self.runs: list = []
        self.safe_calls: list = []

    def run(self, name, params):
        self.runs.append((name, params))
        r = self.canned.get(name)
        return r or SkillResult(skill_name=name, success=False, error="unmocked")

    #: 调制开关。2026-08-05 起标定多了一个前置条件:**调制没开时读到的不是
    #: dI/dV**,所以标定拒绝记录(见 approach._modulation_confirmed_on)。
    #: 这些用例问的是「配置齐了就记标定」,所以默认让调制是开的 —— 把它设成
    #: False 的那一条,测的才是新加的那道闸。
    mod_on: bool = True

    def safe_call(self, method, *args, role="main"):
        self.safe_calls.append((method, args))
        served = {"Signals_ValGet": self.didv, "Bias_Get": self.bias,
                  "ZCtrl_SetpntGet": self.setpoint,
                  "LockIn_ModOnOffGet": (1 if self.mod_on else 0)}
        rv = ("", b"", [served[method]]) if method in served else ("", b"", [])
        return NanonisCallRecord(method=method, args=args, return_value=rv)


def _eng(engaged, needs_auto, setpoint_a=1e-9):
    return SkillResult(skill_name="TryEngageController", success=True,
                       data={"engaged": engaged, "needs_auto_approach": needs_auto,
                             "peak_current_a": 1e-9, "setpoint_a": setpoint_a})


def _reads(current_a=5e-10, setpoint_a=5e-10):
    return {"GetCurrent": SkillResult(skill_name="GetCurrent", success=True,
                                      data={"current_a": current_a}),
            "GetSetpoint": SkillResult(skill_name="GetSetpoint", success=True,
                                       data={"setpoint_a": setpoint_a})}


# ── helper unit ─────────────────────────────────────────────────────────────
def test_helper_records_when_lockin_configured():
    ip.set_profile({"lockin_signal_index": 8, "lockin_mod_amp_v": 0.02})
    ctx = _Ctx(didv=1.5e-3, bias=0.5)
    didv = ApproachTip()._record_didv_calibration(ctx, setpoint_a=1e-10)
    assert didv == pytest.approx(1.5e-3)
    cal = ip.get_calibration()
    assert cal["didv_at_contact_v"] == pytest.approx(1.5e-3)
    assert cal["didv_cal_bias_v"] == 0.5
    assert cal["didv_cal_setpoint_a"] == 1e-10
    assert cal["didv_cal_mod_amp_v"] == 0.02


def test_helper_skips_when_lockin_not_configured():
    ip.set_profile({})                          # no lockin_signal_index
    ctx = _Ctx()
    didv = ApproachTip()._record_didv_calibration(ctx, setpoint_a=1e-10)
    assert didv is None
    assert ip.get_calibration() == {}
    assert ctx.safe_calls == []                  # never even read the lock-in


def test_helper_no_write_without_bias():
    ip.set_profile({"lockin_signal_index": 8})
    ctx = _Ctx(didv=1.5e-3, bias=None)           # Bias_Get serves empty → None
    didv = ApproachTip()._record_didv_calibration(ctx, setpoint_a=1e-10)
    assert didv == pytest.approx(1.5e-3)         # read it...
    assert ip.get_calibration() == {}            # ...but never bound/stored


# ── integration: both success exits write the calibration ───────────────────
def test_already_engaged_exit_records_calibration():
    ip.set_profile({"lockin_signal_index": 8})
    ctx = _Ctx({"TryEngageController": _eng(True, False)}, didv=1.2e-3, bias=0.5)
    res = ApproachTip().execute(ctx, {})
    assert res.success
    assert res.data["didv_at_contact_v"] == pytest.approx(1.2e-3)
    assert ip.get_calibration()["didv_at_contact_v"] == pytest.approx(1.2e-3)


def test_auto_approach_exit_records_calibration():
    ip.set_profile({"lockin_signal_index": 8})
    ctx = _Ctx({"TryEngageController": _eng(False, True),
                "AutoApproach": SkillResult(skill_name="AutoApproach", success=True,
                                            data={"ok": True}),
                **_reads(current_a=5e-10, setpoint_a=5e-10)},
               didv=9e-4, bias=-0.3)
    res = ApproachTip().execute(ctx, {})
    assert res.success
    assert res.data["method"] == "auto_approach"
    assert res.data["didv_at_contact_v"] == pytest.approx(9e-4)
    cal = ip.get_calibration()
    assert cal["didv_at_contact_v"] == pytest.approx(9e-4)
    assert cal["didv_cal_bias_v"] == -0.3


def test_approach_still_succeeds_without_lockin():
    """Regression: no lock-in configured → calibration skipped, 进针 unaffected."""
    ip.set_profile({})
    ctx = _Ctx({"TryEngageController": _eng(True, False)})
    res = ApproachTip().execute(ctx, {})
    assert res.success
    assert res.data["didv_at_contact_v"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


def test_no_calibration_is_recorded_with_the_modulation_off(monkeypatch):
    """新加的那道闸(2026-08-05):调制关着时读到的不是 dI/dV。

    「用完即关」让进针类流程在开跑前自动关调制,于是这条路走到时调制**通常是关的**。
    不拦住,那次改动就会把一个「没有被调制驱动的通道读数」当成 dI/dV 写进持久标定库
    —— 而它之后每次被引用都不会自己声明是假的。
    """
    from mast.core import instrument_profile as ip
    from mast.skills.builtins.approach import ApproachTip

    recorded = []
    monkeypatch.setattr(ip, "get_config",
                        lambda k, d=None: 8 if k == "lockin_signal_index" else d,
                        raising=False)
    monkeypatch.setattr(ip, "set_calibration",
                        lambda *a, **k: recorded.append(a), raising=False)

    ctx = _Ctx()
    ctx.mod_on = False
    assert ApproachTip()._record_didv_calibration(ctx, setpoint_a=1e-9) is None
    assert not recorded, "调制关着时把一个非 dI/dV 的读数写进了标定库"
