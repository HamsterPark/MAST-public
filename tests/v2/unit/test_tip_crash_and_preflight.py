"""Tip-crash state machine + module preflight (⑫, field trace).

Part A — same-region crash tracker breaks the ~5-min in-place spin:
  * FullScan RECORDS a crash at its scan centre.
  * A 2nd crash at that region ESCALATES with an escape directive.
  * A 3rd scan/condition at that region is REFUSED up front (crash_guard).
  * A lateral coarse MotorMove ("换区") clears the blocks.
  * A clean scan clears a single stale crash.

Part B — module preflight surfaces a "module not running" EARLY and clearly,
but fail-open (never a new false-block).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_tip_crash_and_preflight.py -q
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.core.tip_crash_tracker import (
    TipCrashTracker,
    crash_escape_message,
    crash_guard,
    get_tip_crash_tracker,
    reset_tip_crash_tracker,
)
from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.builtins.motor import MotorMove
from mast.skills.composite._preflight import (
    PROBE_TIP_SHAPER,
    module_down_hint,
    preflight_modules,
)
from mast.skills.composite.full_scan import FullScan


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ── Part A: tracker state machine ────────────────────────────────────────

def test_block_at_threshold():
    t = TipCrashTracker(block_threshold=2, tol_m=8e-9)
    assert t.record_crash(1e-9, 2e-9) == 1
    assert not t.is_blocked(1e-9, 2e-9)
    assert t.record_crash(1e-9, 2e-9) == 2
    assert t.is_blocked(1e-9, 2e-9)


def test_nearby_positions_are_same_region():
    t = TipCrashTracker(block_threshold=2, tol_m=8e-9)
    t.record_crash(0.0, 0.0)
    t.record_crash(1e-9, 1e-9)  # within 8 nm → same cell
    assert t.is_blocked(0.0, 0.0)


def test_distinct_regions_counted_separately():
    t = TipCrashTracker(block_threshold=2, tol_m=8e-9)
    t.record_crash(0.0, 0.0)
    t.record_crash(100e-9, 100e-9)  # far away → different cell
    assert not t.is_blocked(0.0, 0.0)
    assert not t.is_blocked(100e-9, 100e-9)


def test_unknown_position_still_accumulates():
    t = TipCrashTracker(block_threshold=2)
    t.record_crash(None, None)
    t.record_crash(None, None)
    assert t.is_blocked(None, None)


def test_lateral_move_clears_all_blocks():
    t = TipCrashTracker(block_threshold=2)
    t.record_crash(0.0, 0.0)
    t.record_crash(0.0, 0.0)
    assert t.is_blocked(0.0, 0.0)
    t.note_recovery()  # coarse move → 换区
    assert not t.is_blocked(0.0, 0.0)


def test_clean_scan_clears_single_region():
    t = TipCrashTracker(block_threshold=2)
    t.record_crash(0.0, 0.0)
    t.record_crash(50e-9, 50e-9)
    t.note_recovery(0.0, 0.0)  # good scan here
    assert t.crash_count(0.0, 0.0) == 0
    assert t.crash_count(50e-9, 50e-9) == 1  # the other region untouched


def test_stale_crash_expires_via_ttl():
    clk = _Clock()
    t = TipCrashTracker(block_threshold=2, ttl_s=1800.0, clock=clk)
    t.record_crash(0.0, 0.0)
    t.record_crash(0.0, 0.0)
    assert t.is_blocked(0.0, 0.0)
    clk.t = 2000.0  # past the 1800 s TTL
    assert not t.is_blocked(0.0, 0.0)


def test_crash_guard_returns_directive_when_blocked():
    reset_tip_crash_tracker()
    tr = get_tip_crash_tracker()
    tr.record_crash(1e-9, 2e-9)
    tr.record_crash(1e-9, 2e-9)
    msg = crash_guard(object(), 1e-9, 2e-9)
    assert msg and "repeated_crash_escape_required" in msg
    # The escape route must name the GUARDED relocation skill. It used to say
    # "WithdrawTip then MotorMove", which sent the agent at the bare primitive —
    # the one that checks only the fine-Z piezo (~1 µm, and passes on unknown
    # state) and does no coarse-Z retract, no pressure check and no watching.
    # Telling a run that has just crashed twice to reach for that is how the
    # third crash happens (2026-07-31).
    assert "RelocateCoarseXY" in msg
    assert "get_coarse_map" in msg, "it must also say how to pick a direction"
    assert "不要直接调 MotorMove" in msg


def test_crash_guard_none_when_clear():
    reset_tip_crash_tracker()
    assert crash_guard(object(), 1e-9, 2e-9) is None


def test_escape_message_mentions_count():
    assert "3" in crash_escape_message(3, 0.0, 0.0)


# ── Part A: FullScan integration ─────────────────────────────────────────

@dataclass
class _ScanCtx:
    """FullScan driver: flat scan_data → crash, varying → clean."""
    scan_data: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    run_log: list = field(default_factory=list)
    state = None

    def run(self, skill_name, params):
        self.run_log.append(skill_name)
        if skill_name == "WaitScanComplete":
            return SkillResult(skill_name=skill_name, success=True,
                               data={"timed_out": False})
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method, *args, role="main"):
        if method == "Scan_FrameDataGrab":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", self.scan_data))
        return NanonisCallRecord(method=method, args=args, return_value=None)

    def emit_progress(self, *_a, **_k):
        pass

    def get_progress(self, _name):
        return None

    def checkpoint_flush(self):
        pass


_SCAN = {"center_x_m": 1e-9, "center_y_m": 2e-9,
         "width_m": 50e-9, "height_m": 50e-9}


def test_fullscan_first_crash_not_yet_blocked():
    reset_tip_crash_tracker()
    r = FullScan().execute(_ScanCtx(scan_data=[0.0, 0.0, 0.0]), dict(_SCAN))
    assert not r.success
    assert "CRASH_DETECTED" in r.error
    assert r.data.get("repeated_crash") is False
    assert get_tip_crash_tracker().crash_count(1e-9, 2e-9) == 1


def test_fullscan_second_crash_escalates():
    reset_tip_crash_tracker()
    FullScan().execute(_ScanCtx(scan_data=[0.0, 0.0, 0.0]), dict(_SCAN))
    r2 = FullScan().execute(_ScanCtx(scan_data=[0.0, 0.0, 0.0]), dict(_SCAN))
    assert not r2.success
    assert r2.data.get("repeated_crash") is True
    assert "repeated_crash_escape_required" in r2.error


def test_fullscan_third_attempt_refused_up_front():
    reset_tip_crash_tracker()
    FullScan().execute(_ScanCtx(scan_data=[0.0, 0.0, 0.0]), dict(_SCAN))
    FullScan().execute(_ScanCtx(scan_data=[0.0, 0.0, 0.0]), dict(_SCAN))
    ctx3 = _ScanCtx(scan_data=[0.0, 0.0, 0.0])
    r3 = FullScan().execute(ctx3, dict(_SCAN))
    assert not r3.success
    assert "repeated_crash_escape_required" in r3.error
    # Refused BEFORE running any scan sub-skill — this is what breaks the spin.
    assert ctx3.run_log == []


def test_lateral_move_reenables_scanning():
    reset_tip_crash_tracker()
    tr = get_tip_crash_tracker()
    tr.record_crash(1e-9, 2e-9)
    tr.record_crash(1e-9, 2e-9)
    assert tr.is_blocked(1e-9, 2e-9)
    # A successful lateral coarse move clears the block (state=None → withdrawn
    # unknown → the move is allowed).
    mv = MotorMove().execute(_ScanCtx(), {"direction": "x+", "steps": 5})
    assert mv.success
    assert not tr.is_blocked(1e-9, 2e-9)
    # And a scan at that region now runs its steps again.
    ctx = _ScanCtx(scan_data=[0.0, 1.0, 2.0, 3.0])  # varying → clean
    r = FullScan().execute(ctx, dict(_SCAN))
    assert r.success
    assert ctx.run_log  # the plan actually ran


def test_clean_scan_clears_prior_single_crash():
    reset_tip_crash_tracker()
    FullScan().execute(_ScanCtx(scan_data=[0.0, 0.0, 0.0]), dict(_SCAN))  # 1 crash
    assert get_tip_crash_tracker().crash_count(1e-9, 2e-9) == 1
    FullScan().execute(_ScanCtx(scan_data=[0.0, 1.0, 2.0, 3.0]), dict(_SCAN))  # clean
    assert get_tip_crash_tracker().crash_count(1e-9, 2e-9) == 0


# ── Part B: module preflight (fail-open) ─────────────────────────────────

class _ProbeCtx:
    def __init__(self, err_by_method=None):
        self._errs = err_by_method or {}

    def safe_call(self, method, *args, role="main"):
        return NanonisCallRecord(method=method, args=args,
                                 error=self._errs.get(method, ""))


def test_preflight_flags_module_not_running():
    ctx = _ProbeCtx({"TipShaper_PropsGet": "Tip shaper module is not running"})
    msg = preflight_modules(ctx, (PROBE_TIP_SHAPER,))
    assert msg and "module_missing" in msg and "Tip Shaper" in msg


def test_preflight_passes_when_module_answers():
    ctx = _ProbeCtx({})  # no error → module present
    assert preflight_modules(ctx, (PROBE_TIP_SHAPER,)) is None


def test_preflight_fails_open_on_comms_down():
    ctx = _ProbeCtx({"TipShaper_PropsGet": "comms_circuit_open: link down"})
    assert preflight_modules(ctx, (PROBE_TIP_SHAPER,)) is None


def test_preflight_fails_open_on_ambiguous_error():
    ctx = _ProbeCtx({"TipShaper_PropsGet": "some transient glitch 0x5"})
    assert preflight_modules(ctx, (PROBE_TIP_SHAPER,)) is None


def test_preflight_accumulates_probe_records():
    acc: list = []
    preflight_modules(_ProbeCtx({}), (PROBE_TIP_SHAPER,), accumulator=acc)
    assert len(acc) == 1


def test_module_down_hint():
    assert module_down_hint("Tip shaper not active", "Tip Shaper")
    assert module_down_hint("", "X") == ""
    assert module_down_hint("comms_circuit_open", "X") == ""
    assert module_down_hint("random error", "X") == ""


def test_tipshape_error_gets_actionable_hint():
    from mast.skills.builtins.tip_shaper import TipShape
    ctx = _ProbeCtx({"TipShaper_PropsSet": "Tip shaper module is not running"})
    # `bias_v` 要显式给：2026-08-11 起，读不到当前偏压而调用方又没给 bias_v 时，
    # TipShape 在下发任何东西之前就拒绝（见下一条测试）。那道守卫比这里要验的
    # 「模块未运行」更早，替身的 safe_call 不返回 return_value ⇒ Bias_Get 读不到，
    # 于是不给 bias_v 就永远走不到 TipShaper_PropsSet 那一步。
    r = TipShape().execute(ctx, {"bias_v": 0.02})
    assert not r.success
    assert "未运行" in r.error or "不可用" in r.error


def test_tipshape_refuses_hardcoded_3v_when_bias_unreadable():
    """当前偏压不可读且调用方未提供替代值时应拒绝执行，不能静默使用固定默认偏压。
    测试固定这一拒绝路径，防止为了省略参数而引入未知工作点的硬件写入。"""
    from mast.skills.builtins.tip_shaper import TipShape
    r = TipShape().execute(_ProbeCtx(), {})
    assert not r.success
    assert "3 V" in r.error
