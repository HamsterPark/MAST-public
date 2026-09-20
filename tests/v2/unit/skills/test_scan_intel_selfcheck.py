"""ScanIntel 的自检入口保持只读。"""

from __future__ import annotations

import pytest

from mast.core import instrument_profile, scan_policy
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.scan_intel_selfcheck import (
    REQUIRED_SKILLS,
    ScanIntelSelfCheck,
)


@pytest.fixture(autouse=True)
def _clean():
    scan_policy.set_policy(None)
    instrument_profile.set_profile({})
    yield
    scan_policy.set_policy(None)
    instrument_profile.set_profile({})


class RigCtx:
    """假仪器。``buffer_semantics`` 决定 Scan_BufferSet(ch,0,0) 是保持还是重置。"""

    def __init__(self, pixels=512, lines=512, buffer_semantics="keep"):
        self.pixels, self.lines = pixels, lines
        self.semantics = buffer_semantics
        self.calls: list[tuple[str, tuple]] = []

    def safe_call(self, method, *args, **kwargs):
        self.calls.append((method, args))
        if method == "Scan_BufferGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=["", b"", [2, [0, 14], self.pixels, self.lines]])
        if method == "Scan_BufferSet":
            px, ln = int(args[1]), int(args[2])
            if px == 0 and ln == 0:
                if self.semantics == "reset":
                    self.pixels = self.lines = 0
            else:
                self.pixels, self.lines = px, ln
            return NanonisCallRecord(method=method, args=args)
        if method == "Scan_FrameGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=["", b"", [1e-7, 2e-7, 1e-7, 1e-7, 30.0]])
        if method == "Piezo_TiltGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=["", b"", [0.25, -0.1]])
        return NanonisCallRecord(method=method, args=args)

    def check_abort(self):
        return False


def _run(ctx, **params):
    return ScanIntelSelfCheck().execute(ctx, params)


# ── 只读性 ───────────────────────────────────────────────────────────────────

def test_default_run_writes_nothing():
    """自检不能改任何东西 —— 它是拿去判断现状的。"""
    ctx = RigCtx()
    _run(ctx)
    writes = [m for m, _ in ctx.calls
              if m.endswith("Set") or m.startswith("Scan_Action")]
    assert writes == [], f"自检发出了写入调用:{writes}"


def test_it_reports_the_live_hardware_state():
    res = _run(RigCtx(pixels=256, lines=256))
    hw = res.data["hardware"]
    assert hw["scan_buffer"]["pixels"] == 256
    assert hw["scan_frame"]["angle_deg"] == 30.0
    assert hw["piezo_tilt"]["tilt_x_deg"] == 0.25


def test_hardware_read_failure_is_reported_not_swallowed():
    class DeadCtx(RigCtx):
        def safe_call(self, method, *args, **kwargs):
            self.calls.append((method, args))
            return NanonisCallRecord(method=method, args=args, error="link down")

    res = _run(DeadCtx())
    assert res.success            # 自检本身仍然完成
    assert "error" in res.data["hardware"]["scan_buffer"]


# ── 注册检查(冻结版里最要紧的一项) ─────────────────────────────────────────

def test_registry_check_finds_all_required_skills():
    res = _run(RigCtx())
    reg = res.data["registry"]
    assert reg["ok"], f"缺技能:{reg.get('missing')}"
    assert reg["total_registered"] > 100


def test_required_list_covers_the_new_layer_and_its_dependencies():
    """少列一个,自检就漏报一条断链。"""
    for name in ("ScanAt", "SetScanBuffer", "TiltProbeCircle", "TiltCalibrate",
                 "AutoTilt", "BiasSettleChange", "ExecuteScanPlan"):
        assert name in REQUIRED_SKILLS
    # 被它们当子步骤调用的既有技能也要在列 —— 断在子步骤上一样是断
    for name in ("ConfigureScan", "StartScan", "WaitScanComplete",
                 "SetBiasRamp", "SetPiezoTilt"):
        assert name in REQUIRED_SKILLS


# ── 待办清单:把「还没做的事」显式说出来 ─────────────────────────────────────

def test_a_fresh_machine_lists_every_outstanding_item():
    res = _run(RigCtx())
    todo = res.data["todo"]
    assert res.data["ready"] is False
    assert any("TiltCalibrate" in t for t in todo)
    assert any("档位表" in t for t in todo)
    assert any("probe_buffer_semantics" in t for t in todo)


def test_tilt_calibration_present_removes_that_item():
    instrument_profile.set_tilt_calibration([[-1.0, 0.0], [0.0, -1.0]], cond=1.0)
    res = _run(RigCtx())
    assert res.data["instrument_profile"]["tilt_calibrated"] is True
    assert not any("TiltCalibrate" in t for t in res.data["todo"])


def test_missing_calibration_says_why_it_matters():
    res = _run(RigCtx())
    note = res.data["instrument_profile"]["tilt_note"]
    assert "AutoTilt 一律跳过" in note
    assert "反方向加倍" in note


def test_customised_policy_removes_that_item():
    scan_policy.set_policy([
        {"name": "mine", "upper_size_m": None, "pixels": 128, "line_time_s": 3.0},
    ])
    res = _run(RigCtx())
    assert res.data["scan_policy"]["customised"] is True
    assert not any("档位表" in t for t in res.data["todo"])


def test_factory_policy_is_flagged_as_a_starting_point_not_truth():
    res = _run(RigCtx())
    assert "按本机实际情况校准" in res.data["scan_policy"]["note"]


def test_still_factory_rig_constants_are_listed():
    res = _run(RigCtx())
    still = res.data["instrument_profile"]["still_factory"]
    assert "z_range_m" in still and "v_tip_max_m_s" in still


def test_filled_rig_constants_drop_off_the_list():
    instrument_profile.set_profile({"z_range_m": 2.2e-6, "v_tip_max_m_s": 5e-7,
                                    "tilt_limit_deg": 3.0})
    res = _run(RigCtx())
    still = res.data["instrument_profile"]["still_factory"]
    assert "z_range_m" not in still and "v_tip_max_m_s" not in still


# ── 档位预览 ─────────────────────────────────────────────────────────────────

def test_preview_shows_what_each_scale_would_use():
    res = _run(RigCtx())
    pv = res.data["scan_policy"]["preview"]
    # 自检按公开档位解析尺度，不将默认档位当作仪器标定。
    assert pv["5nm"]["tier"] == "atomic_verify"
    assert pv["1000nm"]["tier"] == "survey"
    assert pv["50nm"]["pixels"] == 256      # 2026-08-09 改档:默认统一 256


# ── 0/0 语义探测 ─────────────────────────────────────────────────────────────

def test_probe_is_off_by_default():
    res = _run(RigCtx())
    assert "buffer_semantics" not in res.data


def test_probe_detects_keep_semantics():
    ctx = RigCtx(pixels=512, buffer_semantics="keep")
    res = _run(ctx, probe_buffer_semantics=True)
    bs = res.data["buffer_semantics"]
    assert bs["ok"] and bs["semantics"] == "keep"
    assert "与设计假设一致" in bs["note"]
    assert ctx.pixels == 512


def test_probe_detects_reset_semantics_and_restores():
    """如果 0/0 其实是重置,那**直接调 ConfigureScan 会清掉分辨率** —— 这是个
    一直在发生却从没人回读过的动作,必须查出来并把原值还回去。"""
    ctx = RigCtx(pixels=512, buffer_semantics="reset")
    res = _run(ctx, probe_buffer_semantics=True)
    bs = res.data["buffer_semantics"]
    assert bs["semantics"] == "reset"
    assert bs["restored"] is True
    assert "与设计假设相反" in bs["note"]
    assert ctx.pixels == 512, "探测之后没有把分辨率还原"


def test_probe_refuses_when_it_cannot_read_first():
    """读不到当前值就不写 —— 否则复原不回去。"""
    class NoReadCtx(RigCtx):
        def safe_call(self, method, *args, **kwargs):
            self.calls.append((method, args))
            if method == "Scan_BufferGet":
                return NanonisCallRecord(method=method, args=args,
                                         error="no reply")
            return super().safe_call(method, *args, **kwargs)

    ctx = NoReadCtx()
    res = _run(ctx, probe_buffer_semantics=True)
    bs = res.data["buffer_semantics"]
    assert bs["ok"] is False
    assert "未发出任何写入" in bs["error"]
    assert not any(m == "Scan_BufferSet" for m, _ in ctx.calls)


def test_probe_completing_removes_that_todo_item():
    res = _run(RigCtx(), probe_buffer_semantics=True)
    assert not any("probe_buffer_semantics" in t for t in res.data["todo"])


# ── 汇总 ─────────────────────────────────────────────────────────────────────

def test_summary_is_readable_at_a_glance():
    res = _run(RigCtx())
    assert "倾斜标定" in (res.summary or "")
    assert "待办" in (res.summary or "")


def test_a_fully_commissioned_rig_reports_ready():
    instrument_profile.set_profile({"z_range_m": 2.2e-6, "v_tip_max_m_s": 5e-7,
                                    "tilt_limit_deg": 3.0,
                                    "z_noise_floor_m": 1.5e-11})
    instrument_profile.set_tilt_calibration([[-1.0, 0.0], [0.0, -1.0]], cond=1.0)
    scan_policy.set_policy([
        {"name": "mine", "upper_size_m": None, "pixels": 128, "line_time_s": 3.0},
    ])
    res = _run(RigCtx(), probe_buffer_semantics=True)
    assert res.data["todo"] == []
    assert res.data["ready"] is True


def test_skill_is_read_category_and_auto_level():
    from mast.core.types import SafetyLevel, SkillCategory
    meta = ScanIntelSelfCheck().metadata()
    assert meta.category == SkillCategory.READ
    assert meta.safety_level == SafetyLevel.AUTO
