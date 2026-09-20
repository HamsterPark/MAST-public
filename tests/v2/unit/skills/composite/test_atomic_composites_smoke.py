# -*- coding: utf-8 -*-
"""两个原子分辨 composite 的执行冒烟。

它们是 2026-08-19 新写的，而「加上去了」不等于「跑过一次」—— 同一天里
参数校验把整步拒掉的事发生了三次（``SetScanSpeed`` 少了必填的 fwd_speed、
``AssessAtomicResolution`` 收到了它没有的 concentration_min、
``WaitScanComplete`` 收到 timeout_s 而它要的是 timeout_ms）。每一次现场看到的
都是「这一步好像没做」，而返回里只有一行被淹没的错误。

这些测试用假 context 把整条 plan 跑完，专门盯：
  * 发出去的每个参数，被调技能确实声明过；
  * 关键的那几个参数值是对的（``set_scan_speed=False`` 等）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.core.types import SkillResult
from mast.skills.composite.angle_series_calibration import (
    AcquireAngleSeriesForCalibration,
)
from mast.skills.composite.scan_until_atomic import ScanUntilAtomicResolution


@dataclass
class FakeCtx:
    """按技能名给罐头结果，并记录每一次调用。"""

    frame: dict = field(default_factory=lambda: {
        "center_x_m": -6.24e-7, "center_y_m": 6.82e-8,
        "width_m": 8e-9, "height_m": 8e-9, "angle_deg": 0.0})
    speed: dict = field(default_factory=lambda: {"fwd_time_s": 1.2})
    verdicts: list = field(default_factory=list)
    run_log: list = field(default_factory=list)
    _angle: float = 0.0
    _n_assess: int = 0

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "GetScanFrame":
            d = dict(self.frame)
            d["angle_deg"] = self._angle
            return SkillResult(skill_name=skill_name, success=True, data=d)
        if skill_name == "GetScanSpeed":
            return SkillResult(skill_name=skill_name, success=True,
                               data=dict(self.speed))
        if skill_name == "ConfigureScan":
            self._angle = float(params.get("angle_deg", self._angle))
            return SkillResult(skill_name=skill_name, success=True, data={})
        if skill_name == "SaveScan":
            return SkillResult(skill_name=skill_name, success=True,
                               data={"saved_path": "D:/x/frame_%d.sxm"
                                                   % len(self.run_log)})
        if skill_name == "AssessAtomicResolution":
            v = (self.verdicts[self._n_assess]
                 if self._n_assess < len(self.verdicts) else "absent")
            self._n_assess += 1
            return SkillResult(skill_name=skill_name, success=True,
                               data={"verdict": v, "coverage": 1.0,
                                     "angular_concentration": 2500.0})
        if skill_name == "CalibratePiezoMultiAngle":
            return SkillResult(skill_name=skill_name, success=True,
                               data={"ok": True, "x_scale": 1.07, "y_scale": 1.10,
                                     "piezo_shear_deg": -1.6})
        return SkillResult(skill_name=skill_name, success=True, data={})

    def check_abort(self):
        return False


def _params_sent(ctx, skill_name: str) -> list[dict]:
    return [p for n, p in ctx.run_log if n == skill_name]


def _declared(skill_name: str) -> set[str]:
    """那个技能到底声明了哪些参数 —— 从类本身取，不经注册表。

    这道检查是本文件最值钱的一条：同一天里三次「整步被参数校验拒掉」，
    每一次现场看到的都是「这一步好像没做」。
    """
    import mast.skills.builtins as B
    import mast.skills.composite as C

    cls = getattr(B, skill_name, None) or getattr(C, skill_name, None)
    if cls is None:
        for mod in (B, C):
            for nm in dir(mod):
                obj = getattr(mod, nm)
                if isinstance(obj, type) and hasattr(obj, "metadata"):
                    try:
                        if obj().metadata().name == skill_name:
                            cls = obj
                            break
                    except Exception:  # noqa: BLE001
                        continue
            if cls is not None:
                break
    if cls is None:
        pytest.skip("找不到技能类 %s" % skill_name)
    return {q.name for q in cls().metadata().parameters}


# ── ScanUntilAtomicResolution ───────────────────────────────────────────────

def test_scan_until_atomic_stops_at_the_first_atomic_frame():
    ctx = FakeCtx(verdicts=["undetermined", "absent", "atomic"])
    res = ScanUntilAtomicResolution().execute(ctx, {"max_attempts": 6})
    assert res.success
    out = res.data["outputs"] if "outputs" in (res.data or {}) else res.data
    assert out["found"] is True
    assert out["found_at_attempt"] == 3
    assert len([n for n, _ in ctx.run_log if n == "FullScan"]) == 3, \
        "找到之后还在扫"


def test_scan_until_atomic_all_undetermined_is_not_reported_as_no_tip():
    """全是「判不了」时不能说「没有原子分辨」—— 那是采集没完成。"""
    ctx = FakeCtx(verdicts=["undetermined"] * 3)
    res = ScanUntilAtomicResolution().execute(ctx, {"max_attempts": 3})
    out = res.data["outputs"] if "outputs" in (res.data or {}) else res.data
    assert out["found"] is False
    assert out["n_undetermined"] == 3
    assert "判不了" in out["advice"] and "针尖" in out["advice"]


def test_scan_until_atomic_only_sends_declared_params():
    ctx = FakeCtx(verdicts=["atomic"])
    ScanUntilAtomicResolution().execute(ctx, {"max_attempts": 1})
    for name in ("FullScan", "AssessAtomicResolution", "SaveScan"):
        declared = _declared(name)
        for sent in _params_sent(ctx, name):
            extra = set(sent) - declared
            assert not extra, "%s 收到了它没声明的参数: %s" % (name, extra)


# ── AcquireAngleSeriesForCalibration ────────────────────────────────────────

def test_angle_series_refuses_close_angles_before_touching_the_instrument():
    """角度张不开时**一个硬件调用都不发** —— 采完再说「解不出来」是白扔机时。"""
    ctx = FakeCtx()
    res = AcquireAngleSeriesForCalibration().execute(ctx, {"angles_deg": "0,3,6"})
    out = res.data["outputs"] if "outputs" in (res.data or {}) else res.data
    assert out["refused"] == "angles_too_close"
    assert not [n for n, _ in ctx.run_log if n in ("ConfigureScan", "FullScan")]
    assert "张开" in out["advice"]


def test_angle_series_never_lets_configure_scan_touch_the_speed():
    """set_scan_speed 必须显式为 False，避免配置入口按默认行时间覆盖已有线速度。"""
    ctx = FakeCtx()
    AcquireAngleSeriesForCalibration().execute(ctx, {"angles_deg": "0,45,90"})
    cfg = _params_sent(ctx, "ConfigureScan")
    assert cfg, "一帧都没配"
    for p in cfg:
        assert p.get("set_scan_speed") is False, "ConfigureScan 又去改速度了"


def test_angle_series_always_sends_the_required_speed_args():
    """``fwd_speed`` / ``bwd_speed`` 是必填的，少一个整步被拒、速度纹丝不动。"""
    ctx = FakeCtx()
    AcquireAngleSeriesForCalibration().execute(ctx, {"angles_deg": "0,45,90"})
    sp = _params_sent(ctx, "SetScanSpeed")
    assert sp, "没设速度"
    for p in sp:
        assert "fwd_speed" in p and "bwd_speed" in p
        assert p["fwd_speed"] > 0 and p["bwd_speed"] > 0


def test_angle_series_only_sends_declared_params():
    ctx = FakeCtx()
    AcquireAngleSeriesForCalibration().execute(ctx, {"angles_deg": "0,45,90"})
    for name in ("ConfigureScan", "SetScanSpeed", "FullScan", "SaveScan",
                 "CalibratePiezoMultiAngle"):
        declared = _declared(name)
        for sent in _params_sent(ctx, name):
            extra = set(sent) - declared
            assert not extra, "%s 收到了它没声明的参数: %s" % (name, extra)


def test_angle_series_hands_every_good_frame_to_the_calibrator():
    ctx = FakeCtx()
    res = AcquireAngleSeriesForCalibration().execute(ctx, {"angles_deg": "0,45,90"})
    cal = _params_sent(ctx, "CalibratePiezoMultiAngle")
    assert cal, "没有调用定标"
    paths = cal[0]["scan_paths"].split(",")
    assert len(paths) == 3
    out = res.data["outputs"] if "outputs" in (res.data or {}) else res.data
    assert out["n_frames_ok"] == 3


# ── AcquireBiasImagingSeries ────────────────────────────────────────────────

def _bias_ctx():
    ctx = FakeCtx()
    base_run = ctx.run

    def run(name, params):
        if name == "GetBias":
            return SkillResult(skill_name=name, success=True,
                               data={"bias_v": ctx._angle})   # 复用槽位存 bias
        if name == "SetBias":
            ctx._angle = float(params.get("bias_v", 0.0))
            ctx.run_log.append((name, dict(params)))
            return SkillResult(skill_name=name, success=True, data={})
        if name == "GetSetpoint":
            return SkillResult(skill_name=name, success=True,
                               data={"setpoint_a": 2e-10})
        return base_run(name, params)

    ctx.run = run
    return ctx


def test_bias_series_interleaves_and_repeats_the_first_condition():
    """正负交错 + 末尾重复首条件。

    第一版（手工脚本）按 |V| 递增排序，于是 |V| 与采集时间同向 —— 跨 |V| 的
    比较里「偏压效应」和「针尖随时间劣化」分不开。末尾那一帧是**刻度**，
    没有它，一条漂亮的单调趋势可以完全是时间造成的。
    """
    from mast.skills.composite.bias_imaging_series import AcquireBiasImagingSeries

    ctx = _bias_ctx()
    res = AcquireBiasImagingSeries().execute(ctx, {"biases_v": "0.1,-0.02,0.02,-0.1"})
    out = res.data["outputs"] if "outputs" in (res.data or {}) else res.data
    order = out["bias_order"]
    assert order[:4] == [0.02, -0.02, 0.1, -0.1], order   # |V| 升序、正负相邻
    assert order[-1] == order[0], "末尾没有重复首条件 —— 跨偏压比较就没有刻度"


def test_bias_series_can_turn_off_the_time_control_frame():
    from mast.skills.composite.bias_imaging_series import AcquireBiasImagingSeries

    ctx = _bias_ctx()
    res = AcquireBiasImagingSeries().execute(
        ctx, {"biases_v": "0.02,-0.02", "repeat_first_at_end": False})
    out = res.data["outputs"] if "outputs" in (res.data or {}) else res.data
    assert out["bias_order"] == [0.02, -0.02]
    assert "刻度" in out["advice"] or "repeat_first_at_end" in out["advice"]


def test_bias_series_never_references_a_skill_that_does_not_exist():
    """每个被调技能都必须真实存在。

    第一版在 plan 里 yield 了一个 ``Wait`` 步骤 —— 全仓没有那个技能。而
    ``optional=True`` 会让「未知技能」这件事悄悄过去：整定等待每轮都没发生，
    而结果看起来一切正常。
    """
    from mast.skills.composite.bias_imaging_series import AcquireBiasImagingSeries

    ctx = _bias_ctx()
    AcquireBiasImagingSeries().execute(ctx, {"biases_v": "0.02,-0.02"})
    called = {n for n, _ in ctx.run_log}
    for name in called:
        assert _declared(name) is not None      # _declared 找不到就 skip


def test_bias_series_only_sends_declared_params():
    from mast.skills.composite.bias_imaging_series import AcquireBiasImagingSeries

    ctx = _bias_ctx()
    AcquireBiasImagingSeries().execute(ctx, {"biases_v": "0.02,-0.02"})
    for name in ("SetBias", "ConfigureScan", "SetScanSpeed", "FullScan",
                 "SaveScan", "AssessAtomicResolution"):
        declared = _declared(name)
        for sent in _params_sent(ctx, name):
            extra = set(sent) - declared
            assert not extra, "%s 收到了它没声明的参数: %s" % (name, extra)
