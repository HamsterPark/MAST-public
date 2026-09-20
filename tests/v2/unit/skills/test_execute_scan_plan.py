"""ExecuteScanPlan —— 执行多图计划 + 帧间守卫。

守卫正是「批量扫图」与「连着扫 N 张」的全部区别。所以测试的重心在:
针尖事件中止整批、质量差重扫一次、坏帧率提前中止、部分成功的缺口不被吞掉。
"""

from __future__ import annotations

import json

import pytest

from mast.core.types import SkillResult
from mast.skills.composite.execute_scan_plan import (
    BAD_FRAME_RATE_ABORT,
    MAX_RESCANS_PER_FRAME,
    MIN_BAD_FOR_RATE,
    MIN_FRAMES_FOR_RATE,
    ExecuteScanPlan,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("mast.skills.composite.execute_scan_plan.time.sleep",
                        lambda *_a, **_k: None)


def _plan(n=3, **frame_overrides):
    return json.dumps({
        "kind": "survey",
        "n_frames": n,
        "frames": [
            {"index": i, "center_x_m": i * 1e-7, "center_y_m": 0.0,
             "size_m": 1e-7, "label": f"f{i}", **frame_overrides}
            for i in range(n)
        ],
    })


class PlanCtx:
    """按脚本回答 ScanAt 的假上下文。

    ``scan_outcomes`` 是逐次 ScanAt 调用的结果描述:
      "ok" / "bad" / "tip" / "crash"
    """

    def __init__(self, scan_outcomes=None, aborted=False, bias_fail=False):
        self.runs: list[tuple[str, dict]] = []
        self._outcomes = list(scan_outcomes or [])
        self._n = 0
        self.aborted = aborted
        self.bias_fail = bias_fail

    def run(self, skill_name, params, version=None):
        self.runs.append((skill_name, dict(params)))
        if skill_name == "BiasSettleChange":
            if self.bias_fail:
                return SkillResult(skill_name=skill_name, success=False,
                                   error="穿零被拒")
            return SkillResult(skill_name=skill_name, success=True, data={})
        if skill_name != "ScanAt":
            return SkillResult(skill_name=skill_name, success=True, data={})

        outcome = (self._outcomes[self._n] if self._n < len(self._outcomes)
                   else "ok")
        self._n += 1
        if outcome == "tip":
            return SkillResult(skill_name=skill_name, success=True,
                               data={"tip_change_critical": True})
        if outcome == "crash":
            return SkillResult(skill_name=skill_name, success=True,
                               data={"crash_indicator": True})
        if outcome == "bad":
            return SkillResult(skill_name=skill_name, success=False,
                               error="质量不合格", data={})
        return SkillResult(skill_name=skill_name, success=True,
                           data={"saved_path": f"/scans/{self._n}.sxm"})

    def safe_call(self, method, *args, **kwargs):
        from mast.core.types import NanonisCallRecord
        return NanonisCallRecord(method=method, args=args)

    def check_abort(self):
        return self.aborted


def _run(ctx, plan_json, **params):
    params.setdefault("plan_json", plan_json)
    return ExecuteScanPlan().execute(ctx, params)


def _scan_calls(ctx):
    return [p for name, p in ctx.runs if name == "ScanAt"]


# ── 正常路径 ─────────────────────────────────────────────────────────────────

def test_every_planned_frame_is_acquired():
    ctx = PlanCtx()
    res = _run(ctx, _plan(3))
    assert res.success
    assert res.data["done"] == 3 and res.data["bad"] == 0
    assert len(_scan_calls(ctx)) == 3


def test_frame_geometry_reaches_scan_at():
    ctx = PlanCtx()
    _run(ctx, _plan(2))
    calls = _scan_calls(ctx)
    assert calls[0]["center_x_m"] == 0.0
    assert calls[1]["center_x_m"] == pytest.approx(1e-7)
    assert all(c["size_m"] == 1e-7 for c in calls)


def test_per_frame_parameters_are_passed_through():
    plan = json.dumps({"frames": [
        {"index": 0, "center_x_m": 0.0, "center_y_m": 0.0, "size_m": 1e-7,
         "line_time_s": 3.0, "pixels": 1024, "setpoint_a": 5e-11},
    ]})
    ctx = PlanCtx()
    _run(ctx, plan)
    call = _scan_calls(ctx)[0]
    assert call["line_time_s"] == 3.0
    assert call["pixels"] == 1024
    assert call["setpoint_a"] == 5e-11


def test_null_parameters_are_not_forwarded():
    """None 是「保持现值」,不是「设成 0」—— 传下去就把 setpoint 设成 0 了。"""
    plan = json.dumps({"frames": [
        {"index": 0, "center_x_m": 0.0, "center_y_m": 0.0, "size_m": 1e-7,
         "bias_v": None, "setpoint_a": None, "pixels": None},
    ]})
    ctx = PlanCtx()
    _run(ctx, plan)
    call = _scan_calls(ctx)[0]
    assert "bias_v" not in call
    assert "setpoint_a" not in call
    assert "pixels" not in call


def test_scanned_paths_only_lists_successful_frames():
    """只发布成功完成的扫描路径，不能让失败步骤增加结果帧数。"""
    ctx = PlanCtx(scan_outcomes=["ok", "bad", "bad", "ok"])
    res = _run(ctx, _plan(3))
    assert len(res.data["scanned_paths"]) == res.data["done"]


# ── 偏压走安全通道 ───────────────────────────────────────────────────────────

def test_bias_change_goes_through_the_safe_channel():
    """系列里跨零的那一步直接 SetBias 会把针尖推向表面。"""
    ctx = PlanCtx()
    _run(ctx, _plan(1, bias_v=-1.5))
    names = [n for n, _ in ctx.runs]
    assert names[0] == "BiasSettleChange"
    assert ctx.runs[0][1] == {"bias_v": -1.5}
    # 已经设好了就不该再让 ScanAt 设一遍
    assert "bias_v" not in _scan_calls(ctx)[0]


def test_a_refused_bias_change_marks_the_frame_bad_without_scanning():
    ctx = PlanCtx(bias_fail=True)
    res = _run(ctx, _plan(1, bias_v=0.01))
    assert res.data["done"] == 0 and res.data["bad"] == 1
    assert _scan_calls(ctx) == []
    assert "偏压变更被拒" in res.data["frames_bad"][0]["reason"]


# ── 守卫:针尖事件中止整批 ───────────────────────────────────────────────────

def test_a_tip_event_aborts_the_whole_batch():
    """针尖坏了之后每一帧都是废的 —— 继续扫只是浪费机时和针尖寿命。"""
    ctx = PlanCtx(scan_outcomes=["ok", "tip", "ok", "ok"])
    res = _run(ctx, _plan(4))
    assert not res.success
    assert res.data["aborted_at"] == 1
    assert "针尖事件" in res.data["abort_reason"]
    assert len(_scan_calls(ctx)) == 2, "中止后还在继续扫"


def test_a_crash_aborts_the_whole_batch():
    ctx = PlanCtx(scan_outcomes=["ok", "crash", "ok"])
    res = _run(ctx, _plan(3))
    assert not res.success
    assert "撞针" in res.data["abort_reason"]


def test_a_tip_event_is_not_retried():
    """重扫一次对针尖事件毫无意义,只是再废一帧。"""
    ctx = PlanCtx(scan_outcomes=["tip"])
    _run(ctx, _plan(1))
    assert len(_scan_calls(ctx)) == 1


# ── 守卫:质量差重扫一次 ─────────────────────────────────────────────────────

def test_a_poor_frame_is_rescanned_once():
    ctx = PlanCtx(scan_outcomes=["bad", "ok"])
    res = _run(ctx, _plan(1))
    assert res.success
    assert len(_scan_calls(ctx)) == 2
    assert res.data["frames_done"][0]["attempts"] == 2


def test_rescan_budget_is_bounded():
    """行动预算:不无限重试。一直差就标记继续,让坏帧率守卫去判系统性问题。"""
    ctx = PlanCtx(scan_outcomes=["bad"] * 10)
    _run(ctx, _plan(1))
    assert len(_scan_calls(ctx)) == MAX_RESCANS_PER_FRAME + 1


def test_stop_on_bad_quality_aborts_when_asked():
    ctx = PlanCtx(scan_outcomes=["bad", "bad", "ok"])
    res = _run(ctx, _plan(3), stop_on_bad_quality=True)
    assert not res.success
    assert res.data["aborted_at"] == 0


# ── 守卫:坏帧率 ─────────────────────────────────────────────────────────────

def test_a_high_bad_frame_rate_aborts_early():
    """系统性问题(表面脏 / 针尖钝 / 参数不合适)不会因为多扫几张就自己好。"""
    # 每帧两次尝试都坏 → 前 3 帧全坏
    ctx = PlanCtx(scan_outcomes=["bad"] * 20)
    res = _run(ctx, _plan(10))
    assert not res.success
    assert "坏帧率" in res.data["abort_reason"]
    assert res.data["aborted_at"] is not None
    assert res.data["aborted_at"] < 9, "扫完了才中止,守卫没起作用"


def test_a_single_bad_frame_never_aborts_the_batch():
    """一张差图通常是局部的(一粒脏东西、一次瞬时干扰),不是系统性问题。

    n=3 时最小的非零坏帧率就是 33% —— 已经越过 30% 的线。所以光有比例阈值不够,
    还得有最小样本数和最小坏帧数,否则「前三帧里坏了一张」就会中止整批。
    """
    ctx = PlanCtx(scan_outcomes=["bad", "bad", "ok", "ok", "ok", "ok", "ok"])
    res = _run(ctx, _plan(6))
    assert res.success, res.error
    assert res.data["bad"] == 1 and res.data["done"] == 5


def test_the_rate_guard_needs_a_minimum_sample():
    ctx = PlanCtx(scan_outcomes=["bad", "bad", "bad", "bad"])
    res = _run(ctx, _plan(2))
    # 只计划了 2 帧 —— 样本不够,不下系统性结论(仍然是「一帧都没成功」的失败)
    assert res.data["aborted_at"] is None


def test_rate_threshold_is_the_documented_one():
    assert BAD_FRAME_RATE_ABORT == pytest.approx(0.30)
    assert MIN_FRAMES_FOR_RATE >= 5, "样本太小时比例阈值没有意义"
    assert MIN_BAD_FOR_RATE >= 2, "一张坏帧不该中止批次"


# ── 大跳变稳定 ───────────────────────────────────────────────────────────────

def test_a_frame_flagged_for_settling_waits(monkeypatch):
    waits = []
    monkeypatch.setattr("mast.skills.composite.execute_scan_plan.time.sleep",
                        lambda s: waits.append(s))
    ctx = PlanCtx()
    _run(ctx, _plan(1, needs_settle=True))
    assert waits, "规划器标了大跳变却没等"


def test_a_normal_frame_does_not_wait(monkeypatch):
    waits = []
    monkeypatch.setattr("mast.skills.composite.execute_scan_plan.time.sleep",
                        lambda s: waits.append(s))
    _run(PlanCtx(), _plan(1))
    assert waits == []


# ── 中止与部分成功 ───────────────────────────────────────────────────────────

def test_operator_abort_stops_before_the_next_frame():
    ctx = PlanCtx(aborted=True)
    res = _run(ctx, _plan(3))
    assert not res.success
    assert res.data["aborted_at"] == 0
    assert _scan_calls(ctx) == []


def test_partial_success_puts_the_gap_in_the_summary():
    """把 fail_count 埋在 data 里没人读 —— 2026-07-27 修过的同一个坑。"""
    ctx = PlanCtx(scan_outcomes=["ok", "bad", "bad", "ok", "ok", "ok"])
    res = _run(ctx, _plan(4))
    assert res.success, res.error
    assert "未达标" in (res.summary or "")


def test_zero_successful_frames_is_a_hard_failure():
    ctx = PlanCtx(scan_outcomes=["bad"] * 4)
    res = _run(ctx, _plan(2))
    assert not res.success


# ── 输入校验 ─────────────────────────────────────────────────────────────────

def test_malformed_plan_json_is_reported():
    res = _run(PlanCtx(), "{not json")
    assert not res.success and "解析失败" in res.error


def test_empty_plan_is_reported():
    res = _run(PlanCtx(), json.dumps({"frames": []}))
    assert not res.success and "没有任何帧" in res.error


def test_a_frame_with_bad_geometry_is_skipped_not_fatal():
    plan = json.dumps({"frames": [
        {"index": 0, "center_x_m": "x", "center_y_m": 0.0, "size_m": 1e-7},
        {"index": 1, "center_x_m": 0.0, "center_y_m": 0.0, "size_m": 1e-7},
    ]})
    ctx = PlanCtx()
    res = _run(ctx, plan)
    assert res.success
    assert res.data["done"] == 1 and res.data["bad"] == 1
