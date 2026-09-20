"""GraphExecutor + ShapeTipOnSurface regression tests (Phase 7 framework).

ShapeTipOnSurface uses **plan_dynamic** because each attempt's step
sequence depends on prior results (stop early on contact, accept-or-
relocate from roundness). These tests pin:

  1. Single-attempt happy path: wide_scan_path supplied → no fresh
     wide-scan steps emitted; FindFlatRegion → cluster scan →
     AssessClusterRoundness; final result.success on is_round=True.
  2. Multi-attempt path: first attempt's cluster not round → second
     attempt issued; excluded_spots accumulates between attempts.
  3. Contact break: MonitorCurrent reports contact_detected → no
     deeper plunge step emitted in that attempt.
  4. No-contact path: every depth completes without contact → spot is
     excluded, next attempt issues a fresh FindFlatRegion.
  5. Cleanup steps (ZControllerOnOff + ConfigureScan restore) always
     emitted at end of plan_dynamic, regardless of outcome.
  6. Resume: prior progress covering ``get_original_frame`` + first
     attempt's wide-scan steps causes the executor to skip those.
  7. emit_progress fires per step.

Run from repo root::
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_shape_tip_on_surface_graph.py \
        -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
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
from typing import Any, Callable

import pytest

from mast.core.types import SkillResult
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
)
from mast.skills.composite.shape_tip_on_surface import ShapeTipOnSurface


# ── Programmable fake context ────────────────────────────────────────────


@dataclass
class FakeCtx:
    """ExecutionContext stand-in driven by per-call response handlers.

    For each skill_name, ``responder[skill]`` may be either:
      • a SkillResult                → returned every call
      • a callable(params, call_idx) → produces a SkillResult on demand
      • a list of SkillResult        → consumed in order (round-robin
                                       after exhaustion)
    """
    responder: dict[str, Any] = field(default_factory=dict)
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    _counts: dict[str, int] = field(default_factory=dict)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        idx = self._counts.get(skill_name, 0)
        self._counts[skill_name] = idx + 1

        r = self.responder.get(skill_name)
        if isinstance(r, SkillResult):
            return r
        if callable(r):
            return r(dict(params), idx)
        if isinstance(r, list) and r:
            return r[min(idx, len(r) - 1)]
        # Default: skill succeeds with empty data
        return SkillResult(skill_name=skill_name, success=True, data={})

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(
            CompositeProgress.from_dict(progress.to_dict())
        )

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


def _ok(skill: str, data: dict[str, Any] | None = None) -> SkillResult:
    return SkillResult(skill_name=skill, success=True, data=data or {})


def _readback(contact: bool, delta_m: float = -5e-9) -> SkillResult:
    """TipShapeWithReadback result (审查: contact is now judged
    from the in-process Z indent verdict + current jump, not a post-hoc
    MonitorCurrent). contact=True → a permanent Z change ("cluster"); False →
    "no_change"."""
    return _ok("TipShapeWithReadback", {
        "indent": {"verdict": "cluster" if contact else "no_change",
                   "delta_m": delta_m if contact else 0.0},
        "jumps": {"current": {"max_abs_delta": 1e-7 if contact else 1e-13}},
    })


def _frame_data() -> dict[str, Any]:
    """Canned GetScanFrame return."""
    return {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 1e-7, "height_m": 1e-7,
    }


# ── Tests ────────────────────────────────────────────────────────────────


def test_single_attempt_happy_path_with_supplied_scan_path():
    """
    wide_scan_path provided → no fresh wide-scan sub-plan.
    FindFlatRegion succeeds, first plunge depth makes contact,
    cluster scan saved, roundness check passes → success.
    """
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 1e-9, "center_y_m": 2e-9, "rms_m": 1e-11,
        }),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": _readback(True),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.85, "is_round": True,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 3,
        "n_depth_steps": 5,
    })

    assert result.success, result.error
    assert result.data["success_attempt"] == 1
    assert result.data["spot_x_m"] == 1e-9
    assert result.data["spot_y_m"] == 2e-9
    assert result.data["equivalent_axis_ratio"] == 0.85
    assert result.data["attempts"] == 1
    # No fresh wide-scan sub-plan was emitted: the only StartScan call
    # is the cluster-scan one (1), no wide_start.
    names = [n for n, _ in ctx.run_log]
    assert names.count("StartScan") == 1
    # ConfigureScan happens 2× per attempt + once at finalize: cluster
    # configure + cluster_configure(again) + restore = 3.
    # (Phase order: configure_cluster, cluster_configure, restore_frame)
    assert names.count("ConfigureScan") == 3
    # Cleanup steps emitted
    assert "ZControllerOnOff" in names


def test_contact_break_stops_plunge_loop_early():
    """When MonitorCurrent reports contact on plunge step 2, no further
    TipShape / MonitorCurrent steps should be emitted for that attempt."""
    # Sequence the MonitorCurrent results: step1 no-contact, step2 contact
    readback_seq = [_readback(False), _readback(True)]
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11,
        }),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": readback_seq,
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.9, "is_round": True,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 1,
        "n_depth_steps": 5,
    })
    assert result.success
    # Plunge loop should have done exactly 2 TipShape + 2 MonitorCurrent
    # calls and not continued to step 3/4/5.
    names = [n for n, _ in ctx.run_log]
    assert names.count("TipShapeWithReadback") == 2


def test_no_contact_marks_spot_excluded_next_attempt_uses_new_spot():
    """If a full plunge sweep makes no contact, the spot is excluded and
    the next attempt's FindFlatRegion is called with the spot in
    exclude_used_spots."""
    # First attempt: no contact (all 3 monitor calls no-contact).
    # Second attempt: contact on step 1 → eventually success.
    flat_seq = [
        _ok("FindFlatRegion", {
            "center_x_m": 10e-9, "center_y_m": 20e-9, "rms_m": 1e-11,
        }),
        _ok("FindFlatRegion", {
            "center_x_m": 30e-9, "center_y_m": 40e-9, "rms_m": 1e-11,
        }),
    ]
    # 3 (no-contact) + 1 (contact) = 4 MonitorCurrent calls
    readback_seq = [_readback(False), _readback(False), _readback(False), _readback(True)]
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": flat_seq,
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": readback_seq,
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.9, "is_round": True,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 3,
        "n_depth_steps": 3,  # 3 depths per plunge → 3 monitor calls per attempt
    })
    assert result.success, result.error
    assert result.data["success_attempt"] == 2
    # Second FindFlatRegion call should have the first spot in exclude_used_spots
    flat_calls = [
        params for name, params in ctx.run_log if name == "FindFlatRegion"
    ]
    assert len(flat_calls) == 2
    excl = flat_calls[1].get("exclude_used_spots", "")
    # The first spot (10e-9, 20e-9) should be in the exclude list. Python
    # serializes 10e-9 as "1e-08" so check for either format.
    assert "1e-08" in excl or "10e-09" in excl or "10e-9" in excl
    assert "2e-08" in excl or "20e-09" in excl or "20e-9" in excl


def test_failure_after_max_attempts_no_round_cluster():
    """If no attempt yields is_round=True, skill fails with the canonical
    'No round cluster after N attempts' error and attempt_log records
    every attempt."""
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11,
        }),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": _readback(True),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        # Roundness always low (not round)
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.2, "is_round": False,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 2,
        "n_depth_steps": 2,
    })
    assert not result.success
    assert "No round cluster" in result.error
    assert "after 2 attempts" in result.error
    log = result.data.get("attempt_log", [])
    assert len(log) == 2
    for entry in log:
        assert entry["is_round"] is False


def test_finalize_steps_always_emitted():
    """ZControllerOnOff + restore frame always run at end (success or fail)."""
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11,
        }),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": _readback(True),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.2, "is_round": False,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 1,
        "n_depth_steps": 1,
    })
    names = [n for n, _ in ctx.run_log]
    assert "ZControllerOnOff" in names
    # The finalize ConfigureScan (restore frame) should be the last
    # ConfigureScan call in the log.
    last = ctx.run_log[-1]
    assert last[0] == "ConfigureScan"
    assert last[1]["width_m"] == 1e-7  # matches the original frame
    # progress should record finalize steps as completed
    snap = result.data["_progress"]
    assert any(s.startswith("finalize:") for s in snap["completed_steps"])


def test_progress_emitted_on_each_step():
    """emit_progress fires per step + once on plan finish."""
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11,
        }),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": _readback(True),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.85, "is_round": True,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 1, "n_depth_steps": 1,
    })
    # At least N step emits + 1 final emit
    assert len(ctx.emitted) >= len(ctx.run_log) + 1
    assert ctx.emitted[-1].current_step is None


def test_resume_skips_completed_steps():
    """Prior progress covering ``get_original_frame`` causes the executor
    to skip that step (no GetScanFrame call) on re-execution."""
    responder: dict[str, Any] = {
        # If something accidentally runs GetScanFrame again, this fails
        # because the partial_data lookup expects the frame.
        "GetScanFrame": _ok("GetScanFrame", {}),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11,
        }),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": _readback(True),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.9, "is_round": True,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    # Pre-mark get_original_frame as done in prior progress
    prior = CompositeProgress(
        composite_name="ShapeTipOnSurface",
        total_steps=20,
        completed_steps=["get_original_frame"],
        partial_data={
            "original_frame": _frame_data(),
            "attempt_log": [],
            "excluded_spots": [],
        },
    )
    ctx = FakeCtx(responder=responder, prior_progress=prior.to_dict())
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 1, "n_depth_steps": 1,
    })
    assert result.success
    names = [n for n, _ in ctx.run_log]
    # GetScanFrame should NOT have run (resumed-skipped)
    assert "GetScanFrame" not in names


def test_fresh_wide_scan_emitted_when_no_path_provided():
    """If wide_scan_path is empty, attempt 1 issues the wide-scan
    sub-plan (configure → start → wait → save → latest)."""
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/wide.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/wide.sxm"}),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11,
        }),
        "TipShapeWithReadback": _readback(True),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.9, "is_round": True,
        }),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "",  # → trigger fresh wide scan
        "max_attempts": 1, "n_depth_steps": 1,
    })
    assert result.success
    names = [n for n, _ in ctx.run_log]
    # Wide-scan sub-plan: StartScan + WaitScanComplete + SaveScan +
    # GetLatestScanFile all appear at least once.
    assert names.count("StartScan") >= 1
    assert names.count("WaitScanComplete") >= 1
    assert names.count("SaveScan") >= 1
    assert names.count("GetLatestScanFile") >= 1


def test_aggregate_carries_attempt_log_on_failure():
    """Even on failure, attempt_log is in result.data for diagnostics.

    Two attempts both fail at the plunge phase (no contact). v1 re-uses
    the user-supplied wide_scan_path only on attempt 1, then triggers a
    fresh wide scan for subsequent attempts — so we wire StartScan /
    SaveScan / GetLatestScanFile too.
    """
    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11,
        }),
        "ConfigureScan": _ok("ConfigureScan", {}),
        # Never contact
        "TipShapeWithReadback": _readback(False),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/wide.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/wide.sxm"}),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    skill = ShapeTipOnSurface()
    result = skill.execute(ctx, {
        "wide_scan_path": "C:/wide.sxm",
        "max_attempts": 2, "n_depth_steps": 2,
    })
    assert not result.success
    assert "No round cluster" in result.error
    log = result.data.get("attempt_log", [])
    assert len(log) == 2
    for entry in log:
        assert entry["outcome"] == "no_contact"
        assert len(entry["plunge_log"]) == 2  # n_depth_steps


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


def test_finalize_restore_feedback_uses_the_real_parameter_name():
    """收尾那一步得真能跑 —— 「调用发生了」不等于「调用是有效的」。

    ZControllerOnOff 的参数叫 ``enable``；这一步曾传 ``{"on": True}``，
    validate_params 直接判失败，而 optional=True 把失败吞掉：反馈没恢复，
    进度里却记着 finalize 完成，类文档还写着「Always restores feedback」。
    上面那个 test_finalize_steps_always_emitted 只断言了技能名出现过，
    所以它对这个 bug 一直是满意的。
    """
    from mast.skills.builtins.zcontrol import ZControllerOnOff

    responder: dict[str, Any] = {
        "GetScanFrame": _ok("GetScanFrame", _frame_data()),
        "FindFlatRegion": _ok("FindFlatRegion", {
            "center_x_m": 0.0, "center_y_m": 0.0, "rms_m": 1e-11}),
        "ConfigureScan": _ok("ConfigureScan", {}),
        "TipShapeWithReadback": _readback(True),
        "StartScan": _ok("StartScan", {}),
        "WaitScanComplete": _ok("WaitScanComplete", {"timed_out": False}),
        "SaveScan": _ok("SaveScan", {"path": "C:/fake.sxm"}),
        "GetLatestScanFile": _ok("GetLatestScanFile", {"path": "C:/fake.sxm"}),
        "AssessClusterRoundness": _ok("AssessClusterRoundness", {
            "equivalent_axis_ratio": 0.2, "is_round": False}),
        "ZControllerOnOff": _ok("ZControllerOnOff", {}),
    }
    ctx = FakeCtx(responder=responder)
    ShapeTipOnSurface().execute(ctx, {
        "wide_scan_path": "C:/wide.sxm", "max_attempts": 1, "n_depth_steps": 1})

    calls = [p for n, p in ctx.run_log if n == "ZControllerOnOff"]
    assert calls, "收尾没有恢复反馈"
    # 参数必须过得了真技能的校验（fake context 不会校验，所以在这里补上）。
    assert ZControllerOnOff().validate_params(calls[-1]) == []


def test_the_retired_round_threshold_is_refused_by_the_composite_too():
    """``round_threshold`` 在**每一个**入口都要当场报错,不能只在技能里挡。

    2026-08-11:判据换成等效轴比后,旧阈值 0.65 的含义变了 —— 而**两个阈值方向
    相同**(都是越大越圆),所以一个漏改的 0.65 不会崩,只会**悄悄把闸门从
    「长短轴差 25%」放宽到「差 35%」**。这种「能跑的错版本」正是要靠入口处的
    硬拒绝挡掉的。

    composite 这一层单独挡一次,是因为它有**自己的默认值**:只挡技能层的话,
    composite 收下 `round_threshold` 后会用**它自己的** min_axis_ratio 默认值
    跑完,并报告成功 —— 调用方以为自己设的是 0.65,实际跑的是 0.75。
    """
    ctx = FakeCtx(responder=lambda *a, **k: None)
    res = ShapeTipOnSurface().execute(ctx, {
        "wide_scan_path": "C:/wide.sxm", "round_threshold": 0.65})
    assert not res.success
    assert "round_threshold" in (res.error or "")
    assert "min_axis_ratio" in (res.error or ""), "拒绝了却没给处方"
    # **一步硬件动作都不许发生**:阈值不认识就不要开始扎针。
    assert not ctx.run_log, f"拒绝之前已经跑了步骤:{[n for n, _ in ctx.run_log]}"


def test_both_entrances_refuse_the_retired_threshold_not_just_one():
    """两条路各自拒一次 —— **只挡一条等于没挡**。

    * `validate_params` 只在 **agent / LLM** 那条路上跑(`skill_adapter` 调它);
    * 直接 `.execute()` **不经过** `validate_params`,得靠 `plan_dynamic` 顶上
      那道 abort。

    上面那条测试只覆盖了直接调用。这一条单独钉 `validate_params`,
    否则删掉它没有任何测试会红 —— 而 agent 恰恰是最可能照抄旧参数名的那一方。
    """
    errs = ShapeTipOnSurface().validate_params(
        {"wide_scan_path": "C:/w.sxm", "round_threshold": 0.65})
    assert any("round_threshold" in e and "min_axis_ratio" in e for e in errs), errs
    # 不传就不该报这一条(免得这个断言靠「永远报错」变绿)
    assert not any("round_threshold" in e for e in
                   ShapeTipOnSurface().validate_params({"wide_scan_path": "C:/w.sxm"}))
