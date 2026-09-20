"""GraphExecutor + PreScanCheck regression tests (Phase 7 framework).

Pins the following contracts:

  1. ``plan(params)`` returns 6 :class:`CompositeStep` items:
     ConfigureScan -> SetScanSpeed -> StartScan -> WaitScanComplete
     -> SaveScan -> GetLatestScanFile.
  2. Running an empty-context executor walks every step exactly once,
     emits a :class:`CompositeProgress` snapshot whose ``completed_steps``
     length matches the plan, and the post-scan quality eval populates
     ``similarity`` / ``tip_ready`` / ``recommendation`` in the result.
  3. **Resume**: re-running with a prior ``composite_progress`` covering
     the first 3 steps causes the executor to skip them.
  4. ``WaitScanComplete.data["timed_out"] is True`` is converted into a
     hard SkillResult failure (matches v1 behaviour).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_prescan_check_graph.py -x -v
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
from typing import Any

import numpy as np
import pytest


def _textured_line(n: int = 64) -> "np.ndarray":
    """一条**有形貌**的扫描线：台阶 + 起伏 + 噪声，Z 偏置 1 nm。

    见下面 ``fwd_data`` 的注释：纯斜坡在新判据下是「没有信息」，
    而它在旧判据下拿满分 —— 那正是要修的缺陷。
    """
    x = np.arange(n)
    rng = np.random.default_rng(17)
    return (1e-9
            + np.where(x < n // 2, 0.0, 2e-10)
            + 4e-11 * np.sin(2 * np.pi * x / 11.0)
            + rng.normal(0, 5e-12, n))

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
)
from mast.skills.composite.prescan_check import PreScanCheck


# ── Fake ExecutionContext ────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Minimal context — supports run(), safe_call(), + progress hooks."""
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    safe_call_log: list[tuple[str, tuple]] = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    # When True, WaitScanComplete reports a timeout
    wait_times_out: bool = False
    # When set, WaitScanComplete reports a scan stopped part-way (v6.1.3)
    wait_lines_done: int | None = None
    wait_lines_total: int = 512
    # Canned Scan_FrameDataGrab data — fwd / bwd as a list-of-floats.
    #
    # 使用带台阶、起伏与噪声的合成形貌；纯斜坡去趋势后没有有效信息。
    # 缓冲区即便含纹理，也不能据此证明与存盘帧同源或给出针尖合格判决。
    _LINE: list[float] = field(default_factory=lambda: list(_textured_line()))
    fwd_data: list[float] = field(default_factory=lambda: list(_textured_line()))
    bwd_data: list[float] = field(default_factory=lambda: list(_textured_line()))

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "WaitScanComplete":
            done = (self.wait_lines_total if self.wait_lines_done is None
                    else self.wait_lines_done)
            stopped_early = done < self.wait_lines_total
            return SkillResult(
                skill_name=skill_name,
                success=True,
                data={"timed_out": self.wait_times_out, "polls": 1,
                      "stopped_early": stopped_early,
                      "outcome": ("timed_out" if self.wait_times_out
                                  else "stopped_early" if stopped_early
                                  else "completed"),
                      "lines_done": done, "lines_total": self.wait_lines_total,
                      "lines_verified": True},
            )
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.safe_call_log.append((method, args))
        if method == "Scan_FrameDataGrab":
            # args = (channel, direction); 1=forward, 0=backward
            direction = args[1] if len(args) > 1 else 1
            data = self.fwd_data if direction == 1 else self.bwd_data
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", data),
            )
        if method == "Scan_Action":
            return NanonisCallRecord(method=method, args=args, return_value=None)
        return NanonisCallRecord(method=method, args=args, return_value=None)

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(
            CompositeProgress.from_dict(progress.to_dict())
        )

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


# ── Tests ────────────────────────────────────────────────────────────────


EXPECTED_STEPS = ["configure", "set_speed", "start_scan", "wait_scan",
                  "save_scan", "latest_file"]
EXPECTED_SKILLS = ["ConfigureScan", "SetScanSpeed", "StartScan",
                   "WaitScanComplete", "SaveScan", "GetLatestScanFile"]


def test_plan_is_scan_then_save_then_locate_the_file():
    """计划先完成扫描，再存盘并定位文件。

    实时缓冲未必对应刚保存的帧；存盘必须排在等待完成之后，避免保存半帧。
    """
    skill = PreScanCheck()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 10e-9,
    })
    step_ids = [s.step_id for s in plan]
    assert step_ids == EXPECTED_STEPS
    assert [s.skill_name for s in plan] == EXPECTED_SKILLS
    assert step_ids.index("save_scan") > step_ids.index("wait_scan"), (
        "存盘排到了等待之前 —— 存下来的会是半帧")
    # 两步都是 optional：拿不到文件时回落到缓冲并报告弃权，不能冒充合格判决。
    by_id = {s.step_id: s for s in plan}
    assert by_id["save_scan"].optional is True
    assert by_id["latest_file"].optional is True
    assert len({s.step_id for s in plan}) == len(plan)


def test_configure_step_scans_a_square_not_a_strip():
    """预扫描必须覆盖二维方形区域，而不是压缩成细条。

    帧时由行数、每线时间和扫描方向数决定；只减小高度不会缩短既定行数的帧时。
    """
    skill = PreScanCheck()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 100e-9,
    })
    configure = plan[0]
    assert configure.params["height_m"] == pytest.approx(100e-9), (
        "预扫描应覆盖二维方形区域，不能把高度压缩成细条。")
    assert configure.params["height_m"] == configure.params["width_m"]


def test_full_execution_walks_every_step_and_emits_quality():
    """缓冲回落完成所有步骤并提供三态字段；无法确认数据可比时不给出合格判决。"""
    skill = PreScanCheck()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 10e-9,
        "quality_threshold": 0.8,
    })
    assert [name for name, _ in ctx.run_log] == EXPECTED_SKILLS
    assert result.success
    # 替身没有真 .sxm,所以走的是缓冲区那条回落 —— 报告必须说清用了哪条。
    assert result.data["quality_source"] == "buffer"
    assert result.data["tip_ready"] is None
    assert result.data["similarity"] is None
    assert result.data["recommendation"] == "inconclusive"
    # 必须报告不可比的原因、弃权和补测建议，不能把未知解释为需要修针。
    from mast.skills.composite.prescan_check import _BUFFER_NOT_COMPARABLE

    reason = result.data["read_failure"]
    assert reason == _BUFFER_NOT_COMPARABLE
    assert "不可比" in reason
    assert "弃权" in reason
    assert ".sxm" in reason
    assert "不是修针" in reason


def test_progress_snapshot_carries_into_result():
    """`_progress` snapshot is lifted into result.data for the adapter."""
    skill = PreScanCheck()
    ctx = FakeCtx()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 10e-9,
    })
    snap = result.data["_progress"]
    assert isinstance(snap, dict)
    assert snap["composite_name"] == "PreScanCheck"
    assert len(snap["completed_steps"]) == len(EXPECTED_STEPS)
    assert snap["aborted"] is False


def test_wait_timeout_becomes_hard_failure():
    """v1 semantics: WaitScanComplete.timed_out=True → fail + stop scan."""
    skill = PreScanCheck()
    ctx = FakeCtx(wait_times_out=True)
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 10e-9,
    })
    assert not result.success
    assert "timed out" in result.error.lower()
    # Stop the stuck scan via raw safe_call
    methods = [m for m, _ in ctx.safe_call_log]
    assert "Scan_Action" in methods


# ── a pre-scan line that was cut short ────────────────────────────────
#
# These pin that SOMEONE READS `stopped_early`, not that the field exists.
# Every assertion is on PreScanCheck's own outcome.


def _cut_short(**over):
    return FakeCtx(wait_lines_done=3, wait_lines_total=512, **over)


def test_a_pre_scan_line_that_was_cut_short_fails():
    """A line that never finished measures nothing. Letting it through would
    green-light a full scan on a tip that was never actually measured — the
    same fail-open this file already refuses by seeding tip_ready=None."""
    result = PreScanCheck().execute(_cut_short(), {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
    })
    assert not result.success
    assert "中途停止" in result.error
    assert "3/512" in result.error


def test_a_cut_short_pre_scan_is_not_stopped_again():
    """This branch is only reached after Scan_StatusGet read 0, so the scan is
    already stopped. The timeout branch above DOES issue Scan_Action; copying
    it here would be a hardware write on a read-only conclusion."""
    ctx = _cut_short()
    PreScanCheck().execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
    })
    assert "Scan_Action" not in [m for m, _ in ctx.safe_call_log]


def test_a_cut_short_pre_scan_does_not_claim_a_tip_verdict():
    """The worst outcome is not the failure — it is reporting tip_ready off a
    line that was never measured."""
    result = PreScanCheck().execute(_cut_short(), {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
    })
    assert result.data.get("tip_ready") is None
    assert result.data.get("similarity") is None


def test_a_complete_pre_scan_line_still_succeeds():
    """Control. Without it, "cut short fails" could just mean everything fails."""
    result = PreScanCheck().execute(FakeCtx(), {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
    })
    assert result.success


def test_resume_skips_completed_steps():
    """Prior progress covers the first 3 steps → the rest run fresh."""
    skill = PreScanCheck()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 10e-9,
    })
    completed = [s.step_id for s in plan[:3]]
    prior = CompositeProgress(
        composite_name="PreScanCheck",
        total_steps=len(plan),
        completed_steps=list(completed),
        partial_data={},
    )
    ctx = FakeCtx(prior_progress=prior.to_dict())
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 10e-9,
    })
    # 只有还没做过的那几步会重新调子技能。
    sub_names = [name for name, _ in ctx.run_log]
    assert sub_names == EXPECTED_SKILLS[3:]
    snap = result.data["_progress"]
    assert len(snap["completed_steps"]) == len(EXPECTED_STEPS)
    assert result.success


def test_progress_emitted_every_step():
    """emit_progress fires after each successful step + once on finish."""
    skill = PreScanCheck()
    ctx = FakeCtx()
    skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 10e-9,
    })
    # Successful steps and completion emit progress snapshots.
    assert len(ctx.emitted) >= 5
    # Final emit has current_step=None
    assert ctx.emitted[-1].current_step is None


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
