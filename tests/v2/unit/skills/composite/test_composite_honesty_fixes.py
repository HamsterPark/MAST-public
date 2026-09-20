"""Regression tests for composite-skill "honesty" fixes.

Each test pins one finding so the bug cannot silently regress:

  #85  PreScanCheck.prescan_check — a failed hardware read of the pre-scan
       line must report tip_ready=None (inconclusive), NOT tip_ready=True
       (fail-OPEN).
       ⚠️ 这条的后半句原文是「A genuinely-read flat line still returns a real
       0.0」,**两次被推翻,现在两句都不成立**:
         * 2026-08-10 死平的线改成第三态(旧余弦对非零常数线给的是 **1.0**,
           不是 0.0 —— 同一种输入两种相反的判定);
         * 2026-08-14**整条缓冲回落**降级为「判不了」,所以本文件里走
           `Scan_FrameDataGrab` 的用例一律 tip_ready=None。针尖判决只从存盘的
           .sxm 出。
  #140 GridSTS.grid_sts — plan() num_points fallback must match the
       metadata() default (40), not a stale 200 (5x over-acquisition on an
       omitted param).
  #141 DemoScanAndSTS.aggregate — sts_failed is read from the live
       accumulator (single source of truth), not re-derived from
       (sts_total - sts_succeeded).
  #142 FullScan._check_scan_data — post-scan crash check probes multiple
       channels (incl. Z) and records crash_check = ok|skipped|crash. A
       read failure is "skipped", never silently "no crash".
  #143 DemoScanAndSTS._build_sts_positions — sts_count > 5 yields a real
       grid of DISTINCT points, never the center coordinate repeated.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/composite/test_composite_honesty_fixes.py -q
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
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

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite.graph_executor import CompositeProgress
from mast.skills.composite.prescan_check import PreScanCheck
from mast.skills.composite.grid_sts import GridSTS
from mast.skills.composite.full_scan import FullScan
from mast.skills.composite.demo_scan_and_sts import DemoScanAndSTS


# ── Fake ExecutionContext ─────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Context supporting run() (sub-skills) + safe_call() (raw Nanonis).

    ``safe_call_results`` maps a Nanonis method to a list of
    (return_value, error) tuples consumed in order; a method with a single
    entry reuses it for every call.
    """
    run_log: list[tuple[str, dict]] = field(default_factory=list)
    failing_skills: set[str] = field(default_factory=set)
    wait_times_out: bool = False
    safe_calls: list[tuple[str, tuple]] = field(default_factory=list)
    # method -> list of (return_value, error)
    safe_call_results: dict[str, list[tuple[Any, str]]] = field(
        default_factory=dict
    )
    emitted: list[CompositeProgress] = field(default_factory=list)
    flushes: int = 0
    prior_progress: dict | None = None

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name in self.failing_skills:
            return SkillResult(
                skill_name=skill_name, success=False, error="canned failure",
            )
        if skill_name == "WaitScanComplete":
            return SkillResult(
                skill_name=skill_name, success=True,
                data={"timed_out": self.wait_times_out, "polls": 1},
            )
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.safe_calls.append((method, args))
        entries = self.safe_call_results.get(method)
        if entries:
            entry = entries.pop(0) if len(entries) > 1 else entries[0]
            ret, err = entry
            return NanonisCallRecord(
                method=method, args=args, return_value=ret, error=err,
            )
        return NanonisCallRecord(
            method=method, args=args, return_value=None, error="",
        )

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name: str) -> dict | None:
        return self.prior_progress

    def checkpoint_flush(self) -> None:
        self.flushes += 1


def _frame(samples) -> tuple:
    """Build a Scan_FrameDataGrab-style return value: parsed[2] = data."""
    return ("name", (8, 8), np.asarray(samples, dtype=np.float64))


# ══════════════════════════════════════════════════════════════════════════
# #85 — PreScanCheck fail-OPEN on read failure
# ══════════════════════════════════════════════════════════════════════════


def _good_line(n: int = 64) -> np.ndarray:
    """生成含台阶、起伏与噪声的合成扫描线；纯斜坡去趋势后没有可比较形貌。"""
    x = np.arange(n)
    step = np.where(x < n // 2, 0.0, 2e-10)
    ripple = 4e-11 * np.sin(2 * np.pi * x / 11.0)
    rng = np.random.default_rng(5)
    return 1e-9 + step + ripple + rng.normal(0, 5e-12, n)


def test_85_prescan_read_error_is_inconclusive_not_tip_ready():
    """fwd/bwd FrameDataGrab error → tip_ready is None (inconclusive)."""
    ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [(None, "tcp error")],
    })
    skill = PreScanCheck()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
    })
    assert result.success  # the scan steps succeeded
    assert result.data["tip_ready"] is None, (
        "read failure must be inconclusive, not a tip_ready=True verdict"
    )
    assert result.data["similarity"] is None
    assert result.data["recommendation"] == "inconclusive"


def test_85_prescan_exception_is_inconclusive():
    """An exception inside the read path → inconclusive, not fail-OPEN."""
    class BoomCtx(FakeCtx):
        def safe_call(self, method, *args, role="main"):
            raise RuntimeError("hardware exploded")

    ctx = BoomCtx()
    skill = PreScanCheck()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
    })
    assert result.data["tip_ready"] is None
    assert result.data["recommendation"] == "inconclusive"


def test_85_prescan_good_line_is_told_apart_from_a_read_failure():
    """缓冲读取成功与失败必须给出不同原因；成功读取仍不能证明与保存帧同源。"""
    line = _good_line()
    ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [
            (_frame(line), ""),   # fwd
            (_frame(line), ""),   # bwd
        ],
    })
    skill = PreScanCheck()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
        "quality_threshold": 0.8,
    })
    assert result.data["tip_ready"] is None
    assert result.data["similarity"] is None
    assert result.data["recommendation"] == "inconclusive"

    # 对照:同一条流程,读**失败**时那句话必须不一样。
    fail_ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [(None, "tcp error")],
    })
    failed = PreScanCheck().execute(fail_ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9})
    assert (result.data.get("read_failure")
            != failed.data.get("read_failure")), (
        "「读到了但这条路判不了」和「根本没读到」说了同一句话 —— "
        "两者的下一步不同(一个去 .sxm 取帧,一个去查通信)")
    assert "报错" in (failed.data.get("read_failure") or "")


def test_85_prescan_flat_read_line_is_unjudgeable_not_a_bad_tip():
    """恒定的 Z 线缺少判据信息，应该弃权；读失败和无信息分别保留原因。"""
    flat = np.zeros(32)
    ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [
            (_frame(flat), ""),
            (_frame(flat), ""),
        ],
    })
    skill = PreScanCheck()
    result = skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9,
    })
    assert result.data["similarity"] is None
    assert result.data["tip_ready"] is None
    assert result.data["recommendation"] == "inconclusive"
    # 「读到了但判不了」必须与「根本没读到」可区分。
    assert "死平" in result.data.get("unusable_reason", "")


def test_85_a_read_failure_and_a_featureless_line_are_two_sentences():
    """两种 inconclusive 必须分得开 —— 一个要去查通信，一个要去换地方扫。"""
    ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [(None, "tcp error")],
    })
    res = PreScanCheck().execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 10e-9})
    assert res.data["tip_ready"] is None
    assert "unusable_reason" not in res.data, (
        "读失败不该带「这一帧没信息」的说明 —— 那会把人送去看表面，"
        "而问题在通信")


# ══════════════════════════════════════════════════════════════════════════
# #140 — GridSTS num_points fallback divergence
# ══════════════════════════════════════════════════════════════════════════


def test_140_grid_sts_num_points_fallback_matches_metadata_default():
    """Omitted num_points → plan uses 40 (metadata default), not 200."""
    skill = GridSTS()
    meta = skill.metadata()
    meta_default = next(
        p.default for p in meta.parameters if p.name == "num_points"
    )
    assert meta_default == 40

    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0, "spacing_m": 1e-9,
        "nx": 1, "ny": 1,
    })
    configure = next(s for s in plan if s.step_id == "configure")
    assert configure.params["num_points"] == meta_default, (
        "plan() fallback must equal metadata default (single source of truth)"
    )
    assert configure.params["num_points"] != 200


def test_140_grid_sts_explicit_num_points_respected():
    """An explicit num_points still flows through unchanged."""
    skill = GridSTS()
    plan = skill.plan({
        "center_x_m": 0.0, "center_y_m": 0.0, "spacing_m": 1e-9,
        "num_points": 123, "nx": 1, "ny": 1,
    })
    configure = next(s for s in plan if s.step_id == "configure")
    assert configure.params["num_points"] == 123


# ══════════════════════════════════════════════════════════════════════════
# #141 — DemoScanAndSTS aggregate sts_failed single source of truth
# ══════════════════════════════════════════════════════════════════════════


def test_141_aggregate_reads_failed_accumulator_not_recomputed():
    """sts_failed comes straight from the accumulator, not total-succeeded."""
    skill = DemoScanAndSTS()
    progress = CompositeProgress(
        composite_name="DemoScanAndSTS",
        partial_data={
            "sts_total": 5,
            "sts_succeeded": 2,
            "sts_failed": 1,   # only ONE real failure recorded so far
        },
    )
    data = skill.aggregate({}, progress)
    assert data["sts_succeeded"] == 2
    # The buggy recompute would yield 5 - 2 = 3; the accumulator says 1.
    assert data["sts_failed"] == 1, (
        "sts_failed must echo the live accumulator, not (total - succeeded)"
    )


def test_141_move_failure_counts_match_accumulator_end_to_end():
    """End-to-end: every MoveToXY fails → sts_failed == sts_count."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx(failing_skills={"MoveToXY"})
    result = skill.execute(ctx, {"sts_count": 4})
    assert result.data["sts_succeeded"] == 0
    assert result.data["sts_failed"] == 4
    # No AcquireSTS attempted (skipped because its move failed)
    names = [n for n, _ in ctx.run_log]
    assert names.count("AcquireSTS") == 0


def test_141_partial_inflight_does_not_inflate_failed():
    """A point still 'in flight' must not be counted as failed.

    With the old (total - succeeded) recompute, a snapshot taken mid-run
    (succeeded < total, nothing failed yet) reported phantom failures.
    """
    skill = DemoScanAndSTS()
    progress = CompositeProgress(
        composite_name="DemoScanAndSTS",
        partial_data={"sts_total": 5, "sts_succeeded": 3, "sts_failed": 0},
    )
    data = skill.aggregate({}, progress)
    assert data["sts_failed"] == 0  # not 5 - 3 = 2


# ══════════════════════════════════════════════════════════════════════════
# #142 — FullScan crash check: multi-channel + honest skip
# ══════════════════════════════════════════════════════════════════════════


def _run_full_scan(ctx: FakeCtx) -> SkillResult:
    skill = FullScan()
    return skill.execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
    })


def test_142_crash_check_skipped_when_no_channel_readable():
    """All FrameDataGrab reads error → crash_check == 'skipped' (NOT silent ok)."""
    ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [(None, "read error")],
    })
    result = _run_full_scan(ctx)
    assert result.success  # crash check is best-effort, doesn't fail scan
    assert result.data["crash_check"] == "skipped", (
        "an unreadable crash check must be reported as skipped, not hidden"
    )
    # It must have actually attempted multiple channels (ch0 + Z).
    probed = [a for m, a in ctx.safe_calls if m == "Scan_FrameDataGrab"]
    channel_indices = {a[0] for a in probed}
    assert 0 in channel_indices and 14 in channel_indices


def test_142_crash_detected_on_z_channel_even_when_ch0_ok():
    """ch0 looks fine but Z is flat → crash IS detected (multi-channel)."""
    good = np.linspace(0, 1, 64)
    flat = np.zeros(64)

    class ChanCtx(FakeCtx):
        def safe_call(self, method, *args, role="main"):
            self.safe_calls.append((method, args))
            if method == "Scan_FrameDataGrab":
                ch = args[0]
                if ch == 0:
                    return NanonisCallRecord(
                        method=method, args=args,
                        return_value=_frame(good), error="",
                    )
                # Z channel (14) is flat → crash
                return NanonisCallRecord(
                    method=method, args=args,
                    return_value=_frame(flat), error="",
                )
            return NanonisCallRecord(method=method, args=args,
                                     return_value=None, error="")

    ctx = ChanCtx()
    result = _run_full_scan(ctx)
    assert not result.success, "flat Z channel is a crash and must fail the scan"
    assert "CRASH_DETECTED" in (result.error or "")
    assert result.data["crash_indicator"] is True
    assert result.data["crash_check"] == "crash"
    assert result.data["crash_check_channels"]["Z"] == "crash"
    assert result.data["crash_check_channels"]["ch0"] == "ok"


def test_142_crash_check_ok_when_all_channels_have_variance():
    """All channels have real variance → crash_check == 'ok'."""
    good = np.linspace(0, 1, 64)
    ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [(_frame(good), "")],
    })
    result = _run_full_scan(ctx)
    assert result.success
    assert result.data["crash_check"] == "ok"


def test_142_crash_detected_on_nan():
    """NaN in channel data → crash detected."""
    nan_arr = np.array([0.0, 1.0, np.nan, 2.0])
    ctx = FakeCtx(safe_call_results={
        "Scan_FrameDataGrab": [(_frame(nan_arr), "")],
    })
    result = _run_full_scan(ctx)
    assert not result.success
    assert result.data["crash_indicator"] is True


# ══════════════════════════════════════════════════════════════════════════
# #143 — DemoScanAndSTS real grid for sts_count > 5
# ══════════════════════════════════════════════════════════════════════════


def test_143_sts_positions_distinct_for_large_count():
    """sts_count > 5 must yield DISTINCT points, never the center repeated."""
    skill = DemoScanAndSTS()
    for count in (6, 9, 16, 25):
        positions = skill._build_sts_positions(0.0, 0.0, 30e-9, count)
        assert len(positions) == count
        unique = {(round(x, 18), round(y, 18)) for x, y in positions}
        assert len(unique) == count, (
            f"sts_count={count} produced {len(unique)} distinct points "
            f"(duplicates = fake independent measurements)"
        )
        # Center must not be silently repeated as filler.
        center_repeats = sum(1 for x, y in positions if x == 0.0 and y == 0.0)
        assert center_repeats <= 1


def test_143_large_grid_points_within_inner_region():
    """Grid points stay inside the inner square [cx±size/3, cy±size/3]."""
    skill = DemoScanAndSTS()
    size = 60e-9
    offset = size / 3.0
    positions = skill._build_sts_positions(1e-9, -2e-9, size, 9)
    for x, y in positions:
        assert (1e-9 - offset) - 1e-18 <= x <= (1e-9 + offset) + 1e-18
        assert (-2e-9 - offset) - 1e-18 <= y <= (-2e-9 + offset) + 1e-18


def test_143_legacy_layout_preserved_for_small_counts():
    """sts_count <= 5 still uses the historical center + 4 corners layout."""
    skill = DemoScanAndSTS()
    size = 30e-9
    offset = size / 3.0
    positions = skill._build_sts_positions(0.0, 0.0, size, 5)
    expected = [
        (0.0, 0.0),
        (-offset, -offset),
        (offset, -offset),
        (offset, offset),
        (-offset, offset),
    ]
    for (ax, ay), (ex, ey) in zip(positions, expected):
        assert ax == pytest.approx(ex)
        assert ay == pytest.approx(ey)
    # And smaller counts are a prefix of that layout.
    assert skill._build_sts_positions(0.0, 0.0, size, 1) == [(0.0, 0.0)]
    assert len(skill._build_sts_positions(0.0, 0.0, size, 3)) == 3


def test_143_large_count_end_to_end_moves_to_distinct_points():
    """End-to-end: 9-point demo issues 9 MoveToXY to distinct coordinates."""
    skill = DemoScanAndSTS()
    ctx = FakeCtx()
    result = skill.execute(ctx, {"sts_count": 9})
    assert result.success
    moves = [
        (p["x_m"], p["y_m"]) for n, p in ctx.run_log if n == "MoveToXY"
    ]
    assert len(moves) == 9
    assert len({(round(x, 18), round(y, 18)) for x, y in moves}) == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
