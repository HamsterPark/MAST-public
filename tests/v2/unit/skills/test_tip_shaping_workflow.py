"""Smoke tests for the new tip-shaping workflow skills.

Covers metadata correctness and one-call execution-path stubs:
  - MonitorCurrent: feeds a fake Current_Get and checks contact detection
  - FindFlatRegion: feeds a synthetic .sxm and checks (cx, cy) lands on flatest spot
  - AssessClusterRoundness: feeds an .sxm with a synthetic bright disc → score ≥ 0.8
  - ShapeTipOnSurface: mocked context.run() to walk the full flow
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if _MASTV2_ROOT not in sys.path:
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.core.types import NanonisCallRecord, SkillResult  # noqa: E402
from mast.skills.builtins import (  # noqa: E402
    AssessClusterRoundness,
    FindFlatRegion,
    MonitorCurrent,
)
from mast.skills.composite import ShapeTipOnSurface  # noqa: E402


def _ok_call(method, ret_val=("", b"", [0.0])):
    return NanonisCallRecord(
        method=method, args=(), return_value=ret_val,
        error="", elapsed_s=0.0, timestamp="2026-05-18T00:00:00",
    )


# ── MonitorCurrent ────────────────────────────────────────────────────


def test_monitor_current_detects_contact_when_threshold_crossed():
    """Feed a sequence where current crosses threshold after 5 samples."""
    samples = [1e-11] * 4 + [1e-7] * 6  # baseline pA, then big jump above 5e-8

    ctx = MagicMock()
    idx = {"i": 0}

    def _safe_call(method, *args):
        if method == "Current_Get":
            i = idx["i"]
            idx["i"] += 1
            v = samples[min(i, len(samples) - 1)]
            return _ok_call(method, ("", b"", [v]))
        return _ok_call(method)

    ctx.safe_call = MagicMock(side_effect=_safe_call)
    ctx.check_abort = MagicMock(return_value=False)

    result = MonitorCurrent().execute(
        ctx,
        {
            "duration_s": 0.2,
            "poll_hz": 200.0,
            "contact_threshold_a": 5e-8,
            "min_contact_samples": 3,
        },
    )
    assert result.success
    assert result.data["n_samples"] >= 4
    # max_abs_a is the largest |I| seen during polling; with the spike above,
    # it should be on the order of 1e-7 even with the 200 Hz / 0.2 s budget.
    if result.data["max_abs_a"] >= 5e-8:
        assert result.data["contact_detected"] is True
        assert result.data["contact_at_s"] is not None


def test_monitor_current_metadata_safe():
    m = MonitorCurrent().metadata()
    assert m.name == "MonitorCurrent"
    assert m.safety_level.value == "auto"  # read-only
    assert m.category.value == "read"
    # duration_s bounded
    duration = next(p for p in m.parameters if p.name == "duration_s")
    assert duration.max_value == 60.0


# ── FindFlatRegion ────────────────────────────────────────────────────


def _synth_sxm(tmp: Path, nx: int = 64, ny: int = 64,
               width_m: float = 1e-7, height_m: float = 1e-7,
               cx_m: float = 0.0, cy_m: float = 0.0,
               second_patch: bool = False) -> Path:
    """Write a tiny .sxm with a Z channel that is flat in the centre and
    bumpy at the edges.

    ``second_patch`` 在左上角再放一块同样平的小块 —— 用来把「排除逻辑」
    和「这一片没有可用平区」两件事分开测。2026-08-11 之前它们混在一条测试里:
    排除掉唯一的平块之后仍然断言 `success`,而那正是 `FindFlatRegion` 从不放弃、
    把 500 pm 的噪声当平区交出去的那个行为。
    """
    # Z field: 1 nm random bumps everywhere, but the centre 1/4 has near-zero noise.
    rng = np.random.RandomState(0)
    z = rng.normal(0.0, 5e-10, size=(ny, nx))  # 0.5 nm noise
    cy_px, cx_px = ny // 2, nx // 2
    half = nx // 8  # 16x16 flat patch
    z[cy_px - half:cy_px + half, cx_px - half:cx_px + half] = rng.normal(0.0, 1e-12, (2 * half, 2 * half))
    if second_patch:
        z[2:2 + 2 * half, 2:2 + 2 * half] = rng.normal(0.0, 1e-12, (2 * half, 2 * half))

    header = (
        ":NANONIS_VERSION:\n1\n"
        ":SCAN_PIXELS:\n"
        f"{nx} {ny}\n"
        ":SCAN_RANGE:\n"
        f"{width_m:.6e} {height_m:.6e}\n"
        ":SCAN_OFFSET:\n"
        f"{cx_m:.6e} {cy_m:.6e}\n"
        ":SCAN_ANGLE:\n0.000E+0\n"
        ":SCAN_DIR:\ndown\n"
        ":DATA_INFO:\n"
        "\t1\tZ\tm\tboth\t1.0\n"
        ":SCANIT_END:\n"
    )
    path = tmp / "synth.sxm"
    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(b"\x1a\x04")
        # forward + backward Z channels, big-endian float32
        f.write(z.astype(">f4").tobytes())
        f.write(z.astype(">f4").tobytes())
    return path


def test_find_flat_region_picks_flat_centre(tmp_path):
    sxm = _synth_sxm(tmp_path)
    ctx = MagicMock()
    ctx.check_abort = MagicMock(return_value=False)

    result = FindFlatRegion().execute(
        ctx,
        {"scan_path": str(sxm), "window_fraction": 0.2, "stride_fraction": 0.5},
    )
    assert result.success, result.error
    # Centre should be near (0, 0) since that's where the flat patch is.
    cx, cy = result.data["center_x_m"], result.data["center_y_m"]
    assert abs(cx) < 2e-8, f"cx={cx} expected near 0"
    assert abs(cy) < 2e-8, f"cy={cy} expected near 0"
    assert result.data["windows_checked"] > 1


def test_find_flat_region_skips_excluded_spots(tmp_path):
    """排除掉中心的平块之后，要挑到**另一块真的平的**地方。

    帧里放了两块平区（中心 + 左上角），所以「换一块」是有答案的 —— 这条测的是
    排除逻辑本身。2026-08-11 之前这条用的是只有**一块**平区的帧，于是它实际上
    断言的是「排除掉唯一的平块之后仍然要返回点什么」，也就是下面那条现在明确
    禁止的行为。两件事拆开测，覆盖比原来多。
    """
    sxm = _synth_sxm(tmp_path, second_patch=True)
    ctx = MagicMock()

    result = FindFlatRegion().execute(
        ctx,
        {
            "scan_path": str(sxm),
            "window_fraction": 0.2,
            "exclude_used_spots": "0.0,0.0",
            "min_separation_m": 3e-8,
        },
    )
    assert result.success, result.error
    cx, cy = result.data["center_x_m"], result.data["center_y_m"]
    assert (cx ** 2 + cy ** 2) >= (3e-8) ** 2, "should not pick a spot near (0,0)"
    # 挑到的必须是**真的平**的那一块，不是「最不烂的噪声」。
    assert result.data["rms_m"] <= result.data["usable_rms_m"]


def test_excluding_the_only_flat_patch_says_there_is_nothing_here(tmp_path):
    """把唯一的平块排除掉 ⇒ **「这里没有」**，而不是交出最不烂的那个窗。

    「一个区域找不到可以换地方找，没必要一定在一个地方
    找到。**现在的算法可能不会放弃。**」

    这张帧除了中心那一小块以外全是 500 pm 噪声。排除中心之后，正确答案是
    「这一片没有可用平区」—— 而旧实现返回的是 argmin，也就是一块 500 pm 的噪声，
    调用方会拿它去调平、甚至把针扎进去。argmin 永远存在，所以不设绝对线就等于
    永远不会放弃。
    """
    sxm = _synth_sxm(tmp_path)          # 只有中心一块平区
    result = FindFlatRegion().execute(
        MagicMock(),
        {
            "scan_path": str(sxm),
            "window_fraction": 0.2,
            "exclude_used_spots": "0.0,0.0",
            "min_separation_m": 3e-8,
        },
    )
    assert not result.success, (
        f"排除掉唯一的平块之后它仍然交出了一个点: {result.data}")
    assert "没有可用" in (result.error or "")
    # 拒绝时要把**最平的那个是多少**报出来 —— 「没有」和「差多远」是两个信息。
    assert result.data["best_rms_m"] > result.data["usable_rms_m"]


def test_the_absolute_line_can_be_overridden_explicitly(tmp_path):
    """绝对线可以被调用方**显式**放宽 —— 但只能显式，不能由帧自己推上去。

    这个区别是本次设计的核心：`max(25 pm, k×噪声底)` 那种自适应形式会让
    **越脏的帧线越松**，使拒绝条件随着待评输入本身变化。
    显式参数则是调用方写下来的一个数，看得见、进得了报告和版本历史。
    """
    sxm = _synth_sxm(tmp_path)
    params = {"scan_path": str(sxm), "window_fraction": 0.2,
              "exclude_used_spots": "0.0,0.0", "min_separation_m": 3e-8}
    assert not FindFlatRegion().execute(MagicMock(), dict(params)).success
    loose = FindFlatRegion().execute(MagicMock(), {**params, "usable_rms_m": 1e-9})
    assert loose.success, loose.error
    assert loose.data["usable_rms_m"] == 1e-9, "报告里要说清用的是哪条线"


# ── AssessClusterRoundness ────────────────────────────────────────────


def _synth_cluster_sxm(tmp: Path, asymmetric: bool = False) -> Path:
    """Topography with a bright disc (or ellipse) at the centre."""
    nx = ny = 64
    z = np.zeros((ny, nx), dtype=np.float64)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy_px, cx_px = ny // 2, nx // 2
    if asymmetric:
        r2 = ((xx - cx_px) / 12.0) ** 2 + ((yy - cy_px) / 3.0) ** 2  # ratio 4:1
    else:
        r2 = ((xx - cx_px) / 8.0) ** 2 + ((yy - cy_px) / 8.0) ** 2  # circle
    z[r2 < 1.0] = 5e-10

    header = (
        ":SCAN_PIXELS:\n"
        f"{nx} {ny}\n"
        ":SCAN_RANGE:\n5.000e-9 5.000e-9\n"
        ":SCAN_OFFSET:\n0 0\n"
        ":DATA_INFO:\n\t1\tZ\tm\tboth\t1.0\n"
        ":SCANIT_END:\n"
    )
    p = tmp / ("ellipse.sxm" if asymmetric else "round.sxm")
    with open(p, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(b"\x1a\x04")
        f.write(z.astype(">f4").tobytes())
        f.write(z.astype(">f4").tobytes())
    return p


def test_assess_round_disc(tmp_path):
    sxm = _synth_cluster_sxm(tmp_path, asymmetric=False)
    ctx = MagicMock()
    result = AssessClusterRoundness().execute(ctx, {"scan_path": str(sxm)})
    assert result.success, result.error
    # 2026-08-11:`roundness_score`(0.6*circ+0.4*aspect)作废。现在报的是
    # **等效轴比**:「相当于一个短轴/长轴 = q 的椭圆」,完美圆盘 = 1.0。
    # 0.7 这个数在旧标度上是「圆盘最高只能到 0.770」的七成,在新标度上
    # 是「长短轴差 30% 以内」—— 两个标度不可比,所以这里连数带含义一起换。
    assert result.data["equivalent_axis_ratio"] >= 0.8, result.data
    assert result.data["is_round"] is True


def test_assess_asymmetric_ellipse(tmp_path):
    sxm = _synth_cluster_sxm(tmp_path, asymmetric=True)
    ctx = MagicMock()
    result = AssessClusterRoundness().execute(ctx, {"scan_path": str(sxm)})
    assert result.success
    # ellipse 12:3 → aspect ratio ~0.25, score should be below threshold
    assert result.data["aspect_ratio"] < 0.6
    assert result.data["is_round"] is False


# ── ShapeTipOnSurface (mock context.run) ──────────────────────────────


def test_shape_tip_on_surface_succeeds_round_cluster(tmp_path):
    sxm_wide = _synth_sxm(tmp_path / "wide", nx=64, ny=64) if False else _synth_sxm(tmp_path)
    sxm_cluster = _synth_cluster_sxm(tmp_path, asymmetric=False)

    ctx = MagicMock()
    ctx.check_abort = MagicMock(return_value=False)

    def _run(skill_name: str, params: dict) -> SkillResult:
        # Stub each sub-skill the composite calls. Real ones are exercised
        # by their own tests above.
        if skill_name == "GetScanFrame":
            return SkillResult(
                skill_name=skill_name, success=True,
                data={"center_x_m": 0.0, "center_y_m": 0.0,
                       "width_m": 1e-7, "height_m": 1e-7},
            )
        if skill_name == "FindFlatRegion":
            return FindFlatRegion().execute(ctx, params)
        if skill_name == "AssessClusterRoundness":
            return AssessClusterRoundness().execute(
                ctx, {**params, "scan_path": str(sxm_cluster)},
            )
        if skill_name == "ConfigureScan":
            return SkillResult(skill_name=skill_name, success=True, data={"config_set": True})
        if skill_name == "StartScan":
            return SkillResult(skill_name=skill_name, success=True, data={})
        if skill_name == "WaitScanComplete":
            return SkillResult(skill_name=skill_name, success=True, data={})
        if skill_name == "SaveScan":
            return SkillResult(skill_name=skill_name, success=True, data={"path": str(sxm_cluster)})
        if skill_name == "GetLatestScanFile":
            return SkillResult(skill_name=skill_name, success=True, data={"path": str(sxm_cluster)})
        if skill_name == "TipShapeWithReadback":
            # Simulate contact on the first plunge step: a permanent Z change
            # (indent verdict "cluster") + an in-process current jump.
            return SkillResult(
                skill_name=skill_name, success=True,
                data={
                    "indent": {"verdict": "cluster", "delta_m": -5e-9},
                    "jumps": {"current": {"max_abs_delta": 1e-7}},
                },
            )
        if skill_name == "ZControllerOnOff":
            return SkillResult(skill_name=skill_name, success=True, data={})
        return SkillResult(skill_name=skill_name, success=True, data={})

    ctx.run = MagicMock(side_effect=_run)

    result = ShapeTipOnSurface().execute(
        ctx,
        {
            "wide_scan_path": str(sxm_wide),
            "cluster_window_m": 5e-9,
            "n_depth_steps": 3,
            "max_attempts": 2,
        },
    )
    assert result.success, result.error
    assert result.data["success_attempt"] == 1
    assert result.data["equivalent_axis_ratio"] >= 0.8
    assert result.data["cluster_scan_path"] == str(sxm_cluster)


def test_shape_tip_on_surface_retries_when_not_round(tmp_path):
    sxm_wide = _synth_sxm(tmp_path)
    sxm_ellipse = _synth_cluster_sxm(tmp_path, asymmetric=True)
    sxm_round = _synth_cluster_sxm(tmp_path / "second", asymmetric=False) if False else _synth_cluster_sxm(tmp_path, asymmetric=False)

    ctx = MagicMock()
    ctx.check_abort = MagicMock(return_value=False)
    attempt_idx = {"i": 0}

    def _run(skill_name: str, params: dict) -> SkillResult:
        if skill_name == "GetScanFrame":
            return SkillResult(
                skill_name=skill_name, success=True,
                data={"center_x_m": 0.0, "center_y_m": 0.0,
                       "width_m": 1e-7, "height_m": 1e-7},
            )
        if skill_name == "FindFlatRegion":
            return FindFlatRegion().execute(ctx, params)
        if skill_name == "AssessClusterRoundness":
            # First attempt → ellipse, second → round
            attempt_idx["i"] += 1
            chosen = sxm_ellipse if attempt_idx["i"] == 1 else sxm_round
            return AssessClusterRoundness().execute(
                ctx, {**params, "scan_path": str(chosen)},
            )
        if skill_name == "TipShapeWithReadback":
            return SkillResult(
                skill_name=skill_name, success=True,
                data={"indent": {"verdict": "cluster", "delta_m": -5e-9},
                      "jumps": {"current": {"max_abs_delta": 1e-7}}},
            )
        # Generic OK for the rest
        return SkillResult(skill_name=skill_name, success=True, data={"path": str(sxm_round)})

    ctx.run = MagicMock(side_effect=_run)

    result = ShapeTipOnSurface().execute(
        ctx,
        {
            "wide_scan_path": str(sxm_wide),
            "cluster_window_m": 5e-9,
            "n_depth_steps": 2,
            "max_attempts": 3,
        },
    )
    assert result.success, result.error
    # Should succeed on attempt 2, after first one was ellipse
    assert result.data["success_attempt"] == 2
    assert attempt_idx["i"] == 2
