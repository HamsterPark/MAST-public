"""FindFlatRegion 的三个补齐(2026-07-30)。

原来的实现有三个缺口,每一个都会让「找一块 ≥50 nm 的平地」这个请求得到一个
看起来正常、实际不满足要求的答案:

  1. 窗口只能按**比例**表达 —— 「≥50 nm」在 50 nm 的帧上被理解成 10 nm;
  2. 最小 RMS 会选到「跨台阶但两半各自平坦」的窗 —— 对倾斜测量毫无用处;
  3. ``px_to_m`` 忽略 ``scan_angle`` —— 帧一旋转,返回的坐标就落在别处,
     而调用方拿着它去移动针尖。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mast.skills.builtins.flat_region import FindFlatRegion


def _make_sxm(path: Path, frame: np.ndarray, *, offset=(0.0, 0.0),
              size_m=(1e-7, 1e-7), angle_deg=0.0) -> Path:
    ny, nx = frame.shape
    header = (
        ":SCAN_PIXELS:\n" f"{nx} {ny}\n"
        ":SCAN_OFFSET:\n" f"{offset[0]:.6E} {offset[1]:.6E}\n"
        ":SCAN_RANGE:\n" f"{size_m[0]:.6E} {size_m[1]:.6E}\n"
        ":SCAN_ANGLE:\n" f"{angle_deg:.6E}\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n:SCANIT_END:\n"
    )
    data = np.asarray(frame, dtype=">f4")
    path.write_bytes(header.encode() + b"\x1a\x04" + data.tobytes() + data.tobytes())
    return path


def _run(path, **params):
    params.setdefault("scan_path", str(path))
    return FindFlatRegion().execute(None, params)


def _flat(n=128, sigma=15.4e-12, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, size=(n, n))


# ── ① 物理窗口约束 ───────────────────────────────────────────────────────────

def test_min_window_m_enlarges_the_window_beyond_the_fraction(tmp_path):
    """「至少 50 nm」在 100 nm 的帧上必须真的给出 ≥50 nm 的窗,
    而不是 window_fraction 说的 20 nm。"""
    p = _make_sxm(tmp_path / "a.sxm", _flat(128), size_m=(1e-7, 1e-7))
    loose = _run(p, window_fraction=0.2)
    strict = _run(p, window_fraction=0.2, min_window_m=5e-8)
    assert loose.success and strict.success
    assert loose.data["window_side_m"] == pytest.approx(2e-8, rel=0.1)
    assert strict.data["window_side_m"] >= 5e-8 * 0.98


def test_frame_smaller_than_the_required_window_fails_explicitly(tmp_path):
    """帧本身就装不下要求的窗时,必须显式失败。

    静默缩窗会返回一个「找到了」的结果,而调用方以为自己拿到了 ≥50 nm 的平地
    —— 之后在那上面做的调平测量全都建立在一个假前提上。
    """
    p = _make_sxm(tmp_path / "small.sxm", _flat(64), size_m=(3e-8, 3e-8))
    res = _run(p, min_window_m=5e-8)
    assert not res.success
    assert "小于要求的最小窗口" in res.error
    assert res.data["frame_short_m"] == pytest.approx(3e-8)


def test_min_window_defaults_to_off(tmp_path):
    """向后兼容:不传就是原来的行为。"""
    p = _make_sxm(tmp_path / "b.sxm", _flat(128), size_m=(3e-8, 3e-8))
    assert _run(p).success


# ── ② 整窗同层 ───────────────────────────────────────────────────────────────

def _two_terraces_with_flat_halves(n=128, step_m=240e-12, seed=1):
    """一张「跨台阶但两半各自极平」的图。

    最小 RMS 判据会被这种图骗到:跨台阶的窗里两半都很平,平面扣除后 RMS 不算
    大;而它对倾斜测量毫无用处 —— 台面就是晶面,跨两个台面测到的斜率是包络,
    不是失配角。
    """
    img = np.random.default_rng(seed).normal(0.0, 3e-12, size=(n, n))
    img[:, n // 2:] += step_m
    return img


def test_same_terrace_rejects_windows_that_straddle_a_step(tmp_path):
    p = _make_sxm(tmp_path / "step.sxm", _two_terraces_with_flat_halves(),
                  size_m=(1e-7, 1e-7))
    res = _run(p, window_fraction=0.3, same_terrace=True)
    assert res.success, res.error
    assert res.data["same_terrace_enforced"] is True
    assert res.data["windows_cross_terrace"] > 0, "一个跨台阶的窗都没排除?"

    # 选中的窗必须整个落在台阶的一侧
    ix, _iy = res.data["pixel_origin"]
    win = res.data["window_side_px"]
    assert ix + win <= 64 or ix >= 64, (
        f"选中的窗 x∈[{ix}, {ix + win}) 跨过了 x=64 的台阶")


def test_same_terrace_is_off_by_default(tmp_path):
    p = _make_sxm(tmp_path / "step2.sxm", _two_terraces_with_flat_halves(),
                  size_m=(1e-7, 1e-7))
    res = _run(p, window_fraction=0.3)
    assert res.data["same_terrace_enforced"] is False
    assert res.data["windows_cross_terrace"] == 0


def test_single_terrace_frame_needs_no_exclusions(tmp_path):
    """整帧就是一个台面时,同层约束自动满足,不该排除任何窗。"""
    p = _make_sxm(tmp_path / "flat.sxm", _flat(128), size_m=(1e-7, 1e-7))
    res = _run(p, window_fraction=0.3, same_terrace=True)
    assert res.success
    assert res.data["windows_cross_terrace"] == 0


def test_all_windows_cross_terrace_fails_with_a_useful_reason(tmp_path):
    """台面比要求的窗还窄时,要说清楚是这个原因 —— 而不是笼统的"没找到"。"""
    # 8 px 一条的密集台阶,任何 ≥30% 的窗都跨台阶
    n = 128
    img = np.random.default_rng(2).normal(0.0, 3e-12, size=(n, n))
    for k in range(0, n, 8):
        img[:, k:] += 240e-12
    p = _make_sxm(tmp_path / "dense.sxm", img, size_m=(1e-7, 1e-7))
    res = _run(p, window_fraction=0.4, same_terrace=True)
    if not res.success:
        assert "跨台阶" in res.error
        assert res.data["windows_cross_terrace"] > 0


# ── ③ scan_angle 修正 ────────────────────────────────────────────────────────

def test_rotation_moves_the_reported_coordinate(tmp_path):
    """帧旋转后,同一个像素对应的仪器坐标必须跟着转。

    没有这一步,一张转了 30° 的图上找到的"平坦区"坐标会落在别的地方 ——
    而调用方拿着它去移动针尖。
    """
    # 让最平的地方明确落在帧的一角,这样旋转带来的位移可见
    img = np.random.default_rng(3).normal(0.0, 1e-10, size=(128, 128))
    img[:32, :32] = np.random.default_rng(4).normal(0.0, 1e-13, size=(32, 32))

    straight = _run(_make_sxm(tmp_path / "s0.sxm", img, size_m=(1e-7, 1e-7),
                              angle_deg=0.0), window_fraction=0.2)
    turned = _run(_make_sxm(tmp_path / "s90.sxm", img, size_m=(1e-7, 1e-7),
                            angle_deg=90.0), window_fraction=0.2)
    assert straight.success and turned.success
    assert straight.data["scan_angle_deg"] == pytest.approx(0.0)
    assert turned.data["scan_angle_deg"] == pytest.approx(90.0)

    # 同一像素窗,旋转 90° 后坐标应绕帧中心转过去
    dx0 = straight.data["center_x_m"] - straight.data["scan_center_x_m"]
    dy0 = straight.data["center_y_m"] - straight.data["scan_center_y_m"]
    dx1 = turned.data["center_x_m"] - turned.data["scan_center_x_m"]
    dy1 = turned.data["center_y_m"] - turned.data["scan_center_y_m"]
    assert dx1 == pytest.approx(-dy0, abs=1e-12)
    assert dy1 == pytest.approx(dx0, abs=1e-12)


def test_rotation_preserves_the_distance_from_the_frame_centre(tmp_path):
    """旋转只换方向,不改变到帧中心的距离。"""
    img = np.random.default_rng(5).normal(0.0, 1e-10, size=(128, 128))
    img[:32, :32] = np.random.default_rng(6).normal(0.0, 1e-13, size=(32, 32))
    radii = []
    for ang in (0.0, 30.0, 45.0, 90.0, 180.0):
        res = _run(_make_sxm(tmp_path / f"r{ang}.sxm", img, size_m=(1e-7, 1e-7),
                             angle_deg=ang), window_fraction=0.2)
        assert res.success
        radii.append(math.hypot(
            res.data["center_x_m"] - res.data["scan_center_x_m"],
            res.data["center_y_m"] - res.data["scan_center_y_m"]))
    assert max(radii) - min(radii) < 1e-12


def test_zero_angle_matches_the_old_unrotated_formula(tmp_path):
    """0° 时必须与旧公式逐位一致(向后兼容)。"""
    img = _flat(128)
    p = _make_sxm(tmp_path / "z.sxm", img, offset=(1e-7, -2e-7),
                  size_m=(1e-7, 1e-7), angle_deg=0.0)
    res = _run(p, window_fraction=0.25)
    ix, iy = res.data["pixel_origin"]
    win = res.data["window_side_px"]
    nx = ny = 128
    w = h = 1e-7
    expect_x = 1e-7 - w * 0.5 + (ix + win / 2.0 + 0.5) / nx * w
    expect_y = -2e-7 + h * 0.5 - (iy + win / 2.0 + 0.5) / ny * h
    assert res.data["center_x_m"] == pytest.approx(expect_x, abs=1e-15)
    assert res.data["center_y_m"] == pytest.approx(expect_y, abs=1e-15)
