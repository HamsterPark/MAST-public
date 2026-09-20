"""几何退化的扫描帧应弃权，不应直接判为针尖质量差。
测试覆盖独立构造的细条、方形、温和矩形和未知几何。
宽高都应从回包读取，像素数组形状不能替代真实物理比例。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.vision.scan_monitor import ScanVisionMonitor  # noqa: E402


def _mon(w_nm, h_nm):
    """只填几何字段的裸壳 —— 这条判据不该依赖别的任何东西。"""
    m = ScanVisionMonitor.__new__(ScanVisionMonitor)
    m._scan_size_nm = w_nm
    m._scan_height_nm = h_nm
    return m


@pytest.mark.parametrize("w,h,why", [
    (48.0, 3.0, "独立合成细条，16:1"),
    (72.0, 6.0, "独立合成细条，12:1"),
    (3.0, 48.0, "反过来一样退化 —— 判据用 max(w/h, h/w)"),
])
def test_a_degenerate_strip_is_not_judged(w, h, why):
    assert _mon(w, h)._degenerate_frame() is True, (
        f"{why}：{w}×{h} nm 上没有二维形貌可判，"
        "在它上面出的判决是错误信息，不是判决。")


@pytest.mark.parametrize("w,h", [(50.0, 50.0), (100.0, 100.0),
                                 (60.0, 30.0), (30.0, 60.0)])
def test_square_and_mild_rectangles_are_still_judged(w, h):
    """只有这一条在，上一条才不是「把视觉判据关掉了」。

    2:1 的矩形扫描用户偶尔会用，必须照常判。
    """
    assert _mon(w, h)._degenerate_frame() is False, (
        f"{w}×{h} nm 被误判成退化 —— 这道闸伤到了正常扫描")


@pytest.mark.parametrize("w,h", [(None, None), (50.0, None), (None, 2.5),
                                 (0.0, 2.5), (50.0, 0.0), ("junk", 2.5)])
def test_unknown_geometry_does_not_block(w, h):
    """读不到几何 ⇒ **不拦**。

    不知道形状时沉默拦住，会让一台读不回 `Scan_FrameGet` 的机器**永远**拿不到
    视觉判断，而且没有任何一处会说这件事 —— 那正是本仓最贵的那一类失效。
    宁可放一张可疑的图进去（下游还有别的判据），也不要造一个沉默的黑洞。
    """
    assert _mon(w, h)._degenerate_frame() is False


def test_the_threshold_leaves_room_between_the_two_populations():
    """退化阈值应区分测试中的温和矩形与极窄条带，并保留非空中间区间。"""
    assert 2.0 < ScanVisionMonitor.DEGENERATE_ASPECT < 20.0
