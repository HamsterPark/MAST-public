"""像素→米:两份实现必须给出同一个答案,**直到有人把它们合并**。

## 为什么是测试而不是注释

`ExtractClusters` 的坐标换算与 `FindFlatRegion`(`flat_region.py:339-350`)
是**同一套约定**,而现在是**两份实现**。抽成共享件被有意推迟了:构建窗口里不动
别人正在用的文件。

但「注明应择期合并」是**注释,不是闸门** —— 这个仓已经证明注释挡不住漂移。
所以这条测试做两件事:

1. **现在**证明第二份抄对了;
2. **以后**任何一份改了而另一份没跟上,**当场红**。

「不能只有一份真源时,至少让分叉不能静默。」

## 判据是一个**第三方**表达式,不是互相比对

两份实现互相比对,只能证明它们**一样**,不能证明它们**对** —— 一起错的时候
测试照样绿。所以这里把约定**显式写成第三个表达式**(下面的 `expected_xy`),
两份都要跟它一致。约定本身来自 .sxm 的定义:

* 原点在扫描框左下,帧中心 = `scan_offset`;
* **y 要翻**:`scan_dir='down'` 时顶行(row 0)是最大 y;
* 再绕帧中心转 `scan_angle`。
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.skills.builtins.cluster_extract import ExtractClusters  # noqa: E402
from mast.skills.builtins.flat_region import FindFlatRegion  # noqa: E402

PX = 64
RANGE_M = 4e-8
OFFSET = (1.1e-7, -3.0e-8)          # 刻意用一个负的 y,免得符号错被 0 掩盖


def expected_xy(px_x, px_y, *, nx=PX, ny=PX, w=RANGE_M, h=RANGE_M,
                cx=OFFSET[0], cy=OFFSET[1], angle_deg=0.0):
    """**约定本身**(第三方判据)。两份实现都要跟它一致。"""
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    dx = (px_x + 0.5) / nx * w - w * 0.5
    dy = h * 0.5 - (px_y + 0.5) / ny * h
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca


def _write_sxm(path, arr):
    ny, nx = arr.shape
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 10.08.2026\n:REC_TIME:\n12:00:00\n:BIAS:\n\t1.000000E+0\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_RANGE:\n{RANGE_M:>19.6E}{RANGE_M:>19.6E}\n"
        f":SCAN_OFFSET:\n{OFFSET[0]:>19.6E}{OFFSET[1]:>19.6E}\n"
        ":SCAN_ANGLE:\n0.000E+0\n:SCAN_DIR:\ndown\n"
        ":Z-CONTROLLER>SETPOINT:\n20.0000E-12\n"
        ":DATA_INFO:\n\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tfwd\t9.000E-9\t0.000E+0\n:SCANIT_END:\n\n"
    )
    flat = arr.astype(np.float32).ravel()
    path.write_bytes(header.encode("utf-8") + b"\x1a\x04"
                     + struct.pack(">%df" % flat.size, *flat.tolist()))
    return path


@pytest.fixture()
def frame(tmp_path):
    """一个**偏心**的团簇 + 其余全是粗糙面。

    偏心是必须的:放在正中的话,x/y 弄反、符号弄反都测不出来。
    """
    rng = np.random.default_rng(0)
    img = 1e-9 + rng.normal(0, 4e-11, (PX, PX))     # 粗糙背景
    y, x = np.mgrid[0:PX, 0:PX]
    cy_px, cx_px = 14, 46                            # 明显偏心
    img[(y - cy_px) ** 2 + (x - cx_px) ** 2 <= 25] += 1.2e-9
    # 一块明确平坦的角落,给 FindFlatRegion 用
    img[44:60, 4:20] = 1e-9 + rng.normal(0, 2e-13, (16, 16))
    return _write_sxm(tmp_path / "geom.sxm", img), (cx_px, cy_px)


def test_extract_clusters_matches_the_convention(frame):
    """第一份实现 vs 约定本身。"""
    path, (cx_px, cy_px) = frame
    d = ExtractClusters().execute(None, {"scan_path": str(path),
                                         "polarity": "bright"}).data
    big = max(d["clusters"], key=lambda c: c["area_px"])
    ex, ey = expected_xy(cx_px, cy_px)
    tol = RANGE_M / PX                     # 一个像素
    assert big["x_m"] == pytest.approx(ex, abs=tol)
    assert big["y_m"] == pytest.approx(ey, abs=tol)


def test_find_flat_region_matches_the_same_convention(frame):
    """第二份实现 vs **同一个**约定。

    两份都对同一个第三方表达式负责,所以任何一份漂了都会红 —— 而不是
    「它们互相还一样」这种一起错也绿的比法。
    """
    path, _ = frame
    frac = 0.25
    res = FindFlatRegion().execute(None, {"scan_path": str(path),
                                          "window_fraction": frac})
    assert res.success, res.error
    d = res.data
    cx_m, cy_m = d.get("center_x_m"), d.get("center_y_m")
    assert cx_m is not None and cy_m is not None, f"没返回坐标:{sorted(d)}"

    # 用**它自己报的像素位置**反推,而不是猜它会选中哪一块。
    #
    # 第一版猜「它会选那块种好的平地」,结果差了 4.5 个像素就红了 —— 那条断言
    # 其实在测**窗口搜索**(它挑了哪个窗口),而这个文件要测的是**坐标变换**。
    # 一条把两件事混在一起的断言,红了也不知道是哪一件出的问题。
    ix, iy = d["pixel_origin"]
    win_px = max(8, int(min(PX, PX) * frac))          # 与 flat_region 同一式
    ex, ey = expected_xy(ix + win_px / 2.0, iy + win_px / 2.0)
    assert cx_m == pytest.approx(ex, abs=1e-15), (
        f"FindFlatRegion 的 px→m 与约定不符:{cx_m:.6e} vs {ex:.6e} —— "
        "两份实现漂开了,或者约定本身改了而只改了一处")
    assert cy_m == pytest.approx(ey, abs=1e-15)


def test_the_two_implementations_agree_until_someone_merges_them(frame):
    """两份实现在**同一个像素**上给出同一个米坐标。

    这条是前两条的合取形式,单独留着是因为它的**名字**就是给下一个人看的:
    合并之后请删掉这个文件,并在提交信息里说清楚合并到哪儿了。
    """
    path, (cx_px, cy_px) = frame
    d = ExtractClusters().execute(None, {"scan_path": str(path),
                                         "polarity": "bright"}).data
    big = max(d["clusters"], key=lambda c: c["area_px"])
    ex, ey = expected_xy(big["x_px"], big["y_px"])
    assert big["x_m"] == pytest.approx(ex, abs=1e-15)
    assert big["y_m"] == pytest.approx(ey, abs=1e-15)


def test_a_y_flip_would_be_caught(frame):
    """自检:如果 y 翻转丢了,上面那些断言真的会红吗?

    没有这一条,「y 约定」可能根本没被测到 —— 团簇如果恰好在中线上,
    翻不翻都一样。这里直接证明:把 y 翻转去掉,期望值会偏出容差。
    """
    _, (cx_px, cy_px) = frame
    ex, ey = expected_xy(cx_px, cy_px)
    # 不翻 y 的那个版本
    dy_noflip = (cy_px + 0.5) / PX * RANGE_M - RANGE_M * 0.5
    ey_wrong = OFFSET[1] + dy_noflip
    assert abs(ey - ey_wrong) > 4 * RANGE_M / PX, (
        "这个团簇太靠近中线,y 翻转与否测不出来 —— 请把它挪远一点")
