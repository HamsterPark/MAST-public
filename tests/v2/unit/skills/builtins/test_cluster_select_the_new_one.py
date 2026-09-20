"""多团簇时,``AssessClusterRoundness`` 要能被要求评**新扎的那个**。

需要能够指定评估**新扎的那个**团簇，而不是画面里最大的团簇。

修针跑到第三下时,画面里**最大**的那个团簇常常是两轮之前的坑 —— 拿它给针尖打分,
是在给错误的对象打分,而分数看起来完全正常。

簇图是**以刚扎的那个点为中心**扫的,所以「离画面中心最近」就是新扎的那个;
不需要任何坐标变换。这个文件用一张合成图钉住这件事:**中心一个小团、角上一个大团**,
两种 ``select`` 必须选出不同的团。
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.skills.builtins.cluster_roundness import AssessClusterRoundness  # noqa: E402


def _disc(img, cy, cx, r, h):
    ys, xs = np.ogrid[:img.shape[0], :img.shape[1]]
    img[(ys - cy) ** 2 + (xs - cx) ** 2 <= r * r] = h


@pytest.fixture
def two_clusters(tmp_path, monkeypatch):
    """一张 128² 的图:**中心一个小团**(新扎的),**角上一个大团**(旧坑)。

    小团半径 6、大团半径 14 —— 面积差 5 倍以上,所以 "largest" 一定选角上那个。
    """
    img = np.zeros((128, 128), dtype=float)
    rng = np.random.default_rng(5)
    img += rng.normal(0.0, 1e-4, img.shape)      # 一点噪声,免得 std==0
    _disc(img, 64, 64, 6, 1.0)                   # 新扎的:画面中心
    _disc(img, 20, 108, 14, 1.2)                 # 旧坑:右上角,更大更高

    import mast.io.nanonis_files as NF
    monkeypatch.setattr(
        NF, "read_sxm",
        lambda p: {"channels": {"Z": {"forward": img}},
                   "header": {"scan_range": (1e-8, 1e-8)}},
        raising=True)

    p = tmp_path / "two_clusters.sxm"
    p.write_bytes(b"stub - read_sxm is patched")
    return str(p)


def _run(path, **params):
    return AssessClusterRoundness().execute(None, {"scan_path": path, **params})


def test_default_still_grades_the_largest_blob(two_clusters):
    """默认不变 —— 既有调用方给的都是「一个坑一张图」,悄悄改掉它们评的对象
    正是这次要修的那类错误。"""
    res = _run(two_clusters)
    assert res.success, res.error
    assert res.data["n_components"] >= 2
    big = res.data["area_px"]

    small = _run(two_clusters, select="center").data["area_px"]
    assert big > small, (
        f"'largest' 应当选到角上那个大团:largest={big}, center={small}")


def test_select_center_grades_the_one_we_just_made(two_clusters):
    """``select='center'`` 选画面中心那个小团 —— 新扎的那个。"""
    res = _run(two_clusters, select="center")
    assert res.success, res.error

    # 中心那个团半径 6 ⇒ 面积 ≈ π·36 ≈ 113 px;角上那个半径 14 ⇒ ≈ 616 px。
    assert 60 <= res.data["area_px"] <= 220, (
        f"选中的团面积 {res.data['area_px']} px 不像是中心那个小团")


def test_a_single_cluster_reads_the_same_either_way(tmp_path, monkeypatch):
    """只有一个团时两种 select 必须给出**同一个**结果。

    没有这一条,``select='center'`` 哪怕实现成「随便挑一个」也能让上面两条变绿。
    """
    img = np.zeros((96, 96), dtype=float)
    img += np.random.default_rng(2).normal(0.0, 1e-4, img.shape)
    _disc(img, 40, 55, 9, 1.0)

    import mast.io.nanonis_files as NF
    monkeypatch.setattr(NF, "read_sxm",
                        lambda p: {"channels": {"Z": {"forward": img}},
                                   "header": {}}, raising=True)
    p = tmp_path / "one.sxm"
    p.write_bytes(b"stub")

    a = _run(str(p))
    b = _run(str(p), select="center")
    assert a.success and b.success
    assert a.data["area_px"] == b.data["area_px"]
    # 2026-08-11:`roundness_score`(0.6*circ+0.4*aspect)作废,换成等效轴比。
    assert a.data["equivalent_axis_ratio"] == pytest.approx(
        b.data["equivalent_axis_ratio"])
