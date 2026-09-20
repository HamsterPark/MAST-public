"""用独立合成形状核对轴比与长宽比的合取判据，不能让一项补偿另一项。"""
from __future__ import annotations

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

import struct  # noqa: E402

from mast.skills.builtins.cluster_roundness import AssessClusterRoundness  # noqa: E402

PX = 96
RANGE_M = 3e-8


def _write_sxm(path, arr):
    ny, nx = arr.shape
    hdr = (":NANONIS_VERSION:\n2\n:SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
           ":REC_DATE:\n 10.08.2026\n:REC_TIME:\n12:00:00\n:BIAS:\n\t1.0E+0\n"
           f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
           f":SCAN_RANGE:\n{RANGE_M:>19.6E}{RANGE_M:>19.6E}\n"
           f":SCAN_OFFSET:\n{0.0:>19.6E}{0.0:>19.6E}\n"
           ":SCAN_ANGLE:\n0.000E+0\n:SCAN_DIR:\ndown\n"
           ":Z-CONTROLLER>SETPOINT:\n20.0000E-12\n:DATA_INFO:\n"
           "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
           "\t14\tZ\tm\tfwd\t9.000E-9\t0.000E+0\n:SCANIT_END:\n\n")
    f = arr.astype(np.float32).ravel()
    path.write_bytes(hdr.encode() + b"\x1a\x04" + struct.pack(">%df" % f.size, *f.tolist()))
    return path


def _blob_frame(tmp_path, name, mask):
    """一帧:只有 mask 那一块是高的,其余是弱噪声。"""
    rng = np.random.default_rng(0)
    img = 1e-9 + rng.normal(0, 2e-12, (PX, PX))
    img[mask] += 8e-10
    return _write_sxm(tmp_path / f"{name}.sxm", img)


def _disc(r, *, cy=PX // 2, cx=PX // 2):
    y, x = np.mgrid[0:PX, 0:PX]
    return ((y - cy) ** 2 + (x - cx) ** 2) <= r * r


def _assess(tmp_path, name, mask, **kw):
    res = AssessClusterRoundness().execute(
        None, {"scan_path": str(_blob_frame(tmp_path, name, mask)), **kw})
    assert res.success, res.error
    return res.data


def _q(tmp_path, name, mask):
    """等效轴比 —— 现在的主输出。"""
    return _assess(tmp_path, name, mask)["equivalent_axis_ratio"]


# ── 钉「有分辨力」,不钉「等于某个值」 ──────────────────────────────────

def test_the_reading_separates_shapes_that_look_different(tmp_path):
    """喂几个**明显不同圆度**的团簇,读数必须分得开,而且方向要对。

    钉的是**这个判据有分辨力**,不是**它算得对**。
    「恒定的判据和不存在的判据,输出一模一样。」
    """
    rng = np.random.default_rng(3)
    disc = _disc(14)
    ragged = disc ^ ((rng.random((PX, PX)) < 0.20) & disc)     # 啃掉边缘
    streak = np.zeros((PX, PX), bool); streak[46:50, 8:88] = True
    ring = disc & ~_disc(9)                                     # 环:面积小周长长

    vals = {n: _q(tmp_path, n, m) for n, m in
            (("disc", disc), ("ragged", ragged), ("streak", streak), ("ring", ring))}
    assert len({round(v, 3) for v in vals.values()}) >= 3, (
        f"四种明显不同的形状只给出 {len({round(v,3) for v in vals.values()})} "
        f"个不同的读数:{vals} —— 这个判据没有分辨力")
    assert vals["disc"] > vals["ragged"], f"圆盘没有比毛边团更圆:{vals}"
    assert vals["disc"] > vals["streak"], f"圆盘没有比细长条更圆:{vals}"
    assert vals["disc"] > vals["ring"], f"圆盘没有比圆环更圆:{vals}"


def test_a_disc_passes_at_every_size_the_operator_actually_pokes(tmp_path):
    """合成圆盘在不同尺寸下都应通过圆度判据。"""
    for r in (6, 10, 14):
        d = _assess(tmp_path, f"d{r}", _disc(r))
        assert d["is_round"] is True, (
            f"半径 {r} 的圆盘被判「不圆」:轴比 {d['equivalent_axis_ratio']:.3f}、"
            f"aspect {d['aspect_ratio']:.3f}、闸门 {d['min_axis_ratio']}/{d['min_aspect']}")
        assert d["equivalent_axis_ratio"] > 0.85


def test_the_reading_reflects_shape_not_size(tmp_path):
    """不同半径的圆盘读数必须接近 —— 旧判据在这里从 0.53 漂到 0.62,
    于是固定阈值系统性地冤枉小团簇。"""
    vals = [_q(tmp_path, f"s{r}", _disc(r)) for r in (6, 10, 14)]
    assert max(vals) - min(vals) < 0.12, f"读数随半径漂得太厉害:{vals}"


def test_it_is_a_conjunction_not_a_weighted_sum(tmp_path):
    """一项好**不能**补另一项差。

    ⚠️ 这条测试只有落在**两种判决式给出不同答案**的那一段里才有意义。
    第一版用了 amp=0.30 的四瓣:合取判否,而加权和算出来 0.734,**也判否** ——
    于是「把合取改回加权和」这个变异**活了下来**。断言没瞄错对象,
    是**样本落在了两条路还没分岔的地方**。

    amp=0.22 落在分岔里:轴比 0.664(不合格)、aspect 1.000、
    加权和 0.6×0.664+0.4×1.000 = **0.798 ≥ 0.75(合格)**。
    下面把「加权和确实会放它过」也断言出来,这样以后有人改形状参数时,
    测试会先告诉他「你把样本挪出分岔区了」,而不是悄悄失去效力。
    """
    y, x = np.mgrid[0:PX, 0:PX]
    cy = cx = PX // 2
    ang = np.arctan2(y - cy, x - cx)
    rad = np.hypot(y - cy, x - cx)
    clover = rad <= 14.0 * (1.0 + 0.22 * np.cos(4 * ang))       # 四瓣
    d = _assess(tmp_path, "clover", clover)

    q, aspect = d["equivalent_axis_ratio"], d["aspect_ratio"]
    assert aspect > 0.9, (
        f"这个形状的 aspect 是 {aspect:.3f} —— 它没在演示「一项好补另一项差」")
    assert q < d["min_axis_ratio"], f"轴比 {q:.3f} 本身就合格,测不到东西"
    assert 0.6 * q + 0.4 * aspect >= d["min_axis_ratio"], (
        f"加权和算出 {0.6 * q + 0.4 * aspect:.3f},它也会判否 —— "
        "这个样本落在两种判决式**还没分岔**的地方,换一个瓣幅")

    assert d["is_round"] is False, (
        "aspect≈1 的四瓣形状被判「圆」—— 判决又变回加权和了")


def test_a_too_small_blob_is_undecidable_not_false(tmp_path):
    """三态:``is_round`` 可以是 **None**。

    小于 20 px 时像素化本身就能让**完美的圆**读出 0.64–0.77 的轴比。
    在那个尺度上返回 False 是在**用噪声做决策**,而调用方会据此再扎一针。
    """
    d = _assess(tmp_path, "tiny", _disc(2))
    assert d["is_round"] is None, f"太小的团簇给出了 {d['is_round']!r},而不是「判不了」"
    assert d["roundness_undecidable"]
    assert d["equivalent_axis_ratio"] is None
    assert _assess(tmp_path, "big", _disc(10))["is_round"] is True


def test_the_retired_threshold_cannot_come_back_silently(tmp_path):
    """退役的圆度参数应明确拒绝，避免别名继续影响判定。"""
    res = AssessClusterRoundness().execute(
        None, {"scan_path": str(_blob_frame(tmp_path, "rt", _disc(10))),
               "round_threshold": 0.65})
    assert not res.success
    assert "round_threshold" in (res.error or "")
    assert "min_axis_ratio" in (res.error or ""), "拒绝了却没给处方"
    assert res.data.get("use_instead") == "min_axis_ratio"


def test_the_old_weighted_sum_ranked_a_square_above_a_disc():
    """把**被否掉的旧判决式**钉下来,连同它的数字。

    否则「0.6*circ + 0.4*aspect,简单又直观」会带着它的直观回来,
    而它错的地方恰恰不直观。
    """
    circ_disc_sup = 4 * np.pi ** 2 / 64          # 数字化圆盘的上确界
    circ_square = np.pi / 4                      # 轴对齐正方形,精确
    old_disc = 0.6 * circ_disc_sup + 0.4 * 1.0
    old_square = 0.6 * circ_square + 0.4 * 1.0
    assert old_disc == pytest.approx(0.770, abs=0.002)
    assert old_square == pytest.approx(0.871, abs=0.002)
    assert old_square > old_disc, (
        "旧判决式不再把方块排在圆盘前面了 —— 如果口径变了,这段历史说明要一起改")
    assert old_disc > 0.65 > circ_disc_sup, (
        "旧阈值 0.65 与「完美圆盘的 circularity 上确界」的关系变了")
