"""``vision.roundness``:钉**物理事实**,不钉当前实现。

每一条断言下面写的都是「若这个判据被换掉,这句话还该不该成立」。
换成 Feret 径、换成拟合椭圆、换成别的边界走法,这些都还该绿。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.vision.roundness import (  # noqa: E402
    MIN_AREA_PX,
    assess_mask,
    axis_ratio_from_dispersion,
    dispersion_floor,
    dispersion_of_mask,
)


# ── 合成形状 ────────────────────────────────────────────────────────────
def ellipse(area_px: float, ratio: float = 1.0, rot_deg: float = 0.0,
            cx: float = 0.0, cy: float = 0.0, lobes: int = 0,
            lobe_amp: float = 0.0, notch: float = 0.0) -> np.ndarray:
    a = math.sqrt(area_px / (math.pi * ratio))
    n = int(2 * a / min(ratio, 1.0) + 14) | 1
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2.0
    dx, dy = xx - c - cx, yy - c - cy
    t = math.radians(rot_deg)
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    ang = np.arctan2(v, u)
    mod = np.ones_like(ang)
    if lobes:
        mod = mod + lobe_amp * np.cos(lobes * ang)
    if notch:
        mod = mod - notch * np.exp(
            -(np.mod(ang + np.pi, 2 * np.pi) - np.pi) ** 2 / (2 * 0.35 ** 2))
    return np.hypot(u / a, v / (a * ratio)) / np.clip(mod, 0.05, None) <= 1.0


def q_of(mask) -> float:
    r = assess_mask(mask)
    assert r.ok, r.reason
    return r.axis_ratio


# ── 1. 标度:阈值必须有物理含义 ──────────────────────────────────────────

@pytest.mark.parametrize("ratio", [0.95, 0.90, 0.85, 0.80, 0.70, 0.50])
def test_the_reading_is_the_axis_ratio_it_says_it_is(ratio):
    """一个 b/a 的椭圆必须读回 ≈ b/a。

    这是这次重做的**全部意义**:旧判据给一个没有单位的分数,阈值 0.65 只能靠
    「和上次比」来理解,而 0.65 恰好落在圆盘的上确界(0.617)之上 —— 圆的东西
    永远过不去。新判据报的数**自己带物理含义**,阈值可以先定义再看数据。

    这条钉的是标度,不是实现。任何自称在量「圆不圆」的东西,喂给它一个
    2:1 的椭圆都该说「2:1」。
    """
    got = q_of(ellipse(600, ratio=ratio, rot_deg=27.0))
    assert got == pytest.approx(ratio, abs=0.06), (
        f"b/a={ratio} 的椭圆读成了 {got:.3f} —— 报出来的数不是它自称的那个量")


def test_the_analytic_ellipse_mapping_is_monotone_and_invertible():
    """离散 → 轴比 的换算必须单调,否则「更圆」这句话没有意义。"""
    ds = [0.0, 0.018, 0.037, 0.057, 0.079, 0.102, 0.126, 0.247]
    qs = [axis_ratio_from_dispersion(d) for d in ds]
    assert all(a > b for a, b in zip(qs, qs[1:])), qs
    assert qs[0] == 1.0
    # 表上的物理锚点:5% / 10% / 25% / 50% 的长短轴差
    assert axis_ratio_from_dispersion(0.018) == pytest.approx(0.95, abs=0.02)
    assert axis_ratio_from_dispersion(0.037) == pytest.approx(0.90, abs=0.02)
    assert axis_ratio_from_dispersion(0.102) == pytest.approx(0.75, abs=0.02)
    assert axis_ratio_from_dispersion(0.247) == pytest.approx(0.50, abs=0.02)


# ── 2. 尺寸无关:不许把旧的系统性偏差换个地方重建 ────────────────────────

#: 闸门的工作点。物理含义:「不比一个长短轴差 25% 的椭圆更不规则」。
#: 定义在先,数据在后 —— 它**不是**对用户那几帧拟合出来的。
GATE = 0.75


@pytest.mark.parametrize("radius", [3, 4, 5, 6, 8, 10, 14, 20])
def test_a_perfect_disc_never_fails_the_gate_at_any_subpixel_phase(radius):
    """**这条是这次事故的直接对立面,而且必须扫遍亚像素相位。**

    旧判据对完美圆盘从 0.465(小)读到 0.617(大),固定阈值 0.65 因此
    **系统性地**冤枉小团簇 —— 事实上冤枉了所有圆盘。新判据必须对**任何面积、
    任何定心相位**的完美圆盘都读「圆」。

    ⚠️ 只在一个相位上测等于在测运气:同一个完美圆盘在不同亚像素中心下读数
    **实测**摆动如下(最差 / 中位):

        r= 3 (A= 28)  0.757 / 1.000      r= 7 (A=154)  0.937 / 0.956
        r= 4 (A= 52)  0.853 / 0.971      r= 8 (A=201)  0.931 / 0.971
        r= 5 (A= 79)  0.844 / 1.000      r=14 (A=616)  0.972 / 1.000
        r= 6 (A=113)  0.906 / 0.970      r=20 (A=1260) 0.976 / 1.000

    这张表就是 GATE=0.75 的余量证明:**最差相位、最小尺寸也还有 0.007 的余量**,
    而旧判据在同一批圆盘上的余量是**负的**。
    """
    worst = min(
        q_of(ellipse(math.pi * radius ** 2, cx=ox, cy=oy))
        for ox in (0.0, 0.2, 0.4, 0.6, 0.8)
        for oy in (0.0, 0.2, 0.4, 0.6, 0.8))
    assert worst >= GATE, (
        f"r={radius} 的完美圆盘在最差相位下读成轴比 {worst:.3f} < {GATE} —— "
        "判据把面积/相位当成了形状,这正是旧 circularity 的病")


def test_the_floor_is_subtracted_not_ignored():
    """未减 floor 的原始离散**必须**随面积变;减掉之后**必须**基本不变。

    如果哪天有人「简化」掉 floor,这条会红 —— 它钉的是「零点是被推导出来的」
    这件事本身,不是某个具体数值。
    """
    # 单个圆盘的读数会随亚像素定心相位摆动(实测峰谷 0.016),
    # 所以每个面积在几个相位上取平均 —— 否则这条测试测到的是运气。
    offsets = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.37)]
    raws, excesses = [], []
    for area in (25, 100, 2000):
        rs, es = [], []
        for ox, oy in offsets:
            m = ellipse(area, cx=ox, cy=oy)
            rs.append(dispersion_of_mask(m))
            r = assess_mask(m, min_area_px=1)
            es.append(r.excess)
            assert r.floor == pytest.approx(dispersion_floor(int(m.sum())), abs=1e-9)
        raws.append(float(np.mean(rs)))
        excesses.append(float(np.mean(es)))

    raw_span = max(raws) - min(raws)
    exc_span = max(excesses) - min(excesses)
    assert raw_span > 0.05, (
        f"原始离散 {raws} 在这几个面积上没有明显的尺寸依赖 —— "
        "这条测试就没在测它该测的东西了,换几个面积")
    # 断言瞄的是「零点被减掉了」本身,不是某个具体残差值:
    # 若有人把 excess 直接换成 raw,这个比值会跳到 1.0。
    assert exc_span < 0.25 * raw_span, (
        f"减 floor 前尺寸量程 {raw_span:.4f},减完 {exc_span:.4f} —— "
        f"零点没被减掉,固定阈值会继续系统性地冤枉小团簇")


def test_the_floor_is_derived_from_geometry_not_fitted():
    """floor 必须等于「同面积完美圆盘的读数」。

    这是它**不是一个拟合常数**的操作性定义:能被独立重算出来。
    """
    for area in (60, 200, 900):
        m = ellipse(area)
        a = int(m.sum())
        radius = math.sqrt(a / math.pi)
        n = int(2 * radius + 9)
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        c = (n - 1) / 2.0
        independent = dispersion_of_mask(np.hypot(xx - c, yy - c) <= radius)
        assert dispersion_floor(a) == pytest.approx(independent, abs=0.02)


# ── 3. 不变性 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("rot", [0, 15, 30, 45, 60, 90])
def test_rotating_a_shape_does_not_change_how_round_it_is(rot):
    """**旧判据栽在这里**:4πA/P² 对轴对齐的正方形给 π/4=0.785,
    对同一个方块转 45° 给 0.393 —— 一倍的差,全是像素栅格贡献的。

    圆不圆是形状的性质,和它在栅格上怎么摆没有关系。
    """
    assert q_of(ellipse(600, ratio=0.6, rot_deg=rot)) == pytest.approx(0.6, abs=0.07)


def test_the_old_metric_ranked_a_square_above_a_circle():
    """把**被否掉的旧判据**钉成测试,免得它以「更简单」的名义回来。

    4πA/P_crack² 的上确界:轴对齐正方形 = π/4 = 0.7854(精确,与边长无关),
    数字化圆盘 = 4π²/64 = 0.6169。阈值 0.65 卡在两者之间 ⇒
    **圆盘一律不合格、方块一律合格。**
    """
    def crack_circularity(mask):
        p = np.pad(mask, 1)
        c = p[1:-1, 1:-1]
        per = int((c & ~p[:-2, 1:-1]).sum() + (c & ~p[2:, 1:-1]).sum()
                  + (c & ~p[1:-1, :-2]).sum() + (c & ~p[1:-1, 2:]).sum()) or 1
        return 4 * math.pi * int(mask.sum()) / per ** 2

    square = np.ones((21, 21), bool)
    disc = ellipse(2000)
    assert crack_circularity(square) == pytest.approx(math.pi / 4, abs=1e-6)
    assert crack_circularity(disc) < crack_circularity(square)
    assert crack_circularity(disc) < 0.65 < crack_circularity(square), (
        "旧阈值 0.65 不再夹在圆盘上确界与正方形之间了 —— "
        "如果口径变了,这段历史说明要一起改")

    # 新判据不犯这个错:方块比圆盘**更不圆**。
    assert q_of(disc) > q_of(square)


# ── 4. 它看得见 aspect 看不见的东西 ─────────────────────────────────────

def _aspect(mask) -> float:
    ys, xs = np.where(mask)
    eig = np.linalg.eigvalsh(np.cov(np.stack([xs - xs.mean(), ys - ys.mean()])))
    return math.sqrt(max(eig[0], 0.0)) / math.sqrt(max(eig[1], 1e-12))


@pytest.mark.parametrize("kw,label,aspect_blind", [
    (dict(lobes=3, lobe_amp=0.28), "三瓣", True),
    (dict(lobes=4, lobe_amp=0.25), "四瓣", True),
    (dict(notch=0.55), "单边缺口", False),
])
def test_it_sees_lobes_that_second_moments_call_perfectly_round(kw, label, aspect_blind):
    """这是留着这个判据的**唯一理由**。

    `aspect` 在「拉长」那一档比它强(实测 d′ 6.9 vs 3.2),所以两个都留。
    但对**瓣**,二阶矩是真的瞎:三瓣/四瓣形状的长短轴比 ≈1.00,
    因为 m=3 / m=4 的角向分量根本不进二阶矩。

    ⚠️ **单边缺口不属于「aspect 全瞎」那一类** —— 实测 aspect 0.829:
    缺一块会把质心挪走,二阶矩确实动了一点。第一版这条断言写成
    「aspect > 0.85」而红了,红得对:是断言把三种形状混为一谈,不是判据不行。
    所以这里分开钉,**单边缺口钉的是「rd 看得比 aspect 更清楚」**,
    而不是「aspect 看不见」。
    """
    m = ellipse(600, **kw)
    aspect, q = _aspect(m), q_of(m)
    if aspect_blind:
        assert aspect > 0.95, (
            f"{label} 的 aspect 是 {aspect:.3f} —— 二阶矩并没有瞎,"
            "那这个形状就不该被当成「aspect 看不见」的例子")
    assert q < 0.80, (
        f"{label} 被读成轴比 {q:.3f} —— 它没看见这个形状")
    assert q < aspect - 0.10, (
        f"{label}:rd 读 {q:.3f},aspect 读 {aspect:.3f} —— "
        "rd 没有比 aspect 看得更清楚,那它就没有存在价值")


def test_more_deformation_always_reads_less_round():
    """单调性:用户的用法是**比较相继几次哪一次更圆**,
    所以「更扁 ⇒ 分数更低」比任何一个阈值都重要。"""
    qs = [q_of(ellipse(600, ratio=r, rot_deg=17)) for r in (1.0, 0.9, 0.8, 0.7, 0.5)]
    assert all(a > b for a, b in zip(qs, qs[1:])), qs


# ── 5. 判不了的时候说判不了 ─────────────────────────────────────────────

def test_a_too_small_blob_gets_a_reason_not_a_number():
    """A<20 px 时像素化能让**完美的圆**读出 0.64–0.77 的轴比(95 分位,1% 噪声)。
    那个尺度上给一个数比不给更糟 —— 它看起来是测量结果。"""
    r = assess_mask(ellipse(12))
    assert not r.ok and r.axis_ratio is None
    assert "像素" in (r.reason or "")
    assert r.as_dict()["roundness_undecidable"]
    assert assess_mask(ellipse(200)).ok


def test_the_min_area_line_is_where_a_perfect_disc_stops_being_readable():
    """MIN_AREA_PX 不是一个随手写的数:在它以上,完美圆盘还读得出「圆」;
    在它以下,像素化的读数已经掉进会被当成「不圆」的区间。"""
    assert q_of(ellipse(MIN_AREA_PX + 4)) >= 0.75
    below = axis_ratio_from_dispersion(dispersion_floor(12))
    assert below < 0.75, (
        f"A=12 的完美圆盘 floor 相当于轴比 {below:.3f} —— 如果它已经 ≥0.75,"
        "面积下界就该往下挪了")
