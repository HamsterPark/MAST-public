"""正反扫一致性不应依赖像素数或单位缩放。

去趋势范数会随数组大小和物理单位变化，绝对阈值可能误伤小幅值的合成图。
测试覆盖像素数、单位和一维二维路径，不将某个守卫常量直接等同于正确行为。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.vision.tip_metrics import (  # noqa: E402
    _fwd_bwd_instability,
    trace_retrace_correlation,
)

#: 稳定针尖的判定线。判据本身是连续的,这里只要求「不翻类」——
#: 钉一个具体数值会把这条闸门变成第二个「绝对阈值」测试。
_STABLE = 0.05


def _frame(n: int, corrugation_m: float, *, seed: int = 0):
    """一帧「同样的画」,采样到 n×n。空间频率随边长缩放,所以内容与分辨率无关。"""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    return (1e-9
            + corrugation_m * np.sin(2 * np.pi * x / (n / 15))
            + 0.3 * corrugation_m * np.sin(2 * np.pi * y / (n / 11)))


def _pair(n: int, corrugation_m: float, *, noise_frac: float = 0.02):
    """一对稳定针尖该产生的正反扫:同一形貌 + 一点噪声。"""
    a = _frame(n, corrugation_m)
    rng = np.random.default_rng(0)
    return a, a + noise_frac * corrugation_m * rng.normal(0, 1, (n, n))


# ── 单位不变性:数学上必然,曾经不成立 ──────────────────────────────────

@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e9, 1e12])
def test_result_is_invariant_under_a_pure_unit_change(scale):
    """乘一个常数 = 换单位(米 → 纳米 → 皮米)。归一化互相关必须不动。

    这条最尖锐:``instab(x)`` 与 ``instab(x * 1e9)`` 如果不同,那么"这根针尖稳不稳"
    这个物理问题的答案就取决于你**用什么单位写下同一份数据**。
    """
    a, b = _pair(96, 1e-11)
    base = _fwd_bwd_instability(a, b)
    scaled = _fwd_bwd_instability(a * scale, b * scale)
    # 容差 1e-4:``_to_2d`` 把数据转成 **float32**,所以换单位会带来 ~1e-7 的
    # 浮点抖动。那是精度噪声,不是行为差异 —— 而这条要抓的回归是 0.0005 → 1.0,
    # 差了四个数量级。把容差收到 1e-9 只会让这条测试报告 float32 的存在。
    assert scaled == pytest.approx(base, abs=1e-4), (
        f"数据乘以 {scale:g}(纯换单位)之后判据从 {base:.4f} 变成 {scaled:.4f}")


# ── 尺寸不变性 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [32, 48, 64, 96, 128, 256, 512])
def test_a_stable_frame_reads_stable_at_every_frame_size(n):
    """同一幅画采样到不同边长,「稳定」这个结论不许随边长翻转。

    旧守卫下 32/48/64/96/128 px 全部返回 1.0(完全不相关),256/512 px 返回 0.0001
    —— 判据在中间某处**换了答案**,而画没变。
    """
    instab = _fwd_bwd_instability(*_pair(n, 1e-11))
    assert instab < _STABLE, (
        f"{n}×{n} 上一帧稳定的图被判成不一致({instab:.4f}) —— "
        f"边长不该改变这个结论")


@pytest.mark.parametrize("corrugation_m", [1e-12, 5e-12, 2e-11, 1e-10, 1e-9])
def test_a_faint_but_real_corrugation_is_not_called_uncorrelated(corrugation_m):
    """起伏小 ≠ 正反扫不一致。

    1 pm 的起伏在 STM 上是真实存在的信号(Au(111) 单原子台阶 236 pm,原子级起伏
    可到 pm 量级)。判据不该因为"信号弱"就宣布"针尖乱走"—— 那是两件事。
    信噪比不足该由调用方的**弃权门**处理(见 CheckLineQuality 的
    min_corrugation_m),不是由一致性判据谎报成"不一致"。
    """
    instab = _fwd_bwd_instability(*_pair(64, corrugation_m))
    assert instab < _STABLE, (
        f"起伏 {corrugation_m * 1e12:g} pm 的稳定帧被判成不一致({instab:.4f})")


# ── 但真正没有信号的时候,仍然要拒 ──────────────────────────────────────

def test_a_constant_side_is_still_refused():
    """守卫存在的**唯一**正当理由是防除零 —— 那个必须留着。

    一侧是常数 ⇒ 去趋势后范数恰好 0 ⇒ 没有可相关的东西。这时返回 1.0
    (完全不相关)是对的:它是「这里没有信息」,不是「针尖很稳」。
    """
    a, _ = _pair(64, 1e-11)
    constant = np.full((64, 64), 1e-9)
    assert _fwd_bwd_instability(constant, a) == 1.0
    assert _fwd_bwd_instability(a, constant) == 1.0


# ── 一维 / 二维两半必须给同一个答案 ────────────────────────────────────

@pytest.mark.parametrize("n", [64, 256])
def test_the_1d_and_2d_halves_agree_on_the_same_content(n):
    """``trace_retrace_correlation`` 一维用 np.correlate、二维转发给
    ``_fwd_bwd_instability``。两半用**不同的守卫常量**曾是这个 bug 的根:
    一维 1e-30、二维 1e-9,于是同一份物理内容在两条路上答案相反。
    """
    a, b = _pair(n, 1e-11)
    two_d = trace_retrace_correlation(a, b)
    one_d = trace_retrace_correlation(a[n // 2], b[n // 2])
    assert two_d > 0.9, f"二维路径把稳定帧判成 {two_d:.4f}"
    assert one_d > 0.9, f"一维路径把稳定线判成 {one_d:.4f}"


def test_the_guard_constant_is_not_in_metres_territory():
    """防除零阈值应远离测试所覆盖的非零信号幅度。
    结构性检查补充单位缩放行为测试，防止绝对阈值重新进入有效数据范围。"""
    import inspect
    src = inspect.getsource(_fwd_bwd_instability)
    import re
    guards = re.findall(r"na\s*<\s*([0-9.eE+-]+)", src)
    assert guards, "守卫不见了 —— 除零保护是必须留的"
    for g in guards:
        assert float(g) <= 1e-20, (
            f"防除零的 epsilon 写成了 {g} —— 那和真实物理量(米,~1e-9)同量级,"
            f"于是它会在**好数据**上开火。epsilon 要远离数据的量纲。")
