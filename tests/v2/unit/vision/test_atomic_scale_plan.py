"""尺度闸的纯函数层:``plan_scale`` / ``min_pixels_for_scale`` / ``ALL_REASONS``。

S2 偏压序列设计 §5「纯函数层」。三件事:

1. **边界钉死**。满权重档是**严格小于** 0.02 —— 256 px 要 < 5.12 nm、512 px 要
   < 10.24 nm。差一个 ulp 就是「本来判得出的一帧被判成判不了」或反过来。
2. **归一处的证据**。``special_tip_workflow.scale_problem()`` 改调 ``plan_scale``
   之后,它的输出**一字不改**。本文件里那几条中文串是**改之前**跑出来的真实输出
   (2026-08-14 直接从旧实现 dump),不是照着新代码补的 —— 这是「归一处而不是抄
   第二份」唯一站得住的证据。
3. **出局词闭集不许腐烂**。``ALL_REASONS`` 与 ``assess_atomic_phase`` 源码里的字面量
   对账:新增一个出局词而忘了加进闭集 ⇒ 这里红,而不是在下游的三态映射里静默
   落进兜底档。
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from mast.vision import atomic_phase
from mast.vision.atomic_phase import (
    ALL_REASONS,
    SCALE_FULL_NMPP,
    SCALE_OFF_NMPP,
    min_pixels_for_scale,
    plan_scale,
    scale_gate,
)

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of

NM = 1e-9


# ── plan_scale:三态与边界 ────────────────────────────────────────────────────

@pytest.mark.parametrize("size_nm,pixels,expect", [
    # 256 px:满权重的上界是 5.12 nm,**开区间**
    (5.11, 256, "full"),
    (5.12, 256, "reduced"),      # 正好 0.0200 —— 严格小于,过不去
    (5.0, 256, "full"),          # 用户配方(余量只有 2.4%)
    # 512 px:上界翻倍到 10.24 nm,同样开区间
    (10.23, 512, "full"),
    (10.24, 512, "reduced"),
    (5.0, 512, "full"),          # atomic_verify 档:余量 2×
    # 过渡带的另一头是**闭上界** 0.05
    (12.8, 256, "reduced"),      # 正好 0.0500
    (12.81, 256, "off"),
    # 出厂 atomic 档的上半段判不出原子相(设计 §1.4 的关键算术)
    (10.0, 256, "reduced"),
    (50.0, 512, "off"),
])
def test_plan_scale_three_states_at_the_boundaries(size_nm, pixels, expect):
    nmpp, scale, problem = plan_scale(size_nm * NM, pixels)
    assert scale == expect
    assert nmpp == pytest.approx(size_nm / pixels)
    # problem 非空 ⟺ 不是满权重档 —— 「没问题」不许出现在有问题的档上
    assert bool(problem) is (expect != "full")


def test_plan_scale_agrees_with_scale_gate():
    """闸只有一个:``plan_scale`` 的 scale 必须就是 ``scale_gate`` 的输出。"""
    for size_nm in (1.0, 5.0, 5.12, 8.0, 10.24, 12.8, 12.81, 50.0, 200.0):
        for px in (128, 256, 512, 1024):
            nmpp, scale, _ = plan_scale(size_nm * NM, px)
            assert scale == scale_gate(nmpp), (size_nm, px)


@pytest.mark.parametrize("size_m,pixels", [
    (0.0, 256), (-5e-9, 256), (float("nan"), 256), (float("inf"), 256),
    (5e-9, 0), (5e-9, -256), (5e-9, None), ("5nm", 256),
])
def test_plan_scale_refuses_impossible_parameters(size_m, pixels):
    """算不出 nm/px 时 scale 是 ``None`` —— 「不知道」不是「没问题」。

    这一条直接对着 ``unknown_is_not_an_answer``:回一个 "full" 或 "" 会让下游
    以为这组参数没问题,而判据到时候只会说 ``unknown_pixel_size``。
    """
    nmpp, scale, problem = plan_scale(size_m, pixels)
    assert scale is None
    assert nmpp is None
    assert problem, "参数不成立却给了一句「没问题」"


def test_plan_scale_never_rewrites_the_frame():
    """闸只回答「行不行」,不替用户改视野 —— 帧宽是意图,偷偷改掉等于换了被测对象。"""
    sig = inspect.signature(plan_scale)
    assert list(sig.parameters) == ["size_m", "pixels"]
    nmpp, scale, problem = plan_scale(50.0 * NM, 256)
    assert scale == "off"
    # 返回值里没有任何「建议视野」字段能被误当成新的 size 下发
    assert isinstance(nmpp, float) and isinstance(problem, str)


# ── min_pixels_for_scale:floor+1,不是 ceil ──────────────────────────────────

@pytest.mark.parametrize("size_nm,expect", [
    (5.12, 257),     # 5.12/0.02 = 256 整 ⇒ ceil 会给 256(过不了门),要 257
    (10.24, 513),    # 同上,512 px 正好卡在门槛
    (5.0, 251),      # 5.0/0.02 = 250 整
    (8.0, 401),
    (5.11, 256),     # 非整数:floor+1 == ceil
    (2.0, 101),
])
def test_min_pixels_is_the_smallest_that_actually_passes(size_nm, expect):
    n = min_pixels_for_scale(size_nm * NM)
    assert n == expect
    # 自证:算出来的这个数**真的**过得了门,而少一个就过不了。
    assert plan_scale(size_nm * NM, n)[1] == "full"
    assert plan_scale(size_nm * NM, n - 1)[1] != "full"


def test_min_pixels_honours_a_custom_target():
    n = min_pixels_for_scale(12.8 * NM, target=SCALE_OFF_NMPP)
    assert plan_scale(12.8 * NM, n)[0] < SCALE_OFF_NMPP


@pytest.mark.parametrize("bad", [0.0, -1e-9, float("nan"), float("inf"), None, "x"])
def test_min_pixels_returns_zero_for_impossible_input(bad):
    assert min_pixels_for_scale(bad) == 0


def test_min_pixels_rejects_impossible_target():
    assert min_pixels_for_scale(5e-9, target=0.0) == 0
    assert min_pixels_for_scale(5e-9, target=float("nan")) == 0


# ── ALL_REASONS:闭集不许腐烂 ────────────────────────────────────────────────

def test_all_reasons_matches_the_literals_in_the_criterion_source():
    """闭集与 ``assess_atomic_phase`` 源码里的出局词字面量必须一一对应。

    **不是**把常量抄两遍:这里扫的是判据函数**自己**的源码,新增一个
    ``reasons.append("…")`` / ``_fail("…")`` 而没更新 ``ALL_REASONS`` 就当场红。
    """
    src = source_of(atomic_phase.assess_atomic_phase)
    found = set(re.findall(r'_fail\(\s*"([a-z_]+)"', src))
    found |= set(re.findall(r'reasons\.append\(\s*"([a-z_]+)"\s*\)', src))
    assert found == set(ALL_REASONS), (
        f"闭集与源码不符:源码多出 {found - set(ALL_REASONS)}，"
        f"闭集多出 {set(ALL_REASONS) - found}")


def test_all_reasons_has_no_duplicates():
    assert len(ALL_REASONS) == len(set(ALL_REASONS))


def test_warnings_are_not_reasons_except_scale_reduced():
    """``scale_reduced`` 同时是 warning 和 reason(证据不足);其余 warning 不进闭集。"""
    src = source_of(atomic_phase.assess_atomic_phase)
    warns = set(re.findall(r'warns\.append\(\s*"([a-z_]+)"\s*\)', src))
    assert warns & set(ALL_REASONS) == {"scale_reduced"}


# ── 归一处的证据:scale_problem 的输出一字不改 ────────────────────────────────
#
# 下面这些串是 2026-08-14 从**改之前**的 special_tip_workflow.scale_problem()
# 直接 dump 出来的。改成调 plan_scale 之后它们必须逐字符相同 —— 措辞保留、
# 判定归一处。(注意 0.03125 格式化成 "0.0312" 不是 "0.0313":Python 的
# round-half-to-even。照抄真实输出而不是照着心算写,这条差异才不会被写反。)

_FROZEN_SCALE_PROBLEM: dict[tuple[float, int], str] = {
    (5.0, 256): "",
    (2.0, 256): "",
    (10.0, 512): "",
    (8.0, 256): (
        "评估帧 0.0312 nm/px 落在过渡带 [0.02, 0.05] —— 判据会给出结论但证据强度"
        "不足以当验收依据。建议 8 nm 用 401 px 以上。"),
    (5.12, 256): (
        "评估帧 0.0200 nm/px 落在过渡带 [0.02, 0.05] —— 判据会给出结论但证据强度"
        "不足以当验收依据。建议 5.12 nm 用 257 px 以上。"),
    (10.24, 512): (
        "评估帧 0.0200 nm/px 落在过渡带 [0.02, 0.05] —— 判据会给出结论但证据强度"
        "不足以当验收依据。建议 10.24 nm 用 513 px 以上。"),
    (50.0, 256): (
        "评估帧 50 nm / 256 px = 0.1953 nm/px，超过 0.05 nm/px —— 这个尺度上晶格"
        "物理上不可分辨，判据只会说「判不了」。请缩小视野或加大像素数。"),
}


@pytest.mark.parametrize("key", sorted(_FROZEN_SCALE_PROBLEM))
def test_scale_problem_wording_is_unchanged(key):
    from mast.core.special_tip_workflow import AtomicTipWorkflow

    size_nm, pixels = key
    wf = dataclasses.replace(AtomicTipWorkflow(),
                             eval_frame_nm=size_nm, eval_pixels=pixels)
    assert wf.scale_problem() == _FROZEN_SCALE_PROBLEM[key]


def test_scale_problem_and_plan_scale_never_disagree():
    """措辞是两份,**判定只有一份**:有没有问题必须与 plan_scale 的 scale 一致。"""
    from mast.core.special_tip_workflow import AtomicTipWorkflow

    for size_nm in (1.0, 2.0, 5.0, 5.12, 8.0, 10.0, 12.8, 12.81, 50.0):
        for px in (128, 256, 512):
            wf = dataclasses.replace(AtomicTipWorkflow(),
                                     eval_frame_nm=size_nm, eval_pixels=px)
            _, scale, _ = plan_scale(size_nm * NM, px)
            assert bool(wf.scale_problem()) is (scale != "full"), (size_nm, px)


def test_eval_pixel_size_matches_plan_scale():
    """报出来的 nm/px 与用来判定的 nm/px 是同一个数(不是两处各算一遍)。"""
    from mast.core.special_tip_workflow import AtomicTipWorkflow

    for size_nm, px in ((5.0, 256), (8.0, 256), (5.0, 512), (50.0, 256)):
        wf = dataclasses.replace(AtomicTipWorkflow(),
                                 eval_frame_nm=size_nm, eval_pixels=px)
        assert wf.eval_pixel_size_nm() == plan_scale(size_nm * NM, px)[0]


# ── 已知的判据边界:尺度门过不了零信息注入检验 ────────────────────────────────

def _hex_lattice(*, n: int, nmpp: float, period_nm: float = 0.2494,
                 amp: float = 10e-12, noise: float = 2e-12, seed: int = 0):
    """一帧六角晶格的高度图(米)。三组波矢相隔 60°,与判据自己的合成语料同参数。"""
    import math

    import numpy as np

    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xs, ys = x * nmpp, y * nmpp
    k = 2.0 * math.pi / period_nm
    h = np.zeros((n, n), dtype=np.float64)
    for deg in (0.0, 60.0, 120.0):
        th = math.radians(deg)
        h += np.cos(k * (xs * math.cos(th) + ys * math.sin(th)))
    h = h / 3.0 * amp
    return h + np.random.default_rng(seed).normal(0.0, noise, h.shape)


def _upsample_bilinear(h, factor: int = 2):
    """双线性升采样 —— **零新信息**,只是插值。"""
    import numpy as np

    n = h.shape[0]
    src = np.arange(n, dtype=np.float64)
    dst = np.linspace(0.0, n - 1.0, n * factor)
    rows = np.stack([np.interp(dst, src, h[r, :]) for r in range(n)])
    return np.stack([np.interp(dst, src, rows[:, c]) for c in range(rows.shape[1])],
                    axis=1)


@pytest.mark.xfail(
    strict=True,
    reason=("尺度门只看 nm/px,分不开真采样与插值 —— 这是判据的**已知边界**,"
            "不是待修的 bug。钉成 xfail 是因为不写的话下一个人会以为它没有边界。"
            "要让这条转绿,判据得学会问「这些采样点真的携带信息吗」"
            "(例如看高频段的能量塌陷),那是改判据,需要重新过全部四检验。"))
def test_upsampling_does_not_change_atomic_verdict():
    """同一帧的原生版与双线性升采样版,verdict 应当相同 —— **今天不成立**。

    零信息注入检验(设计 §D7-①):10 nm / 256 px = 0.0391 nm/px 落在过渡带
    ⇒ 「判不了」;把它升采样 ×2 记成 512 px,nm/px 变成 0.0195 ⇒ 满权重档
    ⇒ verdict 翻成一个真结论,**纯粹靠插值**。

    仪器侧有等价物:同一帧宽把 pixels 从 256 提到 512 而不动 line_time ——
    驻留砍半,多出来的采样点不携带新信息,而 nm/px 照样变好看。这就是
    ``core.scan_policy`` 的 ``atomic_verify`` 档坚持「加像素必须同比加 line_time」
    的**判据学**理由,不只是护针尖。
    """
    from mast.vision.atomic_phase import assess_atomic_phase

    native = _hex_lattice(n=256, nmpp=10.0 / 256)
    up = _upsample_bilinear(native, 2)

    a = assess_atomic_phase(native, nm_per_px=10.0 / 256)
    b = assess_atomic_phase(up, nm_per_px=10.0 / 512)
    assert a.scale == "reduced" and b.scale == "full"     # 前提:确实翻了档
    assert a.passed == b.passed, (
        f"零信息注入改变了 verdict: 原生 passed={a.passed} {a.reasons} → "
        f"升采样 passed={b.passed} {b.reasons}")


def test_operator_recipe_still_passes_the_gate():
    """用户配方(5 nm / 256 px)的余量只有 2.4% —— 它必须仍在满权重档内。"""
    from mast.core.special_tip_workflow import AtomicTipWorkflow

    wf = AtomicTipWorkflow()
    nmpp, scale, problem = plan_scale(wf.eval_frame_nm * NM, wf.eval_pixels)
    assert scale == "full" and problem == ""
    assert nmpp == pytest.approx(0.01953125)
    assert nmpp / SCALE_FULL_NMPP == pytest.approx(0.9766, abs=1e-4)
