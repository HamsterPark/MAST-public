"""acquired_row_mask 为各视觉调用方提供统一的已采集行判定。

部分帧、NaN、空输入、一维提升及多帧交集应分别覆盖，
避免把尚未采集的数据误当作有效图像并产生质量判决。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.vision.frame_validity import acquired_row_mask  # noqa: E402

N = 16


def _half_scanned(n=N, last_full=7):
    """扫到一半的帧:``last_full`` 之前整行有数,``last_full+1`` 是**正在扫的那一行**
    (一半有数一半 NaN),再往后整行 NaN。"""
    a = np.arange(n * n, dtype=np.float64).reshape(n, n) * 1e-12
    a[last_full + 2:] = np.nan          # 完全没扫到
    a[last_full + 1, n // 2:] = np.nan  # 正在扫的那一行
    return a


# ══════════════════════════════════════════════════════════════════════
# ⭐ 承重的那一条:「**整行**都有限」 ≠ 「这一行**有任何**有限值」
# ══════════════════════════════════════════════════════════════════════

def test_the_row_being_scanned_is_excluded_all_not_any():
    """正在写入、只含部分有限值的行必须排除。

    把含 NaN 的不完整行送入平面或直线拟合会污染结果。显式比较 all(axis=1)
    与 any(axis=1)，证明后者会错误接纳该行，避免测试只验证正确实现的正例。
    """
    a = _half_scanned(last_full=7)
    scanning_row = 8                      # 一半有数、一半 NaN 的那一行

    mask = acquired_row_mask(a)
    assert mask[scanning_row] == False, (  # noqa: E712 — 显式对比,读起来是那句话
        "正在扫的那一行进了掩码 —— 判据被写成了「有任何有限值」。"
        "它一个人就能让整幅拟合返回 NaN。")
    assert mask[:8].all(), "已经扫完整的行被排除了"
    assert not mask[9:].any(), "完全没扫到的行进了掩码"
    assert int(mask.sum()) == 8

    # 被否掉的写法:它**会**把那一行算进来,而且只多算一行 —— 正是 41 vs 42。
    rejected = np.isfinite(a).any(axis=1)
    assert rejected[scanning_row] == True, (  # noqa: E712
        "构造失效:这一行在两种读法下没有区别,本测试分不出 .all 和 .any")
    assert int(rejected.sum()) == int(mask.sum()) + 1


# ══════════════════════════════════════════════════════════════════════
# 目前没有调用方的那几条分支
# ══════════════════════════════════════════════════════════════════════

def test_multiple_frames_take_the_intersection():
    """多帧 ⇒ **交集**:一侧缺的那一行不进掩码。

    理由是物理的,不是防御性编程:正反扫一致性要**逐点比**,只有**两侧都扫到**
    的行才谈得上「一致不一致」。任一侧缺了那一行,这一行就没有可比的东西。
    """
    f = _half_scanned(last_full=7)     # 0..7 完整
    b = _half_scanned(last_full=4)     # 0..4 完整
    mask = acquired_row_mask(f, b)
    assert int(mask.sum()) == 5
    assert mask[:5].all() and not mask[5:].any(), (
        "两帧掩码没有取交集 —— 只有一侧扫到的行被当成可比了")
    # 交集,不是并集:单独看 f 有 8 行
    assert int(acquired_row_mask(f).sum()) == 8


def test_a_1d_line_is_promoted_to_one_row():
    """一维输入(一条扫描线)提升成 ``(1, N)`` —— 「只有 1 行」在这里是正常的。

    ``PreScanCheck`` 的缓冲区那条路交的就是一条线。第一版把「至少几行」的下限
    一律写成 2,于是整条缓冲区路径全部弃权(8 条测试当场变红):
    **一道对合法输入永远开火的门,和没有门一样坏。**
    """
    good = np.arange(N, dtype=np.float64)
    assert acquired_row_mask(good).tolist() == [True]

    bad = good.copy()
    bad[3] = np.nan
    assert acquired_row_mask(bad).tolist() == [False]


def test_degenerate_inputs_return_an_empty_mask_without_raising():
    """空数组 / 维度不对 ⇒ **长度 0 的掩码**,不抛异常。

    调用方拿它去 ``arr[mask]``,一个长度 0 的掩码让「没有可用行」自然地流到
    下游的弃权分支;抛异常则会把「这一帧没准备好」变成「技能坏了」——
    而这两件事指向完全不同的下一步。
    """
    for bad in (np.zeros((0, 0)), np.zeros((0, 5)), np.zeros((2, 2, 2)),
                np.array([])):
        mask = acquired_row_mask(bad)
        assert mask.dtype == np.bool_ and mask.size == 0, bad.shape
    assert acquired_row_mask().size == 0          # 一个参数都不给


# ══════════════════════════════════════════════════════════════════════
# 变异验证:先证明自己动了手,再看测试红不红
# ══════════════════════════════════════════════════════════════════════

def test_mutating_all_to_any_turns_the_load_bearing_test_red():
    """把 ``.all(axis=1)`` 改成 ``.any(axis=1)`` ⇒ 上面第一条**必须红**。

    没有这条,「第一条测试真的挡得住那次改写」只是我们的信念。
    [[mutation_must_prove_it_landed]]:合格的输出要有「变异已应用」+「测试红了」
    两半 —— 今晚已经因为只看后半而误判过一次,所以这里**先断言替换真的落到了
    文件里**(计数 == 1),再跑。
    """
    src = Path(_MASTV2_ROOT) / "mast" / "vision" / "frame_validity.py"
    original = src.read_text(encoding="utf-8")
    needle = "np.isfinite(a).all(axis=1)"
    assert original.count(needle) == 1, (
        f"变异目标不在文件里(找到 {original.count(needle)} 处)—— "
        "实现改过了,这条变异测试已经失效,请更新 needle。")

    mutated = original.replace(needle, "np.isfinite(a).any(axis=1)")
    assert mutated != original, "变异没有落到文件里"
    try:
        src.write_text(mutated, encoding="utf-8")
        # 子进程跑,避免污染本进程已导入的模块
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             f"{Path(__file__)}::test_the_row_being_scanned_is_excluded_all_not_any"],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": _MASTV2_ROOT},
            cwd=str(Path(_MASTV2_ROOT).parent))
        assert proc.returncode != 0, (
            "把 .all 换成 .any 之后那条测试仍然是绿的 —— 它没有钉住这个区分。\n"
            f"stdout:\n{proc.stdout[-2000:]}")
    finally:
        src.write_text(original, encoding="utf-8")

    # 改回来之后必须绿(否则上面那次红说明不了是变异造成的)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{Path(__file__)}::test_the_row_being_scanned_is_excluded_all_not_any"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": _MASTV2_ROOT},
        cwd=str(Path(_MASTV2_ROOT).parent))
    assert proc.returncode == 0, (
        f"恢复之后测试没有变绿,文件可能没恢复干净:\n{proc.stdout[-2000:]}")

# 另一种未采集形态：原始扫描缓冲中的全零行。
def test_all_zero_rows_from_the_live_buffer_are_not_acquired():
    """``.sxm`` 里未扫的行是 NaN，**活体帧缓冲里是零** —— 两种形态。

    只查 NaN 会把那片零当成真数据（零是有限值，``isfinite`` 一路放行），
    于是一张扫到一半的活体帧被当成整帧去判。实测 ``assess_atomic_phase``：
    同一批数据含 40% 未扫时 **162.4**，只取已扫的 143 行 **818.4** —— 差 5 倍。
    而 ``concentration_min`` 出厂只有 20：好针尖的半张帧会掉到线下被判成
    「没有晶格」，上层于是去「修」一根本来就好的针尖。
    """
    img = np.zeros((10, 32))
    img[4:] = np.random.RandomState(0).randn(6, 32) * 1e-11 - 1.5e-7
    got = acquired_row_mask(img)
    assert list(np.flatnonzero(got)) == [4, 5, 6, 7, 8, 9], (
        "活体缓冲里那片零被当成了已扫的行")


def test_an_all_zero_frame_is_left_alone_because_measured_zero_is_not_unmeasured():
    """整帧全零无法区分零信号与未采集，不能据此裁行；只有非零行作对照时，零行才提供空白证据。"""
    got = acquired_row_mask(np.zeros((8, 32)))
    assert got.all(), "整帧全零被裁光了 —— dead_flat 会被误报成 insufficient_data"


def test_zero_rows_and_nan_rows_are_both_dropped_in_one_frame():
    """共用裁行函数应处理 NaN 与零填充混合的输入。"""
    img = np.random.RandomState(1).randn(9, 32) * 1e-11 - 1.5e-7
    img[0:2] = 0.0
    img[7:9] = np.nan
    assert list(np.flatnonzero(acquired_row_mask(img))) == [2, 3, 4, 5, 6]
