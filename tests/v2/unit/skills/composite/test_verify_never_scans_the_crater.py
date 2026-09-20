"""脉冲和下压造成的落点必须同时到达地图与内存避让列表，验证阶段不能回到污染点。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.skills.composite import _tip_phases as tp  # noqa: E402


class _Progress:
    def __init__(self) -> None:
        self.partial_data: dict = {}


class _Executor:
    """只需要 progress.partial_data —— 脏点表就存在那里。"""

    def __init__(self) -> None:
        self.progress = _Progress()


# 独立生成的坐标用于测试扎针后的扫描避让。
_PULSE_XY = (1.25e-7, -2.75e-7)


def test_a_fired_spot_is_remembered():
    ex = _Executor()
    assert tp._dirty(ex) == []
    tp._mark_dirty(ex, *_PULSE_XY)
    assert tp._dirty(ex) == [_PULSE_XY]


def test_marking_the_same_spot_twice_does_not_duplicate_it():
    """幂等 —— 断点续跑会把同一步重放一次。"""
    ex = _Executor()
    tp._mark_dirty(ex, *_PULSE_XY)
    tp._mark_dirty(ex, *_PULSE_XY)
    assert tp._dirty(ex) == [_PULSE_XY]


def test_the_dirty_list_is_shared_across_phases_not_per_phase():
    """关键:记的人(pulse,前缀 A)和用的人(verify,前缀 B)不是同一个阶段。

    从前 `used` 存在 ``f"{prefix}:used"`` 下 —— 按阶段分隔,
    所以 verify 永远看不到 pulse 记的东西。
    """
    ex = _Executor()
    tp._mark_dirty(ex, *_PULSE_XY)
    keys = list(ex.progress.partial_data)
    # 断言的是**没有一个键带阶段前缀**,不是「只有一个键」。
    # map_marker_failures / _why 记录落库失败，见
    # ``_bump_marker_failure``),原来那条 ``keys == [_DIRTY_KEY]`` 当场红了 ——
    # 而红的不是它要防的那件事。钉住关系,别钉住清单长度。
    prefixed = [k for k in keys if ":" in k]
    assert not prefixed, (
        f"脏点表被存进了带阶段前缀的键 {prefixed} —— 那样别的阶段还是读不到。")
    assert tp._DIRTY_KEY in keys
    assert ":" not in tp._DIRTY_KEY


def test_it_survives_a_round_trip_through_json_shaped_state():
    """partial_data 会被 checkpointer 序列化,元组会变成表 —— 读回来还要能用。"""
    ex = _Executor()
    tp._mark_dirty(ex, *_PULSE_XY)
    # 模拟 checkpoint 往返:元组 → 表
    ex.progress.partial_data[tp._DIRTY_KEY] = [
        list(p) for p in ex.progress.partial_data[tp._DIRTY_KEY]]
    assert tp._dirty(ex) == [_PULSE_XY]
    tp._mark_dirty(ex, *_PULSE_XY)          # 往返之后仍须幂等
    assert tp._dirty(ex) == [_PULSE_XY]


def test_verify_and_level_ask_for_the_shared_list_not_an_empty_one():
    """源码级闸门:那两处**不许**再传空表。

    钉在源码上是因为这两行的错误形态就是「字面量 `[]`」—— 一个看起来完全正常、
    读代码时一扫而过的字面量,而它的后果是在坑上判针尖。
    """
    src = Path(tp.__file__).read_text(encoding="utf-8")
    # 注释里可以出现 used=[],代码行不行
    code_lines = [ln for ln in src.splitlines()
                  if "used=[]" in ln and not ln.lstrip().startswith("#")]
    assert not code_lines, (
        "还有调用点在传空的已用落点表:\n  " + "\n  ".join(code_lines)
        + "\n弄脏表面的人和要干净表面的人之间,信息必须通。")


def test_both_damaging_phases_record_into_the_shared_list():
    """pulse 和 poke 都要写进去 —— 只记一个,另一个的坑照样会被扫。"""
    src = Path(tp.__file__).read_text(encoding="utf-8")
    assert src.count("_mark_dirty(executor") >= 2, (
        "只有一个阶段在记脏点。脉冲(150 nm)和扎针(30 nm)都会弄脏表面。")


def test_special_tip_flow_is_fixed_too():
    """同形状在 make_special_tip 里有两处 —— 一起修,不留「下次再说」的。"""
    from mast.skills.composite import make_special_tip as mst

    src = Path(mst.__file__).read_text(encoding="utf-8")
    code_lines = [ln for ln in src.splitlines()
                  if "used=[]" in ln and not ln.lstrip().startswith("#")]
    assert not code_lines, "make_special_tip 里还有传空表的调用点:\n  " + "\n  ".join(code_lines)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
