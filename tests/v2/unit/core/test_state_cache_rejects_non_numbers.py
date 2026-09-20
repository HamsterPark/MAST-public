"""状态缓存只应接受可验证的标量数值。

技能返回中的嵌套元组、非法对象或不可用值不能直接进入缓存并被后续传感器强转。
生产方负责规范化，缓存接收侧也应验证，防止单个调用方遗漏检查污染共享状态。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.state import InstrumentState, coerce_number  # noqa: E402


# ── coerce_number 本身 ─────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    (2.0e-10, 2.0e-10),
    (0, 0.0),
    (-3, -3.0),
    ((2.0e-10,), 2.0e-10),      # 单元素元组的独立合成标量
    ([2.0e-10], 2.0e-10),
    (((2.0e-10,),), 2.0e-10),   # 双层包裹也解得开
])
def test_numbers_and_their_tuple_wrappings_come_through(raw, want):
    assert coerce_number(raw, field="current_a") == pytest.approx(want)


@pytest.mark.parametrize("raw", [
    ("", b"\x00", [1.0]),        # 整个三段信封
    (1.0, 2.0),                  # 两个数 —— 到底是哪一个?判不了
    "2.0e-10",                   # 字符串:能 float(),但来源可疑
    None,
    object(),
    [],
])
def test_anything_else_is_refused_not_guessed(raw):
    """看不懂就返回 None。**不许**挑一个看起来合理的。

    注意字符串也拒:``float("2.0e-10")`` 会成功,但一个字符串出现在这里说明
    上游走错了路,悄悄接住它等于把那个 bug 埋起来。
    """
    assert coerce_number(raw, field="current_a") is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_is_refused(bad):
    """NaN / inf 不是测量值。

    第一版 `coerce_number` 自己实现判据,
    ``float(nan)`` 不抛 ⇒ NaN 被当成合法电流写进缓存 ⇒ 传感器报
    ``status="ok"``、值是 NaN。那是「不知道」伪装成「知道」,比报 error 更坏
    —— error 至少还会被看见。

    改成委托 `mast.io.nanonis_files.scalar_float` 之后才拒掉。
    **删这条之前先回答**:一个 NaN 电流流进阈值比较会得到什么?
    (答案是 `nan >= thr` 恒为 False —— 一条永远「不达标」而不报错的路。)
    """
    assert coerce_number(bad, field="current_a") is None


def test_a_multi_element_reply_is_refused_not_indexed():
    """两个数的回包 ⇒ 拒,**不许**取第 0 个。

    取 seq[0] 会在双通道回包上悄悄挑一路,而挑错哪一路是看不出来的。
    """
    assert coerce_number((1.0, 2.0), field="current_a") is None
    assert coerce_number([1.0, 2.0, 3.0], field="bias_v") is None


def test_bool_is_refused():
    """``float(True) == 1.0`` —— 一个开关状态会变成「电流 1 安培」。

    写进缓存之后,下游没有任何办法把它认出来。
    """
    assert coerce_number(True, field="current_a") is None
    assert coerce_number(False, field="current_a") is None


# ── apply_patch:收侧的闸门 ─────────────────────────────────────────────

def _state() -> InstrumentState:
    st = InstrumentState.__new__(InstrumentState)
    from mast.core.types import HardwareState
    st._cache = HardwareState()
    return st


def test_a_tuple_current_never_reaches_the_cache():
    """元组不能作为可转换为浮点数的标量写入状态缓存。"""
    st = _state()
    st.apply_patch(current_a=1.0e-10)
    st.apply_patch(current_a=(2.0e-10,))
    got = st.snapshot().current_a
    assert isinstance(got, float), (
        f"缓存里的 current_a 是 {got!r}({type(got).__name__}) —— "
        "环境传感器对它 float() 会抛 TypeError,不应让类型错误进入环境故障路径；该路径"
        "会把这个 TypeError 当成环境硬故障:退针 + 挂一个解不开的急停闩。")


def test_a_refused_value_leaves_the_previous_one_alone():
    """拒绝 ≠ 清零。陈的真值好过一个编出来的 0.0。"""
    st = _state()
    st.apply_patch(current_a=1.0e-10)
    st.apply_patch(current_a=("", b"\x00", [9.9]))
    assert st.snapshot().current_a == pytest.approx(1.0e-10), (
        "被拒的写入把上一个真值冲掉了 —— 那等于用「不知道」覆盖了「知道」。")


def test_every_numeric_field_is_guarded_not_just_current():
    """闸门要盖住所有数值字段 —— 只挡 current_a 是在修一个症状。"""
    st = _state()
    st.apply_patch(bias_v=0.5, z_pos_m=1e-7, setpoint_a=5e-11)
    st.apply_patch(bias_v=(1.5,), z_pos_m=("bad",), setpoint_a=[7e-11])
    snap = st.snapshot()
    assert snap.bias_v == pytest.approx(1.5), "1-元组该被解包"
    assert snap.z_pos_m == pytest.approx(1e-7), "非数该被拒,旧值该留着"
    assert snap.setpoint_a == pytest.approx(7e-11)


def test_non_numeric_fields_still_pass_through_untouched():
    """布尔 / 字符串字段不受这道闸门影响 —— 别把它变成一道全局阻塞。"""
    st = _state()
    st.apply_patch(z_controller_on=True, z_controller_name="Z", scan_running=False)
    snap = st.snapshot()
    assert snap.z_controller_on is True
    assert snap.z_controller_name == "Z"
    assert snap.scan_running is False, "False 必须写进去(StopScan 之后就靠它)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
