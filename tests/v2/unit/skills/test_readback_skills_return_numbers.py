"""回读技能必须返回标量数值，不能让单元素元组污染状态缓存和后续监测。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.skills.builtins.bias import GetBias, GetCurrent  # noqa: E402
from mast.skills.builtins.zcontrol import GetZPosition  # noqa: E402


class _Ctx:
    """只回一次 safe_call —— 回什么由测试决定。"""

    def __init__(self, return_value) -> None:
        self._rv = return_value

    def safe_call(self, *a, **kw):
        return SimpleNamespace(error="", return_value=self._rv, method=a[0] if a else "")


#: (技能类, data 里的键)
_READBACKS = [
    (GetCurrent, "current_a"),
    (GetBias, "bias_v"),
    (GetZPosition, "z_pos_m"),
]


def _envelope(body):
    """回包使用协议包装结构。"""
    return ("", b"\x00\x00", body)


@pytest.mark.parametrize("skill_cls,key", _READBACKS)
def test_a_bare_number_comes_through(skill_cls, key):
    r = skill_cls().execute(_Ctx(_envelope([1.25e-10])), {})
    assert r.success is True
    assert r.data[key] == pytest.approx(1.25e-10)
    assert isinstance(r.data[key], float)


@pytest.mark.parametrize("skill_cls,key", _READBACKS)
def test_the_real_machine_tuple_wrapping_is_unwrapped(skill_cls, key):
    """单元素元组回包必须解包成数值。"""
    r = skill_cls().execute(_Ctx(_envelope([(1.25e-10,)])), {})
    assert r.success is True, f"1-元组被当成失败了:{r.error}"
    got = r.data[key]
    assert isinstance(got, float), (
        f"{skill_cls.__name__} 返回了 {got!r}({type(got).__name__}) 而不是一个数。"
        "这个值会进状态缓存,而环境传感器对它 float() 会抛 —— "
        "2026-08-13 那一次的代价是退针 + 锁死整台仪器。")
    assert got == pytest.approx(1.25e-10)


@pytest.mark.parametrize("skill_cls,key", _READBACKS)
def test_a_value_that_is_not_a_number_fails_honestly(skill_cls, key):
    """看不懂的值要**如实报失败**,不许当成一个读数往下传。

    尤其不能报成「电流异常 / 偏压异常」—— 那会把人送去查仪器,
    而问题在解析。
    """
    r = skill_cls().execute(_Ctx(_envelope([{"不是": "数"}])), {})
    assert r.success is False
    assert "不是一个数" in (r.error or ""), (
        f"失败原文没说清是解析问题:{r.error!r}")


@pytest.mark.parametrize("skill_cls,key", _READBACKS)
def test_an_empty_body_does_not_raise_IndexError(skill_cls, key):
    """断连瞬间的截断回包:body 是空表。

    修改前那一行是 ``parsed[2][0]``,对空表直接抛 ``IndexError`` —— 技能以一句
    看不懂的 IndexError 失败。**读不到就是读不到**,该说人话。
    """
    r = skill_cls().execute(_Ctx(_envelope([])), {})
    assert r.success is False
    assert "IndexError" not in (r.error or "")
    assert (r.error or ""), "失败了却没有说明原因"


@pytest.mark.parametrize("skill_cls,key", _READBACKS)
def test_a_truncated_two_part_envelope_is_not_taken_as_a_reading(skill_cls, key):
    """两段回包 ``('', b'…')`` —— 最贴合 2026-08-13 那一秒的形态。

    修改前的兜底分支是 ``else parsed``:形状判据不成立时,它把**整个回包**
    当成读数交出去。那个元组一路裸写进状态缓存,再被环境传感器 ``float()``
    抛成「环境硬故障」,退针 + 挂上一个解不开的急停闩。
    """
    r = skill_cls().execute(_Ctx(("", b"\x00")), {})
    assert r.success is False, (
        f"整个回包被当成了读数:{r.data!r} —— 这正是那次锁机的第一步。")


@pytest.mark.parametrize("skill_cls,key", _READBACKS)
def test_a_two_channel_body_is_refused_not_silently_indexed(skill_cls, key):
    """body 里有两个数 ⇒ 拒。**不许**挑第 0 个。

    挑一路是看不出来的错:数字合理、量纲合理,只是读的是另一个通道。
    """
    r = skill_cls().execute(_Ctx(_envelope([1.0e-10, 5.0e-10])), {})
    assert r.success is False


@pytest.mark.parametrize("skill_cls,key", _READBACKS)
def test_a_flat_reply_without_the_envelope_still_works(skill_cls, key):
    """有些替身/旧路径直接给裸数字,那条兼容分支不许被改坏。"""
    r = skill_cls().execute(_Ctx(3.5e-9), {})
    assert r.success is True
    assert r.data[key] == pytest.approx(3.5e-9)


def test_all_three_patchable_numeric_readbacks_are_covered():
    """新增一个「读回并写进缓存」的技能时,这里要显出它没被覆盖。

    这条不是形式主义:上面那个 bug 之所以能存在,正是因为
    `GetCurrent` 和 `GetBias` 是各写各的两份拷贝,修了一份没人知道还有一份。
    """
    from mast.core.state import InstrumentState
    covered = {key for _, key in _READBACKS}
    # 这三个是「有专门的 Get* 技能、且会回写缓存」的数值字段。
    expected = {"current_a", "bias_v", "z_pos_m"}
    assert covered == expected
    assert expected <= InstrumentState._PATCHABLE_FIELDS
    assert expected <= InstrumentState._NUMERIC_FIELDS, (
        "有字段进得了缓存却不在数值闸门的名单里 —— 那道闸门对它形同虚设。")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
