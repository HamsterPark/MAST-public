"""信号名称列表必须和仪器声明的数量对账，截断不能解释成通道不存在。"""
from __future__ import annotations

from mast.skills.builtins.signals import ListSignalChannels


class _Rec:
    def __init__(self, return_value):
        self.return_value = return_value
        self.error = ""


class _Ctx:
    def __init__(self, declared, names):
        # Signals.NamesGet ResponseTypes = ["i", "i", "*+c"]:
        #   Variables = [names-size-bytes, number-of-signals, [names…]]
        self._rv = ("hdr", b"", [0, declared, names])

    def safe_call(self, verb, *a, **kw):
        assert verb == "Signals_NamesGet"
        return _Rec(self._rv)


def _run(declared, names):
    return ListSignalChannels().execute(_Ctx(declared, names), {})


def test_a_short_decode_is_flagged_against_the_declared_count():
    r = _run(128, [f"Sig {i}" for i in range(51)])
    assert r.success is True          # 部分名单仍然可用，不整条失败
    assert r.data["n_channels"] == 51
    assert r.data["declared_n"] == 128
    assert r.data["truncated"] is True


def test_a_full_table_is_not_flagged():
    """反面：正常时不许报截断，否则这个标志永远在响、等于没有。"""
    r = _run(128, [f"Sig {i}" for i in range(128)])
    assert r.data["truncated"] is False
    assert r.data["declared_n"] == 128


def test_the_declared_count_survives_the_one_tuple_shape():
    """声明数量可能裹在单元素元组中，比较前必须正确解包。"""
    r = _run((128,), [f"Sig {i}" for i in range(51)])
    assert r.data["declared_n"] == 128
    assert r.data["truncated"] is True


def test_an_unreadable_declared_count_is_unknown_not_truncated():
    """把「不知道」当成「坏了」，和把「坏了」当成「正常」是同一种错的两个方向。"""
    r = _run("junk", [f"Sig {i}" for i in range(16)])
    assert r.data["declared_n"] is None
    assert r.data["truncated"] is False


def test_more_names_than_declared_is_not_called_truncated():
    """只朝一个方向判：解出来比声明的多，那是别的毛病，不是截断。"""
    r = _run(4, [f"Sig {i}" for i in range(8)])
    assert r.data["truncated"] is False
