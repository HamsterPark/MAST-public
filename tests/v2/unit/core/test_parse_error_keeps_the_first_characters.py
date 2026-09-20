"""错误文本解析应保留完整开头。

前置字段的游标偏差不能改变尾部自带长度的错误段；测试用合成回包验证
完整文本与退回路径，防止解析偏差产生看似正常但被截断的诊断。"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core import nanonis_patch  # noqa: E402

# ⚠️ 这里拿不到真的 ``Nanonis`` 类:``tests/conftest.py`` 在**任何导入之前**就把
# 整个 ``nanonis_spm`` 换成了 Mock(仓里的测试要能在没有硬件的机器上跑),
# 于是 ``Nanonis.__new__`` 会炸在 mock 内部。
#
# 不需要真类:补丁函数是**纯函数**(只吃 bytes 和一个 int),直接调它就行。
# ``self`` 只在退回原实现时才用到,给 None 即可 —— 而退回那条路本身也有测试。
_parse = nanonis_patch._patched_parseError

_TEXT = b"Error in command: tcplog.statusget: Generate User Event in ProgInterf"


def _reply(text: bytes, lead: int = 12) -> bytes:
    """一条带错误的回包:[前置字段][4B status][4B len][text]。"""
    return b"\x00" * lead + struct.pack(">i", 1) + struct.pack(">i", len(text)) + text


def test_a_correct_index_gives_the_whole_message():
    assert _parse(None, _reply(_TEXT), 12) == _TEXT.decode()


@pytest.mark.parametrize("skew", [1, 2, 5, 14, 30])
def test_a_skewed_index_still_gives_the_whole_message(skew):
    """合成前置游标偏差不应截断位于回包尾部、带独立长度前缀的错误文本。"""
    got = _parse(None, _reply(_TEXT), 12 + skew)
    assert got == _TEXT.decode(), (
        f"index 偏 {skew} 时错误串开头被啃掉了:{got[:30]!r}")


def test_a_long_message_survives():
    """长错误(几百字符)也要完整 —— 长度前缀是 int32,不是单字节。"""
    long_text = b"Error in command: " + b"x" * 500
    assert _parse(None, _reply(long_text), 12 + 7) == long_text.decode()


def test_no_error_still_reads_as_no_error():
    """没有错误的回包必须**退回原实现** —— 那条路本来就答得对(返回空串)。

    这里不能断言「返回空串」:conftest 把 ``nanonis_spm`` 换成了 Mock,
    原实现是个 MagicMock。所以断言的是**退回这件事发生了** ——
    补丁没有对一条没有错误的回包瞎解出一段文本。
    """
    got = _parse(None, b"\x00" * 12, 4)
    assert not isinstance(got, str) or got == "", (
        f"补丁在一条无错误的回包上解出了文本:{got!r}")


def test_a_reply_that_does_not_fit_the_shape_falls_back():
    """对不上就退回原实现 —— **不猜**。"""
    got = _parse(None, b"\x01\x02\x03", 0)
    assert not isinstance(got, str) or "\x01" not in got


def test_it_never_raises_on_garbage():
    """解析器抛异常会替换掉仪器自己的错误文本 —— 那是 nanonis_patch 在修的另一族。"""
    for junk in (b"", b"\x00", b"\xff" * 9, b"not-a-reply"):
        _parse(None, junk, 0)          # 不抛即通过


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
