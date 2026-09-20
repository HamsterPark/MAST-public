# -*- coding: utf-8 -*-
"""带长度前缀的数值数组必须按元素类型解析。

字符串数组后紧跟整数数组时，不能只看加号分派到字符串分支。测试从字节布局
验证类型分派，防止上层因无法读取扫描属性而跳过连续扫描设置。"""
from __future__ import annotations

import struct

import pytest

from mast.core import nanonis_patch as NP


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack(">i", len(b)) + b


def _scan_props_body(continuous: int = 1,
                     mods=("Bias", "Z-Ctrl"),
                     nparams=(3, 4)) -> bytes:
    """按 Scan.PropsGet 的声明造一份回包 body。"""
    mods_blob = b"".join(_s(m) for m in mods)
    params_blob = b"".join(_s(p) for p in ("p1", "p2"))
    return (
        struct.pack(">III", continuous, 0, 0)
        + struct.pack(">i", 5) + b"serie"
        + struct.pack(">i", 0)
        + struct.pack(">i", len(mods_blob)) + struct.pack(">i", len(mods))
        + mods_blob
        + struct.pack(">i", len(nparams)) + struct.pack(">" + "i" * len(nparams), *nparams)
        # Parameters:rows, cols, 然后 rows*cols 个前置长度字符串(2D,无字节数字段)
        + struct.pack(">ii", 1, 2) + params_blob
        + struct.pack(">I", 0)
    )


#: **从模块取,别在测试里手抄。** 手抄的那一份正是这个文件此前钉住的东西 ——
#: 它把最后一个数组写成 1D 的 ``*+c``,于是测试和上游一起错,而且错得一致。
_TYPES = list(NP._SCAN_PROPS_GET_SPEC)


class _Fake:
    """只借解码助手，不连仪器 —— 这个 bug 用纯字节就能完整复现。"""

    displayInfo = 0
    decodeStringPrepended = NP.Nanonis.decodeStringPrepended
    decodeSingularString = NP.Nanonis.decodeSingularString
    decodeArray = NP.Nanonis.decodeArray
    decodeArrayPrepended = NP.Nanonis.decodeArrayPrepended

    @staticmethod
    def parseError(*_a, **_k):
        return ""


def test_star_plus_int_array_does_not_crash_the_whole_reply():
    body = _scan_props_body(continuous=1)
    out = NP._parse_general_response_strict(_Fake(), body, _TYPES)
    v = out[2]
    assert v[0] == 1, "continuous 标志读错了"
    assert v[9] == ["Bias", "Z-Ctrl"], "模块名读错了"
    assert v[11] == [3, 4], "*+i 被当成字符串解析了"


@pytest.mark.parametrize("cont", [0, 1])
def test_continuous_flag_survives_both_states(cont):
    """continuous 是回包的第一个字段，它读对与否决定扫描会不会停。"""
    out = NP._parse_general_response_strict(_Fake(), _scan_props_body(cont), _TYPES)
    assert out[2][0] == cont


def test_malformed_count_does_not_multiply_a_list():
    """前一个字段不是计数时，宁可给空数组也不要拿列表去乘。

    回包一旦错位，``Variables[-1]`` 可能是任何东西。第一版在这里直接
    ``range(n)``，一个非 int 就是 TypeError —— 而错位本身是可以恢复的：
    前面已经解析对的字段（包括 continuous）不该被后面的错误一起带走。
    """
    types = ["i", "*+c", "*+i"]          # 故意让 *+i 前面是字符串列表
    body = struct.pack(">i", 1) + _s("only") + struct.pack(">i", 7)
    out = NP._parse_general_response_strict(_Fake(), body, types)
    assert out[2][0] == 1               # 前面那个字段仍然是对的
    assert out[2][2] == []              # 数不出来就是空，不是崩


def test_star_plus_c_still_reads_strings():
    """回归：``*+c`` 本来是对的，别在修 ``*+i`` 的时候把它带坏。"""
    types = ["i", "i", "*+c"]
    mods = ("AAA", "BB")
    blob = b"".join(_s(m) for m in mods)
    body = struct.pack(">i", len(blob)) + struct.pack(">i", len(mods)) + blob
    out = NP._parse_general_response_strict(_Fake(), body, types)
    assert out[2][2] == ["AAA", "BB"]


# ── status=0 的错误段 = 「一切正常」，不是失败──────────

def test_a_zero_status_error_section_means_success_not_failure():
    """成功回包的错误段应在声明字段结束处解析为零状态和零长度。
    错误段回溯逻辑与 counter 校验必须采用一致的位置解释，不能把数据字节误当错误文本。"""
    import struct

    body = b"\x00" * 2068 + struct.pack('>ii', 0, 0)
    assert NP._reply_ends_in_a_clean_error_section(body, 2068) is True


def test_a_nonzero_status_is_still_a_real_error():
    """status 非 0 ⇒ 那是真的拒绝，不能当成功放行。"""
    import struct

    desc = b"NeedModule"
    body = b"\x00" * 20 + struct.pack('>ii', 1, len(desc)) + desc
    assert NP._reply_ends_in_a_clean_error_section(body, 20) is False
    assert NP._looks_like_a_real_error(body, 20, desc.decode()) is True


def test_coincidental_zeros_do_not_pass_as_a_clean_reply():
    """counter 落错位置时那 8 字节可能碰巧是两个零 —— 长度恒等式挡住它。

    只看 status==0 是不够的：全零的数据段到处都是。必须要求
    ``counter + 8 + desc_len == len(body)``。
    """
    assert NP._reply_ends_in_a_clean_error_section(b"\x00" * 100, 20) is False
