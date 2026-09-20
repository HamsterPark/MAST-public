# -*- coding: utf-8 -*-
"""Scan.PropsGet 同时含一维与二维字符串数组。

一维字段用字节数和元素数表示；二维字段用行列数及逐字符串长度表示。
构造合成参数表验证游标消费全部字段后落在错误段，防止把行数当字节数。"""
from __future__ import annotations

import struct

import pytest

from mast.core import nanonis_patch as NP


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack(">i", len(b)) + b


def _props(continuous: int = 1,
           mods=("Bias", "Z-Controller", "Current", "Lock-in"),
           cols: int = 20,
           series: str = "Au111_%03i",
           comment: str = "",
           with_error_tail: bool = True):
    """按**手册**的字段表造一份 Scan.PropsGet 回包。

    参数名故意造得像真的（``Bias (V)`` 这种），因为字节数是这条 bug 的证据。
    """
    rows = len(mods)
    mods_blob = b"".join(_s(m) for m in mods)
    grid = [["%s p%02d (V)" % (m[:4], c) for c in range(cols)] for m in mods]
    params_blob = b"".join(_s(x) for row in grid for x in row)
    body = (
        struct.pack(">III", continuous, 0, 0)
        + struct.pack(">i", len(series)) + series.encode()
        + struct.pack(">i", len(comment)) + comment.encode()
        + struct.pack(">i", len(mods_blob)) + struct.pack(">i", rows) + mods_blob
        + struct.pack(">i", rows) + struct.pack(">" + "i" * rows, *([cols] * rows))
        + struct.pack(">ii", rows, cols) + params_blob
        + struct.pack(">I", 0)
    )
    if with_error_tail:
        body += struct.pack(">ii", 0, 0)      # status=0, desc_len=0 → 无错误
    return body, grid, len(params_blob)


class _Fake:
    """只借解码助手与真的 parseError —— 这条 bug 用纯字节就能完整复现。"""

    displayInfo = 0
    decodeStringPrepended = NP.Nanonis.decodeStringPrepended
    decodeSingularString = NP.Nanonis.decodeSingularString
    decodeArray = NP.Nanonis.decodeArray
    decodeArrayPrepended = NP.Nanonis.decodeArrayPrepended
    parseError = NP._patched_parseError


@pytest.fixture(autouse=True)
def _restore_the_upstream_parse_error(monkeypatch):
    """把 ``_original_parseError`` 换回真的两行实现。

    ═══════════════════════════════════════════════════════════════════════
    替身把「没有错误」答成了一个真值对象，于是断言在空转
    ═══════════════════════════════════════════════════════════════════════

    ``_patched_parseError`` 先从**尾部**倒推找错误段；找不到就退回上游实现。
    而 ``desc_len == 0``（也就是「一切正常」）恰恰是倒推**必然找不到**的那一种
    ——它的循环从 ``cut=1`` 起步。所以正常回包一定会走到退路上。

    退路是 ``_original_parseError``，它在导入时取自 ``Nanonis.parseError``；而
    测试里 ``nanonis_spm`` 是 mock，那就是个 **MagicMock**。后果不是报错：

    * ``len(mock)`` = 0 ⇒ 「有没有错误」这个判断永远答「没有」；
    * ``mock.startswith(...)`` 返回 mock ⇒ **恒为真**，负例测试照样绿；
    * ``int(mock)`` = 1 ⇒ 从诊断里抠出来的偏移量恒等于 1。

    第二条最要命：本文件那条「上游规格必须失败」的回归测试，第一版就是这么
    **通过的** —— 它什么也没验证。真实现只有两行，就写在 ``_patched_parseError``
    的文档里，照抄回来即可。
    """
    def _upstream(self, response, index):
        return bytes(response)[index + 8:].decode(errors="replace")

    monkeypatch.setattr(NP, "_original_parseError", _upstream)


#: 上游那份（最后一个数组写成 1D）—— 留着作为错误一维声明的负对照。
_UPSTREAM_SPEC = ["I", "I", "I", "i", "*-c", "i", "*-c", "i", "i", "*+c",
                  "i", "*+i", "i", "i", "*+c", "I"]


# ── 修好之后：整包读通 ────────────────────────────────────────────────────

def test_the_reply_parses_clean_end_to_end():
    """counter 正好走到错误段 ⇒ 仪器说「没有错误」⇒ 整包可用。"""
    body, grid, _ = _props(continuous=1)
    err, _raw, v = NP._parse_general_response_strict(
        _Fake(), body, NP._SCAN_PROPS_GET_SPEC)
    assert err == "", "整包应该读通，实际: %r" % err[:200]
    assert v[0] == 1, "continuous 读错了"
    assert v[9] == ["Bias", "Z-Controller", "Current", "Lock-in"]
    assert v[14] == grid, "2D 参数表读错了"


def test_parameters_come_back_as_rows_of_modules():
    """「每一行属于一个模块」是手册的原话，形状要保住。"""
    body, grid, _ = _props(mods=("A", "B", "C"), cols=5)
    v = NP._parse_general_response_strict(
        _Fake(), body, NP._SCAN_PROPS_GET_SPEC)[2]
    assert len(v[14]) == 3 and all(len(r) == 5 for r in v[14])
    assert v[14] == grid


@pytest.mark.parametrize("cont", [0, 1])
def test_continuous_flag_survives_both_states(cont):
    """continuous 读对与否，决定连续扫描能不能被关掉。"""
    body, _g, _p = _props(continuous=cont)
    v = NP._parse_general_response_strict(
        _Fake(), body, NP._SCAN_PROPS_GET_SPEC)[2]
    assert v[0] == cont


# ── 回归：上游那份规格在同一批字节上必须检出未消费字节 ──────────────────────

def test_the_upstream_1d_spec_reproduces_the_field_failure():
    """把二维数组声明为一维会留下未消费字节，解析必须把这种布局错误判为失败。"""
    body, _g, _p = _props()
    err = NP._parse_general_response_strict(_Fake(), body, _UPSTREAM_SPEC)[0]
    assert err.startswith(NP.LAYOUT_MISMATCH_PREFIX), (
        "上游规格在这批字节上本该失败，实际: %r" % err[:200])


def test_the_real_rig_byte_identity_holds():
    """合成参数表应满足未消费字节数恒等式 P−rows+8，用于区分二维行列与一维长度语义。"""
    body, _g, p_bytes = _props()
    rows = 4
    err = NP._parse_general_response_strict(_Fake(), body, _UPSTREAM_SPEC)[0]
    # 诊断文本里带着 offset 与总长
    off = int(err.split("at offset ")[1].split(" ")[0])
    assert len(body) - off == p_bytes - rows + 8


# ── 错位与畸形：宁可给空，别崩、别越界 ────────────────────────────────────

def test_bogus_row_col_counts_do_not_multiply_into_the_buffer():
    """rows×cols×4 超过剩余字节 ⇒ 那不是一对计数，别乘下去。"""
    types = ["i", "i", "*2c"]
    body = struct.pack(">iii", 9999, 9999, 4)
    v = NP._parse_general_response_strict(_Fake(), body, types)[2]
    assert v[2] == []


def test_non_int_counts_yield_an_empty_grid_not_a_crash():
    """前面不是一对 int（回包已错位）⇒ 空表，而且前面读对的字段要留住。"""
    types = ["i", "*+c", "i", "*2c"]
    body = struct.pack(">i", 1) + _s("only") + struct.pack(">i", 2) + _s("x")
    out = NP._parse_general_response_strict(_Fake(), body, types)
    assert out[2][0] == 1        # 前面那个字段仍然是对的
    assert out[2][3] == []


def test_a_truncated_array_stops_at_the_buffer_end():
    """回包被截断时读到哪算哪，不能读出缓冲区，也不能倒退 counter。"""
    types = ["i", "i", "*2c"]
    body = struct.pack(">ii", 2, 2) + _s("aa") + _s("bb") + struct.pack(">i", 40)
    v = NP._parse_general_response_strict(_Fake(), body, types)[2]
    assert v[2][0] == ["aa", "bb"]
    assert v[2][1] == []


def test_counter_advances_by_bytes_not_characters():
    """非 ASCII 参数名：推进必须按**字节**，否则后面每个字段都错位。

    这是 ``decodeStringPrepended`` 里 latin-1/utf-8 之争的同一个坑：只要把
    解码出的字符数喂回字节计数器，一个中文字符就能让整包偏 2 字节。
    """
    types = ["i", "i", "*2c", "I"]
    name = "偏压 (V)"
    body = struct.pack(">ii", 1, 1) + _s(name) + struct.pack(">I", 0xABCD)
    v = NP._parse_general_response_strict(_Fake(), body, types)[2]
    assert v[2] == [[name]]
    assert v[3] == 0xABCD, "counter 按字符数推进了 ⇒ 后面的字段错位"


def test_zero_rows_is_a_valid_empty_answer():
    """一个模块都没配 ⇒ rows=0 ⇒ 空表，不是错误。"""
    types = ["i", "i", "*2c", "I"]
    body = struct.pack(">ii", 0, 7) + struct.pack(">I", 5)
    v = NP._parse_general_response_strict(_Fake(), body, types)[2]
    assert v[2] == [] and v[3] == 5


# ── 规格本身 ──────────────────────────────────────────────────────────────

def test_the_patched_spec_is_installed_on_the_client():
    """光有记号不算数 —— ``Scan_PropsGet`` 得真的用上它。"""
    assert NP.Nanonis.Scan_PropsGet is NP._patched_Scan_PropsGet
    assert NP._SCAN_PROPS_GET_SPEC[14] == "*2c"
    assert NP._SCAN_PROPS_GET_SPEC.count("*+c") == 1, (
        "两个字符串数组的记号又变成同一个了 —— 那正是这条 bug 的形状")
