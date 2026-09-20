"""Scan_BufferGet 应同时接受裸整数与单元素元组形式的通道数组。

对同一合成通道表分别编码两种形状，验证解析一致；
只使用更规整的替身会掩盖真实解析器返回嵌套标量的分支。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# --- v2 package bootstrap (tests live outside MASTv2/) ----------------------
_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.io.nanonis_files import (  # noqa: E402
    channel_ids_from_buffer,
    frame_acquired_lines,
    parse_buffer_get,
    scalar_float,
    scalar_int,
)


# ── 支持的两种回包形态 ───────────────────────────────────────────────────────

#: 合成回包分别覆盖单元素元组与裸整数。
SHAPES = {
    "one_tuple": lambda c: (c,),
    "bare_int": lambda c: c,
}


@pytest.fixture(params=sorted(SHAPES), ids=sorted(SHAPES))
def wrap_ch(request):
    return SHAPES[request.param]


def _body(channels, pixels=512, lines=512, wrap=lambda c: c):
    return [len(channels), [wrap(c) for c in channels], pixels, lines]


def _reply(channels, pixels=512, lines=512, wrap=lambda c: c):
    """完整 return_value = (error_str, raw_bytes, body)。"""
    return ("", b"", _body(channels, pixels, lines, wrap))


# ── channel_ids_from_buffer ─────────────────────────────────────────────────

def test_channel_ids_both_shapes_give_plain_ints(wrap_ch):
    """两种形态解出来的必须是**同一个** int 列表 —— 下游要拿它当通道号用。"""
    ids = channel_ids_from_buffer(_body([0, 30], wrap=wrap_ch))
    assert ids == [0, 30]
    assert all(type(i) is int for i in ids)


def test_channel_ids_synthetic_tuple_array():
    """独立构造的多通道元组数组验证精确解包。"""
    assert channel_ids_from_buffer([3, [(2,), (5,), (9,)], 128, 64]) == [2, 5, 9]


def test_channel_ids_single_channel(wrap_ch):
    assert channel_ids_from_buffer(_body([14], wrap=wrap_ch)) == [14]


def test_channel_ids_empty_selection_is_empty_not_none(wrap_ch):
    """「一个通道都没选」是真实状态,必须与「读不出来」区分开 —— 两者都返回
    ``[]``,但调用侧靠 parse_buffer_get 的 None 区分'回包不可解析'。"""
    assert channel_ids_from_buffer(_body([], wrap=wrap_ch)) == []


def test_channel_ids_accepts_ndarray_forms():
    """nanonis_spm 也可能把 ``*i`` 交成 ndarray。"""
    assert channel_ids_from_buffer([2, np.array([0, 30]), 512, 512]) == [0, 30]
    assert channel_ids_from_buffer([2, [np.int32(0), np.int64(30)], 512, 512]) == [0, 30]


def test_channel_ids_accepts_unwrapped_scalar():
    """单通道没被包成列表时,不能当成"读不到"。"""
    assert channel_ids_from_buffer([1, 7, 512, 512]) == [7]


def test_channel_ids_skips_garbled_element_keeps_the_rest():
    """一个坏元素不该让你损失读得出来的那些通道。"""
    assert channel_ids_from_buffer([3, [(0,), "??", (30,)], 512, 512]) == [0, 30]
    assert channel_ids_from_buffer([3, [(), (30,)], 512, 512]) == [30]


@pytest.mark.parametrize("body", [None, "garbage", b"", 42, [], [2], ("only-one",)])
def test_channel_ids_never_raises_on_junk(body):
    """回包形状意外**绝不能**在 skill 的 execute 里抛异常 —— 那正是原 bug 的
    形状(未被捕获的 TypeError 直接冒到 ScanAt)。"""
    assert channel_ids_from_buffer(body) == []


def test_channel_ids_does_not_iterate_a_string_char_by_char():
    """``"014"`` 不是三个通道。"""
    assert channel_ids_from_buffer([1, "014", 512, 512]) == []


# ── parse_buffer_get ────────────────────────────────────────────────────────

def test_parse_buffer_get_both_shapes(wrap_ch):
    got = parse_buffer_get(_reply([0, 30], 512, 256, wrap=wrap_ch))
    assert got == {"num_channels": 2, "channel_indexes": [0, 30],
                   "pixels": 512, "lines": 256}


def test_parse_buffer_get_reports_declared_num_channels_verbatim():
    """num_channels 报仪器**自己声明**的值,不是 len(解析结果) —— 两者不一致时
    那是一次解析失败,调用方有权看见,不该被抹平。"""
    got = parse_buffer_get(("", b"", [5, [(0,), (30,)], 512, 512]))
    assert got["num_channels"] == 5
    assert got["channel_indexes"] == [0, 30]


@pytest.mark.parametrize("parsed", [
    None, "garbage", b"", [], ["", b""], ("", b"", "not-a-body"),
    ("", b"", [2, [(0,)]]),          # body 短了
])
def test_parse_buffer_get_returns_none_on_unusable_reply(parsed):
    """不可解析 → None。调用侧据此**拒绝写入**(猜一个通道列表写回去会清掉
    用户的采集配置)。"""
    assert parse_buffer_get(parsed) is None


def test_parse_buffer_get_tolerates_wrapped_pixels_and_lines():
    """pixels/lines 同时接受裸整数与单元素元组，避免调用层抛出 TypeError。"""
    got = parse_buffer_get(("", b"", [2, [(0,), (30,)], (512,), (256,)]))
    assert got["pixels"] == 512 and got["lines"] == 256


def test_parse_buffer_get_degrades_pixels_to_none_never_raises():
    """读不出来就是 None,由调用侧决定这意味着什么 —— 但绝不抛。"""
    got = parse_buffer_get(("", b"", [2, [(0,)], "?", None]))
    assert got["pixels"] is None and got["lines"] is None
    assert got["channel_indexes"] == [0]


# ── scalar_int ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    (0, 0), ((0,), 0), ([30], 30), (np.int32(7), 7), (np.array([7]), 7),
    (512, 512), (3.0, 3),
    ("?", None), (None, None), ((), None), ((0, 1), None), (b"0", None),
])
def test_scalar_int(raw, want):
    assert scalar_int(raw) == want


def test_scalar_int_does_not_unwrap_multi_element():
    """``(0, 1)`` 不是一个标量 —— 悄悄取第一个就是在造数据。"""
    assert scalar_int((0, 1)) is None


# ── frame_acquired_lines ────────────────────────────────────────────────────
#
# 合成帧以 NaN 标记未采集行，检验扫描前沿推断。


def _grab_reply(lines: int, done: int, pixels: int = 8):
    """一份 ``Scan_FrameDataGrab`` 回包:异构 body,数据是 2-D ndarray 元素。"""
    arr = np.arange(lines * pixels, dtype=np.float64).reshape(lines, pixels)
    arr[done:, :] = np.nan
    return ("", b"", [5, "Z (m)", lines, pixels, arr, 1])


@pytest.mark.parametrize("lines,done", [(64, 64), (64, 15), (64, 0), (512, 511)])
def test_frame_acquired_lines_counts_the_nan_front(lines, done):
    assert frame_acquired_lines(_grab_reply(lines, done)) == (done, lines)


def test_a_frame_with_no_nan_rows_reads_as_complete():
    """和 ``scan_monitor._measure_acquired_frac`` 刻意不同:那边遇到「一行 NaN
    都没有」返回 None(不肯声称 100%),这里返回 ``(rows, rows)``。本函数回答的是
    「有没有看得见的缺行」,而在一台回填旧数据而不是填 NaN 的机器上,诚实的答案
    就是「看不见缺行」—— 那正好退化成本检查存在之前的行为,而不是造出新的误报。"""
    arr = np.ones((32, 8), dtype=np.float64)
    assert frame_acquired_lines(("", b"", [5, "Z (m)", 32, 8, arr, 1])) == (32, 32)


@pytest.mark.parametrize("reply", [
    None, "", ("", b""), ("", b"", []), ("", b"", [5, "Z (m)", 0, 0]),
])
def test_frame_acquired_lines_returns_none_when_it_cannot_measure(reply):
    """判不了 ≠ 零行。返回 0 会让调用方把一帧好图当成完全没扫。"""
    assert frame_acquired_lines(reply) is None


def test_a_flat_numeric_body_still_yields_a_square_frame():
    """桩/模拟器给扁平数字列表。仍要能量,否则新判据在模拟器上永远「测不到」。"""
    flat = [0.0] * 16                       # 4×4
    assert frame_acquired_lines(("", b"", flat)) == (4, 4)


def _genuine_nanonis(*, patched: bool):
    """从磁盘加载**真的** ``nanonis_spm.Nanonis``,可选装上 §2.21 的补丁。

    ``tests/conftest.py`` 把 ``sys.modules["nanonis_spm"]`` 换成了 MagicMock
    (无硬件即可跑),而 mock 会「通过」任何关于解码值的断言 —— 那正是这条测试要
    防的东西。刻意**不**从 ``tests/v2/unit/core/test_nanonis_patch_decode_array.py``
    里 import 同名 helper:测试文件之间的耦合会让那边一次重构把这边弄红,而
    这里只需要十行。生产逻辑绝不重复,测试夹具可以。
    """
    import importlib.util
    import sysconfig

    # **装好的那一份优先,而不是 sys.path 里撞见的第一份。** 仓里有九份
    # ``nanonis_spm``(``dist/*/_internal/`` 下的冻结构建),今天与装好的那份逐字节
    # 相同,所以扫 sys.path 也碰巧对。但 nanonis_spm 1.0.9 有个每次 pip install
    # 之后都要打的 parser 补丁 —— 只要哪次只打了一边,两份就会分叉,而那时
    # 「先撞见谁」决定这条测试量的是哪一份代码。让顺序由安装位置决定,不由
    # sys.path 当时的样子决定。
    candidates = [Path(sysconfig.get_paths()["purelib"])]
    candidates += [Path(b) for b in sys.path if "_internal" not in b]
    candidates.append(Path(sys.executable).resolve().parents[1])
    for base in candidates:
        cand = base / "nanonis_spm" / "NanonisClass.py"
        if cand.exists():
            break
    else:
        pytest.skip("nanonis_spm package not installed on disk")

    spec = importlib.util.spec_from_file_location("_genuine_nanonis_io", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if patched:
        from mast.core import nanonis_patch
        mod.Nanonis.parseGeneralResponse = nanonis_patch._patched_parseGeneralResponse
        mod.Nanonis.decodeArray = nanonis_patch._patched_decodeArray
        mod.Nanonis.decodeArrayPrepended = nanonis_patch._patched_decodeArrayPrepended
    obj = mod.Nanonis.__new__(mod.Nanonis)   # 解析器不碰 socket
    obj.displayInfo = 0
    return obj


@pytest.mark.parametrize("patched", [False, True], ids=["pristine", "patched"])
def test_frame_acquired_lines_against_the_real_parser_byte_for_byte(patched):
    """字节级:自己拼一份 ``Scan_FrameDataGrab`` 回包,喂给**真的**
    ``parseGeneralResponse``(打补丁前 + 打补丁后各一遍),再交给
    ``frame_acquired_lines``。

    **这条测试钉的是与 §2.21 数组解包补丁的交界。** 1-元组的来源是
    ``decodeArray``(``struct.unpack`` 回 tuple,整条被 append 进去),而
    ``decodeArray`` **只在 ``*`` 数组分支里被调用**。``Scan_FrameDataGrab`` 的
    ResponseTypes 是 ``["i", "*-c", "i", "i", "2f", "I"]`` —— 唯一的 ``*`` 是字符串
    (走 ``decodeSingularString``),**没有数值数组**。另外两条路本来就不产生元组:

    * ``2f`` 分支最后是 ``np.reshape(SentArray, (rows, cols))``,numpy 把那一串
      1-元组摊平成真正的二维 float 数组;
    * 标量分支是 ``Variables.append(Value[0])``,**本来就已经拆开了**。

    所以这条路上的元素补丁前后都是裸值。两种形态都跑,是因为「只钉一种形态」
    正是本文件开头记的那个陷阱。
    """
    import struct

    n = _genuine_nanonis(patched=patched)

    name = b"Z (m)"
    rows, cols = 6, 4
    data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols).copy()
    data[4:, :] = np.nan                    # 最后两行从未采集

    buf = struct.pack(">i", len(name)) + name
    buf += struct.pack(">i", rows) + struct.pack(">i", cols)
    for v in data.ravel():
        buf += struct.pack(">f", v)
    buf += struct.pack(">I", 1)             # 扫描方向
    buf += struct.pack(">i", 0)             # error status
    buf += struct.pack(">i", 0)             # error string length

    parsed = n.parseGeneralResponse(buf, ["i", "*-c", "i", "i", "2f", "I"])
    body = parsed[2]

    # ① 数值标量是裸的,不是 1-元组 —— 这条路不经过 decodeArray。
    assert [x for x in body if isinstance(x, int)] == [len(name), rows, cols, 1]
    assert not [x for x in body if isinstance(x, tuple)]
    # ② 帧是真正的二维 ndarray(np.reshape 的产物,不是 decodeArray 的)。
    frame = [x for x in body if isinstance(x, np.ndarray)]
    assert len(frame) == 1 and frame[0].shape == (rows, cols)
    # ③ 于是行数数得对:6 行里 4 行有数据。
    assert frame_acquired_lines(parsed) == (4, rows)


@pytest.mark.parametrize("patched", [False, True], ids=["pristine", "patched"])
@pytest.mark.parametrize("declared", [7, 3, 900])
def test_a_misaligned_reply_fails_loudly_instead_of_yielding_a_plausible_frame(
        patched, declared):
    """通道名的**声明长度**被写坏时,必须炸,不能安静地给出一个像样的帧。

    为什么这条属于这里:``parseGeneralResponse`` 的 ``*-c`` 分支是
    ``counter += NoOfChars`` —— 字节游标按**声明**长度前进,与字符串解码器实际读到
    多少无关。所以名字长度一错,后面的 rows / cols / ``2f`` 帧**全部错位**。

    §2.21 的补丁把 ``decodeSingularString`` 从「声明长度越界就 IndexError」改成
    「截断」,也就是**把一个响亮的失败改小声了**。要确认的是它没有变成**无声**:
    错位之后如果还能凑出一个二维数组,``frame_acquired_lines`` 就会去数它的 NaN 行,
    而 ``WaitScanComplete`` 会把结果当成「这一帧被中途停止了」—— 一次由解析差异
    伪造出来的截断,正是 ``_measure_lines`` 的两道守卫要挡的东西。

    实测(两版解析器、三种坏长度)都在 ``2f`` 之前就 raise:错位的字节读不出
    合法的 rows/cols。异常类型变了(``IndexError`` → ``struct.error``),
    「会炸」这件事没变 —— 而技能那边只关心「炸没炸」:
    ``safe_call`` 记成 error → ``lines_verified=False`` → 按旧行为放行,不谎报截断。
    """
    import struct

    n = _genuine_nanonis(patched=patched)

    name = b"Z (m)"
    rows, cols = 6, 4
    data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols).copy()
    data[4:, :] = np.nan

    buf = struct.pack(">i", declared) + name        # ← 声明长度与实际不符
    buf += struct.pack(">i", rows) + struct.pack(">i", cols)
    for v in data.ravel():
        buf += struct.pack(">f", v)
    buf += struct.pack(">I", 1)
    buf += struct.pack(">i", 0) + struct.pack(">i", 0)

    try:
        parsed = n.parseGeneralResponse(buf, ["i", "*-c", "i", "i", "2f", "I"])
    except Exception:                                # noqa: BLE001 — 炸了就对了
        return
    # 万一将来某版解析器不再 raise:那也绝不能给出一个「少了几行」的可信答案。
    measured = frame_acquired_lines(parsed)
    assert measured is None or measured[0] == measured[1], (
        f"错位的回包产生了看起来像截断的行数 {measured} —— "
        f"WaitScanComplete 会把它报成 stopped_early"
    )


def test_one_tuple_scalars_degrade_to_unmeasurable_not_to_a_lie():
    """反方向的保险:万一哪天真有一条路把标量也包成 1-元组,而数据又不是二维,
    ``parse_frame_grab`` 读不出 rows/cols —— 那时必须返回 ``None``(判不了),
    绝不能靠「猜一个形状」给出一个看起来像真的行数。"""
    flat = [(float(i),) for i in range(12)]     # 非平方数,拼不出方阵
    body = [(5,), "Z (m)", (3,), (4,), *flat, (1,)]
    assert frame_acquired_lines(("", b"", body)) is None


# ── scalar_float ────────────────────────────────────────────────────────────
#
# `scalar_int` 的兄弟。**拒绝什么才是它的全部价值** —— 一个什么都接受的转换器
# 就是把形状错误变成一个看起来合理的数的机器。


@pytest.mark.parametrize("raw,want", [
    (1.5, 1.5), ((1.5,), 1.5), ([2.5], 2.5), (np.float32(3.5), 3.5),
    (np.array([4.5]), 4.5), (7, 7.0), (((1.5,),), 1.5),      # 嵌套也拆
])
def test_scalar_float_accepts_one_scalar(raw, want):
    assert scalar_float(raw) == pytest.approx(want)


@pytest.mark.parametrize("raw", [
    (1.0, 2.0),          # 多元素:**不许**悄悄取第一个 —— 那会在双通道回包里
    [1.0, 2.0],          # 静默挑一路
    np.array([1.0, 2.0]),
    (), [],              # 空:不是 0.0,是「没有」
    None, "1.5", b"1.5", "abc",
    float("nan"), float("inf"), float("-inf"),
    True, False,         # bool 是 int 的子类,但它不是一个测量值
])
def test_scalar_float_refuses_everything_else(raw):
    assert scalar_float(raw) is None


def test_scalar_float_never_fabricates_zero():
    """两份被替换掉的手写实现在空序列上返回 `0.0`。
    **一个失败的读取变成一个看起来合理的测量值** —— 因此必须明确报告解析失败。"""
    assert scalar_float([]) is None
    assert scalar_float(()) is None


def test_scalar_float_matches_scalar_int_on_refusals():
    """两个兄弟对「拿不到」的判断必须一致,否则同一个回包在两条路上得出不同结论。"""
    for raw in ((1.0, 2.0), [], None, "x", b"x"):
        assert scalar_float(raw) is None
        assert scalar_int(raw) is None
