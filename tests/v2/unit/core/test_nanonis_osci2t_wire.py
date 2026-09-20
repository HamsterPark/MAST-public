"""Osci2T 解析应从自行编码的协议字节开始测试。

直接提供已解码列表会绕过 ResponseTypes 与游标处理。测试构造完整响应和
仅错误段响应，验证字段规格、错误识别与边界处理。"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import struct

import pytest

#: 合成时基表；几何级数的时长用于验证数组长度与二进制索引布局。
TIMEBASES = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
CURRENT_INDEX = 4

#: 仪器拒绝一条命令时的描述文本。全仓判「模块在不在」靠的就是这个子串
#: （`pump._guard`、`zburst._guard` 都写着 `if "NeedModule" in err`）。
NEED_MODULE = b"NeedModule: Oscilloscope 2-Channels module is not loaded"


# ── 回包体的字节 ─────────────────────────────────────────────────────────────

def _error_section(status: int = 0, desc: bytes = b"") -> bytes:
    return struct.pack(">ii", status, len(desc)) + desc


def timebase_body(index_fmt: str, *, index: int = CURRENT_INDEX,
                  values=TIMEBASES) -> bytes:
    """按协议结构生成的合成 Osci2T.TimebaseGet 回包体。

    ``index_fmt`` 是时基索引那个字段的实际宽度：``"H"``（uint16，nanonis_spm
    v1.0.9 自己为 Osci2T 声明的）或 ``"i"``（int32，2026-08-02 那次修复顺手改成
    的值）。**这两条就是本次事故的两个候选**，所以它们必须能各自造出字节来。
    """
    return (struct.pack(">" + index_fmt, index)
            + struct.pack(">i", len(values))
            + b"".join(struct.pack(">f", v) for v in values)
            + _error_section())


def rejected_body(desc: bytes = NEED_MODULE) -> bytes:
    """命令被拒绝时的回包体：**只有错误段**，声明的字段一个都不在。"""
    return _error_section(1, desc)


# ── 真实的 Nanonis 解析器（不是 conftest 那个 MagicMock） ────────────────────

def _real_nanonis_class(*, patched: bool):
    """从磁盘装载**真的** ``nanonis_spm.Nanonis``。

    ``tests/conftest.py`` 把 ``sys.modules["nanonis_spm"]`` 换成了 MagicMock，
    而 MagicMock 会「通过」任何关于解码结果的断言。装载器与
    ``test_nanonis_patch_decode_array.py::_real_nanonis_class`` 保持一致
    （连排除 ``_internal`` 的理由都一样：那是 PyInstaller 的布局标记，冻结包里
    每发一版就多一份同名文件）。
    """
    import importlib.util
    import sysconfig

    candidates = [Path(sysconfig.get_paths()["purelib"])]
    candidates += [Path(b) for b in sys.path if "_internal" not in b]
    candidates.append(Path(sys.executable).resolve().parents[1])
    for base in candidates:
        cand = base / "nanonis_spm" / "NanonisClass.py"
        if cand.exists():
            break
    else:
        pytest.skip("nanonis_spm package not installed on disk")

    spec = importlib.util.spec_from_file_location("_real_nanonis_osci2t", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if patched:
        from mast.core import nanonis_patch
        mod.Nanonis.parseGeneralResponse = nanonis_patch._patched_parseGeneralResponse
        mod.Nanonis.decodeArray = nanonis_patch._patched_decodeArray
        mod.Nanonis.decodeArrayPrepended = nanonis_patch._patched_decodeArrayPrepended
    return mod.Nanonis


def _instance(cls):
    obj = cls.__new__(cls)          # 解析器一个字节都不碰 socket
    obj.displayInfo = 0
    return obj


@pytest.fixture()
def nano():
    return _instance(_real_nanonis_class(patched=True))


@pytest.fixture()
def pristine():
    return _instance(_real_nanonis_class(patched=False))


# ── 拒绝 vs 规格不符：2026-08-09 它们是**逐字相同**的一句话 ──────────────────

def test_a_rejected_command_reports_the_instruments_own_words(nano):
    """模块没加载时,必须能读到 ``NeedModule`` —— 全仓靠它判「模块在不在」。"""
    err, _raw, _vals = nano.parseGeneralResponse(rejected_body(), ["i", "i", "*f"])
    assert "NeedModule" in err
    assert "unpack requires" not in err


def test_before_the_fix_the_rejection_was_an_unreadable_struct_error(pristine):
    """「修好之前长什么样」是观测出来的,不是凭记忆断言的。

    未打补丁的解析器拿 ``["i","i","*f"]`` 去读一段只有错误段的 body:
    前两个 int32 被读成「状态」和「描述长度」,第三个字段于是按**描述长度**个
    float32 去读,一路读出缓冲区。仪器自己那句 ``NeedModule`` 就此丢失。
    """
    with pytest.raises(struct.error, match="requires a buffer"):
        pristine.parseGeneralResponse(rejected_body(), ["i", "i", "*f"])


def test_a_real_reply_read_with_the_wrong_spec_is_not_called_a_missing_module(nano):
    """回包是真的、只是我们读不懂 —— 这句话必须与「模块没加载」分得开。

    两者的处理**完全相反**:前者要改代码,后者要请用户去开前面板。
    一个对两种相反原因给出同一句话的诊断,信息量是零。
    """
    from mast.core.nanonis_patch import LAYOUT_MISMATCH_PREFIX

    err, _raw, _vals = nano.parseGeneralResponse(
        timebase_body("H"), ["i", "i", "*f"])          # 真回包 + 错规格
    assert LAYOUT_MISMATCH_PREFIX in err
    assert "NeedModule" not in err


def test_the_two_causes_no_longer_produce_the_same_message(nano):
    """仪器拒绝与回包布局错误必须保留可区分的诊断信息。"""
    rejected, _, _ = nano.parseGeneralResponse(rejected_body(), ["i", "i", "*f"])
    mismatched, _, _ = nano.parseGeneralResponse(timebase_body("H"), ["i", "i", "*f"])
    assert rejected != mismatched
    assert "unpack requires a buffer of 4 bytes" not in (rejected, mismatched)


def test_salvage_never_invents_an_empty_error(nano):
    """描述为空的拒绝也要说话 —— 空错误串会被 safe_call 判成「没出错」。"""
    err, _raw, _vals = nano.parseGeneralResponse(
        rejected_body(desc=b""), ["i", "i", "*f"])
    assert err.strip(), "一次拒绝被报成了成功"


def test_a_well_formed_reply_is_untouched_by_the_salvage_path(nano):
    """规格对得上时,补丁一个字节都不改 —— salvage 只在异常路径上存在。"""
    err, _raw, vals = nano.parseGeneralResponse(timebase_body("H"), ["H", "i", "*f"])
    assert err == ""
    assert vals[0] == CURRENT_INDEX and vals[1] == len(TIMEBASES)
    assert [round(v, 6) for v in vals[2]] == [round(v, 6) for v in TIMEBASES]


# ── 规格由**探**决定，不由断言决定 ───────────────────────────────────────────

class _WireClient:
    """一台按给定布局说话的假 Osci2T：``quickSend`` 走真实的解析器。

    与 ``test_pump.py`` 的 ``FakePool`` 差一层,而正是那一层出的事:这里伪造的是
    **线上的字节**,ResponseTypes 会被真正执行。
    """

    def __init__(self, body: bytes, *, patched: bool = True):
        self._body = body
        self._nano = _instance(_real_nanonis_class(patched=patched))
        self.sent: list[tuple] = []

    def quickSend(self, cmd, body, body_types, return_types):
        self.sent.append((cmd, tuple(return_types)))
        return tuple(self._nano.parseGeneralResponse(self._body, return_types))


@pytest.mark.parametrize("index_fmt", ["H", "i"])
def test_timebase_get_finds_whichever_layout_the_machine_speaks(index_fmt):
    """两种布局都要能读出**同一张表**。

    我们手上有两个互相矛盾的来源(库自己声明 ``H``、2026-08-02 的修复改成 ``i``,
    而本仓没有 TCP 协议手册可查),所以不硬选一条 —— 探,留下解出来自洽的那条。
    与本文件解决 ``ChsGet``/``ChGet`` 动词名的办法同一个套路。
    """
    from mast.core import nanonis_patch

    client = _WireClient(timebase_body(index_fmt))
    err, _raw, vals = nanonis_patch._patched_Osci2T_TimebaseGet(client)

    assert err == "", err
    assert vals[0] == CURRENT_INDEX
    assert vals[1] == len(TIMEBASES)
    assert [round(v, 6) for v in vals[2]] == [round(v, 6) for v in TIMEBASES]
    assert all(cmd == "Osci2T.TimebaseGet" for cmd, _ in client.sent)


def test_the_resolved_spec_is_cached_per_client():
    """探一次就够。缓存**只在解出自洽结果时**写,猜错的那条永远不会被记住。"""
    from mast.core import nanonis_patch

    client = _WireClient(timebase_body("H"))
    nanonis_patch._patched_Osci2T_TimebaseGet(client)
    first_round = len(client.sent)
    nanonis_patch._patched_Osci2T_TimebaseGet(client)
    assert len(client.sent) == first_round + 1
    assert client._mast_osci2t_tb_spec == ["H", "i", "*f"]


def test_a_wrong_first_candidate_does_not_abort_the_probe():
    """第一个候选解出一句像模像样的错误时,**不许**就此收工。

    用错规格读一份真实回包时,salvage 会把回包头 4 个字节当成「错误状态」——
    那个数其实是时基索引,几乎必然非 0,于是一份好端端的回包被报成一次拒绝。
    提前返回 = 第一个候选猜错就永久退回 Osci1T,而且理由写的是「模块没加载」。
    """
    from mast.core import nanonis_patch

    # 这台机器说 "i" 布局,而候选表里 "H" 排在前面 —— 必须走到第二条。
    client = _WireClient(timebase_body("i"))
    err, _raw, vals = nanonis_patch._patched_Osci2T_TimebaseGet(client)
    assert err == ""
    assert len(client.sent) == 2, "第一个候选就收工了"
    assert client._mast_osci2t_tb_spec == ["i", "i", "*f"]


def test_a_genuine_rejection_survives_every_candidate_and_keeps_its_text():
    """模块真的没加载时,每个候选都会解出**同一段**描述(错误段在最前面,
    与声明的字段无关),所以最后交回去的仍然是仪器自己那句话。"""
    from mast.core import nanonis_patch

    client = _WireClient(rejected_body())
    err, _raw, _vals = nanonis_patch._patched_Osci2T_TimebaseGet(client)
    assert "NeedModule" in err
    assert getattr(client, "_mast_osci2t_tb_spec", None) is None, "猜错的规格被记住了"


def test_coherence_check_rejects_a_table_that_does_not_add_up():
    """自洽判据:``count`` 必须等于数组长度,``index`` 必须落在表内。

    这是**传输层**的判据(只回答「这个 spec 解对了吗」),与
    ``monitoring.pump.check_timebase_table`` 那个面向用户的三态结论刻意分开。
    """
    from mast.core.nanonis_patch import _timebase_reply_is_coherent

    assert _timebase_reply_is_coherent([5, 6, list(TIMEBASES)]) is True
    assert _timebase_reply_is_coherent([327680, 409292, []]) is False   # 错规格的产物
    assert _timebase_reply_is_coherent([9, 6, list(TIMEBASES)]) is False  # 索引越界
    assert _timebase_reply_is_coherent([0, 3, list(TIMEBASES)]) is False  # 数不上
    assert _timebase_reply_is_coherent([0, 0, []]) is False               # 空表


# ── 为什么旧的替身抓不到（把这一课本身钉住） ─────────────────────────────────

def test_a_decoded_level_fake_cannot_tell_the_two_specs_apart():
    """直接伪造解析后列表无法验证 ResponseTypes；字节级替身应覆盖真实解析分支。"""
    class _DecodedLevelFake:
        def __init__(self):
            self.specs_seen: list[tuple] = []

        def quickSend(self, cmd, body, body_types, return_types):
            self.specs_seen.append(tuple(return_types))
            return ["", b"", [CURRENT_INDEX, len(TIMEBASES), list(TIMEBASES)]]

    from mast.core import nanonis_patch

    fake = _DecodedLevelFake()
    err, _raw, vals = nanonis_patch._patched_Osci2T_TimebaseGet(fake)
    assert err == "" and vals[1] == len(TIMEBASES)
    # 它「通过」了 —— 而且是拿第一个候选通过的，无论那个候选对不对。
    assert len(fake.specs_seen) == 1
