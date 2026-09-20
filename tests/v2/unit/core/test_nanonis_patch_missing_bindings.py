"""协议里有、nanonis_spm 里没有的两个绑定（2026-08-04 补）。

两个技能一直在调根本不存在的方法，所以**永远返回失败**（``Method 'X' not found``）：

* ``SetSessionPath``      → ``Util_SessionPathSet``     （协议 p.275）
* ``PLLPerfectUpdateZTC`` → ``PLL_PerfectPLLUpdtZTC``   （协议 p.187）

「命令名存不存在」由 ``tests/v2/unit/skills/test_nanonis_command_names_exist.py``
守着。**这份守的是格式串** —— 那是另一类错，而且只在真机上现形：格式写错不会让
`hasattr` 变假，只会在真的发出去的时候变成 struct 错误或一个被 Nanonis 拒绝的报文，
而那通常发生在用户站在机台边等结果的时候。

格式串照抄库里结构相同的那条，不自己发明（见 ``nanonis_patch`` 里的注释）：
``SessionPathSet`` ← ``Util_SettingsSave``；``PerfectPLLUpdtZTC`` ← ``PLL_FreqShiftAutoCenter``。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_nanonis_patch_missing_bindings.py -q
"""
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


class _FakeNano:
    def __init__(self):
        self.sent: list[tuple] = []

    def quickSend(self, cmd, body, body_types, return_types):
        self.sent.append((cmd, body, body_types, return_types))
        return ["", b"", []]


# ════════════════════════════════════════════════════════════════════════════
# Util.SessionPathSet — 协议 p.275
# ════════════════════════════════════════════════════════════════════════════

def test_session_path_set_matches_the_protocol() -> None:
    from mast.core import nanonis_patch

    fake = _FakeNano()
    nanonis_patch._patched_Util_SessionPathSet(fake, r"D:\data\session", 1)

    cmd, body, body_types, return_types = fake.sent[0]
    assert cmd == "Util.SessionPathSet"
    assert body == [r"D:\data\session", 1]
    # 路径是「长度(int) + 字符串」——库里用 "+*c" 表示这一对，不是 "*-c"（那是回读侧）。
    assert body_types == ["+*c", "I"]
    assert return_types == [], "协议里这条只返回 Error，声明任何返回参数都会错位"


def _library_source() -> str:
    """nanonis_spm 的源码文本 —— **读文件，不 import**。

    conftest 把 ``nanonis_spm`` 换成了 MagicMock（无硬件也要能跑测试），所以
    ``inspect.getsource`` 在这里必然抛 TypeError，而 ``hasattr`` 恒为真。
    凡是要对「库里到底有什么」下断言的，都只能读磁盘上的源码。
    """
    import sysconfig

    import pytest

    # ⚠️ 不能用 ``importlib.util.find_spec``：模块已经在 ``sys.modules`` 里被换成
    # MagicMock，而 mock 没有 ``__spec__`` —— find_spec 直接抛
    # ``ValueError: nanonis_spm.__spec__ is not set``。只能按磁盘路径找。
    # （与 tests/v2/unit/skills/test_nanonis_command_names_exist.py 同一手法。）
    bases = [Path(sysconfig.get_paths()[k]) / "nanonis_spm"
             for k in ("purelib", "platlib")]
    bases += [Path(e) / "nanonis_spm" for e in sys.path if e]
    for base in bases:
        if not base.is_dir():
            continue
        for f in base.rglob("*.py"):
            text = f.read_text(encoding="utf-8", errors="replace")
            if "Util_SettingsSave" in text:
                return text
    pytest.skip("nanonis_spm 未安装在本环境（CI / 精简环境）")


def test_session_path_set_is_shaped_like_its_twin_in_the_library() -> None:
    """与 ``Util_SettingsSave`` 逐字同形 —— 它们的协议参数表就是同一个形状。

    这条钉的是「照抄而不是发明」：格式串一旦自己想当然，错的方式是安静的。
    """
    import re

    src = _library_source()
    m = re.search(r'quickSend\(\s*"Util\.SettingsSave"[^)]*?\)', src, re.S)
    assert m, "库里的 Util_SettingsSave 变样了 —— 照抄的依据没了，请重新对协议"
    assert '"+*c", "I"' in m.group(0).replace("'", '"'), m.group(0)


def test_the_save_previous_flag_is_forwarded_not_hardcoded() -> None:
    """0/1 都要原样送出去 —— 写死成 1 会在每次改 session 时偷偷覆盖上一份设置。"""
    from mast.core import nanonis_patch

    for flag in (0, 1):
        fake = _FakeNano()
        nanonis_patch._patched_Util_SessionPathSet(fake, "/tmp/s", flag)
        assert fake.sent[0][1][1] == flag


# ════════════════════════════════════════════════════════════════════════════
# PLL.PerfectPLLUpdtZTC — 协议 p.187
# ════════════════════════════════════════════════════════════════════════════

def test_perfect_pll_updt_ztc_matches_the_protocol() -> None:
    from mast.core import nanonis_patch

    fake = _FakeNano()
    nanonis_patch._patched_PLL_PerfectPLLUpdtZTC(fake, 1)

    cmd, body, body_types, return_types = fake.sent[0]
    assert cmd == "PLL.PerfectPLLUpdtZTC"
    assert body == [1]
    # Modulator index 是 int ("i")，不是 unsigned int16 ("H") —— 协议对这条写的是
    # (int)，而同模块的 DemodFilterSet 写的是 (unsigned int16)。两者混用会错位。
    assert body_types == ["i"]
    assert return_types == []


def test_the_modulator_index_is_one_based_in_the_call_we_make() -> None:
    """协议原文：「The valid values start from 1」。

    这条不判合法性（那是技能层的事），只钉住**不做 0/1 转换** —— 一次好心的
    ``index - 1`` 会让所有调用打到错误的 PLL 上，而报文本身完全合法。
    """
    from mast.core import nanonis_patch

    fake = _FakeNano()
    nanonis_patch._patched_PLL_PerfectPLLUpdtZTC(fake, 2)
    assert fake.sent[0][1] == [2]


# ════════════════════════════════════════════════════════════════════════════
# 挂载与还原
# ════════════════════════════════════════════════════════════════════════════

def test_both_are_attached_on_import() -> None:
    """``apply()`` 在 import 时跑过 —— 没挂上的话，定义了也调不到。

    ⚠️ 断言的是**身份**不是 ``hasattr``：conftest 把 ``nanonis_spm`` 换成了
    MagicMock，而 ``hasattr(MagicMock(), 任意名)`` 恒为真 —— 用 ``hasattr`` 写的
    版本在这个环境里是一条永远通过的空断言（本文件初稿就是那样，当场改掉）。
    """
    from nanonis_spm import Nanonis

    from mast.core import nanonis_patch

    assert Nanonis.Util_SessionPathSet is nanonis_patch._patched_Util_SessionPathSet
    assert (Nanonis.PLL_PerfectPLLUpdtZTC
            is nanonis_patch._patched_PLL_PerfectPLLUpdtZTC)


def test_revert_removes_them_because_there_is_no_original_to_restore() -> None:
    """这两个是**新增**不是覆盖，还原只能是删掉。

    还原成「原实现」会 AttributeError —— 库里压根没有。``_ADDED_METHODS``
    就是为了把「新增」和「覆盖」两类分开处理。
    """
    from nanonis_spm import Nanonis

    from mast.core import nanonis_patch

    try:
        nanonis_patch.revert()
        assert not hasattr(Nanonis, "Util_SessionPathSet")
        assert not hasattr(Nanonis, "PLL_PerfectPLLUpdtZTC")
    finally:
        nanonis_patch.apply()          # 别把后面的测试坑了
    assert hasattr(Nanonis, "Util_SessionPathSet")


def test_the_added_list_covers_every_binding_that_has_no_original() -> None:
    """``_ADDED_METHODS`` 漏一个，``revert()`` 就会留下一个幽灵方法。

    幽灵方法比缺失更难查：它让 ``hasattr`` 恒真，于是「patch 到底生效没有」这个
    问题再也答不上来。
    """
    import re

    from mast.core import nanonis_patch

    src = Path(nanonis_patch.__file__).read_text(encoding="utf-8")
    attached = set(re.findall(r"^\s*Nanonis\.(\w+)\s*=", src, re.M))
    originals = set(re.findall(r"^_original_(\w+)\s*=", src, re.M))
    # 挂上了、却没有 _original_ 备份的，必须在 _ADDED_METHODS 里
    need = {n for n in attached if n not in originals and n != "send"}
    missing = need - set(nanonis_patch._ADDED_METHODS)
    assert not missing, f"这些挂上了但 revert 删不掉：{sorted(missing)}"
