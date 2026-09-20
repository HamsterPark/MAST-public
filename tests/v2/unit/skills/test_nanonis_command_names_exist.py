"""用 AST 核对 safe_call 命令与库或补丁绑定，绕过会对任意属性返回值的替身。"""
from __future__ import annotations

import ast
import sys
import re
from pathlib import Path

import pytest


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return p
        p = p.parent
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
MAST = ROOT / "MASTv2" / "mast"

#: 文档/注释里的占位符，不是真调用。
_PLACEHOLDERS = {"LITERAL_VERB", "literal"}

#: 2026-08-03 发现时就已经坏了的。**这个集合只许缩小，不许增长。**
#: 修法二选一：① 若 Nanonis 协议确实有这条命令而只是绑定缺失 → 在
#: `core/nanonis_patch.py` 里补上；② 若协议本身没有 → 删掉那个技能，
#: 别把一个永远失败的技能留在 agent 的目录里。
#: **空了**（2026-08-04）。开册时冻住的两个都查清了：
#: ``Util.SessionPathSet``（协议 p.275）与 ``PLL.PerfectPLLUpdtZTC``（协议 p.187）
#: **在 TCP 协议里都存在**，缺的是 nanonis_spm 的绑定 —— 走的是①，已补进
#: ``core/nanonis_patch.py``。
_KNOWN_MISSING: set[str] = set()


def _nanonis_pkg_dir() -> Path:
    """nanonis_spm 的磁盘位置 —— **不 import**。

    conftest 把 `nanonis_spm` 换成了 MagicMock（无硬件也能跑测试），所以
    `import nanonis_spm` 拿到的是 mock：它没有 `__file__`，而且任何属性都存在，
    用它做「这个方法在不在」的判断会**恒真**——正是本测试要防的那种假绿。
    """
    import sysconfig

    for key in ("purelib", "platlib"):
        cand = Path(sysconfig.get_paths()[key]) / "nanonis_spm"
        if cand.is_dir():
            return cand
    for entry in sys.path:
        cand = Path(entry) / "nanonis_spm"
        if cand.is_dir():
            return cand
    pytest.skip("nanonis_spm 未安装在本环境（CI / 精简环境）")


def _library_methods() -> set[str]:
    """nanonis_spm 暴露的方法名，从源码 AST 取（不 import）。"""
    pkg = _nanonis_pkg_dir()
    names: set[str] = set()
    for src in pkg.glob("*.py"):
        tree = ast.parse(src.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


def _patched_methods() -> set[str]:
    """`core/nanonis_patch.py` 真正挂到 ``Nanonis`` 上的方法名。

    取的是 ``Nanonis.<name> = ...`` 赋值，**不是** ``def`` 名 —— 这里踩过一次
    （2026-08-04 当天写、当天发现）：初版抓的是 ``def (\\w+)\\(``，拿到的是
    ``_patched_Util_SessionPathSet``，**带 ``_patched_`` 前缀**，永远匹配不上
    ``safe_call("Util_SessionPathSet")``。于是这个集合恒为一堆匹配不上的名字，
    上面两条断言等于只用了 ``_library_methods()`` —— 一个看着在防护、其实一直
    是空集的守卫。

    赋值才是运行时事实，而且顺带覆盖两件 ``def`` 名答不了的事：
    别名（``Nanonis.Osci2T_ChGet = _patched_Osci2T_ChsGet``），以及
    「定义了但忘了在 ``apply()`` 里挂上」——那种函数存在却调不到。
    """
    src = (MAST / "core" / "nanonis_patch.py").read_text(encoding="utf-8")
    return set(re.findall(r"^\s*Nanonis\.(\w+)\s*=", src, re.M))


def _called_commands() -> dict[str, set[str]]:
    """{命令名: {出现的文件名}}，只取字面量实参的 safe_call。"""
    out: dict[str, set[str]] = {}
    roots = [MAST / "skills", MAST / "core", MAST / "monitoring", MAST / "instruments"]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            text = f.read_text(encoding="utf-8", errors="replace")
            for name in re.findall(r"safe_call\(\s*[\"']([A-Za-z0-9_]+)[\"']", text):
                out.setdefault(name, set()).add(f.name)
    return out


def test_the_scan_actually_finds_commands() -> None:
    """先证明扫描本身有效 —— 一个恒空的扫描会让下面每条断言都白过。"""
    called = _called_commands()
    assert len(called) > 300, f"只扫到 {len(called)} 个命令，扫描逻辑可能坏了"
    assert "ZCtrl_ZPosGet" in called, "连最常用的读 Z 都没扫到"


def test_the_library_surface_is_readable() -> None:
    """同上：库方法集合必须非空，否则「全都不存在」会是个假警报。"""
    lib = _library_methods()
    assert len(lib) > 400, f"只从 nanonis_spm 源码解析出 {len(lib)} 个方法"
    assert "ZCtrl_ZPosGet" in lib


def test_every_command_name_exists() -> None:
    """静态命令名必须对应已定义的接口。"""
    lib = _library_methods() | _patched_methods()
    called = _called_commands()

    missing = {
        name: files for name, files in called.items()
        if name not in lib and name not in _PLACEHOLDERS and name not in _KNOWN_MISSING
    }
    assert not missing, (
        "这些 Nanonis 命令名在 nanonis_spm 里不存在，也没有被 nanonis_patch 补上 —— "
        "技能会以 \"Method not found\" 失败：\n"
        + "\n".join(f"  {n}  ({', '.join(sorted(f))})" for n, f in sorted(missing.items()))
    )


def test_the_known_missing_set_has_not_grown_stale() -> None:
    """_KNOWN_MISSING 只许缩小。修好一个就从集合里删一个。

    若某条已经被 patch 补上或技能已删除，这里会红 —— 提醒把它移出豁免名单，
    免得这个集合变成一张没人再看的旧账。
    """
    lib = _library_methods() | _patched_methods()
    called = set(_called_commands())

    fixed = {n for n in _KNOWN_MISSING if n in lib}
    assert not fixed, f"这些已经能用了，请从 _KNOWN_MISSING 移除：{sorted(fixed)}"

    gone = {n for n in _KNOWN_MISSING if n not in called}
    assert not gone, f"这些已经没有调用方了，请从 _KNOWN_MISSING 移除：{sorted(gone)}"


def test_placeholders_are_really_only_documentation() -> None:
    """豁免名单里的占位符必须只出现在注释/docstring 里，不能是真调用。

    否则「豁免」就成了掩盖。
    """
    for name in _PLACEHOLDERS:
        for root in (MAST / "skills",):
            for f in root.rglob("*.py"):
                text = f.read_text(encoding="utf-8", errors="replace")
                if f'safe_call("{name}"' not in text and f"safe_call('{name}'" not in text:
                    continue
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func
                    if getattr(fn, "attr", None) != "safe_call" or not node.args:
                        continue
                    a0 = node.args[0]
                    assert not (isinstance(a0, ast.Constant) and a0.value == name), (
                        f"{f.name}: {name!r} 出现在真正的 safe_call 里，不是文档占位符"
                    )
