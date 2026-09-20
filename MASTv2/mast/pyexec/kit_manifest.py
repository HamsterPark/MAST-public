"""``mastkit`` —— 拷进会话目录、子进程可以直接 import 的**纯分析模块**。

为什么是拷贝而不是「另写一份」
==============================
2026-07-20 删掉过一个 ``data/formats.py``，因为它是 ``.sxm`` 的**第二套解析实现**
——两套实现会各自漂移，而漂移出来的差异没有任何测试在看。纪律是「全系统只留一套
字节 parser」。

这里拷的是**同一份代码的部署副本**，不是第二套实现（``runner``/``mastdata`` 也是
这么拷的）。让它保持「同一份」的是 :func:`verify` 的 **sha256 逐字节比对** ——
不是行为测试，因为行为测试只覆盖被测到的那条路径。

名单怎么选出来的
================
把每个候选的 **mast 传递闭包**算出来，两条判据卡掉不合格的：

* **闭包必须小且自足** —— ``mast.vision.scan_prep`` 看着很值钱（自动预处理），但
  它的闭包是 41 个模块 629 KiB，把整个 VIGIL 栈拖进来，而那需要 torch，分析运行
  时里没有。同理 ``mast.core.scan_registry`` 会拖进 ``logging.storage`` 的 SQLite
  那一套。两个都排除 —— 想参考它们的算法，去读 ``source/`` 里的源码。
* **闭包里不许有硬件** —— 名单里任何一个模块偷偷 import 了
  ``mast.core.connection`` / ``mast.instruments``，就等于**从后门拆掉 B1**。
  ``test_kit_modules_import_no_hardware`` 每次都重算这件事。

选定结果：**12 个模块 / 226 KiB**，第三方依赖只有 numpy / scipy / matplotlib /
skimage —— 全在分析运行时那六个库之内。

唯一的**可选降级**：``data/quality.py:48`` 在函数体内 import
``mast.vision.atomic_phase``（不在名单里，因为它的闭包会拖进 VIGIL）。调到那个
功能会得到一句清楚的 ImportError，其余一切正常。这类降级由 :func:`closure` 的
第二个返回值列出来，写进给 agent 的说明里 —— 撞上去才发现边界是在浪费它的回合。

目录形状（为什么不用 meta_path finder）
=====================================
按 ``mast/`` 的**原目录结构**拷贝：

    session/mast/io/nanonis_files.py      ← 字节拷贝
    session/mast/io/exp_map.py            ← 字节拷贝
    session/mast/__init__.py              ← 我们写的（说明这是子集）
    session/mastkit.py                    ← 便利别名

这样 ``mast/io/map_analysis.py`` 里那句 ``from mast.io.exp_map import ...`` 就是
一次**普通的包内 import**，既不需要改写源码（改了 sha256 就对不上），也不需要
拓扑序或 meta_path 转发。而 ``import mast.core.connection`` 天然 ``ImportError``
——因为那个文件根本不在这里。B1 由「文件不存在」保证，不由任何运行时逻辑保证。
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

# dotted name -> 是否是包的 __init__（用原文件而不是生成一个空的）
KIT_MODULES: tuple[str, ...] = (
    "mast._runtime_paths",
    "mast.core.types",
    "mast.core.si_quantity",
    "mast.io.nanonis_files",
    "mast.io.exp_map",
    "mast.io.map_analysis",
    "mast.io.product_validity",
    "mast.data",              # 包 __init__：re-export read_sxm/read_dat/read_3ds
    "mast.data.loaders",
    # 下面三个是 mast/data/__init__.py 的**相对导入**拖进来的。第一次算闭包时
    # 漏掉了它们（AST 分析跳过了 `from .x import y`），表现是子进程里
    # `import mast.data.loaders` 抛 ModuleNotFoundError: mast.data.processors。
    # 修的不是名单，是 analyse() —— 它现在也解析相对导入，
    # test_kit_manifest_is_closed 每次重算，名单再漏就会红。
    "mast.data.processors",
    "mast.data.quality",
    "mast.data.visualization",
)

# 这些中间包我们**自己生成**空 __init__（真的那几个会 eager-import 无关的东西：
# mast/__init__.py 为了修 Windows DLL 顺序会 import pyarrow，子进程里没有它）。
_SYNTHETIC_PACKAGES: tuple[str, ...] = ("mast", "mast.core", "mast.io")

_MAST_ROOT_DOC = '''"""MAST 分析模块的**子集** —— 由 mast.pyexec 拷进本会话。

这里只有不碰硬件的解析/分析代码。``mast.core.connection``、``mast.instruments``
之类根本不在这个目录里，所以 import 它们会 ImportError —— 那不是被拦下来的，
是那些文件压根没被拷进来。

想看完整源码（读，不是 import）：本会话的 ``source/`` 目录。
"""
'''

_MASTKIT_DOC = '''"""常用分析入口的便利别名 —— 和 ``mast.*`` 是同一批模块对象。

    from mastkit import read_sxm, read_dat, read_3ds

等价于 ``from mast.io.nanonis_files import ...``。两种写法都行。
"""

from mast.io.nanonis_files import (  # noqa: F401
    read_sxm, read_dat, read_3ds, read_txt, sxm_frame_meta,
)

__all__ = ["read_sxm", "read_dat", "read_3ds", "read_txt", "sxm_frame_meta"]
'''


# 这些前缀出现在闭包里 = 从后门拆掉 B1。
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "mast.instruments", "mast.core.connection", "mast.core.executor",
    "mast.core.safety", "mast.agents", "mast.api", "mast.webui", "mast.chat",
    "mast.buffer", "mast.update", "mast.llm", "mast.monitoring", "mast.conduct",
)


@dataclass(frozen=True)
class ModuleDeps:
    """一个模块的依赖，**按「必须有」和「可选降级」分开**。

    这个区分是载重的：模块级依赖缺一个，整个 import 就炸；函数内的惰性 import
    缺了只是那一个功能用不了，会给出一句清楚的 ImportError。把它们混成一堆，
    要么名单无限膨胀，要么就会像第一次那样漏掉真正必须的那个。
    """

    toplevel_mast: frozenset[str]     # 模块级 —— 必须在名单里
    deferred_mast: frozenset[str]     # 函数/方法体内 —— 缺了是可接受的降级
    third_party: frozenset[str]       # 顶级第三方包名


def analyse(dotted: str) -> ModuleDeps:
    """解析一个模块的 import。**绝对与相对导入都算**。

    相对导入是第一次算闭包时漏掉的东西：``mast/data/__init__.py`` 用
    ``from .processors import ...``，而当时的分析器直接 ``continue`` 掉了带
    ``level`` 的节点，于是名单少了三个模块，子进程里 import 才炸。
    """
    import ast

    src = _resolve(dotted)
    if src is None:
        return ModuleDeps(frozenset(), frozenset(), frozenset())
    tree = ast.parse(src.read_text(encoding="utf-8", errors="replace"))

    # 模块所属的包（相对导入的锚点）
    pkg = dotted if src.name == "__init__.py" else dotted.rsplit(".", 1)[0]

    top: set[str] = set()
    deferred: set[str] = set()
    third: set[str] = set()

    # 哪些节点在函数/类体内 —— 用一次遍历标出来
    inner: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                inner.add(id(sub))

    def _record(name: str, is_inner: bool) -> None:
        if name.startswith("mast"):
            (deferred if is_inner else top).add(name)
        else:
            third.add(name.split(".")[0])

    for node in ast.walk(tree):
        is_inner = id(node) in inner
        if isinstance(node, ast.Import):
            for a in node.names:
                _record(a.name, is_inner)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[:len(base) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if not mod:
                continue
            _record(mod, is_inner)
            if mod.startswith("mast"):
                for a in node.names:      # from mast.io import nanonis_files
                    sub = f"{mod}.{a.name}"
                    if _resolve(sub) is not None:
                        _record(sub, is_inner)

    return ModuleDeps(frozenset(top), frozenset(deferred), frozenset(third))


def closure(seeds=None) -> tuple[set[str], set[str], set[str]]:
    """名单的模块级传递闭包。

    返回 ``(闭包内的模块, 只出现在函数体内的 mast 依赖, 第三方顶级包)``。
    """
    seeds = list(seeds or KIT_MODULES)
    seen: set[str] = set()
    deferred: set[str] = set()
    third: set[str] = set()
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        if cur in seen or _resolve(cur) is None:
            continue
        seen.add(cur)
        d = analyse(cur)
        deferred |= d.deferred_mast
        third |= d.third_party
        stack.extend(x for x in d.toplevel_mast if x not in seen)
    return seen, {m for m in deferred if m not in seen}, third


@dataclass(frozen=True)
class KitFile:
    dotted: str
    src: Path
    rel: str          # 会话目录下的相对路径，如 "mast/io/nanonis_files.py"


def _resolve(dotted: str) -> Path | None:
    """模块 → 它的源**文件**。

    走 ``_srcfiles`` 而不是 ``package_root()``：打包版里 ``.py`` 都在 PYZ 归档
    里，``package_root()/io/nanonis_files.py`` 根本不存在，而我们要拷的正是那个
    文件本身。
    """
    from mast.pyexec._srcfiles import resolve as _r
    return _r(dotted)


def kit_files() -> list[KitFile]:
    """名单里每个模块的 (源文件, 会话内相对路径)。缺文件即抛 —— 静默少一个模块
    会表现成「这个库怎么没有」，是最难查的一类。"""
    out: list[KitFile] = []
    for dotted in KIT_MODULES:
        src = _resolve(dotted)
        if src is None:
            raise FileNotFoundError(f"mastkit 名单里的 {dotted} 找不到源文件")
        rel_parts = dotted.split(".")
        if src.name == "__init__.py":
            rel = "/".join(rel_parts) + "/__init__.py"
        else:
            rel = "/".join(rel_parts) + ".py"
        out.append(KitFile(dotted=dotted, src=src, rel=rel))
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def install(session_dir: Path) -> dict[str, str]:
    """把 mastkit 装进会话目录。返回 {相对路径: sha256}。"""
    session_dir = Path(session_dir)
    digests: dict[str, str] = {}

    for pkg in _SYNTHETIC_PACKAGES:
        d = session_dir / Path(*pkg.split("."))
        d.mkdir(parents=True, exist_ok=True)
        init = d / "__init__.py"
        init.write_text(_MAST_ROOT_DOC if pkg == "mast" else '"""MAST 分析子集。"""\n',
                        encoding="utf-8")

    for kf in kit_files():
        dst = session_dir / kf.rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kf.src, dst)          # 字节拷贝，一个字符都不改
        digests[kf.rel] = sha256(kf.src)

    (session_dir / "mastkit.py").write_text(_MASTKIT_DOC, encoding="utf-8")
    return digests


def verify(session_dir: Path) -> list[str]:
    """会话里那份和源文件是不是**逐字节相同**。返回不一致的相对路径列表。

    这是「同一份代码的部署副本」和「第二套实现」之间唯一的区别所在，所以它必须
    是 sha256 而不是行为比对。
    """
    session_dir = Path(session_dir)
    bad: list[str] = []
    for kf in kit_files():
        dst = session_dir / kf.rel
        if not dst.is_file() or sha256(dst) != sha256(kf.src):
            bad.append(kf.rel)
    return bad


__all__ = ["FORBIDDEN_PREFIXES", "KIT_MODULES", "KitFile", "ModuleDeps",
           "analyse", "closure", "install", "kit_files", "sha256", "verify"]
