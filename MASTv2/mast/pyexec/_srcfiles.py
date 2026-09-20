"""``mast/**.py`` 的**文件**副本从哪来 —— dev 与冻结包各一条路。

为什么需要「文件」而不是 import
==============================
子进程里没有 MAST。要让它 ``import mast.io.nanonis_files``，只能把那个 ``.py``
**拷进会话目录**。所以 pyexec 关心的不是「这个模块能不能 import」，而是
「它的源文件在哪」—— 这两件事在 dev 下重合，在打包版里**完全不重合**：
PyInstaller 把全部 ``.py`` 编译进 ``MAST.exe`` 里的 PYZ 归档，磁盘上一个都没有。

所以打包时必须**另外**把源码作为 data 带进去（A-P1.5：``mast/_src/``）。
本模块是那件事在运行时的对应面。

失败必须响
==========
第一版找不到文件时静默跳过（``if src.is_file()``）。那意味着打包版里
``sitecustomize.py`` 不会被拷进会话 —— **审计钩子不装，保护静默消失**，而一切
看起来完全正常。这类失败是最难发现的一种，所以这里改成抛一个说得清楚的异常。
"""

from __future__ import annotations

from pathlib import Path


class SourceFilesMissing(RuntimeError):
    """拿不到 ``mast`` 的源文件副本 —— 分析环境无法安全地建起来。"""


def source_root() -> Path:
    """``mast/**.py`` 副本的根目录。

    * dev：``mast`` 包目录本身（``.py`` 就在磁盘上）
    * 冻结：``<package_root>/_src``（打包时作为 data 带进来的副本）

    两条都不在就抛 —— 绝不返回一个「看起来像目录」的路径让调用方去踩空。
    """
    from mast._runtime_paths import package_root

    root = package_root()
    frozen = root / "_src"
    if (frozen / "io" / "nanonis_files.py").is_file():
        return frozen
    if (root / "io" / "nanonis_files.py").is_file():
        return root
    raise SourceFilesMissing(
        f"找不到 mast 源文件副本（查过 {frozen} 和 {root}）。\n"
        "打包版需要构建时把 mast/**.py 作为 data 带进 mast/_src/ —— "
        "见 mast2.spec 与 A-P1.5。没有它，分析子进程既装不上审计钩子、"
        "也 import 不到 Nanonis 解析器。"
    )


def resolve(dotted: str) -> Path | None:
    """``mast.io.nanonis_files`` → 那个 ``.py`` 的路径（没有就 None）。"""
    root = source_root()
    rel = dotted[len("mast"):].lstrip(".").replace(".", "/")
    if not rel:
        return None
    for cand in (root / f"{rel}.py", root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def pyexec_file(name: str) -> Path:
    """``mast/pyexec/`` 下某个要被拷进会话的文件。找不到即抛。"""
    p = source_root() / "pyexec" / name
    if not p.is_file():
        raise SourceFilesMissing(
            f"{p} 不存在 —— 分析子进程需要它的文件副本。"
            "打包版请确认 mast/_src/pyexec/ 已随包发出。")
    return p


__all__ = ["SourceFilesMissing", "pyexec_file", "resolve", "source_root"]
