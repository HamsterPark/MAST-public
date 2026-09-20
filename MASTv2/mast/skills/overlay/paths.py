"""覆盖层的目录与命名派生 —— **一条路径规则派生一切**。

目录形状
========
::

    <data_root>/config/skill_overlay/
        overlay.json                  显式启用清单 + 每条选项
        builtins/bias.py              → 覆盖 mast.skills.builtins.bias
        builtins/bias.py.origin.json  eject 写的旁挂 sidecar
        _new/my_thing.py              纯新增（没有对应的内置模块）
        _packs/<pack_id>/…            签名包解出来的只读区（P2）

为什么不复用 ``config/custom_skills/``
=====================================
那条轨的语义是「**新增**一个技能」，而覆盖层的语义是「**替换**一个已有的」。
两者的门禁不同（custom 走 skill_author 的 deny-list，覆盖层不走，见 checks.py 的
威胁模型）、失败语义也不同：custom 加载失败 = 少一个技能；覆盖层加载失败 =
**必须退回内置版**，不能少。同一个目录两套规则，就是「一侧改了另一侧没跟上」的
温床。

顺带白拿一层保护：它在 ``config/`` 下，而 ``mast/update/delta.py:82`` 的
``DEFAULT_EXCLUDE_TOP`` 含 ``"config"`` —— **OTA 永远不会碰它**。

路径规则
========
::

    builtins/bias.py
      → overlay_of  = "mast.skills.builtins.bias"      （去 .py，/ → .，加前缀）
      → 模块名       = "mast.skills._overlay.builtins.bias"
      → 清单 key     = "builtins/bias.py"（posix 相对路径）

    _new/my_thing.py  → overlay_of = None（纯新增）

不做任何编码或转义 —— 路径**就是**语义。多一层映射表就多一处会漂的东西。
"""

from __future__ import annotations

import re
from pathlib import Path

#: 覆盖层加载出来的模块挂在这个包下（**不**替换 ``sys.modules`` 里的内置条目）。
OVERLAY_PKG = "mast.skills._overlay"

#: 内置技能包的前缀 —— 相对路径就是从这里往下算的。
SKILLS_PKG = "mast.skills"

MANIFEST_NAME = "overlay.json"
NEW_DIR = "_new"          # 纯新增（无对应内置模块）
PACKS_DIR = "_packs"      # 签名包解出的只读区
SIDECAR_SUFFIX = ".origin.json"

_SAFE_SEG = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def overlay_dir(*, create: bool = False) -> Path:
    from mast._runtime_paths import project_root

    d = project_root() / "config" / "skill_overlay"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path() -> Path:
    return overlay_dir() / MANIFEST_NAME


def packs_dir(*, create: bool = False) -> Path:
    d = overlay_dir() / PACKS_DIR
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def normalise_rel(rel: str) -> str:
    """清单 key 的规范形式：posix 分隔、去掉前导 ``./``。

    ⚠️ **不能用 ``lstrip("./")``**（第一版就是这么写的，冒烟当场抓到）：
    ``lstrip`` 剥的是**字符集**不是前缀，``"../etc/passwd.py"`` 会被它吃成
    ``"etc/passwd.py"`` —— 于是 :func:`is_valid_rel` 的穿越检查还没跑，
    要查的那两个点已经没了。一个安全检查被它上游的「规范化」提前消化掉，
    而且看起来完全正常。
    """
    s = str(rel or "").replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    return s


def is_valid_rel(rel: str) -> tuple[bool, str]:
    """这个相对路径能不能当覆盖层条目。返回 ``(ok, 原因)``。

    拒绝的三类，每类都是一次真实的越界：绝对路径、``..`` 穿越、非标识符的目录段
    （后者会派生出一个 import 不了的模块名，表现成「加载了但找不到」）。
    """
    s = normalise_rel(rel)
    if not s.endswith(".py"):
        return False, "不是 .py 文件"
    if s.startswith("/") or re.match(r"^[A-Za-z]:", s):
        return False, "不能用绝对路径"
    parts = s[:-3].split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False, "路径里有 . 或 ..（禁止穿越）"
    if parts[0] == PACKS_DIR:
        # _packs/<id>/rest…：第一段是包 id，可以带连字符
        if len(parts) < 3:
            return False, f"{PACKS_DIR}/ 下要形如 {PACKS_DIR}/<包id>/<模块路径>.py"
        rest = parts[2:]
    elif parts[0] == NEW_DIR:
        rest = parts[1:]
        if not rest:
            return False, f"{NEW_DIR}/ 下要有文件名"
    else:
        rest = parts
    for p in rest:
        if not _SAFE_SEG.match(p):
            return False, f"路径段 {p!r} 不是合法的 Python 标识符"
    return True, ""


def overlay_of(rel: str) -> str | None:
    """这个条目覆盖的是哪个内置模块。纯新增返回 ``None``。"""
    s = normalise_rel(rel)
    if not s.endswith(".py"):
        return None
    parts = s[:-3].split("/")
    if parts[0] == NEW_DIR:
        return None
    if parts[0] == PACKS_DIR:
        parts = parts[2:]           # 去掉 _packs/<id>
        if not parts:
            return None
        if parts[0] == NEW_DIR:
            return None
    return ".".join([SKILLS_PKG, *parts])


def rel_for_module(dotted: str) -> str | None:
    """:func:`overlay_of` 的反函数 —— ``mast.skills.builtins.bias`` → ``builtins/bias.py``。

    eject 需要走这个方向（「我要改这个模块」→「文件写到哪」）。它和
    :func:`overlay_of` **必须是同一条规则的两面**，所以放在同一个文件里，由
    ``test_rel_and_module_are_inverses`` 对全部内置模块名往返一遍钉住。
    分成两处写就是「一侧改了，另一侧没跟上」的标准形状。

    不在 ``mast.skills.`` 下的模块返回 ``None`` —— 覆盖层只覆盖技能包。
    """
    d = str(dotted or "").strip()
    prefix = SKILLS_PKG + "."
    if not d.startswith(prefix):
        return None
    rest = d[len(prefix):]
    if not rest:
        return None
    parts = rest.split(".")
    # 这两个是覆盖层自己的保留段，不是可覆盖的内置模块
    if parts[0] in (NEW_DIR, PACKS_DIR) or parts[0] == "_overlay":
        return None
    rel = "/".join(parts) + ".py"
    ok, _why = is_valid_rel(rel)
    return rel if ok else None


def module_name(rel: str) -> str:
    """加载时挂进 ``sys.modules`` 的名字。

    独立命名空间 —— **不**替换 ``sys.modules`` 里的内置条目。替换的后果：
    ``SkillRegistry.discover()`` 的冻结兜底扫的就是 ``sys.modules`` 里
    ``mast.skills.builtins.`` 前缀的东西（``core/registry.py:186-189``），
    于是**任何新建的 SkillRegistry 都会静默继承 overlay** —— 一个「行动在远处」
    的效果，排查时无从下手。

    而且替换**买不到**「直接 import 也生效」：已经持有旧类引用的那些地方不会因此
    改变，只有替换之后才发生的 import 才受影响。付全部代价，只买到不确定的一半。
    """
    s = normalise_rel(rel)[:-3]
    return f"{OVERLAY_PKG}." + s.replace("/", ".").replace("-", "_")


def sidecar_path(rel: str) -> Path:
    return overlay_dir() / (normalise_rel(rel) + SIDECAR_SUFFIX)


def entry_path(rel: str) -> Path:
    return overlay_dir() / normalise_rel(rel)


def is_pack_entry(rel: str) -> bool:
    return normalise_rel(rel).split("/")[0] == PACKS_DIR


def pack_id_of(rel: str) -> str:
    parts = normalise_rel(rel).split("/")
    return parts[1] if len(parts) > 2 and parts[0] == PACKS_DIR else ""


__all__ = ["MANIFEST_NAME", "NEW_DIR", "OVERLAY_PKG", "PACKS_DIR",
           "SIDECAR_SUFFIX", "SKILLS_PKG", "entry_path", "is_pack_entry",
           "is_valid_rel", "manifest_path", "module_name", "normalise_rel",
           "rel_for_module",
           "overlay_dir", "overlay_of", "pack_id_of", "packs_dir", "sidecar_path"]
