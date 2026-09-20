"""把一个内置技能模块「导出到覆盖层」，并告诉用户**哪些入口它管不着**。

为什么需要这个
==============
覆盖层的用法是「把改好的 ``.py`` 放进 ``config/skill_overlay/``」。但用户手上
没有那个 ``.py`` —— 打包版里全部源码都在 ``MAST.exe`` 内嵌的 PYZ 归档里，磁盘上
一个都没有。eject 就是把那份字节从包里取出来，落到覆盖层目录。

它同时做一件**比取文件更重要**的事：扫描出所有**直接 import 这个模块**的地方。
覆盖层换的是注册表里注册的类；一个 ``from mast.skills.builtins.bias import SetBias``
拿到的是模块对象上的旧引用，热重载**换不动它**。不把这些点当场列给用户，
就会出现本仓最典型的那种事故 —— 「我改了它，为什么没反应」，而界面上一切正常。

三条实现判断
============
**不写文件头注释。** 字节原样落盘。加一行 ``# ejected from …`` 之后，「我改过
没有」就要靠「减去头部再算 sha」这种脆逻辑；旁挂一个 ``.origin.json`` sidecar
之后，比对是一次裸 ``sha256``。

**扫描要覆盖「从包导入」。** ``mast/skills/builtins/__init__.py`` 显式 import 了
每一个子模块，所以 ``from mast.skills.builtins import SetBias`` 是一条**同样绕过
注册表**的绑定，而它的 ``node.module`` 是 ``mast.skills.builtins``，不是目标模块。
只按精确模块名匹配会漏掉这一整类 —— 而漏报（说「没有别的入口」其实有）正是这个
功能最坏的失败形态。所以先读目标模块顶层定义了哪些名字，再按名字回捞。

**eject 之后不自动启用。** 沿用 ``custom_loader`` 已确立的纪律：文件在目录里
不等于生效，启用是一次显式动作。取出来 → 改 → 启用 → 重载，四步各自可见。

⚠️ 与计划的一处偏离：计划说构建时用 ``ast`` 预生成 ``mast/_src/_importers.json``
供冻结环境查。实现时没做 —— 冻结下 :func:`~mast.pyexec._srcfiles.source_root`
指向的 ``mast/_src/`` 本来就有全部 ``.py``，直接扫 AST 一两秒，而预生成的索引是
一个**会和源码不同步**的构建期产物。少一个会漂的东西。
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mast.skills.overlay import paths

logger = logging.getLogger(__name__)

#: 扫描时跳过的目录段 —— 它们的 import 是结构性的，不是「绕过注册表的绑定」。
_SCAN_SKIP_DIRS = ("_src", "__pycache__", "_overlay")


@dataclass(frozen=True)
class ImportSite:
    """一处直接绑定 —— 覆盖层管不着的入口。"""

    file: str            # 相对 source_root 的 posix 路径
    line: int
    kind: str            # from_module | from_package | import_module
    names: tuple[str, ...] = ()
    text: str = ""       # 重建给人看的语句
    binds_skill: bool = False   # 绑的是技能类，还是基类/常量/辅助函数

    def as_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "kind": self.kind,
                "names": list(self.names), "text": self.text,
                "binds_skill": self.binds_skill}


@dataclass
class EjectResult:
    ok: bool = False
    reason: str = ""
    dotted: str = ""
    rel: str = ""
    path: str = ""
    sha256: str = ""
    n_bytes: int = 0
    importers: list[ImportSite] = field(default_factory=list)
    skill_classes: list[str] = field(default_factory=list)
    undecidable: list[str] = field(default_factory=list)   # 判断不了是不是技能类
    warning: str = ""    # 一句话总结，UI 直接显示（见 _warning）

    @property
    def n_skill_binds(self) -> int:
        return sum(1 for s in self.importers if s.binds_skill)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "dotted": self.dotted,
                "rel": self.rel, "path": self.path, "sha256": self.sha256,
                "n_bytes": self.n_bytes, "warning": self.warning,
                "skill_classes": list(self.skill_classes),
                "undecidable": list(self.undecidable),
                "n_skill_binds": self.n_skill_binds,
                "importers": [s.as_dict() for s in self.importers]}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _toplevel_names(src: bytes, dotted: str) -> set[str]:
    """目标模块在**顶层**定义了哪些名字（类/函数/赋值）。

    用于把 ``from mast.skills.builtins import SetBias`` 这类「从包导入」回捞成
    这个模块的绑定。语法错的模块返回空集合 —— 那种情况下扫描退化成只按模块名
    匹配，会少报，所以调用方要把这件事说出来而不是当作「没有别的入口」。
    """
    try:
        tree = ast.parse(src, filename=dotted)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


#: 技能类的根基类。``skills/`` 全树实测：BaseSkill 442、CompositeSkillGraph 42、
#: CompositeSkill 1，另有 5 个继承模块内的 ``_TipComposite``（所以要算传递闭包）。
_SKILL_BASES = frozenset({"BaseSkill", "CompositeSkillGraph", "CompositeSkill"})


def _base_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):     # base.BaseSkill → BaseSkill
        return node.attr
    return ""


def _import_map(tree) -> dict[str, tuple[str, str]]:
    """本地名 → (来源模块, 原名)。只收 ``mast.skills.*`` 的绝对导入。"""
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level:
            mod = node.module or ""
            if not mod.startswith(paths.SKILLS_PKG):
                continue
            for a in node.names:
                out[a.asname or a.name] = (mod, a.name)
    return out


def _skill_classes(src: bytes, dotted: str, *, source_root: Path | None = None,
                   _seen: frozenset[str] | None = None) -> tuple[set[str], set[str]]:
    """这个模块里**定义**了哪些技能类。返回 ``(技能类名, 判断不了的类名)``。

    为什么要分清技能类和别的名字：``from mast.skills.base import BaseSkill``
    实测有 111 处，``composite.graph_executor`` 40 处 —— 全是基础设施 import。
    把它们和「``from …bias import SetBias`` 然后 ``SetBias().execute()``」一起
    报给用户，就等于没报：111 行「这些地方不生效」他一条也用不上，而真正
    要紧的那两三行埋在里面。

    两层闭包，缺一层就会**把答案说反**：

    * 模块内 —— ``prepare_noble_tip.py`` 里 3 个类继承同模块的 ``_TipComposite``；
    * 跨模块 —— ``make_special_tip.py`` 里 2 个类继承的 ``_TipComposite`` 是从
      ``prepare_noble_tip`` **import 进来**的。只看模块内闭包，这个模块会被判成
      「一个技能类都没有」，于是 :func:`_warning` 会说出最重的那句「覆盖它一处都
      不会生效」—— 而事实恰恰相反。说反了比不说更糟。

    第二个返回值是**判断不了**的那些：基类既不是已知根基类、又解析不到来源。
    调用方必须把它当成「不知道」，不能折进「不是技能类」——「读不到」被折叠成一个
    具体的值，是本仓记了一整页的那类事故。
    """
    seen = _seen or frozenset()
    if dotted in seen:                       # 环（A 继承 B，B 所在模块又 import A）
        return set(), set()
    seen = seen | {dotted}
    try:
        tree = ast.parse(src, filename=dotted)
    except SyntaxError:
        return set(), set()

    defs: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            defs[node.name] = [_base_name(b) for b in node.bases]
    if not defs:
        return set(), set()

    imported = _import_map(tree)
    skill = {n for n, bs in defs.items() if any(b in _SKILL_BASES for b in bs)}
    changed = True
    while changed:                           # 第一层：模块内
        changed = False
        for n, bs in defs.items():
            if n not in skill and any(b in skill for b in bs):
                skill.add(n)
                changed = True

    # 第二层：基类是从别的技能模块 import 进来的 → 去那边问
    unknown: set[str] = set()
    for n, bs in defs.items():
        if n in skill:
            continue
        for b in bs:
            if not b or b in defs or b in _SKILL_BASES:
                continue
            src_mod = imported.get(b)
            if src_mod is None:
                if b not in ("Exception", "ValueError", "ABC", "Protocol",
                             "dict", "object", "str", "int", "Enum"):
                    unknown.add(n)
                continue
            if _is_skill_name_in(src_mod[0], src_mod[1], source_root, seen):
                skill.add(n)
                break
    changed = True
    while changed:                           # 跨模块解析出的结果再走一遍模块内闭包
        changed = False
        for n, bs in defs.items():
            if n not in skill and any(b in skill for b in bs):
                skill.add(n)
                changed = True
    return skill, unknown - skill


def _is_skill_name_in(mod: str, name: str, source_root: Path | None,
                      seen: frozenset[str]) -> bool:
    from mast.pyexec import _srcfiles
    try:
        p = _srcfiles.resolve(mod)
    except _srcfiles.SourceFilesMissing:
        return False
    if p is None:
        return False
    try:
        sk, _unknown = _skill_classes(p.read_bytes(), mod,
                                      source_root=source_root, _seen=seen)
    except OSError:
        return False
    return name in sk


def _iter_scan_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def scan_direct_importers(dotted: str, *, source_root: Path | None = None,
                          exported: set[str] | None = None,
                          skill_names: set[str] | None = None) -> list[ImportSite]:
    """谁在**直接** import 这个模块 —— 这些入口热重载换不动。

    三种命中形态：

    * ``from mast.skills.builtins.bias import SetBias``  → ``from_module``
    * ``from mast.skills.builtins import SetBias``       → ``from_package``
      （靠 ``exported`` 回捞；见模块 docstring 那条判断）
    * ``import mast.skills.builtins.bias``               → ``import_module``

    目标模块自身、它的包 ``__init__``、以及覆盖层自己都不算 —— 那些是结构性
    import，不是绕过注册表的绑定。
    """
    if source_root is None:
        from mast.pyexec import _srcfiles
        source_root = _srcfiles.source_root()
    root = Path(source_root)

    pkg = dotted.rsplit(".", 1)[0] if "." in dotted else ""
    self_rel = dotted[len("mast"):].lstrip(".").replace(".", "/") + ".py"
    pkg_init_rel = (pkg[len("mast"):].lstrip(".").replace(".", "/") + "/__init__.py"
                    if pkg else "")

    sites: list[ImportSite] = []
    for f in _iter_scan_files(root):
        rel = f.relative_to(root).as_posix()
        if rel in (self_rel, pkg_init_rel):
            continue
        try:
            tree = ast.parse(f.read_bytes(), filename=rel)
        except (SyntaxError, OSError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:          # 相对导入跨不了包，够不到目标
                    continue
                mod = node.module or ""
                names = tuple(a.name for a in node.names)
                if mod == dotted:
                    sites.append(ImportSite(
                        rel, node.lineno, "from_module", names,
                        "from " + mod + " import " + ", ".join(names),
                        _binds(names, skill_names)))
                elif exported and mod == pkg:
                    hit = tuple(n for n in names if n in exported)
                    if hit:
                        sites.append(ImportSite(
                            rel, node.lineno, "from_package", hit,
                            "from " + mod + " import " + ", ".join(hit),
                            _binds(hit, skill_names)))
            elif isinstance(node, ast.Import):
                hit = tuple(a.name for a in node.names if a.name == dotted)
                if hit:
                    # ``import mast.skills.builtins.bias`` 之后 ``bias.SetBias``
                    # 照样是直接绑定，而语句里没有名字可查 —— 模块有技能类就当它绑了。
                    sites.append(ImportSite(
                        rel, node.lineno, "import_module", hit,
                        "import " + hit[0], bool(skill_names)))
    sites.sort(key=lambda s: (not s.binds_skill, s.file, s.line))
    return sites


def _binds(names, skill_names) -> bool:
    return bool(skill_names) and any(n in skill_names for n in names)


def _warning(dotted: str, skills: set[str], sites: list[ImportSite],
             unknown: set[str] | None = None) -> str:
    """扫描结果 → 一句用户能据以决定的话。

    三种形态，区别是**决定不同**，不是措辞不同：

    1. 模块里没有技能类 —— 覆盖它**一处都不会生效**。覆盖层换的是注册表里注册的
       技能类；一个纯常量/辅助函数模块（``_tip_policy.py``、``base.py``）根本不经过
       注册表。这条必须说得最重：它是「改了没反应」里最难自己想明白的一种。
    2. 有技能类，且有人直接绑定了它们 —— 那几个入口继续跑内置版，其余生效。
       用户要决定的是「接受这个部分生效，还是这次得发版本」。
    3. 有技能类且没人直接绑定 —— 覆盖后全线生效，没有需要决定的事。
    """
    binds = [s for s in sites if s.binds_skill]
    if not skills and unknown:
        # 三态的第三态。折进「没有技能类」就会说出最重的那句判断，而它可能是反的。
        return ("判断不了这个模块里有没有技能类：" + "、".join(sorted(unknown)[:4])
                + " 的基类解析不到（多半是从本仓之外或动态构造的）。"
                + "覆盖之前请先确认它们是不是 BaseSkill 的子类 —— 如果不是，"
                "覆盖这个模块不会有任何效果。")
    if not skills:
        n = len(sites)
        head = "这个模块里没有技能类（没有 BaseSkill / CompositeSkillGraph 的子类）。"
        if n:
            return (head + "它是被 " + str(n) + " 处直接 import 的辅助模块 —— 常量、"
                    "函数、基类。覆盖层换的是「注册表里注册的技能类」，而这些 import "
                    "拿的是模块对象上的引用，热重载碰不到它们：覆盖这个模块「一处都"
                    "不会生效」。要改它，仍然得发版本。")
        return head + "覆盖层只对注册进注册表的技能类生效 —— 请确认这确实是你要改的东西。"
    if binds:
        files = sorted({s.file for s in binds})
        return ("有 " + str(len(binds)) + " 处直接 import 了这个模块的技能类（"
                + "、".join(files[:4]) + ("…" if len(files) > 4 else "")
                + "）。那些入口拿的是模块上的旧引用，热重载换不动 —— 它们会继续跑"
                "内置版，其余走注册表的路径会用覆盖版。要一起改，得发版本。")
    return ("没有任何地方直接 import 这个模块的技能类 —— 覆盖之后全线生效。")


def eject(dotted: str, *, overwrite: bool = False) -> EjectResult:
    """把 ``dotted`` 的源码字节原样写进覆盖层目录。**不启用。**

    已存在同名文件时默认拒绝，并在 ``reason`` 里说清楚现有那份是不是还和内置版
    一模一样 —— 「和内置一样」意味着上次导出之后没人改过，覆盖它是安全的；
    「已经不一样」意味着里面有人的改动，那就必须由人来决定。
    """
    res = EjectResult(dotted=dotted)
    rel = paths.rel_for_module(dotted)
    if not rel:
        res.reason = (dotted + " 不是可覆盖的技能模块 —— 覆盖层只覆盖 "
                      + paths.SKILLS_PKG + ".* 下的模块（架构代码改动仍需发版本）。")
        return res
    res.rel = rel

    from mast.pyexec import _srcfiles
    try:
        src_path = _srcfiles.resolve(dotted)
        root = _srcfiles.source_root()
    except _srcfiles.SourceFilesMissing as exc:
        res.reason = str(exc)
        return res
    if src_path is None:
        res.reason = "在 " + str(root) + " 下找不到 " + dotted + " 的源文件。"
        return res

    data = src_path.read_bytes()
    orig_sha = _sha(data)
    res.sha256, res.n_bytes = orig_sha, len(data)

    dest = paths.entry_path(rel)
    res.path = str(dest)
    if dest.exists() and not overwrite:
        try:
            cur = _sha(dest.read_bytes())
        except OSError:
            cur = ""
        same = cur == orig_sha
        res.reason = (
            rel + " 已经在覆盖层里了，" +
            ("而且和内置版逐字节相同（上次导出之后没改过）—— 覆盖它是安全的，"
             "请用 overwrite=True。"
             if same else
             "而且里面「已经有改动」（sha 与内置版不同）。覆盖会丢掉那些改动 —— "
             "请先备份，确认之后再用 overwrite=True。"))
        return res

    exported = _toplevel_names(data, dotted)
    skills, unknown = _skill_classes(data, dotted, source_root=root)
    res.skill_classes = sorted(skills)
    res.undecidable = sorted(unknown)
    res.importers = scan_direct_importers(dotted, source_root=root,
                                          exported=exported, skill_names=skills)
    res.warning = _warning(dotted, skills, res.importers, unknown)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)

    _write_sidecar(rel, dotted=dotted, orig_sha=orig_sha,
                   orig_rel=src_path.relative_to(root).as_posix(),
                   importers=res.importers, skill_classes=res.skill_classes,
                   warning=res.warning)

    res.ok = True
    logger.info("覆盖层 eject：%s → %s（%d 字节，%d 个技能类，%d/%d 处直接 import "
                "绑了技能类）", dotted, rel, len(data), len(skills),
                res.n_skill_binds, len(res.importers))
    if not skills or res.n_skill_binds:
        logger.warning("覆盖层 eject %s：%s", rel, res.warning)
    return res


def _write_sidecar(rel: str, *, dotted: str, orig_sha: str, orig_rel: str,
                   importers: list[ImportSite], skill_classes: list[str],
                   warning: str) -> None:
    from mast.skills.overlay.manifest import atomic_write_json

    from mast.api.version import get_version
    payload = {
        "overlay_of": dotted,
        "orig_sha256": orig_sha,
        "orig_rel": orig_rel,
        "app_version": get_version(),
        "ejected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skill_classes": list(skill_classes),
        "warning": warning,
        "importers": [s.as_dict() for s in importers],
        "note": ("orig_sha256 是导出时内置版的 sha。升级之后用 drift() 复核："
                 "内置版变了而覆盖版没跟上，就意味着这份覆盖会把上游的修复盖回去。"),
    }
    atomic_write_json(paths.sidecar_path(rel), payload)


def read_sidecar(rel: str) -> dict | None:
    """读旁挂 sidecar。

    读不到返回 ``None`` —— 「没有 sidecar」和「sidecar 是空的」是两件事，前者只是
    说明这份覆盖不是 eject 出来的（手写的也完全合法）。
    """
    p = paths.sidecar_path(rel)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


@dataclass(frozen=True)
class DriftReport:
    """内置版在这份覆盖被导出之后变过没有。"""

    rel: str
    known: bool           # 有没有 sidecar 能回答这个问题
    drifted: bool | None  # None = 不知道（没 sidecar / 读不到内置源）
    reason: str = ""
    orig_sha: str = ""
    builtin_sha: str = ""
    ejected_app_version: str = ""

    def as_dict(self) -> dict:
        return {"rel": self.rel, "known": self.known, "drifted": self.drifted,
                "reason": self.reason, "orig_sha": self.orig_sha[:12],
                "builtin_sha": self.builtin_sha[:12],
                "ejected_app_version": self.ejected_app_version}


def drift(rel: str) -> DriftReport:
    """内置版在这份覆盖导出之后改过没有？

    这是「以为生效」的另一个变体，而且更隐蔽：覆盖**确实生效了**，但它基于的是
    三个版本之前的代码，上游后来修的 bug 被它原样盖了回去。这里回答不了就报
    ``drifted=None``，绝不报 ``False`` —— 「不知道」不是「没变」。
    """
    rel = paths.normalise_rel(rel)
    sc = read_sidecar(rel)
    if not sc or not sc.get("orig_sha256"):
        return DriftReport(rel, False, None,
                           "没有 sidecar —— 这份覆盖不是 eject 出来的（手写的也合法），"
                           "无从知道它基于哪一版内置代码。")
    dotted = sc.get("overlay_of") or paths.overlay_of(rel) or ""
    from mast.pyexec import _srcfiles
    try:
        p = _srcfiles.resolve(dotted) if dotted else None
    except _srcfiles.SourceFilesMissing as exc:
        return DriftReport(rel, True, None, str(exc), sc.get("orig_sha256", ""),
                           "", sc.get("app_version", ""))
    if p is None:
        return DriftReport(rel, True, None,
                           "这一版里已经没有 " + dotted + " 了 —— 内置模块被删或改名，"
                           "这份覆盖多半也该跟着处理。",
                           sc.get("orig_sha256", ""), "", sc.get("app_version", ""))
    cur = _sha(p.read_bytes())
    orig = sc.get("orig_sha256", "")
    return DriftReport(
        rel, True, cur != orig,
        ("内置版在导出之后改过 —— 这份覆盖可能把上游的修复盖回去了，"
         "建议重新导出一份再把你的改动挪过去。" if cur != orig else ""),
        orig, cur, sc.get("app_version", ""))


def list_ejectable(*, source_root: Path | None = None) -> list[dict]:
    """哪些内置技能模块可以导出。

    返回按模块名排序的 ``{dotted, rel, n_bytes, already}``。列的是**模块**不是
    技能 —— 覆盖层的单位是文件（一个模块可能定义好几个技能，而它们必须一起换：
    半应用状态正是本仓反复被咬的形状，见 loader.py 的 dropped 检查）。
    """
    if source_root is None:
        from mast.pyexec import _srcfiles
        source_root = _srcfiles.source_root()
    root = Path(source_root)
    skills_dir = root / "skills"
    out: list[dict] = []
    if not skills_dir.is_dir():
        return out
    for f in _iter_scan_files(skills_dir):
        if f.name == "__init__.py":
            continue
        rel_fs = f.relative_to(root).as_posix()             # skills/builtins/bias.py
        dotted = "mast." + rel_fs[:-3].replace("/", ".")
        rel = paths.rel_for_module(dotted)
        if not rel:
            continue
        try:
            n = f.stat().st_size
        except OSError:
            n = 0
        out.append({"dotted": dotted, "rel": rel, "n_bytes": n,
                    "already": paths.entry_path(rel).exists()})
    out.sort(key=lambda d: d["dotted"])
    return out


__all__ = ["DriftReport", "EjectResult", "ImportSite", "drift", "eject",
           "list_ejectable", "read_sidecar", "scan_direct_importers"]
