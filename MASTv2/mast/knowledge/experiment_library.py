"""实验专属文献库 —— 一实验一库，书目权威在实验文件夹。

设计文档：``docs/v2/design/document_and_library_management.md`` §4
上位设计：``docs/v2/design/experiment_folder_persistence.md``（INCREMENTAL-ONLY）

既定的三条前提
------------------

1. **一实验一专属文献库。** 库不跨实验共享；真需要就 :func:`copy_library` 复制过去，
   库数量膨胀不是问题。
2. **整机只有一个大库**（``artifacts/literature_index/``，50k 摘要 + 205 MB 向量；
   全文在 ``data/papers/<slug>/``）。**所有文献库只是大库的指针集合。**
3. 实验**没有结束更没有归档** —— 所以这里没有 ``finalize_*``，没有终态，删除是
   一条事件而不是一次重写。

权威在哪：文件夹赢
------------------

::

    <exp_dir>/library/members.jsonl      ← ★书目权威，append-only 事件日志
    artifacts/literature_libs/registry.json  ← 全局索引 + 缓存

* 写路径**先追加 members.jsonl，后更新 registry**（同 ``chat/export.py`` 的「先写
  文件再推进 state」）。中途崩溃 = registry 落后一点，下次读或 reindex 自动追上。
* 读/冲突裁决：**文件夹赢**。registry 随时可由 :func:`rebuild_registry_from_folders`
  重建。**绝不反向**用 registry 重写文件夹 —— 唯一例外见 :func:`_adopt_once`
  （文件夹里一个字节都还没有的时候），它的存在恰恰是为了不丢东西。
* 因此「复制实验文件夹 = 带走书目」成立。

当前成员集 = 按行序折叠事件
---------------------------

``add`` / ``remove`` / ``set_fulltext`` 三种事件按行序折叠。归一化 work_id 用
``literature_index.canonical_work_id``：裸 ``W123`` 与 ``https://openalex.org/W123``
是同一篇，这个双形重复曾经真的在 registry 里存成两行（2026-07-10）。

**永不抛。** 库函数返回结果 dict（含 ``error`` / ``at_cap`` / ``folder`` 字段），
调用方（@tool / API）据此组织文案 —— 摘取文献失败不该炸掉整个 turn。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mast.knowledge import libraries as _lib

logger = logging.getLogger(__name__)

__all__ = [
    "LIBRARY_SUBDIR",
    "MEMBERS_FILE",
    "REFS_FILE",
    "refs_path",
    "experiment_library_id",
    "library_dir",
    "members_path",
    "active_experiment_id",
    "ensure_experiment_library",
    "folder_members",
    "current_members",
    "add_members",
    "remove_members",
    "set_fulltext",
    "copy_library",
    "resolve_effective_library",
    "rebuild_registry_from_folders",
]

#: 实验文件夹下的子目录名。**已在 ``MANAGED_SUBDIRS`` 与 ``_scaffold_experiment``
#: 两处注册**（两个独立列表；先注册再上写入者，否则 watcher 会去认领我们自己写的
#: 文件 —— 见 ``logging/v2/filestore.py`` 的 ``classify()``）。
LIBRARY_SUBDIR = "library"
MEMBERS_FILE = "members.jsonl"

#: 人读书目视图。**是视图不是版本** —— 每次成员集变化后全量重渲染 + 原子替换，
#: 与 ``plans/<doc>/progress.md`` 同一语义（jsonl 是增量载体，md 是它的投影）。
#:
#: 为什么必须有它：「实验文件夹自包含」这句话的兑现标准是「拷到另一台没装 MAST 的
#: 电脑，靠 ``experiment.json`` + ``README.md`` 就能读懂」。只有 ``members.jsonl``
#: 的话，那台电脑上的人得**自己在脑子里折叠一遍事件日志**才知道这个实验参考了哪些
#: 文献 —— 那不叫读得懂。
REFS_FILE = "refs.md"

#: ``library/`` 自己的身份边车：``{library_id, experiment_id, name, created_at}``。
#:
#: 为什么不只靠 ``experiment.json``：那是实验的身份文件，``library/`` 借它当然可以，
#: 但一旦它缺失（半途崩溃、手工拷了个只含 library/ 的目录），从文件夹重建就只能
#: 猜 id —— 而目录名里只有 8 位，猜不出完整 uuid。有了这个边车，``library/``
#: **单独一个目录就能被解释**。
#:
#: 它是**副本不是权威**：重建时先认 ``experiment.json`` 的 ``id``，认不到才用这里。
LIBRARY_JSON = "library.json"

#: 实验库 id 的前缀。``exp_<id8>`` 是**确定性**的：给同一个 experiment_id 永远推导出
#: 同一个 library_id，所以不需要任何绑定映射表（旧设计的
#: ``experiment_library_binding`` 在新模型下是恒等关系，整张表消失）。
_ID_PREFIX = "exp_"

_VALID_FULLTEXT = ("none", "requested", "ingested")
_VALID_SOURCES = ("openalex", "user_pdf", "user_url", "user_manual")

#: 每个实验一把锁，护住「折叠 → 追加 → 同步 registry」这一段。跨进程靠 append-only
#: 的单行原子追加兜底（两个进程各追加自己的行，折叠结果仍然自洽）。
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(experiment_id: str) -> threading.Lock:
    with _locks_guard:
        lk = _locks.get(experiment_id)
        if lk is None:
            lk = _locks[experiment_id] = threading.Lock()
        return lk


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canon(work_id: str) -> str:
    """归一到裸 ``W…``（``local:`` 与非 OpenAlex id 原样通过）。

    复用 ``libraries._canonical_work_id`` —— 它已经处理了「重依赖缺失时退回内联
    正则」，这个轻量模块不该在 import 时拖进 numpy/pandas/httpx。
    """
    return _lib._canonical_work_id(work_id)


# ── 落盘原语（只允许这两种形态）────────────────────────────────────────
def _append_jsonl(path: Path, obj: dict) -> bool:
    """原子追加一行 JSON + flush。**永不抛**，返回是否写成。

    与 ``documents/store.py`` 的同名函数同款（刻意各写一份而不是 import 私有函数）：
    崩溃最多丢最后一行，读侧丢弃不可解析的行。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
        return True
    except OSError as exc:
        logger.warning("members.jsonl 追加失败 (%s): %r", path, exc)
        return False


def _replace_atomic(path: Path, text: str) -> bool:
    """``.part`` + ``os.replace``。**永不抛**。

    两个用途，都不是「书目权威」：:func:`_write_refs_md` 的人读视图（视图可以反复
    重写），和 :func:`_adopt_once` 的一次性播种。权威 ``members.jsonl`` 走追加。
    """
    tmp = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".part",
                                   dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.warning("members.jsonl 原子替换失败 (%s): %r", path, exc)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


# ── 身份与落点 ────────────────────────────────────────────────────────
def experiment_library_id(experiment_id: str) -> str:
    """``exp_<experiment_id 前 8 位字母数字，小写>``。确定性、幂等可重推导。

    与 ``documents.model.doc_id8`` 同一套规则（同样是「取 id 前 8 位嵌进名字」），
    所以直接复用那个函数而不是再抄一遍。
    """
    from mast.documents.model import doc_id8

    eid = str(experiment_id or "").strip()
    if not eid:
        return ""
    return _ID_PREFIX + doc_id8(eid)


def library_dir(experiment_id: str, *, create: bool = False) -> Path | None:
    """``<exp_dir>/library/``。实验行不存在返回 ``None``。"""
    from mast.documents.paths import exp_dir_for

    eid = str(experiment_id or "").strip()
    if not eid:
        return None
    try:
        exp_dir = exp_dir_for(eid, create=create)
    except Exception as exc:  # noqa: BLE001 — 解析失败退化成「无文件夹」，不抛
        logger.warning("library_dir(%s): 实验目录解析失败 %r", eid, exc)
        return None
    if exp_dir is None:
        return None
    d = exp_dir / LIBRARY_SUBDIR
    if create:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("library_dir(%s): mkdir 失败 %r", eid, exc)
            return None
    return d


def members_path(experiment_id: str, *, create: bool = False) -> Path | None:
    """``<exp_dir>/library/members.jsonl``。拿不到实验目录返回 ``None``。"""
    d = library_dir(experiment_id, create=create)
    return None if d is None else d / MEMBERS_FILE


def refs_path(experiment_id: str, *, create: bool = False) -> Path | None:
    """``<exp_dir>/library/refs.md`` —— 人读书目视图（**不是权威**）。"""
    d = library_dir(experiment_id, create=create)
    return None if d is None else d / REFS_FILE


# ── registry 侧的懒创建 ───────────────────────────────────────────────
def ensure_experiment_library(
    experiment_id: str, *, name: str | None = None,
    registry: "_lib.LibraryRegistry | None" = None,
) -> str:
    """懒创建该实验的专属库，返回 ``library_id``。**幂等**，永不抛。

    刻意**不**在实验创建时同步建库：实验创建有多条路径（工具 / GUI / reindex），
    逐一挂钩既脆弱又会用空库污染 registry。首次真正用到时（lib_add / ingest /
    copy 目标 / 显式 ensure API）走这里一个 choke point，与「空实验不预建目录」
    同哲学。

    库记录带 ``scope="experiment"`` 和 ``experiment_id`` 字段 —— 后者是新增的，
    没有它的 ``scope="experiment"`` 老记录会被 ``libraries`` 按 custom 对待并告警
    （装饰性标签是现状的 bug，不再制造新的）。
    """
    eid = str(experiment_id or "").strip()
    lib_id = experiment_library_id(eid)
    if not lib_id:
        return ""
    display = name or _default_name(eid)
    reg = _lib._reg(registry)
    try:
        reg.ensure_experiment_record(lib_id, experiment_id=eid, name=display)
    except Exception as exc:  # noqa: BLE001 — registry 只是索引，坏了不该阻塞书目
        logger.warning("ensure_experiment_library(%s): registry 写入失败 %r", eid, exc)
    _write_library_json(eid, lib_id, display)
    return lib_id


def _write_library_json(experiment_id: str, library_id: str, name: str) -> None:
    """写 ``library/library.json`` 身份边车。已存在就不动（创建时冻结），永不抛。"""
    d = library_dir(experiment_id, create=True)
    if d is None:
        return
    path = d / LIBRARY_JSON
    if path.exists():
        return
    try:
        from mast.logging.v2.manifest import write_json_atomic
        write_json_atomic(path, {
            "schema_version": "1.0.0",
            "library_id": library_id,
            "experiment_id": experiment_id,
            "name": name,
            "created_at": _now_iso(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("library.json 写入失败 (%s): %r", path, exc)


def _default_name(experiment_id: str) -> str:
    """库的显示名：跟实验同名（拿不到就用 id 前 8 位）。

    只在**创建那一刻**取一次 —— 之后实验改名不追改库名，同「目录名创建时冻结」。
    """
    from mast.documents.model import doc_id8

    try:
        from mast.documents.paths import storage
        exp = storage().get_experiment(experiment_id) or {}
        title = str(exp.get("name") or "").strip()
        if title:
            return f"{title} 文献库"
    except Exception:  # noqa: BLE001
        pass
    return f"实验 {doc_id8(experiment_id)} 文献库"


# ── 事件折叠 ──────────────────────────────────────────────────────────
def _fold(lines: Iterable[str]) -> list[dict[str, Any]]:
    """把事件行折叠成当前成员集（保持首次 add 的顺序）。

    * ``add``：新 key 建行；已存在则原地升级（reason / source / doi），``added_at``
      保留最早那次 —— 与 registry 的 upsert 语义一致。
    * ``remove``：整个 key 移除。**之后再 add 会重新出现**（这是刻意的：删除是事件，
      不是墓碑）。
    * ``set_fulltext``：命中成员就写状态，**同时无条件记进 ``last_ft``**。

    ``last_ft`` 为什么跨 remove 存活（2026-07-29 审查发现）
    ------------------------------------------------------

    它同时解决两件事，而两件事的根子是同一条：**书目是策展决定（可以反复加/删），
    全文是关于这台机器的事实（不因为你把指针删了就消失）。** 两者不该共命运。

    1. ``set_fulltext`` 先到、``add`` 后到（ingest 与「加指针」是两个独立调用，
       谁先谁后取决于调用方）—— 状态要等成员进来时补上；
    2. ``add → set_fulltext → remove → add`` —— 如果 remove 把它一起清掉，第二次
       add 建的新行会显示「没有全文」，而 PDF 其实还躺在 ``papers/<slug>`` 里。
       后果不是记账瑕疵：agent 会据此再发一次 ``request_fulltext``，用户被要求
       上传一份这台机器上已经有的文件。
    """
    cur: dict[str, dict[str, Any]] = {}
    #: work_id → (status, ref)，**只增不减**（remove 不清它）。
    last_ft: dict[str, tuple[str, str | None]] = {}
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue  # 半行（崩在追加中途）—— 丢弃，不让它毒化整份书目
        if not isinstance(ev, dict):
            continue
        wid = _canon(str(ev.get("work_id") or ""))
        if not wid:
            continue
        op = str(ev.get("op") or "").strip().lower()
        at = str(ev.get("at") or "")

        if op == "add":
            existing = cur.get(wid)
            if existing is None:
                m = {
                    "work_id": wid,
                    "doi": str(ev.get("doi") or ""),
                    "added_by": "agent" if str(ev.get("added_by") or "").startswith("agent") else "user",
                    "added_at": at,
                    "reason": str(ev.get("reason") or ""),
                    "source": (ev.get("source") if ev.get("source") in _VALID_SOURCES
                               else "openalex"),
                    "fulltext_status": "none",
                    "fulltext_ref": None,
                }
                if ev.get("copied_from"):
                    m["copied_from"] = str(ev["copied_from"])
                ft = last_ft.get(wid)
                if ft is not None:
                    m["fulltext_status"], m["fulltext_ref"] = ft
                cur[wid] = m
            else:
                if ev.get("reason"):
                    existing["reason"] = str(ev["reason"])
                if ev.get("doi"):
                    existing["doi"] = str(ev["doi"])
                if ev.get("source") in _VALID_SOURCES:
                    existing["source"] = ev["source"]
        elif op == "remove":
            # 刻意不动 last_ft：移除的是**指针**，不是这台机器上的那份 PDF。
            cur.pop(wid, None)
        elif op == "set_fulltext":
            status = ev.get("fulltext_status")
            status = status if status in _VALID_FULLTEXT else "none"
            ref = ev.get("fulltext_ref")
            ref = str(ref) if isinstance(ref, str) and ref else None
            last_ft[wid] = (status, ref)
            if wid in cur:
                cur[wid]["fulltext_status"] = status
                cur[wid]["fulltext_ref"] = ref
    return list(cur.values())


def _read_events(path: Path | None) -> list[str] | None:
    """读事件行。文件不存在返回 ``None``（**区别于空列表** —— 「还没有文件夹权威」
    和「文件夹说这里没有成员」是两件事，后者才允许覆盖 registry 缓存）。"""
    if path is None or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        logger.warning("members.jsonl 读取失败 (%s): %r", path, exc)
        return None


def folder_members(experiment_id: str) -> list[dict[str, Any]] | None:
    """折叠该实验的 ``members.jsonl``；**文件不存在返回 ``None``**。

    ``None`` 与 ``[]`` 的区别是本模块的核心不变式：``None`` = 「文件夹还没有发言权」
    （此时绝不能用它去覆盖 registry 缓存，否则就是反向抹除），``[]`` = 「文件夹说这里
    没有成员」（可以覆盖）。

    ``libraries.LibraryRegistry.get_library`` 走这个函数拿权威成员 —— 所以这里
    **不许回头调 registry**，否则两边互相调用就是死循环。
    """
    events = _read_events(members_path(experiment_id))
    return None if events is None else _fold(events)


def current_members(experiment_id: str, *,
                    registry: "_lib.LibraryRegistry | None" = None) -> list[dict[str, Any]]:
    """该实验专属库的当前成员集。文件夹有就用文件夹（**文件夹赢**），否则退回 registry。"""
    folded = folder_members(experiment_id)
    if folded is not None:
        return folded
    try:
        rec = _lib._reg(registry).get_library(experiment_library_id(experiment_id))
        return [dict(m) for m in (rec.get("members") or [])]
    except Exception:  # noqa: BLE001 — 库还不存在 → 空
        return []


# ── 唯一一次允许的反向写：把 registry 已有成员播种进空文件夹 ────────────
def _adopt_once(experiment_id: str, path: Path,
                registry: "_lib.LibraryRegistry | None" = None) -> None:
    """文件夹里**一个字节都还没有**、而 registry 已有成员时，把它们写成 add 事件。

    这是本模块唯一一次 registry→文件夹的写入，条件严格到不可能丢东西：文件不存在
    才播种。不做的话，一个「先有 registry 成员、后有文件夹」的库（迁移、或
    :func:`copy_library` 崩在两步之间）会在下一次 add 时被折叠成只剩新那一条 ——
    静默丢书目。实盘目前为零，这是防御性的。
    """
    if path.exists():
        return
    try:
        rec = _lib._reg(registry).get_library(experiment_library_id(experiment_id))
    except Exception:  # noqa: BLE001
        return
    members = rec.get("members") or []
    if not members:
        return
    lines = []
    for m in members:
        lines.append(json.dumps({
            "op": "add",
            "work_id": _canon(str(m.get("work_id") or "")),
            "doi": str(m.get("doi") or ""),
            "reason": str(m.get("reason") or ""),
            "source": m.get("source") if m.get("source") in _VALID_SOURCES else "openalex",
            "added_by": m.get("added_by") or "user",
            "at": str(m.get("added_at") or _now_iso()),
            "note": "adopted from registry cache",
        }, ensure_ascii=False))
        if m.get("fulltext_status") in ("requested", "ingested"):
            lines.append(json.dumps({
                "op": "set_fulltext",
                "work_id": _canon(str(m.get("work_id") or "")),
                "fulltext_status": m.get("fulltext_status"),
                "fulltext_ref": m.get("fulltext_ref"),
                "at": _now_iso(),
            }, ensure_ascii=False))
    if _replace_atomic(path, "\n".join(lines) + "\n"):
        logger.info("experiment_library: 已把 registry 的 %d 个成员播种进 %s",
                    len(members), path)


def _sync_registry(experiment_id: str, lib_id: str,
                   registry: "_lib.LibraryRegistry | None" = None) -> None:
    """折叠文件夹 → 刷新 registry 缓存 + 重写 ``refs.md``。**双写的第二步**。

    只在文件夹**真的有文件**时才刷 —— 否则会用一个空集合抹掉 registry 里的成员
    （反向覆盖，正是设计明令禁止的方向）。两个副产物都是 best-effort：写不出去只
    告警，绝不让一次 ``lib_add`` 因为渲染不出视图而失败。
    """
    events = _read_events(members_path(experiment_id))
    if events is None:
        return
    members = _fold(events)
    try:
        _lib._reg(registry).replace_members_cache(lib_id, members)
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry 缓存刷新失败 (%s): %r", lib_id, exc)
    _write_refs_md(experiment_id, lib_id, members)


def _write_refs_md(experiment_id: str, lib_id: str,
                   members: list[dict[str, Any]]) -> bool:
    """重写 ``library/refs.md``（人读书目视图）。**永不抛**，返回是否写成。"""
    d = library_dir(experiment_id, create=True)
    if d is None:
        return False
    try:
        return _replace_atomic(d / REFS_FILE,
                               _render_refs_md(experiment_id, lib_id, members))
    except Exception as exc:  # noqa: BLE001 — 视图渲染失败绝不影响书目本身
        logger.warning("refs.md 渲染失败 (%s): %r", experiment_id, exc)
        return False


def _render_refs_md(experiment_id: str, lib_id: str,
                    members: list[dict[str, Any]]) -> str:
    """渲染书目 markdown。阅读顺序 = 策展顺序。

    只按 ``added_at`` 排，**不加 work_id 做次键**：时间戳精度到秒，一次批量 lib_add
    的几条 ``added_at`` 完全相同，拿 work_id 当次键就会把它们按字母序打乱成
    W100/W200/W300，读起来像是有含义而其实没有。``sorted`` 是稳定的，并列时保持
    ``_fold`` 给出的首次-add 顺序 —— 那才是真正的策展顺序。
    """
    title = ""
    try:
        from mast.documents.paths import storage
        title = str((storage().get_experiment(experiment_id) or {}).get("name") or "")
    except Exception:  # noqa: BLE001 — 没有 DB 也要能渲染（这正是自包含的场景）
        title = ""

    rows = sorted(members, key=lambda m: str(m.get("added_at") or ""))
    lines = [
        f"# 参考文献 —— {title or '实验 ' + experiment_id[:8]}",
        "",
        f"- 文献库：`{lib_id}`",
        f"- 条目数：{len(rows)}",
        f"- 生成时间：{_now_iso()}",
        "",
        "> 这份清单是**指针**：每一条指向本机大库（`artifacts/literature_index/`）",
        "> 里的一篇论文，全文（如果有）在本机 `data/papers/<slug>/`。",
        "> **这两处都不在实验文件夹里** —— 所以在一台没装 MAST 的电脑上，下面的",
        "> `papers/…` 路径打不开，work_id 和 DOI 仍然可以拿去检索原文。",
        "> ",
        "> 本文件由 MAST 从 `members.jsonl` 全量重渲染，**手工修改会被下一次覆盖**；",
        "> 书目的权威是 `members.jsonl`（追加式事件日志）。",
        "",
    ]
    if not rows:
        lines += ["_这个实验还没有收录文献。_", ""]
        return "\n".join(lines)

    lines += [
        "| # | work_id | DOI | 加入理由 | 来源 | 加入时间 | 全文 |",
        "|---|---------|-----|----------|------|----------|------|",
    ]
    for n, m in enumerate(rows, 1):
        ref = m.get("fulltext_ref")
        status = str(m.get("fulltext_status") or "none")
        if status == "ingested" and ref:
            ft = f"本机 `{ref}/`"
        elif status == "requested":
            ft = "已请求"
        else:
            ft = "—"
        src = str(m.get("source") or "")
        if m.get("copied_from"):
            src += f"（复制自 `{m['copied_from']}`）"
        lines.append(
            f"| {n} | `{_md_cell(m.get('work_id'))}` | {_md_cell(m.get('doi')) or '—'} "
            f"| {_md_cell(m.get('reason')) or '—'} | {_md_cell(src)} "
            f"| {_md_cell(m.get('added_at'))} | {ft} |"
        )
    lines.append("")
    return "\n".join(lines)


def _md_cell(value: Any) -> str:
    """把任意值放进 markdown 表格单元格：转义 ``|``，换行压成空格。

    没有这一步，一条带竖线或换行的 ``reason``（模型完全写得出来）会把整张表的列
    对齐弄坏 —— 视图坏掉本身不严重，但它会让人以为书目数据坏了。
    """
    s = str(value or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return s.strip()


# ── 成员增删 ──────────────────────────────────────────────────────────
def add_members(
    experiment_id: str,
    work_ids: list[str] | tuple[str, ...] | str,
    *,
    reason: str = "",
    source: str = "openalex",
    added_by: str = "agent",
    doi_by_work_id: dict[str, str] | None = None,
    copied_from: str = "",
    registry: "_lib.LibraryRegistry | None" = None,
) -> dict[str, Any]:
    """把指针加进实验专属库。返回与 ``libraries.add_members`` 同形的结果 dict。

    ``{"library_id", "added", "skipped", "rejected", "member_count", "at_cap",
    "folder", "error"}``。撞 ``MAX_MEMBERS`` 时**如实上报**（进 ``rejected`` +
    ``at_cap=True``），不静默截断 —— 静默截断会让 agent 以为文献已经在库里了。

    ``copied_from``：由 :func:`copy_library` 填源库 id。它**不写进 ``source``** ——
    ``source`` 记的是这条书目本身的来历（openalex / user_pdf / …），拿 ``"copy:<id>"``
    去盖掉就把真来历弄丢了，而且那个值过不了 registry 与 pydantic 的枚举。两件事
    分两个字段记，都留住。
    """
    if isinstance(work_ids, str):
        work_ids = [work_ids]
    if source not in _VALID_SOURCES:
        source = "openalex"
    doi_by_work_id = doi_by_work_id or {}

    eid = str(experiment_id or "").strip()
    lib_id = ensure_experiment_library(eid, registry=registry)
    if not lib_id:
        return _empty_result("", error="没有实验 id —— 无法解析实验专属库。")

    path = members_path(eid, create=True)
    added: list[str] = []
    skipped: list[str] = []
    rejected: list[str] = []

    with _lock_for(eid):
        if path is None:
            # 实验行不在 / 目录建不出来 → 退回 registry-only（成员不丢，如实标注）。
            res = _registry_fallback_add(
                lib_id, work_ids, reason=reason, source=source,
                added_by=added_by, doi_by_work_id=doi_by_work_id, registry=registry)
            res["folder"] = False
            return res

        _adopt_once(eid, path, registry)
        cur = {m["work_id"]: m for m in _fold(_read_events(path) or [])}
        now = _now_iso()
        for raw in work_ids:
            if not _lib._is_valid_work_id(raw):
                rejected.append(str(raw))
                continue
            wid = _canon(raw)
            already = wid in cur
            if not already and len(cur) >= _lib.MAX_MEMBERS:
                rejected.append(wid)
                continue
            ev = {
                "op": "add", "work_id": wid,
                "doi": str(doi_by_work_id.get(wid, "") or ""),
                "reason": reason, "source": source,
                "added_by": added_by, "at": now,
            }
            if copied_from:
                ev["copied_from"] = copied_from
            # 已在库里也落一条 add 事件：reason / source 可能是新的，而事件日志就是
            # 审计轨迹（谁在什么时候又提了一次这篇）。折叠时原地升级，不会重复成员。
            ok = _append_jsonl(path, ev)
            if already:
                skipped.append(wid)
                continue
            if not ok:
                rejected.append(wid)
                continue
            cur[wid] = {"work_id": wid}
            added.append(wid)

        _sync_registry(eid, lib_id, registry)

    return {
        "library_id": lib_id,
        "added": added,
        "skipped": skipped,
        "rejected": rejected,
        "member_count": len(cur),
        "at_cap": len(cur) >= _lib.MAX_MEMBERS,
        "folder": True,
        "error": "",
    }


def remove_members(
    experiment_id: str,
    work_ids: list[str] | tuple[str, ...] | str,
    *,
    registry: "_lib.LibraryRegistry | None" = None,
) -> dict[str, Any]:
    """从实验专属库移除指针。**删除是事件，不是重写**（INCREMENTAL-ONLY）。

    返回 ``{"library_id", "removed", "member_count", "n_removed", "folder", "error"}``。
    """
    if isinstance(work_ids, str):
        work_ids = [work_ids]
    eid = str(experiment_id or "").strip()
    lib_id = ensure_experiment_library(eid, registry=registry)
    if not lib_id:
        return {"library_id": "", "removed": [], "member_count": 0, "n_removed": 0,
                "folder": False, "error": "没有实验 id —— 无法解析实验专属库。"}

    path = members_path(eid, create=True)
    with _lock_for(eid):
        if path is None:
            # 同 _registry_fallback_add：走 registry-only 的核心，避免互相委托死循环。
            try:
                res = _lib._reg(registry)._remove_members_registry_only(
                    list(work_ids), lib_id)
            except Exception as exc:  # noqa: BLE001
                return {"library_id": lib_id, "removed": [], "member_count": 0,
                        "n_removed": 0, "folder": False, "error": str(exc)}
            res["folder"] = False
            res["error"] = ""
            return res

        _adopt_once(eid, path, registry)
        cur = {m["work_id"] for m in _fold(_read_events(path) or [])}
        now = _now_iso()
        removed: list[str] = []
        for raw in work_ids:
            if not _lib._is_valid_work_id(raw):
                continue
            wid = _canon(raw)
            if wid not in cur:
                continue
            if _append_jsonl(path, {"op": "remove", "work_id": wid, "at": now}):
                removed.append(wid)
                cur.discard(wid)
        _sync_registry(eid, lib_id, registry)

    return {"library_id": lib_id, "removed": removed, "member_count": len(cur),
            "n_removed": len(removed), "folder": True, "error": ""}


def set_fulltext(
    experiment_id: str, work_id: str, status: str = "ingested",
    ref: str | None = None, *,
    registry: "_lib.LibraryRegistry | None" = None,
) -> dict[str, Any]:
    """记录某成员的全文状态 —— 这是 ``fulltext_status`` / ``fulltext_ref`` 从
    「schema 里有槽位但全仓库没人写过」变成活字段的地方。

    ``ref`` 按约定写相对形式 ``papers/<slug>``（全文是机器级资产，不随实验文件夹
    走，所以存相对路径而不是绝对路径 —— 换机器仍可解析）。
    """
    eid = str(experiment_id or "").strip()
    wid = _canon(work_id)
    if not wid:
        return {"ok": False, "error": "work_id 为空。"}
    if status not in _VALID_FULLTEXT:
        status = "none"
    lib_id = ensure_experiment_library(eid, registry=registry)
    if not lib_id:
        return {"ok": False, "error": "没有实验 id —— 无法解析实验专属库。"}

    path = members_path(eid, create=True)
    with _lock_for(eid):
        if path is None:
            ok = _registry_set_fulltext(lib_id, wid, status, ref, registry)
            return {"ok": ok, "library_id": lib_id, "work_id": wid,
                    "fulltext_status": status, "fulltext_ref": ref, "folder": False,
                    "error": "" if ok else "registry 写入失败。"}
        ok = _append_jsonl(path, {
            "op": "set_fulltext", "work_id": wid,
            "fulltext_status": status, "fulltext_ref": ref, "at": _now_iso(),
        })
        if ok:
            _sync_registry(eid, lib_id, registry)
    return {"ok": ok, "library_id": lib_id, "work_id": wid,
            "fulltext_status": status, "fulltext_ref": ref, "folder": True,
            "error": "" if ok else "members.jsonl 追加失败。"}


# ── registry-only 退化路径（拿不到实验文件夹时）────────────────────────
def _empty_result(lib_id: str, *, error: str = "") -> dict[str, Any]:
    return {"library_id": lib_id, "added": [], "skipped": [], "rejected": [],
            "member_count": 0, "at_cap": False, "folder": False, "error": error}


def _registry_fallback_add(lib_id: str, work_ids, *, reason: str, source: str,
                           added_by: str, doi_by_work_id: dict[str, str],
                           registry) -> dict[str, Any]:
    """实验目录拿不到时的退化落点：写进 registry，成员不丢，如实标 ``folder=False``。

    刻意调 ``_add_members_registry_only`` 而不是公开的 ``add_members`` —— 后者会看到
    这是个实验库、把调用**转回本模块**，两边互相委托就是死循环。
    """
    try:
        res = _lib._reg(registry)._add_members_registry_only(
            list(work_ids), lib_id, reason=reason, source=source,
            added_by=("agent" if str(added_by).startswith("agent") else "user"),
            doi_by_work_id=doi_by_work_id)
        res["error"] = ""
        return res
    except Exception as exc:  # noqa: BLE001
        return _empty_result(lib_id, error=str(exc))


def _registry_set_fulltext(lib_id: str, wid: str, status: str, ref: str | None,
                           registry) -> bool:
    try:
        return bool(_lib._reg(registry).set_member_fulltext(lib_id, wid, status, ref))
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry set_fulltext 失败 (%s/%s): %r", lib_id, wid, exc)
        return False


# ── copy 替代跨实验共享 ───────────────────────────────────────────────
def copy_library(
    src_library_id: str, to_experiment_id: str = "", *,
    registry: "_lib.LibraryRegistry | None" = None,
) -> dict[str, Any]:
    """把 *src* 库的成员复制进某实验的专属库（``to_experiment_id`` 空 = 当前实验）。

    定案用**复制**代替共享：库不跨实验共享，膨胀不是问题。

    * 每条记 ``copied_from=<src_id>``（**不动 ``source``** —— 见
      :func:`add_members` 的说明），``reason`` 原样保留。
    * **目标库里已有的条目不动。** 覆写它们的 ``reason`` 会用源库的措辞盖掉目标
      实验自己的批注，那是一次静默的信息损失；「复制」的直觉本来也是「补上缺的」。
    * 全文状态跟着搬 —— ``fulltext_ref`` 指向机器级的 ``papers/<slug>``，两个库
      指的是同一份全文，不产生副本。
    * 撞 ``MAX_MEMBERS`` 如实上报（``at_cap`` / ``rejected``），不静默截断。
    """
    src_id = str(src_library_id or "").strip()
    if not src_id:
        return {"ok": False, "error": "src_library_id 为空。"}

    eid = str(to_experiment_id or "").strip()
    if not eid:
        eid = _active_experiment_id()
    if not eid:
        return {"ok": False, "error": "没有活跃实验，也没给 to_experiment_id —— "
                                      "无法确定复制目标。"}

    dst_id = experiment_library_id(eid)
    if src_id == dst_id:
        return {"ok": False, "error": f"源库与目标库是同一个（{dst_id}）。"}

    # 源可以是任何库：非实验库读 registry，实验库读它自己的文件夹（文件夹赢）。
    src_members = _members_of_any(src_id, registry)
    if src_members is None:
        return {"ok": False, "error": f"没有这个库：{src_id!r}"}
    if not src_members:
        return {"ok": True, "src_library_id": src_id, "library_id": dst_id,
                "copied": [], "skipped": [], "rejected": [], "member_count": 0,
                "at_cap": False, "error": "",
                "note": f"源库 {src_id} 没有成员，什么都没复制。"}

    copied: list[str] = []
    skipped: list[str] = []
    rejected: list[str] = []
    member_count = 0
    at_cap = False
    already = {m["work_id"] for m in current_members(eid, registry=registry)}
    # 逐条 add 而不是批量：每条的 reason / source / doi 都不一样，一次批量调用会把
    # 所有条目盖上同一个 reason。
    for m in src_members:
        wid = _canon(str(m.get("work_id") or ""))
        if not wid:
            continue
        if wid in already:
            skipped.append(wid)
            continue
        res = add_members(
            eid, [wid],
            reason=str(m.get("reason") or ""),
            source=str(m.get("source") or "openalex"),
            added_by=str(m.get("added_by") or "user"),
            doi_by_work_id={wid: str(m.get("doi") or "")},
            copied_from=src_id,
            registry=registry,
        )
        copied.extend(res.get("added") or [])
        rejected.extend(res.get("rejected") or [])
        member_count = int(res.get("member_count") or member_count)
        at_cap = at_cap or bool(res.get("at_cap"))
        if wid in (res.get("added") or []):
            already.add(wid)
            if m.get("fulltext_status") in ("requested", "ingested"):
                set_fulltext(eid, wid, str(m.get("fulltext_status")),
                             m.get("fulltext_ref"), registry=registry)

    if not member_count:
        member_count = len(already)
    note = ""
    if rejected:
        note = (f"有 {len(rejected)} 条没进去" +
                ("（库已到 500 成员上限）。" if at_cap else "（id 非法或写入失败）。"))
    return {"ok": True, "src_library_id": src_id, "library_id": dst_id,
            "to_experiment_id": eid, "copied": copied, "skipped": skipped,
            "rejected": rejected, "member_count": member_count, "at_cap": at_cap,
            "note": note, "error": ""}


def _members_of_any(library_id: str,
                    registry: "_lib.LibraryRegistry | None" = None
                    ) -> list[dict[str, Any]] | None:
    """任意库的当前成员集。库不存在返回 ``None``（区别于「存在但空」）。

    ``get_library`` 已经替实验库折叠过文件夹了，所以这里不必再分叉。
    """
    try:
        rec = _lib._reg(registry).get_library(library_id)
    except Exception:  # noqa: BLE001 — LibraryError / 任何读失败
        return None
    return [dict(m) for m in (rec.get("members") or [])]


# ── 有效库解析（拉模型，每次现读现算）────────────────────────────────
def active_experiment_id() -> str:
    """当前活跃实验 id。拿不到返回 ``""``，**不抛**。"""
    try:
        from mast.documents.paths import current_scope
        eid, _sid = current_scope()
        return str(eid or "")
    except Exception:  # noqa: BLE001
        return ""


#: 旧名，模块内部沿用。
_active_experiment_id = active_experiment_id


def resolve_effective_library(
    *, experiment_id: str | None = None, ensure: bool = False,
    registry: "_lib.LibraryRegistry | None" = None,
) -> tuple[str, str]:
    """**有效库** = f(当前作用域)。返回 ``(library_id, source)``。

    三态::

        有活跃实验  → 该实验专属库                     source="experiment"
        无活跃实验  → registry.active_library_id       source="manual"
        指针失效    → reading 兜底                      source="fallback"

    **每次调用现读现解析** —— 没有订阅者、没有切换事件，所以 ``api/routes/scope.py``
    的 activate 端点零改动就自动跟随。这跟「读当前指针的地方全都每次现读」的既有
    形态一致（``documents.paths.current_scope`` / ``active_scope`` 单行表）。

    ``ensure`` **默认 False，这一点重要**：实验库 id 是从实验 id 确定性推导的
    （``exp_<id8>``），要知道它是哪个**根本不需要先建出来**。所以纯读路径（列库、
    ``lib_list`` 的提示文案、ingest 响应回显）解析完就走，不落任何盘。

    只有写路径（``add_members`` / ``set_fulltext`` / ``copy_library`` / 显式
    ensure 端点）才传 ``ensure=True``。当初把创建塞进解析里，代价是「列一下库」
    也会在实验文件夹里建目录、往 registry 塞一个空库 —— 一个只读操作留下了副作用。
    """
    eid = experiment_id if experiment_id is not None else active_experiment_id()
    eid = str(eid or "").strip()
    if eid:
        lib_id = (ensure_experiment_library(eid, registry=registry) if ensure
                  else experiment_library_id(eid))
        if lib_id:
            return lib_id, "experiment"
    try:
        reg = _lib._reg(registry)
        active = reg.active_library_id
        if active and active in {r["library_id"] for r in reg.list_libraries()}:
            return active, "manual"
    except Exception as exc:  # noqa: BLE001
        logger.warning("resolve_effective_library: registry 不可用 %r", exc)
    return _lib.GLOBAL_LIBRARY_ID, "fallback"


# ── 从文件夹重建 registry（供 reindex）────────────────────────────────
def rebuild_registry_from_folders(*, root: Path | None = None,
                                  registry: "_lib.LibraryRegistry | None" = None) -> int:
    """扫所有实验的 ``library/members.jsonl``，重建 registry 里的实验库条目。

    「只补不改」：**不动非实验库**（reading / custom 的权威本来就在 registry）。
    返回重建的库数量。这是「自包含可证明」对文献库子系统的兑现 —— 删掉
    registry.json 跑一遍 reindex，实验的书目应当完整复原。
    """
    from mast.core.experiment_paths import experiment_root
    from mast.logging.v2.manifest import read_json

    try:
        base = Path(root) if root is not None else experiment_root(create=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rebuild_registry_from_folders: 实验根解析失败 %r", exc)
        return 0
    if not base.is_dir():
        return 0

    reg = _lib._reg(registry)
    n = 0
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        path = d / LIBRARY_SUBDIR / MEMBERS_FILE
        if not path.is_file():
            continue
        # 身份优先认 experiment.json（实验自己的权威），退回 library.json 副本。
        eid = str((read_json(d / "experiment.json") or {}).get("id") or "")
        if not eid:
            eid = str((read_json(path.parent / LIBRARY_JSON) or {}).get("experiment_id") or "")
        if not eid:
            logger.warning("rebuild_registry_from_folders: %s 有 library/ 但 "
                           "experiment.json 与 library.json 都没有 experiment_id "
                           "—— 跳过（目录名只有 8 位，不猜完整 id）", d.name)
            continue
        lib_id = experiment_library_id(eid)
        events = _read_events(path)
        if events is None:
            continue
        try:
            reg.ensure_experiment_record(lib_id, experiment_id=eid,
                                         name=_default_name(eid))
            reg.replace_members_cache(lib_id, _fold(events))
            n += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("rebuild_registry_from_folders(%s): %r", lib_id, exc)
    if n:
        logger.info("experiment_library: 从文件夹重建了 %d 个实验库条目", n)
    return n
