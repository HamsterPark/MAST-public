"""文档在磁盘上的落点解析。

设计文档：``docs/v2/design/document_and_library_management.md`` §2 / §3.6

与 ``core/experiment_paths.py`` 同一条纪律：**每个函数懒解析**，绝不在 import 时
把路径冻结成模块常量（冻结过的教训是 打包版所有数据路径指向
不存在的目录）。

``_unfiled`` 的存在理由（§3.6）
--------------------------------

没有活跃实验时保存文档，三个先例给出的答案不一样：``chat/export`` 无归属就跳过
（不丢东西，因为 DB 里还有一份持久转录），``sample_gate`` 歧义朝放行解（记账损失
不是危险），``_quarantine`` 绝不丢字节、宁可事后认领。

文档属于第三种：拒存的话，LLM 已经生成的整篇内容就只活在对话流里了 —— 这是
**内容损失**，不是记账损失。所以落 ``<root>/_unfiled/documents/``，工具明说
「未归属」，之后可以 claim 认领。``_`` 前缀让 reindex 的 ``startswith("_")``
天然跳过它，不会被误认成一个实验。
"""

from __future__ import annotations

import logging
from pathlib import Path

from mast.core import experiment_paths as ep
from mast.documents.model import ASSETS_DIR, KIND_HOME, normalize_kind

logger = logging.getLogger(__name__)

#: ``<experiment_root>/_unfiled/`` 下的文档区目录名。
UNFILED_DIR = "_unfiled"
UNFILED_DOCS = "documents"

#: 废弃区。**位置即状态** —— 没有 ``discarded`` 字段，因为那会是一个终态字段
#: （INCREMENTAL-ONLY 禁止）。文档住在哪个区就说明了它是什么，和 ``_unfiled``
#: 完全同构：``claim()`` 把它从任一个区搬回某个实验，字节从头到尾没动过。
#:
#: 为什么需要它：``save()`` 在 doc_id 找不到时刻意**另立新文档**而不是报错（内容
#: 损失不可接受）。允许增殖的前提是事后能清理 —— 没有清理路径的话，一个忘传
#: doc_id 的 agent 就能在用户的报告列表里永久留下一串重复。而硬删除同样不行：
#: 「绝不丢字节，宁可事后认领」是 ``_quarantine`` 定下的规矩。
DISCARDED_DIR = "_discarded"


def storage(existing=None):
    """拿一个 ``ExperimentStorage``。

    调用方（runtime / API）手上已经有实例时应当传进来；裸 ``@tool`` 函数没有 ctx
    可用，就地新建一个 —— 同一个 db 文件上多个实例是安全的（WAL + 每次调用
    一条连接），``query_experiment_records`` 早就是这么干的。

    **不缓存实例**：缓存过的实例只在构造时跑 ``_ensure_tables``，测试删库重建
    后就会对着一个没有表的文件发 SQL。构造成本（几条 CREATE IF NOT EXISTS）
    对文档写入这种低频动作完全可以接受。

    库路径**故意**从 ``agents._shared.data_paths.experiment_db_path`` 取（懒 import，
    不成环）：那是 ``MAST_EXPERIMENT_DB`` env > ``<project_root>/experiments/
    mast_experiments.db`` 的唯一解析器，runtime 和 records 都走它。在这里再写一份
    "env > 默认值" 的逻辑会立刻变成第二个真源，而路径分家这类 bug（）
    正是这么来的。``core/runtime.py`` 里也有大量 ``mast.agents._shared`` 的懒 import，
    这个方向在本仓库是既有形态，不是新开的口子。
    """
    if existing is not None:
        return existing
    from mast.agents._shared.data_paths import experiment_db_path
    from mast.logging.storage import ExperimentStorage
    return ExperimentStorage(experiment_db_path())


def current_scope(st=None) -> tuple[str | None, str | None]:
    """当前 ``(experiment_id, sample_id)``。

    先读进程内的 live ``ExperimentLog``（它是内存真源，切换时同步落 DB），拿不到
    再读 ``active_scope`` 单行表 —— 后者让**没有 runtime 的进程**（纯 API 测试、
    离线脚本）也能解析出作用域。两条路都失败返回 ``(None, None)``，**不抛**。
    """
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        if log is not None and log.current_experiment_id:
            return log.current_experiment_id, log.current_sample_id
    except Exception:  # noqa: BLE001 — 作用域解析永远不该让保存失败
        pass
    try:
        row = storage(st).get_active_scope() or {}
        return (row.get("experiment_id") or None), (row.get("sample_id") or None)
    except Exception:  # noqa: BLE001
        return None, None


def exp_dir_for(experiment_id: str, *, create: bool = True, st=None) -> Path | None:
    """实验 id → 实验文件夹。实验行不存在返回 ``None``。

    ``dir_name`` 缺失时**现算并回写 DB**（与 ``runtime._ensure_scope_dirs`` 同一
    手法）：目录名创建时冻结，冻结的那一刻就是这里 —— 之后改实验名只改 name 列，
    目录不动，否则 ``scan_files`` 里所有历史行（append-only，``current_path``
    永远无法 UPDATE）都会指向不存在的路径。
    """
    eid = str(experiment_id or "").strip()
    if not eid:
        return None
    st = storage(st)
    try:
        exp = st.get_experiment(eid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("exp_dir_for(%s): storage read failed: %r", eid, exc)
        return None
    if not exp:
        return None
    dir_name = (exp.get("dir_name") or "").strip()
    if not dir_name:
        dir_name = ep.experiment_dir_name(
            eid, exp.get("name") or "", str(exp.get("start_time") or ""))
        try:
            st.set_experiment_dir_name(eid, dir_name)
        except Exception:  # noqa: BLE001 — 回写失败下次再算，不影响本次落盘
            pass
    return ep.experiment_dir(dir_name, create=create)


def unfiled_docs_dir(*, create: bool = False) -> Path:
    """``<experiment_root>/_unfiled/documents/``。"""
    d = ep.experiment_root(create=create) / UNFILED_DIR / UNFILED_DOCS
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def discarded_docs_dir(*, create: bool = False) -> Path:
    """``<experiment_root>/_discarded/documents/`` —— **无归属**文档的废弃区。

    有归属的文档废弃到**它自己实验文件夹里**（:func:`discarded_home_for`），不是这里。
    """
    d = ep.experiment_root(create=create) / DISCARDED_DIR / UNFILED_DOCS
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def discarded_home_for(experiment_id: str | None, *, create: bool = True,
                       st=None) -> Path:
    """废弃一份文档时它该搬到哪。

    有归属 → ``<exp_dir>/reports/_discarded/``；无归属 → ``<root>/_discarded/documents/``。

    **为什么有归属的要留在实验文件夹里**：``discard`` 刻意保留 ``experiment_id``（区由
    路径表达，不必清归属），所以那份文档逻辑上仍然属于这个实验。要是把它搬到实验根
    一级，「一个实验的所有数据都在一个文件夹里」就破了 —— 把实验文件夹拷到另一台机器，
    被你设为废弃的那几份会**留在原地**，而它们并没有被删除。这个不一致是靠可搬移性
    集成测试抓出来的（拷文件夹后 reindex 只重建出 2 份而磁盘上有 3 份）。
    """
    if experiment_id:
        exp_dir = exp_dir_for(experiment_id, create=create, st=st)
        if exp_dir is not None:
            d = exp_dir / "reports" / DISCARDED_DIR
            if create:
                d.mkdir(parents=True, exist_ok=True)
            return d
    return discarded_docs_dir(create=create)


def doc_home(kind: str, experiment_id: str | None, *, create: bool = True,
             st=None) -> tuple[Path, str]:
    """文档目录的**父目录** + ``root_kind``。

    有归属 → ``<exp>/reports/`` 或 ``<exp>/plans/``（按 kind，见
    ``model.KIND_HOME``，两者都已在 ``MANAGED_SUBDIRS`` 里所以 watcher 无条件
    忽略）；无归属或实验行已消失 → ``_unfiled/documents/``（**扁平**，认领时
    才按 kind 分流到 reports/plans）。
    """
    if experiment_id:
        exp_dir = exp_dir_for(experiment_id, create=create, st=st)
        if exp_dir is not None:
            home = exp_dir / KIND_HOME.get(normalize_kind(kind), "reports")
            if create:
                home.mkdir(parents=True, exist_ok=True)
            return home, "experiment"
    return unfiled_docs_dir(create=create), "unfiled"


def assets_dir_for(home: Path, root_kind: str, *, create: bool = True) -> Path:
    """图池目录。

    实验内是 ``<exp>/reports/_assets/`` —— **一个实验共用一个池**，不是每个文档
    一个：同一张扫描图常被报告和论文同时引用，复制两份既浪费又让「改了图哪些
    文档受影响」无法回答。文档里的链接写成 ``../_assets/x.png``，整个实验文件夹
    搬到别的机器仍然有效（旧实现写 ``../figures/`` 指向全局 data/figures，一搬
    就全断）。

    计划目录（``plans/``）下的文档也用 ``reports/_assets/``：图池按实验分，不按
    子目录分。
    """
    if root_kind == "unfiled":
        d = unfiled_docs_dir(create=create) / ASSETS_DIR
    else:
        # home 可能是 <exp>/plans —— 图池统一挂在 reports/ 下。
        d = home.parent / "reports" / ASSETS_DIR if home.name == "plans" else home / ASSETS_DIR
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def assets_rel_prefix(kind: str) -> str:
    """从文档目录看图池的相对前缀。

    文档住在 ``<home>/<doc-dir>/vNNN.md``，图池在 ``<exp>/reports/_assets/``：

    * ``reports/`` 下的文档 → ``../_assets/``
    * ``plans/`` 下的文档   → ``../../reports/_assets/``

    markdown 渲染器按**文件自己的目录**解析相对路径，所以这个前缀必须从版本
    文件所在目录算起，不是从实验目录算起。
    """
    return ("../../reports/" if KIND_HOME.get(normalize_kind(kind)) == "plans" else "../") + ASSETS_DIR + "/"


def search_roots(*, include_discarded: bool = False,
                 root: Path | None = None) -> list[tuple[Path, str]]:
    """扫描文档时要遍历的所有 ``(父目录, root_kind)``。

    这是 DB 索引未命中时的兜底路径 —— 「文件夹是记录，DB 是可重建索引」这句话
    只有在**真的能从文件夹读出来**时才成立。范围有界：每个实验两个子目录 +
    一个 ``_unfiled``（+ 显式要求时的 ``_discarded``）。

    ``_discarded`` **默认不扫**：废弃的文档不该出现在任何常规列表里。但它随时
    可扫、字节一直在，所以「废弃」是可逆的。
    """
    out: list[tuple[Path, str]] = []
    if root is None:
        try:
            root = ep.experiment_root(create=False)
        except Exception:  # noqa: BLE001
            return out
    root = Path(root)
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        for sub in ("reports", "plans"):
            d = child / sub
            if d.is_dir():
                out.append((d, "experiment"))
        if include_discarded:
            # 有归属的文档废弃在**实验文件夹内**，所以废弃区是每个实验一个。
            d = child / "reports" / DISCARDED_DIR
            if d.is_dir():
                out.append((d, "discarded"))
    unfiled = root / UNFILED_DIR / UNFILED_DOCS
    if unfiled.is_dir():
        out.append((unfiled, "unfiled"))
    if include_discarded:
        discarded = root / DISCARDED_DIR / UNFILED_DOCS
        if discarded.is_dir():
            out.append((discarded, "discarded"))
    return out
