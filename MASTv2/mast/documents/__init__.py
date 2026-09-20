"""实验文档子系统 —— 文献报告 / 实验计划 / 实验报告 / 论文草稿 / 评审报告。

设计文档：``docs/v2/design/document_and_library_management.md``

一句话：**一个文档 = 实验文件夹里的一个目录**（``vNNN.md`` 永不覆盖 +
``doc.json`` 可变头部 + ``versions.jsonl`` 版本权威）。文档必须归属一个主实验
（决定物理落点），可额外声明关联其他实验。一个实验对应 0..N 篇论文、0..N 份
报告、0..N 个计划 —— 一对多是常态。

对外只需要这几个名字::

    from mast.documents import store, KINDS, KIND_LABELS
    res = store().save(text=md, kind="experiment_report", title="Au111 形貌")
    entry = store().resolve_ref("current", kinds=("experiment_report",))
"""

from mast.documents.model import (
    KIND_CODES,
    KIND_HOME,
    KIND_LABELS,
    KINDS,
    DocMeta,
    SaveResult,
    VersionMeta,
    normalize_kind,
)
from mast.documents.store import DocEntry, DocumentStore, reset_caches, store

__all__ = [
    "KINDS",
    "KIND_CODES",
    "KIND_LABELS",
    "KIND_HOME",
    "DocMeta",
    "VersionMeta",
    "SaveResult",
    "DocEntry",
    "DocumentStore",
    "normalize_kind",
    "store",
    "reset_caches",
]
