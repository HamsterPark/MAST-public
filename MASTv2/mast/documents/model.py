"""文档的身份与形状 —— kind 枚举、目录命名、sidecar 数据类。

设计文档：``docs/v2/design/document_and_library_management.md`` §3.1

这里是 kind 的**单一真源**。pydantic 的 ``Literal``、前端的徽章表、
``agents/_shared/artifacts.py`` 的注册表都必须从这里派生或与它对齐 —— 本项目
反复踩「一处定义、多处白名单」的坑（``MANAGED_SUBDIRS`` 与 ``_scaffold_experiment``
是两个独立列表；``SettingsStore.KNOWN_KEYS`` 少一行就静默 no-op）。

身份规则（三条，缺一条就退化成现在的 bug）
--------------------------------------------

1. **doc_id 是身份，title 只是显示名。** 版本族由 doc_id 决定，绝不由 title 的
   slug 决定。此前 ``save_draft`` 拿 LLM 自由填的 title 做文件族：换个措辞就分叉
   成两条历史，两个无关实验撞名就误合并成一条。
2. **目录名创建时冻结。** 改 title 只改 ``doc.json`` + DB，目录不动 —— 与实验
   文件夹同一条规则（``core/experiment_paths.py`` 模块 docstring 第 2 条）。
    目录名里的 ``__<id8>`` 是给人识别的提示，不是反查键：ULID 的短前缀可能
    相同。按 id 找文档必须使用 ``doc.json`` 中的完整 ``doc_id``
   （``store.get()`` 就是这么做的），别按目录名前缀猜。
3. **kind 的权威在 ``doc.json``**，目录名前缀只是给人看的提示。kind 标错时只改
   sidecar，不追改目录（改目录就等于改身份）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from mast.core.experiment_paths import slug as _slug

SCHEMA_VERSION = "1.0.0"

#: 五类文档。顺序即 UI 里的展示顺序。
KINDS: tuple[str, ...] = (
    "literature_report",
    "experiment_plan",
    "experiment_report",
    "paper_draft",
    "review",
)

#: 目录名前缀（人读提示，非权威 —— 见模块 docstring 第 3 条）。
KIND_CODES: dict[str, str] = {
    "literature_report": "lit",
    "experiment_plan": "plan",
    "experiment_report": "rpt",
    "paper_draft": "draft",
    "review": "rev",
}

#: 中文标签，给 UI 徽章和工具返回文案用（前端不必自己再维护一份）。
KIND_LABELS: dict[str, str] = {
    "literature_report": "文献报告",
    "experiment_plan": "实验计划",
    "experiment_report": "实验报告",
    "paper_draft": "论文草稿",
    "review": "评审报告",
}

#: 文档落在实验文件夹的哪个子目录。
#:
#: 计划单独放 ``plans/`` 而不是挤进 ``reports/``：计划有一个被反复原子替换的
#: 活文件（``progress.md`` 进度视图），放进 reports/ 会破坏 README 对该目录
#: 「永不覆盖」的承诺。两个目录都已在 ``MANAGED_SUBDIRS`` 里。
KIND_HOME: dict[str, str] = {
    "literature_report": "reports",
    "experiment_plan": "plans",
    "experiment_report": "reports",
    "paper_draft": "reports",
    "review": "reports",
}

#: 目录名里 slug 的字符预算。加上 ``<码>__<10 位日期>__…__<id8>`` 的固定开销
#: （最多 4+2+10+2+2+8 = 28），再加 ``reports/`` 与实验目录名，仍在 MAX_PATH 内。
SLUG_MAX_DOC = 32

#: 版本文件名。三位零填充 —— 字母序等于版本序，资源管理器里一眼看出顺序。
VERSION_FILE_FMT = "v{v:03d}.md"

DOC_JSON = "doc.json"
VERSIONS_JSONL = "versions.jsonl"
PROGRESS_JSONL = "progress.jsonl"
PROGRESS_MD = "progress.md"

#: 实验内共享图池的目录名。``_`` 前缀让它不会被当成一个文档目录。
ASSETS_DIR = "_assets"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_kind(kind: str, fallback: str = "experiment_report") -> str:
    """把任意输入收敛到合法 kind。**永不抛** —— LLM 会传各种近似写法。"""
    k = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if k in KINDS:
        return k
    # 常见近似写法（模型的自由发挥）→ 正规名。
    alias = {
        "report": "experiment_report",
        "experiment": "experiment_report",
        "draft": "paper_draft",
        "paper": "paper_draft",
        "manuscript": "paper_draft",
        "plan": "experiment_plan",
        "literature": "literature_report",
        "lit_report": "literature_report",
        "literature_review": "literature_report",
        "review_report": "review",
        "peer_review": "review",
    }
    return alias.get(k, fallback if fallback in KINDS else "experiment_report")


def doc_dir_name(kind: str, title: str, doc_id: str, created_at: str = "",
                 *, max_chars: int = SLUG_MAX_DOC) -> str:
    """``<码>__<YYYY-MM-DD>__<slug>__<id8>``。创建时算一次，之后冻结。

    复用 ``experiment_paths.slug``：它已经处理了 Windows 保留设备名、非法字符、
    会被静默吞掉的尾部点/空格、以及 CJK 的字节预算 —— 这些坑不值得再踩一遍。
    """
    code = KIND_CODES.get(normalize_kind(kind), "doc")
    date = (created_at or now_iso())[:10] or "0000-00-00"
    return f"{code}__{date}__{_slug(title, 'untitled', max_chars=max_chars)}__{doc_id8(doc_id)}"


def doc_id8(doc_id: str) -> str:
    """doc_id 的前 8 位（小写，只保留字母数字）。"""
    raw = "".join(ch for ch in str(doc_id or "") if ch.isalnum())
    return (raw[:8] or "0" * 8).lower()


@dataclass
class VersionMeta:
    """``versions.jsonl`` 的一行 —— 版本事实，不可变。"""

    v: int
    file: str = ""
    sha256: str = ""
    words: int = 0
    created_at: str = ""
    created_by: str = ""
    conversation_id: str | None = None
    run_id: str | None = None
    note: str = ""

    def to_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False) + "\n"

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "VersionMeta | None":
        try:
            v = int(obj.get("v"))
        except (TypeError, ValueError):
            return None
        if v <= 0:
            return None
        return cls(
            v=v,
            file=str(obj.get("file") or VERSION_FILE_FMT.format(v=v)),
            sha256=str(obj.get("sha256") or ""),
            words=int(obj.get("words") or 0),
            created_at=str(obj.get("created_at") or ""),
            created_by=str(obj.get("created_by") or ""),
            conversation_id=obj.get("conversation_id") or None,
            run_id=obj.get("run_id") or None,
            note=str(obj.get("note") or ""),
        )


@dataclass
class DocMeta:
    """``doc.json`` —— 可变头部，原子替换。

    刻意**没有** ``status`` / ``finalized_at`` / ``archived``：文档和实验一样
    没有终态，永远可以再来一版（INCREMENTAL-ONLY）。有测试断言这几个键不存在。
    """

    doc_id: str
    kind: str
    title: str
    experiment_id: str | None = None
    sample_id: str | None = None
    related_experiment_ids: list[str] = field(default_factory=list)
    title_history: list[dict] = field(default_factory=list)
    created_at: str = ""
    created_by: str = ""
    conversation_id: str | None = None
    run_id: str | None = None
    target_doc_id: str | None = None
    target_version: int | None = None
    latest_version: int = 0
    updated_at: str = ""
    dir_name: str = ""
    #: ``_unfiled`` 里的文档为 ``"unfiled"``，实验文件夹里的为 ``"experiment"``。
    root_kind: str = "experiment"
    #: 旧 ``<kind>:<stem>`` id 的别名，供 artifacts_edit 的兼容层解析。
    legacy_stem: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "DocMeta | None":
        doc_id = str(obj.get("doc_id") or "").strip()
        if not doc_id:
            return None
        try:
            tv = obj.get("target_version")
            target_version = int(tv) if tv not in (None, "") else None
        except (TypeError, ValueError):
            target_version = None
        return cls(
            doc_id=doc_id,
            kind=normalize_kind(obj.get("kind") or ""),
            title=str(obj.get("title") or ""),
            experiment_id=obj.get("experiment_id") or None,
            sample_id=obj.get("sample_id") or None,
            related_experiment_ids=[str(x) for x in (obj.get("related_experiment_ids") or [])],
            title_history=list(obj.get("title_history") or []),
            created_at=str(obj.get("created_at") or ""),
            created_by=str(obj.get("created_by") or ""),
            conversation_id=obj.get("conversation_id") or None,
            run_id=obj.get("run_id") or None,
            target_doc_id=obj.get("target_doc_id") or None,
            target_version=target_version,
            latest_version=int(obj.get("latest_version") or 0),
            updated_at=str(obj.get("updated_at") or ""),
            dir_name=str(obj.get("dir_name") or ""),
            root_kind=str(obj.get("root_kind") or "experiment"),
            legacy_stem=obj.get("legacy_stem") or None,
        )


@dataclass
class SaveResult:
    """写入结果。``ok=False`` 时 ``error`` 一定非空 —— 调用方（agent 工具）据此
    组织给模型看的文案，而不是抛异常炸掉整个 turn。"""

    ok: bool
    doc_id: str = ""
    version: int = 0
    path: str = ""
    kind: str = ""
    title: str = ""
    experiment_id: str | None = None
    root_kind: str = "experiment"
    created_new: bool = False
    #: 请求的 doc_id 找不到时为 True —— 内容照存（不丢），但明说另立了新文档。
    doc_id_unknown: bool = False
    words: int = 0
    error: str = ""
