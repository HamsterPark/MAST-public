"""文档 API 的传输模型 —— 五类文档的列表 / 正文 / 版本 / 差异 / 写入。

设计文档：``docs/v2/design/document_and_library_management.md`` §3.10

权威在文件夹，这里只是投影
--------------------------

每个字段都能在 ``<exp>/reports/<doc-dir>/doc.json`` 或 ``versions.jsonl`` 里指出
出处。**不新增任何服务端推断出来的状态** —— 尤其没有 ``status`` / ``finalized_at``
/ ``archived``：文档没有终态（INCREMENTAL-ONLY，与 ``experiment.json`` 同一条铁律），
永远可以再来一版。

为什么 ``kind`` 是 ``str`` 而不是 ``Literal``
--------------------------------------------

kind 的单一真源是 ``mast.documents.model.KINDS``。这里如果再写一份 ``Literal``，
就是本项目反复踩的「一处定义、多处白名单」（``MANAGED_SUBDIRS`` 与
``_scaffold_experiment`` 两个独立列表；``SettingsStore.KNOWN_KEYS`` 少一行就静默
no-op）—— 而且更糟：``Literal`` 漏了一个 kind 时，pydantic 会在**响应**校验时
炸掉，一份磁盘上完好的文档因此在 UI 里彻底消失。

改法是把枚举**当数据发下去**：列表响应带 ``kinds: [{value,label}]``，前端渲染
徽章用服务端给的 ``kind_label``，不自己维护对照表。加一个 kind 只需改
``model.py`` 一处。``tests/v2/unit/api/test_documents.py`` 断言这份清单与
``KINDS`` 逐字相同，钉住这个契约。
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class KindOption(BaseModel):
    """一个 kind 的机器值 + 中文标签。**前端的徽章文案来源**（见模块 docstring）。"""

    value: str = Field(..., description="机器值，如 experiment_report")
    label: str = Field("", description="中文标签，如 实验报告")


class VersionInfo(BaseModel):
    """``versions.jsonl`` 的一行 —— 一个版本的不可变事实。

    ``sha256`` 是文档自己的 manifest：文档不进 ``raw/_manifest.jsonl``（那是
    Nanonis 原始测量文件的账本），完整性靠这一列自证（设计 §5.1）。
    """

    version: int = Field(..., description="版本号（1 起）。**版本序恒用这个整数，绝不用 mtime**")
    words: int = 0
    sha256: str = ""
    created_at: str = Field("", description="ISO8601 UTC")
    created_by: str = Field("", description="agent:<id> | operator | unknown")
    note: str = ""
    conversation_id: Optional[str] = None
    run_id: Optional[str] = None


class DocumentSummary(BaseModel):
    """列表里的一行 = 一个文档（不是一个版本）。"""

    doc_id: str = Field(..., description="ULID，稳定身份。改标题不改它")
    kind: str = Field("experiment_report", description="见响应里的 kinds 清单")
    kind_label: str = Field("", description="kind 的中文标签，前端直接显示")
    title: str = ""
    version: int = Field(0, description="最新版本号（= latest_version）")
    versions_count: int = Field(0, description="版本总数")
    experiment_id: Optional[str] = Field(None, description="主归属实验（决定物理落点）")
    experiment_name: str = Field("", description="主归属实验名，来自 experiments 表")
    sample_id: Optional[str] = None
    relation: str = Field(
        "", description="相对查询实验的关系：primary | related | other | unfiled")
    root_kind: str = Field(
        "experiment", description="experiment = 在某个实验文件夹里；unfiled = 无主，待认领")
    verdict: Optional[str] = Field(
        None, description="ACCEPT | REVISE | REJECT —— 仅 review，从正文首行 <!-- verdict: X --> 解析")
    words: int = 0
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""
    conversation_id: Optional[str] = Field(
        None, description="生成它的那次对话（保存那一刻冻结，不跟随后续改挂）")
    run_id: Optional[str] = None
    target_doc_id: Optional[str] = Field(
        None, description="review 评的是哪个文档 —— 取代旧的按 draft_name 前缀猜")
    target_version: Optional[int] = None
    path: str = Field("", description="最新版本文件的绝对路径（用户可直接复制）")
    dir_path: str = Field("", description="文档目录的绝对路径")
    #: 兼容旧 DocumentsPane 的 epoch 秒。新代码请用 ``updated_at``（ISO8601）。
    modified_at: Optional[float] = Field(
        None, description="[deprecated] updated_at 的 epoch 秒形式")


class DocumentsResponse(BaseModel):
    """文档列表。空 = 正常（还没有任何文档），**不是 degraded**。

    ``degraded=True`` 只表示文档层本身不可用（实验根目录读不到之类）。
    """

    documents: List[DocumentSummary] = Field(default_factory=list)
    total: int = Field(0, description="本次返回的行数（已应用 limit）")
    count: int = Field(0, description="[deprecated] total 的旧名")
    kinds: List[KindOption] = Field(
        default_factory=list, description="全部 kind + 标签，前端据此渲染筛选与徽章")
    documents_root: str = Field(
        "", description="当前作用域下文档的落点提示（有实验 → <exp>/reports/；无实验 → _unfiled/documents/）")
    drafts_dir: str = Field("", description="[deprecated] 旧全局 data/drafts；现值同 documents_root")
    reviews_dir: str = Field("", description="[deprecated] 旧全局 data/reviews；现值同 documents_root")
    degraded: bool = False
    detail: str = ""


class DocumentDetail(DocumentSummary):
    """一个文档的正文 + 全部版本。

    ``content`` 是请求版本（缺省最新）的 markdown 正文；``body`` 是它的旧名，
    保留是为了让旧 DocumentsPane 在前端切换前继续工作。
    """

    content: str = Field("", description="markdown 正文")
    body: str = Field("", description="[deprecated] content 的旧名")
    version_requested: Optional[int] = Field(
        None, description="?version= 请求的版本；None = 最新")
    versions: List[VersionInfo] = Field(default_factory=list)
    truncated: bool = Field(False, description="正文超长被截断（病态大文件的护栏）")
    detail: str = ""
    degraded: bool = False


class DocumentVersionsResponse(BaseModel):
    """版本表。**集合 = 目录扫描 ∪ versions.jsonl**（读侧自愈，设计陷阱 ⑬）。"""

    doc_id: str = ""
    versions: List[VersionInfo] = Field(default_factory=list)
    latest_version: int = 0
    total: int = 0
    detail: str = ""
    degraded: bool = False


class DocumentDiffResponse(BaseModel):
    """**任意两版**的 unified diff。

    旧 artifacts_edit 只能比相邻两版（``versions[-2]`` vs ``versions[-1]``），
    「v1 到 v7 一共改了什么」问不出来。
    """

    doc_id: str = ""
    from_version: int = 0
    to_version: int = 0
    from_words: int = 0
    to_words: int = 0
    diff: str = Field("", description="unified diff 文本；无差异为空串")
    changed: bool = False
    versions_available: List[int] = Field(
        default_factory=list, description="该文档现有的全部版本号")
    detail: str = ""
    degraded: bool = False


class DocumentWriteRequest(BaseModel):
    """用户编辑 —— **存为新版本，永不覆盖**。

    智能体自己的版本和用户的修改都留在盘上；``load_draft("current")`` 读到最新
    那一版，所以这次修改**真的会被下一个打开该文档的智能体看到**（这正是旧
    ``artifact_edits`` 内存字典做不到的事）。
    """

    content: str = Field("", description="完整 markdown 正文")
    body: str = Field("", description="[deprecated] content 的旧名；两者取非空的那个")
    note: str = Field("", description="这一版的备注，写进 versions.jsonl")
    base_version: Optional[int] = Field(
        None,
        description="乐观并发：填上你编辑时看到的版本号。与当前最新不符则 409，"
                    "不会用你的文本盖掉别人刚存的那一版")

    def text(self) -> str:
        """正文 —— ``content`` 优先，回退旧字段 ``body``。"""
        return self.content if self.content.strip() else self.body


class DocumentPatchRequest(BaseModel):
    """改可变头部。**不动目录名** —— 目录名创建时冻结（改目录等于改身份）。"""

    title: Optional[str] = Field(None, description="新标题；旧标题进 title_history 留痕")
    kind: Optional[str] = Field(None, description="更正 kind（目录名前缀不追改，doc.json 才是权威）")
    sample_id: Optional[str] = Field(None, description="空串 = 清空")
    related_experiment_ids: Optional[List[str]] = Field(
        None, description="关联实验（这些实验的详情页也能看到本文档）；整表替换")


class DocumentClaimRequest(BaseModel):
    """认领 / 改主归属 —— **物理搬家**（主归属决定落点）。"""

    experiment_id: str = Field(..., description="目标实验（v1 experiment id）")


class DocumentSaveResponse(BaseModel):
    """写操作（PUT / PATCH / claim）的结果。

    ``ok=False`` 时 ``detail`` 一定非空 —— 前端据此提示，而不是看着一个空对象猜。
    """

    ok: bool = False
    doc_id: str = ""
    version: int = Field(0, description="本次写出的版本号；PATCH / claim 不发版本，回最新版本号")
    latest_version: int = 0
    versions_count: int = 0
    path: str = ""
    kind: str = ""
    kind_label: str = ""
    title: str = ""
    experiment_id: Optional[str] = None
    root_kind: str = "experiment"
    words: int = 0
    created_new: bool = Field(False, description="新建了一个文档（而不是给已有文档加一版）")
    doc_id_unknown: bool = Field(
        False, description="请求的 doc_id 找不到，内容照存但另立了新文档（绝不误合并）")
    conflict: bool = Field(False, description="base_version 与当前最新不符（409）")
    detail: str = ""
    degraded: bool = False


class DocumentExportResponse(BaseModel):
    """``/export`` 的降级/JSON 分支。成功路径直接流回文件内容。"""

    ok: bool = False
    doc_id: str = ""
    format: str = ""
    filename: str = ""
    version: int = 0
    export_path: str = Field("", description="HTML 导出的留档路径（时间戳共存，不覆盖上一份）")
    images_inlined: int = 0
    images_missing: int = 0
    content: str = ""
    detail: str = ""
    degraded: bool = False


__all__ = [
    "KindOption",
    "VersionInfo",
    "DocumentSummary",
    "DocumentsResponse",
    "DocumentDetail",
    "DocumentVersionsResponse",
    "DocumentDiffResponse",
    "DocumentWriteRequest",
    "DocumentPatchRequest",
    "DocumentClaimRequest",
    "DocumentSaveResponse",
    "DocumentExportResponse",
]
