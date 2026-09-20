"""产物的**载荷类型** —— 与执行引擎无关的那一半。

为什么它从 ``agents/state.py`` 搬了出来（2026-08-27）
-----------------------------------------------------
``state.py`` 有两种住户：

* 7 个 reducer + ``MASTState`` / ``AgentSubState`` 两张 ``Annotated`` 通道表 ——
  **纯 langgraph 机械**，退出 langgraph 时随图一起走；
* ``DocRef`` / ``CampaignRef`` / ``ScanResult`` / ``AnalysisResult`` 四个 pydantic
  模型 —— **引擎无关**，它们描述的是「一份文档指针长什么样」，与谁在驱动 agent
  没有任何关系。

它们同住一个文件，只是因为通道表要拿它们当注解类型。而这件事的代价，是在隔离副本上
真删一次才看见的：删除清单把 ``state.py`` 整个划掉，于是 ``artifact_channel.py``
—— 一个 663 行、绝大部分与 langgraph 无关的模块 —— 当场 ``ImportError``。

**一个文件里住着两种职责，静态分析看不出来**（它只答得出「谁 import 了谁」，答不出
「删掉之后谁起不来」）。所以这次抽取的目的不是整洁，是让删除那一步**能够只删该删的
东西** —— 否则删 langgraph 会顺手带走一批与 langgraph 毫无关系的类型定义。

``state.py`` 仍然把这四个名字**再导出**一遍，所以既有的 import 一个都不用改。等
``state.py`` 真的被删掉时，那条再导出跟着走，而这里不受影响。

指针，不是正文
--------------
四个模型共享同一条纪律：**带指针，不带正文**。文档正文在实验文件夹里
（``mast.documents``，``vNNN.md``，永不覆写）；扫描数据在磁盘上；campaign 行在 v2
records 库里。state 里只放 id + 一段**有界**的摘要。

界是**校验器**保证的，不是约定 —— 「保持简短」作为约定在本仓的历史成功率是 0。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: 任何随 state 走的自由文本摘要的硬上限。取值让一次满 fan-out（4 条分支 × 每个
#: 产物字段）仍然远低于每次 checkpoint 写入一千 token。由校验器强制，不靠约定。
SUMMARY_MAX_CHARS = 1200

#: 列表型产物载荷的上限（图片路径、异常行）。
LIST_MAX_ITEMS = 40


class DocRef(BaseModel):
    """A POINTER to a versioned document in the experiment folder.

    Never carries the document body — only what a downstream agent needs to
    decide whether to read it, plus enough identity to actually read it back
    (``load_document(doc_id)`` / the tools each agent already has).

    Why a pointer and not the text (2026-07-29): the body lives in
    ``mast.documents`` as ``vNNN.md``, never overwritten, and survives a power
    cut; a body here would be copied on every hop for no gain. ``summary`` is
    bounded by a validator because "keep it short" as a convention has a 100%
    historical failure rate.

    ``doc_id`` is the field that matters most. Before this existed, a document's
    identity survived only as a sentence in the conversation ("下次修订把 doc_id
    传回来") — i.e. on the one channel that compaction erases and a handoff
    evaporates. An identifier that needs a whole prompt section teaching the
    model to carry it by hand IS a missing state field.
    """
    model_config = ConfigDict(frozen=True)
    doc_id: str
    version: int = 0
    kind: str = ""                 # literature_report | experiment_report | paper_draft | review | plan
    title: str = ""
    path: str = ""                 # PATH ONLY — the body lives on disk
    summary: str = ""              # bounded; what the next agent needs to decide
    produced_by: str = ""          # agent id, for provenance in the rendered block

    @field_validator("summary")
    @classmethod
    def _bound_summary(cls, v: str) -> str:
        v = (v or "").strip()
        return v if len(v) <= SUMMARY_MAX_CHARS else v[:SUMMARY_MAX_CHARS] + "…（已截断）"


class CampaignRef(BaseModel):
    """A POINTER to a research campaign — the Campaign layer's "为什么做".

    The row itself lives in the v2 records DB (``logging/v2`` ``campaigns``
    table), which is where it must live: a campaign outlives every run, every
    thread and every checkpoint, and week-to-month reasoning cannot be kept in a
    channel that a compaction pass is allowed to summarise away.

    So this is the same pointer discipline as :class:`DocRef`, one table over:
    ``campaign_id`` is the identity, the rest is a BOUNDED extract so the next
    agent can decide whether to read the full row (``campaign_get``) without
    paying for it on every hop.

    ``plan_request`` is the field that makes this channel worth having. The
    Campaign layer's OUTPUT is a commission — "design me an experiment that would
    discriminate hypothesis A from B" — and before this channel existed the only
    place to put one was the handoff ``reason``, i.e. the one channel this repo
    has already learned not to trust (see ``_shared/artifact_channel`` and
    ``literature/prompts.py``'s 「交接语不是通道」).

    NOT carried: goal_json in full, the experiment list, the claim graph. Those
    are reads against the DB, and a copy here could only go stale.
    """
    model_config = ConfigDict(frozen=True)
    campaign_id: str
    title: str = ""
    hypothesis: str = ""
    #: exploratory | confirmatory | calibration | methodology (the DDL's CHECK set)
    hypothesis_kind: str = ""
    #: draft | running | paused | completed | aborted
    status: str = ""
    #: What the Campaign layer is COMMISSIONING from experiment_design. Bounded.
    plan_request: str = ""
    #: Set when this campaign is a follow-up refining an earlier one (谱系).
    parent_campaign_id: str = ""
    #: 一行人读的**终止判据**摘要（2026-08-27）——「什么算答完了」的机器可判那半。
    #:
    #: 带**定义**不带**结论**：一次求值的结果只会过期（下游看到的是几跳之前的
    #: 状态），而定义在这条纲领的一生里基本不变。同一个理由，完整的 ``goal_json``
    #: 也不进 state（见类 docstring 的 NOT carried 一节）。
    done_when_brief: str = ""

    @field_validator("hypothesis", "plan_request", "done_when_brief")
    @classmethod
    def _bound_text(cls, v: str) -> str:
        v = (v or "").strip()
        return v if len(v) <= SUMMARY_MAX_CHARS else v[:SUMMARY_MAX_CHARS] + "…（已截断）"


class ScanResult(BaseModel):
    """Outcome of a scan job — path only, never the raw image buffer itself."""
    model_config = ConfigDict(frozen=True)
    handle: str                                   # long-running job handle
    status: Literal["queued", "running", "done", "failed"]
    sxm_path: str | None = None                   # PATH ONLY — raw data lives on disk
    duration_s: float | None = None
    warnings: list[str] = Field(default_factory=list)
    frame_idx: int = 0                            # for streaming scan frames


class AnalysisResult(BaseModel):
    """Data-Processing agent output — metrics + figure PATHS, never pixels.

    Already pointer-shaped when it was written (2026-04), which is why it
    survived the 2026-07-29 field audit unchanged apart from the list bounds.
    """
    model_config = ConfigDict(frozen=True)
    sample_id: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    figures: list[str] = Field(default_factory=list)   # PATHS ONLY
    anomalies: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("figures", "anomalies")
    @classmethod
    def _bound_list(cls, v: list[str]) -> list[str]:
        return list(v or [])[:LIST_MAX_ITEMS]

    @field_validator("summary")
    @classmethod
    def _bound_summary(cls, v: str) -> str:
        v = (v or "").strip()
        return v if len(v) <= SUMMARY_MAX_CHARS else v[:SUMMARY_MAX_CHARS] + "…（已截断）"


# ── Models deleted 2026-07-29 (never instantiated, zero importers) ───────────
# ScanRequest / TipAssessment / DraftPaper / ReviewReport / ExperimentPlan.
# They described artifacts that in reality travel differently:
#   * a scan REQUEST is skill parameters, not state;
#   * tip state is authoritative in the vision BufferService (read via
#     read_latest_tip_status), and a stale copy in state could only disagree
#     with it;
#   * draft / review / plan are versioned DOCUMENTS on disk → they now travel
#     as DocRef, so state never holds a second copy of the prose that can drift
#     from the file.
# The state-local ``ExperimentPlan`` was also a NAME COLLISION with the real,
# live ``mast.planning.plan_store.ExperimentPlan`` — two classes, one name, and
# only one of them ever ran. Removing the dead twin removes the ambiguity.


__all__ = [
    "SUMMARY_MAX_CHARS",
    "LIST_MAX_ITEMS",
    "DocRef",
    "CampaignRef",
    "ScanResult",
    "AnalysisResult",
]
