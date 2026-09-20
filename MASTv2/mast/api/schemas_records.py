"""Pydantic request/response models for Domain D — experiment detail, records
drill-down, and operator feedback.

These shapes mirror the real handlers/backends faithfully:

* experiment detail / samples / actions / map_markers / feedback follow the v1
  ``ExperimentStorage`` row shapes (``mast.logging.storage``) as surfaced by
  ``gui.experiment_viewer`` — names/goal_text/status/start_time, sample cards,
  action timeline, spatial markers, and the operator-feedback table.
* the campaigns / actions records drill-down mirrors the richer
  ``ExperimentStoreV2`` payload shaped by ``gui.records_api`` (campaign stats,
  HLC display strings, action params/state_delta).

Per the house rules these are the SINGLE SOURCE OF TYPES for this slice. Every
endpoint has a ``response_model``; write endpoints return typed results that
degrade safely (``ok`` / ``degraded``) when the live core is absent.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── building blocks (mirror the storage row shapes) ────────────────────


class SampleSummary(BaseModel):
    """One sample card under an experiment (mast.logging.storage.samples row)."""

    id: str
    experiment_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    sample_type: Optional[str] = None
    sample_subtype: Optional[str] = None
    action_count: int = 0


class ActionSummary(BaseModel):
    """One action in an experiment / sample timeline.

    Flattened from the v1 ``ActionRecord`` (mast.core.types) as the experiment
    viewer renders it: skill call + success + duration + optional error."""

    id: str
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    timestamp: Optional[str] = None
    skill_name: Optional[str] = None
    skill_version: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    success: Optional[bool] = None
    error: Optional[str] = None
    duration_s: float = 0.0
    context: Optional[str] = None
    approval_source: Optional[str] = None
    # What the skill actually RETURNED and produced. These columns were written
    # as empty for the whole 2026-07-27 session ( — the result
    # dict was discarded, so every action read `"data": {}`, indistinguishable
    # from a skill that returned nothing). They are populated now, but this
    # schema still flattened them away, so the record was fixed in the database
    # and unchanged in the UI — the half of the fix an operator can actually
    # see. `data` is capped by the route: an action's payload can be a whole
    # spectrum, and a timeline of those would be megabytes.
    data: dict[str, Any] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    nanonis_calls: int = 0
    data_truncated: bool = False


class MapMarker(BaseModel):
    """One spatial map marker (mast.logging.storage.map_markers row).

    Coordinates are Nanonis stage frame, METRES. The live scan-map IS the
    experiment record's map (此地图就是实验记录的地图)."""

    id: int
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    kind: str = "move"
    skill_name: str = ""
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    w_m: Optional[float] = None
    h_m: Optional[float] = None
    angle_deg: float = 0.0
    label: str = ""
    status: str = "done"
    source: str = "skill"
    timestamp: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class FeedbackEntry(BaseModel):
    """One operator-feedback row (mast.logging.storage.feedback row).

    kind ∈ rating|comment. A rating tap and a free comment are independent
    rows; both are recorded alongside the experiment record."""

    id: int
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    conversation_id: Optional[str] = None
    kind: str = "rating"
    rating: str = ""
    comment: str = ""
    agent: str = ""
    timestamp: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)
    #: 处理时刻（ISO）。``None`` = 未处理。
    #:
    #: 反馈表原来既没有可见的时间也没有处理标记，于是每一批新反馈都要先做一次
    #: 考古（翻 git log + KNOWN_ISSUES + 实机清单）才知道哪些早就修过 ——
    #: 而有一批在两个方向上都判错过：一条修好了又被重报，一条被当成修好了
    #: 其实只修了一半。
    resolved_at: Optional[str] = None
    #: 在哪个版本处理的，例如 ``v6.2.1``。
    resolved_version: str = ""
    resolved_note: str = ""


# ── GET /api/experiments/{id} ──────────────────────────────────────────


class TipInService(BaseModel):
    """一次实验期间在役的针尖 —— **展示层聚合，不是第二个真源**。

    「针尖记录等是不是也应该在实验记录中?」

    针尖是仪器域的东西（与实验 / 样品**并列**，不是它们的下级 —— 一根针会跨很多次
    实验，一次实验也可能换好几根），所以 ``tips`` 表原地不动。这里只是把「这段时间
    里装的是哪根针」这个本来就能从两张表答出来的问题答出来，并给出跳回针尖卡片的
    id。字段是 ``tips`` 行的**子集**，刻意不全搬 —— 全搬就是在这里造一份会和真源
    漂开的副本。

    ``overlap_exact=False`` 的意思是「时间窗对上了，但对上的那一端是个下界」：
    ``installed_at`` 被人工回填成了纯日期（等价于当天 00:00），或者压根没填、
    退回用了 ``created_at``。两种情况都**偏早**，也就是宁可多列一根也不漏掉一根。
    把它露出来，是为了让「确实在役」和「大概在役」看得出区别。
    """

    id: str
    tip_index: Optional[int] = None
    name: str = ""
    material: str = ""
    fabrication: str = ""
    form: str = ""
    installed_at: Optional[str] = None
    removed_at: Optional[str] = None
    #: 这一行还在役吗（``removed_at IS NULL``）。前端不必自己判空。
    in_service_now: bool = False
    overlap_exact: bool = True


class ExperimentDetail(BaseModel):
    """Full experiment detail: header metadata + samples + actions +
    spatial map markers + operator feedback. ``degraded`` / ``found`` let the
    frontend show an empty-but-not-broken state when storage is unwired or the
    experiment id does not exist."""

    id: str
    name: Optional[str] = None
    goal_text: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    notes: Optional[str] = None
    samples: list[SampleSummary] = Field(default_factory=list)
    actions: list[ActionSummary] = Field(default_factory=list)
    map_markers: list[MapMarker] = Field(default_factory=list)
    feedback: list[FeedbackEntry] = Field(default_factory=list)
    #: 本实验期间在役的针尖。装入时刻早的在前。
    tips: list[TipInService] = Field(default_factory=list)
    found: bool = False
    degraded: bool = False


# ── POST /api/experiments (create) ─────────────────────────────────────


class CreateExperimentRequest(BaseModel):
    """Start a new experiment session. ``thread_id`` becomes the experiment id
    when the live ExperimentLog assigns one (id=thread_id contract)."""

    name: str
    goal: str = ""
    thread_id: Optional[str] = None


class CreateExperimentResult(BaseModel):
    """Result of starting an experiment. ``id`` equals the thread_id of the
    created session when wired (id=thread_id)."""

    ok: bool = False
    id: Optional[str] = None
    degraded: bool = False


# ── POST /api/experiments/{id}/end ─────────────────────────────────────


class EndExperimentRequest(BaseModel):
    """End the current experiment session."""

    status: str = "completed"


class EndExperimentResult(BaseModel):
    ok: bool = False
    id: Optional[str] = None
    status: Optional[str] = None
    degraded: bool = False


# ── POST /api/experiments/{id}/samples ─────────────────────────────────


class CreateSampleRequest(BaseModel):
    """Start a new sample under an experiment."""

    name: str
    description: str = ""
    sample_type: str = ""
    sample_subtype: str = ""


class CreateSampleResult(BaseModel):
    ok: bool = False
    id: Optional[str] = None
    experiment_id: Optional[str] = None
    degraded: bool = False


# ── GET /api/records/campaigns ─────────────────────────────────────────


class CampaignStats(BaseModel):
    """Per-campaign roll-up counts (records_api ``stats`` block)."""

    experiments: int = 0
    actions: int = 0
    observations: int = 0
    scans: int = 0


class GoalProgressView(BaseModel):
    """一条纲领的目标判据现状。

    **裸 ``dict`` 不行**：生成出来的 TS 类型是 ``Record<string, never>``，前端读
    ``.reason`` 就是 ``{}`` —— 字段名从此只活在两边各自的记忆里，改一个名字没有
    任何东西会红。这是「三份手抄的名单总有一份会漏」的类型版本。
    """

    #: done | not_done | unknown。**unknown 不是 not_done** —— 前者要去看为什么
    #: 读不到，后者接着做。
    verdict: str = "unknown"
    #: 人读的一句话。unknown 时说的是**读不到什么**。
    reason: str = ""
    satisfied: int = 0
    total: int = 0


class CampaignSummary(BaseModel):
    """One campaign row of the records drill-down (records_api campaign shape).

    HLC timestamps are surfaced as their UI display form
    ('2026-03-22T15:42:08Z-0009-main')."""

    id: str
    title: Optional[str] = None
    hypothesis: Optional[str] = None
    hypothesis_kind: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    created_by: str = ""
    parent_campaign_id: Optional[str] = None
    stats: CampaignStats = Field(default_factory=CampaignStats)
    #: 目标判据这一刻怎么看（2026-08-28）：
    #: ``{"verdict": done|not_done|unknown, "reason", "satisfied", "total"}``。
    #:
    #: **读即求值**，不是一个存下来的字段 —— 存下来的结论只会过期，而一条纲领的
    #: 时间尺度是周~月。``unknown`` 时 ``reason`` 说的是**判不了的原因**（没写
    #: done_when / 写坏了 / 证据读不到），三者驱动的下一步不同，不能折叠。
    goal_progress: GoalProgressView = Field(default_factory=GoalProgressView)


class CampaignsResponse(BaseModel):
    """Paginated campaigns list. ``page``/``page_size`` echo the request so the
    frontend can render pagination even on a degraded (empty) response."""

    campaigns: list[CampaignSummary] = Field(default_factory=list)
    count: int = 0
    page: int = 1
    page_size: int = 50
    degraded: bool = False


# ── GET /api/records/actions/{id} ──────────────────────────────────────


class ActionDetail(BaseModel):
    """Full action detail for the records drill-down (records_api action shape).

    Carries the richer v2 fields (HLC, params, state_delta) when available; for
    the v1 store it folds in the flattened ActionRecord fields."""

    id: str
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    parent_action_id: Optional[str] = None
    agent_id: Optional[str] = None
    action_type: Optional[str] = None
    skill_name: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    hlc: Optional[str] = None
    timestamp: Optional[str] = None
    status: Optional[str] = None
    duration_ms: Optional[int] = None
    duration_s: Optional[float] = None
    state_delta: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    found: bool = False
    degraded: bool = False


# ── POST /api/feedback ─────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    """Record one operator feedback row alongside the experiment record.

    A ``rating`` tap and a free ``comment`` are independent: pass one or the
    other. Ties to the active experiment / conversation when given."""

    rating: str = ""
    comment: str = ""
    experiment_id: Optional[str] = None
    sample_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class FeedbackResult(BaseModel):
    """Result of recording feedback. ``id`` is the new feedback row id when the
    live storage is wired."""

    ok: bool = False
    id: Optional[int] = None
    kind: Optional[str] = None
    degraded: bool = False


class FeedbackResolveRequest(BaseModel):
    """标 / 取消标一条反馈的处理状态。

    ``resolved=False`` 是刻意支持的：一个症状**又回来了**，与「当初标错了」
    是两回事，用户要能说前者而不抹掉「它曾经被处理过」这个记录。
    """

    resolved: bool = True
    version: str = ""
    note: str = ""


class FeedbackList(BaseModel):
    """Global operator-feedback list (across ALL experiments, newest first),
    including rows with no experiment_id — which the per-experiment drill-down
    (filtered by experiment_id) never surfaces."""

    items: list[FeedbackEntry] = Field(default_factory=list)
    degraded: bool = False


__all__ = [
    "SampleSummary",
    "ActionSummary",
    "MapMarker",
    "FeedbackEntry",
    "FeedbackList",
    "FeedbackResolveRequest",
    "ExperimentDetail",
    "CreateExperimentRequest",
    "CreateExperimentResult",
    "EndExperimentRequest",
    "EndExperimentResult",
    "CreateSampleRequest",
    "CreateSampleResult",
    "CampaignStats",
    "CampaignSummary",
    "CampaignsResponse",
    "ActionDetail",
    "FeedbackRequest",
    "FeedbackResult",
]
