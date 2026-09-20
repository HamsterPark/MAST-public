"""技能市场 / 订阅列表 API 的请求与响应模型。

三条贯穿全文的纪律
==================

1. **「存下来了」和「生效了」是两件事。** agent 的工具表在建图时冻结，改订阅只是
   改了一个 holder —— 模型手上那张表要等一次图重建，而重建可能因任务占用被推迟、
   可能失败、也可能压根没接线。所以每个写响应都带 ``rebuild_note``（人话，UI 逐字
   显示）、``agent_path_pending``、``fingerprint_matches``。这与
   ``schemas_skill_overlay`` 是同一条，因为是同一个事故形态。

2. **``fingerprint_matches`` 是三态。** ``None`` = 判断不了（进程刚起、还没建过
   工具表），**不是** ``False``。

3. **写请求只有增量语义。** 没有「整体替换」字段，这是刻意的：设置页那次事故是
   前端从三张手维护的列表重建整个对象，于是不在列表里的键在用户编辑任何别的
   字段时被删掉。订阅列表有几百条，同样的形状在这里会把人的订阅一次抹平。导入
   走的是 ``/import``，它有自己的 dry-run 与报告。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# 诚实字段（每个写响应都带）
# ─────────────────────────────────────────────────────────────────────────────

class LiveEffectFields(BaseModel):
    """「现在到底生效了没有」—— 三个字段，一个都不能省。"""

    rebuild_note: str = Field(
        default="",
        description="人话说明工具表重建的结果。UI **逐字显示**，不要自己改写")
    agent_path_pending: bool | None = Field(
        default=None,
        description="agent 侧还没跟上（True = 现在还没生效）；None = 判断不了")
    fingerprint_matches: bool | None = Field(
        default=None,
        description="agent 手上那张表 == 注册表现在的样子；None = 还没建过表，"
                    "判断不了 —— **不是** False")


# ─────────────────────────────────────────────────────────────────────────────
# 目录
# ─────────────────────────────────────────────────────────────────────────────

class MarketEntry(BaseModel):
    """市场里的一行。``subscribed`` / ``mandatory`` 是这个功能新加的两列。"""

    name: str = ""
    zh: str = ""
    category: str = ""
    safety: str = ""
    level: int = 0
    source: str = "other"
    source_zh: str = ""
    tags: list[str] = Field(default_factory=list)
    domain: str = "其他"

    subscribed: bool = Field(
        default=True, description="在用户的装载面上吗（未定制时恒 True）")
    mandatory: bool = Field(
        default=False, description="必装项，不可退订（界面上该禁用那个开关）")
    pending_rec_id: str = Field(
        default="", description="有一条待确认的推荐指向它时，那条推荐的 id")


class MarketCatalogResponse(BaseModel):
    total: int = 0
    page: int = 1
    skills: list[MarketEntry] = Field(default_factory=list)
    customised: bool = False
    degraded: bool = False
    reason: str = ""


class MarketStatusResponse(LiveEffectFields):
    customised: bool = False
    subscribed_count: int = 0
    market_total: int = 0
    mandatory: list[str] = Field(default_factory=list)
    #: entries 里有、但注册表里没有的名字。**不剔除、只上报** —— 技能集合是动态的，
    #: 替用户把名字从清单里删掉等于替他改单。这个字段就是那张名单的对账者。
    missing_entries: list[str] = Field(default_factory=list)
    #: 订阅文件读不出来的原因（``""`` = 读得出来，含「文件不存在」）。
    #: 非空时界面要显示一条横幅：现在按**全订阅**在跑，不是按空订阅。
    unreadable: str = ""
    pending_count: int = 0
    store_path: str = ""
    degraded: bool = False
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 写
# ─────────────────────────────────────────────────────────────────────────────

class SubscriptionWriteRequest(BaseModel):
    """增量写。两个字段可以同时给（先加后减）。"""

    subscribe: list[str] = Field(default_factory=list)
    unsubscribe: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class SubscriptionWriteResponse(LiveEffectFields):
    ok: bool = True
    reason: str = ""
    degraded: bool = False

    customised: bool = False
    materialised: bool = Field(
        default=False,
        description="这一次把「全订阅」固化成了明确名单（第一次定制时为 True）")
    changed: list[str] = Field(default_factory=list)
    skipped_mandatory: list[str] = Field(
        default_factory=list, description="被拒绝退订的必装项（连同原因显示给人看）")
    unknown: list[str] = Field(
        default_factory=list, description="请求里注册表不认识的名字（照收不报错，但要说）")
    subscribed_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# 推荐
# ─────────────────────────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    id: str = ""
    skill: str = ""
    by_agent: str = ""
    reason: str = ""
    conversation_id: str = ""
    at: str = ""
    status: str = "pending"
    resolved_at: str | None = None


class RecommendationListResponse(BaseModel):
    pending: list[Recommendation] = Field(default_factory=list)
    #: 已裁决的尾巴。**拒绝也留在这里** —— 留痕的意思是能看见「他拒过」。
    resolved: list[Recommendation] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""


class AuditEntry(BaseModel):
    """订阅面变过一次的记录：什么时候、动了什么、经哪条路。

    ``via`` 不是闭集（``recommendation:<id>`` 带 id），但 ``ui`` / ``import`` /
    ``reset`` / ``materialise`` 是固定词。
    """

    at: str = ""
    action: str = ""
    skills: list[str] = Field(default_factory=list)
    via: str = ""


class AuditResponse(BaseModel):
    """**这个响应是审计流唯一的消费方。**

    它存在的理由值得写下来：第一版把 audit 写进了盘、也写了往返测试，然后就没有
    任何东西读它 —— 一条只写不读的日志，正是本仓
    [[producer_wired_consumer_absent]] 说的那个形状，而我是在引用着那条纪律的
    同一天犯的。前端有一道结构闸门钉着这个端点必须有调用方。
    """

    entries: list[AuditEntry] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""


class RecommendationResolveRequest(BaseModel):
    accept: bool = False

    model_config = {"extra": "forbid"}


class RecommendationResolveResponse(SubscriptionWriteResponse):
    recommendation: Recommendation | None = None
    already_subscribed_by_default: bool = Field(
        default=False,
        description="未定制态下接受 —— 默认已经全订阅，没有把他转成明确名单")


# ─────────────────────────────────────────────────────────────────────────────
# 分享（导出 / 导入）
# ─────────────────────────────────────────────────────────────────────────────

class ManifestEntry(BaseModel):
    name: str = ""
    source: str = "other"
    version: str = ""
    #: 仅 ``user_composite``：内嵌完整 CompositeSpec（纯数据）。
    #: **其余来源只有名字** —— 代码不随单走（manifest-only 红线）。
    spec: dict | None = None


class SubscriptionManifest(BaseModel):
    kind: str = "mast-skill-subscription"
    schema_version: int = 1
    exported_at: str = ""
    machine: str = ""
    app_version: str = ""
    customised: bool = False
    entries: list[ManifestEntry] = Field(default_factory=list)


class ManifestMissing(BaseModel):
    name: str = ""
    source: str = ""
    hint: str = ""


class ImportReport(BaseModel):
    matched: list[str] = Field(default_factory=list)
    missing: list[ManifestMissing] = Field(default_factory=list)
    composites_saved: list[str] = Field(default_factory=list)
    composites_failed: list[dict] = Field(default_factory=list)
    mandatory_added: list[str] = Field(default_factory=list)


class ImportRequest(BaseModel):
    manifest: dict = Field(default_factory=dict)
    mode: str = Field(default="replace", description="replace | merge")
    dry_run: bool = False

    model_config = {"extra": "forbid"}


class ImportResponse(SubscriptionWriteResponse):
    dry_run: bool = False
    report: ImportReport = Field(default_factory=ImportReport)


# ─────────────────────────────────────────────────────────────────────────────
# 实验室中心索引（二期）
# ─────────────────────────────────────────────────────────────────────────────

class LabPublishRequest(BaseModel):
    label: str = Field(default="", description="给别人看的名字，如「qPlus 日常」")
    note: str = Field(default="", description="一句话说明这份列表适合谁")

    model_config = {"extra": "forbid"}


class LabPublishResponse(BaseModel):
    ok: bool = False
    reason: str = ""
    id: str = ""
    status: str = Field(default="", description="服务端的收件状态，通常是 pending_review")
    skill_count: int = 0


class LabIndexEntry(BaseModel):
    """中心索引的一行 —— **只有摘要**，条目本身要单独取回。"""

    id: str = ""
    label: str = ""
    note: str = ""
    machine: str = ""
    exported_at: str = ""
    app_version: str = ""
    skill_count: int = 0
    embedded_specs: int = 0
    status: str = ""
    ts_server: str = ""


class LabIndexResponse(BaseModel):
    subscriptions: list[LabIndexEntry] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""


class LabFetchResponse(BaseModel):
    """取回一份中心索引里的 manifest。

    失败有**自己的字段**，不折叠进 manifest 的某个数据位。第一版把错误塞进了
    ``exported_at``（前端靠 ``kind`` 空不空来判断成没成）—— 那正是本仓
    [[read_failure_folded_into_a_value]] 那一族：一个故障被答成了一个值，而它
    合理得没人会去核。
    """

    ok: bool = False
    reason: str = ""
    manifest: SubscriptionManifest | None = None
