"""新仪器初始化的 API 契约。

设计文档：``docs/v2/design/new_instrument_initialization.md``

这些模型只搬运 :mod:`mast.core.instrument_init` 的目录 + 各真源的当前值。
**没有一个字段是新的存储** —— 值住在 ``instrument_profile`` / ``safety_limits``
覆写 / ``scan_policy`` / ``coarse_drive`` / ``current_monitor`` 里，这里只是把它们
摆到同一张表上。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class InputSpec(BaseModel):
    """怎么渲染这一项的输入框 —— **从后端真源现取，前端不抄**。

    抄一份就会漂：``tests/v2/unit/io/test_instrument_profile_frontend_parity.py``
    存在的理由，正是本仓已经因为「后端一张表、前端另一张表」丢过一次 qPlus 基线。
    """

    type: str = "float"              # float | int | choice | str | table | modules
    min: Optional[float] = None
    max: Optional[float] = None
    choices: list[dict[str, str]] = Field(default_factory=list)


class InitItemModel(BaseModel):
    """目录里的一项 + 它此刻的值与状态。"""

    id: str
    key: str
    store: str
    group: str
    severity: str                    # required | recommended | optional
    label: str
    unit: str = ""
    #: 页面上这一项的**全部**说明，一句话：填什么 + 哪里读。
    #:
    #: 由 ``what`` + ``where`` 合并而来（#39 / #40，第四轮）。改名不是只改内容：
    #: 留旧名字装新东西 = 字段标签说谎，而下游全都信它 —— 这个仓吃过那个亏。
    hint: str = ""
    #: 填错会怎样。**只有 ``safety_critical`` 的项有**，而且一定有。
    consequence: str = ""
    #: 能从仪器读一次来对账的项，写上读法。**只喂「从仪器读一次」按钮，不上屏** ——
    #: 它和 ``hint`` 说的是同一个地方，印两遍就是同一句话说两遍（#40 点名的就是它）。
    probe: str = ""
    has_factory: bool = True
    #: 填错会伤到硬件（或让某条安全判据反向）。只有这一族带 ``consequence``，
    #: 而且常显 —— 页面上除此之外一个字都没有（#11 / #20 / #30 / #39 / #40）。
    safety_critical: bool = False
    unanswered_values: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    depends_values: list[str] = Field(default_factory=list)
    #: 用户**实际存下来**的值（None = 没存过；与「出厂值恰好相同」是两回事）。
    value: Any = None
    #: 出厂默认（没有默认时为 None）。只用于显示「你没填时系统在用什么」。
    factory: Any = None
    status: str = "default"          # set | acknowledged | default | missing | n/a
    #: 仅 ``n/a`` 时非空：指名是上面**哪一项的哪个回答**让这一项失去意义。
    #: 「上面某一项」这种说法把读者留在一道 48 项的谜题前。
    na_reason: str = ""
    complete: bool = False
    input: InputSpec = Field(default_factory=InputSpec)


class InitGroupModel(BaseModel):
    id: str
    title: str
    intro: str


class SeverityCount(BaseModel):
    total: int = 0
    complete: int = 0


class PreampCheck(BaseModel):
    """「跨阻增益」与「满量程」的交叉核对结果。"""

    expected_full_scale_a: Optional[float] = None
    ratio: Optional[float] = None
    consistent: Optional[bool] = None     # None = 缺一个数，算不了（不是「一致」）
    note: str = ""


class DerivedSuggestion(BaseModel):
    """由前放满量程导出的下游建议值。**建议，不自动写入。**"""

    target: str                       # "safety_limits.setpoint_max_a" 等
    value: float
    current: Optional[float] = None
    why: str = ""
    #: True 表示当前值比建议值宽松 —— 这才是危险方向（安全线开得比物理量程大）。
    current_is_looser: bool = False


class InstrumentInitResponse(BaseModel):
    """GET /api/instrument-init —— 「这台机器还差哪些数」。"""

    groups: list[InitGroupModel] = Field(default_factory=list)
    items: list[InitItemModel] = Field(default_factory=list)
    counts: dict[str, SeverityCount] = Field(default_factory=dict)
    outstanding_required: list[str] = Field(default_factory=list)
    #: **权威判据**：有 required 项没有答案。纯内容判据，不看任何标记。
    needs_setup: bool = False
    #: 指纹变了 —— 大概率换了机器（或重做了标定）。补充触发器，读不到不算变。
    fingerprint_changed: bool = False
    #: 三者任一为真就该弹（needs_setup / 指纹变了 / 从未盖过完成戳）。
    should_prompt: bool = False
    completed_at: Optional[float] = None
    rig_fingerprint: Optional[str] = None
    rig_label: str = ""
    acknowledged: list[str] = Field(default_factory=list)
    preamp_check: PreampCheck = Field(default_factory=PreampCheck)
    derived: list[DerivedSuggestion] = Field(default_factory=list)
    #: 安全包络的来源：live_guard | merged | defaults。
    safety_source: str = ""
    #: 安全包络覆写是否还等着重启才生效（None = 说不准）。
    safety_restart_pending: Optional[bool] = None
    degraded: bool = False


class InitApplyRequest(BaseModel):
    """POST /api/instrument-init/apply —— 写一个真源里的一批值。

    ``store`` 必须是既有真源之一；本端点**不新建存储**，它把写入转交给既有的
    ``POST /api/settings`` / ``POST /api/admin/overrides/safety_limits`` 通道，
    然后在 Python 里回读比对。

    数值是**人在输入框里打的**，经 HTTP JSON 直达 —— 这条链路上没有 LLM
    （见 fixes/2026-08-03-tool-call-number-corruption.md 第六点五：比对绝不能交给
    会犯错的那一方）。
    """

    store: str
    values: dict[str, Any] = Field(default_factory=dict)
    #: PIN 门后的 store（coarse_drive / hardware_modules）必需。
    admin_pin: Optional[str] = None


class ValueVerdict(BaseModel):
    """一个值写下去之后，回读回来是什么。"""

    key: str
    requested: Any = None
    stored: Any = None
    #: match（一致）｜clamped（被夹进合法区间）｜dropped（整个没收下）
    #: ｜unverifiable（读不回来）
    verdict: str = "match"
    note: str = ""


class InitApplyResponse(BaseModel):
    ok: bool = False
    store: str = ""
    verdicts: list[ValueVerdict] = Field(default_factory=list)
    #: 有任何一项不是 match —— 前端据此变红，不允许当成保存成功。
    mismatched: bool = False
    #: 安全包络覆写写完仍需重启才生效（KNOWN_ISSUES §1.1）。说实话，不说「已生效」。
    restart_required: bool = False
    pin_required: bool = False
    pin_reason: str = ""
    rejected: dict[str, str] = Field(default_factory=dict)
    degraded: bool = False
    message: str = ""


class AcknowledgeRequest(BaseModel):
    """把一批「用出厂值就对」的项标成已核对。

    有出厂默认的项，「没填」和「填的正好等于默认」在存储层看是同一件事。
    所以完成与否由**明确的核对动作**决定，而不是猜。
    """

    item_ids: list[str] = Field(default_factory=list)
    #: True = 取消核对（重新变回未完成）。
    undo: bool = False


class CompleteRequest(BaseModel):
    """盖「这台机器配过了」的戳。带上当前硬件指纹，换机器时能自动发现。"""

    completed_by: str = ""
    rig_label: str = ""


class InitRecordResponse(BaseModel):
    ok: bool = False
    acknowledged: list[str] = Field(default_factory=list)
    completed_at: Optional[float] = None
    rig_fingerprint: Optional[str] = None
    rig_label: str = ""
    degraded: bool = False
    message: str = ""


class ProbeField(BaseModel):
    """从仪器读回的一个量 + 它与已登记值的对账结论。"""

    key: str
    label: str = ""
    read: Any = None
    configured: Any = None
    #: match | mismatch | unread | not_configured
    verdict: str = "unread"
    note: str = ""


class InitProbeResponse(BaseModel):
    """POST /api/instrument-init/probe —— 从仪器读一次，与登记值对账。

    **只报告不一致，不改任何东西。** 若要自动化只允许收紧
    （`skills/builtins/instrument_limits.py`：能放宽自己限值的 agent 严格来说
    没有限值）。
    """

    ok: bool = False
    fields: list[ProbeField] = Field(default_factory=list)
    rig_fingerprint: Optional[str] = None
    fingerprint_changed: bool = False
    #: Nanonis 自己的 Z 软限位配好了但没启用 —— 见 KNOWN_ISSUES §1.3。
    z_limits_enabled: Optional[bool] = None
    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False
    message: str = ""


class InitBundleResponse(BaseModel):
    """GET /api/instrument-init/export —— 带得走的配置包。

    **不含学习量**（``didv_at_contact_v`` / ``tilt_cal_*`` / qPlus 实测共振）：
    它们是那根针在那台机器上学出来的，搬过去就是一个自信的错值。
    """

    ok: bool = False
    bundle: dict[str, Any] = Field(default_factory=dict)
    stripped: list[str] = Field(default_factory=list)   # 被剥掉的学习量键
    degraded: bool = False


class ImportBundleRequest(BaseModel):
    bundle: dict[str, Any] = Field(default_factory=dict)


class InitImportResponse(BaseModel):
    """POST /api/instrument-init/import —— **只解析，不落盘。**

    导入是「把数字填进表单」，不是「写进硬件」：同型号不等于同一台，前放可能不
    一样，压电标定一定不一样。返回的值进入「待复核」状态，用户逐组确认才写。
    """

    ok: bool = False
    stores: dict[str, Any] = Field(default_factory=dict)
    rig_label: str = ""
    app_version: str = ""
    exported_at: Optional[float] = None
    #: 必须重新签一次才能生效的段（coarse_drive —— 填错了叠堆就废）。
    needs_resign: list[str] = Field(default_factory=list)
    rejected: str = ""
