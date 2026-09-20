"""conduct 指挥层的 REST 契约。

设计:``docs/v2/design/campaign_director_design.md`` §7(API schema 与前端面板)。

## 一个端点渲染整个面板

``GET /api/conducts/{id}`` 是**面板唯一的数据源**。这不是省事,是防一类具体的
缺陷:面板由 N 个端点拼起来时,每个端点各有各的时刻,于是「状态是 RUNNING」
和「等待卡还亮着」可以同时显示,而两者都各自没错。一个端点 = 一个时刻。

WS 帧只做**触发**:收到就 refetch 这个端点。帧里**不放增量状态** —— 总线只重放
最后 100 条,一个跑三天的 conduct 必然丢帧,而按帧累积状态的客户端会带着一个
错的状态一直显示下去。

## 状态词表是闭集 —— 以及它**没有**沿着哪条路传下去(2026-08-15 更正)

``ConductStatus`` / ``OpName`` 用 ``Literal`` 写死,值逐字对应
``mast.conduct.store`` 的 ``STATUSES`` / ``OPS``,由
``tests/v2/unit/conduct/test_api_conducts.py`` 的 parity 测试钉住
(2026-08-20 更正文件名:原来写的 ``test_api_conduct.py`` 少一个 s,那个文件
不存在 —— 一条指向不存在文件的「由 X 钉住」,读的人会以为已经有人在守)。

⚠️ **原来这里写着「so the TS typegen gets a closed enum」,那句话是假的。**
响应字段(``ConductDetail.status`` 等)声明的是 ``str`` 而不是这两个 Literal,
所以 openapi 里根本没有这个枚举,``schema.d.ts`` 拿到的是 ``string`` ——
typegen 一个字都没收到。写下那句话的时候没有人去看生成出来的文件。

字段**故意**保持 ``str``:词表在写入端已经由 ``store.record()`` 强制
(不在闭集就抛),而把它钉进响应模型意味着一行手工改过的 SQLite 就能让面板端点
500 —— 那违背「读端点永不 5xx」。代价是 TS 那边拿不到闭集,于是前端的镜像
(``frontend/src/lib/conduct.ts``)由 ``frontend/test/conduct.test.ts``
**直接读 store.py** 对账。三方(store / 本模块 / TS)因此各有一条通向同一个真源
的检查,而不是串成一条会在中间断掉的链。

## degraded 是什么意思

``cd_enabled`` 默认关(见 :mod:`mast.conduct.settings`),关着的时候引擎根本
不存在。这时读端点回 ``degraded=true`` + 一句为什么,**不是 500、也不是一个
空的正常响应** —— 后者会让「引擎没开」长得和「没有 conduct」一模一样。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

#: 顶层状态。逐字对应 ``mast.conduct.store.STATUSES``。
ConductStatus = Literal[
    "draft", "approved", "running", "waiting_operator", "waiting_condition",
    "yielding", "paused", "halted_estop", "recovery_pending", "completed",
    "aborted",
]

#: 意图。逐字对应 ``mast.conduct.store.OPS``。
OpName = Literal["pause", "resume", "abort", "takeover", "ack", "set_attended",
                 "waive_condition", "override_decision"]


# ── 建 / 批 ─────────────────────────────────────────────────────────

class ConductCreateRequest(BaseModel):
    #: 2026-08-20 由必填改成选填:``from_plan_id`` 那条路上模板是**编译出来的**
    #: (plan 里的绑定块说了算),再让调用方抄一份 spec_id 过来就是两个真源。
    #: 两个都给且不一致 ⇒ 400(明拒),不是静默挑一个。
    spec_id: str = Field("", description="模板标识,见 GET /api/conducts/templates;"
                                         "走 from_plan_id 编译时可以留空")
    experiment_id: str = Field(..., description="绑定的实验;没有实验归属的 conduct,产物没有落点")
    params: dict[str, Any] = Field(default_factory=dict,
                                   description="模板声明的可填参数。**超包络拒绝,不夹紧**")
    attended: Optional[bool] = Field(None, description="值守位;不给就用模板的初值")
    from_plan_id: str = Field("", description="由某个 APPROVED plan 编译而来。"
                                              "给了它而 params 留空 ⇒ 走编译器"
                                              "(mast.conduct.compiler);"
                                              "params 非空 ⇒ 只当来源标注")


class ParamEcho(BaseModel):
    """逐字段回显 —— 让用户看见系统**实际**收到的是什么。"""

    name: str
    value: Any = None
    unit: str = ""
    ok: bool = True
    error: str = ""


class StageSummary(BaseModel):
    stage_id: str = ""
    title: str = ""
    steps: int = 0
    mandatory: bool = True
    capabilities: list[str] = Field(default_factory=list)


class ConductCompileError(BaseModel):
    """编译器的一条错误。``code`` 取自 ``mast.conduct.compiler.COMPILE_ERROR_CODES``。

    **逐码过河,不只是一句话**:前端要按码渲染不同的下一步(``SLOT_UNFILLED``
    → 去方案页填这个槽;``SKILL_UNKNOWN`` → 这台机器上没有这个技能)。把它拍扁
    成 ``errors: list[str]`` 的话,界面只能显示一段红字,而人要做的事完全不同。
    """

    code: str = ""
    stage_id: str = ""
    slot_name: str = ""
    detail: str = ""


class ConductCompileResponse(BaseModel):
    """``POST /api/conducts/compile-from-plan/{plan_id}`` 的干跑结果。

    ## 为什么是 200 而不是 4xx

    这个端点**纯读**:它回答的是「这份方案现在编不编得动」,而「编不动」是一个
    **答上来了的答案**,不是一次失败的请求。真正的 4xx 在
    ``POST /api/conducts``(那一步要建东西)。唯一的例外是 plan 库没接上 ——
    那是「读不到」,回 503 + ``degraded``,免得它长得像「这份方案没问题但空空
    如也」。
    """

    ok: bool = False
    plan_id: str = ""
    #: 认出来的模板。**ok=False 时照样给** —— 一屏错误码需要一个落点,
    #: 而 ``spec`` 本身在编不动时刻意不产出(不部分编译)。
    spec_id: str = ""
    spec_version: int = 0
    title: str = ""
    #: 编出来的参数(ok=False 时是**已解析出的那部分**,仅供回显,不可执行)。
    params: dict[str, Any] = Field(default_factory=dict)
    #: 逐字段回显 —— 与 create 同一张表,同一个渲染件。
    params_echo: list[ParamEcho] = Field(default_factory=list)
    stages_summary: list[StageSummary] = Field(default_factory=list)
    errors: list[ConductCompileError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: **没跑成的检查**。非空 ⇒ ``ok`` 必是 False。与校验器的三态同一个意思:
    #: 「没检查」不许长得像「检查通过」。
    checks_skipped: list[str] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""


class ConductCreateResponse(BaseModel):
    ok: bool = False
    conduct_id: str = ""
    status: str = ""
    params_echo: list[ParamEcho] = Field(default_factory=list)
    stages_summary: list[StageSummary] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    #: 走 ``from_plan_id`` 编译失败时的**闭集码**。``errors`` 里同时有人读的
    #: 一句话,但那一句话没法让界面分支 —— 见 :class:`ConductCompileError`。
    compile_errors: list[ConductCompileError] = Field(default_factory=list)
    degraded: bool = False


class ConductApproveRequest(BaseModel):
    approved_by: str = Field(..., min_length=1,
                             description="批的人。空的不收 —— 审计要的是名字不是 true")


class IgnitionView(BaseModel):
    """supervised 的撤销窗，从**面板的角度**看是什么样。

    ``remaining_s`` 是**算出来的**（``ignite_at - now``），不是存下来的：存剩余
    秒数的话，进程在窗口中间重启会从头再数，而一份 conduct 的时间尺度是
    hour~day，重启是常态。
    """

    by: str = ""
    ignite_at: float = 0.0
    delay_s: float = 0.0
    remaining_s: float = 0.0


class ConductApproveResponse(BaseModel):
    ok: bool = False
    conduct_id: str = ""
    status: str = ""
    #: 这次批准**没有**立刻点火（supervised + agent 批）。响应里此前只有
    #: ``ok=True``，于是「批了就跑」和「批了，十分钟后跑，这段时间能撤回」
    #: 在调用方眼里长得一模一样 —— 而后者正是这一档存在的理由。
    deferred: bool = False
    #: 撤销窗长度（秒）。0 = 立刻点火。
    ignition_delay_s: float = 0.0
    #: approve 时渲染的人读快照(``conduct/<id>/spec_vNNN.md``)。空串 = 没渲染成,
    #: 原因在 ``errors`` 里 —— 不静默。
    spec_doc_path: str = ""
    #: 校验器的**三态**:ok(没发现错误)/ complete(该跑的检查都跑了)。
    #: approve 只看 ``approvable = ok and complete``,「没检查」不许长得像「检查通过」。
    validation_ok: bool = False
    validation_complete: bool = False
    findings: list[str] = Field(default_factory=list)
    checks_skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    #: approve **实测冻结**进 params 的那些槽(今天只有 ``coord_epoch``)。
    #:
    #: 它必须回到屏幕上:用户在新建表单里手填过一个值,而这里把它换掉了 ——
    #: 一次没人看见的覆盖,和一次没发生的覆盖长得一样。读不到当前代次时这一批
    #: 不会是「空的」,而是整个 approve 被 409 拒掉(读不到 ≠ 0)。
    frozen_params: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False


# ── 列表 ────────────────────────────────────────────────────────────

class ConductListRow(BaseModel):
    conduct_id: str = ""
    title: str = ""
    spec_id: str = ""
    status: str = ""
    stage_id: str = ""
    step_id: str = ""
    updated_at: str = ""


class ConductListResponse(BaseModel):
    conducts: list[ConductListRow] = Field(default_factory=list)
    #: 当前那个未了结的(至多一个,单活跃不变式)。
    active_conduct_id: str = ""
    engine_enabled: bool = False
    degraded: bool = False
    reason: str = ""


class ParamSpecRow(BaseModel):
    """一个用户可填参数的完整声明 —— 新建表单据此渲染一个输入格。

    ## 这里**故意没有** ``default``

    模板的 ``ParamSpec.default`` 今天全是 ``None``,而且那是**刻意**的
    (``test_every_physical_number_is_an_operator_parameter`` 钉着):填不出来
    就说明这次实验的工作点还没定,那正是该停下来的时候,而不是让模板替他挑一个
    看起来合理的数。

    把 ``default`` 放进这个契约,等于在表单那一侧摆一个「预填我」的把手 ——
    而一个预填好的工作点会让人以为它已经被谁定过了。``sts_condition`` 是最刺眼
    的一例:出厂只有一个 ``default`` 组,而**它刻意是未标定的**。所以这里只回
    「必填与否」(``required = default is None``),数值本身**不过河** ——
    「移除诱因,别在提示词里说服模型」的同一条:渲染件根本拿不到那个数,
    就不会预填。

    代价写在明处:模板将来真的给了某个参数一个默认值时,用户在表单上看不到
    它是多少,只看到「选填(留空则用模板的默认值)」—— 以及提交后
    ``params_echo`` 里那个**实际生效**的值(``_param_echo`` 用的正是
    ``params.get(name, p.default)``)。
    """

    name: str
    #: ``float`` / ``int`` / ``str`` / ``bool``,逐字对应 ``spec.PARAM_TYPES``。
    type: str = "str"
    unit: str = ""
    #: 包络。**超出 ⇒ 拒绝,绝不夹紧**(``check_params`` 的实现;表单照此提示,
    #: 并且**同样不夹紧** —— 夹紧会把一次越界输入变成一次看起来正常的运行)。
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    help: str = ""
    #: 闭集参数的合法值。非空时 min/max 不参与判定。
    choices: list[Any] = Field(default_factory=list)
    #: ``ParamSpec.default is None`` ⇒ 必填。为什么不回 default 本身,见上。
    required: bool = True


class TemplateRow(BaseModel):
    spec_id: str = ""
    spec_version: int = 0
    title: str = ""
    stages: list[str] = Field(default_factory=list)
    #: 这个模板现在批得下去吗 —— 引用的技能/分析函数全在这台机器上?
    approvable: bool = False
    #: 批不下去的话,缺什么。
    findings: list[str] = Field(default_factory=list)
    checks_skipped: list[str] = Field(default_factory=list)
    #: 这个模板要用户填哪些参数 —— **新建表单据此渲染逐字段输入**。
    #:
    #: 2026-08-15 补:原来这一行不存在,于是「新建一份 conduct」在界面上没有
    #: 任何入口,只剩 curl。那与 approve 是同一个形状(设计 §7 让 approve 只接受
    #: UI 来源 ⇒ 不给按钮 approve 就不存在),而**做不到的动作和不存在的动作,
    #: 在屏幕上长得一模一样**。
    params_schema: list[ParamSpecRow] = Field(default_factory=list)


class TemplateListResponse(BaseModel):
    templates: list[TemplateRow] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""


# ── 旋钮目录(设置页动态渲染)────────────────────────────────────

class KnobChoice(BaseModel):
    """一个有限档位。``value`` 是存进设置的那个数，``label_zh`` 是人看的名字。

    名字的真源在后端（``autonomy.describe``）—— 前端拿到什么就显示什么，不在
    那边再写一份中文档位名。第二处定义迟早会与第一处岔开，而岔开时没有任何
    东西会报错：界面上写着「半自主」，实际存进去的是别的档。
    """

    value: float
    label_zh: str = ""


class ConductKnob(BaseModel):
    """一个可调旋钮的完整描述。

    字段名与 ``EnvHistoryKnob`` / 电流监控的目录**逐字一致** —— 前端有一份共用
    的旋钮渲染件,第三份目录换一套字段名就等于逼它长出第二条分支。
    """

    key: str
    label_zh: str = ""
    hint_zh: str = ""
    min: float = 0.0
    max: float = 0.0
    step: float = 0.0
    default: float = 0.0
    value: float = 0.0
    #: 有限档位的旋钮（今天只有 ``cd_autonomy``）。空 = 这是个连续量，照旧用
    #: 数字框。monitoring / envhistory 的目录不发这个字段，它们的渲染件对空
    #: 列表天然是 no-op。
    choices: list[KnobChoice] = []
    is_bool: bool = False


class ConductConfigResponse(BaseModel):
    """引擎开没开 + 旋钮目录。

    ``enabled`` 与 ``director_running`` 是**两件事**:前者是用户的意愿(设置里
    那个开关),后者是此刻线程活着没有。刚翻开开关而进程还没重启时两者不同,
    而把它们合成一个数会让「设了但没生效」看不出来。
    """

    enabled: bool = False
    director_running: bool = False
    knobs: list[ConductKnob] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""


# ── 面板(GET /conduct/{id})───────────────────────────────────────

class StagePos(BaseModel):
    idx: int = 0
    id: str = ""
    title: str = ""


class StepPos(BaseModel):
    idx: int = 0
    id: str = ""
    title: str = ""
    kind: str = ""
    started_at: Optional[float] = None
    #: 软超时。**不杀步**(TCP 不可杀),只喂停滞告警的阈值。
    timeout_s: Optional[float] = None


class DetourView(BaseModel):
    active: bool = False
    return_stage: str = ""
    return_step: int = 0
    reason: str = ""
    entered_at: Optional[float] = None


class AckView(BaseModel):
    required: bool = False
    at: Optional[float] = None
    by: str = ""


class ConditionView(BaseModel):
    desc: str = ""
    #: Director **真正在用**的阈值与时窗 —— 绑定到 params 的条件在进等待那一刻
    #: 就解析定了。面板显示的必须是这一组,不是模板里的占位值:两个数不一致时,
    #: 库里等的是一个、屏幕上写的是另一个,而两边都各自没错。
    threshold: Optional[float] = None
    stale_after_s: Optional[float] = None
    #: 最近一次读数。``None`` = **读不到**,不是 0。
    current_value: Optional[float] = None
    met: bool = False
    met_since: Optional[float] = None
    #: 读数太旧 / 读不到。stale ≠ 条件不满足 —— 面板要分开显示。
    stale: bool = True
    reading_age_s: Optional[float] = None
    #: 人把 condition 闸标成「证据由我提供」。**持续显示**,不是 ack 那一刻闪一下。
    waived: bool = False
    waived_by: str = ""
    waive_reason: str = ""


class ActiveWaitView(BaseModel):
    wait_id: str = ""
    kind: str = ""
    message: str = ""
    ack: AckView = Field(default_factory=AckView)
    condition: Optional[ConditionView] = None
    #: 还缺哪个闸(人的确认 / 物理条件)。两个证据答两个问题,互不替代。
    lacking: list[str] = Field(default_factory=list)
    entered_at: Optional[float] = None
    last_notified_at: Optional[float] = None
    #: 心愿单里那条请求的 id(重播/重启的幂等就靠它)。
    request_id: str = ""


class GateHistoryRow(BaseModel):
    ts: str = ""
    gate_id: str = ""
    stage_id: str = ""
    verdict: str = ""
    route: str = ""
    note: str = ""
    #: ``rule`` / ``llm``。**面板必须分得开**:一条 LLM 判决与一条 rule 判定在
    #: 上面那五个字段里长得一模一样,而事后要做的核对完全不同。
    kind: str = ""
    #: 真的答了这一题的那个模型(空 = 不是 LLM 判的,或者没记到)。
    #: 记的是**实例上读到的**名字,不是配置里写的 —— 静默回退到另一家 provider
    #: 在本仓发生过,事后从配置根本看不出来。
    llm_model: str = ""
    #: 决策是怎么解析出来的(structured / text_json / call_re / bare_name)。
    #: ``bare_name`` 是标签噪声,一眼能看见才有意义。
    llm_parse_path: str = ""
    #: 走了 escape 没有(模型弃权、或判决器故障)。
    escaped: bool = False
    #: 判决**没发生**时的那句话(超时 / 建不出模型 / 返回值不在闭集里)。
    #: 空 = 判决真的发生了。
    llm_unavailable: str = ""
    #: 这一段已用 / 允许的 LLM 唤醒次数。``None`` = 这条不是 LLM 闸门。
    llm_wakes_used: Optional[int] = None
    llm_wakes_max: Optional[int] = None


class BudgetView(BaseModel):
    #: 实测花销。``None`` = **读不到**(不是花了 0)。
    spent_usd: Optional[float] = None
    cap_usd: float = 0.0
    #: 读不到的时候说清为什么 —— 一个空着的预算条会被读成「没花钱」。
    reason: str = ""
    #: **这条上限今天拦不拦得住东西。** ``False`` 时面板必须把它画成「已声明、
    #: 当前不生效」,而不是一个看起来在守护的数字 —— 一个装得像在拦的闸,比没有
    #: 闸更危险,因为看的人会据此放心。
    enforceable: bool = False
    #: 为什么拦不住(``enforceable=False`` 时必非空)。
    not_enforceable_why: str = ""
    #: 逐币种的实测花销,如 ``{"CNY": 1.23}``。**没有合计** —— 合并需要汇率,
    #: 而本仓不自造汇率。空 dict = 这份 conduct 名下还没有记录;
    #: 与 ``spent_usd is None``(读不到)是两件事。
    measured_by_currency: dict[str, float] = Field(default_factory=dict)


class HeartbeatView(BaseModel):
    at: Optional[float] = None
    age_s: Optional[float] = None
    stalled: bool = False
    threshold_s: float = 0.0
    #: 心跳只证明**决策循环**在转,不证明步在推进 —— 执行一次长步时它本来就不
    #: 更新。所以停滞判据分两支:``in_step`` 时看**这一步**跑了多久(超过
    #: timeout_s + 余量才算停滞),不在步里时看心跳年龄(超过 3×tick 就是循环死了)。
    #: 合成一支的话,一次正常的两小时扫描会天天报警,而报警报久了就没人看。
    in_step: bool = False
    #: 当前步已经跑了多久;``None`` = 读不到开始时刻(**不是 0 秒**)。
    step_elapsed_s: Optional[float] = None
    #: 这个判断是怎么来的 —— 面板直接显示,不让人自己推。
    reason: str = ""


class TimelineRow(BaseModel):
    stage_id: str = ""
    title: str = ""
    status: str = ""      # done | current | pending
    steps_done: int = 0
    steps_total: int = 0


class FolderView(BaseModel):
    """实验文件夹这一侧 —— 人读副本到底有没有在写。"""

    path: str = ""
    spec_doc: str = ""
    #: ``None`` = 读不到行数(不是 0 行)。
    progress_lines: Optional[int] = None
    reason: str = ""


class PendingDecisionView(BaseModel):
    """停在一次闸门判定上,等人说「继续」还是「算了」。

    **闸门的裁决原样带着**(``verdict`` / ``reason``)—— 人推翻的是**它**,
    而事后对账要看得见被推翻的是什么。合成一句「等人处理」的话,一次放行与一次
    正常通过在记录里就长得一样了。
    """

    decision_id: int = 0
    gate_id: str = ""
    stage_id: str = ""
    #: 闸位:entry / step / exit。放行之后往哪走由它决定。
    which: str = ""
    #: 闸门当时判的那个裁决(``wait_operator``)。
    verdict: str = ""
    reason: str = ""


class ConductDetail(BaseModel):
    """面板唯一的数据源。一次响应 = 一个时刻。"""

    ok: bool = False
    conduct_id: str = ""
    experiment_id: str = ""
    spec_id: str = ""
    spec_version: int = 0
    title: str = ""
    status: str = ""
    #: 为什么停 / 为什么中止 / 闩是因为什么挂的。**永不空着转入异常态。**
    status_reason: str = ""
    attended: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    stage: StagePos = Field(default_factory=StagePos)
    step: StepPos = Field(default_factory=StepPos)
    detour: Optional[DetourView] = None
    evidence_epoch: int = 0
    active_wait: Optional[ActiveWaitView] = None
    #: 停在一次**闸门判定**上时,人可以放行的那一条。``None`` = 现在没有可放行的
    #: 判定(在跑、或者停的是一个 wait 步、或者停法根本不是闸门)。
    #:
    #: 面板据此决定「继续」按钮亮不亮 —— 灰的时候也要**说得出为什么灰**。
    pending_decision: Optional[PendingDecisionView] = None
    gates_history: list[GateHistoryRow] = Field(default_factory=list)
    budget: BudgetView = Field(default_factory=BudgetView)
    llm_wakes: dict[str, int] = Field(default_factory=dict)
    heartbeat: HeartbeatView = Field(default_factory=HeartbeatView)
    active_run_id: str = ""
    timeline: list[TimelineRow] = Field(default_factory=list)
    folder: FolderView = Field(default_factory=FolderView)
    approved_by: str = ""
    approved_at: str = ""
    #: 撤销窗还开着的话，它长什么样；已经点火 / 没有窗 ⇒ None。
    ignition: Optional[IgnitionView] = None
    created_at: str = ""
    updated_at: str = ""
    #: 引擎线程活着吗。**关着也读得到状态** —— 关掉一个功能不该连带把
    #: 「看它现在什么样」一起关掉。
    engine_enabled: bool = False
    director_running: bool = False
    #: 本次响应有没有没能算出来的东西(取不到模板、读不到花销……)。
    not_available: list[str] = Field(default_factory=list)
    degraded: bool = False
    reason: str = ""


# ── 意图 ────────────────────────────────────────────────────────────

class OpRequest(BaseModel):
    by: str = Field(..., min_length=1, description="谁按的。审计要名字")
    note: str = Field("", max_length=500)


class AbortRequest(BaseModel):
    by: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1,
                        description="**必填**:一个没有理由的中止,事后没人答得上为什么")


class AckRequest(BaseModel):
    wait_id: str = Field(..., min_length=1,
                         description="每次等待唯一;对旧等待点的确认会被 409 拒掉")
    by: str = Field(..., min_length=1)
    note: str = Field("", max_length=500)


class WaiveRequest(BaseModel):
    wait_id: str = Field(..., min_length=1)
    by: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1,
                        description="**必填**:waive 不是默默放行,面板会持续显示这个标记")


class OverrideDecisionRequest(BaseModel):
    """用户放行**一次**闸门判定。

    ``decision_id`` = 那一次 ``gate_evaluated`` 的 event_id。每判一次换一个 ——
    与 ack 的 ``wait_id`` 同一条纪律,防的是同一件事:**对一个已经翻篇的判定说
    「继续」**。它也因此不是「跳过这道闸」:下一次走到同一道闸,照样重新判。
    """

    decision_id: int = Field(..., ge=1)
    by: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1,
                        description="**必填**:人推翻了一次闸门裁决,"
                                    "面板会持续显示,报告也带")


class AttendedRequest(BaseModel):
    attended: bool
    by: str = Field(..., min_length=1)


class OpResponse(BaseModel):
    ok: bool = False
    op_id: int = 0
    queued: bool = False
    #: abort 专用:per-run abort Event **已经**置位了吗(不等 tick)。
    #: False 且 queued=True = 当前没有在跑的步,下一 tick 生效。
    abort_signalled: bool = False
    #: ack 专用:还缺哪个闸。
    wait_echo: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    degraded: bool = False


__all__ = [
    "ConductStatus", "OpName",
    "ConductCreateRequest", "ConductCreateResponse", "ParamEcho",
    "ConductCompileError", "ConductCompileResponse",
    "StageSummary", "ConductApproveRequest", "ConductApproveResponse",
    "ConductListRow", "ConductListResponse", "ParamSpecRow", "TemplateRow",
    "TemplateListResponse", "ConductKnob", "ConductConfigResponse",
    "StagePos", "StepPos", "DetourView", "AckView",
    "ConditionView", "ActiveWaitView", "GateHistoryRow", "BudgetView",
    "HeartbeatView", "TimelineRow", "FolderView", "ConductDetail",
    "OpRequest", "AbortRequest", "AckRequest", "WaiveRequest",
    "AttendedRequest", "OpResponse",
]
