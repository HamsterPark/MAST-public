"""Pydantic request/response models for Wave A — domain ``settings_admin_write``.

These are the typed contract for the #1-unblock write surface that the TS SPA
lost when the UI was rewritten Gradio→TS: the unified settings write, the
per-agent model/thinking override, and the fine-grained knowledge / skill
guidance / encyclopedia / quick-prompt editors.

The shapes mirror the live (now-removed) Gradio handlers + the kept backends:

* unified settings   → ``webui.settings_store.SettingsStore.update`` (+ the live
  ``config.llm.use`` / ``config.llm.set_thinking`` apply, mirroring
  ``settings_store.apply_to_config``);
* per-agent override  → ``admin.override_store.ConfigOverrideRegistry.set_agent_override``;
* knowledge edit      → ``ConfigOverrideRegistry`` ``KNOWLEDGE_OVERRIDES`` (keyed by ktype);
* guidance edit       → ``GUIDANCE_OVERRIDES`` (``skill_extra`` sub-key per skill);
* encyclopedia edit   → ``ENCYCLOPEDIA_OVERRIDES`` (section keys domains / intents /
  hierarchy / verification);
* quick-prompts       → ``QUICK_PROMPTS`` (``{"prompts": [...]}``).

EVERY response carries a ``degraded`` boolean so a standalone API process (no
live core wired) returns an empty-but-valid body instead of 500-ing.

Safety / merge / business logic NEVER lives here — these models only carry data
to and from the kept core backends (R6). The API layer is a thin relay.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Fine-grained knowledge categories the admin surface can edit. These map to
# top-level keys inside ``knowledge_overrides.json`` (deep-merged over the code
# defaults in ``mast.knowledge``). Kept as a free ``str`` path-param at the route
# so a NEW ktype is reachable without a contract bump, but the canonical set is
# documented here for the frontend type generators.
KnowledgeType = Literal[
    "workflows",
    "fault_diagnosis",
    "hardware_profile",
    "experiment_design",
    "image_databases",
]

# Encyclopedia config sections (top-level keys inside encyclopedia_overrides.json).
EncyclopediaSection = Literal[
    "domains",
    "intents",
    "hierarchy",
    "verification",
]

# 统一设置写入禁止未知字段，避免 extra='ignore' 静默丢弃输入后仍返回成功。
# extra='allow' 同样不合适：额外字段进入存储白名单可能绕过此入口的显式校验。
# 未声明的键应得到明确校验失败；合法写入字段由 SettingsPatch 与请求模型对齐。
# 未落地的写入不能返回 ok=True。
class SettingsWriteRequest(BaseModel):
    """A partial unified settings write (the 设置 tab's single save).

    Only whitelisted keys (``SettingsStore.KNOWN_KEYS``) are persisted, and a
    ``None`` value means 'leave unchanged'. The model alias + thinking level are
    ALSO applied to the live ``config.llm`` (use / set_thinking) when a live
    config is wired, mirroring ``settings_store.apply_to_config``.

    ``extra='forbid'``: a key this model does not have is REFUSED (422), never
    silently dropped — a write that did not land must never answer ``ok=true``.
    The measured incident behind that is in the comment above this class.
    """

    model_config = ConfigDict(extra="forbid")

    model_alias: Optional[str] = None
    thinking: Optional[str] = None
    qa_model: Optional[str] = None
    font_scale: Optional[str] = None
    theme: Optional[str] = None
    codex_live_search: Optional[bool] = None
    voice: Optional[str] = None
    voice_autoplay: Optional[bool] = None
    voice_mode: Optional[str] = None      # "ptt" | "wake" | "duplex"
    voice_narrate: Optional[bool] = None  # speak agent tool executions
    autonomy_mode: Optional[str] = None  # "safe" | "semi" | "auto"
    # Nanonis connection (host + the four TCP roles). These were in KNOWN_KEYS and
    # in the READ schema but not here, so 高级管理 → 硬件's 「保存」 built a body of
    # five keys, pydantic's default extra='ignore' dropped all five, model_dump()
    # was {} , store.update() did nothing — and the response was still
    # ok=True/degraded=False, i.e. a green 「已保存 Nanonis 连接配置。」 toast over a
    # write that never happened (measured 2026-08-10; ui_settings.json was not even
    # created). It stayed invisible because the neighbouring 「重新连接」 button uses
    # a DIFFERENT endpoint with DIFFERENT field names (/api/nanonis/connect,
    # host/port_main) and that path works — so connecting succeeded on the spot and
    # only "will it remember after a restart" was false, a question you cannot ask
    # until the next restart.
    nanonis_host: Optional[str] = None
    nanonis_port_main: Optional[int] = None
    nanonis_port_monitor: Optional[int] = None
    nanonis_port_data: Optional[int] = None
    nanonis_port_emergency: Optional[int] = None
    orchestrator_recursion_limit: Optional[int] = None  # run-task super-step budget (default 250)
    orchestrator_auto_background: Optional[bool] = None  # auto-detach a paired literature survey; default OFF
    # LangGraph 退出的两个切换面（2026-08-26/27）。写入侧与读出侧**必须成对**——
    # 只加一边的结果是「表单存得进读不回」，而这条闸门（test_settings_write_
    # reachability）正是为此存在的：它抓到了我只改 KNOWN_KEYS 的那一版。
    engine_v2_workflow_agent: Optional[bool] = None  # 工作流 agent 节点走 v2 AgentLoop
    engine_v2_private_chat: Optional[bool] = None   # 私聊回合走 v2 ConversationEngineV2（boot 时读）
    engine_v2_background: Optional[bool] = None      # 后台编排走 v2 OrchestratorLoop
    engine_v2_group_chat: Optional[bool] = None      # 群聊编排走 v2（每次 run 读）
    engine_v2_cli: Optional[bool] = None             # `-m mast --instruction` 走 v2
    # Spend ceilings + activation gating (2026-07-30). Three whitelists have to agree
    # for a new setting to be writable at all: SettingsStore.KNOWN_KEYS (else the
    # write is a SILENT no-op — that gap once broke the admin PIN gate), the read
    # schema in schemas.py, and this write schema.
    orchestrator_run_budget_usd: Optional[float] = None  # per-run USD ceiling (default 80.0; 0 = OFF)
    daily_budget_usd: Optional[float] = None  # USD across ALL runs today (default 300.0; 0 = OFF)
    orchestrator_activation_gating: Optional[bool] = None  # park unready agents; default OFF
    wake_max_per_day: Optional[int] = None  # per-experiment auto-wake ceiling/day (default 6; 0 = 不自动唤醒)
    experiment_defaults: Optional[dict[str, Any]] = None  # preferred default scan/spectroscopy params 
    instrument_profile: Optional[dict[str, Any]] = None  # per-rig retract/dI-dV config + learned calibration
    # Coarse-stepper drive ceiling. In admin_pin.GUARDED_KEYS: a write without a
    # valid PIN is refused outright, and with no PIN configured at all it is
    # refused too (fail-closed) — see mast.core.coarse_drive for why this one
    # number is worth its own gate.
    coarse_drive: Optional[dict[str, Any]] = None
    # 修针方案表的用户覆写(2026-08-10)。
    #
    # 这个键在 KNOWN_KEYS 里、resolver 每次 resolve 都读它、两处模块注释都说它
    # 「优先级最高」—— 而**全仓没有任何 API 写得进它**:它从来不是
    # SettingsWriteRequest 上的字段,`instrument_init` 只写自己那一个 key。
    # 一个存在、会被读、却没有写入口的覆写机制。三张白名单必须同时点头才写得
    # 进去(上面那段注释说的),
    # 而这一条缺的是**第三张**。
    #
    # 在 admin_pin.GUARDED_KEYS 里:它能放宽 max_abs_pulse_v / max_poke_depth_m。
    tip_conditioning_overrides: Optional[dict[str, Any]] = None
    scan_policy: Optional[dict[str, Any]] = None  # 按尺度分层的扫描参数表 {"tiers": [...]}
    # 自定义 Z 参数组。只有自定义组住在这里 —— 进针那组在 instrument_profile,
    # 扫图那组在 scan_policy 档位表;ApplyZCtrlPreset 按名解析到各自的真源。
    # 数值是**带 SI 前缀的字符串**("3p" / "180n"),见 mast.core.si_quantity。
    zctrl_presets: Optional[list[dict[str, Any]]] = None
    vision_thresholds: Optional[dict[str, float]] = None  # VIGIL tip-quality discrimination valves
    classical_thresholds: Optional[dict[str, float]] = None  # per-instrument classical tip-quality knobs
    current_monitor: Optional[dict[str, float]] = None  # tunnelling-current monitor knobs (cm_*)
    env_history: Optional[dict[str, float]] = None  # environment-history recorder knobs (eh_*)
    # 多天 conduct 指挥线程的旋钮 (cd_*)。**cd_enabled 默认 0** —— 这是本仓
    # 「加一个可编辑的键永远是双边动作」那条纪律的第 N 次:少了这一行,写进来的
    # conduct 块会被 pydantic 静默丢掉,而响应仍然 ok=true,用户会以为开关翻了。
    # 写完由路由调 mast.conduct.service.apply_settings —— 一个「按了没反应」的
    # 开关比没有开关更危险。
    conduct: Optional[dict[str, float]] = None
    hardware_modules: Optional[dict[str, bool]] = None  # optional/licensed Nanonis modules; all default OFF
    advanced_capabilities: Optional[dict[str, bool]] = None  # powers that step around a protection; all default OFF

    # ── 对话预算 / 精炼 / 归档:13 个「存得进、读得到、写不进」的键 ──────────
    #
    # 这 13 个 2026-08-10 之前都在 KNOWN_KEYS 里、都有程序在读、而**没有任何
    # HTTP 写入口**。它们不是被拒绝的,是被 pydantic 静默丢掉的 —— 响应仍然
    # ok=true。上面那段 extra='forbid' 的注释讲的就是这件事。
    #
    # 取值校验在 routes/settings_admin_write.py 里(与端口同一处),理由也相同:
    # 这几个的**坏值不会当场报错**。``_pos()`` 对 per_run 的 ``v <= 0`` 返回
    # 出厂默认,``_ingest_copy_mode()`` 对不认识的字符串返回 "copy" ——
    # 用户填 0 以为关掉了上限,实际拿到 30,而这件事在任何界面上都看不出来。

    # 单次调用的模型/工具调用上限。**这是会中途砍断长任务的那个预算**:
    # ForgeAuTip 曾被
    # 「Model call limits exceeded: run limit (30/30)」砍断,而用户够不着它。
    # 0 或负数不是「关闭」—— core/runtime.py:_pos 会把它换成出厂默认,所以路由
    # 直接拒(≥1)。要真的放宽就填一个大数。
    chat_model_calls_per_run: Optional[int] = None   # default 30
    chat_tool_calls_per_run: Optional[int] = None    # default 80
    # 跨会话累计上限。这两个 0 = OFF 是**真的** OFF(_pos 的默认就是 0),
    # 所以这里允许 0,只拒负数。
    chat_model_calls_per_thread: Optional[int] = None  # 0 = OFF
    chat_tool_calls_per_thread: Optional[int] = None   # 0 = OFF
    # 每轮工具返回的精炼(before_model)。精炼会改写模型看到的数值文本,
    # 想核对原文时得关得掉。
    tool_refine_enabled: Optional[bool] = None       # default ON
    tool_packs_enabled: Optional[bool] = None        # default ON
    tool_refine_min_chars: Optional[int] = None      # default 600
    # 压缩摘要用的模型别名。空串 = 跟随当前 agent 模型。
    compaction_model: Optional[str] = None
    # 上传全文后是否自动把停住的那轮文献跑完(默认开)。KNOWN_KEYS 的说明里写着
    # 「Set False to keep fulfilment silent」—— 那是一句对用户说的话,而在此
    # 之前全仓没有任何地方 set 得了它。
    literature_fetch_auto_resume: Optional[bool] = None
    # 实验文件夹归档的四个开关 + 一个模式。读侧 2026-08-10 刚修好(七处
    # ``SettingsStore()`` 无参构造实测全死),写侧就是这里。
    ingest_enabled: Optional[bool] = None            # default ON
    ingest_watcher_enabled: Optional[bool] = None    # default ON;改了要重启才起效
    # "copy" | "hardlink"。刻意用 str + 路由里校验,不用 Literal:Literal 的违规是
    # 422,而这个端点的**用户级**拒绝一律走 200 + rejected(前端的单一判据
    # lib/settingsWrite.ts 只认得那个形状)。422 留给「客户端送了一个不存在的键」
    # 那类程序错误。
    ingest_copy_mode: Optional[str] = None  # hardlink 省盘但只能同卷
    env_csv_enabled: Optional[bool] = None           # default ON
    conv_export_enabled: Optional[bool] = None       # default ON

    # Required whenever hardware_modules or advanced_capabilities is present. Never
    # persisted — it is compared (SHA-256) against config/admin_pin.txt and dropped.
    admin_pin: Optional[str] = None


class SettingsWriteResponse(BaseModel):
    """Result of a unified settings write. ``persisted`` echoes the whitelisted
    subset actually stored by the core. ``applied`` lists the keys also pushed to
    the live ``config.llm`` (empty when no live config is wired). ``degraded`` is
    True when no SettingsStore is wired (the write was a no-op)."""

    ok: bool = False
    persisted: dict[str, Any] = Field(default_factory=dict)
    applied: list[str] = Field(default_factory=list)
    degraded: bool = False
    # Set when a PIN-guarded key (hardware_modules / advanced_capabilities) was
    # written without a valid PIN. The write did NOT happen. pin_reason is one of
    # no_pin_set | empty | wrong | unreadable — the UI turns it into a prompt.
    pin_required: bool = False
    pin_reason: str = ""
    # Set only when a capability key changed. The agent's tool list is frozen at
    # graph-build time, so a toggle is NOT live until a rebuild happens; this is
    # the honest string saying what will actually occur ("后台重建中，数秒后生效"
    # / "任务运行中，暂不重建"). Surfaced verbatim in the UI — never swallowed.
    rebuild_note: str = ""
    # key -> 中文原因. Set when a payload key was STRUCTURALLY invalid and the
    # whole write was refused BEFORE persisting (scan_policy: a tier table with
    # overlapping bounds or no catch-all tier is not a table, and persisting it
    # would leave the operator running the factory defaults while the settings
    # file holds their broken one — a difference invisible in every UI.
    # zctrl_presets: same shape of argument, one step harder — those values get
    # written into the Z feedback loop by name later on). Nothing was written
    # when this is non-empty.
    rejected: dict[str, str] = Field(default_factory=dict)


# ── Per-agent model / thinking override ───────────────────────────────────────
class AgentModelOverrideRequest(BaseModel):
    """A per-agent model / thinking override (the Agents tab's LLM picker).

    Only the provided fields are merged into ``agent_overrides.json``; a ``None``
    field leaves that aspect unchanged (mirrors
    ``ConfigOverrideRegistry.set_agent_override``)."""

    model: Optional[str] = None
    thinking: Optional[str] = None


class AgentModelOverrideResponse(BaseModel):
    """Result of a per-agent override write. ``override`` echoes the agent's
    merged ``{model, thinking}`` entry. ``degraded`` True when no registry is
    wired (no-op)."""

    ok: bool = False
    agent_id: str
    override: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False


# ── Generic override-section read (knowledge / guidance / encyclopedia) ───────
class SectionConfigResponse(BaseModel):
    """The effective config for one fine-grained section (defaults merged with
    the persisted override) plus the raw override layer. ``data`` is the
    effective value the frontend renders; ``override`` is only the diff that the
    core persists; ``has_override`` flags whether a saved override exists.
    ``degraded`` True when the live backend/defaults are unavailable."""

    key: str
    data: Any = None
    override: Any = None
    has_override: bool = False
    degraded: bool = False


class SectionConfigWriteRequest(BaseModel):
    """A fine-grained section write. ``data`` is the override payload to persist
    for this section (shape is section-specific and forwarded verbatim to the
    core). An empty ``data`` means 'reset to code defaults' (the core drops the
    section / deletes the override stub)."""

    data: Any = None


class SectionConfigWriteResponse(BaseModel):
    """Result of a fine-grained section write. The core persists + triggers an
    in-process hot-reload (``save_and_reload`` / ``save_or_delete_and_reload``).
    ``reloaded`` reports whether the hot-reload fired. ``degraded`` True when no
    registry is wired (no-op)."""

    ok: bool = False
    key: str
    reloaded: bool = False
    override: Any = None
    degraded: bool = False


# ── Quick prompts ────────────────────────────────────────────────────────────
class QuickPromptsResponse(BaseModel):
    """The persisted quick-prompt list (``quick_prompts.json``'s ``prompts``).
    Each prompt is a free dict (label/text/...). ``degraded`` True when no
    registry is wired."""

    prompts: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    has_override: bool = False
    degraded: bool = False


class QuickPromptsWriteRequest(BaseModel):
    """A whole-list quick-prompt write. An empty ``prompts`` resets to code
    defaults (the core deletes the override file)."""

    prompts: list[dict[str, Any]] = Field(default_factory=list)


class QuickPromptsWriteResponse(BaseModel):
    ok: bool = False
    reloaded: bool = False
    prompts: list[dict[str, Any]] = Field(default_factory=list)
    degraded: bool = False


__all__ = [
    "KnowledgeType",
    "EncyclopediaSection",
    "SettingsWriteRequest",
    "SettingsWriteResponse",
    "AgentModelOverrideRequest",
    "AgentModelOverrideResponse",
    "SectionConfigResponse",
    "SectionConfigWriteRequest",
    "SectionConfigWriteResponse",
    "QuickPromptsResponse",
    "QuickPromptsWriteRequest",
    "QuickPromptsWriteResponse",
]
