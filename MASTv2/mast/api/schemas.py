"""Pydantic request/response models — the SINGLE SOURCE OF TYPES for the API.

These are exported via FastAPI's ``/openapi.json`` and consumed by the frontend
type generators (``openapi-typescript`` → ``schema.d.ts``; ``orval`` → typed
TanStack Query hooks). Per F4 (end-to-end type safety), the two languages must
never drift: a CI gate fails if regenerating from this schema produces a diff.

Phase 2 covers READ-ONLY shapes only. Write/streaming shapes (chat SSE, admin
overrides, experiment CRUD) arrive in Phase 3.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Re-export the core hardware-safety model verbatim so the API and the core
# share ONE definition (no parallel copy that could drift). SafetyLimits already
# lives in mast.config as a BaseModel — it IS a valid response_model as-is.
from mast.config import SafetyLimits as SafetyLimits  # noqa: F401  (re-export)

ThinkingMode = Literal["none", "fixed", "tunable"]


class HealthResponse(BaseModel):
    """Liveness + a snapshot of which optional backends are wired in."""

    status: Literal["ok"] = "ok"
    service: str = "mast-api"
    version: str
    # Which optional core subsystems the running API process has wired. In
    # standalone dev mode most are False (only pure-config endpoints serve real
    # data); when mounted next to the live app they flip True.
    skill_registry_wired: bool = False
    settings_store_wired: bool = False
    experiment_storage_wired: bool = False


class ModelInfo(BaseModel):
    """One row of the model-capability table (config.MODEL_PRESETS + derived)."""

    alias: str = Field(description="short alias used in the UI, e.g. 'glm-5.2'")
    model_id: str = Field(description="provider model id sent on the wire")
    provider: str = Field(description="moonshot | deepseek | qwen | anthropic | minimax | zhipu")
    description: str
    default_max_tokens: int
    thinking_mode: ThinkingMode = Field(
        description="none → never send thinking; fixed → always-on; tunable → off/low/.../max actually changes behaviour"
    )
    output_limit: int = Field(description="max OUTPUT tokens this model accepts")
    input_context: int = Field(description="conservative max INPUT context tokens")
    is_default: bool = False


class ModelsResponse(BaseModel):
    """The full model-capability table for the Settings UI model picker."""

    default_alias: str
    thinking_presets: dict[str, int] = Field(
        description="thinking level name → budget tokens (Anthropic-style); only meaningful for tunable models"
    )
    models: list[ModelInfo]


class SettingsResponse(BaseModel):
    """Persisted UI settings (SettingsStore whitelist). All optional — an unset
    key means 'use the live config default'. Mirrors SettingsStore.KNOWN_KEYS."""

    model_alias: Optional[str] = None
    thinking: Optional[str] = None
    voice: Optional[str] = None
    voice_autoplay: Optional[bool] = None
    voice_mode: Optional[str] = None      # "ptt" | "wake" | "duplex"
    voice_narrate: Optional[bool] = None  # speak agent tool executions
    font_scale: Optional[str] = None
    theme: Optional[str] = None
    nanonis_host: Optional[str] = None
    nanonis_port_main: Optional[int] = None
    nanonis_port_monitor: Optional[int] = None
    nanonis_port_data: Optional[int] = None
    nanonis_port_emergency: Optional[int] = None
    qa_model: Optional[str] = None
    codex_live_search: Optional[bool] = None
    autonomy_mode: Optional[str] = None  # "safe" | "semi" | "auto"
    orchestrator_recursion_limit: Optional[int] = None  # run-task super-step budget (default 250)
    orchestrator_auto_background: Optional[bool] = None  # auto-detach a paired literature survey; default OFF
    # LangGraph 退出的两个切换面（2026-08-26/27）。都是 DEFAULT OFF / live-read /
    # fail-safe OFF：设置缺失或读不出来一律走旧路径。放在 HTTP 面上而不是只留一个
    # 配置文件，是因为真机在别处 —— 翻开关不该需要先 SSH 上去。
    engine_v2_workflow_agent: Optional[bool] = None  # 工作流 agent 节点走 v2 AgentLoop
    engine_v2_private_chat: Optional[bool] = None   # 私聊回合走 v2 ConversationEngineV2（boot 时读）
    engine_v2_background: Optional[bool] = None      # 后台编排走 v2 OrchestratorLoop
    engine_v2_group_chat: Optional[bool] = None      # 群聊编排走 v2（每次 run 读）
    engine_v2_cli: Optional[bool] = None             # `-m mast --instruction` 走 v2
    # Spend ceilings (2026-07-30). The orchestrator's budget gate existed for
    # months with nobody seeding it; these two settings are what feed it.
    #   * run budget bounds ONE run (default $80 — a BACKSTOP, deliberately well
    #     above the measured legitimate worst case of $5.19, because aborting a
    #     long campaign mid-thought is a worse and likelier failure than an
    #     expensive run; the runaway it guards against was $30.66);
    #   * daily budget bounds run COUNT, which no per-run ceiling can (a woken run
    #     is a fresh state with a fresh budget) — see core/wake_scheduler.py.
    # 0 / unset on either one = that dimension is OFF, honestly inert.
    orchestrator_run_budget_usd: Optional[float] = None  # per-run USD ceiling (default 80.0)
    daily_budget_usd: Optional[float] = None  # all-runs-per-day USD ceiling (default 300.0)
    # Environment-driven activation gating: the supervisor may PARK an agent whose
    # upstream inputs do not exist yet, instead of dispatching it into fiction.
    # Default OFF — parking is the one routing behaviour that can look identical to
    # a hang, so it is opt-in (see docs/v2/design/wakeup_scheduling.md).
    orchestrator_activation_gating: Optional[bool] = None  # default OFF
    # Per-EXPERIMENT auto-wake ceiling per local day (default 6). The loop bound for
    # product-driven waking — see core/wake_scheduler.py. 0 = 不自动唤醒;
    # "unlimited" is intentionally not offered.
    wake_max_per_day: Optional[int] = None
    vision_thresholds: Optional[dict[str, float]] = None  # VIGIL tip-quality discrimination valves
    classical_thresholds: Optional[dict[str, float]] = None  # per-instrument classical tip-quality knobs
    current_monitor: Optional[dict[str, float]] = None  # tunnelling-current monitor knobs (cm_*)
    env_history: Optional[dict[str, float]] = None  # environment-history recorder knobs (eh_*)
    # 多天 conduct 指挥线程的旋钮 (cd_*)。cd_enabled 默认 0 —— 关着的时候
    # 那条常驻线程根本不建,逐字节等于这个功能落地之前。见
    # mast/conduct/settings.py 与 docs/v2/design/campaign_director_design.md §3-1。
    conduct: Optional[dict[str, float]] = None
    hardware_modules: Optional[dict[str, bool]] = None  # optional/licensed Nanonis modules; all default OFF
    advanced_capabilities: Optional[dict[str, bool]] = None  # powers that step around a protection; all default OFF
    experiment_defaults: Optional[dict[str, Any]] = None  # operator's preferred default scan/spectroscopy params 
    instrument_profile: Optional[dict[str, Any]] = None  # per-rig retract/dI-dV config + learned calibration
    coarse_drive: Optional[dict[str, Any]] = None  # operator-declared coarse-stepper drive ceiling (admin-PIN guarded)
    scan_policy: Optional[dict[str, Any]] = None  # 按尺度分层的扫描参数表 {"tiers": [...]}
    # 修针方案表的用户覆写(2026-08-10 补齐读写两侧;写侧见
    # schemas_settings_admin_write.SettingsWriteRequest,admin-PIN guarded)。
    tip_conditioning_overrides: Optional[dict[str, Any]] = None
    # ── 对话预算 / 精炼 / 归档(2026-08-10 补齐写侧,这里是配套的读侧) ────────
    # 补写字段而不补读字段 = 一个存得进、界面上看不见的值:表单每次重挂都拿 GET
    # 回来的值重填,读不到就永远显示空的,用户会以为没保存成功而再存一次。
    # test_settings_write_reachability.test_what_you_can_write_you_can_read_back
    # 守着这个方向。
    chat_model_calls_per_run: Optional[int] = None    # 单次调用的模型调用上限(默认 30)
    chat_tool_calls_per_run: Optional[int] = None     # 单次调用的工具调用上限(默认 80)
    chat_model_calls_per_thread: Optional[int] = None  # 跨会话累计;0/缺席 = OFF
    chat_tool_calls_per_thread: Optional[int] = None   # 跨会话累计;0/缺席 = OFF
    tool_refine_enabled: Optional[bool] = None        # 每轮工具返回精炼;默认开
    tool_packs_enabled: Optional[bool] = None         # IC 工具按需加载;默认开
    tool_refine_min_chars: Optional[int] = None       # 只精炼超过这个长度的返回(默认 600)
    compaction_model: Optional[str] = None            # 压缩摘要模型别名;空 = 跟随当前模型
    literature_fetch_auto_resume: Optional[bool] = None  # 全文到货后自动续跑那轮文献;默认开
    ingest_enabled: Optional[bool] = None             # 归档总开关;默认开
    ingest_watcher_enabled: Optional[bool] = None     # 后台兜底扫盘;默认开
    ingest_copy_mode: Optional[str] = None            # "copy" | "hardlink"
    env_csv_enabled: Optional[bool] = None            # 环境读数同时写 CSV;默认开
    conv_export_enabled: Optional[bool] = None        # 对话增量导出;默认开


class HardwareModuleState(BaseModel):
    """One optional (licensed-but-maybe-absent) Nanonis hardware module."""

    id: str
    name: str
    hardware: str          # what you must physically own for this to work
    description: str
    enabled: bool
    skill_count: int
    skills: list[str] = Field(default_factory=list)


class AdvancedCapabilityState(BaseModel):
    """One advanced capability — a power that can step around a protection."""

    id: str
    name: str
    risk: str          # the ONE sentence saying which protection it steps around
    description: str
    enabled: bool
    skill_count: int
    skills: list[str] = Field(default_factory=list)


class AdvancedCapabilitiesResponse(BaseModel):
    """高级 → 高级能力. Every one ships OFF and needs the admin PIN to switch on."""

    capabilities: list[AdvancedCapabilityState] = Field(default_factory=list)
    enabled_count: int = 0
    gated_skill_count: int = 0
    pin_is_set: bool = False   # False ⇒ the toggles cannot be written at all yet


class AdminPinStatusResponse(BaseModel):
    pin_is_set: bool = False


class AdminPinSetRequest(BaseModel):
    """Set or change the admin PIN. Changing one requires the current one."""

    new_pin: str = Field(min_length=4)
    current_pin: Optional[str] = None


class AdminPinSetResponse(BaseModel):
    ok: bool = False
    reason: str = ""     # too_short | wrong_current | unreadable | io_error
    message: str = ""


class HardwareModulesResponse(BaseModel):
    """设置 → 硬件模块. ``rebuild_note`` is the honest UI string from the last
    write — a toggle only takes effect once the agent's tool list is rebuilt."""

    modules: list[HardwareModuleState] = Field(default_factory=list)
    enabled_count: int = 0
    gated_skill_count: int = 0   # skills currently withheld from the agent


class SkillIndexEntry(BaseModel):
    """Lightweight catalog row (client-side fuzzy search). Full cards are fetched
    on demand via /api/skills/{name} (Phase 3)."""

    name: str
    domain: str = "其他"
    source: str = "other"
    safety_level: str = "AUTO"
    composition_level: Optional[str] = None
    summary: Optional[str] = None


class SkillCatalogResponse(BaseModel):
    """Skill index for the Codex browser. ``degraded`` is True when the live
    SkillRegistry is not wired into this API process (standalone dev) — the
    frontend shows an empty-but-not-broken state."""

    index: list[SkillIndexEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class ExperimentSummary(BaseModel):
    """One row of the experiments list (Records tab top level)."""

    id: str
    name: Optional[str] = None
    goal: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[str] = None
    #: The experiment's CURRENT sample (newest active, else newest). The panel
    #: had nothing but the experiment UUID to show and rendered "441ebe7c-2a7",
    #: which the operator could not even identify as an experiment / sample /
    #: conversation id .
    sample_name: Optional[str] = None
    sample_id: Optional[str] = None


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentSummary] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


__all__ = [
    "ThinkingMode",
    "HealthResponse",
    "ModelInfo",
    "ModelsResponse",
    "SettingsResponse",
    "SafetyLimits",
    "SkillIndexEntry",
    "SkillCatalogResponse",
    "ExperimentSummary",
    "ExperimentsResponse",
]
