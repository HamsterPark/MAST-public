"""Persistent UI settings — closes the "settings reset on restart" gap.

The settings inventory (2026-05-30) found that almost every Lab Console setting
— global model / thinking / font / knowledge mode / voice / Nanonis host+ports —
only mutated in-memory config and was **lost on restart** (only theme had
localStorage; sensors + per-agent models had their own JSON). The unified 设置
tab needs ONE durable place for all of them.

``SettingsStore`` is a thin, robust, JSON-backed dict at
``<config_dir>/ui_settings.json`` (atomic temp+replace, corrupt-file-degrades).
It only persists a whitelisted set of keys; the GUI applies them to the live
``MASTConfig`` / clients on startup and calls :meth:`update` after each change.

Pure stdlib, no MASTConfig import — so it's trivially testable and never drags in
the heavy GUI stack.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Whitelist of persistable UI settings (anything else passed in is dropped).
KNOWN_KEYS: frozenset[str] = frozenset({
    "compaction_model",         # compaction summarizer model id; blank = follow current agent model
    "tool_refine_enabled",       # bool; per-turn tool-return refine (before_model); default on
    "tool_packs_enabled",        # bool; IC 工具按需加载(核心40个+按需取包); default on
    "tool_refine_min_chars",     # int; only refine tool returns longer than this; default 600
    "chat_model_calls_per_run",     # int; per-invocation model-call cap (default 30)
    "chat_tool_calls_per_run",      # int; per-invocation tool-call cap (default 80)
    "chat_model_calls_per_thread",  # int; cumulative model-call cap; 0/absent = OFF
    "chat_tool_calls_per_thread",   # int; cumulative tool-call cap; 0/absent = OFF
    "orchestrator_recursion_limit",  # int; run-task super-step budget (default 250)
    "literature_fetch_auto_resume",  # bool; when an operator satisfies a full-text
                            #   request, run one more literature turn on the
                            #   conversation that asked, so the agent finishes what
                            #   it had parked. DEFAULT ON (absent = ON): uploading a
                            #   paper IS the instruction to use it. Set False to keep
                            #   fulfilment silent (board closes, nobody is woken).
    "engine_v2_group_chat",  # bool; drive GROUP (run-task) orchestration on the v2
                            #   OrchestratorLoop. DEFAULT OFF, read per run.
                            #   最大的一个切换面，也是唯一一个会碰硬件的 —— 但 v2
                            #   路径**开跑前就拒绝**把 instrument_control 放进目标
                            #   名单（它的安全中间件还没有 v2 装配路径），所以打开
                            #   它只影响不碰仪器的任务。
    "engine_v2_private_chat",  # bool; drive PRIVATE chat turns on the v2
                            #   ConversationEngineV2 (AgentLoop + chat_messages)
                            #   instead of the LangGraph create_agent graph.
                            #   DEFAULT OFF, and read at BOOT rather than per turn:
                            #   两个引擎交替写同一段历史是双轨期最大的一致性雷区，
                            #   boot 级选择把它整个排除。翻回去要重启（<1 分钟）。
                            #   instrument_control 的私聊在 v2 上被**拒绝**（它的
                            #   硬件安全中间件还没有 v2 装配路径）——见 engine_v2.py。
    "engine_v2_background",  # bool; run a BACKGROUND orchestration on the v2
                            #   OrchestratorLoop instead of the LangGraph supervisor
                            #   graph. DEFAULT OFF. Second switching surface of the
                            #   LangGraph exit, and the orchestrator's first: a
                            #   background run is state-isolated, carries no
                            #   instrument agent, skips HITL entirely, and fails
                            #   without the foreground noticing. Read per spawn, so
                            #   flipping it back takes effect on the next run.
    "engine_v2_workflow_agent",  # bool; run a workflow `agent` node on the v2
                            #   AgentLoop (mast/agentruntime) instead of the
                            #   LangGraph create_agent subgraph. DEFAULT OFF —
                            #   absent = OFF = the old path, byte for byte. This is
                            #   the first switching surface of the LangGraph exit
                            #   (strangler): smallest blast radius (no streaming, no
                            #   HITL, no history, instrument_control excluded), and
                            #   flipping it back takes effect on the next delegation.
    "engine_v2_cli",         # bool; run `python -m mast --instruction …` on the v2
                            #   OrchestratorLoop instead of the LangGraph supervisor
                            #   graph. DEFAULT OFF. 第五个、也是最小的一个切换面：
                            #   CLI 是一次性的离线入口，没有历史、没有 SSE、没有
                            #   刷新续跑，失败最坏是一条指令白跑。
                            #   它也要有开关，是因为删除闸门问的「四个开关全开吗」
                            #   真正想问的是「旧路径已经不是任何一个面的生产了吗」——
                            #   一个没有开关的第五个面会让那个答复失去意义。
    "orchestrator_auto_background",  # bool; auto-detach a literature survey paired
                            #   with instrument_control to a background run so the
                            #   instrument foreground isn't barrier-blocked. DEFAULT
                            #   OFF (conservative); absent = OFF = routing unchanged.
    "orchestrator_run_budget_usd",  # float; per-run USD ceiling (default 80.0; 0 = OFF).
                            #   Feeds the supervisor's budget gate, which existed for
                            #   months with nobody seeding it — see billing/run_meter.
    "daily_budget_usd",     # float; USD across ALL runs since local midnight
                            #   (default 300.0; 0 = OFF). The dimension a per-run
                            #   ceiling cannot see: waking agents multiplies run COUNT.
    "orchestrator_activation_gating",  # bool; let the supervisor PARK an agent whose
                            #   upstream inputs do not exist yet instead of dispatching
                            #   it into fiction. DEFAULT OFF — parking can look exactly
                            #   like a hang, so it is opt-in.
    "wake_max_per_day",     # int; per-EXPERIMENT auto-wake ceiling per local day
                            #   (default 6). THE loop bound for product-driven waking:
                            #   every other guard is per-run and a wake starts a NEW
                            #   run, so they all reset — and every step succeeds, so
                            #   StallGuard (repeated-FAILURE signatures) is blind to
                            #   the cycle. 0 = 不自动唤醒; the settings path clamps at 0
                            #   so "unlimited" is deliberately NOT reachable from the UI.
    "advanced_capabilities",  # dict[str,bool]; powers that can step AROUND a
                            #   protection (script-file I/O, quitting Nanonis,
                            #   multi-pass config files, the blocking wait). ALL
                            #   default OFF and, unlike hardware_modules, they are
                            #   not about hardware you lack — they are about powers
                            #   you have and mostly should not lend to an agent.
                            #   Written ONLY through the PIN-checked admin path.
    "hardware_modules",     # dict[str,bool]; optional/licensed Nanonis hardware
                            #   modules (KPFM, 多探针, OsciHR, 高速扫描器…). ALL
                            #   default OFF: the licences exist, the hardware
                            #   mostly doesn't. A module that is off has its
                            #   skills withheld from the agent's tool list
                            #   entirely (mast.skills.hardware_modules) — so the
                            #   model can't call hardware nobody owns, and the
                            #   ~280-tool IC list doesn't grow for nothing.
                            #   Live-read holder like vision_thresholds, BUT it
                            #   additionally needs an orchestrator rebuild: the
                            #   tool list is frozen at graph-build time.
    "vision_thresholds",    # dict[str,float]; VIGIL tip-quality discrimination
                            #   valves (good/bad cut + Q/N/K/T/S per-dim cuts).
                            #   Live-read by mast.vision.thresholds holder, NOT
                            #   apply_to_config — a change takes effect on the
                            #   next tip assessment, no model reload.
    "classical_thresholds",  # dict[str,float]; per-instrument knobs for the
                            #   network-free classical tip-quality tools (fused
                            #   good/bad + oscillation + I(z) barrier range).
                            #   Live-read by mast.vision.classical_thresholds.
    "current_monitor",      # dict[str,float]; tunnelling-current monitor knobs
                            #   (cm_enabled / segment length / alert thresholds /
                            #   retention). Live-read by mast.monitoring.
                            #   thresholds — a change applies to the next
                            #   segment, no restart.
    "env_history",          # dict[str,float]; environment-history recorder knobs
                            #   (eh_enabled / bucket width / raw retention /
                            #   spectrum cadence / Z burst). Live-read by
                            #   mast.envhistory.thresholds. DEFAULT ON — absent
                            #   means "record", because environment_log growing
                            #   without bound is the existing problem this fixes.
                            #   eh_z_enabled is the one sub-switch that defaults
                            #   OFF: it is the only knob here that issues TCP.
    "conduct",              # dict[str,float]; 多天 conduct 指挥线程的旋钮
                            #   (cd_enabled / cd_stall_grace_s)。Live-read by
                            #   mast.conduct.settings。**cd_enabled 默认 0(关)**:
                            #   在它打开之前,系统里根本没有这条常驻线程 —— 关着
                            #   逐字节等于 M1-c 之前。翻开它要经真机验收(设计
                            #   campaign_director_design.md §9 的 M1)。
                            #   ⚠️ 关掉它**不会**关掉读端点:已有 conduct 的状态
                            #   照样读得到、abort 照样按得下 —— 关一个功能不该连带
                            #   把「停下它」的路一起关掉。
                            #   (2026-08-20 改名:执行层 campaign→conduct,
                            #    campaign 一词归还 logging/v2 的科研纲领。)
    "model_alias",          # global chat model alias, e.g. "kimi-k2.6"
    "thinking",             # global thinking level alias, e.g. "max" / "off"
    "voice",                # TTS voice name or "" / "off"
    "voice_autoplay",       # bool
    "voice_mode",           # ptt | wake | duplex (default voice interaction mode)
    "voice_narrate",        # bool; speak agent tool executions
    "font_scale",           # "小" | "中" | "大"
    "theme",                # "Dark" | "Light"
    "nanonis_host",         # str
    "nanonis_port_main",    # int
    "nanonis_port_monitor", # int
    "nanonis_port_data",    # int
    "nanonis_port_emergency",  # int
    "qa_model",             # 查询助手 model alias (independent of chat model)
    "codex_live_search",    # bool — Skills Codex live (per-keystroke) search toggle
    "autonomy_mode",        # "safe" | "semi" | "auto" — global tip-processing gate
                            #   for the autonomous agent path (safe=no tip work +
                            #   belief, semi=pulses→HITL + shallow shaping, auto=all)
    "experiment_defaults",  # dict; operator's preferred default scan/spectroscopy
                            #   parameters (扫描尺寸/速度/线数/setpoint/bias + notes,
                            #   #147). Live-read holder like vision_thresholds: the
                            #   agents._shared.experiment_prefs middleware injects
                            #   them into IC / experiment_design as PREFERRED
                            #   defaults (hints, not bounds — SafetyLimits still
                            #   enforces). No orchestrator rebuild needed.
    "scan_policy",          # dict {"tiers": [...]}; 按尺度分层的扫描参数表
                            #   (像素 / 每线时间 / setpoint / PI)。档数可变
                            #   (1..8),用户在设置里编辑;空 = 用出厂 4 档模板。
                            #   Live-read holder (mast.core.scan_policy) —— 由
                            #   mast.core.scan_resolver 消费,是**全系统扫图默认
                            #   参数的单一真源**。不需要重建 orchestrator。
                            #   刻意不含 bias:bias 决定探测的电子态,是物理意图
                            #   参数,不是尺度的函数。
    "zctrl_presets",        # list[dict]; 自定义 Z 参数组 (2026-08-03)。
                            #   {name, p_gain, i_gain, setpoint_a?, note?},数值以
                            #   **带 SI 前缀的字符串**存 ("3p" / "180n") —— 见
                            #   mast.core.si_quantity: 前缀强制,掉了前缀就解析失败,
                            #   而裸数字掉了指数仍是合法数字且错一万亿倍。
                            #   Live-read holder (mast.core.zctrl_presets)。
                            #   ⚠️ 这里**只**存自定义组。进针参数在 instrument_profile,
                            #   扫图参数在 scan_policy 档位表 —— 参数组按名解析到
                            #   各自的真源,不在这里复制一份(否则用户改了档位表,
                            #   按名应用的却是旧值)。
    "instrument_profile",   # dict; per-rig hardware facts + learned dI/dV
                            #   calibration (换样品退针方向 / lock-in dI/dV 参数 /
                            #   到样品标定值). Live-read holder
                            #   (mast.core.instrument_profile) injected by
                            #   instrument_profile_mw into IC / experiment_design;
                            #   set_calibration writes learned values back here via
                            #   the persist sink. No orchestrator rebuild needed.
    "instrument_init",      # dict; 新仪器初始化的**完成记录** (2026-08-03)。
                            #   {acknowledged: [...], completed_at, rig_fingerprint,
                            #    completed_by, app_version}。
                            #   ⚠️ 它**一个数值都不存** —— 数值住在各自的真源
                            #   (instrument_profile / safety_limits 覆写 /
                            #   scan_policy / coarse_drive / current_monitor)。
                            #   把 80 多项的值再存一份会得到一张自包含却与真源
                            #   不一致的第二张表。
                            #   刻意与 instrument_profile 同住 ui_settings.json:
                            #   「核对记录」必须与它描述的数据同生共死 —— 值被抹了
                            #   记录也要一起没,否则记录会比数据活得久,而那正是
                            #   「文件存不存在」这类判据的毛病 (KNOWN_ISSUES §3.2)。
                            #   Live-read holder (mast.core.instrument_init)。
    "tip_conditioning_overrides",
                            # dict; 修针方案表的**用户覆写**(2026-07-31)。
                            #   出厂表按「材料 × 制备 × 形态」给参数与安全包络
                            #   (mast.core.tip_conditioning_policy),那些是文献
                            #   起点、标着「待真机标定」;用户实机标出来的真值
                            #   写在这里,优先级高于出厂表、低于调用方显式指定。
                            #   由 tip_conditioning_resolver 延迟读取,不需要
                            #   重建 orchestrator。
                            #   ⚠️ 这个键不在 KNOWN_KEYS 里就会被 update() 静默
                            #   丢弃 —— 用户会看到「填了没生效」而没有任何报错。
    "coarse_drive",         # dict; 粗动马达驱动电压/频率的**本机声明**
                            #   {max_amplitude_v, expected_frequency_hz,
                            #    declared_by, declared_at, notes}。
                            #   **在 admin_pin.GUARDED_KEYS 里** —— 只有输入 admin
                            #   PIN 才能写,没设 PIN 则一律拒写(fail-closed)。
                            #   刻意与 instrument_profile 分开:profile 是用户
                            #   随手调的东西(避让半径之类),整体挂 PIN 太重;而这
                            #   一个数错了压电叠堆就废了,必须单独挂门。
                            #   Live-read holder (mast.core.coarse_drive)。
    # ── 实验文件夹持久化 (2026-07-28) ──────────────────────────────
    # 设计文档 docs/v2/design/experiment_folder_persistence.md
    #
    # NOTE 这里放的是**配置**，不是作用域指针。「当前实验/当前样品」刻意 NOT
    # 在这里：指针指的是 mast_experiments.db 里的行，放在另一个文件里，一旦
    # 数据目录切换 / db 还原 / 多份数据目录，它立刻变成悬空引用。它住在
    # active_scope 单行表里，与被指向的行同生共死。
    "experiment_root",      # str; 所有实验文件夹的父目录。空 = 用默认
                            #   (安装盘的 MAST-Data/experiments)。刻意在代码仓库
                            #   和安装目录之外：实验数据比软件活得久，OTA 更新和
                            #   重装都不该碰到它。Live-read holder
                            #   (mast.core.experiment_paths.set_experiment_root)。
    "ingest_enabled",       # bool (default True); 把 Nanonis 产物复制进实验文件夹
    "ingest_watcher_enabled",  # bool (default True); 后台兜底扫 Nanonis 保存目录，
                            #   把用户手动存的文件也收进来 (source=manual)
    "ingest_copy_mode",     # "copy" | "hardlink"; hardlink 省盘但只能同卷
    "env_csv_enabled",      # bool (default True); 温度等环境读数同时写 CSV
                            #   (实验级按天轮转 + 当前样品级)，供 Excel/Origin 直接打开
    "conv_export_enabled",  # bool (default True); 对话增量导出成 jsonl+md。
                            #   这是唯一能保住被 8000 行上限裁掉的历史的地方
    # ── 墓碑：三个删掉的键（2026-08-10）────────────────────────────────
    #
    # ── 2026-08-24 删除：knowledge_mode（auto | simple | normal | expert）──
    #
    # 这个键是**前端可见、可保存、后端零读者**的幽灵开关。链路是完整的：
    # 设置页有单选框 → POST /api/settings → 这张表 → `config.knowledge.default_mode`
    # ——**然后就没有然后了**。全仓再没有一处读 `config.knowledge.default_mode`。
    #
    # 它本来要驱动的是 v1 的三档知识注入（`mast/knowledge/assembler.py` 的
    # ContextAssembler / PlannerMode）。v2 的裁决是**知识走拉取式**：agent 用
    # `query_knowledge(query, detail_level)` 按需取，而不是把知识预先塞进系统
    # 提示词。理由写在 `docs/v2/design/tip_registry_and_hardware_profile.md`：
    # 「assembler.py 是 D-discarded，全仓零调用者……别接它」。
    #
    # 留着的代价不是三行代码，是**它看起来在起作用**：用户把它从 expert 调到
    # simple，界面确认「已保存」，而注入内容一个字节都不会变。一个不起作用的
    # 开关比没有这个开关更误导人 —— 那是这个仓反复栽的形状。
    #
    # **要把它加回来需要回答**：谁读它？也就是注入侧（`agents/_shared/` 的某个
    # 中间件，或 `prompts/manifest`）有没有一处真的按这个值改变注入内容。只把
    # 键加回白名单而不加那处消费，等于原样恢复这条谎话。
    #
    # 想按角色/按场景裁注入的话，现在有真的机制了：中间件自己的 `AGENTS` 常量
    # （定向注入）与 `tool_packs`（工具按需加载），两者都有双向闸门钉着。

    #   ingest_max_file_mb          「单文件上限，防失控的 3ds 写满盘」
    #   ingest_max_gb               「归档总量配额」
    #   nanonis_follow_session_path 「切样品时自动把 Nanonis 保存目录指向该样品
    #                                的 raw/nanonis/」
    #
    # 三个都实测**全仓零引用**（MASTv2/**.py 与 frontend/src 全搜过，除这张表
    # 自己的定义行外一处读它们的代码都没有）。所以它们不是「有功能但默认关着」
    # ——**那两条分支都不存在**。留着的代价不是浪费三行，是这张表的中文说明在
    # 承诺三样做不到的事：一个来读这张表的人会以为盘满有保护。
    #
    # 删掉它们**不改变任何行为**（没有代码读，且从来没有写入口，所以没有人的
    # ui_settings.json 里会有这三个键；即便有，`_load` 会在下次启动时按白名单
    # 过滤掉）。
    #
    # **要把它们加回来需要回答**：谁在写那道配额？——即 `logging/v2/ingest_sink`
    # 或 `runtime._ingest_submit` 里有没有一处**在复制之前**读到这个数并拒绝。
    # 只加回 KNOWN_KEYS 而不加那处判断，等于把这条注释描述的谎话原样恢复。
    # （`io/nanonis_files.py` 的 MAX_FILE_BYTES 不算：那是**读文件**的守卫，
    #   拦的是把 2 GB 的 .3ds 读进内存，不是归档配额。）
    # 至今没有观测到任何一处读过这三个键。
    # NOTE: per-agent model/thinking overrides are NOT persisted here — they
    # already have a durable home in ConfigOverrideRegistry's agent_overrides.json
    # (shared with the Agents React UI). The 设置 tab writes through that same
    # store so the two surfaces stay in sync; only the single-instance UI prefs
    # above belong in ui_settings.json.
})

_FILENAME = "ui_settings.json"

#: Persisted key → attribute on ``config.nanonis``. ONE list, because two places
#: apply it: :meth:`SettingsStore.apply_to_config` at startup and the
#: ``POST /api/settings`` route live. Keeping it here is what stops those two
#: from drifting into "the restart uses the new port, this session doesn't".
NANONIS_KEY_TO_ATTR: dict[str, str] = {
    "nanonis_host": "host",
    "nanonis_port_main": "port_main",
    "nanonis_port_monitor": "port_monitor",
    "nanonis_port_data": "port_data",
    "nanonis_port_emergency": "port_emergency",
}


def default_config_dir(user_root: str | os.PathLike[str] | None = None) -> Path:
    """``<user_root>/config`` (created). user_root None → cwd."""
    root = Path(user_root) if user_root is not None else Path.cwd()
    d = root / "config"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - best effort
        pass
    return d


# ── the process's ONE store ───────────────────────────────────────────────────
#
# Why this exists at all: seven call sites in ``core/`` wrote ``SettingsStore()``
# — no argument — inside a broad ``except Exception``. ``config_dir`` is a
# REQUIRED positional parameter, so every one of those calls raised TypeError and
# returned its fallback. Six of them were measured on 2026-08-10: none had ever
# read a value the operator wrote. The fallbacks happened to equal the defaults,
# so archiving ran, CSV was written, exports were made — and an operator who
# turned any of it OFF got no feedback that nothing had changed.
#
# The fix is NOT "remember to pass the argument at seven call sites". It is
# having ONE place that answers "which store does this process use", so a call
# site cannot express the question wrongly.
#
# It must be ONE INSTANCE, not one path. ``SettingsStore`` caches the file in
# ``_data`` at construction; a write goes through whichever instance the API
# holds (``bootstrap`` wires ``ctx.settings_store = rt._settings``). A second
# instance built against the same directory would answer from the snapshot it
# loaded at construction and never see that write — i.e. reading the right file
# is not enough, you have to read the same OBJECT the writer mutates.
_process_store: "SettingsStore | None" = None
_process_lock = threading.Lock()
_auto_store: "SettingsStore | None" = None
_auto_store_dir: "Path | None" = None


def set_process_store(store: "SettingsStore | None") -> None:
    """Publish the store this process writes through (``CoreRuntime.setup``).

    After this, :func:`settings_store_for_runtime` hands back the very object the
    ``POST /api/settings`` path mutates, so a setting written at 14:03 is visible
    to the runtime's next read — no restart, no second cache.
    """
    global _process_store
    with _process_lock:
        _process_store = store


def settings_store_for_runtime() -> "SettingsStore":
    """The store for code that has no ``self._settings`` to reach for.

    Returns the published store when there is one. Otherwise it builds one
    against ``<project_root()>/config`` — the same directory ``CoreRuntime.setup``
    uses — which is the honest answer when the runtime never came up (headless
    tooling, a setup that failed): read the operator's file rather than assume
    the defaults.

    ``project_root()`` is re-resolved on every call and the cached instance is
    reused only while it resolves to the same directory. Freezing the path at
    first call is a shape this repo has been bitten by before (huggingface_hub
    fixes its cache dir at import, so a later ``HF_HOME`` does nothing); here it
    would mean a test that redirects ``MAST2_PROJECT_ROOT`` silently keeps
    reading whatever directory the first caller happened to resolve.
    """
    from mast._runtime_paths import project_root

    global _auto_store, _auto_store_dir
    with _process_lock:
        if _process_store is not None:
            return _process_store
        d = default_config_dir(project_root())
        if _auto_store is None or _auto_store_dir != d:
            _auto_store = SettingsStore(d)
            _auto_store_dir = d
        return _auto_store


def reset_process_store() -> None:
    """Forget both the published and the auto-built store (tests)."""
    global _process_store, _auto_store, _auto_store_dir
    with _process_lock:
        _process_store = None
        _auto_store = None
        _auto_store_dir = None


class SettingsStore:
    """JSON-backed persistent UI settings (atomic write, thread-safe)."""

    def __init__(self, config_dir: str | os.PathLike[str]):
        self._dir = Path(config_dir)
        self._path = self._dir / _FILENAME
        self._lock = threading.RLock()
        self._data: dict = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────
    def _load(self) -> None:
        with self._lock:
            self._data = {}
            try:
                if self._path.exists():
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        self._data = {k: v for k, v in raw.items() if k in KNOWN_KEYS}
            except Exception as exc:  # corrupt file → start empty, never crash
                logger.warning("ui_settings load failed (%s); starting empty", exc)
                self._data = {}

    def _save(self) -> None:
        with self._lock:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, self._path)
            except Exception as exc:  # best-effort; never break the GUI
                logger.warning("ui_settings save failed: %s", exc)

    # ── api ───────────────────────────────────────────────────────────
    def load(self) -> dict:
        """Return a copy of the persisted settings (whitelisted keys only)."""
        with self._lock:
            return dict(self._data)

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def update(self, **kwargs) -> dict:
        """Merge whitelisted keys (None values are ignored) + persist."""
        with self._lock:
            changed = False
            for k, v in kwargs.items():
                if k in KNOWN_KEYS and v is not None and self._data.get(k) != v:
                    self._data[k] = v
                    changed = True
            if changed:
                self._save()
            return dict(self._data)

    def set(self, key: str, value) -> dict:
        return self.update(**{key: value})

    # ── apply to a live MASTConfig (startup hydration) ────────────────
    def apply_to_config(self, config) -> dict:
        """Hydrate a MASTConfig from persisted settings. Returns the applied
        subset (for logging). Best-effort: any setter that doesn't exist is
        skipped, so this is forward/backward compatible with config changes.

        Theme + font_scale are CLIENT-side (CSS/localStorage) and are NOT applied
        here — the GUI reads them from load() and seeds the components/JS instead.
        """
        applied: dict = {}
        s = self.load()

        # model alias (provider switches with it via LLMConfig.use)
        ma = s.get("model_alias")
        if ma:
            try:
                if hasattr(config.llm, "use"):
                    config.llm.use(ma)
                else:
                    config.llm.model = ma
                applied["model_alias"] = ma
            except Exception as exc:
                logger.debug("apply model_alias failed: %s", exc)

        # thinking level (Anthropic / MiniMax tunable budget). Applied AFTER
        # model_alias (use() doesn't touch the budget). Without this the
        # operator's chosen thinking level was silently dropped on every restart
        # — config.llm reverted to the default max budget.
        tl = s.get("thinking")
        if tl:
            try:
                if hasattr(config.llm, "set_thinking"):
                    config.llm.set_thinking(tl)
                    applied["thinking"] = tl
            except Exception as exc:
                logger.debug("apply thinking failed: %s", exc)

        # voice
        v = s.get("voice")
        if v is not None:
            try:
                if v in ("", "off", "Off"):
                    config.voice.autoplay = False
                else:
                    config.voice.default_voice = v
                applied["voice"] = v
            except Exception:
                pass
        if s.get("voice_autoplay") is not None:
            try:
                config.voice.autoplay = bool(s["voice_autoplay"])
                applied["voice_autoplay"] = bool(s["voice_autoplay"])
            except Exception:
                pass
        if s.get("voice_mode") in ("ptt", "wake", "duplex"):
            try:
                config.voice.default_mode = s["voice_mode"]
                applied["voice_mode"] = s["voice_mode"]
            except Exception:
                pass
        if s.get("voice_narrate") is not None:
            try:
                config.voice.narrate_execution = bool(s["voice_narrate"])
                applied["voice_narrate"] = bool(s["voice_narrate"])
            except Exception:
                pass

        # nanonis host + ports
        nano = config.nanonis
        for key, attr in NANONIS_KEY_TO_ATTR.items():
            val = s.get(key)
            if val is not None and val != "":
                try:
                    setattr(nano, attr, val)
                    applied[key] = val
                except Exception:
                    pass
        return applied


__all__ = ["SettingsStore", "KNOWN_KEYS", "default_config_dir",
           "NANONIS_KEY_TO_ATTR", "settings_store_for_runtime",
           "set_process_store", "reset_process_store"]
