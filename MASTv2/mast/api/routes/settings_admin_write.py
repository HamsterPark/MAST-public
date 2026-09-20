"""Wave A — domain ``settings_admin_write``: the #1 unblock write surface.

The UI was rewritten Gradio→TS SPA and lost visibility of functionality whose
LOGIC still exists in the Python core. This module re-exposes that write surface
as typed endpoints so the frontend can render it again:

* ``POST /settings``                         — unified UI settings write
  (model_alias / thinking / qa_model / font_scale / theme /
  codex_live_search / voice) via ``webui.settings_store.update`` PLUS the live
  ``config.llm.use`` / ``config.llm.set_thinking`` apply.
* ``POST /agents/{agent_id}/model-override``  — per-agent model + thinking via
  ``ConfigOverrideRegistry.set_agent_override``.
* ``GET|POST /admin/knowledge/{ktype}``       — fine-grained knowledge edit
  (knowledge_overrides.json, keyed by ktype).
* ``GET|POST /admin/guidance/{skill}``        — per-skill guidance edit
  (guidance_overrides.json ``skill_extra`` sub-key).
* ``GET|POST /encyclopedia/config/{section}`` — domains / intents / hierarchy /
  verification edit (encyclopedia_overrides.json).
* ``GET|POST /admin/quick-prompts``           — quick-prompt list edit
  (quick_prompts.json).

The API layer is a THIN passthrough: it only RELAYS to the kept backends
(``webui.settings_store``, ``admin.override_store``, ``config.llm``). NO merge,
NO validation, NO safety logic lives here — that authority stays in core (R6).

GRACEFUL DEGRADATION is mandatory: this app must boot STANDALONE with no live
core wired. Every endpoint checks ctx for the subsystem it needs; if it's absent
or any call raises, it returns a valid empty/degraded body (``degraded: true``)
— never a 500, never a crash on import. Heavy core modules are LAZY-imported
INSIDE the handler in try/except (mirrors routes/admin.py + routes/skills.py).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_settings_admin_write import (
    AgentModelOverrideRequest,
    AgentModelOverrideResponse,
    QuickPromptsResponse,
    QuickPromptsWriteRequest,
    QuickPromptsWriteResponse,
    SectionConfigResponse,
    SectionConfigWriteRequest,
    SectionConfigWriteResponse,
    SettingsWriteRequest,
    SettingsWriteResponse,
)

from mast.api.admin_pin import GUARDED_KEYS, reason_text, verify_pin
# Pure-stdlib (json/os/threading/pathlib) — not one of the heavy core modules the
# module docstring says to lazy-import. It carries the ONE key→attribute table for
# the Nanonis connection so this route and startup hydration cannot drift.
from mast.webui.settings_store import NANONIS_KEY_TO_ATTR

logger = logging.getLogger(__name__)

router = APIRouter(tags=["settings_admin_write"])

# Knowledge ktype → top-level key inside knowledge_overrides.json. The override
# payload is deep-merged over the code defaults by the core; here we only relay.
_KNOWLEDGE_KTYPES = {
    "workflows",
    "fault_diagnosis",
    "hardware_profile",
    "experiment_design",
    "image_databases",
}

# Encyclopedia section → top-level key inside encyclopedia_overrides.json. Maps
# the contract's short names to the store's canonical keys (mirrors
# admin/tabs/encyclopedia/* effective_* helpers).
_ENCYCLOPEDIA_SECTIONS = {
    "domains": "domains",
    "intents": "intent_mapping",
    "hierarchy": "composite_hierarchy",
    "verification": "skill_verification",
}

#: 存下来了，但**当前这个会话还在用老值**的键 → 它在哪儿被读一次。
#:
#: 大多数设置是 live-read holder（改完下一次扫描/下一次 tick 就用新值），这几个
#: 不是：它们在**启动或建图时**读一次就冻住了。不说这句话，用户改完看到绿色的
#: 「已保存」，然后这一轮对话照样死在老的上限上 —— 那正是这次修复要消灭的那种
#: 谎话（值确实存进去了，只是还没在用；「没存进去」和「存了还没生效」必须是两句
#: 不同的话）。
#:
#: 刻意**不**在这里顺手 rebuild：`_request_composite_rebuild` 只重建 orchestrator，
#: 私聊那几张图是 ConversationEngine 缓存的，重建了也换不掉 —— 那会得到一个只对
#: 了一半的「已生效」，比诚实的「要重启」更坏。
_READ_ONCE_AT_STARTUP: dict[str, str] = {
    # runtime._init_experiment_folders 在启动时读它决定建不建 ingest sink；
    # 之后 _ingest_submit 不再问。关掉它当场不停，打开它当场也不开。
    "ingest_enabled": "归档总开关",
    # 下面几个在 _chat_agent_middleware / _chat_call_limits 里读，
    # 而 agent 的图是建好就冻住的。
    "tool_refine_enabled": "工具返回精炼",
    "tool_refine_min_chars": "精炼阈值",
    "compaction_model": "压缩摘要模型",
    "chat_model_calls_per_run": "单次模型调用上限",
    "chat_tool_calls_per_run": "单次工具调用上限",
    "chat_model_calls_per_thread": "累计模型调用上限",
    "chat_tool_calls_per_thread": "累计工具调用上限",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _override_registry(ctx: Any):
    """Best-effort handle to a live ConfigOverrideRegistry.

    Prefers one already wired onto the context (set at integration
    time); otherwise None. We do NOT construct one here in standalone mode — that
    would touch the shared ``config/overrides`` dir and start mutating real files
    from a dev process. Absent ⇒ degrade. Mirrors routes/admin.py."""
    return getattr(ctx, "override_registry", None)


def _live_config(ctx: Any):
    """Best-effort handle to a live MASTConfig (for config.llm.use/set_thinking).

    Looks at the context's wired config / live app. None in standalone mode →
    settings still persist, but the live ``config.llm`` apply is skipped."""
    cfg = getattr(ctx, "config", None)
    if cfg is not None:
        return cfg
    app_handle = getattr(ctx, "app", None) or getattr(ctx, "live_app", None)
    return getattr(app_handle, "config", None) if app_handle is not None else None


def _live_runtime(ctx: Any):
    """Best-effort handle to the live CoreRuntime (bootstrap wires it as
    ``ctx.live_app``). Needed to rebuild the agent tool list after a hardware
    module toggle. None in standalone mode → the toggle persists but is not live
    until the next start, and the route says exactly that. Mirrors
    routes/agents_control.py:_live_app."""
    return getattr(ctx, "live_app", None) or getattr(ctx, "app", None)


def _knowledge_defaults(ktype: str) -> Any:
    """Best-effort code defaults for one knowledge ktype (for GET merge).

    Lazy-imported so a missing/heavy knowledge module degrades to None rather
    than 500. The frontend uses ``data`` (effective) to render; if defaults are
    unavailable it falls back to the raw override alone."""
    try:
        if ktype == "workflows":
            from mast.knowledge.lookups import get_all_categories

            return get_all_categories()
        if ktype == "fault_diagnosis":
            from mast.knowledge import fault_diagnosis as fd  # type: ignore

            return getattr(fd, "FAULT_CATEGORIES", None)
        if ktype == "experiment_design":
            from mast.knowledge import experiment_design as ed  # type: ignore

            return {
                "measurement_strategies": getattr(ed, "MEASUREMENT_STRATEGIES", {}),
                "workflow_sequences": getattr(ed, "WORKFLOW_SEQUENCES", {}),
                "quality_standards": getattr(ed, "QUALITY_STANDARDS", {}),
                "anomaly_response": getattr(ed, "ANOMALY_RESPONSE", {}),
                "reference_experiments": getattr(ed, "REFERENCE_EXPERIMENTS", {}),
            }
        if ktype == "hardware_profile":
            from mast.knowledge import hardware_profile as hp  # type: ignore

            return {
                "piezo_materials": getattr(hp, "PIEZO_MATERIALS", {}),
                "scanner_types": getattr(hp, "SCANNER_TYPES", {}),
                "temperature_scaling": getattr(hp, "TEMPERATURE_SCALING", {}),
                "hysteresis_creep": getattr(hp, "HYSTERESIS_CREEP", {}),
                "motor_types": getattr(hp, "MOTOR_TYPES", {}),
                "woodpecker_approach": getattr(hp, "WOODPECKER_APPROACH", {}),
            }
        if ktype == "image_databases":
            from mast.knowledge import image_databases as idb  # type: ignore

            return {
                "experimental_datasets": getattr(idb, "EXPERIMENTAL_DATASETS", []),
                "simulated_datasets": getattr(idb, "SIMULATED_DATASETS", []),
                "simulation_tools": getattr(idb, "SIMULATION_TOOLS", []),
                "generative_models": getattr(idb, "GENERATIVE_MODELS", []),
                "processing_libraries": getattr(idb, "PROCESSING_LIBRARIES", []),
                "autonomous_frameworks": getattr(idb, "AUTONOMOUS_FRAMEWORKS", []),
            }
    except Exception as exc:  # any import/shape issue ⇒ no defaults (raw only)
        logger.debug("knowledge defaults unavailable (%s): %s", ktype, exc)
    return None


def _encyclopedia_defaults(section_key: str) -> Any:
    """Best-effort code defaults for one encyclopedia section (for GET merge)."""
    try:
        # NB: gui was renamed to webui in the TS rewrite — the old `mast.gui`
        # import silently failed → empty 百科配置. Also accept the URL section
        # names (intents/hierarchy/verification) AND the dict keys.
        from mast.webui.encyclopedia import (  # type: ignore
            COMPOSITE_HIERARCHY,
            DOMAINS,
            INTENT_MAPPING,
            SKILL_VERIFICATION,
        )

        return {
            "domains": DOMAINS,
            "intents": INTENT_MAPPING,
            "intent_mapping": INTENT_MAPPING,
            "hierarchy": COMPOSITE_HIERARCHY,
            "composite_hierarchy": COMPOSITE_HIERARCHY,
            "verification": SKILL_VERIFICATION,
            "skill_verification": SKILL_VERIFICATION,
        }.get(section_key)
    except Exception as exc:
        logger.debug("encyclopedia defaults unavailable (%s): %s", section_key, exc)
        return None


def _merge_default(default: Any, override: Any) -> Any:
    """Best-effort effective value = default with override applied.

    dict over dict → deep_merge (the core's merge semantics); otherwise the
    override (if present) replaces the default. NO business logic — only a
    presentation merge so GET returns what the editor should show."""
    if override is None:
        return copy.deepcopy(default) if default is not None else None
    if isinstance(default, dict) and isinstance(override, dict):
        try:
            from mast.admin.override_store import deep_merge

            return deep_merge(default, override)
        except Exception:
            merged = copy.deepcopy(default)
            merged.update(override)
            return merged
    return copy.deepcopy(override)


# ── Unified settings write ───────────────────────────────────────────────────
@router.post("/settings", response_model=SettingsWriteResponse)
def write_settings(request: Request, body: SettingsWriteRequest) -> SettingsWriteResponse:
    """Unified UI settings write (degrade-safe).

    Persists the whitelisted subset via ``SettingsStore.update`` (the store drops
    anything outside KNOWN_KEYS and ignores None — whitelist authority stays in
    core), THEN applies model_alias + thinking to the live ``config.llm``
    (use / set_thinking) when a live config is wired. Returns the persisted
    subset + the keys actually pushed to the live config."""
    ctx = request.app.state.ctx
    store = ctx.settings_store
    if store is None:
        return SettingsWriteResponse(ok=False, degraded=True)

    # ── PIN guard on the two capability keys ─────────────────────────────────
    # `advanced_capabilities` grants powers that step around a protection (loading a
    # script into a slot the allow-list vets, quitting Nanonis, a scan config MAST
    # cannot read). `hardware_modules` grants DANGEROUS skills the moment a module is
    # switched on — a laser, an RF amplifier, probes that can collide.
    #
    # Both are written from 高级 (the admin page) and both require the PIN. Everything
    # else in 设置 is unguarded: getting the model or the voice wrong costs quality,
    # not safety.
    #
    # This is NOT a defence against the model — the agent has no HTTP tool and no
    # settings tool; its whole surface is Nanonis skills. It is a defence against a
    # human hand: a mis-click, a second person at the bench, a browser tab left open.
    # Fail-closed: no PIN set ⇒ refused, with an instruction to set one.
    payload = body.model_dump(exclude_none=True)
    guarded = sorted(GUARDED_KEYS & set(payload))
    if guarded:
        ok, reason = verify_pin(body.admin_pin)
        if not ok:
            logger.warning("settings write REFUSED (%s): %s", reason, guarded)
            return SettingsWriteResponse(
                ok=False, degraded=False,
                pin_required=True, pin_reason=reason,
                rebuild_note=reason_text(reason),
            )
    payload.pop("admin_pin", None)   # never persist the PIN, not even by accident

    # ── 结构校验必须在持久化之前 ─────────────────────────────────────────────
    # 这条路径是「先存盘,再 live-apply」。扫描档位表如果结构非法(区间重叠、
    # 没有兜底档、档数超限),存盘会成功而 live-apply 会失败 —— 于是用户用着
    # 出厂表,设置文件里躺着他那张坏表,下次启动 hydrate 同样失败。两边不一致
    # 且在任何界面上都看不出来。所以这里整体拒绝,一个字节都不写。
    if body.scan_policy is not None:
        try:
            from mast.core.scan_policy import PolicyRejected, sanitize as _sp_sanitize
            _sp_sanitize(body.scan_policy)
        except PolicyRejected as exc:
            logger.warning("settings write REFUSED (scan_policy 结构非法): %s", exc)
            return SettingsWriteResponse(
                ok=False, degraded=False,
                rejected={"scan_policy": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001 - 校验器本身坏了不该吞掉写入
            logger.warning("scan_policy 校验异常: %s", exc)
            return SettingsWriteResponse(
                ok=False, degraded=False,
                rejected={"scan_policy": f"档位表校验失败: {exc}"},
            )

    # 自定义 Z 参数组:同样「结构非法就一个字节都不写」。这里的理由更硬一层 ——
    # 这些值将来会被 ApplyZCtrlPreset 直接写进 Z 反馈环,存一张坏表等于给以后
    # 每一次按名应用埋一个坏值。
    if body.zctrl_presets is not None:
        try:
            from mast.core.zctrl_presets import (
                PresetRejected,
                sanitize as _zp_sanitize,
            )
            _zp_sanitize(body.zctrl_presets)
        except PresetRejected as exc:
            logger.warning("settings write REFUSED (zctrl_presets 非法): %s", exc)
            return SettingsWriteResponse(
                ok=False, degraded=False,
                rejected={"zctrl_presets": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("zctrl_presets 校验异常: %s", exc)
            return SettingsWriteResponse(
                ok=False, degraded=False,
                rejected={"zctrl_presets": f"参数组校验失败: {exc}"},
            )

    # Nanonis 端口:同样「非法就一个字节都不写」。这里的失败是**延迟**的 ——
    # 存一个 70000 不会当场报错,``apply_to_config`` 下次开机把它 setattr 进
    # config.nanonis,然后连接失败,而错在哪只有翻设置文件才看得出来。
    bad_ports = {
        k: f"端口必须是 1..65535 的整数（收到 {getattr(body, k)}）"
        for k in ("nanonis_port_main", "nanonis_port_monitor",
                  "nanonis_port_data", "nanonis_port_emergency")
        if getattr(body, k) is not None and not (1 <= int(getattr(body, k)) <= 65535)
    }
    if bad_ports:
        logger.warning("settings write REFUSED (端口越界): %s", sorted(bad_ports))
        return SettingsWriteResponse(ok=False, degraded=False, rejected=bad_ports)

    # 对话预算 / 精炼阈值 / 归档模式:同样「非法就一个字节都不写」。
    #
    # 这一组的坏值**也不会当场报错,而且会被悄悄换成另一个数**:
    #   * ``core/runtime.py:_pos()`` 对 per_run 的 ``v <= 0`` 返回**出厂默认** ——
    #     用户填 0 想「关掉上限」,拿到的是 30,而 30 正是砍断长任务的那个数;
    #   * ``_ingest_copy_mode()`` 对不认识的字符串返回 "copy" —— 填错一个字母,
    #     hardlink 就静静地没生效,盘照样满。
    # 存下去之后这两件事在任何界面上都看不出来(GET 回来的是用户填的那个值,
    # 而生效的是另一个)。所以在这里拒。
    bad_values: dict[str, str] = {}
    for k in ("chat_model_calls_per_run", "chat_tool_calls_per_run"):
        v = getattr(body, k)
        if v is not None and int(v) < 1:
            bad_values[k] = (
                f"单次调用上限必须 ≥ 1（收到 {v}）。0 和负数不是「不限」——"
                f"内核会把它换成出厂默认，你会拿到一个自己没选过的上限。"
                f"要放宽就填一个大数。")
    for k in ("chat_model_calls_per_thread", "chat_tool_calls_per_thread"):
        v = getattr(body, k)
        if v is not None and int(v) < 0:
            bad_values[k] = f"跨会话累计上限不能是负数（收到 {v}）。0 = 不限。"
    if body.tool_refine_min_chars is not None and int(body.tool_refine_min_chars) < 1:
        bad_values["tool_refine_min_chars"] = (
            f"精炼阈值必须 ≥ 1（收到 {body.tool_refine_min_chars}）。"
            f"想整个关掉精炼请用 tool_refine_enabled。")
    if body.ingest_copy_mode is not None and body.ingest_copy_mode not in ("copy", "hardlink"):
        bad_values["ingest_copy_mode"] = (
            f"归档方式只能是 copy 或 hardlink（收到 {body.ingest_copy_mode!r}）。"
            f"hardlink 省盘，但源和实验文件夹必须在同一个卷上。")
    if bad_values:
        logger.warning("settings write REFUSED (取值非法): %s", sorted(bad_values))
        return SettingsWriteResponse(ok=False, degraded=False, rejected=bad_values)

    try:
        persisted = store.update(**payload)
    except Exception as exc:  # never 500 → typed degraded result
        logger.warning("unified settings write failed: %s", exc)
        return SettingsWriteResponse(ok=False, degraded=True)

    # Best-effort live apply (model alias switches provider; thinking sets the
    # budget). A missing live config is NOT degraded — the persist succeeded and
    # is re-applied on next startup via settings_store.apply_to_config.
    applied: list[str] = []
    cfg = _live_config(ctx)
    if cfg is not None and getattr(cfg, "llm", None) is not None:
        if body.model_alias:
            try:
                if hasattr(cfg.llm, "use"):
                    cfg.llm.use(body.model_alias)
                else:
                    cfg.llm.model = body.model_alias
                applied.append("model_alias")
            except Exception as exc:
                logger.debug("live apply model_alias failed: %s", exc)
        if body.thinking and hasattr(cfg.llm, "set_thinking"):
            try:
                cfg.llm.set_thinking(body.thinking)
                applied.append("thinking")
            except Exception as exc:
                logger.debug("live apply thinking failed: %s", exc)

    # Nanonis host + ports: apply to the live config too, not just to disk.
    # ``/api/nanonis/connect`` mutates the live config WITHOUT persisting; this
    # path persisted WITHOUT mutating. Between them an operator could save a new
    # host, watch the session keep using the old one, and only get the new one
    # after a restart — while any automatic reconnect in between silently dialled
    # the old address. Nothing is disconnected here: ``reconnect()`` reads
    # ``config.nanonis`` when it runs, so this only decides where the NEXT connect
    # goes. Same key→attribute table apply_to_config uses at startup.
    if any(getattr(body, k) is not None for k in NANONIS_KEY_TO_ATTR):
        nano = getattr(cfg, "nanonis", None) if cfg is not None else None
        if nano is not None:
            for key, attr in NANONIS_KEY_TO_ATTR.items():
                val = getattr(body, key, None)
                if val is None or val == "":
                    continue
                try:
                    setattr(nano, attr, val)
                    applied.append(key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("live apply %s failed: %s", key, exc)

    # Vision tip-quality thresholds: push into the process-level holder so the
    # next tip assessment uses the new cuts (NO model reload). Independent of the
    # live config — the vision layer reads the holder lock-free, never settings.
    if body.vision_thresholds is not None:
        try:
            from mast.vision.thresholds import set_thresholds
            set_thresholds(body.vision_thresholds)
            applied.append("vision_thresholds")
        except Exception as exc:
            logger.debug("live apply vision_thresholds failed: %s", exc)

    # Per-instrument knobs for the network-free classical tip-quality tools —
    # same live-read-holder pattern as vision_thresholds (no model reload).
    if body.classical_thresholds is not None:
        try:
            from mast.vision.classical_thresholds import set_classical_thresholds
            set_classical_thresholds(body.classical_thresholds)
            applied.append("classical_thresholds")
        except Exception as exc:
            logger.debug("live apply classical_thresholds failed: %s", exc)

    # Tunnelling-current monitor knobs — same live-read holder. Includes
    # cm_enabled, so flipping the switch here starts/stops acquisition on the
    # daemon's next poll without a restart.
    if body.current_monitor is not None:
        try:
            from mast.monitoring.thresholds import set_monitor_thresholds
            set_monitor_thresholds(body.current_monitor)
            applied.append("current_monitor")
        except Exception as exc:
            logger.debug("live apply current_monitor failed: %s", exc)

    # Environment-history recorder knobs — same live-read holder, same
    # whole-dict replace semantics (the UI must POST every eh_* key, not just
    # the one it changed; see frontend/src/lib/knobs.ts). Takes effect on the
    # next environment tick / next current segment, no restart.
    if body.env_history is not None:
        try:
            from mast.envhistory.thresholds import set_env_history_thresholds
            set_env_history_thresholds(body.env_history)
            applied.append("env_history")
        except Exception as exc:
            logger.debug("live apply env_history failed: %s", exc)

    # campaign 指挥线程的旋钮 —— 同一个 live-read holder,**外加一次真的起/停**。
    #
    # 只换 holder 是不够的:``cd_enabled`` 决定的是一条**已经在跑的线程**要不要
    # 继续跑。用户把它关掉、以为线程停了,而它还在驱动仪器 —— 那比没有开关更
    # 危险。所以这里紧接着调 ``apply_settings``:关 ⇒ 停线程;开 ⇒ 有 runtime
    # 句柄就当场起,没有就**如实记一条**「下次启动生效」,不假装已经起来了。
    if body.conduct is not None:
        try:
            from mast.conduct.settings import set_conduct_knobs
            set_conduct_knobs(persisted.get("conduct", body.conduct))
            applied.append("conduct")
            from mast.conduct.service import apply_settings as _apply_conduct
            _apply_conduct(_live_runtime(ctx))
        except Exception as exc:
            logger.debug("live apply conduct failed: %s", exc)

    # Experiment default-parameter preferences : push into the process-level
    # holder so the IC / experiment_design agents see the operator's preferred scan
    # size / speed / setpoint / bias on their NEXT turn (NO graph rebuild — the
    # experiment_prefs middleware live-reads the holder). Like vision_thresholds it
    # is independent of config.llm; the agents never import settings.
    if body.experiment_defaults is not None:
        try:
            from mast.agents._shared.experiment_prefs import set_prefs
            set_prefs(persisted.get("experiment_defaults", body.experiment_defaults))
            applied.append("experiment_defaults")
        except Exception as exc:
            logger.debug("live apply experiment_defaults failed: %s", exc)

    # Instrument profile (装置配置 + 学习标定): push into the process holder so the
    # retract/approach skills + the injecting middleware see the new config on the
    # NEXT turn (NO graph rebuild — holder is live-read). The store already merged
    # any learned calibration; take the persisted value so the config write can't
    # clobber a dI/dV calibration the frontend didn't echo back.
    if body.instrument_profile is not None:
        try:
            from mast.core import instrument_profile as _iprof
            _iprof.set_profile(persisted.get("instrument_profile", body.instrument_profile))
            applied.append("instrument_profile")
        except Exception as exc:
            logger.debug("live apply instrument_profile failed: %s", exc)

    # Coarse-stepper drive ceiling: push into its live-read holder so the very
    # next SetMotorFreqAmp / coarse move sees the new declaration. Reaching this
    # line already means the admin PIN check above passed — coarse_drive is in
    # GUARDED_KEYS. Deliberately NOT merged with the persisted value the way
    # instrument_profile is: there is no runtime-learned half here, the whole
    # thing is the operator's statement, and a partial write should replace it
    # rather than silently keep an old ceiling alive under a new declaration.
    if body.coarse_drive is not None:
        try:
            from mast.core import coarse_drive as _cdrive
            _cdrive.set_declaration(persisted.get("coarse_drive", body.coarse_drive))
            applied.append("coarse_drive")
        except Exception as exc:
            logger.warning("live apply coarse_drive failed: %s", exc)

    # 扫描档位表:推进 live-read holder,下一次 ScanAt / resolver 立刻用新值
    # (无需重建 graph —— 参数是执行时查表,不是构建时冻结的)。结构已在持久化
    # 之前校验过,这里不会因为坏表而失败。
    if body.scan_policy is not None:
        try:
            from mast.core import scan_policy as _spolicy
            _spolicy.set_policy(persisted.get("scan_policy", body.scan_policy))
            applied.append("scan_policy")
        except Exception as exc:
            logger.warning("live apply scan_policy failed: %s", exc)

    # 自定义 Z 参数组:同上,执行时查表,不需要重建 graph。
    if body.zctrl_presets is not None:
        try:
            from mast.core import zctrl_presets as _zpresets
            _zpresets.set_presets(
                persisted.get("zctrl_presets", body.zctrl_presets))
            applied.append("zctrl_presets")
        except Exception as exc:
            logger.warning("live apply zctrl_presets failed: %s", exc)

    # The two capability keys (硬件模块 / 高级能力). Two steps, and the SECOND one
    # is the one that actually matters:
    #   1. push the new set into the process-level holder, and
    #   2. rebuild the orchestrator — because the agent's tool list was FROZEN at
    #      graph-build time. Without (2) the switch persists, the holder updates,
    #      the UI shows a satisfying green toggle, and the agent's tool list is
    #      unchanged until the next restart. That is exactly the trap that once
    #      let a deleted composite skill stay callable.
    # The rebuild can honestly decline (a task is mid-run — rebuilding under it
    # would swap the graph beneath a live run), so we return its note verbatim
    # rather than claiming success.
    rebuild_note = ""
    if body.hardware_modules is not None:
        try:
            from mast.skills.hardware_modules import set_enabled
            set_enabled(persisted.get("hardware_modules", body.hardware_modules))
            applied.append("hardware_modules")
        except Exception as exc:
            logger.warning("live apply hardware_modules failed: %s", exc)

    if body.advanced_capabilities is not None:
        try:
            from mast.skills.advanced_capabilities import (
                set_enabled as set_advanced_enabled,
            )
            set_advanced_enabled(
                persisted.get("advanced_capabilities", body.advanced_capabilities))
            applied.append("advanced_capabilities")
        except Exception as exc:
            logger.warning("live apply advanced_capabilities failed: %s", exc)

    if body.hardware_modules is not None or body.advanced_capabilities is not None:
        rt = _live_runtime(ctx)
        req = getattr(rt, "_request_composite_rebuild", None) if rt is not None else None
        if callable(req):
            try:
                rebuild_note = req() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("hardware_modules tool-list rebuild failed: %s", exc)
                rebuild_note = "（agent 工具表重建失败——请重启生效）"
        else:
            # No live orchestrator (headless API, tests). Persisted + holder set;
            # the next graph build picks it up. Say so instead of implying live.
            rebuild_note = "（未连接运行中的 agent；下次启动生效）"

    # 启动时读一次的那几个：写成功了，但这一轮还在用老值。说出来。
    read_once = [v for k, v in _READ_ONCE_AT_STARTUP.items() if k in payload]
    if read_once:
        rebuild_note += f"（{'、'.join(read_once)}下次启动生效）"

    return SettingsWriteResponse(ok=True, persisted=persisted, applied=applied,
                                 degraded=False, rebuild_note=rebuild_note)


# ── Per-agent model / thinking override ───────────────────────────────────────
@router.post(
    "/agents/{agent_id}/model-override", response_model=AgentModelOverrideResponse
)
def set_agent_model_override(
    request: Request, agent_id: str, body: AgentModelOverrideRequest
) -> AgentModelOverrideResponse:
    """Set one agent's model / thinking override (degrade-safe).

    Relays to ``ConfigOverrideRegistry.set_agent_override`` — the core merges
    only the provided fields into agent_overrides.json + persists; the running
    agents pick it up via the registry (read back by GET /agents/snapshot)."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        return AgentModelOverrideResponse(ok=False, agent_id=agent_id, degraded=True)
    try:
        entry = reg.set_agent_override(
            agent_id, model=body.model, thinking=body.thinking
        )
        return AgentModelOverrideResponse(
            ok=True, agent_id=agent_id, override=dict(entry or {}), degraded=False
        )
    except Exception as exc:
        logger.warning("agent model override failed (%s): %s", agent_id, exc)
        return AgentModelOverrideResponse(ok=False, agent_id=agent_id, degraded=True)


# ── Fine-grained knowledge edit ───────────────────────────────────────────────
@router.get("/admin/knowledge/{ktype}", response_model=SectionConfigResponse)
def get_knowledge(request: Request, ktype: str) -> SectionConfigResponse:
    """Read the effective knowledge config for one ktype (defaults + override)."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        # No registry → still surface code defaults so the editor renders.
        default = _knowledge_defaults(ktype)
        return SectionConfigResponse(
            key=ktype, data=default, override=None, has_override=False, degraded=True
        )
    try:
        override = reg.get_knowledge_override(ktype)
        default = _knowledge_defaults(ktype)
        return SectionConfigResponse(
            key=ktype,
            data=_merge_default(default, override),
            override=override,
            has_override=bool(override),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("knowledge read failed (%s): %s", ktype, exc)
        return SectionConfigResponse(key=ktype, degraded=True)


@router.post("/admin/knowledge/{ktype}", response_model=SectionConfigWriteResponse)
def write_knowledge(
    request: Request, ktype: str, body: SectionConfigWriteRequest
) -> SectionConfigWriteResponse:
    """Persist one ktype's knowledge override + hot-reload (degrade-safe).

    Relays to the registry: the per-ktype payload is merged into
    knowledge_overrides.json (keyed by ktype) and saved with an in-process
    reload. Empty payload ⇒ that ktype's override is dropped (reset to default)."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        return SectionConfigWriteResponse(ok=False, key=ktype, degraded=True)
    try:
        from mast.admin.override_store import KNOWLEDGE_OVERRIDES

        data = reg.get_raw(KNOWLEDGE_OVERRIDES) or {}
        if body.data in (None, {}, []):
            data.pop(ktype, None)
        else:
            data[ktype] = body.data
        fired = reg.save_or_delete_and_reload(KNOWLEDGE_OVERRIDES, data)
        saved = reg.get_knowledge_override(ktype)
        return SectionConfigWriteResponse(
            ok=True, key=ktype, reloaded=bool(fired), override=saved, degraded=False
        )
    except Exception as exc:
        logger.warning("knowledge write failed (%s): %s", ktype, exc)
        return SectionConfigWriteResponse(ok=False, key=ktype, degraded=True)


# ── Per-skill guidance edit ───────────────────────────────────────────────────
@router.get("/admin/guidance/{skill}", response_model=SectionConfigResponse)
def get_guidance(request: Request, skill: str) -> SectionConfigResponse:
    """Read one skill's guidance override (skill_extra sub-key)."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        return SectionConfigResponse(key=skill, degraded=True)
    try:
        override = reg.get_guidance_override(skill)
        # Effective view: code SKILL_EXTRA for this skill, override applied.
        default = None
        try:
            from mast.knowledge.skill_guidance import SKILL_EXTRA  # type: ignore

            default = SKILL_EXTRA.get(skill)
        except Exception:
            default = None
        return SectionConfigResponse(
            key=skill,
            data=_merge_default(default, override),
            override=override,
            has_override=bool(override),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("guidance read failed (%s): %s", skill, exc)
        return SectionConfigResponse(key=skill, degraded=True)


@router.post("/admin/guidance/{skill}", response_model=SectionConfigWriteResponse)
def write_guidance(
    request: Request, skill: str, body: SectionConfigWriteRequest
) -> SectionConfigWriteResponse:
    """Persist one skill's guidance override + hot-reload (degrade-safe).

    Relays to the registry: the payload is merged into guidance_overrides.json's
    ``skill_extra`` map under this skill and saved with an in-process reload.
    Empty payload ⇒ that skill's guidance override is dropped."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        return SectionConfigWriteResponse(ok=False, key=skill, degraded=True)
    try:
        from mast.admin.override_store import GUIDANCE_OVERRIDES

        data = reg.get_raw(GUIDANCE_OVERRIDES) or {}
        extra = dict(data.get("skill_extra") or {})
        if body.data in (None, {}, []):
            extra.pop(skill, None)
        else:
            extra[skill] = body.data
        if extra:
            data["skill_extra"] = extra
        else:
            data.pop("skill_extra", None)
        fired = reg.save_or_delete_and_reload(GUIDANCE_OVERRIDES, data)
        saved = reg.get_guidance_override(skill)
        return SectionConfigWriteResponse(
            ok=True, key=skill, reloaded=bool(fired), override=saved, degraded=False
        )
    except Exception as exc:
        logger.warning("guidance write failed (%s): %s", skill, exc)
        return SectionConfigWriteResponse(ok=False, key=skill, degraded=True)


# ── Encyclopedia config edit ──────────────────────────────────────────────────
@router.get("/encyclopedia/config/{section}", response_model=SectionConfigResponse)
def get_encyclopedia_config(request: Request, section: str) -> SectionConfigResponse:
    """Read one encyclopedia section's effective config (defaults + override).

    Sections: domains / intents / hierarchy / verification (mapped to the store's
    canonical keys)."""
    ctx = request.app.state.ctx
    section_key = _ENCYCLOPEDIA_SECTIONS.get(section)
    if section_key is None:
        return SectionConfigResponse(key=section, degraded=True)
    reg = _override_registry(ctx)
    if reg is None:
        default = _encyclopedia_defaults(section_key)
        return SectionConfigResponse(
            key=section, data=default, override=None, has_override=False, degraded=True
        )
    try:
        all_ovr = reg.get_encyclopedia_overrides() or {}
        override = all_ovr.get(section_key)
        default = _encyclopedia_defaults(section_key)
        return SectionConfigResponse(
            key=section,
            data=_merge_default(default, override),
            override=override,
            has_override=section_key in all_ovr,
            degraded=False,
        )
    except Exception as exc:
        logger.warning("encyclopedia read failed (%s): %s", section, exc)
        return SectionConfigResponse(key=section, degraded=True)


@router.post("/encyclopedia/config/{section}", response_model=SectionConfigWriteResponse)
def write_encyclopedia_config(
    request: Request, section: str, body: SectionConfigWriteRequest
) -> SectionConfigWriteResponse:
    """Persist one encyclopedia section's override + hot-reload (degrade-safe).

    Relays to the registry: the section payload is merged into
    encyclopedia_overrides.json under its canonical key and saved with an
    in-process reload. Empty payload ⇒ that section is dropped (reset)."""
    ctx = request.app.state.ctx
    section_key = _ENCYCLOPEDIA_SECTIONS.get(section)
    if section_key is None:
        return SectionConfigWriteResponse(ok=False, key=section, degraded=True)
    reg = _override_registry(ctx)
    if reg is None:
        return SectionConfigWriteResponse(ok=False, key=section, degraded=True)
    try:
        from mast.admin.override_store import ENCYCLOPEDIA_OVERRIDES

        data = reg.get_raw(ENCYCLOPEDIA_OVERRIDES) or {}
        if body.data in (None, {}, []):
            data.pop(section_key, None)
        else:
            data[section_key] = body.data
        fired = reg.save_or_delete_and_reload(ENCYCLOPEDIA_OVERRIDES, data)
        saved = (reg.get_encyclopedia_overrides() or {}).get(section_key)
        return SectionConfigWriteResponse(
            ok=True, key=section, reloaded=bool(fired), override=saved, degraded=False
        )
    except Exception as exc:
        logger.warning("encyclopedia write failed (%s): %s", section, exc)
        return SectionConfigWriteResponse(ok=False, key=section, degraded=True)


# ── Quick prompts ─────────────────────────────────────────────────────────────
@router.get("/admin/quick-prompts", response_model=QuickPromptsResponse)
def get_quick_prompts(request: Request) -> QuickPromptsResponse:
    """Read the persisted quick-prompt list (degrade-safe)."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        return QuickPromptsResponse(degraded=True)
    try:
        prompts = list(reg.get_quick_prompts() or [])
        return QuickPromptsResponse(
            prompts=prompts,
            count=len(prompts),
            has_override=bool(prompts),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("quick-prompts read failed: %s", exc)
        return QuickPromptsResponse(degraded=True)


@router.post("/admin/quick-prompts", response_model=QuickPromptsWriteResponse)
def write_quick_prompts(
    request: Request, body: QuickPromptsWriteRequest
) -> QuickPromptsWriteResponse:
    """Persist the quick-prompt list + hot-reload (degrade-safe).

    Relays to the registry: non-empty ⇒ save_and_reload(QUICK_PROMPTS,
    {"prompts": [...]}); empty ⇒ save_or_delete_and_reload(QUICK_PROMPTS, {})
    (reset to code defaults). Mirrors the old encyclopedia_tab quick-prompts
    sub-tab save / 恢复默认 semantics."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        return QuickPromptsWriteResponse(ok=False, degraded=True)
    try:
        from mast.admin.override_store import QUICK_PROMPTS

        prompts = list(body.prompts or [])
        if prompts:
            fired = reg.save_and_reload(QUICK_PROMPTS, {"prompts": prompts})
        else:
            fired = reg.save_or_delete_and_reload(QUICK_PROMPTS, {})
        saved = list(reg.get_quick_prompts() or [])
        return QuickPromptsWriteResponse(
            ok=True, reloaded=bool(fired), prompts=saved, degraded=False
        )
    except Exception as exc:
        logger.warning("quick-prompts write failed: %s", exc)
        return QuickPromptsWriteResponse(ok=False, degraded=True)
