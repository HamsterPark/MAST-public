"""Gradio-free construction of the live core for the API service (cutover seam).

Builds the full MAST core via ``mast.core.runtime.CoreRuntime`` (extracted from
the old MASTApp, NO Gradio) and wires every singleton into an ``AppContext`` so
the FastAPI service serves real data — chat, hardware, vision, records, agents —
exactly like the validated cutover harness. Used by ``python -m mast.api`` in
live mode and by the launcher after Gradio is gone.

Best-effort per subsystem: a failure wires nothing for it (that endpoint stays
degraded) but never aborts the whole boot.
"""

from __future__ import annotations

import logging
from typing import Optional

from mast.api.context import AppContext

logger = logging.getLogger(__name__)


def build_live_context(config=None) -> AppContext:
    cfg = config
    if cfg is None:
        try:
            from mast.config import MASTConfig

            cfg = MASTConfig()
        except Exception as exc:  # pragma: no cover
            logger.warning("MASTConfig() failed: %s", exc)
            cfg = None

    try:
        from mast.core.runtime import CoreRuntime, _ensure_buffer_for_gui
    except Exception as exc:
        logger.error("CoreRuntime import failed: %s — degraded API only", exc)
        return AppContext()

    rt = CoreRuntime(cfg)
    rt.setup()

    # lazy realtime buffer (may be None until its async start() completes)
    buf = None
    try:
        buf = _ensure_buffer_for_gui(rt)
        if buf is None:
            import time as _time

            for _ in range(12):
                _time.sleep(0.25)
                buf = getattr(rt, "_buffer", None)
                if buf is not None:
                    break
    except Exception as exc:
        logger.debug("buffer start: %s", exc)

    # catalog + agent-tool warming (CoreRuntime.setup wires composite_panel's
    # registry, not builder_api's — that was a build_ui() step; do it here).
    try:
        from mast.webui.builder_api import invalidate_catalog, set_live_registry

        set_live_registry(getattr(rt, "_registry", None))
        invalidate_catalog()
    except Exception as exc:
        logger.debug("catalog wiring: %s", exc)
    try:
        from mast.webui.agents_api import (
            invalidate_agent_tools,
            set_live_registry as set_agent_tools_registry,
            warm_agent_tools,
        )

        # Before this line agents_api built its OWN registry: /agents/tools
        # listed a catalogue no agent ever held (no composites, no custom
        # skills, no bridged agent @tools, no mast.skills.paper). Wire the live
        # one FIRST, then invalidate, then warm — warming before wiring would
        # cache the degraded listing for the life of the process.
        set_agent_tools_registry(getattr(rt, "_registry", None))
        invalidate_agent_tools()
        warm_agent_tools()
    except Exception as exc:
        logger.debug("agent tools warm: %s", exc)

    user_root: Optional[str] = None
    try:
        user_root = str(getattr(getattr(cfg, "paths", None), "project_root", None) or "") or None
    except Exception:
        user_root = None

    ctx = AppContext(user_root=user_root)
    ctx.wire(
        skill_registry=getattr(rt, "_registry", None),
        experiment_storage=getattr(rt, "_storage", None),
        buffer_service=buf or getattr(rt, "_buffer", None),
        settings_store=getattr(rt, "_settings", None),
    )
    # plain attrs the route slices read via getattr(ctx, ...)
    ctx.conversation_engine = getattr(rt, "_conv_engine", None)
    ctx.conversation_store = getattr(rt, "_conv_store", None)
    ctx.connection_pool = getattr(rt, "_pool", None)
    # 2026-08-02 实机抓到：这一行此前不存在。``signals._execution_context()`` 要
    # pool + state + registry 三者齐全才肯建 ExecutionContext，而 ``ctx.state``
    # **全仓从来没有任何地方赋过值** —— 于是它恒为 None，凡是走那条路的端点永远
    # 退化：
    #   /api/experimental/signals   → 返回硬编码的默认通道表（16 条，看着像真的）
    #                                 且 timebases=[]，detail 还写「nanonis not
    #                                 wired」——而 Nanonis 明明连着
    #   /api/coarse-map/selfcheck   → 恒报「内核未就绪(pool/state/registry 未接线)」
    #                                 （粗动子系统的真机调试自检，正是为验收写的）
    ctx.state = getattr(rt, "_state", None)
    ctx.environment_monitor = getattr(rt, "_monitor", None)
    # Nanonis-backed environment sensors are built by the runtime (they need a
    # live pool) — the live-rescan path in routes/admin.py must be able to
    # rebuild them too, or one rescan silently drops them from the monitor.
    ctx.env_extra_sensors = getattr(rt, "_build_nanonis_env_sensors", None)
    ctx.env_history = getattr(rt, "_env_history", None)
    ctx.quickask = getattr(rt, "_quickask", None)  # 查询助手 single-turn helper
    # Agents-UI live overlay hooks (active task / holds / handoffs / interrupts /
    # artifacts) read by routes/agents_topology.py + artifacts_edit.py.
    ctx.agents_snapshot = getattr(rt, "agents_snapshot", None)
    ctx.agents_interrupts = getattr(rt, "agents_interrupts", None)
    ctx.agents_artifacts = getattr(rt, "agents_artifacts", None)
    ctx.config = cfg
    cog = getattr(rt, "_cognition", None)
    ctx.memory_store = getattr(cog, "store", None) if cog else None
    # 整个认知上下文（带语义索引的 remember / recall）。只挂 store 的话，外部 agent
    # 网关写进来的笔记进不了向量索引，内部 agent 的语义召回就找不到它们。
    ctx.cognition = cog
    try:
        from mast.admin.override_store import ConfigOverrideRegistry

        ctx.override_registry = ConfigOverrideRegistry.get()
    except Exception as exc:
        logger.debug("override registry: %s", exc)
    # KNOWN_ISSUES §1.1 — until 2026-08-03 ``register_reload_hook`` had ZERO
    # production subscribers, so every saved admin override reached nothing at
    # all until the next restart while the write endpoint reported success.
    # This is the one place that subscribes: the live SafetyGuard re-merges, the
    # private-chat graphs drop their cache, and the group orchestrator rebuilds
    # in the background (the agent path bakes the envelope into each tool's
    # pydantic schema, so it needs a rebuild, not a re-merge).
    try:
        from mast.admin.reload_wiring import wire_override_reload

        wire_override_reload(rt, getattr(ctx, "override_registry", None))
    except Exception as exc:
        logger.warning("override reload wiring failed: %s", exc)
    # KNOWN_ISSUES §1.2 — the factory safety envelope corresponds to no real
    # machine (the rig: XY half-range 1.7× LARGER than configured, Z travel
    # 4.2× SMALLER). Read the real travel once at boot and say so out loud.
    # REPORT ONLY: it never widens anything — "an agent that can widen its own
    # limits has, in the strict sense, no limits".
    try:
        from mast.core.envelope_reconcile import reconcile_at_boot

        reconcile_at_boot(rt)
    except Exception as exc:
        logger.debug("safety envelope reconcile: %s", exc)
    # 上面那条只报告。这条**会改包络** —— 但只按用户自己登记的仪器常数收紧，
    # 而且只有一个方向。起因：出厂 setpoint_max_a 是 100 nA，真机的前放是
    # ±10 nA，于是「设定点高于量程 → 电流恒小于设定点 → 反馈把 Z 一路推向样品」
    # 这条撞针路径一直敞着，且不报任何错（这是刻意的取舍）。
    # 放在对账之后，是为了让日志里「测到什么」先于「据此改了什么」。
    try:
        from mast.core.safety import report_instrument_clamps

        report_instrument_clamps()
    except Exception as exc:
        logger.debug("instrument clamp report: %s", exc)
    # KNOWN_ISSUES §3.1 — an upgrade overwrites artifacts/literature_index with
    # the factory copy, and the agent's own fetched papers live in the same
    # files. mast2_setup.iss snapshots the directory before overwriting; this
    # merges the snapshot back in. Background: only a machine that just upgraded
    # has a snapshot, and merging a 205 MB index must not sit on the boot path.
    try:
        from mast.knowledge.index_merge import merge_pending_snapshots_async

        merge_pending_snapshots_async()
    except Exception as exc:
        logger.debug("literature index merge: %s", exc)
    try:
        from mast.knowledge.libraries import get_registry as _lib_registry

        ctx.library_registry = _lib_registry()
        ctx.literature_wired = True
    except Exception as exc:
        logger.debug("library registry: %s", exc)
    ctx.live_app = rt
    ctx.app = rt
    # 同一个对象的第三个别名。``routes/scope.py`` 的 ``_runtime()`` 找的是
    # ``app.state.runtime`` 或 ``ctx.runtime``，两个此前**全仓都没有人赋过值** ——
    # 于是它恒返回 None，`/api/scope/folder-health` 永远报
    # `folder_path=null / exists=false / samples=0 / ingest.enabled=false`，
    # 哪怕实验文件夹在磁盘上结构完整（2026-08-02 实机对照确认）；
    # `/api/scope/nanonis-dir` 的读写两个端点同样退化。
    #
    # 三个别名不是好设计（live_app / app / runtime 指同一个 CoreRuntime），但统一
    # 成一个名字要改十几处 route，风险比收益大。真正的教训是下面那条守卫测试：
    # **路由读的名字必须有人挂**，别再靠人肉对齐。
    ctx.runtime = rt
    logger.info("live context built via CoreRuntime")
    return ctx


__all__ = ["build_live_context"]
