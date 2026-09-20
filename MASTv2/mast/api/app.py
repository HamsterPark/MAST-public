"""FastAPI application factory for the MAST typed API.

Phase 2: read-only endpoints, runs standalone next to the live Gradio app.
Phase 3+: ``wire(...)`` shares the live core singletons; WS/SSE + write
endpoints are added; eventually (Phase 5) this app also serves the built TS SPA
and the Gradio app is deleted.

Build the app via the factory so config (CORS origins, wiring) is explicit and
so ``uvicorn --factory`` / tests can construct fresh instances.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mast.api import ws
from mast.api.context import AppContext, set_context
from mast.api.routes import (
    admin,
    agents,
    agents_control,
    agents_topology,
    artifacts_edit,
    builder,
    conducts,
    chat_narration,
    chat_stream,
    chat_voice,
    codex_reference,
    cognition,
    cognition_transcript,
    config_models,
    diagnostics,
    documents,
    env_history,
    environment_readings,
    experiments,
    gallery,
    gallery_figures,
    guidance_ext,
    health,
    instrument_init,
    literature,
    literature_cognition,
    literature_ext2,
    literature_scope,
    monitoring,
    prompts,
    optics,
    orchestrator,
    qa,
    records,
    records_export,
    remote_access,
    scope,
    safety,
    settings,
    settings_admin_write,
    signals,
    skill_market,
    skill_overlay,
    skills,
    skill_exec,
    skills_ext,
    skills_meta,
    tips,
    usage,
    vision,
    vision_hardware,
    voice_ws,
    wishlist,
)
from mast.api.version import get_version

logger = logging.getLogger(__name__)

# Vite dev server origins (frontend `npm run dev`). In production the SPA is
# served same-origin from this app, so CORS is irrelevant there.
_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _add_basic_auth(app: FastAPI, username: str, password: str) -> None:
    """HTTP Basic auth for LAN binding (replaces Gradio's plaintext basic auth).

    NOTE: Basic auth is only safe over TLS — run_service warns + should supply
    ssl_certfile/keyfile when binding non-loopback (the LAN-TLS P0)."""
    import base64
    import secrets

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    expected = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()

    class _BasicAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            hdr = request.headers.get("authorization", "")
            if not (hdr and secrets.compare_digest(hdr, expected)):
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="MAST"'},
                    content="Unauthorized",
                )
            return await call_next(request)

    app.add_middleware(_BasicAuth)


def create_app(
    *,
    context: Optional[AppContext] = None,
    dev_cors: bool = True,
    auth: Optional[tuple[str, str]] = None,
) -> FastAPI:
    ctx = context or AppContext()
    # Wire the chat-abort hook the agents-slice /chat/abort endpoint calls
    # (per-agent threading.Event registry lives in chat_stream).
    ctx.chat_abort = chat_stream.chat_abort_hook
    set_context(ctx)

    app = FastAPI(
        title="MAST API",
        version=get_version(),
        summary="Typed service seam between the TS frontend and the Python core (TS-rewrite Phase 2).",
    )
    app.state.ctx = ctx

    if dev_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_DEV_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if auth is not None:
        _add_basic_auth(app, auth[0], auth[1])

    # All /api routes. Order matters where paths overlap: exact-path routers
    # (skills /skills/catalog) MUST precede path-param routers (skills_ext
    # /skills/{name}) so "catalog" is not captured as a {name}.
    for module in (
        health,
        config_models,
        settings,
        safety,
        skills,
        skills_ext,
        scope,        # 必须在 experiments 之前：/experiments/current
                      # 等字面量段要先于 records 的 /{experiment_id} 匹配
        tips,         # 独立前缀 /tips，与别的路由器不重叠；模块**内部**的
                      # /tips/current 声明在 /tips/{tip_id} 之前
        experiments,
        records,
        agents,
        chat_stream,
        admin,
        cognition,
        literature,
        wishlist,
        vision,
        # ── parity rebuild Wave A (additive endpoints) ──
        settings_admin_write,
        agents_topology,
        records_export,
        # ── 数据图库 (预处理 → 展示 → 人工筛选; docs/v2/design/data_gallery.md) ──
        # 独立前缀 /gallery，与既有路由都不重叠；模块内没有 /{param} 段会去捕获
        # 字面量（本仓踩过两次路由遮蔽）。GET 一律只读，构建在后台线程里跑。
        gallery,
        # 出图（标记帧对比 / 网格逐层 / 拉线谱 / 拼接 / 旋转系列）。前缀 /gallery/figures，
        # 单独一个模块以免碰图库本体的契约；GET 里只有 /preview/… 会写（且只在源文件存在时）。
        gallery_figures,
        chat_voice,
        vision_hardware,
        literature_cognition,
        # ── parity rebuild Wave C (residual endpoints) ──
        skills_meta,
        # ── parity rebuild Wave D (QA + builder write) ──
        qa,
        builder,
        # ── parity rebuild Wave E1 (remaining flagged endpoints) ──
        literature_ext2,
        codex_reference,
        agents_control,
        guidance_ext,
        # ── parity rebuild Wave F1 (properly-built never-done interfaces) ──
        orchestrator,
        artifacts_edit,
        cognition_transcript,
        # ── live right-rail ENVIRONMENT readings relay ──
        environment_readings,
        # ── 信号捕获 + FFT redesign (实验性功能 tab) ──
        signals,
        # 技能覆盖层（改 skill 不发版）。路径刻意用 /skill-overlay
        # 而不是 /skills/overlay —— 后者会被 skills_ext 的
        # /skills/{name} 捕获成 name="overlay"，返回 200 加一个
        # 错的 handler（本仓踩过两次路由遮蔽）。
        skill_overlay,
        # 技能市场 + 订阅列表。前缀 /skill-market 同样是为了躲开
        # skills_ext 的 /skills/{name}（"market" 会被当成一个技能名捕获）。
        # 订阅是**装载面**偏好，不是权限：未订阅的技能仍能被手动执行、被
        # composite 子步与 conduct 的 ctx.run 调到（见模块 docstring）。
        skill_market,
        # ── 光学台 (TERS/THz 位移台手动控制 + 泵浦探测空跑) ──
        optics,
        # ── 报告/评审浏览 (autonomous-run outputs; ) ──
        documents,
        # ── 拒绝台账 (why nothing happened; ) ──
        diagnostics,
        # ── 跨网远程访问就绪状态 (Tailscale; 设置 → 远程访问) ──
        remote_access,
        # ── 用量·花销 (API 花销账本; 独立导航页) ──
        usage,
        # ── 电流监控 (隧道电流分段采集/特征/告警/语料; 独立导航页) ──
        monitoring,
        prompts,
        # ── 环境历史 (温度/真空/液氦/磁场/隧道电流的统计桶 + 噪声谱快照) ──
        # 它同时提供 /experiments/{id}/environment。三段路径，与 records 的
        # 两段 /{experiment_id} 不在同一层，注册顺序不影响匹配。
        env_history,
        # ── 实验专属文献库 (一实验一库; ensure + copy 替代跨实验共享) ──
        literature_scope,
        # ── 仪器 chat 旁白 (长任务跑的时候持续说给用户听; 只读) ──
        # 字面前缀 /chat/*，完全不碰 /agents/{agent_id}/* 的命名空间 ——
        # 路由遮蔽在本仓踩过两次，两次都是 200 + 错的 handler（见模块 docstring）。
        chat_narration,
        # ── 新仪器初始化 (「装到新机器上还差哪些数」的那份清单) ──
        # 独立前缀 /instrument-init，与既有路由都不重叠。它**不新建存储**：
        # 写入转交 settings_admin_write / admin.write_override，本模块只加
        # 「回读比对」那一层 (fixes/2026-08-03 第六点五:比对不能交给会犯错的
        # 那一方 —— 这里连模型都不在链路上)。
        instrument_init,
        # ── 技能直调 (POST /api/skills/{name}/execute) ──
        # 前缀落在既有的 /skills/* 下,但路径是 {name}/execute,与
        # /skills/{name} 不冲突(方法也不同:POST vs GET)。
        # 它复用 ExecutionContext —— 安全闸门/仪器仲裁/中止/状态回写全保留,
        # 唯一绕过的是「模型决定要不要做」。
        skill_exec,
        # ── 多天 conduct 指挥层 (用户面板的后端;引擎默认关) ──
        # 独立前缀 /conducts,与既有路由都不重叠。模块**内部** /conducts/templates
        # 声明在 /conducts/{conduct_id} 之前 —— 否则 "templates" 会被当成一个
        # conduct_id 捕获(本仓踩过两次路由遮蔽,两次都是 200 + 错的 handler)。
        # approve 端点由**自主度策略**裁决谁能批(attended=仅人 /
        # supervised=agent 可批+撤销窗 / autonomous=即批即跑),见
        # mast/conduct/autonomy.py —— 把关在服务端,不靠「不给工具」。
        conducts,
    ):
        app.include_router(module.router, prefix="/api")

    # 旧路径 /api/campaign/* —— 改名后的一版兼容层,handler 与上面同一批。
    app.include_router(conducts.legacy_router, prefix="/api")

    # Realtime channels use full paths (/ws/*, /sse/*, /api/buffer/*) — no prefix.
    app.include_router(ws.router)
    app.include_router(voice_ws.router)  # /ws/voice — full-duplex voice

    # ── 外部 agent 网关 (/api/ext/v1；docs/v2/design/external_agent_gateway.md) ──
    # 独立的子应用：自己的 JSON 错误体与 OpenAPI（/api/ext/v1/openapi.json），主
    # openapi 与前端 schema.d.ts 零改动；上面的认证中间件对它照样生效。必须挂在
    # 下面的 SPA 兜底路由之前 —— 之后注册的 GET 永远匹配不到。
    # 某个构建可以整块不带 mast.api.ext：那时只记一行 WARNING，其余照常。只吞
    # **它自己**缺席的 ModuleNotFoundError；包里真有 bug 要大声报出来，不许静默消失。
    try:
        from mast.api.ext import create_ext_app
    except ModuleNotFoundError as exc:
        if not str(getattr(exc, "name", "") or "").startswith("mast.api.ext"):
            raise
        logger.warning("external-agent gateway not in this build (%s)", exc)
    else:
        try:
            app.mount("/api/ext/v1", create_ext_app(ctx))
        except Exception as exc:  # noqa: BLE001 — 网关坏了不许拖垮整个服务
            logger.error("external-agent gateway failed to mount: %s", exc, exc_info=True)

    # Serve the built TS SPA (production). Registered LAST so explicit /api + /ws
    # routes win. Static assets are served from /assets; EVERY other GET falls back
    # to index.html so client-side routes (/chat, /skills, …) deep-link + refresh
    # correctly (Starlette's StaticFiles html=True does NOT do SPA fallback for
    # arbitrary paths — it 404s, which broke deep links). In dev the Vite server
    # serves the SPA and proxies /api here instead.
    try:
        import sys
        from pathlib import Path

        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        from mast._runtime_paths import project_root

        candidates = [Path(project_root()) / "frontend" / "dist"]
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "frontend" / "dist")
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).parent / "frontend" / "dist")
        dist = next((d for d in candidates if d.is_dir()), None)
        if dist is not None:
            _index = dist / "index.html"
            assets = dist / "assets"
            if assets.is_dir():
                app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

            @app.get("/{full_path:path}", include_in_schema=False)
            async def _spa(full_path: str):
                # /api/* unmatched → real 404 (not the SPA shell).
                if full_path.startswith(("api/", "ws/", "sse/")):
                    return FileResponse(_index, status_code=404)
                cand = dist / full_path
                if full_path and cand.is_file():
                    return FileResponse(cand)  # favicon, etc.
                return FileResponse(_index)  # SPA client route

            logger.info("SPA served from %s (assets + index fallback)", dist)
        else:
            logger.info("SPA dist not found (dev: run `npm --prefix frontend run build`); tried %s",
                        [str(c) for c in candidates])
    except Exception as exc:  # pragma: no cover - static mount is best-effort
        logger.warning("SPA static mount skipped: %s", exc)

    logger.info("MAST API app created (version=%s)", get_version())
    return app


__all__ = ["create_app"]
