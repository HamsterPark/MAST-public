"""外部面子应用的工厂。``mast.api.app.create_app`` 把它挂在 ``/api/ext/v1``。

为什么是子应用而不是往主 app 里 ``include_router``：

* 自己的错误体 —— 一律 ``{"error": code, "detail": ...}``，未知路径回 JSON 404
  （主 app 对未匹配的 ``/api/*`` 回的是 SPA 的 HTML，外部 agent 会把它读成「成功」）；
* 自己的 OpenAPI（``/api/ext/v1/openapi.json``、``/docs``）—— 主 ``frontend/openapi.json``
  与前端 ``schema.d.ts`` 零改动；
* 父应用的认证中间件对挂载的子应用照样生效（中间件包的是整个 ASGI 应用）。
"""

from __future__ import annotations

import ipaddress
import logging
import threading
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mast.api import direct_exec
from mast.api.ext.common import API_VERSION, ExtError
from mast.api.ext.jobs import JobManager

logger = logging.getLogger(__name__)

_DEFAULT_JM: JobManager | None = None
_DEFAULT_JM_LOCK = threading.Lock()


def default_job_manager() -> JobManager:
    """进程级的作业管理器 —— 主 app 被重建（热重载、测试）时作业表不丢。"""
    global _DEFAULT_JM
    with _DEFAULT_JM_LOCK:
        if _DEFAULT_JM is None:
            _DEFAULT_JM = JobManager()
        return _DEFAULT_JM


_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _host_part(netloc: str) -> str:
    """``[::1]:7862`` → ``::1``；``127.0.0.1:7862`` → ``127.0.0.1``；``rig`` → ``rig``。"""
    n = (netloc or "").strip().lower()
    if n.startswith("["):
        return n[1:n.find("]")] if "]" in n else n
    return n.rsplit(":", 1)[0] if n.count(":") == 1 else n


def _is_ip_or_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def browser_refusal(request: Request) -> tuple[int, str, str] | None:
    """外部面只给程序化客户端（MCP server、脚本、curl）用。浏览器发来的请求要过三道判据，
    否则一个网页就能在无认证的回环部署上驱动仪器（安全审计：一个不带请求体的 no-cors POST
    曾经就能触发急停）：

    * 带 ``Origin`` / ``Sec-Fetch-*`` 头的就是浏览器发的（程序化客户端不带这些头）；
    * 浏览器请求的 Host 必须是回环名或 IP 字面量 —— 指向本机的陌生**主机名**是 DNS 重绑定的形状；
    * ``Sec-Fetch-Site`` 只认 ``same-origin`` / ``none``；``Origin`` 必须就是本服务自己。

    另外，所有写方法必须带 ``application/json``：不依赖某个版本 FastAPI 的 ``strict_content_type``
    默认值（旧版把没有 Content-Type 的请求体也当 JSON 解析，那正是绕开 CORS 预检的办法）。
    返回 ``(状态码, 错误码, 说明)`` 或 ``None``（放行）。
    """
    h = request.headers
    if request.method in _WRITE_METHODS:
        ctype = (h.get("content-type") or "").split(";", 1)[0].strip().lower()
        if not (ctype == "application/json" or (ctype.startswith("application/")
                                                 and ctype.endswith("+json"))):
            return (415, "unsupported_media_type",
                    "写请求必须带 Content-Type: application/json 与一个 JSON 请求体（没有内容就发 {}）")
    browserish = "origin" in h or any(k.lower().startswith("sec-fetch-") for k in h.keys())
    if not browserish:
        return None
    host = h.get("host") or ""
    if not _is_ip_or_loopback(_host_part(host)):
        return (403, "cross_origin_refused",
                "浏览器经主机名访问外部面被拒（DNS 重绑定的形状）；用 IP 地址或 127.0.0.1")
    site = (h.get("sec-fetch-site") or "").strip().lower()
    if site and site not in ("same-origin", "none"):
        return (403, "cross_origin_refused", f"跨站请求被拒（Sec-Fetch-Site: {site}）")
    origin = h.get("origin")
    if origin is not None:
        o = urlsplit(origin.strip())
        if o.scheme != request.url.scheme or o.netloc.lower() != host.strip().lower():
            return (403, "cross_origin_refused", f"跨站请求被拒（Origin: {origin[:120]}）")
    return None


def create_ext_app(ctx: Any, *, job_manager: JobManager | None = None) -> FastAPI:
    app = FastAPI(
        title="MAST external-agent API",
        version=API_VERSION,
        summary=("Stable surface for external agents (Claude Code etc.): jobs, briefing, "
                 "skill search, raw data, notes, operator requests, handover."),
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.ctx = ctx
    jm = job_manager or default_job_manager()
    app.state.jobs = jm

    rt = direct_exec.live_runtime(ctx)
    hook = getattr(rt, "add_shutdown_hook", None)
    if callable(hook):
        try:
            hook(jm.shutdown, name="ext-gateway jobs")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ext-gateway: 关停钩子登记失败(%s) —— 关停时在跑的外部作业"
                           "不会被提前请停", exc)

    # 每个响应都带版本头 —— 客户端靠它区分「外部面里的 404（没有这个作业）」与「这个
    # MAST 根本没挂外部面」。错误处理器里也显式带上：未处理异常的 500 由最外层的
    # ServerErrorMiddleware 产出，http 中间件看不到它。
    hdr = {"X-MAST-Ext-Version": API_VERSION}

    @app.middleware("http")
    async def _version_header(request: Request, call_next):
        refusal = browser_refusal(request)
        if refusal is not None:
            status, code, detail = refusal
            logger.warning("ext-gateway: 拒绝 %s %s（%s）origin=%r host=%r", request.method,
                           request.url.path, code, request.headers.get("origin"),
                           request.headers.get("host"))
            return JSONResponse(status_code=status, headers=hdr,
                                content={"error": code, "detail": detail})
        resp = await call_next(request)
        resp.headers["X-MAST-Ext-Version"] = API_VERSION
        return resp

    @app.exception_handler(ExtError)
    async def _ext_error(request: Request, exc: ExtError):
        return JSONResponse(status_code=exc.status, headers=hdr,
                            content=jsonable_encoder(direct_exec.jsonable(exc.body())))

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, headers=hdr, content={
            "error": "invalid_request",
            "detail": direct_exec.jsonable(jsonable_encoder(exc.errors(), custom_encoder={
                Exception: lambda e: str(e)}))})

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        detail = exc.detail if exc.status_code != 404 else (
            f"外部面没有这个端点：{request.url.path}（端点清单见 /api/ext/v1/openapi.json）")
        return JSONResponse(status_code=exc.status_code, headers=hdr,
                            content={"error": code, "detail": str(detail)})

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception):
        logger.exception("ext-gateway: %s %s 未处理异常", request.method, request.url.path)
        return JSONResponse(status_code=500, headers=hdr, content={
            "error": "internal", "detail": f"{type(exc).__name__}: {exc}"[:500]})

    from mast.api.ext import collab, data, jobs, overview, skills

    for module in (overview, jobs, skills, data, collab):
        app.include_router(module.router)
    return app


__all__ = ["create_ext_app", "default_job_manager"]
