"""GET /api/remote-access — cross-network remote-control readiness (Tailscale).

Surfaces, for the in-app 设置 → 远程访问 panel: whether the service is bound for
remote access (0.0.0.0), the scheme/port actually serving, this machine's LAN IP,
and the live Tailscale status + the exact URL(s) to open on ANOTHER computer.

The heavy lifting (CLI detection) is :mod:`mast.net.tailscale`; this route is a
thin adapter that never raises. It's a *sync* def on purpose so FastAPI runs it
in the threadpool — ``tailscale status`` can block a couple of seconds and must
not stall the event loop.
"""
from __future__ import annotations

import socket

from fastapi import APIRouter, Request

from mast.api.schemas_remote_access import (
    RemoteAccessResponse,
    RemoteUrl,
    TailscaleInfo,
)

router = APIRouter(tags=["remote-access"])


def _server_binding(request: Request) -> tuple[str, int, bool]:
    """(scheme, port, bound_all_interfaces).

    Prefers the live uvicorn server's config (the source of truth for the bind
    host + TLS); falls back to the request the client actually reached us on
    (correct for scheme/port even behind the SPA, just blind to the bind host).
    """
    scheme = request.url.scheme
    port = request.url.port or (443 if scheme == "https" else 80)
    bound_all = False
    server = getattr(request.app.state, "uvicorn_server", None)
    cfg = getattr(server, "config", None)
    if cfg is not None:
        host = str(getattr(cfg, "host", "") or "")
        bound_all = host in ("0.0.0.0", "::", "")
        p = getattr(cfg, "port", None)
        if p:
            port = p
        if getattr(cfg, "ssl_certfile", None):
            scheme = "https"
    return scheme, port, bound_all


def _lan_ipv4() -> str:
    """This machine's primary outbound IPv4 (UDP-socket trick), '' if offline."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return ""


@router.get("/remote-access", response_model=RemoteAccessResponse)
def remote_access(request: Request) -> RemoteAccessResponse:
    from mast.net import tailscale as ts

    scheme, port, bound_all = _server_binding(request)
    status = ts.get_status()
    urls = ts.remote_urls(status, port, scheme)

    if not bound_all:
        note = ("远程访问未开启 —— 目前仅本机可访问。在启动器勾选"
                "「启用局域网 / 远程访问」并重启服务即可开放跨网控制。")
    elif not status.installed:
        note = status.hint
    elif not status.ready:
        note = status.hint
    elif urls:
        note = f"就绪 —— 在另一台电脑（登录同一 Tailscale 账号）浏览器打开：{urls[0]['url']}"
    else:
        note = "已绑定远程访问，但暂未取得 Tailscale 地址，请稍候刷新。"

    return RemoteAccessResponse(
        lan_enabled=bound_all,
        scheme=scheme,
        port=port,
        lan_ip=_lan_ipv4(),
        tailscale=TailscaleInfo(
            installed=status.installed,
            running=status.running,
            ready=status.ready,
            backend_state=status.backend_state,
            device_name=status.device_name,
            magic_dns=status.magic_dns,
            tailnet=status.tailnet,
            ips=list(status.self_ips),
            hint=status.hint,
        ),
        urls=[RemoteUrl(label=u["label"], host=u["host"], url=u["url"]) for u in urls],
        note=note,
    )
