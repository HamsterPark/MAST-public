"""Session-auth guard for custom routes mounted on the Gradio app .

Gradio's ``launch(auth=...)`` protects only Gradio's OWN routes — each of them
carries ``Depends(login_check)`` (see gradio/routes.py). Routes we hand-insert
into ``app.routes`` (/agents/*, /api/*, /environment/*, /artifacts/*,
/interrupts/*, /ws/events) and static Mounts (/agents-ui) bypass that check
entirely: on a LAN deployment anyone on the subnet could drive the agent
pipeline — and through it the instrument — with zero credentials
(2026-06-11 builder-project audit, 修复项).

This module re-implements Gradio 6.x's session check against the SAME state
the real login flow populates:

  * ``app.auth`` / ``app.auth_dependency``  — whether auth is enabled at all;
  * ``app.tokens``                          — token → username map filled by /login;
  * ``access-token-{app.cookie_id}`` cookie — the session token
    (plus the ``access-token-unsecure-`` variant, mirroring gradio).

Semantics:
  * auth disabled (localhost default)  → everything passes, zero friction;
  * auth enabled (LAN mode)            → only logged-in sessions pass;
  * resolution error                   → DENY (fail-closed; this guards
    instrument control, not a content page).

``Request`` and ``WebSocket`` both subclass ``HTTPConnection``, so one
resolver serves HTTP routes, WebSocket routes and static mounts.
"""

from __future__ import annotations

import functools
import logging

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def _auth_enabled(app) -> bool:
    return (getattr(app, "auth", None) is not None
            or getattr(app, "auth_dependency", None) is not None)


def _connection_user(conn: HTTPConnection) -> str | None:
    """Resolve the logged-in username, mirroring gradio.routes.get_current_user.

    NOTE: gradio's ``auth_dependency`` contract is Callable[[fastapi.Request],
    str|None]; here it receives the HTTPConnection (Request's base class — and
    a WebSocket on WS routes). Dependencies that only read cookies/headers
    work; one touching Request-only attributes raises → resolves to None →
    deny (fail-closed). MAST itself only uses launch(auth=tuple), so this
    branch is dormant in production."""
    app = conn.scope.get("app")
    if app is None:
        return None
    dep = getattr(app, "auth_dependency", None)
    if dep is not None:
        try:
            return dep(conn)
        except Exception:  # fail-closed
            return None
    cookie_id = getattr(app, "cookie_id", None)
    tokens = getattr(app, "tokens", None) or {}
    token = (conn.cookies.get(f"access-token-{cookie_id}")
             or conn.cookies.get(f"access-token-unsecure-{cookie_id}"))
    return tokens.get(token)


def is_authorized(conn: HTTPConnection) -> bool:
    """True iff auth is disabled OR the connection carries a live session.

    A scope without an ``app`` means the auth state is UNDETERMINABLE — deny
    (fail-closed; review nit: this guard protects instrument control, and a
    bare-ASGI reuse must not silently disable it). Under gradio/uvicorn the
    Starlette root always injects scope['app'], so this never fires in
    production."""
    try:
        app = conn.scope.get("app")
        if app is None:
            logger.error("route auth: no 'app' in ASGI scope — denying "
                         "(auth state undeterminable)")
            return False
        if not _auth_enabled(app):
            return True
        return _connection_user(conn) is not None
    except Exception:  # pragma: no cover — fail-closed
        logger.exception("route auth check failed — denying")
        return False


def authed_route(handler):
    """Wrap a Starlette HTTP endpoint with the Gradio session check.

    Preserves the handler's sync/async nature: Starlette runs sync endpoints
    on its threadpool — wrapping one in an async shim would move its body onto
    the event loop and violate the never-block-the-loop discipline."""
    import inspect
    if inspect.iscoroutinefunction(handler):
        @functools.wraps(handler)
        async def _wrapped_async(request):
            if not is_authorized(request):
                return JSONResponse({"error": "not authenticated"}, status_code=401)
            return await handler(request)
        return _wrapped_async

    @functools.wraps(handler)
    def _wrapped_sync(request):
        if not is_authorized(request):
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        return handler(request)
    return _wrapped_sync


def authed_page(handler):
    """Like authed_route, but for browser-rendered HTML pages: an
    unauthenticated hit redirects to "/" (the gradio login page) instead of
    returning bare JSON the user can't act on."""
    import inspect

    from starlette.responses import RedirectResponse
    if inspect.iscoroutinefunction(handler):
        @functools.wraps(handler)
        async def _wrapped_async(request):
            if not is_authorized(request):
                return RedirectResponse("/", status_code=302)
            return await handler(request)
        return _wrapped_async

    @functools.wraps(handler)
    def _wrapped_sync(request):
        if not is_authorized(request):
            return RedirectResponse("/", status_code=302)
        return handler(request)
    return _wrapped_sync


def authed_ws(handler):
    """Wrap an async Starlette WebSocket endpoint with the session check.

    Closing before ``accept()`` makes uvicorn reject the handshake (HTTP 403),
    so an unauthenticated client never reaches the event stream."""
    @functools.wraps(handler)
    async def _wrapped(websocket):
        if not is_authorized(websocket):
            await websocket.close(code=1008)  # policy violation
            return
        return await handler(websocket)
    return _wrapped


class AuthedStaticFiles(StaticFiles):
    """StaticFiles that honours the Gradio session before serving anything.

    Used for the ``/agents-ui`` mount: after the user logs into Gradio the
    same-origin iframe carries the access-token cookie, so the React bundle
    loads normally; without a session the bundle (and the API surface it
    documents) is not exposed to the subnet."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not is_authorized(HTTPConnection(scope)):
            resp = JSONResponse({"error": "not authenticated"}, status_code=401)
            await resp(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


__all__ = ["authed_route", "authed_page", "authed_ws", "AuthedStaticFiles",
           "is_authorized"]
