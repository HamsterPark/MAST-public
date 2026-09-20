"""HTTP to MAST's external agent API, with every failure turned into a sentence.

An agent that gets ``HTTP=000`` or a page of HTML back can only guess. Every
way a request can go wrong here ends as a :class:`MastError` whose message says
what to check: is the service running, http or https, self-signed certificate,
login, an older MAST without the external API, a proxy in the way.

Proxies are switched off on purpose. MAST is reached on this machine or over a
VPN; a system proxy (common on Windows desktops) would either break loopback
requests or carry instrument control through a third party.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import logging
import os
import re
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request

from . import __version__
from .config import API_PREFIX

log = logging.getLogger("mast_mcp.client")

#: Largest JSON answer read into memory; the tool layer clips what it shows anyway.
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ERROR_BYTES = 256 * 1024
DOWNLOAD_CHUNK = 64 * 1024


class MastError(Exception):
    """A request that produced no usable answer, explained in words.

    ``kind`` is a short machine code (``unreachable``, ``auth``, ``not_found``,
    or the server's own ``error`` code); ``transient`` marks failures where the
    request may or may not have reached MAST (timeouts, dropped connections).
    """

    def __init__(self, kind: str, message: str, *, status: int | None = None,
                 payload=None, transient: bool = False):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        self.payload = payload
        self.transient = transient

    def as_dict(self) -> dict:
        out: dict = {"error": self.kind}
        if self.status is not None:
            out["http_status"] = self.status
        if self.payload is not None:
            out["server"] = self.payload
        return out


class Response:
    def __init__(self, status: int, headers, body: bytes):
        self.status = status
        self.headers = headers
        self.body = body

    def header(self, name: str) -> str:
        if self.headers is None:
            return ""
        return str(self.headers.get(name) or "")

    @property
    def content_type(self) -> str:
        return self.header("Content-Type").lower()


def _query_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _snippet(text: str, limit: int = 200) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def detail_text(payload) -> str:
    """Human text out of a MAST error body (contract shape or FastAPI's own)."""
    if not isinstance(payload, dict):
        return ""
    detail = payload.get("detail")
    if isinstance(detail, list):                      # FastAPI validation errors
        parts = []
        for item in detail[:6]:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in item.get("loc", ()) if x != "body")
                msg = str(item.get("msg") or item)
                parts.append(f"{loc}: {msg}" if loc else msg)
            else:
                parts.append(str(item))
        return "; ".join(parts)
    if isinstance(detail, dict):
        return json.dumps(detail, ensure_ascii=False)[:300]
    if detail:
        return str(detail)
    return str(payload.get("message") or "")


def _ssl_context(verify: bool) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


class MastClient:
    def __init__(self, config):
        self.config = config
        handlers: list = [urllib.request.ProxyHandler({})]
        if config.scheme == "https":
            handlers.append(urllib.request.HTTPSHandler(context=_ssl_context(config.verify_tls)))
        self._opener = urllib.request.build_opener(*handlers)
        self._headers = {
            "User-Agent": f"mast-mcp/{__version__}",
            "X-MAST-Actor": config.actor,
            "X-MAST-Session": config.session,
        }
        if config.user or config.password:
            token = base64.b64encode(f"{config.user}:{config.password}".encode("utf-8"))
            self._headers["Authorization"] = "Basic " + token.decode("ascii")

    # ── plumbing ─────────────────────────────────────────────────────────
    def url_for(self, path: str, query: dict | None = None) -> str:
        url = self.config.api_base + path
        if query:
            pairs = [(k, _query_value(v)) for k, v in query.items()
                     if v is not None and v != "" and v != [] and v != ()]
            if pairs:
                url += "?" + urllib.parse.urlencode(pairs)
        return url

    def _open(self, method: str, path: str, *, query=None, body=None,
              timeout: float = 30.0, accept: str = "application/json"):
        """``(live_response, None)`` for a 2xx, ``(None, Response)`` for an HTTP error."""
        headers = dict(self._headers)
        headers["Accept"] = accept
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(self.url_for(path, query), data=data,
                                     headers=headers, method=method)
        try:
            return self._opener.open(req, timeout=max(0.5, float(timeout))), None
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read(MAX_ERROR_BYTES) or b""
            except (OSError, http.client.HTTPException):
                raw = b""
            finally:
                exc.close()
            return None, Response(exc.code, exc.headers, raw)
        except urllib.error.URLError as exc:
            raise self.transport_error(exc.reason, timeout) from None
        except (OSError, http.client.HTTPException) as exc:
            raise self.transport_error(exc, timeout) from None

    def request(self, method: str, path: str, *, query=None, body=None,
                timeout: float = 30.0) -> Response:
        live, failed = self._open(method, path, query=query, body=body, timeout=timeout)
        if failed is not None:
            return failed
        try:
            with live:
                raw = live.read(MAX_JSON_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise self.transport_error(exc, timeout) from None
        if len(raw) > MAX_JSON_BYTES:
            raise MastError("too_large", f"MAST's answer to {method} {path} is larger than "
                            f"{MAX_JSON_BYTES // (1024 * 1024)} MB; ask for less.")
        return Response(live.status, live.headers, raw)

    def call(self, method: str, path: str, *, query=None, body=None,
             timeout: float = 30.0):
        """JSON answer of a 2xx, or :class:`MastError`."""
        resp = self.request(method, path, query=query, body=body, timeout=timeout)
        if 200 <= resp.status < 300:
            return self._parse_json(method, path, resp)
        raise self.http_error(method, path, resp)

    def _parse_json(self, method: str, path: str, resp: Response):
        text = resp.body.decode("utf-8", errors="replace")
        head = text.lstrip()[:1]
        ctype = resp.content_type
        if not head:
            return {}
        if "json" in ctype or (not ctype and head in "{["):
            try:
                return json.loads(text)
            except ValueError as exc:
                raise MastError("bad_json", f"MAST answered {method} {path} with malformed "
                                f"JSON ({exc}).", status=resp.status) from None
        if "html" in ctype or head == "<":
            raise MastError("not_json", self._html_instead_of_json(method, path),
                            status=resp.status)
        raise MastError("not_json", f"MAST answered {method} {path} with "
                        f"{ctype or 'an untyped body'} instead of JSON: {_snippet(text)}",
                        status=resp.status)

    def _html_instead_of_json(self, method: str, path: str) -> str:
        return (f"MAST at {self.config.url} answered {method} {path} with an HTML page "
                f"instead of JSON. That MAST probably has no external agent API "
                f"({API_PREFIX}): update MAST, or check that MAST_URL is the MAST service "
                "address (port 7862 by default), not some other web server.")

    # ── error shaping ────────────────────────────────────────────────────
    def http_error(self, method: str, path: str, resp: Response) -> MastError:
        text = resp.body.decode("utf-8", errors="replace")
        payload = None
        if "json" in resp.content_type or text.lstrip()[:1] == "{":
            try:
                payload = json.loads(text)
            except ValueError:
                payload = None
        code = payload.get("error") if isinstance(payload, dict) else None
        code = code if isinstance(code, str) and code else None
        detail = detail_text(payload) or _snippet(text) or resp.content_type or "(empty body)"
        status = resp.status
        where = f"{method} {path}"

        if status == 401:
            if self.config.user:
                msg = ("MAST rejected the configured login (MAST_USER / MAST_PASSWORD). "
                       "Check the user name and password of MAST's LAN access.")
            else:
                msg = ("MAST asked for a login: it runs in LAN mode, which uses HTTP Basic. "
                       "Set the plugin options mast_user and mast_password "
                       "(MAST_USER / MAST_PASSWORD).")
            return MastError("auth", msg, status=status, payload=payload)
        if status == 404 and not resp.header("X-MAST-Ext-Version") and (
                payload is None or detail in ("Not Found", "")):
            return MastError("no_ext_api", (
                f"MAST at {self.config.url} has no external agent API ({API_PREFIX}): it is "
                "probably an older MAST, or MAST_URL points at a different service."),
                status=status, payload=payload)
        if status == 404:
            return MastError(code or "not_found", f"Not found ({where}): {detail}",
                             status=status, payload=payload)
        if status == 409:
            return MastError(code or "conflict", f"Conflict ({where}): {detail}",
                             status=status, payload=payload)
        if status == 422:
            return MastError(code or "invalid_request", f"MAST refused {where}: {detail}",
                             status=status, payload=payload)
        if status == 429:
            return MastError(code or "too_many_jobs", (
                f"Too many jobs at once ({detail}). Wait for running jobs to finish "
                "(mast_jobs) before submitting more."), status=status, payload=payload)
        if status == 503:
            missing = payload.get("missing") if isinstance(payload, dict) else None
            extra = f" Missing: {', '.join(map(str, missing))}." if missing else ""
            return MastError(code or "not_wired", (
                f"MAST's external API is up but not fully wired ({detail}).{extra} "
                "Usually MAST is still starting or a subsystem failed to start; "
                "check mast_status in a minute."), status=status, payload=payload)
        if status == 403:
            return MastError(code or "forbidden", f"MAST refused {where}: {detail}",
                             status=status, payload=payload)
        if status >= 500:
            return MastError(code or "server_error",
                             f"MAST server error on {where} (HTTP {status}): {detail}",
                             status=status, payload=payload)
        return MastError(code or f"http_{status}", f"{where} failed (HTTP {status}): {detail}",
                         status=status, payload=payload)

    def transport_error(self, reason, timeout: float) -> MastError:
        url = self.config.url
        https_hint = ("; if MAST runs in LAN mode the address must start with https://"
                      if self.config.scheme == "http" else "")
        if isinstance(reason, ssl.SSLCertVerificationError):
            return MastError("tls_verify", (
                f"The TLS certificate of {url} is not trusted "
                f"({getattr(reason, 'verify_message', '') or reason}). The instrument PC "
                "normally uses a self-signed certificate: set the plugin option verify_tls "
                "to false (MAST_VERIFY_TLS=false), or install that certificate."))
        if isinstance(reason, ssl.SSLError):
            return MastError("tls", (
                f"TLS handshake with {url} failed ({reason}). If MAST on that port serves "
                "plain http, use http:// in MAST_URL."))
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return MastError("timeout", f"MAST at {url} did not answer within {timeout:.0f} s.",
                             transient=True)
        if isinstance(reason, ConnectionRefusedError):
            return MastError("unreachable", (
                f"Cannot connect to MAST at {url} (connection refused). Is the MAST service "
                "running? Check the port too, and http vs https (LAN mode serves https)."))
        if isinstance(reason, socket.gaierror):
            return MastError("unreachable", f"Cannot resolve the host in MAST_URL ({url}): {reason}.")
        if isinstance(reason, (ConnectionResetError, ConnectionAbortedError,
                               http.client.RemoteDisconnected, http.client.BadStatusLine)):
            return MastError("dropped", (
                f"MAST at {url} closed the connection without an HTTP answer{https_hint}."),
                transient=True)
        if isinstance(reason, (OSError, http.client.HTTPException)):
            return MastError("unreachable", (
                f"Cannot reach MAST at {url} ({reason}). Is the service running, and is the "
                "VPN up if MAST is on another machine?"))
        return MastError("unreachable", f"Cannot reach MAST at {url}: {reason}")

    # ── raw downloads ────────────────────────────────────────────────────
    def download(self, path: str, query: dict, dest: str, *, timeout: float,
                 should_stop, deadline: float, clock) -> dict:
        """Stream a data endpoint into ``dest`` (written as ``dest.part``, then renamed).

        Returns ``{bytes, sha256, content_type, headers}``. Nothing is left on
        disk when the download fails, is interrupted or runs out of time.
        """
        live, failed = self._open("GET", path, query=query, timeout=timeout,
                                  accept="application/octet-stream, */*;q=0.1")
        if failed is not None:
            raise self.http_error("GET", path, failed)
        ctype = (live.headers.get("Content-Type") or "").lower()
        tmp = dest + ".part"
        digest = hashlib.sha256()
        total = 0
        try:
            with live:
                if "html" in ctype:
                    raise MastError("not_data", self._html_instead_of_json("GET", path))
                first = b""
                if "json" in ctype:
                    # An error report in disguise is refused; any other JSON is the file.
                    first = live.read(MAX_ERROR_BYTES)
                    try:
                        maybe = json.loads(first.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        maybe = None
                    if isinstance(maybe, dict) and "error" in maybe:
                        raise MastError(str(maybe.get("error")),
                                        f"MAST did not send the file: {detail_text(maybe)}",
                                        status=live.status, payload=maybe)
                with open(tmp, "wb") as fh:
                    chunk = first
                    while True:
                        if chunk:
                            fh.write(chunk)
                            digest.update(chunk)
                            total += len(chunk)
                        if should_stop():
                            raise MastError("interrupted", "The download was interrupted; "
                                            "nothing was saved.")
                        left = deadline - clock()
                        if left <= 0:
                            raise MastError("budget", (
                                f"The download did not finish within this call's time budget "
                                f"({total} bytes so far); nothing was saved. Try again, or "
                                "use a faster link for very large files."), transient=True)
                        _limit_read_wait(live, min(timeout, left))
                        chunk = live.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
            os.replace(tmp, dest)
        except (OSError, http.client.HTTPException) as exc:
            _remove_quietly(tmp)
            if isinstance(exc, (socket.timeout, TimeoutError, ConnectionError,
                                http.client.HTTPException, ssl.SSLError)):
                raise self.transport_error(exc, timeout) from None
            raise MastError("local_io", f"Could not write {dest}: {exc}") from None
        except BaseException:
            _remove_quietly(tmp)
            raise
        return {"bytes": total, "sha256": digest.hexdigest(), "content_type": ctype,
                "headers": live.headers}


def _limit_read_wait(live, seconds: float) -> None:
    """Shrink the socket timeout so one stalled read cannot outlive the call's budget.

    Reaches through ``HTTPResponse.fp`` to the socket; if a Python version lays
    that out differently, the timeout given at connect time stays in force.
    """
    sock = getattr(getattr(getattr(live, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        try:
            sock.settimeout(max(0.5, seconds))
        except OSError:
            pass


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
