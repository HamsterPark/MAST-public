"""Settings from the environment.

=====================  ==========================================================
``MAST_URL``           MAST root, default ``http://127.0.0.1:7862`` (LAN mode: https)
``MAST_USER``          HTTP Basic user (LAN mode only)
``MAST_PASSWORD``      HTTP Basic password
``MAST_VERIFY_TLS``    ``false`` accepts the instrument PC's self-signed certificate
``MAST_ALLOW_REMOTE``  ``true`` allows a MAST that is not on this machine
``MAST_ACTOR``         name the actions are attributed to, default ``claude-code``
``MAST_FETCH_DIR``     where ``mast_fetch`` saves files, default ``.mast-fetch/``
=====================  ==========================================================

Plugin option values reach us through ``${user_config.*}`` substitution. An
option that was never filled in may arrive empty or as the unexpanded
placeholder itself; both mean "not set".

**Loopback only by default.** Controlling the instrument is meant to happen on
this machine or inside a VPN. A MAST URL that points elsewhere is refused until
the user opts in with ``MAST_ALLOW_REMOTE=true``; the server still starts and
lists its tools, and every tool call explains the refusal.
"""
from __future__ import annotations

import ipaddress
import os
import re
import secrets
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:7862"
DEFAULT_ACTOR = "claude-code"
API_PREFIX = "/api/ext/v1"
FETCH_DIRNAME = ".mast-fetch"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_PLACEHOLDER = re.compile(r"^\$\{[^}]*\}$")
_TOKEN_JUNK = re.compile(r"[^a-z0-9_.-]+")


def env_value(environ, key: str) -> str:
    """The stripped value of ``key``; unset, empty and unexpanded all give ``""``."""
    raw = environ.get(key)
    if raw is None:
        return ""
    value = str(raw).strip()
    if _PLACEHOLDER.match(value):
        return ""
    return value


def env_secret(environ, key: str) -> str:
    """Like ``env_value`` but keeps surrounding spaces: they may be part of a password."""
    raw = environ.get(key)
    if raw is None or not str(raw).strip() or _PLACEHOLDER.match(str(raw).strip()):
        return ""
    return str(raw)


def env_flag(environ, key: str, default: bool) -> bool:
    value = env_value(environ, key).lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def sanitize_token(value: str, max_len: int) -> str:
    """Lower-case ``[a-z0-9_.-]`` only (HTTP header values must be plain ASCII)."""
    return _TOKEN_JUNK.sub("-", (value or "").strip().lower()).strip("-.")[:max_len]


def is_loopback_host(host: str) -> bool:
    """``localhost`` or a loopback IP. Names are not resolved: a DNS name that
    happens to point at 127.0.0.1 still counts as remote (the safe side)."""
    h = (host or "").strip().strip("[]").rstrip(".").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Config:
    url: str                    # MAST root, e.g. http://127.0.0.1:7862
    api_base: str               # url + /api/ext/v1
    scheme: str
    host: str
    loopback: bool
    user: str
    password: str
    verify_tls: bool
    allow_remote: bool
    actor: str
    session: str
    fetch_dir: str
    fetch_dir_is_default: bool
    problem: str                # non-empty: every tool call refuses with this text

    def describe(self) -> dict:
        """What is safe to log: everything except the password."""
        return {
            "url": self.url, "actor": self.actor, "session": self.session,
            "auth": bool(self.user), "verify_tls": self.verify_tls,
            "allow_remote": self.allow_remote, "fetch_dir": self.fetch_dir,
            "problem": self.problem,
        }


def load_config(environ=None, *, cwd: str | None = None) -> Config:
    env = os.environ if environ is None else environ
    raw_url = env_value(env, "MAST_URL") or DEFAULT_URL
    url = raw_url.rstrip("/")
    if url.lower().endswith(API_PREFIX):
        url = url[: -len(API_PREFIX)].rstrip("/")

    problem = ""
    scheme = host = ""
    try:
        parts = urlsplit(url)
        scheme = (parts.scheme or "").lower()
        host = (parts.hostname or "").lower()
        parts.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError as exc:
        problem = f"MAST_URL {raw_url!r} is not a valid URL ({exc})."
    if not problem and (scheme not in ("http", "https") or not host):
        problem = (f"MAST_URL {raw_url!r} is not a MAST address; it should look like "
                   f"{DEFAULT_URL} (or https://<host>:7862 in LAN mode).")
    loopback = bool(host) and is_loopback_host(host)

    allow_remote = env_flag(env, "MAST_ALLOW_REMOTE", False)
    if not problem and not loopback and not allow_remote:
        problem = (
            f"MAST_URL points at {host}, which is not this machine. Control of the "
            "instrument is limited to this machine or a VPN, so a remote MAST is refused "
            "by default. If that host is reached over a VPN, opt in with the plugin option "
            "allow_remote (MAST_ALLOW_REMOTE=true). Never expose MAST to the open internet.")

    fetch_dir = env_value(env, "MAST_FETCH_DIR")
    fetch_is_default = not fetch_dir
    if fetch_is_default:
        base = env_value(env, "CLAUDE_PROJECT_DIR") or cwd or os.getcwd()
        fetch_dir = os.path.join(base, FETCH_DIRNAME)
    fetch_dir = os.path.abspath(os.path.expanduser(fetch_dir))

    return Config(
        url=url,
        api_base=url + API_PREFIX,
        scheme=scheme,
        host=host,
        loopback=loopback,
        user=env_value(env, "MAST_USER"),
        password=env_secret(env, "MAST_PASSWORD"),
        verify_tls=env_flag(env, "MAST_VERIFY_TLS", True),
        allow_remote=allow_remote,
        actor=sanitize_token(env_value(env, "MAST_ACTOR"), 48) or DEFAULT_ACTOR,
        session=sanitize_token(env_value(env, "MAST_SESSION"), 64) or secrets.token_hex(4),
        fetch_dir=fetch_dir,
        fetch_dir_is_default=fetch_is_default,
        problem=problem,
    )
