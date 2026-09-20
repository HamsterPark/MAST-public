"""外部面的公共零件：调用方身份、错误体、从请求取活对象。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from mast.api import direct_exec

logger = logging.getLogger(__name__)

#: 外部面契约版本（URL 里的 ``v1``）。v1 之内只增不减。
API_VERSION = "1"

#: 外部 agent 在记录里的署名前缀：v2 ``agent_id``、v1 ``actions.context``、笔记作者、
#: 心愿单请求的 ``agent_id``、组合技能的 ``_author``。**唯一定义处。**
EXT_PREFIX = "ext:"


@dataclass(frozen=True)
class Caller:
    """一次请求的调用方（自报，只用于归属）。"""

    actor: str
    session: str = ""

    @property
    def agent_id(self) -> str:
        return f"{EXT_PREFIX}{self.actor}"

    @property
    def owner(self) -> str:
        """仪器令牌的 owner —— 被拒的一方读到的「现在开车的是谁」。"""
        return f"外部 agent {self.agent_id}"

    @property
    def thread_id(self) -> str:
        """v2 ``actions.thread_id`` 与 v1 ``actions.context``：带上会话，区分同名的两次接入。"""
        return f"{self.agent_id}/{self.session}" if self.session else self.agent_id


def caller_of(request: Request) -> Caller:
    """从请求头取调用方。**永不因它拒绝请求**：缺了就是 ``anonymous``。"""
    actor = direct_exec.actor_slug(request.headers.get("x-mast-actor", "")) or "anonymous"
    session = direct_exec.actor_slug(request.headers.get("x-mast-session", ""), limit=64)
    return Caller(actor=actor, session=session)


class ExtError(Exception):
    """外部面的业务错误 → ``{"error": code, "detail": ..., **extra}`` + 状态码。"""

    def __init__(self, status: int, code: str, detail: str = "", **extra: Any) -> None:
        super().__init__(detail or code)
        self.status = int(status)
        self.code = code
        self.detail = detail
        self.extra = extra

    def body(self) -> dict:
        return {"error": self.code, "detail": self.detail, **self.extra}


# ─────────────────────────────────────────────────────────────────────
# 活对象（全部永不抛；取不到返回 None）
# ─────────────────────────────────────────────────────────────────────

def ctx_of(request: Request) -> Any:
    return getattr(request.app.state, "ctx", None)


def runtime_of(request: Request) -> Any:
    return direct_exec.live_runtime(ctx_of(request))


def registry_of(request: Request) -> Any:
    ctx = ctx_of(request)
    return getattr(ctx, "skill_registry", None) or getattr(ctx, "registry", None)


def storage_of(request: Request) -> Any:
    ctx = ctx_of(request)
    st = getattr(ctx, "experiment_storage", None)
    if st is not None:
        return st
    rt = runtime_of(request)
    return getattr(rt, "_storage", None)


def active_log() -> Any:
    try:
        from mast.logging.experiment_log import get_active_log

        return get_active_log()
    except Exception:  # noqa: BLE001
        return None


def scope_ids() -> tuple[str | None, str | None]:
    log = active_log()
    if log is None:
        return None, None
    try:
        return log.current_experiment_id, log.current_sample_id
    except Exception:  # noqa: BLE001
        return None, None


def cognition_of(request: Request) -> Any:
    ctx = ctx_of(request)
    cog = getattr(ctx, "cognition", None)
    if cog is not None:
        return cog
    rt = runtime_of(request)
    return getattr(rt, "_cognition", None)


def memory_store_of(request: Request) -> Any:
    cog = cognition_of(request)
    st = getattr(cog, "store", None) if cog is not None else None
    if st is not None:
        return st
    return getattr(ctx_of(request), "memory_store", None)


def operating_mode() -> str:
    """``safe`` / ``semi`` / ``auto`` / ``unknown``。

    **未绑定就说 unknown** —— 不走 ``OperatingMode.coerce``：它对读不懂的值 fail-open
    成 AUTO，而把「不知道」说成「全自动」正是这一族缺陷的形状。
    """
    try:
        from mast.core.operating_mode import current_operating_mode

        m = current_operating_mode()
    except Exception:  # noqa: BLE001
        return "unknown"
    if m is None:
        return "unknown"
    return str(getattr(m, "value", m) or "unknown")


__all__ = [
    "API_VERSION",
    "EXT_PREFIX",
    "Caller",
    "ExtError",
    "active_log",
    "caller_of",
    "cognition_of",
    "ctx_of",
    "memory_store_of",
    "operating_mode",
    "registry_of",
    "runtime_of",
    "scope_ids",
    "storage_of",
]
