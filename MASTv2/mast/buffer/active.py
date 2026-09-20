"""Process-level handle to the running BufferService.

The BufferService is created in ``mast/pipeline/main.py`` and passed to the
orchestrator — it is NOT a global singleton. But some producers (notably the
scan-progress vision monitor, which is spawned from a low-level scan skill that
has no reference to the orchestrator's buffer) need to reach it. Rather than
thread the buffer through every ExecutionContext, the pipeline registers the
active instance here once at startup; producers fetch it with
:func:`get_active_buffer`.

Fail-safe: if no buffer is registered (dev / tests / no-hardware mode),
:func:`get_active_buffer` returns ``None`` and producers simply skip publishing.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mast.buffer.service import BufferService

_ACTIVE: "BufferService | None" = None
_LOCK = threading.Lock()


def set_active_buffer(buf: "BufferService | None") -> None:
    """Register (or clear) the process-wide active BufferService.

    Called once by the pipeline after constructing the buffer. Passing ``None``
    clears it (e.g. on shutdown / no-hardware mode)."""
    global _ACTIVE
    with _LOCK:
        _ACTIVE = buf


def get_active_buffer() -> "BufferService | None":
    """Return the registered BufferService, or ``None`` if none is active."""
    with _LOCK:
        return _ACTIVE


__all__ = ["set_active_buffer", "get_active_buffer"]
