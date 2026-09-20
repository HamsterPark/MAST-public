"""Runtime-owned inputs a coarse relocation needs, without importing the runtime.

``RelocateCoarseXY`` has to answer two questions that live outside the skill
layer entirely:

* **"Have we already been where this move would land?"** — needs the recorded
  ``coarse_move`` markers and this rig's step-scale config.
* **"How cold is it?"** — a coarse step travels several times further at 300 K
  than at 4 K, so an odometer entry is only comparable with the next one if the
  temperature is recorded beside it. Never gated on; always recorded.

``ExecutionContext`` deliberately carries neither: it holds the connection pool,
the hardware state, the registry and the abort events, and nothing about
experiment records. Reaching around it into ``CoreRuntime`` from a skill would
invert the layering (``skills`` → ``core`` is fine; ``skills`` → the live app
object is not) and would make the skill untestable without a whole runtime.

So the runtime INJECTS these at startup, the same shape as
``instrument_profile.set_persist_sink`` and ``vacuum_interlock``'s pressure
source. With nothing injected every accessor returns ``None``, and every caller
already has to handle that: an unavailable map degrades the destination check to
"cannot tell — allow", because where the stage has been is a surface-budget
question, not a safety one, and refusing to relocate because a database is
unreachable would strand a run over bookkeeping.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_marker_source: Callable[[], tuple[list, Any] | None] | None = None
_temperature_source: Callable[[], float | None] | None = None

__all__ = [
    "set_marker_source",
    "set_temperature_source",
    "markers_and_config",
    "temperature_k",
]


def set_marker_source(source: Callable[[], tuple[list, Any] | None] | None) -> None:
    """Install ``() -> (marker_rows, CoarseMapConfig)`` for the current scope."""
    global _marker_source
    with _lock:
        _marker_source = source


def set_temperature_source(source: Callable[[], float | None] | None) -> None:
    """Install ``() -> temperature_K | None``."""
    global _temperature_source
    with _lock:
        _temperature_source = source


def markers_and_config() -> tuple[list, Any] | tuple[None, None]:
    """``(rows, cfg)`` for the live experiment/sample, or ``(None, None)``.

    ``(None, None)`` means "cannot tell", not "nothing has happened". A caller
    must not read an empty map out of it — that would turn a broken database
    into "this sample is untouched"."""
    src = _marker_source
    if src is None:
        return None, None
    try:
        out = src()
    except Exception as exc:  # noqa: BLE001 — a broken source is "cannot tell"
        logger.debug("coarse map source failed: %s", exc)
        return None, None
    if not out:
        return None, None
    rows, cfg = out
    return list(rows or []), cfg


def temperature_k() -> float | None:
    src = _temperature_source
    if src is None:
        return None
    try:
        return src()
    except Exception as exc:  # noqa: BLE001
        logger.debug("temperature source failed: %s", exc)
        return None
