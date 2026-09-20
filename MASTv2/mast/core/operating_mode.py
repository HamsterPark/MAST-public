"""Process-level live-read holder for the global operating mode.

Mirrors :mod:`mast.vision.classical_thresholds` (a live-read holder written by
the runtime, read lock-free by the layers that need it). The agent path already
threads the mode through explicit ``get_mode`` callables into its middlewares;
this holder exists for the layers that have **no agent context at all** — the
vision publisher thread, the buffer service, composite skills — which still
produce tip verdicts the agent then consumes.

Why it exists
-------------
SAFE mode's contract is "the tip is fine, just keep running experiments"
(:class:`mast.core.types.OperatingMode`). Until 2026-08-01 that contract was
enforced only at the *agent's* door: the belief block told the model the tip was
good and SafetyGate refused pulses/shaping. The tip verdicts themselves kept
running, so in SAFE the model was told "tip is good" while every tool result,
buffer event and vision note said "tip is bad" — and a CRITICAL
``tip_quality_drop`` still halted the very experiment SAFE told it to focus on.
This holder lets the *verdict producers* honour the same contract, removing the
tip-repair incentive at the source instead of arguing with it in the prompt.

Contract
--------
- ``safe_mode_active()`` is the ONLY predicate callers should use. It is False
  whenever the mode is unknown — unbound holder, missing settings, a source that
  raises. **Fail to "do not override"**: an unbound process (tests, headless
  ``pipeline.main``, offline tools) behaves exactly as it did before this module
  existed.
- SEMI never overrides. SEMI's contract is "shallow shaping allowed, pulses to
  HITL" — it wants real verdicts.
- The bound source is the runtime's ``_current_operating_mode`` (SettingsStore-
  backed, ``RLock``-guarded), so a ``POST /api/settings`` switch takes effect on
  the next verdict with no rebuild.
- Reads happen on the vision publisher thread and inside composite skills, so
  they must never raise and never block: the holder read is an atomic reference
  read and the source call is a single dict get under the store's lock.

This module imports nothing but :mod:`mast.core.types` — ``core`` must not
import ``webui`` (same one-way rule as ``core.experiment_paths``), which is why
the runtime *injects* the source instead of this module reading settings.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from mast.core.types import OperatingMode

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
# Callable returning the current mode as a str / OperatingMode, or None when no
# runtime has bound one (the default — override inactive).
_SOURCE: "Callable[[], OperatingMode | str | None] | None" = None


def bind_mode_source(source: "Callable[[], OperatingMode | str | None] | None") -> None:
    """Bind (or with ``None``, unbind) the callable that reports the live mode.

    Called once from ``CoreRuntime.setup()``. Idempotent — a later bind replaces
    the previous source. Tests MUST unbind in teardown (``bind_mode_source(None)``)
    or the holder leaks into every later test in the same process.
    """
    global _SOURCE
    with _LOCK:
        _SOURCE = source


def current_operating_mode() -> "OperatingMode | None":
    """The live mode, or ``None`` when unknown (unbound / source failed).

    ``None`` is deliberately distinct from ``AUTO``: "nobody told us" is not the
    same claim as "the operator chose auto", and only the former should leave
    every mode-dependent behaviour at its pre-existing default.
    """
    source = _SOURCE  # atomic reference read; no lock needed on the read path
    if source is None:
        return None
    try:
        raw = source()
    except Exception as exc:  # noqa: BLE001 — publisher thread; never propagate
        logger.debug("operating-mode source failed (treating as unknown): %s", exc)
        return None
    if raw is None:
        return None
    return OperatingMode.coerce(raw)


def safe_mode_active() -> bool:
    """True only when the operator has explicitly selected SAFE.

    The single predicate for "suppress tip-quality verdicts". Unknown → False.
    """
    return current_operating_mode() is OperatingMode.SAFE


__all__ = ["bind_mode_source", "current_operating_mode", "safe_mode_active"]
