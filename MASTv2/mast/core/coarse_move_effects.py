"""What must be invalidated the moment the sample stage slides sideways.

A lateral coarse move changes NOTHING that MAST reads. The piezo XY readout keeps
reporting the same numbers; the state snapshot looks identical; no signal drops.
Only the meaning changes: every coordinate recorded before this instant now
addresses a different patch of surface.

``map_markers`` handles that with ``coord_epoch`` (stamped inside ``log_marker``).
But the epoch column only protects things that go THROUGH the marker table.
Everything else that caches a metre-valued, piezo-origin position across the move
keeps its stale value and looks perfectly healthy. This module is the one place
that lists those things, so a future addition has an obvious home instead of
being discovered later as "why is the map drawing a route to nowhere".

Called from all THREE paths that create a ``coarse_move`` row:
  * the recorder (`CoreRuntime._record_map_marker`, a move MAST performed),
  * the agent tool `record_coarse_move` (operator moved by hand, agent backfills),
  * `POST /api/scan-map/coarse-move` (operator moved by hand, backfills in the UI).

Only the first of those used to clear the tip-crash tracker — via
``MotorMove``'s own call — so a hand-made coarse move left the previous region's
crash blocks in force over fresh surface (2026-07-31).

Best-effort by construction: this runs after the marker is already committed, and
a failure here must never turn a successful relocation into a failed one. Every
step is individually guarded and the outcome is returned for logging, not raised.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["on_coarse_move_recorded"]


def on_coarse_move_recorded(*, source: str = "") -> dict[str, str]:
    """Invalidate the caches a lateral coarse move silently poisoned.

    ``source`` is a free-text label for the log line ("skill" / "tool" / "api").
    Returns ``{step: outcome}`` — callers may log it; nobody branches on it.
    """
    out: dict[str, str] = {}

    # 1. The planned route. PlanOverlay is a process-local, in-memory list of
    #    metre-valued steps with NO epoch of its own, and the only thing that
    #    ever cleared it was switching experiment/sample. Left alone across a
    #    coarse move it draws a navigation line onto surface that is no longer
    #    there — and worse, `_log_one_marker` keeps calling `advance()`, so the
    #    steps get consumed by scans of a completely different region.
    try:
        from mast.io.plan_overlay import get_plan_overlay

        overlay = get_plan_overlay()
        had = len(overlay.snapshot() or [])
        overlay.clear()
        out["plan_overlay"] = f"cleared({had})"
    except Exception as exc:  # noqa: BLE001 — never break a completed move
        logger.debug("coarse-move effects: plan overlay clear failed: %s", exc)
        out["plan_overlay"] = "failed"

    # 2. The tip-crash block-list. Quantised to 8 nm cells in the OLD frame, so
    #    after a relocation those cells forbid innocent fresh surface while the
    #    genuinely ruined spots are no longer addressable at all. A lateral move
    #    IS the escape hatch the tracker's own message tells the agent to take,
    #    so arriving somewhere new means starting clean.
    try:
        from mast.core.tip_crash_tracker import get_tip_crash_tracker

        get_tip_crash_tracker().note_recovery()
        out["tip_crash_tracker"] = "cleared"
    except Exception as exc:  # noqa: BLE001
        logger.debug("coarse-move effects: crash tracker clear failed: %s", exc)
        out["tip_crash_tracker"] = "failed"

    logger.info("粗动换区后已失效的缓存 (%s): %s", source or "?", out)
    return out
