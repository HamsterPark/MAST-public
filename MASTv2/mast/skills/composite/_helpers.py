"""Shared helpers for composite skills — K-vendor (Phase 4)."""

# K (Keep) — vendored verbatim from mast/skills/composite/_helpers.py

from __future__ import annotations

import logging
import time

from mast.core.types import NanonisCallRecord

logger = logging.getLogger(__name__)


def wait_scan_complete(
    context,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
    call_accumulator: list[NanonisCallRecord] | None = None,
) -> bool:
    """Poll Scan_StatusGet until scan finishes or timeout.

    Always checks ``context.check_abort()`` each iteration.  If aborted,
    stops the scan and raises ``AbortRequested``.

    Args:
        call_accumulator: If provided, poll NanonisCallRecords are appended
            to this list (typically ``self._all_calls`` in a CompositeSkill).

    Returns:
        ``True`` if scan finished, ``False`` if timeout.

    Raises:
        AbortRequested: if aborted during polling.
    """
    from mast.skills.composite._base import AbortRequested

    max_polls = int(timeout_s / poll_interval_s)
    for _ in range(max_polls):
        time.sleep(poll_interval_s)

        # Abort/pause checkpoint
        if hasattr(context, "check_abort") and context.check_abort():
            context.safe_call("Scan_Action", 1, 0)  # Stop scan (action=1)
            raise AbortRequested("wait_scan_complete")

        rec = context.safe_call("Scan_StatusGet")
        if call_accumulator is not None:
            call_accumulator.append(rec)
        if rec.error:
            continue

        parsed = rec.return_value
        # Scan_StatusGet returns (err, raw, [status]) where status 0 = not scanning
        status = (
            parsed[2][0]
            if isinstance(parsed, (list, tuple)) and len(parsed) > 2
            else parsed
        )
        if status == 0:
            return True

    return False
