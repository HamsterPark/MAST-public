"""Read the tip's lateral position at the instant a skill acts on the surface.

For the operations that leave a PERMANENT MARK — tip forming, a bias pulse —
*where* it happened is as much a part of the result as whether it worked. Those
skills take no position parameter (they act wherever the tip already is), so the
scan map used to place their markers from the cached ``HardwareState`` snapshot,
up to a refresh interval old. The avoidance model that keeps the tip out of
ruined patches is built entirely on those coordinates.

So each such skill reads the position itself, at the moment it fires, and puts it
in its result. The rule when the read fails is to report NOTHING: the recorder
still falls back to the snapshot and stamps ``pos_src`` accordingly, and a marker
that is honestly approximate beats one that is confidently wrong.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def read_tip_xy(context) -> tuple[float, float] | None:
    """Live FolMe tip position in metres, or None if it cannot be read.

    Never raises and never consumes a hardware token beyond the one read: a
    failure here must not change whether the calling skill succeeds. Mirrors
    ``mast.core.state`` for the response shape (``return_value[2]`` is the parsed
    variable list, ``[x, y]``)."""
    try:
        record = context.safe_call("FolMe_XYPosGet", 0)
    except Exception:  # noqa: BLE001 — position bookkeeping is never fatal
        logger.debug("FolMe_XYPosGet raised while stamping tip position",
                     exc_info=True)
        return None
    if record is None or getattr(record, "error", None):
        return None
    ret = getattr(record, "return_value", None)
    if not isinstance(ret, (list, tuple)) or len(ret) < 3:
        return None
    vals = ret[2]
    if not isinstance(vals, (list, tuple)) or len(vals) < 2:
        return None
    try:
        x, y = float(vals[0]), float(vals[1])
    except (TypeError, ValueError):
        return None
    # A non-finite or absurd coordinate is a parse artefact, not a position;
    # letting it through would put a marker light-years from the sample and
    # blow up every extent calculation downstream.
    if not (abs(x) < 1e-3 and abs(y) < 1e-3):
        return None
    if x != x or y != y:  # NaN
        return None
    return (x, y)


def tip_xy_fields(context) -> dict:
    """``{"x_m": …, "y_m": …}`` for a SkillResult's data, or ``{}``.

    Empty on failure so it can be splatted into a result dict unconditionally —
    absent keys mean "could not read", which is exactly what the recorder needs
    to know to fall back."""
    pos = read_tip_xy(context)
    return {} if pos is None else {"x_m": pos[0], "y_m": pos[1]}


__all__ = ["read_tip_xy", "tip_xy_fields"]
