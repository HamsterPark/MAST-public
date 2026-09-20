"""Shared per-point Nanonis signal acquisition for optical scans.

Both ``PumpProbeScan`` (delay sweep) and ``OpticalStageScan`` (stage raster)
step an optical axis and, at each point, read + average the same Nanonis
signals. That per-point machinery lives here so the two skills stay identical
where it matters and diverge only in HOW they move. The atomic
``AcquireSignalPoint`` skill exposes a single such measurement to the agent /
composite builder.

Nothing here touches the instruments registry — motion is the caller's job;
this module only reads Nanonis through ``context.safe_call``.
"""

from __future__ import annotations

import logging
import time

from mast.core.types import NanonisCallRecord
from mast.instruments.base import InstrumentError

logger = logging.getLogger(__name__)

__all__ = [
    "parse_indices",
    "nanonis_scalar",
    "signal_names",
    "mean_std",
    "resolve_lockin_index",
    "PointAcquirer",
]


# ── pure helpers (moved here as the single source; re-exported by callers) ──


def parse_indices(text: str) -> list[int]:
    """Parse a comma/space list of signal indices ('0, 14') → [0, 14],
    de-duplicated in order, each clamped to the valid 0-127 slot range."""
    out: list[int] = []
    for tok in str(text or "").replace(",", " ").split():
        try:
            idx = int(tok)
        except ValueError:
            continue
        if 0 <= idx <= 127 and idx not in out:
            out.append(idx)
    return out


def nanonis_scalar(record: NanonisCallRecord) -> float:
    """Extract the value from a SINGLE-SCALAR Nanonis reply triple
    ``(header, raw_bytes, [value])`` — Current_Get / Signals_ValGet shapes.

    Not for array-returning methods (Signals_ValsGet etc.), whose payload is
    size-prefixed (``[count, [values]]``) — the first numeric there is the
    length header, not a reading."""
    rv = record.return_value
    if isinstance(rv, (list, tuple)) and len(rv) > 2:
        rv = rv[2]
    stack = [rv]
    while stack:
        item = stack.pop(0)
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
    raise ValueError(f"no numeric value in Nanonis reply: {record.return_value!r}")


def signal_names(record: NanonisCallRecord) -> list[str]:
    """Extract the 128-channel name list from a Signals_NamesGet reply."""
    rv = record.return_value
    if isinstance(rv, (list, tuple)) and len(rv) > 2:
        rv = rv[2]
    stack = [rv]
    while stack:
        item = stack.pop(0)
        if (
            isinstance(item, (list, tuple))
            and item
            and all(isinstance(x, (str, bytes)) for x in item)
        ):
            return [x.decode() if isinstance(x, bytes) else str(x) for x in item]
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
    return []


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var**0.5


def resolve_lockin_index(context, index: int, demod: int) -> int:
    """Explicit index wins; otherwise match 'demod <n> … x' in the
    Signals_NamesGet channel names."""
    if index >= 0:
        return index
    record = context.safe_call("Signals_NamesGet")
    if record.error:
        raise InstrumentError(
            f"Signals_NamesGet failed while auto-discovering the lock-in "
            f"signal: {record.error}; pass the signal index explicitly"
        )
    names = signal_names(record)
    needle = f"demod {demod}"
    for i, name in enumerate(names):
        low = " ".join(name.lower().split())
        if needle in low and (" x" in low.split(needle, 1)[1][:4]):
            return i
    raise InstrumentError(
        f"no signal name matches 'Demod {demod} … X' among {len(names)} "
        "channels — pass the signal index explicitly (use ListSignalChannels)"
    )


# ── per-point acquirer ─────────────────────────────────────────────────────


class PointAcquirer:
    """Reads + averages a fixed set of Nanonis signals at the current point.

    Configured once (which signals, how many samples), then :meth:`acquire`
    is called at each scan point. Signals:

    - ``read_current`` → tunnel current via ``Current_Get`` → ``current_a`` /
      ``current_std`` columns;
    - ``lockin_index`` (or None) → a named lock-in demod slot via
      ``Signals_ValGet`` → ``lockin_v`` / ``lockin_std`` (used by
      PumpProbeScan; None for the generic stage scan);
    - ``extra_indices`` → any other slots → ``sig<i>_mean`` / ``sig<i>_std``.

    A pulse trigger is optional and separate: pass ``trigger=(port, line,
    width_s)`` to fire a TTL on a Nanonis digital line before each read
    (matching the lab VIs); :meth:`configure_trigger` sets that line to an
    active-high output once before the scan.
    """

    def __init__(
        self,
        context,
        *,
        read_current: bool = True,
        lockin_index: int | None = None,
        extra_indices=(),
        samples: int = 10,
        interval_s: float = 0.01,
        trigger: tuple[int, int, float] | None = None,
    ):
        self._ctx = context
        self.read_current = bool(read_current)
        self.lockin_index = lockin_index
        self.extra_indices = list(extra_indices)
        self.samples = int(samples)
        self.interval_s = float(interval_s)
        self.trigger = trigger

    # -- schema of the columns this acquirer produces --------------------------

    @property
    def value_columns(self) -> list[str]:
        cols: list[str] = []
        if self.read_current:
            cols += ["current_a", "current_std"]
        if self.lockin_index is not None:
            cols += ["lockin_v", "lockin_std"]
        for xi in self.extra_indices:
            cols += [f"sig{xi}_mean", f"sig{xi}_std"]
        return cols

    @property
    def plot_series(self) -> list[tuple]:
        """(mean_col, std_col, label) per acquired signal, for plotting."""
        s: list[tuple] = []
        if self.read_current:
            s.append(("current_a", "current_std", "Current (A)"))
        if self.lockin_index is not None:
            s.append(("lockin_v", "lockin_std", "Lock-in"))
        for xi in self.extra_indices:
            s.append((f"sig{xi}_mean", f"sig{xi}_std", f"Signal {xi}"))
        return s

    @property
    def is_empty(self) -> bool:
        return not (self.read_current or self.lockin_index is not None or self.extra_indices)

    # -- I/O ------------------------------------------------------------------

    def configure_trigger(self, nanonis_calls: list) -> str | None:
        """Set the trigger line to an active-high output once. Returns an error
        string on failure, or None."""
        if not self.trigger:
            return None
        port, line, _width = self.trigger
        rec = self._ctx.safe_call("DigLines_PropsSet", line, port, 1, 1)
        nanonis_calls.append(rec)
        return rec.error or None

    def acquire(self, nanonis_calls: list) -> dict:
        """Fire the optional trigger, then read+average every configured signal
        at the current point. Appends every call to ``nanonis_calls`` and
        raises :class:`InstrumentError` on any Nanonis error (so the caller can
        keep the partial scan)."""
        if self.trigger:
            port, line, width = self.trigger
            rec = self._ctx.safe_call("DigLines_Pulse", port, [line], width, 0.0, 1, 1)
            nanonis_calls.append(rec)
            if rec.error:
                raise InstrumentError(f"trigger pulse failed: {rec.error}")

        cur: list[float] = []
        li: list[float] = []
        extra: dict[int, list[float]] = {xi: [] for xi in self.extra_indices}
        for k in range(self.samples):
            if k and self.interval_s > 0:
                time.sleep(self.interval_s)
            if self.read_current:
                rec = self._ctx.safe_call("Current_Get")
                nanonis_calls.append(rec)
                if rec.error:
                    raise InstrumentError(f"Current_Get failed: {rec.error}")
                cur.append(nanonis_scalar(rec))
            if self.lockin_index is not None:
                rec = self._ctx.safe_call("Signals_ValGet", self.lockin_index, 0)
                nanonis_calls.append(rec)
                if rec.error:
                    raise InstrumentError(
                        f"Signals_ValGet({self.lockin_index}) failed: {rec.error}"
                    )
                li.append(nanonis_scalar(rec))
            for xi in self.extra_indices:
                rec = self._ctx.safe_call("Signals_ValGet", xi, 0)
                nanonis_calls.append(rec)
                if rec.error:
                    raise InstrumentError(f"Signals_ValGet({xi}) failed: {rec.error}")
                extra[xi].append(nanonis_scalar(rec))

        row: dict = {}
        if self.read_current:
            row["current_a"], row["current_std"] = mean_std(cur)
        if self.lockin_index is not None:
            row["lockin_v"], row["lockin_std"] = mean_std(li)
        for xi in self.extra_indices:
            row[f"sig{xi}_mean"], row[f"sig{xi}_std"] = mean_std(extra[xi])
        return row
