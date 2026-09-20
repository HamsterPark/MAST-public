"""Spend probes that connect the billing ledger to the orchestrator's budget gate.

Why this module exists (2026-07-30)
-----------------------------------
The orchestrator has had a budget hard gate since —
``state["budget_remaining_usd"] <= 0 → END`` — and it was honest about being
CALLER-SEEDED: the graph never metered cost itself. In practice, **no caller ever seeded it.** Nothing in the tree writes that key, so the
gate has been inert for its entire life and the only real bound on one run's cost
was ``recursion_limit``: at 500 super-steps and the measured ~$0.20 per
instrument turn that works out to roughly **$50 per run**.

That is not a theoretical worry. The usage ledger holds a run that spent
**$30.66 over 60 minutes**, and its tail was four identical
``SUP → IC ‖ DP ‖ PW`` cycles that all **succeeded**. No loop guard looks at
money, and StallGuard keys off repeated FAILURE signatures — so a loop made of
successes is structurally invisible to it. Meanwhile the ledger's books were
correct the whole time (1910 real priced calls). The measurement existed, the
gate existed, and the two were never connected. This module connects them.

The accounting caveat, stated plainly
-------------------------------------
``usage_events`` has no ``run_id`` and no ``experiment_id`` column, so "what did
THIS run cost" cannot be answered exactly. What CAN be answered is "what has the
whole system spent since this run started", and that is what :class:`RunMeter`
reports. Both consequences follow from that, and neither is hidden:

  * a concurrent background run, or a private-chat turn, inside the window is
    counted against this run's budget — an **over**-estimate;
  * therefore the gate can fire slightly EARLY, never late.

Over-estimating is the safe direction for a spend ceiling, and the foreground
run-task slot is mutually exclusive (``routes/orchestrator.py``'s ``_busy_claim``),
so in practice the window holds one orchestrator plus whatever the operator does
by hand. This is deliberately NOT dressed up as per-run precision: adding a
run_id column and threading it through every call site is the honest fix if exact
attribution is ever needed, and that is a bigger change than a gate deserves.

Why a SECOND, daily dimension
-----------------------------
A per-run ceiling bounds one run. It does nothing about run COUNT — and the
wake-scheduler (``core/wake_scheduler.py``) exists precisely to create more runs:
ten woken runs at $8 each are $80 and every one of them is individually legal.
:func:`daily_spend_usd` is the dimension that notices that, and it is the only
one that can, because a woken run is a fresh state with a fresh budget.

Currency
--------
The ledger stores every row in the provider's OWN currency, and
``mast.billing.pricing`` deliberately refuses to invent exchange rates for
display. A GATE, however, needs one comparable number. USD-equivalent is computed
here with the configured rate and used **only** for that comparison — never
written back to the ledger, and labelled an estimate wherever it surfaces.

Every probe returns ``None`` when the ledger cannot be read. ``None`` means
"unknown", and every caller must treat it as "leave the gate inert" rather than
as zero — a billing hiccup must not manufacture a fake budget, in either
direction.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

#: Fallback CNY-per-USD if the pricing module cannot be consulted. Only reached
#: when pricing itself fails to import/read; a wrong-but-sane rate keeps the gate
#: roughly right, whereas raising would take the whole run down over billing.
_FALLBACK_USD_TO_CNY = 7.2


def _usd_to_cny() -> float:
    try:
        from mast.billing.pricing import usd_to_cny_rate

        r = float(usd_to_cny_rate() or 0.0)
        return r if r > 0 else _FALLBACK_USD_TO_CNY
    except Exception as exc:  # noqa: BLE001
        logger.debug("run_meter: usd_to_cny_rate unavailable (%s)", exc)
        return _FALLBACK_USD_TO_CNY


def spend_usd(since: float | None = None, until: float | None = None) -> float | None:
    """USD-equivalent spend booked in ``[since, until)``. ``None`` = unreadable.

    Sums every currency bucket the ledger reports, converting non-USD rows at the
    configured rate. Estimated-price rows (``cost_known=0``) are included: for a
    ceiling, an approximate cost is far better information than a silent zero.
    """
    try:
        from mast.billing.ledger import get_ledger

        summary = get_ledger().summary(since=since, until=until)
    except Exception as exc:  # noqa: BLE001 — billing never breaks the caller
        logger.debug("run_meter: ledger summary failed (%s)", exc)
        return None
    buckets = summary.get("by_currency") or {}
    if not isinstance(buckets, dict):
        return None
    rate = _usd_to_cny()
    total = 0.0
    for currency, agg in buckets.items():
        try:
            cost = float((agg or {}).get("cost") or 0.0)
        except (TypeError, ValueError):
            continue
        total += cost if str(currency).upper() == "USD" else cost / rate
    return round(total, 6)


def _local_midnight_ts(now: float | None = None) -> float:
    t = time.time() if now is None else float(now)
    d = datetime.fromtimestamp(t)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def daily_spend_usd(now: float | None = None) -> float | None:
    """USD-equivalent spend since local midnight, across EVERY run and chat.

    The cross-run dimension. A per-run ceiling cannot see run count; this can.
    Local midnight (not a rolling 24 h window) because the operator reasons about
    "today", and a ceiling they cannot predict is a ceiling they will disable.
    """
    return spend_usd(since=_local_midnight_ts(now))


class RunMeter:
    """Tracks one run's budget against ledger spend since the run began.

    Construct at run start, then call :meth:`remaining_usd` on every hop. See the
    module docstring for why "spend since t0" is the honest quantity here and why
    over-attribution is the safe direction.
    """

    def __init__(self, *, budget_usd: float, started_at: float | None = None) -> None:
        self.budget_usd = max(0.0, float(budget_usd or 0.0))
        self.started_at = float(started_at) if started_at is not None else time.time()
        # Ledger rows are timestamped by the CALLING process's clock, the same
        # clock as started_at, so no skew correction is needed. Pulled back by a
        # hair so a call already in flight when the run began still lands inside
        # the window rather than being spent for free.
        self._since = self.started_at - 1.0

    def spent_usd(self) -> float | None:
        """System-wide USD booked since this run started. ``None`` = unreadable."""
        return spend_usd(since=self._since)

    def remaining_usd(self) -> float | None:
        """Budget minus spend, floored at 0.0. ``None`` = unreadable → gate inert.

        Floored rather than allowed negative so the value can be shown to an
        operator without needing a sign convention explained; the gate fires at
        ``<= 0`` either way.
        """
        spent = self.spent_usd()
        if spent is None:
            return None
        return round(max(0.0, self.budget_usd - spent), 6)

    def as_dict(self) -> dict:
        """Snapshot for logs / API. ``spent``/``remaining`` may be ``None``."""
        spent = self.spent_usd()
        return {
            "budget_usd": self.budget_usd,
            "spent_usd": spent,
            "remaining_usd": (None if spent is None
                              else round(max(0.0, self.budget_usd - spent), 6)),
            "started_at": self.started_at,
            "basis": "system-wide spend since run start (see run_meter docstring)",
        }


def make_budget_probe(budget_usd: float,
                      started_at: float | None = None):
    """Return a zero-arg callable giving remaining USD (or ``None`` if unknown).

    This is the shape the orchestrator's ``build(budget_probe=…)`` takes: the
    graph must not import the billing layer (and must not care whether a ledger
    exists at all), so it receives a probe instead. A falsy budget yields ``None``
    — no budget configured means the gate stays inert, which is the pre-existing
    honest default and NOT a $0 ceiling that would end every run instantly.
    """
    try:
        budget = float(budget_usd or 0.0)
    except (TypeError, ValueError):
        budget = 0.0
    if budget <= 0:
        return lambda: None
    meter = RunMeter(budget_usd=budget, started_at=started_at)
    return meter.remaining_usd


__all__ = [
    "RunMeter", "spend_usd", "daily_spend_usd", "make_budget_probe",
]
