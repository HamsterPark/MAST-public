"""用量·花销 — API cost/usage endpoints (mast.billing ledger + price book).

Read-only aggregation + a reset + a view/edit of the price book. All degrade-safe:
the ledger never raises, so an empty or unwired install returns valid zeros.
"""
from __future__ import annotations

import datetime
import time

from fastapi import APIRouter

from mast.api.schemas_usage import (
    CurrencyTotal,
    PriceRow,
    PricingBook,
    PricingUpdate,
    RecentUsage,
    ResetResult,
    UsageAggRow,
    UsageEvent,
    UsageSummary,
)

router = APIRouter(tags=["usage"])


def _since_for(range_: str) -> float | None:
    now = time.time()
    if range_ == "today":
        start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()
    if range_ == "7d":
        return now - 7 * 86400
    if range_ == "30d":
        return now - 30 * 86400
    return None  # "all"


@router.get("/usage/summary", response_model=UsageSummary)
def usage_summary(range: str = "7d") -> UsageSummary:
    from mast.billing.ledger import get_ledger

    since = _since_for(range)
    s = get_ledger().summary(since=since)

    def rows(items: list[dict]) -> list[UsageAggRow]:
        return [UsageAggRow(**r) for r in items]

    return UsageSummary(
        range=range,
        since=s.get("since"),
        until=s.get("until"),
        count=s.get("count", 0),
        by_currency={k: CurrencyTotal(**v) for k, v in (s.get("by_currency") or {}).items()},
        by_provider=rows(s.get("by_provider") or []),
        by_model=rows(s.get("by_model") or []),
        by_source=rows(s.get("by_source") or []),
        by_kind=rows(s.get("by_kind") or []),
        combined_cny=s.get("combined_cny"),
        usd_to_cny=s.get("usd_to_cny"),
    )


@router.get("/usage/recent", response_model=RecentUsage)
def usage_recent(limit: int = 50, range: str = "all") -> RecentUsage:
    from mast.billing.ledger import get_ledger

    since = _since_for(range)
    events = get_ledger().recent(limit=max(1, min(limit, 500)), since=since)
    return RecentUsage(events=[UsageEvent(**e) for e in events])


@router.post("/usage/reset", response_model=ResetResult)
def usage_reset() -> ResetResult:
    from mast.billing.ledger import get_ledger

    return ResetResult(deleted=get_ledger().reset())


@router.get("/usage/pricing", response_model=PricingBook)
def usage_pricing() -> PricingBook:
    from mast.billing.pricing import load_pricing, usd_to_cny_rate

    book = load_pricing()
    return PricingBook(
        models=[PriceRow(model=k, **v.to_dict()) for k, v in sorted(book.items())],
        usd_to_cny=usd_to_cny_rate(),
    )


@router.post("/usage/pricing", response_model=PricingBook)
def usage_pricing_save(body: PricingUpdate) -> PricingBook:
    from mast.billing.pricing import load_pricing, save_pricing_override, usd_to_cny_rate

    models = {
        r.model: {
            "currency": r.currency, "input_per_m": r.input_per_m,
            "output_per_m": r.output_per_m, "char_per_m": r.char_per_m,
            "per_minute": r.per_minute, "note": r.note or "custom",
        }
        for r in body.models
    }
    save_pricing_override(models, usd_to_cny=body.usd_to_cny)
    book = load_pricing(force=True)
    return PricingBook(
        models=[PriceRow(model=k, **v.to_dict()) for k, v in sorted(book.items())],
        usd_to_cny=usd_to_cny_rate(),
    )
