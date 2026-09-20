"""Typed schemas for the 用量·花销 (API cost/usage) endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class UsageAggRow(BaseModel):
    key: str
    currency: str
    count: int
    cost: float
    input_tokens: int = 0
    output_tokens: int = 0
    all_priced: bool = True


class CurrencyTotal(BaseModel):
    cost: float
    count: int
    estimated_cost: float = 0.0


class UsageSummary(BaseModel):
    range: str
    since: Optional[float] = None
    until: Optional[float] = None
    count: int
    by_currency: dict[str, CurrencyTotal] = Field(default_factory=dict)
    by_provider: list[UsageAggRow] = Field(default_factory=list)
    by_model: list[UsageAggRow] = Field(default_factory=list)
    by_source: list[UsageAggRow] = Field(default_factory=list)
    by_kind: list[UsageAggRow] = Field(default_factory=list)
    combined_cny: Optional[float] = None
    usd_to_cny: Optional[float] = None


class UsageEvent(BaseModel):
    ts: float
    kind: str
    provider: str
    model: str
    source: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    chars: int = 0
    seconds: float = 0.0
    cost: float = 0.0
    currency: str = "CNY"
    cost_known: bool = True


class RecentUsage(BaseModel):
    events: list[UsageEvent] = Field(default_factory=list)


class ResetResult(BaseModel):
    deleted: int


class PriceRow(BaseModel):
    model: str
    currency: str = "CNY"
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    char_per_m: float = 0.0
    per_minute: float = 0.0
    note: str = ""


class PricingBook(BaseModel):
    models: list[PriceRow] = Field(default_factory=list)
    usd_to_cny: float = 7.2


class PricingUpdate(BaseModel):
    """Operator price edits — merged over the defaults on save."""
    models: list[PriceRow] = Field(default_factory=list)
    usd_to_cny: Optional[float] = None


__all__ = [
    "UsageAggRow", "CurrencyTotal", "UsageSummary", "UsageEvent",
    "RecentUsage", "ResetResult", "PriceRow", "PricingBook", "PricingUpdate",
]
