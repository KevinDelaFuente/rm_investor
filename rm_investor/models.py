"""Normalized data models shared across the whole app.

Every data-source adapter returns these types, so scoring, valuation, and the UI
never see provider-specific shapes. All financial fields are Optional because free
sources frequently omit them — downstream code must degrade gracefully, never assume.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
class Holding(BaseModel):
    """A raw line from holdings.csv."""
    ticker: str
    shares: float
    cost_basis: float  # per-share average cost


class Quote(BaseModel):
    ticker: str
    price: Optional[float] = None
    previous_close: Optional[float] = None
    currency: str = "USD"
    as_of: Optional[datetime] = None

    @property
    def change(self) -> Optional[float]:
        if self.price is None or self.previous_close is None:
            return None
        return self.price - self.previous_close

    @property
    def change_pct(self) -> Optional[float]:
        if self.change is None or not self.previous_close:
            return None
        return 100.0 * self.change / self.previous_close


class Position(BaseModel):
    """A holding enriched with a live-ish quote and computed P&L."""
    holding: Holding
    quote: Quote

    @property
    def ticker(self) -> str:
        return self.holding.ticker

    @property
    def price(self) -> Optional[float]:
        return self.quote.price

    @property
    def market_value(self) -> Optional[float]:
        if self.price is None:
            return None
        return self.price * self.holding.shares

    @property
    def cost_value(self) -> float:
        return self.holding.cost_basis * self.holding.shares

    @property
    def unrealized_pl(self) -> Optional[float]:
        if self.market_value is None:
            return None
        return self.market_value - self.cost_value

    @property
    def unrealized_pl_pct(self) -> Optional[float]:
        if self.unrealized_pl is None or not self.cost_value:
            return None
        return 100.0 * self.unrealized_pl / self.cost_value

    # weight is portfolio-relative, filled in by the loader once all positions exist
    weight: Optional[float] = None


# --------------------------------------------------------------------------- #
# Reference / market data
# --------------------------------------------------------------------------- #
class Fundamentals(BaseModel):
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"

    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    beta: Optional[float] = None

    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_sales: Optional[float] = None
    price_to_book: Optional[float] = None
    ev_to_ebitda: Optional[float] = None

    eps_trailing: Optional[float] = None
    book_value_per_share: Optional[float] = None
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None      # fraction, e.g. 0.18 == 18%
    earnings_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    ebitda: Optional[float] = None

    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    total_cash: Optional[float] = None
    total_debt: Optional[float] = None

    dividend_rate: Optional[float] = None       # $/share/yr
    dividend_yield: Optional[float] = None       # fraction

    avg_volume: Optional[float] = None


class Rating(BaseModel):
    """Analyst coverage — depth matters, not just the average target."""
    ticker: str
    analyst_count: Optional[int] = None
    target_low: Optional[float] = None
    target_high: Optional[float] = None
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    # recommendation distribution (counts), when available
    strong_buy: Optional[int] = None
    buy: Optional[int] = None
    hold: Optional[int] = None
    sell: Optional[int] = None
    strong_sell: Optional[int] = None
    recommendation_mean: Optional[float] = None   # 1=Strong Buy ... 5=Strong Sell
    recommendation_key: Optional[str] = None
    # month-over-month change in net bullishness, if a trend is available
    trend_delta: Optional[float] = None

    @property
    def target_range_str(self) -> str:
        def f(x):
            return f"${x:,.2f}" if x is not None else "—"
        return f"{f(self.target_low)} / {f(self.target_mean)} / {f(self.target_high)}"


class NewsItem(BaseModel):
    ticker: str
    headline: str
    source: Optional[str] = None
    url: Optional[str] = None
    published: Optional[datetime] = None
    summary: Optional[str] = None
    sentiment: Optional[float] = None   # -1..1 when the source provides it


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


class ActorType(str, Enum):
    CONGRESS = "congress"
    FUND = "fund"


class Trade(BaseModel):
    """A disclosed (lagged, public) trade — congress filing or 13F change."""
    ticker: Optional[str] = None
    issuer: Optional[str] = None    # company name (13F reports issuer, not always ticker)
    cusip: Optional[str] = None
    actor: str                      # person or fund name
    actor_type: ActorType
    side: TradeSide = TradeSide.UNKNOWN
    amount: Optional[str] = None    # free-text range for congress ("$1,001 - $15,000")
    shares_delta: Optional[float] = None   # for 13F position changes
    value: Optional[float] = None
    transaction_date: Optional[date] = None
    disclosed_date: Optional[date] = None
    source: Optional[str] = None


# --------------------------------------------------------------------------- #
# Valuation
# --------------------------------------------------------------------------- #
class FairValue(BaseModel):
    ticker: str
    price: Optional[float] = None
    methods: dict[str, float] = Field(default_factory=dict)   # method -> per-share value
    inputs: dict[str, dict] = Field(default_factory=dict)     # method -> assumptions used
    notes: list[str] = Field(default_factory=list)            # skipped methods / caveats
    low: Optional[float] = None
    base: Optional[float] = None
    high: Optional[float] = None

    @property
    def margin_of_safety(self) -> Optional[float]:
        """(base fair value - price) / price. Positive => undervalued."""
        if self.base is None or not self.price:
            return None
        return (self.base - self.price) / self.price


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class FactorScore(BaseModel):
    name: str
    value: float           # 0..100
    weight: float
    detail: str = ""

    @property
    def contribution(self) -> float:
        return self.value * self.weight


class Score(BaseModel):
    ticker: str
    kind: str              # "sell" | "opportunity" | "moonshot" | "holdings"
    composite: float       # 0..100
    band: str              # e.g. SELL/TRIM/HOLD or STRONG/WATCH/PASS
    factors: list[FactorScore] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)   # risk flags (moonshots) etc.
    # Two-sided holdings scoring (kind == "holdings"); None for one-sided scores.
    buy_score: Optional[float] = None        # 0..100 strength of the add case
    sell_score: Optional[float] = None       # 0..100 strength of the exit case
    conviction: Optional[float] = None       # buy_score - sell_score, -100..+100
