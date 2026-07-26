"""Analysis glue: assemble all data for a ticker, then score it.

`gather()` pulls quote, fundamentals, rating, price history (-> technicals), news,
fair value, and disclosed smart-money trades into one `TickerData` bundle. The scoring
functions take that bundle, so the UI and the scanner share exactly one data path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .datasources.registry import DataHub, get_hub
from .models import Fundamentals, NewsItem, Quote, Rating, Score, Trade
from .models import FairValue
from .scoring.factors import technicals
from .valuation import fair_value as compute_fair_value


@dataclass
class TickerData:
    ticker: str
    quote: Optional[Quote] = None
    fundamentals: Optional[Fundamentals] = None
    rating: Optional[Rating] = None
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    tech: dict = field(default_factory=dict)
    news: list[NewsItem] = field(default_factory=list)
    fair_value: Optional[FairValue] = None
    trades: list[Trade] = field(default_factory=list)


def gather(
    ticker: str,
    hub: Optional[DataHub] = None,
    *,
    with_history: bool = True,
    with_news: bool = True,
    with_trades: bool = True,
    with_valuation: bool = True,
    prefer_yf_ratings: bool = False,
) -> TickerData:
    hub = hub or get_hub()
    ticker = ticker.upper()

    quote = hub.quote(ticker)
    fundamentals = hub.fundamentals(ticker)
    rating = hub.rating(ticker, prefer_yf=prefer_yf_ratings)

    history = hub.history(ticker, period="1y", interval="1d") if with_history else pd.DataFrame()
    tech = technicals(history) if with_history else {}

    news = hub.news(ticker, limit=15) if with_news else []
    trades = hub.smart_money(ticker) if with_trades else []
    fv = compute_fair_value(fundamentals, quote, ticker) if with_valuation else None

    return TickerData(
        ticker=ticker,
        quote=quote,
        fundamentals=fundamentals,
        rating=rating,
        history=history,
        tech=tech,
        news=news,
        fair_value=fv,
        trades=trades,
    )


def score_sell(position, hub: Optional[DataHub] = None) -> tuple[Score, TickerData]:
    from .scoring.sell_signals import score_holding

    data = gather(position.ticker, hub)
    return score_holding(position, data), data


def score_opportunity(ticker: str, hub: Optional[DataHub] = None,
                      data: Optional[TickerData] = None) -> tuple[Score, TickerData]:
    from .scoring.opportunity import score_candidate

    data = data or gather(ticker, hub)
    return score_candidate(ticker, data), data


def score_moonshot_ticker(ticker: str, hub: Optional[DataHub] = None,
                          data: Optional[TickerData] = None) -> tuple[Score, TickerData]:
    from .scoring.moonshot import score_moonshot

    data = data or gather(ticker, hub)
    return score_moonshot(ticker, data), data
