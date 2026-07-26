"""Alpha Vantage adapter — fundamentals (OVERVIEW) + news sentiment (NEWS_SENTIMENT).

Requires ALPHAVANTAGE_API_KEY. Free tier is heavily rate-limited (~25 req/day), so
results are cached aggressively and failures degrade to None/[]. Uses plain `requests`
so the process-wide requests-cache also applies.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import requests

from ..config import get_api_key, get_config
from ..models import Fundamentals, NewsItem
from .base import FundamentalsSource, NewsSource
from .cache import disk_cache

_cfg = get_config()
_FUND_TTL = _cfg.path("cache.fundamentals_ttl_seconds", 86400)
_NEWS_TTL = _cfg.path("cache.news_ttl_seconds", 3600)
_BASE = "https://www.alphavantage.co/query"


def _to_float(v) -> Optional[float]:
    try:
        if v in (None, "None", "-", ""):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


@disk_cache(_FUND_TTL)
def _fetch_overview(ticker: str) -> dict:
    key = get_api_key("ALPHAVANTAGE_API_KEY")
    if not key:
        return {}
    try:
        r = requests.get(
            _BASE,
            params={"function": "OVERVIEW", "symbol": ticker, "apikey": key},
            timeout=15,
        )
        data = r.json()
        return data if isinstance(data, dict) and data.get("Symbol") else {}
    except Exception:
        return {}


@disk_cache(_NEWS_TTL)
def _fetch_news_sentiment(ticker: str, limit: int) -> list[dict]:
    key = get_api_key("ALPHAVANTAGE_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            _BASE,
            params={"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": limit, "apikey": key},
            timeout=20,
        )
        data = r.json()
        return data.get("feed", []) if isinstance(data, dict) else []
    except Exception:
        return []


class AlphaVantageSource(FundamentalsSource, NewsSource):
    name = "alphavantage"

    def get_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        o = _fetch_overview(ticker)
        if not o:
            return None
        return Fundamentals(
            ticker=ticker,
            name=o.get("Name"),
            sector=o.get("Sector"),
            industry=o.get("Industry"),
            market_cap=_to_float(o.get("MarketCapitalization")),
            beta=_to_float(o.get("Beta")),
            trailing_pe=_to_float(o.get("PERatio")),
            forward_pe=_to_float(o.get("ForwardPE")),
            price_to_sales=_to_float(o.get("PriceToSalesRatioTTM")),
            price_to_book=_to_float(o.get("PriceToBookRatio")),
            ev_to_ebitda=_to_float(o.get("EVToEBITDA")),
            eps_trailing=_to_float(o.get("EPS")),
            book_value_per_share=_to_float(o.get("BookValue")),
            revenue=_to_float(o.get("RevenueTTM")),
            profit_margin=_to_float(o.get("ProfitMargin")),
            ebitda=_to_float(o.get("EBITDA")),
            dividend_rate=_to_float(o.get("DividendPerShare")),
            dividend_yield=_to_float(o.get("DividendYield")),
        )

    def get_news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        items: list[NewsItem] = []
        for n in _fetch_news_sentiment(ticker, limit):
            title = n.get("title")
            if not title:
                continue
            # Per-ticker sentiment score, when present.
            sent = None
            for ts in n.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    sent = _to_float(ts.get("ticker_sentiment_score"))
                    break
            if sent is None:
                sent = _to_float(n.get("overall_sentiment_score"))
            published = None
            if n.get("time_published"):
                try:
                    published = datetime.strptime(n["time_published"], "%Y%m%dT%H%M%S")
                except Exception:
                    published = None
            items.append(
                NewsItem(
                    ticker=ticker,
                    headline=title,
                    source=n.get("source"),
                    url=n.get("url"),
                    published=published,
                    summary=n.get("summary"),
                    sentiment=sent,
                )
            )
        return items
