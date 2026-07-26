"""Finnhub adapter — analyst ratings/targets + company news (free tier).

Requires FINNHUB_API_KEY. If the key or the `finnhub` package is absent, every
method degrades to None/[] so the registry can fall back to yfinance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..config import get_api_key, get_config
from ..models import NewsItem, Rating
from .base import NewsSource, RatingsSource
from .cache import disk_cache

_cfg = get_config()
_RATINGS_TTL = _cfg.path("cache.ratings_ttl_seconds", 43200)
_NEWS_TTL = _cfg.path("cache.news_ttl_seconds", 3600)


def _client():
    key = get_api_key("FINNHUB_API_KEY")
    if not key:
        return None
    try:
        import finnhub

        return finnhub.Client(api_key=key)
    except Exception:
        return None


@disk_cache(_RATINGS_TTL)
def _fetch_rating(ticker: str) -> Optional[dict]:
    client = _client()
    if client is None:
        return None
    try:
        target = client.price_target(ticker) or {}
        recs = client.recommendation_trends(ticker) or []
        latest = recs[0] if recs else {}
        prev = recs[1] if len(recs) > 1 else {}
        return {"target": target, "latest": latest, "prev": prev}
    except Exception:
        return None


@disk_cache(_NEWS_TTL)
def _fetch_news(ticker: str, limit: int) -> list[dict]:
    client = _client()
    if client is None:
        return []
    try:
        # Finnhub company_news needs a date window (YYYY-MM-DD).
        from datetime import date, timedelta

        today = date.today()
        start = today - timedelta(days=30)
        return (client.company_news(ticker, _from=start.isoformat(), to=today.isoformat()) or [])[:limit]
    except Exception:
        return []


def _net_bullish(rec: dict) -> Optional[float]:
    if not rec:
        return None
    return (rec.get("strongBuy", 0) + rec.get("buy", 0)) - (rec.get("sell", 0) + rec.get("strongSell", 0))


class FinnhubSource(RatingsSource, NewsSource):
    name = "finnhub"

    def get_rating(self, ticker: str) -> Optional[Rating]:
        data = _fetch_rating(ticker)
        if not data:
            return None
        target = data.get("target") or {}
        latest = data.get("latest") or {}
        prev = data.get("prev") or {}
        count = None
        if latest:
            count = sum(
                int(latest.get(k, 0) or 0)
                for k in ("strongBuy", "buy", "hold", "sell", "strongSell")
            ) or None
        cur_net = _net_bullish(latest)
        prev_net = _net_bullish(prev)
        trend = (cur_net - prev_net) if (cur_net is not None and prev_net is not None) else None
        return Rating(
            ticker=ticker,
            analyst_count=count,
            target_low=target.get("targetLow"),
            target_high=target.get("targetHigh"),
            target_mean=target.get("targetMean"),
            target_median=target.get("targetMedian"),
            strong_buy=latest.get("strongBuy"),
            buy=latest.get("buy"),
            hold=latest.get("hold"),
            sell=latest.get("sell"),
            strong_sell=latest.get("strongSell"),
            trend_delta=trend,
        )

    def get_news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        items: list[NewsItem] = []
        for n in _fetch_news(ticker, limit):
            headline = n.get("headline")
            if not headline:
                continue
            published = None
            if n.get("datetime"):
                try:
                    published = datetime.fromtimestamp(int(n["datetime"]), tz=timezone.utc)
                except Exception:
                    published = None
            items.append(
                NewsItem(
                    ticker=ticker,
                    headline=headline,
                    source=n.get("source"),
                    url=n.get("url"),
                    published=published,
                    summary=n.get("summary"),
                )
            )
        return items
