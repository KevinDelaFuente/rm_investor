"""Financial Modeling Prep (FMP) adapter — one key, many capabilities.

Backs analyst ratings/targets, news, and (via FmpCongressSource) congressional trades.
Requires FMP_API_KEY. FMP recently split into a new `/stable/` API and legacy
`/api/v3|v4/` endpoints; each call tries the stable path first, then legacy, so it
self-heals regardless of which your plan exposes. Degrades to None/[] without a key
or on error, so the registry falls back to yfinance.

FMP free tier is rate-limited (~250 req/day) and US-focused — results are cached.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import requests

from ..config import get_api_key, get_config
from ..models import ActorType, NewsItem, Rating, Trade, TradeSide
from .base import NewsSource, RatingsSource
from .cache import disk_cache

_cfg = get_config()
_RATINGS_TTL = _cfg.path("cache.ratings_ttl_seconds", 43200)
_NEWS_TTL = _cfg.path("cache.news_ttl_seconds", 3600)
_FILINGS_TTL = _cfg.path("cache.filings_ttl_seconds", 604800)
_LOOKBACK = int(_cfg.path("smart_money.congress.lookback_days", 180))
_BASE = "https://financialmodelingprep.com"


def _api_key() -> Optional[str]:
    return get_api_key("FMP_API_KEY")


@disk_cache(_RATINGS_TTL)
def _get(path: str, params: tuple) -> Optional[object]:
    """GET one FMP endpoint. params is a tuple of (k, v) pairs (hashable for cache)."""
    key = _api_key()
    if not key:
        return None
    try:
        q = {k: v for k, v in params}
        q["apikey"] = key
        r = requests.get(f"{_BASE}{path}", params=q, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        # FMP signals plan/endpoint problems with an error dict.
        if isinstance(data, dict) and ("Error Message" in data or "error" in data):
            return None
        return data
    except Exception:
        return None


def _first(candidates: list[tuple[str, tuple]]) -> Optional[object]:
    """Try endpoint variants (stable, then legacy); return the first with data."""
    for path, params in candidates:
        data = _get(path, params)
        if data:
            return data
    return None


def _f(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19] if "T" in fmt or ":" in fmt else s[:10], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _side(txn: Optional[str]) -> TradeSide:
    t = (txn or "").lower()
    if "purchase" in t or "buy" in t or "receive" in t:
        return TradeSide.BUY
    if "sale" in t or "sell" in t or "sold" in t:
        return TradeSide.SELL
    return TradeSide.UNKNOWN


class FmpSource(RatingsSource, NewsSource):
    """Analyst ratings/targets + news via FMP."""

    name = "fmp"

    def get_rating(self, ticker: str) -> Optional[Rating]:
        tk = ticker.upper()
        consensus = _first([
            ("/stable/price-target-consensus", (("symbol", tk),)),
            ("/api/v4/price-target-consensus", (("symbol", tk),)),
        ])
        grades = _first([
            ("/stable/grades-consensus", (("symbol", tk),)),
            ("/api/v4/upgrades-downgrades-consensus", (("symbol", tk),)),
        ])
        con = _one(consensus)
        gr = _one(grades)
        if not con and not gr:
            return None

        count = None
        if gr:
            count = sum(int(gr.get(k, 0) or 0) for k in
                        ("strongBuy", "buy", "hold", "sell", "strongSell")) or None
        return Rating(
            ticker=tk,
            analyst_count=count,
            target_low=_f(con.get("targetLow")) if con else None,
            target_high=_f(con.get("targetHigh")) if con else None,
            target_mean=_f(con.get("targetConsensus")) if con else None,
            target_median=_f(con.get("targetMedian")) if con else None,
            strong_buy=(int(gr["strongBuy"]) if gr and gr.get("strongBuy") is not None else None),
            buy=(int(gr["buy"]) if gr and gr.get("buy") is not None else None),
            hold=(int(gr["hold"]) if gr and gr.get("hold") is not None else None),
            sell=(int(gr["sell"]) if gr and gr.get("sell") is not None else None),
            strong_sell=(int(gr["strongSell"]) if gr and gr.get("strongSell") is not None else None),
        )

    def get_news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        tk = ticker.upper()
        raw = _first([
            ("/stable/news/stock", (("symbols", tk), ("limit", limit))),
            ("/api/v3/stock_news", (("tickers", tk), ("limit", limit))),
        ])
        items: list[NewsItem] = []
        for n in (raw or [])[:limit]:
            title = n.get("title")
            if not title:
                continue
            items.append(
                NewsItem(
                    ticker=tk,
                    headline=title,
                    source=n.get("site") or n.get("publisher"),
                    url=n.get("url"),
                    published=_to_dt(n.get("publishedDate")),
                    summary=n.get("text"),
                )
            )
        return items


_CONGRESS_PAGES = int(_cfg.path("smart_money.congress.pages", 3))
# FMP free tier caps `limit` at 25 and disables paging (page>0 returns nothing).
# Paid tiers allow more — bump page_limit / pages in config accordingly.
_PAGE_LIMIT = int(_cfg.path("smart_money.congress.page_limit", 25))


def _bulk_congress() -> list[Trade]:
    """Fetch recent Senate + House trades in bulk (paged), for local filtering.

    One shared fetch covers every ticker in a scan — O(pages) calls, not O(tickers).
    Each page is cached ~12h at the HTTP layer, so a whole scan reuses it. On the free
    tier only page 0 returns data, so the loop naturally stops after one page.
    """
    rows: list[Trade] = []
    for page in range(_CONGRESS_PAGES):
        senate = _first([
            ("/stable/senate-latest", (("page", page), ("limit", _PAGE_LIMIT))),
            ("/api/v4/senate-trading-rss-feed", (("page", page),)),
        ])
        house = _first([
            ("/stable/house-latest", (("page", page), ("limit", _PAGE_LIMIT))),
            ("/api/v4/senate-disclosure-rss-feed", (("page", page),)),
        ])
        added = False
        if senate:
            rows += _normalize_congress(senate, "Senate")
            added = True
        if house:
            rows += _normalize_congress(house, "House")
            added = True
        if not added:
            break   # no more pages / no data
    return rows


class FmpCongressSource:
    """SmartMoneySource for Senate + House trades via FMP (duck-typed).

    Uses a single bulk fetch (paged) and filters by ticker locally, so scanning N
    tickers costs O(pages) FMP calls, not O(N). Big saver on the free tier.
    """

    name = "fmp_congress"

    def get_trades(self, ticker: Optional[str] = None, limit: int = 200) -> list[Trade]:
        trades = _bulk_congress()
        cutoff = date.today() - timedelta(days=_LOOKBACK)
        trades = [t for t in trades if (t.transaction_date is None or t.transaction_date >= cutoff)]
        if ticker:
            tk = ticker.upper()
            trades = [t for t in trades if t.ticker == tk]
        trades.sort(key=lambda t: t.transaction_date or date.min, reverse=True)
        return trades[:limit]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _one(data) -> Optional[dict]:
    """FMP returns either a list (take first) or a dict."""
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        return data
    return None


def _to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19] if len(s) >= 19 else s[:10], fmt)
        except (ValueError, TypeError):
            continue
    return None


def _actor_name(row: dict) -> str:
    first = row.get("firstName") or ""
    last = row.get("lastName") or ""
    full = f"{first} {last}".strip()
    return full or row.get("representative") or row.get("office") or row.get("name") or "Unknown"


def _normalize_congress(rows, chamber: str) -> list[Trade]:
    out: list[Trade] = []
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        ticker = (row.get("symbol") or "").strip().upper() or None
        out.append(
            Trade(
                ticker=ticker,
                actor=_actor_name(row),
                actor_type=ActorType.CONGRESS,
                side=_side(row.get("type")),
                amount=row.get("amount"),
                transaction_date=_parse_date(row.get("transactionDate")),
                disclosed_date=_parse_date(row.get("disclosureDate")),
                source=f"FMP/{chamber}",
            )
        )
    return out
