"""yfinance adapter — the free, no-key default backing every capability.

Prices are delayed ~15 min (acceptable per project scope). A single cached
`.info` fetch per ticker feeds fundamentals and ratings to minimize calls.
Everything is wrapped so a failed/rate-limited fetch degrades to None/[]/empty.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from ..config import get_config
from ..models import Fundamentals, NewsItem, Quote, Rating
from .base import FundamentalsSource, NewsSource, PriceSource, RatingsSource
from .cache import disk_cache

_cfg = get_config()
_QUOTE_TTL = _cfg.path("cache.quote_ttl_seconds", 900)
_FUND_TTL = _cfg.path("cache.fundamentals_ttl_seconds", 86400)
_NEWS_TTL = _cfg.path("cache.news_ttl_seconds", 3600)


def _safe(d: dict, *keys: str) -> Any:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


@disk_cache(_FUND_TTL)
def _fetch_info(ticker: str) -> dict:
    """Cached `.info` dict (rich but slow/flaky). Empty dict on failure."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        return dict(info) if info else {}
    except Exception:
        return {}


@disk_cache(_QUOTE_TTL)
def _fetch_fast(ticker: str) -> dict:
    """Cached fast_info (last price / previous close) — cheaper than .info."""
    try:
        import yfinance as yf

        fi = yf.Ticker(ticker).fast_info
        # fast_info is a lazy mapping; coerce the fields we need.
        return {
            "last_price": getattr(fi, "last_price", None) or fi.get("lastPrice"),
            "previous_close": getattr(fi, "previous_close", None) or fi.get("previousClose"),
            "currency": getattr(fi, "currency", None) or fi.get("currency") or "USD",
        }
    except Exception:
        return {}


class YFinanceSource(PriceSource, FundamentalsSource, RatingsSource, NewsSource):
    name = "yfinance"

    # ----- Prices ---------------------------------------------------------- #
    def get_quote(self, ticker: str) -> Optional[Quote]:
        fast = _fetch_fast(ticker)
        price = fast.get("last_price")
        prev = fast.get("previous_close")
        currency = fast.get("currency", "USD")

        if price is None:
            # Fall back to the last two closes from history.
            hist = self.get_history(ticker, period="5d", interval="1d")
            if not hist.empty and "Close" in hist:
                closes = hist["Close"].dropna()
                if len(closes):
                    price = float(closes.iloc[-1])
                    prev = float(closes.iloc[-2]) if len(closes) > 1 else price
        if price is None:
            return None
        return Quote(
            ticker=ticker,
            price=float(price),
            previous_close=float(prev) if prev is not None else None,
            currency=currency,
            as_of=datetime.now(timezone.utc),
        )

    @disk_cache(_QUOTE_TTL)
    def _history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            import yfinance as yf

            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        return self._history(ticker, period, interval)

    # ----- Fundamentals ---------------------------------------------------- #
    def get_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        info = _fetch_info(ticker)
        if not info:
            return None
        return Fundamentals(
            ticker=ticker,
            name=_safe(info, "longName", "shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            currency=info.get("currency", "USD"),
            market_cap=info.get("marketCap"),
            shares_outstanding=info.get("sharesOutstanding"),
            beta=info.get("beta"),
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            price_to_sales=_safe(info, "priceToSalesTrailing12Months"),
            price_to_book=info.get("priceToBook"),
            ev_to_ebitda=info.get("enterpriseToEbitda"),
            eps_trailing=info.get("trailingEps"),
            book_value_per_share=info.get("bookValue"),
            revenue=_safe(info, "totalRevenue"),
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            profit_margin=info.get("profitMargins"),
            ebitda=info.get("ebitda"),
            free_cash_flow=info.get("freeCashflow"),
            operating_cash_flow=info.get("operatingCashflow"),
            total_cash=info.get("totalCash"),
            total_debt=info.get("totalDebt"),
            dividend_rate=info.get("dividendRate"),
            dividend_yield=info.get("dividendYield"),
            avg_volume=_safe(info, "averageVolume", "averageDailyVolume10Day"),
        )

    # ----- Ratings --------------------------------------------------------- #
    def get_rating(self, ticker: str) -> Optional[Rating]:
        info = _fetch_info(ticker)
        if not info:
            return None
        rating = Rating(
            ticker=ticker,
            analyst_count=info.get("numberOfAnalystOpinions"),
            target_low=info.get("targetLowPrice"),
            target_high=info.get("targetHighPrice"),
            target_mean=info.get("targetMeanPrice"),
            target_median=info.get("targetMedianPrice"),
            recommendation_mean=info.get("recommendationMean"),
            recommendation_key=info.get("recommendationKey"),
        )
        self._augment_recommendations(ticker, rating)
        return rating

    @staticmethod
    def _augment_recommendations(ticker: str, rating: Rating) -> None:
        """Fill the buy/hold/sell distribution from the recommendations table."""
        try:
            import yfinance as yf

            rec = yf.Ticker(ticker).recommendations
            if rec is None or getattr(rec, "empty", True):
                return
            row = rec.iloc[0]  # most recent period ("0m")

            def g(*names):
                for n in names:
                    if n in row and pd.notna(row[n]):
                        return int(row[n])
                return None

            rating.strong_buy = g("strongBuy")
            rating.buy = g("buy")
            rating.hold = g("hold")
            rating.sell = g("sell")
            rating.strong_sell = g("strongSell")
        except Exception:
            pass

    # ----- News ------------------------------------------------------------ #
    def get_news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        return _fetch_yf_news(ticker, limit)


@disk_cache(_NEWS_TTL)
def _fetch_yf_news(ticker: str, limit: int) -> list[NewsItem]:
    try:
        import yfinance as yf

        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items: list[NewsItem] = []
    for n in raw[:limit]:
        # yfinance news schema shifted; support both flat and nested "content".
        content = n.get("content", n)
        title = content.get("title") or n.get("title")
        if not title:
            continue
        ts = n.get("providerPublishTime")
        published = None
        if ts:
            try:
                published = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except Exception:
                published = None
        url = None
        if isinstance(content.get("canonicalUrl"), dict):
            url = content["canonicalUrl"].get("url")
        url = url or content.get("link") or n.get("link")
        provider = content.get("provider")
        if isinstance(provider, dict):
            provider = provider.get("displayName")
        items.append(
            NewsItem(
                ticker=ticker,
                headline=title,
                source=provider or n.get("publisher"),
                url=url,
                published=published,
                summary=content.get("summary"),
            )
        )
    return items
