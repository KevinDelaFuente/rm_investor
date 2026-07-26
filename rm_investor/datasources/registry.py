"""Provider registry + DataHub facade.

`registry.py` reads config.providers to decide which adapter backs each capability,
then exposes a single `DataHub` that the UI and scoring use. The hub always falls back
to yfinance when a configured (paid/keyed) provider returns nothing, so the app is
useful out of the box with zero API keys.
"""
from __future__ import annotations

import functools
from typing import Optional

import pandas as pd

from ..config import get_config
from ..models import Fundamentals, NewsItem, Quote, Rating, Trade
from .cache import install_http_cache
from .yfinance_source import YFinanceSource

_cfg = get_config()
install_http_cache(_cfg.path("cache.http_ttl_seconds", 900))


def _make(name: str):
    """Instantiate an adapter by config name. Returns None if unavailable."""
    try:
        if name == "yfinance":
            return YFinanceSource()
        if name == "finnhub":
            from .finnhub_source import FinnhubSource

            return FinnhubSource()
        if name == "alphavantage":
            from .alphavantage_source import AlphaVantageSource

            return AlphaVantageSource()
        if name == "fmp":
            from .fmp_source import FmpSource

            return FmpSource()
        if name == "congress":
            from .congress import CongressSource

            return CongressSource()
        if name == "quiver_congress":
            from .quiver_source import QuiverCongressSource

            return QuiverCongressSource()
        if name == "fmp_congress":
            from .fmp_source import FmpCongressSource

            return FmpCongressSource()
        if name == "sec_edgar":
            from .sec_edgar import SecEdgarSource

            return SecEdgarSource()
    except Exception:
        return None
    return None


class DataHub:
    """Single entry point for all market data, with yfinance fallback."""

    def __init__(self):
        p = _cfg.get("providers", {})
        self._yf = YFinanceSource()
        self._price = _make(p.get("price", "yfinance")) or self._yf
        self._fundamentals = _make(p.get("fundamentals", "yfinance"))
        self._ratings = _make(p.get("ratings", "yfinance"))
        self._news = _make(p.get("news", "yfinance"))
        self._smart_money = [
            s for s in (_make(n) for n in (p.get("smart_money") or [])) if s is not None
        ]

    # ----- price / history ------------------------------------------------- #
    def quote(self, ticker: str) -> Optional[Quote]:
        q = self._price.get_quote(ticker)
        return q or (self._yf.get_quote(ticker) if self._price is not self._yf else None)

    def history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        df = self._price.get_history(ticker, period, interval)
        if (df is None or df.empty) and self._price is not self._yf:
            df = self._yf.get_history(ticker, period, interval)
        return df if df is not None else pd.DataFrame()

    # ----- fundamentals ---------------------------------------------------- #
    def fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        f = self._fundamentals.get_fundamentals(ticker) if self._fundamentals else None
        if f is None:
            f = self._yf.get_fundamentals(ticker)
        return f

    # ----- ratings --------------------------------------------------------- #
    def rating(self, ticker: str, prefer_yf: bool = False) -> Optional[Rating]:
        """Analyst rating. prefer_yf=True forces the free yfinance source first
        (used for the 97-holding Sell Signals page to conserve FMP quota)."""
        if prefer_yf:
            yr = self._yf.get_rating(ticker)
            if yr is not None and yr.target_mean is not None:
                return yr
            # yfinance had nothing usable — fall through to the configured provider.
        r = self._ratings.get_rating(ticker) if self._ratings else None
        # Fall back to yfinance if the primary produced nothing usable.
        if r is None or r.target_mean is None:
            yr = self._yf.get_rating(ticker)
            r = r or yr
            if r is not None and r.target_mean is None and yr is not None:
                r = yr
        return r

    # ----- news ------------------------------------------------------------ #
    def news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        items = self._news.get_news(ticker, limit) if self._news else []
        if not items:
            items = self._yf.get_news(ticker, limit)
        return items

    # ----- smart money ----------------------------------------------------- #
    def smart_money(self, ticker: Optional[str] = None, limit: int = 200) -> list[Trade]:
        out: list[Trade] = []
        for src in self._smart_money:
            try:
                out += src.get_trades(ticker, limit)
            except Exception:
                continue
        return out

    @property
    def has_smart_money(self) -> bool:
        return bool(self._smart_money)


@functools.lru_cache(maxsize=1)
def get_hub() -> DataHub:
    return DataHub()
