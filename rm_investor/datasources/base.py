"""Abstract data-source interfaces.

Each capability is a small ABC. Concrete adapters (yfinance, Finnhub, Alpha Vantage,
SEC EDGAR, congress datasets) implement one or more of these. The registry wires the
configured adapter to each capability, so scoring/UI depend only on these interfaces.

Every method must degrade gracefully: return None / [] on failure rather than raising,
because free data sources are flaky and rate-limited.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..models import Fundamentals, NewsItem, Quote, Rating, Trade


class PriceSource(ABC):
    name: str = "price"

    @abstractmethod
    def get_quote(self, ticker: str) -> Optional[Quote]:
        ...

    @abstractmethod
    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Return an OHLCV DataFrame indexed by date (may be empty)."""
        ...


class FundamentalsSource(ABC):
    name: str = "fundamentals"

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        ...


class RatingsSource(ABC):
    name: str = "ratings"

    @abstractmethod
    def get_rating(self, ticker: str) -> Optional[Rating]:
        ...


class NewsSource(ABC):
    name: str = "news"

    @abstractmethod
    def get_news(self, ticker: str, limit: int = 20) -> list[NewsItem]:
        ...


class SmartMoneySource(ABC):
    name: str = "smart_money"

    @abstractmethod
    def get_trades(self, ticker: Optional[str] = None, limit: int = 200) -> list[Trade]:
        """Recent disclosed trades. If ticker is None, return across all names."""
        ...
