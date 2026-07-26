"""Pytest fixtures: synthetic data builders (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from rm_investor.analysis import TickerData
from rm_investor.models import (
    FairValue,
    Fundamentals,
    Holding,
    Position,
    Quote,
    Rating,
)


def make_fundamentals(**kw) -> Fundamentals:
    base = dict(
        ticker="TEST",
        name="Test Co",
        sector="Technology",
        market_cap=5e10,
        shares_outstanding=1e9,
        trailing_pe=20.0,
        eps_trailing=5.0,
        book_value_per_share=25.0,
        revenue=2e10,
        revenue_growth=0.10,
        ebitda=6e9,
        free_cash_flow=4e9,
        total_cash=3e9,
        total_debt=1e9,
        dividend_rate=2.0,
        profit_margin=0.15,
        beta=1.1,
        avg_volume=5_000_000,
    )
    base.update(kw)
    return Fundamentals(**base)


def make_data(**kw) -> TickerData:
    """Assemble a TickerData bundle with sensible defaults, overridable per test."""
    d = TickerData(ticker=kw.get("ticker", "TEST"))
    d.quote = kw.get("quote", Quote(ticker=d.ticker, price=100.0, previous_close=99.0))
    d.fundamentals = kw.get("fundamentals", make_fundamentals(ticker=d.ticker))
    d.rating = kw.get("rating", Rating(ticker=d.ticker, analyst_count=15, target_mean=110.0,
                                       target_low=90.0, target_high=130.0, recommendation_mean=2.2))
    d.tech = kw.get("tech", {"rsi": 55, "above_ma50": 0.03, "above_ma200": 0.10,
                             "ret_3m": 0.08, "vol_ratio": 1.0})
    d.news = kw.get("news", [])
    d.fair_value = kw.get("fair_value", FairValue(ticker=d.ticker, price=d.quote.price, base=115.0,
                                                  low=100.0, high=130.0, methods={"dcf": 115.0}))
    d.trades = kw.get("trades", [])
    return d


@pytest.fixture
def data_factory():
    return make_data


@pytest.fixture
def fundamentals_factory():
    return make_fundamentals


def make_position(ticker="TEST", shares=100, cost_basis=90.0, price=100.0, weight=10.0) -> Position:
    p = Position(holding=Holding(ticker=ticker, shares=shares, cost_basis=cost_basis),
                 quote=Quote(ticker=ticker, price=price, previous_close=price))
    p.weight = weight
    return p


@pytest.fixture
def position_factory():
    return make_position
