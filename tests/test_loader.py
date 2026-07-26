"""Portfolio loader tests — CSV parsing and P&L math (no network)."""
from __future__ import annotations

from rm_investor.models import Holding, Position, Quote
from rm_investor.portfolio.loader import load_holdings, portfolio_summary


def test_load_sample_holdings():
    holdings = load_holdings()  # reads the repo's holdings.csv
    assert len(holdings) >= 1
    assert all(isinstance(h, Holding) for h in holdings)
    assert all(h.shares > 0 for h in holdings)


def test_position_pl_math():
    pos = Position(
        holding=Holding(ticker="ABC", shares=10, cost_basis=100.0),
        quote=Quote(ticker="ABC", price=150.0, previous_close=148.0),
    )
    assert pos.market_value == 1500.0
    assert pos.cost_value == 1000.0
    assert pos.unrealized_pl == 500.0
    assert pos.unrealized_pl_pct == 50.0


def test_portfolio_summary_aggregates():
    positions = [
        Position(holding=Holding(ticker="A", shares=10, cost_basis=100.0),
                 quote=Quote(ticker="A", price=110.0)),
        Position(holding=Holding(ticker="B", shares=5, cost_basis=200.0),
                 quote=Quote(ticker="B", price=180.0)),
    ]
    s = portfolio_summary(positions)
    assert s["market_value"] == 10 * 110 + 5 * 180
    assert s["cost_value"] == 10 * 100 + 5 * 200
    assert s["num_positions"] == 2


def test_missing_price_does_not_crash_summary():
    positions = [
        Position(holding=Holding(ticker="A", shares=10, cost_basis=100.0),
                 quote=Quote(ticker="A", price=None)),
    ]
    s = portfolio_summary(positions)
    assert s["market_value"] == 0.0
    assert s["num_positions"] == 1
