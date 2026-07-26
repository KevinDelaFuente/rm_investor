"""Deterministic scoring tests — synthetic data, no network."""
from __future__ import annotations

from rm_investor.models import FairValue, Quote, Rating, Trade, TradeSide, ActorType, NewsItem
from rm_investor.scoring.factors import pe_pair, high_forward_pe_flag, f_pe_stretch, f_valuation_reasonable
from rm_investor.scoring.opportunity import score_candidate
from rm_investor.scoring.moonshot import score_moonshot
from rm_investor.scoring.sell_signals import score_holding


def test_pe_pair_keeps_positive(fundamentals_factory):
    t, f = pe_pair(fundamentals_factory(trailing_pe=60.0, forward_pe=20.0))
    assert (t, f) == (60.0, 20.0)
    t2, f2 = pe_pair(fundamentals_factory(trailing_pe=-5.0, forward_pe=None))
    assert (t2, f2) == (None, None)


def test_both_pe_contribute_to_stretch(fundamentals_factory):
    """Both trailing and forward count; adding a low forward softens (but doesn't erase) stretch."""
    trailing_only = fundamentals_factory(trailing_pe=60.0, forward_pe=None)
    with_low_fwd = fundamentals_factory(trailing_pe=60.0, forward_pe=18.0)
    s_only = f_pe_stretch(trailing_only, 18.0, 0.10)
    s_both = f_pe_stretch(with_low_fwd, 18.0, 0.10)
    assert s_both.value < s_only.value          # forward softens it
    assert s_both.value > 40                     # but high trailing still counts
    assert "trailing 60.0" in s_both.detail and "fwd 18.0" in s_both.detail


def test_high_forward_pe_is_flagged(fundamentals_factory):
    # Forward P/E of 80 vs baseline 18 -> flagged as expensive even on forward earnings.
    flag = high_forward_pe_flag(fundamentals_factory(forward_pe=80.0), 18.0)
    assert flag is not None and "forward P/E" in flag
    # A reasonable forward P/E is not flagged.
    assert high_forward_pe_flag(fundamentals_factory(forward_pe=22.0), 18.0) is None


def test_high_forward_pe_appears_in_sell_flags(data_factory, position_factory):
    data = data_factory()
    data.fundamentals.forward_pe = 90.0
    pos = position_factory()
    score = score_holding(pos, data)
    assert any("forward P/E" in f for f in score.flags)


def test_strong_sell_case(data_factory, position_factory):
    """Overbought + overvalued + above target + big gain => SELL."""
    data = data_factory(
        quote=Quote(ticker="TEST", price=200.0, previous_close=198.0),
        rating=Rating(ticker="TEST", analyst_count=12, target_mean=150.0,
                      target_low=120.0, target_high=170.0, recommendation_mean=4.0),
        fair_value=FairValue(ticker="TEST", price=200.0, base=100.0, low=90.0, high=120.0,
                             methods={"dcf": 100.0}),
        tech={"rsi": 85, "above_ma50": 0.4, "above_ma200": 0.5, "ret_3m": 0.5, "vol_ratio": 1.2},
    )
    data.fundamentals.trailing_pe = 60.0
    pos = position_factory(cost_basis=50.0, price=200.0)
    score = score_holding(pos, data)
    assert score.band == "SELL"
    assert score.composite >= 67


def test_calm_hold_case(data_factory, position_factory):
    """Below target + undervalued + calm momentum => HOLD."""
    data = data_factory(
        quote=Quote(ticker="TEST", price=100.0, previous_close=100.0),
        rating=Rating(ticker="TEST", analyst_count=10, target_mean=160.0,
                      target_low=140.0, target_high=190.0, recommendation_mean=2.0),
        fair_value=FairValue(ticker="TEST", price=100.0, base=180.0, low=150.0, high=210.0,
                             methods={"dcf": 180.0}),
        tech={"rsi": 45, "above_ma50": -0.02, "above_ma200": 0.0, "ret_3m": 0.02, "vol_ratio": 1.0},
    )
    data.fundamentals.trailing_pe = 12.0
    pos = position_factory(cost_basis=90.0, price=100.0)
    score = score_holding(pos, data)
    assert score.band == "HOLD"
    assert score.composite < 45


def test_strong_opportunity_case(data_factory):
    """High upside + undervalued + cheap + accumulation => STRONG buy."""
    data = data_factory(
        quote=Quote(ticker="TEST", price=100.0, previous_close=99.0),
        rating=Rating(ticker="TEST", analyst_count=20, target_mean=160.0,
                      target_low=130.0, target_high=200.0, recommendation_mean=1.6),
        fair_value=FairValue(ticker="TEST", price=100.0, base=160.0, low=140.0, high=190.0,
                             methods={"dcf": 160.0}),
        tech={"rsi": 55, "above_ma50": 0.05, "above_ma200": 0.15, "ret_3m": 0.2, "vol_ratio": 1.1},
        news=[NewsItem(ticker="TEST", headline="Great quarter", sentiment=0.5)],
        trades=[Trade(ticker="TEST", actor="Rep X", actor_type=ActorType.CONGRESS, side=TradeSide.BUY)],
    )
    data.fundamentals.trailing_pe = 10.0
    score = score_candidate("TEST", data)
    assert score.band == "STRONG"
    assert score.composite >= 67


def test_moonshot_flags_and_score(data_factory, fundamentals_factory):
    """Small, fast-growing, under-followed, breaking out => HIGH tier with risk flags."""
    data = data_factory(
        fundamentals=fundamentals_factory(
            market_cap=4e8, revenue_growth=0.8, profit_margin=-0.2, beta=2.5, avg_volume=150_000,
            total_debt=5e8, total_cash=1e8,
        ),
        rating=Rating(ticker="TEST", analyst_count=1),
        tech={"rsi": 60, "above_ma50": 0.2, "above_ma200": 0.4, "ret_3m": 0.6, "vol_ratio": 1.5},
        trades=[Trade(ticker="TEST", actor="Fund", actor_type=ActorType.FUND, side=TradeSide.BUY)],
    )
    score = score_moonshot("TEST", data)
    assert score.band in ("HIGH", "MEDIUM")
    assert "unprofitable" in score.flags
    assert any("volatility" in f for f in score.flags)


def test_factors_never_crash_on_empty(data_factory, position_factory):
    """Missing data must degrade to a neutral score, not raise."""
    empty = data_factory(
        quote=Quote(ticker="TEST", price=None),
        rating=None,
        fair_value=None,
        tech={},
        news=[],
        trades=[],
    )
    empty.fundamentals = None
    pos = position_factory(price=100.0)
    score = score_holding(pos, empty)
    assert 0 <= score.composite <= 100
    assert score.band in ("SELL", "TRIM", "HOLD")
