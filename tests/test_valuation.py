"""Valuation tests — deterministic, no network."""
from __future__ import annotations

from rm_investor.models import Quote
from rm_investor.valuation import fair_value
from rm_investor.valuation.dcf import dcf_value
from rm_investor.valuation.ddm import ddm_value, graham_number
from rm_investor.valuation.multiples import multiples_value

ASSUMPTIONS = {
    "discount_rate": 0.10,
    "terminal_growth": 0.025,
    "projection_years": 5,
    "default_growth": 0.08,
    "max_growth": 0.20,
    "multiples": {"pe": 18.0, "ps": 3.0, "pb": 2.5, "ev_ebitda": 12.0},
    "method_weights": {"dcf": 0.4, "multiples": 0.4, "ddm": 0.2},
}


def test_dcf_positive_value(fundamentals_factory):
    f = fundamentals_factory(free_cash_flow=4e9, shares_outstanding=1e9,
                             total_cash=3e9, total_debt=1e9, revenue_growth=0.10)
    val, inputs = dcf_value(f, ASSUMPTIONS)
    assert val is not None and val > 0
    assert inputs["growth"] == 0.10


def test_dcf_skips_negative_fcf(fundamentals_factory):
    f = fundamentals_factory(free_cash_flow=-1e9)
    val, inputs = dcf_value(f, ASSUMPTIONS)
    assert val is None
    assert "skipped" in inputs


def test_multiples_median(fundamentals_factory):
    f = fundamentals_factory(eps_trailing=5.0, book_value_per_share=25.0,
                             revenue=2e10, shares_outstanding=1e9, ebitda=6e9,
                             total_debt=1e9, total_cash=3e9)
    val, inputs = multiples_value(f, ASSUMPTIONS)
    assert val is not None and val > 0
    # P/E leg should be EPS * target = 5 * 18 = 90
    assert abs(inputs["value_pe"] - 90.0) < 1e-6


def test_ddm_requires_dividend(fundamentals_factory):
    payer = fundamentals_factory(dividend_rate=2.0)
    val, _ = ddm_value(payer, ASSUMPTIONS)
    assert val is not None and val > 0

    non_payer = fundamentals_factory(dividend_rate=0.0)
    val2, inputs2 = ddm_value(non_payer, ASSUMPTIONS)
    assert val2 is None and "skipped" in inputs2


def test_graham_number(fundamentals_factory):
    f = fundamentals_factory(eps_trailing=5.0, book_value_per_share=25.0)
    gn = graham_number(f)
    assert gn is not None and gn > 0  # sqrt(22.5*5*25) ~= 53


def test_fair_value_blend_and_margin(fundamentals_factory):
    f = fundamentals_factory()
    quote = Quote(ticker=f.ticker, price=50.0)
    fv = fair_value(f, quote, f.ticker)
    assert fv.base is not None and fv.base > 0
    assert fv.low is not None and fv.high is not None and fv.low <= fv.high
    # margin of safety = (base - price) / price
    assert fv.margin_of_safety == (fv.base - 50.0) / 50.0


def test_fair_value_handles_missing_fundamentals():
    fv = fair_value(None, Quote(ticker="X", price=10.0), "X")
    assert fv.base is None
    assert any("no fundamentals" in n for n in fv.notes)
