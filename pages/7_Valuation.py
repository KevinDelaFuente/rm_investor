"""Valuation — DCF + multiples + DDM fair-value range and margin of safety."""
from __future__ import annotations

import streamlit as st

from rm_investor.ui import bootstrap, cached_gather, disclaimer, render_fair_value

st.set_page_config(page_title="Valuation · RM Investor", page_icon="🧮", layout="wide")
bootstrap()
st.title("🧮 Valuation")
disclaimer()

st.write(
    "Estimate a name's **intrinsic value** three ways — discounted cash flow, relative "
    "multiples, and a dividend-discount / Graham sanity check — then blend into a "
    "fair-value **range** and a **margin of safety** vs. price. Assumptions live in "
    "`config.yaml` and every input is shown below."
)

ticker = st.text_input("Ticker", value="AAPL").strip().upper()
if not ticker:
    st.stop()

data = cached_gather(ticker)
if data.fundamentals is None:
    st.error(f"No fundamentals available for {ticker}.")
    st.stop()

st.subheader(f"{data.fundamentals.name or ticker} ({ticker})")
render_fair_value(data.fair_value)

st.divider()
st.markdown("#### Fundamentals used")
f = data.fundamentals
st.json(
    {
        "sector": f.sector,
        "market_cap": f.market_cap,
        "trailing_pe": f.trailing_pe,
        "forward_pe": f.forward_pe,
        "eps_trailing": f.eps_trailing,
        "book_value_per_share": f.book_value_per_share,
        "revenue": f.revenue,
        "revenue_growth": f.revenue_growth,
        "free_cash_flow": f.free_cash_flow,
        "total_cash": f.total_cash,
        "total_debt": f.total_debt,
        "dividend_rate": f.dividend_rate,
        "shares_outstanding": f.shares_outstanding,
    },
    expanded=False,
)
st.caption(
    "Valuation is model-based and only as good as its assumptions. The base is a weighted "
    "blend of the methods that could run; inapplicable methods are skipped, not faked."
)
