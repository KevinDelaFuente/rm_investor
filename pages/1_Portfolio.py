"""Portfolio — positions, live-ish P&L, and allocation."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from rm_investor.ui import bootstrap, cached_portfolio, disclaimer, refresh_button

st.set_page_config(page_title="Portfolio · RM Investor", page_icon="💼", layout="wide")
bootstrap()
st.title("💼 Portfolio")
disclaimer()
refresh_button()

positions, summary = cached_portfolio()

if not positions:
    st.warning("No holdings found. Add rows to `holdings.csv` (ticker, shares, cost_basis).")
    st.stop()

# --- summary metrics -------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Market value", f"${summary['market_value']:,.0f}")
c2.metric("Cost basis", f"${summary['cost_value']:,.0f}")
c3.metric(
    "Unrealized P&L",
    f"${summary['unrealized_pl']:,.0f}",
    f"{summary['unrealized_pl_pct']:+.1f}%" if summary["unrealized_pl_pct"] is not None else None,
)
c4.metric("Positions", summary["num_positions"])

# --- positions table -------------------------------------------------------- #
rows = []
for p in positions:
    rows.append(
        {
            "Ticker": p.ticker,
            "Shares": p.holding.shares,
            "Cost basis": p.holding.cost_basis,
            "Price": p.price,
            "Market value": p.market_value,
            "Unrealized P&L": p.unrealized_pl,
            "P&L %": p.unrealized_pl_pct,
            "Weight %": p.weight,
        }
    )
df = pd.DataFrame(rows)

st.dataframe(
    df.style.format(
        {
            "Cost basis": "${:,.2f}",
            "Price": "${:,.2f}",
            "Market value": "${:,.0f}",
            "Unrealized P&L": "${:,.0f}",
            "P&L %": "{:+.1f}%",
            "Weight %": "{:.1f}%",
        },
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
)

# --- allocation chart ------------------------------------------------------- #
alloc = df.dropna(subset=["Market value"])
if not alloc.empty:
    fig = px.pie(alloc, names="Ticker", values="Market value", title="Allocation by market value")
    st.plotly_chart(fig, use_container_width=True)

missing = [p.ticker for p in positions if p.price is None]
if missing:
    st.caption("No price available for: " + ", ".join(missing))
