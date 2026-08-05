"""RM Investor — Streamlit entrypoint (overview + disclaimer + provider status)."""
from __future__ import annotations

import streamlit as st

from rm_investor.config import get_api_key, get_config
from rm_investor.ui import bootstrap, disclaimer

st.set_page_config(page_title="RM Investor", page_icon="📈", layout="wide")
bootstrap()
cfg = get_config()

st.title("📈 RM Investor")
st.subheader("Portfolio, sell signals, opportunities, valuation & smart-money — on free data")
disclaimer()

st.markdown(
    """
Use the pages in the sidebar:

| Page | What it does |
|------|--------------|
| **1 · Portfolio** | Your positions, live-ish P&L, and allocation |
| **2 · Holdings Signals** | For stocks you own: **SELL / UNDERWEIGHT / HOLD / OVERWEIGHT / STRONG BUY** from a two-sided factor breakdown |
| **3 · Opportunity Scanner** | Ranked buy candidates with analyst coverage, target range & margin of safety |
| **4 · Smart Money** | Disclosed congressional trades & 13F changes, overlap with your book |
| **5 · News** | Aggregated headlines + sentiment per ticker |
| **6 · Moonshots** | High-risk / high-reward small & micro-cap sleeve (risk-flagged) |
| **7 · Valuation** | DCF + multiples + DDM fair-value range and margin of safety |
"""
)

# --- provider / key status -------------------------------------------------- #
st.divider()
st.markdown("#### Data sources")
providers = cfg.get("providers", {})
cola, colb = st.columns(2)
with cola:
    st.write("**Active providers**")
    st.json(providers, expanded=False)
with colb:
    st.write("**API keys detected**")
    st.write(
        {
            "FINNHUB_API_KEY": bool(get_api_key("FINNHUB_API_KEY")),
            "ALPHAVANTAGE_API_KEY": bool(get_api_key("ALPHAVANTAGE_API_KEY")),
        }
    )
    st.caption("No keys required — everything runs on yfinance by default.")

st.divider()
st.caption(
    "Smart-money data is public and lagged: congressional filings under the STOCK Act "
    "(up to ~45 days late) and 13F holdings (~45 days delayed, long U.S. equities only). "
    "It reflects what disclosed smart money did weeks ago — a signal, not live intel."
)
