"""Smart Money — disclosed congressional trades & 13F changes, overlap with your book.

Reminder: this is public, LAGGED disclosure (STOCK Act filings up to ~45 days late; 13F
~45 days delayed, long U.S. equities only). It is a signal of what disclosed smart money
did weeks ago — not insider information.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from rm_investor.datasources.registry import get_hub
from rm_investor.models import TradeSide
from rm_investor.ui import bootstrap, cached_portfolio, disclaimer, refresh_button

st.set_page_config(page_title="Smart Money · RM Investor", page_icon="🏛️", layout="wide")
bootstrap()
st.title("🏛️ Smart Money")
disclaimer()
refresh_button()

st.warning(
    "Public, lagged disclosures — **not insider information**. Congressional filings can be "
    "up to ~45 days late; 13F holdings are ~45 days delayed and long-only U.S. equities."
)

hub = get_hub()
if not hub.has_smart_money:
    st.info("No smart-money sources configured. Enable `congress` and/or `sec_edgar` in config.yaml.")
    st.stop()


@st.cache_data(ttl=86400, show_spinner="Fetching disclosures…")
def _all_trades():
    return hub.smart_money(None, limit=500)


trades = _all_trades()
if not trades:
    st.warning("No disclosures fetched (source may be unreachable). Try Refresh shortly.")
    st.stop()

rows = [
    {
        "Ticker": t.ticker or (t.issuer or "—"),
        "Actor": t.actor,
        "Type": t.actor_type.value,
        "Side": t.side.value,
        "Amount": t.amount or (f"{t.shares_delta:+,.0f} sh" if t.shares_delta else "—"),
        "Transaction date": t.transaction_date,
        "Source": t.source,
    }
    for t in trades
]
df = pd.DataFrame(rows)

# --- overlap with your holdings -------------------------------------------- #
positions, _ = cached_portfolio()
held = {p.ticker for p in positions}
if held:
    overlap = df[df["Ticker"].isin(held)]
    st.markdown("#### Disclosed trades in names **you hold**")
    if overlap.empty:
        st.caption("No disclosed smart-money trades in your current holdings within the lookback window.")
    else:
        st.dataframe(overlap, hide_index=True, use_container_width=True)
    st.divider()

# --- most-bought names ------------------------------------------------------ #
buys = df[df["Side"] == TradeSide.BUY.value]
if not buys.empty:
    top = buys["Ticker"].value_counts().head(15).rename_axis("Ticker").reset_index(name="Buys")
    st.markdown("#### Most-bought tickers (disclosed)")
    st.dataframe(top, hide_index=True, use_container_width=True)

st.markdown("#### All recent disclosures")
st.dataframe(df, hide_index=True, use_container_width=True)
