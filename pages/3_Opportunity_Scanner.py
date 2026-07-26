"""Opportunity Scanner — ranked buy candidates from the universe."""
from __future__ import annotations

import streamlit as st

from rm_investor.ui import (
    bootstrap,
    cached_scan_opportunities,
    disclaimer,
    get_or_compute,
    refresh_button,
    render_score,
    result_is_saved,
    scan_with_progress,
)

st.set_page_config(page_title="Opportunity Scanner · RM Investor", page_icon="🔎", layout="wide")
bootstrap()
st.title("🔎 Opportunity Scanner")
disclaimer()
refresh_button()

st.caption(
    "Scores a bounded universe (watchlist ∪ index sample ∪ recent smart-money buys) on "
    "analyst upside & coverage depth, undervaluation vs. fair value, valuation, momentum, "
    "news, and disclosed smart-money buying. Higher = stronger buy case."
)

df, detail = get_or_compute(
    "opportunity_scan",
    lambda: scan_with_progress(cached_scan_opportunities, "Scanning universe"),
)
if result_is_saved("opportunity_scan"):
    st.caption("Showing saved results — hit **🔄 Refresh data** to re-scan.")

if df.empty:
    st.warning("No candidates scored — data sources may be rate-limited. Try Refresh in a moment.")
    st.stop()

st.dataframe(
    df.style.format(
        {
            "Price": "${:,.2f}",
            "Score": "{:.0f}",
            "Upside %": "{:+.1f}%",
            "Margin of safety %": "{:+.1f}%",
        },
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
)

st.divider()
st.markdown("#### Drill into a candidate")
pick = st.selectbox("Ticker", df["Ticker"].tolist())
if pick and pick in detail:
    score, data = detail[pick]
    render_score(score)
    if data.rating:
        st.caption(
            f"Analyst target range (low/mean/high): {data.rating.target_range_str} · "
            f"{data.rating.analyst_count or '?'} analysts"
        )
