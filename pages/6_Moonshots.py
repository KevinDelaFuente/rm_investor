"""Moonshots — high-risk / high-reward small & micro-cap sleeve (risk-flagged)."""
from __future__ import annotations

import streamlit as st

from rm_investor.ui import (
    bootstrap,
    cached_scan_moonshots,
    disclaimer,
    get_or_compute,
    refresh_button,
    render_score,
    result_is_saved,
    scan_with_progress,
)

st.set_page_config(page_title="Moonshots · RM Investor", page_icon="🚀", layout="wide")
bootstrap()
st.title("🚀 Moonshots")
disclaimer()

st.error(
    "**High risk by design.** These are small/micro-cap, often unprofitable, thinly traded "
    "names with asymmetric upside *and* real chance of large loss. Treat as a small, optional "
    "speculative sleeve — never core holdings. The universe here is a curated sample, not an "
    "exhaustive market sweep."
)
refresh_button()

st.caption(
    "Scored on small size, accelerating revenue growth, being under-followed, momentum "
    "breakouts, and disclosed accumulation. Risk flags are shown alongside every score."
)

df, detail = get_or_compute(
    "moonshot_scan",
    lambda: scan_with_progress(cached_scan_moonshots, "Scanning moonshot universe"),
)
if result_is_saved("moonshot_scan"):
    st.caption("Showing saved results — hit **🔄 Refresh data** to re-scan.")

if df.empty:
    st.warning("No moonshot candidates scored — sources may be rate-limited. Try Refresh shortly.")
    st.stop()

st.dataframe(
    df.style.format(
        {"Price": "${:,.2f}", "Score": "{:.0f}", "Mkt cap $B": "{:.2f}", "Rev growth %": "{:+.0f}%"},
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
