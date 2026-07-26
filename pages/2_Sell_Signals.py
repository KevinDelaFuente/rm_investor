"""Sell Signals — per-holding SELL / TRIM / HOLD with factor breakdown."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from rm_investor.scoring.sell_signals import score_holding
from rm_investor.ui import (
    bootstrap,
    cached_gather_holding,
    cached_portfolio,
    disclaimer,
    band_badge,
    get_or_compute,
    refresh_button,
    render_score,
    result_is_saved,
)

st.set_page_config(page_title="Sell Signals · RM Investor", page_icon="🔔", layout="wide")
bootstrap()
st.title("🔔 Sell Signals")
disclaimer()
refresh_button()

positions, _ = cached_portfolio()
if not positions:
    st.warning("No holdings found. Add rows to `holdings.csv`.")
    st.stop()


def _analyze():
    """Score every holding, with a live progress bar. Runs only on first visit or
    after Refresh — the result is then kept in the session (see get_or_compute)."""
    scored = []
    total = len(positions)
    progress = st.progress(0.0, text=f"Analyzing 0/{total} holdings…")
    status = st.empty()
    start = time.time()

    for i, p in enumerate(positions, start=1):
        status.caption(f"Fetching **{p.ticker}** ({i}/{total})…")
        data = cached_gather_holding(p.ticker)
        score = score_holding(p, data)
        scored.append((p, score, data))

        elapsed = time.time() - start
        eta = (elapsed / i) * (total - i)
        progress.progress(
            i / total,
            text=f"Analyzing {i}/{total} holdings — {elapsed:0.0f}s elapsed"
            + (f", ~{eta:0.0f}s left" if i < total else ""),
        )

    progress.empty()
    status.empty()
    scored.sort(key=lambda x: x[1].composite, reverse=True)
    return scored


scored = get_or_compute("sell_signals", _analyze)
if result_is_saved("sell_signals"):
    st.caption("Showing saved results — hit **🔄 Refresh data** to re-analyze.")

# --- overview table --------------------------------------------------------- #
summary_rows = [
    {
        "Ticker": p.ticker,
        "Signal": s.band,
        "Score": s.composite,
        "P&L %": p.unrealized_pl_pct,
        "Weight %": p.weight,
    }
    for p, s, _ in scored
]
st.dataframe(
    pd.DataFrame(summary_rows).style.format(
        {"Score": "{:.0f}", "P&L %": "{:+.1f}%", "Weight %": "{:.1f}%"}, na_rep="—"
    ),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Higher score = stronger case to trim/sell. Bands are configurable in config.yaml. "
    "Holdings use free yfinance analyst ratings to conserve FMP quota across all positions."
)
st.divider()

# --- per-holding detail ----------------------------------------------------- #
for p, s, data in scored:
    header = f"{p.ticker} — {s.band} ({s.composite:.0f}/100)"
    with st.expander(header, expanded=(s.band == "SELL")):
        st.markdown(band_badge(s.band), unsafe_allow_html=True)
        render_score(s)
        if data.rating and data.rating.target_mean:
            st.caption(
                f"Analyst target range (low/mean/high): {data.rating.target_range_str} · "
                f"{data.rating.analyst_count or '?'} analysts"
            )
