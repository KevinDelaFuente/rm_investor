"""Holdings Signals — for stocks you already own: SELL / UNDERWEIGHT / HOLD /
OVERWEIGHT / STRONG BUY, from a two-sided (buy vs. sell) factor breakdown."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from rm_investor.research import snapshots
from rm_investor.scoring.holdings_signals import score_holding_action
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

st.set_page_config(page_title="Holdings Signals · RM Investor", page_icon="⚖️", layout="wide")
bootstrap()
st.title("⚖️ Holdings Signals")
st.caption(
    "What to do with the stocks you **already own** — trim the stretched, add to the "
    "still-attractive. For names you don't own yet, see **Opportunity Scanner**."
)
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
        score = score_holding_action(p, data)
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
    # Record a point-in-time snapshot. Fair value, P/E and sentiment cannot be
    # reconstructed historically from free data, so forward recording is the only way
    # they ever become validatable. Never raises.
    snapshots.record((p.ticker, s, d) for p, s, d in scored)
    # Most attractive first (highest conviction to add).
    scored.sort(key=lambda x: (x[1].conviction if x[1].conviction is not None else 0), reverse=True)
    return scored


scored = get_or_compute("holdings_signals", _analyze)
if result_is_saved("holdings_signals"):
    st.caption("Showing saved results — hit **🔄 Refresh data** to re-analyze.")

_ORDER = ["STRONG BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"]

# --- action summary --------------------------------------------------------- #
counts = {lbl: sum(1 for _, s, _ in scored if s.band == lbl) for lbl in _ORDER}
cols = st.columns(len(_ORDER))
for col, lbl in zip(cols, _ORDER):
    col.metric(lbl.title(), counts[lbl])

# --- filter ----------------------------------------------------------------- #
chosen = st.multiselect("Filter by signal", _ORDER, default=_ORDER)
view = [(p, s, d) for p, s, d in scored if s.band in chosen]

# --- overview table --------------------------------------------------------- #
summary_rows = [
    {
        "Ticker": p.ticker,
        "Signal": s.band,
        "Conviction": s.conviction,
        "Buy case": s.buy_score,
        "Sell case": s.sell_score,
        "P&L %": p.unrealized_pl_pct,
        "Weight %": p.weight,
    }
    for p, s, _ in view
]
st.dataframe(
    pd.DataFrame(summary_rows).style.format(
        {
            "Conviction": "{:+.0f}",
            "Buy case": "{:.0f}",
            "Sell case": "{:.0f}",
            "P&L %": "{:+.1f}%",
            "Weight %": "{:.1f}%",
        },
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "**Conviction = buy case − sell case** (−100…+100). Each side is a 0–100 weighted "
    "composite where ~50 is neutral, so a gap under ±10 means the two cases effectively "
    "offset (HOLD), and ±30 or more means one side clearly dominates. Thresholds are "
    "**absolute** — they reflect each position's own merits, not a quota per band, so all "
    "holdings can legitimately land in the same bucket. A holding must earn a real buy case "
    "to rate OVERWEIGHT — merely having nothing wrong lands it in HOLD. Configurable under "
    "`scoring.holdings` in config.yaml. Holdings use free yfinance analyst ratings to "
    "conserve FMP quota."
)
st.divider()

# --- per-holding detail ----------------------------------------------------- #
for p, s, data in view:
    conv = s.conviction if s.conviction is not None else 0.0
    header = f"{p.ticker} — {s.band} ({conv:+.0f} conviction)"
    with st.expander(header, expanded=(s.band in ("STRONG BUY", "SELL"))):
        st.markdown(band_badge(s.band), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Conviction", f"{conv:+.0f}")
        c2.metric("Buy case", f"{s.buy_score:.0f}/100" if s.buy_score is not None else "—")
        c3.metric("Sell case", f"{s.sell_score:.0f}/100" if s.sell_score is not None else "—")
        st.caption("▲ = buy-side factors · ▼ = sell-side factors")
        render_score(s)
        if data.rating and data.rating.target_mean:
            st.caption(
                f"Analyst target range (low/mean/high): {data.rating.target_range_str} · "
                f"{data.rating.analyst_count or '?'} analysts"
            )
