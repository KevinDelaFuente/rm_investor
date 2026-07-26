"""Shared Streamlit UI helpers: disclaimer, score/valuation renderers, cached data.

Caching wrappers use st.cache_data so the dashboard stays responsive and avoids
re-hitting free APIs within a session. All heavy data access flows through here.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

from .analysis import TickerData, gather
from .config import get_config
from .models import FairValue, Score
from .portfolio.loader import load_portfolio, portfolio_summary
from .scan.scanner import scan_moonshots, scan_opportunities

_cfg = get_config()


# --------------------------------------------------------------------------- #
# Startup: secrets bridge + optional password gate
# --------------------------------------------------------------------------- #
def _bridge_secrets_to_env() -> None:
    """Copy Streamlit Secrets into os.environ so the pure config layer
    (get_api_key, loader HOLDINGS_CSV) picks up keys and real holdings on
    Streamlit Cloud. Never overrides a value already set via the real
    environment or .env, so local dev is unchanged. No-op when no secrets file
    exists (accessing st.secrets then raises, which we swallow)."""
    try:
        items = list(st.secrets.items())
    except Exception:
        return
    for key, val in items:
        if isinstance(val, str) and not os.environ.get(key):
            os.environ[key] = val


def _password_gate() -> None:
    """Shared-password gate for a private deploy. Open when APP_PASSWORD is unset
    (local dev). Auth persists for the session, so it prompts once across pages."""
    expected = os.environ.get("APP_PASSWORD")
    if not expected or st.session_state.get("_authed"):
        return
    st.title("🔒 RM Investor")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if pw == expected:
            st.session_state["_authed"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def bootstrap() -> None:
    """Run at the top of every page (after set_page_config): bridge secrets into
    the environment, then enforce the optional password gate. Idempotent."""
    _bridge_secrets_to_env()
    _password_gate()

_BAND_COLORS = {
    # sell view: SELL is the "act" state (red)
    "SELL": "#c0392b", "TRIM": "#e67e22", "HOLD": "#27ae60",
    # opportunity: STRONG is good (green)
    "STRONG": "#27ae60", "WATCH": "#e67e22", "PASS": "#7f8c8d",
    # moonshot tiers
    "HIGH": "#8e44ad", "MEDIUM": "#e67e22", "LOW": "#7f8c8d",
}


def disclaimer() -> None:
    st.caption(
        "⚠️ **Not financial advice.** RM Investor is a personal decision-support tool. "
        "Prices are delayed (~15 min). Analyst, valuation, and smart-money signals are "
        "model-based or lagged public disclosures — not recommendations or insider intel. "
        "Do your own research."
    )


def band_badge(band: str) -> str:
    color = _BAND_COLORS.get(band, "#7f8c8d")
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-weight:600'>{band}</span>"
    )


def render_score(score: Score) -> None:
    """Composite gauge + a transparent factor breakdown table."""
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Composite", f"{score.composite:.0f}/100")
        st.markdown(band_badge(score.band), unsafe_allow_html=True)
    with c2:
        rows = [
            {
                "Factor": f.name,
                "Score": round(f.value, 0),
                "Weight": f"{f.weight:.0%}",
                "Contribution": round(f.contribution, 1),
                "Why": f.detail,
            }
            for f in score.factors
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if score.flags:
        st.warning("Risk flags: " + ", ".join(score.flags))


def render_fair_value(fv: Optional[FairValue]) -> None:
    if fv is None:
        st.info("No fair-value estimate available.")
        return
    cols = st.columns(4)
    cols[0].metric("Price", f"${fv.price:,.2f}" if fv.price else "—")
    cols[1].metric("Fair value (base)", f"${fv.base:,.2f}" if fv.base else "—")
    cols[2].metric(
        "Range",
        f"${fv.low:,.0f}–${fv.high:,.0f}" if (fv.low and fv.high) else "—",
    )
    mos = fv.margin_of_safety
    cols[3].metric("Margin of safety", f"{mos * 100:+.0f}%" if mos is not None else "—")

    if fv.methods:
        method_rows = []
        for name, val in fv.methods.items():
            method_rows.append({"Method": name.upper(), "Fair value": f"${val:,.2f}",
                                "Inputs": str(fv.inputs.get(name, {}))})
        st.dataframe(pd.DataFrame(method_rows), hide_index=True, use_container_width=True)
    if fv.notes:
        st.caption("Skipped / caveats: " + " · ".join(fv.notes))


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=_cfg.path("cache.quote_ttl_seconds", 900), show_spinner=False)
def cached_portfolio():
    positions = load_portfolio()
    return positions, portfolio_summary(positions)


@st.cache_data(ttl=_cfg.path("cache.fundamentals_ttl_seconds", 86400), show_spinner="Analyzing…")
def cached_gather(ticker: str) -> TickerData:
    return gather(ticker)


@st.cache_data(ttl=_cfg.path("cache.fundamentals_ttl_seconds", 86400), show_spinner="Analyzing…")
def cached_gather_holding(ticker: str) -> TickerData:
    """Gather for portfolio holdings using free yfinance ratings (conserves FMP quota
    across the 97-name Sell Signals page). Scanners still use FMP via cached_gather."""
    return gather(ticker, prefer_yf_ratings=True)


@st.cache_data(ttl=_cfg.path("cache.fundamentals_ttl_seconds", 86400), show_spinner=False)
def cached_scan_opportunities(_progress=None):
    # _progress is prefixed with "_" so Streamlit doesn't try to hash it; results are
    # cached ~12h, so the bar only animates on a cold run.
    return scan_opportunities(progress=_progress)


@st.cache_data(ttl=_cfg.path("cache.fundamentals_ttl_seconds", 86400), show_spinner=False)
def cached_scan_moonshots(_progress=None):
    return scan_moonshots(progress=_progress)


_RESULT_PREFIX = "rmresult__"


def refresh_button() -> None:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        # Drop stored per-tab results so every tab re-analyzes on next visit.
        for k in list(st.session_state.keys()):
            if k.startswith(_RESULT_PREFIX):
                del st.session_state[k]
        st.rerun()


def get_or_compute(name: str, compute_fn):
    """Return a per-tab result stored in the session, computing (with its progress
    bar) only on the first visit or after Refresh. Navigating back to a tab reuses
    the stored result instantly — no re-analysis, no progress bar replay."""
    key = _RESULT_PREFIX + name
    if key not in st.session_state:
        st.session_state[key] = compute_fn()
        st.session_state[key + "__fresh"] = True
    else:
        st.session_state[key + "__fresh"] = False
    return st.session_state[key]


def result_is_saved(name: str) -> bool:
    """True if the tab is rendering a previously-saved result (not freshly computed)."""
    return st.session_state.get(_RESULT_PREFIX + name + "__fresh") is False


def scan_with_progress(scan_fn, label: str):
    """Run a cached scan while showing a live progress bar on cold runs.

    scan_fn is one of cached_scan_opportunities / cached_scan_moonshots. On a cache
    hit the callback never fires, so the bar simply doesn't appear.
    """
    import time

    bar = st.progress(0.0, text=f"{label}…")
    state = {"start": time.time()}

    def _cb(i: int, total: int, ticker: str):
        elapsed = time.time() - state["start"]
        eta = (elapsed / i) * (total - i) if i else 0
        bar.progress(
            i / total,
            text=f"{label}: {i}/{total} ({ticker}) — {elapsed:0.0f}s"
            + (f", ~{eta:0.0f}s left" if i < total else ""),
        )

    try:
        result = scan_fn(_progress=_cb)
    finally:
        bar.empty()
    return result
