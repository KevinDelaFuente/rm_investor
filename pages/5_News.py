"""News — aggregated headlines + sentiment per ticker."""
from __future__ import annotations

import streamlit as st

from rm_investor.ui import bootstrap, cached_gather, disclaimer

st.set_page_config(page_title="News · RM Investor", page_icon="📰", layout="wide")
bootstrap()
st.title("📰 News & Sentiment")
disclaimer()

st.caption(
    "Headlines from the configured news source. Sentiment scores appear only when the "
    "source provides them (e.g. Alpha Vantage NEWS_SENTIMENT); yfinance gives headlines only."
)

ticker = st.text_input("Ticker", value="AAPL").strip().upper()
if not ticker:
    st.stop()

data = cached_gather(ticker)
news = data.news
if not news:
    st.warning(f"No news returned for {ticker}.")
    st.stop()

scored = [n.sentiment for n in news if n.sentiment is not None]
if scored:
    avg = sum(scored) / len(scored)
    st.metric("Average sentiment", f"{avg:+.2f}", help="-1 (very negative) … +1 (very positive)")

for n in news:
    with st.container(border=True):
        title = f"**[{n.headline}]({n.url})**" if n.url else f"**{n.headline}**"
        st.markdown(title)
        meta = " · ".join(
            x for x in [
                n.source,
                n.published.strftime("%Y-%m-%d %H:%M") if n.published else None,
                f"sentiment {n.sentiment:+.2f}" if n.sentiment is not None else None,
            ] if x
        )
        if meta:
            st.caption(meta)
        if n.summary:
            st.write(n.summary)
