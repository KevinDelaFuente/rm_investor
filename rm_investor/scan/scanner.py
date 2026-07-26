"""Scanner: fetch data for a universe, score each name, rank, and tabulate.

Returns a ranked pandas DataFrame (for the UI table) plus a dict of ticker -> (Score,
TickerData) so a page can drill into any row's factor breakdown. Iterates serially and
relies on the cache layer to stay within free-tier rate limits.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from ..analysis import TickerData, gather
from ..config import get_config
from ..datasources.registry import DataHub, get_hub
from ..models import Score
from ..scoring.moonshot import score_moonshot
from ..scoring.opportunity import score_candidate
from .moonshot_universe import build_moonshot_universe, max_market_cap
from .universe import build_universe

_cfg = get_config()


def _opportunity_row(ticker: str, score: Score, data: TickerData) -> dict:
    q, r, fv, f = data.quote, data.rating, data.fair_value, data.fundamentals
    return {
        "Ticker": ticker,
        "Name": (f.name if f else None),
        "Price": (q.price if q else None),
        "Score": score.composite,
        "Signal": score.band,
        "Upside %": (round((r.target_mean / q.price - 1) * 100, 1)
                     if (r and r.target_mean and q and q.price) else None),
        "Analysts": (r.analyst_count if r else None),
        "Target range": (r.target_range_str if r else "—"),
        "Margin of safety %": (round(fv.margin_of_safety * 100, 1)
                               if (fv and fv.margin_of_safety is not None) else None),
    }


def _moonshot_row(ticker: str, score: Score, data: TickerData) -> dict:
    q, r, f = data.quote, data.rating, data.fundamentals
    return {
        "Ticker": ticker,
        "Name": (f.name if f else None),
        "Price": (q.price if q else None),
        "Score": score.composite,
        "Tier": score.band,
        "Mkt cap $B": (round(f.market_cap / 1e9, 2) if (f and f.market_cap) else None),
        "Rev growth %": (round(f.revenue_growth * 100, 0) if (f and f.revenue_growth is not None) else None),
        "Analysts": (r.analyst_count if r else None),
        "Risk flags": ", ".join(score.flags) if score.flags else "—",
    }


def _scan(
    tickers: list[str],
    score_fn: Callable[[str, TickerData], Score],
    row_fn: Callable[[str, Score, TickerData], dict],
    hub: DataHub,
    cap_ceiling: Optional[float] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    detail: dict = {}
    total = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        if progress is not None:
            progress(idx, total, ticker)
        try:
            data = gather(ticker, hub)
        except Exception:
            continue
        if data.quote is None and data.fundamentals is None:
            continue  # no data at all — skip silently
        if cap_ceiling is not None:
            mc = data.fundamentals.market_cap if data.fundamentals else None
            if mc is not None and mc > cap_ceiling:
                continue  # too big to be a moonshot
        score = score_fn(ticker, data)
        rows.append(row_fn(ticker, score, data))
        detail[ticker] = (score, data)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df, detail


def scan_opportunities(cfg=None, hub: Optional[DataHub] = None, progress=None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or _cfg
    hub = hub or get_hub()
    tickers = build_universe(cfg, hub)
    return _scan(tickers, lambda t, d: score_candidate(t, d, cfg), _opportunity_row, hub, progress=progress)


def scan_moonshots(cfg=None, hub: Optional[DataHub] = None, progress=None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or _cfg
    hub = hub or get_hub()
    tickers = build_moonshot_universe(cfg)
    return _scan(
        tickers,
        lambda t, d: score_moonshot(t, d, cfg),
        _moonshot_row,
        hub,
        cap_ceiling=max_market_cap(cfg),
        progress=progress,
    )
