"""Build the candidate universe for the opportunity scanner.

Universe = configured watchlist  ∪  an index constituent list (shipped CSV)
           ∪  tickers surfaced by recent disclosed smart-money buying (optional),
bounded to `universe.max_size` to respect free-tier rate limits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import get_config
from ..datasources.registry import DataHub, get_hub
from ..models import TradeSide

_cfg = get_config()


def load_ticker_csv(path: Path) -> list[str]:
    if not path or not Path(path).exists():
        return []
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        return [str(t).strip().upper() for t in df[col].dropna().tolist()]
    except Exception:
        return []


def smart_money_buys(hub: DataHub, limit: int = 40) -> list[str]:
    """Tickers with recent disclosed BUYs (congress path yields tickers directly)."""
    tickers: list[str] = []
    for t in hub.smart_money(None, limit=400):
        if t.side == TradeSide.BUY and t.ticker:
            tickers.append(t.ticker)
    # Preserve order, dedup, cap.
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def build_universe(cfg=None, hub: Optional[DataHub] = None) -> list[str]:
    cfg = cfg or _cfg
    hub = hub or get_hub()
    u = cfg.get("universe", {}) or {}

    tickers: list[str] = list(u.get("watchlist", []) or [])
    tickers += load_ticker_csv(cfg.resolve(u.get("index_csv", "")))
    if u.get("include_smart_money_buys") and hub.has_smart_money:
        tickers += smart_money_buys(hub)

    # Dedup preserving order, then bound.
    seen, out = set(), []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    max_size = int(u.get("max_size", 40))
    return out[:max_size]
