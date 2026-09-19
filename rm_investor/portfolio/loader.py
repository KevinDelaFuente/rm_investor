"""Portfolio loading: holdings.csv -> enriched Positions with live-ish P&L."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import get_config
from ..datasources.registry import DataHub, get_hub
from ..models import Holding, Position

_cfg = get_config()


def _rows_to_holdings(df: pd.DataFrame) -> list[Holding]:
    df.columns = [c.strip().lower() for c in df.columns]
    holdings: list[Holding] = []
    for _, row in df.iterrows():
        try:
            ticker = row.get("ticker")
            shares = row.get("shares")
            cost_basis = row.get("cost_basis")

            if ticker is None or pd.isna(ticker):
                continue

            ticker = str(ticker).strip().upper()
            if not ticker:
                continue

            shares = float(shares)
            cost_basis = float(cost_basis)
            if pd.isna(shares) or shares <= 0 or pd.isna(cost_basis):
                continue

            holdings.append(
                Holding(
                    ticker=ticker,
                    shares=shares,
                    cost_basis=cost_basis,
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return holdings


def _holdings_secret() -> Optional[str]:
    """Raw HOLDINGS_CSV from the environment or, on Streamlit Cloud, directly
    from st.secrets. Reading st.secrets here (not just via the bootstrap env
    bridge) makes the loader robust to bridge-vs-cache ordering: even if a cached
    call runs before bootstrap(), the secret is still found."""
    raw = os.environ.get("HOLDINGS_CSV")
    if raw and raw.strip():
        return raw
    try:
        import streamlit as st

        val = st.secrets.get("HOLDINGS_CSV")  # type: ignore[attr-defined]
        if isinstance(val, str) and val.strip():
            return val
    except Exception:
        pass
    return None


def load_holdings(csv_path: Optional[str] = None) -> list[Holding]:
    """Read holdings (columns: ticker, shares, cost_basis).

    Resolution order (when no explicit path is given):
      1. HOLDINGS_CSV env/secret — raw CSV text (Streamlit Cloud, Option B).
      2. configured portfolio.csv_path file — the local, gitignored real holdings.
      3. holdings_sample.csv — safety fallback so a fresh clone is never empty.
    An explicit csv_path (used by tests) always wins and skips the fallbacks.
    """
    if csv_path is None:
        raw = _holdings_secret()
        if raw:
            return _rows_to_holdings(pd.read_csv(io.StringIO(raw)))

    path = Path(csv_path) if csv_path else _cfg.resolve(_cfg.path("portfolio.csv_path", "holdings.csv"))
    if not path.exists():
        sample = _cfg.resolve("holdings_sample.csv")
        if csv_path is None and sample.exists():
            path = sample
        else:
            return []
    return _rows_to_holdings(pd.read_csv(path))


def build_positions(holdings: list[Holding], hub: Optional[DataHub] = None) -> list[Position]:
    """Attach live-ish quotes and compute portfolio weights."""
    hub = hub or get_hub()
    positions: list[Position] = []
    for h in holdings:
        quote = hub.quote(h.ticker)
        if quote is None:
            quote = _empty_quote(h.ticker)
        positions.append(Position(holding=h, quote=quote))

    total_mv = sum((p.market_value or 0.0) for p in positions)
    if total_mv > 0:
        for p in positions:
            p.weight = 100.0 * (p.market_value or 0.0) / total_mv
    return positions


def portfolio_summary(positions: list[Position]) -> dict:
    """Aggregate totals for the header row."""
    market_value = sum((p.market_value or 0.0) for p in positions)
    cost_value = sum(p.cost_value for p in positions)
    pl = market_value - cost_value
    pl_pct = (100.0 * pl / cost_value) if cost_value else None
    return {
        "market_value": market_value,
        "cost_value": cost_value,
        "unrealized_pl": pl,
        "unrealized_pl_pct": pl_pct,
        "num_positions": len(positions),
    }


def load_portfolio(csv_path: Optional[str] = None, hub: Optional[DataHub] = None) -> list[Position]:
    """Convenience: CSV -> Positions in one call."""
    return build_positions(load_holdings(csv_path), hub)


def _empty_quote(ticker: str):
    from ..models import Quote

    return Quote(ticker=ticker, price=None)
