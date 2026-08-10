"""Forward-looking point-in-time recorder.

The factors that matter most to the conviction score — fair value, P/E, news sentiment,
smart money — cannot be reconstructed historically from free data: yfinance returns only
5-6 quarters of financial statements, indexed by period end rather than filing date, and
no sentiment history at all. So they can never be validated from the past.

They can be validated from here forward, but only if we start writing them down now.
Every scoring run appends one row per ticker: each factor's value, the raw inputs behind
it, and a timestamp. In a year this is the dataset that `panel.py` cannot build.

Fire-and-forget by design — `record()` never raises, because a research side-effect must
not be able to break a page render.

DEPLOYMENT CAVEAT: on Streamlit Cloud the filesystem is ephemeral, so snapshots written
there are lost whenever the app restarts or redeploys. The accumulating dataset this
module exists to build is therefore only reliable when the app runs locally, or when
SNAPSHOT_DIR is pointed at durable storage (S3, a database, a mounted volume).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..datasources.cache import CACHE_DIR

SNAPSHOT_DIR: Path = CACHE_DIR / "snapshots"
_KEY = ["date", "ticker", "kind"]


def _path(ts: datetime) -> Path:
    return SNAPSHOT_DIR / f"{ts:%Y-%m}.parquet"


def _row(ticker: str, score, data, ts: datetime) -> dict:
    """Flatten a Score plus the raw inputs that produced it into one record."""
    row: dict = {
        "date": pd.Timestamp(ts).normalize(),
        "ts": pd.Timestamp(ts),
        "ticker": ticker,
        "kind": getattr(score, "kind", None),
        "band": getattr(score, "band", None),
        "composite": getattr(score, "composite", None),
        "conviction": getattr(score, "conviction", None),
        "buy_score": getattr(score, "buy_score", None),
        "sell_score": getattr(score, "sell_score", None),
    }

    # Factor values, keyed by name. The ▲/▼ prefixes from holdings_signals are kept —
    # they distinguish the buy- and sell-side copies of same-named factors.
    for f in getattr(score, "factors", []) or []:
        row[f"f::{f.name}"] = f.value

    # Raw inputs, so a factor's mapping can be re-derived later without re-fetching.
    quote = getattr(data, "quote", None)
    rating = getattr(data, "rating", None)
    fund = getattr(data, "fundamentals", None)
    fv = getattr(data, "fair_value", None)
    tech = getattr(data, "tech", None) or {}

    row["in::price"] = getattr(quote, "price", None)
    row["in::target_mean"] = getattr(rating, "target_mean", None)
    row["in::target_low"] = getattr(rating, "target_low", None)
    row["in::target_high"] = getattr(rating, "target_high", None)
    row["in::analyst_count"] = getattr(rating, "analyst_count", None)
    row["in::rec_mean"] = getattr(rating, "recommendation_mean", None)
    row["in::trend_delta"] = getattr(rating, "trend_delta", None)
    row["in::trailing_pe"] = getattr(fund, "trailing_pe", None)
    row["in::forward_pe"] = getattr(fund, "forward_pe", None)
    row["in::sector"] = getattr(fund, "sector", None)
    row["in::market_cap"] = getattr(fund, "market_cap", None)
    row["in::profit_margin"] = getattr(fund, "profit_margin", None)
    row["in::revenue_growth"] = getattr(fund, "revenue_growth", None)
    row["in::fair_value"] = getattr(fv, "base", None)
    row["in::margin_of_safety"] = getattr(fv, "margin_of_safety", None) if fv else None
    row["in::rsi"] = tech.get("rsi")
    row["in::above_ma50"] = tech.get("above_ma50")
    row["in::above_ma200"] = tech.get("above_ma200")
    row["in::ret_3m"] = tech.get("ret_3m")
    row["in::n_news"] = len(getattr(data, "news", []) or [])
    row["in::n_trades"] = len(getattr(data, "trades", []) or [])
    return row


def record(entries, ts: Optional[datetime] = None) -> Optional[Path]:
    """Append snapshots for `entries` — an iterable of (ticker, score, data).

    One file per calendar month, deduped on (date, ticker, kind) keeping the latest, so
    re-running a page the same day overwrites rather than double-counting. Returns the
    file written, or None if nothing was recorded.
    """
    try:
        ts = ts or datetime.now(timezone.utc)
        rows = [_row(t, s, d, ts) for t, s, d in entries]
        if not rows:
            return None

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = _path(ts)
        fresh = pd.DataFrame(rows)

        if path.exists():
            try:
                fresh = pd.concat([pd.read_parquet(path), fresh], ignore_index=True)
            except Exception:
                pass  # unreadable/partial file: start over rather than lose this run

        fresh = fresh.drop_duplicates(subset=_KEY, keep="last")
        fresh.to_parquet(path, index=False)
        return path
    except Exception:
        # Research telemetry must never break a page render.
        return None


def load(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Read recorded snapshots back into one frame, optionally date-filtered."""
    if not SNAPSHOT_DIR.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(SNAPSHOT_DIR.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)
