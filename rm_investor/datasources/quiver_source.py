"""Quiver Quantitative adapter — congressional trading via API key.

Requires QUIVER_API_KEY (free/paid tiers at quiverquant.com). Congressional trades
carry tickers natively, so they flow straight into the smart-money scoring factors,
the scanner universe, and the portfolio-overlap view.

Public, LAGGED STOCK Act disclosures — a signal of what officials publicly did weeks
ago, not insider intel. Degrades to [] without a key or on any error.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import requests

from ..config import get_api_key, get_config
from ..models import ActorType, Trade, TradeSide
from .cache import disk_cache

_cfg = get_config()
_TTL = _cfg.path("cache.filings_ttl_seconds", 604800)
_LOOKBACK = int(_cfg.path("smart_money.congress.lookback_days", 180))
_BASE = "https://api.quiverquant.com/beta"


def _headers() -> Optional[dict]:
    key = get_api_key("QUIVER_API_KEY")
    if not key:
        return None
    # The quiverquant package authenticates with a "Token" scheme.
    return {"accept": "application/json", "Authorization": f"Token {key}"}


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[: len("2020-01-01T00:00:00") if "T" in fmt else 10], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _side(txn: Optional[str]) -> TradeSide:
    t = (txn or "").lower()
    if "purchase" in t or "buy" in t:
        return TradeSide.BUY
    if "sale" in t or "sell" in t or "sold" in t:
        return TradeSide.SELL
    return TradeSide.UNKNOWN


@disk_cache(_TTL)
def _fetch(path: str) -> list[dict]:
    headers = _headers()
    if not headers:
        return []
    try:
        r = requests.get(f"{_BASE}{path}", headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _normalize(rows: list[dict]) -> list[Trade]:
    out: list[Trade] = []
    for row in rows:
        ticker = (row.get("Ticker") or "").strip().upper() or None
        out.append(
            Trade(
                ticker=ticker,
                actor=row.get("Representative") or row.get("Senator") or row.get("Name") or "Unknown",
                actor_type=ActorType.CONGRESS,
                side=_side(row.get("Transaction") or row.get("Type")),
                amount=row.get("Range") or row.get("Amount"),
                transaction_date=_parse_date(row.get("TransactionDate") or row.get("Traded")),
                disclosed_date=_parse_date(row.get("ReportDate") or row.get("Filed")),
                source=f"Quiver/{row.get('House') or row.get('Chamber') or 'Congress'}",
            )
        )
    return out


class QuiverCongressSource:
    """SmartMoneySource for congressional trades via Quiver (duck-typed)."""

    name = "quiver_congress"

    def get_trades(self, ticker: Optional[str] = None, limit: int = 200) -> list[Trade]:
        if ticker:
            rows = _fetch(f"/historical/congresstrading/{ticker.upper()}")
        else:
            rows = _fetch("/live/congresstrading") or _fetch("/bulk/congresstrading")

        trades = _normalize(rows)
        cutoff = date.today() - timedelta(days=_LOOKBACK)
        trades = [t for t in trades if (t.transaction_date is None or t.transaction_date >= cutoff)]
        if ticker:
            tk = ticker.upper()
            trades = [t for t in trades if t.ticker == tk]
        trades.sort(key=lambda t: t.transaction_date or date.min, reverse=True)
        return trades[:limit]
