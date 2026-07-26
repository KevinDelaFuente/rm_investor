"""Congressional trading adapter — public STOCK Act disclosures.

Sources the community-maintained House & Senate "stock watcher" datasets (public
JSON on S3). These are disclosed, lagged filings (up to ~45 days late) — a signal
of what elected officials publicly did weeks ago, NOT insider information.

Degrades to [] on any network/parse error.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import requests

from ..config import get_config
from ..models import ActorType, Trade, TradeSide
from .cache import disk_cache

_cfg = get_config()
_TTL = _cfg.path("cache.filings_ttl_seconds", 604800)
_C = _cfg.path("smart_money.congress", {}) or {}
_LOOKBACK = int(_C.get("lookback_days", 180))


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _side(txn_type: Optional[str]) -> TradeSide:
    t = (txn_type or "").lower()
    if "purchase" in t or "buy" in t:
        return TradeSide.BUY
    if "sale" in t or "sell" in t:
        return TradeSide.SELL
    return TradeSide.UNKNOWN


@disk_cache(_TTL)
def _fetch(url: str) -> list[dict]:
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "RM-Investor/0.1"})
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _normalize(rows: list[dict], chamber: str) -> list[Trade]:
    trades: list[Trade] = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip().upper()
        if ticker in ("", "--", "N/A"):
            ticker = None
        txn_date = _parse_date(row.get("transaction_date"))
        trades.append(
            Trade(
                ticker=ticker,
                actor=row.get("representative") or row.get("senator") or "Unknown",
                actor_type=ActorType.CONGRESS,
                side=_side(row.get("type")),
                amount=row.get("amount"),
                transaction_date=txn_date,
                disclosed_date=_parse_date(row.get("disclosure_date")),
                source=chamber,
            )
        )
    return trades


class CongressSource:
    """SmartMoneySource for congressional disclosures (duck-typed, no ABC import)."""

    name = "congress"

    def get_trades(self, ticker: Optional[str] = None, limit: int = 200) -> list[Trade]:
        rows: list[Trade] = []
        if _C.get("house_url"):
            rows += _normalize(_fetch(_C["house_url"]), "House")
        if _C.get("senate_url"):
            rows += _normalize(_fetch(_C["senate_url"]), "Senate")

        cutoff = date.today() - timedelta(days=_LOOKBACK)
        rows = [t for t in rows if (t.transaction_date is None or t.transaction_date >= cutoff)]
        if ticker:
            tk = ticker.upper()
            rows = [t for t in rows if t.ticker == tk]

        rows.sort(key=lambda t: t.transaction_date or date.min, reverse=True)
        return rows[:limit]
