"""Relative (multiples) valuation.

Applies target multiples (P/E, P/S, P/B, EV/EBITDA) to the company's own metrics to
derive per-share fair values, then takes their median. Target multiples default to the
market-median values in config; swap in sector/peer medians as you add that data.

Returns (median_value | None, inputs_dict) where inputs holds every sub-method value.
"""
from __future__ import annotations

from statistics import median
from typing import Optional

from ..models import Fundamentals


def multiples_value(fund: Fundamentals, assumptions: dict) -> tuple[Optional[float], dict]:
    m = assumptions.get("multiples", {}) or {}
    shares = fund.shares_outstanding
    subs: dict[str, float] = {}

    # P/E: fair price = EPS * target P/E
    if fund.eps_trailing and fund.eps_trailing > 0 and m.get("pe"):
        subs["pe"] = fund.eps_trailing * float(m["pe"])

    # P/S: fair price = (revenue / shares) * target P/S
    if fund.revenue and shares and shares > 0 and m.get("ps"):
        subs["ps"] = (fund.revenue / shares) * float(m["ps"])

    # P/B: fair price = book value per share * target P/B
    if fund.book_value_per_share and fund.book_value_per_share > 0 and m.get("pb"):
        subs["pb"] = fund.book_value_per_share * float(m["pb"])

    # EV/EBITDA: EV = EBITDA * target; equity = EV - debt + cash; per share
    if fund.ebitda and fund.ebitda > 0 and shares and shares > 0 and m.get("ev_ebitda"):
        ev = fund.ebitda * float(m["ev_ebitda"])
        equity = ev - (fund.total_debt or 0.0) + (fund.total_cash or 0.0)
        if equity > 0:
            subs["ev_ebitda"] = equity / shares

    if not subs:
        return None, {"skipped": "insufficient metrics for any multiple"}

    val = float(median(subs.values()))
    inputs = {"targets": m, **{f"value_{k}": round(v, 2) for k, v in subs.items()}}
    return val, inputs
