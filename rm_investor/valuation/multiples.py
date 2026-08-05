"""Relative (multiples) valuation.

Applies target multiples (P/E, P/S, P/B, EV/EBITDA) to the company's own metrics to
derive per-share fair values, then takes their median.

Target multiples are **sector-aware** and **growth-adjusted** rather than a single
market median. Applying P/E 18 / P/S 3 to every company was the largest source of
fair-value distortion: it values a hypergrowth software name like a utility, so the
resulting "fair value" disagreed with the DCF by up to 20x and pinned most holdings
at "overvalued". Sector targets come from `valuation.sector_multiples`; a growth
tilt then nudges the target up or down for companies growing faster/slower than the
growth already priced into that sector median.

Returns (median_value | None, inputs_dict) where inputs holds every sub-method value
plus the effective targets actually used, so the number stays auditable.
"""
from __future__ import annotations

from statistics import median
from typing import Optional

from ..models import Fundamentals


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def effective_multiples(fund: Fundamentals, assumptions: dict) -> tuple[dict, dict]:
    """Resolve the target multiples for this company: sector median, then growth tilt.

    Returns (targets, meta) where meta explains what was applied.
    """
    base = dict(assumptions.get("multiples", {}) or {})
    by_sector = assumptions.get("sector_multiples", {}) or {}
    meta: dict = {}

    sector = getattr(fund, "sector", None)
    if sector and sector in by_sector:
        base = {**base, **(by_sector[sector] or {})}
        meta["sector"] = sector
    else:
        meta["sector"] = f"{sector or 'unknown'} (market median fallback)"

    adj = assumptions.get("growth_multiple_adjust", {}) or {}
    if adj.get("enabled", False):
        g = fund.revenue_growth
        if g is not None:
            baseline = float(adj.get("baseline_growth", 0.08))
            sens = float(adj.get("sensitivity", 2.0))
            lo = float(adj.get("min_factor", 0.7))
            hi = float(adj.get("max_factor", 1.8))
            factor = _clamp(1.0 + sens * (g - baseline), lo, hi)
            base = {k: v * factor for k, v in base.items()}
            meta["growth_factor"] = round(factor, 3)
            meta["growth"] = round(g, 4)

    return base, meta


def multiples_value(fund: Fundamentals, assumptions: dict) -> tuple[Optional[float], dict]:
    m, meta = effective_multiples(fund, assumptions)
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
    targets = {k: round(v, 2) for k, v in m.items()}
    inputs = {"targets": targets, **meta, **{f"value_{k}": round(v, 2) for k, v in subs.items()}}
    return val, inputs
