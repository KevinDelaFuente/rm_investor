"""Discounted cash flow (DCF) intrinsic value.

Projects free cash flow forward at a growth rate, discounts each year plus a
Gordon-growth terminal value to today, adds net cash, and divides by shares. All
assumptions come from config (with per-ticker overrides) and are returned for display.

Returns (per_share_value | None, inputs_dict). None with a reason when it can't run
(no positive FCF, no share count) — never fabricates a number.
"""
from __future__ import annotations

from typing import Optional

from ..models import Fundamentals


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dcf_value(fund: Fundamentals, assumptions: dict) -> tuple[Optional[float], dict]:
    fcf = fund.free_cash_flow
    shares = fund.shares_outstanding
    inputs: dict = {}

    if not fcf or fcf <= 0:
        return None, {"skipped": "no positive free cash flow"}
    if not shares or shares <= 0:
        return None, {"skipped": "unknown shares outstanding"}

    r = float(assumptions.get("discount_rate", 0.10))
    tg = float(assumptions.get("terminal_growth", 0.025))
    years = int(assumptions.get("projection_years", 5))
    default_g = float(assumptions.get("default_growth", 0.08))
    max_g = float(assumptions.get("max_growth", 0.20))

    # Prefer the company's own growth, capped to avoid runaway extrapolation.
    g = fund.revenue_growth if fund.revenue_growth is not None else default_g
    g = _clamp(g, -0.10, max_g)
    if r <= tg:
        r = tg + 0.02  # keep terminal value finite

    pv_flows = 0.0
    fcf_t = fcf
    for t in range(1, years + 1):
        fcf_t *= (1 + g)
        pv_flows += fcf_t / ((1 + r) ** t)

    terminal = fcf_t * (1 + tg) / (r - tg)
    pv_terminal = terminal / ((1 + r) ** years)

    net_cash = (fund.total_cash or 0.0) - (fund.total_debt or 0.0)
    equity_value = pv_flows + pv_terminal + net_cash
    per_share = equity_value / shares

    inputs = {
        "fcf": fcf,
        "growth": round(g, 4),
        "discount_rate": r,
        "terminal_growth": tg,
        "years": years,
        "net_cash": net_cash,
        "equity_value": equity_value,
    }
    return (per_share if per_share > 0 else None), inputs
