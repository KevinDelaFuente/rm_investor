"""Discounted cash flow (DCF) intrinsic value.

Projects free cash flow forward, discounts each year plus a Gordon-growth terminal
value to today, adds net cash, and divides by shares. All assumptions come from
config (with per-ticker overrides) and are returned for display.

Two refinements over a naive DCF, both aimed at not distorting the signal:

* **Growth fade** - a flat growth cap (20%) was binding on ~2/3 of holdings, valuing
  a 25%-grower and a 60%-grower identically. Instead the company's own growth is
  allowed through (up to a sanity ceiling) and then decays linearly toward a modest
  terminal rate across the projection window. That is both standard practice and
  more discriminating than truncation.
* **Risk-adjusted discount rate** - discounting a speculative micro-cap at the same
  10% as a mega-cap understates its risk. Size and profitability premiums are added
  and the result is clamped.

Returns (per_share_value | None, inputs_dict). None with a reason when it can't run
(no positive FCF, no share count) - never fabricates a number.
"""
from __future__ import annotations

from typing import Optional

from ..models import Fundamentals


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def resolve_discount_rate(fund: Fundamentals, assumptions: dict) -> tuple[float, dict]:
    """Base discount rate plus size / profitability risk premiums."""
    r = float(assumptions.get("discount_rate", 0.10))
    adj = assumptions.get("discount_adjust", {}) or {}
    meta: dict = {}
    if not adj.get("enabled", False):
        return r, meta

    premiums = []
    mcap = fund.market_cap
    if mcap:
        if mcap < float(adj.get("micro_cap_threshold", 3.0e8)):
            premiums.append(("micro cap", float(adj.get("micro_cap_premium", 0.05))))
        elif mcap < float(adj.get("small_cap_threshold", 2.0e9)):
            premiums.append(("small cap", float(adj.get("small_cap_premium", 0.03))))
    eps = fund.eps_trailing
    if eps is not None and eps < 0:
        premiums.append(("unprofitable", float(adj.get("no_profit_premium", 0.02))))

    if premiums:
        r += sum(p for _, p in premiums)
        meta["risk_premiums"] = {name: p for name, p in premiums}
    r = _clamp(r, float(adj.get("min_rate", 0.07)), float(adj.get("max_rate", 0.20)))
    meta["discount_rate_effective"] = round(r, 4)
    return r, meta


def _growth_path(fund: Fundamentals, assumptions: dict, years: int) -> tuple[list[float], dict]:
    """Per-year growth rates. With fade enabled, decay from the company's own growth
    toward `fade_to`; otherwise hold a flat capped rate (legacy behaviour)."""
    default_g = float(assumptions.get("default_growth", 0.08))
    fade = assumptions.get("growth_fade", {}) or {}
    g0 = fund.revenue_growth if fund.revenue_growth is not None else default_g

    if not fade.get("enabled", False):
        max_g = float(assumptions.get("max_growth", 0.20))
        g = _clamp(g0, -0.10, max_g)
        return [g] * years, {"growth": round(g, 4), "growth_mode": "flat"}

    hi = float(fade.get("max_initial_growth", 0.40))
    end = float(fade.get("fade_to", 0.04))
    g0 = _clamp(g0, -0.10, hi)
    if years <= 1:
        path = [g0]
    else:
        step = (g0 - end) / (years - 1)
        path = [g0 - step * t for t in range(years)]
    return path, {
        "growth_initial": round(g0, 4),
        "growth_final": round(path[-1], 4),
        "growth_mode": "fade",
    }


def resolve_fcf(fund: Fundamentals, assumptions: dict) -> tuple[Optional[float], dict]:
    """Free cash flow, with a sanity check against operating cash flow.

    yfinance's `freeCashflow` is intermittently wrong by an order of magnitude for
    large caps (observed: MSFT reporting ~$16B against ~$180B operating cash flow).
    Trusting it blindly produced absurd DCF values and made mega-caps look wildly
    overvalued. When FCF is implausibly small relative to OCF we fall back to a
    conservative fraction of OCF and say so in the inputs, rather than silently
    valuing the company off a broken number.
    """
    fcf = fund.free_cash_flow
    ocf = fund.operating_cash_flow
    chk = assumptions.get("fcf_sanity", {}) or {}
    meta: dict = {}
    if not chk.get("enabled", False) or not ocf or ocf <= 0:
        return fcf, meta

    min_ratio = float(chk.get("min_fcf_to_ocf", 0.25))
    fallback = float(chk.get("ocf_fallback_ratio", 0.65))
    if fcf is None or fcf <= 0 or fcf < ocf * min_ratio:
        meta["fcf_source"] = (
            f"reported FCF {(fcf or 0) / 1e9:.1f}B implausible vs OCF {ocf / 1e9:.1f}B; "
            f"using {fallback:.0%} of OCF"
        )
        return ocf * fallback, meta
    return fcf, meta


def dcf_value(fund: Fundamentals, assumptions: dict) -> tuple[Optional[float], dict]:
    fcf, fcf_meta = resolve_fcf(fund, assumptions)
    shares = fund.shares_outstanding

    if not fcf or fcf <= 0:
        return None, {"skipped": "no positive free cash flow"}
    if not shares or shares <= 0:
        return None, {"skipped": "unknown shares outstanding"}

    tg = float(assumptions.get("terminal_growth", 0.025))
    years = int(assumptions.get("projection_years", 5))

    r, rate_meta = resolve_discount_rate(fund, assumptions)
    if r <= tg:
        r = tg + 0.02  # keep terminal value finite

    growth_path, growth_meta = _growth_path(fund, assumptions, years)

    pv_flows = 0.0
    fcf_t = fcf
    for t, g in enumerate(growth_path, start=1):
        fcf_t *= (1 + g)
        pv_flows += fcf_t / ((1 + r) ** t)

    terminal = fcf_t * (1 + tg) / (r - tg)
    pv_terminal = terminal / ((1 + r) ** years)

    net_cash = (fund.total_cash or 0.0) - (fund.total_debt or 0.0)
    equity_value = pv_flows + pv_terminal + net_cash
    per_share = equity_value / shares

    inputs = {
        "fcf": fcf,
        "discount_rate": round(r, 4),
        "terminal_growth": tg,
        "years": years,
        "net_cash": net_cash,
        "equity_value": equity_value,
        **growth_meta,
        **rate_meta,
        **fcf_meta,
    }
    return (per_share if per_share > 0 else None), inputs
