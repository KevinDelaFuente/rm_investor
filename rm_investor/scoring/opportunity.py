"""Opportunity composite: how strong is the buy case for a candidate?

Combines analyst upside (weighted by coverage depth), undervaluation vs. fair value,
valuation reasonableness, constructive momentum, news, and disclosed smart-money buying
into a STRONG / WATCH / PASS call.
"""
from __future__ import annotations

from ..config import get_config
from ..models import Score
from . import band, build_score, composite
from .factors import (
    f_analyst_upside,
    f_momentum_constructive,
    f_news_positive,
    f_smart_money_buying,
    f_undervaluation,
    f_valuation_reasonable,
    high_forward_pe_flag,
    sector_baseline_pe,
)

_cfg = get_config()


def score_candidate(ticker: str, data, cfg=None) -> Score:
    cfg = cfg or _cfg
    w = cfg.path("scoring.opportunity.weights", {})
    # Sector baseline, matching holdings_signals. Using a flat market P/E of 18 here
    # meant the scanner and the holdings page disagreed on whether a name was expensive.
    baseline_pe = sector_baseline_pe(data.fundamentals, cfg)

    factors = [f for f in [
        f_analyst_upside(data.quote, data.rating, w.get("analyst_upside", 0.25)),
        f_undervaluation(data.fair_value, w.get("undervaluation", 0.25)),
        f_valuation_reasonable(data.fundamentals, baseline_pe, w.get("valuation_reasonable", 0.10)),
        f_momentum_constructive(data.tech, w.get("momentum", 0.15)),
        f_news_positive(data.news, w.get("news_positive", 0.10)),
        f_smart_money_buying(data.trades, w.get("smart_money_buying", 0.15)),
    ] if f is not None]
    value = composite(factors)
    bands = cfg.path("scoring.opportunity.bands", {"strong": 67, "watch": 50})
    label = band(value, bands, ("STRONG", "WATCH", "PASS"))
    flags = [x for x in [high_forward_pe_flag(data.fundamentals, baseline_pe)] if x]
    return build_score(ticker, "opportunity", factors, label, flags=flags)
