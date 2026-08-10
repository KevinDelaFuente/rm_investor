"""Sell-signal composite: when is it a good time to trim or exit a holding?

Combines valuation, price-vs-target, momentum/overextension, analyst deterioration,
concentration, news, and disclosed smart-money selling into a SELL/TRIM/HOLD call.
"""
from __future__ import annotations

from ..config import get_config
from ..models import Position, Score
from . import band, build_score, composite
from .factors import (
    f_analyst_deterioration,
    f_concentration,
    f_momentum_overbought,
    f_news_negative,
    f_overvaluation,
    f_pe_stretch,
    f_price_vs_target,
    f_smart_money_selling,
    high_forward_pe_flag,
    sector_baseline_pe,
)

_cfg = get_config()


def score_holding(position: Position, data, cfg=None) -> Score:
    cfg = cfg or _cfg
    w = cfg.path("scoring.sell.weights", {})
    # Judge P/E against the company's own sector, matching holdings_signals — a flat
    # market P/E of 18 had these pages disagreeing about the same stock.
    baseline_pe = sector_baseline_pe(data.fundamentals, cfg)
    ticker = position.ticker

    factors = [f for f in [
        f_price_vs_target(data.quote, data.rating, w.get("price_vs_target", 0.15)),
        f_overvaluation(data.fair_value, w.get("overvaluation", 0.20)),
        f_pe_stretch(data.fundamentals, baseline_pe, w.get("pe_stretch", 0.10)),
        f_momentum_overbought(data.tech, w.get("momentum_overbought", 0.15)),
        f_analyst_deterioration(data.rating, w.get("analyst_deterioration", 0.15)),
        f_concentration(position, w.get("concentration", 0.10), cfg),
        f_news_negative(data.news, w.get("news_negative", 0.075)),
        f_smart_money_selling(data.trades, w.get("smart_money_selling", 0.075)),
    ] if f is not None]
    value = composite(factors)
    bands = cfg.path("scoring.sell.bands", {"sell": 67, "trim": 45})
    label = band(value, bands, ("SELL", "TRIM", "HOLD"))
    flags = [x for x in [high_forward_pe_flag(data.fundamentals, baseline_pe)] if x]
    return build_score(ticker, "sell", factors, label, flags=flags)
