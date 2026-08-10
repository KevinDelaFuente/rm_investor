"""Holdings signal: for a stock you ALREADY OWN, should you sell, hold, or add?

Unlike `sell_signals` (exit case only) and `opportunity` (candidates you don't own),
this scores each holding on BOTH sides and nets them:

    conviction = buy_composite - sell_composite      # -100 .. +100

The sell side reuses the exit factors (overvaluation, overbought, analyst
deterioration, concentration, negative news, smart-money selling); the buy
side reuses the opportunity factors (analyst upside, undervaluation, reasonable
valuation, constructive momentum, positive news, smart-money buying). Netting
them means "nothing wrong" (low sell score) is NOT mistaken for "actively
attractive" — a holding must earn a positive buy case to be rated ADD.

Factors whose data source is entirely absent return None and are dropped from the list
rather than contributing a neutral 50 (see `factors.py`), so the news pair currently
falls out on the default yfinance provider and its weight redistributes.

Costs no extra API calls: both factor sets read the same gathered TickerData.
"""
from __future__ import annotations

from typing import Optional

from ..config import get_config
from ..models import Position, Score
from . import build_score, composite
from .factors import (
    f_analyst_deterioration,
    f_analyst_upside,
    f_concentration,
    f_momentum_constructive,
    f_momentum_overbought,
    f_news_negative,
    f_news_positive,
    f_overvaluation,
    f_pe_stretch,
    f_price_vs_target,
    f_smart_money_buying,
    f_smart_money_selling,
    f_undervaluation,
    f_valuation_reasonable,
    high_forward_pe_flag,
    sector_baseline_pe,
)

_cfg = get_config()

# Ordered high -> low. Thresholds are on conviction (-100..+100) and are ABSOLUTE:
# each side is a 0-100 composite where ~50 is neutral, so a gap under 10 means the
# two cases effectively offset, and 30+ means one side clearly dominates. They are
# deliberately NOT calibrated to yield a set number of names per band.
_LABELS = ("STRONG BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL")
_DEFAULT_BANDS = {"strong_buy": 30, "overweight": 10, "underweight": -10, "sell": -30}


def band_conviction(conviction: float, bands: Optional[dict] = None) -> str:
    """Map a -100..+100 conviction onto the 5-tier holdings ladder."""
    b = {**_DEFAULT_BANDS, **(bands or {})}
    if conviction >= b["strong_buy"]:
        return "STRONG BUY"
    if conviction >= b["overweight"]:
        return "OVERWEIGHT"
    if conviction > b["underweight"]:
        return "HOLD"
    if conviction > b["sell"]:
        return "UNDERWEIGHT"
    return "SELL"


def score_holding_action(position: Position, data, cfg=None) -> Score:
    """Two-sided verdict for a holding. Returns a Score whose `composite` is the
    0..100 *buy-side* strength and whose `band` is the 5-tier action. The full
    conviction math is exposed via `.conviction`, `.sell_score`, `.buy_score`.
    """
    cfg = cfg or _cfg
    sw = cfg.path("scoring.sell.weights", {})
    bw = cfg.path("scoring.opportunity.weights", {})
    # Judge P/E against the company's own sector, not one market-wide 18.
    baseline_pe = sector_baseline_pe(data.fundamentals, cfg)
    ticker = position.ticker

    sell_factors = [f for f in [
        f_price_vs_target(data.quote, data.rating, sw.get("price_vs_target", 0.15)),
        f_overvaluation(data.fair_value, sw.get("overvaluation", 0.20)),
        f_pe_stretch(data.fundamentals, baseline_pe, sw.get("pe_stretch", 0.10)),
        f_momentum_overbought(data.tech, sw.get("momentum_overbought", 0.15)),
        f_analyst_deterioration(data.rating, sw.get("analyst_deterioration", 0.15)),
        f_concentration(position, sw.get("concentration", 0.10), cfg),
        f_news_negative(data.news, sw.get("news_negative", 0.075)),
        f_smart_money_selling(data.trades, sw.get("smart_money_selling", 0.075)),
    ] if f is not None]
    buy_factors = [f for f in [
        f_analyst_upside(data.quote, data.rating, bw.get("analyst_upside", 0.25)),
        f_undervaluation(data.fair_value, bw.get("undervaluation", 0.25)),
        f_valuation_reasonable(data.fundamentals, baseline_pe, bw.get("valuation_reasonable", 0.10)),
        f_momentum_constructive(data.tech, bw.get("momentum", 0.15)),
        f_news_positive(data.news, bw.get("news_positive", 0.10)),
        f_smart_money_buying(data.trades, bw.get("smart_money_buying", 0.15)),
    ] if f is not None]

    sell_score = composite(sell_factors)
    buy_score = composite(buy_factors)
    conviction = round(buy_score - sell_score, 1)

    bands = cfg.path("scoring.holdings.bands", None)
    label = band_conviction(conviction, bands)

    flags = [x for x in [high_forward_pe_flag(data.fundamentals, baseline_pe)] if x]

    # Show both sides in the breakdown, prefixed so the source is obvious.
    shown = (
        [f.model_copy(update={"name": f"▲ {f.name}"}) for f in buy_factors]
        + [f.model_copy(update={"name": f"▼ {f.name}"}) for f in sell_factors]
    )
    score = build_score(ticker, "holdings", shown, label, flags=flags)
    # build_score recomputes composite over the merged factor list; override with
    # the buy-side strength and attach the two-sided detail.
    score.composite = buy_score
    score.conviction = conviction
    score.sell_score = sell_score
    score.buy_score = buy_score
    return score
