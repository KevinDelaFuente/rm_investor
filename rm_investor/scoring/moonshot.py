"""Moonshot composite: asymmetric high-risk/high-reward small & micro-caps.

A SEPARATE weight profile from the core Opportunity score — it rewards small size,
accelerating growth, being under-followed, momentum breakouts, and disclosed
accumulation. Risk flags (unprofitable, volatile, thinly traded, indebted) are surfaced
alongside the score, never hidden. This is a small, speculative sleeve by design.
"""
from __future__ import annotations

from ..config import get_config
from ..models import Score
from . import band, build_score, composite
from .factors import (
    f_accumulation,
    f_momentum_breakout,
    f_revenue_growth,
    f_small_cap,
    f_under_followed,
    moonshot_risk_flags,
)

_cfg = get_config()


def score_moonshot(ticker: str, data, cfg=None) -> Score:
    cfg = cfg or _cfg
    w = cfg.path("scoring.moonshot.weights", {})

    factors = [
        f_small_cap(data.fundamentals, w.get("small_cap", 0.20)),
        f_revenue_growth(data.fundamentals, w.get("revenue_growth", 0.25)),
        f_under_followed(data.rating, w.get("under_followed", 0.15)),
        f_momentum_breakout(data.tech, w.get("momentum_breakout", 0.20)),
        f_accumulation(data.trades, w.get("accumulation", 0.20)),
    ]
    value = composite(factors)
    bands = cfg.path("scoring.moonshot.bands", {"high": 65, "medium": 45})
    label = band(value, bands, ("HIGH", "MEDIUM", "LOW"))
    flags = moonshot_risk_flags(data.fundamentals, data.tech)
    return build_score(ticker, "moonshot", factors, label, flags=flags)
