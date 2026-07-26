"""Dividend Discount Model (Gordon growth) + Graham-number sanity check.

DDM only applies to dividend payers; returns None otherwise. The Graham number is a
classic conservative heuristic (sqrt(22.5 * EPS * book value per share)) offered as a
sanity check, not a precise target.
"""
from __future__ import annotations

import math
from typing import Optional

from ..models import Fundamentals


def ddm_value(fund: Fundamentals, assumptions: dict) -> tuple[Optional[float], dict]:
    d0 = fund.dividend_rate
    if not d0 or d0 <= 0:
        return None, {"skipped": "not a dividend payer"}

    r = float(assumptions.get("discount_rate", 0.10))
    # Dividend growth: conservative — half the general default, capped below r.
    g = 0.5 * float(assumptions.get("default_growth", 0.08))
    if r - g < 0.02:
        g = r - 0.02
    if r <= g:
        return None, {"skipped": "discount rate <= dividend growth"}

    d1 = d0 * (1 + g)
    value = d1 / (r - g)
    return value, {"dividend": d0, "growth": round(g, 4), "discount_rate": r}


def graham_number(fund: Fundamentals) -> Optional[float]:
    eps = fund.eps_trailing
    bvps = fund.book_value_per_share
    if not eps or eps <= 0 or not bvps or bvps <= 0:
        return None
    return math.sqrt(22.5 * eps * bvps)
