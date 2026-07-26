"""Deterministic scoring: composites over transparent 0-100 factors."""
from __future__ import annotations

from ..models import FactorScore, Score


def composite(factors: list[FactorScore]) -> float:
    """Weight-normalized 0-100 composite (robust if weights don't sum to 1)."""
    total_w = sum(f.weight for f in factors)
    if total_w <= 0:
        return 0.0
    return round(sum(f.contribution for f in factors) / total_w, 1)


def band(value: float, thresholds: dict, labels: tuple[str, str, str]) -> str:
    """Map a composite to a 3-tier label given {hi_key, lo_key} thresholds."""
    hi_key, lo_key = list(thresholds.keys())[:2]
    if value >= thresholds[hi_key]:
        return labels[0]
    if value >= thresholds[lo_key]:
        return labels[1]
    return labels[2]


def build_score(ticker: str, kind: str, factors: list[FactorScore], band_label: str,
                flags: list[str] | None = None) -> Score:
    return Score(
        ticker=ticker,
        kind=kind,
        composite=composite(factors),
        band=band_label,
        factors=factors,
        flags=flags or [],
    )
