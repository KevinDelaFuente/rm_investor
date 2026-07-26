"""Build the moonshot universe: small/micro-cap, under-followed names.

Starts from a shipped small-cap list and (optionally) enforces the market-cap ceiling
during scanning. Bounded to `moonshot.max_size`. This is a curated sample, NOT an
exhaustive market sweep — that limit is surfaced in the UI.
"""
from __future__ import annotations

from typing import Optional

from ..config import get_config
from .universe import load_ticker_csv

_cfg = get_config()


def build_moonshot_universe(cfg=None) -> list[str]:
    cfg = cfg or _cfg
    m = cfg.get("moonshot", {}) or {}
    tickers = load_ticker_csv(cfg.resolve(m.get("universe_csv", "")))

    seen, out = set(), []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out[: int(m.get("max_size", 40))]


def max_market_cap(cfg=None) -> float:
    cfg = cfg or _cfg
    return float((cfg.get("moonshot", {}) or {}).get("max_market_cap", 3e9))
