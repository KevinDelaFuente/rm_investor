"""Research utilities: point-in-time panel construction, IC statistics, and the
forward-looking snapshot recorder.

Nothing here is imported by the Streamlit pages' hot path except `snapshots.record`,
which is fire-and-forget.
"""
from __future__ import annotations

from .ic import quantile_spread, rank_ic, study, summarize
# Exported under a distinct name: re-exporting it as `neutralize` would shadow the
# `neutralize` submodule, so `from rm_investor.research import neutralize` would hand
# back the function instead of the module.
from .neutralize import compare as compare_neutralized
from .neutralize import neutralize as neutralize_factors
from .panel import FEATURES, RISK_FACTORS, build_panel, fetch_grades, fetch_history

__all__ = [
    "FEATURES",
    "RISK_FACTORS",
    "build_panel",
    "compare_neutralized",
    "fetch_grades",
    "fetch_history",
    "neutralize_factors",
    "quantile_spread",
    "rank_ic",
    "study",
    "summarize",
]
