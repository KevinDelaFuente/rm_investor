"""Information-coefficient statistics over a point-in-time panel.

The IC is the cross-sectional rank correlation between a factor's value at date `t` and
realised forward returns. Averaged across many dates it answers the question a scoring
model must answer before anyone allocates on it: does this factor rank future returns,
and how reliably?

Reading the output:
  * `mean_ic`   — average Spearman rank IC. 0.02-0.05 is a normal, useful equity factor;
                  above ~0.10 on free data usually means a bug or look-ahead, not alpha.
  * `ic_ir`     — mean / std of the IC series. Consistency, not magnitude.
  * `t_adj`     — overlap-adjusted t-stat. USE THIS ONE, not `t_naive`. Monthly sampling
                  of an h-month forward return produces h-fold overlapping observations,
                  so the naive t-stat overstates significance by roughly sqrt(h).
  * `hit_rate`  — fraction of dates with IC > 0.
  * `q_spread`  — mean forward return of the top quintile minus the bottom.

A negative mean IC is not a dead factor — it is a correctly-signed factor pointing the
wrong way. That distinction is the entire purpose of running this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NAMES_PER_DATE = 10


def rank_ic(
    panel: pd.DataFrame,
    feature: str,
    horizon: int,
    min_names: int = MIN_NAMES_PER_DATE,
) -> pd.Series:
    """Per-date Spearman rank IC between `feature` and the `horizon`-month return.

    Computed as Pearson correlation of the ranks, which is the definition of Spearman
    and avoids a scipy dependency (`Series.corr(method="spearman")` imports scipy).

    No winsorizing: ranks are already outlier-insensitive. Dates with too thin a
    cross-section are dropped rather than contributing noise.
    """
    target = f"fwd_ret_{horizon}m"
    if feature not in panel or target not in panel:
        return pd.Series(dtype="float64")

    out: dict[pd.Timestamp, float] = {}
    for date, cross in panel.groupby(level="date"):
        pair = cross[[feature, target]].dropna()
        if len(pair) < min_names:
            continue
        # A constant cross-section has no ranking information (and yields NaN).
        if pair[feature].nunique() < 2 or pair[target].nunique() < 2:
            continue
        out[date] = pair[feature].rank().corr(pair[target].rank())

    return pd.Series(out, dtype="float64").sort_index().dropna()


def quantile_spread(
    panel: pd.DataFrame,
    feature: str,
    horizon: int,
    q: int = 5,
    min_names: int = MIN_NAMES_PER_DATE,
) -> float:
    """Mean forward return of the top quantile minus the bottom, averaged over dates."""
    target = f"fwd_ret_{horizon}m"
    if feature not in panel or target not in panel:
        return np.nan

    spreads: list[float] = []
    for _, cross in panel.groupby(level="date"):
        pair = cross[[feature, target]].dropna()
        if len(pair) < max(min_names, q * 2) or pair[feature].nunique() < q:
            continue
        try:
            buckets = pd.qcut(pair[feature].rank(method="first"), q, labels=False)
        except ValueError:
            continue
        top = pair[target][buckets == q - 1].mean()
        bot = pair[target][buckets == 0].mean()
        if pd.notna(top) and pd.notna(bot):
            spreads.append(float(top - bot))

    return float(np.mean(spreads)) if spreads else np.nan


def summarize(ic: pd.Series, horizon: int) -> dict:
    """Collapse an IC series to the numbers worth putting in a table."""
    n = int(ic.notna().sum())
    if n < 2:
        return {
            "n_periods": n, "mean_ic": np.nan, "std_ic": np.nan,
            "ic_ir": np.nan, "t_naive": np.nan, "t_adj": np.nan, "hit_rate": np.nan,
        }

    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    ir = mean / std if std > 0 else np.nan
    t_naive = ir * np.sqrt(n) if pd.notna(ir) else np.nan
    # Monthly sampling of an h-month return overlaps h-fold; deflate accordingly.
    t_adj = t_naive / np.sqrt(max(horizon, 1)) if pd.notna(t_naive) else np.nan

    return {
        "n_periods": n,
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": ir,
        "t_naive": t_naive,
        "t_adj": t_adj,
        "hit_rate": float((ic > 0).mean()),
    }


def study(
    panel: pd.DataFrame,
    features: list[str],
    horizons: tuple[int, ...] = (1, 3, 6, 12),
    q: int = 5,
) -> pd.DataFrame:
    """Full feature x horizon table, sorted by absolute overlap-adjusted t-stat."""
    rows = []
    for feature in features:
        if feature not in panel:
            continue
        for h in horizons:
            ic = rank_ic(panel, feature, h)
            row = {"feature": feature, "horizon_m": h}
            row.update(summarize(ic, h))
            row["q_spread"] = quantile_spread(panel, feature, h, q=q)
            row["coverage"] = float(panel[feature].notna().mean())
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    return table.reindex(
        table["t_adj"].abs().sort_values(ascending=False, na_position="last").index
    ).reset_index(drop=True)
