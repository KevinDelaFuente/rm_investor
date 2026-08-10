"""Strip risk-factor loadings out of a signal before measuring its IC.

A raw information coefficient cannot tell skill from exposure. A "factor" that merely
selects high-beta, high-volatility names will post a superb IC in a rising market and
destroy capital in a falling one — it is not alpha, it is leverage wearing a signal's
clothes. The IC study on this book produced exactly that pattern: `tgt_dispersion_90d`
scored the highest mean IC of any analyst factor (+0.074 at 12m, quintile spread +0.37)
across a universe stuffed with high-volatility growth names in an eight-year bull run.

Neutralization answers the question directly. For each rebalance date, regress the
factor cross-sectionally on sector dummies plus the risk exposures, and keep the
residual — the part of the factor that is NOT explained by what it was loading on. If
the residual IC survives, the signal is real. If it collapses, the factor was a risk
proxy and belongs nowhere near a conviction score.

    raw IC survives neutralization      -> plausible alpha, worth weighting
    raw IC collapses toward zero        -> it was beta/sector/size all along
    residual IC exceeds raw IC          -> a real signal that was being masked by
                                           an offsetting risk exposure

Ordinary least squares per date via `numpy.linalg.lstsq`, so no statsmodels dependency.
Dates whose cross-section is too thin or whose design matrix is rank-deficient yield NaN
residuals rather than fabricated ones.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .panel import RISK_FACTORS

MIN_NAMES_FOR_FIT = 20
SUFFIX = "_neut"
# Below this raw |IC|, the decay ratio is dividing by noise and is not reportable.
MIN_IC_FOR_DECAY = 0.005


def _design(cross: pd.DataFrame, risk_cols: Sequence[str], sector_col: Optional[str]) -> pd.DataFrame:
    """Intercept + z-scored risk exposures + sector dummies, for one date."""
    parts = [pd.Series(1.0, index=cross.index, name="_const")]

    for col in risk_cols:
        if col not in cross:
            continue
        v = pd.to_numeric(cross[col], errors="coerce")
        # Median/IQR standardisation: robust to the fat tails these exposures have.
        med = v.median()
        scale = v.quantile(0.75) - v.quantile(0.25)
        if pd.isna(med) or not scale or not np.isfinite(scale):
            continue
        parts.append(((v - med) / scale).clip(-5, 5).rename(col))

    if sector_col and sector_col in cross:
        sectors = cross[sector_col].astype("object").fillna("__unknown__")
        if sectors.nunique() > 1:
            # drop_first avoids perfect collinearity with the intercept.
            dummies = pd.get_dummies(sectors, prefix="sec", drop_first=True, dtype=float)
            parts.append(dummies)

    return pd.concat(parts, axis=1)


def _residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """OLS residuals of y on X, computed only where both are present.

    Rank deficiency is deliberately tolerated. Correlated risk exposures (volatility and
    beta are near-collinear on real data) make the *coefficients* non-unique, but the
    projection onto the column space — and therefore the residual — is still unique, and
    the residual is all we want. `lstsq` returns the minimum-norm solution, so bailing
    out on rank would throw away perfectly good neutralizations.
    """
    out = pd.Series(np.nan, index=y.index, dtype="float64")
    ok = y.notna() & X.notna().all(axis=1)
    if int(ok.sum()) < MIN_NAMES_FOR_FIT:
        return out

    Xv = X[ok].to_numpy(dtype="float64")
    yv = y[ok].to_numpy(dtype="float64")
    # A constant factor has no variance to explain; leave it as NaN rather than
    # returning an all-zero "residual" that would read as a dead signal.
    if not np.isfinite(Xv).all() or not np.isfinite(yv).all() or yv.std() == 0:
        return out

    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    out.loc[ok] = yv - Xv @ beta
    return out


def neutralize(
    panel: pd.DataFrame,
    features: Iterable[str],
    risk_cols: Sequence[str] = RISK_FACTORS,
    sector_col: Optional[str] = "sector",
    suffix: str = SUFFIX,
) -> pd.DataFrame:
    """Return `panel` with an extra `<feature><suffix>` column per feature.

    Residuals are computed within each date, so the output is directly comparable to the
    raw column and can be fed straight to `ic.rank_ic`.
    """
    features = [f for f in features if f in panel]
    if not features:
        return panel.copy()

    out = panel.copy()
    for f in features:
        out[f + suffix] = np.nan

    for date, cross in panel.groupby(level="date"):
        X = _design(cross, risk_cols, sector_col)
        if X.shape[1] < 2:          # nothing but the intercept: nothing to strip
            continue
        for f in features:
            resid = _residualize(pd.to_numeric(cross[f], errors="coerce"), X)
            out.loc[resid.index, f + suffix] = resid.to_numpy()

    return out


def compare(
    panel: pd.DataFrame,
    features: Iterable[str],
    horizons: Iterable[int] = (1, 3, 6, 12),
    risk_cols: Sequence[str] = RISK_FACTORS,
    sector_col: Optional[str] = "sector",
) -> pd.DataFrame:
    """Raw vs neutralized IC side by side — the table that settles 'is it alpha?'.

    `ic_decay` is the fraction of raw IC destroyed by neutralization. Near 1.0 means the
    factor was almost entirely a risk proxy; negative means neutralization *strengthened*
    the signal, i.e. a real effect was being masked by an offsetting risk exposure.
    It is left NaN when the raw IC is too close to zero for the ratio to mean anything —
    dividing a small residual by a near-zero baseline produces impressive nonsense.
    """
    from .ic import rank_ic, summarize

    features = [f for f in features if f in panel]
    neut = neutralize(panel, features, risk_cols=risk_cols, sector_col=sector_col)

    rows = []
    for f in features:
        for h in horizons:
            raw = summarize(rank_ic(neut, f, h), h)
            res = summarize(rank_ic(neut, f + SUFFIX, h), h)
            raw_ic, res_ic = raw["mean_ic"], res["mean_ic"]
            decay = np.nan
            if pd.notna(raw_ic) and pd.notna(res_ic) and abs(raw_ic) >= MIN_IC_FOR_DECAY:
                decay = 1.0 - (abs(res_ic) / abs(raw_ic))
            rows.append({
                "feature": f,
                "horizon_m": h,
                "raw_ic": raw_ic,
                "neut_ic": res_ic,
                "raw_t": raw["t_adj"],
                "neut_t": res["t_adj"],
                "ic_decay": decay,
                "n_periods": res["n_periods"],
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.reindex(
        table["neut_t"].abs().sort_values(ascending=False, na_position="last").index
    ).reset_index(drop=True)
