"""Point-in-time factor panel — the input to the IC study.

Builds a long DataFrame indexed by (date, ticker) holding, for every rebalance date,
each candidate factor's value alongside forward returns at several horizons. Feeding
that to `ic.py` answers the only question that matters for a scoring model: does this
factor actually rank future returns?

THE INVARIANT: every feature at rebalance date `t` is computed from data timestamped
`<= t`. Forward-return columns are the deliberate exception — they are the label, not a
feature. `tests/test_research.py` enforces this by rebuilding features from a truncated
history and demanding an exact match; if that test fails the whole study is worthless.

Price features are computed by calling `scoring.factors.technicals()` on a truncated
history, so the study measures *the production code path*, not a reimplementation that
might quietly disagree with it.

Analyst features come from yfinance `upgrades_downgrades`, which carries dated per-firm
rows back to ~2012 with `currentPriceTarget` / `priorPriceTarget`. Because each row is
dated, the price at publication can be recovered from price history — so an analyst's
implied return is measured against the price they actually saw, not today's. That
de-anchoring is the point: consensus `targetMeanPrice` is a rolling ~12-month average, so
a stock that has fallen hard shows large "upside" only because the targets are stale.
`tgt_upside_stale` reproduces that naive construction on purpose, as the control.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..datasources.cache import disk_cache
from ..scoring.factors import technicals

# Research pulls are large and change slowly; a day is plenty.
_RAW_TTL = 86_400

DEFAULT_HORIZONS = (1, 3, 6, 12)      # months
FRESH_WINDOW_DAYS = 90                 # "recent" analyst action
STALE_WINDOW_DAYS = 365                # the naive consensus lookback, for the control
HALF_LIFE_DAYS = 45.0                  # recency decay on analyst actions

_TRADING_DAYS_MONTH = 21
_TRADING_DAYS_YEAR = 252

FEATURES = (
    # --- price: the current model's momentum inputs, then the replicated alternative
    "rsi_14",
    "dist_ma50",
    "dist_ma200",
    "mom_3m",
    "mom_12_1",
    "vol_ratio",
    # --- analyst: naive control first, then the de-anchored / revision variants
    "tgt_upside_stale",
    "tgt_upside_fresh_90d",
    "tgt_revision_90d",
    "tgt_action_net_90d",
    "grade_net_90d",
    "tgt_dispersion_90d",
    "analyst_count_90d",
)

# Risk exposures to neutralize against, NOT candidate signals. A raw IC conflates skill
# with risk: a factor that merely selects high-beta names looks brilliant in a bull
# sample. Everything here is reconstructed from price history, so it is genuinely
# point-in-time — unlike `sector`, see `fetch_sector`.
RISK_FACTORS = ("beta_252d", "vol_252d", "log_dvol")

MARKET_PROXY = "SPY"



# --------------------------------------------------------------------------- #
# Raw pulls (cached to disk; ~2 calls per ticker for a whole study)
# --------------------------------------------------------------------------- #
def _naive(idx: pd.Index) -> pd.Index:
    """Drop timezone info so price history and grade dates are comparable.

    yfinance returns tz-aware daily bars but tz-naive `GradeDate`s; comparing the two
    raises, and silently coercing would misalign the publication-price lookup.
    """
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        return idx.tz_localize(None)
    return idx


@disk_cache(_RAW_TTL)
def fetch_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """Daily OHLCV, split/dividend adjusted. Empty frame on any failure."""
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.copy()
    hist.index = _naive(hist.index)
    return hist.sort_index()


@disk_cache(_RAW_TTL)
def fetch_grades(ticker: str) -> pd.DataFrame:
    """Dated analyst actions: Firm, ToGrade, FromGrade, Action, priceTargetAction,
    currentPriceTarget, priorPriceTarget. Empty frame when unavailable."""
    try:
        import yfinance as yf

        gr = yf.Ticker(ticker).upgrades_downgrades
    except Exception:
        return pd.DataFrame()
    if gr is None or getattr(gr, "empty", True):
        return pd.DataFrame()
    gr = gr.copy()
    gr.index = _naive(pd.DatetimeIndex(gr.index))
    return gr.sort_index()


@disk_cache(_RAW_TTL, cache_empty=True)
def fetch_sector(ticker: str) -> Optional[str]:
    """The company's sector.

    CAVEAT — this is the one non-point-in-time input in the panel. yfinance exposes only
    the *current* classification, so a company that changed sector is labelled with
    today's. Sector reassignment is rare and slow, so this is a mild and acceptable
    contamination for neutralization purposes; it would not be acceptable as a signal.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        return info.get("sector")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #
def risk_features(hist: pd.DataFrame, market: pd.Series, t: pd.Timestamp) -> dict:
    """Trailing beta, realised volatility and dollar volume — all strictly from
    price data at or before `t`, so fully point-in-time.

    These exist to be regressed *out* of the signals, not to be scored. `vol_252d` in
    particular is the direct test of whether an apparently strong factor is really just
    selecting volatile names in a rising market.
    """
    out = {"beta_252d": np.nan, "vol_252d": np.nan, "log_dvol": np.nan}
    past = hist.loc[:t]
    if past.empty or "Close" not in past:
        return out
    closes = past["Close"].dropna()
    if len(closes) < 60:
        return out

    rets = closes.pct_change().dropna().iloc[-_TRADING_DAYS_YEAR:]
    if len(rets) >= 60:
        out["vol_252d"] = float(rets.std() * np.sqrt(_TRADING_DAYS_YEAR))
        if market is not None and not market.empty:
            mrets = market.loc[:t].pct_change().dropna()
            joined = pd.concat([rets, mrets], axis=1, join="inner").dropna()
            if len(joined) >= 60:
                r, m = joined.iloc[:, 0], joined.iloc[:, 1]
                var_m = float(m.var())
                if var_m > 0:
                    out["beta_252d"] = float(r.cov(m) / var_m)

    if "Volume" in past:
        vol = past["Volume"].dropna()
        n = min(len(vol), 60)
        if n >= 20:
            dvol = float((closes.iloc[-n:] * vol.iloc[-n:]).mean())
            if dvol > 0:
                out["log_dvol"] = float(np.log(dvol))
    return out


def price_features(hist: pd.DataFrame, t: pd.Timestamp) -> dict:
    """Momentum/technical features from history strictly at or before `t`.

    Delegates to `scoring.factors.technicals` so the study scores the same computation
    the live model uses, then adds 12-1 momentum (the standard cross-sectional
    construction, which the production model does not currently compute).
    """
    past = hist.loc[:t]
    out = {k: np.nan for k in ("rsi_14", "dist_ma50", "dist_ma200", "mom_3m", "mom_12_1", "vol_ratio")}
    if past.empty or "Close" not in past:
        return out

    tech = technicals(past)
    out["rsi_14"] = tech.get("rsi") if tech.get("rsi") is not None else np.nan
    out["dist_ma50"] = tech.get("above_ma50") if tech.get("above_ma50") is not None else np.nan
    out["dist_ma200"] = tech.get("above_ma200") if tech.get("above_ma200") is not None else np.nan
    out["mom_3m"] = tech.get("ret_3m") if tech.get("ret_3m") is not None else np.nan
    out["vol_ratio"] = tech.get("vol_ratio") if tech.get("vol_ratio") is not None else np.nan

    # 12-1: skip the most recent month to avoid the short-horizon reversal effect.
    closes = past["Close"].dropna()
    if len(closes) > _TRADING_DAYS_YEAR:
        recent = float(closes.iloc[-_TRADING_DAYS_MONTH])
        year_ago = float(closes.iloc[-_TRADING_DAYS_YEAR])
        if year_ago:
            out["mom_12_1"] = recent / year_ago - 1
    return out


def _implied_returns(rows: pd.DataFrame, closes: pd.Series) -> pd.Series:
    """Each analyst's forecast return measured against the price on their publish date.

    This is the de-anchoring step. `currentPriceTarget / price_today - 1` conflates the
    analyst's view with everything the stock has done since they published it.
    """
    px = pd.Series(
        [closes.asof(ts) for ts in rows.index],
        index=rows.index,
        dtype="float64",
    )
    tgt = pd.to_numeric(rows["currentPriceTarget"], errors="coerce")
    valid = px.notna() & (px > 0) & tgt.notna() & (tgt > 0)
    return (tgt / px - 1).where(valid)


def _decay_weights(index: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Series:
    age_days = (t - index).days.to_numpy(dtype="float64")
    return pd.Series(0.5 ** (age_days / HALF_LIFE_DAYS), index=index)


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna()
    if not ok.any():
        return np.nan
    w = weights[ok]
    return float((values[ok] * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def analyst_features(grades: pd.DataFrame, hist: pd.DataFrame, t: pd.Timestamp) -> dict:
    """Analyst-action features from rows published at or before `t`."""
    out = {
        k: np.nan
        for k in (
            "tgt_upside_stale",
            "tgt_upside_fresh_90d",
            "tgt_revision_90d",
            "tgt_action_net_90d",
            "grade_net_90d",
            "tgt_dispersion_90d",
            "analyst_count_90d",
        )
    }
    if grades is None or grades.empty or hist.empty or "Close" not in hist:
        return out

    past = grades.loc[:t]
    if past.empty:
        return out
    closes = hist.loc[:t, "Close"].dropna()
    if closes.empty:
        return out
    price_now = float(closes.iloc[-1])

    # --- control: the naive consensus the production model uses today ---------- #
    stale = past.loc[past.index >= t - pd.Timedelta(days=STALE_WINDOW_DAYS)]
    if not stale.empty and price_now > 0:
        tgt = pd.to_numeric(stale["currentPriceTarget"], errors="coerce")
        tgt = tgt[tgt > 0]
        if not tgt.empty:
            out["tgt_upside_stale"] = float(tgt.mean()) / price_now - 1

    fresh = past.loc[past.index >= t - pd.Timedelta(days=FRESH_WINDOW_DAYS)]
    out["analyst_count_90d"] = float(len(fresh))
    if fresh.empty:
        return out

    w = _decay_weights(fresh.index, t)

    # --- de-anchored upside ---------------------------------------------------- #
    implied = _implied_returns(fresh, closes)
    out["tgt_upside_fresh_90d"] = _wmean(implied, w)
    # Dispersion needs at least two views to mean anything.
    if implied.notna().sum() >= 2:
        out["tgt_dispersion_90d"] = float(implied.std())

    # --- target revision ------------------------------------------------------- #
    # `priorPriceTarget` is 0.0 on "Announces" rows (initiation, no prior view) — those
    # carry no revision information and must not become an infinite percentage change.
    cur = pd.to_numeric(fresh["currentPriceTarget"], errors="coerce")
    prior = pd.to_numeric(fresh["priorPriceTarget"], errors="coerce")
    ok = cur.notna() & prior.notna() & (prior > 0) & (cur > 0)
    rev = ((cur - prior) / prior).where(ok)
    out["tgt_revision_90d"] = _wmean(rev, w)

    # --- action counts --------------------------------------------------------- #
    if "priceTargetAction" in fresh:
        act = fresh["priceTargetAction"].astype("string").str.strip().str.lower()
        raises, lowers = (act == "raises"), (act == "lowers")
        n = int(raises.sum() + lowers.sum())
        if n:
            out["tgt_action_net_90d"] = float(raises.sum() - lowers.sum()) / n

    if "Action" in fresh:
        a = fresh["Action"].astype("string").str.strip().str.lower()
        up, down = (a == "up"), (a == "down")
        n = int(up.sum() + down.sum())
        if n:
            out["grade_net_90d"] = float(up.sum() - down.sum()) / n

    return out


# --------------------------------------------------------------------------- #
# Forward returns (the label — deliberately looks ahead)
# --------------------------------------------------------------------------- #
def forward_returns(hist: pd.DataFrame, t: pd.Timestamp, horizons: Iterable[int]) -> dict:
    out = {f"fwd_ret_{h}m": np.nan for h in horizons}
    if hist.empty or "Close" not in hist:
        return out
    closes = hist["Close"].dropna()
    if closes.empty:
        return out
    p0 = closes.asof(t)
    if pd.isna(p0) or p0 <= 0:
        return out
    last = closes.index[-1]
    for h in horizons:
        t1 = t + pd.DateOffset(months=h)
        # Don't fabricate a return from the final available bar when the horizon
        # extends past the data — that would bias recent periods toward zero.
        if t1 > last:
            continue
        p1 = closes.asof(t1)
        if pd.notna(p1) and p1 > 0:
            out[f"fwd_ret_{h}m"] = float(p1) / float(p0) - 1
    return out


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def month_ends(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Month-end rebalance dates, tolerant of the pandas 'M' -> 'ME' rename."""
    for freq in ("ME", "M"):
        try:
            return pd.date_range(start=start, end=end, freq=freq)
        except ValueError:
            continue
    raise ValueError("could not build a month-end date range")


def build_panel(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: Optional[str | pd.Timestamp] = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    progress=None,
    with_risk: bool = True,
) -> pd.DataFrame:
    """Long panel indexed by (date, ticker): every feature plus forward returns.

    With `with_risk`, also attaches point-in-time risk exposures (`RISK_FACTORS`) and
    `sector`, so `neutralize.py` can strip risk-factor loadings out of the signals
    before IC is measured.

    `progress` is an optional callable(done, total, ticker) for CLI feedback.
    """
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start = pd.Timestamp(start)
    dates = month_ends(start, end)
    horizons = tuple(horizons)

    tickers = [t.upper() for t in tickers]
    rows: list[dict] = []
    total = len(tickers)

    market = pd.Series(dtype="float64")
    if with_risk:
        mkt_hist = fetch_history(MARKET_PROXY)
        if not mkt_hist.empty and "Close" in mkt_hist:
            market = mkt_hist["Close"].dropna()

    for i, ticker in enumerate(tickers, start=1):
        if progress:
            progress(i, total, ticker)
        hist = fetch_history(ticker)
        if hist.empty:
            continue
        grades = fetch_grades(ticker)
        sector = fetch_sector(ticker) if with_risk else None

        first_bar = hist.index[0]
        for t in dates:
            # Need enough history for the longest lookback (12-1 momentum).
            if t < first_bar + pd.Timedelta(days=400):
                continue
            row = {"date": t, "ticker": ticker}
            row.update(price_features(hist, t))
            row.update(analyst_features(grades, hist, t))
            if with_risk:
                row.update(risk_features(hist, market, t))
                row["sector"] = sector
            row.update(forward_returns(hist, t, horizons))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return panel
