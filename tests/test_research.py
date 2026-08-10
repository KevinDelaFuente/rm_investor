"""Tests for the research harness. No network: every fixture is synthetic.

The look-ahead guard is the reason this file exists. An IC study that can see the future
produces beautiful, confident, worthless numbers — and it fails silently. Everything else
here is secondary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rm_investor.research import ic as ic_mod
from rm_investor.research import neutralize as neut_mod
from rm_investor.research import panel as panel_mod


# --------------------------------------------------------------------------- #
# Synthetic builders
# --------------------------------------------------------------------------- #
def make_history(n_days: int = 900, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    """Daily OHLCV with a random walk — enough history for 12-1 momentum."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n_days)))
    return pd.DataFrame(
        {"Close": closes, "Volume": rng.integers(1_000_000, 5_000_000, n_days)},
        index=idx,
    )


def make_grades(dates: list[str], current: list[float], prior: list[float],
                action: str = "main", pt_action: str = "Raises") -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="GradeDate")
    return pd.DataFrame(
        {
            "Firm": [f"Firm{i}" for i in range(len(dates))],
            "ToGrade": ["Buy"] * len(dates),
            "FromGrade": ["Hold"] * len(dates),
            "Action": [action] * len(dates),
            "priceTargetAction": [pt_action] * len(dates),
            "currentPriceTarget": current,
            "priorPriceTarget": prior,
        },
        index=idx,
    ).sort_index()


# --------------------------------------------------------------------------- #
# THE look-ahead guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t_str", ["2021-06-30", "2022-03-31", "2022-12-30"])
def test_price_features_cannot_see_the_future(t_str):
    """Features at t must be identical whether or not post-t data exists.

    If this fails, every IC number the harness produces is contaminated.
    """
    hist = make_history()
    t = pd.Timestamp(t_str)

    from_full = panel_mod.price_features(hist, t)
    from_truncated = panel_mod.price_features(hist.loc[:t], t)

    assert set(from_full) == set(from_truncated)
    for key in from_full:
        a, b = from_full[key], from_truncated[key]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, rel=1e-12), f"{key} leaked future data"


@pytest.mark.parametrize("t_str", ["2021-06-30", "2022-06-30"])
def test_analyst_features_cannot_see_the_future(t_str):
    hist = make_history()
    t = pd.Timestamp(t_str)
    grades = make_grades(
        ["2021-05-15", "2021-06-20", "2021-08-01", "2022-06-01", "2023-01-05"],
        current=[120.0, 130.0, 999.0, 140.0, 999.0],
        prior=[110.0, 120.0, 130.0, 135.0, 140.0],
    )

    from_full = panel_mod.analyst_features(grades, hist, t)
    from_truncated = panel_mod.analyst_features(grades.loc[:t], hist.loc[:t], t)

    for key in from_full:
        a, b = from_full[key], from_truncated[key]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b, rel=1e-12), f"{key} leaked future data"


def test_mom_12_1_matches_hand_calculation():
    """Pin the 12-1 construction: skip the last month, look back a year."""
    hist = make_history()
    t = hist.index[-1]
    closes = hist["Close"]
    expected = float(closes.iloc[-21]) / float(closes.iloc[-252]) - 1
    assert panel_mod.price_features(hist, t)["mom_12_1"] == pytest.approx(expected)


def test_forward_returns_are_not_features():
    """The label must never appear in the feature list."""
    assert not any(f.startswith("fwd_ret") for f in panel_mod.FEATURES)


def test_forward_return_is_none_past_end_of_data():
    """Better a missing label than a fabricated one from the last available bar."""
    hist = make_history()
    t = hist.index[-1] - pd.Timedelta(days=10)
    out = panel_mod.forward_returns(hist, t, (1, 12))
    assert pd.isna(out["fwd_ret_12m"])


# --------------------------------------------------------------------------- #
# Analyst feature correctness
# --------------------------------------------------------------------------- #
def test_upside_is_measured_against_the_publication_price():
    """De-anchoring: a target set months ago is judged against the price back then.

    This is the whole point of the fresh-vs-stale split. A stock that has fallen since
    publication must NOT show inflated upside.
    """
    hist = make_history()
    t = pd.Timestamp("2021-06-30")
    pub = pd.Timestamp("2021-06-15")
    closes = hist.loc[:t, "Close"].dropna()
    px_at_pub = float(closes.asof(pub))

    grades = make_grades([str(pub.date())], current=[px_at_pub * 1.20], prior=[px_at_pub])
    out = panel_mod.analyst_features(grades, hist, t)

    # Single fresh target, so the decay weight cancels: exactly +20%.
    assert out["tgt_upside_fresh_90d"] == pytest.approx(0.20, abs=1e-9)


def test_zero_prior_target_does_not_produce_infinity():
    """`priorPriceTarget` is 0.0 on initiation rows — a naive pct-change would blow up."""
    hist = make_history()
    t = pd.Timestamp("2021-06-30")
    grades = make_grades(["2021-06-10"], current=[150.0], prior=[0.0],
                         pt_action="Announces")
    out = panel_mod.analyst_features(grades, hist, t)
    assert pd.isna(out["tgt_revision_90d"])
    assert np.isfinite(out["tgt_upside_fresh_90d"])


def test_empty_grades_yield_nan_not_crash():
    hist = make_history()
    out = panel_mod.analyst_features(pd.DataFrame(), hist, pd.Timestamp("2021-06-30"))
    assert all(pd.isna(v) for v in out.values())


def test_stale_and_fresh_diverge_after_a_drawdown():
    """The control must behave the way the production model does — badly.

    Targets set at a much higher price, then a crash: the stale consensus shows large
    upside purely from lag, while the de-anchored measure keeps the analyst's real view.
    """
    idx = pd.bdate_range("2020-01-01", periods=600)
    closes = np.concatenate([np.full(500, 100.0), np.full(100, 60.0)])
    hist = pd.DataFrame({"Close": closes, "Volume": 1_000_000}, index=idx)

    t = idx[540]                       # 40 business days after the crash
    pub = idx[520]                     # published post-crash, at 60
    grades = make_grades([str(pub.date())], current=[66.0], prior=[60.0])

    out = panel_mod.analyst_features(grades, hist, t)
    assert out["tgt_upside_fresh_90d"] == pytest.approx(0.10, abs=1e-9)
    assert out["tgt_upside_stale"] == pytest.approx(0.10, abs=1e-9)

    # Now the same target published BEFORE the crash, when the stock was at 100.
    pub_pre = idx[480]
    grades_pre = make_grades([str(pub_pre.date())], current=[110.0], prior=[100.0])
    out_pre = panel_mod.analyst_features(grades_pre, hist, idx[500 + 20])

    # Stale reads ~+83% upside (110 vs 60) purely from lag; de-anchored still reads +10%.
    assert out_pre["tgt_upside_stale"] > 0.7
    assert out_pre["tgt_upside_fresh_90d"] == pytest.approx(0.10, abs=1e-9)


# --------------------------------------------------------------------------- #
# IC harness self-tests
# --------------------------------------------------------------------------- #
def _panel_from(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(["date", "ticker"]).sort_index()


def test_perfect_feature_scores_ic_one():
    """A feature that ranks forward returns exactly must produce IC = 1."""
    rows = []
    for d in pd.date_range("2021-01-31", periods=6, freq="ME"):
        for i in range(20):
            rows.append({"date": d, "ticker": f"T{i}", "feat": float(i),
                         "fwd_ret_1m": float(i) / 100})
    ic = ic_mod.rank_ic(_panel_from(pd.DataFrame(rows)), "feat", 1)
    assert len(ic) == 6
    assert ic.mean() == pytest.approx(1.0)


def test_inverted_feature_scores_ic_minus_one():
    """A sign-inverted factor must read as -1, not as noise — that distinction is
    the difference between 'drop this' and 'flip this'."""
    rows = []
    for d in pd.date_range("2021-01-31", periods=6, freq="ME"):
        for i in range(20):
            rows.append({"date": d, "ticker": f"T{i}", "feat": float(i),
                         "fwd_ret_1m": -float(i) / 100})
    ic = ic_mod.rank_ic(_panel_from(pd.DataFrame(rows)), "feat", 1)
    assert ic.mean() == pytest.approx(-1.0)


def test_thin_cross_sections_are_dropped():
    rows = [{"date": pd.Timestamp("2021-01-31"), "ticker": f"T{i}",
             "feat": float(i), "fwd_ret_1m": float(i)} for i in range(4)]
    assert ic_mod.rank_ic(_panel_from(pd.DataFrame(rows)), "feat", 1).empty


def test_t_stat_is_deflated_for_overlapping_returns():
    """A 12-month horizon sampled monthly overlaps 12-fold; t_adj must reflect that."""
    ic = pd.Series([0.05] * 36, index=pd.date_range("2021-01-31", periods=36, freq="ME"))
    ic = ic + np.linspace(-0.01, 0.01, 36)     # give it non-zero variance
    s1 = ic_mod.summarize(ic, horizon=1)
    s12 = ic_mod.summarize(ic, horizon=12)
    assert s1["t_naive"] == pytest.approx(s12["t_naive"])
    assert s12["t_adj"] == pytest.approx(s1["t_adj"] / np.sqrt(12))


def test_quantile_spread_sign_follows_the_feature():
    rows = []
    for d in pd.date_range("2021-01-31", periods=6, freq="ME"):
        for i in range(20):
            rows.append({"date": d, "ticker": f"T{i}", "feat": float(i),
                         "fwd_ret_1m": float(i) / 100})
    spread = ic_mod.quantile_spread(_panel_from(pd.DataFrame(rows)), "feat", 1, q=5)
    assert spread > 0


def test_summarize_handles_degenerate_series():
    out = ic_mod.summarize(pd.Series(dtype="float64"), horizon=1)
    assert out["n_periods"] == 0
    assert pd.isna(out["mean_ic"])


# --------------------------------------------------------------------------- #
# Risk features + neutralization
# --------------------------------------------------------------------------- #
def test_beta_recovers_a_planted_exposure():
    """A stock built as exactly 2x the market must measure beta ~= 2."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=400)
    mrets = rng.normal(0.0003, 0.010, len(idx))
    srets = 2.0 * mrets                                  # pure 2x, no idiosyncratic noise

    market = pd.Series(100 * np.exp(np.cumsum(mrets)), index=idx)
    hist = pd.DataFrame(
        {"Close": 50 * np.exp(np.cumsum(srets)), "Volume": 1_000_000}, index=idx
    )
    out = panel_mod.risk_features(hist, market, idx[-1])
    assert out["beta_252d"] == pytest.approx(2.0, abs=0.05)
    assert out["vol_252d"] > 0
    assert np.isfinite(out["log_dvol"])


def test_risk_features_cannot_see_the_future():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=600)
    mrets = rng.normal(0.0003, 0.010, len(idx))
    market = pd.Series(100 * np.exp(np.cumsum(mrets)), index=idx)
    hist = make_history(n_days=600, seed=11)
    hist.index = idx
    t = idx[400]

    full = panel_mod.risk_features(hist, market, t)
    trunc = panel_mod.risk_features(hist.loc[:t], market.loc[:t], t)
    for k in full:
        if pd.isna(full[k]) and pd.isna(trunc[k]):
            continue
        assert full[k] == pytest.approx(trunc[k], rel=1e-12), f"{k} leaked future data"


def _risk_panel(n_dates=8, n_names=40, alpha_strength=0.0, seed=5):
    """Panel where forward return is driven by beta, plus optional true alpha.

    `feat_beta`  is a pure beta proxy — neutralization must destroy its IC.
    `feat_alpha` carries genuine independent signal — neutralization must preserve it.
    """
    rng = np.random.default_rng(seed)
    sectors = ["Tech", "Energy", "Health", "Financials"]
    rows = []
    for d in pd.date_range("2021-01-31", periods=n_dates, freq="ME"):
        for i in range(n_names):
            beta = rng.uniform(0.5, 2.5)
            alpha = rng.normal(0, 1)
            rows.append({
                "date": d,
                "ticker": f"T{i}",
                "sector": sectors[i % len(sectors)],
                "beta_252d": beta,
                "vol_252d": 0.2 * beta,
                "log_dvol": rng.normal(18, 1),
                "feat_beta": beta + rng.normal(0, 0.01),        # ~pure risk proxy
                "feat_alpha": alpha,
                # Market rose over the sample, so beta alone ranks returns.
                "fwd_ret_1m": 0.05 * beta + alpha_strength * alpha + rng.normal(0, 0.005),
            })
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def test_neutralization_destroys_a_pure_beta_proxy():
    """The headline test: a factor that is only beta must not survive.

    This is the check that would have caught `tgt_dispersion_90d` looking like the best
    analyst factor in the study.
    """
    panel = _risk_panel(alpha_strength=0.0)
    out = neut_mod.neutralize(panel, ["feat_beta"])

    raw = ic_mod.rank_ic(out, "feat_beta", 1).mean()
    residual = ic_mod.rank_ic(out, "feat_beta" + neut_mod.SUFFIX, 1).mean()

    assert raw > 0.8, "setup is wrong: the raw factor should look excellent"
    assert abs(residual) < 0.3, f"beta proxy survived neutralization (resid IC {residual:.3f})"


def test_neutralization_preserves_genuine_alpha():
    """The converse: a real signal orthogonal to risk must come through intact."""
    panel = _risk_panel(alpha_strength=0.05)
    out = neut_mod.neutralize(panel, ["feat_alpha"])

    raw = ic_mod.rank_ic(out, "feat_alpha", 1).mean()
    residual = ic_mod.rank_ic(out, "feat_alpha" + neut_mod.SUFFIX, 1).mean()
    assert residual > 0.5 * raw, "neutralization ate a genuine signal"


def test_neutralize_is_computed_within_each_date():
    """Residuals must be cross-sectional, never pooled across time."""
    panel = _risk_panel(n_dates=4, n_names=30)
    out = neut_mod.neutralize(panel, ["feat_alpha"])
    col = "feat_alpha" + neut_mod.SUFFIX
    for _, cross in out.groupby(level="date"):
        vals = cross[col].dropna()
        if len(vals) >= neut_mod.MIN_NAMES_FOR_FIT:
            # OLS with an intercept forces each date's residuals to mean zero.
            assert abs(float(vals.mean())) < 1e-8


def test_thin_cross_section_yields_nan_not_a_fabricated_residual():
    panel = _risk_panel(n_dates=2, n_names=5)
    out = neut_mod.neutralize(panel, ["feat_alpha"])
    assert out["feat_alpha" + neut_mod.SUFFIX].notna().sum() == 0


def test_compare_table_reports_decay():
    panel = _risk_panel(alpha_strength=0.0)
    table = neut_mod.compare(panel, ["feat_beta"], horizons=(1,))
    assert not table.empty
    row = table.iloc[0]
    assert row["raw_ic"] > 0.8
    # Nearly all of the raw IC should be attributed to risk exposure.
    assert row["ic_decay"] > 0.6
