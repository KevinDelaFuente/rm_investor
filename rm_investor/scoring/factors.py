"""Factor calculators — the transparent building blocks of every score.

Each function returns a `FactorScore` with a 0-100 value and a human-readable detail
string, so the UI can always explain *why* a name scored the way it did. Factors never
raise: missing data yields a neutral 50 with a "no data" note, so one gap never sinks a
whole score.

A few factors instead return `None` when their data source provides nothing at all
(currently the news-sentiment pair). That is not the same as missing one input: inside a
weight-normalized composite a constant 50 silently shrinks every other factor's
influence, so a permanently-dead factor must be omitted rather than defaulted. Callers
filter `None` out of the factor list; `composite()` then redistributes the weight.

Convention: higher value = stronger evidence for that factor's thesis
  * sell factors      -> higher = stronger reason to SELL
  * opportunity factors -> higher = stronger reason to BUY
  * moonshot factors  -> higher = more attractive speculative setup
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..models import (
    FactorScore,
    FairValue,
    Fundamentals,
    NewsItem,
    Position,
    Quote,
    Rating,
    Trade,
    TradeSide,
)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _lin(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Linear interpolation from (x0,y0)->(x1,y1), clamped to [0,100]."""
    if x1 == x0:
        return _clamp((y0 + y1) / 2)
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return _clamp(y)


def _fs(name: str, value: float, weight: float, detail: str) -> FactorScore:
    return FactorScore(name=name, value=round(_clamp(value), 1), weight=weight, detail=detail)


# --------------------------------------------------------------------------- #
# Technical helpers
# --------------------------------------------------------------------------- #
def rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    if closes is None or len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    val = 100 - (100 / (1 + rs))
    last = val.dropna()
    return float(last.iloc[-1]) if len(last) else None


def technicals(history: pd.DataFrame) -> dict:
    """Extract RSI, MA distances, momentum and volume trend from OHLCV history."""
    out = {"rsi": None, "above_ma50": None, "above_ma200": None, "ret_3m": None, "vol_ratio": None}
    if history is None or history.empty or "Close" not in history:
        return out
    closes = history["Close"].dropna()
    if len(closes) < 20:
        return out
    price = float(closes.iloc[-1])
    out["rsi"] = rsi(closes)
    if len(closes) >= 50:
        ma50 = float(closes.rolling(50).mean().iloc[-1])
        out["above_ma50"] = (price / ma50 - 1) if ma50 else None
    if len(closes) >= 200:
        ma200 = float(closes.rolling(200).mean().iloc[-1])
        out["above_ma200"] = (price / ma200 - 1) if ma200 else None
    if len(closes) >= 63:
        out["ret_3m"] = price / float(closes.iloc[-63]) - 1
    if "Volume" in history:
        vol = history["Volume"].dropna()
        if len(vol) >= 50:
            recent = float(vol.iloc[-20:].mean())
            base = float(vol.iloc[-50:].mean())
            out["vol_ratio"] = (recent / base) if base else None
    return out


def _avg_sentiment(news: list[NewsItem]) -> Optional[float]:
    vals = [n.sentiment for n in (news or []) if n.sentiment is not None]
    return sum(vals) / len(vals) if vals else None


def _net_smart_money(trades: list[Trade]) -> tuple[int, int]:
    buys = sum(1 for t in trades if t.side == TradeSide.BUY)
    sells = sum(1 for t in trades if t.side == TradeSide.SELL)
    return buys, sells


# --------------------------------------------------------------------------- #
# SELL factors
# --------------------------------------------------------------------------- #
def f_price_vs_target(quote: Quote, rating: Optional[Rating], w: float) -> FactorScore:
    if not rating or not rating.target_mean or not quote or not quote.price:
        return _fs("Price vs target", 50, w, "no analyst target available")
    ratio = quote.price / rating.target_mean
    val = _lin(ratio, 0.8, 20, 1.2, 100)  # 20% below target -> 20; 20% above -> 100
    return _fs(
        "Price vs target",
        val,
        w,
        f"price ${quote.price:,.2f} vs mean target ${rating.target_mean:,.2f} "
        f"({(ratio - 1) * 100:+.0f}%)",
    )


def f_overvaluation(fv: Optional[FairValue], w: float) -> FactorScore:
    if not fv or fv.base is None or fv.margin_of_safety is None:
        return _fs("Overvaluation", 50, w, "no fair-value estimate")
    mos = fv.margin_of_safety
    # Range is deliberately wide (-60%..+30%). A narrow -25%..+25% band saturated at
    # 100 for most of the book, so the heaviest sell factor became a constant bias
    # instead of a signal. Model-based fair values carry real uncertainty, so only a
    # large discrepancy should max the score.
    val = _lin(mos, -0.60, 100, 0.30, 0)
    return _fs("Overvaluation", val, w, f"margin of safety {mos * 100:+.0f}% vs fair value ${fv.base:,.2f}")


def pe_pair(fund: Optional[Fundamentals]) -> tuple[Optional[float], Optional[float]]:
    """Return (trailing_pe, forward_pe), keeping only positive values."""
    if fund is None:
        return None, None
    t = fund.trailing_pe if (fund.trailing_pe and fund.trailing_pe > 0) else None
    f = fund.forward_pe if (fund.forward_pe and fund.forward_pe > 0) else None
    return t, f


def sector_baseline_pe(fund: Optional[Fundamentals], cfg=None) -> float:
    """Baseline P/E to judge a company against — its sector's, not the market's.

    Comparing every stock to one market P/E of 18 penalises whole sectors on
    principle (software is structurally above 18; energy structurally below), which
    biases the valuation factors rather than informing them.
    """
    from ..config import get_config

    cfg = cfg or get_config()
    default = float(cfg.path("valuation.multiples.pe", 18.0))
    sector = getattr(fund, "sector", None) if fund else None
    if not sector:
        return default
    table = cfg.path("valuation.sector_multiples", {}) or {}
    entry = table.get(sector) or {}
    try:
        return float(entry.get("pe", default))
    except (TypeError, ValueError):
        return default


def high_forward_pe_flag(fund: Optional[Fundamentals], baseline_pe: float) -> Optional[str]:
    """Flag a forward P/E that is expensive even on *forward* earnings.

    A high forward P/E means a lot of future growth is already priced in — worth
    noting explicitly, since the prefer-forward view could otherwise excuse it.
    """
    _, f = pe_pair(fund)
    if f is not None and f > max(40.0, 2.5 * baseline_pe):
        return f"very high forward P/E ({f:.0f})"
    return None


def f_pe_stretch(fund: Optional[Fundamentals], baseline_pe: float, w: float) -> FactorScore:
    """Sell factor: both trailing and forward P/E contribute; higher = more stretched."""
    t, f = pe_pair(fund)
    if t is None and f is None:
        return _fs("Valuation stretch (P/E)", 55, w, "no positive P/E")
    scores, parts = [], []
    if t is not None:
        scores.append(_lin(t / baseline_pe, 0.5, 15, 2.0, 100))
        parts.append(f"trailing {t:.1f}")
    if f is not None:
        scores.append(_lin(f / baseline_pe, 0.5, 15, 2.0, 100))
        parts.append(f"fwd {f:.1f}")
    val = sum(scores) / len(scores)
    detail = " / ".join(parts) + f" vs baseline {baseline_pe:.0f}"
    return _fs("Valuation stretch (P/E)", val, w, detail)


def f_momentum_overbought(tech: dict, w: float) -> FactorScore:
    r = tech.get("rsi")
    above200 = tech.get("above_ma200")
    if r is None:
        return _fs("Overextension / momentum", 50, w, "insufficient price history")
    val = _lin(r, 30, 15, 80, 100)  # RSI 30 -> 15, 80 -> 100
    if above200 is not None and above200 > 0.30:
        val = _clamp(val + 10)
    return _fs(
        "Overextension / momentum",
        val,
        w,
        f"RSI {r:.0f}" + (f", {above200 * 100:+.0f}% vs 200d MA" if above200 is not None else ""),
    )


def f_analyst_deterioration(rating: Optional[Rating], w: float) -> FactorScore:
    if not rating:
        return _fs("Analyst deterioration", 50, w, "no ratings data")
    val = None
    detail_bits = []
    if rating.recommendation_mean is not None:
        val = _lin(rating.recommendation_mean, 1.0, 0, 5.0, 100)  # 1=StrongBuy ->0, 5=Sell ->100
        detail_bits.append(f"rec mean {rating.recommendation_mean:.1f}")
    if rating.trend_delta is not None:
        adj = _lin(-rating.trend_delta, -3, 20, 3, 90)
        val = adj if val is None else (val + adj) / 2
        detail_bits.append(f"trend Δ {rating.trend_delta:+.0f}")
    if val is None:
        return _fs("Analyst deterioration", 50, w, "no consensus signal")
    return _fs("Analyst deterioration", val, w, ", ".join(detail_bits))


def f_concentration(position: Optional[Position], w: float, cfg=None) -> FactorScore:
    """Sell factor: position size versus the book. Higher weight = stronger trim case.

    This deliberately does NOT look at unrealized gain. The previous version scored a
    doubled position at 90 on the sell side, which is the disposition effect encoded as
    alpha: your cost basis is a fact about when you bought, not about the stock's forward
    return, so two managers holding identical shares would get different signals. Position
    weight is a genuine risk input and stays; P&L is shown on the holdings table instead.

    Anchors assume a diversified book (equal weight ~1% across ~100 names) and are
    tunable via `scoring.concentration` in config.yaml.
    """
    if position is None or position.weight is None:
        return _fs("Concentration", 50, w, "no position weight")

    from ..config import get_config

    cfg = cfg or get_config()
    lo = float(cfg.path("scoring.concentration.low_pct", 1.0))
    hi = float(cfg.path("scoring.concentration.high_pct", 15.0))
    weight = position.weight
    val = _lin(weight, lo, 30, hi, 90)
    return _fs("Concentration", val, w, f"weight {weight:.1f}% of book")



def f_news_negative(news: list[NewsItem], w: float) -> Optional[FactorScore]:
    """Sell factor. Returns None — not a neutral 50 — when nothing scored the sentiment.

    A constant 50 is not neutral inside a weighted composite; it is dilution. The
    configured provider (yfinance) never populates `NewsItem.sentiment`, so this factor
    spent every run contributing a fixed value that shrank every live factor's influence,
    and did so asymmetrically (0.10 weight on the buy side vs 0.075 on the sell side).
    `composite()` normalizes by total weight, so omitting the factor redistributes
    cleanly. It re-enables by itself the moment a sentiment-capable provider is used.
    """
    s = _avg_sentiment(news)
    if s is None:
        return None
    val = _lin(s, -1, 100, 1, 0)  # negative sentiment -> high sell
    return _fs("News sentiment", val, w, f"avg sentiment {s:+.2f}")


def f_smart_money_selling(trades: list[Trade], w: float) -> FactorScore:
    buys, sells = _net_smart_money(trades)
    if buys + sells == 0:
        return _fs("Smart money selling", 50, w, "no disclosed trades")
    frac_sell = sells / (buys + sells)
    val = _lin(frac_sell, 0, 20, 1, 90)
    return _fs("Smart money selling", val, w, f"{sells} sells / {buys} buys (disclosed)")


# --------------------------------------------------------------------------- #
# OPPORTUNITY factors
# --------------------------------------------------------------------------- #
def f_analyst_upside(quote: Quote, rating: Optional[Rating], w: float) -> FactorScore:
    if not rating or not rating.target_mean or not quote or not quote.price:
        return _fs("Analyst upside", 45, w, "no analyst target")
    upside = rating.target_mean / quote.price - 1
    val = _lin(upside, -0.20, 10, 0.40, 95)
    # Confidence from coverage depth: thin coverage pulls toward neutral (0.4..1.0).
    n = rating.analyst_count or 0
    conf = max(0.4, min(1.0, 0.4 + 0.6 * min(n, 15) / 15))
    val = 50 + (val - 50) * conf
    return _fs(
        "Analyst upside",
        val,
        w,
        f"{upside * 100:+.0f}% to mean target · {n or '?'} analysts · range {rating.target_range_str}",
    )


def f_undervaluation(fv: Optional[FairValue], w: float) -> FactorScore:
    if not fv or fv.base is None or fv.margin_of_safety is None:
        return _fs("Undervaluation", 45, w, "no fair-value estimate")
    mos = fv.margin_of_safety
    # Mirrors f_overvaluation's widened range so the buy side doesn't saturate either.
    val = _lin(mos, -0.60, 5, 0.40, 95)
    return _fs("Undervaluation", val, w, f"margin of safety {mos * 100:+.0f}% vs fair value ${fv.base:,.2f}")


def f_valuation_reasonable(fund: Optional[Fundamentals], baseline_pe: float, w: float) -> FactorScore:
    """Opportunity factor: both trailing and forward P/E contribute; cheaper = higher."""
    t, f = pe_pair(fund)
    if t is None and f is None:
        return _fs("Valuation reasonable", 45, w, "no positive P/E")
    scores, parts = [], []
    for pe, lbl in ((t, "trailing"), (f, "fwd")):
        if pe is not None:
            scores.append(_lin(pe / baseline_pe, 0.5, 90, 2.0, 10))  # cheap -> high
            parts.append(f"{lbl} {pe:.1f}")
    val = sum(scores) / len(scores)
    return _fs("Valuation reasonable", val, w, " / ".join(parts) + f" vs baseline {baseline_pe:.0f}")


def f_momentum_constructive(tech: dict, w: float) -> FactorScore:
    r = tech.get("rsi")
    above50 = tech.get("above_ma50")
    if r is None:
        return _fs("Momentum", 50, w, "insufficient history")
    # Reward uptrend, penalize overbought extremes.
    val = 50.0
    bits = [f"RSI {r:.0f}"]
    if above50 is not None:
        val = _lin(above50, -0.15, 25, 0.15, 80)
        bits.append(f"{above50 * 100:+.0f}% vs 50d MA")
    if r > 75:
        val = _clamp(val - 20)
        bits.append("overbought")
    return _fs("Momentum", val, w, ", ".join(bits))


def f_news_positive(news: list[NewsItem], w: float) -> Optional[FactorScore]:
    """Opportunity factor. Returns None when unscored — see `f_news_negative`."""
    s = _avg_sentiment(news)
    if s is None:
        return None
    val = _lin(s, -1, 0, 1, 100)
    return _fs("News sentiment", val, w, f"avg sentiment {s:+.2f}")


def f_smart_money_buying(trades: list[Trade], w: float) -> FactorScore:
    buys, sells = _net_smart_money(trades)
    if buys + sells == 0:
        return _fs("Smart money buying", 50, w, "no disclosed trades")
    frac_buy = buys / (buys + sells)
    val = _lin(frac_buy, 0, 10, 1, 90)
    return _fs("Smart money buying", val, w, f"{buys} buys / {sells} sells (disclosed)")


# --------------------------------------------------------------------------- #
# MOONSHOT factors
# --------------------------------------------------------------------------- #
def f_small_cap(fund: Optional[Fundamentals], w: float) -> FactorScore:
    if not fund or not fund.market_cap:
        return _fs("Small-cap size", 50, w, "unknown market cap")
    cap_b = fund.market_cap / 1e9
    val = _lin(cap_b, 0.1, 100, 5.0, 25)  # $100M -> 100, $5B -> 25
    return _fs("Small-cap size", val, w, f"market cap ${cap_b:.2f}B")


def f_revenue_growth(fund: Optional[Fundamentals], w: float) -> FactorScore:
    if not fund or fund.revenue_growth is None:
        return _fs("Revenue growth", 40, w, "unknown growth")
    g = fund.revenue_growth
    val = _lin(g, 0, 25, 1.0, 100)  # 0% -> 25, +100% -> 100
    return _fs("Revenue growth", val, w, f"revenue growth {g * 100:+.0f}%")


def f_under_followed(rating: Optional[Rating], w: float) -> FactorScore:
    n = rating.analyst_count if (rating and rating.analyst_count is not None) else None
    if n is None:
        return _fs("Under-followed", 70, w, "little/no coverage (undiscovered)")
    val = _lin(n, 0, 95, 15, 20)  # fewer analysts -> higher
    return _fs("Under-followed", val, w, f"{n} analysts covering")


def f_momentum_breakout(tech: dict, w: float) -> FactorScore:
    ret3 = tech.get("ret_3m")
    vol = tech.get("vol_ratio")
    if ret3 is None:
        return _fs("Momentum breakout", 45, w, "insufficient history")
    val = _lin(ret3, -0.2, 20, 0.6, 95)
    bits = [f"3m return {ret3 * 100:+.0f}%"]
    if vol is not None and vol > 1.2:
        val = _clamp(val + 10)
        bits.append(f"volume {vol:.1f}x")
    return _fs("Momentum breakout", val, w, ", ".join(bits))


def f_accumulation(trades: list[Trade], w: float) -> FactorScore:
    buys, sells = _net_smart_money(trades)
    if buys + sells == 0:
        return _fs("Accumulation", 50, w, "no disclosed trades")
    val = _lin(buys - sells, -3, 20, 5, 95)
    return _fs("Accumulation", val, w, f"{buys} buys / {sells} sells (disclosed)")


def moonshot_risk_flags(fund: Optional[Fundamentals], tech: dict) -> list[str]:
    flags: list[str] = []
    if fund:
        if fund.profit_margin is not None and fund.profit_margin < 0:
            flags.append("unprofitable")
        if fund.beta is not None and fund.beta > 1.8:
            flags.append(f"high volatility (β {fund.beta:.1f})")
        if fund.avg_volume is not None and fund.avg_volume < 200_000:
            flags.append("thin liquidity")
        if (
            fund.total_debt is not None
            and fund.total_cash is not None
            and fund.total_debt > 2 * max(fund.total_cash, 1)
        ):
            flags.append("high debt vs cash")
        if fund.market_cap is not None and fund.market_cap < 3e8:
            flags.append("micro-cap")
        if fund.forward_pe is not None and fund.forward_pe > 40:
            flags.append(f"very high forward P/E ({fund.forward_pe:.0f})")
    return flags
