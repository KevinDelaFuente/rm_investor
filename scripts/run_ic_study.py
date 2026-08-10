"""Run the factor IC study and write the results table.

    python scripts/run_ic_study.py --universe holdings,sp500 --start 2018-01-01

Builds a point-in-time panel over the chosen universe, computes rank IC at each
horizon for every candidate factor, prints the table and writes it to
`.cache/research/ic_<date>.csv`.

The two comparisons this exists to settle:
  * `mom_12_1` vs `rsi_14` / `dist_ma50` — is the production momentum factor
    sign-inverted relative to the standard construction?
  * `tgt_upside_fresh_90d` vs `tgt_upside_stale` — does measuring an analyst's target
    against the price they actually saw beat the stale rolling consensus?
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_investor.config import get_config           # noqa: E402
from rm_investor.datasources.cache import CACHE_DIR  # noqa: E402
from rm_investor.portfolio.loader import load_holdings  # noqa: E402
from rm_investor.research import FEATURES, build_panel, study  # noqa: E402
from rm_investor.research.neutralize import compare  # noqa: E402

OUT_DIR = CACHE_DIR / "research"


def resolve_universe(spec: str) -> list[str]:
    """Expand a comma-separated universe spec into a deduped ticker list."""
    cfg = get_config()
    tickers: list[str] = []
    for part in [p.strip().lower() for p in spec.split(",") if p.strip()]:
        if part == "holdings":
            tickers += [h.ticker for h in load_holdings()]
        elif part == "sp500":
            path = cfg.resolve(cfg.path("universe.index_csv", "rm_investor/data/sp500_sample.csv"))
            if path.exists():
                tickers += pd.read_csv(path)["ticker"].astype(str).tolist()
        elif part == "watchlist":
            tickers += list(cfg.path("universe.watchlist", []) or [])
        else:
            tickers.append(part.upper())

    seen, out = set(), []
    for t in (x.strip().upper() for x in tickers if x and str(x).strip()):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Factor IC study over a point-in-time panel.")
    ap.add_argument("--universe", default="holdings,sp500",
                    help="comma-separated: holdings, sp500, watchlist, or bare tickers")
    ap.add_argument("--start", default="2018-01-01", help="first rebalance date")
    ap.add_argument("--end", default=None, help="last rebalance date (default: today)")
    ap.add_argument("--horizons", default="1,3,6,12", help="forward-return horizons in months")
    ap.add_argument("--quantiles", type=int, default=5, help="buckets for the spread stat")
    ap.add_argument("--no-neutralize", action="store_true",
                    help="skip the risk-neutralized IC comparison")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    tickers = resolve_universe(args.universe)
    if not tickers:
        print("No tickers resolved — check --universe.", file=sys.stderr)
        return 1

    print(f"Universe: {len(tickers)} tickers | rebalance monthly from {args.start}")
    print("NOTE: both holdings.csv and sp500_sample.csv are TODAY's members, so these")
    print("      results carry survivorship bias. Directional, not publication-grade.\n")

    def progress(i: int, total: int, ticker: str) -> None:
        pct = 100.0 * i / total
        print(f"\r  fetching {i}/{total} ({pct:5.1f}%) {ticker:<8}", end="", flush=True)

    panel = build_panel(tickers, start=args.start, end=args.end,
                        horizons=horizons, progress=progress)
    print()

    if panel.empty:
        print("Panel is empty — no usable history.", file=sys.stderr)
        return 1

    n_dates = panel.index.get_level_values("date").nunique()
    n_names = panel.index.get_level_values("ticker").nunique()
    print(f"Panel: {len(panel):,} rows | {n_dates} dates | {n_names} tickers\n")

    table = study(panel, list(FEATURES), horizons=horizons, q=args.quantiles)
    if table.empty:
        print("No IC could be computed — cross-sections too thin.", file=sys.stderr)
        return 1

    shown = table[["feature", "horizon_m", "n_periods", "mean_ic", "ic_ir",
                   "t_adj", "hit_rate", "q_spread", "coverage"]].copy()
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(shown.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    print("\n  mean_ic  : average cross-sectional rank IC (0.02-0.05 = a normal useful factor)")
    print("  t_adj    : t-stat deflated by sqrt(horizon) for overlapping returns — use this")
    print("  q_spread : mean forward return, top quintile minus bottom")
    print("  A NEGATIVE mean_ic means the factor works but is pointed the wrong way.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"ic_{date.today():%Y%m%d}.csv"
    table.to_csv(out, index=False)
    panel.to_parquet(OUT_DIR / f"panel_{date.today():%Y%m%d}.parquet")
    print(f"\nWrote {out}")

    # --- risk-neutralized IC ------------------------------------------------- #
    if not args.no_neutralize and "sector" in panel.columns:
        print("\n" + "=" * 78)
        print("RISK-NEUTRALIZED IC — factor residualized on sector + beta + vol + size")
        print("A factor whose IC collapses here was loading on risk, not producing alpha.")
        print("=" * 78)
        comp = compare(panel, list(FEATURES), horizons=horizons)
        if comp.empty:
            print("Neutralized comparison unavailable (cross-sections too thin).")
        else:
            with pd.option_context("display.width", 200, "display.max_rows", None):
                print(comp.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
            print("\n  ic_decay : fraction of raw IC destroyed by neutralization.")
            print("             ~1.0 = pure risk proxy;  ~0.0 = genuinely orthogonal signal.")
            neut_out = OUT_DIR / f"ic_neutralized_{date.today():%Y%m%d}.csv"
            comp.to_csv(neut_out, index=False)
            print(f"\nWrote {neut_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
