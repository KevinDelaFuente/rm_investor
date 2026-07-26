"""Import a Merrill 'Holdings_*.csv' export (Symbol/Quantity/Price/Value/UnrealizedGL
layout) into holdings.csv.

Unlike ExportData_*.csv (handled by import_merrill.py), this export has a real
header row with a dedicated 'Symbol' column and NO cost-basis column. We derive
per-share cost from:  cost_total = Value($) - UnrealizedGain($);  per_share = cost_total / Quantity.

Usage:
    python scripts/import_holdings_export.py "<path>\\Holdings_07242026.csv" --dry-run
    python scripts/import_holdings_export.py "<path>" --out holdings.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_SKIP_NAME_HINTS = (
    "DIRECT DEPOSIT", "MONEY MARKET", "SWEEP", "CASH", "PENDING", "DEPOSIT PROGRM",
    "BANK DEP", "PRESERVATION",
)
_TICKER_FIXUPS = {"BRKB": "BRK-B", "BRKA": "BRK-A"}


def _money(s: str | None) -> float | None:
    """'1,745.00' / '(171.15)' / '--' -> float | None. Parens = negative."""
    if s is None:
        return None
    s = s.strip().strip('"').strip()
    if s in ("", "--", "N/A"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").strip()
    if s.startswith("-"):
        neg = True
        s = s[1:]
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def convert(input_path: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    with open(input_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    def pick(d: dict, *names: str) -> str | None:
        for n in names:
            for k in d:
                if k and k.strip().lower() == n:
                    return d[k]
        for n in names:
            for k in d:
                if k and n in k.strip().lower():
                    return d[k]
        return None

    holdings: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for r in rows:
        symbol = (pick(r, "symbol") or "").strip().strip('"').upper()
        name = (pick(r, "security description") or "").strip().strip('"').upper()
        if not symbol or not re.fullmatch(r"[A-Z]{1,6}", symbol):
            skipped.append((symbol or name or "?", "no usable ticker"))
            continue
        if any(h in name for h in _SKIP_NAME_HINTS):
            skipped.append((symbol, "cash/sweep/non-equity"))
            continue

        qty = _money(pick(r, "quantity"))
        value = _money(pick(r, "value ($)", "value"))
        gain = _money(pick(r, "unrealized gain/loss ($)", "unrealized gain/loss"))
        if not qty or qty <= 0:
            skipped.append((symbol, "no quantity"))
            continue
        if value is None or gain is None:
            skipped.append((symbol, "missing value/gain -> can't derive cost"))
            continue

        cost_total = value - gain
        if cost_total <= 0:
            skipped.append((symbol, f"non-positive derived cost ({cost_total:.2f})"))
            continue

        ticker = _TICKER_FIXUPS.get(symbol, symbol)
        holdings.append({
            "ticker": ticker,
            "shares": qty,
            "cost_basis": round(cost_total / qty, 4),
        })

    return holdings, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Import a Merrill Holdings_*.csv into holdings.csv")
    ap.add_argument("input")
    ap.add_argument("--out", default="holdings.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    holdings, skipped = convert(input_path)
    holdings.sort(key=lambda h: h["ticker"])

    print(f"Parsed {len(holdings)} positions; skipped {len(skipped)} rows.")
    if skipped:
        print("\nSkipped:")
        for sym, why in skipped:
            print(f"  - {sym[:24]:24} ({why})")

    if args.dry_run:
        print("\n[dry-run] Preview (first 12):")
        for h in holdings[:12]:
            print(f"  {h['ticker']:6} shares={h['shares']:<8} cost/sh=${h['cost_basis']}")
        print("  ...")
        return

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "shares", "cost_basis"])
        w.writeheader()
        w.writerows(holdings)
    print(f"\nWrote {len(holdings)} positions -> {out_path}")


if __name__ == "__main__":
    main()
