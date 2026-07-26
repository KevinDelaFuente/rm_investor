"""Import a Merrill Lynch (or similar) brokerage CSV export into holdings.csv.

Broker exports vary, but Merrill's "ExportData_*.csv" has:
  * preamble/summary rows before a "Positions" header row,
  * a first column like "AAPL APPLE INC" (ticker + name),
  * Quantity, Price, ... and a **total** "Cost basis" column (not per-share).

Our holdings.csv needs: ticker, shares, cost_basis (PER SHARE). This script:
  * finds the Positions header row,
  * extracts the leading ticker token,
  * divides total cost basis by quantity -> per-share cost,
  * skips cash/sweep/non-equity rows and anything without a usable ticker,
  * writes holdings.csv (and prints a summary + skipped rows).

Usage:
    python scripts/import_merrill.py "C:\\path\\to\\ExportData_....csv"
    python scripts/import_merrill.py <input.csv> --out holdings.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Rows whose ticker/description clearly aren't tradable equities we can price.
_SKIP_TICKER_RE = re.compile(r"^\d+$")  # numeric "tickers" e.g. sweep account ids
_SKIP_NAME_HINTS = (
    "DIRECT DEPOSIT", "MONEY MARKET", "SWEEP", "CASH", "PENDING", "DEPOSIT PROGRM",
    "BANK DEP", "PRESERVATION",
)


def _money(s: str) -> float | None:
    """Parse '$1,745.00' / '-$293.35' / '12.3633' -> float. '--' -> None."""
    if s is None:
        return None
    s = s.strip().strip('"')
    if s in ("", "--", "N/A"):
        return None
    neg = s.startswith("-")
    s = s.replace("-", "").replace("$", "").replace(",", "").strip()
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _qty(s: str) -> float | None:
    return _money(s)  # same cleaning (handles "1,300")


def _find_header(rows: list[list[str]]) -> int:
    for i, row in enumerate(rows):
        if row and row[0].strip().strip('"').lower() == "positions":
            return i
    # Fallback: first row containing both "Quantity" and "Cost basis".
    for i, row in enumerate(rows):
        joined = ",".join(c.lower() for c in row)
        if "quantity" in joined and "cost basis" in joined:
            return i
    raise SystemExit("Could not find the 'Positions' header row in the export.")


def _ticker_from(cell: str) -> str | None:
    """'AAPL APPLE INC' -> 'AAPL'; 'BRKB BERKSHIRE...' -> 'BRKB' (normalized later)."""
    cell = cell.strip().strip('"')
    if not cell:
        return None
    token = cell.split()[0].upper()
    # Basic sanity: tickers are 1-5 alpha (allow a trailing letter class already merged).
    if not re.fullmatch(r"[A-Z]{1,6}", token):
        return None
    return token


# Broker sometimes concatenates share-class into the symbol (BRKB -> BRK-B).
_TICKER_FIXUPS = {
    "BRKB": "BRK-B",
    "BRKA": "BRK-A",
}


def convert(input_path: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    with open(input_path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))

    start = _find_header(rows)
    header = [c.strip().strip('"').lower() for c in rows[start]]

    def col(name: str) -> int:
        for i, h in enumerate(header):
            if h == name:
                return i
        for i, h in enumerate(header):
            if name in h:
                return i
        return -1

    i_qty, i_cost = col("quantity"), col("cost basis")
    if i_qty < 0 or i_cost < 0:
        raise SystemExit(f"Missing Quantity/Cost basis columns. Header was: {header}")

    holdings: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for row in rows[start + 1:]:
        if not row or not row[0].strip().strip('"'):
            continue
        first = row[0].strip().strip('"')
        if first.lower() == "total":
            break

        ticker = _ticker_from(first)
        name_upper = first.upper()
        if ticker is None or _SKIP_TICKER_RE.match(ticker):
            skipped.append((first, "no usable ticker"))
            continue
        if any(h in name_upper for h in _SKIP_NAME_HINTS):
            skipped.append((first, "cash/sweep/non-equity"))
            continue

        qty = _qty(row[i_qty]) if i_qty < len(row) else None
        total_cost = _money(row[i_cost]) if i_cost < len(row) else None
        if not qty or qty <= 0:
            skipped.append((first, "no quantity"))
            continue
        if total_cost is None or total_cost <= 0:
            skipped.append((first, "no cost basis"))
            continue

        ticker = _TICKER_FIXUPS.get(ticker, ticker)
        per_share = round(total_cost / qty, 4)
        holdings.append({"ticker": ticker, "shares": qty, "cost_basis": per_share})

    return holdings, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Import a Merrill brokerage CSV into holdings.csv")
    ap.add_argument("input", help="Path to the broker export CSV")
    ap.add_argument("--out", default="holdings.csv", help="Output holdings CSV (default: holdings.csv)")
    ap.add_argument("--dry-run", action="store_true", help="Print result, don't write")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    holdings, skipped = convert(input_path)
    holdings.sort(key=lambda h: h["ticker"])

    print(f"Parsed {len(holdings)} positions; skipped {len(skipped)} rows.")
    if skipped:
        print("\nSkipped:")
        for name, why in skipped:
            print(f"  - {name[:48]:48} ({why})")

    if args.dry_run:
        print("\n[dry-run] Preview (first 10):")
        for h in holdings[:10]:
            print(f"  {h['ticker']:6} shares={h['shares']:<8} cost/sh=${h['cost_basis']}")
        return

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "shares", "cost_basis"])
        w.writeheader()
        w.writerows(holdings)
    print(f"\nWrote {len(holdings)} positions -> {out_path}")


if __name__ == "__main__":
    main()
