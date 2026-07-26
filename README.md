.venv\Scripts\activate# 📈 RM Investor

A personal, **free-data** financial assistant: understand your portfolio, get transparent
**sell signals**, **scan for new opportunities**, estimate **intrinsic value**, follow disclosed
**smart-money** moves, and screen high-risk **moonshots** — all in a Streamlit dashboard.

> ⚠️ **Not financial advice.** RM Investor is a decision-support tool. Prices are delayed
> (~15 min). Analyst, valuation, and smart-money signals are model-based or lagged public
> disclosures — never recommendations or insider intel. Do your own research.

---

## What it does

| Page | Description |
|------|-------------|
| **Portfolio** | Loads `holdings.csv`, shows live-ish P&L and allocation. |
| **Sell Signals** | Per-holding **SELL / TRIM / HOLD** from a transparent, weighted factor set. |
| **Opportunity Scanner** | Ranks a bounded universe by analyst upside (+ coverage depth & target range), undervaluation, momentum, news, and smart-money buying. |
| **Smart Money** | Disclosed congressional trades + hedge-fund 13F changes; overlap with your book. |
| **News** | Aggregated headlines + sentiment (when the source provides it). |
| **Moonshots** | Separate, risk-flagged small/micro-cap sleeve for asymmetric upside. |
| **Valuation** | DCF + multiples + DDM/Graham → fair-value **range** and **margin of safety**. |

Everything is **deterministic** (rules/metrics, no LLM) and **explainable** — every score shows
its factor breakdown.

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # optional — everything works with no keys
streamlit run app.py
```

Edit `holdings.csv` with your positions (`ticker, shares, cost_basis`).

## Data sources (free → paid)

Runs out of the box on **yfinance** (no key). Optional free keys unlock more:

- `FINNHUB_API_KEY` — analyst ratings/targets + company news
- `ALPHAVANTAGE_API_KEY` — fundamentals + news **sentiment**

Smart money uses public **SEC EDGAR** 13F filings and community **congressional** disclosure
datasets. Everything is behind a **pluggable adapter layer** (`rm_investor/datasources/`): switch
a provider in `config.yaml` to move to paid/real-time sources (Polygon, Alpaca, Finnhub paid,
EODHD, Mboum) without touching scoring or the UI.

## Configuration

All knobs live in `config.yaml`: active providers, cache TTLs, the scan/moonshot universes,
**valuation assumptions** (discount rate, growth, terminal growth, target multiples), and the
**scoring weights & bands** for each profile. Nothing is hard-coded.

## Architecture

```
data adapters (pluggable)  →  DataHub  →  analysis.gather()  →  scoring / valuation  →  Streamlit UI
```

- `datasources/` — adapters + registry (yfinance, finnhub, alphavantage, sec_edgar, congress)
- `valuation/` — dcf, multiples, ddm/graham → blended `FairValue`
- `scoring/` — factors + sell / opportunity / moonshot composites
- `scan/` — universe construction + scanner
- `pages/` — the seven Streamlit pages

## Tests

```bash
pytest tests/
```

Deterministic, no network — synthetic data drives scoring, valuation, and loader assertions.

## Caveats

- Free tiers are rate-limited and prices are delayed ~15 min; results are cached (`.cache/`).
- Smart-money data is lagged (~45 days) and partial (13F = long U.S. equities only).
- Valuation is model-based — sensitive to assumptions; the UI shows a range and every input.
- Moonshots are speculative by design; the shipped universe is a curated **sample**, not an
  exhaustive market sweep.
