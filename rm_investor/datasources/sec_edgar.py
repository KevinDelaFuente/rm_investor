"""SEC EDGAR 13F adapter — hedge-fund / institutional holdings (free, no key).

Opt-in (add "sec_edgar" to providers.smart_money). For each configured fund CIK we
fetch the two most recent 13F-HR filings, parse their information tables, and compute
per-issuer position changes (new / added / trimmed / exited).

13F data is ~45 days delayed and shows only long U.S. equity positions — a lagged,
partial signal, never live intel. EDGAR requires a descriptive User-Agent.

Note: 13F reports issuers by CUSIP, not ticker. We surface the issuer name; ticker-level
filtering is best-effort (only when an issuer name maps cleanly). Degrades to [] on error.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from ..config import get_config, sec_user_agent
from ..models import ActorType, Trade, TradeSide
from .cache import disk_cache

_cfg = get_config()
_TTL = _cfg.path("cache.filings_ttl_seconds", 604800)
_FUNDS = _cfg.path("smart_money.funds", []) or []
_HEADERS = {"User-Agent": sec_user_agent(), "Accept-Encoding": "gzip, deflate"}


def _get(url: str) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=30)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None


@disk_cache(_TTL)
def _recent_13f(cik: str) -> list[dict]:
    """Return [{accession, date}] for the fund's recent 13F-HR filings (newest first)."""
    cik10 = str(cik).zfill(10)
    r = _get(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if not r:
        return []
    try:
        recent = r.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        out = []
        for form, accn, dt in zip(forms, accns, dates):
            if form.startswith("13F-HR"):
                out.append({"accession": accn.replace("-", ""), "date": dt})
        return out
    except Exception:
        return []


@disk_cache(_TTL)
def _holdings(cik: str, accession: str) -> dict[str, dict]:
    """Parse a 13F information table -> {cusip: {issuer, value, shares}}."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}"
    idx = _get(f"{base}/index.json")
    if not idx:
        return {}
    try:
        items = idx.json().get("directory", {}).get("item", [])
    except Exception:
        return {}

    # The information table is an .xml file that is not the primary_doc.
    xml_name = None
    for it in items:
        n = (it.get("name") or "").lower()
        if n.endswith(".xml") and "primary_doc" not in n:
            xml_name = it["name"]
            if "info" in n or "table" in n:
                break
    if not xml_name:
        return {}

    r = _get(f"{base}/{xml_name}")
    if not r:
        return {}
    return _parse_infotable(r.text)


def _parse_infotable(xml_text: str) -> dict[str, dict]:
    holdings: dict[str, dict] = {}
    try:
        # Strip namespaces for simpler tag matching.
        cleaned = re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", xml_text)
        cleaned = re.sub(r"<(/?)(\w+):", r"<\1", cleaned)
        root = ET.fromstring(cleaned)
    except Exception:
        return {}

    for info in root.iter("infoTable"):
        issuer = info.findtext("nameOfIssuer") or ""
        cusip = (info.findtext("cusip") or "").strip()
        value = _num(info.findtext("value"))
        shares = None
        shrs = info.find("shrsOrPrnAmt")
        if shrs is not None:
            shares = _num(shrs.findtext("sshPrnamt"))
        if not cusip:
            continue
        agg = holdings.setdefault(cusip, {"issuer": issuer.strip(), "value": 0.0, "shares": 0.0})
        agg["value"] += value or 0.0
        agg["shares"] += shares or 0.0
    return holdings


def _num(s) -> Optional[float]:
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Issuer-name -> ticker matching (13F reports CUSIP + name, not ticker).
# Uses SEC's free company_tickers.json — no key. Imperfect for some names;
# unmatched issuers keep ticker=None and still show by name.
# --------------------------------------------------------------------------- #
_SUFFIX_TOKENS = {
    "THE", "CO", "COMPANY", "INC", "INCORPORATED", "CORP", "CORPORATION", "LTD", "LIMITED",
    "PLC", "LP", "LLC", "HOLDINGS", "HLDGS", "HOLDING", "GROUP", "GRP", "CLASS", "CL",
    "COMMON", "STOCK", "SHS", "SHARES", "NEW", "ADR", "ADS", "SPONSORED", "COM", "NV",
    "SA", "AG", "TR", "TRUST", "REIT", "ORD", "OF", "AND",
}


def _norm_name(name: str) -> str:
    s = (name or "").upper()
    s = re.sub(r"/[A-Z]{2,3}/", " ", s)        # strip /DE/ state-of-incorporation markers
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _SUFFIX_TOKENS]
    while toks and len(toks[-1]) == 1:         # drop trailing class letters (A/B/...)
        toks.pop()
    return " ".join(toks).strip()


@disk_cache(_TTL)
def _sec_ticker_map() -> dict:
    """Build {normalized company name: ticker} from SEC's company_tickers.json."""
    r = _get("https://www.sec.gov/files/company_tickers.json")
    if not r:
        return {}
    try:
        data = r.json()
    except Exception:
        return {}
    out: dict[str, str] = {}
    for item in (data.values() if isinstance(data, dict) else []):
        title, ticker = item.get("title"), item.get("ticker")
        if title and ticker:
            out.setdefault(_norm_name(title), ticker.upper())
    return out


def _match_ticker(issuer: Optional[str]) -> Optional[str]:
    if not issuer:
        return None
    mapping = _sec_ticker_map()
    if not mapping:
        return None
    key = _norm_name(issuer)
    if key in mapping:
        return mapping[key]
    parts = key.split()
    for n in (3, 2, 1):   # progressively shorter prefix fallback
        if len(parts) >= n:
            pref = " ".join(parts[:n])
            if pref in mapping:
                return mapping[pref]
    return None


class SecEdgarSource:
    """SmartMoneySource for 13F filings (duck-typed)."""

    name = "sec_edgar"

    def get_trades(self, ticker: Optional[str] = None, limit: int = 200) -> list[Trade]:
        trades: list[Trade] = []
        for fund in _FUNDS:
            trades += self._fund_changes(fund.get("name", "Fund"), str(fund.get("cik", "")))
        # Now that issuer names are matched to tickers, filter on ticker first.
        if ticker:
            tk = ticker.upper()
            trades = [t for t in trades if (t.ticker == tk) or (t.issuer and tk in t.issuer.upper())]
        trades.sort(key=lambda t: abs(t.value or 0), reverse=True)
        return trades[:limit]

    def _fund_changes(self, name: str, cik: str) -> list[Trade]:
        if not cik:
            return []
        filings = _recent_13f(cik)
        if not filings:
            return []
        latest = _holdings(cik, filings[0]["accession"])
        prior = _holdings(cik, filings[1]["accession"]) if len(filings) > 1 else {}
        if not latest:
            return []

        out: list[Trade] = []
        for cusip, cur in latest.items():
            prev_shares = prior.get(cusip, {}).get("shares", 0.0)
            delta = (cur.get("shares") or 0.0) - prev_shares
            if delta > 0:
                side = TradeSide.BUY
            elif delta < 0:
                side = TradeSide.SELL
            else:
                side = TradeSide.UNKNOWN
            out.append(
                Trade(
                    ticker=_match_ticker(cur.get("issuer")),
                    issuer=cur.get("issuer"),
                    cusip=cusip,
                    actor=name,
                    actor_type=ActorType.FUND,
                    side=side,
                    shares_delta=delta,
                    value=cur.get("value"),
                    disclosed_date=None,
                    source=f"13F {filings[0]['date']}",
                )
            )
        # Exited positions (in prior, absent now).
        for cusip, prev in prior.items():
            if cusip not in latest:
                out.append(
                    Trade(
                        ticker=_match_ticker(prev.get("issuer")),
                        issuer=prev.get("issuer"),
                        cusip=cusip,
                        actor=name,
                        actor_type=ActorType.FUND,
                        side=TradeSide.SELL,
                        shares_delta=-(prev.get("shares") or 0.0),
                        value=0.0,
                        source=f"13F {filings[0]['date']} (exited)",
                    )
                )
        return out
