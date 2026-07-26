"""Valuation blend: combine DCF + multiples + DDM into a FairValue range.

`fair_value()` runs each applicable method, weights the available ones (renormalizing
config weights over whatever ran), and reports a low/base/high range plus margin of
safety. Methods that can't run are recorded in `notes` — never faked. Every input is
surfaced so a fair-value number is auditable, not a black box.
"""
from __future__ import annotations

from typing import Optional

from ..config import get_config
from ..models import FairValue, Fundamentals, Quote
from .dcf import dcf_value
from .ddm import ddm_value, graham_number
from .multiples import multiples_value

_cfg = get_config()


def _assumptions(ticker: str) -> dict:
    base = dict(_cfg.get("valuation", {}) or {})
    overrides = (base.get("overrides") or {}).get(ticker, {})
    merged = {**base, **overrides}
    return merged


def fair_value(
    fund: Optional[Fundamentals],
    quote: Optional[Quote] = None,
    ticker: Optional[str] = None,
) -> FairValue:
    tk = ticker or (fund.ticker if fund else "?")
    price = quote.price if quote else None
    fv = FairValue(ticker=tk, price=price)

    if fund is None:
        fv.notes.append("no fundamentals available — valuation skipped")
        return fv

    assumptions = _assumptions(tk)
    weights: dict[str, float] = dict(assumptions.get("method_weights", {}) or {})

    # Run each method.
    for method, fn in (("dcf", dcf_value), ("multiples", multiples_value), ("ddm", ddm_value)):
        val, inputs = fn(fund, assumptions)
        fv.inputs[method] = inputs
        if val is not None and val > 0:
            fv.methods[method] = round(val, 2)
        else:
            reason = inputs.get("skipped", "not applicable")
            fv.notes.append(f"{method}: {reason}")

    # Graham number as a labelled sanity check (not weighted into the base).
    gn = graham_number(fund)
    if gn is not None:
        fv.inputs["graham"] = {"value": round(gn, 2)}
        fv.methods["graham"] = round(gn, 2)

    weighted_methods = {k: v for k, v in fv.methods.items() if k in weights}

    # DDM only belongs in the weighted base for materially dividend-paying names.
    # For low-yield growth stocks it distorts the blend, so keep it shown but excluded.
    if "ddm" in weighted_methods:
        div_yield = None
        if fund.dividend_rate and price:
            div_yield = fund.dividend_rate / price
        if div_yield is None or div_yield < 0.02:
            del weighted_methods["ddm"]
            fv.notes.append("ddm: shown but excluded from blend (dividend yield < 2%)")

    if weighted_methods:
        total_w = sum(weights[k] for k in weighted_methods)
        fv.base = round(
            sum(v * weights[k] for k, v in weighted_methods.items()) / total_w, 2
        )
    elif fv.methods:
        # Only unweighted methods (e.g. graham) survived — use their mean.
        vals = list(fv.methods.values())
        fv.base = round(sum(vals) / len(vals), 2)

    # Range reflects the methods that form the base (serious valuations), not the
    # excluded DDM / Graham sanity checks — so it isn't misleadingly wide.
    range_source = weighted_methods or fv.methods
    if range_source:
        vals = list(range_source.values())
        fv.low = round(min(vals), 2)
        fv.high = round(max(vals), 2)
    else:
        fv.notes.append("no valuation method could run")

    return fv
