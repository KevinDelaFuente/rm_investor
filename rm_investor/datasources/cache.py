"""Caching utilities to keep free-tier APIs happy.

Two layers:
  * `install_http_cache()` — a process-wide requests-cache so anything using the
    `requests` library (Finnhub, Alpha Vantage, SEC, congress datasets) is cached
    transparently on disk with a TTL.
  * `disk_cache(ttl)` — a decorator that memoizes arbitrary function results
    (e.g. yfinance calls, which don't go through plain `requests`) to a pickle
    on disk keyed by function name + args, expiring after `ttl` seconds.

Both are best-effort: if the cache backend can't initialize, calls still work,
just uncached.
"""
from __future__ import annotations

import functools
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Callable

from ..config import ROOT

CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

_http_installed = False


def install_http_cache(ttl_seconds: int = 900) -> None:
    """Install a global requests-cache. Safe to call repeatedly."""
    global _http_installed
    if _http_installed:
        return
    try:
        import requests_cache

        requests_cache.install_cache(
            cache_name=str(CACHE_DIR / "http_cache"),
            backend="sqlite",
            expire_after=ttl_seconds,
            allowable_methods=("GET", "POST"),
            stale_if_error=True,
        )
        _http_installed = True
    except Exception:
        # No requests-cache available — proceed without HTTP caching.
        pass


def _key(func_name: str, args: tuple, kwargs: dict) -> str:
    raw = repr((func_name, args, sorted(kwargs.items()))).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _is_empty(result: Any) -> bool:
    """Treat None / empty containers / empty DataFrames as failures worth retrying."""
    if result is None:
        return True
    try:
        import pandas as pd

        if isinstance(result, pd.DataFrame):
            return result.empty
    except Exception:
        pass
    if isinstance(result, (dict, list, tuple, set, str)):
        return len(result) == 0
    return False


def disk_cache(ttl_seconds: int, cache_empty: bool = False) -> Callable:
    """Memoize a function's return value to disk with a TTL.

    Only use for picklable return values. On any cache error we fall back to
    calling the wrapped function directly. Empty/failed results (None, empty dict/
    list/DataFrame) are NOT cached by default, so a transient fetch failure never
    gets pinned for the whole TTL.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                path = CACHE_DIR / f"{func.__name__}_{_key(func.__name__, args, kwargs)}.pkl"
                if path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds:
                    with open(path, "rb") as fh:
                        return pickle.load(fh)
            except Exception:
                path = None  # fall through to recompute

            result = func(*args, **kwargs)

            try:
                if path is not None and (cache_empty or not _is_empty(result)):
                    with open(path, "wb") as fh:
                        pickle.dump(result, fh)
            except Exception:
                pass
            return result

        return wrapper

    return decorator


def clear_cache() -> int:
    """Delete all on-disk cache files. Returns count removed."""
    removed = 0
    for p in CACHE_DIR.glob("*.pkl"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    return removed
