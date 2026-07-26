"""Configuration loader: merges config.yaml with .env secrets.

Provides a single `get_config()` accessor (cached) returning a `Config` object with
convenient dotted/keyed access, plus `get_api_key()` for provider credentials.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = two levels up from this file (rm_investor/config.py -> repo root)
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

load_dotenv(ROOT / ".env")  # no-op if the file is absent


class Config:
    """Thin wrapper over the parsed YAML dict with dict + attribute access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def path(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested value, e.g. cfg.path('scoring.sell.bands.sell')."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def resolve(self, relative: str) -> Path:
        """Resolve a config-relative path (e.g. a data CSV) against the repo root."""
        p = Path(relative)
        return p if p.is_absolute() else (ROOT / p)


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh) or {})


def get_api_key(name: str) -> str | None:
    """Fetch a secret from the environment (loaded from .env)."""
    val = os.environ.get(name)
    return val.strip() if val else None


def sec_user_agent() -> str:
    """SEC EDGAR requires a descriptive UA; fall back to a generic one."""
    return get_api_key("SEC_USER_AGENT") or "RM Investor research contact@example.com"
