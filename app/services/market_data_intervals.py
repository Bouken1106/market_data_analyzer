"""Interval normalization helpers for market-data services."""

from __future__ import annotations

from typing import Any

DAILY_INTERVAL_ALIASES = frozenset({"1day", "1d", "day"})


def normalized_interval(value: Any) -> str:
    return str(value or "").strip().lower()


def is_daily_interval(value: Any) -> bool:
    return normalized_interval(value) in DAILY_INTERVAL_ALIASES
