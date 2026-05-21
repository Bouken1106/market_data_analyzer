"""Shared helpers for portfolio-related service payloads."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..utils import finite_float_or_none, normalize_symbol


def positive_price_or_none(value: Any) -> float | None:
    return finite_float_or_none(value, minimum=0.0, strict_minimum=True)


def price_map_from_rows(rows: Any, *, include_missing: bool = False) -> dict[str, float | None]:
    price_map: dict[str, float | None] = {}
    if not isinstance(rows, list):
        return price_map

    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        price = positive_price_or_none(row.get("price"))
        if price is None and not include_missing:
            continue
        price_map[symbol] = price
    return price_map


def apply_market_value_weights(
    rows: Iterable[dict[str, Any]],
    total_market_value: float,
    *,
    market_value_key: str = "market_value",
    weight_key: str = "weight",
    scale: float = 100.0,
) -> None:
    if total_market_value <= 0:
        return
    for item in rows:
        market_value = item.get(market_value_key)
        if isinstance(market_value, (int, float)):
            item[weight_key] = (float(market_value) / total_market_value) * scale
