"""Post-processing helpers for valuation payload results."""

from __future__ import annotations

from statistics import median
from typing import Any

from .valuation_numeric import positive_float


def valuations_with_upside(valuations: list[dict[str, Any]], current_price: float | None) -> list[dict[str, Any]]:
    price = positive_float(current_price)
    enriched: list[dict[str, Any]] = []
    for item in valuations:
        row = dict(item)
        theoretical = positive_float(row.get("theoretical_price"))
        row["upside_pct"] = ((theoretical / price) - 1.0) * 100.0 if theoretical is not None and price else None
        enriched.append(row)
    return enriched


def valuation_summary(valuations: list[dict[str, Any]], current_price: float | None) -> dict[str, Any]:
    prices = [
        value
        for value in (positive_float(item.get("theoretical_price")) for item in valuations)
        if value is not None
    ]
    calculated = len(prices)
    price = positive_float(current_price)
    if not prices:
        return {
            "calculated_count": 0,
            "method_count": len(valuations),
            "median_price": None,
            "median_upside_pct": None,
            "min_price": None,
            "max_price": None,
        }
    median_price = float(median(prices))
    return {
        "calculated_count": calculated,
        "method_count": len(valuations),
        "median_price": median_price,
        "median_upside_pct": ((median_price / price) - 1.0) * 100.0 if price else None,
        "min_price": min(prices),
        "max_price": max(prices),
    }
