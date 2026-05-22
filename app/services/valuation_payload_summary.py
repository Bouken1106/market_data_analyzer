"""Post-processing helpers for valuation payload results."""

from __future__ import annotations

from statistics import median
from typing import Any

from ..utils import percent_change
from .valuation_numeric import positive_float


def valuations_with_upside(valuations: list[dict[str, Any]], current_price: float | None) -> list[dict[str, Any]]:
    price = positive_float(current_price)
    enriched: list[dict[str, Any]] = []
    for item in valuations:
        row = dict(item)
        theoretical = positive_float(row.get("theoretical_price"))
        row["upside_pct"] = percent_change(theoretical, price)
        enriched.append(row)
    return enriched


def valuation_summary(valuations: list[dict[str, Any]], current_price: float | None) -> dict[str, Any]:
    calculated_rows = [
        item
        for item in valuations
        if positive_float(item.get("theoretical_price")) is not None
    ]
    standard_rows = [item for item in calculated_rows if item.get("is_standard_candidate") is True]
    fallback_rows = [
        item
        for item in calculated_rows
        if item.get("valuation_role") != "downside_reference"
    ]
    summary_rows = standard_rows or fallback_rows or calculated_rows
    prices = _filter_price_outliers(
        [
            value
            for value in (positive_float(item.get("theoretical_price")) for item in summary_rows)
            if value is not None
        ]
    )
    calculated = len(calculated_rows)
    price = positive_float(current_price)
    if not prices:
        return {
            "calculated_count": 0,
            "method_count": len(valuations),
            "standard_candidate_count": 0,
            "median_price": None,
            "median_upside_pct": None,
            "min_price": None,
            "max_price": None,
        }
    median_price = float(median(prices))
    return {
        "calculated_count": calculated,
        "method_count": len(valuations),
        "standard_candidate_count": len(standard_rows),
        "median_price": median_price,
        "median_upside_pct": percent_change(median_price, price),
        "min_price": min(prices),
        "max_price": max(prices),
        "used_standard_candidates": bool(standard_rows),
    }


def _filter_price_outliers(prices: list[float]) -> list[float]:
    if len(prices) < 3:
        return prices
    center = float(median(prices))
    return [value for value in prices if center * 0.10 <= value <= center * 10.0]
