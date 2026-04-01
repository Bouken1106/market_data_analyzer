"""Metric helpers for watchlist commentary generation."""

from __future__ import annotations

import math
from typing import Any

from ..utils import finite_float_or_none


def safe_float(value: Any) -> float | None:
    return finite_float_or_none(value)


def format_signed_percent(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}%"


def compute_watch_metrics(symbol: str, sparkline_item: dict[str, Any] | None) -> dict[str, Any]:
    latest_close = safe_float(sparkline_item.get("latest_close")) if isinstance(sparkline_item, dict) else None
    previous_close = safe_float(sparkline_item.get("previous_close")) if isinstance(sparkline_item, dict) else None

    trend_raw = sparkline_item.get("trend_30d") if isinstance(sparkline_item, dict) else []
    trend_closes: list[float] = []
    if isinstance(trend_raw, list):
        for raw_value in trend_raw:
            close_value = safe_float(raw_value)
            if close_value is None or close_value <= 0:
                continue
            trend_closes.append(close_value)

    day_change_pct: float | None = None
    if latest_close is not None and previous_close is not None and previous_close > 0:
        day_change_pct = ((latest_close - previous_close) / previous_close) * 100

    return_30d_pct: float | None = None
    if len(trend_closes) >= 2 and trend_closes[0] > 0:
        return_30d_pct = ((trend_closes[-1] - trend_closes[0]) / trend_closes[0]) * 100

    daily_returns: list[float] = []
    for idx in range(1, len(trend_closes)):
        prev_close = trend_closes[idx - 1]
        curr_close = trend_closes[idx]
        if prev_close <= 0:
            continue
        daily_returns.append((curr_close / prev_close) - 1.0)

    volatility_30d_pct: float | None = None
    if daily_returns:
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((item - mean_return) ** 2 for item in daily_returns) / len(daily_returns)
        volatility_30d_pct = math.sqrt(max(variance, 0.0)) * 100

    return {
        "symbol": symbol,
        "day_change_pct": day_change_pct,
        "return_30d_pct": return_30d_pct,
        "volatility_30d_pct": volatility_30d_pct,
        "day_change_text": format_signed_percent(day_change_pct),
        "return_30d_text": format_signed_percent(return_30d_pct),
        "volatility_30d_text": format_percent(volatility_30d_pct),
    }


def metrics_payload(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": item["symbol"],
            "day_change_pct": item["day_change_pct"],
            "return_30d_pct": item["return_30d_pct"],
            "volatility_30d_pct": item["volatility_30d_pct"],
            "day_change_text": item["day_change_text"],
            "return_30d_text": item["return_30d_text"],
            "volatility_30d_text": item["volatility_30d_text"],
        }
        for item in metrics
    ]
