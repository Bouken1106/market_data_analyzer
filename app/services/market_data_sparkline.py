"""Shared helpers for compact daily-close sparkline payloads."""

from __future__ import annotations

from typing import Any, Callable

from ..utils import finite_float_or_none

SparklineValue = tuple[str, float]
CloseParser = Callable[[Any], float | None]


def positive_close_value(value: Any) -> float | None:
    return finite_float_or_none(value, minimum=0.0, strict_minimum=True)


def daily_close_values(
    points: list[dict[str, Any]],
    *,
    parse_close: CloseParser = positive_close_value,
    date_only: bool = False,
) -> list[SparklineValue]:
    values: list[SparklineValue] = []
    for point in points:
        raw_timestamp = str(point.get("t") or "").strip()
        if not raw_timestamp:
            continue
        close_value = parse_close(point.get("c"))
        if close_value is None:
            continue
        timestamp = raw_timestamp.split(" ")[0] if date_only else raw_timestamp
        values.append((timestamp, close_value))
    values.sort(key=lambda item: item[0], reverse=True)
    return values


def completed_daily_values(values: list[SparklineValue], *, today_iso: str) -> list[SparklineValue]:
    if len(values) < 2:
        return []
    start_index = 1 if values[0][0].startswith(today_iso) else 0
    completed = values[start_index:]
    return completed if len(completed) >= 2 else []


def build_daily_sparkline_payload(
    *,
    symbol: str,
    completed: list[SparklineValue],
    max_points: int,
    current_price: float | None,
    reference_close: float | None,
    updated_at: Any,
    source: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if len(completed) < 2:
        return None

    latest_date, latest_close = completed[0]
    previous_date, previous_close = completed[1]
    resolved_reference_close = reference_close if reference_close is not None else previous_close
    change_abs = None
    change_pct = None
    if current_price is not None and resolved_reference_close is not None and resolved_reference_close > 0:
        change_abs = current_price - resolved_reference_close
        change_pct = (change_abs / resolved_reference_close) * 100.0

    recent_desc = completed[:max_points]
    recent_asc = list(reversed(recent_desc))
    trend_values = [point[1] for point in recent_asc]
    payload = {
        "symbol": symbol,
        "latest_close": latest_close,
        "latest_close_date": latest_date,
        "previous_close": previous_close,
        "previous_close_date": previous_date,
        "current_price": current_price,
        "reference_close": resolved_reference_close,
        "change_abs": change_abs,
        "change_pct": change_pct,
        "updated_at": updated_at,
        "trend_30d": trend_values,
        "trend_from": recent_asc[0][0],
        "trend_to": recent_asc[-1][0],
        "points": len(trend_values),
        "source": source,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def provider_from_points(points: list[dict[str, Any]], *, default_provider: str) -> str:
    providers = {
        str(point.get("_src") or "").strip().lower()
        for point in points
        if isinstance(point, dict) and str(point.get("_src") or "").strip()
    }
    if len(providers) == 1:
        return next(iter(providers))
    if len(providers) > 1:
        return "mixed"
    return default_provider
