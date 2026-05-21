"""Payload-shaping helpers for market-data queries."""

from __future__ import annotations

from typing import Any

from ..utils import first_finite_float, utc_datetime_or_none


def pick_float(payload: dict[str, Any], *keys: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    return first_finite_float(*(payload.get(key) for key in keys))


def pick_string(payload: dict[str, Any], *keys: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return None


def merge_quote_payloads_with_source(
    primary: dict[str, Any],
    primary_name: str,
    secondary: dict[str, Any],
    secondary_name: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(primary, dict) and not isinstance(secondary, dict):
        return {}, {}
    out: dict[str, Any] = {}
    detail: dict[str, str] = {}
    keys = {
        "symbol",
        "name",
        "instrument_name",
        "exchange",
        "price",
        "close",
        "previous_close",
        "prev_close",
        "open",
        "high",
        "low",
        "volume",
        "bid",
        "ask",
        "timestamp",
        "datetime",
    }
    for key in keys:
        first = primary.get(key) if isinstance(primary, dict) else None
        second = secondary.get(key) if isinstance(secondary, dict) else None
        if first not in (None, ""):
            out[key] = first
            detail[key] = primary_name
        elif second not in (None, ""):
            out[key] = second
            detail[key] = secondary_name
    return out, detail


def series_source_descriptor(points: list[dict[str, Any]]) -> str:
    if not points:
        return "none"
    providers: set[str] = set()
    for item in points:
        src = str(item.get("_src", "")).strip().lower()
        if src:
            providers.add(src)
    if not providers:
        return "unknown"
    if len(providers) == 1:
        return next(iter(providers))
    return "mixed"


def parse_timestamp(raw: Any) -> str | None:
    parsed = utc_datetime_or_none(raw)
    return parsed.isoformat() if parsed is not None else None


def best_updated_at(
    quote_payload: dict[str, Any],
    intraday_points: list[dict[str, Any]],
    day_points: list[dict[str, Any]],
) -> str | None:
    candidates = [
        parse_timestamp(quote_payload.get("timestamp")) if isinstance(quote_payload, dict) else None,
        parse_timestamp(quote_payload.get("datetime")) if isinstance(quote_payload, dict) else None,
        parse_timestamp(intraday_points[-1]["t"]) if intraday_points else None,
        parse_timestamp(day_points[-1]["t"]) if day_points else None,
    ]
    for item in candidates:
        if item:
            return item
    return None


def build_market_item(symbol: str, latest: float | None, previous: float | None) -> dict[str, Any]:
    change_abs = None
    change_pct = None
    if latest is not None and previous is not None and previous > 0:
        change_abs = latest - previous
        change_pct = (change_abs / previous) * 100
    return {
        "symbol": symbol,
        "price": latest,
        "change_abs": change_abs,
        "change_pct": change_pct,
    }


def is_fmp_error(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("status") == "error":
        return True
    message = str(payload.get("Error Message", "")).strip()
    return bool(message)


def delay_note(provider: str) -> str:
    if provider == "both":
        return "Combined feed: Twelve Data + Financial Modeling Prep."
    if provider == "fmp":
        return "Financial Modeling Prep free plan feed."
    return "Twelve Data Basic plan (delayed feed may apply)."
