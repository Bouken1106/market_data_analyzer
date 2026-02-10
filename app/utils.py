"""General-purpose utility functions."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .config import (
    API_LIMIT_PER_DAY,
    API_LIMIT_PER_MIN,
    DAILY_BUDGET_UTILIZATION,
    ML_HISTORY_MAX_MONTHS,
    ML_HISTORY_MIN_MONTHS,
    PER_MIN_LIMIT_UTILIZATION,
    REST_MIN_POLL_INTERVAL_SEC,
    SYMBOL_PATTERN,
)


# ---------------------------------------------------------------------------
# Symbol handling
# ---------------------------------------------------------------------------

def normalize_symbols(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        tokens = [item.strip().upper() for item in raw.split(",")]
    else:
        tokens = [str(item).strip().upper() for item in raw]

    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in tokens:
        if not symbol:
            continue
        if not SYMBOL_PATTERN.match(symbol):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)

    return normalized


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def to_iso8601(value: Any) -> str:
    if isinstance(value, (int, float)):
        parsed = _datetime_from_unix(value)
        if parsed is not None:
            return parsed.isoformat()
    if isinstance(value, str) and value:
        return value
    return datetime.now(timezone.utc).isoformat()


def _datetime_from_unix(value: Any) -> datetime | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None

    # Support unix values in second/ms/us/ns order of magnitude.
    abs_value = abs(numeric)
    if abs_value >= 1e18:
        numeric /= 1_000_000_000
    elif abs_value >= 1e15:
        numeric /= 1_000_000
    elif abs_value >= 1e12:
        numeric /= 1_000

    try:
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

def effective_rest_requests_per_minute() -> float:
    # Respect both per-minute and per-day limits, then keep a safety margin.
    minute_cap = API_LIMIT_PER_MIN * PER_MIN_LIMIT_UTILIZATION
    day_cap_as_rpm = (API_LIMIT_PER_DAY * DAILY_BUDGET_UTILIZATION) / (24 * 60)
    return max(0.05, min(minute_cap, day_cap_as_rpm))


def rest_request_spacing_seconds() -> int:
    rpm = effective_rest_requests_per_minute()
    return max(REST_MIN_POLL_INTERVAL_SEC, math.ceil(60 / rpm))


def fallback_interval_seconds(symbol_count: int) -> int:
    spacing = rest_request_spacing_seconds()
    if symbol_count <= 0:
        return spacing
    # One full cycle means each tracked symbol is refreshed once.
    return symbol_count * spacing


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def ok_json_response(**payload: Any) -> JSONResponse:
    return JSONResponse({"ok": True, **payload})


# ---------------------------------------------------------------------------
# ML validation
# ---------------------------------------------------------------------------

def normalize_ml_history_months(months: int) -> int:
    value = int(months)
    if value < ML_HISTORY_MIN_MONTHS or value > ML_HISTORY_MAX_MONTHS:
        raise HTTPException(
            status_code=400,
            detail=f"months は {ML_HISTORY_MIN_MONTHS}〜{ML_HISTORY_MAX_MONTHS} の範囲で指定してください。",
        )
    return value
