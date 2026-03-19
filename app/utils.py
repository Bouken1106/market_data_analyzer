"""General-purpose utility functions."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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

def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def is_valid_symbol(value: Any) -> bool:
    symbol = normalize_symbol(value)
    return bool(symbol and SYMBOL_PATTERN.match(symbol))


def normalize_symbols(raw: str | Iterable[Any], *, max_items: int | None = None) -> list[str]:
    if max_items is not None and int(max_items) <= 0:
        return []
    tokens = raw.split(",") if isinstance(raw, str) else raw

    normalized: list[str] = []
    seen: set[str] = set()
    for item in tokens:
        symbol = normalize_symbol(item)
        if not symbol:
            continue
        if not SYMBOL_PATTERN.match(symbol):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
        if max_items is not None and len(normalized) >= int(max_items):
            break

    return normalized


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_iso8601(value: Any) -> str:
    if isinstance(value, (int, float)):
        parsed = _datetime_from_unix(value)
        if parsed is not None:
            return parsed.isoformat()
    if isinstance(value, str) and value:
        return value
    return utc_now_iso()


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
# JSON helpers
# ---------------------------------------------------------------------------

def clone_json_like(value: Any) -> Any:
    return deepcopy(value)


def read_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_file(path: Path, payload: Any, *, compact: bool = False) -> None:
    dump_kwargs: dict[str, Any] = {"ensure_ascii": False}
    if compact:
        dump_kwargs["separators"] = (",", ":")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, **dump_kwargs),
        encoding="utf-8",
    )


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
