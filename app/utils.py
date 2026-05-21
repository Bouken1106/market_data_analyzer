"""General-purpose utility functions."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from fastapi.responses import JSONResponse

from .config import (
    API_LIMIT_PER_DAY,
    API_LIMIT_PER_MIN,
    DAILY_BUDGET_UTILIZATION,
    PER_MIN_LIMIT_UTILIZATION,
    REST_MIN_POLL_INTERVAL_SEC,
    SYMBOL_PATTERN,
)


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Object helpers
# ---------------------------------------------------------------------------

def cached_attr(owner: Any, name: str, factory: Callable[[], T]) -> T:
    value = getattr(owner, name, None)
    if value is None:
        value = factory()
        setattr(owner, name, value)
    return value


def exception_detail_text(exc: BaseException) -> str:
    detail = getattr(exc, "detail", None)
    return str(detail or exc)


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
# Numeric helpers
# ---------------------------------------------------------------------------

def finite_float_or_none(
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if minimum is None:
        return numeric
    if strict_minimum and numeric <= minimum:
        return None
    if not strict_minimum and numeric < minimum:
        return None
    return numeric


def first_finite_float(
    *values: Any,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float | None:
    for value in values:
        parsed = finite_float_or_none(value, minimum=minimum, strict_minimum=strict_minimum)
        if parsed is not None:
            return parsed
    return None


def scaled_ratio(
    numerator: Any,
    denominator: Any,
    *,
    scale: float = 1.0,
    require_positive_denominator: bool = True,
) -> float | None:
    numerator_value = finite_float_or_none(numerator)
    denominator_value = finite_float_or_none(denominator)
    if numerator_value is None or denominator_value is None:
        return None
    if require_positive_denominator:
        if denominator_value <= 0:
            return None
    elif denominator_value == 0:
        return None
    return (numerator_value / denominator_value) * scale


def percent_of(numerator: Any, denominator: Any) -> float | None:
    return scaled_ratio(numerator, denominator, scale=100.0)


def change_abs_percent(current: Any, previous: Any) -> tuple[float | None, float | None]:
    current_value = finite_float_or_none(current)
    previous_value = finite_float_or_none(previous)
    if current_value is None or previous_value is None or previous_value <= 0:
        return None, None
    change_abs = current_value - previous_value
    return change_abs, percent_of(change_abs, previous_value)


def percent_change(current: Any, previous: Any) -> float | None:
    _change_abs, change_pct = change_abs_percent(current, previous)
    return change_pct


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_iso8601(value: Any) -> str:
    if isinstance(value, (int, float)):
        parsed = datetime_from_unix(value)
        if parsed is not None:
            return parsed.isoformat()
    if isinstance(value, str) and value:
        return value
    return utc_now_iso()


def datetime_from_unix(value: Any) -> datetime | None:
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


def datetime_from_iso8601(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_datetime_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime_from_unix(value)

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return datetime_from_unix(text)
    numeric = finite_float_or_none(text)
    if numeric is not None:
        return datetime_from_unix(numeric)
    return datetime_from_iso8601(text)


def date_or_none(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    date_text = text.split(" ")[0]
    try:
        return date.fromisoformat(date_text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def iso_date_or_none(value: Any) -> str | None:
    parsed = date_or_none(value)
    return parsed.isoformat() if parsed is not None else None


def date_key_or_none(value: Any) -> str | None:
    parsed = iso_date_or_none(value)
    if parsed is not None:
        return parsed
    text = str(value or "").strip()
    if not text:
        return None
    return text.split(" ")[0]


def epoch_from_iso8601(value: Any) -> float | None:
    parsed = datetime_from_iso8601(value)
    return parsed.timestamp() if parsed is not None else None


_datetime_from_unix = datetime_from_unix


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
