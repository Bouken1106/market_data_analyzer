"""Runtime and provider helpers for historical market-data queries."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable

from ..market_session import infer_country_from_symbol
from .market_data_intervals import is_daily_interval

_JQUANTS_COVERAGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})")


def _queries_module():
    from . import market_data_queries as module

    return module


def runtime_value(name: str, default: Any) -> Any:
    return getattr(_queries_module(), name, default)


def resolve_runtime_fetcher(name: str, default: Callable[..., Any]) -> Callable[..., Any]:
    return runtime_value(name, default)


def normalize_jquants_code(symbol: str) -> str | None:
    normalized = str(symbol or "").strip().upper()
    if normalized.endswith(".T"):
        normalized = normalized[:-2]
    normalized = normalized.strip()
    if normalized.isdigit() and len(normalized) in {4, 5}:
        return normalized
    return None


def should_use_jquants_for_symbol(symbol: str, interval: str, *, api_key: str) -> bool:
    if not str(api_key or "").strip():
        return False
    if not is_daily_interval(interval):
        return False
    return infer_country_from_symbol(symbol) == "JAPAN" and normalize_jquants_code(symbol) is not None


def extract_jquants_coverage_window(message: Any) -> tuple[date, date] | None:
    matched = _JQUANTS_COVERAGE_RE.search(str(message or ""))
    if matched is None:
        return None
    try:
        return date.fromisoformat(matched.group(1)), date.fromisoformat(matched.group(2))
    except ValueError:
        return None


def bound_jquants_request_dates(
    *,
    start_date: str | None,
    end_date: str | None,
    coverage_window: tuple[date, date],
) -> tuple[str, str] | None:
    coverage_start, coverage_end = coverage_window
    try:
        requested_start = date.fromisoformat(start_date) if start_date else coverage_start
        requested_end = date.fromisoformat(end_date) if end_date else coverage_end
    except ValueError:
        return None

    bounded_start = max(requested_start, coverage_start)
    bounded_end = min(requested_end, coverage_end)
    if bounded_start > bounded_end:
        return None
    return bounded_start.isoformat(), bounded_end.isoformat()


def clamp_jquants_request_dates(
    *,
    start_date: str | None,
    end_date: str | None,
    coverage_message: Any,
) -> tuple[str, str] | None:
    coverage_window = extract_jquants_coverage_window(coverage_message)
    if coverage_window is None:
        return None

    bounded_dates = bound_jquants_request_dates(
        start_date=start_date,
        end_date=end_date,
        coverage_window=coverage_window,
    )
    if bounded_dates is None:
        return None

    clamped_start, clamped_end = bounded_dates
    if clamped_start == str(start_date or "") and clamped_end == str(end_date or ""):
        return None
    return clamped_start, clamped_end


def is_jquants_rate_limit_message(message: Any) -> bool:
    return "rate limit exceeded" in str(message or "").strip().lower()


def is_jquants_invalid_api_key_message(message: Any) -> bool:
    normalized = str(message or "").strip().lower()
    return "api key is invalid or expired" in normalized


def build_stooq_historical_detail(
    *,
    mode: str,
    points: int,
    error: str | None = None,
) -> dict[str, Any]:
    detail = {
        "mode": mode,
        "dataset": "historical_daily",
        "provider": "stooq",
        "points": points,
    }
    if error:
        detail["error"] = error
    return detail


def build_jquants_historical_detail(*, points: int) -> dict[str, Any]:
    return {
        "mode": "jquants",
        "dataset": "historical_daily",
        "provider": "jquants",
        "points": points,
    }


def build_combined_historical_detail(
    *,
    td_points: list[dict[str, Any]],
    fmp_points: list[dict[str, Any]],
    merged_points: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": "both",
        "dataset": "historical_daily",
        "merge_policy": "twelvedata_overrides_fmp_on_same_timestamp",
        "providers": {
            "twelvedata_points": len(td_points),
            "fmp_points": len(fmp_points),
            "merged_points": len(merged_points),
        },
    }


def build_standard_historical_detail(*, provider: str, points: int) -> dict[str, Any]:
    provider_name = "fmp" if provider == "fmp" else "twelvedata"
    return {
        "mode": provider,
        "dataset": "historical_daily",
        "provider": provider_name,
        "points": points,
    }
