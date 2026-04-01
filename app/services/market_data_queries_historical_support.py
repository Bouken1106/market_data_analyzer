"""Pure helpers for historical query orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from fastapi import HTTPException

from ..config import (
    HISTORICAL_MAX_POINTS,
    HISTORICAL_MAX_YEARS,
    ML_HISTORY_MAX_MONTHS,
    SYMBOL_PATTERN,
    TIME_SERIES_MAX_OUTPUTSIZE,
)


@dataclass(frozen=True)
class HistoricalRequest:
    symbol: str
    source_mode: str
    years: int
    months: int | None
    fetch_full_history: bool
    cache_key: tuple[str, str, str]
    start_date: date
    end_date: date
    outputsize: int

    @property
    def start_date_iso(self) -> str:
        return self.start_date.isoformat()

    @property
    def end_date_iso(self) -> str:
        return self.end_date.isoformat()


def build_no_historical_data_detail(
    *,
    symbol: str,
    source_mode: str,
    source_detail: dict[str, Any] | None,
    allow_api_fallback: bool,
) -> str:
    detail = source_detail if isinstance(source_detail, dict) else {}
    provider = str(detail.get("provider") or source_mode or "provider").strip().lower() or "provider"
    mode = str(detail.get("mode") or "").strip().lower()
    error_text = str(detail.get("error") or "").strip()

    if provider == "stooq":
        if mode == "stooq_fetch_failed":
            return f"Stooq daily CSV fetch failed for {symbol}." + (f" {error_text}" if error_text else "")
        if mode == "stooq_empty":
            return f"Stooq daily CSV returned no rows for {symbol}."
        if mode == "stooq_empty_range":
            return f"Stooq daily CSV had no rows in the requested date range for {symbol}."
        if not allow_api_fallback:
            return f"Stooq daily data unavailable for {symbol}, and API fallback is disabled."

    if error_text:
        return error_text
    return "No historical data found for this symbol."


def is_daily_interval(interval: str) -> bool:
    return str(interval).strip().lower() in {"1day", "1d", "day"}


def build_historical_request(
    *,
    symbol: str,
    years: int,
    months: int | None,
    source_preference: str | None,
    today: date | None = None,
) -> HistoricalRequest:
    normalized = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(normalized):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")

    source_mode = str(source_preference or "").strip().lower() or "provider"
    requested_years = max(1, int(years))
    fetch_full_history = months is None and requested_years > HISTORICAL_MAX_YEARS
    resolved_years = requested_years if fetch_full_history else max(1, min(requested_years, HISTORICAL_MAX_YEARS))
    resolved_months = None if months is None else max(1, min(int(months), ML_HISTORY_MAX_MONTHS))

    if resolved_months is None:
        cache_key = (
            (normalized, "years:max", f"source:{source_mode}")
            if fetch_full_history
            else (normalized, f"years:{resolved_years}", f"source:{source_mode}")
        )
    else:
        cache_key = (normalized, f"months:{resolved_months}", f"source:{source_mode}")

    end_date = today or date.today()
    if resolved_months is None:
        requested_days = (365 * resolved_years) + (resolved_years // 4)
    else:
        requested_days = (31 * resolved_months) + 7
    start_date = end_date - timedelta(days=requested_days)
    estimated_points = max(200, int(requested_days * 0.8))
    outputsize = (
        0
        if fetch_full_history
        else min(TIME_SERIES_MAX_OUTPUTSIZE, max(HISTORICAL_MAX_POINTS, estimated_points))
    )

    return HistoricalRequest(
        symbol=normalized,
        source_mode=source_mode,
        years=resolved_years,
        months=resolved_months,
        fetch_full_history=fetch_full_history,
        cache_key=cache_key,
        start_date=start_date,
        end_date=end_date,
        outputsize=outputsize,
    )


def build_historical_payload(
    *,
    request: HistoricalRequest,
    points: list[dict[str, Any]],
    source_detail: dict[str, Any],
    provider: str,
    interval: str,
) -> dict[str, Any]:
    return {
        "symbol": request.symbol,
        "years": request.years,
        "months": request.months,
        "interval": interval,
        "from": points[0]["t"],
        "to": points[-1]["t"],
        "count": len(points),
        "points": points,
        "source": (
            "stooq-live"
            if str(source_detail.get("provider") or "").strip().lower() == "stooq"
            else f"{provider}-live"
        ),
        "source_detail": source_detail,
    }


def slice_daily_points(
    points: list[dict[str, Any]],
    *,
    start_date: str | None,
    end_date: str | None,
    outputsize: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in points:
        point_date = str(item.get("t") or "").split(" ")[0]
        if not point_date:
            continue
        if start_date and point_date < start_date:
            continue
        if end_date and point_date > end_date:
            continue
        filtered.append(dict(item))
    if outputsize > 0 and len(filtered) > outputsize:
        filtered = filtered[-outputsize:]
    return filtered
