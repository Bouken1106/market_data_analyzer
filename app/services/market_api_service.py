"""Application-level helpers shared by market API routes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Any

from fastapi import HTTPException

from ..config import LOGGER, settings
from ..validation import require_symbols
from .market_data_sparkline import (
    build_daily_sparkline_payload,
    completed_daily_values,
    daily_close_values,
    provider_from_points,
)
from .relationship_analysis import build_relationship_analysis


@dataclass(frozen=True)
class RelationshipRequest:
    symbols: list[str]
    months: int
    window_days: int
    top_pairs: int
    refresh: bool


def clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value), maximum))


def build_relationship_request(
    *,
    symbols: str,
    months: int,
    window_days: int,
    top_pairs: int,
    refresh: bool,
) -> RelationshipRequest:
    target_symbols = require_symbols(
        symbols,
        min_count=2,
        max_count=12,
        empty_detail="At least two valid symbols are required.",
        max_detail="You can request up to 12 symbols at once.",
    )
    return RelationshipRequest(
        symbols=target_symbols,
        months=clamp_int(months, minimum=3, maximum=60),
        window_days=clamp_int(window_days, minimum=20, maximum=252),
        top_pairs=clamp_int(top_pairs, minimum=1, maximum=20),
        refresh=bool(refresh),
    )


async def gather_relationship_points(
    hub: Any,
    request: RelationshipRequest,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, str]]]:
    responses = await asyncio.gather(
        *[
            hub.historical_payload(symbol=symbol, months=request.months, refresh=request.refresh)
            for symbol in request.symbols
        ],
        return_exceptions=True,
    )

    points_by_symbol: dict[str, list[dict[str, object]]] = {}
    skipped: list[dict[str, str]] = []
    for symbol, item in zip(request.symbols, responses):
        if isinstance(item, Exception):
            detail = getattr(item, "detail", None)
            skipped.append({"symbol": symbol, "reason": str(detail or item)})
            continue
        points = item.get("points") if isinstance(item, dict) else None
        if isinstance(points, list) and points:
            points_by_symbol[symbol] = points
            continue
        skipped.append({"symbol": symbol, "reason": "No historical points returned."})

    return points_by_symbol, skipped


def build_relationship_payload(
    *,
    request: RelationshipRequest,
    points_by_symbol: dict[str, list[dict[str, object]]],
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    if len(points_by_symbol) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough symbols returned aligned historical data for relationship analysis.",
        )

    try:
        analysis = build_relationship_analysis(
            points_by_symbol,
            window_days=request.window_days,
            top_pairs=request.top_pairs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "requested_symbols": request.symbols,
        "analyzed_symbols": analysis["symbols"],
        "skipped_symbols": skipped,
        "months": request.months,
        **analysis,
    }


def require_basic_watchlist_symbols(symbols: str) -> list[str]:
    max_symbols = settings.provider.max_basic_symbols
    return require_symbols(
        symbols,
        min_count=2,
        max_count=max_symbols,
        empty_detail="At least two valid symbols are required.",
        max_detail=f"You can request up to {max_symbols} symbols at once.",
    )


def build_eod_sparkline_item(
    *,
    symbol: str,
    points: list[dict[str, Any]],
    source_detail: dict[str, Any] | None,
    default_provider: str = "stooq",
) -> dict[str, Any] | None:
    values = daily_close_values(points, date_only=True)
    completed = completed_daily_values(values, today_iso=date.today().isoformat())
    if len(completed) < 2:
        return None

    provider = str((source_detail or {}).get("provider") or "").strip().lower()
    if not provider:
        provider = provider_from_points(points, default_provider=default_provider)
    return build_daily_sparkline_payload(
        symbol=symbol,
        completed=completed,
        max_points=settings.overview.sparkline_points,
        current_price=completed[0][1],
        reference_close=completed[1][1],
        updated_at=completed[0][0],
        source=f"{provider}_eod",
        extra_fields={
            "price_mode": "eod_close",
            "source_detail": {
                "provider": provider,
                "mode": "eod_close",
                "base": source_detail or {},
            },
        },
    )


async def gather_eod_sparkline_items(
    hub: Any,
    *,
    symbols: list[str],
    refresh: bool,
) -> list[dict[str, Any]]:
    provider_name = str(getattr(hub, "provider", "") or "provider").strip().lower() or "provider"
    provider_responses = await asyncio.gather(
        *[
            hub.historical_payload(
                symbol=symbol,
                months=2,
                refresh=refresh,
            )
            for symbol in symbols
        ],
        return_exceptions=True,
    )

    items_by_symbol: dict[str, dict[str, Any]] = {}
    fallback_symbols: list[str] = []
    for symbol, response in zip(symbols, provider_responses):
        if isinstance(response, Exception):
            detail = getattr(response, "detail", None)
            LOGGER.warning("Provider EOD sparkline fetch failed for %s: %s", symbol, detail or response)
            fallback_symbols.append(symbol)
            continue
        points = response.get("points") if isinstance(response, dict) else None
        if not isinstance(points, list) or not points:
            fallback_symbols.append(symbol)
            continue
        item = build_eod_sparkline_item(
            symbol=symbol,
            points=[dict(point) for point in points if isinstance(point, dict)],
            source_detail=response.get("source_detail") if isinstance(response, dict) else None,
            default_provider=provider_name,
        )
        if item is None:
            fallback_symbols.append(symbol)
            continue
        items_by_symbol[symbol] = item

    if not fallback_symbols:
        return [items_by_symbol[symbol] for symbol in symbols if symbol in items_by_symbol]

    stooq_responses = await asyncio.gather(
        *[
            hub.historical_payload(
                symbol=symbol,
                years=1,
                refresh=refresh,
                source_preference="stooq",
                allow_api_fallback=False,
            )
            for symbol in fallback_symbols
        ],
        return_exceptions=True,
    )

    for symbol, response in zip(fallback_symbols, stooq_responses):
        if isinstance(response, Exception):
            detail = getattr(response, "detail", None)
            LOGGER.warning("EOD sparkline fetch failed for %s: %s", symbol, detail or response)
            continue
        points = response.get("points") if isinstance(response, dict) else None
        if not isinstance(points, list) or not points:
            continue
        item = build_eod_sparkline_item(
            symbol=symbol,
            points=[dict(point) for point in points if isinstance(point, dict)],
            source_detail=response.get("source_detail") if isinstance(response, dict) else None,
            default_provider="stooq",
        )
        if item is not None:
            items_by_symbol[symbol] = item
    return [items_by_symbol[symbol] for symbol in symbols if symbol in items_by_symbol]


async def gather_cached_eod_sparkline_items(
    hub: Any,
    *,
    symbols: list[str],
) -> list[dict[str, Any]]:
    store = getattr(hub, "full_daily_history_store", None)
    if store is None:
        return []

    items: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            points = await store.get(symbol, copy=True)
        except Exception as exc:
            LOGGER.warning("Cached EOD sparkline read failed for %s: %s", symbol, exc)
            continue
        if not points:
            continue
        item = build_eod_sparkline_item(
            symbol=symbol,
            points=[dict(point) for point in points if isinstance(point, dict)],
            source_detail={"provider": provider_from_points(points, default_provider="cache")},
            default_provider="cache",
        )
        if item is not None:
            items.append(item)
    return items


def latest_watchlist_commentary_payload(ui_state_store: Any) -> dict[str, Any]:
    payload = ui_state_store.get_watchlist_commentary() if ui_state_store else None
    if isinstance(payload, dict):
        return payload
    return {
        "comment": None,
        "generated_at": None,
        "model": settings.lmstudio.lmstudio_model,
        "symbols": [],
    }


def persist_watchlist_commentary(ui_state_store: Any, payload: dict[str, Any]) -> None:
    if ui_state_store is None:
        return
    try:
        ui_state_store.set_watchlist_commentary(payload)
    except Exception as exc:
        LOGGER.warning("Failed to persist watchlist commentary: %s", exc)
