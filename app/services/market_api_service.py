"""Application-level helpers shared by market API routes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from ..config import LMSTUDIO_MODEL, LOGGER, MAX_BASIC_SYMBOLS
from ..validation import require_symbols
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
    return require_symbols(
        symbols,
        min_count=2,
        max_count=MAX_BASIC_SYMBOLS,
        empty_detail="At least two valid symbols are required.",
        max_detail=f"You can request up to {MAX_BASIC_SYMBOLS} symbols at once.",
    )


def latest_watchlist_commentary_payload(ui_state_store: Any) -> dict[str, Any]:
    payload = ui_state_store.get_watchlist_commentary() if ui_state_store else None
    if isinstance(payload, dict):
        return payload
    return {
        "comment": None,
        "generated_at": None,
        "model": LMSTUDIO_MODEL,
        "symbols": [],
    }


def persist_watchlist_commentary(ui_state_store: Any, payload: dict[str, Any]) -> None:
    if ui_state_store is None:
        return
    try:
        ui_state_store.set_watchlist_commentary(payload)
    except Exception as exc:
        LOGGER.warning("Failed to persist watchlist commentary: %s", exc)
