"""Market data and stream routes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import (
    HISTORICAL_DEFAULT_YEARS,
    LOGGER,
    LMSTUDIO_MODEL,
    MAX_BASIC_SYMBOLS,
)
from ..models import SymbolUpdateRequest
from ..services.relationship_analysis import build_relationship_analysis
from ..services.watchlist_commentary import build_watchlist_commentary_payload
from ..utils import normalize_symbols, ok_json_response
from .deps import HubDep, SymbolCatalogStoreDep, UiStateStoreDep
from .validators import require_symbols

router = APIRouter()


@dataclass(frozen=True)
class RelationshipRequest:
    symbols: list[str]
    months: int
    window_days: int
    top_pairs: int
    refresh: bool


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value), maximum))


def _build_relationship_request(
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
        months=_clamp_int(months, minimum=3, maximum=60),
        window_days=_clamp_int(window_days, minimum=20, maximum=252),
        top_pairs=_clamp_int(top_pairs, minimum=1, maximum=20),
        refresh=bool(refresh),
    )


async def _gather_relationship_points(
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


def _build_relationship_response(
    *,
    request: RelationshipRequest,
    points_by_symbol: dict[str, list[dict[str, object]]],
    skipped: list[dict[str, str]],
) -> JSONResponse:
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

    return ok_json_response(
        requested_symbols=request.symbols,
        analyzed_symbols=analysis["symbols"],
        skipped_symbols=skipped,
        months=request.months,
        **analysis,
    )


def _require_basic_watchlist_symbols(symbols: str) -> list[str]:
    return require_symbols(
        symbols,
        min_count=2,
        max_count=MAX_BASIC_SYMBOLS,
        empty_detail="At least two valid symbols are required.",
        max_detail=f"You can request up to {MAX_BASIC_SYMBOLS} symbols at once.",
    )


def _latest_watchlist_commentary_payload(ui_state_store: Any) -> dict[str, Any]:
    payload = ui_state_store.get_watchlist_commentary() if ui_state_store else None
    if isinstance(payload, dict):
        return payload
    return {
        "comment": None,
        "generated_at": None,
        "model": LMSTUDIO_MODEL,
        "symbols": [],
    }


def _persist_watchlist_commentary(ui_state_store: Any, payload: dict[str, Any]) -> None:
    if ui_state_store is None:
        return
    try:
        ui_state_store.set_watchlist_commentary(payload)
    except Exception as exc:
        LOGGER.warning("Failed to persist watchlist commentary: %s", exc)


@router.get("/api/snapshot")
async def snapshot(hub: HubDep) -> JSONResponse:
    payload = await hub.snapshot_payload()
    return JSONResponse(payload)


@router.post("/api/symbols")
async def update_symbols(req: SymbolUpdateRequest, hub: HubDep) -> JSONResponse:
    symbols = normalize_symbols(req.symbols)
    await hub.set_symbols(symbols)
    rows = await hub.current_rows(symbols)
    return ok_json_response(
        symbols=symbols,
        status=await hub.status_payload(),
        rows=rows,
    )


@router.get("/api/credits")
async def credits(hub: HubDep, refresh: bool = False) -> JSONResponse:
    if refresh:
        status = await hub.refresh_api_credits()
    else:
        status = await hub.status_payload()
    note = (
        "refresh=true fetches exact daily remaining credits via /api_usage and consumes 1 API credit."
        if getattr(hub, "provider", "") in {"twelvedata", "both"}
        else "Current provider does not expose Twelve Data /api_usage credits."
    )
    return ok_json_response(
        status=status,
        note=note,
    )


@router.get("/api/symbol-catalog")
async def symbol_catalog(
    symbol_catalog_store: SymbolCatalogStoreDep,
    refresh: bool = False,
    cache_only: bool = False,
) -> JSONResponse:
    payload = await symbol_catalog_store.get_catalog(refresh=refresh, cache_only=cache_only)
    return ok_json_response(**payload)


@router.get("/api/historical/{symbol}")
async def historical(
    symbol: str,
    hub: HubDep,
    years: int = HISTORICAL_DEFAULT_YEARS,
    refresh: bool = False,
) -> JSONResponse:
    payload = await hub.historical_payload(symbol=symbol, years=years, refresh=refresh)
    return ok_json_response(**payload)


@router.get("/api/relationships")
async def relationships(
    symbols: str,
    hub: HubDep,
    months: int = 12,
    window_days: int = 60,
    top_pairs: int = 10,
    refresh: bool = False,
) -> JSONResponse:
    request = _build_relationship_request(
        symbols=symbols,
        months=months,
        window_days=window_days,
        top_pairs=top_pairs,
        refresh=refresh,
    )
    points_by_symbol, skipped = await _gather_relationship_points(
        hub,
        request,
    )
    return _build_relationship_response(
        request=request,
        points_by_symbol=points_by_symbol,
        skipped=skipped,
    )


@router.get("/api/security-overview/{symbol}")
async def security_overview(
    symbol: str,
    hub: HubDep,
    refresh: bool = False,
    include_intraday: bool = True,
    include_market: bool = True,
    include_qqq: bool = True,
) -> JSONResponse:
    payload = await hub.security_overview_payload(
        symbol=symbol,
        refresh=refresh,
        include_intraday=include_intraday,
        include_market=include_market,
        include_qqq=include_qqq,
    )
    return ok_json_response(**payload)


@router.get("/api/sparkline")
async def sparkline(symbols: str, hub: HubDep, refresh: bool = False) -> JSONResponse:
    target_symbols = require_symbols(
        symbols,
        max_count=MAX_BASIC_SYMBOLS,
        max_detail=f"You can request up to {MAX_BASIC_SYMBOLS} symbols at once.",
    )

    items = await hub.sparkline_payload(target_symbols, refresh=refresh)
    return ok_json_response(
        symbols=target_symbols,
        items=items,
    )


@router.get("/api/watchlist-commentary")
async def watchlist_commentary(
    symbols: str,
    hub: HubDep,
    ui_state_store: UiStateStoreDep,
    refresh: bool = False,
) -> JSONResponse:
    target_symbols = _require_basic_watchlist_symbols(symbols)
    payload = await build_watchlist_commentary_payload(hub, target_symbols, refresh=refresh)
    _persist_watchlist_commentary(ui_state_store, payload)
    return ok_json_response(**payload)


@router.get("/api/watchlist-commentary/latest")
async def watchlist_commentary_latest(ui_state_store: UiStateStoreDep) -> JSONResponse:
    payload = _latest_watchlist_commentary_payload(ui_state_store)
    return ok_json_response(**payload)


@router.get("/api/stream")
async def stream(request: Request, hub: HubDep) -> StreamingResponse:
    queue = hub.register_listener()

    async def event_generator():
        initial_payload = await hub.snapshot_payload()
        yield f"data: {json.dumps(initial_payload)}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            hub.unregister_listener(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
