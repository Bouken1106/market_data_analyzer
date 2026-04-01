"""Market data routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import HISTORICAL_DEFAULT_YEARS, MAX_BASIC_SYMBOLS
from ..models import SymbolUpdateRequest
from ..services.market_api_service import (
    build_relationship_payload,
    build_relationship_request,
    gather_relationship_points,
    latest_watchlist_commentary_payload,
    persist_watchlist_commentary,
    require_basic_watchlist_symbols,
)
from ..services.watchlist_commentary import build_watchlist_commentary_payload
from ..utils import normalize_symbols, ok_json_response
from ..validation import require_symbols
from .deps import HubDep, SymbolCatalogStoreDep, UiStateStoreDep
from .market_stream import build_market_stream_response

router = APIRouter()


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
    request = build_relationship_request(
        symbols=symbols,
        months=months,
        window_days=window_days,
        top_pairs=top_pairs,
        refresh=refresh,
    )
    points_by_symbol, skipped = await gather_relationship_points(
        hub,
        request,
    )
    return ok_json_response(
        **build_relationship_payload(
            request=request,
            points_by_symbol=points_by_symbol,
            skipped=skipped,
        )
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
    target_symbols = require_basic_watchlist_symbols(symbols)
    payload = await build_watchlist_commentary_payload(hub, target_symbols, refresh=refresh)
    persist_watchlist_commentary(ui_state_store, payload)
    return ok_json_response(**payload)


@router.get("/api/watchlist-commentary/latest")
async def watchlist_commentary_latest(ui_state_store: UiStateStoreDep) -> JSONResponse:
    payload = latest_watchlist_commentary_payload(ui_state_store)
    return ok_json_response(**payload)


@router.get("/api/stream")
async def stream(request: Request, hub: HubDep) -> StreamingResponse:
    return build_market_stream_response(request, hub)
