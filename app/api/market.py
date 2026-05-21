"""Market data routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import settings
from ..models import SymbolUpdateRequest, WatchlistStateRequest
from ..services.market_api_service import (
    build_relationship_payload,
    build_relationship_request,
    gather_cached_eod_sparkline_items,
    gather_eod_sparkline_items,
    gather_relationship_points,
    latest_watchlist_commentary_payload,
    persist_watchlist_commentary,
    require_basic_watchlist_symbols,
)
from ..services.valuation_service import build_valuation_payload
from ..services.watchlist_commentary import build_watchlist_commentary_payload
from ..services.watchlist_state import (
    SUPPORTED_WATCHLIST_NAMESPACES,
    UnsupportedWatchlistNamespace,
    normalize_watchlist_namespace,
)
from ..utils import normalize_symbols, ok_json_response
from ..validation import require_symbols
from .deps import HubDep, SymbolCatalogStoreDep, UiStateStoreDep
from .market_stream import build_market_stream_response

router = APIRouter()


def _normalize_watchlist_namespace(namespace: str | None) -> str:
    try:
        return normalize_watchlist_namespace(namespace, allowed=SUPPORTED_WATCHLIST_NAMESPACES)
    except UnsupportedWatchlistNamespace:
        raise HTTPException(status_code=400, detail="Unsupported watchlist namespace.")


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


@router.get("/api/watchlist-state")
async def get_watchlist_state(
    ui_state_store: UiStateStoreDep,
    namespace: str | None = None,
) -> JSONResponse:
    normalized_namespace = _normalize_watchlist_namespace(namespace)
    return ok_json_response(
        namespace=normalized_namespace,
        symbols=ui_state_store.get_symbols(namespace=normalized_namespace),
    )


@router.post("/api/watchlist-state")
async def update_watchlist_state(
    req: WatchlistStateRequest,
    ui_state_store: UiStateStoreDep,
) -> JSONResponse:
    normalized_namespace = _normalize_watchlist_namespace(req.namespace)
    symbols = normalize_symbols(req.symbols)
    ui_state_store.set_symbols(symbols, namespace=normalized_namespace)
    return ok_json_response(
        namespace=normalized_namespace,
        symbols=ui_state_store.get_symbols(namespace=normalized_namespace),
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
    country: str | None = None,
) -> JSONResponse:
    payload = await symbol_catalog_store.get_catalog(
        refresh=refresh,
        cache_only=cache_only,
        country=country,
    )
    return ok_json_response(**payload)


@router.get("/api/historical/{symbol}")
async def historical(
    symbol: str,
    hub: HubDep,
    years: int = settings.historical.historical_default_years,
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


@router.post("/api/security-overview/{symbol}/clear-cache")
async def clear_security_overview_cache(symbol: str, hub: HubDep) -> JSONResponse:
    payload = await hub.clear_symbol_overview_cache(symbol)
    return ok_json_response(**payload)


@router.get("/api/fmp-reference/{symbol}")
async def fmp_reference(
    symbol: str,
    hub: HubDep,
    refresh: bool = False,
    cache_only: bool = False,
) -> JSONResponse:
    payload = await hub.fmp_reference_payload(symbol, refresh=refresh, cache_only=cache_only)
    return ok_json_response(**payload)


@router.post("/api/fmp-reference/{symbol}/clear-cache")
async def clear_fmp_reference_cache(symbol: str, hub: HubDep) -> JSONResponse:
    payload = await hub.clear_fmp_reference_cache(symbol)
    return ok_json_response(**payload)


@router.get("/api/valuation/{symbol}")
async def valuation(
    symbol: str,
    hub: HubDep,
    refresh: bool = False,
    cache_only: bool = True,
    fair_per: float | None = None,
    fair_pbr: float | None = None,
    fair_psr: float | None = None,
    fair_ev_sales: float | None = None,
    fair_ev_ebitda: float | None = None,
    fair_ev_fcf: float | None = None,
    fair_p_fcf: float | None = None,
    target_dividend_yield: float | None = None,
    risk_free_rate: float | None = None,
    equity_risk_premium: float = 0.055,
    terminal_growth_rate: float = 0.01,
    fcf_growth_rate: float = 0.02,
    forecast_years: int = 5,
) -> JSONResponse:
    payload = await build_valuation_payload(
        hub,
        symbol,
        refresh=refresh,
        cache_only=cache_only,
        fair_per=fair_per,
        fair_pbr=fair_pbr,
        fair_psr=fair_psr,
        fair_ev_sales=fair_ev_sales,
        fair_ev_ebitda=fair_ev_ebitda,
        fair_ev_fcf=fair_ev_fcf,
        fair_p_fcf=fair_p_fcf,
        target_dividend_yield=target_dividend_yield,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        terminal_growth_rate=terminal_growth_rate,
        fcf_growth_rate=fcf_growth_rate,
        forecast_years=forecast_years,
    )
    return ok_json_response(**payload)


@router.get("/api/sparkline")
async def sparkline(
    symbols: str,
    hub: HubDep,
    refresh: bool = False,
    eod_only: bool = False,
    eod_cache_only: bool = False,
) -> JSONResponse:
    max_symbols = settings.provider.max_basic_symbols
    target_symbols = require_symbols(
        symbols,
        max_count=max_symbols,
        max_detail=f"You can request up to {max_symbols} symbols at once.",
    )

    if eod_cache_only:
        items = await gather_cached_eod_sparkline_items(hub, symbols=target_symbols)
    elif eod_only:
        items = await gather_eod_sparkline_items(hub, symbols=target_symbols, refresh=refresh)
    else:
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
