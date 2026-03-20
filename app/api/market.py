"""Market data and stream routes."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import (
    HISTORICAL_DEFAULT_YEARS,
    LMSTUDIO_MODEL,
    MAX_BASIC_SYMBOLS,
)
from ..models import (
    MarketDataLabOnboardingRequest,
    MarketDataLabStateRequest,
    SymbolUpdateRequest,
)
from ..services.watchlist_commentary import build_watchlist_commentary_payload
from ..utils import (
    fallback_interval_seconds,
    normalize_symbols,
    ok_json_response,
    rest_request_spacing_seconds,
)
from .deps import HubDep, SymbolCatalogStoreDep, UiStateStoreDep

router = APIRouter()

MARKET_DATA_LAB_HELP_TEXTS = {
    "provider": "データプロバイダーは株価や銘柄情報の取得元です。表示が止まったらどの提供元を使っているかを先に確認してください。無料版では取得範囲や更新頻度に上限があります。",
    "update_mode": "更新モードは画面がどうやって値を取り直すかを示します。このページではウォッチリストを壊さないために REST 定期更新を優先します。無料版では高速連打より間隔を空けた更新が安定します。",
    "change_pct": "前日比は前回終値に対する現在値の変化率です。上昇率だけでなく値幅と合わせて見ると動きの強さを判断しやすくなります。無料版では遅延や取得失敗時に最新約定ではなく近い値を使うことがあります。",
    "volume": "出来高はその日に売買された株数です。20日平均と比べると注目度の変化が見やすくなります。無料版では板情報までは取れないため売買の細かい内訳は見えません。",
    "vwap": "VWAP はその日の出来高加重平均価格です。現在値が VWAP より上か下かで当日の売買位置をざっくり見ます。無料版では日中足が欠けると未対応表示になることがあります。",
    "ma": "MA は移動平均線で、ここでは 20 日と 50 日を表示します。現在値と平均線の位置関係を見ると短中期の流れを把握しやすくなります。無料版では長い期間の追加指標は絞って表示します。",
    "atr": "ATR は値動きの大きさを表す平均的なレンジです。数値が大きいほど一日のブレ幅が大きい銘柄と考えます。無料版では日足履歴が不足すると算出できません。",
    "beta_corr": "ベータと相関は SPY など市場全体に対する連動度です。指数と一緒に動きやすいかを確認したいときに使います。無料版では簡易計算のみで期間や比較先は限定されます。",
    "cache": "キャッシュは直前に取得した結果を再利用して API 消費を抑える仕組みです。更新時刻を見れば今の表示がどれくらい新しいか判断できます。無料版ではこの仕組みで呼び出し回数を節約します。",
    "credits": "クレジットはその日やその分に使える API 呼び出し残量の目安です。残量が少ないときは監視銘柄を減らすか更新間隔を延ばしてください。無料版では上限が小さいので連続更新に向きません。",
    "fallback": "フォールバックは通常取得が不安定なときに別の取り方へ切り替えることです。接続が不安定でも画面全体を止めないための保険として見てください。無料版では未対応項目が増える代わりにページ継続を優先します。",
}

MARKET_DATA_LAB_UNSUPPORTED_FEATURES = [
    {"key": "order_book", "label": "板情報", "reason": "無料プロバイダー未対応"},
    {"key": "news_headlines", "label": "ニュース", "reason": "このページの対象外"},
    {"key": "corporate_events", "label": "企業イベント", "reason": "無料モードでは簡易表示のみ"},
    {"key": "earnings_calendar", "label": "決算カレンダー", "reason": "初版では未統合"},
]


def _market_data_lab_provider_label(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "both":
        return "Both"
    if normalized == "fmp":
        return "FMP Free"
    return "Twelve Data Basic"


def _market_data_lab_update_mode_label(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized in {"twelvedata", "both"}:
        return "REST polling (WebSocket available on main Monitor)"
    return "REST polling"


def _market_data_lab_watchlist_state(ui_state_store, hub) -> dict[str, object]:
    state = ui_state_store.get_market_data_lab_state()
    watchlist_symbols = state.get("watchlist_symbols") if isinstance(state, dict) else []
    if isinstance(watchlist_symbols, list) and watchlist_symbols:
        return state

    inherited_symbols = []
    try:
        inherited_symbols = ui_state_store.get_symbols()
    except Exception:
        inherited_symbols = []
    if not inherited_symbols:
        inherited_symbols = list(getattr(hub, "symbols", []) or [])
    if not inherited_symbols:
        return state

    return ui_state_store.set_market_data_lab_state(
        {
            "watchlist_symbols": inherited_symbols[:MAX_BASIC_SYMBOLS],
            "last_viewed_symbol": inherited_symbols[0],
        }
    )


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


@router.get("/api/market-data-lab/bootstrap")
async def market_data_lab_bootstrap(hub: HubDep, ui_state_store: UiStateStoreDep) -> JSONResponse:
    state = _market_data_lab_watchlist_state(ui_state_store, hub)
    watchlist_symbols = normalize_symbols(state.get("watchlist_symbols", []), max_items=MAX_BASIC_SYMBOLS)
    current_symbols_count = len(watchlist_symbols)
    provider = str(getattr(hub, "provider", "") or "").strip().lower()
    status = await hub.status_payload()
    return ok_json_response(
        provider=provider,
        provider_mode_label=_market_data_lab_provider_label(provider),
        update_mode_label=_market_data_lab_update_mode_label(provider),
        max_symbols=MAX_BASIC_SYMBOLS,
        current_symbols_count=current_symbols_count,
        supports_websocket=provider in {"twelvedata", "both"},
        supports_rest=True,
        supports_fmp_reference=bool(getattr(hub, "fmp_api_key", "")),
        rest_min_poll_interval_sec=rest_request_spacing_seconds(),
        recommended_poll_interval_sec=fallback_interval_seconds(max(1, current_symbols_count)),
        unsupported_features=MARKET_DATA_LAB_UNSUPPORTED_FEATURES,
        help_texts=MARKET_DATA_LAB_HELP_TEXTS,
        onboarding_enabled=not bool(state.get("onboarding_dismissed")),
        state=state,
        status=status,
        configured_sources={
            "fmp": bool(getattr(hub, "fmp_api_key", "")),
            "twelvedata": bool(getattr(hub, "twelvedata_api_key", "")),
        },
    )


@router.get("/api/market-data-lab/onboarding")
async def market_data_lab_onboarding(ui_state_store: UiStateStoreDep) -> JSONResponse:
    return ok_json_response(**ui_state_store.get_market_data_lab_onboarding())


@router.post("/api/market-data-lab/onboarding")
async def market_data_lab_set_onboarding(
    req: MarketDataLabOnboardingRequest,
    ui_state_store: UiStateStoreDep,
) -> JSONResponse:
    payload = ui_state_store.set_market_data_lab_onboarding(req.dismissed)
    return ok_json_response(**payload)


@router.post("/api/market-data-lab/state")
async def market_data_lab_set_state(
    req: MarketDataLabStateRequest,
    ui_state_store: UiStateStoreDep,
) -> JSONResponse:
    state = ui_state_store.set_market_data_lab_state(
        {
            "watchlist_symbols": req.watchlist_symbols,
            "last_viewed_symbol": req.last_viewed_symbol,
            "chart_interval": req.chart_interval,
        }
    )
    watchlist_symbols = normalize_symbols(state.get("watchlist_symbols", []), max_items=MAX_BASIC_SYMBOLS)
    return ok_json_response(
        state=state,
        current_symbols_count=len(watchlist_symbols),
        recommended_poll_interval_sec=fallback_interval_seconds(max(1, len(watchlist_symbols))),
    )


@router.get("/api/market-data-lab/quotes")
async def market_data_lab_quotes(symbols: str, hub: HubDep) -> JSONResponse:
    target_symbols = normalize_symbols(symbols, max_items=MAX_BASIC_SYMBOLS)
    if not target_symbols:
        raise HTTPException(status_code=400, detail="At least one valid symbol is required.")
    items = await hub.market_data_lab_quotes_payload(target_symbols)
    return ok_json_response(
        symbols=target_symbols,
        items=items,
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


@router.get("/api/security-overview/{symbol}/intraday")
async def security_overview_intraday(symbol: str, hub: HubDep, refresh: bool = False) -> JSONResponse:
    payload = await hub.security_overview_payload(
        symbol=symbol,
        refresh=refresh,
        include_intraday=True,
        include_market=False,
        include_qqq=False,
    )
    return ok_json_response(
        symbol=payload.get("symbol"),
        technical={
            "vwap_1m": payload.get("technical", {}).get("vwap_1m") if isinstance(payload.get("technical"), dict) else None,
            "vwap_5m": payload.get("technical", {}).get("vwap_5m") if isinstance(payload.get("technical"), dict) else None,
        },
        charts={
            "1min": payload.get("charts", {}).get("1min") if isinstance(payload.get("charts"), dict) else [],
            "5min": payload.get("charts", {}).get("5min") if isinstance(payload.get("charts"), dict) else [],
        },
        source=payload.get("source"),
    )


@router.post("/api/security-overview/{symbol}/clear-cache")
async def clear_security_overview_cache(symbol: str, hub: HubDep) -> JSONResponse:
    payload = await hub.clear_symbol_overview_cache(symbol=symbol)
    return ok_json_response(**payload)


@router.get("/api/fmp-reference/{symbol}")
async def fmp_reference(symbol: str, hub: HubDep, refresh: bool = False, cache_only: bool = False) -> JSONResponse:
    payload = await hub.fmp_reference_payload(symbol=symbol, refresh=refresh, cache_only=cache_only)
    return ok_json_response(**payload)


@router.post("/api/fmp-reference/{symbol}/clear-cache")
async def clear_fmp_reference_cache(symbol: str, hub: HubDep) -> JSONResponse:
    payload = await hub.clear_fmp_reference_cache(symbol=symbol)
    return ok_json_response(**payload)


@router.get("/api/sparkline")
async def sparkline(symbols: str, hub: HubDep, refresh: bool = False) -> JSONResponse:
    target_symbols = normalize_symbols(symbols)
    if not target_symbols:
        raise HTTPException(status_code=400, detail="At least one valid symbol is required.")
    if len(target_symbols) > MAX_BASIC_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"You can request up to {MAX_BASIC_SYMBOLS} symbols at once.")

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
    target_symbols = normalize_symbols(symbols)
    if len(target_symbols) < 2:
        raise HTTPException(status_code=400, detail="At least two valid symbols are required.")
    if len(target_symbols) > MAX_BASIC_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"You can request up to {MAX_BASIC_SYMBOLS} symbols at once.")

    payload = await build_watchlist_commentary_payload(hub, target_symbols, refresh=refresh)
    try:
        ui_state_store.set_watchlist_commentary(payload)
    except Exception:
        pass
    return ok_json_response(**payload)


@router.get("/api/watchlist-commentary/latest")
async def watchlist_commentary_latest(ui_state_store: UiStateStoreDep) -> JSONResponse:
    payload = ui_state_store.get_watchlist_commentary() if ui_state_store else None
    if not isinstance(payload, dict):
        return ok_json_response(comment=None, generated_at=None, model=LMSTUDIO_MODEL, symbols=[])
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
