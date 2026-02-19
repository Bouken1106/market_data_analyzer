"""Market data and stream routes."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import (
    HISTORICAL_DEFAULT_YEARS,
    LMSTUDIO_API_KEY,
    LMSTUDIO_CHAT_COMPLETIONS_URL,
    LMSTUDIO_MODEL,
    LMSTUDIO_TIMEOUT_SEC,
    MAX_BASIC_SYMBOLS,
)
from ..models import SymbolUpdateRequest
from ..utils import normalize_symbols, ok_json_response
from .deps import HubDep, SymbolCatalogStoreDep, UiStateStoreDep

router = APIRouter()


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_signed_percent(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}%"


def _compute_watch_metrics(symbol: str, sparkline_item: dict[str, Any] | None) -> dict[str, Any]:
    latest_close = _safe_float(sparkline_item.get("latest_close")) if isinstance(sparkline_item, dict) else None
    previous_close = _safe_float(sparkline_item.get("previous_close")) if isinstance(sparkline_item, dict) else None

    trend_raw = sparkline_item.get("trend_30d") if isinstance(sparkline_item, dict) else []
    trend_closes: list[float] = []
    if isinstance(trend_raw, list):
        for raw_value in trend_raw:
            close_value = _safe_float(raw_value)
            if close_value is None or close_value <= 0:
                continue
            trend_closes.append(close_value)

    day_change_pct: float | None = None
    if latest_close is not None and previous_close is not None and previous_close > 0:
        day_change_pct = ((latest_close - previous_close) / previous_close) * 100

    return_30d_pct: float | None = None
    if len(trend_closes) >= 2 and trend_closes[0] > 0:
        return_30d_pct = ((trend_closes[-1] - trend_closes[0]) / trend_closes[0]) * 100

    daily_returns: list[float] = []
    for idx in range(1, len(trend_closes)):
        prev_close = trend_closes[idx - 1]
        curr_close = trend_closes[idx]
        if prev_close <= 0:
            continue
        daily_returns.append((curr_close / prev_close) - 1.0)

    volatility_30d_pct: float | None = None
    if daily_returns:
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((item - mean_return) ** 2 for item in daily_returns) / len(daily_returns)
        volatility_30d_pct = math.sqrt(max(variance, 0.0)) * 100

    return {
        "symbol": symbol,
        "day_change_pct": day_change_pct,
        "return_30d_pct": return_30d_pct,
        "volatility_30d_pct": volatility_30d_pct,
        "day_change_text": _format_signed_percent(day_change_pct),
        "return_30d_text": _format_signed_percent(return_30d_pct),
        "volatility_30d_text": _format_percent(volatility_30d_pct),
    }


def _build_watchlist_prompt(current_date: str, metrics: list[dict[str, Any]]) -> str:
    lines = [
        f"現在({current_date})の銘柄の情報は以下の通りです",
        "",
        "銘柄\t前日比\t30日リターン\t30日ボラティリティ",
    ]
    for item in metrics:
        lines.append(
            f"{item['symbol']}\t{item['day_change_text']}\t{item['return_30d_text']}\t{item['volatility_30d_text']}"
        )
    lines.extend(
        [
            "",
            "これらの銘柄の中から2つ特徴的な銘柄を選び、それらに対する情報を一言で短くまとめて下さい。"
            "(与えられてない情報を用いた分析は不要で、まとめの文章のみ書け)",
        ]
    )
    return "\n".join(lines)


async def _request_lmstudio_commentary(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"

    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }

    timeout = httpx.Timeout(LMSTUDIO_TIMEOUT_SEC, connect=min(10.0, LMSTUDIO_TIMEOUT_SEC))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(LMSTUDIO_CHAT_COMPLETIONS_URL, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LM Studio request failed: {exc}") from exc

    try:
        result = response.json()
    except ValueError:
        result = {}

    if response.status_code >= 400:
        detail: str | None = None
        if isinstance(result, dict):
            error = result.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "").strip() or None
            elif isinstance(error, str):
                detail = error.strip() or None
            if detail is None:
                detail = str(result.get("detail") or "").strip() or None
        raise HTTPException(
            status_code=502,
            detail=f"LM Studio error: {detail or f'HTTP {response.status_code}'}",
        )

    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="LM Studio returned an invalid response format.")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=502, detail="LM Studio response does not include choices.")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    commentary = str(content or "").strip()
    if not commentary:
        raise HTTPException(status_code=502, detail="LM Studio returned an empty commentary.")
    return commentary


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

    sparkline_items = await hub.sparkline_payload(target_symbols, refresh=refresh)
    items_by_symbol: dict[str, dict[str, Any]] = {}
    for item in sparkline_items:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        items_by_symbol[symbol] = item

    metrics = [
        _compute_watch_metrics(symbol, items_by_symbol.get(symbol))
        for symbol in target_symbols
    ]
    current_date = datetime.now(timezone.utc).astimezone().date().isoformat()
    prompt = _build_watchlist_prompt(current_date=current_date, metrics=metrics)
    commentary = await _request_lmstudio_commentary(prompt)

    payload = {
        "symbols": target_symbols,
        "current_date": current_date,
        "model": LMSTUDIO_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comment": commentary,
        "prompt": prompt,
        "metrics": [
            {
                "symbol": item["symbol"],
                "day_change_pct": item["day_change_pct"],
                "return_30d_pct": item["return_30d_pct"],
                "volatility_30d_pct": item["volatility_30d_pct"],
                "day_change_text": item["day_change_text"],
                "return_30d_text": item["return_30d_text"],
                "volatility_30d_text": item["volatility_30d_text"],
            }
            for item in metrics
        ],
    }
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
