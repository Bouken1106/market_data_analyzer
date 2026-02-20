"""Market data and stream routes."""

from __future__ import annotations

import asyncio
import json
import math
import re
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
_WATCHLIST_MAX_COMMENT_LEN = 80
_JSON_DECODER = json.JSONDecoder()
_WATCHLIST_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "watchlist_commentary",
        "schema": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "comment": {"type": "string"},
                        },
                        "required": ["symbol", "comment"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["picks"],
            "additionalProperties": False,
        },
    },
}


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
            "上記の銘柄だけを対象に、特徴的な2銘柄を選ぶ。",
            "必ずJSONのみで返すこと。形式:",
            '{"picks":[{"symbol":"AAPL","comment":"..."} , {"symbol":"NVDA","comment":"..."}]}',
            "制約: symbolは表内の銘柄のみ、commentは日本語1文、簡潔、余計な説明禁止。",
        ]
    )
    return "\n".join(lines)


def _extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    for idx, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed, _ = _JSON_DECODER.raw_decode(raw_text[idx:])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_comment_line(text: str) -> str:
    compact = " ".join(str(text).replace("\r", "\n").split())
    compact = re.sub(r"[*_`>#]+", "", compact).strip()
    return compact[:_WATCHLIST_MAX_COMMENT_LEN].strip()


def _commentary_from_json(raw_text: str, valid_symbols: list[str]) -> str | None:
    payload = _extract_first_json_object(raw_text)
    if not isinstance(payload, dict):
        return None

    picks = payload.get("picks")
    if not isinstance(picks, list):
        return None

    symbol_set = {symbol.upper() for symbol in valid_symbols}
    selected: list[str] = []
    used_symbols: set[str] = set()
    for item in picks:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if symbol not in symbol_set or symbol in used_symbols:
            continue
        comment = _normalize_comment_line(item.get("comment") or "")
        if not comment:
            continue
        selected.append(f"{symbol}: {comment}")
        used_symbols.add(symbol)
        if len(selected) >= 2:
            break

    if len(selected) < 2:
        return None
    return "\n".join(selected)


def _fallback_commentary(raw_text: str, valid_symbols: list[str]) -> str:
    symbol_set = {symbol.upper() for symbol in valid_symbols}
    lines = [line.strip() for line in str(raw_text).replace("\r", "\n").split("\n")]

    accepted: list[str] = []
    used_symbols: set[str] = set()
    for raw_line in lines:
        if not raw_line:
            continue
        lowered = raw_line.lower()
        if lowered.startswith(("alright", "i need to", "let me", "first,", "first ", "second,", "third,")):
            continue

        tokens = re.findall(r"[A-Z]{1,6}(?:\.[A-Z]{1,5})?", raw_line.upper())
        symbol = ""
        for token in tokens:
            if token in symbol_set and token not in used_symbols:
                symbol = token
                break
        if not symbol:
            continue

        line = _normalize_comment_line(raw_line)
        if not line:
            continue
        accepted.append(f"{symbol}: {line}")
        used_symbols.add(symbol)
        if len(accepted) >= 2:
            break

    if len(accepted) >= 2:
        return "\n".join(accepted)
    if accepted:
        return accepted[0]
    return "コメントの整形に失敗しました。再実行してください。"


async def _request_lmstudio_commentary(prompt: str, valid_symbols: list[str]) -> tuple[str, str]:
    headers = {"Content-Type": "application/json"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"

    timeout = httpx.Timeout(LMSTUDIO_TIMEOUT_SEC, connect=min(10.0, LMSTUDIO_TIMEOUT_SEC))
    base_messages = [
        {
            "role": "system",
            "content": (
                "あなたは株式ウォッチリスト要約アシスタントです。"
                "思考過程や分析手順は一切出力しない。"
                "常にJSONのみを返す。"
                '形式は {"picks":[{"symbol":"...","comment":"..."},{"symbol":"...","comment":"..."}]} のみ。'
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    async def _chat(
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, int, str | None, str]:
        payload: dict[str, Any] = {
            "model": LMSTUDIO_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(LMSTUDIO_CHAT_COMPLETIONS_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"LM Studio request failed: {exc}") from exc

        try:
            result = response.json()
        except ValueError:
            result = {}

        error_message: str | None = None
        if isinstance(result, dict):
            error = result.get("error")
            if isinstance(error, dict):
                error_message = str(error.get("message") or "").strip() or None
            elif isinstance(error, str):
                error_message = error.strip() or None
            if error_message is None:
                error_message = str(result.get("detail") or "").strip() or None

        if response.status_code >= 400:
            return "", response.status_code, error_message, LMSTUDIO_MODEL

        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="LM Studio returned an invalid response format.")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise HTTPException(status_code=502, detail="LM Studio response does not include choices.")

        model_name = str(result.get("model") or "").strip() or LMSTUDIO_MODEL
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        return str(content or "").strip(), response.status_code, None, model_name

    raw_commentary, status_code, error_detail, used_model = await _chat(
        base_messages,
        max_tokens=320,
        response_format=_WATCHLIST_RESPONSE_FORMAT,
    )
    if status_code >= 400:
        # Some LM Studio model backends ignore or reject response_format; retry without it.
        if status_code in {400, 404, 422}:
            raw_commentary, status_code, error_detail, used_model = await _chat(
                base_messages,
                max_tokens=320,
                response_format=None,
            )
        if status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"LM Studio error: {error_detail or f'HTTP {status_code}'}",
            )

    if not raw_commentary:
        raise HTTPException(status_code=502, detail="LM Studio returned an empty commentary.")

    commentary = _commentary_from_json(raw_commentary, valid_symbols)
    if commentary:
        return commentary, used_model

    repair_messages = [
        {
            "role": "system",
            "content": (
                "あなたはJSON整形器です。"
                "入力文を要約し、指定形式JSONのみを返す。"
                "説明文や前置きは出力禁止。"
            ),
        },
        {
            "role": "user",
            "content": (
                "有効な銘柄: "
                + ",".join(valid_symbols)
                + "\n次の文章を2銘柄の短評JSONへ整形してください。"
                + '\n形式: {"picks":[{"symbol":"...","comment":"..."},{"symbol":"...","comment":"..."}]}'
                + "\n文章:\n"
                + raw_commentary
            ),
        },
    ]
    repaired_commentary, repair_status, repair_error, repair_model = await _chat(
        repair_messages,
        max_tokens=220,
        response_format=_WATCHLIST_RESPONSE_FORMAT,
    )
    if repair_status >= 400 and repair_status in {400, 404, 422}:
        repaired_commentary, repair_status, repair_error, repair_model = await _chat(
            repair_messages,
            max_tokens=220,
            response_format=None,
        )
    if repair_status < 400:
        repaired = _commentary_from_json(repaired_commentary, valid_symbols)
        if repaired:
            return repaired, repair_model

    if repair_status >= 400 and not raw_commentary:
        raise HTTPException(
            status_code=502,
            detail=f"LM Studio error: {repair_error or f'HTTP {repair_status}'}",
        )

    return _fallback_commentary(raw_commentary, valid_symbols), used_model


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
    commentary, used_model = await _request_lmstudio_commentary(prompt, target_symbols)

    payload = {
        "symbols": target_symbols,
        "current_date": current_date,
        "model": used_model,
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
