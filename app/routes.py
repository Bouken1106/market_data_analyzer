"""FastAPI route definitions for Market Data Analyzer."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import (
    HISTORICAL_DEFAULT_YEARS,
    MAX_BASIC_SYMBOLS,
    ML_HISTORY_DEFAULT_MONTHS,
)
from .ml.catalog import ML_MODEL_CATALOG
from .ml.pipelines import (
    _cancel_check_for_job,
    _normalize_ml_history_months,
    _parse_compare_models,
    _progress_callback_for_job,
    _run_ml_comparison_job,
    _run_patchtst_job,
    _run_patchtst_pipeline,
    _run_quantile_lstm_job,
    _run_quantile_lstm_pipeline,
)
from .models import (
    MlComparisonJobRequest,
    PaperPortfolioResetRequest,
    PaperTradeRequest,
    QuantileLstmJobRequest,
    SymbolUpdateRequest,
)
from .utils import normalize_symbols, ok_json_response


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# --- Hub / job store singletons (injected by main.py) ---

_hub: Any = None
_symbol_catalog_store: Any = None
_ml_job_store: Any = None
_paper_portfolio_store: Any = None


def init_routes(hub: Any, symbol_catalog_store: Any, ml_job_store: Any, paper_portfolio_store: Any) -> None:
    """Called from main.py to inject singletons."""
    global _hub, _symbol_catalog_store, _ml_job_store, _paper_portfolio_store
    _hub = hub
    _symbol_catalog_store = symbol_catalog_store
    _ml_job_store = ml_job_store
    _paper_portfolio_store = paper_portfolio_store


def _to_valid_price(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


async def _resolve_trade_price(symbol: str, explicit_price: float | None) -> tuple[float, str]:
    if explicit_price is not None:
        parsed = _to_valid_price(explicit_price)
        if parsed is None:
            raise HTTPException(status_code=400, detail="price must be greater than 0.")
        return parsed, "manual"

    rows = await _hub.current_rows([symbol])
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="Current market price is unavailable. Set price manually.",
        )
    latest = rows[0] if isinstance(rows[0], dict) else {}
    parsed = _to_valid_price(latest.get("price"))
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail="Current market price is unavailable. Set price manually.",
        )
    return parsed, "market"


async def _paper_portfolio_payload() -> dict[str, Any]:
    state = await _paper_portfolio_store.get_state()
    positions_raw = state.get("positions") if isinstance(state, dict) else {}
    if not isinstance(positions_raw, dict):
        positions_raw = {}

    symbols = sorted(str(symbol).upper().strip() for symbol in positions_raw.keys())
    rows = await _hub.current_rows(symbols) if symbols else []
    price_map: dict[str, float | None] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        price_map[symbol] = _to_valid_price(row.get("price"))

    positions: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    has_market_value = False

    for symbol in symbols:
        item = positions_raw.get(symbol, {})
        quantity = _to_valid_price(item.get("quantity")) or 0.0
        avg_cost = _to_valid_price(item.get("avg_cost")) or 0.0
        if quantity <= 0:
            continue
        cost_basis = quantity * avg_cost
        total_cost_basis += cost_basis
        last_price = price_map.get(symbol)
        market_value = None
        unrealized_pnl = None
        unrealized_pnl_pct = None
        if last_price is not None:
            has_market_value = True
            market_value = quantity * last_price
            total_market_value += market_value
            unrealized_pnl = market_value - cost_basis
            if cost_basis > 0:
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100

        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "cost_basis": cost_basis,
                "last_price": last_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "weight": None,
            }
        )

    if total_market_value > 0:
        for item in positions:
            market_value = item.get("market_value")
            if isinstance(market_value, (int, float)):
                item["weight"] = (float(market_value) / total_market_value) * 100

    cash = _to_valid_price(state.get("cash")) or 0.0
    initial_cash = _to_valid_price(state.get("initial_cash")) or cash
    equity = cash + total_market_value
    unrealized_total = total_market_value - total_cost_basis if has_market_value else None
    total_return_pct = ((equity - initial_cash) / initial_cash * 100) if initial_cash > 0 else None

    trades = state.get("trades") if isinstance(state.get("trades"), list) else []
    recent_trades = []
    for item in reversed(trades[-50:]):
        if isinstance(item, dict):
            recent_trades.append(dict(item))

    return {
        "initial_cash": initial_cash,
        "cash": cash,
        "market_value": total_market_value,
        "equity": equity,
        "cost_basis": total_cost_basis,
        "unrealized_pnl": unrealized_total,
        "total_return_pct": total_return_pct,
        "positions": positions,
        "recent_trades": recent_trades,
        "trade_count": len(trades),
        "updated_at": state.get("updated_at"),
    }


# --- Page routes ---

@router.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/compare-lab", include_in_schema=False)
async def compare_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "compare_lab.html")


@router.get("/ml-lab", include_in_schema=False)
async def ml_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "ml_lab.html")


@router.get("/strategy-lab", include_in_schema=False)
async def strategy_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "strategy_lab.html")


@router.get("/historical/{symbol}", include_in_schema=False)
async def historical_page(symbol: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "historical.html")


# --- API routes ---

@router.get("/api/snapshot")
async def snapshot() -> JSONResponse:
    payload = await _hub.snapshot_payload()
    return JSONResponse(payload)


@router.post("/api/symbols")
async def update_symbols(req: SymbolUpdateRequest) -> JSONResponse:
    symbols = normalize_symbols(req.symbols)
    await _hub.set_symbols(symbols)
    rows = await _hub.current_rows(symbols)
    return ok_json_response(
        symbols=symbols,
        status=await _hub.status_payload(),
        rows=rows,
    )


@router.get("/api/credits")
async def credits(refresh: bool = False) -> JSONResponse:
    if refresh:
        status = await _hub.refresh_api_credits()
    else:
        status = await _hub.status_payload()
    note = (
        "refresh=true fetches exact daily remaining credits via /api_usage and consumes 1 API credit."
        if getattr(_hub, "provider", "") in {"twelvedata", "both"}
        else "Current provider does not expose Twelve Data /api_usage credits."
    )
    return ok_json_response(
        status=status,
        note=note,
    )


@router.get("/api/portfolio")
async def paper_portfolio() -> JSONResponse:
    payload = await _paper_portfolio_payload()
    return ok_json_response(**payload)


@router.post("/api/portfolio/trades")
async def paper_trade(req: PaperTradeRequest) -> JSONResponse:
    symbols = normalize_symbols([req.symbol])
    if not symbols:
        raise HTTPException(status_code=400, detail="Invalid symbol format.")
    symbol = symbols[0]
    side = str(req.side or "").lower().strip()

    try:
        quantity = float(req.quantity)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="quantity must be greater than 0.") from None
    if not math.isfinite(quantity) or quantity <= 0:
        raise HTTPException(status_code=400, detail="quantity must be greater than 0.")

    execution_price, execution_source = await _resolve_trade_price(symbol, req.price)
    try:
        trade = await _paper_portfolio_store.apply_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    payload = await _paper_portfolio_payload()
    return ok_json_response(
        trade={**trade, "execution_source": execution_source},
        **payload,
    )


@router.post("/api/portfolio/reset")
async def paper_portfolio_reset(req: PaperPortfolioResetRequest) -> JSONResponse:
    if req.initial_cash is not None:
        try:
            initial_cash = float(req.initial_cash)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="initial_cash must be greater than 0.") from None
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise HTTPException(status_code=400, detail="initial_cash must be greater than 0.")
    await _paper_portfolio_store.reset(initial_cash=req.initial_cash)
    payload = await _paper_portfolio_payload()
    return ok_json_response(**payload)


@router.get("/api/symbol-catalog")
async def symbol_catalog(refresh: bool = False) -> JSONResponse:
    payload = await _symbol_catalog_store.get_catalog(refresh=refresh)
    return ok_json_response(**payload)


@router.get("/api/historical/{symbol}")
async def historical(symbol: str, years: int = HISTORICAL_DEFAULT_YEARS, refresh: bool = False) -> JSONResponse:
    payload = await _hub.historical_payload(symbol=symbol, years=years, refresh=refresh)
    return ok_json_response(**payload)


@router.get("/api/security-overview/{symbol}")
async def security_overview(
    symbol: str,
    refresh: bool = False,
    include_intraday: bool = True,
    include_market: bool = True,
    include_qqq: bool = True,
) -> JSONResponse:
    payload = await _hub.security_overview_payload(
        symbol=symbol,
        refresh=refresh,
        include_intraday=include_intraday,
        include_market=include_market,
        include_qqq=include_qqq,
    )
    return ok_json_response(**payload)


@router.get("/api/security-overview/{symbol}/intraday")
async def security_overview_intraday(symbol: str, refresh: bool = False) -> JSONResponse:
    payload = await _hub.security_overview_payload(
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
async def clear_security_overview_cache(symbol: str) -> JSONResponse:
    payload = await _hub.clear_symbol_overview_cache(symbol=symbol)
    return ok_json_response(**payload)


@router.get("/api/fmp-reference/{symbol}")
async def fmp_reference(symbol: str, refresh: bool = False, cache_only: bool = False) -> JSONResponse:
    payload = await _hub.fmp_reference_payload(symbol=symbol, refresh=refresh, cache_only=cache_only)
    return ok_json_response(**payload)


@router.post("/api/fmp-reference/{symbol}/clear-cache")
async def clear_fmp_reference_cache(symbol: str) -> JSONResponse:
    payload = await _hub.clear_fmp_reference_cache(symbol=symbol)
    return ok_json_response(**payload)


@router.get("/api/ml/models")
async def ml_models() -> JSONResponse:
    return ok_json_response(models=ML_MODEL_CATALOG)


@router.get("/api/ml/quantile-lstm")
async def quantile_lstm_forecast(
    symbol: str,
    months: int = ML_HISTORY_DEFAULT_MONTHS,
    sequence_length: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 80,
    patience: int = 10,
    representative_days: int = 5,
    seed: int = 42,
    refresh: bool = False,
) -> JSONResponse:
    payload = await _run_quantile_lstm_pipeline(
        symbol=symbol,
        months=months,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        representative_days=representative_days,
        seed=seed,
        refresh=refresh,
    )
    return ok_json_response(**payload)


@router.get("/api/ml/patchtst")
async def patchtst_forecast(
    symbol: str,
    months: int = ML_HISTORY_DEFAULT_MONTHS,
    sequence_length: int = 256,
    hidden_size: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 40,
    patience: int = 8,
    representative_days: int = 5,
    seed: int = 42,
    refresh: bool = False,
) -> JSONResponse:
    payload = await _run_patchtst_pipeline(
        symbol=symbol,
        months=months,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        representative_days=representative_days,
        seed=seed,
        refresh=refresh,
    )
    return ok_json_response(**payload)


@router.post("/api/ml/quantile-lstm/jobs")
async def start_quantile_lstm_job(req: QuantileLstmJobRequest) -> JSONResponse:
    symbol = normalize_symbols([req.symbol])
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbolを入力してください。")
    req.symbol = symbol[0]
    req.months = _normalize_ml_history_months(req.months)

    job_id = _ml_job_store.create(kind="quantile_lstm", symbol=req.symbol)
    asyncio.create_task(_run_quantile_lstm_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@router.post("/api/ml/patchtst/jobs")
async def start_patchtst_job(req: QuantileLstmJobRequest) -> JSONResponse:
    symbol = normalize_symbols([req.symbol])
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbolを入力してください。")
    req.symbol = symbol[0]
    req.months = _normalize_ml_history_months(req.months)

    job_id = _ml_job_store.create(kind="patchtst_quantile", symbol=req.symbol)
    asyncio.create_task(_run_patchtst_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@router.post("/api/ml/compare/jobs")
async def start_ml_compare_job(req: MlComparisonJobRequest) -> JSONResponse:
    symbols = normalize_symbols(req.symbols) if str(req.symbols or "").strip() else []
    if not symbols:
        from .ml.catalog import ML_COMPARE_DEFAULT_SYMBOLS
        symbols = ML_COMPARE_DEFAULT_SYMBOLS
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one valid symbol is required.")
    req.symbols = ",".join(symbols)
    req.months = _normalize_ml_history_months(req.months)

    selected_models = _parse_compare_models(req.models)
    if not selected_models:
        raise HTTPException(status_code=400, detail="At least one valid model is required.")
    req.models = ",".join(selected_models)

    job_id = _ml_job_store.create(kind="ml_compare", symbol="MULTI")
    asyncio.create_task(_run_ml_comparison_job(job_id, req))
    return ok_json_response(job_id=job_id, status="queued")


@router.get("/api/ml/jobs/{job_id}")
async def ml_job_status(job_id: str) -> JSONResponse:
    payload = _ml_job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**payload)


@router.post("/api/ml/jobs/{job_id}/cancel")
async def ml_job_cancel(job_id: str) -> JSONResponse:
    payload = _ml_job_store.request_cancel(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**payload)


@router.get("/api/sparkline")
async def sparkline(symbols: str, refresh: bool = False) -> JSONResponse:
    target_symbols = normalize_symbols(symbols)
    if not target_symbols:
        raise HTTPException(status_code=400, detail="At least one valid symbol is required.")
    if len(target_symbols) > MAX_BASIC_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"You can request up to {MAX_BASIC_SYMBOLS} symbols at once.")

    items = await _hub.sparkline_payload(target_symbols, refresh=refresh)
    return ok_json_response(
        symbols=target_symbols,
        items=items,
    )


@router.get("/api/stream")
async def stream(request: Request) -> StreamingResponse:
    queue = _hub.register_listener()

    async def event_generator():
        initial_payload = await _hub.snapshot_payload()
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
            _hub.unregister_listener(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
