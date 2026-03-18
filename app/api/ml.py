"""ML forecast and job routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..config import ML_HISTORY_DEFAULT_MONTHS
from ..ml.catalog import ML_MODEL_CATALOG
from ..ml.stock_page import StockMlPageService
from ..ml.pipelines import (
    _parse_compare_models,
    _run_ml_comparison_job,
    _run_patchtst_job,
    _run_patchtst_pipeline,
    _run_quantile_lstm_job,
    _run_quantile_lstm_pipeline,
)
from ..models import (
    MlComparisonJobRequest,
    QuantileLstmJobRequest,
    StockMlModelAdoptionRequest,
    StockMlPageActionRequest,
)
from ..utils import normalize_ml_history_months, normalize_symbols, ok_json_response
from .deps import HubDep, MlJobStoreDep, StockMlPageStoreDep

router = APIRouter()


def _stock_ml_page_service(hub: HubDep, stock_ml_page_store: StockMlPageStoreDep) -> StockMlPageService:
    return StockMlPageService(
        full_daily_history_store=hub.full_daily_history_store,
        page_store=stock_ml_page_store,
    )


def _normalize_job_symbol(raw_symbol: str) -> str:
    symbols = normalize_symbols([raw_symbol])
    if not symbols:
        raise HTTPException(status_code=400, detail="Symbolを入力してください。")
    return symbols[0]


@router.get("/api/ml/models")
async def ml_models() -> JSONResponse:
    return ok_json_response(models=ML_MODEL_CATALOG)


@router.get("/api/ml/stock-page")
async def stock_ml_page_snapshot(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    prediction_date: str | None = None,
    universe_filter: str = "jp_large_cap_stooq_v1",
    model_family: str = "LightGBM Classifier",
    feature_set: str = "base_v1",
    cost_buffer: float = 0.0,
    run_note: str = "",
    refresh: bool = False,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.build_snapshot(
        prediction_date=prediction_date,
        universe_filter=universe_filter,
        model_family=model_family,
        feature_set=feature_set,
        cost_buffer=cost_buffer,
        run_note=run_note,
        refresh=refresh,
    )
    return ok_json_response(**payload)


@router.post("/api/ml/stock-page/actions/refresh")
async def stock_ml_page_refresh(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.refresh_data(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        run_note=req.run_note,
    )
    return ok_json_response(**payload)


@router.post("/api/ml/stock-page/actions/run-inference")
async def stock_ml_page_run_inference(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.run_inference(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        run_note=req.run_note,
        refresh=req.refresh,
    )
    return ok_json_response(**payload)


@router.post("/api/ml/stock-page/actions/create-training-job")
async def stock_ml_page_create_training_job(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.create_training_job(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        run_note=req.run_note,
        refresh=req.refresh,
    )
    return ok_json_response(**payload)


@router.post("/api/ml/stock-page/actions/adopt-model")
async def stock_ml_page_adopt_model(
    req: StockMlModelAdoptionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.adopt_model(
        model_version=req.model_version,
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        run_note=req.run_note,
        refresh=req.refresh,
    )
    return ok_json_response(**payload)


@router.get("/api/ml/quantile-lstm")
async def quantile_lstm_forecast(
    hub: HubDep,
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
        hub=hub,
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
    hub: HubDep,
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
        hub=hub,
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
async def start_quantile_lstm_job(
    req: QuantileLstmJobRequest,
    hub: HubDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    req.symbol = _normalize_job_symbol(req.symbol)
    req.months = normalize_ml_history_months(req.months)

    job_id = ml_job_store.create(kind="quantile_lstm", symbol=req.symbol)
    asyncio.create_task(_run_quantile_lstm_job(job_id, req, hub=hub, ml_job_store=ml_job_store))
    return ok_json_response(job_id=job_id, status="queued")


@router.post("/api/ml/patchtst/jobs")
async def start_patchtst_job(
    req: QuantileLstmJobRequest,
    hub: HubDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    req.symbol = _normalize_job_symbol(req.symbol)
    req.months = normalize_ml_history_months(req.months)

    job_id = ml_job_store.create(kind="patchtst_quantile", symbol=req.symbol)
    asyncio.create_task(_run_patchtst_job(job_id, req, hub=hub, ml_job_store=ml_job_store))
    return ok_json_response(job_id=job_id, status="queued")


@router.post("/api/ml/compare/jobs")
async def start_ml_compare_job(
    req: MlComparisonJobRequest,
    hub: HubDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    symbols = normalize_symbols(req.symbols) if str(req.symbols or "").strip() else []
    if not symbols:
        from ..ml.catalog import ML_COMPARE_DEFAULT_SYMBOLS

        symbols = ML_COMPARE_DEFAULT_SYMBOLS
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one valid symbol is required.")
    req.symbols = ",".join(symbols)
    req.months = normalize_ml_history_months(req.months)

    selected_models = _parse_compare_models(req.models)
    if not selected_models:
        raise HTTPException(status_code=400, detail="At least one valid model is required.")
    req.models = ",".join(selected_models)

    job_id = ml_job_store.create(kind="ml_compare", symbol="MULTI")
    asyncio.create_task(_run_ml_comparison_job(job_id, req, hub=hub, ml_job_store=ml_job_store))
    return ok_json_response(job_id=job_id, status="queued")


@router.get("/api/ml/jobs/{job_id}")
async def ml_job_status(job_id: str, ml_job_store: MlJobStoreDep) -> JSONResponse:
    payload = ml_job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**payload)


@router.post("/api/ml/jobs/{job_id}/cancel")
async def ml_job_cancel(job_id: str, ml_job_store: MlJobStoreDep) -> JSONResponse:
    payload = ml_job_store.request_cancel(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**payload)
