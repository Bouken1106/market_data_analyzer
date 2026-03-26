"""ML forecast and job routes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..config import LOGGER, ML_HISTORY_DEFAULT_MONTHS
from ..ml.catalog import ML_MODEL_CATALOG
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
    StockMlPageQueryRequest,
)
from ..utils import normalize_ml_history_months, normalize_symbols, ok_json_response, utc_now_iso
from .deps import HubDep, MlJobStoreDep, StockMlPageStoreDep
from .ml_support import (
    StockMlPageContext,
    backtest_payload,
    build_job_response_payload,
    normalize_job_symbol,
    ops_status_payload,
    prediction_daily_payload,
    stock_model_registry_payload,
    training_job_status_payload,
)

router = APIRouter()
StockMlPageQueryDep = Annotated[StockMlPageQueryRequest, Depends()]


def _stock_page_context(
    request: StockMlPageQueryRequest | StockMlPageActionRequest | StockMlModelAdoptionRequest,
    *,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> StockMlPageContext:
    return StockMlPageContext.from_request(
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        request=request,
    )


async def _stock_page_snapshot(
    request: StockMlPageQueryRequest | StockMlPageActionRequest | StockMlModelAdoptionRequest,
    *,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> dict[str, Any]:
    return await _stock_page_context(
        request,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
    ).snapshot()


async def _stock_page_snapshot_response(
    request: StockMlPageQueryRequest | StockMlPageActionRequest | StockMlModelAdoptionRequest,
    *,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> JSONResponse:
    payload = await _stock_page_snapshot(
        request,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
    )
    if transform is not None:
        payload = transform(payload)
    return ok_json_response(**payload)


async def _stock_page_action_response(
    request: StockMlPageActionRequest | StockMlModelAdoptionRequest,
    *,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    action: str,
    **extra: Any,
) -> JSONResponse:
    payload = await _stock_page_context(
        request,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
    ).call_named(action, **extra)
    return ok_json_response(**payload)


def _queue_stock_page_job(
    *,
    kind: str,
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
    include_config_hash: bool = False,
) -> JSONResponse:
    job_id = ml_job_store.create(kind=kind, symbol="JP")
    asyncio.create_task(
        _run_stock_page_job(
            job_id=job_id,
            kind=kind,
            req=req,
            hub=hub,
            stock_ml_page_store=stock_ml_page_store,
            ml_job_store=ml_job_store,
        )
    )

    payload: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "status": "QUEUED",
        "status_raw": "queued",
    }
    if include_config_hash:
        payload["config_hash"] = req.stock_page_config_hash()
    else:
        payload["accepted_at"] = utc_now_iso()
    return ok_json_response(**payload)


async def _run_stock_page_job(
    *,
    job_id: str,
    kind: str,
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> None:
    context = _stock_page_context(req, hub=hub, stock_ml_page_store=stock_ml_page_store)
    ml_job_store.update(job_id, status="running", progress=15, message="ジョブを実行しています。")
    stage_name = "prepare"
    try:
        if kind == "stock_prediction_run":
            stage_name = "run_inference"
            snapshot = await context.call_named(
                "run_inference",
                confirm_regenerate=req.confirm_regenerate,
            )
            result = prediction_daily_payload(snapshot)
        elif kind == "stock_training_job":
            stage_name = "create_training_job"
            snapshot = await context.call_named("create_training_job")
            result = {
                "metrics": snapshot.get("train", {}).get("compare_rows", []),
                "summary": snapshot.get("train", {}).get("summary_cards", []),
                "folds": snapshot.get("train", {}).get("folds", []),
                "logs": snapshot.get("ops", {}).get("logs", []),
            }
        elif kind == "stock_backtest_run":
            stage_name = "run_backtest"
            snapshot = await context.call_named("run_backtest")
            result = backtest_payload(snapshot)
        else:
            raise HTTPException(status_code=400, detail="Unsupported stock ML job kind.")
        ml_job_store.complete(job_id, result)
    except HTTPException as exc:
        detail = str(exc.detail)
        ml_job_store.fail(
            job_id,
            detail,
            error_detail={
                "stage_name": stage_name,
                "error_code": f"HTTP_{exc.status_code}",
                "message": detail,
                "retryable": exc.status_code >= 500,
            },
            message=detail,
        )
    except Exception as exc:  # pragma: no cover - defensive background task guard
        LOGGER.exception("Stock ML page job failed: %s", exc)
        ml_job_store.fail(
            job_id,
            str(exc),
            error_detail={
                "stage_name": stage_name,
                "error_code": "UNEXPECTED_ERROR",
                "message": str(exc),
                "retryable": True,
            },
            message="内部エラーにより失敗しました。",
        )


@router.get("/api/ml/models")
async def ml_models(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    query: StockMlPageQueryDep,
    scope: str | None = None,
) -> JSONResponse:
    normalized_scope = str(scope or "").strip().lower()
    if normalized_scope not in {"stock", "stock-page", "registry"}:
        return ok_json_response(models=ML_MODEL_CATALOG)
    return await _stock_page_snapshot_response(
        query,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        transform=stock_model_registry_payload,
    )


@router.get("/api/ml/predictions/daily")
async def stock_ml_prediction_daily(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    query: StockMlPageQueryDep,
) -> JSONResponse:
    return await _stock_page_snapshot_response(
        query,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        transform=prediction_daily_payload,
    )


@router.post("/api/ml/predictions/run")
async def stock_ml_prediction_run(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    context = _stock_page_context(req, hub=hub, stock_ml_page_store=stock_ml_page_store)
    snapshot = await context.snapshot()
    context.service._ensure_action_allowed(snapshot, "run_inference")
    generation = context.service._prediction_run_payload(snapshot)
    existing = stock_ml_page_store.find_prediction_run(generation_key=generation["generation_key"])
    if existing is not None and not req.confirm_regenerate:
        generated_at = existing.get("generated_at")
        raise HTTPException(
            status_code=409,
            detail=(
                "同一条件の prediction_daily は既に生成済みです。"
                f" 最終生成: {generated_at}。再生成する場合は確認してください。"
            ),
        )
    return _queue_stock_page_job(
        kind="stock_prediction_run",
        req=req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        ml_job_store=ml_job_store,
    )


@router.post("/api/ml/training/jobs")
async def stock_ml_training_job_create(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    context = _stock_page_context(req, hub=hub, stock_ml_page_store=stock_ml_page_store)
    snapshot = await context.snapshot()
    context.service._ensure_action_allowed(snapshot, "create_training_job")
    return _queue_stock_page_job(
        kind="stock_training_job",
        req=req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        ml_job_store=ml_job_store,
        include_config_hash=True,
    )


@router.get("/api/ml/training/jobs/{job_id}")
async def stock_ml_training_job_status(job_id: str, ml_job_store: MlJobStoreDep) -> JSONResponse:
    payload = ml_job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**training_job_status_payload(payload))


@router.get("/api/ml/backtests")
async def stock_ml_backtests(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    query: StockMlPageQueryDep,
) -> JSONResponse:
    return await _stock_page_snapshot_response(
        query,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        transform=backtest_payload,
    )


@router.post("/api/ml/backtests/run")
async def stock_ml_backtests_run(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    context = _stock_page_context(req, hub=hub, stock_ml_page_store=stock_ml_page_store)
    snapshot = await context.snapshot()
    context.service._ensure_action_allowed(snapshot, "run_backtest")
    return _queue_stock_page_job(
        kind="stock_backtest_run",
        req=req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        ml_job_store=ml_job_store,
    )


@router.post("/api/ml/models/{model_version}/adopt")
async def stock_ml_model_adopt_alias(
    model_version: str,
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    payload = await _stock_page_context(
        req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
    ).call_named("adopt_model", model_version=model_version)
    return ok_json_response(
        result="adopted",
        adopted_version=model_version,
        models=payload.get("models", {}).get("rows", []),
    )


@router.get("/api/ml/ops/status")
async def stock_ml_ops_status(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    query: StockMlPageQueryDep,
) -> JSONResponse:
    return await _stock_page_snapshot_response(
        query,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        transform=ops_status_payload,
    )


@router.get("/api/ml/stock-page")
async def stock_ml_page_snapshot(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    query: StockMlPageQueryDep,
) -> JSONResponse:
    return await _stock_page_snapshot_response(
        query,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
    )


@router.post("/api/ml/stock-page/actions/refresh")
async def stock_ml_page_refresh(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    return await _stock_page_action_response(
        req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        action="refresh_data",
    )


@router.post("/api/ml/stock-page/actions/run-inference")
async def stock_ml_page_run_inference(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    return await _stock_page_action_response(
        req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        action="run_inference",
        confirm_regenerate=req.confirm_regenerate,
    )


@router.post("/api/ml/stock-page/actions/create-training-job")
async def stock_ml_page_create_training_job(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    return await _stock_page_action_response(
        req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        action="create_training_job",
    )


@router.post("/api/ml/stock-page/actions/adopt-model")
async def stock_ml_page_adopt_model(
    req: StockMlModelAdoptionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    return await _stock_page_action_response(
        req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        action="adopt_model",
        model_version=req.model_version,
    )


@router.post("/api/ml/stock-page/actions/export-csv")
async def stock_ml_page_export_csv(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    return await _stock_page_action_response(
        req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        action="export_csv",
        search_query=req.search_query,
    )


@router.post("/api/ml/stock-page/actions/export-report")
async def stock_ml_page_export_report(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    return await _stock_page_action_response(
        req,
        hub=hub,
        stock_ml_page_store=stock_ml_page_store,
        action="export_report",
        search_query=req.search_query,
    )


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
    req.symbol = normalize_job_symbol(req.symbol)
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
    req.symbol = normalize_job_symbol(req.symbol)
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
    return ok_json_response(**build_job_response_payload(payload, include_status_code=True))


@router.post("/api/ml/jobs/{job_id}/cancel")
async def ml_job_cancel(job_id: str, ml_job_store: MlJobStoreDep) -> JSONResponse:
    payload = ml_job_store.request_cancel(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**build_job_response_payload(payload, include_status_code=True))
