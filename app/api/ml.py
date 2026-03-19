"""ML forecast and job routes."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..config import LOGGER, ML_HISTORY_DEFAULT_MONTHS
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
from ..utils import normalize_ml_history_months, normalize_symbols, ok_json_response, utc_now_iso
from .deps import HubDep, MlJobStoreDep, StockMlPageStoreDep

router = APIRouter()


def _spec_job_status(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    mapping = {
        "queued": "QUEUED",
        "running": "RUNNING",
        "cancelling": "RUNNING",
        "completed": "SUCCEEDED",
        "failed": "FAILED",
        "cancelled": "CANCELLED",
    }
    return mapping.get(normalized, "UNKNOWN")


def _job_response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["status_code"] = _spec_job_status(payload.get("status"))
    return result


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


def _stock_page_kwargs(
    *,
    prediction_date: str | None,
    universe_filter: str,
    model_family: str,
    feature_set: str,
    cost_buffer: float,
    train_window_months: int,
    gap_days: int,
    valid_window_months: int,
    random_seed: int,
    train_note: str,
    run_note: str,
    refresh: bool = False,
) -> dict[str, Any]:
    return {
        "prediction_date": prediction_date,
        "universe_filter": universe_filter,
        "model_family": model_family,
        "feature_set": feature_set,
        "cost_buffer": cost_buffer,
        "train_window_months": train_window_months,
        "gap_days": gap_days,
        "valid_window_months": valid_window_months,
        "random_seed": random_seed,
        "train_note": train_note,
        "run_note": run_note,
        "refresh": refresh,
    }


def _prediction_daily_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    dashboard = snapshot.get("dashboard", {})
    filters = snapshot.get("filters", {})
    models = snapshot.get("models", {})
    default_versions = models.get("default_versions", {})
    model_version = default_versions.get(filters.get("model_family"), "")
    rows = []
    for item in dashboard.get("rows", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "prediction_date": dashboard.get("prediction_date"),
                "target_date": dashboard.get("target_date"),
                "code": item.get("code"),
                "score_cls": item.get("score_cls"),
                "prob_up": item.get("prob_up"),
                "score_rank": item.get("score_rank"),
                "expected_return": item.get("expected_return"),
                "model_version": model_version,
                "feature_version": dashboard.get("feature_version"),
                "data_version": dashboard.get("data_version"),
                "warnings": item.get("warnings", []),
            }
        )
    return {
        "prediction_date": dashboard.get("prediction_date"),
        "target_date": dashboard.get("target_date"),
        "model_version": model_version,
        "feature_version": dashboard.get("feature_version"),
        "data_version": dashboard.get("data_version"),
        "rows": rows,
    }


def _training_job_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") or {}
    error_detail = payload.get("error_detail") or {}
    raw_status = str(payload.get("status") or "")
    return {
        "job_id": payload.get("job_id"),
        "status": _spec_job_status(raw_status),
        "status_raw": raw_status,
        "progress": payload.get("progress"),
        "message": payload.get("message"),
        "metrics": result.get("metrics"),
        "summary": result.get("summary"),
        "folds": result.get("folds"),
        "logs": result.get("logs"),
        "error": payload.get("error"),
        "stage_name": error_detail.get("stage_name"),
        "error_code": error_detail.get("error_code"),
        "retryable": error_detail.get("retryable"),
        "updated_at": payload.get("updated_at"),
    }


def _ml_job_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    error_detail = payload.get("error_detail") or {}
    raw_status = str(payload.get("status") or "")
    return {
        "job_id": payload.get("job_id"),
        "kind": payload.get("kind"),
        "symbol": payload.get("symbol"),
        "status": _spec_job_status(raw_status),
        "status_raw": raw_status,
        "progress": payload.get("progress"),
        "message": payload.get("message"),
        "result": payload.get("result"),
        "error": payload.get("error"),
        "stage_name": error_detail.get("stage_name"),
        "error_code": error_detail.get("error_code"),
        "retryable": error_detail.get("retryable"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def _backtest_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    backtest = snapshot.get("backtest", {})
    return {
        "summary": backtest.get("summary_cards", []),
        "compare": backtest.get("compare_rows", []),
        "curve": {
            "labels": backtest.get("equity_labels", []),
            "series": backtest.get("equity_series", []),
        },
        "monthly_returns": backtest.get("monthly_returns", []),
        "daily_return_distribution": backtest.get("daily_return_distribution", {}),
        "exceptions": backtest.get("exceptions", []),
    }


def _ops_status_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    ops = snapshot.get("ops", {})
    return {
        "pipeline_states": ops.get("pipeline", []),
        "summary": ops.get("summary_cards", []),
        "coverage_breakdown": ops.get("coverage_breakdown", []),
        "score_drift_distribution": ops.get("score_drift_distribution", {}),
        "alerts": ops.get("alerts", []),
        "logs": ops.get("logs", []),
    }


def _stock_model_registry_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    models = snapshot.get("models", {})
    return {
        "models": models.get("rows", []),
        "adopted_model_version": models.get("adopted_model_version"),
        "default_versions": models.get("default_versions", {}),
    }


def _stock_page_config_hash(req: StockMlPageActionRequest) -> str:
    payload = {
        "prediction_date": req.prediction_date,
        "universe_filter": req.universe_filter,
        "model_family": req.model_family,
        "feature_set": req.feature_set,
        "cost_buffer": req.cost_buffer,
        "train_window_months": req.train_window_months,
        "gap_days": req.gap_days,
        "valid_window_months": req.valid_window_months,
        "random_seed": req.random_seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest[:12]


async def _run_stock_page_job(
    *,
    job_id: str,
    kind: str,
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> None:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    ml_job_store.update(job_id, status="running", progress=15, message="ジョブを実行しています。")
    stage_name = "prepare"
    kwargs = _stock_page_kwargs(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        run_note=req.run_note,
        refresh=req.refresh,
    )
    try:
        if kind == "stock_prediction_run":
            stage_name = "run_inference"
            snapshot = await service.run_inference(confirm_regenerate=req.confirm_regenerate, **kwargs)
            result = _prediction_daily_payload(snapshot)
        elif kind == "stock_training_job":
            stage_name = "create_training_job"
            snapshot = await service.create_training_job(**kwargs)
            result = {
                "metrics": snapshot.get("train", {}).get("compare_rows", []),
                "summary": snapshot.get("train", {}).get("summary_cards", []),
                "folds": snapshot.get("train", {}).get("folds", []),
                "logs": snapshot.get("ops", {}).get("logs", []),
            }
        elif kind == "stock_backtest_run":
            stage_name = "run_backtest"
            snapshot = await service.build_snapshot(**kwargs)
            service._ensure_action_allowed(snapshot, "run_backtest")
            result = _backtest_payload(snapshot)
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
    scope: str | None = None,
    prediction_date: str | None = None,
    universe_filter: str = "jp_large_cap_stooq_v1",
    model_family: str = "LightGBM Classifier",
    feature_set: str = "base_v1",
    cost_buffer: float = 0.0,
    train_window_months: int = 12,
    gap_days: int = 5,
    valid_window_months: int = 1,
    random_seed: int = 42,
    train_note: str = "",
    run_note: str = "",
    refresh: bool = False,
) -> JSONResponse:
    normalized_scope = str(scope or "").strip().lower()
    if normalized_scope not in {"stock", "stock-page", "registry"}:
        return ok_json_response(models=ML_MODEL_CATALOG)

    service = _stock_ml_page_service(hub, stock_ml_page_store)
    snapshot = await service.build_snapshot(
        prediction_date=prediction_date,
        universe_filter=universe_filter,
        model_family=model_family,
        feature_set=feature_set,
        cost_buffer=cost_buffer,
        train_window_months=train_window_months,
        gap_days=gap_days,
        valid_window_months=valid_window_months,
        random_seed=random_seed,
        train_note=train_note,
        run_note=run_note,
        refresh=refresh,
    )
    return ok_json_response(**_stock_model_registry_payload(snapshot))


@router.get("/api/ml/predictions/daily")
async def stock_ml_prediction_daily(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    prediction_date: str | None = None,
    universe_filter: str = "jp_large_cap_stooq_v1",
    model_family: str = "LightGBM Classifier",
    feature_set: str = "base_v1",
    cost_buffer: float = 0.0,
    train_window_months: int = 12,
    gap_days: int = 5,
    valid_window_months: int = 1,
    random_seed: int = 42,
    train_note: str = "",
    run_note: str = "",
    refresh: bool = False,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    snapshot = await service.build_snapshot(
        prediction_date=prediction_date,
        universe_filter=universe_filter,
        model_family=model_family,
        feature_set=feature_set,
        cost_buffer=cost_buffer,
        train_window_months=train_window_months,
        gap_days=gap_days,
        valid_window_months=valid_window_months,
        random_seed=random_seed,
        train_note=train_note,
        run_note=run_note,
        refresh=refresh,
    )
    return ok_json_response(**_prediction_daily_payload(snapshot))


@router.post("/api/ml/predictions/run")
async def stock_ml_prediction_run(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    snapshot = await service.build_snapshot(**_stock_page_kwargs(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        run_note=req.run_note,
        refresh=req.refresh,
    ))
    service._ensure_action_allowed(snapshot, "run_inference")
    generation = service._prediction_run_payload(snapshot)
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
    job_id = ml_job_store.create(kind="stock_prediction_run", symbol="JP")
    asyncio.create_task(
        _run_stock_page_job(
            job_id=job_id,
            kind="stock_prediction_run",
            req=req,
            hub=hub,
            stock_ml_page_store=stock_ml_page_store,
            ml_job_store=ml_job_store,
        )
    )
    return ok_json_response(
        job_id=job_id,
        kind="stock_prediction_run",
        status="QUEUED",
        status_raw="queued",
        accepted_at=utc_now_iso(),
    )


@router.post("/api/ml/training/jobs")
async def stock_ml_training_job_create(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    snapshot = await service.build_snapshot(**_stock_page_kwargs(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        run_note=req.run_note,
        refresh=req.refresh,
    ))
    service._ensure_action_allowed(snapshot, "create_training_job")
    job_id = ml_job_store.create(kind="stock_training_job", symbol="JP")
    asyncio.create_task(
        _run_stock_page_job(
            job_id=job_id,
            kind="stock_training_job",
            req=req,
            hub=hub,
            stock_ml_page_store=stock_ml_page_store,
            ml_job_store=ml_job_store,
        )
    )
    return ok_json_response(
        job_id=job_id,
        kind="stock_training_job",
        status="QUEUED",
        status_raw="queued",
        config_hash=_stock_page_config_hash(req),
    )


@router.get("/api/ml/training/jobs/{job_id}")
async def stock_ml_training_job_status(job_id: str, ml_job_store: MlJobStoreDep) -> JSONResponse:
    payload = ml_job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**_training_job_status_payload(payload))


@router.get("/api/ml/jobs/{job_id}")
async def ml_job_status(job_id: str, ml_job_store: MlJobStoreDep) -> JSONResponse:
    payload = ml_job_store.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**_ml_job_status_payload(payload))


@router.get("/api/ml/backtests")
async def stock_ml_backtests(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    prediction_date: str | None = None,
    universe_filter: str = "jp_large_cap_stooq_v1",
    model_family: str = "LightGBM Classifier",
    feature_set: str = "base_v1",
    cost_buffer: float = 0.0,
    train_window_months: int = 12,
    gap_days: int = 5,
    valid_window_months: int = 1,
    random_seed: int = 42,
    train_note: str = "",
    run_note: str = "",
    refresh: bool = False,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    snapshot = await service.build_snapshot(
        prediction_date=prediction_date,
        universe_filter=universe_filter,
        model_family=model_family,
        feature_set=feature_set,
        cost_buffer=cost_buffer,
        train_window_months=train_window_months,
        gap_days=gap_days,
        valid_window_months=valid_window_months,
        random_seed=random_seed,
        train_note=train_note,
        run_note=run_note,
        refresh=refresh,
    )
    return ok_json_response(**_backtest_payload(snapshot))


@router.post("/api/ml/backtests/run")
async def stock_ml_backtests_run(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    ml_job_store: MlJobStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    snapshot = await service.build_snapshot(**_stock_page_kwargs(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        run_note=req.run_note,
        refresh=req.refresh,
    ))
    service._ensure_action_allowed(snapshot, "run_backtest")
    job_id = ml_job_store.create(kind="stock_backtest_run", symbol="JP")
    asyncio.create_task(
        _run_stock_page_job(
            job_id=job_id,
            kind="stock_backtest_run",
            req=req,
            hub=hub,
            stock_ml_page_store=stock_ml_page_store,
            ml_job_store=ml_job_store,
        )
    )
    return ok_json_response(
        job_id=job_id,
        kind="stock_backtest_run",
        status="QUEUED",
        status_raw="queued",
        accepted_at=utc_now_iso(),
    )


@router.post("/api/ml/models/{model_version}/adopt")
async def stock_ml_model_adopt_alias(
    model_version: str,
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.adopt_model(
        model_version=model_version,
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        run_note=req.run_note,
        refresh=req.refresh,
    )
    return ok_json_response(
        result="adopted",
        adopted_version=model_version,
        models=payload.get("models", {}).get("rows", []),
    )


@router.get("/api/ml/ops/status")
async def stock_ml_ops_status(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    prediction_date: str | None = None,
    universe_filter: str = "jp_large_cap_stooq_v1",
    model_family: str = "LightGBM Classifier",
    feature_set: str = "base_v1",
    cost_buffer: float = 0.0,
    train_window_months: int = 12,
    gap_days: int = 5,
    valid_window_months: int = 1,
    random_seed: int = 42,
    train_note: str = "",
    run_note: str = "",
    refresh: bool = False,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    snapshot = await service.build_snapshot(
        prediction_date=prediction_date,
        universe_filter=universe_filter,
        model_family=model_family,
        feature_set=feature_set,
        cost_buffer=cost_buffer,
        train_window_months=train_window_months,
        gap_days=gap_days,
        valid_window_months=valid_window_months,
        random_seed=random_seed,
        train_note=train_note,
        run_note=run_note,
        refresh=refresh,
    )
    return ok_json_response(**_ops_status_payload(snapshot))


@router.get("/api/ml/stock-page")
async def stock_ml_page_snapshot(
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
    prediction_date: str | None = None,
    universe_filter: str = "jp_large_cap_stooq_v1",
    model_family: str = "LightGBM Classifier",
    feature_set: str = "base_v1",
    cost_buffer: float = 0.0,
    train_window_months: int = 12,
    gap_days: int = 5,
    valid_window_months: int = 1,
    random_seed: int = 42,
    train_note: str = "",
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
        train_window_months=train_window_months,
        gap_days=gap_days,
        valid_window_months=valid_window_months,
        random_seed=random_seed,
        train_note=train_note,
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
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
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
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        confirm_regenerate=req.confirm_regenerate,
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
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
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
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        run_note=req.run_note,
        refresh=req.refresh,
    )
    return ok_json_response(**payload)


@router.post("/api/ml/stock-page/actions/export-csv")
async def stock_ml_page_export_csv(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.export_csv(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        search_query=req.search_query,
        run_note=req.run_note,
        refresh=req.refresh,
    )
    return ok_json_response(**payload)


@router.post("/api/ml/stock-page/actions/export-report")
async def stock_ml_page_export_report(
    req: StockMlPageActionRequest,
    hub: HubDep,
    stock_ml_page_store: StockMlPageStoreDep,
) -> JSONResponse:
    service = _stock_ml_page_service(hub, stock_ml_page_store)
    payload = await service.export_report(
        prediction_date=req.prediction_date,
        universe_filter=req.universe_filter,
        model_family=req.model_family,
        feature_set=req.feature_set,
        cost_buffer=req.cost_buffer,
        train_window_months=req.train_window_months,
        gap_days=req.gap_days,
        valid_window_months=req.valid_window_months,
        random_seed=req.random_seed,
        train_note=req.train_note,
        search_query=req.search_query,
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
    return ok_json_response(**_job_response_payload(payload))


@router.post("/api/ml/jobs/{job_id}/cancel")
async def ml_job_cancel(job_id: str, ml_job_store: MlJobStoreDep) -> JSONResponse:
    payload = ml_job_store.request_cancel(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ok_json_response(**_job_response_payload(payload))
