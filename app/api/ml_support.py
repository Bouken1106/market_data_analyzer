"""Shared helpers for ML API routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..ml.stock_page import StockMlPageService
from ..stock_ml_page_params import StockMlPageParams
from .validators import require_symbol


def spec_job_status(status: str | None) -> str:
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


def build_job_response_payload(
    payload: dict[str, Any],
    *,
    normalize_status: bool = False,
    include_status_code: bool = False,
) -> dict[str, Any]:
    error_detail = payload.get("error_detail") or {}
    raw_status = str(payload.get("status") or "")
    response = {
        "job_id": payload.get("job_id"),
        "kind": payload.get("kind"),
        "symbol": payload.get("symbol"),
        "status": spec_job_status(raw_status) if normalize_status else raw_status,
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
    if include_status_code:
        response["status_code"] = spec_job_status(raw_status)
    return response


def training_job_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") or {}
    response = build_job_response_payload(payload, normalize_status=True)
    response.update(
        metrics=result.get("metrics"),
        summary=result.get("summary"),
        folds=result.get("folds"),
        logs=result.get("logs"),
    )
    return response


def normalize_job_symbol(raw_symbol: str) -> str:
    return require_symbol(raw_symbol, detail="Symbolを入力してください。")


def selected_model_version(snapshot: dict[str, Any]) -> str:
    dashboard = snapshot.get("dashboard", {})
    model_version = str(dashboard.get("model_version") or "").strip()
    if model_version:
        return model_version
    for item in dashboard.get("summary_cards", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("label") or "").strip() != "model_version":
            continue
        value = str(item.get("value") or "").strip()
        if value:
            return value
    filters = snapshot.get("filters", {})
    models = snapshot.get("models", {})
    return str(models.get("default_versions", {}).get(filters.get("model_family"), "")).strip()


def prediction_daily_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    dashboard = snapshot.get("dashboard", {})
    model_version = selected_model_version(snapshot)
    rows = []
    for item in dashboard.get("rows", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "prediction_date": dashboard.get("prediction_date"),
                "target_date": dashboard.get("target_date"),
                "code": item.get("code"),
                "company_name": item.get("company_name"),
                "score_cls": item.get("score_cls"),
                "prob_up": item.get("prob_up"),
                "score_rank": item.get("score_rank"),
                "expected_return": item.get("expected_return"),
                "model_version": model_version,
                "feature_version": dashboard.get("feature_version"),
                "data_version": dashboard.get("data_version"),
                "sector33_code": item.get("sector33_code"),
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


def backtest_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
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


def ops_status_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    ops = snapshot.get("ops", {})
    return {
        "pipeline_states": ops.get("pipeline", []),
        "summary": ops.get("summary_cards", []),
        "coverage_breakdown": ops.get("coverage_breakdown", []),
        "score_drift_distribution": ops.get("score_drift_distribution", {}),
        "alerts": ops.get("alerts", []),
        "logs": ops.get("logs", []),
    }


def stock_model_registry_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    models = snapshot.get("models", {})
    return {
        "models": models.get("rows", []),
        "adopted_model_version": models.get("adopted_model_version"),
        "default_versions": models.get("default_versions", {}),
    }


@dataclass(frozen=True)
class StockMlPageContext:
    params: StockMlPageParams
    service: StockMlPageService

    @classmethod
    def from_request(
        cls,
        *,
        hub: Any,
        stock_ml_page_store: Any,
        request: Any,
    ) -> "StockMlPageContext":
        return cls(
            params=request.stock_page_params(),
            service=StockMlPageService(
                full_daily_history_store=hub.full_daily_history_store,
                page_store=stock_ml_page_store,
            ),
        )

    async def call(
        self,
        method: Callable[..., Awaitable[dict[str, Any]]],
        **extra: Any,
    ) -> dict[str, Any]:
        return await method(**self.params.service_kwargs(), **extra)

    async def call_named(self, method_name: str, **extra: Any) -> dict[str, Any]:
        method = getattr(self.service, method_name)
        return await self.call(method, **extra)

    async def snapshot(self) -> dict[str, Any]:
        return await self.call_named("build_snapshot")
