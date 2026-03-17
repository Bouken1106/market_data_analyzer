"""ML training pipelines and job-runner helpers."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Callable

from fastapi import HTTPException

from ..config import (
    LOGGER,
    ML_EVAL_MONTHS,
    ML_HISTORY_DEFAULT_MONTHS,
    ML_SPLIT_EVAL_DAYS,
    ML_SPLIT_TRAIN_VAL_RATIO,
)
from ..models import MlComparisonJobRequest, MlJobCancelledError, QuantileLstmJobRequest
from ..utils import normalize_ml_history_months, normalize_symbols
from .catalog import ML_COMPARE_ALLOWED_MODELS, ML_COMPARE_DEFAULT_SYMBOLS


async def _run_quantile_lstm_pipeline(
    *,
    hub: Any,
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
    split_eval_days: int | None = None,
    split_train_val_ratio: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from ..quantile_lstm import run_quantile_lstm_forecast

    return await _run_ml_pipeline(
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
        split_eval_days=split_eval_days,
        split_train_val_ratio=split_train_val_ratio,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        trainer=run_quantile_lstm_forecast,
        training_start_message="モデル学習を開始します。",
        error_log_message="Quantile LSTM training failed for %s",
        error_detail="Quantile LSTM training failed.",
    )


def _build_training_config_payload(
    *,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    representative_days: int,
    seed: int,
    split_eval_days: int | None,
    split_train_val_ratio: float | None,
) -> dict[str, Any]:
    effective_split_eval_days = int(split_eval_days) if split_eval_days is not None else ML_SPLIT_EVAL_DAYS
    effective_split_train_val_ratio = (
        float(split_train_val_ratio) if split_train_val_ratio is not None else ML_SPLIT_TRAIN_VAL_RATIO
    )
    return {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "representative_days": representative_days,
        "seed": seed,
        "split_eval_days": max(1, effective_split_eval_days),
        "split_train_val_ratio": max(0.5, min(0.95, effective_split_train_val_ratio)),
    }


async def _run_ml_pipeline(
    *,
    hub: Any,
    symbol: str,
    months: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    representative_days: int,
    seed: int,
    refresh: bool,
    split_eval_days: int | None,
    split_train_val_ratio: float | None,
    progress_callback: Callable[[float, str], None] | None,
    cancel_check: Callable[[], None] | None,
    trainer: Callable[
        [list[dict[str, Any]], dict[str, Any], Callable[[float, str], None] | None, Callable[[], None] | None],
        dict[str, Any],
    ],
    training_start_message: str,
    error_log_message: str,
    error_detail: str,
) -> dict[str, Any]:
    if cancel_check is not None:
        cancel_check()

    if progress_callback is not None:
        progress_callback(2, "ヒストリカルデータを取得しています。")

    effective_months = normalize_ml_history_months(months)

    historical_data = await hub.historical_payload(symbol=symbol, months=effective_months, refresh=refresh)
    points = historical_data.get("points")
    if not isinstance(points, list):
        raise HTTPException(status_code=502, detail="Unexpected historical payload format.")

    config_payload = _build_training_config_payload(
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
        split_eval_days=split_eval_days,
        split_train_val_ratio=split_train_val_ratio,
    )

    if progress_callback is not None:
        progress_callback(5, training_start_message)

    try:
        model_payload = await asyncio.to_thread(
            trainer,
            points,
            config_payload,
            progress_callback,
            cancel_check,
        )
    except MlJobCancelledError:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception(error_log_message, symbol, exc_info=exc)
        raise HTTPException(status_code=500, detail=error_detail) from exc

    if progress_callback is not None:
        progress_callback(100, "結果を整形しました。")

    return {
        "symbol": historical_data.get("symbol"),
        "months": effective_months,
        "historical_source": historical_data.get("source"),
        **model_payload,
    }


async def _run_patchtst_pipeline(
    *,
    hub: Any,
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
    split_eval_days: int | None = None,
    split_train_val_ratio: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from ..patchtst_quantile import run_patchtst_forecast

    return await _run_ml_pipeline(
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
        split_eval_days=split_eval_days,
        split_train_val_ratio=split_train_val_ratio,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        trainer=run_patchtst_forecast,
        training_start_message="PatchTST学習を開始します。",
        error_log_message="PatchTST training failed for %s",
        error_detail="PatchTST training failed.",
    )


# ---------------------------------------------------------------------------
# Job-level helpers (callbacks and wrappers)
# ---------------------------------------------------------------------------

def _progress_callback_for_job(job_id: str, ml_job_store: Any):
    def _cb(progress: float, message: str) -> None:
        if ml_job_store.is_cancel_requested(job_id):
            raise MlJobCancelledError("ML job cancelled.")
        current = ml_job_store.get(job_id)
        if current is None:
            return
        status = str(current.get("status") or "")
        if status in {"completed", "failed", "cancelled"}:
            return
        next_status = "cancelling" if status == "cancelling" else "running"
        ml_job_store.update(
            job_id,
            status=next_status,
            progress=max(0.0, min(100.0, float(progress))),
            message=message,
        )

    return _cb


def _cancel_check_for_job(job_id: str, ml_job_store: Any):
    def _check() -> None:
        if ml_job_store.is_cancel_requested(job_id):
            raise MlJobCancelledError("ML job cancelled.")

    return _check


def _parse_compare_models(raw_models: str) -> list[str]:
    tokens = [item.strip().lower() for item in str(raw_models or "").split(",")]
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in seen:
            continue
        if token not in ML_COMPARE_ALLOWED_MODELS:
            continue
        selected.append(token)
        seen.add(token)
    return selected


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if isinstance(v, (int, float))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _as_aligned_arrays(left: list[Any], right: list[Any]) -> tuple[list[float], list[float]]:
    n = min(len(left), len(right))
    if n <= 0:
        return [], []
    out_left: list[float] = []
    out_right: list[float] = []
    for idx in range(n):
        try:
            l_value = float(left[idx])
            r_value = float(right[idx])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(l_value) or not math.isfinite(r_value):
            continue
        out_left.append(l_value)
        out_right.append(r_value)
    return out_left, out_right


def _compute_loss_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _safe_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    fan_chart = payload.get("fan_chart") if isinstance(payload, dict) else None
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    splits = payload.get("splits") if isinstance(payload, dict) else None
    training = payload.get("training") if isinstance(payload, dict) else None

    actual_returns = fan_chart.get("actual_returns") if isinstance(fan_chart, dict) else []
    q50_returns = fan_chart.get("q50_returns") if isinstance(fan_chart, dict) else []
    actual_prices = fan_chart.get("actual_prices") if isinstance(fan_chart, dict) else []
    q50_prices = fan_chart.get("q50_prices") if isinstance(fan_chart, dict) else []

    y_ret, yhat_ret = _as_aligned_arrays(actual_returns if isinstance(actual_returns, list) else [], q50_returns if isinstance(q50_returns, list) else [])
    y_price, yhat_price = _as_aligned_arrays(actual_prices if isinstance(actual_prices, list) else [], q50_prices if isinstance(q50_prices, list) else [])

    mae_return = float(sum(abs(a - b) for a, b in zip(y_ret, yhat_ret)) / len(y_ret)) if y_ret else None
    rmse_return = (
        float(math.sqrt(sum((a - b) ** 2 for a, b in zip(y_ret, yhat_ret)) / len(y_ret)))
        if y_ret else None
    )
    mae_price = float(sum(abs(a - b) for a, b in zip(y_price, yhat_price)) / len(y_price)) if y_price else None
    rmse_price = (
        float(math.sqrt(sum((a - b) ** 2 for a, b in zip(y_price, yhat_price)) / len(y_price)))
        if y_price else None
    )

    mape_terms = [
        abs((actual - pred) / actual) * 100.0
        for actual, pred in zip(y_price, yhat_price)
        if abs(actual) > 1e-12
    ]
    smape_terms = [
        (2.0 * abs(actual - pred) / (abs(actual) + abs(pred))) * 100.0
        for actual, pred in zip(y_price, yhat_price)
        if (abs(actual) + abs(pred)) > 1e-12
    ]

    test_split = splits.get("test") if isinstance(splits, dict) else {}
    return {
        "test_count": _safe_int(test_split.get("count")),
        "test_from": test_split.get("from"),
        "test_to": test_split.get("to"),
        "epochs_trained": _safe_int(training.get("epochs_trained")) if isinstance(training, dict) else 0,
        "best_val_pinball_loss": _safe_float(training.get("best_val_pinball_loss")) if isinstance(training, dict) else None,
        "mean_pinball_loss": _safe_float(metrics.get("mean_pinball_loss")) if isinstance(metrics, dict) else None,
        "coverage_90": _safe_float(metrics.get("coverage_90")) if isinstance(metrics, dict) else None,
        "coverage_50": _safe_float(metrics.get("coverage_50")) if isinstance(metrics, dict) else None,
        "mae_return": mae_return,
        "rmse_return": rmse_return,
        "mae_price": mae_price,
        "rmse_price": rmse_price,
        "mape_price_pct": float(sum(mape_terms) / len(mape_terms)) if mape_terms else None,
        "smape_price_pct": float(sum(smape_terms) / len(smape_terms)) if smape_terms else None,
    }


async def _run_ml_comparison_job(job_id: str, req: MlComparisonJobRequest, *, hub: Any, ml_job_store: Any) -> None:
    try:
        raw_symbols = normalize_symbols(req.symbols) if str(req.symbols or "").strip() else []
        symbols = raw_symbols or ML_COMPARE_DEFAULT_SYMBOLS
        if not symbols:
            raise HTTPException(status_code=400, detail="比較対象シンボルが空です。")

        selected_models = _parse_compare_models(req.models)
        if not selected_models:
            raise HTTPException(status_code=400, detail="比較対象モデルが空です。")

        split_eval_days = ML_SPLIT_EVAL_DAYS
        split_train_val_ratio = ML_SPLIT_TRAIN_VAL_RATIO

        ml_job_store.update(job_id, status="running", progress=1, message="比較ジョブを開始しました。")
        cancel_check = _cancel_check_for_job(job_id, ml_job_store)

        rows: list[dict[str, Any]] = []
        task_items = [(symbol, model_id) for symbol in symbols for model_id in selected_models]
        task_count = len(task_items)

        for task_idx, (symbol, model_id) in enumerate(task_items):
            cancel_check()
            progress_start = 5 + int((task_idx * 88) / max(1, task_count))
            progress_end = 5 + int(((task_idx + 1) * 88) / max(1, task_count))

            def _task_progress(progress: float, message: str) -> None:
                cancel_check()
                clamped = max(0.0, min(100.0, float(progress)))
                mapped = progress_start + (((progress_end - progress_start) * clamped) / 100.0)
                ml_job_store.update(
                    job_id,
                    status="running",
                    progress=max(1.0, min(99.0, mapped)),
                    message=f"[{task_idx + 1}/{task_count}] {symbol} | {model_id}: {message}",
                )

            try:
                if model_id == "quantile_lstm":
                    payload = await _run_quantile_lstm_pipeline(
                        hub=hub,
                        symbol=symbol,
                        months=req.months,
                        sequence_length=req.sequence_length,
                        hidden_size=req.hidden_size,
                        num_layers=req.num_layers,
                        dropout=req.dropout,
                        learning_rate=req.learning_rate,
                        batch_size=req.batch_size,
                        max_epochs=req.max_epochs,
                        patience=req.patience,
                        representative_days=3,
                        seed=req.seed,
                        refresh=req.refresh,
                        split_eval_days=split_eval_days,
                        split_train_val_ratio=split_train_val_ratio,
                        progress_callback=_task_progress,
                        cancel_check=cancel_check,
                    )
                elif model_id == "patchtst_quantile":
                    payload = await _run_patchtst_pipeline(
                        hub=hub,
                        symbol=symbol,
                        months=req.months,
                        sequence_length=req.sequence_length,
                        hidden_size=req.hidden_size,
                        num_layers=req.num_layers,
                        dropout=req.dropout,
                        learning_rate=req.learning_rate,
                        batch_size=req.batch_size,
                        max_epochs=req.max_epochs,
                        patience=req.patience,
                        representative_days=3,
                        seed=req.seed,
                        refresh=req.refresh,
                        split_eval_days=split_eval_days,
                        split_train_val_ratio=split_train_val_ratio,
                        progress_callback=_task_progress,
                        cancel_check=cancel_check,
                    )
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")

                rows.append(
                    {
                        "symbol": symbol,
                        "model_id": model_id,
                        "status": "ok",
                        "error": None,
                        "metrics": _compute_loss_metrics(payload),
                    }
                )
            except MlJobCancelledError:
                raise
            except HTTPException as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "model_id": model_id,
                        "status": "failed",
                        "error": str(exc.detail),
                        "metrics": None,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "model_id": model_id,
                        "status": "failed",
                        "error": str(exc),
                        "metrics": None,
                    }
                )

        cancel_check()
        summary_by_model: list[dict[str, Any]] = []
        for model_id in selected_models:
            model_rows = [row for row in rows if row.get("model_id") == model_id and row.get("status") == "ok"]
            metrics_list = [row.get("metrics") for row in model_rows if isinstance(row.get("metrics"), dict)]
            summary_by_model.append(
                {
                    "model_id": model_id,
                    "success_count": len(model_rows),
                    "mean_pinball_loss": _mean_or_none([item.get("mean_pinball_loss") for item in metrics_list]),
                    "mean_mae_return": _mean_or_none([item.get("mae_return") for item in metrics_list]),
                    "mean_rmse_return": _mean_or_none([item.get("rmse_return") for item in metrics_list]),
                    "mean_mae_price": _mean_or_none([item.get("mae_price") for item in metrics_list]),
                    "mean_rmse_price": _mean_or_none([item.get("rmse_price") for item in metrics_list]),
                    "mean_mape_price_pct": _mean_or_none([item.get("mape_price_pct") for item in metrics_list]),
                    "mean_smape_price_pct": _mean_or_none([item.get("smape_price_pct") for item in metrics_list]),
                    "mean_coverage_90": _mean_or_none([item.get("coverage_90") for item in metrics_list]),
                    "mean_coverage_50": _mean_or_none([item.get("coverage_50") for item in metrics_list]),
                }
            )

        success_count = len([row for row in rows if row.get("status") == "ok"])
        failed_count = len(rows) - success_count
        result = {
            "symbols": symbols,
            "models": selected_models,
            "config": {
                "months": int(req.months),
                "sequence_length": int(req.sequence_length),
                "hidden_size": int(req.hidden_size),
                "num_layers": int(req.num_layers),
                "dropout": float(req.dropout),
                "learning_rate": float(req.learning_rate),
                "batch_size": int(req.batch_size),
                "max_epochs": int(req.max_epochs),
                "patience": int(req.patience),
                "seed": int(req.seed),
                "refresh": bool(req.refresh),
            },
            "evaluation_policy": {
                "eval_months": ML_EVAL_MONTHS,
                "eval_days_approx": ML_SPLIT_EVAL_DAYS,
                "test_window": "latest 2 months (relative to the latest available trading date)",
                "train_val_split": "remaining history split by 4:1",
                "train_ratio": split_train_val_ratio,
                "val_ratio": 1.0 - split_train_val_ratio,
            },
            "summary_by_model": summary_by_model,
            "rows": rows,
            "success_count": success_count,
            "failed_count": failed_count,
        }
        ml_job_store.complete(job_id, result=result)
    except MlJobCancelledError:
        ml_job_store.mark_cancelled(job_id)
    except HTTPException as exc:
        ml_job_store.fail(job_id, error=str(exc.detail))
    except Exception as exc:
        ml_job_store.fail(job_id, error=str(exc))


async def _run_quantile_lstm_job(job_id: str, req: QuantileLstmJobRequest, *, hub: Any, ml_job_store: Any) -> None:
    try:
        if ml_job_store.is_cancel_requested(job_id):
            ml_job_store.mark_cancelled(job_id, message="ジョブ開始前に停止しました。")
            return
        ml_job_store.update(job_id, status="running", progress=1, message="ジョブを開始しました。")
        result = await _run_quantile_lstm_pipeline(
            hub=hub,
            symbol=req.symbol,
            months=req.months,
            sequence_length=req.sequence_length,
            hidden_size=req.hidden_size,
            num_layers=req.num_layers,
            dropout=req.dropout,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size,
            max_epochs=req.max_epochs,
            patience=req.patience,
            representative_days=req.representative_days,
            seed=req.seed,
            refresh=req.refresh,
            progress_callback=_progress_callback_for_job(job_id, ml_job_store),
            cancel_check=_cancel_check_for_job(job_id, ml_job_store),
        )
        ml_job_store.complete(job_id, result=result)
    except MlJobCancelledError:
        ml_job_store.mark_cancelled(job_id)
    except HTTPException as exc:
        ml_job_store.fail(job_id, error=str(exc.detail))
    except Exception as exc:
        ml_job_store.fail(job_id, error=str(exc))


async def _run_patchtst_job(job_id: str, req: QuantileLstmJobRequest, *, hub: Any, ml_job_store: Any) -> None:
    try:
        if ml_job_store.is_cancel_requested(job_id):
            ml_job_store.mark_cancelled(job_id, message="ジョブ開始前に停止しました。")
            return
        ml_job_store.update(job_id, status="running", progress=1, message="ジョブを開始しました。")
        result = await _run_patchtst_pipeline(
            hub=hub,
            symbol=req.symbol,
            months=req.months,
            sequence_length=req.sequence_length,
            hidden_size=req.hidden_size,
            num_layers=req.num_layers,
            dropout=req.dropout,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size,
            max_epochs=req.max_epochs,
            patience=req.patience,
            representative_days=req.representative_days,
            seed=req.seed,
            refresh=req.refresh,
            progress_callback=_progress_callback_for_job(job_id, ml_job_store),
            cancel_check=_cancel_check_for_job(job_id, ml_job_store),
        )
        ml_job_store.complete(job_id, result=result)
    except MlJobCancelledError:
        ml_job_store.mark_cancelled(job_id)
    except HTTPException as exc:
        ml_job_store.fail(job_id, error=str(exc.detail))
    except Exception as exc:
        ml_job_store.fail(job_id, error=str(exc))
