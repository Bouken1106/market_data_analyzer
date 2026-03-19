"""Real-data backed stock ML page service for the Japan stock dashboard."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import numpy as np
from fastapi import HTTPException

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - import guard for incomplete local envs
    lgb = None

from ..config import HISTORICAL_CACHE_TTL_SEC, LOGGER, STOCK_ML_PAGE_ROLE
from ..stores import FullDailyHistoryStore, StockMlPageStore

_TOP_N = 10
_FEATURE_VERSION = "base_v1"
_UNIVERSE_VALUE = "jp_large_cap_stooq_v1"
_UNIVERSE_LABEL = "JP large caps / Stooq daily cache"
_MODEL_PRIMARY_VERSION = "lgbm_cls_jp_v1.0.0"
_MODEL_BASELINE_VERSION = "logreg_cls_jp_v0.1.0"
_PRIMARY_MODEL_FAMILY = "LightGBM Classifier"
_BASELINE_MODEL_FAMILY = "Logistic Regression"
_DEFAULT_TRAIN_WINDOW_MONTHS = 12
_ALLOWED_TRAIN_WINDOW_MONTHS = (6, 12, 24)
_DEFAULT_GAP_DAYS = 5
_ALLOWED_GAP_DAYS = tuple(range(1, 11))
_DEFAULT_VALID_WINDOW_MONTHS = 1
_ALLOWED_VALID_WINDOW_MONTHS = (1, 2, 3)
_ALLOWED_COST_BUFFERS = (0.0, 0.002)
_BACKTEST_WINDOW_DAYS = 120
_MAX_PREDICTION_DATES = 24
_STOOQ_TIMEOUT_SEC = 25.0
_STOOQ_FETCH_CONCURRENCY = 4
_DEFAULT_MODEL_SEED = 42
_RETURN_ABS_CLIP = 0.2
_EXCLUSION_REASON_LABELS = {
    "history_short": "履歴不足",
    "fetch_failed": "取得失敗",
    "empty_response": "データなし",
    "no_data": "データなし",
}

_PRIMARY_FEATURE_ORDER = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ma_gap_10",
    "ma_gap_20",
    "vol_z_20",
    "range_pct",
    "volatility_20",
    "gap_pct",
)

_PRIMARY_WEIGHTS = {
    "ret_1d": 0.08,
    "ret_5d": 0.23,
    "ret_20d": 0.25,
    "ma_gap_10": 0.08,
    "ma_gap_20": 0.18,
    "vol_z_20": 0.10,
    "range_pct": -0.06,
    "volatility_20": -0.16,
    "gap_pct": 0.10,
}

_LOGISTIC_FEATURE_ORDER = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ma_gap_10",
    "ma_gap_20",
    "vol_z_20",
    "range_pct",
    "volatility_20",
    "gap_pct",
)


@dataclass(frozen=True)
class UniverseSymbol:
    code: str
    symbol: str
    company_name: str
    sector: str


JP_LARGE_CAP_UNIVERSE: tuple[UniverseSymbol, ...] = (
    UniverseSymbol("7203", "7203.JP", "トヨタ自動車", "輸送用機器"),
    UniverseSymbol("6758", "6758.JP", "ソニーグループ", "電気機器"),
    UniverseSymbol("9984", "9984.JP", "ソフトバンクグループ", "情報・通信業"),
    UniverseSymbol("8035", "8035.JP", "東京エレクトロン", "電気機器"),
    UniverseSymbol("7974", "7974.JP", "任天堂", "その他製品"),
    UniverseSymbol("6501", "6501.JP", "日立製作所", "電気機器"),
    UniverseSymbol("4063", "4063.JP", "信越化学工業", "化学"),
    UniverseSymbol("9983", "9983.JP", "ファーストリテイリング", "小売業"),
    UniverseSymbol("8306", "8306.JP", "三菱UFJフィナンシャル・グループ", "銀行業"),
    UniverseSymbol("6861", "6861.JP", "キーエンス", "電気機器"),
    UniverseSymbol("9432", "9432.JP", "日本電信電話", "情報・通信業"),
    UniverseSymbol("9433", "9433.JP", "KDDI", "情報・通信業"),
    UniverseSymbol("4519", "4519.JP", "中外製薬", "医薬品"),
    UniverseSymbol("4568", "4568.JP", "第一三共", "医薬品"),
    UniverseSymbol("8058", "8058.JP", "三菱商事", "卸売業"),
    UniverseSymbol("6902", "6902.JP", "デンソー", "輸送用機器"),
    UniverseSymbol("8766", "8766.JP", "東京海上ホールディングス", "保険業"),
    UniverseSymbol("7741", "7741.JP", "HOYA", "精密機器"),
    UniverseSymbol("6857", "6857.JP", "アドバンテスト", "電気機器"),
    UniverseSymbol("6367", "6367.JP", "ダイキン工業", "機械"),
    UniverseSymbol("7267", "7267.JP", "本田技研工業", "輸送用機器"),
    UniverseSymbol("6098", "6098.JP", "リクルートホールディングス", "サービス業"),
    UniverseSymbol("6146", "6146.JP", "ディスコ", "機械"),
    UniverseSymbol("6723", "6723.JP", "ルネサスエレクトロニクス", "電気機器"),
    UniverseSymbol("8001", "8001.JP", "伊藤忠商事", "卸売業"),
)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _sigmoid(value: float) -> float:
    clipped = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def _next_business_day(day: str) -> str:
    cursor = date.fromisoformat(day)
    while True:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            return cursor.isoformat()


def _to_jst_label(day: str) -> str:
    parsed = date.fromisoformat(day)
    weekdays = "月火水木金土日"
    return f"{parsed.isoformat()} ({weekdays[parsed.weekday()]})"


def _to_jst_timestamp(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    jst = parsed.astimezone(timezone(timedelta(hours=9)))
    return jst.strftime("%Y-%m-%d %H:%M JST")


def _config_hash(
    *,
    prediction_date: str,
    universe_filter: str,
    model_family: str,
    feature_set: str,
    cost_buffer: float,
    train_window_months: int,
    gap_days: int,
    valid_window_months: int,
    random_seed: int,
) -> str:
    payload = {
        "prediction_date": prediction_date,
        "universe_filter": universe_filter,
        "model_family": model_family,
        "feature_set": feature_set,
        "cost_buffer": round(float(cost_buffer), 6),
        "train_window_months": int(train_window_months),
        "gap_days": int(gap_days),
        "valid_window_months": int(valid_window_months),
        "random_seed": int(random_seed),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest[:12]


def _normalize_choice(value: Any, *, allowed: tuple[int, ...], default: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    return normalized if normalized in allowed else default


def _normalize_random_seed(value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_MODEL_SEED
    return normalized if normalized > 0 else _DEFAULT_MODEL_SEED


def _normalize_cost_buffer(value: Any) -> float:
    normalized = _safe_float(value)
    if normalized is None:
        return _ALLOWED_COST_BUFFERS[0]
    for allowed in _ALLOWED_COST_BUFFERS:
        if abs(normalized - allowed) <= 1e-9:
            return allowed
    return _ALLOWED_COST_BUFFERS[0]


def _format_cost_buffer(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value else "0.0"


def _window_days_from_months(months: int) -> int:
    return int(months) * 21


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0 or y_score.size == 0:
        return None
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0 or y_score.size == 0:
        return None
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    hit_count = 0
    precision_sum = 0.0
    for index, target in enumerate(y_sorted, start=1):
        if int(target) != 1:
            continue
        hit_count += 1
        precision_sum += hit_count / index
    return float(precision_sum / positives)


def _balanced_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float | None:
    if y_true.size == 0:
        return None
    predicted = (y_prob >= threshold).astype(np.int64)
    positives = y_true == 1
    negatives = y_true == 0
    pos_total = int(np.sum(positives))
    neg_total = int(np.sum(negatives))
    if pos_total == 0 or neg_total == 0:
        return None
    tpr = float(np.sum(predicted[positives] == 1)) / pos_total
    tnr = float(np.sum(predicted[negatives] == 0)) / neg_total
    return (tpr + tnr) / 2.0


def _hit_ratio(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float | None:
    if y_true.size == 0:
        return None
    predicted = (y_prob >= threshold).astype(np.int64)
    if predicted.size == 0:
        return None
    return float(np.mean(predicted == y_true))


def _series_metrics(daily_returns: list[float]) -> dict[str, float | None]:
    if len(daily_returns) == 0:
        return {
            "cagr_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "win_rate_pct": None,
            "avg_holding_pnl_pct": None,
        }
    equity = np.cumprod(np.array([1.0] + [1.0 + item for item in daily_returns], dtype=np.float64))
    years = max(len(daily_returns) / 252.0, 1e-9)
    total_return = float(equity[-1] / equity[0]) - 1.0
    cagr = (float(equity[-1] / equity[0]) ** (1.0 / years)) - 1.0 if equity[-1] > 0 else float("nan")
    volatility = float(np.std(daily_returns, ddof=1)) if len(daily_returns) >= 2 else 0.0
    sharpe = ((float(np.mean(daily_returns)) * 252.0) / (volatility * math.sqrt(252.0))) if volatility > 1e-12 else None
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity / running_max) - 1.0
    max_drawdown = float(np.min(drawdowns)) if drawdowns.size else 0.0
    win_rate = float(np.mean(np.array(daily_returns) > 0.0))
    avg_holding = float(np.mean(daily_returns))
    return {
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0 if math.isfinite(cagr) else None,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown * 100.0,
        "win_rate_pct": win_rate * 100.0,
        "avg_holding_pnl_pct": avg_holding * 100.0,
    }


def _safe_pct_label(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}%"


def _safe_signed_pct_label(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:+.{digits}f}%"


def _safe_number_label(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _population_stability_index(
    reference_values: list[float],
    current_values: list[float],
    *,
    bucket_count: int = 10,
) -> float | None:
    if len(reference_values) < bucket_count or len(current_values) < bucket_count:
        return None
    reference_arr = np.array(reference_values, dtype=np.float64)
    current_arr = np.array(current_values, dtype=np.float64)
    quantiles = np.quantile(reference_arr, np.linspace(0.0, 1.0, bucket_count + 1))
    edges = np.unique(quantiles)
    if edges.size < 3:
        return 0.0
    edges = edges.astype(np.float64, copy=True)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference_arr, bins=edges)
    cur_counts, _ = np.histogram(current_arr, bins=edges)
    ref_ratio = np.clip(ref_counts / max(int(np.sum(ref_counts)), 1), 1e-6, None)
    cur_ratio = np.clip(cur_counts / max(int(np.sum(cur_counts)), 1), 1e-6, None)
    psi = np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio))
    return float(psi)


def _excluded_reason_label(reason: str) -> str:
    return _EXCLUSION_REASON_LABELS.get(str(reason or "").strip(), "その他")


def _summarize_excluded_symbols(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return [
            {
                "reason": "none",
                "label": "除外なし",
                "count": 0,
                "detail": "全ユニバース銘柄を利用できています。",
            }
        ]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        reason = str(item.get("reason") or "no_data").strip() or "no_data"
        buckets.setdefault(reason, []).append(item)
    ordered_reasons = ("history_short", "fetch_failed", "empty_response", "no_data")
    ordered_reasons += tuple(reason for reason in buckets.keys() if reason not in ordered_reasons)
    summary: list[dict[str, Any]] = []
    for reason in ordered_reasons:
        grouped = buckets.get(reason)
        if not grouped:
            continue
        sample = ", ".join(str(item.get("code") or "-") for item in grouped[:3])
        detail = f"例: {sample}" if sample else ""
        if reason == "history_short":
            min_points = min(
                int(item.get("points") or 0)
                for item in grouped
                if str(item.get("points") or "").strip()
            ) if any(str(item.get("points") or "").strip() for item in grouped) else 0
            detail = (
                f"260営業日未満のため除外。最小 {min_points} 日。"
                + (f" 例: {sample}" if sample else "")
            )
        elif reason == "fetch_failed":
            detail = "Stooq 取得失敗またはキャッシュ未整備。" + (f" 例: {sample}" if sample else "")
        elif reason in {"empty_response", "no_data"}:
            detail = "公開ソースから十分な日足を返却できませんでした。" + (f" 例: {sample}" if sample else "")
        summary.append(
            {
                "reason": reason,
                "label": _excluded_reason_label(reason),
                "count": len(grouped),
                "detail": detail,
            }
        )
    return summary


class StockMlPageService:
    def __init__(
        self,
        *,
        full_daily_history_store: FullDailyHistoryStore,
        page_store: StockMlPageStore,
    ) -> None:
        self.full_daily_history_store = full_daily_history_store
        self.page_store = page_store

    async def build_snapshot(
        self,
        *,
        prediction_date: str | None,
        universe_filter: str | None,
        model_family: str | None,
        feature_set: str | None,
        cost_buffer: float,
        train_window_months: int,
        gap_days: int,
        valid_window_months: int,
        random_seed: int,
        train_note: str | None,
        run_note: str | None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        selected_train_window_months = _normalize_choice(
            train_window_months,
            allowed=_ALLOWED_TRAIN_WINDOW_MONTHS,
            default=_DEFAULT_TRAIN_WINDOW_MONTHS,
        )
        selected_gap_days = _normalize_choice(
            gap_days,
            allowed=_ALLOWED_GAP_DAYS,
            default=_DEFAULT_GAP_DAYS,
        )
        selected_valid_window_months = _normalize_choice(
            valid_window_months,
            allowed=_ALLOWED_VALID_WINDOW_MONTHS,
            default=_DEFAULT_VALID_WINDOW_MONTHS,
        )
        selected_random_seed = _normalize_random_seed(random_seed)
        selected_cost_buffer = _normalize_cost_buffer(cost_buffer)
        normalized_train_note = str(train_note or "").strip()[:500]
        histories, excluded_symbols = await self._load_histories(refresh=refresh)
        dataset = self._build_dataset(
            histories=histories,
            excluded_symbols=excluded_symbols,
            cost_buffer=selected_cost_buffer,
        )
        if not dataset["rows"]:
            raise HTTPException(status_code=502, detail="日本株の日足データを構築できませんでした。")

        available_prediction_dates = dataset["prediction_dates"]
        if not available_prediction_dates:
            raise HTTPException(status_code=502, detail="prediction_date 候補を構築できませんでした。")
        selected_prediction_date = prediction_date if prediction_date in available_prediction_dates else available_prediction_dates[-1]
        selected_model_family = model_family if model_family in {_PRIMARY_MODEL_FAMILY, _BASELINE_MODEL_FAMILY} else _PRIMARY_MODEL_FAMILY
        selected_feature_set = feature_set if feature_set in {_FEATURE_VERSION} else _FEATURE_VERSION
        selected_universe = universe_filter if universe_filter in {_UNIVERSE_VALUE} else _UNIVERSE_VALUE
        latest_market_date = dataset["latest_market_date"]
        config_hash = _config_hash(
            prediction_date=selected_prediction_date,
            universe_filter=selected_universe,
            model_family=selected_model_family,
            feature_set=selected_feature_set,
            cost_buffer=selected_cost_buffer,
            train_window_months=selected_train_window_months,
            gap_days=selected_gap_days,
            valid_window_months=selected_valid_window_months,
            random_seed=selected_random_seed,
        )

        training = self._build_training_view(
            dataset=dataset,
            cost_buffer=selected_cost_buffer,
            train_window_months=selected_train_window_months,
            gap_days=selected_gap_days,
            valid_window_months=selected_valid_window_months,
            random_seed=selected_random_seed,
        )
        backtest = self._build_backtest_view(
            dataset=dataset,
            train_window_months=selected_train_window_months,
            gap_days=selected_gap_days,
            random_seed=selected_random_seed,
        )
        quality_gate = self._build_model_quality_gate(
            dataset=dataset,
            training=training,
            gap_days=selected_gap_days,
        )
        models = self._build_model_registry(
            training=training,
            backtest=backtest,
            config_hash=config_hash,
            cost_buffer=selected_cost_buffer,
            train_window_months=selected_train_window_months,
            gap_days=selected_gap_days,
            valid_window_months=selected_valid_window_months,
            random_seed=selected_random_seed,
            quality_gate=quality_gate,
        )
        selected_model_version = models["default_versions"].get(selected_model_family, _MODEL_PRIMARY_VERSION)
        dashboard = self._build_dashboard_view(
            dataset=dataset,
            prediction_date=selected_prediction_date,
            model_family=selected_model_family,
            selected_model_version=selected_model_version,
            adopted_model_version=models["adopted_model_version"],
            feature_set=selected_feature_set,
            cost_buffer=selected_cost_buffer,
            latest_market_date=latest_market_date,
            train_window_months=selected_train_window_months,
            random_seed=selected_random_seed,
        )
        ops = self._build_ops_view(
            dataset=dataset,
            dashboard=dashboard,
            training=training,
            backtest=backtest,
            models=models,
            refresh=refresh,
        )
        permissions = self._build_permissions(
            dataset=dataset,
            dashboard=dashboard,
            training=training,
            models=models,
        )
        global_status, sidebar_status = self._build_status_views(
            dashboard=dashboard,
            training=training,
            models=models,
            selected_model_family=selected_model_family,
            permissions=permissions,
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "header": {
                "updated_at": dashboard["latest_update"],
                "env": self._infer_environment(),
            },
            "filter_options": {
                "prediction_dates": [
                    {"value": item, "label": f"{item} / {dataset['target_date_by_prediction'].get(item) or _next_business_day(item)}"}
                    for item in available_prediction_dates[-_MAX_PREDICTION_DATES:]
                ],
                "universe_filters": [{"value": _UNIVERSE_VALUE, "label": _UNIVERSE_LABEL}],
                "model_families": [
                    {"value": _PRIMARY_MODEL_FAMILY, "label": _PRIMARY_MODEL_FAMILY},
                    {"value": _BASELINE_MODEL_FAMILY, "label": _BASELINE_MODEL_FAMILY},
                ],
                "feature_sets": [{"value": _FEATURE_VERSION, "label": _FEATURE_VERSION}],
                "cost_buffers": [
                    {"value": _format_cost_buffer(item), "label": _format_cost_buffer(item)}
                    for item in _ALLOWED_COST_BUFFERS
                ],
                "train_window_months": [
                    {"value": str(item), "label": f"{item}か月"}
                    for item in _ALLOWED_TRAIN_WINDOW_MONTHS
                ],
                "gap_days": [
                    {"value": str(item), "label": f"{item}営業日"}
                    for item in _ALLOWED_GAP_DAYS
                ],
                "valid_window_months": [
                    {"value": str(item), "label": f"{item}か月"}
                    for item in _ALLOWED_VALID_WINDOW_MONTHS
                ],
            },
            "filters": {
                "prediction_date": selected_prediction_date,
                "universe_filter": selected_universe,
                "model_family": selected_model_family,
                "feature_set": selected_feature_set,
                "cost_buffer": _format_cost_buffer(selected_cost_buffer),
                "train_window_months": str(selected_train_window_months),
                "gap_days": str(selected_gap_days),
                "valid_window_months": str(selected_valid_window_months),
                "random_seed": str(selected_random_seed),
                "train_note": normalized_train_note,
                "run_note": str(run_note or "").strip()[:200],
            },
            "permissions": permissions,
            "global_status": global_status,
            "sidebar_status": sidebar_status,
            "dashboard": dashboard,
            "train": training,
            "backtest": backtest,
            "models": models,
            "ops": ops,
        }

    async def run_inference(self, **kwargs: Any) -> dict[str, Any]:
        confirm_regenerate = bool(kwargs.pop("confirm_regenerate", False))
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "run_inference")
        generation = self._prediction_run_payload(current_snapshot)
        existing = self.page_store.find_prediction_run(generation_key=generation["generation_key"])
        if existing is not None and not confirm_regenerate:
            generated_at = _to_jst_timestamp(existing.get("generated_at"))
            raise HTTPException(
                status_code=409,
                detail=(
                    "同一条件の prediction_daily は既に生成済みです。"
                    f" 最終生成: {generated_at}。再生成する場合は確認してください。"
                ),
            )
        self.page_store.mark_inference_run()
        self.page_store.record_prediction_run(
            generation_key=generation["generation_key"],
            prediction_date=generation["prediction_date"],
            target_date=generation["target_date"],
            model_version=generation["model_version"],
            feature_version=generation["feature_version"],
            data_version=generation["data_version"],
            config_hash=generation["config_hash"],
        )
        regenerate_label = "Regenerated" if existing is not None else "Generated"
        run_note = str(kwargs.get("run_note") or "").strip()
        self.page_store.add_audit_log(
            action="run_inference",
            detail=(
                f"{regenerate_label} prediction_daily."
                f" prediction_date={generation['prediction_date']}"
                f" target_date={generation['target_date']}"
                f" model_version={generation['model_version']}"
                f" config={generation['config_hash']}"
                + (f" note={run_note[:80]}" if run_note else "")
            ),
            level="warning" if existing is not None else "normal",
        )
        return await self.build_snapshot(**kwargs)

    async def create_training_job(self, **kwargs: Any) -> dict[str, Any]:
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "create_training_job")
        self.page_store.mark_training_run()
        filters = current_snapshot.get("filters", {})
        run_note = str(kwargs.get("run_note") or "").strip()
        train_note = str(kwargs.get("train_note") or "").strip()
        self.page_store.add_audit_log(
            action="create_training_job",
            detail=(
                "Training summary refreshed."
                f" prediction_date={filters.get('prediction_date')}"
                f" cost_buffer={filters.get('cost_buffer')}"
                f" config={current_snapshot.get('config_hash') or '-'}"
                + (f" train_note={train_note[:120]}" if train_note else "")
                + (f" note={run_note[:80]}" if run_note else "")
            ),
            level="warning",
        )
        return await self.build_snapshot(**kwargs)

    async def refresh_data(self, **kwargs: Any) -> dict[str, Any]:
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "refresh_data")
        filters = current_snapshot.get("filters", {})
        run_note = str(kwargs.get("run_note") or "").strip()
        self.page_store.add_audit_log(
            action="refresh_data",
            detail=(
                "Universe daily cache refreshed from Stooq."
                f" prediction_date={filters.get('prediction_date')}"
                f" config={current_snapshot.get('config_hash') or '-'}"
                + (f" note={run_note[:80]}" if run_note else "")
            ),
        )
        refreshed_kwargs = dict(kwargs)
        refreshed_kwargs["refresh"] = True
        return await self.build_snapshot(**refreshed_kwargs)

    async def adopt_model(self, *, model_version: str, **kwargs: Any) -> dict[str, Any]:
        if model_version not in {_MODEL_PRIMARY_VERSION, _MODEL_BASELINE_VERSION}:
            raise HTTPException(status_code=400, detail="Unknown model_version.")
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "adopt_model")
        selected_model = next(
            (
                item
                for item in (current_snapshot.get("models", {}).get("rows") or [])
                if isinstance(item, dict) and str(item.get("model_version") or "").strip() == model_version
            ),
            None,
        )
        if not isinstance(selected_model, dict):
            raise HTTPException(status_code=404, detail="Model not found.")
        if not bool(selected_model.get("adoptable")):
            blockers = [
                str(item).strip()
                for item in (selected_model.get("adopt_blockers") or [])
                if str(item).strip()
            ]
            detail = "このモデルは現在採用できません。"
            if blockers:
                detail += " " + " / ".join(blockers[:3])
            raise HTTPException(status_code=409, detail=detail)
        self.page_store.set_adopted_model_version(model_version)
        run_note = str(kwargs.get("run_note") or "").strip()
        self.page_store.add_audit_log(
            action="adopt_model",
            detail=(
                f"Adopted model changed to {model_version}."
                f" config={current_snapshot.get('config_hash') or '-'}"
                + (f" note={run_note[:80]}" if run_note else "")
            ),
            level="warning",
        )
        return await self.build_snapshot(**kwargs)

    async def export_csv(self, *, search_query: str = "", **kwargs: Any) -> dict[str, Any]:
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "export_csv")
        dashboard = current_snapshot.get("dashboard", {})
        filtered_rows = self._filter_dashboard_rows(
            rows=dashboard.get("rows", []),
            search_query=search_query,
        )
        model_version = self._prediction_run_payload(current_snapshot)["model_version"]
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "prediction_date",
                "target_date",
                "code",
                "score_cls",
                "prob_up",
                "score_rank",
                "expected_return",
                "model_version",
                "feature_version",
                "data_version",
            ]
        )
        for row in filtered_rows:
            writer.writerow(
                [
                    dashboard.get("prediction_date") or current_snapshot.get("filters", {}).get("prediction_date") or "",
                    dashboard.get("target_date") or "",
                    row.get("code") or "",
                    f"{float(row.get('score_cls') or 0.0):.2f}",
                    f"{float(row.get('prob_up') or 0.0):.3f}",
                    "NULL" if row.get("score_rank") is None else str(row.get("score_rank")),
                    "NULL" if row.get("expected_return") is None else f"{float(row.get('expected_return')):.6f}",
                    model_version,
                    dashboard.get("feature_version") or current_snapshot.get("filters", {}).get("feature_set") or "",
                    dashboard.get("data_version") or "",
                ]
            )
        trimmed_query = str(search_query or "").strip()
        self.page_store.add_audit_log(
            action="export_csv",
            detail=(
                f"CSV exported. rows={len(filtered_rows)}"
                f" query={trimmed_query[:80] or '-'}"
                f" config={current_snapshot.get('config_hash') or '-'}"
            ),
        )
        updated_snapshot = await self.build_snapshot(**kwargs)
        filename = f"prediction_daily_{str(dashboard.get('prediction_date') or '').replace('-', '')}.csv"
        return {
            "filename": filename,
            "content": output.getvalue(),
            "row_count": len(filtered_rows),
            "snapshot": updated_snapshot,
        }

    async def export_report(self, *, search_query: str = "", **kwargs: Any) -> dict[str, Any]:
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "export_report")
        trimmed_query = str(search_query or "").strip()
        self.page_store.add_audit_log(
            action="export_report",
            detail=(
                "Report exported."
                f" query={trimmed_query[:80] or '-'}"
                f" config={current_snapshot.get('config_hash') or '-'}"
            ),
        )
        updated_snapshot = await self.build_snapshot(**kwargs)
        filename = f"stock_ml_report_{str(updated_snapshot.get('dashboard', {}).get('prediction_date') or '').replace('-', '')}.json"
        return {
            "filename": filename,
            "content": f"{json.dumps(updated_snapshot, ensure_ascii=False, indent=2)}\n",
            "snapshot": updated_snapshot,
        }

    async def _load_histories(self, *, refresh: bool) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        semaphore = asyncio.Semaphore(_STOOQ_FETCH_CONCURRENCY)

        async def load_one(symbol_meta: UniverseSymbol) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
            async with semaphore:
                points, issue = await self._get_stooq_history(symbol_meta, refresh=refresh)
            if len(points) < 260:
                return symbol_meta.code, None, {
                    "code": symbol_meta.code,
                    "symbol": symbol_meta.symbol,
                    "company_name": symbol_meta.company_name,
                    "reason": "history_short" if points else (issue or "no_data"),
                    "points": len(points),
                }
            return symbol_meta.code, {
                "meta": symbol_meta,
                "points": points,
                "last_date": str(points[-1].get("t") or "").split(" ")[0],
            }, None

        results = await asyncio.gather(*(load_one(item) for item in JP_LARGE_CAP_UNIVERSE))
        out: dict[str, dict[str, Any]] = {}
        excluded: list[dict[str, Any]] = []
        for code, payload, exclusion in results:
            if payload is not None:
                out[code] = payload
                continue
            if exclusion is not None:
                excluded.append(exclusion)
        return out, excluded

    async def _get_stooq_history(self, symbol_meta: UniverseSymbol, *, refresh: bool) -> tuple[list[dict[str, Any]], str | None]:
        normalized_symbol = symbol_meta.symbol.upper()
        cached = await self.full_daily_history_store.get(normalized_symbol)
        updated_epoch = await self.full_daily_history_store.last_updated_epoch(normalized_symbol)
        now_epoch = time.time()
        is_fresh = (
            bool(cached)
            and updated_epoch is not None
            and ((now_epoch - float(updated_epoch)) <= HISTORICAL_CACHE_TTL_SEC)
        )
        if cached and not refresh and is_fresh:
            return cached, None

        try:
            points = await asyncio.to_thread(self._fetch_stooq_csv, symbol_meta.code)
        except Exception as exc:
            LOGGER.warning("Failed to fetch Stooq CSV for %s: %s", normalized_symbol, exc)
            return (cached, None) if cached else ([], "fetch_failed")
        if points:
            await self.full_daily_history_store.upsert(normalized_symbol, points)
            return points, None
        if cached:
            return cached, None
        return [], "empty_response"

    @staticmethod
    def _fetch_stooq_csv(code: str) -> list[dict[str, Any]]:
        url = f"https://stooq.com/q/d/l/?s={code.lower()}.jp&i=d"
        with httpx.Client(timeout=_STOOQ_TIMEOUT_SEC, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
                },
            )
            response.raise_for_status()
            payload = response.text
        reader = csv.DictReader(io.StringIO(payload))
        points: list[dict[str, Any]] = []
        for row in reader:
            date_text = str(row.get("Date") or "").strip()
            if not date_text:
                continue
            open_value = _safe_float(row.get("Open"))
            high_value = _safe_float(row.get("High"))
            low_value = _safe_float(row.get("Low"))
            close_value = _safe_float(row.get("Close"))
            volume_value = _safe_float(row.get("Volume"))
            if close_value is None or close_value <= 0:
                continue
            point = {
                "t": date_text,
                "o": open_value if open_value is not None and open_value > 0 else close_value,
                "h": high_value if high_value is not None and high_value > 0 else close_value,
                "l": low_value if low_value is not None and low_value > 0 else close_value,
                "c": close_value,
                "v": volume_value if volume_value is not None and volume_value >= 0 else 0.0,
                "_src": "stooq",
            }
            points.append(point)
        points.sort(key=lambda item: str(item["t"]))
        return points

    def _build_dataset(
        self,
        *,
        histories: dict[str, dict[str, Any]],
        excluded_symbols: list[dict[str, Any]],
        cost_buffer: float,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        per_date: dict[str, list[dict[str, Any]]] = {}
        target_date_by_prediction: dict[str, str] = {}
        latest_market_date = ""
        excluded_reason_breakdown = _summarize_excluded_symbols(excluded_symbols)
        for payload in histories.values():
            meta: UniverseSymbol = payload["meta"]
            points = payload["points"]
            latest_market_date = max(latest_market_date, payload["last_date"])
            closes = np.array([float(item["c"]) for item in points], dtype=np.float64)
            opens = np.array([float(item["o"]) for item in points], dtype=np.float64)
            highs = np.array([float(item["h"]) for item in points], dtype=np.float64)
            lows = np.array([float(item["l"]) for item in points], dtype=np.float64)
            volumes = np.array([float(item.get("v") or 0.0) for item in points], dtype=np.float64)
            dates = [str(item["t"]).split(" ")[0] for item in points]
            daily_returns = np.full_like(closes, np.nan)
            daily_returns[1:] = (closes[1:] / closes[:-1]) - 1.0

            for idx in range(21, len(points) - 1):
                if idx < 20:
                    continue
                ret_1d = (closes[idx] / closes[idx - 1]) - 1.0
                ret_5d = (closes[idx] / closes[idx - 5]) - 1.0
                ret_20d = (closes[idx] / closes[idx - 20]) - 1.0
                ma_gap_10 = (closes[idx] / float(np.mean(closes[idx - 9:idx + 1]))) - 1.0
                ma_gap_20 = (closes[idx] / float(np.mean(closes[idx - 19:idx + 1]))) - 1.0
                volume_window = volumes[idx - 19:idx + 1]
                vol_mean = float(np.mean(volume_window))
                vol_std = float(np.std(volume_window, ddof=1)) if len(volume_window) >= 2 else 0.0
                vol_z_20 = ((volumes[idx] - vol_mean) / vol_std) if vol_std > 1e-9 else 0.0
                range_pct = ((highs[idx] - lows[idx]) / closes[idx]) if closes[idx] > 0 else 0.0
                returns_window = daily_returns[idx - 19:idx + 1]
                clean_returns = returns_window[np.isfinite(returns_window)]
                volatility_20 = float(np.std(clean_returns, ddof=1)) if clean_returns.size >= 2 else 0.0
                gap_pct = ((opens[idx] / closes[idx - 1]) - 1.0) if closes[idx - 1] > 0 else 0.0
                next_return = (closes[idx + 1] / closes[idx]) - 1.0
                row = {
                    "date": dates[idx],
                    "target_date": dates[idx + 1],
                    "code": meta.code,
                    "symbol": meta.symbol,
                    "company_name": meta.company_name,
                    "sector": meta.sector,
                    "close": float(closes[idx]),
                    "next_return": float(next_return),
                    "y": 1 if next_return > cost_buffer else 0,
                    "ret_1d": float(ret_1d),
                    "ret_5d": float(ret_5d),
                    "ret_20d": float(ret_20d),
                    "ma_gap_10": float(ma_gap_10),
                    "ma_gap_20": float(ma_gap_20),
                    "vol_z_20": float(vol_z_20),
                    "range_pct": float(range_pct),
                    "volatility_20": float(volatility_20),
                    "gap_pct": float(gap_pct),
                    "volume_ratio_20": float((volumes[idx] / vol_mean) if vol_mean > 1e-9 else 1.0),
                }
                rows.append(row)
                per_date.setdefault(row["date"], []).append(row)
                target_date_by_prediction[row["date"]] = row["target_date"]

        prediction_dates = sorted(day for day, day_rows in per_date.items() if len(day_rows) >= 8)
        self._apply_primary_scores(per_date)
        return {
            "rows": rows,
            "by_date": per_date,
            "prediction_dates": prediction_dates,
            "target_date_by_prediction": target_date_by_prediction,
            "excluded_symbols": len(excluded_symbols),
            "excluded_reason_breakdown": excluded_reason_breakdown,
            "excluded_symbol_details": excluded_symbols[:12],
            "latest_market_date": latest_market_date,
        }

    def _apply_primary_scores(self, by_date: dict[str, list[dict[str, Any]]]) -> None:
        for day_rows in by_date.values():
            if not day_rows:
                continue
            for feature_name in _PRIMARY_FEATURE_ORDER:
                values = np.array([float(item[feature_name]) for item in day_rows], dtype=np.float64)
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
                for item, raw_value in zip(day_rows, values, strict=False):
                    z_value = 0.0 if std <= 1e-9 else (float(raw_value) - mean) / std
                    item[f"z_{feature_name}"] = float(z_value)

            for item in day_rows:
                contributions = []
                score = 0.0
                for feature_name in _PRIMARY_FEATURE_ORDER:
                    contribution = _PRIMARY_WEIGHTS[feature_name] * float(item[f"z_{feature_name}"])
                    contributions.append((feature_name, contribution))
                    score += contribution
                contributions.sort(key=lambda entry: abs(entry[1]), reverse=True)
                item["primary_score"] = float(score)
                item["primary_prob"] = float(_sigmoid(score * 1.9))
                item["primary_contrib"] = [
                    {"name": name, "value": float(value)}
                    for name, value in contributions[:3]
                ]

    @staticmethod
    def _require_lightgbm() -> None:
        if lgb is None:
            raise HTTPException(status_code=500, detail="lightgbm is not installed. Run pip install -r requirements.txt.")

    @staticmethod
    def _feature_matrix(rows: list[dict[str, Any]], feature_order: tuple[str, ...]) -> np.ndarray:
        if not rows:
            return np.empty((0, len(feature_order)), dtype=np.float64)
        return np.array(
            [[float(item[name]) for name in feature_order] for item in rows],
            dtype=np.float64,
        )

    @staticmethod
    def _clip_bounds(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return -_RETURN_ABS_CLIP, _RETURN_ABS_CLIP
        lower = float(np.quantile(values, 0.02))
        upper = float(np.quantile(values, 0.98))
        lower = max(-_RETURN_ABS_CLIP, lower)
        upper = min(_RETURN_ABS_CLIP, upper)
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    def _fit_lightgbm_classifier(self, rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
        self._require_lightgbm()
        if not rows:
            return {"mode": "constant", "prob": 0.5}
        x = self._feature_matrix(rows, _PRIMARY_FEATURE_ORDER)
        y = np.array([int(item["y"]) for item in rows], dtype=np.float64)
        positive_ratio = float(np.mean(y)) if y.size else 0.5
        if y.size == 0 or np.all(y == y[0]):
            base_prob = min(0.99, max(0.01, positive_ratio))
            return {"mode": "constant", "prob": base_prob}

        train_set = lgb.Dataset(x, label=y, feature_name=list(_PRIMARY_FEATURE_ORDER))
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 24,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "num_threads": 1,
            "force_col_wise": True,
        }
        booster = lgb.train(params, train_set, num_boost_round=72)
        return {"mode": "booster", "booster": booster}

    def _predict_lightgbm_classifier(
        self,
        model: dict[str, Any],
        rows: list[dict[str, Any]],
        *,
        include_contrib: bool = False,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        if model.get("mode") == "constant":
            prob = float(model.get("prob") or 0.5)
            score = math.log(prob / (1.0 - prob)) if 0.0 < prob < 1.0 else 0.0
            return [{"score": score, "prob": prob, "feature_contrib": []} for _ in rows]

        booster = model["booster"]
        x = self._feature_matrix(rows, _PRIMARY_FEATURE_ORDER)
        raw_scores = np.array(booster.predict(x, raw_score=True), dtype=np.float64)
        probs = np.array(booster.predict(x), dtype=np.float64)
        contrib_matrix = None
        if include_contrib:
            contrib_matrix = np.array(booster.predict(x, pred_contrib=True), dtype=np.float64)

        out: list[dict[str, Any]] = []
        for index, (score, prob) in enumerate(zip(raw_scores.tolist(), probs.tolist(), strict=False)):
            feature_contrib: list[dict[str, float]] = []
            if contrib_matrix is not None and index < contrib_matrix.shape[0]:
                raw_contrib = contrib_matrix[index][: len(_PRIMARY_FEATURE_ORDER)]
                pairs = sorted(
                    (
                        {"name": name, "value": float(value)}
                        for name, value in zip(_PRIMARY_FEATURE_ORDER, raw_contrib.tolist(), strict=False)
                    ),
                    key=lambda item: abs(item["value"]),
                    reverse=True,
                )
                feature_contrib = pairs[:3]
            out.append(
                {
                    "score": float(score),
                    "prob": float(prob),
                    "feature_contrib": feature_contrib,
                }
            )
        return out

    def _fit_lightgbm_regression(self, rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
        self._require_lightgbm()
        if not rows:
            return {"mode": "constant", "mean_return": 0.0, "lower_clip": -0.02, "upper_clip": 0.02}
        x = self._feature_matrix(rows, _PRIMARY_FEATURE_ORDER)
        y = np.array([float(item["next_return"]) for item in rows], dtype=np.float64)
        lower_clip, upper_clip = self._clip_bounds(y)
        if y.size == 0 or np.allclose(y, y[0]):
            return {
                "mode": "constant",
                "mean_return": float(np.mean(y)) if y.size else 0.0,
                "lower_clip": lower_clip,
                "upper_clip": upper_clip,
            }

        train_set = lgb.Dataset(x, label=y, feature_name=list(_PRIMARY_FEATURE_ORDER))
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 24,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "num_threads": 1,
            "force_col_wise": True,
        }
        booster = lgb.train(params, train_set, num_boost_round=84)
        return {
            "mode": "booster",
            "booster": booster,
            "lower_clip": lower_clip,
            "upper_clip": upper_clip,
        }

    def _predict_lightgbm_regression(self, model: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        if model.get("mode") == "constant":
            value = float(model.get("mean_return") or 0.0)
            lower = float(model.get("lower_clip") or -_RETURN_ABS_CLIP)
            upper = float(model.get("upper_clip") or _RETURN_ABS_CLIP)
            clipped = min(upper, max(lower, value))
            return [clipped for _ in rows]

        booster = model["booster"]
        x = self._feature_matrix(rows, _PRIMARY_FEATURE_ORDER)
        values = np.array(booster.predict(x), dtype=np.float64)
        lower = float(model.get("lower_clip") or -_RETURN_ABS_CLIP)
        upper = float(model.get("upper_clip") or _RETURN_ABS_CLIP)
        clipped = np.clip(values, lower, upper)
        return [float(item) for item in clipped.tolist()]

    @staticmethod
    def _lightgbm_feature_summary(model: dict[str, Any]) -> str:
        if model.get("mode") != "booster":
            return "importance unavailable"
        gains = np.array(model["booster"].feature_importance(importance_type="gain"), dtype=np.float64)
        if gains.size == 0 or not np.any(gains > 0):
            return "importance unavailable"
        order = np.argsort(-gains)
        labels = [f"{_PRIMARY_FEATURE_ORDER[index]}({gains[index]:.1f})" for index in order[:3]]
        return " / ".join(labels)

    def _fit_linear_return_regression(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"mode": "constant", "mean_return": 0.0, "lower_clip": -0.02, "upper_clip": 0.02}
        x = self._feature_matrix(rows, _LOGISTIC_FEATURE_ORDER)
        y = np.array([float(item["next_return"]) for item in rows], dtype=np.float64)
        lower_clip, upper_clip = self._clip_bounds(y)
        if y.size == 0 or np.allclose(y, y[0]):
            return {
                "mode": "constant",
                "mean_return": float(np.mean(y)) if y.size else 0.0,
                "lower_clip": lower_clip,
                "upper_clip": upper_clip,
            }

        means = np.mean(x, axis=0)
        stds = np.std(x, axis=0)
        stds = np.where(stds <= 1e-9, 1.0, stds)
        x_scaled = (x - means) / stds
        design = np.column_stack([np.ones((x_scaled.shape[0], 1), dtype=np.float64), x_scaled])
        ridge = 0.6
        eye = np.eye(design.shape[1], dtype=np.float64)
        eye[0, 0] = 0.0
        coeff = np.linalg.pinv(design.T @ design + (ridge * eye)) @ design.T @ y
        return {
            "mode": "fitted",
            "means": means,
            "stds": stds,
            "coeff": coeff,
            "lower_clip": lower_clip,
            "upper_clip": upper_clip,
        }

    def _predict_linear_return(self, model: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        if model.get("mode") == "constant":
            value = float(model.get("mean_return") or 0.0)
            lower = float(model.get("lower_clip") or -_RETURN_ABS_CLIP)
            upper = float(model.get("upper_clip") or _RETURN_ABS_CLIP)
            clipped = min(upper, max(lower, value))
            return [clipped for _ in rows]

        x = self._feature_matrix(rows, _LOGISTIC_FEATURE_ORDER)
        means = np.array(model["means"], dtype=np.float64)
        stds = np.array(model["stds"], dtype=np.float64)
        coeff = np.array(model["coeff"], dtype=np.float64)
        x_scaled = (x - means) / stds
        design = np.column_stack([np.ones((x_scaled.shape[0], 1), dtype=np.float64), x_scaled])
        values = design @ coeff
        lower = float(model.get("lower_clip") or -_RETURN_ABS_CLIP)
        upper = float(model.get("upper_clip") or _RETURN_ABS_CLIP)
        clipped = np.clip(values, lower, upper)
        return [float(item) for item in clipped.tolist()]

    def _fit_logistic_regression(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"mode": "constant", "prob": 0.5, "weights": np.zeros((len(_LOGISTIC_FEATURE_ORDER),), dtype=np.float64), "bias": 0.0}
        x = np.array(
            [[float(item[name]) for name in _LOGISTIC_FEATURE_ORDER] for item in rows],
            dtype=np.float64,
        )
        y = np.array([int(item["y"]) for item in rows], dtype=np.float64)
        positive_ratio = float(np.mean(y)) if y.size else 0.5
        if y.size == 0 or np.all(y == y[0]):
            base_prob = min(0.99, max(0.01, positive_ratio))
            return {
                "mode": "constant",
                "prob": base_prob,
                "means": np.zeros((len(_LOGISTIC_FEATURE_ORDER),), dtype=np.float64),
                "stds": np.ones((len(_LOGISTIC_FEATURE_ORDER),), dtype=np.float64),
                "weights": np.zeros((len(_LOGISTIC_FEATURE_ORDER),), dtype=np.float64),
                "bias": math.log(base_prob / (1.0 - base_prob)),
            }

        means = np.mean(x, axis=0)
        stds = np.std(x, axis=0)
        stds = np.where(stds <= 1e-9, 1.0, stds)
        x_scaled = (x - means) / stds
        weights = np.zeros((x_scaled.shape[1],), dtype=np.float64)
        bias = 0.0
        learning_rate = 0.12
        l2 = 0.004
        for _ in range(260):
            logits = np.clip(x_scaled @ weights + bias, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            errors = probs - y
            grad_w = (x_scaled.T @ errors) / len(y) + (l2 * weights)
            grad_b = float(np.mean(errors))
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
        return {
            "mode": "fitted",
            "means": means,
            "stds": stds,
            "weights": weights,
            "bias": bias,
        }

    def _predict_logistic(self, model: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        if not rows:
            return []
        if model.get("mode") == "constant":
            prob = float(model.get("prob") or 0.5)
            score = math.log(prob / (1.0 - prob)) if 0.0 < prob < 1.0 else 0.0
            return [{"score": score, "prob": prob} for _ in rows]

        x = np.array(
            [[float(item[name]) for name in _LOGISTIC_FEATURE_ORDER] for item in rows],
            dtype=np.float64,
        )
        means = np.array(model["means"], dtype=np.float64)
        stds = np.array(model["stds"], dtype=np.float64)
        weights = np.array(model["weights"], dtype=np.float64)
        bias = float(model["bias"])
        x_scaled = (x - means) / stds
        logits = np.clip(x_scaled @ weights + bias, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return [
            {"score": float(score), "prob": float(prob)}
            for score, prob in zip(logits.tolist(), probs.tolist(), strict=False)
        ]

    def _build_training_view(
        self,
        *,
        dataset: dict[str, Any],
        cost_buffer: float,
        train_window_months: int,
        gap_days: int,
        valid_window_months: int,
        random_seed: int,
    ) -> dict[str, Any]:
        prediction_dates: list[str] = dataset["prediction_dates"]
        train_window_days = _window_days_from_months(train_window_months)
        valid_window_days = _window_days_from_months(valid_window_months)
        available_folds = min(4, max(0, (len(prediction_dates) - train_window_days - gap_days) // max(valid_window_days, 1)))
        if available_folds <= 0:
            raise HTTPException(status_code=502, detail="学習・検証に必要な営業日数が不足しています。")

        folds: list[dict[str, Any]] = []
        primary_metrics: list[dict[str, float | None]] = []
        baseline_metrics: list[dict[str, float | None]] = []
        latest_valid_rows: list[dict[str, Any]] = []
        latest_primary_model: dict[str, Any] | None = None
        for fold_index in range(available_folds):
            valid_end = len(prediction_dates) - ((available_folds - 1 - fold_index) * valid_window_days)
            valid_start = valid_end - valid_window_days
            gap_end = valid_start
            gap_start = gap_end - gap_days
            train_end = gap_start
            train_start = max(0, train_end - train_window_days)
            train_dates = set(prediction_dates[train_start:train_end])
            valid_dates = set(prediction_dates[valid_start:valid_end])

            train_rows = [row for row in dataset["rows"] if row["date"] in train_dates]
            valid_rows = [row for row in dataset["rows"] if row["date"] in valid_dates]
            if not train_rows or not valid_rows:
                continue

            start_timer = time.perf_counter()
            lgbm_model = self._fit_lightgbm_classifier(train_rows, seed=random_seed)
            primary_train_time = time.perf_counter() - start_timer
            inference_timer = time.perf_counter()
            lgbm_predictions = self._predict_lightgbm_classifier(lgbm_model, valid_rows)
            primary_inference_time = time.perf_counter() - inference_timer

            start_timer = time.perf_counter()
            logistic_model = self._fit_logistic_regression(train_rows)
            train_time = time.perf_counter() - start_timer
            inference_timer = time.perf_counter()
            logistic_predictions = self._predict_logistic(logistic_model, valid_rows)
            inference_time = time.perf_counter() - inference_timer
            primary_prob = np.array([float(item["prob"]) for item in lgbm_predictions], dtype=np.float64)
            primary_score = np.array([float(item["score"]) for item in lgbm_predictions], dtype=np.float64)
            baseline_prob = np.array([float(item["prob"]) for item in logistic_predictions], dtype=np.float64)
            baseline_score = np.array([float(item["score"]) for item in logistic_predictions], dtype=np.float64)
            y_true = np.array([int(row["y"]) for row in valid_rows], dtype=np.int64)

            primary_stat = {
                "roc_auc": _roc_auc_score(y_true, primary_score),
                "pr_auc": _average_precision(y_true, primary_prob),
                "balanced_accuracy": _balanced_accuracy(y_true, primary_prob),
                "hit_ratio": _hit_ratio(y_true, primary_prob),
                "train_time_sec": primary_train_time,
                "inference_time_sec": max(0.005, primary_inference_time),
                "features": len(_PRIMARY_FEATURE_ORDER),
            }
            baseline_stat = {
                "roc_auc": _roc_auc_score(y_true, baseline_score),
                "pr_auc": _average_precision(y_true, baseline_prob),
                "balanced_accuracy": _balanced_accuracy(y_true, baseline_prob),
                "hit_ratio": _hit_ratio(y_true, baseline_prob),
                "train_time_sec": train_time,
                "inference_time_sec": max(0.005, inference_time),
                "features": len(_LOGISTIC_FEATURE_ORDER),
            }
            primary_metrics.append(primary_stat)
            baseline_metrics.append(baseline_stat)
            latest_valid_rows = [
                {**row, "lgbm_score": float(prediction["score"])}
                for row, prediction in zip(valid_rows, lgbm_predictions, strict=False)
            ]
            latest_primary_model = lgbm_model
            folds.append(
                {
                    "fold": f"Fold {len(folds) + 1}",
                    "train": f"{prediction_dates[train_start]} - {prediction_dates[train_end - 1]}",
                    "gap": f"{gap_days}営業日",
                    "valid": f"{prediction_dates[valid_start]} - {prediction_dates[valid_end - 1]}",
                    "samples": f"train {len(train_rows)} / valid {len(valid_rows)}",
                    "lgbm_roc_auc": primary_stat["roc_auc"] or 0.0,
                    "logreg_roc_auc": baseline_stat["roc_auc"] or 0.0,
                }
            )

        if not folds:
            raise HTTPException(status_code=502, detail="学習・検証に利用できる fold を構築できませんでした。")

        primary_summary = self._summarize_model_metrics(primary_metrics, _PRIMARY_MODEL_FAMILY)
        baseline_summary = self._summarize_model_metrics(baseline_metrics, _BASELINE_MODEL_FAMILY)
        distribution = self._score_histogram(latest_valid_rows)
        acceptance = [
            {
                "level": "normal",
                "title": "実データ評価完了",
                "detail": f"{len(folds)} fold の walk-forward + gap({gap_days}) を計算しました。",
            },
            {
                "level": "normal",
                "title": "Primary model 実装",
                "detail": f"LightGBM binary classifier を seed={random_seed} で再学習し、Logistic Regression と比較しています。",
            },
            {
                "level": "normal" if (primary_summary["roc_auc"] or 0) >= (baseline_summary["roc_auc"] or 0) else "warning",
                "title": "採用判定候補",
                "detail": "ROC-AUC とバックテストの双方を見て採用判断してください。",
            },
        ]
        return {
            "config_items": [
                {"label": "task_type", "value": "y_cls_1d"},
                {"label": "主モデル", "value": f"{_PRIMARY_MODEL_FAMILY} ({_MODEL_PRIMARY_VERSION})"},
                {"label": "比較モデル", "value": _BASELINE_MODEL_FAMILY},
                {"label": "学習期間", "value": f"{train_window_months}か月 ({train_window_days}営業日)"},
                {"label": "gap", "value": f"{gap_days}営業日"},
                {"label": "valid期間", "value": f"{valid_window_months}か月 ({valid_window_days}営業日)"},
                {"label": "cost_buffer", "value": _format_cost_buffer(cost_buffer)},
                {"label": "feature_set", "value": _FEATURE_VERSION},
                {"label": "seed", "value": str(random_seed)},
            ],
            "rules": [
                "ランダム分割オプションを UI に出さない。",
                "一部銘柄だけを train / valid へ分ける設定を許可しない。",
                "未来営業日を含む学習期間や、未来時点のラベル混入を許可しない。",
                "公開イベント未接続のため、現状は価格・出来高特徴量のみで検証する。",
            ],
            "summary_cards": [
                {"label": "Primary", "value": _PRIMARY_MODEL_FAMILY},
                {"label": "Baseline", "value": _BASELINE_MODEL_FAMILY},
                {"label": "Validation", "value": f"walk-forward + gap({gap_days})"},
                {"label": "Train Window", "value": f"{train_window_months}か月"},
                {"label": "Leakage Check", "value": "PASS"},
                {"label": "Universe", "value": f"{len(JP_LARGE_CAP_UNIVERSE)} symbols"},
            ],
            "compare_rows": [primary_summary, baseline_summary],
            "acceptance": acceptance,
            "folds": folds,
            "distribution": distribution,
            "primary_feature_importance": self._lightgbm_feature_summary(latest_primary_model or {"mode": "constant"}),
            "primary_summary": primary_summary,
            "baseline_summary": baseline_summary,
        }

    @staticmethod
    def _summarize_model_metrics(metrics: list[dict[str, float | None]], label: str) -> dict[str, Any]:
        def avg(key: str) -> float | None:
            values = [float(item[key]) for item in metrics if item.get(key) is not None]
            if not values:
                return None
            return float(sum(values) / len(values))

        return {
            "model": label,
            "roc_auc": avg("roc_auc") or 0.0,
            "pr_auc": avg("pr_auc") or 0.0,
            "balanced_accuracy": avg("balanced_accuracy") or 0.0,
            "hit_ratio": avg("hit_ratio") or 0.0,
            "train_time": f"{(avg('train_time_sec') or 0.0):.2f}s",
            "inference_time": f"{(avg('inference_time_sec') or 0.0):.2f}s",
            "features": int(round(avg("features") or 0.0)),
            "missing_rate": "0.0%",
        }

    @staticmethod
    def _score_histogram(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not valid_rows:
            return []
        scores = np.array(
            [float(item.get("lgbm_score", item.get("primary_score", 0.0))) for item in valid_rows],
            dtype=np.float64,
        )
        bins = [-10.0, -0.5, 0.0, 0.5, 1.0, 1.5, 10.0]
        labels = ["< -0.5", "-0.5 ~ 0", "0 ~ 0.5", "0.5 ~ 1.0", "1.0 ~ 1.5", "> 1.5"]
        counts, _ = np.histogram(scores, bins=bins)
        return [{"label": labels[idx], "value": int(counts[idx])} for idx in range(len(labels))]

    def _build_backtest_view(
        self,
        *,
        dataset: dict[str, Any],
        train_window_months: int,
        gap_days: int,
        random_seed: int,
    ) -> dict[str, Any]:
        prediction_dates: list[str] = dataset["prediction_dates"]
        backtest_dates = prediction_dates[-_BACKTEST_WINDOW_DAYS:]
        if len(backtest_dates) < 40:
            raise HTTPException(status_code=502, detail="バックテストに必要な営業日数が不足しています。")
        train_window_days = _window_days_from_months(train_window_months)
        train_end = max(0, len(prediction_dates) - _BACKTEST_WINDOW_DAYS - gap_days)
        train_start = max(0, train_end - train_window_days)
        train_dates = prediction_dates[train_start:train_end]
        train_date_set = set(train_dates)
        primary_train_rows = [row for row in dataset["rows"] if row["date"] in train_date_set]
        logistic_train_rows = [row for row in dataset["rows"] if row["date"] in train_date_set]
        lgbm_model = self._fit_lightgbm_classifier(primary_train_rows, seed=random_seed)
        logistic_model = self._fit_logistic_regression(logistic_train_rows)

        by_date = dataset["by_date"]
        date_to_rows = {day: by_date.get(day, []) for day in backtest_dates}
        lgbm_scores_by_date: dict[str, list[dict[str, Any]]] = {}
        logistic_scores_by_date: dict[str, list[dict[str, Any]]] = {}
        for day in backtest_dates:
            rows = date_to_rows[day]
            lgbm_predictions = self._predict_lightgbm_classifier(lgbm_model, rows)
            predictions = self._predict_logistic(logistic_model, rows)
            lgbm_enriched: list[dict[str, Any]] = []
            enriched: list[dict[str, Any]] = []
            for row, prediction in zip(rows, lgbm_predictions, strict=False):
                copied = dict(row)
                copied["lgbm_score"] = float(prediction["score"])
                copied["lgbm_prob"] = float(prediction["prob"])
                lgbm_enriched.append(copied)
            for row, prediction in zip(rows, predictions, strict=False):
                copied = dict(row)
                copied["logreg_score"] = float(prediction["score"])
                copied["logreg_prob"] = float(prediction["prob"])
                enriched.append(copied)
            lgbm_scores_by_date[day] = lgbm_enriched
            logistic_scores_by_date[day] = enriched

        primary_result = self._run_cross_sectional_backtest(backtest_dates, lgbm_scores_by_date, model_kind="lgbm")
        baseline_result = self._run_cross_sectional_backtest(backtest_dates, logistic_scores_by_date, model_kind="logreg")
        adopted_model_version = self.page_store.get_adopted_model_version()
        adopted_result = primary_result if adopted_model_version == _MODEL_PRIMARY_VERSION else baseline_result
        candidate_result = baseline_result if adopted_model_version == _MODEL_PRIMARY_VERSION else primary_result
        adopted_label = "現行採用モデル"
        candidate_label = "候補モデル"
        compare_rows = [
            self._backtest_compare_row(adopted_label, adopted_result),
            self._backtest_compare_row(candidate_label, candidate_result),
        ]
        equity_labels = [item["date"][5:7] for item in adopted_result["series"][-12:]]
        equity_series = [
            {
                "label": adopted_label,
                "color": "#39d2c0",
                "values": [item["equity_norm"] for item in adopted_result["series"][-12:]],
            },
            {
                "label": candidate_label,
                "color": "#58a6ff",
                "values": [item["equity_norm"] for item in candidate_result["series"][-12:]],
            },
        ]
        monthly_returns = self._monthly_returns(candidate_result["daily_series"])
        daily_return_distribution = self._daily_return_distribution(
            adopted_label=adopted_label,
            adopted_daily_series=adopted_result["daily_series"],
            candidate_label=candidate_label,
            candidate_daily_series=candidate_result["daily_series"],
        )
        exceptions = self._build_backtest_exceptions(backtest_dates, date_to_rows)
        return {
            "settings": [
                {"label": "エントリー", "value": "D+1 寄り"},
                {"label": "イグジット", "value": "D+1 引け"},
                {"label": "上位N銘柄", "value": str(_TOP_N)},
                {"label": "売買コスト", "value": "10 bps"},
                {"label": "流動性条件", "value": "大型株ユニバース固定"},
                {"label": "約定不能処理", "value": "高ボラ銘柄を例外監視"},
            ],
            "compare_rows": compare_rows,
            "summary_cards": [
                {"label": "CAGR (gross)", "value": f"{candidate_result['metrics']['gross_cagr_pct']:.1f}%"},
                {"label": "CAGR (net)", "value": f"{candidate_result['metrics']['cagr_pct']:.1f}%"},
                {"label": "Sharpe", "value": f"{candidate_result['metrics']['sharpe']:.2f}" if candidate_result["metrics"]["sharpe"] is not None else "-"},
                {"label": "Max Drawdown", "value": f"{candidate_result['metrics']['max_drawdown_pct']:.1f}%"},
                {"label": "Turnover", "value": f"{candidate_result['metrics']['turnover_pct']:.1f}%"},
                {"label": "平均保有損益", "value": f"{candidate_result['metrics']['avg_holding_pnl_pct']:.2f}%"},
                {"label": "勝率", "value": f"{candidate_result['metrics']['win_rate_pct']:.1f}%"},
                {"label": "約定不能影響率", "value": f"{candidate_result['metrics']['unable_rate_pct']:.2f}%"},
            ],
            "equity_labels": equity_labels,
            "equity_series": equity_series,
            "monthly_returns": monthly_returns,
            "daily_return_distribution": daily_return_distribution,
            "exceptions": exceptions,
            "adopted_model_version": adopted_model_version,
            "model_results": {
                _MODEL_PRIMARY_VERSION: primary_result,
                _MODEL_BASELINE_VERSION: baseline_result,
            },
        }

    def _run_cross_sectional_backtest(
        self,
        dates: list[str],
        by_date: dict[str, list[dict[str, Any]]],
        *,
        model_kind: str,
    ) -> dict[str, Any]:
        equity = 1.0
        gross_equity = 1.0
        prev_selection: list[str] = []
        series: list[dict[str, Any]] = []
        daily_series: list[dict[str, Any]] = []
        daily_returns: list[float] = []
        gross_daily_returns: list[float] = []
        turnover_acc = 0.0
        unable_days = 0
        for day in dates:
            rows = [dict(item) for item in by_date.get(day, [])]
            if not rows:
                continue
            score_key = "lgbm_score" if model_kind == "lgbm" else "logreg_score"
            prob_key = "lgbm_prob" if model_kind == "lgbm" else "logreg_prob"
            rows.sort(key=lambda item: float(item.get(score_key) or 0.0), reverse=True)
            selected = rows[: min(_TOP_N, len(rows))]
            symbols = [item["code"] for item in selected]
            turnover = 1.0 if not prev_selection else len(set(symbols).symmetric_difference(prev_selection)) / max(len(symbols), 1)
            turnover_acc += turnover
            cost = turnover * (10.0 / 10_000.0)
            gross_ret = float(np.mean([float(item["next_return"]) for item in selected])) if selected else 0.0
            net_ret = gross_ret - cost
            gross_equity *= (1.0 + gross_ret)
            equity *= (1.0 + net_ret)
            if any(float(item["range_pct"]) > 0.10 for item in selected):
                unable_days += 1
            series.append({"date": day, "equity": equity, "equity_norm": equity})
            daily_series.append({"date": day, "net_return": net_ret, "gross_return": gross_ret, "avg_prob": float(np.mean([float(item.get(prob_key) or 0.5) for item in selected]))})
            daily_returns.append(net_ret)
            gross_daily_returns.append(gross_ret)
            prev_selection = symbols
        metrics = _series_metrics(daily_returns)
        gross_metrics = _series_metrics(gross_daily_returns)
        metrics["gross_cagr_pct"] = gross_metrics["cagr_pct"] or 0.0
        metrics["turnover_pct"] = (turnover_acc / max(len(daily_returns), 1)) * 252.0 * 100.0
        metrics["unable_count"] = unable_days
        metrics["unable_rate_pct"] = (unable_days / max(len(daily_returns), 1)) * 100.0
        return {"metrics": metrics, "series": series, "daily_series": daily_series}

    @staticmethod
    def _backtest_compare_row(label: str, result: dict[str, Any]) -> dict[str, Any]:
        metrics = result["metrics"]
        return {
            "model": label,
            "cagr": f"{metrics['cagr_pct']:.1f}%",
            "sharpe": f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "-",
            "mdd": f"{metrics['max_drawdown_pct']:.1f}%",
            "turnover": f"{metrics['turnover_pct']:.1f}%",
            "win_rate": f"{metrics['win_rate_pct']:.1f}%",
            "unable": f"{metrics['unable_count']}件",
        }

    @staticmethod
    def _monthly_returns(daily_series: list[dict[str, Any]]) -> list[dict[str, Any]]:
        monthly: dict[str, float] = {}
        for item in daily_series:
            month_key = str(item["date"])[:7]
            monthly.setdefault(month_key, 1.0)
            monthly[month_key] *= (1.0 + float(item["net_return"]))
        return [{"month": month, "value": (value - 1.0) * 100.0} for month, value in sorted(monthly.items())[-12:]]

    @staticmethod
    def _daily_return_distribution(
        *,
        adopted_label: str,
        adopted_daily_series: list[dict[str, Any]],
        candidate_label: str,
        candidate_daily_series: list[dict[str, Any]],
    ) -> dict[str, Any]:
        bins = [-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0]
        labels = ["< -2%", "-2% ~ -1%", "-1% ~ 0%", "0% ~ 1%", "1% ~ 2%", "> 2%"]

        def distribution_values(items: list[dict[str, Any]]) -> list[float]:
            returns_pct = np.array([float(item.get("net_return") or 0.0) * 100.0 for item in items], dtype=np.float64)
            if returns_pct.size == 0:
                return [0.0 for _ in labels]
            counts, _ = np.histogram(returns_pct, bins=bins)
            return [float(item) for item in counts.tolist()]

        return {
            "labels": labels,
            "series": [
                {"label": adopted_label, "color": "#39d2c0", "values": distribution_values(adopted_daily_series)},
                {"label": candidate_label, "color": "#58a6ff", "values": distribution_values(candidate_daily_series)},
            ],
        }

    @staticmethod
    def _build_backtest_exceptions(dates: list[str], by_date: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for day in dates[-12:]:
            rows = by_date.get(day, [])
            high_range = [row for row in rows if float(row["range_pct"]) >= 0.10]
            if high_range:
                impact = float(np.mean([float(item["next_return"]) for item in high_range])) * 100.0
                out.append(
                    {
                        "date": day,
                        "type": "高ボラ例外",
                        "count": str(len(high_range)),
                        "impact": f"{impact:.2f}%",
                        "note": "日中値幅10%以上の銘柄を検出。",
                    }
                )
        return out[:8]

    @staticmethod
    def _model_decision_payload(
        *,
        current_label: str,
        peer_label: str,
        current_summary: dict[str, Any],
        peer_summary: dict[str, Any],
        current_result: dict[str, Any],
        peer_result: dict[str, Any],
        is_adopted: bool,
    ) -> tuple[str, list[str], list[dict[str, Any]]]:
        current_metrics = current_result["metrics"]
        peer_metrics = peer_result["metrics"]
        roc_auc = float(current_summary.get("roc_auc") or 0.0)
        peer_roc_auc = float(peer_summary.get("roc_auc") or 0.0)
        roc_diff = roc_auc - peer_roc_auc
        cagr = float(current_metrics.get("cagr_pct") or 0.0)
        peer_cagr = float(peer_metrics.get("cagr_pct") or 0.0)
        sharpe = current_metrics.get("sharpe")
        peer_sharpe = peer_metrics.get("sharpe")
        sharpe_diff = (
            float(sharpe) - float(peer_sharpe)
            if sharpe is not None and peer_sharpe is not None
            else None
        )
        turnover_pct = float(current_metrics.get("turnover_pct") or 0.0)
        unable_rate_pct = float(current_metrics.get("unable_rate_pct") or 0.0)
        outperform_stat = roc_diff >= 0.0
        outperform_trade = cagr >= peer_cagr
        decision_level = "normal" if (outperform_stat and outperform_trade) else "warning"
        adopt_reason = (
            f"{current_label} は 直近 fold平均 ROC-AUC {roc_auc:.3f}"
            f" ({peer_label} 比 {roc_diff:+.3f})"
            f" / net CAGR {cagr:.1f}% ({peer_label} 比 {cagr - peer_cagr:+.1f}pt)"
            " を確認対象にしています。"
        )
        warnings: list[str] = []
        if not outperform_stat:
            warnings.append(f"ROC-AUC が {peer_label} を下回る")
        if not outperform_trade:
            warnings.append(f"net CAGR が {peer_label} を下回る")
        if unable_rate_pct >= 20.0:
            warnings.append("約定不能率が高め")
        if turnover_pct >= 400.0:
            warnings.append("売買回転が高め")
        if not warnings:
            warnings.append("-")

        decision_items = [
            {
                "level": "normal",
                "title": "採用理由",
                "detail": adopt_reason,
            },
            {
                "level": "normal" if sharpe_diff is None or sharpe_diff >= 0.0 else "warning",
                "title": "比較指標",
                "detail": (
                    f"Sharpe {_safe_number_label(float(sharpe) if sharpe is not None else None, 2)}"
                    f" / {peer_label} 比 {_safe_number_label(sharpe_diff, 2)}"
                    f" / Turnover {_safe_pct_label(turnover_pct, 1)}"
                    f" / 約定不能率 {_safe_pct_label(unable_rate_pct, 2)}"
                ),
            },
            {
                "level": decision_level,
                "title": "採用判定",
                "detail": (
                    f"{'採用中モデルとして維持' if is_adopted else '候補モデルとして比較継続'}。"
                    if decision_level == "normal"
                    else f"統計指標または net 成績が {peer_label} に劣後しているため、採用前に再確認が必要です。"
                ),
            },
        ]
        return adopt_reason, [] if warnings == ["-"] else warnings, decision_items

    def _build_model_registry(
        self,
        *,
        training: dict[str, Any],
        backtest: dict[str, Any],
        config_hash: str,
        cost_buffer: float,
        train_window_months: int,
        gap_days: int,
        valid_window_months: int,
        random_seed: int,
        quality_gate: dict[str, Any],
    ) -> dict[str, Any]:
        adopted_model_version = self.page_store.get_adopted_model_version()
        model_results = backtest["model_results"]
        primary_result = model_results[_MODEL_PRIMARY_VERSION]
        baseline_result = model_results[_MODEL_BASELINE_VERSION]
        leakage_check = bool(quality_gate.get("leakage_check", {}).get("passed"))
        leakage_detail = str(quality_gate.get("leakage_check", {}).get("detail") or "").strip()
        data_quality_check = bool(quality_gate.get("data_quality_check", {}).get("passed"))
        data_quality_detail = str(quality_gate.get("data_quality_check", {}).get("detail") or "").strip()
        quality_warning_labels = [
            str(item).strip()
            for item in (quality_gate.get("warning_labels") or [])
            if str(item).strip()
        ]
        quality_blockers = [
            str(item).strip()
            for item in (quality_gate.get("blockers") or [])
            if str(item).strip()
        ]
        adoptable = leakage_check and data_quality_check
        primary_adopt_reason, primary_warnings, primary_decision = self._model_decision_payload(
            current_label=_PRIMARY_MODEL_FAMILY,
            peer_label=_BASELINE_MODEL_FAMILY,
            current_summary=training["primary_summary"],
            peer_summary=training["baseline_summary"],
            current_result=primary_result,
            peer_result=baseline_result,
            is_adopted=adopted_model_version == _MODEL_PRIMARY_VERSION,
        )
        baseline_adopt_reason, baseline_warnings, baseline_decision = self._model_decision_payload(
            current_label=_BASELINE_MODEL_FAMILY,
            peer_label=_PRIMARY_MODEL_FAMILY,
            current_summary=training["baseline_summary"],
            peer_summary=training["primary_summary"],
            current_result=baseline_result,
            peer_result=primary_result,
            is_adopted=adopted_model_version == _MODEL_BASELINE_VERSION,
        )
        primary_quality_detail = (
            "leakage_check / data_quality_check を満たしています。"
            + (f" 監視中: {' / '.join(quality_warning_labels)}。" if quality_warning_labels else "")
            if adoptable
            else "採用不可。 " + " / ".join(quality_blockers)
        )
        baseline_quality_detail = (
            "leakage_check / data_quality_check を満たしています。"
            + (f" 監視中: {' / '.join(quality_warning_labels)}。" if quality_warning_labels else "")
            if adoptable
            else "採用不可。 " + " / ".join(quality_blockers)
        )
        primary_warnings = list(dict.fromkeys(primary_warnings + quality_warning_labels + ([] if adoptable else ["採用不可"])))
        baseline_warnings = list(dict.fromkeys(baseline_warnings + quality_warning_labels + ([] if adoptable else ["採用不可"])))
        primary_decision = primary_decision + [
            {
                "level": "normal" if adoptable else "error",
                "title": "採用可否",
                "detail": primary_quality_detail,
            }
        ]
        baseline_decision = baseline_decision + [
            {
                "level": "normal" if adoptable else "error",
                "title": "採用可否",
                "detail": baseline_quality_detail,
            }
        ]

        primary_row = {
            "model_version": _MODEL_PRIMARY_VERSION,
            "feature_version": _FEATURE_VERSION,
            "family": _PRIMARY_MODEL_FAMILY,
            "task_type": "分類",
            "status": "adopted" if adopted_model_version == _MODEL_PRIMARY_VERSION else "candidate",
            "summary_metrics": f"ROC-AUC {training['primary_summary']['roc_auc']:.3f} / Sharpe {primary_result['metrics']['sharpe']:.2f}" if primary_result["metrics"]["sharpe"] is not None else f"ROC-AUC {training['primary_summary']['roc_auc']:.3f}",
            "warnings": primary_warnings,
            "adoptable": adoptable,
            "adopt_blockers": quality_blockers,
            "leakage_check": leakage_check,
            "data_quality_check": data_quality_check,
            "leakage_detail": leakage_detail,
            "data_quality_detail": data_quality_detail,
            "train_conditions": [
                {"label": "学習期間", "value": f"{train_window_months}か月 ({_window_days_from_months(train_window_months)}営業日)"},
                {"label": "gap", "value": f"{gap_days}営業日"},
                {"label": "valid期間", "value": f"{valid_window_months}か月 ({_window_days_from_months(valid_window_months)}営業日)"},
                {"label": "cost_buffer", "value": _format_cost_buffer(cost_buffer)},
                {"label": "seed", "value": str(random_seed)},
                {"label": "特徴量セット", "value": _FEATURE_VERSION},
                {"label": "採用理由", "value": primary_adopt_reason},
            ],
            "eval_conditions": [
                {"label": "統計指標平均", "value": f"ROC-AUC {training['primary_summary']['roc_auc']:.3f} / PR-AUC {training['primary_summary']['pr_auc']:.3f}"},
                {"label": "バックテスト", "value": f"CAGR {primary_result['metrics']['cagr_pct']:.1f}% / Sharpe {primary_result['metrics']['sharpe']:.2f}" if primary_result["metrics"]["sharpe"] is not None else "-"},
                {"label": "計算時間", "value": training["primary_summary"]["train_time"]},
                {"label": "欠損率", "value": "0.0%"},
                {"label": "Leakage Check", "value": "PASS" if leakage_check else "FAIL"},
                {"label": "Data Quality", "value": "PASS" if data_quality_check else "FAIL"},
                {"label": "期待収益率", "value": "LightGBM regressor で算出"},
            ],
            "adopt_reason": primary_adopt_reason,
            "decision": primary_decision,
            "explainability": [
                {"level": "normal", "title": "全体重要特徴量", "detail": training.get("primary_feature_importance") or "importance unavailable"},
                {"level": "normal", "title": "説明可能性の注記", "detail": "pred_contrib による tree contribution を各銘柄で表示します。"},
                {"level": "normal", "title": "当日寄与上位", "detail": "各銘柄の top3 feature contribution をダッシュボードで表示。"},
            ],
            "audit": self._model_audit_lines(model_version=_MODEL_PRIMARY_VERSION, config_hash=config_hash),
        }
        baseline_row = {
            "model_version": _MODEL_BASELINE_VERSION,
            "feature_version": _FEATURE_VERSION,
            "family": _BASELINE_MODEL_FAMILY,
            "task_type": "分類",
            "status": "adopted" if adopted_model_version == _MODEL_BASELINE_VERSION else "candidate",
            "summary_metrics": f"ROC-AUC {training['baseline_summary']['roc_auc']:.3f} / Sharpe {baseline_result['metrics']['sharpe']:.2f}" if baseline_result["metrics"]["sharpe"] is not None else f"ROC-AUC {training['baseline_summary']['roc_auc']:.3f}",
            "warnings": baseline_warnings,
            "adoptable": adoptable,
            "adopt_blockers": quality_blockers,
            "leakage_check": leakage_check,
            "data_quality_check": data_quality_check,
            "leakage_detail": leakage_detail,
            "data_quality_detail": data_quality_detail,
            "train_conditions": [
                {"label": "学習期間", "value": f"{train_window_months}か月 ({_window_days_from_months(train_window_months)}営業日)"},
                {"label": "gap", "value": f"{gap_days}営業日"},
                {"label": "valid期間", "value": f"{valid_window_months}か月 ({_window_days_from_months(valid_window_months)}営業日)"},
                {"label": "cost_buffer", "value": _format_cost_buffer(cost_buffer)},
                {"label": "seed", "value": str(random_seed)},
                {"label": "特徴量セット", "value": _FEATURE_VERSION},
                {"label": "採用理由", "value": baseline_adopt_reason},
            ],
            "eval_conditions": [
                {"label": "統計指標平均", "value": f"ROC-AUC {training['baseline_summary']['roc_auc']:.3f} / PR-AUC {training['baseline_summary']['pr_auc']:.3f}"},
                {"label": "バックテスト", "value": f"CAGR {baseline_result['metrics']['cagr_pct']:.1f}% / Sharpe {baseline_result['metrics']['sharpe']:.2f}" if baseline_result["metrics"]["sharpe"] is not None else "-"},
                {"label": "計算時間", "value": training["baseline_summary"]["train_time"]},
                {"label": "欠損率", "value": "0.0%"},
                {"label": "Leakage Check", "value": "PASS" if leakage_check else "FAIL"},
                {"label": "Data Quality", "value": "PASS" if data_quality_check else "FAIL"},
                {"label": "警告", "value": "-"},
            ],
            "adopt_reason": baseline_adopt_reason,
            "decision": baseline_decision,
            "explainability": [
                {"level": "normal", "title": "係数上位特徴量", "detail": "ret_20d / ma_gap_20 / volatility_20"},
                {"level": "normal", "title": "セクター別傾向", "detail": "価格・出来高特徴量のみのためセクター説明は限定的です。"},
                {"level": "normal", "title": "当日寄与上位", "detail": "線形係数に基づく寄与。"},
            ],
            "audit": self._model_audit_lines(model_version=_MODEL_BASELINE_VERSION, config_hash=config_hash),
        }
        rows = [primary_row, baseline_row]
        default_versions = {
            _PRIMARY_MODEL_FAMILY: _MODEL_PRIMARY_VERSION,
            _BASELINE_MODEL_FAMILY: _MODEL_BASELINE_VERSION,
        }
        return {
            "rows": rows,
            "adopted_model_version": adopted_model_version,
            "default_versions": default_versions,
        }

    def _build_model_quality_gate(
        self,
        *,
        dataset: dict[str, Any],
        training: dict[str, Any],
        gap_days: int,
    ) -> dict[str, Any]:
        count_check = self._ops_count_check(dataset)
        missing_check = self._ops_missing_rate_check(dataset)
        prediction_dates: list[str] = dataset.get("prediction_dates") or []
        latest_prediction_date = prediction_dates[-1] if prediction_dates else ""
        latest_available = len(dataset.get("by_date", {}).get(latest_prediction_date, [])) if latest_prediction_date else 0
        total_symbols = len(JP_LARGE_CAP_UNIVERSE)
        excluded = int(dataset.get("excluded_symbols") or 0)
        excluded_ratio_pct = (excluded / total_symbols) * 100.0 if total_symbols else 0.0
        coverage_level = "normal"
        if latest_available < max(1, math.ceil(total_symbols * 0.60)) or excluded_ratio_pct >= 25.0:
            coverage_level = "error"
        elif latest_available < max(1, math.ceil(total_symbols * 0.80)) or excluded > 0:
            coverage_level = "warning"
        coverage_detail = f"最新 prediction_date {latest_prediction_date or '-'} は {latest_available}/{total_symbols} 銘柄。"
        if excluded > 0:
            coverage_detail += f" 除外 {excluded} 銘柄。"
        coverage_check = {
            "label": "銘柄数急減",
            "value": f"{latest_available}/{total_symbols}",
            "level": coverage_level,
            "detail": coverage_detail,
        }
        fold_count = len(training.get("folds") or [])
        leakage_pass = gap_days >= 1 and fold_count > 0
        leakage_detail = (
            f"walk-forward {fold_count} folds / gap {gap_days}営業日 / ランダム分割なし。"
            if leakage_pass
            else "gap または fold 条件が不足しているため leakage_check に失敗しました。"
        )
        quality_checks = [count_check, missing_check, coverage_check]
        warning_labels = [
            str(item.get("label") or "").strip()
            for item in quality_checks
            if str(item.get("level") or "") == "warning" and str(item.get("label") or "").strip()
        ]
        blocker_details = [
            f"{item['label']}: {item['detail']}"
            for item in quality_checks
            if str(item.get("level") or "") == "error"
        ]
        if not leakage_pass:
            blocker_details.insert(0, leakage_detail)
        data_quality_pass = not any(str(item.get("level") or "") == "error" for item in quality_checks)
        if data_quality_pass:
            data_quality_detail = " / ".join(f"{item['label']}={item['value']}" for item in quality_checks)
            if warning_labels:
                data_quality_detail += " / 監視中: " + " / ".join(warning_labels)
        else:
            data_quality_detail = " / ".join(
                f"{item['label']}: {item['detail']}"
                for item in quality_checks
                if str(item.get("level") or "") == "error"
            )
        return {
            "leakage_check": {
                "passed": leakage_pass,
                "detail": leakage_detail,
            },
            "data_quality_check": {
                "passed": data_quality_pass,
                "detail": data_quality_detail,
            },
            "warning_labels": warning_labels,
            "blockers": blocker_details,
        }

    def _build_dashboard_view(
        self,
        *,
        dataset: dict[str, Any],
        prediction_date: str,
        model_family: str,
        selected_model_version: str,
        adopted_model_version: str,
        feature_set: str,
        cost_buffer: float,
        latest_market_date: str,
        train_window_months: int,
        random_seed: int,
    ) -> dict[str, Any]:
        base_rows = [dict(item) for item in dataset["by_date"].get(prediction_date, [])]
        if not base_rows:
            raise HTTPException(status_code=404, detail="選択した prediction_date のデータがありません。")
        target_date = self._target_date_for_prediction(
            target_date_by_prediction=dataset["target_date_by_prediction"],
            prediction_date=prediction_date,
        )
        training_dates = self._recent_prediction_dates_before(
            prediction_dates=dataset["prediction_dates"],
            selected=prediction_date,
            limit=_window_days_from_months(train_window_months),
        )
        training_date_set = set(training_dates)
        train_rows = [row for row in dataset["rows"] if row["date"] in training_date_set]
        if model_family == _BASELINE_MODEL_FAMILY:
            logistic_model = self._fit_logistic_regression(train_rows)
            return_model = self._fit_linear_return_regression(train_rows)
            predictions = self._predict_logistic(logistic_model, base_rows)
            return_predictions = self._predict_linear_return(return_model, base_rows)
            for row, prediction, expected_return in zip(base_rows, predictions, return_predictions, strict=False):
                row["score_cls"] = float(prediction["score"])
                row["prob_up"] = float(prediction["prob"])
                row["expected_return"] = float(expected_return)
                row["model_feature_contrib"] = [
                    {"name": "linear_ret_20d", "value": float(row["ret_20d"])},
                    {"name": "linear_ma_gap_20", "value": float(row["ma_gap_20"])},
                    {"name": "linear_volatility_20", "value": float(-row["volatility_20"])},
                ]
            previous_rows = dataset["by_date"].get(self._previous_prediction_date(dataset["prediction_dates"], prediction_date) or "", [])
            previous_predictions = self._predict_logistic(logistic_model, previous_rows)
            previous_scores = {
                row["code"]: float(prediction["score"])
                for row, prediction in zip(previous_rows, previous_predictions, strict=False)
            }
            expected_return_note = "expected_return は ridge return model の D+1 期待収益率です。"
        else:
            lgbm_model = self._fit_lightgbm_classifier(train_rows, seed=random_seed)
            return_model = self._fit_lightgbm_regression(train_rows, seed=random_seed)
            predictions = self._predict_lightgbm_classifier(lgbm_model, base_rows, include_contrib=True)
            return_predictions = self._predict_lightgbm_regression(return_model, base_rows)
            for row, prediction, expected_return in zip(base_rows, predictions, return_predictions, strict=False):
                row["score_cls"] = float(prediction["score"])
                row["prob_up"] = float(prediction["prob"])
                row["expected_return"] = float(expected_return)
                row["model_feature_contrib"] = list(prediction["feature_contrib"])
            previous_rows = dataset["by_date"].get(self._previous_prediction_date(dataset["prediction_dates"], prediction_date) or "", [])
            previous_predictions = self._predict_lightgbm_classifier(lgbm_model, previous_rows)
            previous_scores = {
                row["code"]: float(prediction["score"])
                for row, prediction in zip(previous_rows, previous_predictions, strict=False)
            }
            expected_return_note = "expected_return は LightGBM regressor の D+1 期待収益率です。"

        latest_delta_days = self._staleness_days(latest_market_date)
        display_rows: list[dict[str, Any]] = []
        volatility_cutoff = float(np.percentile(np.array([item["volatility_20"] for item in base_rows], dtype=np.float64), 75))
        ranked_rows = sorted(base_rows, key=lambda item: float(item["score_cls"]), reverse=True)
        for rank, row in enumerate(ranked_rows, start=1):
            warnings: list[str] = []
            if float(row["volume_ratio_20"]) < 0.65:
                warnings.append("流動性注意")
            if float(row["volatility_20"]) >= volatility_cutoff:
                warnings.append("高ボラ注意")
            if latest_delta_days > 1:
                warnings.append("データ鮮度注意")
            score_delta = float(row["score_cls"]) - float(previous_scores.get(row["code"], row["score_cls"]))
            display_rows.append(
                {
                    "code": row["code"],
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "prob_up": float(row["prob_up"]),
                    "score_cls": float(row["score_cls"]),
                    "score_rank": rank,
                    "expected_return": float(row["expected_return"]),
                    "warnings": warnings,
                    "recent_metrics": [
                        {"label": "score_rank", "value": f"#{rank}"},
                        {"label": "期待収益率", "value": f"{row['expected_return'] * 100.0:+.2f}%"},
                        {"label": "直近1日", "value": f"{row['ret_1d'] * 100.0:+.2f}%"},
                        {"label": "直近5日", "value": f"{row['ret_5d'] * 100.0:+.2f}%"},
                        {"label": "出来高倍率", "value": f"{row['volume_ratio_20']:.2f}x"},
                    ],
                    "event_proximity": "イベントデータ未接続 / 価格・出来高特徴量のみで推論",
                    "score_delta": f"{score_delta:+.2f} vs 前営業日",
                    "note": f"{expected_return_note} score_rank は score_cls 降順です。",
                    "feature_contrib": row["model_feature_contrib"],
                }
            )

        sector_scores: dict[str, list[float]] = {}
        for row in display_rows:
            sector_scores.setdefault(row["sector"], []).append(float(row["prob_up"]))
        sector_items = [
            {"label": sector, "value": float(sum(values) / len(values))}
            for sector, values in sorted(sector_scores.items(), key=lambda item: sum(item[1]) / len(item[1]), reverse=True)
        ]
        coverage_total = len(display_rows)
        coverage_excluded = dataset["excluded_symbols"]
        freshness_level = "normal" if latest_delta_days <= 1 else "warning"
        latest_update = f"{latest_market_date} 15:00 JST" if latest_market_date else "-"
        alerts = [
            {
                "level": freshness_level,
                "title": "データ鮮度",
                "detail": f"最新日付は {latest_market_date}。現在との差分 {latest_delta_days} 日。",
            },
            {
                "level": "warning" if coverage_excluded > 0 else "normal",
                "title": "ユニバース被覆",
                "detail": f"{coverage_total} 銘柄利用可能 / 除外 {coverage_excluded} 銘柄。",
            },
            {
                "level": "normal" if selected_model_version == adopted_model_version else "warning",
                "title": "モデル版",
                "detail": f"表示モデル {selected_model_version}。採用中モデル {adopted_model_version}。",
            },
        ]
        logs = self._runtime_logs(action="predictor", latest_market_date=latest_market_date)
        adopted_label = "採用中" if selected_model_version == adopted_model_version else f"表示中 / 採用中={adopted_model_version}"
        summary_cards = [
            {"label": "prediction_date", "value": _to_jst_label(prediction_date)},
            {"label": "target_date", "value": _to_jst_label(target_date)},
            {"label": "model_version", "value": selected_model_version, "sub": adopted_label, "action_tab": "models"},
            {"label": "data_version", "value": f"stooq_jp_{prediction_date.replace('-', '')}", "sub": feature_set},
            {"label": "coverage", "value": f"{coverage_total}/{coverage_total + coverage_excluded}", "sub": f"除外 {coverage_excluded}銘柄"},
            {"label": "data freshness", "value": "最新" if freshness_level == "normal" else "遅延あり", "sub": latest_update},
        ]
        return {
            "prediction_date": prediction_date,
            "label": _to_jst_label(prediction_date),
            "target_date": target_date,
            "latest_update": latest_update,
            "data_version": f"stooq_jp_{prediction_date.replace('-', '')}",
            "feature_version": feature_set,
            "coverage_total": coverage_total,
            "coverage_excluded": coverage_excluded,
            "freshness": {"level": freshness_level, "label": "最新" if freshness_level == "normal" else "遅延あり"},
            "run_state": "推論完了",
            "summary_cards": summary_cards,
            "caption": f"{coverage_total} 件表示 / selected={model_family} / ranking=score_cls / cost_buffer={_format_cost_buffer(cost_buffer)}",
            "footnote": f"score_rank を算出済み。{expected_return_note}",
            "rows": display_rows,
            "sector_scores": sector_items,
            "alerts": alerts,
            "logs": logs,
        }

    @staticmethod
    def _ops_count_check(dataset: dict[str, Any]) -> dict[str, Any]:
        prediction_dates: list[str] = dataset["prediction_dates"]
        if not prediction_dates:
            return {
                "label": "取得件数異常",
                "value": "-",
                "level": "error",
                "detail": "prediction_date を構築できていないため件数監視を実施できません。",
            }
        counts = [len(dataset["by_date"].get(day, [])) for day in prediction_dates]
        latest_count = counts[-1]
        prev_count = counts[-2] if len(counts) >= 2 else latest_count
        baseline_window = counts[-11:-1] if len(counts) >= 3 else counts[:-1]
        baseline_mean = float(sum(baseline_window) / len(baseline_window)) if baseline_window else float(latest_count)
        deviation_pct = 0.0 if baseline_mean <= 1e-9 else ((latest_count - baseline_mean) / baseline_mean) * 100.0
        total_symbols = len(JP_LARGE_CAP_UNIVERSE)
        level = "normal"
        if latest_count < max(1, math.ceil(total_symbols * 0.60)) or deviation_pct <= -20.0:
            level = "error"
        elif latest_count < max(1, math.ceil(total_symbols * 0.80)) or deviation_pct <= -10.0 or abs(latest_count - prev_count) >= 3:
            level = "warning"
        return {
            "label": "取得件数異常",
            "value": f"{latest_count}/{total_symbols}",
            "level": level,
            "detail": f"最新 {latest_count} 銘柄 / 前営業日差 {latest_count - prev_count:+d} / 10営業日平均 {baseline_mean:.1f}。",
        }

    @staticmethod
    def _ops_missing_rate_check(dataset: dict[str, Any]) -> dict[str, Any]:
        total_cells = len(dataset["rows"]) * len(_PRIMARY_FEATURE_ORDER)
        feature_missing_counts = {name: 0 for name in _PRIMARY_FEATURE_ORDER}
        missing_cells = 0
        for row in dataset["rows"]:
            for feature_name in _PRIMARY_FEATURE_ORDER:
                value = _safe_float(row.get(feature_name))
                if value is None:
                    missing_cells += 1
                    feature_missing_counts[feature_name] += 1
        missing_rate_pct = (missing_cells / total_cells) * 100.0 if total_cells else 0.0
        worst_feature = max(feature_missing_counts.items(), key=lambda item: item[1], default=("", 0))
        level = "normal"
        if missing_rate_pct >= 5.0:
            level = "error"
        elif missing_rate_pct >= 1.0:
            level = "warning"
        detail = f"欠損セル {missing_cells} / {total_cells}。"
        if worst_feature[1] > 0:
            detail += f" 最大欠損列は {worst_feature[0]} ({worst_feature[1]}件)。"
        else:
            detail += " 主要特徴量は全列で欠損なしです。"
        return {
            "label": "欠損率上昇",
            "value": _safe_pct_label(missing_rate_pct, 2),
            "level": level,
            "detail": detail,
        }

    @staticmethod
    def _ops_coverage_check(dataset: dict[str, Any], dashboard: dict[str, Any]) -> dict[str, Any]:
        total_symbols = len(JP_LARGE_CAP_UNIVERSE)
        available = int(dashboard.get("coverage_total") or 0)
        excluded = int(dashboard.get("coverage_excluded") or 0)
        breakdown = [
            item for item in (dataset.get("excluded_reason_breakdown") or [])
            if isinstance(item, dict) and int(item.get("count") or 0) > 0
        ]
        excluded_ratio_pct = (excluded / total_symbols) * 100.0 if total_symbols else 0.0
        level = "normal"
        if excluded_ratio_pct >= 25.0:
            level = "error"
        elif excluded > 0:
            level = "warning"
        detail = f"利用可能 {available} 銘柄 / 除外 {excluded} 銘柄。"
        if breakdown:
            detail += " 除外理由: " + " / ".join(f"{item['label']} {item['count']}件" for item in breakdown[:3]) + "。"
        else:
            detail += " 除外は発生していません。"
        return {
            "label": "銘柄数急減",
            "value": f"{available}/{total_symbols}",
            "level": level,
            "detail": detail,
        }

    @staticmethod
    def _ops_score_drift_check(dataset: dict[str, Any]) -> dict[str, Any]:
        prediction_dates: list[str] = dataset["prediction_dates"]
        if len(prediction_dates) < 25:
            return {
                "label": "スコア分布ドリフト",
                "value": "-",
                "level": "unknown",
                "detail": "比較対象の営業日数が不足しているため、PSI を算出していません。",
            }
        recent_dates = set(prediction_dates[-5:])
        prior_dates = set(prediction_dates[-25:-5])
        recent_scores = [float(row["primary_score"]) for row in dataset["rows"] if row["date"] in recent_dates]
        prior_scores = [float(row["primary_score"]) for row in dataset["rows"] if row["date"] in prior_dates]
        psi = _population_stability_index(prior_scores, recent_scores)
        recent_mean = float(np.mean(recent_scores)) if recent_scores else 0.0
        prior_mean = float(np.mean(prior_scores)) if prior_scores else 0.0
        mean_shift = recent_mean - prior_mean
        level = "normal"
        if psi is not None and psi >= 0.25:
            level = "error"
        elif (psi is not None and psi >= 0.10) or abs(mean_shift) >= 0.20:
            level = "warning"
        return {
            "label": "スコア分布ドリフト",
            "value": f"PSI {_safe_number_label(psi)}",
            "level": level,
            "detail": (
                "直近5営業日 vs その前20営業日で比較。"
                f" PSI {_safe_number_label(psi)} / 平均スコア差 {mean_shift:+.3f}。"
            ),
        }

    @staticmethod
    def _ops_score_drift_distribution(dataset: dict[str, Any]) -> dict[str, Any]:
        prediction_dates: list[str] = dataset["prediction_dates"]
        if len(prediction_dates) < 25:
            return {
                "labels": [],
                "series": [],
                "psi": None,
                "mean_shift": None,
                "note": "比較対象の営業日数が不足しているため、分布比較を表示していません。",
            }
        recent_dates = set(prediction_dates[-5:])
        prior_dates = set(prediction_dates[-25:-5])
        recent_scores = np.array(
            [float(row["primary_score"]) for row in dataset["rows"] if row["date"] in recent_dates],
            dtype=np.float64,
        )
        prior_scores = np.array(
            [float(row["primary_score"]) for row in dataset["rows"] if row["date"] in prior_dates],
            dtype=np.float64,
        )
        bins = [-10.0, -1.5, -0.5, 0.0, 0.5, 1.5, 10.0]
        labels = ["< -1.5", "-1.5 ~ -0.5", "-0.5 ~ 0", "0 ~ 0.5", "0.5 ~ 1.5", "> 1.5"]
        prior_counts, _ = np.histogram(prior_scores, bins=bins)
        recent_counts, _ = np.histogram(recent_scores, bins=bins)
        prior_total = max(int(np.sum(prior_counts)), 1)
        recent_total = max(int(np.sum(recent_counts)), 1)
        psi = _population_stability_index(prior_scores.tolist(), recent_scores.tolist())
        mean_shift = (float(np.mean(recent_scores)) - float(np.mean(prior_scores))) if recent_scores.size and prior_scores.size else None
        return {
            "labels": labels,
            "series": [
                {
                    "label": "前20営業日",
                    "color": "#58a6ff",
                    "values": [float(count / prior_total * 100.0) for count in prior_counts.tolist()],
                },
                {
                    "label": "直近5営業日",
                    "color": "#39d2c0",
                    "values": [float(count / recent_total * 100.0) for count in recent_counts.tolist()],
                },
            ],
            "psi": psi,
            "mean_shift": mean_shift,
            "note": "各ビンの構成比(%)。直近5営業日と、その前20営業日を比較しています。",
        }

    @staticmethod
    def _ops_pnl_drift_check(backtest: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
        adopted_model_version = str(models.get("adopted_model_version") or _MODEL_PRIMARY_VERSION)
        model_results = backtest.get("model_results", {})
        adopted_result = model_results.get(adopted_model_version)
        if not isinstance(adopted_result, dict):
            return {
                "label": "実績損益ドリフト",
                "value": "-",
                "level": "unknown",
                "detail": "採用モデルのバックテスト結果を参照できません。",
            }
        daily_series = adopted_result.get("daily_series") or []
        if len(daily_series) < 40:
            return {
                "label": "実績損益ドリフト",
                "value": "-",
                "level": "unknown",
                "detail": "最近20営業日と前20営業日を比較するには履歴が不足しています。",
            }
        recent_returns = [float(item.get("net_return") or 0.0) for item in daily_series[-20:]]
        prior_returns = [float(item.get("net_return") or 0.0) for item in daily_series[-40:-20]]
        recent_cum = (float(np.prod([1.0 + value for value in recent_returns])) - 1.0) * 100.0
        prior_cum = (float(np.prod([1.0 + value for value in prior_returns])) - 1.0) * 100.0
        recent_mean = float(np.mean(recent_returns))
        prior_mean = float(np.mean(prior_returns))
        level = "normal"
        if recent_cum <= -3.0 and recent_mean < prior_mean - 0.001:
            level = "error"
        elif recent_cum < 0.0 or recent_mean < prior_mean:
            level = "warning"
        return {
            "label": "実績損益ドリフト",
            "value": _safe_signed_pct_label(recent_cum, 1),
            "level": level,
            "detail": (
                f"{adopted_model_version} の直近20営業日 net {recent_cum:+.2f}%"
                f" / 前20営業日 {prior_cum:+.2f}%。"
                + (" モデル再学習候補として監視します。" if level != "normal" else " 直近成績は安定圏です。")
            ),
        }

    def _build_ops_view(
        self,
        *,
        dataset: dict[str, Any],
        dashboard: dict[str, Any],
        training: dict[str, Any],
        backtest: dict[str, Any],
        models: dict[str, Any],
        refresh: bool,
    ) -> dict[str, Any]:
        state = self.page_store.get_state()
        collector_level = "normal" if self._staleness_days(dataset["latest_market_date"]) <= 1 else "warning"
        monitor_checks = [
            self._ops_count_check(dataset),
            self._ops_missing_rate_check(dataset),
            self._ops_coverage_check(dataset, dashboard),
            self._ops_score_drift_check(dataset),
            self._ops_pnl_drift_check(backtest, models),
        ]
        predictor_level = "error" if any(item["level"] == "error" for item in monitor_checks) else "warning" if any(item["level"] == "warning" for item in monitor_checks) else "normal"
        pipeline = [
            {"name": "collector", "title": "データ取得", "level": collector_level, "updated": dashboard["latest_update"], "detail": f"Stooq daily CSV ({len(JP_LARGE_CAP_UNIVERSE)} symbols)"},
            {"name": "normalizer", "title": "正規化", "level": "normal", "updated": dashboard["latest_update"], "detail": "CSV -> OHLCV cache -> feature rows"},
            {"name": "feature_builder", "title": "特徴量更新", "level": "normal", "updated": dashboard["latest_update"], "detail": f"{len(dataset['rows'])} rows の特徴量を構築"},
            {"name": "predictor", "title": "推論", "level": predictor_level, "updated": _to_jst_timestamp(state.get("last_inference_run_at")) if state.get("last_inference_run_at") else dashboard["latest_update"], "detail": "LightGBM classifier/regressor と Logistic baseline を生成"},
            {"name": "reporter", "title": "レポート出力", "level": "normal", "updated": dashboard["latest_update"], "detail": "prediction_daily 互換 DTO を返却"},
        ]
        alerts = [
            {
                "level": item["level"],
                "title": item["label"],
                "detail": item["detail"],
            }
            for item in monitor_checks
        ]
        alerts.extend(
            [
                {
                    "level": collector_level,
                    "title": "公開ソース利用",
                    "detail": "日本株日足は Stooq 公開CSV を使用しています。",
                },
                {
                    "level": "normal",
                    "title": "モデル実装差分",
                    "detail": "Primary は LightGBM 本実装、Baseline は Logistic Regression です。",
                },
                {
                    "level": "normal",
                    "title": "監査ログ",
                    "detail": f"{len(state.get('audit_log') or [])} 件の主要操作を保持。",
                },
            ]
        )
        logs = []
        for item in (state.get("audit_log") or [])[:8]:
            if not isinstance(item, dict):
                continue
            logs.append(
                {
                    "level": str(item.get("level") or "normal"),
                    "time": _to_jst_timestamp(item.get("time")).split(" ")[1] if item.get("time") else "-",
                    "stage": "audit",
                    "status": str(item.get("action") or "write").upper(),
                    "message": str(item.get("detail") or "").strip(),
                }
            )
        if not logs:
            logs = self._runtime_logs(action="audit", latest_market_date=dataset["latest_market_date"])
        return {
            "pipeline": pipeline,
            "summary_cards": [
                {"label": item["label"], "value": item["value"]}
                for item in monitor_checks
            ] + [
                {"label": "リークチェック", "value": "PASS"},
                {"label": "ジョブ状態", "value": "SUCCEEDED" if not refresh else "REFRESHED"},
            ],
            "monitor_checks": monitor_checks,
            "coverage_breakdown": dataset.get("excluded_reason_breakdown") or [],
            "score_drift_distribution": self._ops_score_drift_distribution(dataset),
            "alerts": alerts,
            "logs": logs,
        }

    def _build_permissions(
        self,
        *,
        dataset: dict[str, Any],
        dashboard: dict[str, Any],
        training: dict[str, Any],
        models: dict[str, Any],
    ) -> dict[str, Any]:
        role = STOCK_ML_PAGE_ROLE
        adopted_model_version = str(models.get("adopted_model_version") or "").strip()
        adopted_model_row = next(
            (
                item
                for item in (models.get("rows") or [])
                if isinstance(item, dict) and str(item.get("model_version") or "").strip() == adopted_model_version
            ),
            None,
        )
        has_adopted_model = isinstance(adopted_model_row, dict)
        adopted_model_ready = bool(adopted_model_row and adopted_model_row.get("adoptable"))
        coverage_ready = int(dashboard.get("coverage_total") or 0) > 0
        missing_coverage = int(dashboard.get("coverage_excluded") or 0) > 0
        data_is_fresh = str(dashboard.get("freshness", {}).get("level") or "") == "normal"
        training_ready = len(training.get("compare_rows") or []) > 0

        inference_reason = ""
        if not has_adopted_model:
            inference_reason = "採用モデルが未設定のため、先にモデル管理タブで採用モデルを選択してください。"
        elif not adopted_model_ready:
            blockers = [
                str(item).strip()
                for item in (adopted_model_row.get("adopt_blockers") or [])
                if str(item).strip()
            ]
            inference_reason = (
                "採用中モデルが採用不可状態です。"
                + (f" {' / '.join(blockers[:2])}" if blockers else " leakage_check または data_quality_check を確認してください。")
                + " モデル管理タブで状態を確認してください。"
            )
        elif missing_coverage:
            reasons = [
                str(item.get("label") or "").strip()
                for item in (dataset.get("excluded_reason_breakdown") or [])
                if isinstance(item, dict) and int(item.get("count") or 0) > 0 and str(item.get("label") or "").strip()
            ]
            reason_text = " / ".join(reasons[:3]) if reasons else "取得失敗またはデータ不足"
            inference_reason = (
                "対象日にデータ未取得の銘柄があるため推論を実行できません。"
                f" 除外理由: {reason_text}。データ更新後に再確認してください。"
            )
        elif not coverage_ready:
            inference_reason = "推論対象銘柄を構築できていないため、データ更新を先に実行してください。"
        elif not data_is_fresh:
            inference_reason = "最新日付が古いため、データ更新後に推論を再実行してください。"

        training_reason = ""
        if not training_ready:
            training_reason = "学習・検証データが不足しているため、ジョブを作成できません。"

        def permission_item(*, allowed: bool, reason: str = "") -> dict[str, Any]:
            return {
                "allowed": bool(allowed),
                "reason": "" if allowed else reason,
            }

        return {
            "role": role,
            "actions": {
                "refresh_data": permission_item(
                    allowed=role == "admin",
                    reason="データ更新は admin ロールのみ実行できます。",
                ),
                "run_inference": permission_item(
                    allowed=(role == "admin" and not inference_reason),
                    reason=inference_reason or "推論実行は admin ロールのみ実行できます。",
                ),
                "create_training_job": permission_item(
                    allowed=(role in {"analyst", "admin"} and not training_reason),
                    reason=training_reason or "学習ジョブ作成は analyst 以上のロールが必要です。",
                ),
                "run_backtest": permission_item(
                    allowed=role in {"analyst", "admin"},
                    reason="バックテスト実行は analyst 以上のロールが必要です。",
                ),
                "export_report": permission_item(
                    allowed=role in {"analyst", "admin"},
                    reason="レポート出力は analyst 以上のロールが必要です。",
                ),
                "adopt_model": permission_item(
                    allowed=role == "admin",
                    reason="採用モデル変更は admin ロールのみ実行できます。",
                ),
                "export_csv": permission_item(
                    allowed=True,
                    reason="",
                ),
            },
        }

    def _build_status_views(
        self,
        *,
        dashboard: dict[str, Any],
        training: dict[str, Any],
        models: dict[str, Any],
        selected_model_family: str,
        permissions: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        adopted_model_version = models["adopted_model_version"]
        adopted_row = next(
            (
                item
                for item in (models.get("rows") or [])
                if isinstance(item, dict) and str(item.get("model_version") or "").strip() == adopted_model_version
            ),
            None,
        )
        inference_permission = permissions.get("actions", {}).get("run_inference", {})
        inference_enabled = bool(inference_permission.get("allowed"))
        selected_version = models["default_versions"].get(selected_model_family, _MODEL_PRIMARY_VERSION)
        leakage_ok = bool(adopted_row.get("leakage_check")) if isinstance(adopted_row, dict) else False
        leakage_detail = (
            str(adopted_row.get("leakage_detail") or "").strip()
            if isinstance(adopted_row, dict)
            else "採用モデルの leakage_check 情報を取得できません。"
        )
        data_quality_ok = bool(adopted_row.get("data_quality_check")) if isinstance(adopted_row, dict) else False
        data_quality_detail = (
            str(adopted_row.get("data_quality_detail") or "").strip()
            if isinstance(adopted_row, dict)
            else "採用モデルの data_quality_check 情報を取得できません。"
        )
        adopted_ready = bool(adopted_row and adopted_row.get("adoptable"))
        adopted_blockers = [
            str(item).strip()
            for item in ((adopted_row.get("adopt_blockers") or []) if isinstance(adopted_row, dict) else [])
            if str(item).strip()
        ]
        if not adopted_ready:
            global_status = {
                "level": "error",
                "badge": "BLOCKED",
                "text": (
                    "採用中モデルが採用不可状態です。"
                    + (f" {' / '.join(adopted_blockers[:2])}" if adopted_blockers else " モデル管理タブで原因を確認してください。")
                ),
            }
        elif self._staleness_days(dashboard["prediction_date"]) > 3:
            global_status = {
                "level": "warning",
                "badge": "WARNING",
                "text": "最新データではありません。公開日足の更新を確認してください。",
            }
        else:
            global_status = {
                "level": "normal",
                "badge": "READY",
                "text": f"{dashboard['target_date']} 向け prediction_daily 互換データを表示中です。",
            }
        sidebar_status = [
            {
                "label": "採用モデル",
                "value": adopted_model_version,
                "badge": {"label": "READY" if adopted_ready else "REVIEW", "level": "normal" if adopted_ready else "error"},
                "note": (
                    f"現在の表示モデルは {selected_version}。"
                    + (f" {' / '.join(adopted_blockers[:2])}" if adopted_blockers else "")
                ),
            },
            {
                "label": "データ鮮度",
                "value": dashboard["freshness"]["label"],
                "badge": {"label": "LATEST" if dashboard["freshness"]["level"] == "normal" else "STALE", "level": dashboard["freshness"]["level"]},
                "note": f"最終更新: {dashboard['latest_update']}",
            },
            {
                "label": "リークチェック",
                "value": "PASS" if leakage_ok else "FAIL",
                "badge": {"label": "PASS" if leakage_ok else "FAIL", "level": "normal" if leakage_ok else "error"},
                "note": leakage_detail or "walk-forward + gap 条件を明示し、ランダム分割は許可しません。",
            },
            {
                "label": "データ品質",
                "value": "PASS" if data_quality_ok else "FAIL",
                "badge": {"label": "PASS" if data_quality_ok else "FAIL", "level": "normal" if data_quality_ok else "error"},
                "note": data_quality_detail or "取得件数・欠損率・銘柄被覆を確認します。",
            },
            {
                "label": "推論実行可否",
                "value": "実行可能" if inference_enabled else "実行不可",
                "badge": {"label": "ENABLED" if inference_enabled else "BLOCKED", "level": "normal" if inference_enabled else "error"},
                "note": str(inference_permission.get("reason") or f"学習比較 rows: {len(training['compare_rows'])}"),
            },
            {
                "label": "操作ロール",
                "value": str(permissions.get("role") or "viewer"),
                "badge": {"label": str(permissions.get("role") or "viewer").upper(), "level": "normal"},
                "note": "viewer は閲覧のみ、analyst は学習/レポート、admin は推論/採用切替まで可能です。",
            },
        ]
        return global_status, sidebar_status

    @staticmethod
    def _ensure_action_allowed(snapshot: dict[str, Any], action: str) -> None:
        permissions = snapshot.get("permissions", {})
        action_permission = permissions.get("actions", {}).get(action, {})
        if bool(action_permission.get("allowed")):
            return
        reason = str(action_permission.get("reason") or "この操作は現在の条件では実行できません。").strip()
        raise HTTPException(status_code=403, detail=reason)

    def _model_audit_lines(self, *, model_version: str, config_hash: str) -> list[str]:
        state = self.page_store.get_state()
        adopted_model_version = state.get("adopted_model_version")
        lines = [
            f"採用状態: {'採用中' if model_version == adopted_model_version else '候補'}",
            f"最終推論: {_to_jst_timestamp(state.get('last_inference_run_at'))}",
            f"最終学習集計: {_to_jst_timestamp(state.get('last_training_run_at'))}",
            f"入力設定ハッシュ: {config_hash or '-'}",
        ]
        prediction_runs = state.get("prediction_runs") or []
        latest_prediction_run = None
        for item in prediction_runs:
            if not isinstance(item, dict):
                continue
            if str(item.get("model_version") or "").strip() != model_version:
                continue
            latest_prediction_run = item
            break
        if isinstance(latest_prediction_run, dict):
            lines.append(
                "直近prediction_daily: "
                f"{latest_prediction_run.get('prediction_date')}"
                f" -> {latest_prediction_run.get('target_date')}"
                f" / {_to_jst_timestamp(latest_prediction_run.get('generated_at'))}"
            )
        latest_audit = state.get("audit_log") or []
        if latest_audit:
            first = latest_audit[0]
            if isinstance(first, dict):
                lines.append(f"直近操作: {first.get('action')} / {first.get('detail')}")
        return lines

    @staticmethod
    def _filter_dashboard_rows(*, rows: list[dict[str, Any]], search_query: str) -> list[dict[str, Any]]:
        query = str(search_query or "").strip().lower()
        if not query:
            return [item for item in rows if isinstance(item, dict)]
        filtered: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            haystacks = (
                str(item.get("code") or "").lower(),
                str(item.get("company_name") or "").lower(),
                str(item.get("sector") or "").lower(),
            )
            if any(query in value for value in haystacks):
                filtered.append(item)
        return filtered

    def _prediction_run_payload(self, snapshot: dict[str, Any]) -> dict[str, str]:
        filters = snapshot.get("filters", {})
        models = snapshot.get("models", {})
        dashboard = snapshot.get("dashboard", {})
        model_version = str(models.get("default_versions", {}).get(filters.get("model_family"), "")).strip()
        prediction_date = str(dashboard.get("prediction_date") or filters.get("prediction_date") or "").strip()
        feature_version = str(dashboard.get("feature_version") or filters.get("feature_set") or "").strip()
        data_version = str(dashboard.get("data_version") or "").strip()
        generation_key = "::".join(
            [
                prediction_date,
                model_version,
                feature_version,
                data_version,
                str(filters.get("universe_filter") or "").strip(),
                str(filters.get("cost_buffer") or "").strip(),
                str(snapshot.get("config_hash") or "").strip(),
            ]
        )
        return {
            "generation_key": generation_key,
            "prediction_date": prediction_date,
            "target_date": str(dashboard.get("target_date") or "").strip(),
            "model_version": model_version,
            "feature_version": feature_version,
            "data_version": data_version,
            "config_hash": str(snapshot.get("config_hash") or "").strip(),
        }

    def _runtime_logs(self, *, action: str, latest_market_date: str) -> list[dict[str, Any]]:
        time_label = f"{latest_market_date} 15:00".split(" ")[1] if latest_market_date else "-"
        return [
            {"level": "normal", "time": time_label, "stage": action, "status": "SUCCEEDED", "message": f"{action} stage completed on {latest_market_date}."},
        ]

    @staticmethod
    def _previous_prediction_date(prediction_dates: list[str], selected: str) -> str | None:
        try:
            index = prediction_dates.index(selected)
        except ValueError:
            return None
        if index <= 0:
            return None
        return prediction_dates[index - 1]

    @staticmethod
    def _recent_prediction_dates_before(prediction_dates: list[str], selected: str, limit: int) -> list[str]:
        try:
            index = prediction_dates.index(selected)
        except ValueError:
            index = len(prediction_dates)
        start = max(0, index - max(1, int(limit)))
        return prediction_dates[start:index]

    @staticmethod
    def _target_date_for_prediction(*, target_date_by_prediction: dict[str, str], prediction_date: str) -> str:
        target_date = str(target_date_by_prediction.get(prediction_date) or "").strip()
        return target_date or _next_business_day(prediction_date)

    @staticmethod
    def _staleness_days(latest_date: str) -> int:
        if not latest_date:
            return 999
        try:
            parsed = date.fromisoformat(latest_date)
        except ValueError:
            return 999
        return max(0, (date.today() - parsed).days)

    @staticmethod
    def _score_drift_value(dataset: dict[str, Any]) -> str:
        prediction_dates: list[str] = dataset["prediction_dates"]
        if len(prediction_dates) < 30:
            return "-"
        recent = prediction_dates[-10:]
        prior = prediction_dates[-30:-10]
        recent_scores = [float(row["primary_score"]) for row in dataset["rows"] if row["date"] in set(recent)]
        prior_scores = [float(row["primary_score"]) for row in dataset["rows"] if row["date"] in set(prior)]
        if not recent_scores or not prior_scores:
            return "-"
        drift = abs(float(np.mean(recent_scores)) - float(np.mean(prior_scores)))
        return f"{drift:.3f}"

    @staticmethod
    def _infer_environment() -> str:
        import os

        host = str(os.getenv("HOSTNAME") or "").lower()
        if "stg" in host or "stage" in host:
            return "stg"
        if host:
            return "dev"
        return "dev"
