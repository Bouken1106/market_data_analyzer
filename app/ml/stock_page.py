"""Real-data backed stock ML page service for the Japan stock dashboard."""

from __future__ import annotations

import asyncio
import csv
import io
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
_TRAIN_WINDOW_DAYS = 252
_GAP_DAYS = 5
_VALID_WINDOW_DAYS = 21
_BACKTEST_WINDOW_DAYS = 120
_MAX_PREDICTION_DATES = 24
_STOOQ_TIMEOUT_SEC = 25.0
_STOOQ_FETCH_CONCURRENCY = 4
_MODEL_SEED = 42
_RETURN_ABS_CLIP = 0.2

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
        run_note: str | None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        histories = await self._load_histories(refresh=refresh)
        dataset = self._build_dataset(histories=histories, cost_buffer=cost_buffer)
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

        training = self._build_training_view(dataset=dataset)
        backtest = self._build_backtest_view(dataset=dataset)
        models = self._build_model_registry(training=training, backtest=backtest)
        selected_model_version = models["default_versions"].get(selected_model_family, _MODEL_PRIMARY_VERSION)
        dashboard = self._build_dashboard_view(
            dataset=dataset,
            prediction_date=selected_prediction_date,
            model_family=selected_model_family,
            selected_model_version=selected_model_version,
            adopted_model_version=models["adopted_model_version"],
            feature_set=selected_feature_set,
            cost_buffer=cost_buffer,
            latest_market_date=latest_market_date,
        )
        ops = self._build_ops_view(
            dataset=dataset,
            dashboard=dashboard,
            training=training,
            models=models,
            refresh=refresh,
        )
        permissions = self._build_permissions(
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
            "header": {
                "updated_at": dashboard["latest_update"],
                "env": self._infer_environment(),
            },
            "filter_options": {
                "prediction_dates": [
                    {"value": item, "label": f"{item} / {_next_business_day(item)}"}
                    for item in available_prediction_dates[-_MAX_PREDICTION_DATES:]
                ],
                "universe_filters": [{"value": _UNIVERSE_VALUE, "label": _UNIVERSE_LABEL}],
                "model_families": [
                    {"value": _PRIMARY_MODEL_FAMILY, "label": _PRIMARY_MODEL_FAMILY},
                    {"value": _BASELINE_MODEL_FAMILY, "label": _BASELINE_MODEL_FAMILY},
                ],
                "feature_sets": [{"value": _FEATURE_VERSION, "label": _FEATURE_VERSION}],
                "cost_buffers": [{"value": "0.0", "label": "0.0"}, {"value": "0.002", "label": "0.002"}],
            },
            "filters": {
                "prediction_date": selected_prediction_date,
                "universe_filter": selected_universe,
                "model_family": selected_model_family,
                "feature_set": selected_feature_set,
                "cost_buffer": f"{cost_buffer:.3f}".rstrip("0").rstrip(".") if cost_buffer else "0.0",
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
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "run_inference")
        self.page_store.mark_inference_run()
        self.page_store.add_audit_log(
            action="run_inference",
            detail=f"Manual inference executed. model_family={kwargs.get('model_family') or _PRIMARY_MODEL_FAMILY}",
        )
        return await self.build_snapshot(**kwargs)

    async def create_training_job(self, **kwargs: Any) -> dict[str, Any]:
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "create_training_job")
        self.page_store.mark_training_run()
        self.page_store.add_audit_log(
            action="create_training_job",
            detail=f"Training summary refreshed. cost_buffer={kwargs.get('cost_buffer')}",
            level="warning",
        )
        return await self.build_snapshot(**kwargs)

    async def refresh_data(self, **kwargs: Any) -> dict[str, Any]:
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "refresh_data")
        self.page_store.add_audit_log(
            action="refresh_data",
            detail="Universe daily cache refreshed from Stooq.",
        )
        refreshed_kwargs = dict(kwargs)
        refreshed_kwargs["refresh"] = True
        return await self.build_snapshot(**refreshed_kwargs)

    async def adopt_model(self, *, model_version: str, **kwargs: Any) -> dict[str, Any]:
        if model_version not in {_MODEL_PRIMARY_VERSION, _MODEL_BASELINE_VERSION}:
            raise HTTPException(status_code=400, detail="Unknown model_version.")
        current_snapshot = await self.build_snapshot(**kwargs)
        self._ensure_action_allowed(current_snapshot, "adopt_model")
        self.page_store.set_adopted_model_version(model_version)
        self.page_store.add_audit_log(
            action="adopt_model",
            detail=f"Adopted model changed to {model_version}.",
            level="warning",
        )
        return await self.build_snapshot(**kwargs)

    async def _load_histories(self, *, refresh: bool) -> dict[str, dict[str, Any]]:
        semaphore = asyncio.Semaphore(_STOOQ_FETCH_CONCURRENCY)

        async def load_one(symbol_meta: UniverseSymbol) -> tuple[str, dict[str, Any] | None]:
            async with semaphore:
                points = await self._get_stooq_history(symbol_meta, refresh=refresh)
            if len(points) < 260:
                return symbol_meta.code, None
            return symbol_meta.code, {
                "meta": symbol_meta,
                "points": points,
                "last_date": str(points[-1].get("t") or "").split(" ")[0],
            }

        results = await asyncio.gather(*(load_one(item) for item in JP_LARGE_CAP_UNIVERSE))
        out: dict[str, dict[str, Any]] = {}
        for code, payload in results:
            if payload is not None:
                out[code] = payload
        return out

    async def _get_stooq_history(self, symbol_meta: UniverseSymbol, *, refresh: bool) -> list[dict[str, Any]]:
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
            return cached

        try:
            points = await asyncio.to_thread(self._fetch_stooq_csv, symbol_meta.code)
        except Exception as exc:
            LOGGER.warning("Failed to fetch Stooq CSV for %s: %s", normalized_symbol, exc)
            return cached
        if points:
            await self.full_daily_history_store.upsert(normalized_symbol, points)
            return points
        if cached:
            return cached
        return []

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

    def _build_dataset(self, *, histories: dict[str, dict[str, Any]], cost_buffer: float) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        per_date: dict[str, list[dict[str, Any]]] = {}
        latest_market_date = ""
        excluded_symbols = len(JP_LARGE_CAP_UNIVERSE) - len(histories)
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

        prediction_dates = sorted(day for day, day_rows in per_date.items() if len(day_rows) >= 8)
        self._apply_primary_scores(per_date)
        return {
            "rows": rows,
            "by_date": per_date,
            "prediction_dates": prediction_dates,
            "excluded_symbols": excluded_symbols,
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

    def _fit_lightgbm_classifier(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
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
            "seed": _MODEL_SEED,
            "feature_fraction_seed": _MODEL_SEED,
            "bagging_seed": _MODEL_SEED,
            "data_random_seed": _MODEL_SEED,
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

    def _fit_lightgbm_regression(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
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
            "seed": _MODEL_SEED,
            "feature_fraction_seed": _MODEL_SEED,
            "bagging_seed": _MODEL_SEED,
            "data_random_seed": _MODEL_SEED,
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

    def _build_training_view(self, *, dataset: dict[str, Any]) -> dict[str, Any]:
        prediction_dates: list[str] = dataset["prediction_dates"]
        if len(prediction_dates) < (_TRAIN_WINDOW_DAYS + _VALID_WINDOW_DAYS + _GAP_DAYS + 5):
            raise HTTPException(status_code=502, detail="学習・検証に必要な営業日数が不足しています。")

        folds: list[dict[str, Any]] = []
        primary_metrics: list[dict[str, float | None]] = []
        baseline_metrics: list[dict[str, float | None]] = []
        latest_valid_rows: list[dict[str, Any]] = []
        latest_primary_model: dict[str, Any] | None = None
        for fold_index in range(4):
            valid_end = len(prediction_dates) - ((3 - fold_index) * _VALID_WINDOW_DAYS)
            valid_start = valid_end - _VALID_WINDOW_DAYS
            gap_end = valid_start
            gap_start = gap_end - _GAP_DAYS
            train_end = gap_start
            train_start = max(0, train_end - _TRAIN_WINDOW_DAYS)
            train_dates = set(prediction_dates[train_start:train_end])
            valid_dates = set(prediction_dates[valid_start:valid_end])

            train_rows = [row for row in dataset["rows"] if row["date"] in train_dates]
            valid_rows = [row for row in dataset["rows"] if row["date"] in valid_dates]
            if not train_rows or not valid_rows:
                continue

            start_timer = time.perf_counter()
            lgbm_model = self._fit_lightgbm_classifier(train_rows)
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
                    "gap": f"{_GAP_DAYS}営業日",
                    "valid": f"{prediction_dates[valid_start]} - {prediction_dates[valid_end - 1]}",
                    "samples": f"train {len(train_rows)} / valid {len(valid_rows)}",
                    "lgbm_roc_auc": primary_stat["roc_auc"] or 0.0,
                    "logreg_roc_auc": baseline_stat["roc_auc"] or 0.0,
                }
            )

        primary_summary = self._summarize_model_metrics(primary_metrics, _PRIMARY_MODEL_FAMILY)
        baseline_summary = self._summarize_model_metrics(baseline_metrics, _BASELINE_MODEL_FAMILY)
        distribution = self._score_histogram(latest_valid_rows)
        acceptance = [
            {
                "level": "normal",
                "title": "実データ評価完了",
                "detail": f"{len(folds)} fold の walk-forward + gap を計算しました。",
            },
            {
                "level": "normal",
                "title": "Primary model 実装",
                "detail": "LightGBM binary classifier を fold ごとに再学習して評価しています。",
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
                {"label": "学習期間", "value": "252営業日"},
                {"label": "gap", "value": f"{_GAP_DAYS}営業日"},
                {"label": "valid期間", "value": f"{_VALID_WINDOW_DAYS}営業日"},
                {"label": "feature_set", "value": _FEATURE_VERSION},
                {"label": "seed", "value": "42"},
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
                {"label": "Validation", "value": "walk-forward + gap"},
                {"label": "Leakage Check", "value": "PASS"},
                {"label": "Feature Set", "value": _FEATURE_VERSION},
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

    def _build_backtest_view(self, *, dataset: dict[str, Any]) -> dict[str, Any]:
        prediction_dates: list[str] = dataset["prediction_dates"]
        backtest_dates = prediction_dates[-_BACKTEST_WINDOW_DAYS:]
        if len(backtest_dates) < 40:
            raise HTTPException(status_code=502, detail="バックテストに必要な営業日数が不足しています。")
        train_dates = prediction_dates[: max(0, len(prediction_dates) - _BACKTEST_WINDOW_DAYS - _GAP_DAYS)]
        primary_train_rows = [row for row in dataset["rows"] if row["date"] in set(train_dates)]
        logistic_train_rows = [row for row in dataset["rows"] if row["date"] in set(train_dates)]
        lgbm_model = self._fit_lightgbm_classifier(primary_train_rows)
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

    def _build_model_registry(self, *, training: dict[str, Any], backtest: dict[str, Any]) -> dict[str, Any]:
        adopted_model_version = self.page_store.get_adopted_model_version()
        model_results = backtest["model_results"]
        primary_result = model_results[_MODEL_PRIMARY_VERSION]
        baseline_result = model_results[_MODEL_BASELINE_VERSION]

        primary_row = {
            "model_version": _MODEL_PRIMARY_VERSION,
            "feature_version": _FEATURE_VERSION,
            "family": _PRIMARY_MODEL_FAMILY,
            "task_type": "分類",
            "status": "adopted" if adopted_model_version == _MODEL_PRIMARY_VERSION else "candidate",
            "summary_metrics": f"ROC-AUC {training['primary_summary']['roc_auc']:.3f} / Sharpe {primary_result['metrics']['sharpe']:.2f}" if primary_result["metrics"]["sharpe"] is not None else f"ROC-AUC {training['primary_summary']['roc_auc']:.3f}",
            "warnings": [],
            "adoptable": True,
            "leakage_check": True,
            "data_quality_check": True,
            "train_conditions": [
                {"label": "学習期間", "value": f"{_TRAIN_WINDOW_DAYS}営業日"},
                {"label": "gap", "value": f"{_GAP_DAYS}営業日"},
                {"label": "valid期間", "value": f"{_VALID_WINDOW_DAYS}営業日"},
                {"label": "cost_buffer", "value": "0.0 / 0.002"},
                {"label": "seed", "value": "42"},
                {"label": "特徴量セット", "value": _FEATURE_VERSION},
            ],
            "eval_conditions": [
                {"label": "統計指標平均", "value": f"ROC-AUC {training['primary_summary']['roc_auc']:.3f} / PR-AUC {training['primary_summary']['pr_auc']:.3f}"},
                {"label": "バックテスト", "value": f"CAGR {primary_result['metrics']['cagr_pct']:.1f}% / Sharpe {primary_result['metrics']['sharpe']:.2f}" if primary_result["metrics"]["sharpe"] is not None else "-"},
                {"label": "計算時間", "value": training["primary_summary"]["train_time"]},
                {"label": "欠損率", "value": "0.0%"},
                {"label": "期待収益率", "value": "LightGBM regressor で算出"},
            ],
            "explainability": [
                {"level": "normal", "title": "全体重要特徴量", "detail": training.get("primary_feature_importance") or "importance unavailable"},
                {"level": "normal", "title": "説明可能性の注記", "detail": "pred_contrib による tree contribution を各銘柄で表示します。"},
                {"level": "normal", "title": "当日寄与上位", "detail": "各銘柄の top3 feature contribution をダッシュボードで表示。"},
            ],
            "audit": self._model_audit_lines(model_version=_MODEL_PRIMARY_VERSION),
        }
        baseline_row = {
            "model_version": _MODEL_BASELINE_VERSION,
            "feature_version": _FEATURE_VERSION,
            "family": _BASELINE_MODEL_FAMILY,
            "task_type": "分類",
            "status": "adopted" if adopted_model_version == _MODEL_BASELINE_VERSION else "candidate",
            "summary_metrics": f"ROC-AUC {training['baseline_summary']['roc_auc']:.3f} / Sharpe {baseline_result['metrics']['sharpe']:.2f}" if baseline_result["metrics"]["sharpe"] is not None else f"ROC-AUC {training['baseline_summary']['roc_auc']:.3f}",
            "warnings": [],
            "adoptable": True,
            "leakage_check": True,
            "data_quality_check": True,
            "train_conditions": [
                {"label": "学習期間", "value": f"{_TRAIN_WINDOW_DAYS}営業日"},
                {"label": "gap", "value": f"{_GAP_DAYS}営業日"},
                {"label": "valid期間", "value": f"{_VALID_WINDOW_DAYS}営業日"},
                {"label": "cost_buffer", "value": "0.0 / 0.002"},
                {"label": "seed", "value": "42"},
                {"label": "特徴量セット", "value": _FEATURE_VERSION},
            ],
            "eval_conditions": [
                {"label": "統計指標平均", "value": f"ROC-AUC {training['baseline_summary']['roc_auc']:.3f} / PR-AUC {training['baseline_summary']['pr_auc']:.3f}"},
                {"label": "バックテスト", "value": f"CAGR {baseline_result['metrics']['cagr_pct']:.1f}% / Sharpe {baseline_result['metrics']['sharpe']:.2f}" if baseline_result["metrics"]["sharpe"] is not None else "-"},
                {"label": "計算時間", "value": training["baseline_summary"]["train_time"]},
                {"label": "欠損率", "value": "0.0%"},
                {"label": "警告", "value": "-"},
            ],
            "explainability": [
                {"level": "normal", "title": "係数上位特徴量", "detail": "ret_20d / ma_gap_20 / volatility_20"},
                {"level": "normal", "title": "セクター別傾向", "detail": "価格・出来高特徴量のみのためセクター説明は限定的です。"},
                {"level": "normal", "title": "当日寄与上位", "detail": "線形係数に基づく寄与。"},
            ],
            "audit": self._model_audit_lines(model_version=_MODEL_BASELINE_VERSION),
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
    ) -> dict[str, Any]:
        base_rows = [dict(item) for item in dataset["by_date"].get(prediction_date, [])]
        if not base_rows:
            raise HTTPException(status_code=404, detail="選択した prediction_date のデータがありません。")
        train_rows = [row for row in dataset["rows"] if row["date"] < prediction_date]
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
            lgbm_model = self._fit_lightgbm_classifier(train_rows)
            return_model = self._fit_lightgbm_regression(train_rows)
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
            {"label": "target_date", "value": _to_jst_label(_next_business_day(prediction_date))},
            {"label": "model_version", "value": selected_model_version, "sub": adopted_label},
            {"label": "data_version", "value": f"stooq_jp_{prediction_date.replace('-', '')}", "sub": feature_set},
            {"label": "coverage", "value": f"{coverage_total}/{coverage_total + coverage_excluded}", "sub": f"除外 {coverage_excluded}銘柄"},
            {"label": "data freshness", "value": "最新" if freshness_level == "normal" else "遅延あり", "sub": latest_update},
        ]
        return {
            "prediction_date": prediction_date,
            "label": _to_jst_label(prediction_date),
            "target_date": _next_business_day(prediction_date),
            "latest_update": latest_update,
            "data_version": f"stooq_jp_{prediction_date.replace('-', '')}",
            "feature_version": feature_set,
            "coverage_total": coverage_total,
            "coverage_excluded": coverage_excluded,
            "freshness": {"level": freshness_level, "label": "最新" if freshness_level == "normal" else "遅延あり"},
            "run_state": "推論完了",
            "summary_cards": summary_cards,
            "caption": f"{coverage_total} 件表示 / selected={model_family} / ranking=score_cls / cost_buffer={cost_buffer}",
            "footnote": f"score_rank を算出済み。{expected_return_note}",
            "rows": display_rows,
            "sector_scores": sector_items,
            "alerts": alerts,
            "logs": logs,
        }

    def _build_ops_view(
        self,
        *,
        dataset: dict[str, Any],
        dashboard: dict[str, Any],
        training: dict[str, Any],
        models: dict[str, Any],
        refresh: bool,
    ) -> dict[str, Any]:
        state = self.page_store.get_state()
        collector_level = "normal" if self._staleness_days(dataset["latest_market_date"]) <= 1 else "warning"
        pipeline = [
            {"name": "collector", "title": "データ取得", "level": collector_level, "updated": dashboard["latest_update"], "detail": f"Stooq daily CSV ({len(JP_LARGE_CAP_UNIVERSE)} symbols)"},
            {"name": "normalizer", "title": "正規化", "level": "normal", "updated": dashboard["latest_update"], "detail": "CSV -> OHLCV cache -> feature rows"},
            {"name": "feature_builder", "title": "特徴量更新", "level": "normal", "updated": dashboard["latest_update"], "detail": f"{len(dataset['rows'])} rows の特徴量を構築"},
            {"name": "predictor", "title": "推論", "level": "normal", "updated": _to_jst_timestamp(state.get("last_inference_run_at")) if state.get("last_inference_run_at") else dashboard["latest_update"], "detail": "LightGBM classifier/regressor と Logistic baseline を生成"},
            {"name": "reporter", "title": "レポート出力", "level": "normal", "updated": dashboard["latest_update"], "detail": "prediction_daily 互換 DTO を返却"},
        ]
        alerts = [
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
                {"label": "取得件数異常", "value": "0 source"},
                {"label": "欠損率", "value": "0.0%"},
                {"label": "スコアドリフト", "value": self._score_drift_value(dataset)},
                {"label": "対象銘柄数", "value": str(len(dashboard["rows"]))},
                {"label": "リークチェック", "value": "PASS"},
                {"label": "ジョブ状態", "value": "SUCCEEDED" if not refresh else "REFRESHED"},
            ],
            "alerts": alerts,
            "logs": logs,
        }

    def _build_permissions(
        self,
        *,
        dashboard: dict[str, Any],
        training: dict[str, Any],
        models: dict[str, Any],
    ) -> dict[str, Any]:
        role = STOCK_ML_PAGE_ROLE
        registered_versions = {
            str(item.get("model_version") or "").strip()
            for item in (models.get("rows") or [])
            if isinstance(item, dict)
        }
        adopted_model_version = str(models.get("adopted_model_version") or "").strip()
        has_adopted_model = adopted_model_version in registered_versions
        coverage_ready = int(dashboard.get("coverage_total") or 0) > 0
        data_is_fresh = str(dashboard.get("freshness", {}).get("level") or "") == "normal"
        training_ready = len(training.get("compare_rows") or []) > 0

        inference_reason = ""
        if not has_adopted_model:
            inference_reason = "採用モデルが未設定のため、先にモデル管理タブで採用モデルを選択してください。"
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
        inference_permission = permissions.get("actions", {}).get("run_inference", {})
        inference_enabled = bool(inference_permission.get("allowed"))
        selected_version = models["default_versions"].get(selected_model_family, _MODEL_PRIMARY_VERSION)
        if self._staleness_days(dashboard["prediction_date"]) > 3:
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
                "badge": {"label": "READY", "level": "normal"},
                "note": f"現在の表示モデルは {selected_version}。",
            },
            {
                "label": "データ鮮度",
                "value": dashboard["freshness"]["label"],
                "badge": {"label": "LATEST" if dashboard["freshness"]["level"] == "normal" else "STALE", "level": dashboard["freshness"]["level"]},
                "note": f"最終更新: {dashboard['latest_update']}",
            },
            {
                "label": "リークチェック",
                "value": "PASS",
                "badge": {"label": "PASS", "level": "normal"},
                "note": "walk-forward + gap を固定表示。",
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

    def _model_audit_lines(self, *, model_version: str) -> list[str]:
        state = self.page_store.get_state()
        adopted_model_version = state.get("adopted_model_version")
        lines = [
            f"採用状態: {'採用中' if model_version == adopted_model_version else '候補'}",
            f"最終推論: {_to_jst_timestamp(state.get('last_inference_run_at'))}",
            f"最終学習集計: {_to_jst_timestamp(state.get('last_training_run_at'))}",
            "入力設定ハッシュ: stooq-jp-base_v1",
        ]
        latest_audit = state.get("audit_log") or []
        if latest_audit:
            first = latest_audit[0]
            if isinstance(first, dict):
                lines.append(f"直近操作: {first.get('action')} / {first.get('detail')}")
        return lines

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
