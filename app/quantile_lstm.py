from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

QUANTILE_STEP = 0.001
TAIL_QUANTILE = 0.0001
QUANTILES = np.concatenate(
    (
        np.array([TAIL_QUANTILE], dtype=np.float32),
        np.arange(QUANTILE_STEP, 1.0, QUANTILE_STEP, dtype=np.float32),
        np.array([1.0 - TAIL_QUANTILE], dtype=np.float32),
    )
)
FEATURE_NAMES = [
    "log_ret_1d",
    "log_ret_oc",
    "log_range_hl",
    "log_gap_open_prev_close",
    "volume_log_change",
    "ret_mean_5",
    "ret_std_5",
    "ret_mean_20",
    "ret_std_20",
    "mom_5",
    "mom_20",
    "range_ratio",
    "volume_z_20",
]
ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], None]


def _emit_progress(progress_callback: ProgressCallback | None, progress: float, message: str) -> None:
    if progress_callback is None:
        return
    safe_progress = max(0.0, min(100.0, float(progress)))
    progress_callback(safe_progress, str(message))


def _run_cancel_check(cancel_check: CancelCheck | None) -> None:
    if cancel_check is None:
        return
    cancel_check()


@dataclass(frozen=True)
class QuantileLstmConfig:
    sequence_length: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    min_delta: float = 1e-5
    representative_days: int = 5
    seed: int = 42

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "QuantileLstmConfig":
        payload = payload or {}

        def int_value(name: str, default: int, minimum: int, maximum: int) -> int:
            raw = payload.get(name, default)
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                parsed = default
            return max(minimum, min(maximum, parsed))

        def float_value(name: str, default: float, minimum: float, maximum: float) -> float:
            raw = payload.get(name, default)
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                parsed = default
            return max(minimum, min(maximum, parsed))

        return cls(
            sequence_length=int_value("sequence_length", default=60, minimum=20, maximum=1024),
            hidden_size=int_value("hidden_size", default=64, minimum=16, maximum=2048),
            num_layers=int_value("num_layers", default=2, minimum=1, maximum=12),
            dropout=float_value("dropout", default=0.2, minimum=0.0, maximum=0.6),
            learning_rate=float_value("learning_rate", default=1e-3, minimum=1e-5, maximum=1e-1),
            batch_size=int_value("batch_size", default=64, minimum=8, maximum=2048),
            max_epochs=int_value("max_epochs", default=80, minimum=10, maximum=2000),
            patience=int_value("patience", default=10, minimum=2, maximum=400),
            min_delta=float_value("min_delta", default=1e-5, minimum=0.0, maximum=1e-2),
            representative_days=int_value("representative_days", default=5, minimum=1, maximum=12),
            seed=int_value("seed", default=42, minimum=1, maximum=100000),
        )


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class QuantileLstmModel(nn.Module):
    def __init__(self, input_size: int, config: QuantileLstmConfig) -> None:
        super().__init__()
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.out = nn.Linear(config.hidden_size, len(QUANTILES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.out(last_hidden)


def run_quantile_lstm_forecast(
    points: list[dict[str, Any]],
    config_payload: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    _run_cancel_check(cancel_check)
    config = QuantileLstmConfig.from_payload(config_payload)
    payload = config_payload or {}
    split_eval_days = 0
    split_train_val_ratio = 0.8
    try:
        split_eval_days = max(0, int(payload.get("split_eval_days", 0)))
    except (TypeError, ValueError):
        split_eval_days = 0
    try:
        split_train_val_ratio = float(payload.get("split_train_val_ratio", 0.8))
    except (TypeError, ValueError):
        split_train_val_ratio = 0.8
    split_train_val_ratio = max(0.5, min(0.95, split_train_val_ratio))
    _emit_progress(progress_callback, 5, "学習準備を開始しました。")
    _set_seed(config.seed)

    _run_cancel_check(cancel_check)
    _emit_progress(progress_callback, 10, "特徴量を生成しています。")
    dates, features, closes = _build_feature_matrix(points)
    if len(dates) < (config.sequence_length + 30):
        raise ValueError(
            "ヒストリカルデータが不足しています。sequence_length + 30 営業日以上が必要です。"
        )

    _run_cancel_check(cancel_check)
    _emit_progress(progress_callback, 18, "系列データを構築しています。")
    seq_features, targets, target_dates, base_closes, realized_closes = _build_sequences(
        dates=dates,
        features=features,
        closes=closes,
        sequence_length=config.sequence_length,
    )

    _run_cancel_check(cancel_check)
    _emit_progress(progress_callback, 22, "train/val/test を分割しています。")
    if split_eval_days > 0:
        train_idx, val_idx, test_idx = _split_time_series_indices_recent_window(
            target_dates=target_dates,
            eval_days=split_eval_days,
            train_ratio=split_train_val_ratio,
        )
    else:
        train_idx, val_idx, test_idx = _split_time_series_indices(
            sample_count=len(target_dates),
            train_ratio=0.75,
            val_ratio=0.15,
        )

    train_x = seq_features[train_idx]
    val_x = seq_features[val_idx]
    test_x = seq_features[test_idx]
    train_y = targets[train_idx]
    val_y = targets[val_idx]
    test_y = targets[test_idx]

    _run_cancel_check(cancel_check)
    _emit_progress(progress_callback, 28, "特徴量スケーリングを実行しています。")
    scaled_train_x, scaled_val_x, scaled_test_x = _scale_features(train_x, val_x, test_x)

    _run_cancel_check(cancel_check)
    _emit_progress(progress_callback, 30, "LSTM 学習を開始します。")
    model, device, best_val_loss, epochs_trained = _train_model(
        train_x=scaled_train_x,
        train_y=train_y,
        val_x=scaled_val_x,
        val_y=val_y,
        config=config,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    _run_cancel_check(cancel_check)
    _emit_progress(progress_callback, 86, "推論と評価指標を計算しています。")
    test_pred_quantiles = _predict_quantiles_sorted(model=model, device=device, features=scaled_test_x)
    quantiles = QUANTILES.astype(np.float64)

    mean_pinball = _pinball_loss_np(test_y.astype(np.float64), test_pred_quantiles, quantiles)
    q05_index = _nearest_quantile_index(quantiles, target_tau=0.05)
    q25_index = _nearest_quantile_index(quantiles, target_tau=0.25)
    q50_index = _nearest_quantile_index(quantiles, target_tau=0.50)
    q75_index = _nearest_quantile_index(quantiles, target_tau=0.75)
    q95_index = _nearest_quantile_index(quantiles, target_tau=0.95)

    q05 = test_pred_quantiles[:, q05_index]
    q25 = test_pred_quantiles[:, q25_index]
    q50 = test_pred_quantiles[:, q50_index]
    q75 = test_pred_quantiles[:, q75_index]
    q95 = test_pred_quantiles[:, q95_index]

    coverage_90 = float(np.mean((test_y >= q05) & (test_y <= q95)))
    coverage_50 = float(np.mean((test_y >= q25) & (test_y <= q75)))

    test_base_close = base_closes[test_idx]
    test_realized_close = realized_closes[test_idx]
    test_pred_price_quantiles = test_base_close[:, None] * np.exp(test_pred_quantiles)
    scaler_mean, scaler_std = _fit_feature_scaler(train_x)

    latest_input = features[-config.sequence_length :].astype(np.float32).reshape(1, config.sequence_length, -1)
    latest_input_scaled = _apply_feature_scaler(latest_input, scaler_mean, scaler_std)
    next_return_quantiles = _predict_quantiles_sorted(model=model, device=device, features=latest_input_scaled)[0]
    next_price_quantiles = closes[-1] * np.exp(next_return_quantiles)
    next_cdf_at_zero = _estimate_cdf_at_zero(next_return_quantiles, quantiles)
    next_up_prob = float(max(0.0, min(1.0, 1.0 - next_cdf_at_zero)))
    next_down_prob = float(max(0.0, min(1.0, next_cdf_at_zero)))
    investment_projection_60d = _project_rational_investment_60d(
        next_return_quantiles=next_return_quantiles,
        seed=config.seed,
        horizon_days=60,
        initial_capital=10000.0,
    )
    realized_backtest_60d = _backtest_recent_days(
        pred_return_quantiles=test_pred_quantiles,
        realized_log_returns=test_y.astype(np.float64),
        dates=target_dates[test_idx],
        lookback_days=60,
        initial_capital=10000.0,
    )

    _run_cancel_check(cancel_check)
    _emit_progress(progress_callback, 95, "可視化データを整形しています。")
    representative_curves = _build_representative_curves(
        quantiles=quantiles,
        target_dates=target_dates[test_idx],
        pred_returns=test_pred_quantiles,
        pred_prices=test_pred_price_quantiles,
        actual_returns=test_y.astype(np.float64),
        actual_prices=test_realized_close.astype(np.float64),
        representative_days=config.representative_days,
    )

    fan_chart = {
        "dates": [d.isoformat() for d in target_dates[test_idx]],
        "actual_returns": _to_float_list(test_y),
        "actual_prices": _to_float_list(test_realized_close),
        "base_close": _to_float_list(test_base_close),
        "q05_returns": _to_float_list(q05),
        "q25_returns": _to_float_list(q25),
        "q50_returns": _to_float_list(q50),
        "q75_returns": _to_float_list(q75),
        "q95_returns": _to_float_list(q95),
        "q05_prices": _to_float_list(test_pred_price_quantiles[:, q05_index]),
        "q25_prices": _to_float_list(test_pred_price_quantiles[:, q25_index]),
        "q50_prices": _to_float_list(test_pred_price_quantiles[:, q50_index]),
        "q75_prices": _to_float_list(test_pred_price_quantiles[:, q75_index]),
        "q95_prices": _to_float_list(test_pred_price_quantiles[:, q95_index]),
    }

    _emit_progress(progress_callback, 100, "学習と推論が完了しました。")
    return {
        "config": {
            "sequence_length": config.sequence_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "representative_days": config.representative_days,
            "seed": config.seed,
            "feature_names": FEATURE_NAMES,
            "quantiles": _to_float_list(quantiles),
        },
        "metrics": {
            "mean_pinball_loss": mean_pinball,
            "test_pinball_loss": mean_pinball,
            "test_10pct_pinball_loss": mean_pinball,
            "coverage_90": coverage_90,
            "coverage_50": coverage_50,
        },
        "splits": {
            "train": _split_meta(target_dates, train_idx),
            "val": _split_meta(target_dates, val_idx),
            "test": _split_meta(target_dates, test_idx),
        },
        "quantile_function": {
            "taus": _to_float_list(quantiles),
            "curves": representative_curves,
        },
        "fan_chart": fan_chart,
        "training": {
            "epochs_trained": epochs_trained,
            "best_val_pinball_loss": float(best_val_loss),
            "device": str(device),
        },
        "backtest_60d": realized_backtest_60d,
        "next_day_forecast": {
            "as_of_date": dates[-1].isoformat() if isinstance(dates[-1], date) else str(dates[-1]),
            "target_date": _next_business_day(dates[-1]).isoformat() if isinstance(dates[-1], date) else None,
            "current_close": float(closes[-1]),
            "up_probability": next_up_prob,
            "down_probability": next_down_prob,
            "taus": _to_float_list(quantiles),
            "return_quantiles": _to_float_list(next_return_quantiles),
            "price_quantiles": _to_float_list(next_price_quantiles),
            "q05_return": float(next_return_quantiles[q05_index]),
            "q50_return": float(next_return_quantiles[q50_index]),
            "q95_return": float(next_return_quantiles[q95_index]),
            "q05_price": float(next_price_quantiles[q05_index]),
            "q50_price": float(next_price_quantiles[q50_index]),
            "q95_price": float(next_price_quantiles[q95_index]),
            "investment_60d": investment_projection_60d,
        },
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.clip(numerator, 1e-12, None) / np.clip(denominator, 1e-12, None))
    ratio[~np.isfinite(ratio)] = 0.0
    return ratio


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        out[idx] = float(np.mean(segment)) if segment.size else 0.0
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        if segment.size <= 1:
            out[idx] = 0.0
            continue
        out[idx] = float(np.std(segment, ddof=1))
    return out


def _fill_volume(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float64)
    finite_mask = np.isfinite(out)
    if not finite_mask.any():
        return np.zeros_like(out, dtype=np.float64)

    first_valid = int(np.argmax(finite_mask))
    if first_valid > 0:
        out[:first_valid] = out[first_valid]
    for idx in range(first_valid + 1, out.shape[0]):
        if not np.isfinite(out[idx]):
            out[idx] = out[idx - 1]
    out[~np.isfinite(out)] = 0.0
    out = np.clip(out, a_min=0.0, a_max=None)
    return out


def _build_feature_matrix(points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parsed_dates: list[date] = []
    open_values: list[float] = []
    high_values: list[float] = []
    low_values: list[float] = []
    close_values: list[float] = []
    volume_values: list[float] = []

    for item in points:
        if not isinstance(item, dict):
            continue
        dt_raw = str(item.get("t") or item.get("datetime") or "").strip()
        if not dt_raw:
            continue
        try:
            dt = date.fromisoformat(dt_raw[:10])
        except ValueError:
            continue

        close_price = _to_float(item.get("c", item.get("close")))
        if close_price is None or close_price <= 0:
            continue

        open_price = _to_float(item.get("o", item.get("open")))
        high_price = _to_float(item.get("h", item.get("high")))
        low_price = _to_float(item.get("l", item.get("low")))

        if open_price is None or open_price <= 0:
            open_price = close_price
        if high_price is None or high_price <= 0:
            high_price = max(open_price, close_price)
        if low_price is None or low_price <= 0:
            low_price = min(open_price, close_price)

        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        if low_price <= 0:
            low_price = min(open_price, close_price)
            if low_price <= 0:
                low_price = close_price

        volume_value = _to_float(item.get("v", item.get("volume")))
        volume_values.append(float("nan") if volume_value is None else volume_value)

        parsed_dates.append(dt)
        open_values.append(open_price)
        high_values.append(high_price)
        low_values.append(low_price)
        close_values.append(close_price)

    if len(parsed_dates) < 280:
        raise ValueError("モデル学習に必要なOHLCVデータが不足しています。")

    dates = np.array(parsed_dates, dtype=object)
    open_arr = np.array(open_values, dtype=np.float64)
    high_arr = np.array(high_values, dtype=np.float64)
    low_arr = np.array(low_values, dtype=np.float64)
    close_arr = np.array(close_values, dtype=np.float64)
    volume_arr = _fill_volume(np.array(volume_values, dtype=np.float64))

    prev_close = np.roll(close_arr, 1)
    prev_close[0] = close_arr[0]
    prev_volume = np.roll(volume_arr, 1)
    prev_volume[0] = volume_arr[0]

    log_ret_1d = _safe_log_ratio(close_arr, prev_close)
    log_ret_1d[0] = 0.0
    log_ret_oc = _safe_log_ratio(close_arr, open_arr)
    log_range_hl = _safe_log_ratio(high_arr, low_arr)
    log_gap = _safe_log_ratio(open_arr, prev_close)
    volume_log_change = np.log1p(volume_arr) - np.log1p(prev_volume)
    volume_log_change[0] = 0.0

    ret_mean_5 = _rolling_mean(log_ret_1d, 5)
    ret_std_5 = _rolling_std(log_ret_1d, 5)
    ret_mean_20 = _rolling_mean(log_ret_1d, 20)
    ret_std_20 = _rolling_std(log_ret_1d, 20)

    mom_5 = np.zeros_like(close_arr, dtype=np.float64)
    mom_20 = np.zeros_like(close_arr, dtype=np.float64)
    if close_arr.shape[0] > 5:
        mom_5[5:] = _safe_log_ratio(close_arr[5:], close_arr[:-5])
    if close_arr.shape[0] > 20:
        mom_20[20:] = _safe_log_ratio(close_arr[20:], close_arr[:-20])

    range_ratio = (high_arr - low_arr) / np.clip(close_arr, 1e-12, None)
    volume_mean_20 = _rolling_mean(volume_arr, 20)
    volume_std_20 = _rolling_std(volume_arr, 20)
    volume_z_20 = (volume_arr - volume_mean_20) / np.clip(volume_std_20, 1e-8, None)

    features = np.column_stack(
        [
            log_ret_1d,
            log_ret_oc,
            log_range_hl,
            log_gap,
            volume_log_change,
            ret_mean_5,
            ret_std_5,
            ret_mean_20,
            ret_std_20,
            mom_5,
            mom_20,
            range_ratio,
            volume_z_20,
        ]
    ).astype(np.float32)

    features[~np.isfinite(features)] = 0.0
    return dates, features, close_arr


def _build_sequences(
    dates: np.ndarray,
    features: np.ndarray,
    closes: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(features.shape[0])
    feature_dim = int(features.shape[1])
    sample_count = n_rows - sequence_length
    if sample_count <= 0:
        raise ValueError("系列長がデータ数を超えています。")

    seq_x = np.zeros((sample_count, sequence_length, feature_dim), dtype=np.float32)
    targets = np.zeros(sample_count, dtype=np.float32)
    target_dates = np.empty(sample_count, dtype=object)
    base_closes = np.zeros(sample_count, dtype=np.float64)
    realized_closes = np.zeros(sample_count, dtype=np.float64)

    for sample_idx, t in enumerate(range(sequence_length - 1, n_rows - 1)):
        seq_x[sample_idx] = features[(t - sequence_length + 1) : (t + 1)]
        targets[sample_idx] = float(math.log(max(closes[t + 1], 1e-12) / max(closes[t], 1e-12)))
        target_dates[sample_idx] = dates[t + 1]
        base_closes[sample_idx] = closes[t]
        realized_closes[sample_idx] = closes[t + 1]

    return seq_x, targets, target_dates, base_closes, realized_closes


def _split_time_series_indices(
    sample_count: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_count < 30:
        raise ValueError("分割に必要なサンプル数が不足しています。")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("train_ratio/val_ratio が不正です。")

    train_end = int(sample_count * train_ratio)
    val_end = int(sample_count * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, sample_count - 2))
    val_end = max(train_end + 1, min(val_end, sample_count - 1))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, sample_count, dtype=np.int64)
    return train_idx, val_idx, test_idx


def _split_time_series_indices_recent_window(
    target_dates: np.ndarray,
    eval_days: int,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(target_dates)
    if sample_count < 30:
        raise ValueError("分割に必要なサンプル数が不足しています。")

    dates = [item for item in target_dates]
    if not dates:
        raise ValueError("時系列日付が取得できませんでした。")

    latest = dates[-1]
    if not isinstance(latest, date):
        raise ValueError("時系列日付の形式が不正です。")

    eval_start = latest - timedelta(days=max(1, int(eval_days)))
    test_start = next((idx for idx, d in enumerate(dates) if isinstance(d, date) and d >= eval_start), sample_count - 1)
    test_start = max(1, min(test_start, sample_count - 1))
    test_count = sample_count - test_start
    if test_count < 20:
        raise ValueError("評価期間（直近2か月）のサンプル数が不足しています。")

    train_val_count = test_start
    if train_val_count < 10:
        raise ValueError("学習/検証期間のサンプル数が不足しています。")

    train_end = int(train_val_count * float(train_ratio))
    train_end = max(1, min(train_end, train_val_count - 1))
    val_end = train_val_count

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(test_start, sample_count, dtype=np.int64)
    return train_idx, val_idx, test_idx


def _fit_feature_scaler(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = train_x.shape[-1]
    train_2d = train_x.reshape(-1, feature_dim)
    mean = train_2d.mean(axis=0, keepdims=True)
    std = train_2d.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _apply_feature_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    out = out.astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _scale_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = _fit_feature_scaler(train_x)
    return (
        _apply_feature_scaler(train_x, mean, std),
        _apply_feature_scaler(val_x, mean, std),
        _apply_feature_scaler(test_x, mean, std),
    )


def _pinball_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    errors = y_true.unsqueeze(1) - y_pred
    return torch.maximum(quantiles * errors, (quantiles - 1.0) * errors).mean()


def _pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray) -> float:
    errors = y_true[:, None] - y_pred
    loss = np.maximum(quantiles[None, :] * errors, (quantiles[None, :] - 1.0) * errors)
    return float(np.mean(loss))


def _evaluate(
    model: QuantileLstmModel,
    loader: DataLoader,
    quantiles: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = _pinball_loss_torch(batch_y, pred, quantiles)
            losses.append(float(loss.item()))
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def _train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: QuantileLstmConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> tuple[QuantileLstmModel, torch.device, float, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileLstmModel(input_size=train_x.shape[-1], config=config).to(device)

    train_loader = DataLoader(
        SequenceDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceDataset(val_x, val_y),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    quantiles = torch.tensor(QUANTILES, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    patience_counter = 0
    epochs_trained = 0
    train_steps_per_epoch = max(1, len(train_loader))
    total_train_steps = max(1, config.max_epochs * train_steps_per_epoch)
    last_emit_at = 0.0
    last_progress = 29.0

    for epoch in range(config.max_epochs):
        _run_cancel_check(cancel_check)
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader, start=1):
            _run_cancel_check(cancel_check)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = _pinball_loss_torch(batch_y, pred, quantiles)
            loss.backward()
            optimizer.step()

            completed_steps = (epoch * train_steps_per_epoch) + min(step, train_steps_per_epoch)
            train_progress = 30.0 + (completed_steps / total_train_steps) * 55.0
            train_progress = max(30.0, min(85.0, train_progress))
            now = time.monotonic()
            if (
                (train_progress > (last_progress + 0.05))
                or ((now - last_emit_at) >= 0.8)
                or (step == train_steps_per_epoch)
            ):
                _emit_progress(
                    progress_callback,
                    train_progress,
                    f"LSTM学習中: epoch {epoch + 1}/{config.max_epochs} (batch {step}/{train_steps_per_epoch})",
                )
                last_progress = max(last_progress, train_progress)
                last_emit_at = now

        _run_cancel_check(cancel_check)
        val_loss = _evaluate(model=model, loader=val_loader, quantiles=quantiles, device=device)
        epochs_trained = epoch + 1
        training_progress = 30.0 + (epochs_trained / max(1, config.max_epochs)) * 55.0
        training_progress = max(last_progress, min(85.0, training_progress))
        _emit_progress(
            progress_callback,
            training_progress,
            f"学習中: epoch {epochs_trained}/{config.max_epochs} (val pinball={val_loss:.6f})",
        )

        if val_loss + config.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                _emit_progress(
                    progress_callback,
                    training_progress,
                    f"Early stopping: {epochs_trained} epoch で終了しました。",
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, device, best_val_loss, epochs_trained


def _predict_quantiles_sorted(
    model: QuantileLstmModel,
    device: torch.device,
    features: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        pred = model(x).detach().cpu().numpy().astype(np.float64)
    # Quantile crossing handling: enforce monotonic order after inference.
    return np.sort(pred, axis=1)


def _to_float_list(values: np.ndarray | list[float]) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).tolist()]


def _nearest_quantile_index(quantiles: np.ndarray, target_tau: float) -> int:
    if quantiles.size == 0:
        return 0
    distances = np.abs(quantiles.astype(np.float64) - float(target_tau))
    return int(np.argmin(distances))


def _split_meta(target_dates: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    if indices.size == 0:
        return {"count": 0, "from": None, "to": None}
    first_dt = target_dates[int(indices[0])]
    last_dt = target_dates[int(indices[-1])]
    return {
        "count": int(indices.size),
        "from": first_dt.isoformat() if isinstance(first_dt, date) else None,
        "to": last_dt.isoformat() if isinstance(last_dt, date) else None,
    }


def _build_representative_curves(
    quantiles: np.ndarray,
    target_dates: np.ndarray,
    pred_returns: np.ndarray,
    pred_prices: np.ndarray,
    actual_returns: np.ndarray,
    actual_prices: np.ndarray,
    representative_days: int,
) -> list[dict[str, Any]]:
    count = int(target_dates.shape[0])
    if count == 0:
        return []

    rep_count = max(1, min(representative_days, count))
    selected = np.unique(np.linspace(0, count - 1, num=rep_count, dtype=int))

    curves: list[dict[str, Any]] = []
    for idx in selected:
        dt = target_dates[idx]
        curves.append(
            {
                "date": dt.isoformat() if isinstance(dt, date) else str(dt),
                "taus": _to_float_list(quantiles),
                "return_quantiles": _to_float_list(pred_returns[idx]),
                "price_quantiles": _to_float_list(pred_prices[idx]),
                "actual_return": float(actual_returns[idx]),
                "actual_price": float(actual_prices[idx]),
            }
        )
    return curves


def _next_business_day(value: date) -> date:
    out = value
    while True:
        out = out + timedelta(days=1)
        if out.weekday() < 5:
            return out


def _estimate_cdf_at_zero(quantile_values: np.ndarray, quantiles: np.ndarray) -> float:
    if quantile_values.size == 0:
        return 0.5
    if 0.0 < quantile_values[0]:
        return 0.0
    if 0.0 > quantile_values[-1]:
        return 1.0

    for idx in range(quantile_values.shape[0]):
        q_value = float(quantile_values[idx])
        tau_value = float(quantiles[idx])
        if q_value == 0.0:
            return tau_value
        if q_value > 0.0:
            left_idx = max(0, idx - 1)
            left_q = float(quantile_values[left_idx])
            left_tau = float(quantiles[left_idx])
            if q_value == left_q:
                return tau_value
            weight = (0.0 - left_q) / (q_value - left_q)
            return float(left_tau + ((tau_value - left_tau) * weight))

    return 0.5


def _project_rational_investment_60d(
    next_return_quantiles: np.ndarray,
    seed: int,
    horizon_days: int,
    initial_capital: float,
) -> dict[str, Any]:
    if next_return_quantiles.size == 0:
        return {
            "horizon_days": int(horizon_days),
            "initial_capital": float(initial_capital),
            "optimal_stock_fraction": 0.0,
            "expected_return": 0.0,
            "expected_capital": float(initial_capital),
            "median_return": 0.0,
            "p10_return": 0.0,
            "p90_return": 0.0,
            "median_capital": float(initial_capital),
            "p10_capital": float(initial_capital),
            "p90_capital": float(initial_capital),
            "profit_probability": 0.5,
            "assumption": "distribution_fixed_and_iid",
        }

    asset_simple_returns = np.expm1(next_return_quantiles.astype(np.float64))
    best_fraction, cap_fraction = _risk_capped_return_max_fraction(next_return_quantiles)
    expected_simple_return = float(np.mean(asset_simple_returns))
    best_expected_log_growth = float(
        np.mean(np.log(np.clip(1.0 + (best_fraction * asset_simple_returns), 1e-12, None)))
    )

    horizon = max(1, int(horizon_days))
    expected_return = float(np.exp(best_expected_log_growth * horizon) - 1.0)

    rng = np.random.default_rng(seed=seed)
    path_count = 20000
    sampled_idx = rng.integers(0, asset_simple_returns.shape[0], size=(path_count, horizon))
    sampled_asset_returns = asset_simple_returns[sampled_idx]
    wealth_paths = np.prod(1.0 + (best_fraction * sampled_asset_returns), axis=1) * float(initial_capital)
    returns_paths = (wealth_paths / float(initial_capital)) - 1.0

    p10_return = float(np.quantile(returns_paths, 0.10))
    p50_return = float(np.quantile(returns_paths, 0.50))
    p90_return = float(np.quantile(returns_paths, 0.90))
    expected_capital = float(np.mean(wealth_paths))
    p10_capital = float(np.quantile(wealth_paths, 0.10))
    p50_capital = float(np.quantile(wealth_paths, 0.50))
    p90_capital = float(np.quantile(wealth_paths, 0.90))
    profit_probability = float(np.mean(returns_paths > 0.0))

    return {
        "horizon_days": int(horizon),
        "initial_capital": float(initial_capital),
        "optimal_stock_fraction": best_fraction,
        "cap_stock_fraction": cap_fraction,
        "expected_return": expected_return,
        "expected_daily_return": expected_simple_return,
        "expected_capital": expected_capital,
        "median_return": p50_return,
        "p10_return": p10_return,
        "p90_return": p90_return,
        "median_capital": p50_capital,
        "p10_capital": p10_capital,
        "p90_capital": p90_capital,
        "profit_probability": profit_probability,
        "assumption": "distribution_fixed_and_iid_with_risk_cap",
        "risk_rule": {
            "max_loss_fraction_per_day": 0.01,
            "max_probability": 0.03,
        },
    }


def _optimal_fraction_log_growth(asset_simple_returns: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    best_fraction = 0.0
    best_expected_log_growth = -np.inf
    for fraction in grid:
        wealth_step = 1.0 + (fraction * asset_simple_returns)
        wealth_step = np.clip(wealth_step, 1e-12, None)
        expected_log_growth = float(np.mean(np.log(wealth_step)))
        if expected_log_growth > best_expected_log_growth:
            best_expected_log_growth = expected_log_growth
            best_fraction = float(fraction)
    return best_fraction, best_expected_log_growth


def _risk_capped_return_max_fraction(
    predicted_log_return_quantiles: np.ndarray,
    max_loss_fraction: float = 0.01,
    max_prob: float = 0.03,
) -> tuple[float, float]:
    predicted_simple_quantiles = np.expm1(predicted_log_return_quantiles.astype(np.float64))
    q_tail = _interpolate_quantile(QUANTILES.astype(np.float64), predicted_simple_quantiles, max_prob)
    cap_fraction = _risk_cap_fraction_from_tail_quantile(q_tail, max_loss_fraction=max_loss_fraction)
    expected_simple_return = float(np.mean(predicted_simple_quantiles))
    allocation = cap_fraction if expected_simple_return > 0 else 0.0
    allocation = float(max(0.0, min(1.0, allocation)))
    return allocation, float(cap_fraction)


def _interpolate_quantile(taus: np.ndarray, values: np.ndarray, target_tau: float) -> float:
    if taus.size == 0 or values.size == 0:
        return 0.0
    if target_tau <= float(taus[0]):
        return float(values[0])
    if target_tau >= float(taus[-1]):
        return float(values[-1])

    for idx in range(1, taus.shape[0]):
        left_tau = float(taus[idx - 1])
        right_tau = float(taus[idx])
        if target_tau > right_tau:
            continue
        left_value = float(values[idx - 1])
        right_value = float(values[idx])
        if right_tau == left_tau:
            return right_value
        weight = (target_tau - left_tau) / (right_tau - left_tau)
        return float(left_value + ((right_value - left_value) * weight))

    return float(values[-1])


def _risk_cap_fraction_from_tail_quantile(q_tail_simple_return: float, max_loss_fraction: float = 0.01) -> float:
    if not np.isfinite(q_tail_simple_return):
        return 0.0
    if q_tail_simple_return >= 0.0:
        return 1.0
    # Constraint:
    # P(portfolio loss >= max_loss_fraction) <= 3%
    # with portfolio return = f * R, long-only, cash return = 0
    # => f <= max_loss_fraction / (-q_0.03)
    cap = max_loss_fraction / max(1e-12, -float(q_tail_simple_return))
    return float(max(0.0, min(1.0, cap)))


def _backtest_recent_days(
    pred_return_quantiles: np.ndarray,
    realized_log_returns: np.ndarray,
    dates: np.ndarray,
    lookback_days: int,
    initial_capital: float,
) -> dict[str, Any]:
    if pred_return_quantiles.shape[0] == 0:
        return {
            "days": 0,
            "initial_capital": float(initial_capital),
            "final_capital_strategy": float(initial_capital),
            "final_capital_buy_hold": float(initial_capital),
            "final_return_strategy": 0.0,
            "final_return_buy_hold": 0.0,
            "outperformance": 0.0,
            "path": [],
        }

    sample_count = pred_return_quantiles.shape[0]
    window = max(1, min(int(lookback_days), sample_count))
    start_idx = sample_count - window

    strategy_capital = float(initial_capital)
    buy_hold_capital = float(initial_capital)
    path: list[dict[str, Any]] = []
    allocation_sum = 0.0
    cap_sum = 0.0
    zero_position_days = 0
    capped_days = 0

    for idx in range(start_idx, sample_count):
        fraction, cap_fraction = _risk_capped_return_max_fraction(pred_return_quantiles[idx])
        realized_simple_return = float(np.expm1(realized_log_returns[idx]))

        strategy_capital *= max(1e-12, 1.0 + (fraction * realized_simple_return))
        buy_hold_capital *= max(1e-12, 1.0 + realized_simple_return)
        allocation_sum += float(fraction)
        cap_sum += float(cap_fraction)
        if fraction <= 1e-12:
            zero_position_days += 1
        if cap_fraction < 0.999999:
            capped_days += 1

        dt = dates[idx]
        path.append(
            {
                "date": dt.isoformat() if isinstance(dt, date) else str(dt),
                "allocation_stock": float(fraction),
                "cap_stock": float(cap_fraction),
                "realized_return": realized_simple_return,
                "strategy_capital": float(strategy_capital),
                "buy_hold_capital": float(buy_hold_capital),
                "cash_capital": float(initial_capital),
            }
        )

    final_return_strategy = (strategy_capital / float(initial_capital)) - 1.0
    final_return_buy_hold = (buy_hold_capital / float(initial_capital)) - 1.0

    return {
        "days": int(window),
        "initial_capital": float(initial_capital),
        "from": path[0]["date"] if path else None,
        "to": path[-1]["date"] if path else None,
        "final_capital_strategy": float(strategy_capital),
        "final_capital_buy_hold": float(buy_hold_capital),
        "final_return_strategy": float(final_return_strategy),
        "final_return_buy_hold": float(final_return_buy_hold),
        "outperformance": float(final_return_strategy - final_return_buy_hold),
        "avg_allocation_stock": float(allocation_sum / max(1, window)),
        "avg_cap_stock": float(cap_sum / max(1, window)),
        "zero_position_days": int(zero_position_days),
        "capped_days": int(capped_days),
        "risk_rule": {
            "max_loss_fraction_per_day": 0.01,
            "max_probability": 0.03,
        },
        "path": path,
    }
