from __future__ import annotations

import calendar
import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

QUANTILES = np.arange(0.01, 1.0, 0.01, dtype=np.float32)
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
            sequence_length=int_value("sequence_length", default=60, minimum=20, maximum=240),
            hidden_size=int_value("hidden_size", default=64, minimum=16, maximum=256),
            num_layers=int_value("num_layers", default=2, minimum=1, maximum=4),
            dropout=float_value("dropout", default=0.2, minimum=0.0, maximum=0.6),
            learning_rate=float_value("learning_rate", default=1e-3, minimum=1e-5, maximum=1e-1),
            batch_size=int_value("batch_size", default=64, minimum=8, maximum=512),
            max_epochs=int_value("max_epochs", default=80, minimum=10, maximum=400),
            patience=int_value("patience", default=10, minimum=2, maximum=80),
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


def run_quantile_lstm_forecast(points: list[dict[str, Any]], config_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    config = QuantileLstmConfig.from_payload(config_payload)
    _set_seed(config.seed)

    dates, features, closes = _build_feature_matrix(points)
    if len(dates) < (config.sequence_length + 190):
        raise ValueError(
            "ヒストリカルデータが不足しています。少なくとも約9か月以上の営業日データが必要です。"
        )

    seq_features, targets, target_dates, base_closes, realized_closes = _build_sequences(
        dates=dates,
        features=features,
        closes=closes,
        sequence_length=config.sequence_length,
    )

    split = _split_by_recent_months(target_dates, lookback_months=60)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    train_x = seq_features[train_idx]
    val_x = seq_features[val_idx]
    test_x = seq_features[test_idx]
    train_y = targets[train_idx]
    val_y = targets[val_idx]
    test_y = targets[test_idx]

    scaled_train_x, scaled_val_x, scaled_test_x = _scale_features(train_x, val_x, test_x)

    model, device, best_val_loss, epochs_trained = _train_model(
        train_x=scaled_train_x,
        train_y=train_y,
        val_x=scaled_val_x,
        val_y=val_y,
        config=config,
    )

    test_pred_quantiles = _predict_quantiles_sorted(model=model, device=device, features=scaled_test_x)
    quantiles = QUANTILES.astype(np.float64)

    mean_pinball = _pinball_loss_np(test_y.astype(np.float64), test_pred_quantiles, quantiles)
    q05_index = int(0.05 * 100) - 1
    q25_index = int(0.25 * 100) - 1
    q50_index = int(0.50 * 100) - 1
    q75_index = int(0.75 * 100) - 1
    q95_index = int(0.95 * 100) - 1

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


def _subtract_months(value: date, months: int) -> date:
    year = value.year
    month = value.month - months
    while month <= 0:
        year -= 1
        month += 12
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _split_by_recent_months(target_dates: np.ndarray, lookback_months: int = 60) -> dict[str, np.ndarray]:
    if target_dates.shape[0] < 40:
        raise ValueError("分割に必要なサンプル数が不足しています。")

    last_date = target_dates[-1]
    if not isinstance(last_date, date):
        raise ValueError("日付データの形式が不正です。")

    window_start = _subtract_months(last_date, max(12, lookback_months))
    test_start = _subtract_months(last_date, 3)
    val_start = _subtract_months(last_date, 6)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for idx, dt in enumerate(target_dates):
        if dt < window_start:
            continue
        if dt >= test_start:
            test_idx.append(idx)
        elif dt >= val_start:
            val_idx.append(idx)
        else:
            train_idx.append(idx)

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            "train/val/test の時系列分割に必要なデータが不足しています。`years` を増やしてください。"
        )

    return {
        "train_idx": np.array(train_idx, dtype=np.int64),
        "val_idx": np.array(val_idx, dtype=np.int64),
        "test_idx": np.array(test_idx, dtype=np.int64),
    }


def _scale_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_dim = train_x.shape[-1]
    train_2d = train_x.reshape(-1, feature_dim)
    mean = train_2d.mean(axis=0, keepdims=True)
    std = train_2d.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    def normalize(values: np.ndarray) -> np.ndarray:
        out = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        out = out.astype(np.float32)
        out[~np.isfinite(out)] = 0.0
        return out

    return normalize(train_x), normalize(val_x), normalize(test_x)


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

    for epoch in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = _pinball_loss_torch(batch_y, pred, quantiles)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate(model=model, loader=val_loader, quantiles=quantiles, device=device)
        epochs_trained = epoch + 1

        if val_loss + config.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
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
