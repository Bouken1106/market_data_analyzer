from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_QUANTILES = np.arange(0.001, 1.0, 0.001, dtype=np.float32)
ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class FeatureConfig:
    date_candidates: tuple[str, ...] = ("date", "datetime", "timestamp", "time")
    price_candidates: tuple[str, ...] = ("adj_close", "adjusted_close", "close")
    volume_candidates: tuple[str, ...] = ("volume", "vol")
    open_candidates: tuple[str, ...] = ("open",)
    high_candidates: tuple[str, ...] = ("high",)
    low_candidates: tuple[str, ...] = ("low",)
    close_candidates: tuple[str, ...] = ("close",)
    include_ohlc_features: bool = True
    fillna_method: Literal["drop", "ffill"] = "drop"


@dataclass(frozen=True)
class PatchTSTConfig:
    input_length: int = 256
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    channel_independence: bool = True
    enforce_monotonic: bool = True
    crossing_penalty_lambda: float = 0.0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    min_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_epochs: int = 40
    patience: int = 8
    min_delta: float = 1e-5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    use_amp: bool = True
    seed: int = 42
    device: str | None = None


@dataclass(frozen=True)
class PatchTSTRuntimeConfig:
    model_config: PatchTSTConfig
    representative_days: int = 5

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "PatchTSTRuntimeConfig":
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

        input_length = int_value("sequence_length", default=256, minimum=32, maximum=512)
        d_model = int_value("hidden_size", default=128, minimum=32, maximum=256)
        n_heads = 4 if d_model >= 128 else 2
        if d_model % n_heads != 0:
            n_heads = 2
        n_layers = int_value("num_layers", default=3, minimum=1, maximum=6)
        dropout = float_value("dropout", default=0.1, minimum=0.0, maximum=0.6)
        learning_rate = float_value("learning_rate", default=1e-3, minimum=1e-5, maximum=1e-1)
        batch_size = int_value("batch_size", default=32, minimum=4, maximum=256)
        max_epochs = int_value("max_epochs", default=40, minimum=5, maximum=300)
        patience = int_value("patience", default=8, minimum=2, maximum=80)
        seed = int_value("seed", default=42, minimum=1, maximum=100000)
        representative_days = int_value("representative_days", default=5, minimum=1, maximum=12)

        model_config = PatchTSTConfig(
            input_length=input_length,
            patch_len=16,
            stride=8,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=max(64, d_model * 2),
            dropout=dropout,
            channel_independence=True,
            enforce_monotonic=True,
            crossing_penalty_lambda=0.0,
            train_ratio=0.7,
            val_ratio=0.15,
            batch_size=batch_size,
            min_batch_size=4,
            gradient_accumulation_steps=1,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=1e-5,
            learning_rate=learning_rate,
            weight_decay=1e-4,
            grad_clip=1.0,
            num_workers=0,
            use_amp=True,
            seed=seed,
            device=None,
        )
        return cls(model_config=model_config, representative_days=representative_days)


@dataclass(frozen=True)
class PreparedData:
    dates: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    features: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class WindowedData:
    x: np.ndarray
    y: np.ndarray
    base_prices: np.ndarray
    target_prices: np.ndarray
    target_dates: np.ndarray


@dataclass
class PatchTSTArtifact:
    model: "PatchTSTQuantileModel"
    config: PatchTSTConfig
    feature_config: FeatureConfig
    quantiles: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    feature_names: list[str]
    device: torch.device


def _emit_progress(progress_callback: ProgressCallback | None, progress: int, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(max(0, min(100, int(progress))), str(message))


class RollingWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.targets = torch.from_numpy(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class PatchTSTQuantileModel(nn.Module):
    def __init__(
        self,
        input_length: int,
        input_channels: int,
        quantile_count: int,
        config: PatchTSTConfig,
    ) -> None:
        super().__init__()
        if input_length < config.patch_len:
            raise ValueError("input_length must be >= patch_len.")
        if config.patch_len <= 0 or config.stride <= 0:
            raise ValueError("patch_len and stride must be > 0.")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.input_length = int(input_length)
        self.input_channels = int(input_channels)
        self.patch_len = int(config.patch_len)
        self.stride = int(config.stride)
        self.d_model = int(config.d_model)
        self.quantile_count = int(quantile_count)
        self.channel_independence = bool(config.channel_independence)
        self.enforce_monotonic = bool(config.enforce_monotonic)

        self.n_patches = 1 + (self.input_length - self.patch_len) // self.stride
        if self.n_patches <= 0:
            raise ValueError("n_patches must be positive. Check input_length/patch_len/stride.")

        patch_embed_in = self.patch_len if self.channel_independence else self.patch_len * self.input_channels
        self.patch_embed = nn.Linear(patch_embed_in, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.post_norm = nn.LayerNorm(self.d_model)
        head_out_dim = self.quantile_count + 1 if (self.enforce_monotonic and self.quantile_count > 1) else self.quantile_count
        self.head = nn.Linear(self.d_model, head_out_dim)

        nn.init.trunc_normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_length, channels]
        if x.ndim != 3:
            raise ValueError("Expected x shape [batch, input_length, channels].")
        batch_size, seq_len, channels = x.shape
        if seq_len != self.input_length:
            raise ValueError(f"Expected input_length={self.input_length}, got {seq_len}.")
        if channels != self.input_channels:
            raise ValueError(f"Expected input_channels={self.input_channels}, got {channels}.")

        x = x.transpose(1, 2)  # [batch, channels, length]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [b, c, n_patches, patch_len]
        n_patches = patches.size(-2)

        if self.channel_independence:
            patches = patches.reshape(batch_size * channels, n_patches, self.patch_len)
            tokens = self.patch_embed(patches)
            tokens = tokens + self.position_embedding[:, :n_patches, :]
            tokens = self.embed_dropout(tokens)
            encoded = self.encoder(tokens)
            encoded = self.post_norm(encoded)
            summary = encoded.mean(dim=1).reshape(batch_size, channels, self.d_model).mean(dim=1)
        else:
            patches = patches.permute(0, 2, 1, 3).reshape(batch_size, n_patches, channels * self.patch_len)
            tokens = self.patch_embed(patches)
            tokens = tokens + self.position_embedding[:, :n_patches, :]
            tokens = self.embed_dropout(tokens)
            encoded = self.encoder(tokens)
            encoded = self.post_norm(encoded)
            summary = encoded.mean(dim=1)

        raw_quantiles = self.head(summary)
        if self.enforce_monotonic and self.quantile_count > 1:
            base = raw_quantiles[:, :1]
            span = F.softplus(raw_quantiles[:, 1:2]) + 1e-6
            logits = raw_quantiles[:, 2:]
            weights = F.softmax(logits, dim=1)
            increments = span * weights
            tail = base + torch.cumsum(increments, dim=1)
            return torch.cat([base, tail], dim=1)
        return raw_quantiles


def _canonical_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    column_map = {_canonical_column_name(col): col for col in df.columns}
    for candidate in candidates:
        hit = column_map.get(_canonical_column_name(candidate))
        if hit is not None:
            return hit
    return None


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_timeseries_dataframe(df: pd.DataFrame, feature_config: FeatureConfig | None = None) -> PreparedData:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if len(df) == 0:
        raise ValueError("df is empty.")

    cfg = feature_config or FeatureConfig()
    work = df.copy()

    date_col = _find_column(work, cfg.date_candidates)
    if date_col is not None:
        work["__date__"] = pd.to_datetime(work[date_col], errors="coerce")
    elif isinstance(work.index, pd.DatetimeIndex):
        work["__date__"] = pd.to_datetime(work.index, errors="coerce")
    else:
        raise ValueError("No date column found and index is not DatetimeIndex.")

    price_col = _find_column(work, cfg.price_candidates)
    if price_col is None:
        raise ValueError(f"Price column not found. Tried: {cfg.price_candidates}")

    volume_col = _find_column(work, cfg.volume_candidates)
    open_col = _find_column(work, cfg.open_candidates)
    high_col = _find_column(work, cfg.high_candidates)
    low_col = _find_column(work, cfg.low_candidates)
    close_col = _find_column(work, cfg.close_candidates)

    work["__price__"] = pd.to_numeric(work[price_col], errors="coerce")
    work["__volume__"] = (
        pd.to_numeric(work[volume_col], errors="coerce")
        if volume_col is not None
        else pd.Series(np.nan, index=work.index, dtype=np.float64)
    )
    work["__open__"] = (
        pd.to_numeric(work[open_col], errors="coerce")
        if open_col is not None
        else pd.Series(np.nan, index=work.index, dtype=np.float64)
    )
    work["__high__"] = (
        pd.to_numeric(work[high_col], errors="coerce")
        if high_col is not None
        else pd.Series(np.nan, index=work.index, dtype=np.float64)
    )
    work["__low__"] = (
        pd.to_numeric(work[low_col], errors="coerce")
        if low_col is not None
        else pd.Series(np.nan, index=work.index, dtype=np.float64)
    )
    work["__close__"] = (
        pd.to_numeric(work[close_col], errors="coerce")
        if close_col is not None
        else work["__price__"]
    )

    work = work.dropna(subset=["__date__", "__price__"]).sort_values("__date__").reset_index(drop=True)
    if len(work) < 32:
        raise ValueError("Too few rows after basic cleaning.")

    price = work["__price__"].where(work["__price__"] > 0.0)
    volume = work["__volume__"].where(work["__volume__"] > 0.0)
    open_ = work["__open__"].where(work["__open__"] > 0.0)
    high = work["__high__"].where(work["__high__"] > 0.0)
    low = work["__low__"].where(work["__low__"] > 0.0)
    close = work["__close__"].where(work["__close__"] > 0.0)

    log_price = np.log(price)
    log_ret_1d = log_price.diff()

    feature_map: dict[str, pd.Series] = {"log_ret_1d": log_ret_1d}

    if volume_col is not None:
        log_volume = np.log(volume)
        feature_map["log_volume"] = log_volume
        feature_map["log_volume_change"] = log_volume.diff()
    else:
        zeros = pd.Series(0.0, index=work.index, dtype=np.float64)
        feature_map["log_volume"] = zeros
        feature_map["log_volume_change"] = zeros

    if cfg.include_ohlc_features:
        if open_col is not None and close_col is not None:
            feature_map["log_ret_oc"] = np.log(close / open_)
            feature_map["log_gap_open_prev_close"] = np.log(open_ / close.shift(1))
        if high_col is not None and low_col is not None:
            feature_map["log_range_hl"] = np.log(high / low)

    feature_df = pd.DataFrame(feature_map)
    merged = pd.DataFrame(
        {
            "date": work["__date__"],
            "price": price,
            "target_return": log_ret_1d,
        }
    )
    merged = pd.concat([merged, feature_df], axis=1)

    feature_names = list(feature_map.keys())
    if cfg.fillna_method == "ffill":
        merged[feature_names] = merged[feature_names].ffill()
    merged = merged.dropna(subset=["date", "price", "target_return"] + feature_names).reset_index(drop=True)
    if len(merged) < 32:
        raise ValueError("Too few rows after feature generation.")

    return PreparedData(
        dates=merged["date"].to_numpy(),
        prices=merged["price"].to_numpy(dtype=np.float64),
        returns=merged["target_return"].to_numpy(dtype=np.float64),
        features=merged[feature_names].to_numpy(dtype=np.float32),
        feature_names=feature_names,
    )


def build_rolling_windows(prepared: PreparedData, input_length: int) -> WindowedData:
    if input_length < 8:
        raise ValueError("input_length must be >= 8.")
    if len(prepared.dates) <= input_length:
        raise ValueError("Not enough rows to build rolling windows.")

    sample_count = len(prepared.dates) - input_length
    feature_count = prepared.features.shape[1]
    x = np.empty((sample_count, input_length, feature_count), dtype=np.float32)
    y = np.empty(sample_count, dtype=np.float32)
    base_prices = np.empty(sample_count, dtype=np.float64)
    target_prices = np.empty(sample_count, dtype=np.float64)
    target_dates = np.empty(sample_count, dtype=object)

    for i in range(sample_count):
        end_idx = i + input_length - 1
        x[i] = prepared.features[i : i + input_length]
        y[i] = np.float32(prepared.returns[end_idx + 1])  # r_{t+1}
        base_prices[i] = prepared.prices[end_idx]  # P_t
        target_prices[i] = prepared.prices[end_idx + 1]  # P_{t+1}
        target_dates[i] = prepared.dates[end_idx + 1]

    return WindowedData(x=x, y=y, base_prices=base_prices, target_prices=target_prices, target_dates=target_dates)


def split_time_series_indices(
    sample_count: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_count < 30:
        raise ValueError("Need at least 30 rolling samples.")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("train_ratio/val_ratio are invalid.")

    train_end = int(sample_count * train_ratio)
    val_end = int(sample_count * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, sample_count - 2))
    val_end = max(train_end + 1, min(val_end, sample_count - 1))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, sample_count, dtype=np.int64)
    return train_idx, val_idx, test_idx


def fit_feature_scaler(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = train_x.reshape(-1, train_x.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    error = y_true.unsqueeze(1) - y_pred
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def quantile_crossing_penalty(y_pred: torch.Tensor) -> torch.Tensor:
    if y_pred.size(1) < 2:
        return y_pred.new_tensor(0.0)
    return torch.relu(y_pred[:, :-1] - y_pred[:, 1:]).mean()


def _resolve_device(device_hint: str | None) -> torch.device:
    if device_hint is not None:
        return torch.device(device_hint)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return ("out of memory" in message) or ("can't allocate memory" in message)


def _build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        RollingWindowDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _train_one_attempt(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    quantiles: np.ndarray,
    config: PatchTSTConfig,
    device: torch.device,
    verbose: bool,
) -> tuple[PatchTSTQuantileModel, list[dict[str, float]], float]:
    model = PatchTSTQuantileModel(
        input_length=config.input_length,
        input_channels=train_x.shape[-1],
        quantile_count=int(len(quantiles)),
        config=config,
    ).to(device)

    pin_memory = bool(device.type == "cuda")
    train_loader = _build_loader(
        x=train_x,
        y=train_y,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _build_loader(
        x=val_x,
        y=val_y,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    q_tensor = torch.from_numpy(quantiles.astype(np.float32)).to(device)
    grad_acc_steps = max(1, int(config.gradient_accumulation_steps))

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (batch_x, batch_y) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            batch_y = batch_y.to(device, non_blocking=pin_memory)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(batch_x)
                loss = pinball_loss(batch_y, pred, q_tensor)
                if (not config.enforce_monotonic) and config.crossing_penalty_lambda > 0.0:
                    loss = loss + (config.crossing_penalty_lambda * quantile_crossing_penalty(pred))

            loss_for_backward = loss / grad_acc_steps
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if (step % grad_acc_steps == 0) or (step == len(train_loader)):
                if config.grad_clip > 0.0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size_now = int(batch_x.size(0))
            train_loss_sum += float(loss.detach().cpu()) * batch_size_now
            train_count += batch_size_now

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=pin_memory)
                batch_y = batch_y.to(device, non_blocking=pin_memory)
                pred = model(batch_x)
                loss = pinball_loss(batch_y, pred, q_tensor)
                if (not config.enforce_monotonic) and config.crossing_penalty_lambda > 0.0:
                    loss = loss + (config.crossing_penalty_lambda * quantile_crossing_penalty(pred))
                batch_size_now = int(batch_x.size(0))
                val_loss_sum += float(loss.detach().cpu()) * batch_size_now
                val_count += batch_size_now

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if verbose:
            print(
                f"[Epoch {epoch:03d}] train_pinball={train_loss:.6f} "
                f"val_pinball={val_loss:.6f} batch_size={config.batch_size}"
            )

        if val_loss + config.min_delta < best_val:
            best_val = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


def _train_with_memory_fallback(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    quantiles: np.ndarray,
    config: PatchTSTConfig,
    device: torch.device,
    verbose: bool,
) -> tuple[PatchTSTQuantileModel, list[dict[str, float]], float, int]:
    batch_candidates: list[int] = []
    current = max(1, int(config.batch_size))
    min_batch = max(1, int(config.min_batch_size))
    while current >= min_batch:
        if current not in batch_candidates:
            batch_candidates.append(current)
        if current == min_batch:
            break
        current = max(min_batch, current // 2)

    last_error: RuntimeError | None = None
    for batch_size in batch_candidates:
        attempt_cfg = replace(config, batch_size=batch_size)
        try:
            model, history, best_val = _train_one_attempt(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                quantiles=quantiles,
                config=attempt_cfg,
                device=device,
                verbose=verbose,
            )
            return model, history, best_val, batch_size
        except RuntimeError as exc:
            if _is_oom_error(exc):
                last_error = exc
                if verbose:
                    print(f"OOM detected with batch_size={batch_size}. Retrying with smaller batch.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError("Training failed due to memory pressure at all batch sizes.") from last_error


@torch.no_grad()
def predict_quantiles_from_array(
    model: PatchTSTQuantileModel,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, model.quantile_count), dtype=np.float32)

    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, len(x), max(1, batch_size)):
        end = min(len(x), start + max(1, batch_size))
        batch_x = torch.from_numpy(x[start:end].astype(np.float32, copy=False)).to(device)
        pred = model(batch_x).detach().cpu().numpy()
        if not model.enforce_monotonic:
            pred = np.sort(pred, axis=1)
        preds.append(pred.astype(np.float32, copy=False))
    return np.concatenate(preds, axis=0)


def _evaluate_pinball_on_array(
    model: PatchTSTQuantileModel,
    x: np.ndarray,
    y: np.ndarray,
    quantiles: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    pred = predict_quantiles_from_array(model=model, x=x, device=device, batch_size=batch_size).astype(np.float64)
    y_col = y.astype(np.float64)[:, None]
    error = y_col - pred
    q_row = quantiles.astype(np.float64)[None, :]
    pinball = np.maximum(q_row * error, (q_row - 1.0) * error)
    return float(pinball.mean())


def train_patchtst_quantile(
    df: pd.DataFrame,
    config: PatchTSTConfig | None = None,
    feature_config: FeatureConfig | None = None,
    quantiles: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = config or PatchTSTConfig()
    feat_cfg = feature_config or FeatureConfig()
    probs = np.asarray(quantiles if quantiles is not None else DEFAULT_QUANTILES, dtype=np.float32)
    if probs.ndim != 1 or len(probs) < 3:
        raise ValueError("quantiles must be a 1D array with at least 3 values.")
    if np.any(np.diff(probs) <= 0.0):
        raise ValueError("quantiles must be strictly increasing.")

    _set_seed(cfg.seed)
    prepared = prepare_timeseries_dataframe(df, feat_cfg)
    windows = build_rolling_windows(prepared=prepared, input_length=cfg.input_length)
    train_idx, val_idx, test_idx = split_time_series_indices(
        sample_count=len(windows.y),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    train_x = windows.x[train_idx]
    val_x = windows.x[val_idx]
    test_x = windows.x[test_idx]
    train_y = windows.y[train_idx]
    val_y = windows.y[val_idx]
    test_y = windows.y[test_idx]

    scaler_mean, scaler_std = fit_feature_scaler(train_x)
    train_x_scaled = apply_feature_scaler(train_x, scaler_mean, scaler_std)
    val_x_scaled = apply_feature_scaler(val_x, scaler_mean, scaler_std)
    test_x_scaled = apply_feature_scaler(test_x, scaler_mean, scaler_std)

    device = _resolve_device(cfg.device)
    if verbose:
        print(
            f"Device={device}, samples(train/val/test)="
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}, features={train_x.shape[-1]}"
        )

    model, history, best_val_loss, used_batch_size = _train_with_memory_fallback(
        train_x=train_x_scaled,
        train_y=train_y,
        val_x=val_x_scaled,
        val_y=val_y,
        quantiles=probs,
        config=cfg,
        device=device,
        verbose=verbose,
    )

    eval_batch = max(used_batch_size, 64)
    val_pinball = _evaluate_pinball_on_array(
        model=model,
        x=val_x_scaled,
        y=val_y,
        quantiles=probs,
        device=device,
        batch_size=eval_batch,
    )
    test_pinball = _evaluate_pinball_on_array(
        model=model,
        x=test_x_scaled,
        y=test_y,
        quantiles=probs,
        device=device,
        batch_size=eval_batch,
    )

    test_return_quantiles = predict_quantiles_from_array(
        model=model,
        x=test_x_scaled,
        device=device,
        batch_size=eval_batch,
    ).astype(np.float64)
    test_price_quantiles = windows.base_prices[test_idx][:, None] * np.exp(test_return_quantiles)

    artifact = PatchTSTArtifact(
        model=model,
        config=replace(cfg, batch_size=used_batch_size),
        feature_config=feat_cfg,
        quantiles=probs.astype(np.float32),
        scaler_mean=scaler_mean.astype(np.float32),
        scaler_std=scaler_std.astype(np.float32),
        feature_names=prepared.feature_names,
        device=device,
    )
    return {
        "artifact": artifact,
        "history": history,
        "metrics": {
            "best_val_pinball_loss": float(best_val_loss),
            "val_pinball_loss": float(val_pinball),
            "test_pinball_loss": float(test_pinball),
            "used_batch_size": int(used_batch_size),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        },
        "splits": {
            "train_dates": windows.target_dates[train_idx],
            "val_dates": windows.target_dates[val_idx],
            "test_dates": windows.target_dates[test_idx],
        },
        "test": {
            "dates": windows.target_dates[test_idx],
            "base_prices": windows.base_prices[test_idx],
            "actual_prices": windows.target_prices[test_idx],
            "actual_returns": windows.y[test_idx],
            "pred_return_quantiles": test_return_quantiles,
            "pred_price_quantiles": test_price_quantiles,
            "quantiles": probs.astype(np.float32),
        },
    }


def _next_business_day(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    next_day = ts + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    return next_day.normalize()


def _to_float_list(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(v) for v in arr]


def _nearest_quantile_index(quantiles: np.ndarray, target_tau: float) -> int:
    return int(np.argmin(np.abs(np.asarray(quantiles, dtype=np.float64) - float(target_tau))))


def _split_meta(dates: np.ndarray) -> dict[str, Any]:
    if len(dates) == 0:
        return {"count": 0, "from": None, "to": None}
    first = pd.Timestamp(dates[0]).date().isoformat()
    last = pd.Timestamp(dates[-1]).date().isoformat()
    return {"count": int(len(dates)), "from": first, "to": last}


def _estimate_cdf_at_zero(return_quantiles: np.ndarray, quantiles: np.ndarray) -> float:
    x = np.asarray(return_quantiles, dtype=np.float64)
    p = np.asarray(quantiles, dtype=np.float64)
    if x.size == 0:
        return 0.5
    if 0.0 <= x[0]:
        return 0.0
    if 0.0 >= x[-1]:
        return 1.0
    return float(np.interp(0.0, x, p))


def _build_representative_curves(
    quantiles: np.ndarray,
    target_dates: np.ndarray,
    pred_returns: np.ndarray,
    pred_prices: np.ndarray,
    actual_returns: np.ndarray,
    actual_prices: np.ndarray,
    representative_days: int,
) -> list[dict[str, Any]]:
    count = len(target_dates)
    if count == 0:
        return []
    rep = max(1, min(int(representative_days), count))
    indices = np.linspace(0, count - 1, rep, dtype=np.int64)
    curves: list[dict[str, Any]] = []
    for idx in indices:
        curves.append(
            {
                "date": pd.Timestamp(target_dates[idx]).date().isoformat(),
                "actual_return": float(actual_returns[idx]),
                "actual_price": float(actual_prices[idx]),
                "return_quantiles": _to_float_list(pred_returns[idx]),
                "price_quantiles": _to_float_list(pred_prices[idx]),
            }
        )
    return curves


def _backtest_recent_days(
    pred_return_quantiles: np.ndarray,
    realized_log_returns: np.ndarray,
    dates: np.ndarray,
    lookback_days: int = 60,
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    if len(dates) == 0:
        return {
            "days": 0,
            "from": None,
            "to": None,
            "final_capital_strategy": float(initial_capital),
            "final_capital_buy_hold": float(initial_capital),
            "final_return_strategy": 0.0,
            "final_return_buy_hold": 0.0,
            "outperformance": 0.0,
            "avg_allocation_stock": 0.0,
            "avg_cap_stock": 0.0,
            "capped_days": 0,
            "path": [],
        }

    n = len(dates)
    start = max(0, n - int(lookback_days))
    pred = np.asarray(pred_return_quantiles[start:], dtype=np.float64)
    realized = np.asarray(realized_log_returns[start:], dtype=np.float64)
    sub_dates = dates[start:]

    if pred.ndim != 2 or len(pred) == 0:
        return {
            "days": 0,
            "from": None,
            "to": None,
            "final_capital_strategy": float(initial_capital),
            "final_capital_buy_hold": float(initial_capital),
            "final_return_strategy": 0.0,
            "final_return_buy_hold": 0.0,
            "outperformance": 0.0,
            "avg_allocation_stock": 0.0,
            "avg_cap_stock": 0.0,
            "capped_days": 0,
            "path": [],
        }

    q50_idx = pred.shape[1] // 2
    strategy_capital = float(initial_capital)
    buy_hold_capital = float(initial_capital)
    cash_capital = float(initial_capital)

    allocations: list[float] = []
    cap_allocations: list[float] = []
    capped_days = 0
    path: list[dict[str, Any]] = []

    for i in range(len(sub_dates)):
        row = pred[i]
        median_ret = float(row[q50_idx])
        down_prob = float(np.mean(row <= -0.01))
        if median_ret <= 0.0:
            allocation = 0.0
        else:
            # 1日-1%超損失の確率が3%を超えない範囲を優先。
            cap = max(0.0, min(1.0, (0.03 - down_prob) / 0.03))
            allocation = cap
        if allocation < 0.999:
            capped_days += 1

        growth = float(np.exp(realized[i]))
        strategy_capital = (strategy_capital * allocation * growth) + (strategy_capital * (1.0 - allocation))
        buy_hold_capital = buy_hold_capital * growth
        allocations.append(allocation)
        cap_allocations.append(allocation)

        path.append(
            {
                "date": pd.Timestamp(sub_dates[i]).date().isoformat(),
                "allocation_stock": float(allocation),
                "strategy_capital": float(strategy_capital),
                "buy_hold_capital": float(buy_hold_capital),
                "cash_capital": float(cash_capital),
            }
        )

    final_return_strategy = (strategy_capital / initial_capital) - 1.0
    final_return_buy_hold = (buy_hold_capital / initial_capital) - 1.0
    return {
        "days": int(len(sub_dates)),
        "from": pd.Timestamp(sub_dates[0]).date().isoformat(),
        "to": pd.Timestamp(sub_dates[-1]).date().isoformat(),
        "final_capital_strategy": float(strategy_capital),
        "final_capital_buy_hold": float(buy_hold_capital),
        "final_return_strategy": float(final_return_strategy),
        "final_return_buy_hold": float(final_return_buy_hold),
        "outperformance": float(final_return_strategy - final_return_buy_hold),
        "avg_allocation_stock": float(np.mean(allocations)) if allocations else 0.0,
        "avg_cap_stock": float(np.mean(cap_allocations)) if cap_allocations else 0.0,
        "capped_days": int(capped_days),
        "path": path,
    }


def _points_to_dataframe(points: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "date": item.get("t"),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "adj_close": item.get("c"),
                "volume": item.get("v"),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No usable historical rows.")
    return df


def run_patchtst_forecast(
    points: list[dict[str, Any]],
    config_payload: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    runtime_cfg = PatchTSTRuntimeConfig.from_payload(config_payload)
    config = runtime_cfg.model_config

    _emit_progress(progress_callback, 5, "PatchTSTの学習準備を開始しました。")
    df = _points_to_dataframe(points)
    if len(df) < (config.input_length + 190):
        raise ValueError("ヒストリカルデータが不足しています。少なくとも約9か月以上の営業日データが必要です。")

    _emit_progress(progress_callback, 15, "PatchTSTを学習しています。")
    trained = train_patchtst_quantile(
        df=df,
        config=config,
        quantiles=DEFAULT_QUANTILES,
        verbose=False,
    )
    artifact: PatchTSTArtifact = trained["artifact"]

    _emit_progress(progress_callback, 82, "推論結果を整形しています。")
    forecast = predict_next_day_quantiles(df=df, artifact=artifact)

    quantiles = np.asarray(trained["test"]["quantiles"], dtype=np.float64)
    test_dates = np.asarray(trained["test"]["dates"], dtype=object)
    test_returns = np.asarray(trained["test"]["actual_returns"], dtype=np.float64)
    test_prices = np.asarray(trained["test"]["actual_prices"], dtype=np.float64)
    test_pred_returns = np.asarray(trained["test"]["pred_return_quantiles"], dtype=np.float64)
    test_pred_prices = np.asarray(trained["test"]["pred_price_quantiles"], dtype=np.float64)

    q05_idx = _nearest_quantile_index(quantiles, 0.05)
    q25_idx = _nearest_quantile_index(quantiles, 0.25)
    q50_idx = _nearest_quantile_index(quantiles, 0.50)
    q75_idx = _nearest_quantile_index(quantiles, 0.75)
    q95_idx = _nearest_quantile_index(quantiles, 0.95)

    q05 = test_pred_returns[:, q05_idx]
    q25 = test_pred_returns[:, q25_idx]
    q50 = test_pred_returns[:, q50_idx]
    q75 = test_pred_returns[:, q75_idx]
    q95 = test_pred_returns[:, q95_idx]

    coverage_90 = float(np.mean((test_returns >= q05) & (test_returns <= q95)))
    coverage_50 = float(np.mean((test_returns >= q25) & (test_returns <= q75)))

    curves = _build_representative_curves(
        quantiles=quantiles,
        target_dates=test_dates,
        pred_returns=test_pred_returns,
        pred_prices=test_pred_prices,
        actual_returns=test_returns,
        actual_prices=test_prices,
        representative_days=runtime_cfg.representative_days,
    )
    fan_chart = {
        "dates": [pd.Timestamp(d).date().isoformat() for d in test_dates],
        "actual_returns": _to_float_list(test_returns),
        "actual_prices": _to_float_list(test_prices),
        "base_close": _to_float_list(trained["test"]["base_prices"]),
        "q05_returns": _to_float_list(test_pred_returns[:, q05_idx]),
        "q25_returns": _to_float_list(test_pred_returns[:, q25_idx]),
        "q50_returns": _to_float_list(test_pred_returns[:, q50_idx]),
        "q75_returns": _to_float_list(test_pred_returns[:, q75_idx]),
        "q95_returns": _to_float_list(test_pred_returns[:, q95_idx]),
        "q05_prices": _to_float_list(test_pred_prices[:, q05_idx]),
        "q25_prices": _to_float_list(test_pred_prices[:, q25_idx]),
        "q50_prices": _to_float_list(test_pred_prices[:, q50_idx]),
        "q75_prices": _to_float_list(test_pred_prices[:, q75_idx]),
        "q95_prices": _to_float_list(test_pred_prices[:, q95_idx]),
    }

    next_ret_q = np.asarray(forecast["return_quantiles"], dtype=np.float64)
    next_price_q = np.asarray(forecast["price_quantiles"], dtype=np.float64)
    cdf0 = _estimate_cdf_at_zero(next_ret_q, quantiles)
    up_prob = float(max(0.0, min(1.0, 1.0 - cdf0)))
    down_prob = float(max(0.0, min(1.0, cdf0)))
    backtest_60d = _backtest_recent_days(
        pred_return_quantiles=test_pred_returns,
        realized_log_returns=test_returns,
        dates=test_dates,
        lookback_days=60,
        initial_capital=10000.0,
    )
    q05_next_idx = _nearest_quantile_index(quantiles, 0.05)
    q50_next_idx = _nearest_quantile_index(quantiles, 0.50)
    q95_next_idx = _nearest_quantile_index(quantiles, 0.95)

    train_dates = np.asarray(trained["splits"]["train_dates"], dtype=object)
    val_dates = np.asarray(trained["splits"]["val_dates"], dtype=object)
    test_dates_out = np.asarray(trained["splits"]["test_dates"], dtype=object)

    _emit_progress(progress_callback, 100, "PatchTSTの学習と推論が完了しました。")
    return {
        "config": {
            "sequence_length": int(config.input_length),
            "patch_len": int(config.patch_len),
            "stride": int(config.stride),
            "hidden_size": int(config.d_model),
            "n_heads": int(config.n_heads),
            "num_layers": int(config.n_layers),
            "dropout": float(config.dropout),
            "learning_rate": float(config.learning_rate),
            "batch_size": int(artifact.config.batch_size),
            "max_epochs": int(config.max_epochs),
            "patience": int(config.patience),
            "representative_days": int(runtime_cfg.representative_days),
            "seed": int(config.seed),
            "feature_names": list(artifact.feature_names),
            "quantiles": _to_float_list(quantiles),
        },
        "metrics": {
            "mean_pinball_loss": float(trained["metrics"]["test_pinball_loss"]),
            "coverage_90": coverage_90,
            "coverage_50": coverage_50,
        },
        "splits": {
            "train": _split_meta(train_dates),
            "val": _split_meta(val_dates),
            "test": _split_meta(test_dates_out),
        },
        "quantile_function": {
            "taus": _to_float_list(quantiles),
            "curves": curves,
        },
        "fan_chart": fan_chart,
        "training": {
            "epochs_trained": int(len(trained["history"])),
            "best_val_pinball_loss": float(trained["metrics"]["best_val_pinball_loss"]),
            "device": str(artifact.device),
        },
        "backtest_60d": backtest_60d,
        "next_day_forecast": {
            "as_of_date": str(forecast["last_date"]),
            "target_date": str(forecast["next_business_day"]),
            "current_close": float(forecast["last_price"]),
            "up_probability": up_prob,
            "down_probability": down_prob,
            "taus": _to_float_list(quantiles),
            "return_quantiles": _to_float_list(next_ret_q),
            "price_quantiles": _to_float_list(next_price_q),
            "q05_return": float(next_ret_q[q05_next_idx]),
            "q50_return": float(next_ret_q[q50_next_idx]),
            "q95_return": float(next_ret_q[q95_next_idx]),
            "q05_price": float(next_price_q[q05_next_idx]),
            "q50_price": float(next_price_q[q50_next_idx]),
            "q95_price": float(next_price_q[q95_next_idx]),
            "investment_60d": {
                "simulated": False,
                "message": "PatchTST endpoint does not include Monte Carlo investment projection.",
            },
        },
    }


def predict_next_day_quantiles(df: pd.DataFrame, artifact: PatchTSTArtifact) -> dict[str, Any]:
    prepared = prepare_timeseries_dataframe(df, artifact.feature_config)
    if prepared.features.shape[1] != len(artifact.feature_names):
        raise ValueError(
            "Feature count mismatch between training and inference data. "
            f"train={len(artifact.feature_names)}, infer={prepared.features.shape[1]}"
        )
    if len(prepared.features) < artifact.config.input_length:
        raise ValueError("Not enough rows for inference window.")

    recent_x = prepared.features[-artifact.config.input_length :]
    recent_x = apply_feature_scaler(recent_x[None, :, :], artifact.scaler_mean, artifact.scaler_std)
    pred_returns = predict_quantiles_from_array(
        model=artifact.model,
        x=recent_x,
        device=artifact.device,
        batch_size=1,
    )[0].astype(np.float64)

    last_price = float(prepared.prices[-1])
    pred_prices = last_price * np.exp(pred_returns)
    last_date = pd.Timestamp(prepared.dates[-1]).normalize()
    next_business_day = _next_business_day(last_date)

    return {
        "last_date": last_date.date().isoformat(),
        "next_business_day": next_business_day.date().isoformat(),
        "last_price": last_price,
        "quantiles": artifact.quantiles.astype(np.float64),
        "return_quantiles": pred_returns,
        "price_quantiles": pred_prices,
    }


def plot_quantile_curve(
    quantiles: np.ndarray,
    quantile_values: np.ndarray,
    ylabel: str = "Price",
    ax: Any | None = None,
) -> tuple[Any, Any]:
    q = np.asarray(quantiles, dtype=np.float64)
    values = np.asarray(quantile_values, dtype=np.float64)
    if q.shape != values.shape:
        raise ValueError("quantiles and quantile_values must have the same shape.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(q * 100.0, values, color="#0B6E4F", linewidth=2.0)
    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Quantile Curve")
    ax.grid(alpha=0.3)
    return fig, ax


def approximate_pdf_from_quantiles(
    quantiles: np.ndarray,
    quantile_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(quantiles, dtype=np.float64)
    qv = np.asarray(quantile_values, dtype=np.float64)
    if p.shape != qv.shape:
        raise ValueError("quantiles and quantile_values must have the same shape.")
    if p.ndim != 1:
        raise ValueError("quantiles must be 1D.")
    if np.any(np.diff(p) <= 0.0):
        raise ValueError("quantiles must be strictly increasing.")

    qv = np.maximum.accumulate(qv)
    dq_dp = np.gradient(qv, p, edge_order=1)
    dq_dp = np.clip(dq_dp, 1e-8, None)
    pdf = 1.0 / dq_dp

    area = np.trapezoid(pdf, qv)
    if np.isfinite(area) and area > 0.0:
        pdf = pdf / area
    return qv, pdf


def plot_pdf_approximation(
    quantiles: np.ndarray,
    quantile_values: np.ndarray,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    x_grid, pdf = approximate_pdf_from_quantiles(quantiles, quantile_values)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(x_grid, pdf, color="#C85C5C", linewidth=2.0)
    ax.fill_between(x_grid, 0.0, pdf, color="#C85C5C", alpha=0.15)
    ax.set_xlabel("Price")
    ax.set_ylabel("PDF (approx.)")
    ax.set_title("Approximate PDF from Quantile Function")
    ax.grid(alpha=0.3)
    return fig, ax


def _build_dummy_dataframe(n_days: int = 1400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    base_returns = rng.normal(loc=0.0003, scale=0.014, size=n_days)
    regime_shock = rng.normal(loc=0.0, scale=0.004, size=n_days)
    log_price = np.cumsum(base_returns + regime_shock)
    adj_close = 100.0 * np.exp(log_price)

    close = adj_close * np.exp(rng.normal(loc=0.0, scale=0.002, size=n_days))
    open_ = close * np.exp(rng.normal(loc=0.0, scale=0.003, size=n_days))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(loc=0.0, scale=0.004, size=n_days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(loc=0.0, scale=0.004, size=n_days)))
    volume = rng.lognormal(mean=15.0, sigma=0.35, size=n_days).astype(np.int64)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj_close,
            "volume": volume,
        }
    )


def demo_train_predict_plot() -> None:
    # 8GB メモリを意識した軽量寄り設定。
    config = PatchTSTConfig(
        input_length=256,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=2,
        n_layers=2,
        ff_dim=128,
        batch_size=16,
        min_batch_size=4,
        gradient_accumulation_steps=2,
        max_epochs=8,
        patience=3,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        channel_independence=True,
    )

    df = _build_dummy_dataframe(n_days=1400, seed=11)
    trained = train_patchtst_quantile(df=df, config=config, verbose=True)
    artifact: PatchTSTArtifact = trained["artifact"]
    forecast = predict_next_day_quantiles(df, artifact)

    q = forecast["quantiles"]
    price_q = forecast["price_quantiles"]
    q001 = int(np.argmin(np.abs(q - 0.001)))
    q500 = int(np.argmin(np.abs(q - 0.500)))
    q999 = int(np.argmin(np.abs(q - 0.999)))

    print(f"Next business day: {forecast['next_business_day']}")
    print(
        "Predicted price quantiles: "
        f"q0.1%={price_q[q001]:.4f}, q50%={price_q[q500]:.4f}, q99.9%={price_q[q999]:.4f}"
    )
    print("Metrics:", trained["metrics"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_quantile_curve(q, price_q, ylabel="Price", ax=axes[0])
    plot_pdf_approximation(q, price_q, ax=axes[1])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_train_predict_plot()
