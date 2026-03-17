"""Shared NumPy helpers for rolling-window ML pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_feature_windows(
    features: np.ndarray,
    window_size: int,
    *,
    sample_count: int | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if values.shape[0] < window_size:
        raise ValueError("window_size exceeds the available rows.")

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    if sample_count is not None:
        safe_count = max(0, min(int(sample_count), int(windows.shape[0])))
        windows = windows[:safe_count]
    return np.ascontiguousarray(windows, dtype=dtype)


def fit_feature_scaler_3d(
    train_x: np.ndarray,
    *,
    min_std: float = 1e-8,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_x, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("train_x must be a 3D array.")
    flat = values.reshape(-1, values.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < float(min_std), 1.0, std)
    return mean.astype(dtype, copy=False), std.astype(dtype, copy=False)


def apply_feature_scaler_3d(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    dtype: Any = np.float32,
    fill_non_finite: float | None = 0.0,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64).reshape(1, 1, -1)
    std_arr = np.asarray(std, dtype=np.float64).reshape(1, 1, -1)
    out = ((array - mean_arr) / std_arr).astype(dtype, copy=False)
    if fill_non_finite is not None:
        out[~np.isfinite(out)] = float(fill_non_finite)
    return out
