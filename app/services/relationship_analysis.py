"""Relationship analysis helpers for multi-symbol historical data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ohlcv import close_values_by_date
from ..utils import finite_float_or_none


@dataclass(frozen=True)
class RelationshipDataset:
    symbols: list[str]
    price_dates: list[str]
    return_dates: list[str]
    prices: np.ndarray
    returns: np.ndarray
    correlation: np.ndarray
    covariance: np.ndarray
    symbol_indices: dict[str, int]


def _build_price_matrix(points_by_symbol: dict[str, list[dict[str, Any]]]) -> tuple[list[str], np.ndarray, list[str]]:
    normalized: dict[str, dict[str, float]] = {}
    common_dates: set[str] | None = None
    ordered_symbols: list[str] = []

    for symbol, points in points_by_symbol.items():
        series = close_values_by_date(points)
        if not series:
            continue
        normalized[symbol] = series
        ordered_symbols.append(symbol)
        series_dates = set(series.keys())
        common_dates = series_dates if common_dates is None else (common_dates & series_dates)

    if not ordered_symbols or not common_dates:
        return [], np.empty((0, 0), dtype=np.float64), []

    price_dates = sorted(common_dates)
    prices = np.asarray(
        [[normalized[symbol][date] for symbol in ordered_symbols] for date in price_dates],
        dtype=np.float64,
    )
    return price_dates, prices, ordered_symbols


def _compute_returns(prices: np.ndarray) -> np.ndarray:
    if prices.ndim != 2 or prices.shape[0] < 2:
        return np.empty((0, 0), dtype=np.float64)
    previous = prices[:-1, :]
    current = prices[1:, :]
    return (current / previous) - 1.0


def _matrix_to_rows(symbols: list[str], matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, symbol in enumerate(symbols):
        values: list[float | None] = []
        for col_index in range(len(symbols)):
            value = float(matrix[row_index, col_index])
            values.append(value if np.isfinite(value) else None)
        rows.append({"symbol": symbol, "values": values})
    return rows


def _safe_scalar(value: float | np.floating[Any]) -> float | None:
    return finite_float_or_none(value)


def _window_corr(x: np.ndarray, y: np.ndarray, window: int) -> float | None:
    size = min(int(window), int(x.shape[0]), int(y.shape[0]))
    if size < 3:
        return None
    corr = np.corrcoef(x[-size:], y[-size:])[0, 1]
    return _safe_scalar(corr)


def _rolling_corr_series(
    dates: list[str],
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> list[dict[str, Any]]:
    size = int(window)
    if size < 3 or len(dates) < size or x.shape[0] < size or y.shape[0] < size:
        return []

    series: list[dict[str, Any]] = []
    for end in range(size, len(dates) + 1):
        corr = np.corrcoef(x[end - size : end], y[end - size : end])[0, 1]
        series.append({"date": dates[end - 1], "value": _safe_scalar(corr)})
    return series


def _spread_stats(
    left_prices: np.ndarray,
    right_prices: np.ndarray,
    window: int,
) -> dict[str, float | None]:
    size = min(int(window), int(left_prices.shape[0]), int(right_prices.shape[0]))
    if size < 3:
        return {
            "latest_log_spread": None,
            "spread_mean": None,
            "spread_std": None,
            "spread_zscore": None,
        }

    spread = np.log(left_prices[-size:]) - np.log(right_prices[-size:])
    mean = float(np.mean(spread))
    std = float(np.std(spread, ddof=1)) if size >= 2 else float("nan")
    latest = float(spread[-1])
    zscore = (latest - mean) / std if np.isfinite(std) and std > 1e-12 else float("nan")
    return {
        "latest_log_spread": _safe_scalar(latest),
        "spread_mean": _safe_scalar(mean),
        "spread_std": _safe_scalar(std),
        "spread_zscore": _safe_scalar(zscore),
    }


def _beta(left_returns: np.ndarray, right_returns: np.ndarray) -> float | None:
    if left_returns.shape[0] < 2 or right_returns.shape[0] < 2:
        return None
    variance = float(np.var(right_returns, ddof=1))
    if not np.isfinite(variance) or variance <= 1e-12:
        return None
    covariance = float(np.cov(left_returns, right_returns, ddof=1)[0, 1])
    return _safe_scalar(covariance / variance)


def _prepare_dataset(points_by_symbol: dict[str, list[dict[str, Any]]]) -> RelationshipDataset:
    price_dates, prices, symbols = _build_price_matrix(points_by_symbol)
    if not price_dates or prices.shape[0] < 4 or len(symbols) < 2:
        raise ValueError("Not enough aligned historical data to analyze relationships.")

    returns = _compute_returns(prices)
    if returns.shape[0] < 3:
        raise ValueError("Not enough return observations to analyze relationships.")

    correlation = np.atleast_2d(np.asarray(np.corrcoef(returns, rowvar=False), dtype=np.float64))
    covariance = np.atleast_2d(np.asarray(np.cov(returns, rowvar=False, ddof=1), dtype=np.float64))
    return RelationshipDataset(
        symbols=symbols,
        price_dates=price_dates,
        return_dates=price_dates[1:],
        prices=prices,
        returns=returns,
        correlation=correlation,
        covariance=covariance,
        symbol_indices={symbol: index for index, symbol in enumerate(symbols)},
    )


def _build_pair_candidates(dataset: RelationshipDataset, *, window_days: int) -> list[dict[str, Any]]:
    pair_candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(dataset.symbols):
        for right_index in range(left_index + 1, len(dataset.symbols)):
            right = dataset.symbols[right_index]
            left_returns = dataset.returns[:, left_index]
            right_returns = dataset.returns[:, right_index]
            left_prices = dataset.prices[:, left_index]
            right_prices = dataset.prices[:, right_index]
            corr = float(dataset.correlation[left_index, right_index])
            pair_candidates.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": _safe_scalar(corr),
                    "abs_correlation": abs(corr) if np.isfinite(corr) else None,
                    "covariance": _safe_scalar(dataset.covariance[left_index, right_index]),
                    "beta_left_to_right": _beta(left_returns, right_returns),
                    "corr_20d": _window_corr(left_returns, right_returns, 20),
                    "corr_60d": _window_corr(left_returns, right_returns, 60),
                    "corr_120d": _window_corr(left_returns, right_returns, 120),
                    "observations": int(dataset.returns.shape[0]),
                    **_spread_stats(left_prices, right_prices, window_days),
                }
            )

    pair_candidates.sort(
        key=lambda item: (
            -float(item["abs_correlation"]) if isinstance(item.get("abs_correlation"), (int, float)) else math.inf,
            str(item.get("left") or ""),
            str(item.get("right") or ""),
        )
    )
    return pair_candidates


def _build_rolling_pairs(
    dataset: RelationshipDataset,
    pair_candidates: list[dict[str, Any]],
    *,
    rolling_pair_limit: int,
    window_days: int,
) -> list[dict[str, Any]]:
    rolling_pairs: list[dict[str, Any]] = []
    for item in pair_candidates[: max(1, int(rolling_pair_limit))]:
        left = str(item["left"])
        right = str(item["right"])
        left_index = dataset.symbol_indices[left]
        right_index = dataset.symbol_indices[right]
        rolling_pairs.append(
            {
                "left": left,
                "right": right,
                "series": _rolling_corr_series(
                    dataset.return_dates,
                    dataset.returns[:, left_index],
                    dataset.returns[:, right_index],
                    max(3, int(window_days)),
                ),
            }
        )
    return rolling_pairs


def _summarize_symbol_correlation(dataset: RelationshipDataset) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    avg_abs_corr_by_symbol: list[dict[str, Any]] = []
    for index, symbol in enumerate(dataset.symbols):
        values = [abs(float(dataset.correlation[index, col])) for col in range(len(dataset.symbols)) if col != index]
        mean_abs_corr = float(np.mean(values)) if values else float("nan")
        avg_abs_corr_by_symbol.append({"symbol": symbol, "mean_abs_correlation": _safe_scalar(mean_abs_corr)})

    strongest = max(
        avg_abs_corr_by_symbol,
        key=lambda item: float(item["mean_abs_correlation"]) if isinstance(item.get("mean_abs_correlation"), (int, float)) else -1.0,
        default=None,
    )
    weakest = min(
        avg_abs_corr_by_symbol,
        key=lambda item: float(item["mean_abs_correlation"]) if isinstance(item.get("mean_abs_correlation"), (int, float)) else math.inf,
        default=None,
    )
    return avg_abs_corr_by_symbol, strongest, weakest


def _average_abs_correlation(dataset: RelationshipDataset) -> float | None:
    off_diag_values = [
        abs(float(dataset.correlation[row, col]))
        for row in range(len(dataset.symbols))
        for col in range(row + 1, len(dataset.symbols))
        if np.isfinite(dataset.correlation[row, col])
    ]
    return _safe_scalar(float(np.mean(off_diag_values))) if off_diag_values else None


def build_relationship_analysis(
    points_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    window_days: int = 60,
    top_pairs: int = 10,
    rolling_pair_limit: int = 3,
) -> dict[str, Any]:
    dataset = _prepare_dataset(points_by_symbol)
    pair_candidates = _build_pair_candidates(dataset, window_days=window_days)
    rolling_pairs = _build_rolling_pairs(
        dataset,
        pair_candidates,
        rolling_pair_limit=rolling_pair_limit,
        window_days=window_days,
    )
    _, strongest, weakest = _summarize_symbol_correlation(dataset)

    return {
        "symbols": dataset.symbols,
        "data_summary": {
            "from": dataset.price_dates[0],
            "to": dataset.price_dates[-1],
            "price_points": int(dataset.prices.shape[0]),
            "return_points": int(dataset.returns.shape[0]),
            "window_days": int(window_days),
        },
        "summary": {
            "average_abs_correlation": _average_abs_correlation(dataset),
            "most_connected_symbol": strongest,
            "most_diversifying_symbol": weakest,
        },
        "correlation_matrix": _matrix_to_rows(dataset.symbols, dataset.correlation),
        "covariance_matrix": _matrix_to_rows(dataset.symbols, dataset.covariance),
        "pair_candidates": pair_candidates[: max(1, int(top_pairs))],
        "rolling_correlations": rolling_pairs,
    }
