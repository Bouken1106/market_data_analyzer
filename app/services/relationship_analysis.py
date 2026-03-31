"""Relationship analysis helpers for multi-symbol historical data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..strategy_engine import build_price_matrix, compute_returns


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
    parsed = float(value)
    if not np.isfinite(parsed):
        return None
    return parsed


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
        corr = np.corrcoef(x[end - size:end], y[end - size:end])[0, 1]
        series.append(
            {
                "date": dates[end - 1],
                "value": _safe_scalar(corr),
            }
        )
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
    beta = covariance / variance
    return _safe_scalar(beta)


def build_relationship_analysis(
    points_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    window_days: int = 60,
    top_pairs: int = 10,
    rolling_pair_limit: int = 3,
) -> dict[str, Any]:
    price_dates, prices, symbols = build_price_matrix(points_by_symbol)
    if not price_dates or prices.shape[0] < 4 or len(symbols) < 2:
        raise ValueError("Not enough aligned historical data to analyze relationships.")

    returns = compute_returns(prices)
    return_dates = price_dates[1:]
    if returns.shape[0] < 3:
        raise ValueError("Not enough return observations to analyze relationships.")

    correlation = np.corrcoef(returns, rowvar=False)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    correlation = np.atleast_2d(np.asarray(correlation, dtype=np.float64))
    covariance = np.atleast_2d(np.asarray(covariance, dtype=np.float64))

    pair_candidates: list[dict[str, Any]] = []
    for left_index in range(len(symbols)):
        for right_index in range(left_index + 1, len(symbols)):
            left_returns = returns[:, left_index]
            right_returns = returns[:, right_index]
            left_prices = prices[:, left_index]
            right_prices = prices[:, right_index]
            corr = float(correlation[left_index, right_index])
            pair_candidates.append(
                {
                    "left": symbols[left_index],
                    "right": symbols[right_index],
                    "correlation": _safe_scalar(corr),
                    "abs_correlation": abs(corr) if np.isfinite(corr) else None,
                    "covariance": _safe_scalar(covariance[left_index, right_index]),
                    "beta_left_to_right": _beta(left_returns, right_returns),
                    "corr_20d": _window_corr(left_returns, right_returns, 20),
                    "corr_60d": _window_corr(left_returns, right_returns, 60),
                    "corr_120d": _window_corr(left_returns, right_returns, 120),
                    "observations": int(returns.shape[0]),
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

    rolling_pairs: list[dict[str, Any]] = []
    for item in pair_candidates[: max(1, int(rolling_pair_limit))]:
        left = str(item["left"])
        right = str(item["right"])
        left_index = symbols.index(left)
        right_index = symbols.index(right)
        rolling_pairs.append(
            {
                "left": left,
                "right": right,
                "series": _rolling_corr_series(
                    return_dates,
                    returns[:, left_index],
                    returns[:, right_index],
                    max(3, int(window_days)),
                ),
            }
        )

    avg_abs_corr_by_symbol: list[dict[str, Any]] = []
    for index, symbol in enumerate(symbols):
        values = [abs(float(correlation[index, col])) for col in range(len(symbols)) if col != index]
        mean_abs_corr = float(np.mean(values)) if values else float("nan")
        avg_abs_corr_by_symbol.append(
            {
                "symbol": symbol,
                "mean_abs_correlation": _safe_scalar(mean_abs_corr),
            }
        )

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

    off_diag_values = [
        abs(float(correlation[row, col]))
        for row in range(len(symbols))
        for col in range(row + 1, len(symbols))
        if np.isfinite(correlation[row, col])
    ]

    return {
        "symbols": symbols,
        "data_summary": {
            "from": price_dates[0],
            "to": price_dates[-1],
            "price_points": int(prices.shape[0]),
            "return_points": int(returns.shape[0]),
            "window_days": int(window_days),
        },
        "summary": {
            "average_abs_correlation": _safe_scalar(float(np.mean(off_diag_values))) if off_diag_values else None,
            "most_connected_symbol": strongest,
            "most_diversifying_symbol": weakest,
        },
        "correlation_matrix": _matrix_to_rows(symbols, correlation),
        "covariance_matrix": _matrix_to_rows(symbols, covariance),
        "pair_candidates": pair_candidates[: max(1, int(top_pairs))],
        "rolling_correlations": rolling_pairs,
    }
