"""Portfolio allocation and cost-aware backtest utilities for Strategy Lab."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np


TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class AllocationPlan:
    symbol: str
    weight: float


def _to_close_map(points: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in points:
        if not isinstance(item, dict):
            continue
        raw_t = str(item.get("t") or "").strip()
        if not raw_t:
            continue
        date_key = raw_t.split(" ")[0]
        try:
            _ = date.fromisoformat(date_key)
        except ValueError:
            continue
        try:
            close_value = float(item.get("c"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(close_value) or close_value <= 0:
            continue
        out[date_key] = close_value
    return out


def build_price_matrix(
    points_by_symbol: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], np.ndarray, list[str]]:
    """Build aligned close-price matrix using common timestamps only."""
    maps = {
        symbol: close_map
        for symbol, points in points_by_symbol.items()
        if (close_map := _to_close_map(points))
    }

    if not maps:
        return [], np.empty((0, 0), dtype=np.float64), []

    symbols = sorted(maps.keys())
    common_dates = set.intersection(*(set(maps[symbol].keys()) for symbol in symbols))
    if not common_dates:
        return [], np.empty((0, len(symbols)), dtype=np.float64), symbols

    ordered_dates = sorted(common_dates)
    matrix = np.empty((len(ordered_dates), len(symbols)), dtype=np.float64)
    for col, symbol in enumerate(symbols):
        close_map = maps[symbol]
        matrix[:, col] = np.fromiter(
            (close_map[day] for day in ordered_dates),
            dtype=np.float64,
            count=len(ordered_dates),
        )

    return ordered_dates, matrix, symbols


def compute_returns(prices: np.ndarray) -> np.ndarray:
    if prices.shape[0] < 2:
        return np.empty((0, prices.shape[1] if prices.ndim == 2 else 0), dtype=np.float64)
    return (prices[1:] / prices[:-1]) - 1.0


def _safe_normalize(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        n = int(weights.shape[0])
        if n <= 0:
            return weights
        return np.full((n,), 1.0 / n, dtype=np.float64)
    return weights / total


def _apply_max_weight(weights: np.ndarray, max_weight: float) -> np.ndarray:
    clipped = np.clip(weights, 0.0, max_weight)
    for _ in range(8):
        clipped = _safe_normalize(clipped)
        over = clipped > max_weight
        if not np.any(over):
            break
        clipped[over] = max_weight
    return _safe_normalize(clipped)


def target_weights(
    method: str,
    returns_window: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    n = int(returns_window.shape[1])
    if n <= 0:
        return np.empty((0,), dtype=np.float64)

    clean = returns_window[np.all(np.isfinite(returns_window), axis=1)]
    if clean.shape[0] < 2:
        return np.full((n,), 1.0 / n, dtype=np.float64)

    method_key = str(method or "").strip().lower()
    if method_key == "equal_weight":
        raw = np.full((n,), 1.0 / n, dtype=np.float64)
    elif method_key == "inverse_volatility":
        vol = np.std(clean, axis=0, ddof=1)
        inv_vol = np.where((vol > 1e-12) & np.isfinite(vol), 1.0 / vol, 0.0)
        raw = _safe_normalize(inv_vol)
    elif method_key == "min_variance":
        cov = np.cov(clean, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        cov += np.eye(n, dtype=np.float64) * 1e-6
        ones = np.ones((n,), dtype=np.float64)
        try:
            inv_cov_ones = np.linalg.solve(cov, ones)
        except np.linalg.LinAlgError:
            inv_cov_ones = ones
        raw = np.maximum(inv_cov_ones, 0.0)
        raw = _safe_normalize(raw)
    else:
        raise ValueError("Unknown allocation method.")

    return _apply_max_weight(raw, max_weight=max_weight)


def _is_rebalance_day(freq: str, previous_day: str | None, current_day: str) -> bool:
    if previous_day is None:
        return True
    prev = date.fromisoformat(previous_day)
    cur = date.fromisoformat(current_day)
    key = str(freq or "monthly").strip().lower()
    if key == "daily":
        return True
    if key == "weekly":
        return cur.isocalendar()[:2] != prev.isocalendar()[:2]
    if key == "quarterly":
        prev_q = ((prev.month - 1) // 3) + 1
        cur_q = ((cur.month - 1) // 3) + 1
        return prev.year != cur.year or prev_q != cur_q
    return prev.year != cur.year or prev.month != cur.month


def _performance_metrics(equity_curve: np.ndarray, daily_returns: np.ndarray) -> dict[str, float | None]:
    if equity_curve.size < 2:
        return {
            "total_return_pct": None,
            "cagr_pct": None,
            "volatility_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "win_rate_pct": None,
        }

    start = float(equity_curve[0])
    end = float(equity_curve[-1])
    total_return = (end / start) - 1.0 if start > 0 else np.nan
    years = max((daily_returns.size / TRADING_DAYS_PER_YEAR), 1e-9)
    cagr = (end / start) ** (1.0 / years) - 1.0 if start > 0 and end > 0 else np.nan
    vol = float(np.std(daily_returns, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_returns.size >= 2 else np.nan
    mean_daily = float(np.mean(daily_returns)) if daily_returns.size > 0 else np.nan
    sharpe = ((mean_daily * TRADING_DAYS_PER_YEAR) / vol) if vol and np.isfinite(vol) and vol > 1e-12 else np.nan

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / running_max) - 1.0
    max_dd = float(np.min(drawdowns)) if drawdowns.size else np.nan
    wins = float(np.mean(daily_returns > 0.0)) if daily_returns.size else np.nan

    def pct_or_none(value: float) -> float | None:
        if not np.isfinite(value):
            return None
        return float(value * 100.0)

    def num_or_none(value: float) -> float | None:
        if not np.isfinite(value):
            return None
        return float(value)

    return {
        "total_return_pct": pct_or_none(total_return),
        "cagr_pct": pct_or_none(cagr),
        "volatility_pct": pct_or_none(vol),
        "sharpe": num_or_none(sharpe),
        "max_drawdown_pct": pct_or_none(max_dd),
        "win_rate_pct": pct_or_none(wins),
    }


def run_backtest(
    symbols: list[str],
    return_dates: list[str],
    returns: np.ndarray,
    method: str,
    lookback_days: int,
    rebalance_frequency: str,
    rebalance_threshold_pct: float,
    max_weight: float,
    initial_capital: float,
    transaction_cost_rate: float,
) -> dict[str, Any]:
    n_assets = len(symbols)
    horizon = int(returns.shape[0])
    if n_assets <= 0 or horizon <= 0:
        raise ValueError("Not enough data to run backtest.")

    equity = float(initial_capital)
    weights = np.zeros((n_assets,), dtype=np.float64)
    weight_sum = 0.0
    equity_curve = np.empty(horizon + 1, dtype=np.float64)
    equity_curve[0] = equity
    daily_returns = np.zeros(horizon, dtype=np.float64)
    turnover_sum = 0.0
    cost_sum = 0.0
    rebalance_events = 0
    last_rebalance_date: str | None = None
    threshold = max(0.0, float(rebalance_threshold_pct) / 100.0)
    lookback = max(20, int(lookback_days))

    for idx in range(horizon):
        day = return_dates[idx]
        should_rebalance = False
        if idx >= lookback:
            scheduled = _is_rebalance_day(rebalance_frequency, last_rebalance_date, day)
            desired = target_weights(
                method=method,
                returns_window=returns[idx - lookback:idx, :],
                max_weight=max_weight,
            )
            drift = float(np.max(np.abs(weights - desired))) if weights.size else 0.0
            should_rebalance = scheduled or drift >= threshold or weight_sum <= 1e-12
            if should_rebalance:
                turnover = float(np.sum(np.abs(desired - weights)))
                cost = equity * turnover * transaction_cost_rate
                if cost > 0 and np.isfinite(cost):
                    equity = max(0.0, equity - cost)
                    cost_sum += cost
                turnover_sum += turnover
                weights = desired
                weight_sum = float(np.sum(weights))
                rebalance_events += 1
                last_rebalance_date = day

        day_vector = returns[idx, :]
        day_ret = float(np.dot(weights, day_vector)) if weight_sum > 1e-12 else 0.0
        if np.isfinite(day_ret):
            equity *= (1.0 + day_ret)
        else:
            day_ret = 0.0

        gross = weights * (1.0 + day_vector)
        gross_sum = float(np.sum(gross))
        if gross_sum > 1e-12 and np.all(np.isfinite(gross)):
            weights = gross / gross_sum
            weight_sum = 1.0
        else:
            weights.fill(0.0)
            weight_sum = 0.0

        equity_curve[idx + 1] = equity
        daily_returns[idx] = day_ret

    metrics = _performance_metrics(equity_curve, daily_returns)
    years = max(daily_returns.size / TRADING_DAYS_PER_YEAR, 1e-9)
    annual_turnover = turnover_sum / years
    latest_weights = {symbol: float(weights[pos]) for pos, symbol in enumerate(symbols)}

    return {
        "metrics": {
            **metrics,
            "final_equity": float(equity_curve[-1]) if equity_curve.size else None,
            "avg_annual_turnover": float(annual_turnover) if np.isfinite(annual_turnover) else None,
            "total_trade_cost": float(cost_sum),
            "trade_cost_pct": float((cost_sum / initial_capital) * 100.0) if initial_capital > 0 else None,
            "rebalance_count": int(rebalance_events),
        },
        "series": [
            {"date": return_dates[i], "equity": float(equity_curve[i + 1])}
            for i in range(len(return_dates))
        ],
        "latest_weights": latest_weights,
    }


def estimate_window_stats(
    symbols: list[str],
    returns_window: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    if returns_window.size == 0 or weights.size == 0:
        return {
            "expected_return_pct": None,
            "expected_volatility_pct": None,
            "symbols": [AllocationPlan(symbol=symbol, weight=0.0).__dict__ for symbol in symbols],
        }
    mean_daily = np.mean(returns_window, axis=0)
    cov = np.cov(returns_window, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    exp_daily = float(np.dot(weights, mean_daily))
    exp_annual = ((1.0 + exp_daily) ** TRADING_DAYS_PER_YEAR) - 1.0
    variance = float(np.dot(weights, np.dot(cov, weights)))
    vol_annual = np.sqrt(max(0.0, variance)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "expected_return_pct": float(exp_annual * 100.0) if np.isfinite(exp_annual) else None,
        "expected_volatility_pct": float(vol_annual * 100.0) if np.isfinite(vol_annual) else None,
        "symbols": [AllocationPlan(symbol=symbol, weight=float(weights[i])).__dict__ for i, symbol in enumerate(symbols)],
    }


def buy_and_hold_backtest(
    return_dates: list[str],
    returns: np.ndarray,
    initial_capital: float,
) -> dict[str, Any]:
    if returns.ndim != 1:
        returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    equity = float(initial_capital)
    horizon = int(returns.shape[0])
    equity_curve = np.empty(horizon + 1, dtype=np.float64)
    equity_curve[0] = equity
    series: list[dict[str, Any]] = []
    daily = np.zeros(horizon, dtype=np.float64)
    for idx, value in enumerate(returns):
        r = float(value) if np.isfinite(value) else 0.0
        equity *= (1.0 + r)
        daily[idx] = r
        equity_curve[idx + 1] = equity
        if idx < len(return_dates):
            series.append({"date": return_dates[idx], "equity": float(equity)})
    metrics = _performance_metrics(equity_curve, daily)
    return {
        "metrics": {
            **metrics,
            "final_equity": float(equity),
        },
        "series": series,
    }
