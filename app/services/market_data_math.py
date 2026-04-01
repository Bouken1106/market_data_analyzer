"""Numeric helpers shared by market-data query services."""

from __future__ import annotations

import math
from typing import Any

from ..ohlcv import latest_session_points


def moving_average(points: list[dict[str, Any]], window: int) -> float | None:
    closes = [item["c"] for item in points if isinstance(item.get("c"), (int, float))]
    if len(closes) < window or window <= 0:
        return None
    sample = closes[-window:]
    return sum(sample) / window


def atr(points: list[dict[str, Any]], window: int = 14) -> float | None:
    if len(points) < window + 1:
        return None
    trs: list[float] = []
    prev_close = points[0]["c"]
    for item in points[1:]:
        high = item.get("h")
        low = item.get("l")
        close = item.get("c")
        if not isinstance(high, (int, float)) or not isinstance(low, (int, float)) or not isinstance(close, (int, float)):
            prev_close = close if isinstance(close, (int, float)) else prev_close
            continue
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if tr >= 0:
            trs.append(tr)
        prev_close = close
    if len(trs) < window:
        return None
    sample = trs[-window:]
    return sum(sample) / window


def intraday_vwap(points: list[dict[str, Any]]) -> float | None:
    if not points:
        return None
    latest_session = latest_session_points(points)
    if not latest_session:
        return None
    pv_sum = 0.0
    v_sum = 0.0
    for item in latest_session:
        close = item.get("c")
        volume = item.get("v")
        if not isinstance(close, (int, float)) or not isinstance(volume, (int, float)) or volume <= 0:
            continue
        pv_sum += close * volume
        v_sum += volume
    if v_sum <= 0:
        return None
    return pv_sum / v_sum


def daily_returns(points: list[dict[str, Any]], max_len: int) -> dict[str, float]:
    closes: list[tuple[str, float]] = []
    for item in points:
        raw_t = str(item.get("t", "")).strip()
        close = item.get("c")
        if not raw_t or not isinstance(close, (int, float)) or close <= 0:
            continue
        closes.append((raw_t.split(" ")[0], close))
    if len(closes) < 2:
        return {}
    target = closes[-(max_len + 1) :]
    out: dict[str, float] = {}
    for idx in range(1, len(target)):
        date_key, close_value = target[idx]
        prev_close = target[idx - 1][1]
        if prev_close <= 0:
            continue
        out[date_key] = (close_value / prev_close) - 1
    return out


def beta_and_corr(
    symbol_points: list[dict[str, Any]],
    benchmark_points: list[dict[str, Any]],
    *,
    max_len: int = 60,
    min_overlap: int = 20,
) -> tuple[float | None, float | None]:
    symbol_returns = daily_returns(symbol_points, max_len=max_len)
    benchmark_returns = daily_returns(benchmark_points, max_len=max_len)
    common_dates = sorted(set(symbol_returns.keys()) & set(benchmark_returns.keys()))
    if len(common_dates) < min_overlap:
        return None, None

    x = [benchmark_returns[d] for d in common_dates]
    y = [symbol_returns[d] for d in common_dates]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    cov = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x, y)) / max(1, len(x) - 1)
    var_x = sum((xv - mean_x) ** 2 for xv in x) / max(1, len(x) - 1)
    var_y = sum((yv - mean_y) ** 2 for yv in y) / max(1, len(y) - 1)
    if var_x <= 0 or var_y <= 0:
        return None, None
    beta = cov / var_x
    corr = cov / math.sqrt(var_x * var_y)
    return beta, corr
