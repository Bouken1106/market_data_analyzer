"""Optional long-short evaluation helpers for the lead-lag module."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .signals import SignalObservation

_TRADING_DAYS_PER_YEAR = 252.0


def _safe_pct(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value * 100.0)


def evaluate_long_short(
    observations: tuple[SignalObservation, ...],
    *,
    quantile_q: float,
) -> dict[str, Any]:
    """Evaluate a simple equal-weight long-short portfolio from predicted JP signals."""

    daily_rows: list[dict[str, Any]] = []
    gross_returns: list[float] = []
    equity_curve: list[dict[str, Any]] = []
    equity = 1.0

    for observation in observations:
        frame = pd.DataFrame(
            {
                "signal": observation.predicted,
                "realized": observation.realized,
            }
        ).dropna()
        if frame.empty:
            continue

        ranked = frame.assign(symbol=frame.index.astype(str)).sort_values(
            ["signal", "symbol"],
            ascending=[False, True],
        )
        bucket_size = int(math.floor(len(ranked) * quantile_q))
        if bucket_size < 1:
            continue

        longs = ranked.head(bucket_size)
        shorts = ranked.tail(bucket_size)
        gross_return = float(longs["realized"].mean() - shorts["realized"].mean())
        equity *= 1.0 + gross_return
        gross_returns.append(gross_return)

        daily_rows.append(
            {
                "signal_date": observation.signal_date.date().isoformat(),
                "target_date": observation.target_date.date().isoformat(),
                "breadth": int(len(ranked)),
                "bucket_size": bucket_size,
                "gross_return": gross_return,
                "long_symbols": list(longs.index),
                "short_symbols": list(shorts.index),
            }
        )
        equity_curve.append(
            {
                "target_date": observation.target_date.date().isoformat(),
                "equity": equity,
            }
        )

    if not gross_returns:
        return {
            "summary": {
                "annual_return_pct": None,
                "annual_volatility_pct": None,
                "return_risk_ratio": None,
                "max_drawdown_pct": None,
                "signal_days": 0,
                "average_breadth": None,
            },
            "daily_rows": daily_rows,
            "equity_curve": equity_curve,
        }

    returns_array = np.asarray(gross_returns, dtype=np.float64)
    annual_return = float(np.mean(returns_array) * _TRADING_DAYS_PER_YEAR)
    annual_volatility = (
        float(np.std(returns_array, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if returns_array.size >= 2
        else float("nan")
    )
    return_risk_ratio = annual_return / annual_volatility if annual_volatility and np.isfinite(annual_volatility) else np.nan

    equity_array = np.asarray([1.0] + [row["equity"] for row in equity_curve], dtype=np.float64)
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array / running_max) - 1.0
    max_drawdown = float(np.min(drawdowns)) if drawdowns.size else np.nan

    return {
        "summary": {
            "annual_return_pct": _safe_pct(annual_return),
            "annual_volatility_pct": _safe_pct(annual_volatility),
            "return_risk_ratio": float(return_risk_ratio) if np.isfinite(return_risk_ratio) else None,
            "max_drawdown_pct": _safe_pct(max_drawdown),
            "signal_days": len(daily_rows),
            "average_breadth": float(np.mean([row["breadth"] for row in daily_rows])) if daily_rows else None,
        },
        "daily_rows": daily_rows,
        "equity_curve": equity_curve,
    }
