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


def _empty_summary() -> dict[str, Any]:
    return {
        "annual_return_pct": None,
        "annual_volatility_pct": None,
        "return_risk_ratio": None,
        "max_drawdown_pct": None,
        "signal_days": 0,
        "average_breadth": None,
        "range": {
            "from": None,
            "to": None,
        },
        "signal_range": {
            "from": None,
            "to": None,
        },
        "target_range": {
            "from": None,
            "to": None,
        },
    }


def _summarize_daily_rows(daily_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not daily_rows:
        return _empty_summary()

    gross_returns = np.asarray([float(row["gross_return"]) for row in daily_rows], dtype=np.float64)
    equity = 1.0
    equity_curve: list[float] = []
    for gross_return in gross_returns:
        equity *= 1.0 + float(gross_return)
        equity_curve.append(equity)

    annual_return = float(np.mean(gross_returns) * _TRADING_DAYS_PER_YEAR)
    annual_volatility = (
        float(np.std(gross_returns, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if gross_returns.size >= 2
        else float("nan")
    )
    return_risk_ratio = annual_return / annual_volatility if annual_volatility and np.isfinite(annual_volatility) else np.nan

    equity_array = np.asarray([1.0] + equity_curve, dtype=np.float64)
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array / running_max) - 1.0
    max_drawdown = float(np.min(drawdowns)) if drawdowns.size else np.nan

    return {
        "annual_return_pct": _safe_pct(annual_return),
        "annual_volatility_pct": _safe_pct(annual_volatility),
        "return_risk_ratio": float(return_risk_ratio) if np.isfinite(return_risk_ratio) else None,
        "max_drawdown_pct": _safe_pct(max_drawdown),
        "signal_days": len(daily_rows),
        "average_breadth": float(np.mean([float(row["breadth"]) for row in daily_rows])) if daily_rows else None,
        "range": {
            "from": daily_rows[0].get("signal_date"),
            "to": daily_rows[-1].get("signal_date"),
        },
        "signal_range": {
            "from": daily_rows[0].get("signal_date"),
            "to": daily_rows[-1].get("signal_date"),
        },
        "target_range": {
            "from": daily_rows[0].get("target_date"),
            "to": daily_rows[-1].get("target_date"),
        },
    }


def _select_recent_rows(daily_rows: list[dict[str, Any]], *, months: int = 1) -> list[dict[str, Any]]:
    if not daily_rows:
        return []

    latest_signal_date = pd.Timestamp(daily_rows[-1]["signal_date"])
    cutoff = latest_signal_date - pd.DateOffset(months=months)
    recent_rows = [row for row in daily_rows if pd.Timestamp(row["signal_date"]) >= cutoff]
    return recent_rows or daily_rows[-1:]


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
            "summary": _empty_summary(),
            "recent_1m_summary": _empty_summary(),
            "daily_rows": daily_rows,
            "equity_curve": equity_curve,
        }

    summary = _summarize_daily_rows(daily_rows)
    recent_1m_rows = _select_recent_rows(daily_rows, months=1)
    recent_1m_summary = _summarize_daily_rows(recent_1m_rows)

    return {
        "summary": summary,
        "recent_1m_summary": recent_1m_summary,
        "daily_rows": daily_rows,
        "equity_curve": equity_curve,
    }
