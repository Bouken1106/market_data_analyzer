"""Data preparation utilities for the lead-lag PCA workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_adapter import HistoricalPointBatch
from .schemas import LeadLagConfig

_MIN_CFULL_OBSERVATIONS_CAP = 20


@dataclass(frozen=True)
class PreparedLeadLagDataset:
    """Prepared return matrices and aligned metadata."""

    us_symbols: tuple[str, ...]
    jp_symbols: tuple[str, ...]
    combined_symbols: tuple[str, ...]
    rcc_all: pd.DataFrame
    z_all: pd.DataFrame
    z_us: pd.DataFrame
    roc_jp: pd.DataFrame
    cfull_source: pd.DataFrame
    candidate_signal_dates: tuple[pd.Timestamp, ...]
    next_target_by_signal_date: dict[pd.Timestamp, pd.Timestamp]
    excluded_symbols: dict[str, str]
    fetch_failures: dict[str, str]
    point_counts: dict[str, int]


def _points_to_price_series(points: list[dict[str, Any]], field: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for item in points:
        if not isinstance(item, dict):
            continue
        day = str(item.get("t") or "").split(" ")[0]
        if not day:
            continue
        try:
            index_value = pd.Timestamp(day)
            price = float(item.get(field))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price) or price <= 0:
            continue
        values[index_value] = price
    if not values:
        return pd.Series(dtype=np.float64)
    return pd.Series(values, dtype=np.float64).sort_index()


def _build_price_frame(symbols: tuple[str, ...], points_by_symbol: dict[str, list[dict[str, Any]]], field: str) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        series = _points_to_price_series(points_by_symbol.get(symbol, []), field)
        if not series.empty:
            columns[symbol] = series[~series.index.duplicated(keep="last")]
    if not columns:
        return pd.DataFrame(dtype=np.float64)
    return pd.DataFrame(columns).sort_index()


def compute_available_rolling_zscores(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Standardize each series using its own previous ``window`` valid observations."""

    zscores = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=np.float64)
    lookback = max(1, int(window))
    for column in returns.columns:
        valid = returns[column].dropna()
        if valid.empty:
            continue
        rolling_mean = valid.shift(1).rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = valid.shift(1).rolling(window=lookback, min_periods=lookback).std(ddof=0)
        standardized = (valid - rolling_mean) / rolling_std.replace(0.0, np.nan)
        zscores.loc[standardized.index, column] = standardized
    return zscores


def _filter_symbols(
    symbols: tuple[str, ...],
    returns: pd.DataFrame,
    *,
    rolling_window_days: int,
    cfull_mask: pd.Series,
    excluded_symbols: dict[str, str],
    require_target: pd.DataFrame | None = None,
) -> tuple[str, ...]:
    included: list[str] = []
    min_total_observations = rolling_window_days + 1
    min_cfull_observations = max(2, min(_MIN_CFULL_OBSERVATIONS_CAP, rolling_window_days))

    for symbol in symbols:
        if symbol in excluded_symbols:
            continue
        if symbol not in returns.columns:
            excluded_symbols[symbol] = "No return history available."
            continue

        total_count = int(returns[symbol].notna().sum())
        cfull_count = int(returns.loc[cfull_mask, symbol].notna().sum())
        if total_count < min_total_observations:
            excluded_symbols[symbol] = (
                f"Insufficient close-to-close history: total={total_count}, required>={min_total_observations}."
            )
            continue
        if cfull_count < min_cfull_observations:
            excluded_symbols[symbol] = (
                f"Insufficient Cfull observations: period_count={cfull_count}, required>={min_cfull_observations}."
            )
            continue
        if require_target is not None:
            target_count = int(require_target[symbol].notna().sum()) if symbol in require_target.columns else 0
            if target_count < 2:
                excluded_symbols[symbol] = "Insufficient open-to-close target history."
                continue
        included.append(symbol)

    return tuple(included)


def _build_next_target_mapping(
    signal_dates: tuple[pd.Timestamp, ...],
    roc_jp: pd.DataFrame,
) -> dict[pd.Timestamp, pd.Timestamp]:
    active_target_dates = roc_jp.dropna(how="all").index
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    if active_target_dates.empty:
        return mapping

    for signal_date in signal_dates:
        next_index = active_target_dates.searchsorted(signal_date, side="right")
        if next_index >= len(active_target_dates):
            continue
        mapping[signal_date] = active_target_dates[next_index]
    return mapping


def prepare_leadlag_dataset(config: LeadLagConfig, batch: HistoricalPointBatch) -> PreparedLeadLagDataset:
    """Convert raw OHLC points into aligned return matrices for the algorithm."""

    excluded_symbols = dict(batch.failures)
    close_us = _build_price_frame(config.us_symbols, batch.points_by_symbol, "c")
    close_jp = _build_price_frame(config.jp_symbols, batch.points_by_symbol, "c")
    open_jp = _build_price_frame(config.jp_symbols, batch.points_by_symbol, "o")

    if close_us.empty:
        raise ValueError("No U.S. close series could be prepared from the fetched data.")
    if close_jp.empty or open_jp.empty:
        raise ValueError("No Japan open/close series could be prepared from the fetched data.")

    rcc_us = close_us.div(close_us.shift(1)).subtract(1.0)
    rcc_jp = close_jp.div(close_jp.shift(1)).subtract(1.0)
    roc_jp = close_jp.div(open_jp).subtract(1.0)

    cfull_start = pd.Timestamp(config.cfull_start)
    cfull_end = pd.Timestamp(config.cfull_end)
    cfull_us_mask = (rcc_us.index >= cfull_start) & (rcc_us.index <= cfull_end)
    cfull_jp_mask = (rcc_jp.index >= cfull_start) & (rcc_jp.index <= cfull_end)

    included_us = _filter_symbols(
        config.us_symbols,
        rcc_us,
        rolling_window_days=config.rolling_window_days,
        cfull_mask=pd.Series(cfull_us_mask, index=rcc_us.index),
        excluded_symbols=excluded_symbols,
    )
    included_jp = _filter_symbols(
        config.jp_symbols,
        rcc_jp,
        rolling_window_days=config.rolling_window_days,
        cfull_mask=pd.Series(cfull_jp_mask, index=rcc_jp.index),
        excluded_symbols=excluded_symbols,
        require_target=roc_jp,
    )

    if not included_us:
        raise ValueError("No eligible U.S. symbols remain after data sufficiency checks.")
    if not included_jp:
        raise ValueError("No eligible Japan symbols remain after data sufficiency checks.")

    rcc_all = pd.concat([rcc_us.loc[:, list(included_us)], rcc_jp.loc[:, list(included_jp)]], axis=1).sort_index()
    if rcc_all.dropna(how="all").empty:
        raise ValueError("The combined return matrix is empty after filtering symbols.")

    z_all = compute_available_rolling_zscores(rcc_all, config.rolling_window_days)
    z_us = z_all.loc[:, list(included_us)]
    roc_jp_active = roc_jp.loc[:, list(included_jp)].sort_index()

    cfull_source = rcc_all.loc[(rcc_all.index >= cfull_start) & (rcc_all.index <= cfull_end)]
    min_cfull_rows = max(2, min(10, config.rolling_window_days))
    if cfull_source.dropna(how="all").shape[0] < min_cfull_rows:
        raise ValueError(
            "Too few observations are available in the requested Cfull period. "
            "Adjust cfull_start/cfull_end or the symbol universe."
        )

    signal_dates = tuple(index_value for index_value in z_us.dropna(how="any").index)
    next_target_by_signal_date = _build_next_target_mapping(signal_dates, roc_jp_active)
    candidate_signal_dates = tuple(
        index_value for index_value in signal_dates if index_value in next_target_by_signal_date
    )
    if not candidate_signal_dates:
        raise ValueError("No valid signal dates were found after aligning U.S. inputs to next Japan sessions.")

    return PreparedLeadLagDataset(
        us_symbols=included_us,
        jp_symbols=included_jp,
        combined_symbols=included_us + included_jp,
        rcc_all=rcc_all,
        z_all=z_all,
        z_us=z_us,
        roc_jp=roc_jp_active,
        cfull_source=cfull_source,
        candidate_signal_dates=candidate_signal_dates,
        next_target_by_signal_date=next_target_by_signal_date,
        excluded_symbols=excluded_symbols,
        fetch_failures=dict(batch.failures),
        point_counts=dict(batch.point_counts),
    )
