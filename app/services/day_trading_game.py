"""Day-trading game data assembly backed by yfinance intraday history."""

from __future__ import annotations

import asyncio
import math
import random
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from ..utils import finite_float_or_none, normalize_symbol

YFINANCE_INTERVAL = "15m"
YFINANCE_PERIOD = "60d"
GAME_SESSION_DAYS = 3

US_GAME_SYMBOLS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "AVGO",
    "JPM",
    "XOM",
    "UNH",
)

JP_GAME_SYMBOLS: tuple[str, ...] = (
    "7203.T",
    "6758.T",
    "9984.T",
    "8306.T",
    "9432.T",
    "6861.T",
    "6098.T",
    "8035.T",
    "4063.T",
    "8058.T",
    "6501.T",
    "7974.T",
)

JP_GAME_SYMBOL_NAMES: dict[str, str] = {
    "7203.T": "Toyota Motor",
    "6758.T": "Sony Group",
    "9984.T": "SoftBank Group",
    "8306.T": "Mitsubishi UFJ Financial Group",
    "9432.T": "NTT",
    "6861.T": "Keyence",
    "6098.T": "Recruit Holdings",
    "8035.T": "Tokyo Electron",
    "4063.T": "Shin-Etsu Chemical",
    "8058.T": "Mitsubishi Corp.",
    "6501.T": "Hitachi",
    "7974.T": "Nintendo",
}

MARKET_CONFIGS: dict[str, dict[str, Any]] = {
    "us": {
        "label": "US",
        "timezone": "America/New_York",
        "currency": "USD",
        "currency_symbol": "$",
        "currency_digits": 2,
        "symbols": US_GAME_SYMBOLS,
        "min_candles": 12,
    },
    "jp": {
        "label": "Japan",
        "timezone": "Asia/Tokyo",
        "currency": "JPY",
        "currency_symbol": "¥",
        "currency_digits": 0,
        "symbols": JP_GAME_SYMBOLS,
        "symbol_names": JP_GAME_SYMBOL_NAMES,
        "min_candles": 8,
    },
}

HistoryFetcher = Callable[[str], Any]


class DayTradingGameError(Exception):
    """Base class for game data errors."""


class DayTradingGameDependencyError(DayTradingGameError):
    """Raised when the yfinance dependency is unavailable."""


class DayTradingGameDataError(DayTradingGameError):
    """Raised when yfinance does not provide usable intraday data."""


class DayTradingGameRequestError(DayTradingGameError):
    """Raised when a caller supplied invalid game inputs."""


async def build_day_trading_session(
    *,
    market: str = "us",
    symbol: str | None = None,
    rng: random.Random | None = None,
    fetch_history: HistoryFetcher | None = None,
) -> dict[str, Any]:
    """Build one randomly selected multi-day replay session from 15-minute yfinance bars."""

    market_key = str(market or "us").strip().lower()
    config = MARKET_CONFIGS.get(market_key)
    if config is None:
        supported = ", ".join(sorted(MARKET_CONFIGS))
        raise DayTradingGameRequestError(f"Unsupported market. Choose one of: {supported}.")

    selected_symbols = _candidate_symbols(config, symbol=symbol)
    chooser = rng or random.SystemRandom()
    chooser.shuffle(selected_symbols)
    fetcher = fetch_history or fetch_yfinance_history

    failures: list[str] = []
    for candidate in selected_symbols:
        try:
            history = await _maybe_await(fetcher(candidate))
            candles_by_date = _candles_by_session_date(history, timezone_name=str(config["timezone"]))
            eligible = _eligible_session_date_windows(
                candles_by_date,
                timezone_name=str(config["timezone"]),
                min_candles=int(config["min_candles"]),
                session_days=GAME_SESSION_DAYS,
            )
            if not eligible:
                raise DayTradingGameDataError(
                    f"No complete {GAME_SESSION_DAYS}-day 15-minute trading windows were available."
                )
            date_keys = chooser.choice(eligible)
            candles = [
                candle
                for date_key in date_keys
                for candle in candles_by_date[date_key]
            ]
            return _build_session_payload(
                market_key=market_key,
                config=config,
                symbol=candidate,
                symbol_name=_symbol_name(config, candidate),
                date_keys=date_keys,
                candles=candles,
            )
        except DayTradingGameDependencyError:
            raise
        except DayTradingGameError as exc:
            failures.append(f"{candidate}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive around provider internals
            failures.append(f"{candidate}: {exc}")

    detail = "No usable yfinance 15-minute data was available."
    if failures:
        detail = f"{detail} Last errors: {'; '.join(failures[-3:])}"
    raise DayTradingGameDataError(detail)


async def fetch_yfinance_history(symbol: str) -> pd.DataFrame:
    """Fetch 15-minute bars for a symbol via yfinance without blocking the event loop."""

    return await asyncio.to_thread(_fetch_yfinance_history_sync, symbol)


def _fetch_yfinance_history_sync(symbol: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in integration environments
        raise DayTradingGameDependencyError("yfinance is not installed. Run `pip install -r requirements.txt`.") from exc

    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(
            period=YFINANCE_PERIOD,
            interval=YFINANCE_INTERVAL,
            auto_adjust=False,
            prepost=False,
            actions=False,
            timeout=20,
        )
    except TypeError:
        ticker = yf.Ticker(symbol)
        return ticker.history(
            period=YFINANCE_PERIOD,
            interval=YFINANCE_INTERVAL,
            auto_adjust=False,
            prepost=False,
            actions=False,
        )


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _candidate_symbols(config: dict[str, Any], *, symbol: str | None) -> list[str]:
    if symbol:
        normalized = normalize_symbol(symbol)
        if not normalized:
            raise DayTradingGameRequestError("Invalid symbol.")
        return [normalized]
    return list(config["symbols"])


def _symbol_name(config: dict[str, Any], symbol: str) -> str | None:
    symbol_names = config.get("symbol_names")
    if not isinstance(symbol_names, dict):
        return None
    name = symbol_names.get(symbol.upper())
    if not name:
        return None
    return str(name)


def _symbol_label(symbol: str, symbol_name: str | None) -> str:
    if symbol_name:
        return f"{symbol_name} ({symbol})"
    return symbol


def _candles_by_session_date(history: Any, *, timezone_name: str) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise DayTradingGameDataError("Empty yfinance history.")

    frame = _normalize_history_columns(history)
    tz = ZoneInfo(timezone_name)
    candles_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, row in frame.iterrows():
        timestamp = _timestamp_from_row(index, row, tz=tz)
        if timestamp is None:
            continue

        candle = _candle_from_row(timestamp, row)
        if candle is None:
            continue

        candles_by_date[candle["date"]].append(candle)

    for candles in candles_by_date.values():
        candles.sort(key=lambda item: str(item["timestamp"]))

    if not candles_by_date:
        raise DayTradingGameDataError("No usable OHLC rows were found.")
    return dict(candles_by_date)


def _normalize_history_columns(history: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[Any, str] = {}
    for column in history.columns:
        normalized = _normalize_column_name(column)
        if normalized:
            rename_map[column] = normalized
    return history.rename(columns=rename_map)


def _normalize_column_name(column: Any) -> str | None:
    known = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume",
        "median": "median",
        "q1": "q1",
        "quartile1": "q1",
        "q3": "q3",
        "quartile3": "q3",
        "datetime": "datetime",
        "date": "datetime",
    }
    parts = column if isinstance(column, tuple) else (column,)
    for part in parts:
        key = str(part or "").strip().lower().replace(" ", "_").replace("-", "_")
        compact = key.replace("_", "")
        if key in known:
            return known[key]
        if compact in known:
            return known[compact]
    return None


def _timestamp_from_row(index: Any, row: pd.Series, *, tz: ZoneInfo) -> pd.Timestamp | None:
    raw = index
    if raw is None or (isinstance(raw, (int, float)) and "datetime" in row):
        raw = row.get("datetime")
    try:
        timestamp = pd.Timestamp(raw)
    except (TypeError, ValueError):
        raw = row.get("datetime")
        try:
            timestamp = pd.Timestamp(raw)
        except (TypeError, ValueError):
            return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(tz)
    return timestamp.tz_convert(tz)


def _candle_from_row(timestamp: pd.Timestamp, row: pd.Series) -> dict[str, Any] | None:
    open_price = _positive_float(row.get("open"))
    high_price = _positive_float(row.get("high"))
    low_price = _positive_float(row.get("low"))
    close_price = _positive_float(row.get("close"))
    execution_price, execution_method = _execution_price(row)
    if execution_price is None:
        return None

    open_price = open_price if open_price is not None else execution_price
    close_price = close_price if close_price is not None else execution_price
    high_candidates = [value for value in (high_price, open_price, close_price, execution_price) if value is not None]
    low_candidates = [value for value in (low_price, open_price, close_price, execution_price) if value is not None]
    if not high_candidates or not low_candidates:
        return None

    high_price = max(high_candidates)
    low_price = min(low_candidates)
    if high_price <= 0 or low_price <= 0:
        return None

    volume = finite_float_or_none(row.get("volume"), minimum=0.0)
    timestamp_iso = timestamp.isoformat()
    return {
        "timestamp": timestamp_iso,
        "date": timestamp.date().isoformat(),
        "time": timestamp.strftime("%H:%M"),
        "open": round(open_price, 6),
        "high": round(high_price, 6),
        "low": round(low_price, 6),
        "close": round(close_price, 6),
        "volume": round(volume, 3) if volume is not None else None,
        "execution_price": round(execution_price, 6),
        "execution_price_method": execution_method,
    }


def _execution_price(row: pd.Series) -> tuple[float | None, str]:
    close_price = _positive_float(row.get("close"))
    if close_price is not None:
        return close_price, "close"

    median_value = _positive_float(row.get("median"))
    if median_value is not None:
        return median_value, "median"

    q1 = _positive_float(row.get("q1"))
    q3 = _positive_float(row.get("q3"))
    if q1 is not None and q3 is not None:
        return (q1 + q3) / 2.0, "iqr_midpoint"

    ohlc_values = [
        value
        for value in (
            _positive_float(row.get("open")),
            _positive_float(row.get("high")),
            _positive_float(row.get("low")),
        )
        if value is not None
    ]
    if ohlc_values:
        midpoint = _median(ohlc_values)
        if midpoint is not None:
            return midpoint, "ohlc_median"

    return None, "unavailable"


def _positive_float(value: Any) -> float | None:
    return finite_float_or_none(value, minimum=0.0, strict_minimum=True)


def _median(values: Iterable[float]) -> float | None:
    sorted_values = sorted(value for value in values if math.isfinite(value))
    if not sorted_values:
        return None
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _eligible_session_dates(
    candles_by_date: dict[str, list[dict[str, Any]]],
    *,
    timezone_name: str,
    min_candles: int,
) -> list[str]:
    today = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    complete = [
        date_key
        for date_key, candles in candles_by_date.items()
        if len(candles) >= min_candles and date_key < today
    ]
    if complete:
        return sorted(complete)
    return sorted(date_key for date_key, candles in candles_by_date.items() if len(candles) >= min_candles)


def _eligible_session_date_windows(
    candles_by_date: dict[str, list[dict[str, Any]]],
    *,
    timezone_name: str,
    min_candles: int,
    session_days: int,
) -> list[tuple[str, ...]]:
    if session_days <= 1:
        return [
            (date_key,)
            for date_key in _eligible_session_dates(
                candles_by_date,
                timezone_name=timezone_name,
                min_candles=min_candles,
            )
        ]

    today = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    eligible_dates = sorted(
        date_key
        for date_key, candles in candles_by_date.items()
        if len(candles) >= min_candles
    )
    complete_dates = [date_key for date_key in eligible_dates if date_key < today]
    complete_windows = _session_date_windows(complete_dates, session_days=session_days)
    if complete_windows:
        return complete_windows
    return _session_date_windows(eligible_dates, session_days=session_days)


def _session_date_windows(date_keys: list[str], *, session_days: int) -> list[tuple[str, ...]]:
    if len(date_keys) < session_days:
        return []
    return [
        tuple(date_keys[start:start + session_days])
        for start in range(0, len(date_keys) - session_days + 1)
    ]


def _build_session_payload(
    *,
    market_key: str,
    config: dict[str, Any],
    symbol: str,
    symbol_name: str | None,
    date_keys: tuple[str, ...],
    candles: list[dict[str, Any]],
) -> dict[str, Any]:
    first = candles[0]
    last = candles[-1]
    start_date = date_keys[0]
    end_date = date_keys[-1]
    date_range_label = start_date if start_date == end_date else f"{start_date} to {end_date}"
    return {
        "game_id": uuid.uuid4().hex,
        "market": market_key,
        "market_label": config["label"],
        "symbol": symbol,
        "symbol_name": symbol_name,
        "symbol_label": _symbol_label(symbol, symbol_name),
        "date": start_date,
        "start_date": start_date,
        "end_date": end_date,
        "date_range": date_range_label,
        "session_dates": list(date_keys),
        "session_day_count": len(date_keys),
        "timezone": config["timezone"],
        "currency": config["currency"],
        "currency_symbol": config["currency_symbol"],
        "currency_digits": config["currency_digits"],
        "interval": YFINANCE_INTERVAL,
        "period": YFINANCE_PERIOD,
        "source": "yfinance",
        "execution_price_rule": "close, then median, then IQR midpoint, then OHLC median",
        "candle_count": len(candles),
        "session_start": first["time"],
        "session_end": last["time"],
        "candles": candles,
    }
