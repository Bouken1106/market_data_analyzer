"""Day-trading game data assembly backed by yfinance history."""

from __future__ import annotations

import asyncio
import inspect
import math
import random
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from ..utils import finite_float_or_none, normalize_symbol

DEFAULT_GAME_MODE = "intraday"
MAX_PRICE_DISPLAY_DIGITS = 4
SCORE_EPSILON = 1e-12

GAME_MODE_PROFILES: dict[str, dict[str, Any]] = {
    "intraday": {
        "label": "15m",
        "interval": "15m",
        "period": "60d",
        "session_days": 3,
        "min_candles": None,
        "moving_averages": (
            {"key": "short", "label": "MA5", "window": 5},
            {"key": "mid", "label": "MA20", "window": 20},
        ),
        "step_label": "Next 15m",
        "chart_label": "15-minute OHLC chart",
        "data_error_label": "15-minute trading windows",
    },
    "daily": {
        "label": "Daily",
        "interval": "1d",
        "period": "2y",
        "session_days": 30,
        "min_candles": 1,
        "moving_averages": (
            {"key": "short", "label": "MA5", "window": 5},
            {"key": "mid", "label": "MA25", "window": 25},
        ),
        "step_label": "Next Day",
        "chart_label": "Daily OHLC chart",
        "data_error_label": "daily trading windows",
    },
}

TRADE_MODE_PROFILES: dict[str, dict[str, str]] = {
    "long_only": {
        "label": "Long Only",
        "score_label": "S_L",
    },
    "long_short": {
        "label": "Long/Short",
        "score_label": "S_LS",
    },
}

YFINANCE_INTERVAL = GAME_MODE_PROFILES[DEFAULT_GAME_MODE]["interval"]
YFINANCE_PERIOD = GAME_MODE_PROFILES[DEFAULT_GAME_MODE]["period"]
GAME_SESSION_DAYS = GAME_MODE_PROFILES[DEFAULT_GAME_MODE]["session_days"]

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
    "BRK-B",
    "LLY",
    "V",
    "MA",
    "COST",
    "NFLX",
    "WMT",
    "ORCL",
    "HD",
    "PG",
    "JNJ",
    "ABBV",
    "BAC",
    "KO",
    "MRK",
    "CRM",
    "CSCO",
    "CVX",
    "PEP",
    "ADBE",
    "TMO",
    "ACN",
    "MCD",
    "ABT",
    "LIN",
    "WFC",
    "DIS",
    "GE",
    "INTU",
    "IBM",
    "NOW",
    "QCOM",
    "TXN",
    "AMGN",
    "CAT",
    "VZ",
    "ISRG",
    "PFE",
    "PM",
    "SPGI",
    "RTX",
    "HON",
    "NEE",
    "UBER",
    "LOW",
    "GS",
    "AXP",
    "BKNG",
    "BLK",
    "T",
    "PANW",
    "AMAT",
    "PGR",
    "UNP",
    "SYK",
    "TJX",
    "GILD",
    "DE",
    "VRTX",
    "LRCX",
    "MU",
    "SCHW",
    "MDT",
    "ADI",
    "BA",
    "COP",
    "C",
    "CB",
    "ETN",
    "PLTR",
    "SHOP",
    "CRWD",
    "SNOW",
    "COIN",
    "PYPL",
    "NKE",
    "SBUX",
    "LULU",
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
    "8411.T",
    "8316.T",
    "8766.T",
    "7267.T",
    "6902.T",
    "4519.T",
    "4502.T",
    "4503.T",
    "4568.T",
    "7741.T",
    "6954.T",
    "6981.T",
    "6594.T",
    "6273.T",
    "6146.T",
    "7735.T",
    "3382.T",
    "9983.T",
    "9433.T",
    "9434.T",
    "8001.T",
    "8031.T",
    "8015.T",
    "6301.T",
    "7011.T",
    "7012.T",
    "7013.T",
    "9101.T",
    "9104.T",
    "9107.T",
    "8801.T",
    "8802.T",
    "1925.T",
    "1928.T",
    "7269.T",
    "7270.T",
    "9020.T",
    "9021.T",
    "9022.T",
    "2914.T",
    "4452.T",
    "4661.T",
    "4901.T",
    "6701.T",
    "6702.T",
    "6723.T",
    "6762.T",
    "7751.T",
    "7832.T",
    "9024.T",
    "9201.T",
    "9202.T",
    "9501.T",
    "9502.T",
    "9503.T",
    "4689.T",
    "4755.T",
    "2413.T",
    "3659.T",
    "3697.T",
    "4385.T",
    "5401.T",
    "5411.T",
    "5713.T",
    "5802.T",
    "6326.T",
    "6367.T",
    "6503.T",
    "6645.T",
    "6920.T",
    "9843.T",
    "9613.T",
    "9735.T",
)

US_GAME_SYMBOL_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet",
    "TSLA": "Tesla",
    "AMD": "Advanced Micro Devices",
    "AVGO": "Broadcom",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "UNH": "UnitedHealth Group",
    "BRK-B": "Berkshire Hathaway",
    "LLY": "Eli Lilly",
    "V": "Visa",
    "MA": "Mastercard",
    "COST": "Costco Wholesale",
    "NFLX": "Netflix",
    "WMT": "Walmart",
    "ORCL": "Oracle",
    "HD": "Home Depot",
    "PG": "Procter & Gamble",
    "JNJ": "Johnson & Johnson",
    "ABBV": "AbbVie",
    "BAC": "Bank of America",
    "KO": "Coca-Cola",
    "MRK": "Merck",
    "CRM": "Salesforce",
    "CSCO": "Cisco Systems",
    "CVX": "Chevron",
    "PEP": "PepsiCo",
    "ADBE": "Adobe",
    "TMO": "Thermo Fisher Scientific",
    "ACN": "Accenture",
    "MCD": "McDonald's",
    "ABT": "Abbott Laboratories",
    "LIN": "Linde",
    "WFC": "Wells Fargo",
    "DIS": "Walt Disney",
    "GE": "GE Aerospace",
    "INTU": "Intuit",
    "IBM": "IBM",
    "NOW": "ServiceNow",
    "QCOM": "Qualcomm",
    "TXN": "Texas Instruments",
    "AMGN": "Amgen",
    "CAT": "Caterpillar",
    "VZ": "Verizon",
    "ISRG": "Intuitive Surgical",
    "PFE": "Pfizer",
    "PM": "Philip Morris",
    "SPGI": "S&P Global",
    "RTX": "RTX",
    "HON": "Honeywell",
    "NEE": "NextEra Energy",
    "UBER": "Uber",
    "LOW": "Lowe's",
    "GS": "Goldman Sachs",
    "AXP": "American Express",
    "BKNG": "Booking Holdings",
    "BLK": "BlackRock",
    "T": "AT&T",
    "PANW": "Palo Alto Networks",
    "AMAT": "Applied Materials",
    "PGR": "Progressive",
    "UNP": "Union Pacific",
    "SYK": "Stryker",
    "GILD": "Gilead Sciences",
    "DE": "Deere",
    "VRTX": "Vertex Pharmaceuticals",
    "LRCX": "Lam Research",
    "MU": "Micron Technology",
    "SCHW": "Charles Schwab",
    "ADI": "Analog Devices",
    "BA": "Boeing",
    "COP": "ConocoPhillips",
    "C": "Citigroup",
    "CB": "Chubb",
    "ETN": "Eaton",
    "TJX": "TJX Companies",
    "MDT": "Medtronic",
    "PLTR": "Palantir",
    "SHOP": "Shopify",
    "CRWD": "CrowdStrike",
    "SNOW": "Snowflake",
    "COIN": "Coinbase",
    "PYPL": "PayPal",
    "NKE": "Nike",
    "SBUX": "Starbucks",
    "LULU": "Lululemon Athletica",
}

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
    "8411.T": "Mizuho Financial Group",
    "8316.T": "Sumitomo Mitsui Financial Group",
    "8766.T": "Tokio Marine Holdings",
    "7267.T": "Honda Motor",
    "6902.T": "Denso",
    "4519.T": "Chugai Pharmaceutical",
    "4502.T": "Takeda Pharmaceutical",
    "4503.T": "Astellas Pharma",
    "4568.T": "Daiichi Sankyo",
    "7741.T": "HOYA",
    "6954.T": "Fanuc",
    "6981.T": "Murata Manufacturing",
    "6594.T": "Nidec",
    "6273.T": "SMC",
    "6146.T": "Disco",
    "7735.T": "Screen Holdings",
    "3382.T": "Seven & i Holdings",
    "9983.T": "Fast Retailing",
    "9433.T": "KDDI",
    "9434.T": "SoftBank Corp.",
    "8001.T": "Itochu",
    "8031.T": "Mitsui & Co.",
    "8015.T": "Toyota Tsusho",
    "6301.T": "Komatsu",
    "7011.T": "Mitsubishi Heavy Industries",
    "7012.T": "Kawasaki Heavy Industries",
    "7013.T": "IHI",
    "9101.T": "Nippon Yusen",
    "9104.T": "Mitsui O.S.K. Lines",
    "9107.T": "Kawasaki Kisen Kaisha",
    "8801.T": "Mitsui Fudosan",
    "8802.T": "Mitsubishi Estate",
    "1925.T": "Daiwa House Industry",
    "1928.T": "Sekisui House",
    "7269.T": "Suzuki Motor",
    "7270.T": "Subaru",
    "9020.T": "East Japan Railway",
    "9021.T": "West Japan Railway",
    "9022.T": "Central Japan Railway",
    "2914.T": "Japan Tobacco",
    "4452.T": "Kao",
    "4661.T": "Oriental Land",
    "4901.T": "Fujifilm Holdings",
    "6701.T": "NEC",
    "6702.T": "Fujitsu",
    "6723.T": "Renesas Electronics",
    "6762.T": "TDK",
    "7751.T": "Canon",
    "7832.T": "Bandai Namco Holdings",
    "9024.T": "Seibu Holdings",
    "9201.T": "Japan Airlines",
    "9202.T": "ANA Holdings",
    "9501.T": "Tokyo Electric Power",
    "9502.T": "Chubu Electric Power",
    "9503.T": "Kansai Electric Power",
    "4689.T": "LY Corp.",
    "4755.T": "Rakuten Group",
    "2413.T": "M3",
    "3659.T": "Nexon",
    "3697.T": "SHIFT",
    "4385.T": "Mercari",
    "5401.T": "Nippon Steel",
    "5411.T": "JFE Holdings",
    "5713.T": "Sumitomo Metal Mining",
    "5802.T": "Sumitomo Electric Industries",
    "6326.T": "Kubota",
    "6367.T": "Daikin Industries",
    "6503.T": "Mitsubishi Electric",
    "6645.T": "Omron",
    "6920.T": "Lasertec",
    "9843.T": "Nitori Holdings",
    "9613.T": "NTT Data Group",
    "9735.T": "Secom",
}

MARKET_CONFIGS: dict[str, dict[str, Any]] = {
    "us": {
        "label": "US",
        "timezone": "America/New_York",
        "currency": "USD",
        "currency_symbol": "$",
        "currency_digits": 2,
        "symbols": US_GAME_SYMBOLS,
        "symbol_names": US_GAME_SYMBOL_NAMES,
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

HistoryFetcher = Callable[..., Any]


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
    mode: str = DEFAULT_GAME_MODE,
    symbol: str | None = None,
    rng: random.Random | None = None,
    fetch_history: HistoryFetcher | None = None,
) -> dict[str, Any]:
    """Build one randomly selected replay session from yfinance bars."""

    market_key = str(market or "us").strip().lower()
    config = MARKET_CONFIGS.get(market_key)
    if config is None:
        supported = ", ".join(sorted(MARKET_CONFIGS))
        raise DayTradingGameRequestError(f"Unsupported market. Choose one of: {supported}.")
    mode_key, profile = _mode_profile(mode)

    selected_symbols = _candidate_symbols(config, symbol=symbol)
    chooser = rng or random.SystemRandom()
    chooser.shuffle(selected_symbols)
    fetcher = fetch_history or fetch_yfinance_history

    failures: list[str] = []
    for candidate in selected_symbols:
        try:
            history = await _call_history_fetcher(fetcher, candidate, profile=profile)
            candles_by_date = _candles_by_session_date(history, timezone_name=str(config["timezone"]))
            _apply_moving_averages(candles_by_date, moving_averages=profile["moving_averages"])
            min_candles = profile["min_candles"]
            if min_candles is None:
                min_candles = int(config["min_candles"])
            eligible = _eligible_session_date_windows(
                candles_by_date,
                timezone_name=str(config["timezone"]),
                min_candles=int(min_candles),
                session_days=int(profile["session_days"]),
            )
            if not eligible:
                raise DayTradingGameDataError(
                    f"No complete {profile['session_days']}-day {profile['data_error_label']} were available."
                )
            eligible = _prefer_windows_with_initial_moving_averages(
                eligible,
                candles_by_date,
                moving_averages=profile["moving_averages"],
            )
            date_keys = chooser.choice(eligible)
            candles = [
                candle
                for date_key in date_keys
                for candle in candles_by_date[date_key]
            ]
            return _build_session_payload(
                market_key=market_key,
                mode_key=mode_key,
                profile=profile,
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

    detail = f"No usable yfinance {profile['label']} data was available."
    if failures:
        detail = f"{detail} Last errors: {'; '.join(failures[-3:])}"
    raise DayTradingGameDataError(detail)


async def fetch_yfinance_history(
    symbol: str,
    *,
    interval: str = YFINANCE_INTERVAL,
    period: str = YFINANCE_PERIOD,
) -> pd.DataFrame:
    """Fetch bars for a symbol via yfinance without blocking the event loop."""

    return await asyncio.to_thread(_fetch_yfinance_history_sync, symbol, interval=interval, period=period)


def _fetch_yfinance_history_sync(symbol: str, *, interval: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in integration environments
        raise DayTradingGameDependencyError("yfinance is not installed. Run `pip install -r requirements.txt`.") from exc

    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            actions=False,
            timeout=20,
        )
    except TypeError:
        ticker = yf.Ticker(symbol)
        return ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            actions=False,
        )


def _mode_profile(mode: str) -> tuple[str, dict[str, Any]]:
    normalized = str(mode or DEFAULT_GAME_MODE).strip().lower().replace("-", "_")
    aliases = {
        "15m": "intraday",
        "intraday": "intraday",
        "day": "daily",
        "daily": "daily",
        "1d": "daily",
    }
    mode_key = aliases.get(normalized)
    if mode_key is None:
        supported = ", ".join(sorted(GAME_MODE_PROFILES))
        raise DayTradingGameRequestError(f"Unsupported mode. Choose one of: {supported}.")
    return mode_key, GAME_MODE_PROFILES[mode_key]


async def _call_history_fetcher(fetcher: HistoryFetcher, symbol: str, *, profile: dict[str, Any]) -> Any:
    if fetcher is fetch_yfinance_history:
        return await fetch_yfinance_history(
            symbol,
            interval=str(profile["interval"]),
            period=str(profile["period"]),
        )

    if _accepts_fetcher_options(fetcher):
        return await _maybe_await(
            fetcher(
                symbol,
                interval=str(profile["interval"]),
                period=str(profile["period"]),
            )
        )
    return await _maybe_await(fetcher(symbol))


def _accepts_fetcher_options(fetcher: HistoryFetcher) -> bool:
    try:
        signature = inspect.signature(fetcher)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters.values()
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters) or all(
        key in signature.parameters for key in ("interval", "period")
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


def calculate_day_trading_scoring(candles: Iterable[dict[str, Any]]) -> dict[str, Any]:
    closes = [_positive_float(candle.get("close")) for candle in candles]
    if not closes or any(close is None for close in closes):
        return _empty_scoring_metadata()

    prices = [float(close) for close in closes if close is not None]
    base_price = prices[0]
    deltas = [prices[index + 1] - prices[index] for index in range(len(prices) - 1)]

    buy_hold_return = (prices[-1] - prices[0]) / base_price
    long_lower_return = min(0.0, buy_hold_return)
    long_max_return = sum(max(delta, 0.0) for delta in deltas) / base_price
    long_denominator = long_max_return - long_lower_return
    long_short_max_return = _long_short_max_profit(prices) / base_price

    return {
        "base_price": _clean_score_float(base_price),
        "buy_hold_return": _clean_score_float(buy_hold_return),
        "long_only": {
            "lower_return": _clean_score_float(long_lower_return),
            "max_return": _clean_score_float(long_max_return),
            "denominator": _clean_score_float(long_denominator),
            "undefined": math.isclose(long_denominator, 0.0, abs_tol=SCORE_EPSILON),
        },
        "long_short": {
            "max_return": _clean_score_float(long_short_max_return),
            "undefined": math.isclose(long_short_max_return, 0.0, abs_tol=SCORE_EPSILON),
        },
    }


def _empty_scoring_metadata() -> dict[str, Any]:
    return {
        "base_price": None,
        "buy_hold_return": None,
        "long_only": {
            "lower_return": None,
            "max_return": None,
            "denominator": None,
            "undefined": True,
        },
        "long_short": {
            "max_return": None,
            "undefined": True,
        },
    }


def _long_short_max_profit(prices: list[float]) -> float:
    if not prices:
        return 0.0

    flat = 0.0
    long = -prices[0]
    short = prices[0]
    for price in prices[1:]:
        next_flat = max(flat, long + price, short - price)
        next_long = max(long, flat - price)
        next_short = max(short, flat + price)
        flat, long, short = next_flat, next_long, next_short
    return 0.0 if math.isclose(flat, 0.0, abs_tol=SCORE_EPSILON) else flat


def _clean_score_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if math.isclose(value, 0.0, abs_tol=SCORE_EPSILON):
        return 0.0
    return float(value)


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


def _apply_moving_averages(
    candles_by_date: dict[str, list[dict[str, Any]]],
    *,
    moving_averages: Iterable[dict[str, Any]],
) -> None:
    definitions = [
        (str(config["key"]), int(config["window"]))
        for config in moving_averages
        if int(config.get("window", 0)) > 0
    ]
    if not definitions:
        return

    ordered = [
        candle
        for date_key in sorted(candles_by_date)
        for candle in candles_by_date[date_key]
    ]
    sums = {key: 0.0 for key, _window in definitions}
    queues = {key: [] for key, _window in definitions}
    for candle in ordered:
        close = finite_float_or_none(candle.get("close"), minimum=0.0, strict_minimum=True)
        values: dict[str, float | None] = {}
        if close is None:
            for key, _window in definitions:
                sums[key] = 0.0
                queues[key].clear()
                values[key] = None
            candle["moving_averages"] = values
            continue

        for key, window in definitions:
            queue = queues[key]
            queue.append(close)
            sums[key] += close
            if len(queue) > window:
                sums[key] -= queue.pop(0)
            values[key] = round(sums[key] / window, 6) if len(queue) == window else None
        candle["moving_averages"] = values


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


def _prefer_windows_with_initial_moving_averages(
    windows: list[tuple[str, ...]],
    candles_by_date: dict[str, list[dict[str, Any]]],
    *,
    moving_averages: Iterable[dict[str, Any]],
) -> list[tuple[str, ...]]:
    required_keys = [str(config["key"]) for config in moving_averages if config.get("key")]
    if not required_keys:
        return windows

    preferred = [
        window
        for window in windows
        if _window_starts_with_moving_averages(window, candles_by_date, required_keys=required_keys)
    ]
    return preferred or windows


def _window_starts_with_moving_averages(
    window: tuple[str, ...],
    candles_by_date: dict[str, list[dict[str, Any]]],
    *,
    required_keys: list[str],
) -> bool:
    if not window:
        return False
    first_date_candles = candles_by_date.get(window[0]) or []
    if not first_date_candles:
        return False
    values = first_date_candles[0].get("moving_averages")
    if not isinstance(values, dict):
        return False
    return all(finite_float_or_none(values.get(key)) is not None for key in required_keys)


def _build_session_payload(
    *,
    market_key: str,
    mode_key: str,
    profile: dict[str, Any],
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
    currency_digits = int(config["currency_digits"])
    return {
        "game_id": uuid.uuid4().hex,
        "market": market_key,
        "market_label": config["label"],
        "mode": mode_key,
        "mode_label": profile["label"],
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
        "currency_digits": currency_digits,
        "price_digits": _price_display_digits(candles, fallback_digits=currency_digits),
        "interval": profile["interval"],
        "period": profile["period"],
        "source": "yfinance",
        "execution_price_rule": "close, then median, then IQR midpoint, then OHLC median",
        "step_label": profile["step_label"],
        "chart_label": profile["chart_label"],
        "moving_averages": list(profile["moving_averages"]),
        "trade_modes": [
            {"key": key, **value}
            for key, value in TRADE_MODE_PROFILES.items()
        ],
        "scoring": calculate_day_trading_scoring(candles),
        "candle_count": len(candles),
        "session_start": first["date"] if mode_key == "daily" else first["time"],
        "session_end": last["date"] if mode_key == "daily" else last["time"],
        "candles": candles,
    }


def _price_display_digits(candles: Iterable[dict[str, Any]], *, fallback_digits: int) -> int:
    fallback = max(0, min(MAX_PRICE_DISPLAY_DIGITS, int(fallback_digits)))
    observed = fallback
    for candle in candles:
        for key in ("open", "high", "low", "close", "execution_price"):
            observed = max(observed, _decimal_places(candle.get(key)))
            if observed >= MAX_PRICE_DISPLAY_DIGITS:
                return MAX_PRICE_DISPLAY_DIGITS
    return observed


def _decimal_places(value: Any) -> int:
    numeric = finite_float_or_none(value)
    if numeric is None:
        return 0
    rounded = round(numeric, 6)
    if math.isclose(rounded, round(rounded), abs_tol=1e-9):
        return 0
    text = f"{rounded:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return min(MAX_PRICE_DISPLAY_DIGITS, len(text.rsplit(".", 1)[1]))
