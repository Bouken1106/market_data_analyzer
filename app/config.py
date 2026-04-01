"""Application-wide configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from pathlib import Path
from typing import Pattern

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _float_env(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("market-data-analyzer")


@dataclass(frozen=True)
class AppSettings:
    supported_data_providers: frozenset[str]
    data_provider: str
    twelve_data_api_key: str
    fmp_api_key: str
    jquants_api_key: str
    jquants_min_request_interval_sec: float
    jquants_rate_limit_backoff_sec: float
    api_key: str
    max_basic_symbols: int
    symbol_pattern: Pattern[str]
    ws_url_template: str
    rest_price_url: str
    quote_url: str
    api_usage_url: str
    stocks_list_url: str
    time_series_url: str
    earliest_timestamp_url: str
    fmp_quote_url: str
    fmp_stock_list_url: str
    fmp_stock_list_legacy_url: str
    fmp_historical_eod_url: str
    jquants_daily_bars_url: str
    api_limit_per_min: int
    api_limit_per_day: int
    daily_budget_utilization: float
    per_min_limit_utilization: float
    rest_min_poll_interval_sec: int
    market_closed_sleep_sec: int
    symbol_catalog_country: str
    symbol_catalog_ttl_sec: int
    symbol_catalog_max_items: int
    app_dir: Path
    symbol_catalog_cache_path: Path
    last_price_cache_path: Path
    full_daily_history_cache_dir: Path
    fmp_reference_cache_dir: Path
    paper_portfolio_cache_path: Path
    ui_state_cache_path: Path
    paper_initial_cash: float
    auto_refresh_on_startup: bool
    historical_default_years: int
    historical_max_years: int
    historical_cache_ttl_sec: int
    historical_interval: str
    historical_max_points: int
    time_series_max_outputsize: int
    full_history_chunk_years: int
    full_history_max_chunks: int
    daily_diff_min_recheck_sec: int
    beta_market_recheck_sec: int
    fmp_reference_cache_ttl_sec: int
    fmp_profile_url: str
    fmp_key_metrics_ttm_url: str
    fmp_ratios_ttm_url: str
    fmp_income_statement_url: str
    fmp_balance_sheet_url: str
    fmp_cash_flow_url: str
    fmp_dividend_adjusted_price_url: str
    fmp_dividends_url: str
    fmp_splits_url: str
    overview_cache_ttl_sec: int
    sparkline_cache_ttl_sec: int
    sparkline_points: int
    lmstudio_base_url: str
    lmstudio_chat_completions_url: str
    lmstudio_model: str
    lmstudio_api_key: str
    lmstudio_timeout_sec: float
    ml_history_min_months: int
    ml_history_max_months: int
    symbol_country_map_raw: str
    default_symbols_raw: str


def _resolve_data_provider(supported_data_providers: frozenset[str]) -> str:
    configured = os.getenv("MARKET_DATA_PROVIDER", "twelvedata").strip().lower()
    if configured in supported_data_providers:
        return configured
    LOGGER.warning(
        "Unsupported MARKET_DATA_PROVIDER=%s. Falling back to twelvedata.",
        configured,
    )
    return "twelvedata"


def _load_settings() -> AppSettings:
    supported_data_providers = frozenset({"twelvedata", "fmp", "both"})
    data_provider = _resolve_data_provider(supported_data_providers)
    twelve_data_api_key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    fmp_api_key = os.getenv("FMP_API_KEY", "").strip()
    jquants_api_key = os.getenv("JQUANTS_API_KEY", "").strip()
    symbol_pattern = re.compile(r"^[A-Z0-9.\-]{1,15}$")

    if data_provider == "twelvedata":
        api_key = twelve_data_api_key
    elif data_provider == "fmp":
        api_key = fmp_api_key
    else:
        api_key = twelve_data_api_key or fmp_api_key

    app_dir = Path(__file__).resolve().parent
    lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").strip().rstrip("/")
    return AppSettings(
        supported_data_providers=supported_data_providers,
        data_provider=data_provider,
        twelve_data_api_key=twelve_data_api_key,
        fmp_api_key=fmp_api_key,
        jquants_api_key=jquants_api_key,
        jquants_min_request_interval_sec=_float_env("JQUANTS_MIN_REQUEST_INTERVAL_SEC", default=12.0, minimum=0.0),
        jquants_rate_limit_backoff_sec=_float_env("JQUANTS_RATE_LIMIT_BACKOFF_SEC", default=30.0, minimum=1.0),
        api_key=api_key,
        max_basic_symbols=8,
        symbol_pattern=symbol_pattern,
        ws_url_template="wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}",
        rest_price_url="https://api.twelvedata.com/price",
        quote_url="https://api.twelvedata.com/quote",
        api_usage_url="https://api.twelvedata.com/api_usage",
        stocks_list_url="https://api.twelvedata.com/stocks",
        time_series_url="https://api.twelvedata.com/time_series",
        earliest_timestamp_url="https://api.twelvedata.com/earliest_timestamp",
        fmp_quote_url="https://financialmodelingprep.com/stable/quote",
        fmp_stock_list_url="https://financialmodelingprep.com/stable/stock-list",
        fmp_stock_list_legacy_url="https://financialmodelingprep.com/api/v3/stock/list",
        fmp_historical_eod_url="https://financialmodelingprep.com/stable/historical-price-eod/full",
        jquants_daily_bars_url="https://api.jquants.com/v2/equities/bars/daily",
        api_limit_per_min=_int_env("API_LIMIT_PER_MIN", default=8, minimum=1),
        api_limit_per_day=_int_env("API_LIMIT_PER_DAY", default=800, minimum=1),
        daily_budget_utilization=_float_env("DAILY_BUDGET_UTILIZATION", default=0.75, minimum=0.1),
        per_min_limit_utilization=_float_env("PER_MIN_LIMIT_UTILIZATION", default=0.9, minimum=0.1),
        rest_min_poll_interval_sec=_int_env("REST_MIN_POLL_INTERVAL_SEC", default=30, minimum=10),
        market_closed_sleep_sec=_int_env("MARKET_CLOSED_SLEEP_SEC", default=60, minimum=10),
        symbol_catalog_country=os.getenv("SYMBOL_CATALOG_COUNTRY", "United States").strip() or "United States",
        symbol_catalog_ttl_sec=_int_env("SYMBOL_CATALOG_TTL_SEC", default=86400, minimum=60),
        symbol_catalog_max_items=_int_env("SYMBOL_CATALOG_MAX_ITEMS", default=25000, minimum=1000),
        app_dir=app_dir,
        symbol_catalog_cache_path=app_dir / "cache" / "us_stock_symbol_catalog.json",
        last_price_cache_path=app_dir / "cache" / "last_prices.json",
        full_daily_history_cache_dir=app_dir / "cache" / "daily_history",
        fmp_reference_cache_dir=app_dir / "cache" / "fmp_reference",
        paper_portfolio_cache_path=app_dir / "cache" / "paper_portfolio.json",
        ui_state_cache_path=app_dir / "cache" / "ui_state.json",
        paper_initial_cash=_float_env("PAPER_INITIAL_CASH", default=1_000_000, minimum=1),
        auto_refresh_on_startup=_bool_env("AUTO_REFRESH_ON_STARTUP", default=False),
        historical_default_years=_int_env("HISTORICAL_DEFAULT_YEARS", default=5, minimum=1),
        historical_max_years=_int_env("HISTORICAL_MAX_YEARS", default=10, minimum=1),
        historical_cache_ttl_sec=_int_env("HISTORICAL_CACHE_TTL_SEC", default=43200, minimum=60),
        historical_interval=os.getenv("HISTORICAL_INTERVAL", "1day").strip() or "1day",
        historical_max_points=_int_env("HISTORICAL_MAX_POINTS", default=2000, minimum=100),
        time_series_max_outputsize=_int_env("TIME_SERIES_MAX_OUTPUTSIZE", default=5000, minimum=100),
        full_history_chunk_years=_int_env("FULL_HISTORY_CHUNK_YEARS", default=15, minimum=1),
        full_history_max_chunks=_int_env("FULL_HISTORY_MAX_CHUNKS", default=20, minimum=1),
        daily_diff_min_recheck_sec=_int_env("DAILY_DIFF_MIN_RECHECK_SEC", default=21600, minimum=60),
        beta_market_recheck_sec=_int_env("BETA_MARKET_RECHECK_SEC", default=86400, minimum=300),
        fmp_reference_cache_ttl_sec=_int_env("FMP_REFERENCE_CACHE_TTL_SEC", default=43200, minimum=300),
        fmp_profile_url="https://financialmodelingprep.com/stable/profile",
        fmp_key_metrics_ttm_url="https://financialmodelingprep.com/stable/key-metrics-ttm",
        fmp_ratios_ttm_url="https://financialmodelingprep.com/stable/ratios-ttm",
        fmp_income_statement_url="https://financialmodelingprep.com/stable/income-statement",
        fmp_balance_sheet_url="https://financialmodelingprep.com/stable/balance-sheet-statement",
        fmp_cash_flow_url="https://financialmodelingprep.com/stable/cash-flow-statement",
        fmp_dividend_adjusted_price_url="https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted",
        fmp_dividends_url="https://financialmodelingprep.com/stable/dividends",
        fmp_splits_url="https://financialmodelingprep.com/stable/splits",
        overview_cache_ttl_sec=_int_env("OVERVIEW_CACHE_TTL_SEC", default=120, minimum=10),
        sparkline_cache_ttl_sec=_int_env("SPARKLINE_CACHE_TTL_SEC", default=21600, minimum=300),
        sparkline_points=_int_env("SPARKLINE_POINTS", default=30, minimum=10),
        lmstudio_base_url=lmstudio_base_url,
        lmstudio_chat_completions_url=os.getenv(
            "LMSTUDIO_CHAT_COMPLETIONS_URL",
            f"{lmstudio_base_url}/chat/completions",
        ).strip(),
        lmstudio_model=os.getenv("LMSTUDIO_MODEL", "ministral-3-3b").strip() or "ministral-3-3b",
        lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY", "").strip(),
        lmstudio_timeout_sec=_float_env("LMSTUDIO_TIMEOUT_SEC", default=25.0, minimum=3.0),
        ml_history_min_months=3,
        ml_history_max_months=60,
        symbol_country_map_raw=os.getenv("SYMBOL_COUNTRY_MAP", ""),
        default_symbols_raw=os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA"),
    )


settings = _load_settings()

# ---------------------------------------------------------------------------
# Data provider / API key
# ---------------------------------------------------------------------------

SUPPORTED_DATA_PROVIDERS = settings.supported_data_providers
DATA_PROVIDER = settings.data_provider
TWELVE_DATA_API_KEY = settings.twelve_data_api_key
FMP_API_KEY = settings.fmp_api_key
JQUANTS_API_KEY = settings.jquants_api_key
JQUANTS_MIN_REQUEST_INTERVAL_SEC = settings.jquants_min_request_interval_sec
JQUANTS_RATE_LIMIT_BACKOFF_SEC = settings.jquants_rate_limit_backoff_sec
API_KEY = settings.api_key

# ---------------------------------------------------------------------------
# Symbol constraints
# ---------------------------------------------------------------------------

MAX_BASIC_SYMBOLS = settings.max_basic_symbols
SYMBOL_PATTERN = settings.symbol_pattern

# ---------------------------------------------------------------------------
# Twelve Data API URLs
# ---------------------------------------------------------------------------

WS_URL_TEMPLATE = settings.ws_url_template
REST_PRICE_URL = settings.rest_price_url
QUOTE_URL = settings.quote_url
API_USAGE_URL = settings.api_usage_url
STOCKS_LIST_URL = settings.stocks_list_url
TIME_SERIES_URL = settings.time_series_url
EARLIEST_TIMESTAMP_URL = settings.earliest_timestamp_url

# ---------------------------------------------------------------------------
# Financial Modeling Prep API URLs
# ---------------------------------------------------------------------------

FMP_QUOTE_URL = settings.fmp_quote_url
FMP_STOCK_LIST_URL = settings.fmp_stock_list_url
FMP_STOCK_LIST_LEGACY_URL = settings.fmp_stock_list_legacy_url
FMP_HISTORICAL_EOD_URL = settings.fmp_historical_eod_url

# ---------------------------------------------------------------------------
# J-Quants API URLs
# ---------------------------------------------------------------------------

JQUANTS_DAILY_BARS_URL = settings.jquants_daily_bars_url

# ---------------------------------------------------------------------------
# Rate-limiting / budget
# ---------------------------------------------------------------------------

API_LIMIT_PER_MIN = settings.api_limit_per_min
API_LIMIT_PER_DAY = settings.api_limit_per_day
DAILY_BUDGET_UTILIZATION = settings.daily_budget_utilization
PER_MIN_LIMIT_UTILIZATION = settings.per_min_limit_utilization
REST_MIN_POLL_INTERVAL_SEC = settings.rest_min_poll_interval_sec
MARKET_CLOSED_SLEEP_SEC = settings.market_closed_sleep_sec

# ---------------------------------------------------------------------------
# Symbol catalog
# ---------------------------------------------------------------------------

SYMBOL_CATALOG_COUNTRY = settings.symbol_catalog_country
SYMBOL_CATALOG_TTL_SEC = settings.symbol_catalog_ttl_sec
SYMBOL_CATALOG_MAX_ITEMS = settings.symbol_catalog_max_items

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

_APP_DIR = settings.app_dir
SYMBOL_CATALOG_CACHE_PATH = settings.symbol_catalog_cache_path
LAST_PRICE_CACHE_PATH = settings.last_price_cache_path
FULL_DAILY_HISTORY_CACHE_DIR = settings.full_daily_history_cache_dir
FMP_REFERENCE_CACHE_DIR = settings.fmp_reference_cache_dir
PAPER_PORTFOLIO_CACHE_PATH = settings.paper_portfolio_cache_path
UI_STATE_CACHE_PATH = settings.ui_state_cache_path
PAPER_INITIAL_CASH = settings.paper_initial_cash
AUTO_REFRESH_ON_STARTUP = settings.auto_refresh_on_startup

# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

HISTORICAL_DEFAULT_YEARS = settings.historical_default_years
HISTORICAL_MAX_YEARS = settings.historical_max_years
HISTORICAL_CACHE_TTL_SEC = settings.historical_cache_ttl_sec
HISTORICAL_INTERVAL = settings.historical_interval
HISTORICAL_MAX_POINTS = settings.historical_max_points
TIME_SERIES_MAX_OUTPUTSIZE = settings.time_series_max_outputsize
FULL_HISTORY_CHUNK_YEARS = settings.full_history_chunk_years
FULL_HISTORY_MAX_CHUNKS = settings.full_history_max_chunks
DAILY_DIFF_MIN_RECHECK_SEC = settings.daily_diff_min_recheck_sec
BETA_MARKET_RECHECK_SEC = settings.beta_market_recheck_sec
FMP_REFERENCE_CACHE_TTL_SEC = settings.fmp_reference_cache_ttl_sec

# ---------------------------------------------------------------------------
# FMP fundamental/reference endpoints
# ---------------------------------------------------------------------------

FMP_PROFILE_URL = settings.fmp_profile_url
FMP_KEY_METRICS_TTM_URL = settings.fmp_key_metrics_ttm_url
FMP_RATIOS_TTM_URL = settings.fmp_ratios_ttm_url
FMP_INCOME_STATEMENT_URL = settings.fmp_income_statement_url
FMP_BALANCE_SHEET_URL = settings.fmp_balance_sheet_url
FMP_CASH_FLOW_URL = settings.fmp_cash_flow_url
FMP_DIVIDEND_ADJUSTED_PRICE_URL = settings.fmp_dividend_adjusted_price_url
FMP_DIVIDENDS_URL = settings.fmp_dividends_url
FMP_SPLITS_URL = settings.fmp_splits_url

# ---------------------------------------------------------------------------
# Overview / Sparkline
# ---------------------------------------------------------------------------

OVERVIEW_CACHE_TTL_SEC = settings.overview_cache_ttl_sec
SPARKLINE_CACHE_TTL_SEC = settings.sparkline_cache_ttl_sec
SPARKLINE_POINTS = settings.sparkline_points

# ---------------------------------------------------------------------------
# Local LLM (LM Studio)
# ---------------------------------------------------------------------------

LMSTUDIO_BASE_URL = settings.lmstudio_base_url
LMSTUDIO_CHAT_COMPLETIONS_URL = settings.lmstudio_chat_completions_url
LMSTUDIO_MODEL = settings.lmstudio_model
LMSTUDIO_API_KEY = settings.lmstudio_api_key
LMSTUDIO_TIMEOUT_SEC = settings.lmstudio_timeout_sec

ML_HISTORY_MIN_MONTHS = settings.ml_history_min_months
ML_HISTORY_MAX_MONTHS = settings.ml_history_max_months

# ---------------------------------------------------------------------------
# Default symbols
# ---------------------------------------------------------------------------

SYMBOL_COUNTRY_MAP_RAW = settings.symbol_country_map_raw
DEFAULT_SYMBOLS_RAW = settings.default_symbols_raw
