"""Application-wide configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("market-data-analyzer")

# ---------------------------------------------------------------------------
# Data provider / API key
# ---------------------------------------------------------------------------

SUPPORTED_DATA_PROVIDERS = {"twelvedata", "fmp", "both"}
DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "twelvedata").strip().lower()
if DATA_PROVIDER not in SUPPORTED_DATA_PROVIDERS:
    LOGGER.warning(
        "Unsupported MARKET_DATA_PROVIDER=%s. Falling back to twelvedata.",
        DATA_PROVIDER,
    )
    DATA_PROVIDER = "twelvedata"

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
if DATA_PROVIDER == "twelvedata":
    API_KEY = TWELVE_DATA_API_KEY
elif DATA_PROVIDER == "fmp":
    API_KEY = FMP_API_KEY
else:
    API_KEY = TWELVE_DATA_API_KEY or FMP_API_KEY

# ---------------------------------------------------------------------------
# Symbol constraints
# ---------------------------------------------------------------------------

MAX_BASIC_SYMBOLS = 8
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,15}$")

# ---------------------------------------------------------------------------
# Twelve Data API URLs
# ---------------------------------------------------------------------------

WS_URL_TEMPLATE = "wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}"
REST_PRICE_URL = "https://api.twelvedata.com/price"
QUOTE_URL = "https://api.twelvedata.com/quote"
API_USAGE_URL = "https://api.twelvedata.com/api_usage"
STOCKS_LIST_URL = "https://api.twelvedata.com/stocks"
TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
EARLIEST_TIMESTAMP_URL = "https://api.twelvedata.com/earliest_timestamp"

# ---------------------------------------------------------------------------
# Financial Modeling Prep API URLs
# ---------------------------------------------------------------------------

FMP_QUOTE_URL = "https://financialmodelingprep.com/stable/quote"
FMP_STOCK_LIST_URL = "https://financialmodelingprep.com/stable/stock-list"
FMP_STOCK_LIST_LEGACY_URL = "https://financialmodelingprep.com/api/v3/stock/list"
FMP_HISTORICAL_EOD_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

# ---------------------------------------------------------------------------
# Rate-limiting / budget
# ---------------------------------------------------------------------------

API_LIMIT_PER_MIN = _int_env("API_LIMIT_PER_MIN", default=8, minimum=1)
API_LIMIT_PER_DAY = _int_env("API_LIMIT_PER_DAY", default=800, minimum=1)
DAILY_BUDGET_UTILIZATION = _float_env("DAILY_BUDGET_UTILIZATION", default=0.75, minimum=0.1)
PER_MIN_LIMIT_UTILIZATION = _float_env("PER_MIN_LIMIT_UTILIZATION", default=0.9, minimum=0.1)
REST_MIN_POLL_INTERVAL_SEC = _int_env("REST_MIN_POLL_INTERVAL_SEC", default=30, minimum=10)
MARKET_CLOSED_SLEEP_SEC = _int_env("MARKET_CLOSED_SLEEP_SEC", default=60, minimum=10)

# ---------------------------------------------------------------------------
# Symbol catalog
# ---------------------------------------------------------------------------

SYMBOL_CATALOG_COUNTRY = os.getenv("SYMBOL_CATALOG_COUNTRY", "United States").strip() or "United States"
SYMBOL_CATALOG_TTL_SEC = _int_env("SYMBOL_CATALOG_TTL_SEC", default=86400, minimum=60)
SYMBOL_CATALOG_MAX_ITEMS = _int_env("SYMBOL_CATALOG_MAX_ITEMS", default=25000, minimum=1000)

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parent
SYMBOL_CATALOG_CACHE_PATH = _APP_DIR / "cache" / "us_stock_symbol_catalog.json"
LAST_PRICE_CACHE_PATH = _APP_DIR / "cache" / "last_prices.json"
FULL_DAILY_HISTORY_CACHE_DIR = _APP_DIR / "cache" / "daily_history"
FMP_REFERENCE_CACHE_DIR = _APP_DIR / "cache" / "fmp_reference"

# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

HISTORICAL_DEFAULT_YEARS = _int_env("HISTORICAL_DEFAULT_YEARS", default=5, minimum=1)
HISTORICAL_MAX_YEARS = _int_env("HISTORICAL_MAX_YEARS", default=10, minimum=1)
HISTORICAL_CACHE_TTL_SEC = _int_env("HISTORICAL_CACHE_TTL_SEC", default=43200, minimum=60)
HISTORICAL_INTERVAL = os.getenv("HISTORICAL_INTERVAL", "1day").strip() or "1day"
HISTORICAL_MAX_POINTS = _int_env("HISTORICAL_MAX_POINTS", default=2000, minimum=100)
TIME_SERIES_MAX_OUTPUTSIZE = _int_env("TIME_SERIES_MAX_OUTPUTSIZE", default=5000, minimum=100)
FULL_HISTORY_CHUNK_YEARS = _int_env("FULL_HISTORY_CHUNK_YEARS", default=15, minimum=1)
FULL_HISTORY_MAX_CHUNKS = _int_env("FULL_HISTORY_MAX_CHUNKS", default=20, minimum=1)
DAILY_DIFF_MIN_RECHECK_SEC = _int_env("DAILY_DIFF_MIN_RECHECK_SEC", default=21600, minimum=60)
BETA_MARKET_RECHECK_SEC = _int_env("BETA_MARKET_RECHECK_SEC", default=86400, minimum=300)
FMP_REFERENCE_CACHE_TTL_SEC = _int_env("FMP_REFERENCE_CACHE_TTL_SEC", default=43200, minimum=300)

# ---------------------------------------------------------------------------
# FMP fundamental/reference endpoints
# ---------------------------------------------------------------------------

FMP_PROFILE_URL = "https://financialmodelingprep.com/stable/profile"
FMP_KEY_METRICS_TTM_URL = "https://financialmodelingprep.com/stable/key-metrics-ttm"
FMP_RATIOS_TTM_URL = "https://financialmodelingprep.com/stable/ratios-ttm"
FMP_INCOME_STATEMENT_URL = "https://financialmodelingprep.com/stable/income-statement"
FMP_BALANCE_SHEET_URL = "https://financialmodelingprep.com/stable/balance-sheet-statement"
FMP_CASH_FLOW_URL = "https://financialmodelingprep.com/stable/cash-flow-statement"
FMP_DIVIDEND_ADJUSTED_PRICE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
FMP_DIVIDENDS_URL = "https://financialmodelingprep.com/stable/dividends"
FMP_SPLITS_URL = "https://financialmodelingprep.com/stable/splits"

# ---------------------------------------------------------------------------
# Overview / Sparkline
# ---------------------------------------------------------------------------

OVERVIEW_CACHE_TTL_SEC = _int_env("OVERVIEW_CACHE_TTL_SEC", default=120, minimum=10)
SPARKLINE_CACHE_TTL_SEC = _int_env("SPARKLINE_CACHE_TTL_SEC", default=21600, minimum=300)
SPARKLINE_POINTS = _int_env("SPARKLINE_POINTS", default=30, minimum=10)

# ---------------------------------------------------------------------------
# ML defaults
# ---------------------------------------------------------------------------

ML_HISTORY_DEFAULT_MONTHS = 60
ML_HISTORY_MIN_MONTHS = 3
ML_HISTORY_MAX_MONTHS = 60
ML_EVAL_MONTHS = 2
ML_SPLIT_EVAL_DAYS = ML_EVAL_MONTHS * 31
ML_SPLIT_TRAIN_VAL_RATIO = 0.8

# ---------------------------------------------------------------------------
# Default symbols
# ---------------------------------------------------------------------------

SYMBOL_COUNTRY_MAP_RAW = os.getenv("SYMBOL_COUNTRY_MAP", "")
DEFAULT_SYMBOLS_RAW = os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA")
