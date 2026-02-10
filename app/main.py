"""Market Data Analyzer – application entry point.

This file wires together all components and creates the FastAPI ``app``
instance used by Uvicorn.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import (
    LAST_PRICE_CACHE_PATH,
    FULL_DAILY_HISTORY_CACHE_DIR,
    MAX_BASIC_SYMBOLS,
    SYMBOL_CATALOG_CACHE_PATH,
    SYMBOL_CATALOG_TTL_SEC,
)
from .hub import MarketDataHub
from .ml.job_store import MlJobStore
from .ml.pipelines import set_hub, set_ml_job_store
from .routes import init_routes, router
from .stores import FullDailyHistoryStore, LastPriceStore, SymbolCatalogStore
from .utils import normalize_symbols

# ---------------------------------------------------------------------------
# Resolve environment variables
# ---------------------------------------------------------------------------

API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
DEFAULT_SYMBOLS = normalize_symbols(
    os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA")
)
if not DEFAULT_SYMBOLS:
    DEFAULT_SYMBOLS = ["AAPL"]
DEFAULT_SYMBOLS = DEFAULT_SYMBOLS[:MAX_BASIC_SYMBOLS]

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY is required. Set it in your environment or .env file.")

# ---------------------------------------------------------------------------
# Instantiate stores & hub
# ---------------------------------------------------------------------------

last_price_store = LastPriceStore(cache_path=LAST_PRICE_CACHE_PATH)
full_daily_history_store = FullDailyHistoryStore(cache_dir=FULL_DAILY_HISTORY_CACHE_DIR)
symbol_catalog_store = SymbolCatalogStore(
    api_key=API_KEY,
    cache_path=SYMBOL_CATALOG_CACHE_PATH,
    ttl_sec=SYMBOL_CATALOG_TTL_SEC,
)
ml_job_store = MlJobStore(max_jobs=120)

hub = MarketDataHub(
    api_key=API_KEY,
    symbols=DEFAULT_SYMBOLS,
    last_price_store=last_price_store,
    full_daily_history_store=full_daily_history_store,
)

# Inject singletons where needed
set_hub(hub)
set_ml_job_store(ml_job_store)
init_routes(hub=hub, symbol_catalog_store=symbol_catalog_store, ml_job_store=ml_job_store)


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    await hub.start()
    try:
        yield
    finally:
        await hub.stop()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Market Data Analyzer", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)
