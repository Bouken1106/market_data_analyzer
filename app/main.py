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
    DATA_PROVIDER,
    FMP_API_KEY,
    FMP_REFERENCE_CACHE_DIR,
    LAST_PRICE_CACHE_PATH,
    PAPER_INITIAL_CASH,
    PAPER_PORTFOLIO_CACHE_PATH,
    UI_STATE_CACHE_PATH,
    FULL_DAILY_HISTORY_CACHE_DIR,
    MAX_BASIC_SYMBOLS,
    SYMBOL_CATALOG_CACHE_PATH,
    SYMBOL_CATALOG_TTL_SEC,
    TWELVE_DATA_API_KEY,
)
from .hub import MarketDataHub
from .ml.job_store import MlJobStore
from .routes import init_routes, router
from .stores import (
    FmpReferenceStore,
    FullDailyHistoryStore,
    LastPriceStore,
    PaperPortfolioStore,
    SymbolCatalogStore,
    UiStateStore,
)
from .utils import normalize_symbols

# ---------------------------------------------------------------------------
# Resolve environment variables
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = normalize_symbols(
    os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA")
)
if not DEFAULT_SYMBOLS:
    DEFAULT_SYMBOLS = ["AAPL"]
DEFAULT_SYMBOLS = DEFAULT_SYMBOLS[:MAX_BASIC_SYMBOLS]

if DATA_PROVIDER == "twelvedata" and not TWELVE_DATA_API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY is required. Set it in your environment or .env file.")
if DATA_PROVIDER == "fmp" and not FMP_API_KEY:
    raise RuntimeError("FMP_API_KEY is required. Set it in your environment or .env file.")
if DATA_PROVIDER == "both":
    missing: list[str] = []
    if not TWELVE_DATA_API_KEY:
        missing.append("TWELVE_DATA_API_KEY")
    if not FMP_API_KEY:
        missing.append("FMP_API_KEY")
    if missing:
        raise RuntimeError(f"{', '.join(missing)} is required when MARKET_DATA_PROVIDER=both.")

# ---------------------------------------------------------------------------
# Instantiate stores & hub
# ---------------------------------------------------------------------------

last_price_store = LastPriceStore(cache_path=LAST_PRICE_CACHE_PATH)
full_daily_history_store = FullDailyHistoryStore(cache_dir=FULL_DAILY_HISTORY_CACHE_DIR)
fmp_reference_store = FmpReferenceStore(cache_dir=FMP_REFERENCE_CACHE_DIR)
paper_portfolio_store = PaperPortfolioStore(
    cache_path=PAPER_PORTFOLIO_CACHE_PATH,
    default_initial_cash=PAPER_INITIAL_CASH,
)
symbol_catalog_store = SymbolCatalogStore(
    provider=DATA_PROVIDER,
    twelvedata_api_key=TWELVE_DATA_API_KEY,
    fmp_api_key=FMP_API_KEY,
    cache_path=SYMBOL_CATALOG_CACHE_PATH,
    ttl_sec=SYMBOL_CATALOG_TTL_SEC,
)
ml_job_store = MlJobStore(max_jobs=120)
ui_state_store = UiStateStore(cache_path=UI_STATE_CACHE_PATH)
persisted_symbols = ui_state_store.get_symbols()
initial_symbols = persisted_symbols if persisted_symbols else DEFAULT_SYMBOLS
if not persisted_symbols:
    ui_state_store.set_symbols(initial_symbols)

hub = MarketDataHub(
    provider=DATA_PROVIDER,
    twelvedata_api_key=TWELVE_DATA_API_KEY,
    fmp_api_key=FMP_API_KEY,
    symbols=initial_symbols,
    last_price_store=last_price_store,
    full_daily_history_store=full_daily_history_store,
    fmp_reference_store=fmp_reference_store,
    ui_state_store=ui_state_store,
)

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

init_routes(
    app,
    hub=hub,
    symbol_catalog_store=symbol_catalog_store,
    ml_job_store=ml_job_store,
    paper_portfolio_store=paper_portfolio_store,
    ui_state_store=ui_state_store,
)


@app.middleware("http")
async def disable_monitor_asset_cache(request, call_next):
    response = await call_next(request)
    path = request.url.path
    no_cache_paths = {
        "/",
        "/ml-lab",
        "/static/app.js",
        "/static/app.terminal.js",
        "/static/ml_lab.js",
        "/static/page_menu.js",
        "/static/app.monitor.20260211b.js",
        "/static/page_menu.20260211b.js",
        "/static/styles.css",
        "/static/index.html",
    }
    if path in no_cache_paths:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)
