"""Market Data Analyzer application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import (
    DATA_PROVIDER,
    DEFAULT_SYMBOLS_RAW,
    FMP_API_KEY,
    FMP_REFERENCE_CACHE_DIR,
    FULL_DAILY_HISTORY_CACHE_DIR,
    LAST_PRICE_CACHE_PATH,
    MAX_BASIC_SYMBOLS,
    PAPER_INITIAL_CASH,
    PAPER_PORTFOLIO_CACHE_PATH,
    STOCK_ML_PAGE_STATE_CACHE_PATH,
    SYMBOL_CATALOG_CACHE_PATH,
    SYMBOL_CATALOG_TTL_SEC,
    TWELVE_DATA_API_KEY,
    UI_STATE_CACHE_PATH,
)
from .hub import MarketDataHub
from .ml.job_store import MlJobStore
from .routes import init_routes, router
from .stores import (
    FmpReferenceStore,
    FullDailyHistoryStore,
    LastPriceStore,
    PaperPortfolioStore,
    StockMlPageStore,
    SymbolCatalogStore,
    UiStateStore,
)
from .utils import normalize_symbols

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE_PATHS = frozenset(
    {
        "/",
        "/market-data-lab",
        "/ml-lab",
        "/static/app.js",
        "/static/app.terminal.js",
        "/static/market_data_lab.js",
        "/static/market_data_lab.html",
        "/static/ml_lab.js",
        "/static/page_menu.js",
        "/static/app.monitor.20260211b.js",
        "/static/page_menu.20260211b.js",
        "/static/styles.css",
        "/static/index.html",
    }
)


@dataclass(frozen=True)
class AppServices:
    hub: MarketDataHub
    symbol_catalog_store: SymbolCatalogStore
    ml_job_store: MlJobStore
    paper_portfolio_store: PaperPortfolioStore
    ui_state_store: UiStateStore
    stock_ml_page_store: StockMlPageStore


def resolve_default_symbols() -> list[str]:
    symbols = normalize_symbols(DEFAULT_SYMBOLS_RAW, max_items=MAX_BASIC_SYMBOLS)
    return symbols or ["AAPL"]


def validate_provider_configuration() -> None:
    if DATA_PROVIDER == "twelvedata" and not TWELVE_DATA_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is required. Set it in your environment or .env file.")
    if DATA_PROVIDER == "fmp" and not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is required. Set it in your environment or .env file.")
    if DATA_PROVIDER != "both":
        return

    missing: list[str] = []
    if not TWELVE_DATA_API_KEY:
        missing.append("TWELVE_DATA_API_KEY")
    if not FMP_API_KEY:
        missing.append("FMP_API_KEY")
    if missing:
        raise RuntimeError(f"{', '.join(missing)} is required when MARKET_DATA_PROVIDER=both.")


def resolve_initial_symbols(ui_state_store: UiStateStore) -> list[str]:
    persisted_symbols = ui_state_store.get_symbols()
    if persisted_symbols:
        return persisted_symbols

    initial_symbols = resolve_default_symbols()
    ui_state_store.set_symbols(initial_symbols)
    return initial_symbols


def build_services() -> AppServices:
    validate_provider_configuration()

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
    stock_ml_page_store = StockMlPageStore(cache_path=STOCK_ML_PAGE_STATE_CACHE_PATH)
    initial_symbols = resolve_initial_symbols(ui_state_store)

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
    return AppServices(
        hub=hub,
        symbol_catalog_store=symbol_catalog_store,
        ml_job_store=ml_job_store,
        paper_portfolio_store=paper_portfolio_store,
        ui_state_store=ui_state_store,
        stock_ml_page_store=stock_ml_page_store,
    )


def create_lifespan(hub: MarketDataHub):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await hub.start()
        try:
            yield
        finally:
            await hub.stop()

    return lifespan


def register_no_cache_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def disable_monitor_asset_cache(request, call_next):
        response = await call_next(request)
        if request.url.path in NO_CACHE_PATHS:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


def create_app(services: AppServices | None = None) -> FastAPI:
    resolved_services = services or build_services()
    app = FastAPI(title="Market Data Analyzer", lifespan=create_lifespan(resolved_services.hub))

    init_routes(
        app,
        hub=resolved_services.hub,
        symbol_catalog_store=resolved_services.symbol_catalog_store,
        ml_job_store=resolved_services.ml_job_store,
        paper_portfolio_store=resolved_services.paper_portfolio_store,
        ui_state_store=resolved_services.ui_state_store,
        stock_ml_page_store=resolved_services.stock_ml_page_store,
    )
    register_no_cache_middleware(app)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(router)
    return app


services = build_services()
hub = services.hub
app = create_app(services)
