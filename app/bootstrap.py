"""Application bootstrap helpers for stateful services."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class AppServices:
    hub: MarketDataHub
    symbol_catalog_store: SymbolCatalogStore
    paper_portfolio_store: PaperPortfolioStore
    ui_state_store: UiStateStore
    stock_ml_page_store: StockMlPageStore
    ml_job_store: MlJobStore


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
    ui_state_store = UiStateStore(cache_path=UI_STATE_CACHE_PATH)
    stock_ml_page_store = StockMlPageStore(cache_path=STOCK_ML_PAGE_STATE_CACHE_PATH)
    ml_job_store = MlJobStore(max_jobs=120)
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
        paper_portfolio_store=paper_portfolio_store,
        ui_state_store=ui_state_store,
        stock_ml_page_store=stock_ml_page_store,
        ml_job_store=ml_job_store,
    )
