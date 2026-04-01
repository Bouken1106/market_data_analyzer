"""Application bootstrap helpers for stateful services."""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .hub import MarketDataHub
from .stores import (
    FmpReferenceStore,
    FullDailyHistoryStore,
    LastPriceStore,
    PaperPortfolioStore,
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


def resolve_default_symbols() -> list[str]:
    symbols = normalize_symbols(settings.default_symbols_raw, max_items=settings.max_basic_symbols)
    return symbols or ["AAPL"]


def validate_provider_configuration() -> None:
    if settings.data_provider == "twelvedata" and not settings.twelve_data_api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is required. Set it in your environment or .env file.")
    if settings.data_provider == "fmp" and not settings.fmp_api_key:
        raise RuntimeError("FMP_API_KEY is required. Set it in your environment or .env file.")
    if settings.data_provider != "both":
        return

    missing: list[str] = []
    if not settings.twelve_data_api_key:
        missing.append("TWELVE_DATA_API_KEY")
    if not settings.fmp_api_key:
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

    last_price_store = LastPriceStore(cache_path=settings.last_price_cache_path)
    full_daily_history_store = FullDailyHistoryStore(cache_dir=settings.full_daily_history_cache_dir)
    fmp_reference_store = FmpReferenceStore(cache_dir=settings.fmp_reference_cache_dir)
    paper_portfolio_store = PaperPortfolioStore(
        cache_path=settings.paper_portfolio_cache_path,
        default_initial_cash=settings.paper_initial_cash,
    )
    symbol_catalog_store = SymbolCatalogStore(
        provider=settings.data_provider,
        twelvedata_api_key=settings.twelve_data_api_key,
        fmp_api_key=settings.fmp_api_key,
        cache_path=settings.symbol_catalog_cache_path,
        ttl_sec=settings.symbol_catalog_ttl_sec,
    )
    ui_state_store = UiStateStore(cache_path=settings.ui_state_cache_path)
    initial_symbols = resolve_initial_symbols(ui_state_store)

    hub = MarketDataHub(
        provider=settings.data_provider,
        twelvedata_api_key=settings.twelve_data_api_key,
        fmp_api_key=settings.fmp_api_key,
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
    )
