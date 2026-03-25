"""Shared FastAPI dependency helpers for API routers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request

APP_STATE_HUB = "market_data_hub"
APP_STATE_SYMBOL_CATALOG_STORE = "symbol_catalog_store"
APP_STATE_PAPER_PORTFOLIO_STORE = "paper_portfolio_store"
APP_STATE_UI_STATE_STORE = "ui_state_store"
APP_STATE_STOCK_ML_PAGE_STORE = "stock_ml_page_store"
APP_STATE_ML_JOB_STORE = "ml_job_store"


def init_routes(
    app: FastAPI,
    *,
    hub: Any,
    symbol_catalog_store: Any,
    paper_portfolio_store: Any,
    ui_state_store: Any,
    stock_ml_page_store: Any | None = None,
    ml_job_store: Any | None = None,
) -> None:
    """Register shared service objects on application state."""
    for key, value in (
        (APP_STATE_HUB, hub),
        (APP_STATE_SYMBOL_CATALOG_STORE, symbol_catalog_store),
        (APP_STATE_PAPER_PORTFOLIO_STORE, paper_portfolio_store),
        (APP_STATE_UI_STATE_STORE, ui_state_store),
        (APP_STATE_STOCK_ML_PAGE_STORE, stock_ml_page_store),
        (APP_STATE_ML_JOB_STORE, ml_job_store),
    ):
        if value is not None:
            setattr(app.state, key, value)


def _get_app_state_or_500(request: Request, key: str, label: str) -> Any:
    value = getattr(request.app.state, key, None)
    if value is None:
        raise HTTPException(status_code=500, detail=f"{label} is not initialized.")
    return value


def _build_state_dependency(key: str, label: str):
    def _dependency(request: Request) -> Any:
        return _get_app_state_or_500(request, key, label)

    return _dependency


_get_hub = _build_state_dependency(APP_STATE_HUB, "hub")
_get_symbol_catalog_store = _build_state_dependency(APP_STATE_SYMBOL_CATALOG_STORE, "symbol catalog store")
_get_paper_portfolio_store = _build_state_dependency(APP_STATE_PAPER_PORTFOLIO_STORE, "paper portfolio store")
_get_ui_state_store = _build_state_dependency(APP_STATE_UI_STATE_STORE, "ui state store")
_get_stock_ml_page_store = _build_state_dependency(APP_STATE_STOCK_ML_PAGE_STORE, "stock ML page store")
_get_ml_job_store = _build_state_dependency(APP_STATE_ML_JOB_STORE, "ML job store")

HubDep = Annotated[Any, Depends(_get_hub)]
SymbolCatalogStoreDep = Annotated[Any, Depends(_get_symbol_catalog_store)]
PaperPortfolioStoreDep = Annotated[Any, Depends(_get_paper_portfolio_store)]
UiStateStoreDep = Annotated[Any, Depends(_get_ui_state_store)]
StockMlPageStoreDep = Annotated[Any, Depends(_get_stock_ml_page_store)]
MlJobStoreDep = Annotated[Any, Depends(_get_ml_job_store)]
