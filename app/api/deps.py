"""Shared FastAPI dependency helpers for API routers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request

APP_STATE_HUB = "market_data_hub"
APP_STATE_SYMBOL_CATALOG_STORE = "symbol_catalog_store"
APP_STATE_ML_JOB_STORE = "ml_job_store"
APP_STATE_PAPER_PORTFOLIO_STORE = "paper_portfolio_store"
APP_STATE_UI_STATE_STORE = "ui_state_store"


def init_routes(
    app: FastAPI,
    *,
    hub: Any,
    symbol_catalog_store: Any,
    ml_job_store: Any,
    paper_portfolio_store: Any,
    ui_state_store: Any,
) -> None:
    """Register shared service objects on application state."""
    app.state.__setattr__(APP_STATE_HUB, hub)
    app.state.__setattr__(APP_STATE_SYMBOL_CATALOG_STORE, symbol_catalog_store)
    app.state.__setattr__(APP_STATE_ML_JOB_STORE, ml_job_store)
    app.state.__setattr__(APP_STATE_PAPER_PORTFOLIO_STORE, paper_portfolio_store)
    app.state.__setattr__(APP_STATE_UI_STATE_STORE, ui_state_store)


def _get_app_state_or_500(request: Request, key: str, label: str) -> Any:
    value = getattr(request.app.state, key, None)
    if value is None:
        raise HTTPException(status_code=500, detail=f"{label} is not initialized.")
    return value


def _get_hub(request: Request) -> Any:
    return _get_app_state_or_500(request, APP_STATE_HUB, "hub")


def _get_symbol_catalog_store(request: Request) -> Any:
    return _get_app_state_or_500(request, APP_STATE_SYMBOL_CATALOG_STORE, "symbol catalog store")


def _get_ml_job_store(request: Request) -> Any:
    return _get_app_state_or_500(request, APP_STATE_ML_JOB_STORE, "ML job store")


def _get_paper_portfolio_store(request: Request) -> Any:
    return _get_app_state_or_500(request, APP_STATE_PAPER_PORTFOLIO_STORE, "paper portfolio store")


def _get_ui_state_store(request: Request) -> Any:
    return _get_app_state_or_500(request, APP_STATE_UI_STATE_STORE, "ui state store")


HubDep = Annotated[Any, Depends(_get_hub)]
SymbolCatalogStoreDep = Annotated[Any, Depends(_get_symbol_catalog_store)]
MlJobStoreDep = Annotated[Any, Depends(_get_ml_job_store)]
PaperPortfolioStoreDep = Annotated[Any, Depends(_get_paper_portfolio_store)]
UiStateStoreDep = Annotated[Any, Depends(_get_ui_state_store)]
