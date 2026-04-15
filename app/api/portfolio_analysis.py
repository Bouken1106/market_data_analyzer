"""Saved portfolio analysis API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models import PortfolioAnalysisRequest, SavedPortfolioRequest
from ..services.portfolio_analysis import analyze_saved_portfolio, resolve_region_holdings
from ..utils import ok_json_response
from .deps import HubDep, PortfolioAnalysisStoreDep, SymbolCatalogStoreDep

router = APIRouter()


@router.get("/api/portfolio-analysis/portfolios")
async def list_saved_portfolios(portfolio_analysis_store: PortfolioAnalysisStoreDep) -> JSONResponse:
    return ok_json_response(portfolios=await portfolio_analysis_store.list_portfolios())


@router.post("/api/portfolio-analysis/portfolios")
async def save_saved_portfolio(
    req: SavedPortfolioRequest,
    portfolio_analysis_store: PortfolioAnalysisStoreDep,
    symbol_catalog_store: SymbolCatalogStoreDep,
) -> JSONResponse:
    try:
        jp_holdings = await resolve_region_holdings(
            req.jp_holdings,
            region="jp",
            symbol_catalog_store=symbol_catalog_store,
        )
        us_holdings = await resolve_region_holdings(
            req.us_holdings,
            region="us",
            symbol_catalog_store=symbol_catalog_store,
        )
        portfolio = await portfolio_analysis_store.save_portfolio(
            portfolio_id=req.portfolio_id,
            name=req.name,
            jp_holdings=jp_holdings,
            us_holdings=us_holdings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    return ok_json_response(
        portfolio=portfolio,
        portfolios=await portfolio_analysis_store.list_portfolios(),
    )


@router.delete("/api/portfolio-analysis/portfolios/{portfolio_id}")
async def delete_saved_portfolio(
    portfolio_id: str,
    portfolio_analysis_store: PortfolioAnalysisStoreDep,
) -> JSONResponse:
    deleted = await portfolio_analysis_store.delete_portfolio(portfolio_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Saved portfolio not found.")
    return ok_json_response(portfolios=await portfolio_analysis_store.list_portfolios())


@router.post("/api/portfolio-analysis/analyze")
async def analyze_portfolio(
    req: PortfolioAnalysisRequest,
    hub: HubDep,
    symbol_catalog_store: SymbolCatalogStoreDep,
) -> JSONResponse:
    try:
        jp_holdings = await resolve_region_holdings(
            req.jp_holdings,
            region="jp",
            symbol_catalog_store=symbol_catalog_store,
        )
        us_holdings = await resolve_region_holdings(
            req.us_holdings,
            region="us",
            symbol_catalog_store=symbol_catalog_store,
        )
        payload = await analyze_saved_portfolio(
            hub,
            jp_holdings=jp_holdings,
            us_holdings=us_holdings,
            lookback_days=req.lookback_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return ok_json_response(**payload)
