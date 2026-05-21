"""Saved portfolio analysis API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models import PortfolioAnalysisDraftRequest, PortfolioAnalysisRequest, SavedPortfolioRequest
from ..services.portfolio_analysis import analyze_saved_portfolio
from ..services.portfolio_holdings import resolve_region_holdings
from ..utils import ok_json_response
from .deps import HubDep, PortfolioAnalysisStoreDep, SymbolCatalogStoreDep

router = APIRouter()


async def _resolve_request_holdings(
    *,
    jp_holdings: object,
    us_holdings: object,
    symbol_catalog_store: object,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    resolved_jp = await resolve_region_holdings(
        jp_holdings,
        region="jp",
        symbol_catalog_store=symbol_catalog_store,
    )
    resolved_us = await resolve_region_holdings(
        us_holdings,
        region="us",
        symbol_catalog_store=symbol_catalog_store,
    )
    return resolved_jp, resolved_us


@router.get("/api/portfolio-analysis/portfolios")
async def list_saved_portfolios(portfolio_analysis_store: PortfolioAnalysisStoreDep) -> JSONResponse:
    return ok_json_response(portfolios=await portfolio_analysis_store.list_portfolios())


@router.get("/api/portfolio-analysis/draft")
async def get_portfolio_draft(portfolio_analysis_store: PortfolioAnalysisStoreDep) -> JSONResponse:
    return ok_json_response(draft=await portfolio_analysis_store.get_draft())


@router.post("/api/portfolio-analysis/draft")
async def save_portfolio_draft(
    req: PortfolioAnalysisDraftRequest,
    portfolio_analysis_store: PortfolioAnalysisStoreDep,
) -> JSONResponse:
    draft = await portfolio_analysis_store.save_draft(
        portfolio_id=req.portfolio_id,
        name=req.name,
        lookback_days=req.lookback_days,
        jp_rows=[row.model_dump() for row in req.jp_rows],
        us_rows=[row.model_dump() for row in req.us_rows],
    )
    return ok_json_response(draft=draft)


@router.post("/api/portfolio-analysis/portfolios")
async def save_saved_portfolio(
    req: SavedPortfolioRequest,
    portfolio_analysis_store: PortfolioAnalysisStoreDep,
    symbol_catalog_store: SymbolCatalogStoreDep,
) -> JSONResponse:
    try:
        jp_holdings, us_holdings = await _resolve_request_holdings(
            jp_holdings=req.jp_holdings,
            us_holdings=req.us_holdings,
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
        jp_holdings, us_holdings = await _resolve_request_holdings(
            jp_holdings=req.jp_holdings,
            us_holdings=req.us_holdings,
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
