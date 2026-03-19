"""Paper portfolio API routes."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models import PaperPortfolioResetRequest, PaperTradeRequest
from ..services.paper_portfolio import paper_portfolio_payload, resolve_trade_price
from ..utils import normalize_symbols, ok_json_response
from .deps import HubDep, PaperPortfolioStoreDep

router = APIRouter()


@router.get("/api/portfolio")
async def paper_portfolio(hub: HubDep, paper_portfolio_store: PaperPortfolioStoreDep) -> JSONResponse:
    payload = await paper_portfolio_payload(hub, paper_portfolio_store)
    return ok_json_response(**payload)


@router.post("/api/portfolio/trades")
async def paper_trade(
    req: PaperTradeRequest,
    hub: HubDep,
    paper_portfolio_store: PaperPortfolioStoreDep,
) -> JSONResponse:
    symbols = normalize_symbols([req.symbol])
    if not symbols:
        raise HTTPException(status_code=400, detail="Invalid symbol format.")
    symbol = symbols[0]
    side = str(req.side or "").lower().strip()

    try:
        quantity = float(req.quantity)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="quantity must be greater than 0.") from None
    if not math.isfinite(quantity) or quantity <= 0:
        raise HTTPException(status_code=400, detail="quantity must be greater than 0.")

    execution_price, execution_source = await resolve_trade_price(hub, symbol, req.price)
    try:
        trade = await paper_portfolio_store.apply_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    payload = await paper_portfolio_payload(hub, paper_portfolio_store)
    return ok_json_response(
        trade={**trade, "execution_source": execution_source},
        **payload,
    )


@router.post("/api/portfolio/reset")
async def paper_portfolio_reset(
    req: PaperPortfolioResetRequest,
    hub: HubDep,
    paper_portfolio_store: PaperPortfolioStoreDep,
) -> JSONResponse:
    if req.initial_cash is not None:
        try:
            initial_cash = float(req.initial_cash)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="initial_cash must be greater than 0.") from None
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise HTTPException(status_code=400, detail="initial_cash must be greater than 0.")
    await paper_portfolio_store.reset(initial_cash=req.initial_cash)
    payload = await paper_portfolio_payload(hub, paper_portfolio_store)
    return ok_json_response(**payload)
