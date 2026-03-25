"""Paper portfolio API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models import PaperPortfolioResetRequest, PaperTradeRequest
from ..services.paper_portfolio import paper_portfolio_payload, resolve_trade_price
from ..utils import ok_json_response
from .deps import HubDep, PaperPortfolioStoreDep
from .validators import require_positive_float, require_symbol

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
    symbol = require_symbol(req.symbol, detail="Invalid symbol format.")
    side = str(req.side or "").lower().strip()
    quantity = require_positive_float(req.quantity, detail="quantity must be greater than 0.")

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
    initial_cash = None
    if req.initial_cash is not None:
        initial_cash = require_positive_float(req.initial_cash, detail="initial_cash must be greater than 0.")
    await paper_portfolio_store.reset(initial_cash=initial_cash)
    payload = await paper_portfolio_payload(hub, paper_portfolio_store)
    return ok_json_response(**payload)
