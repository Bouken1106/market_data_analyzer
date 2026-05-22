"""Top-level route composition for Market Data Analyzer."""

from __future__ import annotations

from fastapi import APIRouter

from .api.day_trading_game import router as day_trading_game_router
from .api.leadlag import router as leadlag_router
from .api.market import router as market_router
from .api.pages import router as pages_router
from .api.portfolio import router as portfolio_router
from .api.portfolio_analysis import router as portfolio_analysis_router

router = APIRouter()
for child_router in (
    pages_router,
    day_trading_game_router,
    leadlag_router,
    market_router,
    portfolio_router,
    portfolio_analysis_router,
):
    router.include_router(child_router)
