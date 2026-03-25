"""Top-level route composition for Market Data Analyzer."""

from __future__ import annotations

from fastapi import APIRouter

from .api.leadlag import router as leadlag_router
from .api.market import router as market_router
from .api.ml import router as ml_router
from .api.pages import router as pages_router
from .api.portfolio import router as portfolio_router
from .api.strategy import router as strategy_router

router = APIRouter()
for child_router in (
    pages_router,
    leadlag_router,
    market_router,
    portfolio_router,
    strategy_router,
    ml_router,
):
    router.include_router(child_router)
