"""Top-level route composition for Market Data Analyzer."""

from __future__ import annotations

from fastapi import APIRouter

from .api.deps import init_routes
from .api.market import router as market_router
from .api.ml import router as ml_router
from .api.pages import router as pages_router
from .api.portfolio import router as portfolio_router
from .api.strategy import router as strategy_router

router = APIRouter()
router.include_router(pages_router)
router.include_router(market_router)
router.include_router(portfolio_router)
router.include_router(strategy_router)
router.include_router(ml_router)

