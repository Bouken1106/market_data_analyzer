"""Static page routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@router.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/compare-lab", include_in_schema=False)
async def compare_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "compare_lab.html")


@router.get("/market-data-lab", include_in_schema=False)
async def market_data_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "market_data_lab.html")


@router.get("/ml-lab", include_in_schema=False)
async def ml_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "ml_lab.html")


@router.get("/strategy-lab", include_in_schema=False)
async def strategy_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "strategy_lab.html")


@router.get("/historical/{symbol}", include_in_schema=False)
async def historical_page(symbol: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "historical.html")
