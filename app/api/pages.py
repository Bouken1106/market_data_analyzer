"""Static page routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
STATIC_PAGES = (
    ("/", "index.html"),
    ("/market-data-lab", "market_data_lab.html"),
    ("/ml-lab", "ml_lab.html"),
    ("/strategy-lab", "strategy_lab.html"),
    ("/compare-lab", "compare_lab.html"),
    ("/leadlag-lab", "leadlag_lab.html"),
)


def _static_page(filename: str) -> FileResponse:
    return FileResponse(STATIC_DIR / filename)


def _build_static_page_handler(filename: str):
    async def _handler() -> FileResponse:
        return _static_page(filename)

    return _handler


for route_path, filename in STATIC_PAGES:
    router.add_api_route(
        route_path,
        _build_static_page_handler(filename),
        methods=["GET"],
        include_in_schema=False,
        name=filename.removesuffix(".html"),
    )


@router.get("/historical/{symbol}", include_in_schema=False)
async def historical_page(symbol: str) -> FileResponse:
    del symbol
    return _static_page("historical.html")
