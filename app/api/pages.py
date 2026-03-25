"""Static page routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..static_pages import HISTORICAL_PAGE_ROUTE, STATIC_PAGES

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


def _static_page(filename: str) -> FileResponse:
    return FileResponse(STATIC_DIR / filename)


def _build_static_page_handler(filename: str):
    async def _handler() -> FileResponse:
        return _static_page(filename)

    return _handler


for page in STATIC_PAGES:
    router.add_api_route(
        page.route_path,
        _build_static_page_handler(page.filename),
        methods=["GET"],
        include_in_schema=False,
        name=page.route_name,
    )


@router.get(HISTORICAL_PAGE_ROUTE, include_in_schema=False)
async def historical_page(symbol: str) -> FileResponse:
    del symbol
    return _static_page("historical.html")
