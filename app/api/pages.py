"""Static page routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..paths import static_file_path
from ..static_pages import HISTORICAL_PAGE_FILE, HISTORICAL_PAGE_ROUTE, STATIC_PAGES

router = APIRouter()


def _static_page(filename: str) -> FileResponse:
    return FileResponse(static_file_path(filename))


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
    return _static_page(HISTORICAL_PAGE_FILE)
