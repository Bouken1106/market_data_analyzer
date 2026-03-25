"""Shared static page metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StaticPage:
    route_path: str
    filename: str

    @property
    def route_name(self) -> str:
        return self.filename.removesuffix(".html")


STATIC_PAGES: tuple[StaticPage, ...] = (
    StaticPage("/", "index.html"),
    StaticPage("/market-data-lab", "market_data_lab.html"),
    StaticPage("/ml-lab", "ml_lab.html"),
    StaticPage("/strategy-lab", "strategy_lab.html"),
    StaticPage("/compare-lab", "compare_lab.html"),
    StaticPage("/leadlag-lab", "leadlag_lab.html"),
)

HISTORICAL_PAGE_ROUTE = "/historical/{symbol}"
HISTORICAL_PAGE_PATH_PREFIX = "/historical/"
NO_CACHE_PATHS = frozenset(
    {
        *(page.route_path for page in STATIC_PAGES),
        "/static/app.terminal.js",
        "/static/styles.css",
        "/static/index.html",
    }
)


def is_no_cache_path(path: str) -> bool:
    return path in NO_CACHE_PATHS or path.startswith(HISTORICAL_PAGE_PATH_PREFIX)
