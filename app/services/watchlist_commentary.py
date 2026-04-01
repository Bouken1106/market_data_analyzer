"""Watchlist commentary entrypoints for the market API."""

from __future__ import annotations

from typing import Any

from .watchlist_commentary_service import WatchlistCommentaryService

_watchlist_commentary_service = WatchlistCommentaryService()


async def build_watchlist_commentary_payload(
    hub: Any,
    symbols: list[str],
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    return await _watchlist_commentary_service.build_payload(hub, symbols, refresh=refresh)


__all__ = ["build_watchlist_commentary_payload"]
