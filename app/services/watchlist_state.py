"""Shared helpers for persisted watchlist state."""

from __future__ import annotations

from typing import Iterable

DEFAULT_WATCHLIST_NAMESPACE = "us"
SUPPORTED_WATCHLIST_NAMESPACES = frozenset({"us", "jp"})


class UnsupportedWatchlistNamespace(ValueError):
    """Raised when a watchlist namespace is outside the accepted set."""


def normalize_watchlist_namespace(
    namespace: str | None,
    *,
    allowed: Iterable[str] | None = None,
) -> str:
    normalized = str(namespace or "").strip().lower() or DEFAULT_WATCHLIST_NAMESPACE
    if allowed is not None:
        allowed_namespaces = {str(item).strip().lower() for item in allowed}
        if normalized not in allowed_namespaces:
            raise UnsupportedWatchlistNamespace("Unsupported watchlist namespace.")
    return normalized
