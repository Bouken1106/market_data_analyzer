"""Shared request validation helpers for API routes."""

from __future__ import annotations

import math
from typing import Any, Iterable

from fastapi import HTTPException

from ..utils import normalize_symbols


def require_symbols(
    raw: str | Iterable[Any],
    *,
    min_count: int = 1,
    max_count: int | None = None,
    empty_detail: str = "At least one valid symbol is required.",
    max_detail: str | None = None,
) -> list[str]:
    symbols = normalize_symbols(raw)
    if len(symbols) < min_count:
        raise HTTPException(status_code=400, detail=empty_detail)
    if max_count is not None and len(symbols) > max_count:
        detail = max_detail or f"You can request up to {max_count} symbols at once."
        raise HTTPException(status_code=400, detail=detail)
    return symbols


def require_symbol(raw: Any, *, detail: str = "Invalid symbol format.") -> str:
    return require_symbols([raw], empty_detail=detail, max_count=1)[0]


def require_positive_float(value: Any, *, detail: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=detail) from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise HTTPException(status_code=400, detail=detail)
    return parsed


def require_non_negative_float(value: Any, *, detail: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=detail) from None
    if not math.isfinite(parsed) or parsed < 0:
        raise HTTPException(status_code=400, detail=detail)
    return parsed
