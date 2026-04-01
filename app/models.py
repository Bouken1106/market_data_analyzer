"""Pydantic request models and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from pydantic import BaseModel


@dataclass(frozen=True)
class MarketSession:
    tz: ZoneInfo
    open_minutes: int
    close_minutes: int
    weekdays: frozenset[int]


class SymbolUpdateRequest(BaseModel):
    symbols: str


class PaperTradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: float | None = None


class PaperPortfolioResetRequest(BaseModel):
    initial_cash: float | None = None
