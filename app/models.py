"""Pydantic request models and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class MarketSession:
    tz: ZoneInfo
    open_minutes: int
    close_minutes: int
    weekdays: frozenset[int]


class SymbolUpdateRequest(BaseModel):
    symbols: str


class WatchlistStateRequest(BaseModel):
    namespace: str | None = None
    symbols: list[str] = Field(default_factory=list)


class PaperTradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: float | None = None


class PaperPortfolioResetRequest(BaseModel):
    initial_cash: float | None = None


class PortfolioHoldingRequest(BaseModel):
    symbol: str
    quantity: float


class SavedPortfolioRequest(BaseModel):
    portfolio_id: str | None = None
    name: str
    jp_holdings: list[PortfolioHoldingRequest] = Field(default_factory=list)
    us_holdings: list[PortfolioHoldingRequest] = Field(default_factory=list)


class PortfolioAnalysisRequest(BaseModel):
    jp_holdings: list[PortfolioHoldingRequest] = Field(default_factory=list)
    us_holdings: list[PortfolioHoldingRequest] = Field(default_factory=list)
    lookback_days: int | None = 252


class PortfolioDraftRowRequest(BaseModel):
    symbol: str = ""
    quantity: str = ""


class PortfolioAnalysisDraftRequest(BaseModel):
    portfolio_id: str | None = None
    name: str = ""
    lookback_days: int | None = 252
    jp_rows: list[PortfolioDraftRowRequest] = Field(default_factory=list)
    us_rows: list[PortfolioDraftRowRequest] = Field(default_factory=list)
