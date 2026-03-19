"""Pydantic request models, dataclasses, and custom exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from .config import ML_HISTORY_DEFAULT_MONTHS


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSession:
    tz: ZoneInfo
    open_minutes: int
    close_minutes: int
    weekdays: frozenset[int]


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class SymbolUpdateRequest(BaseModel):
    symbols: str


class QuantileLstmJobRequest(BaseModel):
    symbol: str
    months: int = ML_HISTORY_DEFAULT_MONTHS
    sequence_length: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    representative_days: int = 5
    seed: int = 42
    refresh: bool = False


class MlComparisonJobRequest(BaseModel):
    symbols: str = "AAPL,MSFT,GOOG,JPM,XOM,UNH,WMT,META,LLY,BRK.B,NVDA,HD"
    models: str = "quantile_lstm,patchtst_quantile"
    months: int = ML_HISTORY_DEFAULT_MONTHS
    sequence_length: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    seed: int = 42
    refresh: bool = False


class PaperTradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: float | None = None


class PaperPortfolioResetRequest(BaseModel):
    initial_cash: float | None = None


class StrategyEvaluationRequest(BaseModel):
    symbols: str = "AAPL,MSFT,NVDA,AMZN,GOOGL"
    method: str = "inverse_volatility"
    months: int = 36
    lookback_days: int = 126
    rebalance_frequency: str = "monthly"
    rebalance_threshold_pct: float = 5.0
    max_weight: float = 0.35
    initial_capital: float = 1_000_000.0
    commission_bps: float = 2.0
    slippage_bps: float = 3.0
    benchmark_symbol: str = "SPY"
    min_trade_value: float = 100.0
    refresh: bool = False


class StockMlPageActionRequest(BaseModel):
    prediction_date: str | None = None
    universe_filter: str = "jp_large_cap_stooq_v1"
    model_family: str = "LightGBM Classifier"
    feature_set: str = "base_v1"
    cost_buffer: float = 0.0
    train_window_months: int = 12
    gap_days: int = 5
    valid_window_months: int = 1
    random_seed: int = 42
    train_note: str = ""
    run_note: str = ""
    search_query: str = ""
    confirm_regenerate: bool = False
    refresh: bool = False


class StockMlModelAdoptionRequest(StockMlPageActionRequest):
    model_version: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MlJobCancelledError(Exception):
    pass
