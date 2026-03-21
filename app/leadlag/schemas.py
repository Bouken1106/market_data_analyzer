"""Request models and typed config for the lead-lag module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel

from ..utils import normalize_symbols
from .defaults import (
    DEFAULT_CFULL_END,
    DEFAULT_CFULL_START,
    DEFAULT_CYCLICAL_SYMBOLS,
    DEFAULT_DEFENSIVE_SYMBOLS,
    DEFAULT_HISTORY_YEARS,
    DEFAULT_JP_SYMBOLS,
    DEFAULT_LAMBDA_REG,
    DEFAULT_N_COMPONENTS,
    DEFAULT_QUANTILE_Q,
    DEFAULT_ROLLING_WINDOW_DAYS,
    DEFAULT_US_SYMBOLS,
)


@dataclass(frozen=True)
class LeadLagConfig:
    """Runtime configuration for one lead-lag analysis run."""

    us_symbols: tuple[str, ...]
    jp_symbols: tuple[str, ...]
    rolling_window_days: int
    lambda_reg: float
    n_components: int
    quantile_q: float
    cfull_start: str
    cfull_end: str
    cyclical_symbols: frozenset[str]
    defensive_symbols: frozenset[str]
    refresh: bool
    include_backtest: bool
    include_transfer_matrix: bool
    history_years: int

    @property
    def all_symbols(self) -> tuple[str, ...]:
        return self.us_symbols + self.jp_symbols


class LeadLagAnalysisRequest(BaseModel):
    """HTTP request model for lead-lag analysis."""

    us_symbols: str = ",".join(DEFAULT_US_SYMBOLS)
    jp_symbols: str = ",".join(DEFAULT_JP_SYMBOLS)
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS
    lambda_reg: float = DEFAULT_LAMBDA_REG
    n_components: int = DEFAULT_N_COMPONENTS
    quantile_q: float = DEFAULT_QUANTILE_Q
    cfull_start: str = DEFAULT_CFULL_START
    cfull_end: str = DEFAULT_CFULL_END
    cyclical_symbols: str = ",".join(sorted(DEFAULT_CYCLICAL_SYMBOLS))
    defensive_symbols: str = ",".join(sorted(DEFAULT_DEFENSIVE_SYMBOLS))
    refresh: bool = False
    include_backtest: bool = True
    include_transfer_matrix: bool = False
    history_years: int = DEFAULT_HISTORY_YEARS


def build_leadlag_config(req: LeadLagAnalysisRequest) -> LeadLagConfig:
    """Validate a request and convert it into a frozen runtime config."""

    us_symbols = tuple(normalize_symbols(req.us_symbols))
    if not us_symbols:
        raise ValueError("At least one valid U.S. symbol is required.")

    jp_symbols = tuple(normalize_symbols(req.jp_symbols))
    if not jp_symbols:
        raise ValueError("At least one valid Japan symbol is required.")

    rolling_window_days = int(req.rolling_window_days)
    if rolling_window_days < 2:
        raise ValueError("rolling_window_days must be at least 2.")

    lambda_reg = float(req.lambda_reg)
    if lambda_reg < 0.0 or lambda_reg > 1.0:
        raise ValueError("lambda_reg must be within [0, 1].")

    n_components = int(req.n_components)
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    quantile_q = float(req.quantile_q)
    if quantile_q <= 0.0 or quantile_q >= 0.5:
        raise ValueError("quantile_q must be greater than 0 and less than 0.5.")

    cfull_start = date.fromisoformat(str(req.cfull_start))
    cfull_end = date.fromisoformat(str(req.cfull_end))
    if cfull_start >= cfull_end:
        raise ValueError("cfull_start must be earlier than cfull_end.")

    cyclical_symbols = frozenset(normalize_symbols(req.cyclical_symbols))
    defensive_symbols = frozenset(normalize_symbols(req.defensive_symbols))
    if not cyclical_symbols:
        raise ValueError("At least one cyclical symbol label is required.")
    if not defensive_symbols:
        raise ValueError("At least one defensive symbol label is required.")

    history_years = max(5, int(req.history_years))

    return LeadLagConfig(
        us_symbols=us_symbols,
        jp_symbols=jp_symbols,
        rolling_window_days=rolling_window_days,
        lambda_reg=lambda_reg,
        n_components=n_components,
        quantile_q=quantile_q,
        cfull_start=cfull_start.isoformat(),
        cfull_end=cfull_end.isoformat(),
        cyclical_symbols=cyclical_symbols,
        defensive_symbols=defensive_symbols,
        refresh=bool(req.refresh),
        include_backtest=bool(req.include_backtest),
        include_transfer_matrix=bool(req.include_transfer_matrix),
        history_years=history_years,
    )
