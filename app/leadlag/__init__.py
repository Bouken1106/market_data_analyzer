"""Lead-lag strategy package based on subspace-regularized PCA."""

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
from .schemas import LeadLagAnalysisRequest, LeadLagConfig, build_leadlag_config
from .service import LeadLagService

__all__ = [
    "DEFAULT_CFULL_END",
    "DEFAULT_CFULL_START",
    "DEFAULT_CYCLICAL_SYMBOLS",
    "DEFAULT_DEFENSIVE_SYMBOLS",
    "DEFAULT_HISTORY_YEARS",
    "DEFAULT_JP_SYMBOLS",
    "DEFAULT_LAMBDA_REG",
    "DEFAULT_N_COMPONENTS",
    "DEFAULT_QUANTILE_Q",
    "DEFAULT_ROLLING_WINDOW_DAYS",
    "DEFAULT_US_SYMBOLS",
    "LeadLagAnalysisRequest",
    "LeadLagConfig",
    "LeadLagService",
    "build_leadlag_config",
]
