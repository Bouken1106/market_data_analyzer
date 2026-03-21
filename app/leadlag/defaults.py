"""Paper-aligned defaults for the lead-lag PCA strategy."""

from __future__ import annotations

DEFAULT_US_SYMBOLS: tuple[str, ...] = (
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)

DEFAULT_JP_SYMBOLS: tuple[str, ...] = (
    "1617.T",
    "1618.T",
    "1619.T",
    "1620.T",
    "1621.T",
    "1622.T",
    "1623.T",
    "1624.T",
    "1625.T",
    "1626.T",
    "1627.T",
    "1628.T",
    "1629.T",
    "1630.T",
    "1631.T",
    "1632.T",
    "1633.T",
)

# These labels follow the paper text and are intentionally left configurable.
DEFAULT_CYCLICAL_SYMBOLS = frozenset(
    {
        "XLB",
        "XLE",
        "XLF",
        "XLRE",
        "1618.T",
        "1625.T",
        "1629.T",
        "1631.T",
    }
)

DEFAULT_DEFENSIVE_SYMBOLS = frozenset(
    {
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "1617.T",
        "1621.T",
        "1627.T",
        "1630.T",
    }
)

DEFAULT_CFULL_START = "2010-01-01"
DEFAULT_CFULL_END = "2014-12-31"
DEFAULT_ROLLING_WINDOW_DAYS = 60
DEFAULT_LAMBDA_REG = 0.9
DEFAULT_N_COMPONENTS = 3
DEFAULT_QUANTILE_Q = 0.3
DEFAULT_HISTORY_YEARS = 30
