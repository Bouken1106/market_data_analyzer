"""Errors shared by valuation data-source adapters."""


class ValuationDataError(RuntimeError):
    """Raised when a requested free data source cannot return usable data."""
