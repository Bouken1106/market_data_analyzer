"""EDINET adapters for valuation metrics."""

from __future__ import annotations

import csv
import io
import zipfile

from ..utils import normalize_symbol
from .valuation_fact_pickers import EdinetFactPicker
from .valuation_models import MARKET_JP, FinancialMetrics
from .valuation_numeric import (
    abs_or_none as _abs_or_none,
    first_present_text as _first_present_text,
    parse_float as _parse_float,
    sub_optional as _sub_optional,
    sum_optional as _sum_optional,
)


def parse_edinet_xbrl_to_csv_zip(content: bytes) -> dict[str, float]:
    facts: dict[str, float] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        for name in names:
            if "XBRL_TO_CSV" not in name and "xbrl_to_csv" not in name.lower():
                continue
            raw = archive.read(name)
            text = raw.decode("utf-16", errors="ignore")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t")
            for row in reader:
                key = _first_present_text(row, "要素ID", "Element ID", "element_id")
                value = _first_present_text(row, "値", "Value", "value")
                numeric = _parse_float(value)
                if key and numeric is not None:
                    facts[key] = numeric
    return facts


def normalize_edinet_metrics(symbol: str, facts: dict[str, float], *, doc_id: str) -> FinancialMetrics:
    picker = EdinetFactPicker(facts)
    operating_cf = picker.value("NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromOperatingActivities")
    capex = _abs_or_none(
        picker.value(
            "PurchaseOfPropertyPlantAndEquipment",
            "PaymentsForPurchaseOfPropertyPlantAndEquipment",
            "PurchaseOfIntangibleAssets",
        )
    )
    cash = picker.value("CashAndCashEquivalents", "CashAndCashEquivalentsIFRS", "CashAndDeposits")
    debt = _sum_optional(
        picker.value("BondsAndBorrowingsCurrent", "ShortTermBorrowings", "CurrentPortionOfLongTermBorrowings"),
        picker.value("BondsAndBorrowingsNonCurrent", "LongTermBorrowings", "LongTermDebt"),
    )
    revenue = picker.value("NetSales", "Revenue", "RevenueIFRS")
    operating_income = picker.value("OperatingIncome", "OperatingProfitLoss", "OperatingProfitIFRS")
    net_income = picker.value("ProfitLossAttributableToOwnersOfParent", "NetIncome", "ProfitLoss")
    equity = picker.value("EquityAttributableToOwnersOfParent", "NetAssets", "Equity")
    total_assets = picker.value("Assets", "TotalAssets")
    total_liabilities = picker.value("Liabilities", "TotalLiabilities")
    current_assets = picker.value("CurrentAssets", "AssetsCurrent")
    shares = picker.value(
        "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        "TotalNumberOfIssuedShares",
    )

    return FinancialMetrics(
        symbol=normalize_symbol(symbol),
        market=MARKET_JP,
        currency="JPY",
        revenue=revenue,
        operating_income=operating_income,
        ebit=operating_income,
        net_income=net_income,
        operating_cash_flow=operating_cf,
        capital_expenditure=capex,
        free_cash_flow=_sub_optional(operating_cf, capex),
        cash_and_equivalents=cash,
        interest_bearing_debt=debt,
        total_liabilities=total_liabilities,
        current_assets=current_assets,
        total_assets=total_assets,
        equity=equity,
        shareholders_equity=equity,
        shares_outstanding=shares,
        data_sources=(f"EDINET:{doc_id}:csv",),
        raw={"edinet_doc_id": doc_id},
    )
