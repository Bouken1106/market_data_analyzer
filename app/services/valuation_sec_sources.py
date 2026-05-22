"""SEC EDGAR adapters for valuation metrics."""

from __future__ import annotations

from typing import Any

import httpx

from ..utils import normalize_symbol
from .valuation_errors import ValuationDataError
from .valuation_fact_pickers import SecFactPicker
from .valuation_models import MARKET_US, FinancialMetrics
from .valuation_numeric import (
    abs_or_none as _abs_or_none,
    div_optional as _div_optional,
    first_present,
    parse_float as _parse_float,
    sub_optional as _sub_optional,
    sum_optional as _sum_optional,
)


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


async def fetch_sec_cik_for_ticker(client: httpx.AsyncClient, symbol: str) -> int:
    ticker_response = await client.get(SEC_COMPANY_TICKERS_URL, headers={"Host": "www.sec.gov"})
    ticker_response.raise_for_status()
    payload = ticker_response.json()
    normalized = normalize_symbol(symbol)
    rows = payload.values() if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if normalize_symbol(row.get("ticker")) == normalized:
            cik = _parse_float(row.get("cik_str"))
            if cik is not None:
                return int(cik)
    raise ValuationDataError(f"SEC CIK not found for ticker {normalized}.")


def normalize_sec_company_facts(symbol: str, cik: int, payload: dict[str, Any]) -> FinancialMetrics:
    facts = payload.get("facts") if isinstance(payload, dict) else {}
    entity_name = payload.get("entityName") if isinstance(payload, dict) else None
    latest_duration = SecFactPicker(facts, annual_only=False)
    latest_annual = SecFactPicker(facts, annual_only=True)
    instant = SecFactPicker(facts, instant_only=True)

    revenue = latest_annual.value(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        unit="USD",
    )
    gross_profit = latest_annual.value("GrossProfit", unit="USD")
    operating_income = latest_annual.value("OperatingIncomeLoss", unit="USD")
    ebit = operating_income or latest_annual.value(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        unit="USD",
    )
    depreciation = latest_annual.value(
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortizationExpense",
        unit="USD",
    )
    net_income = latest_annual.value("NetIncomeLoss", "ProfitLoss", unit="USD")
    operating_cf = latest_annual.value("NetCashProvidedByUsedInOperatingActivities", unit="USD")
    capex = _abs_or_none(latest_annual.value("PaymentsToAcquirePropertyPlantAndEquipment", unit="USD"))
    dividends_paid = _abs_or_none(
        latest_annual.value("PaymentsOfDividends", "PaymentsOfOrdinaryDividends", unit="USD")
    )
    share_repurchases = _abs_or_none(
        latest_annual.value("PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity", unit="USD")
    )
    proceeds = latest_annual.value(
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromBorrowings",
        "ProceedsFromDebtNetOfIssuanceCosts",
        unit="USD",
    )
    repayments = latest_annual.value(
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
        unit="USD",
    )
    net_borrowing = _sub_optional(proceeds, repayments)

    cash = instant.value(
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        unit="USD",
    )
    short_term_investments = instant.value("ShortTermInvestments", unit="USD")
    debt = first_present(
        instant.value("LongTermDebtAndFinanceLeaseObligations", unit="USD"),
        _sum_optional(
            instant.value("ShortTermBorrowings", "ShortTermDebt", "LongTermDebtCurrent", unit="USD"),
            instant.value("LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent", unit="USD"),
        ),
    )
    total_assets = instant.value("Assets", unit="USD")
    current_assets = instant.value("AssetsCurrent", unit="USD")
    total_liabilities = instant.value("Liabilities", unit="USD")
    equity = instant.value(
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        unit="USD",
    )
    shares = first_present(
        instant.value("EntityCommonStockSharesOutstanding", unit="shares", taxonomy="dei"),
        latest_duration.value(
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingDiluted",
            unit="shares",
        ),
        latest_duration.value("WeightedAverageNumberOfSharesOutstandingBasic", unit="shares"),
    )
    eps = latest_annual.value("EarningsPerShareDiluted", "EarningsPerShareBasic", unit="USD/shares")
    interest_expense = _abs_or_none(latest_annual.value("InterestExpenseNonOperating", "InterestExpense", unit="USD"))
    tax_expense = latest_annual.value("IncomeTaxExpenseBenefit", unit="USD")
    pretax_income = latest_annual.value(
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        unit="USD",
    )
    net_income_history = tuple(latest_annual.history("NetIncomeLoss", "ProfitLoss", unit="USD", max_items=5))
    revenue_history = tuple(
        latest_annual.history(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            unit="USD",
            max_items=5,
        )
    )
    eps_history = tuple(
        latest_annual.history("EarningsPerShareDiluted", "EarningsPerShareBasic", unit="USD/shares", max_items=5)
    )

    return FinancialMetrics(
        symbol=normalize_symbol(symbol),
        market=MARKET_US,
        currency="USD",
        company_name=str(entity_name or "") or None,
        fiscal_date=latest_annual.latest_end_date(),
        revenue=revenue,
        gross_profit=gross_profit,
        operating_income=operating_income,
        ebit=ebit,
        depreciation_and_amortization=depreciation,
        ebitda=_sum_optional(ebit, depreciation),
        net_income=net_income,
        eps=eps,
        operating_cash_flow=operating_cf,
        capital_expenditure=capex,
        free_cash_flow=_sub_optional(operating_cf, capex),
        net_borrowing=net_borrowing,
        cash_and_equivalents=cash,
        short_term_investments=short_term_investments,
        interest_bearing_debt=debt,
        total_liabilities=total_liabilities,
        current_assets=current_assets,
        total_assets=total_assets,
        equity=equity,
        shareholders_equity=equity,
        shares_outstanding=shares,
        dividends_paid=dividends_paid,
        share_repurchases=share_repurchases,
        dividend_per_share=_div_optional(dividends_paid, shares),
        interest_expense=interest_expense,
        income_tax_expense=tax_expense,
        income_before_tax=pretax_income,
        net_income_history=net_income_history,
        revenue_history=revenue_history,
        eps_history=eps_history,
        data_sources=(f"SEC:companyfacts:CIK{cik:010d}",),
        raw={"cik": cik},
    )
