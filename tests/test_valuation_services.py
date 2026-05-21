from __future__ import annotations

import unittest

from app.services.valuation_data_sources import (
    merge_financial_metrics,
    normalize_jquants_metrics,
    normalize_sec_company_facts,
    parse_fred_latest_rate,
    parse_mof_jgb_10y_csv,
)
from app.services.valuation_models import (
    ComparableMultiples,
    FinancialMetrics,
    ValuationAssumptions,
    calculate_valuation_report,
)


class ValuationServiceTest(unittest.TestCase):
    def test_calculates_supported_operating_company_methods(self) -> None:
        metrics = FinancialMetrics(
            symbol="EXM",
            market="US",
            currency="USD",
            price=100.0,
            shares_outstanding=100.0,
            revenue=10_000.0,
            operating_income=900.0,
            ebitda=1_000.0,
            net_income=500.0,
            eps=5.0,
            operating_cash_flow=800.0,
            capital_expenditure=200.0,
            cash_and_equivalents=1_000.0,
            short_term_investments=100.0,
            interest_bearing_debt=500.0,
            total_liabilities=3_000.0,
            current_assets=3_600.0,
            total_assets=5_000.0,
            equity=2_000.0,
            dividend_per_share=1.0,
            beta=1.0,
            risk_free_rate=0.02,
            data_sources=("unit",),
        )
        multiples = ComparableMultiples(
            fair_per=15.0,
            fair_pbr=2.0,
            fair_psr=1.5,
            fair_ev_sales=1.2,
            fair_ev_ebitda=8.0,
            fair_ev_fcf=12.0,
            fair_p_fcf=14.0,
            target_dividend_yield=0.02,
            source="test_peer_median",
        )

        report = calculate_valuation_report(metrics, multiples=multiples)
        by_method = {item.method_name: item for item in report.valuations}

        self.assertEqual(by_method["実績PER法"].theoretical_price, 75.0)
        self.assertEqual(by_method["PBR法"].theoretical_price, 40.0)
        self.assertEqual(by_method["PSR法"].theoretical_price, 150.0)
        self.assertAlmostEqual(by_method["EV/EBITDA法"].theoretical_price or 0, 86.0)
        self.assertIsNone(by_method["予想PER法"].theoretical_price)
        self.assertEqual(by_method["予想PER法"].unavailable_reason, "forecast EPS is missing or non-positive")
        self.assertIsNotNone(by_method["簡易DCF法"].theoretical_price)

    def test_sector_rules_exclude_bank_ev_and_fcf_methods(self) -> None:
        metrics = FinancialMetrics(
            symbol="BANK",
            market="JP",
            currency="JPY",
            sector="銀行業",
            price=1000.0,
            shares_outstanding=100.0,
            equity=100_000.0,
            net_income=5_000.0,
            eps=50.0,
            ebitda=10_000.0,
            operating_cash_flow=8_000.0,
            capital_expenditure=1_000.0,
            cash_and_equivalents=20_000.0,
            interest_bearing_debt=10_000.0,
            beta=1.0,
            risk_free_rate=0.01,
        )
        multiples = ComparableMultiples(fair_pbr=0.8, fair_ev_ebitda=7.0, fair_p_fcf=10.0)

        report = calculate_valuation_report(metrics, multiples=multiples)
        by_method = {item.method_name: item for item in report.valuations}

        self.assertIsNotNone(by_method["PBR法"].theoretical_price)
        self.assertEqual(by_method["EV/EBITDA法"].unavailable_reason, "sector rule excludes this method")
        self.assertEqual(by_method["P/FCF法"].unavailable_reason, "sector rule excludes this method")

    def test_jquants_summary_normalizes_japan_metrics(self) -> None:
        info_rows = [
            {
                "CompanyName": "テスト株式会社",
                "Sector33CodeName": "情報・通信業",
                "MarketCodeName": "プライム",
            }
        ]
        statements = [
            {
                "DisclosedDate": "2025-05-10",
                "DisclosureNumber": "1",
                "TypeOfCurrentPeriod": "FY",
                "CurrentPeriodEndDate": "2025-03-31",
                "NetSales": "100000",
                "OperatingProfit": "12000",
                "Profit": "8000",
                "EarningsPerShare": "80",
                "CashFlowsFromOperatingActivities": "9000",
                "CashAndEquivalents": "30000",
                "Equity": "50000",
                "TotalAssets": "90000",
                "BookValuePerShare": "500",
                "ResultDividendPerShareAnnual": "20",
                "ForecastEarningsPerShare": "90",
                "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock": "1000",
                "NumberOfTreasuryStockAtTheEndOfFiscalYear": "100",
            }
        ]

        metrics = normalize_jquants_metrics("1234.T", info_rows, statements)

        self.assertEqual(metrics.symbol, "1234.T")
        self.assertEqual(metrics.market, "JP")
        self.assertEqual(metrics.shares_outstanding, 900.0)
        self.assertEqual(metrics.forecast_eps, 90.0)
        self.assertIn("J-Quants:fins/statements", metrics.data_sources)

    def test_sec_companyfacts_normalizes_us_metrics(self) -> None:
        payload = {
            "entityName": "Example Inc.",
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {"shares": [{"val": 100, "end": "2025-12-31", "form": "10-K", "filed": "2026-01-31"}]}
                    }
                },
                "us-gaap": {
                    "Revenues": {
                        "units": {"USD": [{"val": 1000, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-01-31", "end": "2025-12-31"}]}
                    },
                    "OperatingIncomeLoss": {
                        "units": {"USD": [{"val": 120, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-01-31", "end": "2025-12-31"}]}
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"val": 80, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-01-31", "end": "2025-12-31"},
                                {"val": 70, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-01-31", "end": "2024-12-31"},
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"val": 90, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-01-31", "end": "2025-12-31"}]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"val": 20, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-01-31", "end": "2025-12-31"}]}
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {"USD": [{"val": 200, "end": "2025-12-31", "filed": "2026-01-31"}]}
                    },
                    "LongTermDebtNoncurrent": {
                        "units": {"USD": [{"val": 50, "end": "2025-12-31", "filed": "2026-01-31"}]}
                    },
                    "Assets": {
                        "units": {"USD": [{"val": 500, "end": "2025-12-31", "filed": "2026-01-31"}]}
                    },
                    "Liabilities": {
                        "units": {"USD": [{"val": 300, "end": "2025-12-31", "filed": "2026-01-31"}]}
                    },
                    "StockholdersEquity": {
                        "units": {"USD": [{"val": 200, "end": "2025-12-31", "filed": "2026-01-31"}]}
                    },
                },
            },
        }

        metrics = normalize_sec_company_facts("EXM", 123, payload)

        self.assertEqual(metrics.company_name, "Example Inc.")
        self.assertEqual(metrics.revenue, 1000.0)
        self.assertEqual(metrics.free_cash_flow, 70.0)
        self.assertEqual(metrics.shares_outstanding, 100.0)
        self.assertEqual(metrics.net_income_history, (80.0, 70.0))

    def test_rate_parsers_and_metric_merge(self) -> None:
        fred_rate = parse_fred_latest_rate({"observations": [{"date": "2026-01-02", "value": "."}, {"value": "4.25"}]})
        mof_rate = parse_mof_jgb_10y_csv("基準日,1年,10年\n2026/01/01,0.2,1.5\n")
        merged = merge_financial_metrics(
            FinancialMetrics(symbol="AAPL", market="US", revenue=100.0, data_sources=("primary",)),
            FinancialMetrics(symbol="AAPL", market="US", price=10.0, revenue=200.0, data_sources=("supplement",)),
        )

        self.assertEqual(fred_rate, 0.0425)
        self.assertEqual(mof_rate, 0.015)
        self.assertEqual(merged.revenue, 100.0)
        self.assertEqual(merged.price, 10.0)
        self.assertEqual(merged.data_sources, ("primary", "supplement"))


if __name__ == "__main__":
    unittest.main()
