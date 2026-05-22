from __future__ import annotations

import asyncio
import unittest

from app.services.valuation_data_sources import (
    merge_financial_metrics,
    normalize_edinet_metrics,
    normalize_jquants_metrics,
    normalize_sec_company_facts,
    parse_fred_latest_rate,
    parse_mof_jgb_10y_csv,
)
from app.services.valuation_service import build_valuation_payload
from app.services.valuation_models import (
    ComparableMultiples,
    FinancialMetrics,
    ValuationAssumptions,
    calculate_valuation_report,
)


class _FakeValuationHub:
    async def security_overview_payload(self, **kwargs):
        return {
            "symbol": kwargs["symbol"],
            "source": "test-overview",
            "price": {"current": 100.0},
            "market": {"beta_60d_vs_spy": 1.1},
        }

    async def fmp_reference_payload(self, symbol: str, *, refresh: bool = False, cache_only: bool = False):
        del refresh, cache_only
        return {
            "symbol": symbol,
            "source": "test-fmp",
            "profile": {
                "company_name": "Example Inc.",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 10_000.0,
                "beta": 1.2,
            },
            "financials": {
                "ratios_ttm": {
                    "pe_ratio_ttm": 20.0,
                    "pb_ratio_ttm": 3.0,
                    "ps_ratio_ttm": 4.0,
                    "roe_ttm": 0.2,
                },
                "key_metrics_ttm": {
                    "eps_ttm": 5.0,
                    "book_value_per_share_ttm": 30.0,
                    "dividend_yield_ttm": 0.01,
                },
                "income_statement_latest": {
                    "date": "2025-12-31",
                    "revenue": 1_000.0,
                    "operating_income": 120.0,
                    "net_income": 500.0,
                    "eps": 5.0,
                },
                "balance_sheet_latest": {
                    "cash_and_short_term_investments": 1_000.0,
                    "total_debt": 500.0,
                    "total_liabilities": 3_000.0,
                    "total_assets": 5_000.0,
                    "total_equity": 2_000.0,
                },
                "cash_flow_latest": {
                    "operating_cash_flow": 800.0,
                    "capital_expenditure": -200.0,
                    "free_cash_flow": 600.0,
                },
            },
        }


class _FakeStore:
    async def get(self, symbol: str):
        return {
            "symbol": symbol,
            "source": "stale-store",
            "profile": {"company_name": "Stale Inc.", "market_cap": 1_000.0},
            "financials": {
                "ratios_ttm": {},
                "key_metrics_ttm": {},
                "income_statement_latest": {"net_income": 10.0, "eps": 1.0},
                "balance_sheet_latest": {},
                "cash_flow_latest": {},
            },
        }


class _FakeRefreshAwareHub(_FakeValuationHub):
    def __init__(self) -> None:
        self.fmp_reference_store = _FakeStore()
        self.calls: list[dict[str, object]] = []

    async def fmp_reference_payload(self, symbol: str, *, refresh: bool = False, cache_only: bool = False):
        self.calls.append({"symbol": symbol, "refresh": refresh, "cache_only": cache_only})
        payload = await super().fmp_reference_payload(symbol, refresh=refresh, cache_only=cache_only)
        payload["profile"]["company_name"] = "Fresh Inc."
        return payload


def _sec_duration_fact(value: float) -> dict[str, object]:
    return {
        "val": value,
        "start": "2025-01-01",
        "end": "2025-12-31",
        "form": "10-K",
        "fp": "FY",
        "fy": 2025,
        "filed": "2026-02-01",
    }


def _sec_instant_fact(value: float) -> dict[str, object]:
    return {
        "val": value,
        "end": "2025-12-31",
        "form": "10-K",
        "fp": "FY",
        "fy": 2025,
        "filed": "2026-02-01",
    }


def _peer_metrics(
    symbol: str,
    *,
    price: float,
    revenue: float,
    operating_income: float,
    net_income: float,
    eps: float,
) -> FinancialMetrics:
    operating_cash_flow = operating_income * 0.8
    capex = 250.0
    return FinancialMetrics(
        symbol=symbol,
        market="US",
        price=price,
        shares_outstanding=100.0,
        revenue=revenue,
        gross_profit=revenue * 0.35,
        operating_income=operating_income,
        ebit=operating_income,
        ebitda=operating_income + 100.0,
        depreciation_and_amortization=100.0,
        net_income=net_income,
        eps=eps,
        operating_cash_flow=operating_cash_flow,
        capital_expenditure=capex,
        free_cash_flow=operating_cash_flow - capex,
        cash_and_equivalents=300.0,
        interest_bearing_debt=1_300.0,
        equity=3_500.0,
        beta=1.0,
        risk_free_rate=0.02,
        revenue_history=(revenue, revenue * 0.99, revenue * 0.98, revenue * 0.97),
        eps_history=(eps, eps * 0.98, eps * 0.97, eps * 0.96),
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
        self.assertEqual(by_method["成長・収益性補正PER法"].theoretical_price, 75.0)
        self.assertAlmostEqual(by_method["成長・収益性補正EV/EBITDA法"].theoretical_price or 0, 86.0)
        self.assertAlmostEqual(by_method["成長・収益性補正EV/Sales法"].theoretical_price or 0, 126.0)
        self.assertEqual(
            by_method["成長・収益性補正PER法"].assumptions["quality_adjustment_source"],
            "neutral_fallback_no_peer_quality",
        )
        self.assertFalse(by_method["成長・収益性補正PER法"].is_standard_candidate)
        self.assertAlmostEqual(by_method["EV/EBITDA法"].theoretical_price or 0, 86.0)
        self.assertIsNone(by_method["予想PER法"].theoretical_price)
        self.assertEqual(by_method["予想PER法"].unavailable_reason, "forecast EPS is missing or non-positive")
        self.assertIsNotNone(by_method["簡易DCF法"].theoretical_price)

    def test_gordon_growth_uses_forecast_dividend_as_next_dividend(self) -> None:
        metrics = FinancialMetrics(
            symbol="DIV",
            market="US",
            price=100.0,
            dividend_per_share=1.0,
            forecast_dividend_per_share=2.0,
            beta=1.0,
            risk_free_rate=0.02,
        )
        assumptions = ValuationAssumptions(dividend_growth_rate=0.03)

        report = calculate_valuation_report(metrics, assumptions=assumptions)
        by_method = {item.method_name: item for item in report.valuations}

        gordon = by_method["ゴードン成長モデル"]
        self.assertAlmostEqual(gordon.theoretical_price or 0, 2.0 / (0.067 - 0.03))
        self.assertEqual(gordon.used_data["next_dividend"], 2.0)
        self.assertEqual(gordon.used_data["dividend_source"], "forecast")

    def test_standard_intrinsic_value_methods_use_quality_growth_roic_and_reverse_dcf(self) -> None:
        metrics = FinancialMetrics(
            symbol="TGT",
            market="US",
            currency="USD",
            price=100.0,
            shares_outstanding=100.0,
            revenue=10_000.0,
            gross_profit=4_500.0,
            operating_income=1_300.0,
            ebit=1_300.0,
            ebitda=1_500.0,
            depreciation_and_amortization=200.0,
            net_income=800.0,
            eps=8.0,
            operating_cash_flow=1_100.0,
            capital_expenditure=300.0,
            free_cash_flow=800.0,
            cash_and_equivalents=500.0,
            short_term_investments=100.0,
            interest_bearing_debt=1_000.0,
            equity=4_200.0,
            total_assets=7_000.0,
            dividends_paid=200.0,
            share_repurchases=100.0,
            beta=1.0,
            risk_free_rate=0.02,
            revenue_history=(10_000.0, 9_800.0, 9_600.0, 9_400.0),
            eps_history=(8.0, 7.8, 7.6, 7.4),
            free_cash_flow_history=(800.0, 760.0, 730.0, 700.0),
            data_sources=("unit",),
        )
        peers = [
            _peer_metrics("P1", price=85.0, revenue=9_000.0, operating_income=900.0, net_income=500.0, eps=5.0),
            _peer_metrics("P2", price=90.0, revenue=9_500.0, operating_income=950.0, net_income=550.0, eps=5.5),
            _peer_metrics("P3", price=95.0, revenue=9_200.0, operating_income=850.0, net_income=480.0, eps=4.8),
        ]

        report = calculate_valuation_report(metrics, peers=peers)
        by_method = {item.method_name: item for item in report.valuations}

        self.assertIsNotNone(by_method["成長・収益性補正PER法"].theoretical_price)
        self.assertIsNotNone(by_method["成長・収益性補正EV/EBITDA法"].theoretical_price)
        self.assertIsNotNone(by_method["成長・収益性補正EV/Sales法"].theoretical_price)
        self.assertIsNotNone(by_method["Justified PER法"].theoretical_price)
        self.assertIsNotNone(by_method["Justified PBR法"].theoretical_price)
        self.assertIsNotNone(by_method["ROIC・再投資率DCF法"].theoretical_price)
        self.assertGreater(by_method["成長・収益性補正PER法"].used_data["adjusted_fair_per"], 0)
        self.assertTrue(by_method["ROIC・再投資率DCF法"].is_standard_candidate)
        self.assertIsNotNone(report.diagnostics["standard_valuation"]["standard_theoretical_price"])
        self.assertTrue(report.diagnostics["reverse_dcf"]["is_calculated"])
        self.assertIn("current_price_implied_fcff_growth_rate", report.diagnostics["reverse_dcf"])

    def test_negative_working_capital_release_does_not_force_decline_growth(self) -> None:
        metrics = FinancialMetrics(
            symbol="WC",
            market="US",
            price=100.0,
            shares_outstanding=100.0,
            revenue=10_000.0,
            operating_income=1_200.0,
            ebit=1_200.0,
            depreciation_and_amortization=200.0,
            net_income=800.0,
            eps=8.0,
            operating_cash_flow=1_200.0,
            capital_expenditure=220.0,
            working_capital_change=-1_000.0,
            cash_and_equivalents=500.0,
            interest_bearing_debt=800.0,
            equity=4_000.0,
            dividends_paid=200.0,
            share_repurchases=200.0,
            beta=1.0,
            risk_free_rate=0.04,
        )

        report = calculate_valuation_report(metrics)
        by_method = {item.method_name: item for item in report.valuations}

        self.assertIsNone(report.metrics["sustainable_growth_rate"])
        self.assertEqual(by_method["Justified PER法"].used_data["growth_rate"], 0.02)
        self.assertEqual(by_method["ROIC・再投資率DCF法"].unavailable_reason, "reinvestment amount or reinvestment rate is missing")

    def test_reverse_dcf_can_solve_forecast_growth_above_wacc(self) -> None:
        metrics = FinancialMetrics(
            symbol="HIGH",
            market="US",
            price=200.0,
            shares_outstanding=100.0,
            free_cash_flow=1_000.0,
            wacc=0.10,
            data_sources=("unit",),
        )

        report = calculate_valuation_report(metrics)
        reverse = report.diagnostics["reverse_dcf"]

        self.assertTrue(reverse["is_calculated"])
        self.assertGreater(reverse["current_price_implied_fcff_growth_rate"], 0.10)
        self.assertEqual(reverse["growth_search_high"], 1.0)

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

    def test_sec_net_borrowing_does_not_treat_stock_buybacks_as_debt_repayment(self) -> None:
        payload = {
            "entityName": "Example Inc.",
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {"shares": [_sec_instant_fact(100.0)]},
                    },
                },
                "us-gaap": {
                    "NetIncomeLoss": {"units": {"USD": [_sec_duration_fact(1000.0)]}},
                    "ProceedsFromIssuanceOfLongTermDebt": {"units": {"USD": [_sec_duration_fact(300.0)]}},
                    "RepaymentsOfLongTermDebt": {"units": {"USD": [_sec_duration_fact(100.0)]}},
                    "PaymentsForRepurchaseOfCommonStock": {"units": {"USD": [_sec_duration_fact(900.0)]}},
                },
            },
        }

        metrics = normalize_sec_company_facts("EXM", 123, payload)

        self.assertEqual(metrics.net_borrowing, 200.0)

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

    def test_edinet_facts_normalize_japan_metrics(self) -> None:
        metrics = normalize_edinet_metrics(
            "7203.T",
            {
                "NetSales": 1000.0,
                "OperatingIncome": 120.0,
                "NetIncome": 80.0,
                "NetCashProvidedByUsedInOperatingActivities": 100.0,
                "PurchaseOfPropertyPlantAndEquipment": -20.0,
                "CashAndCashEquivalents": 200.0,
                "LongTermBorrowings": 50.0,
                "TotalAssets": 500.0,
                "Equity": 220.0,
                "TotalNumberOfIssuedShares": 10.0,
            },
            doc_id="S100TEST",
        )

        self.assertEqual(metrics.symbol, "7203.T")
        self.assertEqual(metrics.market, "JP")
        self.assertEqual(metrics.free_cash_flow, 80.0)
        self.assertEqual(metrics.interest_bearing_debt, 50.0)
        self.assertEqual(metrics.data_sources, ("EDINET:S100TEST:csv",))

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

    def test_build_valuation_payload_normalizes_fmp_reference_for_ui(self) -> None:
        payload = asyncio.run(
            build_valuation_payload(
                _FakeValuationHub(),
                "aapl",
                cache_only=False,
                fair_per=15.0,
                fair_pbr=2.0,
                risk_free_rate=0.04,
            )
        )

        by_method = {item["method_name"]: item for item in payload["valuations"]}
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["current_price"], 100.0)
        self.assertEqual(payload["metrics"]["shares_outstanding"], 100.0)
        self.assertEqual(by_method["実績PER法"]["theoretical_price"], 75.0)
        self.assertEqual(by_method["PBR法"]["theoretical_price"], 60.0)
        self.assertIn("upside_pct", by_method["実績PER法"])
        self.assertGreater(payload["summary"]["calculated_count"], 0)

    def test_build_valuation_payload_uses_reference_service_when_refresh_allowed(self) -> None:
        hub = _FakeRefreshAwareHub()

        payload = asyncio.run(build_valuation_payload(hub, "aapl", cache_only=False))

        self.assertEqual(payload["company_name"], "Fresh Inc.")
        self.assertEqual(payload["assumptions"]["risk_free_rate"], 0.0457)
        self.assertEqual(payload["assumptions"]["equity_risk_premium"], 0.047)
        self.assertEqual(payload["input_status"]["risk_free_rate_source"], "market_default:2026-05-22")
        self.assertEqual(hub.calls[0], {"symbol": "AAPL", "refresh": False, "cache_only": False})


if __name__ == "__main__":
    unittest.main()
