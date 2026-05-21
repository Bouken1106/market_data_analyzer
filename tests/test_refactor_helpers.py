from __future__ import annotations

import unittest

from app.ohlcv import close_values_by_date, point_date_key, point_positive_close
from app.services.market_data_historical_jquants import JQuantsHistoricalClient
from app.services.market_data_math import atr, beta_and_corr, daily_returns, intraday_vwap, moving_average
from app.services.market_data_overview_ops import MarketDataOverviewOps
from app.services.market_data_sparkline import (
    build_daily_sparkline_payload,
    completed_daily_values,
    daily_close_values,
    provider_from_points,
)
from app.services.market_data_queries import MarketDataQueriesMixin
from app.services.market_data_queries_overview_support import (
    OverviewInputs,
    OverviewRequest,
    build_overview_payload,
    build_overview_source_detail,
)
from app.services.market_data_queries_reference import FmpReferenceData
from app.services.paper_portfolio import _apply_position_weights, _build_position_rows, _portfolio_summary
from app.services.portfolio_common import apply_market_value_weights, positive_price_or_none, price_map_from_rows
from app.services.valuation_payload_inputs import (
    ValuationPayloadOptions,
    build_comparable_multiples,
    build_valuation_assumptions,
    resolve_risk_free_rate,
)
from app.services.valuation_payload_metrics import financial_metrics_from_payloads
from app.services.valuation_payload_summary import valuation_summary, valuations_with_upside
from app.services.valuation_numeric import (
    abs_div_optional,
    add_optional,
    clean_finite_dict,
    first_dict,
    first_present,
    median_positive,
    parse_float,
    positive_div,
    positive_float,
    sum_optional,
)
from app.services.watchlist_commentary_parser import commentary_from_json, fallback_commentary
from app.services.watchlist_commentary_prompt import build_watchlist_prompt
from app.services.watchlist_state import normalize_watchlist_namespace


class _OverviewOwner:
    def _try_parse_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _pick_float(self, payload, *keys):
        for key in keys:
            value = payload.get(key) if isinstance(payload, dict) else None
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _best_updated_at(self, quote, _intraday_points, _day_points):
        return quote.get("updated_at") if isinstance(quote, dict) else None

    def _series_source_descriptor(self, points):
        return points[0].get("_src", "unknown") if points else "empty"


class RefactorHelperTest(unittest.TestCase):
    def test_ohlcv_point_helpers_extract_common_shapes(self) -> None:
        point = {"datetime": "2024-01-03 15:00:00", "price": "105.5"}
        self.assertEqual(point_date_key(point), "2024-01-03")
        self.assertEqual(point_positive_close(point), 105.5)
        self.assertEqual(
            close_values_by_date(
                [
                    {"date": "2024-01-02", "close": "100"},
                    point,
                    {"timestamp": "2024-01-04T00:00:00Z", "price": 0},
                ]
            ),
            {"2024-01-02": 100.0, "2024-01-03": 105.5},
        )

    def test_watchlist_namespace_helper_defaults_and_normalizes(self) -> None:
        self.assertEqual(normalize_watchlist_namespace(None), "us")
        self.assertEqual(normalize_watchlist_namespace(" JP "), "jp")

    def test_market_data_math_helpers(self) -> None:
        points = [
            {"c": 10.0, "h": 11.0, "l": 9.0},
            {"c": 12.0, "h": 13.0, "l": 10.0},
            {"c": 14.0, "h": 15.0, "l": 12.0},
        ]
        self.assertEqual(moving_average(points, 2), 13.0)
        self.assertAlmostEqual(atr(points, 2), 3.0)
        self.assertEqual(moving_average([{"c": "10"}, {"c": float("nan")}, {"c": "14"}], 2), 12.0)
        self.assertAlmostEqual(
            atr(
                [
                    {"c": "10.0", "h": "11.0", "l": "9.0"},
                    {"c": "12.0", "h": "13.0", "l": "10.0"},
                    {"c": "14.0", "h": "15.0", "l": "12.0"},
                ],
                2,
            ),
            3.0,
        )
        string_returns = daily_returns(
            [
                {"t": "2024-01-01 15:00:00", "c": "100"},
                {"t": "2024-01-02 15:00:00", "c": "110"},
                {"t": "2024-01-03 15:00:00", "c": "bad"},
            ],
            max_len=5,
        )
        self.assertEqual(set(string_returns), {"2024-01-02"})
        self.assertAlmostEqual(string_returns["2024-01-02"], 0.1)
        self.assertEqual(
            intraday_vwap(
                [
                    {"t": "2024-01-03 09:30:00", "c": "10", "v": "100"},
                    {"t": "2024-01-03 09:31:00", "c": "20", "v": "300"},
                ]
            ),
            17.5,
        )

        left = [{"t": "2024-01-01", "c": 100.0}, {"t": "2024-01-02", "c": 110.0}, {"t": "2024-01-03", "c": 118.0}]
        right = [{"t": "2024-01-01", "c": 50.0}, {"t": "2024-01-02", "c": 60.0}, {"t": "2024-01-03", "c": 63.0}]
        beta, corr = beta_and_corr(left, right, max_len=5, min_overlap=2)
        self.assertIsNotNone(beta)
        self.assertIsNotNone(corr)

    def test_watchlist_commentary_parser_handles_json_and_fallback(self) -> None:
        raw_json = '{"picks":[{"symbol":"AAPL","comment":"強い値動き。"},{"symbol":"NVDA","comment":"出来高が目立つ。"}]}'
        parsed = commentary_from_json(raw_json, ["AAPL", "NVDA", "MSFT"])
        self.assertEqual(parsed, "AAPL: 強い値動き。\nNVDA: 出来高が目立つ。")

        fallback = fallback_commentary("AAPL strong move\nNVDA volume expanding", ["AAPL", "NVDA"])
        self.assertIn("AAPL:", fallback)
        self.assertIn("NVDA:", fallback)

    def test_watchlist_prompt_lists_metrics(self) -> None:
        prompt = build_watchlist_prompt(
            current_date="2026-04-01",
            metrics=[
                {
                    "symbol": "AAPL",
                    "day_change_text": "+1.23%",
                    "return_30d_text": "+5.67%",
                    "volatility_30d_text": "2.34%",
                }
            ],
        )
        self.assertIn("AAPL", prompt)
        self.assertIn("2026-04-01", prompt)

    def test_build_overview_source_detail_summarizes_field_and_chart_sources(self) -> None:
        source_detail = build_overview_source_detail(
            quote={
                "_source_provider": "both",
                "_source_detail": {"bid": "fmp", "ask": "fmp"},
            },
            day_points=[{"_src": "fmp"}],
            m1_points=[{"_src": "twelvedata"}],
            m5_points=[{"_src": "twelvedata"}],
            spy_points=[{"_src": "stooq"}],
            qqq_points=[],
            price_context={
                "current_price_source": "twelvedata",
                "previous_close_source": "fmp",
                "day_open_source": "intraday_1min",
                "day_high_source": "intraday_1min",
                "day_low_source": "intraday_1min",
                "day_volume_source": "daily_series(fmp)",
            },
            series_source_descriptor=lambda points: points[0].get("_src", "empty") if points else "empty",
        )

        self.assertEqual(source_detail["quote_provider"], "both")
        self.assertEqual(source_detail["chart_sources"]["1min"], "twelvedata")
        self.assertEqual(source_detail["chart_sources"]["SPY"], "stooq")
        self.assertEqual(source_detail["fields"]["price.current"], "twelvedata")
        self.assertEqual(source_detail["fields"]["spread.ask"], "fmp")

    def test_build_overview_payload_assembles_sections(self) -> None:
        payload = build_overview_payload(
            request=OverviewRequest(symbol="AAPL", include_intraday=True, include_market=True, include_qqq=False),
            inputs=OverviewInputs(
                quote={"name": "Apple Inc.", "exchange": "NASDAQ"},
                day_points=[{"t": "2024-01-03", "c": 104.0}],
                m1_points=[{"t": "2024-01-03 09:30:00", "c": 103.0}],
                m5_points=[],
                market_context=None,
            ),
            provider="both",
            price_context={
                "current_price": 105.0,
                "previous_close": 100.0,
                "change_abs": 5.0,
                "change_pct": 5.0,
                "day_open": 101.0,
                "day_high": 106.0,
                "day_low": 99.0,
                "gap_abs": 1.0,
                "gap_pct": 1.0,
                "day_volume": 1200.0,
                "avg_volume_20": 1000.0,
                "avg_volume_ratio": 1.2,
                "turnover": 126000.0,
                "bid": 104.9,
                "ask": 105.1,
                "spread_abs": 0.2,
                "spread_pct": 0.19,
            },
            technical={"ma_short_20": 99.0},
            market_payload={"spy": {"symbol": "SPY"}},
            source_detail={"quote_provider": "both"},
            pick_string=lambda payload, *keys: next((payload.get(key) for key in keys if payload.get(key)), None),
            best_updated_at=lambda quote, intraday_points, day_points: "2024-01-03T15:30:00Z",
            delay_note=lambda: "delayed",
        )

        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["name"], "Apple Inc.")
        self.assertEqual(payload["price"]["current"], 105.0)
        self.assertEqual(payload["price"]["updated_at"], "2024-01-03T15:30:00Z")
        self.assertEqual(payload["volume"]["turnover"], 126000.0)
        self.assertEqual(payload["market"]["spy"]["symbol"], "SPY")
        self.assertEqual(payload["source"], "both-live")
        self.assertEqual(payload["source_detail"]["quote_provider"], "both")

    def test_jquants_helpers_build_request_and_extract_values(self) -> None:
        client = JQuantsHistoricalClient(owner=object())

        params = client._build_request_params(
            code="1617",
            start_date="2024-01-01",
            end_date="2024-01-31",
            pagination_key="next",
        )
        values = client._extract_daily_quote_values(
            {
                "dailyBars": [
                    {"Date": "2024-01-05", "Open": 101.0, "High": 102.0, "Low": 100.0, "Close": 101.5, "Volume": 10}
                ]
            }
        )
        normalized = client._normalize_values(values or [])

        self.assertEqual(
            params,
            {"code": "1617", "from": "2024-01-01", "to": "2024-01-31", "pagination_key": "next"},
        )
        self.assertEqual(normalized[0]["t"], "2024-01-05")
        self.assertEqual(normalized[0]["_src"], "jquants")

    def test_paper_portfolio_helpers_build_rows_and_summary(self) -> None:
        positions, total_market_value, total_cost_basis, has_market_value = _build_position_rows(
            {
                "AAPL": {"quantity": 2, "avg_cost": 100},
                "MSFT": {"quantity": 1, "avg_cost": 50},
            },
            {"AAPL": 125.0, "MSFT": 55.0},
        )
        _apply_position_weights(positions, total_market_value)
        summary = _portfolio_summary(
            state={"cash": 700.0, "initial_cash": 1_000.0},
            total_market_value=total_market_value,
            total_cost_basis=total_cost_basis,
            has_market_value=has_market_value,
        )

        self.assertEqual(total_market_value, 305.0)
        self.assertEqual(total_cost_basis, 250.0)
        self.assertTrue(has_market_value)
        self.assertAlmostEqual(positions[0]["weight"], 250.0 / 305.0 * 100.0)
        self.assertAlmostEqual(positions[1]["weight"], 55.0 / 305.0 * 100.0)
        self.assertEqual(summary["equity"], 1_005.0)
        self.assertEqual(summary["unrealized_pnl"], 55.0)

    def test_portfolio_common_helpers_normalize_prices_and_weights(self) -> None:
        rows = [
            {"symbol": " aapl ", "price": "123.45"},
            {"symbol": "MSFT", "price": "not-a-number"},
            {"symbol": "", "price": 99},
            object(),
        ]

        self.assertEqual(positive_price_or_none("0"), None)
        self.assertEqual(price_map_from_rows(rows), {"AAPL": 123.45})
        self.assertEqual(price_map_from_rows(rows, include_missing=True), {"AAPL": 123.45, "MSFT": None})

        weighted = [
            {"market_value": 30.0, "weight": None},
            {"market_value": 70.0, "weight": None},
            {"market_value": None, "weight": None},
        ]
        apply_market_value_weights(weighted, 100.0)

        self.assertEqual(weighted[0]["weight"], 30.0)
        self.assertEqual(weighted[1]["weight"], 70.0)
        self.assertIsNone(weighted[2]["weight"])

    def test_market_data_overview_ops_helpers_build_completed_snapshot(self) -> None:
        ops = MarketDataOverviewOps(owner=_OverviewOwner())
        points = [
            {"t": "2026-04-03", "c": 110.0, "_src": "fmp"},
            {"t": "2026-04-02", "c": 108.0, "_src": "fmp"},
            {"t": "2026-04-01", "c": 106.0, "_src": "fmp"},
        ]

        values = ops._daily_close_values(points)
        completed = ops._completed_daily_values(values, today_iso="2026-04-03")
        snapshot = ops._build_sparkline_snapshot(
            symbol="AAPL",
            points=points,
            completed=completed,
            quote={"close": 111.0, "updated_at": "2026-04-03T00:00:00Z"},
        )

        self.assertEqual(values[0], ("2026-04-03", 110.0))
        self.assertEqual(completed[0], ("2026-04-02", 108.0))
        self.assertEqual(snapshot["latest_close"], 108.0)
        self.assertEqual(snapshot["reference_close"], 106.0)
        self.assertEqual(snapshot["change_abs"], 5.0)
        self.assertEqual(snapshot["trend_30d"], [106.0, 108.0])
        self.assertEqual(snapshot["source"], "fmp")

    def test_sparkline_helpers_normalize_completed_daily_values_and_provider(self) -> None:
        points = [
            {"t": "2026-04-03 15:00:00", "c": 110.0, "_src": "fmp"},
            {"t": "2026-04-02", "c": "108.5", "_src": "fmp"},
            {"t": "2026-04-01", "c": "bad", "_src": "fmp"},
            {"t": "2026-03-31", "c": 106.0, "_src": "fmp"},
        ]

        values = daily_close_values(points, date_only=True)
        completed = completed_daily_values(values, today_iso="2026-04-03")
        payload = build_daily_sparkline_payload(
            symbol="AAPL",
            completed=completed,
            max_points=30,
            current_price=109.0,
            reference_close=None,
            updated_at="2026-04-03T00:00:00Z",
            source="fmp",
            extra_fields={"price_mode": "quote"},
        )

        self.assertEqual(values[0], ("2026-04-03", 110.0))
        self.assertEqual(completed, [("2026-04-02", 108.5), ("2026-03-31", 106.0)])
        self.assertEqual(payload["reference_close"], 106.0)
        self.assertEqual(payload["change_abs"], 3.0)
        self.assertEqual(payload["trend_30d"], [106.0, 108.5])
        self.assertEqual(payload["price_mode"], "quote")
        self.assertEqual(provider_from_points(points, default_provider="cache"), "fmp")

    def test_valuation_payload_helpers_build_inputs_and_metrics(self) -> None:
        options = ValuationPayloadOptions(fair_per=12.0, fair_pbr=-1.0, equity_risk_premium=0.08, forecast_years=99)
        multiples = build_comparable_multiples(options)
        assumptions = build_valuation_assumptions(options)
        metrics = financial_metrics_from_payloads(
            "AAPL",
            market="US",
            overview_payload={"source": "overview", "price": {"current": 100.0}, "market": {"beta_60d_vs_spy": 1.1}},
            fmp_payload={
                "source": "fmp",
                "profile": {"company_name": "Example Inc.", "market_cap": 10_000.0},
                "financials": {
                    "ratios_ttm": {"pe_ratio_ttm": 20.0},
                    "key_metrics_ttm": {"eps_ttm": 5.0, "dividend_yield_ttm": 0.01},
                    "income_statement_latest": {"revenue": 1_000.0, "net_income": 500.0},
                    "balance_sheet_latest": {"cash_and_short_term_investments": 1_000.0, "total_debt": 500.0},
                    "cash_flow_latest": {"capital_expenditure": -200.0, "free_cash_flow": 600.0},
                },
            },
            risk_free_rate=resolve_risk_free_rate("US", None),
        )

        self.assertEqual(multiples.fair_per, 12.0)
        self.assertEqual(multiples.fair_pbr, 2.0)
        self.assertEqual(assumptions.forecast_years, 20)
        self.assertEqual(metrics.shares_outstanding, 100.0)
        self.assertEqual(metrics.dividend_per_share, 1.0)
        self.assertEqual(metrics.beta, 1.1)

        valuations = valuations_with_upside(
            [{"method_name": "A", "theoretical_price": 120.0}, {"method_name": "B", "theoretical_price": None}],
            current_price=100.0,
        )
        summary = valuation_summary(valuations, current_price=100.0)

        self.assertAlmostEqual(valuations[0]["upside_pct"], 20.0)
        self.assertIsNone(valuations[1]["upside_pct"])
        self.assertEqual(summary["calculated_count"], 1)
        self.assertEqual(summary["median_price"], 120.0)

    def test_valuation_numeric_helpers_parse_financial_payload_values(self) -> None:
        self.assertEqual(parse_float("1,234.5"), 1234.5)
        self.assertIsNone(parse_float("NaN"))
        self.assertIsNone(positive_float("0"))
        self.assertEqual(add_optional("1", None), 1.0)
        self.assertEqual(abs_div_optional("-10", "2"), 5.0)
        self.assertEqual(positive_div("10", "2"), 5.0)
        self.assertEqual(sum_optional("1", None, "2.5"), 3.5)
        self.assertEqual(clean_finite_dict({"a": 1.0, "b": float("nan"), "c": None}), {"a": 1.0})
        self.assertEqual(median_positive([None, 5, 1, 3]), 3.0)
        self.assertEqual(first_present(None, "", {}, 0, "fallback"), 0)
        self.assertEqual(first_dict({"data": [{"value": 1}]}), {"value": 1})

    def test_fmp_reference_payload_builders_shape_sections(self) -> None:
        queries = MarketDataQueriesMixin()
        data = FmpReferenceData(
            profile={
                "companyName": "Example Inc.",
                "exchangeShortName": "NASDAQ",
                "sector": "Technology",
                "mktCap": "123456",
                "beta": "1.25",
            },
            ratios={"peRatioTTM": "20.5", "returnOnEquityTTM": "0.31"},
            metrics={"epsTTM": "5.25", "dividendYieldTTM": "0.01"},
            income={"date": "2025-12-31", "revenue": "1000", "netIncome": "200"},
            balance_sheet={"date": "2025-12-31", "totalAssets": "5000", "totalDebt": "700"},
            cash_flow={"date": "2025-12-31", "operatingCashFlow": "300", "freeCashFlow": "250"},
            historical=[
                {"date": "2025-01-02", "close": "100", "adjClose": "99", "volume": "1000"},
                {"date": "2025-01-03", "close": "110", "adjustedClose": "108.9", "volume": "1200"},
            ],
            dividends=[{"date": "2025-01-15", "dividend": "0.25", "adjDividend": "0.24"}],
            splits=[{"date": "2025-02-01", "numerator": "2", "denominator": "1"}],
        )

        payload = queries._build_fmp_reference_payload(symbol="EXM", data=data)

        self.assertEqual(payload["symbol"], "EXM")
        self.assertEqual(payload["profile"]["company_name"], "Example Inc.")
        self.assertEqual(payload["profile"]["market_cap"], 123456.0)
        self.assertEqual(payload["financials"]["ratios_ttm"]["pe_ratio_ttm"], 20.5)
        self.assertEqual(payload["financials"]["key_metrics_ttm"]["eps_ttm"], 5.25)
        self.assertEqual(payload["financials"]["income_statement_latest"]["net_income"], 200.0)
        self.assertEqual(payload["adjusted_prices"]["latest_adjustment_factor"], 108.9 / 110.0)
        self.assertEqual(payload["corporate_actions"]["dividends"][0]["dividend"], 0.25)
        self.assertEqual(payload["corporate_actions"]["splits"][0]["numerator"], 2.0)


if __name__ == "__main__":
    unittest.main()
