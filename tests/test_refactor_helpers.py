from __future__ import annotations

import unittest

from app.services.market_data_historical_jquants import JQuantsHistoricalClient
from app.services.market_data_math import atr, beta_and_corr, moving_average
from app.services.market_data_overview_ops import MarketDataOverviewOps
from app.services.market_data_queries_overview_support import (
    OverviewInputs,
    OverviewRequest,
    build_overview_payload,
    build_overview_source_detail,
)
from app.services.paper_portfolio import _apply_position_weights, _build_position_rows, _portfolio_summary
from app.services.watchlist_commentary_parser import commentary_from_json, fallback_commentary
from app.services.watchlist_commentary_prompt import build_watchlist_prompt


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
    def test_market_data_math_helpers(self) -> None:
        points = [
            {"c": 10.0, "h": 11.0, "l": 9.0},
            {"c": 12.0, "h": 13.0, "l": 10.0},
            {"c": 14.0, "h": 15.0, "l": 12.0},
        ]
        self.assertEqual(moving_average(points, 2), 13.0)
        self.assertAlmostEqual(atr(points, 2), 3.0)

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


if __name__ == "__main__":
    unittest.main()
