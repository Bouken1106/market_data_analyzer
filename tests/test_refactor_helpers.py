from __future__ import annotations

import unittest

from app.services.market_data_math import atr, beta_and_corr, moving_average
from app.services.watchlist_commentary_parser import commentary_from_json, fallback_commentary
from app.services.watchlist_commentary_prompt import build_watchlist_prompt


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


if __name__ == "__main__":
    unittest.main()
