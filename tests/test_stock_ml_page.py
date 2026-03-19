import unittest

from app.ml.stock_page import StockMlPageService


class StockMlPageServiceHelpersTest(unittest.TestCase):
    def test_backtest_summary_cards_follow_focus_result(self) -> None:
        result = {
            "metrics": {
                "gross_cagr_pct": 12.3,
                "cagr_pct": 9.8,
                "sharpe": 1.234,
                "max_drawdown_pct": -6.7,
                "turnover_pct": 123.4,
                "avg_holding_pnl_pct": 0.56,
                "win_rate_pct": 54.3,
                "unable_rate_pct": 1.25,
            }
        }

        cards = StockMlPageService._backtest_summary_cards(
            focus_label="現行採用モデル",
            result=result,
        )

        self.assertEqual(cards[0], {"label": "CAGR (gross)", "value": "12.3%", "sub": "現行採用モデル"})
        self.assertEqual(cards[1], {"label": "CAGR (net)", "value": "9.8%", "sub": "現行採用モデル"})
        self.assertEqual(cards[2], {"label": "Sharpe", "value": "1.23", "sub": "現行採用モデル"})

    def test_ops_summary_cards_report_fail_and_idle_when_no_job_history(self) -> None:
        cards = StockMlPageService._ops_summary_cards(
            monitor_checks=[{"label": "取得件数異常", "value": "25/25"}],
            leakage_ok=False,
            state={"audit_log": []},
            refresh=False,
        )

        self.assertEqual(cards[-2], {"label": "リークチェック", "value": "FAIL"})
        self.assertEqual(cards[-1], {"label": "ジョブ状態", "value": "IDLE"})

    def test_ops_job_status_uses_error_audit_level(self) -> None:
        label = StockMlPageService._ops_job_status_label(
            state={"audit_log": [{"level": "error", "action": "run_inference"}]},
            refresh=False,
        )

        self.assertEqual(label, "FAILED")


if __name__ == "__main__":
    unittest.main()
