import tempfile
import unittest
from pathlib import Path

from app.ml.stock_page import StockMlPageService
from app.stores.stock_ml_page import StockMlPageStore


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

        self.assertEqual(cards[0]["label"], "CAGR (gross)")
        self.assertEqual(cards[0]["value"], "12.3%")
        self.assertEqual(cards[0]["sub"], "現行採用モデル")
        self.assertIn("help", cards[0])
        self.assertEqual(cards[1]["label"], "CAGR (net)")
        self.assertEqual(cards[1]["value"], "9.8%")
        self.assertEqual(cards[1]["sub"], "現行採用モデル")
        self.assertIn("help", cards[1])
        self.assertEqual(cards[2]["label"], "Sharpe")
        self.assertEqual(cards[2]["value"], "1.23")
        self.assertEqual(cards[2]["sub"], "現行採用モデル")
        self.assertIn("help", cards[2])

    def test_ops_summary_cards_report_fail_and_idle_when_no_job_history(self) -> None:
        cards = StockMlPageService._ops_summary_cards(
            monitor_checks=[{"label": "取得件数異常", "value": "25/25"}],
            leakage_ok=False,
            state={"audit_log": []},
            refresh=False,
        )

        self.assertEqual(cards[-2]["label"], "リークチェック")
        self.assertEqual(cards[-2]["value"], "FAIL")
        self.assertIn("help", cards[-2])
        self.assertEqual(cards[-1]["label"], "ジョブ状態")
        self.assertEqual(cards[-1]["value"], "IDLE")
        self.assertIn("help", cards[-1])

    def test_ops_summary_cards_include_help_for_coverage_monitor(self) -> None:
        cards = StockMlPageService._ops_summary_cards(
            monitor_checks=[{"label": "銘柄数急減", "value": "22/25"}],
            leakage_ok=True,
            state={"audit_log": []},
            refresh=False,
        )

        self.assertEqual(cards[0]["label"], "銘柄数急減")
        self.assertTrue(cards[0]["help"])

    def test_ops_job_status_uses_error_audit_level(self) -> None:
        label = StockMlPageService._ops_job_status_label(
            state={"audit_log": [{"level": "error", "action": "run_inference"}]},
            refresh=False,
        )

        self.assertEqual(label, "FAILED")

    def test_backtest_exceptions_include_multiple_exception_types(self) -> None:
        exceptions = StockMlPageService._build_backtest_exceptions(
            dates=["2026-03-16", "2026-03-17"],
            by_date={
                "2026-03-16": [
                    {"code": "7203", "range_pct": 0.12, "ret_1d": 0.09, "volume_ratio_20": 1.10},
                    {"code": "6758", "range_pct": 0.11, "ret_1d": -0.09, "volume_ratio_20": 1.00},
                ],
                "2026-03-17": [
                    {"code": "9984", "range_pct": 0.001, "ret_1d": 0.00, "volume_ratio_20": 0.01},
                ],
            },
            excluded_reason_breakdown=[
                {"label": "取得失敗", "count": 2, "detail": "Stooq 取得失敗またはキャッシュ未整備。 例: 8306, 9432"}
            ],
        )

        self.assertEqual([item["type"] for item in exceptions[:4]], ["ストップ高疑い", "ストップ安疑い", "売買停止疑い", "データ欠損除外"])

    def test_cross_sectional_backtest_uses_intraday_return_and_excludes_unable_names(self) -> None:
        service = StockMlPageService(full_daily_history_store=None, page_store=None)

        result = service._run_cross_sectional_backtest(
            ["2026-03-16"],
            {
                "2026-03-16": [
                    {
                        "code": "7203",
                        "lgbm_score": 2.0,
                        "lgbm_prob": 0.8,
                        "next_return": 0.20,
                        "next_intraday_return": 0.05,
                        "range_pct": 0.02,
                        "volume_ratio_20": 1.10,
                    },
                    {
                        "code": "6758",
                        "lgbm_score": 1.0,
                        "lgbm_prob": 0.7,
                        "next_return": 0.30,
                        "next_intraday_return": 0.10,
                        "range_pct": 0.12,
                        "volume_ratio_20": 1.00,
                    },
                ]
            },
            model_kind="lgbm",
        )

        self.assertAlmostEqual(result["daily_series"][0]["gross_return"], 0.05)
        self.assertAlmostEqual(result["daily_series"][0]["net_return"], 0.049)
        self.assertEqual(result["metrics"]["unable_count"], 1)
        self.assertAlmostEqual(result["metrics"]["unable_rate_pct"], 50.0)

    def test_cross_sectional_backtest_does_not_charge_cost_when_every_pick_is_unable(self) -> None:
        service = StockMlPageService(full_daily_history_store=None, page_store=None)

        result = service._run_cross_sectional_backtest(
            ["2026-03-16"],
            {
                "2026-03-16": [
                    {
                        "code": "9984",
                        "lgbm_score": 2.0,
                        "lgbm_prob": 0.9,
                        "next_return": 0.25,
                        "next_intraday_return": 0.08,
                        "range_pct": 0.15,
                        "volume_ratio_20": 1.20,
                    },
                ]
            },
            model_kind="lgbm",
        )

        self.assertAlmostEqual(result["daily_series"][0]["gross_return"], 0.0)
        self.assertAlmostEqual(result["daily_series"][0]["net_return"], 0.0)
        self.assertEqual(result["metrics"]["unable_count"], 1)


class StockMlPageStoreTest(unittest.TestCase):
    def test_add_audit_log_preserves_structured_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StockMlPageStore(Path(tmpdir) / "stock_ml_page_state.json")
            store.add_audit_log(
                action="adopt_model",
                detail="Adopted model changed a -> b.",
                level="warning",
                actor="role:admin",
                config_hash="abc123def456",
                job_kind="adopt_model",
                settings={"prediction_date": "2026-03-17", "model_family": "LightGBM Classifier"},
                before_model_version="model_a",
                after_model_version="model_b",
                compare_metrics={"before_summary": "ROC-AUC 0.51", "after_summary": "ROC-AUC 0.58"},
            )

            entry = store.get_state()["audit_log"][0]
            self.assertEqual(entry["actor"], "role:admin")
            self.assertEqual(entry["config_hash"], "abc123def456")
            self.assertEqual(entry["job_kind"], "adopt_model")
            self.assertEqual(entry["before_model_version"], "model_a")
            self.assertEqual(entry["after_model_version"], "model_b")
            self.assertEqual(entry["settings"]["prediction_date"], "2026-03-17")
            self.assertEqual(entry["compare_metrics"]["after_summary"], "ROC-AUC 0.58")


if __name__ == "__main__":
    unittest.main()
