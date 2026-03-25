import unittest

from app.models import StockMlPageActionRequest, StockMlPageQueryRequest
from app.static_pages import HISTORICAL_PAGE_PATH_PREFIX, STATIC_PAGES, is_no_cache_path
from app.stock_ml_page_params import StockMlPageParams


class StockMlPageRequestHelpersTest(unittest.TestCase):
    def test_query_request_builds_snapshot_kwargs_from_shared_fields(self) -> None:
        req = StockMlPageQueryRequest(
            prediction_date="2026-03-18",
            universe_filter="jp_large_cap_stooq_v1",
            model_family="LightGBM Classifier",
            feature_set="base_v1",
            cost_buffer=0.002,
            train_window_months=24,
            gap_days=3,
            valid_window_months=2,
            random_seed=7,
            train_note="train memo",
            run_note="run memo",
            refresh=True,
        )

        self.assertEqual(
            req.stock_page_kwargs(),
            StockMlPageParams(
                prediction_date="2026-03-18",
                universe_filter="jp_large_cap_stooq_v1",
                model_family="LightGBM Classifier",
                feature_set="base_v1",
                cost_buffer=0.002,
                train_window_months=24,
                gap_days=3,
                valid_window_months=2,
                random_seed=7,
                train_note="train memo",
                run_note="run memo",
                refresh=True,
            ).service_kwargs(),
        )

    def test_action_request_builds_snapshot_kwargs_without_action_only_fields(self) -> None:
        req = StockMlPageActionRequest(
            prediction_date="2026-03-18",
            universe_filter="jp_large_cap_stooq_v1",
            model_family="LightGBM Classifier",
            feature_set="base_v1",
            cost_buffer=0.002,
            train_window_months=24,
            gap_days=3,
            valid_window_months=2,
            random_seed=7,
            train_note="train memo",
            run_note="run memo",
            search_query="7203",
            confirm_regenerate=True,
            refresh=True,
        )

        self.assertEqual(
            req.stock_page_kwargs(),
            StockMlPageParams(
                prediction_date="2026-03-18",
                universe_filter="jp_large_cap_stooq_v1",
                model_family="LightGBM Classifier",
                feature_set="base_v1",
                cost_buffer=0.002,
                train_window_months=24,
                gap_days=3,
                valid_window_months=2,
                random_seed=7,
                train_note="train memo",
                run_note="run memo",
                refresh=True,
            ).service_kwargs(),
        )

    def test_stock_page_params_from_mapping_ignores_action_only_fields(self) -> None:
        params = StockMlPageParams.from_mapping(
            {
                "prediction_date": "2026-03-18",
                "universe_filter": "jp_large_cap_stooq_v1",
                "model_family": "LightGBM Classifier",
                "feature_set": "base_v1",
                "cost_buffer": 0.002,
                "train_window_months": 24,
                "gap_days": 3,
                "valid_window_months": 2,
                "random_seed": 7,
                "train_note": "train memo",
                "run_note": "run memo",
                "refresh": True,
                "search_query": "7203",
                "confirm_regenerate": True,
            }
        )

        self.assertEqual(
            params,
            StockMlPageParams(
                prediction_date="2026-03-18",
                universe_filter="jp_large_cap_stooq_v1",
                model_family="LightGBM Classifier",
                feature_set="base_v1",
                cost_buffer=0.002,
                train_window_months=24,
                gap_days=3,
                valid_window_months=2,
                random_seed=7,
                train_note="train memo",
                run_note="run memo",
                refresh=True,
            ),
        )

    def test_action_request_config_hash_ignores_notes_and_action_flags(self) -> None:
        base_kwargs = {
            "prediction_date": "2026-03-18",
            "universe_filter": "jp_large_cap_stooq_v1",
            "model_family": "LightGBM Classifier",
            "feature_set": "base_v1",
            "cost_buffer": 0.0,
            "train_window_months": 12,
            "gap_days": 5,
            "valid_window_months": 1,
            "random_seed": 42,
        }

        first = StockMlPageActionRequest(
            **base_kwargs,
            train_note="first note",
            run_note="first run",
            search_query="7203",
            confirm_regenerate=True,
            refresh=True,
        )
        second = StockMlPageActionRequest(
            **base_kwargs,
            train_note="second note",
            run_note="second run",
            search_query="6758",
            confirm_regenerate=False,
            refresh=False,
        )

        self.assertEqual(first.stock_page_config_hash(), second.stock_page_config_hash())


class StaticPagesHelperTest(unittest.TestCase):
    def test_static_pages_and_historical_page_are_no_cache(self) -> None:
        self.assertTrue(all(is_no_cache_path(page.route_path) for page in STATIC_PAGES))
        self.assertTrue(is_no_cache_path(f"{HISTORICAL_PAGE_PATH_PREFIX}AAPL"))
        self.assertFalse(is_no_cache_path("/api/snapshot"))


if __name__ == "__main__":
    unittest.main()
