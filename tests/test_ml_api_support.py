import asyncio
import unittest
from unittest.mock import AsyncMock

from app.api.ml_support import (
    StockMlPageContext,
    build_job_response_payload,
    prediction_daily_payload,
    training_job_status_payload,
)
from app.models import StockMlPageQueryRequest


class _FakeHub:
    def __init__(self) -> None:
        self.full_daily_history_store = object()


class MlApiSupportTest(unittest.TestCase):
    def test_build_job_response_payload_keeps_raw_status_and_adds_status_code(self) -> None:
        payload = {
            "job_id": "job-1",
            "kind": "quantile_lstm",
            "symbol": "AAPL",
            "status": "completed",
            "progress": 100,
            "message": "done",
            "result": {"metric": 1.23},
            "error": None,
            "error_detail": {"stage_name": "train", "error_code": None, "retryable": False},
            "created_at": "2026-03-26T00:00:00+00:00",
            "updated_at": "2026-03-26T00:01:00+00:00",
        }

        response = build_job_response_payload(payload, include_status_code=True)

        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["status_raw"], "completed")
        self.assertEqual(response["status_code"], "SUCCEEDED")
        self.assertEqual(response["stage_name"], "train")

    def test_training_job_status_payload_normalizes_status_and_extracts_sections(self) -> None:
        payload = {
            "job_id": "job-2",
            "status": "running",
            "progress": 45,
            "message": "training",
            "result": {
                "metrics": [{"label": "roc_auc", "value": 0.61}],
                "summary": [{"label": "best_model", "value": "lgbm"}],
                "folds": [{"fold": 1, "score": 0.6}],
                "logs": [{"stage": "train", "status": "RUNNING"}],
            },
            "error": None,
            "error_detail": {"stage_name": "train", "error_code": None, "retryable": False},
            "updated_at": "2026-03-26T00:01:00+00:00",
        }

        response = training_job_status_payload(payload)

        self.assertEqual(response["status"], "RUNNING")
        self.assertEqual(response["status_raw"], "running")
        self.assertEqual(response["metrics"][0]["label"], "roc_auc")
        self.assertEqual(response["summary"][0]["value"], "lgbm")
        self.assertEqual(response["folds"][0]["fold"], 1)
        self.assertEqual(response["logs"][0]["stage"], "train")

    def test_prediction_daily_payload_uses_summary_card_model_version_fallback(self) -> None:
        snapshot = {
            "dashboard": {
                "prediction_date": "2026-03-18",
                "target_date": "2026-03-19",
                "summary_cards": [{"label": "model_version", "value": "lgbm_cls_jp_v1.1.0"}],
                "feature_version": "base_v1",
                "data_version": "stooq_jp_20260318",
                "rows": [
                    {
                        "code": "7203",
                        "company_name": "トヨタ自動車",
                        "score_cls": 0.42,
                        "prob_up": 0.61,
                        "score_rank": 3,
                        "expected_return": 0.012,
                        "sector33_code": "3700",
                        "warnings": [],
                    }
                ],
            },
            "filters": {"model_family": "LightGBM Classifier"},
            "models": {"default_versions": {"LightGBM Classifier": "lgbm_cls_jp_v1.0.0"}},
        }

        payload = prediction_daily_payload(snapshot)

        self.assertEqual(payload["model_version"], "lgbm_cls_jp_v1.1.0")
        self.assertEqual(payload["rows"][0]["model_version"], "lgbm_cls_jp_v1.1.0")
        self.assertEqual(payload["rows"][0]["code"], "7203")

    def test_stock_ml_page_context_passes_stock_page_params_to_service_method(self) -> None:
        request = StockMlPageQueryRequest(
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
        context = StockMlPageContext.from_request(
            hub=_FakeHub(),
            stock_ml_page_store=object(),
            request=request,
        )
        context.service.build_snapshot = AsyncMock(return_value={"dashboard": {"rows": []}})

        result = asyncio.run(context.snapshot())

        self.assertEqual(result, {"dashboard": {"rows": []}})
        context.service.build_snapshot.assert_awaited_once_with(**request.stock_page_kwargs())


if __name__ == "__main__":
    unittest.main()
