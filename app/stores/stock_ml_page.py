"""Persistent state for the stock ML page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import clone_json_like, utc_now_iso
from .json_state import JsonStateStore

_PRIMARY_MODEL_VERSION = "lgbm_cls_jp_v1.0.0"
_LEGACY_PRIMARY_MODEL_VERSION = "lgbm_cls_jp_v0.1.0_proxy"


class StockMlPageStore(JsonStateStore):
    def __init__(self, cache_path: Path, *, max_logs: int = 120) -> None:
        super().__init__(cache_path, log_label="stock ML page state cache", compact=True)
        self.max_logs = max(20, int(max_logs))
        self._state: dict[str, Any] = {
            "adopted_model_version": _PRIMARY_MODEL_VERSION,
            "last_inference_run_at": None,
            "last_training_run_at": None,
            "updated_at": None,
            "audit_log": [],
            "prediction_runs": [],
        }
        self._load_from_disk()

    def get_state(self) -> dict[str, Any]:
        return clone_json_like(self._state)

    def get_adopted_model_version(self) -> str:
        value = str(self._state.get("adopted_model_version") or "").strip()
        if value == _LEGACY_PRIMARY_MODEL_VERSION:
            return _PRIMARY_MODEL_VERSION
        return value or _PRIMARY_MODEL_VERSION

    def set_adopted_model_version(self, model_version: str) -> None:
        value = str(model_version or "").strip()
        if not value:
            return
        self._state["adopted_model_version"] = value
        self._touch_and_write()

    def mark_inference_run(self) -> None:
        self._state["last_inference_run_at"] = utc_now_iso()
        self._touch_and_write()

    def mark_training_run(self) -> None:
        self._state["last_training_run_at"] = utc_now_iso()
        self._touch_and_write()

    def add_audit_log(self, *, action: str, detail: str, level: str = "normal") -> None:
        timestamp = utc_now_iso()
        logs = self._state.setdefault("audit_log", [])
        if not isinstance(logs, list):
            logs = []
            self._state["audit_log"] = logs
        logs.insert(
            0,
            {
                "time": timestamp,
                "action": str(action or "").strip() or "unknown",
                "detail": str(detail or "").strip(),
                "level": str(level or "normal").strip() or "normal",
            },
        )
        del logs[self.max_logs:]
        self._touch_and_write()

    def find_prediction_run(self, *, generation_key: str) -> dict[str, str] | None:
        normalized_key = str(generation_key or "").strip()
        if not normalized_key:
            return None
        runs = self._state.get("prediction_runs")
        if not isinstance(runs, list):
            return None
        for item in runs:
            if not isinstance(item, dict):
                continue
            if str(item.get("generation_key") or "").strip() != normalized_key:
                continue
            return clone_json_like(item)
        return None

    def record_prediction_run(
        self,
        *,
        generation_key: str,
        prediction_date: str,
        target_date: str,
        model_version: str,
        feature_version: str,
        data_version: str,
        config_hash: str,
    ) -> None:
        normalized_key = str(generation_key or "").strip()
        if not normalized_key:
            return
        runs = self._state.setdefault("prediction_runs", [])
        if not isinstance(runs, list):
            runs = []
            self._state["prediction_runs"] = runs
        cleaned_runs = [
            item
            for item in runs
            if isinstance(item, dict) and str(item.get("generation_key") or "").strip() != normalized_key
        ]
        cleaned_runs.insert(
            0,
            {
                "generation_key": normalized_key,
                "prediction_date": str(prediction_date or "").strip(),
                "target_date": str(target_date or "").strip(),
                "model_version": str(model_version or "").strip(),
                "feature_version": str(feature_version or "").strip(),
                "data_version": str(data_version or "").strip(),
                "config_hash": str(config_hash or "").strip(),
                "generated_at": utc_now_iso(),
            },
        )
        self._state["prediction_runs"] = cleaned_runs[: self.max_logs]
        self._touch_and_write()

    def _load_from_disk(self) -> None:
        payload = self._read_state_dict()
        if payload is None:
            return
        adopted_model_version = str(payload.get("adopted_model_version") or "").strip()
        if adopted_model_version:
            self._state["adopted_model_version"] = (
                _PRIMARY_MODEL_VERSION if adopted_model_version == _LEGACY_PRIMARY_MODEL_VERSION else adopted_model_version
            )
        for key in ("last_inference_run_at", "last_training_run_at", "updated_at"):
            value = payload.get(key)
            if isinstance(value, str):
                self._state[key] = value
        logs = payload.get("audit_log")
        if isinstance(logs, list):
            cleaned: list[dict[str, str]] = []
            for item in logs[: self.max_logs]:
                if not isinstance(item, dict):
                    continue
                cleaned.append(
                    {
                        "time": str(item.get("time") or "").strip(),
                        "action": str(item.get("action") or "").strip(),
                        "detail": str(item.get("detail") or "").strip(),
                        "level": str(item.get("level") or "normal").strip() or "normal",
                    }
                )
            self._state["audit_log"] = cleaned
        runs = payload.get("prediction_runs")
        if isinstance(runs, list):
            cleaned_runs: list[dict[str, str]] = []
            for item in runs[: self.max_logs]:
                if not isinstance(item, dict):
                    continue
                generation_key = str(item.get("generation_key") or "").strip()
                if not generation_key:
                    continue
                cleaned_runs.append(
                    {
                        "generation_key": generation_key,
                        "prediction_date": str(item.get("prediction_date") or "").strip(),
                        "target_date": str(item.get("target_date") or "").strip(),
                        "model_version": str(item.get("model_version") or "").strip(),
                        "feature_version": str(item.get("feature_version") or "").strip(),
                        "data_version": str(item.get("data_version") or "").strip(),
                        "config_hash": str(item.get("config_hash") or "").strip(),
                        "generated_at": str(item.get("generated_at") or "").strip(),
                    }
                )
            self._state["prediction_runs"] = cleaned_runs

    def _touch_and_write(self) -> None:
        self._touch_and_write_state(self._state)
