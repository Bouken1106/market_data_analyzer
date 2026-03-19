"""Persistent state for the stock ML page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import LOGGER
from ..utils import clone_json_like, read_json_file, utc_now_iso, write_json_file

_PRIMARY_MODEL_VERSION = "lgbm_cls_jp_v1.0.0"
_LEGACY_PRIMARY_MODEL_VERSION = "lgbm_cls_jp_v0.1.0_proxy"


class StockMlPageStore:
    def __init__(self, cache_path: Path, *, max_logs: int = 120) -> None:
        self.cache_path = cache_path
        self.max_logs = max(20, int(max_logs))
        self._state: dict[str, Any] = {
            "adopted_model_version": _PRIMARY_MODEL_VERSION,
            "last_inference_run_at": None,
            "last_training_run_at": None,
            "updated_at": None,
            "audit_log": [],
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

    def _load_from_disk(self) -> None:
        payload = read_json_file(self.cache_path)
        if not isinstance(payload, dict):
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

    def _touch_and_write(self) -> None:
        self._state["updated_at"] = utc_now_iso()
        self._write_to_disk()

    def _write_to_disk(self) -> None:
        try:
            write_json_file(self.cache_path, self._state, compact=True)
        except Exception as exc:
            LOGGER.warning("Failed to write stock ML page state cache: %s", exc)
