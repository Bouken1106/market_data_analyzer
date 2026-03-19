"""Thread-safe store for tracking ML training jobs."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any


class MlJobStore:
    def __init__(self, max_jobs: int = 100) -> None:
        self.max_jobs = max(20, max_jobs)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._terminal_statuses = frozenset({"completed", "failed", "cancelled"})

    def create(self, kind: str, symbol: str) -> str:
        job_id = uuid.uuid4().hex
        now_iso = datetime.now(timezone.utc).isoformat()
        payload = {
            "job_id": job_id,
            "kind": kind,
            "symbol": symbol.upper().strip(),
            "status": "queued",
            "progress": 0,
            "message": "ジョブを作成しました。",
            "result": None,
            "error": None,
            "error_detail": None,
            "cancel_requested": False,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        with self._lock:
            self._jobs[job_id] = payload
            self._trim_no_lock()
        return job_id

    def update(self, job_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return None
            item.update(changes)
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            return dict(item)

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return
            if item.get("cancel_requested"):
                item.update(
                    status="cancelled",
                    progress=0,
                    message="停止しました。",
                    result=None,
                    error=None,
                    error_detail=None,
                )
            else:
                item.update(
                    status="completed",
                    progress=100,
                    message="完了しました。",
                    result=result,
                    error=None,
                    error_detail=None,
                )
            item["updated_at"] = datetime.now(timezone.utc).isoformat()

    def fail(
        self,
        job_id: str,
        error: str,
        *,
        error_detail: dict[str, Any] | None = None,
        message: str = "失敗しました。",
    ) -> None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return
            if item.get("cancel_requested"):
                item.update(
                    status="cancelled",
                    progress=0,
                    message="停止しました。",
                    error=None,
                    error_detail=None,
                    result=None,
                )
            else:
                item.update(
                    status="failed",
                    message=message,
                    error=error,
                    error_detail=error_detail,
                    result=None,
                )
            item["updated_at"] = datetime.now(timezone.utc).isoformat()

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return False
            return bool(item.get("cancel_requested"))

    def request_cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return None
            status = str(item.get("status") or "")
            if status in self._terminal_statuses:
                return dict(item)

            item["cancel_requested"] = True
            if status == "queued":
                item.update(
                    status="cancelled",
                    progress=0,
                    message="停止しました。",
                    result=None,
                    error=None,
                    error_detail=None,
                )
            else:
                item.update(
                    status="cancelling",
                    progress=0,
                    message="停止を要求しました。",
                    result=None,
                    error=None,
                    error_detail=None,
                )
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            return dict(item)

    def mark_cancelled(self, job_id: str, message: str = "停止しました。") -> None:
        self.update(
            job_id,
            status="cancelled",
            progress=0,
            message=message,
            result=None,
            error=None,
            error_detail=None,
            cancel_requested=True,
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            if not item:
                return None
            return dict(item)

    def _trim_no_lock(self) -> None:
        if len(self._jobs) <= self.max_jobs:
            return
        sorted_items = sorted(self._jobs.items(), key=lambda pair: pair[1].get("updated_at", ""))
        remove_count = len(self._jobs) - self.max_jobs
        for idx in range(remove_count):
            self._jobs.pop(sorted_items[idx][0], None)
