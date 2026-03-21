"""Lead-lag page and API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..leadlag import (
    DEFAULT_CFULL_END,
    DEFAULT_CFULL_START,
    DEFAULT_CYCLICAL_SYMBOLS,
    DEFAULT_DEFENSIVE_SYMBOLS,
    DEFAULT_HISTORY_YEARS,
    DEFAULT_JP_SYMBOLS,
    DEFAULT_LAMBDA_REG,
    DEFAULT_N_COMPONENTS,
    DEFAULT_QUANTILE_Q,
    DEFAULT_ROLLING_WINDOW_DAYS,
    DEFAULT_US_SYMBOLS,
    LeadLagAnalysisRequest,
    LeadLagService,
    build_leadlag_config,
)
from ..utils import ok_json_response
from .deps import HubDep

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@router.get("/leadlag-lab", include_in_schema=False)
async def leadlag_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "leadlag_lab.html")


@router.get("/api/leadlag/config")
async def leadlag_config() -> JSONResponse:
    return ok_json_response(
        defaults={
            "us_symbols": list(DEFAULT_US_SYMBOLS),
            "jp_symbols": list(DEFAULT_JP_SYMBOLS),
            "rolling_window_days": DEFAULT_ROLLING_WINDOW_DAYS,
            "lambda_reg": DEFAULT_LAMBDA_REG,
            "n_components": DEFAULT_N_COMPONENTS,
            "quantile_q": DEFAULT_QUANTILE_Q,
            "cfull_start": DEFAULT_CFULL_START,
            "cfull_end": DEFAULT_CFULL_END,
            "history_years": DEFAULT_HISTORY_YEARS,
            "cyclical_symbols": sorted(DEFAULT_CYCLICAL_SYMBOLS),
            "defensive_symbols": sorted(DEFAULT_DEFENSIVE_SYMBOLS),
        }
    )


@router.post("/api/leadlag/analyze")
async def leadlag_analyze(req: LeadLagAnalysisRequest, hub: HubDep) -> JSONResponse:
    try:
        config = build_leadlag_config(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    service = LeadLagService(hub)
    try:
        payload = await service.analyze(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return ok_json_response(**payload)
