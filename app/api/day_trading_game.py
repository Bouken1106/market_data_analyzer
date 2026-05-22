"""Day-trading game API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..services.day_trading_game import (
    DayTradingGameDataError,
    DayTradingGameDependencyError,
    DayTradingGameRequestError,
    build_day_trading_session,
)
from ..utils import ok_json_response

router = APIRouter()


@router.get("/api/day-trading-game/session")
async def day_trading_game_session(
    market: str = "us",
    mode: str = "intraday",
    symbol: str | None = None,
) -> JSONResponse:
    try:
        payload = await build_day_trading_session(market=market, mode=mode, symbol=symbol)
    except DayTradingGameRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except DayTradingGameDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except DayTradingGameDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return ok_json_response(**payload)
