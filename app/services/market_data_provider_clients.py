"""HTTP client adapters for external market-data providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class JsonApiResponse:
    response: httpx.Response
    payload: Any


class TwelveDataClient:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = str(api_key or "").strip()

    async def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> JsonApiResponse:
        request_params = dict(params or {})
        request_params["apikey"] = self._api_key
        response = await self._client.get(url, params=request_params)
        return JsonApiResponse(response=response, payload=response.json())

    async def get_price(self, url: str, *, symbol: str) -> JsonApiResponse:
        return await self.get_json(url, params={"symbol": symbol})

    async def get_quote(self, url: str, *, symbol: str) -> JsonApiResponse:
        return await self.get_json(url, params={"symbol": symbol})

    async def get_time_series(
        self,
        url: str,
        *,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> JsonApiResponse:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "order": "ASC",
            "outputsize": outputsize,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self.get_json(url, params=params)

    async def get_earliest_timestamp(
        self,
        url: str,
        *,
        symbol: str,
        interval: str,
    ) -> JsonApiResponse:
        return await self.get_json(url, params={"symbol": symbol, "interval": interval})

    async def get_symbol_catalog(self, url: str, *, country: str) -> JsonApiResponse:
        return await self.get_json(url, params={"country": country})


class FmpClient:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = str(api_key or "").strip()

    async def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        request_params = dict(params or {})
        request_params["apikey"] = self._api_key
        response = await self._client.get(url, params=request_params)
        return response.json()

    async def get_quote(self, url: str, *, symbol: str) -> Any:
        return await self.get_json(url, params={"symbol": symbol})

    async def get_historical_eod(
        self,
        url: str,
        *,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {"symbol": symbol}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        return await self.get_json(url, params=params)

    async def get_symbol_catalog(self, url: str) -> Any:
        return await self.get_json(url)


def owner_twelvedata_client(owner: Any, client: httpx.AsyncClient) -> TwelveDataClient:
    return TwelveDataClient(client, getattr(owner, "twelvedata_api_key", ""))


def owner_fmp_client(owner: Any, client: httpx.AsyncClient) -> FmpClient:
    return FmpClient(client, getattr(owner, "fmp_api_key", ""))
