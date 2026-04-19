"""Polymarket Gamma API client — market metadata, no auth required."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

GAMMA_BASE = "https://gamma-api.polymarket.com"


class GammaClient:
    def __init__(self, base_url: str = GAMMA_BASE, timeout: float = 10.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, http2=True)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 10))
    def get_market(self, market_id: str) -> dict[str, Any]:
        r = self._client.get(f"/markets/{market_id}")
        r.raise_for_status()
        return r.json()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(1, 10))
    def list_markets(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        r = self._client.get("/markets", params={"limit": limit, "offset": offset})
        r.raise_for_status()
        return list(r.json())

    def close(self) -> None:
        self._client.close()
