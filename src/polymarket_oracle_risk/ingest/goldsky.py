"""UMA subgraph client via Goldsky (blueprint §2.2).

Endpoint IDs rotate; keep them in env vars, not hard-coded.
"""

from __future__ import annotations

import os
from typing import Any

from gql import Client, gql
from gql.transport.httpx import HTTPXTransport


def _url(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


DEFAULT_POLYGON_OOV2 = (
    "https://api.goldsky.com/api/public/project_clcxirpj1c5vx2aton4ou0iin/"
    "subgraphs/polygon-optimistic-oracle-v2/1.1.0/gn"
)
DEFAULT_POLYGON_MOOV2 = (
    "https://api.goldsky.com/api/public/project_clcxirpj1c5vx2aton4ou0iin/"
    "subgraphs/polygon-managed-optimistic-oracle-v2/1.0.5/gn"
)

REQUESTS_QUERY = gql(
    """
    query OracleRequests($first: Int!, $skip: Int!) {
      requests(first: $first, skip: $skip, orderBy: time, orderDirection: desc) {
        id
        identifier
        ancillaryData
        time
        requester
        proposer
        disputer
        bond
        proposedPrice
        settlementPrice
        state
      }
    }
    """
)


def goldsky_client(url: str | None = None, *, managed: bool = False) -> Client:
    if url is None:
        url = _url(
            "GOLDSKY_POLYGON_MOOV2_URL" if managed else "GOLDSKY_POLYGON_OOV2_URL",
            DEFAULT_POLYGON_MOOV2 if managed else DEFAULT_POLYGON_OOV2,
        )
    return Client(transport=HTTPXTransport(url=url), fetch_schema_from_transport=False)


def fetch_requests(client: Client, first: int = 100, skip: int = 0) -> list[dict[str, Any]]:
    data = client.execute(REQUESTS_QUERY, variable_values={"first": first, "skip": skip})
    return list(data.get("requests", []))
