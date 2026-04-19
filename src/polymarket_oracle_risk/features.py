"""Feature vector construction (blueprint §2.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "liveness_hours",
    "log1p_bond_usdc",
    "subjectivity_score",
    "resolution_source_concreteness",
    "hhi_uma_top10",
    "log1p_market_volume_usd",
    "log1p_max_single_wallet_position_usd",
    "price_distance_from_extremes",
    "hours_to_resolution",
    "price_moved_pct_24h",
    "proposer_whitelisted",
    "similar_market_dispute_rate",
    "llm_grok_disagrees_with_market",
)


@dataclass(frozen=True)
class OracleFeatures:
    resolution_type: str
    liveness_seconds: int
    bond_usdc: float
    subjectivity_score: float
    resolution_source_concreteness: float
    hhi_uma_top10: float
    market_volume_usd: float
    max_single_wallet_position_usd: float
    price_distance_from_extremes: float
    hours_to_resolution: float
    price_moved_pct_24h: float
    proposer_whitelisted: bool
    similar_market_dispute_rate: float
    llm_grok_disagrees_with_market: bool

    def __post_init__(self) -> None:
        for name in (
            "subjectivity_score",
            "resolution_source_concreteness",
            "hhi_uma_top10",
            "price_distance_from_extremes",
            "similar_market_dispute_rate",
        ):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name}={val} must be in [0, 1]")
        for name in (
            "bond_usdc",
            "market_volume_usd",
            "max_single_wallet_position_usd",
            "hours_to_resolution",
        ):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name}={val} must be non-negative")


def feature_vector(f: OracleFeatures) -> np.ndarray:
    """Float64 array in stable column order matching :data:`FEATURE_NAMES`."""
    return np.array(
        [
            f.liveness_seconds / 3600.0,
            np.log1p(f.bond_usdc),
            f.subjectivity_score,
            f.resolution_source_concreteness,
            f.hhi_uma_top10,
            np.log1p(f.market_volume_usd),
            np.log1p(f.max_single_wallet_position_usd),
            f.price_distance_from_extremes,
            f.hours_to_resolution,
            f.price_moved_pct_24h,
            float(f.proposer_whitelisted),
            f.similar_market_dispute_rate,
            float(f.llm_grok_disagrees_with_market),
        ],
        dtype=np.float64,
    )


def feature_matrix(features_list: list[OracleFeatures]) -> np.ndarray:
    """Stack a list of :class:`OracleFeatures` into shape ``(N, D)``."""
    return np.vstack([feature_vector(f) for f in features_list])
