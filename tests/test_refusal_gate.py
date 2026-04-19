"""Refusal gate."""

from __future__ import annotations

from polymarket_oracle_risk.features import OracleFeatures
from polymarket_oracle_risk.refusal_gate import Signal, gate
from polymarket_oracle_risk.scorer import RiskScore


def _f() -> OracleFeatures:
    return OracleFeatures(
        resolution_type="UMA_OOV2",
        liveness_seconds=7200,
        bond_usdc=750.0,
        subjectivity_score=0.0,
        resolution_source_concreteness=1.0,
        hhi_uma_top10=0.0,
        market_volume_usd=1_000_000.0,
        max_single_wallet_position_usd=0.0,
        price_distance_from_extremes=0.5,
        hours_to_resolution=24.0,
        price_moved_pct_24h=0.0,
        proposer_whitelisted=True,
        similar_market_dispute_rate=0.0,
        llm_grok_disagrees_with_market=False,
    )


def _risk(mean: float, lo: float | None = None, hi: float | None = None) -> RiskScore:
    lo = lo if lo is not None else max(0.0, mean - 0.05)
    hi = hi if hi is not None else min(1.0, mean + 0.05)
    return RiskScore("m", mean=mean, lo_95=lo, hi_95=hi, model_version="t", features=_f())


def test_hard_refuse_above_threshold() -> None:
    sig = Signal("m", edge=0.05, size=1000.0)
    assert gate(sig, _risk(0.40), capital=10_000) is None


def test_hard_refuse_wide_posterior() -> None:
    sig = Signal("m", edge=0.05, size=1000.0)
    # mean below threshold but CI width huge → still refuse
    assert gate(sig, _risk(0.10, 0.0, 0.80), capital=10_000) is None


def test_caps_applied_in_tiers() -> None:
    sig = Signal("m", edge=0.05, size=1_000_000.0)
    lowest = gate(sig, _risk(0.05), capital=10_000)
    mid = gate(sig, _risk(0.20), capital=10_000)
    high = gate(sig, _risk(0.30), capital=10_000)
    assert lowest is not None and mid is not None and high is not None
    assert lowest.size >= mid.size >= high.size


def test_zero_capital_refuses() -> None:
    sig = Signal("m", edge=0.05, size=100.0)
    assert gate(sig, _risk(0.05), capital=0) is None


def test_zero_edge_returns_zero_size() -> None:
    sig = Signal("m", edge=0.0, size=100.0)
    result = gate(sig, _risk(0.05), capital=10_000)
    assert result is not None
    assert result.size == 0.0
