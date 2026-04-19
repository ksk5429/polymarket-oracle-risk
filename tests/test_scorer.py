"""Scorer / posterior / refusal-gate tests."""

from __future__ import annotations

import numpy as np
import pytest

from polymarket_oracle_risk.features import OracleFeatures, feature_matrix, feature_vector
from polymarket_oracle_risk.scorer import PosteriorModel, RiskScore, score_market
from polymarket_oracle_risk.train import FitSummary


def _features(subjectivity: float = 0.2, whitelisted: bool = True) -> OracleFeatures:
    return OracleFeatures(
        resolution_type="UMA_OOV2",
        liveness_seconds=7200,
        bond_usdc=750.0,
        subjectivity_score=subjectivity,
        resolution_source_concreteness=0.8,
        hhi_uma_top10=0.15,
        market_volume_usd=500_000.0,
        max_single_wallet_position_usd=25_000.0,
        price_distance_from_extremes=0.10,
        hours_to_resolution=48.0,
        price_moved_pct_24h=0.03,
        proposer_whitelisted=whitelisted,
        similar_market_dispute_rate=0.02,
        llm_grok_disagrees_with_market=False,
    )


def test_feature_vector_shape() -> None:
    x = feature_vector(_features())
    assert x.shape == (13,)


def test_feature_matrix_shape() -> None:
    X = feature_matrix([_features(), _features(0.8)])
    assert X.shape == (2, 13)


def test_feature_validation_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _features(subjectivity=1.5)


def test_prior_scoring_without_posterior() -> None:
    score = score_market(market_id="m", features=_features())
    # Sigmoid(-3) ~= 0.047
    assert 0.02 < score.mean < 0.10
    assert score.hi_95 > score.lo_95
    assert "prior" in score.model_version


def test_posterior_scorer_with_fake_samples() -> None:
    rng = np.random.default_rng(0)
    samples = {
        "beta": rng.normal(size=(2000, 13)).astype(np.float64),
        "alpha": rng.normal(-3.0, 2.0, size=2000),
    }
    fit = FitSummary(num_samples=2000, num_warmup=500, num_chains=1, posterior_samples=samples)
    model = PosteriorModel(summary=fit)

    high_subj = _features(subjectivity=0.95, whitelisted=False)
    low_subj = _features(subjectivity=0.05, whitelisted=True)
    s_high = score_market(market_id="h", features=high_subj, posterior=model)
    s_low = score_market(market_id="l", features=low_subj, posterior=model)
    assert 0 <= s_high.lo_95 <= s_high.mean <= s_high.hi_95 <= 1
    assert 0 <= s_low.lo_95 <= s_low.mean <= s_low.hi_95 <= 1


def test_posterior_predict_many_shape() -> None:
    rng = np.random.default_rng(1)
    samples = {
        "beta": rng.normal(size=(500, 13)).astype(np.float64),
        "alpha": rng.normal(-3.0, 2.0, size=500),
    }
    fit = FitSummary(num_samples=500, num_warmup=100, num_chains=1, posterior_samples=samples)
    model = PosteriorModel(summary=fit)
    X = feature_matrix([_features() for _ in range(4)])
    out = model.predict_many(X)
    assert out.shape == (4,)
    assert np.all((out >= 0) & (out <= 1))


def test_risk_score_width_property() -> None:
    s = RiskScore("m", mean=0.20, lo_95=0.10, hi_95=0.40, model_version="v", features=_features())
    assert s.width == pytest.approx(0.30)
    assert s.ci_95 == (0.10, 0.40)
