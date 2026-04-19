"""Posterior-backed risk scorer.

Uses samples from :func:`polymarket_oracle_risk.train.fit_posterior` to produce
a posterior-mean probability + a 95% credible interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polymarket_oracle_risk.features import OracleFeatures, feature_vector
from polymarket_oracle_risk.train import FitSummary


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class PosteriorModel:
    summary: FitSummary

    def predict_one(self, features: OracleFeatures) -> tuple[float, float, float]:
        """Return ``(mean, lo_95, hi_95)`` for a single feature vector."""
        x = feature_vector(features)
        beta = self.summary.posterior_samples["beta"]  # (S, D)
        alpha = self.summary.posterior_samples["alpha"]  # (S,)
        logits = alpha + beta @ x
        probs = _sigmoid(logits)
        return (
            float(np.mean(probs)),
            float(np.quantile(probs, 0.025)),
            float(np.quantile(probs, 0.975)),
        )

    def predict_many(self, xs: np.ndarray) -> np.ndarray:
        """Return posterior-mean probability for each row of ``xs``; shape (N,).

        Shapes: alpha (S,), beta (S, D), xs (N, D).
        xs @ beta.T has shape (N, S); alpha broadcasts as (1, S) → logits (N, S).
        Mean over the posterior axis (axis=1) collapses to (N,).
        """
        beta = self.summary.posterior_samples["beta"]
        alpha = self.summary.posterior_samples["alpha"]
        logits = alpha[None, :] + xs @ beta.T  # (N, S)
        probs = _sigmoid(logits)
        return np.mean(probs, axis=1)


@dataclass(frozen=True)
class RiskScore:
    market_id: str
    mean: float
    lo_95: float
    hi_95: float
    model_version: str
    features: OracleFeatures

    @property
    def ci_95(self) -> tuple[float, float]:
        return (self.lo_95, self.hi_95)

    @property
    def width(self) -> float:
        return self.hi_95 - self.lo_95


_DEFAULT_PRIOR_MEAN = 1.0 / (1.0 + np.exp(3.0))  # sigmoid(-3)
_DEFAULT_PRIOR_CI = (0.02, 0.20)


def score_market(
    *,
    market_id: str,
    features: OracleFeatures,
    posterior: PosteriorModel | None = None,
    model_version: str = "v0.1.0",
) -> RiskScore:
    """If a fitted posterior is provided, use it; otherwise fall back to the
    prior-predictive mean with a wide CI."""
    if posterior is None:
        return RiskScore(
            market_id=market_id,
            mean=_DEFAULT_PRIOR_MEAN,
            lo_95=_DEFAULT_PRIOR_CI[0],
            hi_95=_DEFAULT_PRIOR_CI[1],
            model_version=f"{model_version}-prior",
            features=features,
        )

    mean, lo, hi = posterior.predict_one(features)
    return RiskScore(
        market_id=market_id,
        mean=mean,
        lo_95=lo,
        hi_95=hi,
        model_version=model_version,
        features=features,
    )
