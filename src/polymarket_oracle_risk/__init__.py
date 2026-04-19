"""Risk analyzer for Polymarket oracle (UMA OO) resolutions."""

from importlib.metadata import PackageNotFoundError, version

from polymarket_oracle_risk.features import OracleFeatures, feature_vector
from polymarket_oracle_risk.refusal_gate import Signal, gate
from polymarket_oracle_risk.scorer import PosteriorModel, RiskScore, score_market
from polymarket_oracle_risk.subjectivity import (
    ZELENSKYY_RESOLUTION_TEXT,
    SubjectivityResult,
    rule_based_score,
    score_subjectivity,
)
from polymarket_oracle_risk.train import fit_posterior

try:
    __version__ = version("polymarket-oracle-risk")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ZELENSKYY_RESOLUTION_TEXT",
    "OracleFeatures",
    "PosteriorModel",
    "RiskScore",
    "Signal",
    "SubjectivityResult",
    "__version__",
    "feature_vector",
    "fit_posterior",
    "gate",
    "rule_based_score",
    "score_market",
    "score_subjectivity",
]
