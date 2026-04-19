"""Position-sizing refusal gate (blueprint §2.5).

Thresholds are conservative defaults. CVaR-style refusal: fat-tailed loss
distributions mean we refuse when ``P(loss > 50%) > 5%``, not just on marginal
EV. The CI width from :class:`RiskScore` is folded into the decision so we
refuse wide-CI cases even when the mean is below the hard threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeVar

from polymarket_oracle_risk.scorer import RiskScore

HARD_REFUSE_MEAN_THRESHOLD = 0.35
HARD_REFUSE_CI_WIDTH_THRESHOLD = 0.50

POSITION_CAP_PCT: dict[float, float] = {0.15: 1.00, 0.25: 0.30, 0.35: 0.10}


@dataclass(frozen=True)
class Signal:
    market_id: str
    edge: float
    size: float


S = TypeVar("S", bound=Signal)


def gate(signal: S, risk: RiskScore, capital: float) -> S | None:
    """Return an adjusted signal, or ``None`` if the market is a hard refuse.

    - Hard refuse when posterior mean risk exceeds the threshold.
    - Hard refuse when posterior is too wide to trust.
    - Otherwise, shrink size by tier cap and risk-adjusted edge.
    """
    if risk.mean > HARD_REFUSE_MEAN_THRESHOLD:
        return None
    if risk.width > HARD_REFUSE_CI_WIDTH_THRESHOLD:
        return None
    if signal.edge == 0:
        return replace(signal, size=0.0)
    if capital <= 0:
        return None

    cap = next(
        (v for k, v in sorted(POSITION_CAP_PCT.items()) if risk.mean <= k),
        POSITION_CAP_PCT[max(POSITION_CAP_PCT)],
    )
    adjusted_edge = signal.edge * (1.0 - risk.mean)
    max_size = capital * cap * (adjusted_edge / signal.edge)
    return replace(signal, size=min(signal.size, max_size))
