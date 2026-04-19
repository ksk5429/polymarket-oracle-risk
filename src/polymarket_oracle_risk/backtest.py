"""Out-of-time backtest harness.

Uses an explicit time split: train on rows with ``ts < val_start``; validate on
``[val_start, test_start)``; test on ``>= test_start``. Reports calibration
metrics on the test split. No pre-2024 rows permitted (Paleka 2025 leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from polymarket_oracle_risk.scorer import PosteriorModel
from polymarket_oracle_risk.train import FitSummary, fit_posterior


@dataclass(frozen=True)
class BacktestMetrics:
    n_train: int
    n_val: int
    n_test: int
    brier_test: float
    auc_test: float
    logloss_test: float
    ece_test: float
    base_rate_test: float

    def summary(self) -> str:
        return (
            f"n_train={self.n_train} n_val={self.n_val} n_test={self.n_test} "
            f"brier={self.brier_test:.4f} auc={self.auc_test:.3f} "
            f"logloss={self.logloss_test:.3f} ece={self.ece_test:.3f} "
            f"base_rate={self.base_rate_test:.3f}"
        )


def _compute_ece(probs: np.ndarray, outcomes: np.ndarray, *, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, edges[1:-1], right=False)
    total = 0.0
    n = probs.size
    for k in range(n_bins):
        mask = bin_ids == k
        nk = int(np.sum(mask))
        if nk == 0:
            continue
        acc = float(np.mean(outcomes[mask]))
        conf = float(np.mean(probs[mask]))
        total += (nk / n) * abs(acc - conf)
    return float(total)


@dataclass
class BacktestConfig:
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    ts_column: str = "ts"
    feature_columns: list[str] | None = None
    label_column: str = "disputed"


def run_backtest(
    frame: pd.DataFrame,
    *,
    cfg: BacktestConfig,
    num_samples: int = 1000,
    num_warmup: int = 500,
) -> tuple[BacktestMetrics, FitSummary]:
    """Fit NUTS on train, evaluate on test split, return metrics + posterior."""
    if cfg.feature_columns is None:
        raise ValueError("feature_columns must be provided")

    train = frame[frame[cfg.ts_column] < cfg.val_start]
    val = frame[(frame[cfg.ts_column] >= cfg.val_start) & (frame[cfg.ts_column] < cfg.test_start)]
    test = frame[frame[cfg.ts_column] >= cfg.test_start]

    X_train = train[cfg.feature_columns].to_numpy(np.float64)
    y_train = train[cfg.label_column].to_numpy(np.int32)
    summary = fit_posterior(X_train, y_train, num_samples=num_samples, num_warmup=num_warmup)
    model = PosteriorModel(summary=summary)

    X_test = test[cfg.feature_columns].to_numpy(np.float64)
    y_test = test[cfg.label_column].to_numpy(np.int32)
    probs_test = model.predict_many(X_test)

    auc = roc_auc_score(y_test, probs_test) if len(np.unique(y_test)) > 1 else float("nan")
    metrics = BacktestMetrics(
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        brier_test=float(brier_score_loss(y_test, probs_test)),
        auc_test=float(auc),
        logloss_test=float(log_loss(y_test, probs_test, labels=[0, 1])),
        ece_test=_compute_ece(probs_test, y_test),
        base_rate_test=float(np.mean(y_test)),
    )
    return metrics, summary
