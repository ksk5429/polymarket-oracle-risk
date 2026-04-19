"""NumPyro posterior fit + backtest smoke-tests.

These tests run NUTS with small ``num_samples`` so CI stays fast. They verify
shapes and sanity bounds, not statistical efficiency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polymarket_oracle_risk.backtest import BacktestConfig, run_backtest
from polymarket_oracle_risk.train import fit_posterior


@pytest.fixture
def toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 200
    d = 13
    X = rng.normal(size=(n, d))
    # True logits use a hidden sparse beta; only feature 2 + 4 matter
    true_beta = np.zeros(d)
    true_beta[2] = 1.8
    true_beta[4] = -1.2
    logits = -2.0 + X @ true_beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(np.int32)
    return X, y


def test_fit_posterior_returns_expected_shapes(toy_data) -> None:
    X, y = toy_data
    summary = fit_posterior(X, y, num_samples=200, num_warmup=200)
    assert "beta" in summary.posterior_samples
    assert "alpha" in summary.posterior_samples
    assert summary.posterior_samples["beta"].shape == (200, 13)
    assert summary.posterior_samples["alpha"].shape == (200,)


def test_fit_posterior_recovers_signed_effects(toy_data) -> None:
    X, y = toy_data
    summary = fit_posterior(X, y, num_samples=400, num_warmup=400)
    mean_beta = summary.posterior_samples["beta"].mean(axis=0)
    # Non-trivial signal recovery: feature 2 positive, feature 4 negative
    assert mean_beta[2] > 0
    assert mean_beta[4] < 0


def test_fit_summary_save_load_roundtrip(tmp_path, toy_data) -> None:
    from polymarket_oracle_risk.train import FitSummary

    X, y = toy_data
    summary = fit_posterior(X, y, num_samples=100, num_warmup=100)
    path = tmp_path / "s.pkl"
    summary.save(path)
    loaded = FitSummary.load(path)
    assert loaded.num_samples == summary.num_samples
    assert np.allclose(loaded.posterior_samples["alpha"], summary.posterior_samples["alpha"])


def test_backtest_produces_finite_metrics(toy_data) -> None:
    X, y = toy_data
    ts = pd.date_range("2024-01-01", periods=len(y), freq="D", tz="UTC")
    feats = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feats)
    df["disputed"] = y
    df["ts"] = ts
    cfg = BacktestConfig(
        val_start=pd.Timestamp("2024-04-01", tz="UTC"),
        test_start=pd.Timestamp("2024-06-01", tz="UTC"),
        feature_columns=feats,
    )
    metrics, _ = run_backtest(df, cfg=cfg, num_samples=200, num_warmup=200)
    assert metrics.n_test > 0
    assert 0.0 <= metrics.brier_test <= 1.0
    assert metrics.ece_test >= 0.0
