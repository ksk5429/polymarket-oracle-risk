"""NumPyro NUTS posterior fit over the Bayesian-logistic dispute model.

Priors (blueprint §2.4):
  β ~ Normal(0, 1)           weight prior
  α ~ Normal(-3, 2)          intercept prior (~2% base rate)

For 13 features and <50 high-profile dispute observations, the posterior is
wide by design. Callers should report credible intervals, not point values.

JAX / NumPyro are imported lazily inside :func:`fit_posterior` so merely
importing the package (e.g. for the CLI's ``score`` or ``subjectivity``
subcommands) does not pay the cost of loading them.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FitSummary:
    num_samples: int
    num_warmup: int
    num_chains: int
    posterior_samples: dict[str, np.ndarray]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_samples": self.num_samples,
            "num_warmup": self.num_warmup,
            "num_chains": self.num_chains,
            "posterior_samples": {k: np.asarray(v) for k, v in self.posterior_samples.items()},
        }
        with path.open("wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> FitSummary:
        with path.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301 — local file
        return cls(
            num_samples=int(data["num_samples"]),
            num_warmup=int(data["num_warmup"]),
            num_chains=int(data["num_chains"]),
            posterior_samples={k: np.asarray(v) for k, v in data["posterior_samples"].items()},
        )


def fit_posterior(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_samples: int = 2000,
    num_warmup: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
) -> FitSummary:
    """Run NUTS; return posterior samples as a :class:`FitSummary`.

    JAX and NumPyro are imported lazily because their import time is dominated
    by XLA initialization.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def _model(X, y=None):
        d = X.shape[1]
        beta = numpyro.sample("beta", dist.Normal(jnp.zeros(d), jnp.ones(d)))
        alpha = numpyro.sample("alpha", dist.Normal(-3.0, 2.0))
        logits = alpha + X @ beta
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    X_j = jnp.asarray(X, dtype=jnp.float32)
    y_j = jnp.asarray(y.astype(np.int32))
    kernel = NUTS(_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), X=X_j, y=y_j)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return FitSummary(
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        posterior_samples=samples,
    )


def prior_predictive_mean(d: int) -> float:
    """Closed-form prior-predictive mean for the default priors.

    With α ~ N(-3, 2) and β ~ N(0, I), E[sigmoid(α + 0·β)] ≈ sigmoid(-3) ≈ 0.047.
    """
    return float(1.0 / (1.0 + np.exp(3.0)))
