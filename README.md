# polymarket-oracle-risk

> **Note**: This directory is the source tree for a **separate public repo** —
> `github.com/ksk5429/polymarket-oracle-risk`. It lives inside the private
> `kfish` monorepo only for scaffolding convenience. Before going public:
>
> 1. `git subtree split --prefix=polymarket-oracle-risk -b split-oracle-risk`
> 2. Push `split-oracle-risk` to the new repo's `main`.
> 3. Verify no private keys, trading strategies, or internal data references
>    leak through into the public history.

Risk analyzer for **Polymarket oracle (UMA Optimistic Oracle v2)** resolutions.
Scores markets on manipulation/governance-attack risk using a Bayesian logistic
model over subjective-resolution, whale-concentration, and liveness features.

## Install

```bash
pip install polymarket-oracle-risk
```

## Usage

```python
from polymarket_oracle_risk import score_market

risk = score_market(market_id="0x...")
print(risk.mean, risk.ci_95)
```

## Methodology

Features span:

- `resolution_type` — UMA_OOV2 vs UMA_MOOV2 (post-UMIP-189) vs CHAINLINK
- `subjectivity_score` — Claude-scored, Zelenskyy regression-test anchored
- `hhi_uma_top10` — voter-concentration Herfindahl
- `market_volume_usd` + `max_single_wallet_position_usd` — attack incentive
- `price_distance_from_extremes`, `hours_to_resolution`, etc.

Posteriors fit with NumPyro NUTS using `Normal(0, 1)` weight priors and
`Normal(-3, 2)` intercept (low base rate). Per-category dispute rates use a
`Beta(1, 50)` beta-binomial prior.

Backtest window: Q3 2024 – Q1 2026; out-of-time validation split.

## Refusal gate

Default thresholds (tune to your risk appetite):

| `risk.mean` | Position cap vs unconstrained Kelly |
|---|---|
| ≤ 0.15 | 100% |
| ≤ 0.25 | 30% |
| ≤ 0.35 | 10% |
| > 0.35 | **refuse** |

## Limitations

- <50 high-profile dispute observations → wide posteriors
- Managed Proposer (UMIP-189, Aug 2025) is a regime change; segment training
- HIP-4 has no dispute history yet — scores for HIP-4 markets are low confidence
- UMA top-10 voter numbers are community-derived; re-verify from the subgraph

## License

MIT © Kyeong Sun Kim
