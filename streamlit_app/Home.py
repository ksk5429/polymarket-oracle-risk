"""Streamlit dashboard — Polymarket oracle-risk overview.

Data path (optional): a read-only DuckDB snapshot at
``data/parquet/risk_scores.parquet``. When the file is missing, we surface a
synthetic sample so the dashboard renders on Streamlit Community Cloud without
live data — essential for the "first public impression" launch window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from polymarket_oracle_risk.features import OracleFeatures
from polymarket_oracle_risk.scorer import score_market
from polymarket_oracle_risk.subjectivity import ZELENSKYY_RESOLUTION_TEXT, rule_based_score

st.set_page_config(page_title="polymarket-oracle-risk", layout="wide")

st.title("🪤 Polymarket Oracle Risk")
st.caption(
    "Posterior-backed manipulation-risk scorer for UMA-resolved Polymarket markets. "
    "Snapshots refresh nightly from the public backtest pipeline."
)

DATA_PATH = Path("data/parquet/risk_scores.parquet")


def _load_or_synth() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    rng = np.random.default_rng(0)
    n = 60
    rows = []
    for i in range(n):
        f = OracleFeatures(
            resolution_type="UMA_OOV2",
            liveness_seconds=rng.choice([3600, 7200, 172_800]),
            bond_usdc=float(rng.choice([500, 750, 5_000])),
            subjectivity_score=float(rng.beta(2, 4)),
            resolution_source_concreteness=float(rng.beta(4, 2)),
            hhi_uma_top10=float(rng.beta(2, 8)),
            market_volume_usd=float(rng.lognormal(12.5, 1.2)),
            max_single_wallet_position_usd=float(rng.lognormal(9, 1.5)),
            price_distance_from_extremes=float(rng.uniform(0.05, 0.5)),
            hours_to_resolution=float(rng.uniform(2, 240)),
            price_moved_pct_24h=float(rng.normal(0, 0.05)),
            proposer_whitelisted=bool(rng.random() < 0.6),
            similar_market_dispute_rate=float(rng.beta(1, 50)),
            llm_grok_disagrees_with_market=bool(rng.random() < 0.1),
        )
        r = score_market(market_id=f"sample-{i:03d}", features=f)
        rows.append(
            {
                "market_id": r.market_id,
                "risk_mean": r.mean,
                "risk_lo_95": r.lo_95,
                "risk_hi_95": r.hi_95,
                "volume_usd": f.market_volume_usd,
                "subjectivity": f.subjectivity_score,
                "hhi_uma_top10": f.hhi_uma_top10,
                "hours_to_resolution": f.hours_to_resolution,
            }
        )
    return pd.DataFrame(rows)


df = _load_or_synth()

with st.sidebar:
    st.header("Filters")
    max_risk = st.slider("Max risk mean", 0.0, 1.0, 0.35, 0.01)
    min_volume = st.number_input("Min volume (USD)", min_value=0, value=50_000, step=25_000)
    st.divider()
    st.subheader("Zelenskyy regression")
    rb = rule_based_score(ZELENSKYY_RESOLUTION_TEXT)
    st.metric("Subjectivity score", f"{rb.score:.2f}", help=rb.reasoning)
    st.caption("Must score ≥ 0.75 for the package to be considered in-calibration.")

filtered = df[(df["risk_mean"] <= max_risk) & (df["volume_usd"] >= min_volume)].copy()
filtered = filtered.sort_values("risk_mean")

col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Top candidates")
    st.dataframe(
        filtered[["market_id", "risk_mean", "risk_lo_95", "risk_hi_95", "volume_usd"]]
        .reset_index(drop=True)
        .head(20),
        use_container_width=True,
        column_config={
            "risk_mean": st.column_config.ProgressColumn(
                "risk mean", format="%.3f", min_value=0.0, max_value=1.0
            ),
        },
    )
with col2:
    st.subheader("Risk vs subjectivity")
    fig = px.scatter(
        filtered,
        x="subjectivity",
        y="risk_mean",
        size="volume_usd",
        color="hhi_uma_top10",
        hover_data=["market_id", "hours_to_resolution"],
        color_continuous_scale="Turbo",
        labels={
            "subjectivity": "Subjectivity score",
            "risk_mean": "Posterior mean risk",
            "hhi_uma_top10": "HHI (UMA top-10)",
        },
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Source: synthetic sample used for UX; replace with a live snapshot by dropping "
    "a `risk_scores.parquet` into `data/parquet/`."
)
