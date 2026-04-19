"""`polymarket-oracle-risk` CLI.

Subcommands:
  score         — score a single market (uses demo features if no JSON provided)
  subjectivity  — run the subjectivity heuristic on arbitrary text
  demo          — emit JSON risk for the Zelenskyy case study
"""

from __future__ import annotations

import argparse
import json
import sys

from polymarket_oracle_risk.features import OracleFeatures
from polymarket_oracle_risk.scorer import score_market
from polymarket_oracle_risk.subjectivity import ZELENSKYY_RESOLUTION_TEXT, rule_based_score


def _demo_features(subjectivity: float = 0.20) -> OracleFeatures:
    return OracleFeatures(
        resolution_type="UMA_OOV2",
        liveness_seconds=7200,
        bond_usdc=750.0,
        subjectivity_score=subjectivity,
        resolution_source_concreteness=0.8,
        hhi_uma_top10=0.18,
        market_volume_usd=500_000.0,
        max_single_wallet_position_usd=25_000.0,
        price_distance_from_extremes=0.10,
        hours_to_resolution=48.0,
        price_moved_pct_24h=0.03,
        proposer_whitelisted=True,
        similar_market_dispute_rate=0.02,
        llm_grok_disagrees_with_market=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polymarket-oracle-risk")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="score a market by id using demo features")
    p_score.add_argument("market_id")
    p_score.add_argument("--subjectivity", type=float, default=0.20)

    p_subj = sub.add_parser("subjectivity", help="rule-based subjectivity score on text")
    p_subj.add_argument("--text", required=False, default=None)

    sub.add_parser("demo", help="run the Zelenskyy-suit case study demo")

    args = parser.parse_args(argv)

    if args.cmd == "score":
        f = _demo_features(args.subjectivity)
        result = score_market(market_id=args.market_id, features=f)
        json.dump(
            {
                "market_id": result.market_id,
                "risk_mean": round(result.mean, 4),
                "ci_95": [round(result.lo_95, 4), round(result.hi_95, 4)],
                "width": round(result.width, 4),
                "model_version": result.model_version,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    if args.cmd == "subjectivity":
        text = args.text or sys.stdin.read()
        result = rule_based_score(text)
        json.dump(
            {
                "score": result.score,
                "reasoning": result.reasoning,
                "named_source": result.named_source,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    if args.cmd == "demo":
        subj = rule_based_score(ZELENSKYY_RESOLUTION_TEXT)
        f = _demo_features(subjectivity=subj.score)
        result = score_market(market_id="zelenskyy-suit", features=f)
        json.dump(
            {
                "subjectivity": {
                    "score": subj.score,
                    "reasoning": subj.reasoning,
                    "named_source": subj.named_source,
                },
                "risk": {
                    "mean": round(result.mean, 4),
                    "ci_95": [round(result.lo_95, 4), round(result.hi_95, 4)],
                    "width": round(result.width, 4),
                    "model_version": result.model_version,
                },
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
