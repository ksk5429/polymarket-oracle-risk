"""CLI smoke tests."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from polymarket_oracle_risk.cli import main


def test_score_subcommand_emits_json() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["score", "demo-market-1"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["market_id"] == "demo-market-1"
    assert 0 <= data["risk_mean"] <= 1
    assert len(data["ci_95"]) == 2


def test_subjectivity_subcommand() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["subjectivity", "--text", "Resolves at the discretion of moderators."])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["score"] >= 0.85


def test_demo_subcommand_covers_full_pipeline() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["demo"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["subjectivity"]["score"] >= 0.75
    assert 0 <= data["risk"]["mean"] <= 1
