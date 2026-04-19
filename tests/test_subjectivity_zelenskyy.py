"""Zelenskyy-suit regression tests."""

from __future__ import annotations

import pytest

from polymarket_oracle_risk.subjectivity import (
    ZELENSKYY_RESOLUTION_TEXT,
    SubjectivityResult,
    rule_based_score,
    score_subjectivity,
)


def test_zelenskyy_rule_based_score_at_or_above_075() -> None:
    """The core invariant: the Zelenskyy text must register as highly subjective."""
    result = rule_based_score(ZELENSKYY_RESOLUTION_TEXT)
    assert result.score >= 0.75
    assert result.named_source is False


def test_named_source_text_scores_low() -> None:
    text = (
        "This market resolves YES if the official NOAA snowfall measurement "
        "at station ABCD exceeds 4 inches by the end date."
    )
    result = rule_based_score(text)
    assert result.score <= 0.35
    assert result.named_source is True


def test_pure_judgment_text_scores_max() -> None:
    text = "Resolves at the discretion of Polymarket moderators."
    result = rule_based_score(text)
    assert result.score >= 0.85


def test_llm_path_uses_client_when_provided() -> None:
    class _FakeLLM:
        def complete(self, *, system: str, user: str, json_mode: bool = True) -> object:
            # emulate LLMResponse.text
            class _R:
                text = '{"score": 0.92, "reasoning": "no named source; pure judgment", "named_source": false}'

            return _R()

    result = score_subjectivity("anything", client=_FakeLLM())
    assert result.score == pytest.approx(0.92)
    assert not result.named_source


def test_llm_path_falls_back_when_client_misbehaves() -> None:
    class _BrokenLLM:
        def complete(self, *, system: str, user: str, json_mode: bool = True) -> object:
            class _R:
                text = "totally not json"

            return _R()

    result = score_subjectivity(ZELENSKYY_RESOLUTION_TEXT, client=_BrokenLLM())
    assert isinstance(result, SubjectivityResult)
    # Rule-based fallback must still score the Zelenskyy case correctly
    assert result.score >= 0.75
