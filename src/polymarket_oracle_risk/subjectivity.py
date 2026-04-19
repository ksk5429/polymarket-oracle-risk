"""Subjectivity scoring (blueprint §2.4).

Two paths:
  1. Rule-based heuristic — deterministic, cheap, runs in CI.
  2. LLM-backed scorer — Claude/GPT prompt that must be ≥ 0.75 on the
     Zelenskyy-suit regression test.

The rule-based score is always available; the LLM result supersedes it when a
client is provided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

ZELENSKYY_RESOLUTION_TEXT = (
    "This market resolves YES if, by the deadline, there is a consensus of "
    "credible reporting that the described event occurred. Resolution is based "
    "on the judgment of Polymarket's moderation team applying the above rule."
)


SUBJECTIVITY_RUBRIC = """\
You are rating how subjective a prediction-market resolution criterion is.

Return strict JSON:
  {"score": <float in [0, 1]>, "reasoning": <str>, "named_source": <bool>}

Scoring guide:
  0.00 — named authoritative source with objective threshold (e.g., "NOAA snowfall measurement at station X exceeds 4 inches").
  0.25 — named authoritative source with minor interpretation (e.g., a headline containing a named keyword).
  0.50 — unnamed source aggregate with objective threshold.
  0.75 — "consensus of credible reporting" or similar vague attribution with binary judgment.
  1.00 — pure judgment with no named source at all.

Rate strictly. Do not be charitable. Return ONLY the JSON object; no prose.
"""


@dataclass(frozen=True)
class SubjectivityResult:
    score: float
    reasoning: str
    named_source: bool


@runtime_checkable
class SubjectivityLLM(Protocol):
    """Anything with a ``complete(system, user, json_mode=True)`` callable works.
    The :class:`kfish_common.llm.ClaudeClient` satisfies this by construction.
    """

    def complete(self, *, system: str, user: str, json_mode: bool = True) -> object: ...


def _to_dict(raw: object) -> dict:
    """Normalize ``LLMResponse`` or dict into a plain dict."""
    if isinstance(raw, dict):
        return raw
    text = getattr(raw, "text", "")
    if not isinstance(text, str):
        raise TypeError("LLM response has no text attribute")
    import json

    return json.loads(text)


_CONSENSUS_PATTERNS = (
    r"consensus of credible reporting",
    r"generally accepted",
    r"widely reported",
    r"public consensus",
    r"judgment of .*moderation",
    r"reasonable interpretation",
)

_NAMED_SOURCE_PATTERNS = (
    r"according to\s+[A-Z][A-Za-z ]{2,}",
    r"\bNOAA\b",
    r"\bNASA\b",
    r"\bCDC\b",
    r"\bFED\b",
    r"\bfederal reserve\b",
    r"official [a-z]+ report",
    r"CoinGecko",
    r"Bloomberg",
)


def rule_based_score(text: str) -> SubjectivityResult:
    """Fast deterministic fallback; mirrors the LLM rubric."""
    lowered = text.lower()
    consensus_hits = sum(1 for p in _CONSENSUS_PATTERNS if re.search(p, lowered))
    named_hits = sum(1 for p in _NAMED_SOURCE_PATTERNS if re.search(p, text, flags=re.IGNORECASE))

    if named_hits >= 1 and consensus_hits == 0:
        score = 0.20
        reasoning = f"found {named_hits} named-source markers with no consensus language"
        named = True
    elif consensus_hits >= 1:
        score = 0.80 if named_hits == 0 else 0.55
        reasoning = (
            f"{consensus_hits} consensus-language markers and {named_hits} named sources — "
            "subjective resolution likely"
        )
        named = named_hits > 0
    elif "judgment" in lowered or "at the discretion" in lowered:
        score = 0.90
        reasoning = "explicit discretion/judgment language"
        named = False
    else:
        score = 0.40
        reasoning = "no explicit consensus or named-source signals"
        named = named_hits > 0
    return SubjectivityResult(score=round(score, 2), reasoning=reasoning, named_source=named)


def score_subjectivity(
    text: str,
    *,
    client: SubjectivityLLM | None = None,
) -> SubjectivityResult:
    """LLM-path when a client is provided, rule-based fallback otherwise."""
    if client is None:
        return rule_based_score(text)
    raw = client.complete(system=SUBJECTIVITY_RUBRIC, user=text, json_mode=True)
    try:
        payload = _to_dict(raw)
    except (ValueError, TypeError):
        # LLM misbehaved — keep going with rule-based
        return rule_based_score(text)
    try:
        return SubjectivityResult(
            score=float(payload["score"]),
            reasoning=str(payload.get("reasoning", "")),
            named_source=bool(payload.get("named_source", False)),
        )
    except (KeyError, ValueError, TypeError):
        return rule_based_score(text)
