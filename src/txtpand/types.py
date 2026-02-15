"""Type definitions for txtpand."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MatchTier(Enum):
    """How a candidate was matched."""

    EXACT = "exact"
    PREFIX = "prefix"
    FUZZY = "fuzzy"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A word candidate with its composite score."""

    word: str
    score: float
    tier: MatchTier
    prefix_score: float = 0.0
    edit_similarity: float = 0.0
    frequency: float = 0.0
    length_penalty: float = 0.0
    context_bonus: float = 0.0


@dataclass(slots=True)
class TokenResult:
    """Expansion result for a single token."""

    original: str
    expanded: str
    confidence: float
    candidates: list[ScoredCandidate] = field(default_factory=list)
    tier: MatchTier = MatchTier.EXACT
    ambiguous: bool = False
    llm_resolved: bool = False


@dataclass(slots=True)
class ExpansionReport:
    """Full expansion result with diagnostics."""

    input: str
    expanded: str
    tokens: list[TokenResult] = field(default_factory=list)
    confidence: float = 0.0
    spaceless: bool = False
    segments: list[str] | None = None
    llm_used: bool = False
    elapsed_ms: float = 0.0
