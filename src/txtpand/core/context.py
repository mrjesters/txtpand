"""Bigram-based context resolver for candidate disambiguation.

Re-ranks candidates based on co-occurrence with neighboring words.
"""

from __future__ import annotations

from txtpand.config import TxtpandConfig
from txtpand.corpus.loader import get_bigrams
from txtpand.types import ScoredCandidate


class ContextResolver:
    """Re-ranks candidates using bigram context."""

    def __init__(self, config: TxtpandConfig | None = None) -> None:
        self.config = config or TxtpandConfig()
        self._bigrams: dict[str, float] | None = None

    def _load_bigrams(self) -> dict[str, float]:
        if self._bigrams is None:
            self._bigrams = get_bigrams()
        return self._bigrams

    def rescore(
        self,
        candidates: list[ScoredCandidate],
        prev_word: str | None = None,
        next_word: str | None = None,
    ) -> list[ScoredCandidate]:
        """Re-rank candidates by adding bigram context bonus.

        Args:
            candidates: Scored candidates from matcher.
            prev_word: The word before this token (if known).
            next_word: The word after this token (if known).

        Returns:
            Re-scored and re-sorted candidates.
        """
        if not candidates or (prev_word is None and next_word is None):
            return candidates

        bigrams = self._load_bigrams()
        max_bigram = max(bigrams.values()) if bigrams else 1.0
        weight = self.config.context_bonus_weight

        rescored: list[ScoredCandidate] = []
        for c in candidates:
            bonus = 0.0

            if prev_word:
                key = f"{prev_word.lower()}_{c.word}"
                if key in bigrams:
                    bonus += bigrams[key] / max_bigram

            if next_word:
                key = f"{c.word}_{next_word.lower()}"
                if key in bigrams:
                    bonus += bigrams[key] / max_bigram

            if bonus > 0:
                new_score = c.score + weight * bonus
                rescored.append(ScoredCandidate(
                    word=c.word,
                    score=new_score,
                    tier=c.tier,
                    prefix_score=c.prefix_score,
                    edit_similarity=c.edit_similarity,
                    frequency=c.frequency,
                    length_penalty=c.length_penalty,
                    context_bonus=bonus,
                ))
            else:
                rescored.append(c)

        rescored.sort(key=lambda c: -c.score)
        return rescored
