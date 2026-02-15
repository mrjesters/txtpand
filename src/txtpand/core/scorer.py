"""Candidate scoring and ambiguity detection."""

from __future__ import annotations

from txtpand.config import TxtpandConfig
from txtpand.types import ScoredCandidate


class Scorer:
    """Picks the best candidate and detects ambiguity."""

    def __init__(self, config: TxtpandConfig | None = None) -> None:
        self.config = config or TxtpandConfig()

    def pick_best(
        self, candidates: list[ScoredCandidate]
    ) -> tuple[ScoredCandidate | None, bool]:
        """Pick the best candidate and flag if ambiguous.

        Returns:
            (best_candidate, is_ambiguous)
            - best_candidate is None if no candidates meet minimum confidence.
            - is_ambiguous is True if top-2 are within the ambiguity margin.
        """
        if not candidates:
            return None, False

        # Already sorted by score descending
        top = candidates[0]

        if top.score < self.config.min_confidence:
            return None, True

        # Check ambiguity: are top-2 within margin?
        if len(candidates) >= 2:
            second = candidates[1]
            margin = top.score - second.score
            if margin < self.config.ambiguity_margin:
                return top, True

        return top, False
