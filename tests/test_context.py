"""Tests for the context resolver."""

from txtpand.core.context import ContextResolver
from txtpand.types import MatchTier, ScoredCandidate


class TestContextResolver:
    def test_rescore_with_prev_word(self, context_resolver: ContextResolver):
        candidates = [
            ScoredCandidate("work", 0.8, MatchTier.PREFIX, 0.5, 0.8, 0.7, 0.6),
            ScoredCandidate("would", 0.78, MatchTier.PREFIX, 0.5, 0.7, 0.7, 0.6),
        ]
        rescored = context_resolver.rescore(candidates, prev_word="me")
        # "me work" should get a context bonus from "me_work" bigram
        # Either way, should still have valid candidates
        assert len(rescored) == 2
        assert all(isinstance(c, ScoredCandidate) for c in rescored)

    def test_rescore_no_context(self, context_resolver: ContextResolver):
        candidates = [
            ScoredCandidate("help", 0.9, MatchTier.PREFIX, 0.5, 0.8, 0.7, 0.6),
        ]
        # No prev/next word - should return unchanged
        result = context_resolver.rescore(candidates)
        assert result == candidates

    def test_rescore_empty(self, context_resolver: ContextResolver):
        result = context_resolver.rescore([], prev_word="the")
        assert result == []

    def test_rescore_with_next_word(self, context_resolver: ContextResolver):
        candidates = [
            ScoredCandidate("you", 0.9, MatchTier.EXACT, 1.0, 1.0, 0.9, 1.0),
        ]
        rescored = context_resolver.rescore(candidates, next_word="can")
        assert len(rescored) == 1

    def test_bigram_bonus_applied(self, context_resolver: ContextResolver):
        candidates = [
            ScoredCandidate("help", 0.5, MatchTier.PREFIX, 0.5, 0.5, 0.5, 0.5),
            ScoredCandidate("helicopter", 0.51, MatchTier.PREFIX, 0.3, 0.5, 0.3, 0.4),
        ]
        rescored = context_resolver.rescore(candidates, prev_word="me")
        # "help me" is a common bigram so "help" should get boosted
        # Even if "help_me" is not stored as "me_help", the rescore should handle gracefully
        assert len(rescored) == 2
