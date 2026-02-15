"""Tests for the scorer."""

from txtpand.core.scorer import Scorer
from txtpand.config import TxtpandConfig
from txtpand.types import MatchTier, ScoredCandidate


class TestScorer:
    def test_pick_best_clear_winner(self, scorer: Scorer):
        candidates = [
            ScoredCandidate("help", 0.9, MatchTier.PREFIX, 0.8, 0.9, 0.7, 0.6),
            ScoredCandidate("hello", 0.5, MatchTier.PREFIX, 0.8, 0.5, 0.5, 0.5),
        ]
        best, ambiguous = scorer.pick_best(candidates)
        assert best is not None
        assert best.word == "help"
        assert not ambiguous

    def test_pick_best_ambiguous(self, scorer: Scorer):
        candidates = [
            ScoredCandidate("work", 0.80, MatchTier.PREFIX, 0.5, 0.8, 0.7, 0.6),
            ScoredCandidate("would", 0.75, MatchTier.PREFIX, 0.5, 0.7, 0.7, 0.6),
        ]
        best, ambiguous = scorer.pick_best(candidates)
        assert best is not None
        assert ambiguous  # Within 0.15 margin

    def test_pick_best_empty(self, scorer: Scorer):
        best, ambiguous = scorer.pick_best([])
        assert best is None
        assert not ambiguous

    def test_pick_best_low_confidence(self):
        scorer = Scorer(TxtpandConfig(min_confidence=0.5))
        candidates = [
            ScoredCandidate("xyz", 0.3, MatchTier.FUZZY, 0.1, 0.2, 0.1, 0.1),
        ]
        best, ambiguous = scorer.pick_best(candidates)
        assert best is None
        assert ambiguous

    def test_single_candidate(self, scorer: Scorer):
        candidates = [
            ScoredCandidate("help", 0.9, MatchTier.PREFIX, 0.8, 0.9, 0.7, 0.6),
        ]
        best, ambiguous = scorer.pick_best(candidates)
        assert best is not None
        assert best.word == "help"
        assert not ambiguous

    def test_margin_boundary(self):
        # Exactly at margin boundary
        config = TxtpandConfig(ambiguity_margin=0.15)
        scorer = Scorer(config)
        candidates = [
            ScoredCandidate("a", 0.80, MatchTier.PREFIX, 0.5, 0.8, 0.7, 0.6),
            ScoredCandidate("b", 0.65, MatchTier.PREFIX, 0.5, 0.7, 0.7, 0.6),
        ]
        best, ambiguous = scorer.pick_best(candidates)
        # Margin is exactly 0.15, so NOT ambiguous (margin < threshold)
        assert not ambiguous
