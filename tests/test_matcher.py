"""Tests for the matcher."""

from txtpand.core.matcher import Matcher
from txtpand.config import TxtpandConfig
from txtpand.types import MatchTier


class TestMatcher:
    def test_exact_match(self, matcher: Matcher):
        candidates = matcher.match("hello")
        assert len(candidates) >= 1
        assert candidates[0].word == "hello"
        assert candidates[0].tier == MatchTier.PASSTHROUGH

    def test_prefix_match(self, matcher: Matcher):
        candidates = matcher.match("hel")
        words = [c.word for c in candidates]
        assert "help" in words or "hello" in words
        # At least one should be PREFIX tier
        prefix_candidates = [c for c in candidates if c.tier == MatchTier.PREFIX]
        assert len(prefix_candidates) > 0

    def test_known_word(self, matcher: Matcher):
        assert matcher.is_known_word("the")
        assert matcher.is_known_word("help")
        assert not matcher.is_known_word("xyzzy")

    def test_short_abbreviation(self, matcher: Matcher):
        candidates = matcher.match("cn")
        words = [c.word for c in candidates]
        assert "can" in words

    def test_candidates_sorted_by_score(self, matcher: Matcher):
        candidates = matcher.match("wo")
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_custom_abbreviations(self):
        config = TxtpandConfig(abbreviations={"k8s": "kubernetes"})
        m = Matcher(config)
        m.build()
        candidates = m.match("k8s")
        assert len(candidates) == 1
        assert candidates[0].word == "kubernetes"

    def test_custom_words(self):
        config = TxtpandConfig(custom_words={"foobar": 5.0})
        m = Matcher(config)
        m.build()
        candidates = m.match("foobar")
        assert candidates[0].word == "foobar"

    def test_build_with_extra(self):
        m = Matcher()
        m.build(
            extra_words={"zxytest": 5.0},
            abbreviations={"zt": "zxytest"},
        )
        candidates = m.match("zt")
        assert candidates[0].word == "zxytest"

    def test_empty_token(self, matcher: Matcher):
        candidates = matcher.match("")
        # Should handle gracefully
        assert isinstance(candidates, list)

    def test_single_char(self, matcher: Matcher):
        candidates = matcher.match("a")
        assert len(candidates) >= 1
        # "a" is a known word
        assert candidates[0].word == "a"

    def test_edit_distance_static(self):
        assert Matcher._edit_distance("kitten", "sitting", 10) == 3
        assert Matcher._edit_distance("hello", "hello", 5) == 0
        assert Matcher._edit_distance("abc", "xyz", 3) == 3
