"""Tests for the spaceless mode segmenter."""

from txtpand.core.segmenter import Segmenter
from txtpand.config import TxtpandConfig


class TestSegmenter:
    def setup_method(self):
        self.seg = Segmenter()

    def test_simple_words(self):
        result = self.seg.segment("helloworld")
        assert result == ["hello", "world"]

    def test_common_phrase(self):
        result = self.seg.segment("thankyou")
        # Should split into "thank" + "you"
        assert "thank" in result
        assert "you" in result

    def test_empty(self):
        assert self.seg.segment("") == []

    def test_single_word(self):
        result = self.seg.segment("hello")
        assert result == ["hello"]

    def test_the_quick(self):
        result = self.seg.segment("thequick")
        assert "the" in result
        assert "quick" in result or any("qui" in s for s in result)

    def test_known_words_joined(self):
        result = self.seg.segment("canyouhelp")
        words = result
        assert "can" in words
        assert "you" in words
        assert "help" in words

    def test_abbreviated_fragments(self):
        """The key feature: segmenting abbreviated text, not just full words."""
        result = self.seg.segment("cnyhelme")
        # Should produce segments that when expanded give "can you help me"
        # The segmenter just finds boundaries - expansion happens later
        assert len(result) >= 3  # At least cn/y/hel/me or similar

    def test_with_articles(self):
        result = self.seg.segment("iamhere")
        assert "i" in result
        assert "am" in result
        assert "here" in result

    def test_longer_input(self):
        result = self.seg.segment("workonfewthings")
        # Should have reasonable segments
        assert len(result) >= 2
        assert "work" in result or "on" in result

    def test_preserves_all_characters(self):
        """Verify no characters are lost in segmentation."""
        text = "helloworld"
        result = self.seg.segment(text)
        assert "".join(result) == text

    def test_preserves_chars_complex(self):
        text = "canihelpyou"
        result = self.seg.segment(text)
        assert "".join(result) == text
