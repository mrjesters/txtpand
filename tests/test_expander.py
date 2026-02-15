"""Integration tests for the expander."""

import txtpand
from txtpand.core.expander import Expander
from txtpand.config import TxtpandConfig


class TestExpander:
    def setup_method(self):
        self.expander = Expander()

    def test_simple_expansion(self):
        result = self.expander.expand("hel me")
        assert "help" in result or "hello" in result
        assert "me" in result

    def test_known_words_passthrough(self):
        result = self.expander.expand("the quick brown fox")
        assert result == "the quick brown fox"

    def test_empty_input(self):
        assert self.expander.expand("") == ""
        assert self.expander.expand("  ") == "  "

    def test_detailed_report(self):
        report = self.expander.expand_detailed("hel me")
        assert report.input == "hel me"
        assert report.expanded
        assert report.confidence > 0
        assert report.elapsed_ms >= 0
        assert len(report.tokens) == 2

    def test_spaceless_mode(self):
        result = self.expander.expand("helloworld", spaceless=True)
        assert "hello" in result
        assert "world" in result

    def test_spaceless_detailed(self):
        report = self.expander.expand_detailed("helloworld", spaceless=True)
        assert report.spaceless is True
        assert report.segments is not None
        assert len(report.segments) >= 2

    def test_punctuation_preserved(self):
        result = self.expander.expand("hel me!")
        assert result.endswith("!")

    def test_url_preserved(self):
        result = self.expander.expand("chk https://example.com")
        assert "https://example.com" in result

    def test_custom_abbreviations(self):
        exp = Expander(TxtpandConfig(abbreviations={"k8s": "kubernetes"}))
        result = exp.expand("k8s")
        assert result == "kubernetes"

    def test_add_words(self):
        exp = Expander()
        exp.add_words({"xyzfoo": 5.0})
        result = exp.expand("xyzfoo")
        assert result == "xyzfoo"

    def test_add_abbreviations(self):
        exp = Expander()
        exp.add_abbreviations({"tf": "terraform"})
        result = exp.expand("tf")
        assert result == "terraform"

    def test_can_you_help(self):
        result = self.expander.expand("cn y hel me")
        words = result.split()
        assert "can" in words
        assert "help" in words or "hello" in words
        assert "me" in words

    def test_mixed_with_code(self):
        result = self.expander.expand("run `git commit` plz")
        assert "`git commit`" in result

    def test_numbers_preserved(self):
        result = self.expander.expand("v 2 things")
        assert "2" in result


class TestModuleAPI:
    def test_expand_function(self):
        result = txtpand.expand("hel me")
        assert "me" in result

    def test_expand_detailed_function(self):
        report = txtpand.expand_detailed("hel me")
        assert isinstance(report, txtpand.ExpansionReport)
        assert report.expanded

    def test_version(self):
        assert txtpand.__version__ == "0.1.0"
