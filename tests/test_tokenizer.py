"""Tests for the tokenizer."""

from txtpand.core.tokenizer import Token, tokenize


class TestTokenize:
    def test_simple_words(self):
        tokens = tokenize("cn y hel me")
        assert len(tokens) == 4
        assert [t.text for t in tokens] == ["cn", "y", "hel", "me"]
        assert all(t.expandable for t in tokens)

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_preserves_punctuation(self):
        tokens = tokenize("hello, world!")
        assert len(tokens) == 2
        assert tokens[0].text == "hello"
        assert tokens[0].trailing_punct == ","
        assert tokens[1].text == "world"
        assert tokens[1].trailing_punct == "!"

    def test_leading_punctuation(self):
        tokens = tokenize("(hello) [world]")
        assert tokens[0].leading_punct == "("
        assert tokens[0].text == "hello"
        assert tokens[0].trailing_punct == ")"
        assert tokens[1].leading_punct == "["
        assert tokens[1].text == "world"
        assert tokens[1].trailing_punct == "]"

    def test_url_preserved(self):
        tokens = tokenize("chk https://example.com plz")
        assert len(tokens) == 3
        assert tokens[1].text == "https://example.com"
        assert not tokens[1].expandable

    def test_code_preserved(self):
        tokens = tokenize("run `git commit` plz")
        assert len(tokens) == 3
        assert tokens[1].text == "`git commit`"
        assert not tokens[1].expandable

    def test_quoted_preserved(self):
        tokens = tokenize('set it to "hello world" plz')
        assert any(t.text == '"hello world"' and not t.expandable for t in tokens)

    def test_numbers_not_expandable(self):
        tokens = tokenize("v 2 things")
        num_token = next(t for t in tokens if t.text == "2")
        assert not num_token.expandable

    def test_mixed_content(self):
        tokens = tokenize("cn y hel me at https://x.com?")
        expandable = [t for t in tokens if t.expandable]
        assert len(expandable) == 5  # cn, y, hel, me, at

    def test_with_expansion(self):
        tok = Token(text="hel", expandable=True, leading_punct="(", trailing_punct=")")
        assert tok.with_expansion("help") == "(help)"

    def test_email_preserved(self):
        tokens = tokenize("snd to user@example.com plz")
        email = next(t for t in tokens if "example" in t.text)
        assert not email.expandable
