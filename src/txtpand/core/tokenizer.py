"""Tokenizer for shorthand text.

Splits input into tokens while preserving:
- Punctuation attached to words
- URLs
- Code snippets (backtick-delimited)
- Quoted strings
- Numbers
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns that should NOT be expanded
_URL_RE = re.compile(
    r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE
)
_CODE_RE = re.compile(r"`[^`]+`")
_QUOTED_RE = re.compile(r'"[^"]*"|\'[^\']*\'')
_NUMBER_RE = re.compile(r"^\d+\.?\d*$")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Punctuation that can be stripped from word boundaries
_LEADING_PUNCT = re.compile(r"^([(\[{\"']+)")
_TRAILING_PUNCT = re.compile(r"([)\]}.!?,;:\"']+)$")


@dataclass(frozen=True, slots=True)
class Token:
    """A single token from the input text."""

    text: str
    expandable: bool
    leading_punct: str = ""
    trailing_punct: str = ""

    def with_expansion(self, expanded: str) -> str:
        """Reconstruct token with original punctuation."""
        return f"{self.leading_punct}{expanded}{self.trailing_punct}"


def tokenize(text: str) -> list[Token]:
    """Split shorthand text into tokens.

    Preserves URLs, code blocks, quoted strings, and punctuation context.
    Only alphabetic tokens are marked as expandable.
    """
    if not text or not text.strip():
        return []

    # First, identify protected spans (URLs, code, quotes, emails)
    protected: list[tuple[int, int, str]] = []
    for pattern in (_URL_RE, _CODE_RE, _QUOTED_RE, _EMAIL_RE):
        for m in pattern.finditer(text):
            protected.append((m.start(), m.end(), m.group()))

    # Sort by start position
    protected.sort(key=lambda x: x[0])

    # Build tokens by walking through the text
    tokens: list[Token] = []
    pos = 0

    for start, end, matched in protected:
        # Process text before this protected span
        if pos < start:
            chunk = text[pos:start]
            tokens.extend(_tokenize_chunk(chunk))
        # Add the protected span as a non-expandable token
        tokens.append(Token(text=matched, expandable=False))
        pos = end

    # Process remaining text
    if pos < len(text):
        tokens.extend(_tokenize_chunk(text[pos:]))

    return tokens


def _tokenize_chunk(chunk: str) -> list[Token]:
    """Tokenize a chunk of unprotected text."""
    tokens: list[Token] = []
    words = chunk.split()

    for word in words:
        if not word:
            continue

        # Check if it's a number
        if _NUMBER_RE.match(word):
            tokens.append(Token(text=word, expandable=False))
            continue

        # Strip leading/trailing punctuation
        leading = ""
        trailing = ""

        m = _LEADING_PUNCT.match(word)
        if m:
            leading = m.group(1)
            word = word[len(leading):]

        m = _TRAILING_PUNCT.search(word)
        if m:
            trailing = m.group(1)
            word = word[: -len(trailing)]

        if not word:
            # Pure punctuation
            tokens.append(Token(text=leading + trailing, expandable=False))
            continue

        # Expandable if alphabetic or alphanumeric (for abbreviations like "k8s")
        expandable = word.isalpha() or (word.isalnum() and any(c.isalpha() for c in word))

        tokens.append(Token(
            text=word,
            expandable=expandable,
            leading_punct=leading,
            trailing_punct=trailing,
        ))

    return tokens
