"""Word segmenter for spaceless mode.

Uses Viterbi-style dynamic programming over character positions.
Handles both full words ("thequickbrown") and abbreviated fragments ("thqckbrwn").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from txtpand.config import TxtpandConfig
from txtpand.corpus.loader import get_bigrams, get_words
from txtpand.exceptions import SegmentationError


@dataclass(frozen=True, slots=True)
class _DPEntry:
    """A DP table entry for Viterbi segmentation."""

    score: float
    back: int  # backpointer to start of last segment
    word: str  # the word/abbreviation chosen at this position


# Penalty applied to abbreviation matches (vs exact dictionary words)
_ABBREV_PENALTY = 0.4

# Bonus for longer exact-match segments (strongly discourages fragmenting real words)
_EXACT_LENGTH_BONUS = 0.8

# Bonus for longer abbreviation segments
_ABBREV_LENGTH_BONUS = 0.15

# Penalty for very short (1-char) segments that aren't real words
_SHORT_PENALTY = 3.0


class Segmenter:
    """Segment spaceless text into words/abbreviations using DP."""

    def __init__(self, config: TxtpandConfig | None = None) -> None:
        self.config = config or TxtpandConfig()
        self._words: dict[str, float] | None = None
        self._bigrams: dict[str, float] | None = None
        self._max_freq: float = 7.0
        self._max_bigram: float = 7.0

    def _load(self) -> None:
        if self._words is None:
            self._words = get_words()
            if self.config.custom_words:
                self._words = {**self._words, **self.config.custom_words}
            self._max_freq = max(self._words.values()) if self._words else 7.0

            self._bigrams = get_bigrams()
            self._max_bigram = max(self._bigrams.values()) if self._bigrams else 7.0

    def segment(self, text: str) -> list[str]:
        """Find optimal word boundaries in spaceless text.

        Uses Viterbi-style DP. For each position i in the text, tries all
        substrings text[j:i] and scores them as either:
        1. Exact dictionary word → score = log_freq(word)
        2. Prefix of a word → score = log_freq(best_match) * abbreviation_penalty
        3. Plus bigram bonus with previous segment

        Args:
            text: Spaceless input like "cnyhelmewoonfethin"

        Returns:
            List of segments like ["cn", "y", "hel", "me", "wo", "on", "fe", "thin"]
        """
        self._load()
        assert self._words is not None
        assert self._bigrams is not None

        if not text:
            return []

        text = text.lower().strip()
        n = len(text)
        max_word_len = min(self.config.max_word_length, n)

        # DP table: best[i] = best way to segment text[0:i]
        # best[0] = empty prefix (score 0, no backpointer)
        best: list[_DPEntry | None] = [None] * (n + 1)
        best[0] = _DPEntry(score=0.0, back=0, word="")

        for i in range(1, n + 1):
            for length in range(1, min(max_word_len, i) + 1):
                j = i - length
                if best[j] is None:
                    continue

                substr = text[j:i]
                prev_entry = best[j]

                # Score this substring
                word_score, matched_word = self._score_substring(substr)
                if word_score <= -999:
                    continue  # No plausible match

                # Bigram bonus with previous segment
                bigram_bonus = 0.0
                if prev_entry.word and matched_word:
                    bkey = f"{prev_entry.word}_{matched_word}"
                    if bkey in self._bigrams:
                        bigram_bonus = (
                            self._bigrams[bkey] / self._max_bigram * 0.5
                        )

                total = prev_entry.score + word_score + bigram_bonus

                if best[i] is None or total > best[i].score:
                    best[i] = _DPEntry(
                        score=total, back=j, word=matched_word or substr
                    )

        # Backtrack to recover segmentation
        if best[n] is None:
            raise SegmentationError(f"Cannot segment: {text!r}")

        segments: list[str] = []
        pos = n
        while pos > 0:
            entry = best[pos]
            assert entry is not None
            start = entry.back
            segments.append(text[start:pos])
            pos = start

        segments.reverse()
        return segments

    def _score_substring(self, substr: str) -> tuple[float, str | None]:
        """Score a substring as a potential word/abbreviation.

        Returns (score, matched_word). Score of -999 means no match.
        """
        assert self._words is not None

        # 1. Exact dictionary match — strongly preferred
        if substr in self._words:
            freq = self._words[substr]
            score = freq / self._max_freq
            # Strong bonus for longer exact matches to prevent fragmenting real words
            score += len(substr) * _EXACT_LENGTH_BONUS
            return score, substr

        # 2. Check if it's a prefix of any word
        best_prefix_score = -999.0
        best_prefix_word: str | None = None

        for word, freq in self._words.items():
            if word.startswith(substr) and len(word) > len(substr):
                # It's a prefix - score based on frequency and how much of the word is typed
                prefix_ratio = len(substr) / len(word)
                score = (freq / self._max_freq) * _ABBREV_PENALTY * prefix_ratio
                # Small bonus for longer abbreviations
                score += len(substr) * _ABBREV_LENGTH_BONUS
                if score > best_prefix_score:
                    best_prefix_score = score
                    best_prefix_word = word

        if best_prefix_word is not None:
            # Penalize single-character abbreviations that aren't real words
            if len(substr) == 1 and substr not in self._words:
                best_prefix_score -= _SHORT_PENALTY
            return best_prefix_score, best_prefix_word

        # 3. No match at all
        # Still allow single characters as fallback with heavy penalty
        if len(substr) == 1:
            return -2.0, None

        return -999.0, None
