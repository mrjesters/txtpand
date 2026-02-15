"""3-tier matching: exact → prefix (trie) → fuzzy (edit distance).

Finds candidate expansions for abbreviated tokens.
"""

from __future__ import annotations

from txtpand.config import TxtpandConfig
from txtpand.corpus.loader import get_words
from txtpand.types import MatchTier, ScoredCandidate


class TrieNode:
    """Simple trie node for prefix matching."""

    __slots__ = ("children", "word", "freq")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.word: str | None = None
        self.freq: float = 0.0


class Matcher:
    """Three-tier candidate matcher for abbreviated tokens."""

    def __init__(self, config: TxtpandConfig | None = None) -> None:
        self.config = config or TxtpandConfig()
        self._words: dict[str, float] = {}
        self._trie_root = TrieNode()
        self._max_freq = 7.0
        self._built = False

    def build(
        self,
        extra_words: dict[str, float] | None = None,
        abbreviations: dict[str, str] | None = None,
    ) -> None:
        """Build index from corpus + optional custom words."""
        self._words = dict(get_words())

        if extra_words:
            self._words.update(extra_words)
        if self.config.custom_words:
            self._words.update(self.config.custom_words)

        self._abbreviations: dict[str, str] = {}
        if abbreviations:
            self._abbreviations.update(abbreviations)
        if self.config.abbreviations:
            self._abbreviations.update(self.config.abbreviations)

        # Build trie
        self._trie_root = TrieNode()
        for word, freq in self._words.items():
            self._trie_insert(word, freq)

        self._max_freq = max(self._words.values()) if self._words else 7.0
        self._built = True

    def _trie_insert(self, word: str, freq: float) -> None:
        node = self._trie_root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.word = word
        node.freq = freq

    def _trie_prefix_search(self, prefix: str, top_k: int) -> list[tuple[str, float]]:
        """Find top-k words starting with prefix, sorted by frequency."""
        node = self._trie_root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        # BFS to collect all words under this prefix
        results: list[tuple[str, float]] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.word is not None:
                results.append((current.word, current.freq))
            stack.extend(current.children.values())

        # Sort by frequency descending, take top_k
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def is_known_word(self, token: str) -> bool:
        """Check if token is an exact known word."""
        if not self._built:
            self.build()
        return token.lower() in self._words

    def match(self, token: str) -> list[ScoredCandidate]:
        """Find candidate expansions for a token using 3-tier matching.

        Returns scored candidates sorted by composite score.
        """
        if not self._built:
            self.build()

        token_lower = token.lower()

        # Check custom abbreviation overrides first
        if token_lower in self._abbreviations:
            target = self._abbreviations[token_lower]
            freq = self._words.get(target, 3.0)
            return [ScoredCandidate(
                word=target,
                score=1.0,
                tier=MatchTier.EXACT,
                prefix_score=1.0,
                edit_similarity=1.0,
                frequency=freq / self._max_freq,
                length_penalty=1.0,
            )]

        # Tier 1: Exact match
        if token_lower in self._words:
            freq = self._words[token_lower]
            return [ScoredCandidate(
                word=token_lower,
                score=1.0,
                tier=MatchTier.PASSTHROUGH,
                prefix_score=1.0,
                edit_similarity=1.0,
                frequency=freq / self._max_freq,
                length_penalty=1.0,
            )]

        candidates: list[ScoredCandidate] = []

        # Tier 2: Prefix match
        prefix_results = self._trie_prefix_search(
            token_lower, self.config.top_k_prefix
        )
        for word, freq in prefix_results:
            if word == token_lower:
                continue  # Already handled in exact match
            prefix_score = len(token_lower) / len(word)
            length_penalty = self._length_penalty(token_lower, word)
            norm_freq = freq / self._max_freq

            score = (
                self.config.weight_prefix * prefix_score
                + self.config.weight_edit * 1.0  # Perfect start match
                + self.config.weight_frequency * norm_freq
                + self.config.weight_length * length_penalty
            )
            candidates.append(ScoredCandidate(
                word=word,
                score=score,
                tier=MatchTier.PREFIX,
                prefix_score=prefix_score,
                edit_similarity=1.0,
                frequency=norm_freq,
                length_penalty=length_penalty,
            ))

        # Tier 3: Fuzzy match (edit distance)
        if len(token_lower) >= self.config.min_fuzzy_length:
            fuzzy_candidates = self._fuzzy_match(token_lower)
            candidates.extend(fuzzy_candidates)

        # Deduplicate (prefer higher score)
        seen: dict[str, ScoredCandidate] = {}
        for c in candidates:
            if c.word not in seen or c.score > seen[c.word].score:
                seen[c.word] = c
        candidates = sorted(seen.values(), key=lambda c: -c.score)

        return candidates

    def _fuzzy_match(self, token: str) -> list[ScoredCandidate]:
        """Fuzzy matching using edit distance."""
        max_dist = max(1, int(len(token) * self.config.max_edit_distance_ratio))
        candidates: list[ScoredCandidate] = []

        try:
            from rapidfuzz import fuzz, process

            results = process.extract(
                token,
                self._words.keys(),
                scorer=fuzz.ratio,
                limit=self.config.top_k_fuzzy,
                score_cutoff=40,
            )
            for word, ratio, _ in results:
                if word == token:
                    continue
                freq = self._words[word]
                edit_sim = ratio / 100.0
                prefix_score = self._prefix_overlap(token, word)
                length_penalty = self._length_penalty(token, word)
                norm_freq = freq / self._max_freq

                score = (
                    self.config.weight_prefix * prefix_score
                    + self.config.weight_edit * edit_sim
                    + self.config.weight_frequency * norm_freq
                    + self.config.weight_length * length_penalty
                )
                candidates.append(ScoredCandidate(
                    word=word,
                    score=score,
                    tier=MatchTier.FUZZY,
                    prefix_score=prefix_score,
                    edit_similarity=edit_sim,
                    frequency=norm_freq,
                    length_penalty=length_penalty,
                ))
        except ImportError:
            # Fall back to simple Levenshtein without rapidfuzz
            candidates = self._simple_fuzzy(token, max_dist)

        return candidates

    def _simple_fuzzy(self, token: str, max_dist: int) -> list[ScoredCandidate]:
        """Simple fuzzy matching without rapidfuzz dependency.

        Uses a prefix-filtered approach to avoid scanning entire dictionary.
        """
        candidates: list[ScoredCandidate] = []

        # Only check words that share at least the first character
        for word, freq in self._words.items():
            if abs(len(word) - len(token)) > max_dist:
                continue
            if not word or word[0] != token[0]:
                continue

            dist = self._edit_distance(token, word, max_dist)
            if dist <= max_dist:
                max_len = max(len(token), len(word))
                edit_sim = 1.0 - dist / max_len if max_len > 0 else 0.0
                prefix_score = self._prefix_overlap(token, word)
                length_penalty = self._length_penalty(token, word)
                norm_freq = freq / self._max_freq

                score = (
                    self.config.weight_prefix * prefix_score
                    + self.config.weight_edit * edit_sim
                    + self.config.weight_frequency * norm_freq
                    + self.config.weight_length * length_penalty
                )
                candidates.append(ScoredCandidate(
                    word=word,
                    score=score,
                    tier=MatchTier.FUZZY,
                    prefix_score=prefix_score,
                    edit_similarity=edit_sim,
                    frequency=norm_freq,
                    length_penalty=length_penalty,
                ))

        candidates.sort(key=lambda c: -c.score)
        return candidates[: self.config.top_k_fuzzy]

    @staticmethod
    def _edit_distance(s1: str, s2: str, max_dist: int) -> int:
        """Bounded Levenshtein distance."""
        if abs(len(s1) - len(s2)) > max_dist:
            return max_dist + 1

        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1] + [0] * len(s2)
            min_val = curr[0]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
                min_val = min(min_val, curr[j + 1])
            if min_val > max_dist:
                return max_dist + 1
            prev = curr

        return prev[len(s2)]

    @staticmethod
    def _prefix_overlap(token: str, word: str) -> float:
        """Compute prefix overlap ratio."""
        common = 0
        for a, b in zip(token, word):
            if a == b:
                common += 1
            else:
                break
        return common / len(word) if word else 0.0

    @staticmethod
    def _length_penalty(token: str, word: str) -> float:
        """Penalize candidates whose length is very different from token.

        A word is ideal if it's a natural expansion of the abbreviation
        (e.g., "hel" → "help" is better than "hel" → "helicopter").
        """
        ratio = len(token) / len(word) if word else 0.0
        # Sweet spot: abbreviation is 40-80% of full word length
        if 0.4 <= ratio <= 1.0:
            return 1.0 - abs(0.6 - ratio) * 0.5
        return max(0.0, 1.0 - abs(0.6 - ratio))
