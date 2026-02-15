"""Main orchestrator — wires the expansion pipeline together."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from txtpand.config import TxtpandConfig
from txtpand.core.context import ContextResolver
from txtpand.core.matcher import Matcher
from txtpand.core.scorer import Scorer
from txtpand.core.segmenter import Segmenter
from txtpand.core.tokenizer import tokenize
from txtpand.types import ExpansionReport, MatchTier, TokenResult

if TYPE_CHECKING:
    from txtpand.llm.fallback import LLMFallback


class Expander:
    """Main text expansion engine.

    Orchestrates: tokenizer → matcher → context → scorer → (optional LLM fallback).
    """

    def __init__(
        self,
        config: TxtpandConfig | None = None,
        llm_fallback: LLMFallback | None = None,
    ) -> None:
        self.config = config or TxtpandConfig()
        self._matcher = Matcher(self.config)
        self._context = ContextResolver(self.config)
        self._scorer = Scorer(self.config)
        self._segmenter = Segmenter(self.config)
        self._llm_fallback = llm_fallback
        self._built = False

    def _ensure_built(self) -> None:
        if not self._built:
            self._matcher.build()
            self._built = True

    def add_words(self, words: dict[str, float]) -> None:
        """Add custom words with frequencies."""
        self.config.custom_words.update(words)
        self._built = False

    def add_abbreviations(self, abbreviations: dict[str, str]) -> None:
        """Add custom abbreviation mappings."""
        self.config.abbreviations.update(abbreviations)
        self._built = False

    def expand(self, text: str, spaceless: bool = False) -> str:
        """Expand shorthand text to full English.

        Args:
            text: Shorthand input.
            spaceless: If True, first segment the text to find word boundaries.

        Returns:
            Expanded text string.
        """
        report = self.expand_detailed(text, spaceless=spaceless)
        return report.expanded

    def expand_detailed(
        self, text: str, spaceless: bool = False
    ) -> ExpansionReport:
        """Expand with full diagnostic report."""
        self._ensure_built()
        start = time.monotonic()

        if not text or not text.strip():
            return ExpansionReport(input=text, expanded=text)

        segments: list[str] | None = None

        if spaceless:
            # Phase 1: Segment spaceless input
            segments = self._segmenter.segment(text)
            # Reconstruct as spaced input for the rest of the pipeline
            spaced_text = " ".join(segments)
        else:
            spaced_text = text

        # Phase 2: Tokenize
        tokens = tokenize(spaced_text)
        if not tokens:
            return ExpansionReport(input=text, expanded=text, spaceless=spaceless)

        # Phase 3: Match + Context + Score each token
        token_results: list[TokenResult] = []
        expanded_words: list[str] = []
        ambiguous_indices: list[int] = []

        for idx, token in enumerate(tokens):
            if not token.expandable:
                # Non-expandable tokens pass through as-is
                tr = TokenResult(
                    original=token.text,
                    expanded=token.text,
                    confidence=1.0,
                    tier=MatchTier.PASSTHROUGH,
                )
                token_results.append(tr)
                expanded_words.append(token.text)
                continue

            # Passthrough known words
            if self.config.passthrough_known_words and self._matcher.is_known_word(token.text):
                candidates = self._matcher.match(token.text)
                tr = TokenResult(
                    original=token.text,
                    expanded=token.text.lower(),
                    confidence=1.0,
                    candidates=candidates,
                    tier=MatchTier.PASSTHROUGH,
                )
                token_results.append(tr)
                expanded_words.append(token.with_expansion(token.text.lower()))
                continue

            # Get candidates
            candidates = self._matcher.match(token.text)

            if not candidates:
                # No match — keep original
                tr = TokenResult(
                    original=token.text,
                    expanded=token.text,
                    confidence=0.0,
                )
                token_results.append(tr)
                expanded_words.append(token.with_expansion(token.text))
                continue

            # Context-based rescoring
            prev_word = self._get_prev_expanded(token_results)
            next_word = self._peek_next_token(tokens, idx)
            candidates = self._context.rescore(candidates, prev_word, next_word)

            # Pick best
            best, ambiguous = self._scorer.pick_best(candidates)

            if best is None:
                tr = TokenResult(
                    original=token.text,
                    expanded=token.text,
                    confidence=0.0,
                    candidates=candidates,
                    ambiguous=True,
                )
                token_results.append(tr)
                expanded_words.append(token.with_expansion(token.text))
                ambiguous_indices.append(idx)
                continue

            tr = TokenResult(
                original=token.text,
                expanded=best.word,
                confidence=best.score,
                candidates=candidates,
                tier=best.tier,
                ambiguous=ambiguous,
            )
            token_results.append(tr)
            expanded_words.append(token.with_expansion(best.word))

            if ambiguous:
                ambiguous_indices.append(idx)

        # Phase 4: LLM
        llm_used = False
        if self._llm_fallback and self.config.llm_enabled:
            # Always-on polish: send full sentence to LLM for correction
            dict_expanded = " ".join(expanded_words)
            try:
                polished = self._llm_fallback.polish(text, dict_expanded)
                if polished != dict_expanded:
                    expanded_words = polished.split()
                    # Mark all ambiguous tokens as LLM-resolved
                    for idx in ambiguous_indices:
                        if idx < len(token_results):
                            token_results[idx].llm_resolved = True
                            token_results[idx].ambiguous = False
                    llm_used = True
            except Exception:
                pass  # LLM failure is non-fatal — dictionary result stands

        # Build final output
        expanded = " ".join(expanded_words)
        confidence = self._overall_confidence(token_results)

        elapsed = (time.monotonic() - start) * 1000

        return ExpansionReport(
            input=text,
            expanded=expanded,
            tokens=token_results,
            confidence=confidence,
            spaceless=spaceless,
            segments=segments,
            llm_used=llm_used,
            elapsed_ms=elapsed,
        )

    def _get_prev_expanded(self, results: list[TokenResult]) -> str | None:
        """Get the most recent expanded word for context."""
        for r in reversed(results):
            if r.expanded and r.expanded.isalpha():
                return r.expanded
        return None

    def _peek_next_token(self, tokens: list, idx: int) -> str | None:
        """Peek at the next expandable token for context hints."""
        for i in range(idx + 1, min(idx + 3, len(tokens))):
            if tokens[i].expandable:
                text = tokens[i].text.lower()
                if self._matcher.is_known_word(text):
                    return text
        return None

    def _resolve_ambiguous(
        self,
        token_results: list[TokenResult],
        expanded_words: list[str],
        tokens: list,
        ambiguous_indices: list[int],
    ) -> bool:
        """Use LLM fallback to resolve ambiguous tokens."""
        assert self._llm_fallback is not None

        ambiguous_tokens = [
            (idx, token_results[idx]) for idx in ambiguous_indices
        ]
        context = " ".join(expanded_words)

        resolutions = self._llm_fallback.resolve(ambiguous_tokens, context)

        if resolutions:
            for idx, resolved_word in resolutions.items():
                token_results[idx].expanded = resolved_word
                token_results[idx].llm_resolved = True
                token_results[idx].ambiguous = False
                expanded_words[idx] = tokens[idx].with_expansion(resolved_word)
            return True

        return False

    @staticmethod
    def _overall_confidence(results: list[TokenResult]) -> float:
        """Compute overall confidence as weighted average."""
        if not results:
            return 0.0
        expandable = [r for r in results if r.tier != MatchTier.PASSTHROUGH or r.confidence < 1.0]
        if not expandable:
            return 1.0
        return sum(r.confidence for r in results) / len(results)
